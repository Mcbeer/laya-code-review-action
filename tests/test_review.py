from pathlib import Path

from laya_review import evaluate, report
from laya_review.review import Verdict, review
from laya_review.ruleset import load_ruleset, parse_ruleset

REPO = Path(__file__).resolve().parents[1]


class FakeJudge:
    """Returns a fixed p(violation) per rule id, keyed by a marker in the state."""

    state_budget = 1000

    def __init__(self, table):
        self.table = table
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def violation_probabilities(self, state, rules):
        self.calls.append((state, [r.id for r in rules]))
        marker = next((m for m in self.table if m in state), None)
        return {r.id: self.table.get(marker, {}).get(r.id, 0.0) for r in rules}


RULESET = parse_ruleset(
    {
        "model": {"repo": "org/model", "revision": "abc"},
        "rules": [
            {"id": "block", "severity": "error", "question": "q", "violation": "y", "compliant": "n"},
            {"id": "warn", "question": "q", "violation": "y", "compliant": "n"},
            {"id": "py-only", "question": "q", "violation": "y", "compliant": "n", "paths": ["*.py"]},
        ],
    }
)


def diff(path, added):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n ctx\n+{added}\n"


def by_id(result):
    return {r.rule.id: r for r in result.results}


def test_error_rule_fail_blocks():
    judge = FakeJudge({"BAD": {"block": 0.9}})
    result = review(diff("a.py", "BAD"), RULESET, judge)
    assert result.outcome is Verdict.FAIL
    assert by_id(result)["block"].findings[0].path == "a.py"


def test_warning_fail_does_not_block_and_middle_band_is_unsure():
    judge = FakeJudge({"X": {"warn": 0.95, "block": 0.5}})
    result = review(diff("a.py", "X"), RULESET, judge)
    rules = by_id(result)
    assert rules["warn"].verdict is Verdict.FAIL
    assert rules["block"].verdict is Verdict.UNSURE
    assert result.outcome is Verdict.UNSURE


def test_rule_verdict_is_worst_chunk_and_path_filter_applies():
    judge = FakeJudge({"LOW": {"block": 0.1}, "HIGH": {"block": 0.85}})
    text = diff("a.ts", "LOW") + diff("b.ts", "HIGH")
    result = review(text, RULESET, judge)
    block = by_id(result)["block"]
    assert block.verdict is Verdict.FAIL
    assert block.max_probability == 0.85
    assert [f.path for f in block.findings] == ["b.ts"]
    assert all("py-only" not in ids for _, ids in judge.calls)
    assert by_id(result)["py-only"].max_probability is None


def test_empty_diff_passes():
    result = review("", RULESET, FakeJudge({}))
    assert result.outcome is Verdict.PASS
    assert result.chunks == 0


def test_reports_render():
    result = review(diff("a.py", "BAD"), RULESET, FakeJudge({"BAD": {"block": 0.9}}))
    as_dict = report.to_dict(result)
    assert as_dict["outcome"] == "fail"
    assert as_dict["rules"][0]["findings"][0]["probability"] == 0.9
    markdown = report.to_markdown(result)
    assert "Laya review: FAIL" in markdown
    assert "`a.py`" in markdown


def test_eval_scores_shipped_cases_with_perfect_judge():
    ruleset = load_ruleset(REPO / ".laya" / "rules.yaml")
    cases = evaluate.load_cases(REPO / "eval" / "cases.yaml", ruleset)

    class Oracle(FakeJudge):
        def violation_probabilities(self, state, rules):
            case = next(c for c in cases if c.body in state and c.rule_id == rules[0].id)
            return {rules[0].id: 1.0 if case.expect is Verdict.FAIL else 0.0}

    scores = evaluate.evaluate(ruleset, cases, Oracle({}))
    assert {s.rule_id for s in scores} == {r.id for r in ruleset.rules}
    assert all(s.accuracy == 1.0 and s.brier == 0.0 and s.auroc == 1.0 for s in scores)


def test_eval_auroc_is_threshold_free():
    ruleset = load_ruleset(REPO / ".laya" / "rules.yaml")
    rule = ruleset.rules[0]
    cases = [
        evaluate.Case(rule.id, Verdict.FAIL, "a", "@@ -1 +1 @@", "+v1"),
        evaluate.Case(rule.id, Verdict.PASS, "a", "@@ -1 +1 @@", "+c1"),
    ]
    # Both inside the unsure band, but correctly ordered.
    judge = FakeJudge({"v1": {rule.id: 0.6}, "c1": {rule.id: 0.4}})
    [score] = evaluate.evaluate(ruleset, cases, judge)
    assert score.unsure == 2 and score.accuracy is None
    assert score.auroc == 1.0
