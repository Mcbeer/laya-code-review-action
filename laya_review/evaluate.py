"""Measure each rule against labelled hunks before trusting it in CI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .diff import render_state
from .judge import Judge
from .review import Verdict, classify
from .ruleset import Rule, Ruleset, RulesetError

_EXPECT = {"violation": Verdict.FAIL, "compliant": Verdict.PASS}


@dataclass(frozen=True)
class Case:
    rule_id: str
    expect: Verdict
    path: str
    hunk: str
    body: str


@dataclass(frozen=True)
class RuleScore:
    rule_id: str
    total: int
    correct: int
    wrong: int
    unsure: int
    brier: float
    auroc: float | None  # threshold-free: can p(violation) rank violations above compliant?
    probabilities: tuple[tuple[str, float], ...]  # (expected verdict, p) per case, for threshold tuning

    @property
    def accuracy(self) -> float | None:
        """Accuracy on the cases the rule actually decided."""
        decided = self.correct + self.wrong
        return self.correct / decided if decided else None


def load_cases(path: str | Path, ruleset: Ruleset) -> list[Case]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    known = {rule.id for rule in ruleset.rules}
    cases = []
    for i, item in enumerate((raw or {}).get("cases", [])):
        where = f"case #{i + 1}"
        if item.get("rule") not in known:
            raise RulesetError(f"{where}: unknown rule {item.get('rule')!r}")
        if item.get("expect") not in _EXPECT:
            raise RulesetError(f"{where}: expect must be 'violation' or 'compliant'")
        hunk, _, body = str(item.get("diff", "")).strip("\n").partition("\n")
        if not hunk.startswith("@@") or not body:
            raise RulesetError(f"{where}: diff must start with an @@ hunk header and have a body")
        cases.append(
            Case(item["rule"], _EXPECT[item["expect"]], item.get("path", "file"), hunk, body)
        )
    return cases


def evaluate(ruleset: Ruleset, cases: list[Case], judge: Judge) -> list[RuleScore]:
    rules = {rule.id: rule for rule in ruleset.rules}
    outcomes: dict[str, list[tuple[Verdict, Verdict, float]]] = {rid: [] for rid in rules}
    for case in cases:
        rule = rules[case.rule_id]
        state = render_state(case.path, case.hunk, case.body)
        p = judge.violation_probabilities(state, [rule])[rule.id]
        outcomes[rule.id].append((case.expect, classify(rule, p), p))
    return [_score(rules[rid], outcomes[rid]) for rid in rules if outcomes[rid]]


def _score(rule: Rule, rows: list[tuple[Verdict, Verdict, float]]) -> RuleScore:
    correct = sum(1 for expect, got, _ in rows if got is expect)
    unsure = sum(1 for _, got, _ in rows if got is Verdict.UNSURE)
    brier = sum((p - (1.0 if expect is Verdict.FAIL else 0.0)) ** 2 for expect, _, p in rows)
    return RuleScore(
        rule_id=rule.id,
        total=len(rows),
        correct=correct,
        wrong=len(rows) - correct - unsure,
        unsure=unsure,
        brier=brier / len(rows),
        auroc=_auroc(rows),
        probabilities=tuple((expect.value, p) for expect, _, p in rows),
    )


def _auroc(rows: list[tuple[Verdict, Verdict, float]]) -> float | None:
    positives = [p for expect, _, p in rows if expect is Verdict.FAIL]
    negatives = [p for expect, _, p in rows if expect is Verdict.PASS]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if pos > neg else 0.5 if pos == neg else 0.0 for pos in positives for neg in negatives)
    return wins / (len(positives) * len(negatives))


def to_markdown(scores: list[RuleScore]) -> str:
    lines = [
        "| Rule | Cases | Correct | Wrong | Unsure | Accuracy (decided) | Brier | AUROC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in scores:
        acc = "n/a" if s.accuracy is None else f"{s.accuracy:.2f}"
        auroc = "n/a" if s.auroc is None else f"{s.auroc:.2f}"
        lines.append(
            f"| `{s.rule_id}` | {s.total} | {s.correct} | {s.wrong} | {s.unsure} | {acc} "
            f"| {s.brier:.3f} | {auroc} |"
        )
    return "\n".join(lines) + "\n"


def to_dict(scores: list[RuleScore]) -> list[dict[str, Any]]:
    return [
        {
            "rule": s.rule_id,
            "total": s.total,
            "correct": s.correct,
            "wrong": s.wrong,
            "unsure": s.unsure,
            "accuracy": s.accuracy,
            "brier": round(s.brier, 4),
            "auroc": None if s.auroc is None else round(s.auroc, 4),
            "cases": [{"expect": e, "probability": round(p, 4)} for e, p in s.probabilities],
        }
        for s in scores
    ]
