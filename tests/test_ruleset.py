from pathlib import Path

import pytest

from laya_review.ruleset import RulesetError, Severity, load_ruleset, parse_ruleset

REPO = Path(__file__).resolve().parents[1]


def base(**rule_overrides):
    rule = {"id": "r1", "question": "q?", "violation": "yes", "compliant": "no", **rule_overrides}
    return {"model": {"repo": "org/model", "revision": "abc"}, "rules": [rule]}


def test_defaults():
    rule = parse_ruleset(base()).rules[0]
    assert rule.severity is Severity.WARNING
    assert (rule.pass_at, rule.fail_at) == (0.2, 0.8)
    assert rule.applies_to("any/path/file.py")


def test_question_spec_is_two_option_choice():
    spec = parse_ruleset(base()).rules[0].question_spec()
    assert spec == {"type": "choice", "instructions": "q?", "criteria": {"A": "yes", "B": "no"}}


def test_paths_filter():
    rule = parse_ruleset(base(paths=["*.py"])).rules[0]
    assert rule.applies_to("src/deep/a.py")
    assert not rule.applies_to("src/a.ts")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"severity": "fatal"}, "severity"),
        ({"fail_at": 0.3, "pass_at": 0.5}, "pass_at must be below"),
        ({"fail_at": 1.5}, "thresholds"),
        ({"question": ""}, "'question'"),
        ({"paths": []}, "'paths'"),
    ],
)
def test_rejects_bad_rules(overrides, message):
    with pytest.raises(RulesetError, match=message):
        parse_ruleset(base(**overrides))


def test_rejects_duplicate_ids_and_missing_model():
    raw = base()
    raw["rules"].append(dict(raw["rules"][0]))
    with pytest.raises(RulesetError, match="duplicate"):
        parse_ruleset(raw)
    with pytest.raises(RulesetError, match="model"):
        parse_ruleset({"rules": base()["rules"]})


def test_shipped_ruleset_is_valid():
    ruleset = load_ruleset(REPO / ".laya" / "rules.yaml")
    assert ruleset.rules
