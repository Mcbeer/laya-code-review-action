"""Apply a ruleset to a diff and aggregate per-rule verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .diff import Chunk, chunk_diff
from .judge import Judge
from .ruleset import ModelPin, Rule, Ruleset, Severity


class Verdict(str, Enum):
    """Ordered from best to worst so `max` picks the worst."""

    PASS = "pass"
    UNSURE = "unsure"
    FAIL = "fail"

    @property
    def rank(self) -> int:
        return _RANK[self]


_RANK = {Verdict.PASS: 0, Verdict.UNSURE: 1, Verdict.FAIL: 2}


def classify(rule: Rule, probability: float) -> Verdict:
    if probability >= rule.fail_at:
        return Verdict.FAIL
    if probability <= rule.pass_at:
        return Verdict.PASS
    return Verdict.UNSURE


@dataclass(frozen=True)
class Finding:
    path: str
    hunk: str
    probability: float
    verdict: Verdict


@dataclass(frozen=True)
class RuleResult:
    rule: Rule
    verdict: Verdict
    max_probability: float | None
    findings: tuple[Finding, ...]  # non-passing chunks, worst first


@dataclass(frozen=True)
class Report:
    outcome: Verdict
    model: ModelPin
    chunks: int
    results: tuple[RuleResult, ...]


def review(diff_text: str, ruleset: Ruleset, judge: Judge) -> Report:
    chunks = chunk_diff(diff_text, judge.count_tokens, judge.state_budget)
    findings: dict[str, list[Finding]] = {rule.id: [] for rule in ruleset.rules}
    maxima: dict[str, float] = {}

    for chunk in chunks:
        rules = [rule for rule in ruleset.rules if rule.applies_to(chunk.path)]
        probabilities = judge.violation_probabilities(chunk.as_state(), rules)
        for rule in rules:
            p = probabilities[rule.id]
            maxima[rule.id] = max(p, maxima.get(rule.id, 0.0))
            verdict = classify(rule, p)
            if verdict is not Verdict.PASS:
                findings[rule.id].append(_finding(chunk, p, verdict))

    results = tuple(
        _rule_result(rule, maxima.get(rule.id), findings[rule.id]) for rule in ruleset.rules
    )
    return Report(
        outcome=_outcome(results), model=ruleset.model, chunks=len(chunks), results=results
    )


def _finding(chunk: Chunk, probability: float, verdict: Verdict) -> Finding:
    return Finding(path=chunk.path, hunk=chunk.hunk, probability=probability, verdict=verdict)


def _rule_result(rule: Rule, max_p: float | None, findings: list[Finding]) -> RuleResult:
    ordered = tuple(sorted(findings, key=lambda f: (-f.probability, f.path, f.hunk)))
    verdict = max((f.verdict for f in ordered), key=lambda v: v.rank, default=Verdict.PASS)
    return RuleResult(rule=rule, verdict=verdict, max_probability=max_p, findings=ordered)


def _outcome(results: tuple[RuleResult, ...]) -> Verdict:
    """Only `error` rules decide the outcome; `warning` rules are reported only."""
    blocking = (r.verdict for r in results if r.rule.severity is Severity.ERROR)
    return max(blocking, key=lambda v: v.rank, default=Verdict.PASS)
