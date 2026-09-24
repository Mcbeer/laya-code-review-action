"""Load and validate the review ruleset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

# Each rule is asked as a two-option `choice` with neutral keys. The Laya model
# card warns that `noul` can follow its true/false labels instead of the input.
VIOLATION_KEY = "A"
COMPLIANT_KEY = "B"


class RulesetError(ValueError):
    """Raised when a ruleset file is malformed."""


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ModelPin:
    repo: str
    revision: str


@dataclass(frozen=True)
class Rule:
    id: str
    question: str
    violation: str
    compliant: str
    severity: Severity
    fail_at: float
    pass_at: float
    paths: tuple[str, ...]

    def applies_to(self, path: str) -> bool:
        """fnmatch semantics: `*` also matches `/`, so `*.py` matches any Python file."""
        return any(fnmatch(path, pattern) for pattern in self.paths)

    def question_spec(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.question,
            "criteria": {VIOLATION_KEY: self.violation, COMPLIANT_KEY: self.compliant},
        }


@dataclass(frozen=True)
class Ruleset:
    model: ModelPin
    rules: tuple[Rule, ...]


_DEFAULT_FAIL_AT = 0.8
_DEFAULT_PASS_AT = 0.2


def load_ruleset(path: str | Path) -> Ruleset:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse_ruleset(raw)


def parse_ruleset(raw: Any) -> Ruleset:
    if not isinstance(raw, dict):
        raise RulesetError("ruleset must be a mapping")
    model = _parse_model(raw.get("model"))
    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise RulesetError("ruleset needs a non-empty 'rules' list")
    rules = tuple(_parse_rule(r) for r in raw_rules)
    ids = [r.id for r in rules]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise RulesetError(f"duplicate rule ids: {', '.join(duplicates)}")
    return Ruleset(model=model, rules=rules)


def _parse_model(raw: Any) -> ModelPin:
    if not isinstance(raw, dict):
        raise RulesetError("ruleset needs a 'model' mapping with 'repo' and 'revision'")
    repo, revision = raw.get("repo"), raw.get("revision")
    if not _is_text(repo) or not _is_text(revision):
        raise RulesetError("'model.repo' and 'model.revision' must be non-empty strings")
    return ModelPin(repo=repo, revision=revision)


def _parse_rule(raw: Any) -> Rule:
    if not isinstance(raw, dict):
        raise RulesetError("each rule must be a mapping")
    rule_id = raw.get("id")
    if not _is_text(rule_id):
        raise RulesetError("each rule needs a non-empty 'id'")
    for key in ("question", "violation", "compliant"):
        if not _is_text(raw.get(key)):
            raise RulesetError(f"rule {rule_id!r}: '{key}' must be a non-empty string")

    try:
        severity = Severity(raw.get("severity", Severity.WARNING.value))
    except ValueError:
        raise RulesetError(f"rule {rule_id!r}: severity must be 'error' or 'warning'") from None

    fail_at = raw.get("fail_at", _DEFAULT_FAIL_AT)
    pass_at = raw.get("pass_at", _DEFAULT_PASS_AT)
    if not all(isinstance(t, (int, float)) and 0.0 <= t <= 1.0 for t in (fail_at, pass_at)):
        raise RulesetError(f"rule {rule_id!r}: thresholds must be numbers in [0, 1]")
    if pass_at >= fail_at:
        raise RulesetError(f"rule {rule_id!r}: pass_at must be below fail_at")

    paths = raw.get("paths", ["*"])
    if not isinstance(paths, list) or not paths or not all(_is_text(p) for p in paths):
        raise RulesetError(f"rule {rule_id!r}: 'paths' must be a non-empty list of globs")

    return Rule(
        id=rule_id,
        question=raw["question"],
        violation=raw["violation"],
        compliant=raw["compliant"],
        severity=severity,
        fail_at=float(fail_at),
        pass_at=float(pass_at),
        paths=tuple(paths),
    )


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
