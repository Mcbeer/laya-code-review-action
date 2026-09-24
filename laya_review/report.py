"""Render a review report as JSON or Markdown."""

from __future__ import annotations

from typing import Any

from .review import Report, Verdict

_ICON = {Verdict.PASS: "PASS", Verdict.UNSURE: "UNSURE", Verdict.FAIL: "FAIL"}


def to_dict(report: Report) -> dict[str, Any]:
    return {
        "outcome": report.outcome.value,
        "model": {"repo": report.model.repo, "revision": report.model.revision},
        "chunks": report.chunks,
        "rules": [
            {
                "id": r.rule.id,
                "severity": r.rule.severity.value,
                "verdict": r.verdict.value,
                "max_probability": _round(r.max_probability),
                "fail_at": r.rule.fail_at,
                "pass_at": r.rule.pass_at,
                "findings": [
                    {
                        "path": f.path,
                        "hunk": f.hunk,
                        "probability": _round(f.probability),
                        "verdict": f.verdict.value,
                    }
                    for f in r.findings
                ],
            }
            for r in report.results
        ],
    }


def to_markdown(report: Report) -> str:
    lines = [
        f"## Laya review: {_ICON[report.outcome]}",
        "",
        f"Model `{report.model.repo}@{report.model.revision[:12]}`, {report.chunks} chunk(s) judged.",
        "Only `error` rules decide the outcome; `unsure` means a human should look.",
        "",
        "| Rule | Severity | Verdict | Max p(violation) | Thresholds |",
        "|---|---|---|---|---|",
    ]
    for r in report.results:
        max_p = "n/a" if r.max_probability is None else f"{r.max_probability:.3f}"
        lines.append(
            f"| `{r.rule.id}` | {r.rule.severity.value} | {_ICON[r.verdict]} | {max_p} "
            f"| pass <= {r.rule.pass_at}, fail >= {r.rule.fail_at} |"
        )

    flagged = [r for r in report.results if r.findings]
    if flagged:
        lines += ["", "### Findings"]
    for r in flagged:
        lines += ["", f"**`{r.rule.id}`**: {r.rule.question}", ""]
        for f in r.findings:
            lines.append(f"- {_ICON[f.verdict]} p={f.probability:.3f} `{f.path}` `{f.hunk}`")
    return "\n".join(lines) + "\n"


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
