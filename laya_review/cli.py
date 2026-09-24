"""Command line entry point: `laya-review review` and `laya-review eval`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evaluate, report
from .judge import LayaJudge
from .review import Verdict, review
from .ruleset import load_ruleset


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laya-review")
    sub = parser.add_subparsers(required=True)

    rev = sub.add_parser("review", help="judge a unified diff against the ruleset")
    rev.add_argument("--ruleset", required=True)
    rev.add_argument("--diff", required=True, help="path to a unified diff, or - for stdin")
    rev.add_argument("--json-out", help="write the JSON report here")
    rev.add_argument("--summary-out", help="append the Markdown report here (e.g. $GITHUB_STEP_SUMMARY)")
    rev.add_argument("--device", default="cpu")
    rev.set_defaults(handler=_review)

    ev = sub.add_parser("eval", help="score each rule against labelled cases")
    ev.add_argument("--ruleset", required=True)
    ev.add_argument("--cases", required=True)
    ev.add_argument("--json-out")
    ev.add_argument("--device", default="cpu")
    ev.set_defaults(handler=_eval)
    return parser


def _review(args: argparse.Namespace) -> int:
    ruleset = load_ruleset(args.ruleset)
    diff_text = sys.stdin.read() if args.diff == "-" else Path(args.diff).read_text(encoding="utf-8")
    result = review(diff_text, ruleset, LayaJudge(ruleset.model, device=args.device))

    markdown = report.to_markdown(result)
    print(markdown)
    if args.summary_out:
        with open(args.summary_out, "a", encoding="utf-8") as f:
            f.write(markdown)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_dict(result), indent=2) + "\n")
    return 1 if result.outcome is Verdict.FAIL else 0


def _eval(args: argparse.Namespace) -> int:
    ruleset = load_ruleset(args.ruleset)
    cases = evaluate.load_cases(args.cases, ruleset)
    scores = evaluate.evaluate(ruleset, cases, LayaJudge(ruleset.model, device=args.device))
    print(evaluate.to_markdown(scores))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(evaluate.to_dict(scores), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
