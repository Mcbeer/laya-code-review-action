# laya-review

Deterministic, ruleset-driven PR review using the
[Laya](https://huggingface.co/convaiinnovations/laya) System One decision model.

Laya does not generate text. For every changed hunk it answers each rule in
`.laya/rules.yaml` as a two-option question and returns `p(violation)`. The same
diff, ruleset, model revision and library versions give byte-identical results.

Background and measurements: [docs/research/laya-pr-review-feasibility.md](docs/research/laya-pr-review-feasibility.md).

> **Status: experimental.** On the seed eval set most rules rank violations
> above compliant code, but probabilities sit in the middle band, and realistic
> multi-issue hunks are misjudged. Every rule ships as `warning`, so CI never
> blocks. Grow `eval/cases.yaml` before promoting anything to `error`.

## How it works

1. `git diff` of the PR is split per hunk into chunks that fit the model's
   state budget (~290 tokens on the English checkpoint). Deleted files, binary
   files and hunks without added lines are skipped.
2. Each chunk is judged against every rule whose `paths` match its file, in one
   forward pass.
3. Per chunk: `p >= fail_at` is fail, `p <= pass_at` is pass, anything between is
   unsure. A rule's verdict is its worst chunk.
4. Only `severity: error` rules decide the outcome. Exit code 1 on fail.

## Usage

```sh
pip install --index-url https://download.pytorch.org/whl/cpu -c constraints.txt torch
pip install -c constraints.txt -e ".[dev]"

git diff origin/main...HEAD > pr.diff
laya-review review --ruleset .laya/rules.yaml --diff pr.diff --json-out laya-report.json

# Measure rules against labelled hunks: accuracy, Brier score, AUROC, and
# per-case probabilities for picking thresholds.
laya-review eval --ruleset .laya/rules.yaml --cases eval/cases.yaml --json-out eval.json

pytest   # unit tests, no model download
```

First run downloads ~808 MB of weights to `~/.cache/huggingface`.

## Writing rules

- One narrow, semantic judgement per rule. Mechanical checks (secrets, lint,
  tests touched) belong in gitleaks, semgrep or linters, which are exact.
- Add labelled cases for each rule to `eval/cases.yaml`. Use AUROC to see
  whether the rule has signal at all, and the per-case probabilities to set
  `pass_at` / `fail_at`.
- Changing the model revision, `constraints.txt` or a rule's wording changes
  scores. Re-run the eval.

## CI

`.github/workflows/laya-review.yml` runs on `pull_request` with a read-only
token, caches the weights, writes the report to the job summary and uploads
`laya-report.json` as an artifact.
