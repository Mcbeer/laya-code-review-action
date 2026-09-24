# Laya for PR review on GitHub Actions — feasibility research

Date: 2026-09-24
Question: Can we run `convaiinnovations/laya` on a GitHub Actions runner, feed it a PR diff, and score the PR on quality indexes?

**Short answer: Yes, it runs, but not as a "code reviewer".** Laya is a
*classifier/scorer* that returns calibrated probabilities for questions we
define. It never generates text, so it cannot write review comments. It fits
the "score the PR on quality indexes" half of the goal. It does not fit a
"review the PR" goal unless something else writes the comments.

## 1. What Laya actually is

| Fact | Source |
|---|---|
| HF `pipeline_tag` is `text-classification`, license Apache-2.0 | HF model API, `https://huggingface.co/api/models/convaiinnovations/laya` |
| "Non-autoregressive System 1 decision model… returns typed answers with calibrated probabilities in a single forward pass… It never generates text" | Model card, `README.md` |
| Input = a *state* (text/JSON) + *typed questions*: `choice`, `score` (ordinal), `noul` (yes/no probability) | Model card, Quickstart |
| Answer space defined at request time; no retraining needed for new schemas | Model card, Architecture ("Option markers") |
| Three checkpoints: `laya` (ModernBERT-large, 421M, 512 ctx), `multilingual` (mmBERT-base, 322M, 1024 ctx, up to 8,192 with `max_len=8192`), `typed-decisions` (421M, 1024 ctx) | Model card, checkpoint table |
| English weights ~808 MB, multilingual ~647 MB | Model card, Single-Model Mode |
| `pip install laya`, v0.3.20, Python >=3.10, deps `torch>=2.0`, `transformers>=4.48` | PyPI JSON, `https://pypi.org/pypi/laya/json` |
| Optional extras: `onnx`, `serve`, `mcp`, `fast` (TileLang GPU) | PyPI JSON `requires_dist` |

### Honest limits stated by the authors (all from the model card, "Honest Limits")

- Base checkpoints are "near chance on typed-decisions zero-shot" (0.362 vs 0.461 majority baseline). "Laya is a fast base to specialise, not a zero-shot decision engine."
- English checkpoint: 512 tokens total, `head_max_len=192`, so **~320 tokens for the state**. A diff must be chunked.
- `score` (ordinal) is the weakest primitive (SST-5 0.372).
- `noul` can follow its `false:/true:` labels instead of the input; authors recommend a two-option `choice` with neutral keys instead.
- `action.act_probability` carries no usable signal; gate on `confidence`.
- Ships over-confident; needs per-(question type, option count) temperature refit on your own data (ECE 0.466 -> 0.081).
- Nothing in the card mentions source code or diffs as a trained/evaluated domain. Benchmarks are intent, NLI, news, emotion, banking, email/ticket triage.

## 2. Can it run on a GitHub-hosted runner?

Runner specs (GitHub docs, `https://docs.github.com/en/actions/reference/runners/github-hosted-runners`):

| Repo type | `ubuntu-latest` | 
|---|---|
| Public | 4 CPU, 16 GB RAM, 14 GB SSD |
| Private | 2 CPU, 8 GB RAM, 14 GB SSD |

No GPU on standard runners. Model card reports CPU latency with `Router(preload=True)` of **193–464 ms** per request and a cold checkpoint build of ~7.4 s median on CPU.

A 421M-param model (~808 MB weights) plus CPU PyTorch fits comfortably in 8 GB RAM and 14 GB disk. **Verdict: feasible on standard runners, including private-repo 2-CPU ones.**

### Local smoke test (evidence, not a GitHub run)

Run on an Apple Silicon Mac, CPU, `laya==0.3.20`, `torch 2.14.0`, English checkpoint, fresh venv (`/tmp/laya-poc/poc.py`):

- First download + load: **39.2 s** (13 s of that download).
- 5 questions per diff, one forward pass: **435–532 ms**.

Two tiny diffs: a clean typed `add()` + test, and a bad one with a hardcoded API key, string-concatenated SQL, bare `except: pass`, no tests. Questions were written as 2-option `choice` per the authors' advice.

| Question | Clean diff | Bad diff | Correct? |
|---|---|---|---|
| hardcoded secret | B (no) 0.83 | A (yes) 0.77 | yes / yes |
| SQL by concatenation | B (no) 0.92 | A (yes) 0.83 | yes / yes |
| tests added | A (yes) 0.77 | A (yes) 0.87 | yes / **no** |
| errors swallowed | ok 0.69 | swallowed 0.57 | yes / yes (weak) |
| overall quality (score 0–2) | 1.40 | 1.48 | **no discrimination** |

Reading: pattern-like, concrete questions ("is there a hardcoded key?") show signal zero-shot. Holistic questions ("overall quality") and the "tests added" question did not. This matches the card's warnings. Sample size is 2, so treat it as a sanity check, not an evaluation. A GitHub runner CPU will be slower than an M-series Mac.

## 3. Recommended architecture

```
on: pull_request
  └─ job (ubuntu-latest, permissions: contents: read)
       1. checkout, `git diff origin/<base>...HEAD`
       2. split diff per file, then per hunk, to <= ~300 tokens each
       3. restore HF cache (actions/cache on ~/.cache/huggingface) to skip the 808 MB download
       4. laya.load(...) once, then batch-predict every chunk with a fixed question set
       5. aggregate: per index, take max/mean probability across chunks
       6. write $GITHUB_STEP_SUMMARY + upload JSON artifact; optional check-run / label
```

Design notes:

- **Quality indexes = narrow, concrete `choice` questions**, one per index (secrets, SQL/command injection, swallowed errors, debug prints / TODOs left in, tests touched, public API changed, large-function added). Avoid a single "overall quality" `score`; derive the PR score from the index probabilities.
- **Chunking is mandatory** on the English checkpoint (~320 state tokens). Option: the `multilingual` checkpoint with `max_len=8192` for whole files, but the card says accuracy drops past ~4k tokens and the `Router` routes Latin-script text (code) to English unless you force `model="multilingual"`.
- **Cheap structural facts (tests touched, lines changed, files touched) should come from `git diff --stat`, not the model.** The smoke test already got "tests added" wrong.
- **Calibration / fine-tuning** is the path to real accuracy. The authors ship a Kaggle 2×T4 fine-tuning notebook (linked from the card). Training data could come from past PRs labelled by reviewers or by a stronger LLM acting as teacher.
- **Fork PRs / security**: Laya only reads the diff as data, so plain `pull_request` with a read-only token is enough to compute scores. Posting PR comments from fork PRs needs a write token. Do not solve that by running PR code under `pull_request_target`. Use the job summary or a follow-up `workflow_run` job for commenting. (GitHub's documented hardening guidance; verify at implementation time.)
- **Pin the model revision** (`sha` `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` at time of writing) so scores are reproducible between runs.

## 4. Risks / open questions

1. Code is out of Laya's evaluated domain. Zero-shot accuracy on diffs is unknown beyond the 2-sample smoke test. The first milestone should be a small labelled eval set (~50–100 hunks) before trusting any score.
2. Probabilities ship over-confident; a raw 0.8 is not 80%. Temperature fitting needs labelled data.
3. Laya flags things; it cannot explain them. If human-readable review comments are required, pair it with a generative LLM. Laya then acts as the cheap router/gate deciding *which* hunks deserve LLM review.
4. Model metadata shows 0 downloads with a very recent `lastModified`. Young project; API churn likely. Pin `laya==0.3.20`.

## 5. Ruleset-driven review and determinism

Goal clarified: Laya does not write free-form review. It judges each diff chunk against **a ruleset we define**, so the AI review gives the same verdict for the same input.

### Vendor positioning

TypeSafe's launch post for Jev (`https://typesafe.ai/blog/introducing-system-one-models-and-jev`) does not mention code review specifically. It positions System One models as "AI-Powered Workflows / smart if-statements" and "Verify everything. Score, judge, verify, guardrail", with outputs "defined in advance" and "More consistent: returns similar answers for similar inputs". The Laya card says it uses the same RLCD approach and ships a Jev-compatible API (`laya-serve`, `POST /v1/systemone`). A ruleset -> typed questions -> verdict pipeline matches that positioning. Note the post's own evals use frontier LLM outputs as the reference answer, not ground truth.

### Determinism: measured

Local CPU test (`/tmp/laya-poc/det.py`, same setup as section 2):

- Same diff, 5 calls in one process: **byte-identical** answers.
- Two separate processes: identical output hash (`746cb4681f0d2afc`).
- Inserting one blank line into the diff: every verdict kept the same, but probabilities moved by up to **0.026**. The "errors swallowed" rule sat at 0.57, so a threshold of 0.5 is only ~0.07 away from flipping.

What that means:

1. **Deterministic != correct.** A wrong answer is wrong every time (e.g. "tests added" in section 2). Determinism makes errors *reproducible*, which helps debugging and fine-tuning. It does not reduce them.
2. **Reproducibility needs pinning:** model revision sha, `laya`, `torch`, and `transformers` versions, and runner image. Floating-point results can differ across CPU types and library builds. This was not measured across machines. Expect tiny probability drift, not verdict changes, except near thresholds.
3. **Input normalisation matters.** Chunking must be deterministic: same split, same order, same context lines. Otherwise the same logical change scores differently.
4. **Use bands, not a single threshold:** e.g. `p >= 0.8` fail, `p <= 0.2` pass, anything in between = "needs human". Laya's `confidence` field is the gate the authors recommend.

### Where Laya fits in a ruleset (and where it does not)

For rules that are mechanically checkable, existing static tools are fully deterministic and more accurate than an ML classifier:

| Rule kind | Example | Better tool |
|---|---|---|
| Pattern / syntax | hardcoded secret, bare `except`, `console.log` left in | gitleaks, semgrep, linters |
| Structural fact | tests touched, lines changed, files changed | `git diff --stat`, path globs |
| **Fuzzy / semantic** | "change matches PR description", "logs sensitive user data", "names describe behaviour", "adds error handling appropriate to context", "business logic in a view layer" | **Laya** |

Recommendation: the ruleset file tags each rule with an engine (`static` or `laya`). Laya only answers rules no regex or AST can express. Otherwise we pay ML error rates for problems that already have exact solutions.

### Tooling to build

1. **Ruleset schema** (YAML): `id`, `engine`, `question`, `criteria` (2-option `choice` preferred, per the card's `noul` warning), `severity`, `fail_threshold`, `pass_threshold`, `applies_to` globs.
2. **Diff chunker**: per-file -> per-hunk, fixed context, token-bounded to the checkpoint budget (~320 state tokens on English root).
3. **Runner script**: load pinned model once, batch all rules per chunk in one forward pass, aggregate by max probability per rule across chunks, emit JSON + markdown summary.
4. **Eval harness**: labelled hunks per rule, reports accuracy/ECE per rule. It gates whether a rule is allowed to fail CI, and later feeds temperature fitting / fine-tuning.
5. **Workflow**: `pull_request`, read-only token, cache HF weights, job summary + artifact.

## 6. First build: measured results

The tool (`laya_review/`, see `README.md`) runs on CPU (`device="cpu"`). The section 2 latency numbers were measured with Laya's automatic device choice, which picks MPS on Apple Silicon, so they were not pure CPU.

Seed eval: `eval/cases.yaml`, 6 single-issue hunks per rule (3 violation, 3 compliant), default bands 0.2 / 0.8:

| Rule | Unsure | AUROC | p range, violations | p range, compliant |
|---|---|---|---|---|
| sensitive-data-logging | 6/6 | 1.00 | 0.575–0.704 | 0.392–0.482 |
| swallowed-errors | 6/6 | 0.89 | 0.645–0.749 | 0.432–0.650 |
| hardcoded-environment-config | 5/6 | 1.00 | 0.537–0.647 | 0.195–0.383 |
| commented-out-code | 3/6 | 0.67 | 0.164–0.646 | 0.136–0.356 |

On single-issue hunks the model mostly ranks violations above compliant code, but its probabilities cluster around 0.4–0.7. Fixed 0.2 / 0.8 bands therefore return "unsure". Thresholds must be set per rule from eval data, and 6 cases is too few to set them without overfitting.

Realistic multi-issue diff (`/tmp/sample.diff`): one hunk logs a password *and* swallows an exception; a second file hardcodes a DB URL. Results:

- `sensitive-data-logging`: **0.885 on the DB URL hunk** (wrong), 0.352 on the password-logging hunk (should fail).
- `swallowed-errors`: 0.446 on the hunk with `except Exception: pass`, lower than the 0.519 on the unrelated config hunk.

Two runs gave byte-identical JSON. **Deterministic, and wrong on the case that matters.** Zero-shot, the base checkpoint does not reliably judge mixed real-world hunks. Options, cheapest first:

1. Smaller chunks (e.g. added lines only, or a few lines of context) so each state carries fewer concepts.
2. Tune rule wording and criteria against a larger eval set.
3. Fine-tune on labelled hunks (the authors' notebook); the card reports 0.362 -> 0.766 from fine-tuning on another domain.

## 7. Suggested next step

Build a proof-of-concept workflow + a `score_pr.py` script with 5–7 `choice` indexes, run it against ~20 real historical PRs, and compare with human judgement before investing in fine-tuning.
