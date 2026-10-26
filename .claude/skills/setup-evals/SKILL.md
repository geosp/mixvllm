---
name: setup-evals
description: Create the private model-evaluation suite for this cluster from the public task specs in evals/ - generate each code challenge's answer key (reference solution, hidden tests, mutants, stress cases, independent solver), verify it with spark-cluster eval verify, fill in the vision answers and run a baseline. Use when spark-cluster eval reports no suite or an unverified suite, or the user asks to set up, create, regenerate or repair the evals.
---

# Set up the evaluation suite

`spark-cluster eval` grades models with a private suite on the head (`docker/cluster.yml`'s
`evals_path`, usually `/mnt/shared/evals`). The repo only has the public task specs in `evals/`:
prompts, settings, media and a `SPEC.md` per code challenge. This skill writes the missing answer
keys **on the head**, and checks them with `uv run spark-cluster eval verify` before any model is
graded with them.

Tell the user once, at the start: generating answer keys sends the prompts and the generated
answers through Claude's API, and results from this suite are only comparable with other results
from the same suite (`eval report` tags different suite fingerprints).

## Never

- Never write answer-key files (`hidden_tests.py`, `reference.py`, `mutants/`, `brute.py`,
  `stress_cases.py`, `stress_expected.json`, `expected.md`, expected answers) into this repo, a
  commit message, an issue or anywhere public. They live only under `evals_path` on the head.
- Never overwrite an existing suite file without asking: `eval init` never does, and neither should
  you. A suite marked `"origin": "curated"` is hand-validated; leave it alone.
- Never weaken a hidden test just to make the reference pass. When they disagree, one of them
  misreads the spec: re-read `prompt.md` and decide which.

## 1. Create the folder

```
uv run spark-cluster eval init
```
Copies the public specs to `evals_path` (keeping anything already there), puts the generic grader
(`grader.py`, from `spark_cluster/evalkit/grader_template.py`) into each code challenge, and lists
which suites are ready. Only `code` suites need generating; `challenge3` reuses challenge2's
grader, and `speed` and `tools` need nothing.

Write files on the head over SSH, e.g. `ssh <head> 'cat > /mnt/shared/evals/suites/challenge1/reference.py' <<'EOF' ... EOF`
(`<head>` is the first node's `ssh` value in `docker/cluster.yml`). Read
`spark_cluster/evalkit/grader_template.py`'s docstring first: it defines the file formats below.

## 2. For each code challenge that isn't ready

Read its `prompt.md` and `SPEC.md` (in `evals/suites/<name>/`), then write, in its folder on the
head:

1. **`reference.py`**: a careful, straightforward solution of the prompt (standard library
   only). Correctness over speed, but it must stay within the timeout in `SPEC.md`.
2. **`hidden_tests.py`**:
   - `FUNCTION`, `TIMEOUT` (from `SPEC.md`), and `CASES = [(description, args, expected), ...]`
     with at least 15 cases (more if `SPEC.md` recommends), covering every `[category-id]` in
     `SPEC.md`. List the ids with their cases in the module docstring: `verify` checks they appear.
   - Work out each `expected` from the prompt by hand. Don't generate expected values by running
     your reference: then a bug in it becomes a "correct" answer.
   - `judge(args, outcome, expected)` returning `None` for a pass or a short reason. `outcome` is a
     `sandbox.Outcome` (`kind` ok/raised/timeout/crash, `value`, `raised("ValueError")` for an
     exception or subclass). Values arrive as JSON: tuples become lists, `Fraction` stays
     `Fraction`, other objects become `sandbox.Other`. Follow `SPEC.md`'s pass rules exactly
     (e.g. type checks, validating a placement rather than comparing it to one answer).
3. **`mutants/m1_<bug>.py` … at least 5**: copies of `reference.py`, each with one realistic bug
   that a weaker model might make (`SPEC.md` suggests some). Each must still load and run.
4. **If `SPEC.md` describes a stress test:**
   - `stress_cases.py` with `generate(seed)` (a few thousand cases from `random.Random(seed)`),
     `expected(reference_module, args)` and `judge_stress(args, outcome, expected)` returning
     `ok`/`missed`/`impossible`/`invalid`/`error`.
   - `brute.py` with `expected(args)` computed **a different way** than the reference (e.g. a plain
     exhaustive search), and optionally `applicable(args)` to skip cases too big for it.
   - Don't write `stress_expected.json` yourself: `verify` builds it from the reference.
5. Run `uv run spark-cluster eval verify <name>` and fix whatever it reports:
   - **reference below 100%**: a test's expected value or the judge is wrong, or the reference is.
     Decide from the prompt.
   - **a mutant isn't caught**: add a case aimed at that bug.
   - **reference and brute.py disagree**: one of them is wrong. Check the example it prints by hand.
   - **missing categories / too few cases**: add cases.

   Repeat until it prints `PASSED`. Any later edit changes the fingerprint and needs `verify` again.

## 3. Vision answers

For `suites/vision`, write `expected.md` and add an `"expected"` string to each question in its
`suite.json` (on the head only). Read `accuracy_plot.png` yourself for the chart questions, and the
generators `make_video.py` and `make_audio.py` for what the clip and the audio contain. Record the
specific facts a correct answer must contain, and note that an HTTP 400 about audio or video means
the model has no such input ("not supported", not wrong).

## 4. Check and baseline

1. `uv run spark-cluster eval init` again: every suite should now show ready.
2. Sanity check (recommended): write two or three deliberately different solutions yourself
   (one good, one subtly wrong) to a scratch folder on the head and grade them with the suite's
   grader. The good one should pass, the wrong one shouldn't.
3. With the user's go-ahead (it loads the cluster for ~30 minutes):
   `uv run spark-cluster eval run --tag baseline`, on the model currently served. This is the
   reference row other models are compared with.
4. Remind the user that the suite exists only on the head: suggest copying `evals_path` somewhere
   private (not the public repo) as a backup.
