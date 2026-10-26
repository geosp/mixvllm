# Evaluating a model on the Spark cluster

How to compare a model against the current lineup with the same tests used to pick the default
(October 2026). Every test talks to the `spark` alias on the cluster, so it measures whatever model
is loaded. Two ways to run it:

- **`/evaluate-model <model>`** in Claude Code: the skill in `.claude/skills/evaluate-model/` does
  the whole job. It researches the model, adds a registry entry, loads it, runs the suite, tries a
  few settings, grades the vision answers, reports a verdict and restores the default.
- **`uv run spark-cluster eval ...`** by hand, as below.

## The suite and where it lives

The test content is **private**: it lives on the head at `cluster.yml`'s `evals_path`
(`/mnt/shared/evals`), not in this public repo. Publishing the hidden tests and reference
solutions would let future models train on the answers.

| Suite | What it measures | Graded by |
|---|---|---|
| `speed` | Prose and code tok/s (512 tokens forced), time for a short answer, draft-token acceptance | numbers |
| `tools` | One `get_weather` tool call comes back as a structured `tool_calls` entry | pass/fail |
| `challenge1` | Exact arithmetic expression evaluator (precedence, unary minus, `^`, errors): 45 hidden tests. Moderate; strong models all pass. | hidden tests |
| `challenge2` | Smallest shipping box for up to 6 items in 3D with rotations, with a valid placement: 18 hidden tests plus a 4,587-case stress test (fixed seed) against an exhaustive reference. Runs twice by default (seeds 1 and 2). | hidden tests + stress |
| `challenge3` | The challenge 2 task, but the model also writes its own unit tests. They run against its code, and failures go back to it for up to 3 fix rounds. The final code gets the hidden tests and the stress test. | hidden tests + stress |
| `vision` | Two chart questions, a video and an audio clip | a person or the agent, against `suites/vision/expected.md` |

The stress test reports *missed fits* (said no box fits when one does), *impossible claimed*,
*invalid arrangements* and *errors/timeouts*. Missed fits are what separate models.

**Grading is isolated.** The model's code runs in its own locked-down container (`--network none
--read-only --cap-drop ALL`, unprivileged user, 2 GB, 64 processes) that contains only its own
`solution.py` and a small shim. The graders run outside it, send each test case's arguments over a
pipe and check the answers themselves. So the model's code has no hidden tests, reference solution
or expected answers it could read. The runner also reports a count of risky calls in the code
(`eval`, `exec`, `os`, `subprocess`, `socket`, `open(`). These graders were checked against the
original ones: all 37 archived October solutions get identical scores.

## No suite yet? Generate one

The repo has the public half of every task in `evals/`: prompts, settings, media and a `SPEC.md`
per code challenge that says how to grade it. To build the private half:

1. `uv run spark-cluster eval init`: creates `evals_path` on the head from those specs (it never
   overwrites anything) and lists which suites still need an answer key.
2. `/setup-evals` in Claude Code. For each code challenge Claude writes a reference solution,
   hidden tests (expected values worked out from the prompt, not from the reference), at least
   five deliberately broken **mutants**, and for the stress test a case generator plus an
   independent **brute-force solver**. It then runs `uv run spark-cluster eval verify`, which only
   passes a suite when:
   - the reference scores 100%
   - grading twice gives the same result
   - every mutant is caught
   - the reference and the brute-force solver agree on at least 500 stress cases
   - every category in `SPEC.md` has hidden tests
3. A baseline run of your current model.

`eval run` refuses to grade with an answer key that hasn't passed `verify` for its current
contents. Your own hand-built suite is marked `"origin": "curated"` and verifies on its reference
alone.

How well this works: a suite generated from the public specs alone (October 2026) passed
`verify`, then graded the 37 archived October model solutions. It reached the same verdicts as the
curated suite on all of them: the same solutions passed, the same ones failed, with nearly the
same stress numbers. Its new hidden tests also caught three real bugs the curated tests missed
(a tab accepted as a space, and a missed fit for a three-item set).

## Running it by hand

1. **Load the model** (skip if it's already loaded).
   - Add a registry entry in `docker/config/model_registry.yml`: copy the closest existing one and
     keep `served-model-name: [<own name>, spark]`.
   - Check it with `MODEL_NAME=<name> python docker/config/launch.py --dry-run`.
   - Commit, push, `uv run spark-cluster pull`.
   - `uv run spark-cluster download <name>` if the weights aren't on disk.
   - `uv run spark-cluster model <name>`: restarts the cluster, about 10 minutes.
2. **Run the suite:**
   ```bash
   uv run spark-cluster eval run --tag <name>_default
   ```
   It starts a detached run on the head and shows a live table: each suite's state, time,
   thinking progress, result and stress numbers. A model can think for 15+ minutes. Ctrl+C only
   stops watching; the run keeps going.

   | Command | |
   |---|---|
   | `eval run [suites…]` | all suites by default; name some to run only those |
   | `eval watch [run]` | reattach (default: the latest run) |
   | `eval status` | recent runs: done, running, stopped |
   | `eval stop [run]` | end a run and its sandbox containers |
   | `eval report` | results table across runs (`--model`, `--run`, `--last`) |
   | `eval grade <run> vision 3/4 --note "..."` | record the vision grade |

   While a run is going, `up`, `down`, `restart` and `model` refuse to run, because a restart
   mid-run would spoil its results.
3. **Settings worth varying**, each as its own tagged run (usually only the code suites):

   | Option | Effect |
   |---|---|
   | `--effort low` | sends `reasoning_effort` (default: the server's, from the registry's `default-chat-template-kwargs`) |
   | `--ctk '{"enable_thinking": false}'` | sends `chat_template_kwargs` (Gemma thinking on/off) |
   | `--budget 10000` | `thinking_token_budget` (challenge 2) |
   | `--repeat N`, `--seed S` | repeats per suite; repeat *k* uses seed `S+k-1` |
   | `--prompt-suffix '…'` | appends text to the code prompts |

   Effort names differ per model: Qwen accepts `low | medium | xhigh`, DeepSeek `low | high | max`,
   gpt-oss `low | medium | high` (its server default `medium` can't be changed). Speculative
   decoding (MTP / DSpark) changes speed only, not output. Flash-Next went from 27–33 to 43–51
   tok/s with MTP (2 drafts).
4. **Grade the vision answers.** On the head, compare
   `runs/<run>/vision_answers.json` with `suites/vision/expected.md`, then
   `uv run spark-cluster eval grade <run> vision <correct>/<total>`.
5. **Record the outcome** (below).

Every run is stored on the head under `runs/<run-id>/`: its config (including the registry entry,
image and git commit), events, each answer, reasoning, extracted solution and grade. Each run also
adds one row per suite and repeat to `results.csv`.

## Reading the results

- **Other users share the cluster.** vLLM batches everyone's requests, so while other sessions are
  generating, each request gets fewer tokens per second: Flash-Next measured 39.5 code tok/s with
  other users active and 51 on an idle server. The speed suite samples the server's load while it
  measures and the report marks shared results `(shared: up to N other requests)`; compare those
  with each other, not with idle numbers. Correctness scores aren't affected by load.
- **Compare within one suite.** Every result records its suite's fingerprint (a hash of the
  prompt, hidden tests, grader and stress cases), and `eval run` prints them. Results from
  different suites (someone else's, or an older version of yours) aren't comparable. `eval report`
  tags them `[a]`/`[b]` with a footnote, and `--suite-hash` shows one version only.
- **One run is one sample.** Requests carry a seed (repeat *k* uses seed *k*), but vLLM's results
  aren't exactly reproducible when other requests share the batch, so the same seed can still give
  a different answer on a busy cluster. That's why challenge 2 runs twice: gpt-oss at `medium`
  crashed once, then missed 22, then 5, and in October 2026 Flash-Next missed 0 twice but 5 twice
  in a later run. Report each run, not just the best.
- **Speed vs time to answer.** tok/s is the generation rate. Total time also depends on how much
  the model thinks (Flash-Next wrote 7K tokens on one run and 16K on another).
- **A higher effort level isn't always better.** Qwen3.8-27B at `medium`/`xhigh` never finished
  challenge 2 within 32K tokens. A result of `finish=length` with no code block means it ran out of
  tokens while thinking.
- **Challenge 3's fix loop rarely triggers**: models usually write tests their own code passes.
  What it shows is how many tests target the hard cases (tight multi-item packings), and whether
  the final stress result improves on the single-shot one.
- **A challenge 3 round can run out of tokens.** If a round thinks until `max_tokens` without
  writing code, the next round only asks for both code blocks again, so it is a fresh attempt
  rather than a fix, and it uses up one of the fix rounds. The watch table lists each round
  (`round 1: 32,000 tokens, out of tokens, no code`) and the report adds `N round(s) out of
  tokens`. On 2026-10-25 Flash-Next spent round 1 that way, then wrote code in round 2 and fixed
  2 failing tests of its own in round 3: 18/18, 0 missed fits, 16m45s.

## Baseline to compare against (vLLM 0.32, server-default effort unless noted)

| Model | Prose / code tok/s | Short reply | Challenge 1 | Challenge 2 (missed fits) | Challenge 3 | Vision |
|---|---|---|---|---|---|---|
| **qwen3.8-flash-next** (default; MTP 2, 0.70) | 43.1 / 50.8 | 2.7 s | 45/45 | 18/18, **0** (twice) | 18/18, **0**, 19 own tests, 907 s | passed |
| qwen3.8-27b (`low`) | 34 / 44 | 3.2 s | 45/45, 97 s | 18/18, 5 (twice) | — | passed |
| gpt-oss-120b (`medium`) | 55 / 58 | 2.5 s | 45/45, 75 s | crash / 22 / 5; `high`: 0, 299 s | 18/18, 5, 91 s | no vision |
| gemma-4-26b-a4b (thinking off) | 44 / 50 | 1.4 s | 45/45, 126 s | 18/18, 18 and 22 | 18/18, 55, 47 s | weak (video wrong) |
| deepseek-v4-flash (`low`) | 26 / 35 | 6.3 s | 45/45, 233 s | 18/18, 5, 413 s | — | no vision |

These October numbers predate `spark-cluster eval`; newer runs are in `uv run spark-cluster eval
report`. Tool calling works on all five. Flash-Next without MTP was 27–29 / 33 tok/s and 5.3 s.

## Recording the outcome

Add the model's row to the table above and to the "Which model to run" table in
`docker/README.md`. If it replaces a model, remove the old registry entry, and ask before deleting
its weights from `/mnt/shared/huggingface`.
