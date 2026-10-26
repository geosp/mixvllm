---
name: evaluate-model
description: Evaluate a model on the Spark cluster against the current lineup - research it, add a registry entry, load it, run the private test suite with spark-cluster eval, try a few settings, grade the vision answers, report a verdict and restore the default. Use when the user asks to evaluate, test, try, benchmark or compare a model on the cluster, e.g. "/evaluate-model glm-5.3-flash".
---

# Evaluate a model on the cluster

You are running an evaluation the way it was done by hand in October 2026, using the tooling in this
repo. `spark-cluster` (run as `uv run spark-cluster ...` from the repo root on the control machine)
does everything that touches the cluster. Do not SSH in to start, stop or edit things yourself,
except to read files of a run (below).

Read `docs/model-evaluations.md` first: it explains the suites, the baseline table and how to read
results. The suite itself (prompts, hidden tests, answer keys) is private, on the head at
`cluster.yml`'s `evals_path` (`/mnt/shared/evals`). Never copy any of it into this repo or into a
commit message: the repo is public.

## 0. Is there a suite?

If `spark-cluster eval run` reports no suite, or a suite that isn't verified ("not ready to grade"),
run the `/setup-evals` skill first. It builds and verifies the private suite.

## 1. Confirm before touching the cluster

Loading another model restarts the cluster: the `spark` endpoint (Open WebUI, anything else using
it) is offline for about 10 minutes per switch, twice in total (to the candidate and back). Tell the
user that, note the current model (`uv run spark-cluster status`, or `MODEL_NAME` in
`docker/cluster.yml`), and wait for a go-ahead.

If the requested model is the one already loaded, skip steps 2–3 and 8: evaluate it as it is.

## 2. Prepare the registry entry

1. Research the model: the model card on Hugging Face (size, quantized builds that fit two GB10
   nodes with 121 GB unified memory each, recommended `vllm serve` flags, reasoning and tool-call
   parsers, speculative decoding), the minimum vLLM version, and known DGX Spark / GB10 issues.
   Say plainly if it can't fit or needs a different vLLM image.
2. Draft an entry in `docker/config/model_registry.yml` by copying the closest existing entry.
   Keep `served-model-name: [<registry-key>, spark]`, set `gpu-memory-utilization` explicitly
   (0.55–0.85; the vLLM default is too high on unified memory), and make
   `tensor-parallel-size` × `pipeline-parallel-size` equal the node count.
3. Check it: `MODEL_NAME=<key> python docker/config/launch.py --dry-run`.
4. Show the entry to the user. After they agree: commit (plain message, **no Claude attribution
   or Co-Authored-By lines** in this project), push, then `uv run spark-cluster pull`.
5. Weights: if they aren't on disk, `spark-cluster model` refuses and says so. Then run
   `uv run spark-cluster download <key>` (on the head only; one copy over NFS). Large models
   take a while: tell the user the size first.
6. If the model needs another `VLLM_IMAGE`, ask before changing it in `docker/cluster.yml`. It
   applies to every model, so note that it must be changed back afterwards.

## 3. Load it

`uv run spark-cluster model <key>`. It runs the checks, restarts in rank order and follows progress
until the API answers (about 10 minutes). If it fails:
- Read the cause with `uv run spark-cluster logs` (head) or `uv run spark-cluster logs spark-02`.
- Fix it with a registry change, explain the change to the user, commit/push/pull, and try again.
  At most 3 attempts; then stop, restore the previous model (step 8) and report what failed.

## 4. Run the suite

```
uv run spark-cluster eval run --tag <key>_default
```
This starts a detached run on the head and follows it live. It runs speed, tool calling,
challenge 1, challenge 2 (twice, seeds 1 and 2), challenge 3 (the fix loop) and the vision/audio
questions. It takes 20–60 minutes, mostly the model thinking. If you lose the watch, reattach with
`uv run spark-cluster eval watch`; `eval status` lists runs; `eval stop` ends one. While a run is
going, `up/down/restart/model` refuse to run: don't work around that.

For long runs, start with `--no-watch` and follow with a background `eval watch` so you can keep
talking to the user.

## 5. Try other settings (only if informative, at most 3 variants)

Each variant is its own tagged run; usually only the code suites are worth repeating:
```
uv run spark-cluster eval run challenge1 challenge2 --tag <key>_low --effort low
uv run spark-cluster eval run challenge2 --tag <key>_nothink --ctk '{"enable_thinking": false}'
```
Effort names differ per model (Qwen `low|medium|xhigh`, DeepSeek `low|high|max`, gpt-oss
`low|medium|high`; Gemma uses `--ctk '{"enable_thinking": ...}'`). Worth trying when: the default
run hit `finish=length` (ran out of tokens while thinking), when it was very slow, or when the
model card recommends a setting. Speculative-decoding changes (draft tokens) are registry changes
and need a restart: only try them if speed is the open question, and ask first.

## 6. Grade the vision and audio answers

The runner records answers but does not grade them. Read them and the answer key:
```
ssh <head> cat /mnt/shared/evals/runs/<run-id>/vision_answers.json /mnt/shared/evals/suites/vision/expected.md
```
(`<head>` is the first node's `ssh` value in `docker/cluster.yml`.) Count an answer as correct only
if it gets the specific facts right; "partly" counts as wrong. An HTTP 400 about audio or video
means the model has no such input: count it as not supported and say so. Record the grade:
```
uv run spark-cluster eval grade <run-id> vision 3/4 --by agent --note "video: said right to left"
```

## 7. Report

1. `uv run spark-cluster eval report` shows the new runs next to earlier ones (the model
   `cluster.yml` serves is bold).
2. Give the user a short verdict against the current default: **replace**, **keep as an option**
   (and for what), or **drop**, with the numbers that decide it. What separates models:
   challenge 2 missed fits (lower is better; one run is one sample, so compare both seeds), time to
   answer, speed, vision, tool calling. A higher effort is not automatically better.
3. With the user's agreement: add the model's row to the baseline table in
   `docs/model-evaluations.md` and to "Which model to run" in `docker/README.md`, and a short
   comment with the decisive numbers on its registry entry. If it is dropped, remove the registry
   entry; ask before deleting weights from `/mnt/shared/huggingface`. Commit as in step 2.

## 8. Restore

Unless the user chose the new model as the default: `uv run spark-cluster model <previous-key>`
(and revert `VLLM_IMAGE` if you changed it). Tell the user when the cluster is back.

## Rules

- Tell the user every time the cluster goes down and when it is back.
- Never edit a node's `docker/node/.env` by hand; change `docker/cluster.yml` and let
  `spark-cluster` sync it.
- Never `pkill -f` anything; use `spark-cluster eval stop` for runs.
- Never put suite content (prompts, tests, reference solutions, expected answers) in this repo,
  in commits or in public places.
- Commit messages in this project carry no Claude attribution lines.
