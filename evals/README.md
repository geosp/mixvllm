# Evaluation task specs (public)

The tasks `spark-cluster eval` runs against the loaded model: prompts, settings and media, plus a
`SPEC.md` per code challenge that says how answers must be graded. **No answer keys live here.**
The hidden tests, reference solutions, mutants, stress cases and expected answers are generated
into the private suite folder on the head (`cluster.yml`'s `evals_path`) and must never be
committed: this repo is public, and models trained on public code could learn the answers.

| Suite | Kind | What it measures |
|---|---|---|
| `speed` | speed | prose and code tok/s, short-answer time, draft-token acceptance |
| `tools` | tools | a structured tool call |
| `challenge1` | code | exact arithmetic expression evaluator |
| `challenge2` | code | smallest shipping box (3D packing), with a stress test |
| `challenge3` | fix_loop | the challenge 2 task with the model's own tests and up to 3 fix rounds |
| `vision` | media | chart, video and audio questions |

## Setting up a suite

```bash
uv run spark-cluster eval init      # create evals_path on the head from these specs (never overwrites)
```
Then run `/setup-evals` in Claude Code. It writes each code challenge's answer key and checks it
with `uv run spark-cluster eval verify`. A suite only grades models once its reference scores
100%, every deliberately broken mutant is caught, and (for the stress test) an independent solver
agrees with the reference. Finish with a baseline run of your current model.

Results carry the suite's fingerprint: scores from different suites (yours, someone else's, or an
older version of yours) aren't comparable, and `eval report` tags them apart.

Generating an answer key sends the prompts and the generated answers through Claude's API.

See [docs/model-evaluations.md](../docs/model-evaluations.md).
