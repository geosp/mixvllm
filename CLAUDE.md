# CLAUDE.md - MixVLLM Project Guide

## Project Overview

MixVLLM serves large language models with **vLLM** through an OpenAI-compatible API. Models are
defined once in a registry and launched the same way everywhere. The main deployment is a 2-node
DGX Spark cluster that runs one model with tensor parallelism across both nodes over RDMA/RoCE,
using vLLM's built-in multi-node launcher (no Ray). The cluster is defined in one tracked file,
`docker/cluster.yml`, and run from the control machine with the `spark-cluster` command.

## Repository Structure

```
docker/
  cluster.yml                 # THE cluster: nodes (order = rank), MODEL_NAME, VLLM_IMAGE, NCCL_*
  node/docker-compose.yml     # one compose file for every node; .env generated, secrets.env by hand
  config/model_registry.yml   # one entry per model; keys become `vllm serve` flags (default qwen3.8-flash-next)
  config/launch.py            # builds `vllm serve` from the registry + CLUSTER_* env (standalone, PyYAML only)
  stand_alone/                # single machine (no RDMA, publishes :8000)
  open-webui/                 # Open WebUI + Open Terminal
spark_cluster/                # `spark-cluster` CLI (uv project at the repo root)
  cli.py                      # commands, lock, up/down, readiness summary
  config.py                   # loads cluster.yml, renders each node's .env, loads launch.py by path
  remote.py                   # SSH with retries, parallel across nodes
  checks.py, node_probe.py    # `check`; node_probe.py runs on the nodes with stdlib python3
  progress.py                 # container state + startup stage from `docker logs`, rich tables
tests/                        # pytest, no SSH needed (fixtures in tests/data/)
docs/                         # maintenance notes, startups.csv (one row per `up`)
multi_node_gpu_cluster_with_rdma/setup.md   # RoCE host setup
```

## Tech Stack

- **Cluster image:** `vllm/vllm-openai:v0.32.0` (upstream; entrypoint `/usr/bin/env`), set once as
  `VLLM_IMAGE` in `docker/cluster.yml`
- **spark-cluster:** Python ≥ 3.11, PyYAML and `rich` only, managed with **uv** (`uv.lock` is committed)
- **NCCL** over **RDMA/RoCE** between nodes; model folder shared over NFS/RDMA from spark-01
- **Docker Compose** on each node

## Common Commands

Run on the control machine, anywhere in the repo (or `uv tool install --editable .` once):

```bash
uv run spark-cluster status            # per node: state, image, commit, startup stage; /health
uv run spark-cluster check             # commits, registry, GPU split, model files, memory, host setup
uv run spark-cluster sync --dry-run    # diff of each node's rendered docker/node/.env
uv run spark-cluster up | down | restart
uv run spark-cluster model <name>      # edit MODEL_NAME in cluster.yml and restart (offers rollback)
uv run spark-cluster download <name>   # into the shared HF folder, on the head only
uv run spark-cluster pull              # git pull --ff-only on every node
uv run spark-cluster logs [node]
uv run pytest
MODEL_NAME=gpt-oss-120b python docker/config/launch.py --dry-run   # the vllm serve command
```

## Key Architecture Notes

- Never edit a node's `docker/node/.env`: it's generated from `cluster.yml` by `spark-cluster sync`.
  `secrets.env` (HF token) is per node, by hand, gitignored.
- Each node runs `python3 /config/launch.py` with `CLUSTER_NNODES`, `CLUSTER_NODE_RANK`,
  `CLUSTER_MASTER_ADDR` (first node's mesh IP), `CLUSTER_MASTER_PORT` → `--nnodes N --node-rank R`
  (rank > 0 `--headless`), rendezvous on 192.168.100.1:29501. Rank 0 starts first.
- The registry is read from each node's checkout, so the nodes must be on the same commit as the
  control machine (`check` enforces it). Changes to `docker/config` must be committed, pushed and
  pulled before `up`.
- A model's `tensor-parallel-size` × `pipeline-parallel-size` must equal the node count.
- Default model `qwen3.8-flash-next` (MTP 2 drafts, 0.70 memory); `qwen3.8-27b` when fast
  restarts or spare memory matter. The lineup is in `docker/README.md` ("Which model to run").
- Startup takes ~11 min for Flash-Next (two weight-loading passes, ~7 min); `status` shows the
  stage and shard progress, `up` prints a readiness summary and appends it to `docs/startups.csv`.
- Changing commands take `/tmp/spark-cluster.lock` on the head (`--force-unlock` for a dead run).
- Before starting, `up` drops the page cache and compacts memory on every node through a privileged
  container (no sudo needed); page cache counts against vLLM's free-memory check on unified memory.
- API: OpenAI-compatible on http://10.0.0.96:8000; every model is also served as `spark`.

## Hardware Context

- Nodes: NVIDIA DGX Spark — GB10 Grace Blackwell, 121 GB unified memory, DGX OS 7.6, kernel
  6.17.0-1032, driver 595.99.02 (see `docs/cluster-update-10-20-2026.md`)
- RDMA mesh on 192.168.100.x (`roceP2p1s0f0` / `enP2p1s0f0np0`), management on 10.0.0.x
- spark-01 exports `/mnt/shared` (local NVMe) over NFS/RDMA; spark-02 mounts it with
  `proto=rdma` and 16 MB read-ahead (`nfs-shared-readahead.service`)
- sudo on the nodes asks for a password: give the user ready-to-paste commands for sudo steps
