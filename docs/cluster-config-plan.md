# Cluster config file + spark-cluster command

## Context
Today the cluster is two per-role folders (`docker/head/`, `docker/worker/`). Each has a near-identical compose file and a hand-edited `.env` on its own node. Settings that must match (model, image, NCCL tuning) are copied by hand, and they drift: the nodes ended up on different commits, and old `RAY_*` variables are still lying around. Every start or stop means SSHing to each node in the right order. A restart takes about 10 minutes, and the only way to follow it is to read logs on each node.

Goal: one tracked `docker/cluster.yml` that lists the machines and holds every shared setting, plus a `spark-cluster` command on the control machine that pushes per-node settings and starts, stops, checks and reports on the whole cluster. Adding a machine becomes adding an entry.

Both nodes' current `.env` files are identical except for `VLLM_HOST_IP` (the node's mesh IP) and `NCCL_NET_PLUGIN=none`, which is set on the head only. The compose files differ only in:
- rank and master address
- compile-cache folder
- container name
- the worker's `apparmor:unconfined`
- the worker's `NVIDIA_VISIBLE_DEVICES` (interpolated, but `all` on both)

### What the 2026-10-24 restarts showed
- **Startup takes about 10 minutes.** Roughly 7 minutes is reading the 124 GiB of weights in two passes. The head took 418 s from local NVMe; the worker took 437 s over NFS once read-ahead was raised. Warmup and CUDA graph capture add another 2.3 minutes. Progress reporting matters more than raw speed.
- **The model folder is shared over NFS** (`/mnt/shared`, exported by spark-01 over RDMA). One copy per model is the right trade-off. It also means two nodes must never download the same model at once.
- **spark-02 needs 16 MB NFS read-ahead** for the worker to load at near local-disk speed. It's set by the `nfs-shared-readahead.service` systemd unit on spark-02 and isn't recorded in the repo.
- **SSH from the control machine occasionally fails to look up a node's name.** Commands must retry rather than treat that as a node failure.

## Design

### 1. `docker/cluster.yml` (tracked, the single source)
```yaml
repo_path: ~/projects/mixvllm          # checkout on every node
api_url: http://10.0.0.96:8000         # head's API, used for health checks
master_port: 29501

env:                                   # identical on every node
  MODEL_NAME: qwen3.8-flash-next
  VLLM_IMAGE: vllm/vllm-openai:v0.32.0 # one image for all models
  VLLM_ENTRYPOINT: /usr/bin/env
  HF_HOME: /mnt/shared/huggingface
  HARMONY_ENCODINGS: /mnt/shared/harmony/encodings
  TIKTOKEN_ENCODINGS_BASE: /etc/encodings
  NCCL_IB_HCA: roceP2p1s0f0
  # ... the rest of today's NCCL_* values, with their "why" comments moved here

nodes:                                 # list order = rank; the first node is the head
  - name: spark-01
    ssh: spark-01.mixwarecs-home.net   # full name or IP: the short name failed to resolve once
    mesh_ip: 192.168.100.1
    env: {NCCL_NET_PLUGIN: none}       # per-node overrides
  - name: spark-02
    ssh: spark-02.mixwarecs-home.net
    mesh_ip: 192.168.100.2
```
The image stays cluster-wide. If a model needs a different vLLM version, note it in that model's registry comment and change `VLLM_IMAGE` when switching to it.

### 2. `docker/node/docker-compose.yml` (replaces `head/` and `worker/`)
- One compose file for every node. `env_file: [./.env, ./secrets.env]`.
- Interpolated from the generated `.env`:
  - `image: ${VLLM_IMAGE}`
  - `entrypoint: ${VLLM_ENTRYPOINT}`
  - `container_name: ${CONTAINER_NAME}` (`vllm-server` on rank 0, `vllm-worker` elsewhere)
  - compile cache at `/mnt/shared/vllm-compile-cache/${NODE_NAME}`
- The `environment:` block keeps only fixed or derived values:
  - `HF_HOME=/root/.cache/huggingface`, `CUDA_VISIBLE_DEVICES=0`
  - the `CLUSTER_*` values from `.env`
  - `GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}`
- The NCCL variables arrive through `env_file`.
- Devices, volumes, ulimits, caps and GPU reservation stay the same. Apply `apparmor:unconfined` on all nodes.
- `docker/node/.env` is **generated** and gitignored, with a "do not edit" header.
- `docker/node/secrets.env` is hand-maintained per node and gitignored: `HUGGING_FACE_HUB_TOKEN`. Add `docker/*/secrets.env` to `.gitignore`.
- A plain `docker compose up -d` on a node still works by hand.

### 3. `spark-cluster` command (runs on the control machine)
- **A small uv project at the repo root**, run as `uv run spark-cluster <command>` from anywhere inside the repo:
  ```toml
  # pyproject.toml
  [project]
  name = "spark-cluster"
  version = "0.1.0"
  requires-python = ">=3.11"
  dependencies = ["pyyaml>=6", "rich>=13"]

  [project.scripts]
  spark-cluster = "spark_cluster.cli:main"

  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"
  ```
- **Code in `spark_cluster/`:**
  - `cli.py`: commands and argument parsing
  - `config.py`: loads `cluster.yml` and renders each node's `.env`
  - `remote.py`: SSH with retries, run in parallel across nodes
  - `checks.py`: everything `check` verifies
  - `progress.py`: reads each node's log stage and draws the `rich` tables
- **Optional global command:** `uv tool install --editable .` puts `spark-cluster` on the PATH, usable outside the repo, and code edits apply immediately.
- **Lock file:** commit `uv.lock`, so every machine gets the same dependency versions. Remove `uv.lock` and `.python-version` from `.gitignore`; they're leftovers from the old Python projects.
- **Dependencies stay light:** PyYAML and `rich` only, unlike the removed `mixvllm_server`, which pulled in vLLM and torch.
- **`rich` for output:**
  - `status` prints a table: one row per node with state, image, commit, stage and elapsed time.
  - `up` shows the same table as a live display, refreshed every few seconds. The stage column includes shard progress (e.g. `loading weights · pass 1 · 6/11`).
  - The readiness summary is a final table.
  - Errors and warnings print as coloured panels.
  - If the output isn't a terminal (e.g. piped to a file), it falls back to plain lines.

| Command | Behaviour |
|---|---|
| `status` | Per node: container state, image, restart count, git commit, clean or dirty. **Current stage with elapsed time**, read from each node's log: loading weights (pass N, shard x/11), warmup/graph capture, ready. Plus `/health` and `/v1/models`. |
| `check` | Every node is on the same commit as local `HEAD`, with no local changes in `docker/config` or `docker/node`. `MODEL_NAME` exists in the registry, and its `tensor-parallel-size` × `pipeline-parallel-size` equals the node count. The model's files are complete in the shared folder. A rough memory check passes (below). The host setup checks pass (below). `secrets.env` exists on every node. |
| `sync` | Render each node's `.env` and write it over SSH, only if it changed. `--dry-run` prints the diff instead. |
| `download <model>` | Download a registry model into the shared Hugging Face folder **on the head only** (one copy over NFS, so the nodes can't race each other). |
| `up` | `check` → `sync` → `docker pull` the image on all nodes in parallel → head `up -d`, then the other nodes in rank order. Then follow progress (as in `status`) until `/health` answers, with a 25-minute limit. Fail fast if a container exits or restarts. Ctrl+C stops watching but leaves the cluster running. |
| `down` | Reverse rank order. |
| `restart` | `down` then `up`. |
| `logs [node]` | `ssh -t … docker logs -f --tail=50` (defaults to the head). |
| `pull` | `git pull --ff-only` on every node; refuses if a node has local changes. |
| `model <name>` | Check the name against the registry and that its files are complete (otherwise suggest `download`). Rewrite only the `MODEL_NAME:` line in `cluster.yml` (keeps comments), then `restart`. If the new model doesn't come up healthy, offer to switch back to the previous model and restart, rather than leaving the cluster down. |

**Readiness summary,** printed when `up` finishes:
- per-node model load time and total
- warmup time
- total time from the head's start to "Application startup complete"
- the NCCL transport line (`NET/IB … RoCE`, or a warning if it fell back to TCP)
- one test request with time to first token

Each summary is also appended as one row to **`docs/startups.csv`** (date, model, commit, image, per-node load times, warmup, total, NCCL transport, time to first token). Comparing restarts, e.g. before and after a tuning change, then comes from data rather than memory.

**Host setup checks,** run by `check` on every node over SSH:
- the RDMA device (`/dev/infiniband`) is present
- on nodes reading the model folder over NFS: `/mnt/shared` is mounted with `proto=rdma`, and read-ahead on that mount is ≥ 16384 KB

**Rough memory check:** the model's checkpoint size ÷ (`tensor-parallel-size` × `pipeline-parallel-size`) must fit within `gpu-memory-utilization` × 121 GB per node. This only warns, but it catches an obviously wrong setting in seconds rather than after a 7-minute load.

Per-node values `spark-cluster` writes into `.env`:
- `NODE_NAME`, `CONTAINER_NAME`
- `VLLM_HOST_IP` (the node's `mesh_ip`)
- `CLUSTER_NNODES`, `CLUSTER_NODE_RANK`
- `CLUSTER_MASTER_ADDR` (the first node's `mesh_ip`), `CLUSTER_MASTER_PORT`
- the shared `env`, then the node's own `env` (node values win)

`spark_cluster` reuses the registry-loading logic from `docker/config/launch.py` (`load_registry`, `get_model_config`) instead of duplicating it. It loads that file by path with `importlib`, because `launch.py` must stay a standalone file: it runs inside the vLLM container, where it needs only PyYAML. Every SSH call retries briefly on connection and name-lookup errors.

**One run at a time:** commands that change the cluster (`up`, `down`, `restart`, `sync`, `model`) take a lock file on the head (`/tmp/spark-cluster.lock`, holding the control machine's name and process ID, removed on exit). A second run refuses with a message saying who holds the lock; a stale lock whose process is gone can be cleared with `--force-unlock`.

**Tests:** `tests/` with pytest (a dev dependency), run with `uv run pytest`. They cover the parts that don't need SSH:
- rendering each node's `.env` from a sample `cluster.yml`, including per-node overrides and derived `CLUSTER_*` values
- `check` logic for the registry, GPU split and memory estimate, using sample registry entries
- stage parsing from saved log lines: shard progress, "Loading weights took", graph capture, "Application startup complete", and the `NET/IB` transport line

### 4. Docs and cleanup
- Delete `docker/head/` and `docker/worker/`, including their READMEs and `.env.example` files.
- Add `docker/node/secrets.env.example`.
- Update `docker/README.md`:
  - the `cluster.yml` format and the `spark-cluster` commands
  - "adding a node": the registry's `tensor-parallel-size` must match the node count
  - a **host setup** section: NFS export over RDMA from spark-01, the RDMA mount on the other nodes, and the read-ahead systemd unit (with its contents)
- Update the root `README.md` layout and run steps (`uv run spark-cluster …`).
- `.gitignore`: stop ignoring `uv.lock` and `.python-version`.
- Drop the unused `RAY_*` variables.
- **Bring spark-01's untracked notes into the repo.** `docker/README.md` already links to `docs/cluster-update-10-20-2026.md`, which exists only on spark-01, so the link is broken elsewhere. Commit it and `docs/cluster-update-26.03-runbook.md`. Leave out `docs/beegfs-migration.log` (536 KB): archive it outside the repo or delete it.
- **`CLAUDE.md`:** take spark-01's untracked copy, update it for the new layout (`cluster.yml`, `docker/node/`, `spark-cluster`, `vllm/vllm-openai:v0.32.0`; no `mixvllm_server`/`mixvllm-chat`), and commit it at the repo root.
- Before spark-01 next pulls, move its untracked copies of these three files aside. Git refuses to overwrite untracked files that a pull would add at the same paths.
- `stand_alone/` stays as it is: it has no RDMA devices and publishes port 8000, so it's out of scope.

## Build order
0. Bring spark-01's notes and an updated `CLAUDE.md` into the repo (independent of the rest; fixes the broken README link now).
1. **`pyproject.toml`, the `spark_cluster/` skeleton and `status`**, including stage reporting. It's read-only and works against today's head/worker containers, so it's useful immediately.
2. `cluster.yml`, `docker/node/docker-compose.yml`, `sync --dry-run`, `check`, with the unit tests.
3. Migration (below).
4. `up` (with the readiness summary and `docs/startups.csv`), `down`, `restart`, `logs`, `pull`, `download`, `model` (with rollback), and the lock.

## Migration (live cluster, done once)
1. Record each running container's env: `docker inspect -f '{{.Config.Env}}'` on both nodes.
2. On each node, create `docker/node/secrets.env` from the token line in the old `.env` (`chmod 600`). This is a server-side copy; the token is never printed.
3. Stop the old projects: `docker compose down` in `worker/`, then `head/`. The new compose project name `node` would otherwise clash on container names.
4. Commit and push; `spark-cluster pull`; then `spark-cluster up`.
5. After it's healthy, delete the old `docker/head/.env` and `docker/worker/.env` on the nodes. They're ignored, untracked leftovers holding the token.

The compile-cache folders move from `head`/`worker` to `spark-01`/`spark-02`. The first start after migration recompiles from scratch and is noticeably slower; later starts use the new cache. The old folders can be deleted afterwards (root-owned, needs sudo).

## Later (not in this round)
- `bench`: a fixed short benchmark (time to first token, tokens per second at 1 and 8 parallel requests) appended to `docs/benchmarks.csv` with the model, commit and image.
- `show <model>`: the registry entry and the resolved `vllm serve` command for the current node count.
- A per-model image override, if models ever need different vLLM versions often.

## Verification
- `uv run pytest` passes.
- `spark-cluster status` against the current head/worker containers reports the right state, stage and timings.
- `spark-cluster sync --dry-run`: for each node, the rendered `.env` matches today's effective values, apart from the dropped `RAY_*` variables.
- `spark-cluster check` passes on the live cluster, and fails clearly for:
  - an invalid `MODEL_NAME` in a scratch copy of `cluster.yml`
  - a registry model whose files aren't downloaded
  - read-ahead temporarily set back to 128 on spark-02, then restored
- After `spark-cluster up`:
  - compare each container's `docker inspect` env with migration step 1 (expected differences: the `RAY_*` variables, the compile-cache path, the env order)
  - `/v1/models` lists `qwen3.8-flash-next` and `spark`
  - the readiness summary shows `NET/IB … RoCE` and a successful test request
- A `spark-cluster down` and `spark-cluster up` round trip, which adds a row to `docs/startups.csv`.
- A second `spark-cluster restart` started while one is running refuses because of the lock.
