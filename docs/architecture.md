# Architecture

How the mixvllm cluster is put together: the machines and networks, how one configuration file
becomes a running `vllm serve` on every node, what happens during a start, and how the
`spark-cluster` command is built. For day-to-day commands see
[docker/README.md](../docker/README.md); for the RoCE host setup see
[multi_node_gpu_cluster_with_rdma/setup.md](../multi_node_gpu_cluster_with_rdma/setup.md).

## 1. System overview

Two DGX Spark nodes (GB10 Grace Blackwell, one GPU and 121 GB of unified memory each) serve one
model together with tensor parallelism across the nodes. A control machine (any Linux box with
`uv` and SSH access) drives them. Each node runs `vllm serve` with its own rank, and vLLM's
built-in multi-process launcher coordinates them.

```mermaid
flowchart LR
    CL["Clients<br/>Open WebUI, curl, SDKs"]
    SC["Control machine<br/>spark-cluster + docker/cluster.yml"]

    subgraph CLUSTER["DGX Spark nodes · RDMA mesh 192.168.100.x"]
        direction TB
        subgraph S1["spark-01 · rank 0 · 10.0.0.96"]
            V1["vllm-server<br/>API :8000 + TP shard 0"]
            NVME[("/mnt/shared<br/>local NVMe")]
        end
        subgraph S2["spark-02 · rank 1 · 10.0.0.22"]
            V2["vllm-worker<br/>headless, TP shard 1"]
            MNT[("/mnt/shared<br/>NFS mount")]
        end
        V1 <-- "NCCL over RoCE v2" --> V2
        NVME -- "NFS over RDMA" --> MNT
    end

    SC -- "SSH + HTTP" --> CLUSTER
    CL -- "OpenAI API<br/>http://10.0.0.96:8000/v1" --> V1
```

| | spark-01 | spark-02 |
|---|---|---|
| Rank | 0 (head) | 1 |
| Management IP | 10.0.0.96 | 10.0.0.22 |
| Mesh (RDMA) IP | 192.168.100.1 | 192.168.100.2 |
| Container | `vllm-server` | `vllm-worker` |
| Serves the API | yes, `:8000` | no (`--headless`) |
| `/mnt/shared` | local NVMe, exported over NFS/RDMA | NFS/RDMA mount, 16 MB read-ahead |

### Three kinds of traffic

| Traffic | Network | Path |
|---|---|---|
| API requests | management (10.0.0.x) | clients → spark-01:8000 |
| Tensors between GPUs | RDMA mesh (192.168.100.x) | NCCL over RoCE v2 (`roceP2p1s0f0`), about 111 Gb/s; bootstrap over `enP2p1s0f0np0` |
| Model weights, caches | RDMA mesh | NFS v4.1 over RDMA from spark-01's `/mnt/shared` |
| Control | management | SSH and HTTP from the control machine |

The model folder lives once, on spark-01's NVMe, and spark-02 reads it over NFS/RDMA. With
read-ahead raised to 16 MB, spark-02 loads weights about as fast as spark-01 does locally. One copy
per model also means only one node may download a model at a time (`spark-cluster download` runs
on the head only).

## 2. Configuration: from one file to a running process

Two tracked files describe everything: `docker/cluster.yml` (the machines and every shared
setting) and `docker/config/model_registry.yml` (one entry per model). Nothing on a node is edited
by hand except the token file.

```mermaid
flowchart TB
    subgraph REPO["Tracked in git"]
        CY["docker/cluster.yml<br/>nodes, MODEL_NAME, VLLM_IMAGE,<br/>HF_HOME, NCCL_*"]
        REG["docker/config/model_registry.yml<br/>one entry per model"]
        LP["docker/config/launch.py"]
        DC["docker/node/docker-compose.yml"]
    end

    subgraph CTRL["Control machine"]
        SYNC["spark-cluster sync<br/>(config.py: render_env)"]
    end

    subgraph NODE["Each node: repo_path/docker/node/"]
        ENV[".env<br/>GENERATED"]
        SEC["secrets.env<br/>by hand, chmod 600"]
        COMPOSE["docker compose up -d"]
    end

    subgraph CONT["Container"]
        ENVV["environment"]
        LAUNCH["python3 /config/launch.py"]
        SERVE["vllm serve MODEL ...flags<br/>--nnodes N --node-rank R ..."]
    end

    CY --> SYNC
    SYNC -- "SSH, only if changed" --> ENV
    ENV --> COMPOSE
    SEC --> COMPOSE
    DC --> COMPOSE
    COMPOSE --> ENVV
    ENVV -- "MODEL_NAME, CLUSTER_*" --> LAUNCH
    REG -- "mounted read-only at /config" --> LAUNCH
    LP -- "mounted read-only at /config" --> LAUNCH
    LAUNCH --> SERVE
```

### What goes into a node's `.env`

`config.node_env()` builds it in this order, later entries winning:

1. **Derived per node** (`cluster.yml` must not set these): `NODE_NAME`, `CONTAINER_NAME`
   (`vllm-server` on rank 0, `vllm-worker` elsewhere), `VLLM_HOST_IP` (the node's `mesh_ip`),
   `CLUSTER_NNODES`, `CLUSTER_NODE_RANK` (list position), `CLUSTER_MASTER_ADDR` (the first node's
   `mesh_ip`), `CLUSTER_MASTER_PORT`.
2. **Shared** `env:` from `cluster.yml`: `MODEL_NAME`, `VLLM_IMAGE`, `VLLM_ENTRYPOINT`, host paths,
   all `NCCL_*`.
3. **The node's own** `env:`, e.g. `NCCL_NET_PLUGIN: none` on spark-01.

The `.env` serves two purposes in compose: it supplies `${...}` interpolation in the compose file
(image, entrypoint, container name, volume paths) and, through `env_file`, the container's
environment. The compose file's own `environment:` block adds fixed values (`HF_HOME` inside the
container, `CUDA_VISIBLE_DEVICES=0`, `GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}`) and repeats the
`CLUSTER_*` values with `:?` guards, so a node without a synced `.env` refuses to start.

### From `MODEL_NAME` to `vllm serve`

`launch.py` looks up `MODEL_NAME` in the registry and turns every key except `description` into a
flag (`true` → bare flag, `false` → omitted, list → one flag with several values; `model` is
positional). With `CLUSTER_NNODES > 1` it appends:

```
--distributed-executor-backend mp --nnodes N --node-rank R
--master-addr <CLUSTER_MASTER_ADDR> --master-port <CLUSTER_MASTER_PORT>
[--headless]   # every rank but 0
```

`launch.py` stays a standalone file because it runs inside the vLLM container with only PyYAML.
`spark-cluster` loads it by path (`importlib`) and reuses `load_registry` and `get_model_config`,
so the registry is parsed one way everywhere. Because each node reads the registry from its own
checkout, all nodes must be on the same commit; `spark-cluster check` enforces it.

### Inside a node container

```mermaid
graph LR
    subgraph HOST["Host"]
        CFG["repo/docker/config"]
        HF["/mnt/shared/huggingface"]
        HAR["/mnt/shared/harmony/encodings"]
        CC["/mnt/shared/vllm-compile-cache/NODE_NAME"]
        IB["/dev/infiniband<br/>/sys/class/infiniband, /sys/class/net, /etc/rdma"]
    end
    subgraph C["Container (network_mode host, ipc host)"]
        C1["/config (ro)"]
        C2["/root/.cache/huggingface"]
        C3["/etc/encodings (ro)<br/>TIKTOKEN_ENCODINGS_BASE"]
        C4["/root/.cache/vllm<br/>torch.compile + FlashInfer autotune"]
        C5["RDMA verbs devices"]
    end
    CFG --> C1
    HF --> C2
    HAR --> C3
    CC --> C4
    IB --> C5
```

The container runs with host networking (the API binds :8000 directly on spark-01; NCCL and the
rendezvous use the mesh interfaces), `memlock: -1`, capabilities `IPC_LOCK`, `SYS_NICE`,
`SYS_RESOURCE`, and `apparmor:unconfined`. The compile cache is per node so the two nodes never
write the same files over NFS.

## 3. Starting the cluster (`spark-cluster up`)

```mermaid
sequenceDiagram
    autonumber
    participant U as spark-cluster (control)
    participant H as spark-01 (rank 0)
    participant W as spark-02 (rank 1)
    participant API as API :8000

    U->>H: take lock /tmp/spark-cluster.lock
    U->>H: check (probe over SSH)
    U->>W: check (probe over SSH)
    U->>H: sync .env (if changed)
    U->>W: sync .env (if changed)
    par pull image
        U->>H: docker pull VLLM_IMAGE
    and
        U->>W: docker pull VLLM_IMAGE
    end
    par free memory
        U->>H: drop page cache + compact (privileged container)
    and
        U->>W: drop page cache + compact
    end
    U->>H: docker compose up -d
    U->>W: docker compose up -d
    W->>H: rendezvous 192.168.100.1:29501
    Note over H,W: NCCL init over RoCE (NET/IB)
    Note over H,W: load weights (1–2 passes), warmup, CUDA graph capture
    H->>API: Application startup complete
    loop every 5 s until /health answers (25 min limit)
        U->>H: container state + filtered log
        U->>W: container state + filtered log
        U->>API: GET /health
    end
    U->>API: streamed test request (time to first token)
    U->>U: readiness summary → docs/startups.csv
    U->>H: release lock
```

Notes on the steps:

- **Check** refuses to start on any failure: nodes on a different commit than the control
  machine, local changes in `docker/config` or `docker/node`, a model missing from the registry,
  `tensor-parallel-size × pipeline-parallel-size` not equal to the node count, model files
  incomplete, no RDMA device, NFS not over RDMA or read-ahead below 16 MB, or `secrets.env`
  missing. The memory estimate (checkpoint ÷ GPUs vs `gpu-memory-utilization × 121 GiB`) only
  warns.
- **Free memory.** vLLM compares free memory with `gpu-memory-utilization` when the engine starts,
  and on unified memory the page cache counts as used: after a `down`, each node typically shows
  60–65 GiB free until the cache is dropped, then about 119 GiB. `/proc/sys/vm` is not namespaced,
  so a privileged container (the vLLM image, already on every node) writes the host's
  `drop_caches` and `compact_memory`; the `docker` group already allows this, so no sudo is
  needed.
- **Order.** Rank 0 starts first because it hosts the rendezvous; the others connect to it.
  `down` stops in reverse order.
- **Fail fast.** The watch stops with the node's last error line if a container exits, restarts
  or disappears. Ctrl+C stops watching but leaves the cluster starting.

### Startup stages (as `status` reports them)

`progress.parse_log()` reads `docker logs --timestamps --since <container start>`, filtered on the
node to the marker lines below, and moves each node forward through these stages:

```mermaid
stateDiagram-v2
    [*] --> starting: container running
    starting --> loading: "Loading model from scratch"
    state loading {
        [*] --> pass
        pass --> pass: "Initializing model took" (next pass)<br/>shard x/N (rank 0 only)
    }
    loading --> warmup: "Model loading took … seconds"
    state warmup {
        [*] --> kernels
        kernels --> graphs: "Capturing CUDA graphs"
        graphs --> kernels: kernel warmup / autotune
    }
    warmup --> engine_ready: "Free memory on device" / "init engine"
    engine_ready --> ready: rank 0 · "Application startup complete"
    engine_ready --> ready: other ranks (headless)
    ready --> [*]
```

Measured on `qwen3.8-flash-next` (124 GiB checkpoint, read in two passes): about 6.5 minutes of
loading, 3.5 minutes of warmup and graph capture, about 11 minutes in total. `gpt-oss-120b`
(61 GiB, one pass) is ready in a little over 2 minutes. Every `up` appends its timings to
`docs/startups.csv` on the control machine (gitignored: a local history, not shared).

The **readiness summary** derives:

| Field | From |
|---|---|
| model load per node | "Model loading took … seconds" |
| warmup | rank 0's "Application startup complete" − the latest "Model loading took" timestamp |
| total | rank 0's "Application startup complete" − rank 0's container start |
| NCCL | rank 0's `NET/IB : Using …` line (a `NET/Socket` line means TCP fallback and is flagged) |
| first token | one streamed `/v1/chat/completions` request to model `spark` |

## 4. Switching models (`spark-cluster model <name>`)

```mermaid
flowchart TD
    A["model NEW"] --> B{"NEW in registry?"}
    B -- no --> X1["refuse: list known models"]
    B -- yes --> C{"TP × PP = node count?"}
    C -- no --> X2["refuse"]
    C -- yes --> D{"files complete on head?"}
    D -- no --> X3["refuse: run download NEW"]
    D -- yes --> E{"NEW = current?"}
    E -- yes --> X4["nothing to do"]
    E -- no --> F["take lock"]
    F --> G["rewrite only the MODEL_NAME: line<br/>in cluster.yml (comments kept)"]
    G --> H["down, then up"]
    H --> I{"healthy?"}
    I -- yes --> Z["done: summary + startups.csv"]
    I -- "failed" --> J{"switch back? (terminal prompt)"}
    J -- yes --> K["restore OLD in cluster.yml<br/>down, then up"]
    J -- no --> Y["leave cluster.yml at NEW,<br/>cluster not serving"]
```

The change to `cluster.yml` is local to the control machine; commit it if the new model should
stay the default. Every registry entry serves its own name plus the alias `spark`, so clients
that ask for `spark` keep working across switches.

## 5. The `spark-cluster` command

A small uv project at the repo root (`pyproject.toml`, `uv.lock`), depending only on PyYAML and
`rich`. It runs on the control machine; the nodes need only `python3`, git and Docker.

```mermaid
graph TD
    CLI["cli.py<br/>commands, lock, up/down/restart/model,<br/>watch loop, readiness summary"]
    CFG["config.py<br/>load cluster.yml, render .env,<br/>set_model_name, load launch.py by path"]
    REM["remote.py<br/>ssh with retries, parallel()"]
    CHK["checks.py<br/>registry, GPU split, memory, host facts"]
    PRB["node_probe.py<br/>stdlib only, runs ON the node"]
    PRG["progress.py<br/>container state, log stages,<br/>rich tables, API status"]
    LP["docker/config/launch.py"]

    CLI --> CFG
    CLI --> CHK
    CLI --> PRG
    CLI --> REM
    CHK --> REM
    PRG --> REM
    CHK -. "sent over ssh stdin:<br/>python3 - args" .-> PRB
    CFG -. importlib .-> LP
```

Design points:

- **One SSH round trip per node per poll.** `status` runs a single script per node that prints the
  git commit, `docker inspect` state and the log filtered on the node (`grep -E` on
  `progress.LOG_FILTER`), so long-running containers don't ship hours of request logs.
- **Probes need nothing installed.** `node_probe.py` is sent over stdin to the node's `python3`
  and prints JSON: commit, local changes, `secrets.env` presence, RDMA devices, the mount behind
  `HF_HOME` (from `/proc/self/mountinfo`, including `proto=rdma`), its read-ahead
  (`/sys/class/bdi/<dev>/read_ahead_kb`), and on the head whether the model's files are complete
  in the Hugging Face cache.
- **Retries.** SSH errors that look transient (name lookup, refused, timed out, reset) are retried
  up to 4 attempts with 2, 4 and 8 s back-off; the control machine occasionally fails to resolve a
  node's name.
- **One changing run at a time.** `up`, `down`, `restart`, `sync` and `model` create
  `/tmp/spark-cluster.lock` on the head atomically (`set -C`), holding the control machine's name,
  PID and time, and remove it on exit if it is still theirs. A second run refuses and names the
  holder; `--force-unlock` clears a lock left by a dead run.
- **Plain output when piped.** With a terminal, `rich` tables and a live view; without one, one line
  per node, printed only when a node's state or stage changes.
- **Tests without hardware.** `uv run pytest` covers `.env` rendering, the check logic and stage
  parsing; the log fixtures in `tests/data/` are real startups captured with `LOG_FILTER`.

## 6. Failure modes and where they show

| Symptom | Where it shows | Usual cause |
|---|---|---|
| `check`: commit differs | before any start | a node wasn't pulled (`spark-cluster pull`) |
| `check`: read-ahead below 16384 KB | before any start | `nfs-shared-readahead.service` didn't run on spark-02 |
| head stuck after loading, `/health` never answers | `status`, watch timeout | a worker isn't running or can't reach 192.168.100.1:29501 |
| `NET/Socket` instead of `NET/IB` | `status` warning, readiness summary | NCCL fell back to TCP: RDMA device or HCA setting |
| container exits during start | watch stops with the last error line | memory (utilization too high), model or image mismatch |
| "another spark-cluster run holds the lock" | any changing command | a run in progress, or a dead one (`--force-unlock`) |
| `docker compose up` refuses on a node | by hand on the node | `.env` never synced (`spark-cluster sync`) |

## 7. Single machine

[docker/stand_alone/](../docker/stand_alone/) runs the same `launch.py` and registry on one
machine without the cluster pieces: no RDMA devices, no `CLUSTER_*` variables (so plain
`vllm serve`), and port 8000 published instead of host networking. It is outside `spark-cluster`.
