# mixvllm

Docker setup for serving LLMs with [vLLM](https://docs.vllm.ai/), plus Open WebUI as the chat front end.
Models are defined once in a registry and launched the same way on every deployment.

## Layout

```
docker/
├── config/
│   ├── model_registry.yml   # one entry per model; keys become `vllm serve` flags
│   └── launch.py            # reads MODEL_NAME, builds and runs the `vllm serve` command
├── head/                    # 2-node cluster, rank 0 (serves the API on :8000)
├── worker/                  # 2-node cluster, rank 1 (headless)
├── stand_alone/             # single machine
└── open-webui/              # Open WebUI + Open Terminal (sandboxed shell for the model)
multi_node_gpu_cluster_with_rdma/
└── setup.md                 # RoCE / RDMA host setup
```

Every deployment mounts `docker/config` read-only at `/config` and runs `python3 /config/launch.py`.
See [docker/README.md](docker/README.md) for the registry format, per-node `.env` settings, and NCCL tuning.

## Running

**Two-node cluster** (head first, then worker; set `MODEL_NAME` identically in each node's `.env`):

```bash
cd docker/head && docker compose up -d     # on the head node
cd docker/worker && docker compose up -d   # on the worker node
```

**Single machine:**

```bash
cd docker/stand_alone
MODEL_NAME=<registry key> HF_TOKEN=<token> docker compose up -d
```

The registry entry's `tensor-parallel-size` must not exceed the GPUs on that machine.

**Check the command a model would run** (needs only Python with PyYAML):

```bash
MODEL_NAME=gpt-oss-120b python docker/config/launch.py --dry-run
```

## Using the model

The server exposes an OpenAI-compatible API on port 8000. Every model is also served as `spark`,
so clients keep working when the loaded model changes.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "spark", "messages": [{"role": "user", "content": "Hello!"}]}'
```

For a chat UI, start [docker/open-webui/](docker/open-webui/) and add `http://<host>:8000/v1` as an
OpenAI connection. Its Open Terminal container gives the model a shell: connect it under
Admin Settings → Integrations → Open Terminal with URL `http://open-terminal:8000` and the
`OPEN_TERMINAL_API_KEY` from `docker/open-webui/.env`.
