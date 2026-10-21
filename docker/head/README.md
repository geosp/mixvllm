# Head node (spark-01, rank 0)

Runs `vllm-server`: the OpenAI-compatible API on port 8000 plus tensor-parallel shard 0.
It also hosts the multi-node rendezvous on `192.168.100.1:29501`, so start it **before** the
worker.

```bash
docker compose up -d
docker logs -f --tail=20 vllm-server     # wait for "Application startup complete"
curl http://10.0.0.96:8000/health
```

- Model: `MODEL_NAME` in `.env` (must match the worker's), resolved through
  `../config/model_registry.yml` by `../config/launch.py`.
- Multi-node settings: `CLUSTER_*` in `docker-compose.yml` (rank 0, master = `VLLM_HOST_IP`).
- Compile cache: `/mnt/shared/vllm-compile-cache/head`.

Architecture, `.env` settings, NCCL tuning and troubleshooting: see [../README.md](../README.md).
