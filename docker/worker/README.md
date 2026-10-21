# Worker node (spark-02, rank 1)

Runs `vllm-worker`: headless vLLM holding tensor-parallel shard 1. It has no API; it connects
to the head's rendezvous at `192.168.100.1:29501`, so start the head **first**.

```bash
docker compose up -d
docker logs -f --tail=20 vllm-worker
```

- Model: `MODEL_NAME` in `.env` must match the head's (compose refuses to start without it).
- Multi-node settings: `CLUSTER_*` in `docker-compose.yml` (rank 1, master 192.168.100.1;
  override with `CLUSTER_MASTER_ADDR` in `.env`).
- Weights and tokenizer files come from spark-01 over NFS/RDMA (`/mnt/shared`).
- Compile cache: `/mnt/shared/vllm-compile-cache/worker`.
- The "NCCL_NET_PLUGIN variable is not set" compose warning is harmless.

Architecture, `.env` settings, NCCL tuning and troubleshooting: see [../README.md](../README.md).
