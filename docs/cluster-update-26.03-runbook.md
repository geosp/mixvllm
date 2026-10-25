# DGX Spark Cluster Update Runbook — 26.02 → 26.03

> Historical record: written for the old `docker/head` + `docker/worker` layout. The cluster now runs from [`docker/cluster.yml`](../docker/cluster.yml) with `spark-cluster` (see [docker/README.md](../docker/README.md)).

**Date written:** 2026-10-20  
**Executed from:** your laptop (not spark-01 or spark-02 — both nodes will reboot)  
**Estimated time:** ~45–60 min including reboots and model reload

---

## Cluster at a Glance

| | spark-01 (HEAD) | spark-02 (WORKER) |
|---|---|---|
| **Management IP** | 10.0.0.96 | 10.0.0.22 |
| **RDMA mesh IP** | 192.168.100.1 | 192.168.100.2 |
| **SSH** | `ssh geo@10.0.0.96` | `ssh geo@10.0.0.22` |
| **GPU** | NVIDIA GB10 | NVIDIA GB10 |
| **RAM** | 121 GB unified | 121 GB unified |
| **NVMe** | 3.7 TB (root + all data) | 931 GB (root only) |
| **Role** | Ray head + vLLM API + NFS server | Ray worker + NFS client |

**RDMA interface (both nodes):** `enP2p1s0f0np0` / `roceP2p1s0f0` — MTU 9000  
**NFS exports from spark-01:** `/mnt/shared` and `/mnt/nfs-rdma-test` → spark-02 via `proto=rdma,port=20049`  
**All model weights live on spark-01** at `/mnt/shared/huggingface/`

---

## What This Update Does

| Component | Before | After |
|-----------|--------|-------|
| DGX Spark OS release | 26.02.1 | 26.03.1 |
| Docker CE | 29.1.3 | 29.2.1 |
| Docker Compose plugin | 5.0.1 | 5.0.2 |
| nvidia-nvme-options | 25.12-1 | 26.02-1 |
| nv-common-apis | 26.01-1 | 26.03-2 |
| netplan.io | 1.1.2-8ubuntu1~24.04.1 | 1.1.2-8ubuntu1~24.04.2 |
| vLLM container image | 26.02-py3 | 26.03-py3 (if available in registry) |
| Kernel | 6.17.0-1014-nvidia | unchanged (already latest) |
| NVIDIA driver | 590.48.01 | unchanged unless OTA bumps it |

**Why `dgx-spark-ota-update-meta` is the right update mechanism:**  
NVIDIA packages the kernel modules, drivers, nvme options, nvidia-fs (GPU Direct Storage), and DGX system services as a tested bundle. Running a bare `apt upgrade` without the OTA meta-package can leave components at mismatched versions. The OTA package pulls everything to a consistent, validated 26.03 state.

---

## Pre-flight Checks (from your laptop)

```bash
# Confirm both nodes are reachable
ping -c 2 10.0.0.96    # spark-01
ping -c 2 10.0.0.22    # spark-02

# SSH in to confirm access works
ssh geo@10.0.0.96 "hostname && uname -r && nvidia-smi | head -4"
ssh geo@10.0.0.22 "hostname && uname -r && nvidia-smi | head -4"

# Check current container state on spark-01
ssh geo@10.0.0.96 "docker ps --format 'table {{.Names}}\t{{.Status}}'"

# Check current container state on spark-02
ssh geo@10.0.0.22 "docker ps --format 'table {{.Names}}\t{{.Status}}'"

# Check RDMA is up on both
ssh geo@10.0.0.96 "rdma link | grep ACTIVE"
ssh geo@10.0.0.22 "rdma link | grep ACTIVE"
```

Expected: both nodes show `roceP2p1s0f0/1 state ACTIVE physical_state LINK_UP`

---

## Step 1 — Stop All Containers

**Order:** spark-02 first (worker), then spark-01 (head/API).

### spark-02
```bash
ssh geo@10.0.0.22 "
  cd /home/geo/projects/mixvllm/docker/worker && docker compose down
  docker stop nvidia-gpu-exporter node-exporter 2>/dev/null || true
  docker ps
"
```
Expected: empty container list.

### spark-01
```bash
ssh geo@10.0.0.96 "
  cd /home/geo/projects/mixvllm/docker/head && docker compose down
  docker stop nvidia-gpu-exporter node-exporter 2>/dev/null || true
  docker ps
"
```
Expected: empty container list.

---

## Step 2 — Update spark-02 (Worker)

spark-01 remains up during this step so NFS can be cleanly unmounted.

```bash
ssh geo@10.0.0.22
```

Once inside spark-02:

```bash
# 1. Unmount NFS mounts cleanly before any package changes
sudo umount /mnt/shared
sudo umount /mnt/nfs-rdma-test

# Confirm unmounted
mount | grep rdma   # should return nothing

# 2. Fetch latest package lists
sudo apt-get update

# 3. Install DGX OTA meta-package — drives all NVIDIA-interdependent upgrades
#    (nvidia-nvme-options, nvidia-fs modules, DGX services, etc.)
sudo apt-get install -y dgx-spark-ota-update-meta

# 4. Upgrade remaining packages (Docker, netplan, fwupd, libarchive, etc.)
sudo apt-get upgrade -y

# 5. Remove old / orphaned packages
sudo apt-get autoremove -y --purge

# 6. Confirm what changed
dpkg -l | grep dgx-spark-ota
# Expected: 26.03.1

# 7. Reboot
sudo reboot
```

**Wait for spark-02 to come back** (~2–3 min). Then verify from your laptop:

```bash
# Wait until SSH is available again
ssh geo@10.0.0.22 "uname -r && nvidia-smi | head -5 && rdma link"
```

Expected:
- Kernel: `6.17.0-1014-nvidia` (unchanged, was already latest)
- GPU: GB10 visible in nvidia-smi
- RDMA: `roceP2p1s0f0/1 state ACTIVE physical_state LINK_UP`

Verify NFS auto-remounted (fstab has `_netdev`):
```bash
ssh geo@10.0.0.22 "df -h /mnt/shared && mount | grep rdma"
```
Expected: `/mnt/shared` shows `192.168.100.1:/mnt/shared` mounted via `proto=rdma`.

---

## Step 3 — Update spark-01 (Head)

Now spark-02 is updated. Stop NFS clients are already gone (spark-02 has remounted but spark-01 is about to reboot, which is fine — the fstab `_netdev` flag handles remount after spark-01 comes back).

```bash
ssh geo@10.0.0.96
```

Once inside spark-01:

```bash
# 1. Fetch latest package lists
sudo apt-get update

# 2. OTA meta-package
sudo apt-get install -y dgx-spark-ota-update-meta

# 3. Upgrade remaining packages
sudo apt-get upgrade -y

# 4. Remove old / orphaned packages
sudo apt-get autoremove -y --purge

# 5. Confirm
dpkg -l | grep dgx-spark-ota
# Expected: 26.03.1

# 6. Reboot
sudo reboot
```

**Your SSH session to spark-01 will disconnect** — this is expected.

**Wait for spark-01 to come back** (~2–3 min). Then from your laptop:

```bash
ssh geo@10.0.0.96 "uname -r && nvidia-smi | head -5 && rdma link && showmount -e localhost"
```

Expected:
- Kernel unchanged, GPU visible, RDMA ACTIVE
- `showmount` lists `/mnt/shared` and `/mnt/nfs-rdma-test` still exported to `192.168.100.2`

Then verify spark-02's NFS remounted successfully:
```bash
ssh geo@10.0.0.22 "df -h /mnt/shared"
```

---

## Step 4 — Check for 26.03 vLLM Container Image

The private registry is `docker-registry.mixwarecs-home.net:5000`. Check whether the 26.03 image is available:

```bash
# Try to pull on spark-01 first
ssh geo@10.0.0.96 "docker pull docker-registry.mixwarecs-home.net:5000/nvidia/vllm:26.03-py3"
```

**If it succeeds:** proceed to update docker-compose files (Step 5).  
**If it fails** (`manifest unknown` or connection error): the image isn't published yet — skip Step 5 and stay on 26.02. The OS update is still complete and worthwhile.

---

## Step 5 — Update Container Image Tag (only if 26.03 image pulled successfully)

Two files to update (same change in each — swap `26.02-py3` → `26.03-py3`):

### spark-01 head compose
```
/home/geo/projects/mixvllm/docker/head/docker-compose.yml
```
Line 3: `image: docker-registry.mixwarecs-home.net:5000/nvidia/vllm:26.02-py3` → `26.03-py3`  
Line 88: same change for the `vllm:` service

```bash
ssh geo@10.0.0.96 "
  sed -i 's|nvidia/vllm:26.02-py3|nvidia/vllm:26.03-py3|g' \
    /home/geo/projects/mixvllm/docker/head/docker-compose.yml
  grep 'image:' /home/geo/projects/mixvllm/docker/head/docker-compose.yml
"
```

### spark-02 worker compose
```
/home/geo/projects/mixvllm/docker/worker/docker-compose.yml
```
```bash
ssh geo@10.0.0.22 "
  sed -i 's|nvidia/vllm:26.02-py3|nvidia/vllm:26.03-py3|g' \
    /home/geo/projects/mixvllm/docker/worker/docker-compose.yml
  grep 'image:' /home/geo/projects/mixvllm/docker/worker/docker-compose.yml
"
```

Pull image on spark-02 as well:
```bash
ssh geo@10.0.0.22 "docker pull docker-registry.mixwarecs-home.net:5000/nvidia/vllm:26.03-py3"
```

### Clear vLLM compile cache (important after container update)

A new container version ships a new vLLM build and its compiled CUDA graphs will be incompatible with the previous cache. Clear it to avoid silent corruption:

```bash
ssh geo@10.0.0.96 "
  du -sh /mnt/shared/vllm-compile-cache/
  rm -rf /mnt/shared/vllm-compile-cache/*
  echo 'Cache cleared'
"
```

---

## Step 6 — Start Containers

**Order:** spark-02 worker first, then spark-01 head + vLLM.

### spark-02 — Start Ray worker
```bash
ssh geo@10.0.0.22 "
  cd /home/geo/projects/mixvllm/docker/worker
  docker compose up -d
  docker start nvidia-gpu-exporter node-exporter 2>/dev/null || true
"

# Watch until worker joins the cluster
ssh geo@10.0.0.22 "docker logs -f ray-worker --tail=5"
```
Wait for: `✅ Ray WORKER joined cluster`  
Press Ctrl+C to exit the log follow.

### spark-01 — Start Ray head + vLLM
```bash
ssh geo@10.0.0.96 "
  cd /home/geo/projects/mixvllm/docker/head
  docker compose up -d
  docker start nvidia-gpu-exporter node-exporter 2>/dev/null || true
"

# Watch Ray head come up
ssh geo@10.0.0.96 "docker logs -f ray-head --tail=5"
```
Wait for: `✅ Ray HEAD started`  
Press Ctrl+C.

```bash
# Watch vLLM load the model (this takes several minutes for gpt-oss-120b)
ssh geo@10.0.0.96 "docker logs -f vllm-server --tail=20"
```
Watch for progress through:
- `Parse safetensors files: 100%` — weights loaded
- `Using max model len 131072`
- CUDA graph capture (takes longest — ~5–10 min for 120B)
- `Application startup complete` — API is ready

Press Ctrl+C when you see the API ready message.

---

## Step 7 — Verification

Run all of these from your laptop:

### RDMA links
```bash
ssh geo@10.0.0.96 "rdma link"
ssh geo@10.0.0.22 "rdma link"
```
Expected on both: `roceP2p1s0f0/1 state ACTIVE physical_state LINK_UP`

### NFS-over-RDMA on spark-02
```bash
ssh geo@10.0.0.22 "mount | grep rdma"
```
Expected: both `/mnt/shared` and `/mnt/nfs-rdma-test` with `proto=rdma`

### Ray cluster has both GPUs
```bash
ssh geo@10.0.0.96 "
  docker exec ray-head python3 -c \"
import subprocess
r = subprocess.run(['ray','list','nodes','--address=192.168.100.1:6379'], capture_output=True, text=True)
print(r.stdout or r.stderr[:400])
\"
"
```
Alternatively check Ray dashboard at `http://10.0.0.96:8265` in your browser.

### vLLM API health
```bash
curl http://10.0.0.96:8000/health
curl http://10.0.0.96:8000/v1/models | python3 -m json.tool
```
Expected: `{"status":"ok"}` and model list showing `openai/gpt-oss-120b`

### Quick inference test
```bash
curl http://10.0.0.96:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 30
  }'
```

---

## Troubleshooting

### NFS doesn't remount on spark-02 after spark-01 reboot
```bash
ssh geo@10.0.0.22 "sudo mount -a"
# If that fails:
ssh geo@10.0.0.22 "sudo mount 192.168.100.1:/mnt/shared /mnt/shared -o proto=rdma,port=20049,vers=4.1"
```

### RDMA port shows DOWN after reboot
```bash
# Check the link
ssh geo@10.0.0.96 "rdma link; ip link show enP2p1s0f0np0"
# Bring it up manually if needed
ssh geo@10.0.0.96 "sudo ip link set enP2p1s0f0np0 up; sudo ip link set enP2p1s0f0np0 mtu 9000"
```

### vLLM fails to start — NCCL / Ray errors
The `NCCL_IB_GDR_DMABUF_ENABLE=0` flag in the `.env` files is critical for GB10 unified memory.  
If you see DMABUF hangs or `NCCL error: unhandled system error`, confirm it is set:
```bash
ssh geo@10.0.0.96 "grep DMABUF /home/geo/projects/mixvllm/docker/head/.env"
ssh geo@10.0.0.22 "grep DMABUF /home/geo/projects/mixvllm/docker/worker/.env"
```
Both must show `NCCL_IB_GDR_DMABUF_ENABLE=0`.

### Ray worker doesn't join cluster
Check that spark-01 Ray head is listening:
```bash
ssh geo@10.0.0.96 "ss -tlnp | grep 6379"
```
Then restart just the worker:
```bash
ssh geo@10.0.0.22 "cd /home/geo/projects/mixvllm/docker/worker && docker compose restart"
```

### Container image 26.03 pulled but vLLM won't load model
The compile cache may have old artifacts. Clear it:
```bash
ssh geo@10.0.0.96 "rm -rf /mnt/shared/vllm-compile-cache/*"
```
Then restart the vllm-server container.

### Roll back to 26.02 containers
If 26.03 containers are broken, revert image tags:
```bash
ssh geo@10.0.0.96 "
  sed -i 's|nvidia/vllm:26.03-py3|nvidia/vllm:26.02-py3|g' \
    /home/geo/projects/mixvllm/docker/head/docker-compose.yml
"
ssh geo@10.0.0.22 "
  sed -i 's|nvidia/vllm:26.03-py3|nvidia/vllm:26.02-py3|g' \
    /home/geo/projects/mixvllm/docker/worker/docker-compose.yml
"
```
The 26.02 images are still cached locally on both nodes. The OS-level update (26.03.1) is independent and stays.

---

## Key File Paths Reference

| File | Node | Purpose |
|------|------|---------|
| `/home/geo/projects/mixvllm/docker/head/docker-compose.yml` | spark-01 | Ray head + vLLM service definitions |
| `/home/geo/projects/mixvllm/docker/head/.env` | spark-01 | Model name, NCCL config, IP addresses |
| `/home/geo/projects/mixvllm/docker/worker/docker-compose.yml` | spark-02 | Ray worker service definition |
| `/home/geo/projects/mixvllm/docker/worker/.env` | spark-02 | NCCL config, head address |
| `/home/geo/projects/mixvllm/docker/config/model_registry.yml` | spark-01 | Model → vLLM args mapping |
| `/home/geo/projects/mixvllm/docker/config/launch.py` | spark-01 | vLLM launcher script (reads registry) |
| `/mnt/shared/huggingface/hub/` | spark-01 | All cached model weights |
| `/mnt/shared/vllm-compile-cache/` | spark-01 | vLLM Inductor/CUDA compile cache |
| `/etc/exports` | spark-01 | NFS export config |
| `/etc/fstab` | spark-02 | NFS client mount config (proto=rdma) |
