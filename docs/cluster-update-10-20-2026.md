# DGX Spark Cluster Update: Handoff

> Historical record: written for the old `docker/head` + `docker/worker` layout. The cluster now runs from [`docker/cluster.yml`](../docker/cluster.yml) with `spark-cluster` (see [docker/README.md](../docker/README.md)).

**Last updated:** 2026-10-20 (evening)  
**Status:** **All done.** OS, firmware, kernel and driver updated and verified on both nodes. The container update is also done: the cluster runs `nvidia/vllm:26.09-py3` (vLLM 0.29.0) **without Ray**, and serves `openai/gpt-oss-120b` with tensor parallelism 2 across both nodes (see [Container update](#container-update-done)).  
**Next action:** Only the optional [follow-ups](#follow-ups-and-cleanup-after-vllm-works).

---

## How to work with this user

- **Go one step at a time and wait for feedback** before the next step. The user asked for this explicitly.
- **The user runs every `sudo` command.** sudo on the nodes asks for a password, and the agent can't type it. Give the user ready-to-paste blocks, ask for the output, then verify it yourself.
- The agent may run **non-sudo, read-only checks** and Docker commands (`geo` is in the `docker` group) over SSH. Use hostnames, not IPs:
  `ssh -o BatchMode=yes geo@spark-01 '…'` / `ssh -o BatchMode=yes geo@spark-02 '…'`
  (`ssh geo@10.0.0.x` fails host-key verification from the agent's environment.)
- **Check state yourself rather than assuming a step was done.** Twice today a step the user was asked to do hadn't happened: the NFS unmount, and `grub-reboot` before a reboot. Check before every reboot.
- Run long sudo work inside `tmux` on the node.
- Use `-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold` on apt runs so local config files are kept. `/etc/nfs.conf` holds the RDMA settings.

---

## Cluster at a glance

| | spark-01 (HEAD) | spark-02 (WORKER) |
|---|---|---|
| Hardware | **NVIDIA DGX Spark** (Founders Edition) | **ASUS Ascent GX10** (GB10 partner unit) |
| Management IP | 10.0.0.96 | 10.0.0.22 |
| RDMA mesh IP | 192.168.100.1 | 192.168.100.2 |
| GPU / RAM | GB10 / 121 GB unified | GB10 / 121 GB unified |
| Disk | 3.7 TB (root + all data) | 916 GB (root only) |
| Role | Ray head + vLLM API (:8000) + **NFS server** | Ray worker + **NFS client** |

- RDMA interface: `enP2p1s0f0np0` / `roceP2p1s0f0`, MTU 9000
- NFS: spark-01 exports `/mnt/shared` and `/mnt/nfs-rdma-test` to 192.168.100.2. spark-02's fstab mounts them with `proto=rdma,port=20049,vers=4.1,_netdev` as **hard** mounts.
- Model weights: `/mnt/shared/huggingface/` on spark-01. Model: `openai/gpt-oss-120b`, with tensor parallelism 2 across both nodes through Ray.
- Project: `/home/geo/projects/mixvllm/docker/{head,worker,config}` on both nodes. It's a git repo; the compose files are tracked.

---

## Current state (verified on both nodes)

| Item | Both nodes |
|---|---|
| DGX OS / OTA | **DGX OS 7.6.0**, `dgx-spark-ota-update-meta` **26.09.2**, Ubuntu 24.04.5 |
| Kernel | **6.17.0-1032-nvidia**, set as the GRUB default (`saved_entry`) |
| NVIDIA driver | **595.99.02** (DKMS, `nvidia-dkms-595-open`), built for 1032 |
| nvidia-fs | 2.28.4 |
| Docker | 29.6.2, Compose plugin 5.2.0, containerd 2.2.5, nvidia-container-toolkit 1.20.1, `nvidia` runtime present |
| Kernel cmdline | includes `kho=off` (from `/etc/default/grub.d/nvidia-spark-kho.cfg`, installed by OTA 26.09.2) |
| memlock | unlimited (`/etc/security/limits.d/99-nv-spark-limits.conf`, from the `nvidia-spark-limits` package) |
| apt | 0 pending upgrades, no reboot required, `dpkg --audit` clean |
| RDMA | ACTIVE/LINK_UP, MTU 9000, jumbo ping OK, `ib_write_bw` **≈111.7 Gb/s in both directions** |
| NFS | spark-01 `portlist` = `rdma 20049`, both exports active. spark-02 has **2/2 mounts with `proto=rdma`**. |
| Containers | **none running**. The compose stacks are down, and `nvidia-gpu-exporter` and `node-exporter` are stopped. |

**Firmware (different vendors, so the versions can't match; both are on the latest available):**
- spark-01 (NVIDIA): EC 0x03000508, UEFI 0x02009b0b, USB-C PD 0x00000516
- spark-02 (ASUS): EC 0x02000007, SoC/UEFI 0x03000008, USB-C PD 0x00000516

**Container images present locally on both nodes** (tagged `docker-registry.mixwarecs-home.net:5000/nvidia/vllm:<tag>`):
- `26.09-py3`: vLLM 0.29.0, CUDA 13.4.1, build 452757715 (same on both nodes). **In use.** No Ray inside.
- `26.03-py3`: vLLM 0.17.1, CUDA 13.2.0, Ray 2.54.0. Kept as a fallback.
- `26.02-py3`: vLLM 0.15.1, CUDA 13.1.1, Ray 2.54.0. What the cluster ran before today.
- Older tags: 25.09, 25.10, 25.12.

**Package differences between the nodes (none affect the cluster):**
- spark-02 has 387 extra Node.js packages (Ubuntu's Node 18). spark-01 has Node 20 from NodeSource.
- spark-01 has `cuda-keyring`, which adds NVIDIA's CUDA apt repository, and a leftover `nvidia-dkms-580-open`. See [Follow-ups](#follow-ups--cleanup).

---

## What was done today (chronological)

1. **Pre-flight (read-only).**
   - Found **vLLM crash-looping before the update**: `vllm-server` had restarted 1,406 times, and `/health` gave no response.
   - Cause: Ray saw only 1 GPU ("Tensor parallel size (2) exceeds available GPUs (1)"). The worker had restarted after a spark-02 reboot and never rejoined.
   - `ray status` on the head also errored (`get_cluster_metadata` → `.decode` on None).
   - The user chose to update first and fix this afterwards; a clean restart in the correct order should resolve it.
2. **Snapshots and backups** in `~/pre-26.03/` on both nodes:
   - kernel, driver, `dpkg -l`, `dgx-release`, `docker images`, `docker ps`, RDMA link
   - a copy of `mixvllm/docker/`
   - `etc-backup.tgz` (`nfs.conf`, `exports`, `fstab`, `netplan`, `docker/daemon.json`)
3. **The private registry is down.** `docker-registry.mixwarecs-home.net` (10.0.0.212:5000) answers neither ping nor connections. The `26.03-py3` image was pulled from `nvcr.io` and retagged locally instead.
4. **Stopped all containers** in this order: `vllm-server`, then the worker stack, then the head stack, then the exporters.
5. **OS upgrade, both nodes.**
   - Ran `apt-get update`, then `apt-get install dgx-spark-ota-update-meta`, then `apt-get dist-upgrade` with confold.
   - The fresh package lists jumped straight to **OTA 26.09.2**, not the 26.03.1 originally planned. The user accepted this ("why not the latest").
   - Driver **590 → 595.99.02** came from **Ubuntu's archive** (`noble-updates/restricted`), not from NVIDIA's OTA: `nvidia-driver-590-open` became an empty placeholder package that depends on 595.
   - Docker went to 29.6.2.
   - The 26.09.2 update also installed **NVIDIA's mitigation for the kernel 7.0 RDMA regression**: `kho=off` plus unlimited memlock.
6. **Firmware, both nodes** (`fwupdmgr refresh --force && fwupdmgr update`, then reboot). All updates applied successfully.
   - When fwupd offered to upload a report to LVFS, it was rejected with "machine ID banned". This is harmless and optional; use `--no-unreported-check` to skip the prompt.
7. **Kernel investigation.** The 26.09 update had installed **7.0.0-1019** without its headers, so the NVIDIA driver isn't built for it. GRUB was pinned to 6.17.0-1014, and an `apt-mark hold` sits on `linux-image-nvidia-hwe-24.04`. Both pins date from before today; no record of who set them.
   - NVIDIA's [DGX Spark Update Advisory](https://forums.developer.nvidia.com/t/dgx-spark-update-advisory/383254) (Sept 14, 2026) tells multi-node/RoCE users to **hold off on 7.0.0-1019**, which has an NCCL `ibv_reg_mr` ENOMEM regression. The fix is `kho=off`.
   - There are also community reports of a 50–60% slowdown on 7.0.
   - **Decision: don't use 7.0.**
8. **Kernel 6.17.0-1014 → 6.17.0-1032, both nodes.**
   - Installed the same 7 packages that 1014 had: image, modules, headers, `modules-nvidia-fs`, tools, `nvidia-6.17-headers`, `nvidia-6.17-tools`.
   - DKMS built 595.99.02 for 1032 automatically.
   - Booted 1032 once with `grub-reboot`, verified it, then made it permanent with `grub-set-default`.
   - spark-02 went first as the test node. NFS on spark-02 was unmounted before spark-01 rebooted, and remounted afterwards.
   - Why 1032: **639 CVE fixes** since 1014 (built Mar 17, 2026), plus RDMA fixes ("Fix stale RoCE GIDs during netdev events"), ConnectX-7 (mlx5) fixes, NFS fixes, and a DGX Spark iGPU change. Community reports say it works for 2-node GB10 NCCL/RoCE.
   - There's a separate report of a one-way RDMA slowdown on the ASUS GX10 with kernel 1029. **Tested: not present on 1032**, with symmetric 111.7 Gb/s.
   - The kernel error log on 1032 is identical to 1014's, which are known platform quirks: `of_root node is NULL`, MPAM, CPU energy-model messages.
9. **Final comparison of both nodes:** identical on every runtime component (table above).

### Decisions and reasons, so they aren't reopened
- **Kernel stays on 6.17.0-1032.** Not 7.0.0-1019 (NVIDIA advisory, regressions, no newer 7.0 kernel available). The 7.0 kernel is installed but unused, and its headers are deliberately *not* installed.
- **The driver stays on 595.99.02**, the user's choice.
  - NVIDIA's official DGX Spark release notes list **580.159.03** with DGX OS 7.5.0, but that page hasn't been updated for 7.6.
  - The cluster previously ran Ubuntu's 590 driver.
  - **610/615** exist in the repository but were rejected: not chosen by NVIDIA for Spark, and future OTAs would fight them.
  - If NCCL problems appear, the fallback is `nvidia-driver-580-open` (580.178.04 in the repository).
- **Both nodes must be identical** in software. Firmware differs by vendor and that's accepted.
- **LibreOffice stays.** Removing it also removes `nvidia-system-station-apps`. Optional cleanup for later.

---

## Facts about the setup (verified on the nodes)

- **Compose services (since 2026-10-20, no Ray):** one `vllm` service per node, `restart: unless-stopped`, `command: python /config/launch.py`.
  - Head: container `vllm-server`, rank 0, API on :8000, rendezvous on `192.168.100.1:29501`.
  - Worker: container `vllm-worker`, rank 1, `--headless`. Expect a harmless compose warning: "NCCL_NET_PLUGIN variable is not set".
  - **Start the head before the worker.** Details are in `docker/README.md`.
- **The image tag appears only in the compose files** (once in each). Not in `.env`.
- **`MODEL_NAME` must match** in `docker/head/.env` and `docker/worker/.env`. Both `.env` files are untracked (gitignored).
- **vLLM args** (0.29.0): `dtype bfloat16`, `max_model_len 131072`, `tensor_parallel_size 2`, `gpu_memory_utilization 0.55`, `distributed_executor_backend mp`, `nnodes 2`.
- **Compile cache:** mounted per node at `/mnt/shared/vllm-compile-cache/{head,worker}`. The directory is owned by root, so you need `sudo` to clear it. The old 26.02 entries in the top level are ignored.
- The 26.09 container logs "CUDA Forward Compatibility mode ENABLED" (built for driver 615, running on 595). That's expected.
- **GRUB:** `GRUB_DEFAULT=saved`, and the `nv-no-grubmenu` package hides the menu. Entry string format:
  `Advanced options for DGX OS GNU/Linux>DGX OS GNU/Linux, with Linux <kver>`
  Kernels installed: 7.0.0-1019, 6.17.0-1032 (default), 6.17.0-1014 (fallback), 6.14.x, 6.11.0-1016.
- **apt holds (both nodes):** `linux-image-nvidia-hwe-24.04`, `linux-image-6.11.0-1016-nvidia`, `linux-headers-6.11.0-1016-nvidia`. The hold alone doesn't stop 7.0 being pulled in, because `linux-modules-nvidia-fs-nvidia-hwe-24.04` depends on it. The GRUB default is what keeps the nodes on 6.17.
- **NFS hard mounts:** if spark-01 reboots while spark-02 has the shares mounted, I/O on spark-02 freezes until spark-01 is back. Always unmount on spark-02 first (`sudo umount /mnt/shared /mnt/nfs-rdma-test`) and remount afterwards (`sudo mount -a`).
- **RDMA bandwidth test** (no sudo): run `ib_write_bw -d roceP2p1s0f0 -R -F -q 4 -D 5 --report_gbits` on the server, and the same command plus `<server IP>` on the client. Expected result: ≈111.7 Gb/s.

---

## Container update (done)

Done on the evening of 2026-10-20. Commits `b9d355e`, `e7edeb6`, `ecd2c84`, `da5c665`.

1. **Pulled `nvcr.io/nvidia/vllm:26.09-py3` on both nodes** and retagged it under the registry name. Build 452757715 on both.
2. **Checked the config against vLLM 0.29.**
   - The GPU smoke test passed on both nodes.
   - Every option in `model_registry.yml` still exists.
   - `python -m vllm.entrypoints.openai.api_server` is deprecated in 0.29, so `launch.py` now runs `vllm serve <model>`.
   - The default `--gpu-memory-utilization` in 0.29 is 0.92 (not about 1.0). Every registry entry sets its own value anyway.
3. **Switched the image tags** to `26.09-py3`.
4. **Found that 26.09 ships without Ray**, so the Ray-based compose files couldn't start (`ray: command not found`).
   - **Decision:** use vLLM's built-in multi-node launcher instead of adding Ray back. That's the path NVIDIA ships and tests, and it removes the Ray worker-rejoin failure behind the 1,406-restart crash loop.
   - Ray never carried the GPU traffic; NCCL over RoCE does, in both setups.
5. **Rewrote the compose files** (one `vllm` service per node) and taught `launch.py` the `CLUSTER_*` variables.
   - **spark-02 `.env`:** added `MODEL_NAME=gpt-oss-120b` and fixed `HARMONY_ENCODINGS` to `/mnt/shared/harmony/encodings`.
   - **Both `.env` files:** commented out `MASTER_ADDR`, `MASTER_PORT=8000`, `RANK=1` and `WORLD_SIZE=2`. Nothing used them, but PyTorch reads those names.
   - **Backups** are in `~/pre-26.03/*.env.before-mp*` on each node.
6. **Compile cache:** the old compose files never mounted `/mnt/shared/vllm-compile-cache`, so the old step 4 (moving it aside) was skipped. It's now mounted per node (`head/`, `worker/`).
7. **Verified:**
   - `/health` returns 200, `/v1/models` lists `openai/gpt-oss-120b`, and a chat completion works.
   - NCCL 2.31.2 uses `NET/IB` over `roceP2p1s0f0` (RoCE), not a TCP fallback.
   - No `ibv_reg_mr` or ENOMEM errors.
   - Single-stream decode: **57 tok/s** (512 tokens, one request). There's no earlier number to compare with, because the cluster was crash-looping before the update.
   - **Compile cache test** (two full restarts):
     - `torch.compile` went from 10.9 s to 0.7 s.
     - Total time to ready was 143 s, then 183 s. Weight loading dominates startup and varies with page cache; the cache saves only about 10 s.
     - FlashInfer autotune results weren't reused ("Loaded 0 configs"), so autotune runs on every start.

### Starting the cluster (new procedure)
```bash
ssh geo@spark-01 "cd /home/geo/projects/mixvllm/docker/head && docker compose up -d && docker start nvidia-gpu-exporter node-exporter"
ssh geo@spark-02 "cd /home/geo/projects/mixvllm/docker/worker && docker compose up -d && docker start nvidia-gpu-exporter node-exporter"
ssh geo@spark-01 "docker logs -f --tail=20 vllm-server"    # wait for: Application startup complete
curl http://10.0.0.96:8000/health
```
To stop the cluster, run `docker compose down` on the worker first, then on the head.

### Rollback (if 26.09 misbehaves)
The Ray-based compose files only work with an image that contains Ray (26.03 or 26.02):
1. `docker compose down` on the worker, then on the head.
2. On both nodes, check out the pre-migration compose files and launcher at commit `e7edeb6`, then change the tag to `26.03-py3`:
   `git checkout e7edeb6 -- docker/head docker/worker docker/config && sed -i 's|vllm:26.09-py3|vllm:26.03-py3|' docker/*/docker-compose.yml`
3. Uncomment `RANK`/`WORLD_SIZE`/`MASTER_*` in the `.env` files if you want them back (nothing uses them).
4. Start in the old Ray order: head `ray-node`, worker, check `docker exec ray-head ray status` (2 nodes, 2.0 GPU), then head `vllm`.

**Kernel fallback:** boot 1014 once with `sudo grub-reboot "…with Linux 6.17.0-1014-nvidia"`, or make it permanent with `grub-set-default`. **Driver fallback:** 580.178.04.

---

## Follow-ups and cleanup (after vLLM works)

- **Leftover packages:** `sudo apt-get autoremove` (dry run first) to remove the 590 driver leftovers, the 6.11 tools, and old firmware packages. Remove `nvidia-dkms-580-open` on spark-01.
- **`cuda-keyring` exists only on spark-01.** Either remove it or add it to spark-02, so future upgrades stay identical.
- **Make the kernel pin sturdier:** an apt preferences file blocking `linux-*-nvidia-hwe-24.04` at version 7.*. Holds alone let 7.0 be pulled in through `nvidia-fs`.
- **Bring the private registry back** (10.0.0.212:5000), then push `26.09-py3` (and `26.03-py3`) to it.
- ~~**Fix the worker script.**~~ No longer needed: the Ray worker script is gone.
- **Drop `ray` from `mixvllm_server/pyproject.toml`.** It's listed as a dependency but no code imports it.
- **Remove the leftover `RAY_*` variables** from both `.env` files. They're unused now.
- **Optional:** add `NCCL_NET_PLUGIN=none` to spark-02's `.env` to match the head and silence the compose warning.
- **Optional:** remove the stopped `ray-worker` container on spark-01. It's 6 months old, on the 25.09 image.
- **Optional:** remove LibreOffice. This also removes `nvidia-system-station-apps`, and a future OTA may reinstall it.
- **Revisit kernel 7.0** only when NVIDIA ships a fixed kernel or updates its advisory. Test it on spark-02 with `grub-reboot`, after installing `linux-headers-7.0.0-<ver>-nvidia`.
- **The repo copy `docs/cluster-update-26.03-runbook.md` is out of date.** Replace it with this file.

---

## Key file paths

| File | Node | Purpose |
|---|---|---|
| `/home/geo/projects/mixvllm/docker/README.md` | both | How the cluster runs: start order, launcher, NCCL settings, troubleshooting |
| `/home/geo/projects/mixvllm/docker/head/docker-compose.yml` | spark-01 | `vllm` service, rank 0 (`vllm-server`, API :8000) |
| `/home/geo/projects/mixvllm/docker/head/.env` | spark-01 | `MODEL_NAME`, NCCL config (`NCCL_IB_GDR_DMABUF_ENABLE=0` is required), IPs |
| `/home/geo/projects/mixvllm/docker/worker/docker-compose.yml` | spark-02 | `vllm` service, rank 1 (`vllm-worker`, headless) |
| `/home/geo/projects/mixvllm/docker/worker/.env` | spark-02 | `MODEL_NAME` (must match the head), NCCL config |
| `/home/geo/projects/mixvllm/docker/config/model_registry.yml` | both (git) | Model → vLLM args |
| `/home/geo/projects/mixvllm/docker/config/launch.py` | both (git) | Builds `vllm serve` + multi-node flags |
| `/mnt/shared/vllm-compile-cache/{head,worker}/` | spark-01 (NFS) | Per-node vLLM compile cache (root-owned) |
| `/etc/nfs.conf` | spark-01 | `rdma=y`, `rdma-port=20049` (lines 68–69) |
| `/etc/exports` | spark-01 | Both exports to 192.168.100.2 |
| `/etc/fstab` | spark-02 | NFS mounts (`proto=rdma,port=20049,vers=4.1,_netdev`) |
| `/etc/default/grub.d/nvidia-spark-kho.cfg` | both | Adds `kho=off` |
| `/etc/security/limits.d/99-nv-spark-limits.conf` | both | memlock unlimited |
| `~/pre-26.03/` | both | Pre-update snapshot and `/etc` + compose backups |

## Sources
- [NVIDIA: DGX Spark Update Advisory (kernel 7.0.0-1019, `kho=off`)](https://forums.developer.nvidia.com/t/dgx-spark-update-advisory/383254)
- [Kernel 7.0.0-1019 NCCL/RoCE ENOMEM; 6.17.0-1032 works](https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023)
- [ASUS GX10 one-way RDMA regression on 6.17.0-1029](https://forums.developer.nvidia.com/t/severe-one-way-rdma-performance-regression-on-asus-ascent-gx10-with-kernel-6-17-0-1029-nvidia/379303)
- [NVIDIA DGX Spark Release Notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
- [NVIDIA vLLM Release 26.09 notes](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-09.html)
