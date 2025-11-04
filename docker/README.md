# Docker Folder Documentation

This folder contains Docker Compose configurations and environment examples for deploying MixVLLM services in different modes. It is organized into three subfolders:

## Structure
- `head/`
- `stand_alone/`
- `worker/`

---

## 1. `head/`
**Purpose:** Configuration for the Ray head node in a distributed setup.

**Contents:**
- `docker-compose.yml`: Defines the Ray head node service, including GPU, RDMA, and NCCL settings. Uses environment variables from `.env`.
- `.env.example`: Example environment file for cluster, NCCL, CUDA, and Hugging Face cache settings.

**Key Features:**
- Uses NVIDIA GPU and RDMA devices for high-performance distributed inference.
- Extensive NCCL and Ray environment configuration for cluster networking and debugging.

---

## 2. `stand_alone/`
**Purpose:** Standalone deployment for model and terminal servers.

**Contents:**
- `docker-compose.yml`: Defines two services:
  - `model-server`: Runs a vLLM model server with GPU support and Hugging Face cache.
  - `terminal-server`: Runs a terminal server (image: `ghcr.io/geosp/mixvllm:main`).

**Key Features:**
- Simple deployment for local or single-node use.
- GPU resources reserved for model server.
- Hugging Face token support via environment variable.

---

## 3. `worker/`
**Purpose:** Configuration for Ray worker nodes in a distributed setup.

**Contents:**
- `docker-compose.yml`: Defines the Ray worker node service, similar to the head node but with `RAY_MODE=worker`.
- `.env.exmple`: Example environment file for worker-specific settings.

**Key Features:**
- Inherits NCCL, CUDA, and Ray settings for distributed operation.
- GPU and RDMA device configuration for high-performance networking.

---

## Usage
- Copy `.env.example` or `.env.exmple` to `.env` and adjust values for your cluster.
- Use `docker-compose up` in the desired subfolder to start services.
- For distributed setups, start the head node first, then workers.

---

## Notes
- Ensure RDMA and GPU devices are available and properly configured on your host machines.
- Hugging Face cache paths and tokens should be set according to your environment.
- For more details, see each subfolder's `docker-compose.yml` and environment example files.
