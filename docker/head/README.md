# Head Node Docker Configuration

## Overview
This document details the Docker configuration for the Ray head node in a distributed GPU inference setup using vLLM with RDMA (Remote Direct Memory Access) networking. The setup is optimized for multi-node GPU clusters with high-speed interconnects.

## Key Components

### 1. Base Container Configuration
```yaml
services:
  ray-node:
    image: docker-registry.mixwarecs-home.net:5000/nvidia/vllm:25.09-py3
    container_name: ray-${RAY_MODE}
    network_mode: host
    ipc: host
    runtime: nvidia
```
- Custom NVIDIA vLLM image
- Host networking for direct interface access
- IPC host mode for container shared memory
- NVIDIA runtime for GPU access

### 2. RDMA and Network Configuration
**Device Mounts:**
```yaml
devices:
  - /dev/infiniband/rdma_cm
  - /dev/infiniband/uverbs0
  - /dev/infiniband:/dev/infiniband
```

**System Files:**
```yaml
volumes:
  - /sys/class/infiniband:/sys/class/infiniband:ro
  - /sys/class/net:/sys/class/net:ro
  - /etc/rdma:/etc/rdma:ro
```

### 3. Environment Configuration
**Cluster Settings:**
- `MASTER_ADDR=spark-01-mesh`: Head node hostname
- `MASTER_PORT=8000`: Model server port
- `WORLD_SIZE=2`: Total cluster GPUs

**RDMA/NCCL Configuration:**
- `NCCL_IB_HCA=roceP2p1s0f0`: RDMA device
- `NCCL_SOCKET_IFNAME=enP2p1s0f0np0`: Network interface
- `NCCL_NET_GDR_LEVEL=5`: GPUDirect RDMA enabled
- QoS settings (TC, SL, timeout, retry)

### 4. Model Configuration
Model settings are managed through the `model_registry.yml` file in the parent directory:
```yaml
models:
  gpt-oss-20b:
    model: openai/gpt-oss-20b
    dtype: float16
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.35
    max_num_seqs: 8
    description: Lightweight
```

The `launch_model.sh` script automatically loads these configurations based on the `MODEL_NAME` environment variable.

### 5. Performance Settings
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - MODEL_NAME=gpt-oss-20b  # References model_registry.yml
```

### 6. Resource Limits
```yaml
ulimits:
  memlock: -1        # Unlimited for RDMA
  stack: 67108864    # Deep model support
```

## Network Architecture

### High-Speed Network
- Dedicated RDMA network (192.168.100.x/24)
- RoCE (RDMA over Converged Ethernet)
- Expected bandwidth: ~12 GB/s for large messages

### Ray Cluster Setup
- Head node as cluster manager
- `RAY_ADDRESS=192.168.100.1:6379` for workers
- Mesh network for worker connections

## Performance Optimization

### 1. Network Settings
- MTU: 9000 for throughput
- QoS optimization for RDMA
- Direct GPU-to-GPU transfer

### 2. Resource Management
- 35% GPU memory utilization
- Controlled batch processing
- Unlimited locked memory (RDMA)

### 3. Monitoring
- NCCL debug logging
- Performance metrics
- System monitoring

## Model Distribution
- Tensor parallelism support
- NCCL for GPU communication
- Multi-GPU model optimization

## Model Configuration Management

### 1. Model Registry
The `model_registry.yml` in the parent directory manages model configurations:
- Model paths and identifiers
- Data types and precision settings
- Tensor parallelism configuration
- Memory and batch size settings

### 2. Dynamic Model Loading
The `launch_model.sh` script provides:
- Dynamic configuration loading
- Environment-based model selection
- Consistent deployment across nodes
- Automatic parameter management

### 3. Usage
1. Select model via environment:
   ```bash
   # In .env file
   MODEL_NAME=gpt-oss-20b
   ```
2. Configuration is automatically loaded
3. Model server launched with optimal settings

## Usage

1. Copy `.env.example` to `.env`
2. Adjust environment variables for your setup
3. Start the container:
   ```bash
   docker-compose up -d
   ```
4. Monitor logs:
   ```bash
   docker-compose logs -f
   ```

## Requirements
- NVIDIA GPUs with RDMA support
- RDMA-capable network interface
- Docker with NVIDIA runtime
- Proper RDMA driver installation

This configuration is optimized for distributed inference of large language models, utilizing RDMA for efficient inter-node communication and Ray for distributed task management.