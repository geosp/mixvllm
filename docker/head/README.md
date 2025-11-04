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

### 4. Performance Settings
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - GPU_MEMORY_UTILIZATION=0.35  # Conservative setting
  - MAX_NUM_SEQS=16             # Batch size control
```

### 5. Resource Limits
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