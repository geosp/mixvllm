# Worker Node Docker Configuration

## Overview
This document details the Docker configuration for Ray worker nodes in a distributed GPU inference setup. The worker nodes connect to the head node and participate in distributed model serving using vLLM with RDMA networking.

## Key Components

### 1. Base Container Configuration
```yaml
services:
  ray-node:
    image: docker-registry.mixwarecs-home.net:5000/nvidia/vllm:25.09-py3
    container_name: ray-${RAY_MODE}
    network_mode: host
    ipc: host
    restart: unless-stopped
    runtime: nvidia
```
- Same base image as head node
- Host networking for RDMA access
- Automatic restart policy
- NVIDIA runtime for GPU access

### 2. RDMA and Network Configuration
**Device Access:**
```yaml
devices:
  - /dev/infiniband/rdma_cm
  - /dev/infiniband/uverbs0
  - /dev/infiniband:/dev/infiniband
```

**System Mounts:**
```yaml
volumes:
  - /sys/class/infiniband:/sys/class/infiniband:ro
  - /sys/class/net:/sys/class/net:ro
  - /etc/rdma:/etc/rdma:ro
  - /tmp/ray:/tmp/ray
```

### 3. Environment Settings
**Core Settings:**
- `RAY_MODE=worker`: Node role
- `VLLM_HOST_IP`: Worker node IP (192.168.100.x)
- `RAY_ADDRESS`: Head node address (192.168.100.1:6379)

**NCCL/RDMA Configuration:**
```yaml
environment:
  # NCCL / RDMA
  - NCCL_IB_HCA=${NCCL_IB_HCA}
  - NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
  - NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}
  - NCCL_NET_PLUGIN=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
```

**Performance Settings:**
```yaml
  - NCCL_IB_TC=106          # Traffic Class
  - NCCL_IB_SL=3           # Service Level
  - NCCL_IB_TIMEOUT=23     # Extended timeout
  - NCCL_IB_RETRY_CNT=7    # Retry attempts
```

### 4. Security and Resource Limits
```yaml
security_opt:
  - apparmor:unconfined

ulimits:
  memlock: -1
  stack: 67108864

cap_add:
  - IPC_LOCK
  - SYS_NICE
  - SYS_RESOURCE
```

### 5. GPU Resource Management
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Startup Process

The worker node follows a specific startup sequence:
1. Displays configuration information
2. Performs GPU and RDMA checks
3. Waits for head node availability
4. Joins the Ray cluster
5. Maintains operation with log monitoring

```bash
command: >
  bash -c '
    echo "Ray WORKER Node Starting";
    # ... diagnostic checks ...
    ray start \
      --address=${RAY_ADDRESS} \
      --node-ip-address=${VLLM_HOST_IP} \
      --num-gpus=1 \
      --verbose;
```

## Usage

1. Copy `.env.exmple` to `.env`
2. Configure environment variables:
   - Set `VLLM_HOST_IP` to worker's IP
   - Verify `RAY_ADDRESS` points to head node
   - Configure RDMA interface settings
3. Start the worker:
   ```bash
   docker-compose up -d
   ```
4. Monitor logs:
   ```bash
   docker-compose logs -f
   ```

## Important Notes

### 1. Environment Variables
The worker inherits NCCL environment variables for RoCE via:
```yaml
VLLM_RAY_WORKER_ENV_VARS=NCCL_IB_DISABLE,NCCL_IB_HCA,NCCL_IB_GID_INDEX,...
```

### 2. Resource Access
- Requires direct access to RDMA devices
- Needs GPU visibility
- Must be able to reach head node on the mesh network

### 3. Performance
- Uses GPUDirect RDMA for optimal performance
- Configured for high-bandwidth inter-node communication
- Optimized NCCL settings for stability

## Requirements
- NVIDIA GPU with RDMA support
- RDMA-capable network interface
- Proper RDMA drivers and configuration
- Network connectivity to head node

This configuration enables worker nodes to participate in distributed model serving with optimal performance and reliability.