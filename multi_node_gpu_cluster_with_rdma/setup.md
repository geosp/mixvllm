# Multi-Node GPU Cluster with RDMA Setup Guide

This guide documents the complete process for setting up inter-node GPU communication using RDMA over Converged Ethernet (RoCE) between NVIDIA DGX Spark nodes.

## Important: Network and Hostname Configuration

The nodes have multiple network interfaces and hostnames:
- **Main hostnames**: spark-01, spark-02 (10.0.0.x network - regular ethernet)
- **Mesh hostnames**: spark-01-mesh, spark-02-mesh (192.168.100.x network - high-speed RDMA)
- **RDMA traffic uses the 192.168.100.x mesh network exclusively**

When running MPI/NCCL commands, always use the mesh network IP addresses (192.168.100.1, 192.168.100.2) for best reliability.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Overview](#environment-overview)
3. [Initial System Check](#initial-system-check)
4. [Installing Dependencies](#installing-dependencies)
5. [System Optimization](#system-optimization)
6. [Building NCCL Tests](#building-nccl-tests)
7. [Testing Inter-Node Communication](#testing-inter-node-communication)
8. [Making Configuration Persistent](#making-configuration-persistent)
9. [Verification](#verification)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements
- 2× NVIDIA DGX Spark nodes (or similar)
- NVIDIA GPUs with GPUDirect RDMA support
- Mellanox ConnectX NICs (or similar RDMA-capable NICs)
- RoCE/InfiniBand cable connecting the nodes
- CUDA 13.0+ installed

### Software Requirements
- Ubuntu 22.04+ (ARM64 in this case)
- NVIDIA drivers installed
- Basic networking configured

## Environment Overview

### Test Environment Specifications
- **Nodes**: spark-01-mesh (192.168.100.1), spark-02-mesh (192.168.100.2)
- **GPUs**: NVIDIA GB10 (1 per node)
- **CUDA**: 13.0.88
- **NICs**: Mellanox ConnectX-7 (4 ports per node, 1 cable connected)
- **Network**: 192.168.100.x/24 for high-speed RDMA mesh traffic
- **Expected Performance**: 12+ GB/s inter-node bandwidth

## Initial System Check

### Step 1: Verify GPU and CUDA Installation

Run on both nodes:

```bash
# Check hostname
hostname

# Check GPU detection
nvidia-smi -L

# Check CUDA version
nvcc --version

# Check GPU-NIC topology
nvidia-smi topo -m
```

Expected output shows GPU with NODE-level connection to NICs (good for GPUDirect RDMA).

### Step 2: Identify RDMA Interfaces

```bash
# List all network interfaces
ip link show

# Check for Mellanox devices
lspci | grep -i mellanox

# Check RDMA devices
ls /sys/class/infiniband/

# Check RDMA link status
rdma link show
```

Identify active RDMA interfaces. In our case:
- `roceP2p1s0f0` → `enP2p1s0f0np0` (active, used for RDMA)

### Step 3: Verify Network Connectivity

```bash
# From spark-01-mesh (192.168.100.1)
ping -c 1 spark-02-mesh  # or use IP: 192.168.100.2
ping -c 1 192.168.100.2

# From spark-02-mesh (192.168.100.2)
ping -c 1 spark-01-mesh  # or use IP: 192.168.100.1
ping -c 1 192.168.100.1
```

Note: The hosts file should have:
```bash
192.168.100.1   spark-01-mesh
192.168.100.2   spark-02-mesh
```

## Installing Dependencies

Run on **both nodes**:

### Step 1: Install Build Tools and MPI

```bash
sudo apt update
sudo apt install -y git build-essential openmpi-bin libopenmpi-dev

# Verify MPI installation
which mpirun
mpirun --version
```

### Step 2: Install NCCL

```bash
# Add NVIDIA repository (if not already added)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Install NCCL
sudo apt update
sudo apt install -y libnccl2 libnccl-dev

# Verify NCCL installation
ls /usr/include/nccl.h
```

## System Optimization

Run on **both nodes**:

### Step 1: Configure Network Buffers

```bash
# Set network buffer sizes
sudo sysctl -w net.core.rmem_max=268435456
sudo sysctl -w net.core.wmem_max=268435456

# Make permanent
echo "net.core.rmem_max=268435456" | sudo tee /etc/sysctl.d/90-rdma-tuning.conf
echo "net.core.wmem_max=268435456" | sudo tee -a /etc/sysctl.d/90-rdma-tuning.conf
```

### Step 2: Configure Memory Limits

```bash
# Check current limit
ulimit -l

# Set unlimited memlock
echo -e "* soft memlock unlimited\n* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf

# Note: Logout/login required for this to take effect
```

### Step 3: Set MTU to 9000

```bash
# Set MTU on RDMA interfaces (adjust interface names as needed)
sudo ip link set dev enP2p1s0f0np0 mtu 9000

# Verify
ip link show enP2p1s0f0np0 | grep mtu
```

## Building NCCL Tests

Run on **both nodes**:

```bash
cd /tmp
rm -rf nccl-tests

# Clone NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests

# Build with MPI support
make MPI=1 MPI_HOME=/usr CUDA_HOME=/usr/local/cuda CC=mpicc CXX=mpicxx

# Install
sudo mkdir -p /usr/local/nccl-tests
sudo cp -r build/* /usr/local/nccl-tests/

# Verify
ls /usr/local/nccl-tests/ | grep all_reduce
/usr/local/nccl-tests/all_reduce_perf --help | head -5
```

## Testing Inter-Node Communication

### Step 1: Setup Passwordless SSH

On spark-01 (main hostname):

```bash
# Generate SSH key if needed
ssh-keygen -t ed25519

# Copy to spark-02-mesh (using mesh network IP)
ssh-copy-id geo@192.168.100.2

# Test
ssh geo@192.168.100.2 hostname
```

Repeat from spark-02 to spark-01-mesh:
```bash
ssh-copy-id geo@192.168.100.1
ssh geo@192.168.100.1 hostname
```

Add to ~/.ssh/config for convenience:
```bash
Host spark-01-mesh
    Hostname 192.168.100.1
    User geo

Host spark-02-mesh
    Hostname 192.168.100.2
    User geo
```

### Step 2: Run NCCL Performance Test

From spark-01 (using the mesh network):

```bash
# Set environment to avoid X11 issues
export DISPLAY=""

# Run inter-node all-reduce test using IP addresses
mpirun -np 2 -H 192.168.100.1,192.168.100.2 \
  --mca btl_tcp_if_include enP2p1s0f0np0 \
  --mca oob_tcp_if_include enP2p1s0f0np0 \
  -x DISPLAY="" \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_HCA=roceP2p1s0f0 \
  -x NCCL_SOCKET_IFNAME=enP2p1s0f0np0 \
  -x NCCL_NET_GDR_LEVEL=5 \
  /usr/local/nccl-tests/all_reduce_perf -b 8 -e 512M -f 2 -g 1
```

Note: Using IP addresses (192.168.100.1, 192.168.100.2) is more reliable than hostnames for MPI.

Expected results:
- Should see "NET/IB : Using [0]roceP2p1s0f0:1/RoCE"
- Large message bandwidth: ~12 GB/s
- No TCP fallback warnings

## Making Configuration Persistent

### Step 1: Persistent Network Configuration

On **both nodes**, create netplan configuration for persistent MTU:

```bash
# Check existing netplan files
ls /etc/netplan/

# If your RDMA interface isn't configured with MTU 9000, create/edit:
sudo tee /etc/netplan/90-rdma-mtu.yaml << 'EOF'
network:
  version: 2
  ethernets:
    enP2p1s0f0np0:
      renderer: NetworkManager
      mtu: 9000
EOF

sudo chmod 600 /etc/netplan/90-rdma-mtu.yaml
sudo netplan apply
```

### Step 2: Disable Unused Interfaces

Create service to disable unused NICs on **both nodes**:

```bash
# Create systemd service
sudo tee /etc/systemd/system/disable-unused-nics.service << 'EOF'
[Unit]
Description=Disable Unused RDMA NICs
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/disable-unused-nics.sh

[Install]
WantedBy=multi-user.target
EOF

# Create the script
sudo tee /usr/local/bin/disable-unused-nics.sh << 'EOF'
#!/bin/bash
# Force unused interfaces down
ip link set dev enp1s0f0np0 down 2>/dev/null
ip link set dev enp1s0f1np1 down 2>/dev/null
ip link set dev enP2p1s0f1np1 down 2>/dev/null
echo "Disabled unused NICs at $(date)"
EOF

sudo chmod +x /usr/local/bin/disable-unused-nics.sh
sudo systemctl daemon-reload
sudo systemctl enable disable-unused-nics.service
sudo systemctl start disable-unused-nics.service
```

### Step 3: RDMA Optimization Service

Create optimization service on **both nodes**:

```bash
# Create systemd service
sudo tee /etc/systemd/system/rdma-optimize.service << 'EOF'
[Unit]
Description=RDMA Performance Optimization
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/rdma-optimize.sh

[Install]
WantedBy=multi-user.target
EOF

# Create optimization script
sudo tee /usr/local/bin/rdma-optimize.sh << 'EOF'
#!/bin/bash
# RDMA Performance Optimization

# Set network buffer sizes
sysctl -w net.core.rmem_max=268435456
sysctl -w net.core.wmem_max=268435456

# Ensure MTU is 9000 on active interface
ip link set dev enP2p1s0f0np0 mtu 9000 2>/dev/null

# Try to load optional peer memory modules
modprobe nv_peer_mem 2>/dev/null || true
modprobe nvidia-peermem 2>/dev/null || true

echo "RDMA optimization applied at $(date)"
EOF

sudo chmod +x /usr/local/bin/rdma-optimize.sh
sudo systemctl daemon-reload
sudo systemctl enable rdma-optimize.service
sudo systemctl start rdma-optimize.service
```

### Step 4: NCCL Environment Variables

Set NCCL environment variables on **both nodes**:

```bash
# Create NCCL environment file
sudo tee /etc/profile.d/nccl-env.sh << 'EOF'
# NCCL Environment Variables for Multi-node GPU
export NCCL_IB_HCA=roceP2p1s0f0
export NCCL_SOCKET_IFNAME=enP2p1s0f0np0
export NCCL_NET_GDR_LEVEL=5
# Optional: Uncomment for debugging
# export NCCL_DEBUG=INFO
EOF

sudo chmod +x /etc/profile.d/nccl-env.sh

# Load for current session
source /etc/profile.d/nccl-env.sh
```

### Step 5: Create Test Script

Create test script on **both nodes**:

```bash
tee ~/test_rdma_cluster.sh << 'EOF'
#!/bin/bash
echo "==================================="
echo "RDMA Cluster Configuration Test"
echo "==================================="
echo ""

# Check environment variables
echo "1. NCCL Environment Variables:"
echo "   NCCL_IB_HCA=$NCCL_IB_HCA"
echo "   NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "   NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL"
echo ""

# Check network settings
echo "2. Network Settings:"
echo -n "   Buffer sizes: "
sysctl net.core.rmem_max net.core.wmem_max | grep -oE "[0-9]+" | tail -2 | xargs
echo -n "   MTU on enP2p1s0f0np0: "
ip link show enP2p1s0f0np0 | grep -oE "mtu [0-9]+" | awk '{print $2}'
echo ""

# Check interface status
echo "3. Interface Status:"
for iface in enp1s0f0np0 enP2p1s0f0np0; do
    state=$(ip link show $iface 2>/dev/null | grep -oE "state [A-Z]+" | awk '{print $2}')
    echo "   $iface: ${state:-not found}"
done
echo ""

# Quick RDMA test
echo "4. Quick RDMA Test (if on spark-01):"
# Note: spark-01 is the main hostname, spark-01-mesh is for the 192.168.100.x network
if [[ "$(hostname)" =~ ^spark-01 ]]; then
    echo "   Running bandwidth test on mesh network..."
    mpirun -np 2 -H 192.168.100.1,192.168.100.2 \
        --mca btl_tcp_if_include enP2p1s0f0np0 \
        --mca oob_tcp_if_include enP2p1s0f0np0 \
        -x DISPLAY="" \
        /usr/local/nccl-tests/all_reduce_perf -b 256M -e 512M -f 2 -g 1 2>&1 | \
        grep -E "512M|Avg bus"
else
    echo "   (Run from spark-01 for RDMA test)"
fi
EOF

chmod +x ~/test_rdma_cluster.sh
```

## Verification

### Final Test

After completing all steps on both nodes:

1. Run the test script on both nodes:
```bash
~/test_rdma_cluster.sh
```

2. From spark-01, run full bandwidth test:
```bash
mpirun -np 2 -H 192.168.100.1,192.168.100.2 \
  --mca btl_tcp_if_include enP2p1s0f0np0 \
  --mca oob_tcp_if_include enP2p1s0f0np0 \
  -x DISPLAY="" \
  /usr/local/nccl-tests/all_reduce_perf -b 256M -e 512M -f 2 -g 1 2>&1 | \
  grep -E "512M|Avg bus"
```

Expected output:
```
# Avg bus bandwidth    : ~12.04 GB/s
```

### After Reboot

The configuration should persist after reboot. To verify:

1. Reboot both nodes
2. Run `~/test_rdma_cluster.sh` on both nodes
3. Run the bandwidth test from spark-01

## Troubleshooting

### Common Issues and Solutions

#### 1. SSH Authorization Errors
```bash
# Fix SSH key permissions
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# Ensure passwordless SSH using mesh network IPs
ssh-copy-id geo@192.168.100.2  # from spark-01
ssh-copy-id geo@192.168.100.1  # from spark-02
```

#### 2. NCCL Falls Back to TCP
- Check RDMA device is active: `rdma link show`
- Verify correct interface names in NCCL_IB_HCA
- Ensure MTU 9000 is set end-to-end
- Check firewall isn't blocking RDMA ports

#### 3. Low Bandwidth
- Verify MTU 9000: `ip link show enP2p1s0f0np0 | grep mtu`
- Check cable quality and connection
- Ensure no other traffic on the RDMA network
- Verify GPUDirect is enabled: `nvidia-smi topo -m`

#### 4. MPI Connection Issues
- Disable firewall temporarily for testing: `sudo ufw disable`
- Specify network interface explicitly with `--mca` options
- Use IP addresses (192.168.100.1, 192.168.100.2) instead of hostnames for reliability
- Check /etc/hosts has correct mesh network entries:
  ```
  192.168.100.1   spark-01-mesh
  192.168.100.2   spark-02-mesh
  ```

#### 5. Build Errors
- For MPI headers not found: Use `CC=mpicc CXX=mpicxx` in make command
- For NCCL headers not found: Install libnccl-dev package
- For CUDA issues: Verify CUDA_HOME=/usr/local/cuda is correct

## Next Steps

With the cluster configured, you can now:

1. **Run Distributed PyTorch Training**
```python
# Example PyTorch distributed initialization
import torch.distributed as dist
dist.init_process_group(backend='nccl')
```

2. **Deploy vLLM for Distributed Inference**
```bash
# On spark-01 (using mesh network for coordination)
vllm serve model_name \
  --tensor-parallel-size 2 \
  --distributed-init-method tcp://192.168.100.1:29500

# On spark-02 (connecting to spark-01-mesh)
vllm serve model_name \
  --tensor-parallel-size 2 \
  --distributed-init-method tcp://192.168.100.1:29500
```

3. **Scale Your Workloads**
- Use the 12 GB/s bandwidth for model parallelism
- Distribute large models across nodes
- Implement data parallel training

## Performance Expectations

With this setup, you should achieve:
- **Small messages (< 1KB)**: Low bandwidth due to overhead
- **Medium messages (1MB-100MB)**: 2-10 GB/s
- **Large messages (> 100MB)**: 11-12+ GB/s
- **Average bandwidth**: ~3.6-3.7 GB/s across all sizes

## Additional Resources

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [NVIDIA Collective Communications Library](https://github.com/NVIDIA/nccl)
- [NCCL Tests Repository](https://github.com/NVIDIA/nccl-tests)

## Notes

- This setup uses a single cable between nodes
- The configuration focuses on `roceP2p1s0f0` interface
- Unused interfaces are disabled to prevent interference
- All settings persist across reboots
- Environment tested on NVIDIA DGX Spark with Ubuntu 24.04 ARM64

---
*Configuration tested and verified to achieve 12+ GB/s inter-node GPU bandwidth*