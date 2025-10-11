# Docker Deployment Guide

This directory contains Docker Compose configurations for running mixvllm in containerized environments.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU(s) with CUDA support
- Sufficient VRAM for your chosen model:
  - Phi-3 Mini: ~8GB (1 GPU)
  - GPT-OSS-20B: ~20GB (1-2 GPUs recommended)
  - Llama 70B: ~40GB+ (2 GPUs with tensor parallelism)

### Software Requirements
- Docker Engine 20.10+
- Docker Compose v2.0+
- NVIDIA Container Toolkit (nvidia-docker2)

#### Installing NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Verify installation:**
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. HuggingFace Authentication

For gated models (Llama, GPT-OSS, etc.), you need HuggingFace authentication.

**Option A: Login via CLI (Recommended)**
```bash
# Install HuggingFace CLI if not already installed
pip install huggingface-hub[cli]

# Login (token will be cached in ~/.cache/huggingface/token)
huggingface-cli login
```

**Option B: Manual token file**
```bash
# Create token file manually
echo "hf_your_token_here" > ~/.cache/huggingface/token
```

**Option C: Environment variable**
```bash
# Set HF_TOKEN in your shell
export HF_TOKEN=hf_your_token_here
```

The Docker container will automatically use the cached token via the volume mount.

### 2. Configure GPU Count

Edit [docker-compose.yml](docker-compose.yml) line 32 to match your hardware:

```yaml
- GPU_COUNT=2  # Change to 1 for single GPU, 2 for dual GPU, etc.
```

Or override at runtime:
```bash
GPU_COUNT=1 docker compose up
```

### 3. Start the Server

```bash
# From the docker/ directory
cd docker

# Start with default settings (gpt-oss-20b, 2 GPUs)
docker compose up

# Or customize
MODEL=phi3-mini GPU_COUNT=1 docker compose up
```

The server will:
- Download the model (first run only, cached afterwards)
- Load the model into GPU(s)
- Start the API server on http://localhost:8000
- Start the web terminal on http://localhost:8888 (if enabled)

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 5. Access Web Terminal

Open your browser to http://localhost:8888 for an interactive terminal with the chat client.

## Configuration Options

### Environment Variables

Customize behavior by setting environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `gpt-oss-20b` | Model config name (matches YAML filename in configs/) |
| `GPU_COUNT` | `2` | Number of GPUs for tensor parallelism |
| `HF_TOKEN` | (from cache) | HuggingFace token for gated models |
| `ENABLE_TERMINAL` | `--enable-terminal` | Set to empty string to disable web terminal |

### Examples

**Single GPU with Phi-3 Mini:**
```bash
MODEL=phi3-mini GPU_COUNT=1 docker compose up
```

**Dual GPU with Llama 70B:**
```bash
MODEL=llama-70b-tp2 GPU_COUNT=2 docker compose up
```

**Disable web terminal:**
```bash
ENABLE_TERMINAL="" docker compose up
```

**Custom HuggingFace token:**
```bash
HF_TOKEN=hf_custom_token docker compose up
```

## Available Models

Pre-configured models in `configs/`:

- `phi3-mini` - Microsoft Phi-3 Mini (4K context, ~8GB VRAM)
- `llama-7b` - Llama 2 7B (requires HF token)
- `llama-70b-tp2` - Llama 2 70B with 2-GPU tensor parallelism (requires HF token)
- `gpt-oss-20b` - OpenAI GPT-OSS 20B (requires HF token)

## Volumes

The compose file mounts two important volumes:

### Model Cache (`~/.cache/huggingface`)
- **Purpose**: Persistent model storage and HuggingFace authentication
- **Location**: `~/.cache/huggingface:/root/.cache/huggingface`
- **Contents**:
  - Downloaded model weights
  - HuggingFace token (from `huggingface-cli login`)
  - Model configs and tokenizers

**Benefits:**
- Models download once, persist across container restarts
- No need to pass `HF_TOKEN` as environment variable
- Faster startup after first run

## Ports

| Port | Service | Description |
|------|---------|-------------|
| 8000 | Model API | OpenAI-compatible REST API |
| 8888 | Web Terminal | Browser-based terminal with chat client |

## Build Approach

The Docker setup uses a hybrid approach:
- **Pre-built base image**: Uses `ghcr.io/geosp/mixvllm:latest` from GitHub Container Registry
- **Runtime git cloning**: Container clones the latest code from `https://github.com/geosp/mixvllm.git` when starting
- **Automatic updates**: Always runs the latest version without rebuilding the base image

### Key Features:
- **Lightweight base images**: CUDA runtime and dependencies pre-installed
- **Fresh code**: Clones latest repository code at container startup
- **GPU acceleration**: Full CUDA support with NVIDIA Container Toolkit
- **Flexible configuration**: Environment variables for model selection and GPU count

## Troubleshooting

### GPU Not Detected

**Error:** `CUDA is not available` or `No GPUs found`

**Solution:**
```bash
# Verify NVIDIA Container Toolkit is installed
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon.json includes nvidia runtime
cat /etc/docker/daemon.json
# Should contain:
# {
#   "runtimes": {
#     "nvidia": {
#       "path": "nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   }
# }
```

### Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce GPU count: `GPU_COUNT=1 docker compose up`
2. Use a smaller model: `MODEL=phi3-mini docker compose up`
3. Edit config file to reduce `gpu_memory_utilization` or `max_model_len`

### Authentication Errors

**Error:** `401 Unauthorized` or `Access to model requires authentication`

**Solutions:**
1. Verify HuggingFace login:
   ```bash
   huggingface-cli whoami
   ```
2. Check token file exists:
   ```bash
   cat ~/.cache/huggingface/token
   ```
3. Request access to gated models on HuggingFace website
4. Pass token explicitly:
   ```bash
   HF_TOKEN=hf_your_token docker compose up
   ```

### Port Already in Use

**Error:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution:**
Edit `docker-compose.yml` to use different ports:
```yaml
ports:
  - "8001:8000"  # Map host port 8001 to container port 8000
  - "8889:8888"  # Map host port 8889 to container port 8888
```

### Container Exits Immediately

**Solution:**
Check logs for errors:
```bash
docker compose logs mixvllm-server
```

Common issues:
- Missing HF_TOKEN for gated model
- Invalid model name
- Insufficient VRAM

## Advanced Usage

### Running in Background

```bash
# Start detached
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Custom Compose Files

Create additional compose files for different scenarios:

**docker-compose.dev.yml** - Development with code mounting:
```yaml
version: '3.8'
services:
  mixvllm-server:
    extends:
      file: docker-compose.yml
      service: mixvllm-server
    volumes:
      - ../:/app  # Mount source code
```

Use it:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Building Locally

If you want to build the image locally instead of using the pre-built one:

```bash
# Build from project root
cd ..
docker build -t mixvllm:local .

# Update docker-compose.yml to use local image
# image: mixvllm:local
```

## Production Considerations

### Security

- **Web Terminal**: Only enable in trusted environments (no authentication by default)
- **HF Token**: Use volume-mounted token file instead of environment variable
- **Network**: Consider running behind reverse proxy (nginx, traefik)
- **Firewall**: Restrict ports 8000/8888 to trusted networks

### Performance

- **Model Cache**: Use SSD/NVMe for `~/.cache/huggingface` volume
- **GPU Memory**: Monitor with `nvidia-smi` and adjust `gpu_memory_utilization` in configs
- **Batch Size**: vLLM automatically optimizes based on available memory

### Monitoring

Add health checks and monitoring:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 120s
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/geosp/mixvllm/issues
- Check main [README.md](../README.md) for project documentation
- vLLM docs: https://docs.vllm.ai/

## License

See main project [LICENSE](../LICENSE) file.
