# vLLM Model Serving CLI

This directory contains the command-line interface for serving vLLM models with configurable parameters for GPU usage, server settings, and model configuration.

## Overview

The CLI provides a comprehensive interface to start vLLM servers with support for:

- **Multi-GPU Configuration**: Single or dual GPU setups
- **Model Management**: HuggingFace models or local paths
- **Server Configuration**: Host, port, and API settings
- **Performance Tuning**: Tensor parallelism, quantization, and memory optimization
- **Configuration Files**: YAML-based configuration for complex setups

## Architecture

- **`serve_model.py`** - Main CLI entry point with argument parsing and server initialization

## Dependencies

The CLI integrates with:
- **`mixvllm.inference.config`** - Configuration management
- **`mixvllm.inference.server`** - Server implementation

## Usage

### Basic Usage

```bash
# Serve a model with default settings
uv run mixvllm-serve --model microsoft/Phi-3-mini-4k-instruct

# Use a configuration file
uv run mixvllm-serve --config configs/phi3-mini.yaml
```

### Advanced Configuration

```bash
# Dual GPU setup with tensor parallelism
uv run mixvllm-serve \
  --model microsoft/Phi-3-mini-4k-instruct \
  --gpus 2 \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

### Configuration File

Create a YAML configuration file for complex setups:

```yaml
# configs/phi3-mini.yaml
model: microsoft/Phi-3-mini-4k-instruct
gpus: 2
tensor_parallel_size: 2
host: 0.0.0.0
port: 8000
max_model_len: 4096
gpu_memory_utilization: 0.9
trust_remote_code: true
```

## Command Line Options

### Model Options
- `--model`: HuggingFace model name or local path
- `--trust-remote-code`: Enable for custom models
- `--config`: Path to YAML configuration file

### GPU Configuration
- `--gpus`: Number of GPUs (1 or 2)
- `--tensor-parallel-size`: Tensor parallelism size
- `--gpu-memory-utilization`: GPU memory usage (0.0-1.0)

### Server Settings
- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 8000)
- `--max-model-len`: Maximum model sequence length

### Performance Options
- `--dtype`: Data type (auto, float16, bfloat16, float32)
- `--quantization`: Quantization method
- `--enforce-eager`: Disable CUDA graphs for debugging

## Integration

After starting the server, use the chat client to interact:

```bash
# Start server
uv run mixvllm-serve --model microsoft/Phi-3-mini-4k-instruct

# In another terminal, start chat client
uv run mixvllm-chat --enable-mcp --mcp-config configs/mcp_servers.yaml
```

## Examples

See the `configs/` directory for example configuration files:
- `phi3-mini.yaml` - Phi-3 mini model configuration
- `llama-70b-tp2.yaml` - LLaMA 70B with tensor parallelism
- `gpt-oss-20b.yaml` - GPT OSS model configuration