# mixvllm_server

High-performance, configurable model serving backend for vLLM, supporting advanced GPU management, and robust configuration.

---

## Overview

`mixvllm_server` provides a modular Python backend for serving large language models (LLMs) using vLLM. It features:

- Flexible configuration via YAML and CLI
- GPU-aware model serving with tensor parallelism and quantization
- Pydantic-based config validation
- Subprocess management for server lifecycle

---

## Project Structure

```
mixvllm_server/
├── src/
│   ├── cli/
│   │   ├── serve_model.py      # CLI entry point for model serving
│   │   └── README.md           # CLI documentation
│   ├── inference/
│   │   ├── config.py           # Pydantic config models
│   │   ├── server.py           # vLLM server wrapper & lifecycle
│   │   ├── utils.py            # Config utilities & GPU detection
│   │   └── README.md           # Inference server documentation
│   └── config/                 # Example configs
└── pyproject.toml
```

---

## Key Components

### 1. Configuration System (`config.py`)
- Uses Pydantic for type-safe, validated configuration.
- Supports model, inference, server, and generation parameters.
- Loads from YAML, merges with CLI overrides.

### 2. Server Lifecycle (`server.py`)
- Builds vLLM CLI command from config.
- Starts/stops vLLM server as subprocess.
- Monitors health and logs output/errors.

### 3. CLI Integration (`serve_model.py`)
- Loads config from YAML and CLI.
- Merges/validates configuration.
- Starts server and monitors process.

### 4. GPU Detection (`utils.py`)
- Uses PyTorch to detect available GPUs and memory.
- Validates GPU availability before server start.

---

## Configuration

- YAML config supports all model, inference, server, and generation options.
- CLI arguments override YAML for flexible deployment.
- Example config:
  ```yaml
  model:
    name: "meta-llama/Llama-2-70b-hf"
    trust_remote_code: true
  inference:
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.9
    dtype: "bfloat16"
    quantization: "awq"
  server:
    host: "0.0.0.0"
    port: 8000
    workers: 1
  generation_defaults:
    temperature: 0.7
    max_tokens: 512
    top_p: 0.9
    top_k: 40
  ```

---

## Dependencies

- `pydantic` (config validation)
- `pyyaml` (YAML parsing)
- `torch` (GPU detection)
- `vllm` (model serving backend)

---

## Performance & Testing

- Efficient GPU memory management and tensor parallelism.
- Subprocess isolation for robust server control.
- Unit and integration tests for config and server.

---

## Future Enhancements

- Dynamic config reloading
- Health/metrics endpoints
- GPU utilization tracking

---
