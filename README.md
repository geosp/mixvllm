# vLLM Development Environment

Development environment for vLLM with dual-GPU tensor parallelism support.

## Hardware

- **GPUs**: 2x NVIDIA GeForce RTX 3090 Ti (24GB VRAM each)
- **CUDA Version**: 12.8
- **Driver**: 570.172.08

## Setup

This project was initialized using `create_project.sh` at the root.

### Install Dependencies

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Or install without dev dependencies
uv sync
```

### Activate Environment

```bash
# Run commands with uv (recommended)
uv run python script.py

# Or activate the virtual environment
source .venv/bin/activate
```

## Project Structure

```
.
├── create_project.sh     # Project bootstrap script
├── .claude/              # Temporary and experimental code
│   ├── experiments/      # Model testing and prototyping
│   ├── benchmarks/       # Performance benchmarks
│   └── scratch/          # Quick tests and scratchpad
├── src/                  # Production code
│   ├── inference/        # vLLM inference wrappers
│   └── utils/            # Shared utilities
├── configs/              # Model and server configurations
└── tests/                # Test suite
```

## Quick Start

### 1. Test GPU Detection

```bash
uv run python .claude/experiments/test_gpu.py
```

### 2. Test vLLM Installation

```bash
uv run python .claude/experiments/test_vllm.py
```

### 3. Run a Model with Tensor Parallelism

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,  # Use both GPUs
    gpu_memory_utilization=0.90,
    trust_remote_code=True
)

outputs = llm.generate("Hello, my name is")
print(outputs[0].outputs[0].text)
```

## Configuration

See `configs/example_model.yaml` for a complete configuration template.

## Development

```bash
# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Auto-formatting
uv run black src/
```

## Tensor Parallelism Notes

With 2x RTX 3090 Ti (24GB each = 48GB total):
- Can run 70B models in FP16 (requires ~140GB, use quantization)
- Can run 70B models in 4-bit quantization comfortably
- Can run 34B models in FP16 easily
- Communication overhead between GPUs is minimal on PCIe 4.0

## Troubleshooting

**Out of Memory Errors:**
- Reduce `gpu_memory_utilization` (try 0.85 or 0.80)
- Use quantization (4-bit or 8-bit)
- Reduce `max_model_len`

**Slow Inference:**
- Check GPU utilization with `nvidia-smi`
- Verify both GPUs are being used
- Ensure PCIe link is running at full speed

