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

**401 Unauthorized Errors:**
- Set `HF_TOKEN` environment variable with your HuggingFace token
- For gated models, request access on the HuggingFace model page
- Verify token has read permissions: `huggingface-cli whoami`
- Some models require accepting terms/conditions on HuggingFace

## Authentication

Some models require authentication to access from HuggingFace. If you encounter `401 Unauthorized` errors, you need to:

### HuggingFace Token Setup

1. **Get a token**: Visit https://huggingface.co/settings/tokens to create an access token
2. **Set environment variable**:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. **Or login via CLI**:
   ```bash
   huggingface-cli login
   ```

### Gated Models

Some models (especially from OpenAI, Meta, etc.) are **gated repositories** that require:
- ✅ Valid HuggingFace account
- ✅ Explicit access approval on the model page
- ✅ Proper authentication token

**Example with authentication:**
```bash
HF_TOKEN=$HF_TOKEN uv run mixvllm-serve --config configs/gpt-oss-20b.yaml
```

**Models that may require authentication:**
- `openai/gpt-oss-20b` (gated)
- `meta-llama/Llama-2-*` (gated)
- `meta-llama/Llama-3-*` (gated)

**Public models (no auth required):**
- `microsoft/Phi-3-mini-4k-instruct`
- Most Microsoft and Google models

## Model Serving

Serve vLLM models with the `serve_model.py` script, which provides an OpenAI-compatible API server.

### Basic Usage

```bash
# Serve Phi-3 Mini on single GPU (no auth required)
uv run mixvllm-serve --model microsoft/Phi-3-mini-4k-instruct --gpus 1

# Serve Llama 2 70B with tensor parallelism (requires HF_TOKEN)
HF_TOKEN=$HF_TOKEN uv run mixvllm-serve --model meta-llama/Llama-2-70b-hf --gpus 2 --trust-remote-code
```

### Using Configuration Files

```bash
# Use predefined configurations
uv run mixvllm-serve --config configs/phi3-mini.yaml          # No auth required
uv run mixvllm-serve --config configs/llama-7b.yaml           # May require HF_TOKEN
HF_TOKEN=$HF_TOKEN uv run mixvllm-serve --config configs/llama-70b-tp2.yaml  # Requires HF_TOKEN
HF_TOKEN=$HF_TOKEN uv run mixvllm-serve --config configs/gpt-oss-20b.yaml    # Requires HF_TOKEN

# Override config with CLI options
uv run mixvllm-serve --config configs/phi3-mini.yaml --port 8080
```

### Advanced Options

```bash
HF_TOKEN=$HF_TOKEN uv run mixvllm-serve \
  --model meta-llama/Llama-2-70b-hf \
  --gpus 2 \
  --gpu-memory 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --temperature 0.8 \
  --max-tokens 1024
```

### API Usage

Once running, the server provides an OpenAI-compatible API:

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Chat Client

The `mixvllm-chat` command provides a CLI chat interface for interactive conversations with your served models. It features rich terminal formatting and enhanced input handling similar to modern CLI applications.

```bash
# Install dependencies (if not already done)
uv sync

# Start chatting with default settings
uv run mixvllm-chat

# Connect to specific server and model
uv run mixvllm-chat --base-url http://localhost:8000 --model microsoft/Phi-3-mini-4k-instruct

# Enable streaming responses
uv run mixvllm-chat --stream --temperature 0.8
```

### Chat Client Features

- **Rich Terminal UI**: Beautiful formatting with colors, panels, and markdown rendering
- **Conversation Context**: Maintains chat history for coherent conversations
- **Command Support**: `/help`, `/clear`, `/history`, `/quit`
- **Enhanced Input**: History-based auto-completion and navigation (with prompt_toolkit)
- **Streaming Support**: Real-time response streaming with live updates
- **Model Auto-detection**: Automatically detects available models from server
- **Error Handling**: Clear error messages with appropriate formatting

### Dependencies

The chat client uses these optional libraries for enhanced UI:
- `rich`: Beautiful terminal formatting and colors
- `prompt_toolkit`: Enhanced input with history and completion
- `requests`: HTTP client for API calls

If these libraries are not available, the client falls back to basic text output.

### Example Chat Session

```
✓ Connected to vLLM server at http://localhost:8000
✓ Auto-selected model: microsoft/Phi-3-mini-4k-instruct

╭─ Welcome ─────────────────────────────────────────────────────────────────╮
│                                                                            │
│ 🤖 vLLM Chat Client                                                        │
│                                                                            │
│ Configuration:                                                             │
│ • Server: http://localhost:8000                                            │
│ • Model: microsoft/Phi-3-mini-4k-instruct                                  │
│                                                                            │
│ Commands: /help, /clear, /history, /quit                                   │
│ Type your message and press Enter to chat!                                 │
│                                                                            │
╰────────────────────────────────────────────────────────────────────────────╯

You: Hello! How are you today?
╭─ 🤖 Assistant ─────────────────────────────────────────────────────────────╮
│ Hello! I'm doing well, thank you for asking. I'm here and ready to help   │
│ you with any questions or tasks you might have. How can I assist you      │
│ today?                                                                     │
╰────────────────────────────────────────────────────────────────────────────╯

You: Tell me about machine learning
╭─ 🤖 Assistant ─────────────────────────────────────────────────────────────╮
│ Machine learning is a fascinating field that involves teaching computers  │
│ to learn from data and make predictions or decisions without being        │
│ explicitly programmed for each specific task. It's a subset of artificial │
│ intelligence that focuses on algorithms and statistical models that can   │
│ improve their performance as they are exposed to more data.               │
│                                                                            │
│ There are several main types of machine learning:                          │
│                                                                            │
│ 1. **Supervised Learning**: The algorithm learns from labeled training    │
│    data to make predictions on new, unseen data. Examples include          │
│    classification (like spam detection) and regression (like predicting    │
│    house prices).                                                          │
│                                                                            │
│ 2. **Unsupervised Learning**: The algorithm finds patterns in data        │
│    without labeled examples. This includes clustering (grouping similar    │
│    data points) and dimensionality reduction.                              │
│                                                                            │
│ 3. **Reinforcement Learning**: An agent learns through trial and error by │
│    interacting with an environment, receiving rewards or penalties for     │
│    actions.                                                                │
│                                                                            │
│ Machine learning has applications in many fields including computer        │
│ vision, natural language processing, recommendation systems, autonomous    │
│ vehicles, medical diagnosis, and financial trading.                        │
╰────────────────────────────────────────────────────────────────────────────╯

You: /history
╭─ 📝 Conversation History ──────────────────────────────────────────────────╮
│ ┏━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
│ ┃ Turn ┃ Role         ┃ Content                                         ┃ │
│ ┡━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ │
│ │ 1    │ User         │ Hello! How are you today?                       │
│ │ 2    │ Assistant    │ Hello! I'm doing well, thank you for asking. ... │
│ │ 3    │ User         │ Tell me about machine learning                  │
│ │ 4    │ Assistant    │ Machine learning is a fascinating field that... │
│ └──────┴──────────────┴─────────────────────────────────────────────────┘ │
╰────────────────────────────────────────────────────────────────────────────╯

You: /quit
👋 Goodbye!
```

## Enhanced Chat Client with MCP Tools

The `mixvllm-chat` command provides an advanced chat client with MCP (Model Context Protocol) tool integration, enabling the LLM to call external tools during conversations.

### Features

- **MCP Tool Integration**: Weather queries and other MCP tools
- **Tool Discovery Display**: Shows available MCP tools on startup
- **Dual Modes**: Simple chat or agent mode with tool calling
- **Rich Terminal UI**: Enhanced formatting with panels and colors
- **Conversation Context**: Maintains chat history
- **Streaming Support**: Real-time response streaming
- **Command System**: `/help`, `/clear`, `/history`, `/mcp`, `/quit`

### Installation

Install additional dependencies for MCP support:

```bash
uv sync
```

### Usage

**Note**: Since vLLM serves only one model at a time, the `--model` parameter is optional. The client will automatically detect and use the model currently loaded on the server.

#### Simple Chat Mode (Default)

```bash
# Basic chat with vLLM server (auto-detects model)
uv run mixvllm-chat

# Connect to specific server (auto-detects model)
uv run mixvllm-chat --base-url http://localhost:8000

# Specify model explicitly (optional)
uv run mixvllm-chat --base-url http://localhost:8000 --model microsoft/Phi-3-mini-4k-instruct
```

#### MCP Agent Mode

```bash
# Enable MCP tools for weather queries (auto-detects model)
uv run mixvllm-chat --enable-mcp

# Full configuration with custom MCP config
uv run mixvllm-chat \
  --enable-mcp \
  --mcp-config configs/mcp_servers.yaml \
  --base-url http://localhost:8000 \
  --stream \
  --temperature 0.8
```

### MCP Tools Available

When MCP mode is enabled, the following tools are available:

- **Weather Queries**: Get current weather, forecasts, and historical data
- **Location Support**: Supports city names and coordinates
- **Units**: Celsius or Fahrenheit temperature units

### Example MCP Conversation

```
✓ Connected to vLLM server at http://localhost:8000
✓ Auto-selected model: microsoft/Phi-3-mini-4k-instruct
✓ MCP tools enabled (2 tools available)

╭─ Welcome ─────────────────────────────────────────────────────────────────╮
│ 🤖 Enhanced vLLM Chat Client (with MCP tools)                             │
│                                                                           │
│ Configuration:                                                            │
│ • Server: http://localhost:8000                                           │
│ • Model: microsoft/Phi-3-mini-4k-instruct                                 │
│ • MCP Tools: Enabled                                                      │
│                                                                           │
│ Available MCP Tools (2):                                                  │
│ • weather_get_hourly_weather - Get hourly weather forecast for a location│
│   using Open-Meteo API (Weather information and forecasts)               │
│ • weather_geocode_location - Get coordinates and timezone information for│
│   a location. (Weather information and forecasts)                         │
│                                                                           │
│ Commands: /help, /clear, /history, /mcp, /quit                            │
│ Type your message and press Enter to chat!                                │
╰────────────────────────────────────────────────────────────────────────────╯

You: What's the weather like in New York?
╭─ 🌤️ Assistant (with tools) ───────────────────────────────────────────────╮
│ The user is asking about the weather in New York. I should use the        │
│ weather_get_weather tool to get current weather information.              │
│                                                                           │
│ Tool Call: weather_get_weather(location="New York", units="celsius")      │
│                                                                           │
│ Tool Result: [weather] Weather for New York: 22°C, Partly Cloudy, Wind 5  │
│ km/h                                                                     │
│                                                                           │
│ Current weather in New York: 22°C with partly cloudy conditions and light │
│ winds at 5 km/h.                                                          │
╰────────────────────────────────────────────────────────────────────────────╯

You: /mcp
╭─ 🔧 MCP Integration Status ───────────────────────────────────────────────╮
│ ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓ │
│ ┃ Server  ┃ Status                                        ┃ Tools       ┃ │
│ ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩ │
│ │ weather │ ✓ Connected (2 tools)                         │ get_hourly_ │
│ │         │                                               │ weather,    │
│ │         │                                               │ geocode_loc │
│ │         │                                               │ ation       │
│ └─────────┴───────────────────────────────────────────────┴─────────────┘ │
╰────────────────────────────────────────────────────────────────────────────╯
```

