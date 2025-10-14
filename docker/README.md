# MixVLLM Terminal and vLLM Integration

This project provides a lightweight terminal server (`mixvllm-terminal`) that integrates seamlessly with the vLLM model server (`vllm/vllm-openai:latest`).

## Terminal Server Overview

The terminal server offers a web-based interface for interacting with the vLLM model server. It is designed to be lightweight and avoids heavy machine learning dependencies. Key features include:

- **Web Interface**: Built using `xterm.js` for the frontend and `terminado` for the backend.
- **Auto-Start**: Automatically connects to the model server when a terminal session begins.
- **Customizable**: Configurable host, port, and auto-start behavior.

## Integration with vLLM

The `docker-compose.yml` file integrates the terminal server with the vLLM model server. The `model-server` service uses the official `vllm/vllm-openai:latest` image, while the `terminal-server` service uses the `mixvllm-terminal` image. Communication between the two services is facilitated via HTTP, using the `MODEL_SERVER_URL` environment variable.

## How It Works

1. **Model Server**:
   - Runs the vLLM model server, exposing an OpenAI-compatible REST API on port 8000.
   - Supports GPU acceleration and tensor parallelism for efficient model inference.

2. **Terminal Server**:
   - Provides a web-based terminal interface on port 8888.
   - Connects to the model server to send and receive chat completions.

## Quick Start

Follow the steps in the `docker-compose.yml` file to start both the model server and terminal server. Once running, access the terminal interface at `http://localhost:8888` and the model API at `http://localhost:8000`.

## Environment Variable Configuration

### MODEL_SERVER_URL

The `MODEL_SERVER_URL` environment variable has been added to the `docker-compose.yml` file for the `terminal-server` service. This variable points to the `model-server` service, ensuring that the terminal server knows where to send requests. By default, it is set to:

```yaml
MODEL_SERVER_URL=http://model-server:8000
```

This allows the `DEFAULT_BASE_URL` in the `chat` script to dynamically adapt to the Docker Compose setup.