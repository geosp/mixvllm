#!/bin/bash

# Strict error handling
set -e

# Print welcome message
echo "===== MIXVLLM CONTAINER ENTRYPOINT ====="
echo "Running as user: $(id)"
echo "Current directory: $(pwd)"
echo "PATH: $PATH"

# Display Python and UV versions
echo "Python version: $(python --version 2>&1)"
echo "UV version: $(uv --version 2>&1)"

# Check for HuggingFace token
if [ -n "$HF_TOKEN" ]; then
  echo "HuggingFace token detected. Configuring credentials..."
  mkdir -p ~/.huggingface
  echo -e "{\n  \"api_key\": \"$HF_TOKEN\"\n}" > ~/.huggingface/token
else
  echo "Warning: HF_TOKEN environment variable not set. Private models may not be accessible."
fi

# Environment check
echo "Checking environment and dependencies..."
if ! command -v mixvllm-serve &>/dev/null; then
  echo "ERROR: mixvllm-serve not found in PATH. Installation may have failed."
  echo "PATH: $PATH"
  echo "Attempting to reinstall dependencies..."
  
  # We should already be in the correct directory
  echo "Installing from $(pwd)..."
  uv pip install --user -e .
  
  if ! command -v mixvllm-serve &>/dev/null; then
    echo "ERROR: Installation failed. Please check logs."
    exit 1
  fi
fi

echo "MixVLLM environment is ready!"

# Check for command line arguments
if [ "$#" -eq 0 ]; then
  # Default behavior: just wait
  echo "Container is running and waiting for commands."
  echo "Use 'docker exec -it mixvllm-server mixvllm-serve' to start the server."
  echo "Or specify a command in docker-compose.yml."
elif [ "$1" = "tail" ] && [ "$2" = "-f" ] && [ "$3" = "/dev/null" ]; then
  # Default CMD from Dockerfile - just keep container running
  echo "Container is running in idle mode."
  echo "Use 'docker exec -it mixvllm-server mixvllm-serve' to start the server."
  exec "$@"
else
  # Run the provided command
  echo "Running command: $@"
  exec "$@"
fi