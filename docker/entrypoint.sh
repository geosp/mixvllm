#!/bin/bash

# Strict error handling
set -e

# Print welcome message
echo "===== MIXVLLM CONTAINER ENTRYPOINT ====="
echo "Running as user: $(id)"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"

# Set up HuggingFace token if provided
if [ -n "$HF_TOKEN" ]; then
  echo "Setting up HuggingFace token..."
  mkdir -p ~/.huggingface
  echo -e "{\n  \"api_key\": \"$HF_TOKEN\"\n}" > ~/.huggingface/token
fi

# Verify that the commands are available
if ! command -v mixvllm-serve &>/dev/null; then
  echo "WARNING: mixvllm-serve not found in PATH. Installation may have failed."
  echo "PATH: $PATH"
  echo "Checking installed packages:"
  python -m pip list
  
  echo "Attempting to reinstall package..."
  cd /app/mixvllm
  uv pip install -e .
  
  echo "Adding scripts directory to PATH as a fallback..."
  export PATH=$PATH:/app/mixvllm
  chmod +x /app/mixvllm/mixvllm-serve /app/mixvllm/mixvllm-chat
fi

# Simple argument handling
if [ "$#" -eq 0 ] || [ "$1" = "tail" ] && [ "$2" = "-f" ] && [ "$3" = "/dev/null" ]; then
  # Default behavior: keep container running
  echo "Container is running in idle mode."
  echo "Use 'docker exec -it mixvllm-server /app/mixvllm/mixvllm-serve' to start the server."
  echo "Or specify a command in docker-compose.yml."
  
  # Keep container running
  exec tail -f /dev/null
else
  # Run whatever command was provided
  echo "Running command: $@"
  exec "$@"
fi