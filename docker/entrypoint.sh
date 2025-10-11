#!/bin/bash

# Strict error handling
set -e

# Activate virtual environment if it exists
if [ -f "/home/mixvllm/.venv/bin/activate" ]; then
  echo "Activating virtual environment..."
  . /home/mixvllm/.venv/bin/activate
fi

# Print welcome message
echo "===== MIXVLLM CONTAINER ENTRYPOINT ====="
echo "Running as user: $(id)"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"
echo "Python executable: $(which python)"

# Set up HuggingFace token if provided
if [ -n "$HF_TOKEN" ]; then
  echo "Setting up HuggingFace token..."
  mkdir -p ~/.huggingface
  echo -e "{\n  \"api_key\": \"$HF_TOKEN\"\n}" > ~/.huggingface/token
fi

# Verify that the commands are available
if [ ! -f "/app/mixvllm/serve" ] || [ ! -f "/app/mixvllm/chat" ]; then
  echo "ERROR: Convenience scripts not found. Installation may have failed."
  echo "PATH: $PATH"
  echo "Checking for scripts:"
  ls -la /app/mixvllm/
  
  echo "Adding scripts directory to PATH as a fallback..."
  export PATH=$PATH:/app/mixvllm
fi

# Simple argument handling
if [ "$#" -eq 0 ] || [ "$1" = "tail" ] && [ "$2" = "-f" ] && [ "$3" = "/dev/null" ]; then
  # Default behavior: keep container running
  echo "Container is running in idle mode."
  echo "Use 'docker exec -it mixvllm-server /app/mixvllm/serve --model <model-name>' to start the server."
  echo "Or specify a command in docker-compose.yml."
  
  # Keep container running
  exec tail -f /dev/null
else
  # Run whatever command was provided
  echo "Running command: $@"
  exec "$@"
fi