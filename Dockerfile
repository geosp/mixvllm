# Universal CUDA + UV Base Image with non-root user
# Provides NVIDIA CUDA 12.8 runtime with Python 3.11 and UV package manager
# Automatically installs dependencies from pyproject.toml

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

LABEL maintainer="Geovanny Fajardo <gffajardo@gmail.com>"
LABEL description="MixVLLM - vLLM inference server with auto-dependency installation"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv (Universal Python package manager)
RUN pip install --no-cache-dir uv

# Create a non-root user with sudo privileges
RUN groupadd --gid 1000 mixvllm && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash mixvllm && \
    echo "mixvllm ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/mixvllm && \
    chmod 0440 /etc/sudoers.d/mixvllm

# Create app directories and set ownership
RUN mkdir -p /app/mixvllm && chown -R mixvllm:mixvllm /app

# Set working directory
WORKDIR /app/mixvllm

# Switch to non-root user
USER mixvllm

# Set environment variables for CUDA and virtual environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}:/home/mixvllm/.venv/bin:/home/mixvllm/.local/bin
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV VIRTUAL_ENV=/home/mixvllm/.venv

# Copy the local repository contents into the container
# We'll do this during the docker build with the context
COPY --chown=mixvllm:mixvllm . /app/mixvllm/

# The pyproject.toml already has the README.md reference removed
# No need to modify it further

# Create a virtual environment for the mixvllm user
RUN uv venv /home/mixvllm/.venv

# Activate the virtual environment and install the package in development mode
RUN . /home/mixvllm/.venv/bin/activate && \
    cd /app/mixvllm && \
    uv pip install -e .

# Make the convenience scripts executable
RUN chmod +x /app/mixvllm/serve /app/mixvllm/chat

# Add the scripts to the PATH
ENV PATH=${PATH}:/app/mixvllm

# Expose common ports
# 8000: Default vLLM/model server port
# 8888: Web terminal port
# 3000: Common MCP server port
EXPOSE 8000 8888 3000

# Verify the installation and path
RUN ls -la /app/mixvllm/serve && \
    ls -la /app/mixvllm/chat && \
    echo "PATH: $PATH"

# Verify and make the entrypoint script executable
RUN ls -la /app/mixvllm/docker/entrypoint.sh && \
    chmod +x /app/mixvllm/docker/entrypoint.sh

ENTRYPOINT ["/app/mixvllm/docker/entrypoint.sh"]

# Default command keeps the container running and waiting for instructions
# This allows you to execute commands via docker exec or specify a command in docker-compose
CMD ["tail", "-f", "/dev/null"]
