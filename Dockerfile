# Universal CUDA + UV Base Image
# Provides NVIDIA CUDA 12.8 runtime with Python 3.11 and UV package manager
# Suitable for vLLM, MCP servers, and any GPU-accelerated Python applications

FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

LABEL maintainer="Geovanny Fajardo <gffajardo@gmail.com>"
LABEL description="Universal CUDA 12.8 + Python 3.11 + UV base image for GPU workloads"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
# - python3.11: Main Python interpreter
# - python3.11-dev: Development headers for compiling Python extensions
# - python3-pip: Package installer (used to install uv)
# - git: Required for installing packages from git repositories
# - curl: For downloading resources
# - build-essential: GCC, G++, Make - required for compiling native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python and python3
# This ensures 'python' and 'python3' commands use Python 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv (Universal Python package manager)
# uv is a fast, modern Python package installer and runner
# https://github.com/astral-sh/uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Expose common ports
# 8000: Default vLLM/model server port
# 8888: Web terminal port
# 3000: Common MCP server port
EXPOSE 8000 8888 3000

# Set environment variables for CUDA
# These help CUDA applications find the correct libraries
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Default entrypoint uses uv for running commands
# This can be overridden in docker-compose or docker run
# Examples:
#   docker run image uv run python script.py
#   docker run image uv pip install package
ENTRYPOINT ["uv"]

# Default command (can be overridden)
CMD ["--help"]
