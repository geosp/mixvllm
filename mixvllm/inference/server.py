"""Server wrapper for vLLM model serving.

This module provides functions to validate the environment,
prepare server arguments, and start the vLLM OpenAI-compatible server.
"""

import argparse
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

from .config import ServeConfig
from .utils import get_gpu_info


def validate_environment(config: ServeConfig) -> None:
    """Validate the environment before starting the server.

    Args:
        config: Server configuration

    Raises:
        RuntimeError: If CUDA is not available
        ValueError: If insufficient GPUs
    """
    gpu_info = get_gpu_info()

    if not gpu_info['cuda_available']:
        raise RuntimeError(
            "CUDA is not available. GPU required for vLLM.\n"
            "Please ensure CUDA drivers are installed and GPUs are available."
        )

    if config.inference.tensor_parallel_size > gpu_info['gpu_count']:
        raise ValueError(
            f"Requested {config.inference.tensor_parallel_size} GPUs "
            f"but only {gpu_info['gpu_count']} available"
        )

    print("✓ Environment validation passed")
    print(f"  GPUs available: {gpu_info['gpu_count']}")
    for gpu in gpu_info['gpus']:
        print(f"    GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']} GB)")


def prepare_server_args(config: ServeConfig) -> List[str]:
    """Prepare command line arguments for vLLM server.

    Args:
        config: Server configuration

    Returns:
        List of command line arguments
    """
    args = [
        '--model', config.model.name,
        '--tensor-parallel-size', str(config.inference.tensor_parallel_size),
        '--gpu-memory-utilization', str(config.inference.gpu_memory_utilization),
        '--max-model-len', str(config.inference.max_model_len),
        '--host', config.server.host,
        '--port', str(config.server.port),
    ]

    if config.model.trust_remote_code:
        args.append('--trust-remote-code')

    if config.inference.dtype:
        args.extend(['--dtype', config.inference.dtype])

    if config.inference.quantization:
        args.extend(['--quantization', config.inference.quantization])

    return args


def start_terminal_server_thread(config: ServeConfig, project_root: str) -> threading.Thread:
    """Start the terminal server in a separate thread.

    Args:
        config: Server configuration
        project_root: Root directory of the project

    Returns:
        Thread running the terminal server
    """
    from .terminal_server import start_terminal_server

    # Build model server URL
    model_server_url = f"http://localhost:{config.server.port}"

    # Create thread
    terminal_thread = threading.Thread(
        target=start_terminal_server,
        args=(config.terminal, model_server_url, project_root),
        daemon=True,  # Daemon thread will exit when main thread exits
        name="TerminalServer"
    )

    # Start thread
    terminal_thread.start()

    return terminal_thread


def start_server(config: ServeConfig) -> None:
    """Start the vLLM server with the given configuration.

    Args:
        config: Server configuration

    Raises:
        RuntimeError: If server fails to start
    """
    print("\n🚀 Starting vLLM server...")

    # Validate environment
    validate_environment(config)

    # Prepare arguments
    args_list = prepare_server_args(config)

    print(f"📋 Server configuration:")
    print(f"  Model: {config.model.name}")
    print(f"  GPUs: {config.inference.tensor_parallel_size}")
    print(f"  GPU Memory: {config.inference.gpu_memory_utilization}")
    print(f"  Max Length: {config.inference.max_model_len}")
    print(f"  Host: {config.server.host}:{config.server.port}")

    # Start terminal server if enabled
    terminal_thread = None
    if config.terminal.enabled:
        try:
            # Get project root (directory containing mixvllm-chat script)
            project_root = str(Path(__file__).parent.parent.parent)

            print(f"\n🖥️  Terminal server enabled:")
            print(f"  Host: {config.terminal.host}:{config.terminal.port}")
            print(f"  Auto-start chat: {config.terminal.auto_start_chat}")

            terminal_thread = start_terminal_server_thread(config, project_root)

            # Give terminal server a moment to start
            import time
            time.sleep(1)

        except Exception as e:
            print(f"\n⚠️  Warning: Failed to start terminal server: {e}")
            print("    Continuing with model server only...")

    # Build command
    cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server'] + args_list

    print(f"\n⚡ Running command: {' '.join(cmd)}")
    print("\n" + "="*60)
    print("Server starting... Press Ctrl+C to stop")
    print("="*60 + "\n")

    try:
        # Start server
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed with exit code {e.returncode}")
        raise RuntimeError(f"vLLM server failed to start: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Server interrupted by user")
        return

    print("\n✅ Server stopped successfully")