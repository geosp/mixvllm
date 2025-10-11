#!/usr/bin/env python3
"""CLI entry point for vLLM model serving.

This script provides a command-line interface to serve vLLM models
with configurable parameters for GPU usage, server settings, and
model configuration.
"""

import argparse
import sys
from pathlib import Path

# Add mixvllm to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mixvllm.inference.config import ServeConfig
from mixvllm.inference.server import start_server


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Serve vLLM models with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    # Model options
    parser.add_argument(
        '--model',
        type=str,
        help='HuggingFace model name or path (required if no --config)'
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Trust remote code for custom models'
    )

    # Inference options
    parser.add_argument(
        '--gpus',
        type=int,
        choices=[1, 2],
        help='Number of GPUs for tensor parallelism'
    )
    parser.add_argument(
        '--gpu-memory',
        type=float,
        help='GPU memory utilization (0.1-1.0)'
    )
    parser.add_argument(
        '--max-model-len',
        type=int,
        help='Maximum model context length'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        help='Data type for model weights'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        help='Quantization method (awq, gptq, or null)'
    )

    # Server options
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Server host address'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Server port number'
    )

    # Generation defaults
    parser.add_argument(
        '--temperature',
        type=float,
        help='Sampling temperature (0.0-2.0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        help='Top-p sampling parameter (0.0-1.0)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--presence-penalty',
        type=float,
        help='Presence penalty'
    )
    parser.add_argument(
        '--frequency-penalty',
        type=float,
        help='Frequency penalty'
    )

    args = parser.parse_args()

    # Load base configuration
    if args.config:
        try:
            config = ServeConfig.from_yaml(args.config)
            print(f"✓ Loaded configuration from {args.config}")
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            sys.exit(1)
    else:
        # Create minimal config - requires model
        if not args.model:
            print("❌ Error: --model is required when not using --config")
            print("\nExamples:")
            print("  uv run python serve_model.py --model microsoft/Phi-3-mini-4k-instruct --gpus 1")
            print("  uv run python serve_model.py --config configs/phi3-mini.yaml")
            sys.exit(1)

        config = ServeConfig(
            model={'name': args.model},
            inference={},
            server={}
        )
        print("✓ Using default configuration with CLI overrides")

    # Merge CLI arguments
    cli_args = vars(args)
    cli_args.pop('config', None)  # Remove config file arg

    try:
        config = config.merge_cli_args(**cli_args)
        print("✓ Configuration merged successfully")
    except Exception as e:
        print(f"❌ Error merging configuration: {e}")
        sys.exit(1)

    # Start the server
    try:
        start_server(config)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()