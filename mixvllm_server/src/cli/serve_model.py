#!/usr/bin/env python3
"""CLI entry point for vLLM model serving.

This script provides a command-line interface to serve vLLM models
with configurable parameters for GPU usage, server settings, and
model configuration.

The CLI supports two modes of operation:
1. Configuration file mode: Load settings from a YAML file with optional CLI overrides
2. Direct mode: Specify all parameters via command-line arguments

Example Usage:
    # Using a configuration file
    $ python serve_model.py --config configs/phi3-mini.yaml

    # Direct specification with CLI arguments
    $ python serve_model.py --model microsoft/Phi-3-mini-4k-instruct --gpus 1

    # Configuration file with CLI overrides
    $ python serve_model.py --config configs/phi3-mini.yaml --port 8080

Architecture:
    This module acts as the CLI entry point and orchestrates:
    - Argument parsing and validation
    - Configuration loading from YAML or CLI args
    - Configuration merging (base config + CLI overrides)
    - Server startup via the inference.server module
"""

import argparse
import sys
from pathlib import Path

# Add mixvllm to path
# This allows the script to import mixvllm modules regardless of installation location
# by adding the parent directory to Python's module search path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..inference.config import ServeConfig
from ..inference.server import start_server


def main() -> None:
    """Main CLI function that orchestrates the model serving workflow.

    This function performs the following operations in sequence:
    1. Parses command-line arguments using argparse
    2. Loads configuration from YAML file or creates minimal config
    3. Validates required parameters (model name if no config file)
    4. Merges CLI arguments with base configuration
    5. Starts the vLLM server with the final configuration

    The function implements a configuration hierarchy:
        YAML file config (base) <- CLI arguments (overrides)

    Exit Codes:
        0: Success (server starts successfully)
        1: Error (configuration loading, validation, or server startup failure)

    Raises:
        SystemExit: Exits with code 1 on any configuration or startup error
    """
    # ========================================================================
    # STEP 1: Argument Parser Setup
    # ========================================================================
    # Create argument parser with automatic default value display
    # The ArgumentDefaultsHelpFormatter shows default values in help text
    parser = argparse.ArgumentParser(
        description="Serve vLLM models with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ========================================================================
    # STEP 2: Define Command-Line Arguments
    # ========================================================================

    # ------------------------------------------------------------------------
    # Configuration File Option
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--config',
        type=str,  # Expects a string path
        help='Path to YAML configuration file'
    )
    # What it does: Loads a YAML file containing pre-configured settings
    # Why use it: Easier than specifying many CLI args; enables version-controlled configs
    # Example: --config configs/phi3-mini.yaml

    # ------------------------------------------------------------------------
    # Model Options
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--model',
        type=str,  # Expects a string: HuggingFace model ID or local path
        help='HuggingFace model name or path (required if no --config)'
    )
    # What it does: Specifies which LLM to load
    # Format options:
    #   - HuggingFace ID: "microsoft/Phi-3-mini-4k-instruct"
    #   - Local path: "/path/to/model/directory"
    # Required: Yes, unless --config provides it
    # Example: --model microsoft/Phi-3-mini-4k-instruct

    parser.add_argument(
        '--trust-remote-code',
        action='store_true',  # Boolean flag: True if present, False if absent
        help='Trust remote code for custom models'
    )
    # What it does: Allows execution of custom Python code from model repository
    # When needed: Some models (like Phi-3) include custom modeling code
    # Security: Only use with trusted models (code could be malicious)
    # Default: False (safer, but some models won't load)
    # Example: --trust-remote-code

    # ------------------------------------------------------------------------
    # Inference Options (GPU and Model Configuration)
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--gpus',
        type=int,  # Expects an integer
        choices=[1, 2],  # Validation: only 1 or 2 allowed
        help='Number of GPUs for tensor parallelism'
    )
    # What it does: Enables tensor parallelism across multiple GPUs
    # How it works: Splits model layers across GPUs to fit larger models
    # When to use:
    #   - 1 GPU: Model fits in single GPU memory (simpler, faster)
    #   - 2 GPUs: Model too large for 1 GPU, or need more memory
    # Note: More GPUs = communication overhead between devices
    # Example: --gpus 2

    parser.add_argument(
        '--gpu-memory',
        type=float,  # Expects a decimal number
        help='GPU memory utilization (0.1-1.0)'
    )
    # What it does: Controls what fraction of GPU memory vLLM can use
    # Range: 0.1 (10%) to 1.0 (100%)
    # How to choose:
    #   - 0.9 (90%): Recommended default, leaves room for CUDA overhead
    #   - 0.7-0.8: Safe if running other GPU processes
    #   - 1.0: Maximum performance, but may cause OOM errors
    # Trade-off: Higher = more KV cache = larger batch sizes = better throughput
    # Example: --gpu-memory 0.9

    parser.add_argument(
        '--max-model-len',
        type=int,  # Expects an integer (number of tokens)
        help='Maximum model context length'
    )
    # What it does: Sets the maximum sequence length (context window)
    # Impact:
    #   - Longer context = can process more text at once
    #   - Longer context = more GPU memory needed
    # Typical values:
    #   - 2048: Small context, saves memory
    #   - 4096: Standard for many models
    #   - 8192+: Long context, needs significant VRAM
    # Note: Cannot exceed model's trained context length
    # Example: --max-model-len 4096

    parser.add_argument(
        '--dtype',
        type=str,  # Expects a string data type name
        help='Data type for model weights'
    )
    # What it does: Sets the precision for model weight storage
    # Options:
    #   - 'float32': Full precision (32-bit) - highest quality, most memory
    #   - 'float16': Half precision (16-bit) - good balance
    #   - 'bfloat16': Brain float16 - better range than float16, newer GPUs
    #   - 'auto': Let vLLM choose based on model config
    # Trade-offs:
    #   - Lower precision = less memory, faster compute, tiny quality loss
    #   - Higher precision = more memory, slower, marginally better
    # Recommendation: bfloat16 for Ampere+ GPUs, float16 otherwise
    # Example: --dtype bfloat16

    parser.add_argument(
        '--quantization',
        type=str,  # Expects quantization method name or 'null'
        help='Quantization method (awq, gptq, or null)'
    )
    # What it does: Reduces model weights to lower bit precision (e.g., 4-bit)
    # Methods:
    #   - 'awq': Activation-aware Weight Quantization - preserves important weights
    #   - 'gptq': GPT Quantization - general purpose quantization
    #   - 'null' or None: No quantization (full model precision)
    # Benefits:
    #   - 4x smaller memory footprint (e.g., 13B model in ~4GB vs 16GB)
    #   - Faster inference on supported hardware
    #   - Small quality degradation (usually <2% accuracy loss)
    # Requirements: Model must be pre-quantized in that format
    # Example: --quantization awq

    # ------------------------------------------------------------------------
    # Server Options (Network Configuration)
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--host',
        type=str,  # Expects an IP address string
        default='0.0.0.0',  # Default: listen on all network interfaces
        help='Server host address'
    )
    # What it does: Sets the network interface to bind the server to
    # Options:
    #   - '0.0.0.0': Listen on all interfaces (accessible from network)
    #   - '127.0.0.1' or 'localhost': Only local machine access
    #   - Specific IP: Bind to specific network interface
    # Security: '0.0.0.0' exposes server to network - ensure firewall rules
    # Example: --host 127.0.0.1 (local only)

    parser.add_argument(
        '--port',
        type=int,  # Expects an integer port number
        default=8000,  # Default port
        help='Server port number'
    )
    # What it does: Sets the TCP port the HTTP server listens on
    # Valid range: 1024-65535 (avoid privileged ports <1024)
    # Common choices:
    #   - 8000: Default web dev port
    #   - 8080: Alternative HTTP port
    #   - Custom: Any unused port
    # Note: Port must not be in use by another service
    # Example: --port 8080

    # ------------------------------------------------------------------------
    # Generation Defaults (Text Generation Parameters)
    # ------------------------------------------------------------------------
    # These parameters control how the model generates text responses
    # They can be overridden per-request via the API

    parser.add_argument(
        '--temperature',
        type=float,  # Expects a decimal number
        help='Sampling temperature (0.0-2.0)'
    )
    # What it does: Controls randomness in text generation
    # How it works: Scales the logits before sampling
    # Values:
    #   - 0.0: Deterministic (always picks highest probability token)
    #   - 0.7: Balanced creativity and coherence (good default)
    #   - 1.0: Unmodified probabilities
    #   - 1.5+: Very creative but potentially incoherent
    # Use cases:
    #   - Low (0.1-0.3): Factual tasks, code generation
    #   - Medium (0.7-1.0): Creative writing, conversations
    #   - High (1.2-2.0): Experimental, very creative outputs
    # Example: --temperature 0.7

    parser.add_argument(
        '--max-tokens',
        type=int,  # Expects an integer
        help='Maximum tokens to generate'
    )
    # What it does: Limits the length of generated responses
    # Unit: Tokens (roughly 0.75 words in English)
    # Purpose:
    #   - Prevents runaway generation
    #   - Controls API costs
    #   - Ensures timely responses
    # Typical values:
    #   - 100-200: Short responses
    #   - 500-1000: Medium responses
    #   - 2000+: Long-form content
    # Note: Generation may stop earlier due to stop sequences
    # Example: --max-tokens 512

    parser.add_argument(
        '--top-p',
        type=float,  # Expects a decimal between 0 and 1
        help='Top-p sampling parameter (0.0-1.0)'
    )
    # What it does: Nucleus sampling - considers only top tokens with cumulative probability p
    # How it works: Sorts tokens by probability, keeps only top % that sum to p
    # Values:
    #   - 0.9: Consider top 90% probability mass (good default)
    #   - 1.0: Consider all tokens (no filtering)
    #   - 0.5: Very conservative, only most likely tokens
    # Comparison to temperature:
    #   - Temperature: Flattens/sharpens entire distribution
    #   - Top-p: Truncates long tail of unlikely tokens
    # Recommendation: Use top_p OR temperature, not both aggressively
    # Example: --top-p 0.9

    parser.add_argument(
        '--top-k',
        type=int,  # Expects an integer
        help='Top-k sampling parameter'
    )
    # What it does: Limits sampling to the k most probable tokens
    # How it works: Sorts tokens by probability, keeps only top k
    # Values:
    #   - 40-50: Balanced (common default)
    #   - 1: Greedy (always pick most probable)
    #   - 100+: More diverse
    # Difference from top-p:
    #   - Top-k: Fixed number of tokens
    #   - Top-p: Variable number (probability-based)
    # Note: Often used together with top_p for better control
    # Example: --top-k 40

    parser.add_argument(
        '--presence-penalty',
        type=float,  # Expects a decimal (can be negative)
        help='Presence penalty'
    )
    # What it does: Penalizes tokens that have already appeared in the text
    # Range: -2.0 to 2.0
    # Effect:
    #   - Positive (e.g., 0.5): Encourages new topics, reduces repetition
    #   - 0.0: No penalty (default)
    #   - Negative: Encourages staying on topic
    # How it works: Subtracts penalty from logits of any token that appeared
    # Difference from frequency_penalty: Only cares if token appeared (not how many times)
    # Use case: Reducing repetitive phrases, encouraging topic diversity
    # Example: --presence-penalty 0.5

    parser.add_argument(
        '--frequency-penalty',
        type=float,  # Expects a decimal (can be negative)
        help='Frequency penalty'
    )
    # What it does: Penalizes tokens based on how often they've appeared
    # Range: -2.0 to 2.0
    # Effect:
    #   - Positive (e.g., 0.5): Reduces repetition proportional to frequency
    #   - 0.0: No penalty (default)
    #   - Negative: Encourages repetition of common patterns
    # How it works: Penalty increases with each occurrence of a token
    # Difference from presence_penalty: Scales with frequency (stronger for repeated tokens)
    # Use case: Preventing word/phrase repetition in longer text
    # Example: --frequency-penalty 0.3

    # ========================================================================
    # STEP 3: Parse Arguments
    # ========================================================================
    # Parse all command-line arguments into an args namespace object
    # Each argument becomes an attribute: args.model, args.gpus, etc.
    args = parser.parse_args()

    # ========================================================================
    # STEP 4: Load Base Configuration
    # ========================================================================
    # Two paths:
    # Path A: Load from YAML file (if --config provided)
    # Path B: Create minimal config from scratch (requires --model)

    if args.config:
        # PATH A: Configuration File Mode
        # ----------------------------------------------------------------
        # Load a YAML file containing structured configuration
        # This is the preferred method for production deployments
        try:
            # ServeConfig.from_yaml() reads and validates the YAML file
            # Returns a ServeConfig object with all settings populated
            config = ServeConfig.from_yaml(args.config)
            print(f"✓ Loaded configuration from {args.config}")
        except Exception as e:
            # File not found, invalid YAML, or validation errors
            print(f"❌ Error loading config file: {e}")
            sys.exit(1)  # Exit with error code
    else:
        # PATH B: Direct CLI Mode
        # ----------------------------------------------------------------
        # Create configuration from scratch using only CLI arguments
        # Requires at minimum a --model argument

        # Validation: --model is mandatory in this mode
        if not args.model:
            print("❌ Error: --model is required when not using --config")
            print("\nExamples:")
            print("  uv run python serve_model.py --model microsoft/Phi-3-mini-4k-instruct --gpus 1")
            print("  uv run python serve_model.py --config configs/phi3-mini.yaml")
            sys.exit(1)  # Exit with error code

        # Create minimal ServeConfig with just the model name
        # Other fields (inference, server) start as empty dicts
        # They'll be populated when CLI args are merged
        config = ServeConfig(
            model={'name': args.model},  # Required: model specification
            inference={},                 # Will be filled from CLI args
            server={}                     # Will be filled from CLI args
        )
        print("✓ Using default configuration with CLI overrides")

    # ========================================================================
    # STEP 5: Merge CLI Arguments with Base Configuration
    # ========================================================================
    # CLI arguments override values from the YAML config (if any)
    # This allows: base settings in YAML + environment-specific overrides via CLI

    # Convert args namespace to dictionary for processing
    # vars(args) returns: {'model': 'phi-3', 'gpus': 2, 'config': 'file.yaml', ...}
    cli_args = vars(args)

    # Remove the 'config' key since it's not a model/server setting
    # It's only used to locate the YAML file, not part of the final config
    cli_args.pop('config', None)

    try:
        # merge_cli_args() intelligently merges CLI args into the config:
        # - Converts CLI arg names to config structure (e.g., '--gpu-memory' -> inference.gpu_memory)
        # - Only overrides values that were explicitly provided on CLI (not None)
        # - Validates merged configuration
        # Returns a new ServeConfig object with merged settings
        config = config.merge_cli_args(**cli_args)
        print("✓ Configuration merged successfully")
    except Exception as e:
        # Validation errors, type mismatches, or invalid combinations
        print(f"❌ Error merging configuration: {e}")
        sys.exit(1)

    # ========================================================================
    # STEP 6: Start the vLLM Server
    # ========================================================================
    # Launch the HTTP server with the final configuration
    # This is a blocking call - it runs until interrupted (Ctrl+C)

    try:
        # start_server() performs:
        # 1. Initialize vLLM engine with model and inference settings
        # 2. Load model weights into GPU(s)
        # 3. Start HTTP server on specified host:port
        # 4. Register API endpoints (completions, chat, health checks)
        # 5. Begin processing requests
        start_server(config)

        # Note: This line is never reached in normal operation
        # The server runs indefinitely until terminated
    except KeyboardInterrupt:
        # User pressed Ctrl+C - graceful shutdown
        print("\n✓ Server stopped by user")
        sys.exit(0)
    except Exception as e:
        # Startup errors: GPU OOM, model loading failure, port in use, etc.
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


# ============================================================================
# Script Entry Point
# ============================================================================
if __name__ == '__main__':
    # This is the standard Python idiom for script entry points
    # What it does:
    #   - __name__ is a special variable set by Python
    #   - When script is run directly: __name__ == '__main__'
    #   - When script is imported: __name__ == 'mixvllm.cli.serve_model'
    #
    # Why use it:
    #   - Allows the file to be both a runnable script AND an importable module
    #   - Prevents main() from running when someone imports this file
    #
    # Without this guard:
    #   from mixvllm.cli.serve_model import ServeConfig  # Would start server!
    #
    # With this guard:
    #   from mix_vllm.cli.serve_model import ServeConfig  # Safe, no side effects
    #   python serve_model.py  # Runs main() and starts server
    main()