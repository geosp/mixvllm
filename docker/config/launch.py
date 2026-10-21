#!/usr/bin/env python3
"""
vLLM launcher - reads model config from registry YAML and launches vLLM.
Usage: MODEL_NAME=gpt-oss-20b python launch.py [--dry-run]

Multi-node (vLLM's built-in mp launcher, no Ray) is enabled by CLUSTER_NNODES > 1.
Every node runs this script with the same MODEL_NAME; rank 0 serves the API and
the other ranks run headless:
  CLUSTER_NNODES=2 CLUSTER_NODE_RANK=0|1 CLUSTER_MASTER_ADDR=192.168.100.1
  CLUSTER_MASTER_PORT=29501 (optional)
"""

import os
import shlex
import sys
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError:
    print("❌ PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


REGISTRY_PATH = Path(__file__).parent / "model_registry.yml"


def load_registry():
    """Load and parse the YAML registry."""
    if not REGISTRY_PATH.exists():
        print(f"❌ Registry not found: {REGISTRY_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(REGISTRY_PATH) as f:
        try:
            registry = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"❌ Failed to parse registry: {e}", file=sys.stderr)
            sys.exit(1)

    return registry


def list_available_models(registry):
    """Print available models."""
    models = registry.get("models", {})
    print("Available models:", file=sys.stderr)
    for name, config in models.items():
        desc = config.get("description", "N/A")
        print(f"  • {name}: {desc}", file=sys.stderr)


def get_model_config(registry, model_name):
    """Get config for a specific model."""
    models = registry.get("models", {})

    if model_name not in models:
        print(f"❌ Model '{model_name}' not found in registry", file=sys.stderr)
        list_available_models(registry)
        sys.exit(1)

    config = models[model_name]

    # Validate required fields
    if "model" not in config:
        print(f"❌ Missing 'model' field for {model_name}", file=sys.stderr)
        sys.exit(1)

    return config


def config_to_cli_args(config):
    """
    Convert config dict to vLLM CLI arguments.
    
    Rules:
    - Skip 'description' (metadata only)
    - Skip 'model' (passed positionally to `vllm serve`)
    - For boolean True: add flag with no value (e.g., --trust-remote-code)
    - For boolean False: skip it
    - For lists: add flag followed by each item (e.g., --served-model-name qwen spark)
    - For other values: add flag with value (e.g., --model openai/gpt-oss-20b)
    """
    args = []

    for key, value in config.items():
        # Skip metadata and the positional model
        if key in ("description", "model"):
            continue

        flag = f"--{key}"

        if isinstance(value, bool):
            # Boolean flag: only add if True
            if value:
                args.append(flag)
        elif isinstance(value, list):
            # Multi-value flag: one flag, several values
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            # Value flag: add both flag and value
            args.extend([flag, str(value)])

    return args


def cluster_cli_args():
    """
    Build multi-node args from CLUSTER_* env vars (empty list for single node).
    Rank 0 hosts the API server; other ranks add --headless.
    """
    nnodes = int(os.getenv("CLUSTER_NNODES", "1"))
    if nnodes <= 1:
        return []

    rank = int(os.getenv("CLUSTER_NODE_RANK", "0"))
    master_addr = os.getenv("CLUSTER_MASTER_ADDR")
    if not master_addr:
        print("❌ CLUSTER_MASTER_ADDR must be set when CLUSTER_NNODES > 1", file=sys.stderr)
        sys.exit(1)

    args = [
        "--distributed-executor-backend", "mp",
        "--nnodes", str(nnodes),
        "--node-rank", str(rank),
        "--master-addr", master_addr,
        "--master-port", os.getenv("CLUSTER_MASTER_PORT", "29501"),
    ]
    if rank > 0:
        args.append("--headless")

    return args


def main():
    # Parse CLI arguments
    dry_run = "--dry-run" in sys.argv

    # Get model name from environment variable
    model_name = os.getenv("MODEL_NAME")

    if not model_name:
        print("❌ MODEL_NAME environment variable not set", file=sys.stderr)
        print("Usage: MODEL_NAME=gpt-oss-20b [--dry-run] python launch.py", file=sys.stderr)
        sys.exit(1)

    # Load registry and get model config
    registry = load_registry()
    config = get_model_config(registry, model_name)

    # Convert config to CLI args
    cli_args = config_to_cli_args(config) + cluster_cli_args()

    # Build command (`python -m vllm.entrypoints.openai.api_server` is deprecated since vLLM 0.29)
    cmd = ["vllm", "serve", config["model"]]
    cmd.extend(cli_args)

    # Print what we're about to run
    mode = "DRY-RUN" if dry_run else "LAUNCH"
    print(f"🚀 {mode}: {model_name}")
    print(f"   Model URI: {config['model']}")
    print(f"   Registry: {REGISTRY_PATH}")
    print()
    print(f"▶️  Command:")
    print(f"   {shlex.join(cmd)}")  # shell-quoted, so it can be copy-pasted
    print()

    if dry_run:
        print("✅ Dry-run successful (registry valid)")
        sys.exit(0)

    # Launch vLLM
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"❌ vLLM exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"❌ Launch failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()