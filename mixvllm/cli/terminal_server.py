#!/usr/bin/env python3
"""
Standalone terminal server for MixVLLM.

This script runs a web-based terminal interface that can connect to any
OpenAI-compatible model server (vLLM, Ollama, etc.).

Usage:
    python -m mixvllm.cli.terminal_server --model-server-url http://localhost:8000

Or after installation:
    mixvllm-terminal-server --model-server-url http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mixvllm.inference.terminal_server import start_terminal_server
from mixvllm.inference.terminal_config import TerminalConfig


def main():
    parser = argparse.ArgumentParser(
        description="Standalone web terminal server for MixVLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-server-url',
        required=True,
        help='URL of the model server to connect to (e.g., http://localhost:8000)'
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Terminal server host address'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8888,
        help='Terminal server port'
    )

    parser.add_argument(
        '--no-auto-chat',
        action='store_true',
        help='Disable auto-starting chat in terminal'
    )

    args = parser.parse_args()

    # Create terminal configuration
    config = TerminalConfig(
        enabled=True,
        host=args.host,
        port=args.port,
        auto_start_chat=not args.no_auto_chat
    )

    print("🚀 Starting MixVLLM Terminal Server...")
    print(f"📋 Configuration:")
    print(f"  Model Server: {args.model_server_url}")
    print(f"  Terminal Host: {args.host}:{args.port}")
    print(f"  Auto-start Chat: {config.auto_start_chat}")
    print(f"\n🌐 Open your browser to: http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop the server\n")

    try:
        # Get project root for script access
        project_root = str(Path(__file__).parent.parent.parent)

        # Start the terminal server
        start_terminal_server(
            config=config,
            model_server_url=args.model_server_url,
            project_root=project_root
        )

    except KeyboardInterrupt:
        print("\n✓ Terminal server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting terminal server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()