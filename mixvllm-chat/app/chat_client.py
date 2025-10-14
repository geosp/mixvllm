#!/usr/bin/env python3
"""
Standalone chat client script.

This script provides a minimal chat client that uses the existing
app.client components instead of reimplementing functionality.
"""

import argparse
import sys
import os
import logging
import asyncio

# Import the existing client components
from .client.config import ChatConfig
from .client.chat_client import ChatClient

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style


def create_prompt_session():
    """Create an enhanced prompt session with history and styling."""
    style = Style.from_dict({
        'prompt': 'bold cyan',
    })
    history_file = os.path.expanduser("~/.mixvllm_chat_history")
    return PromptSession(
        history=FileHistory(history_file),
        style=style,
        message="You: "
    )


def main():
    """Main CLI function using the existing ChatClient components."""
    parser = argparse.ArgumentParser(
        description="Enhanced CLI chat client for vLLM server with MCP tool support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base-url',
        default="http://localhost:8000",
        help='vLLM server base URL'
    )
    parser.add_argument(
        '--model',
        help='Model name (if not specified, will try to detect from server)'
    )
    parser.add_argument(
        '--enable-mcp',
        action='store_true',
        help='Enable MCP (Model Context Protocol) tools'
    )
    parser.add_argument(
        '--mcp-config',
        help='Path to MCP servers configuration file'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=4096,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming responses'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Updated logging configuration to log to a file
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='mixvllm-chat.log',  # Log to file
            filemode='w'  # Overwrite the log file on each run
        )
        logging.debug("Debug mode enabled")

    try:
        # Create configuration using the existing ChatConfig
        config = ChatConfig.from_args(args)

        # Initialize the existing ChatClient (handles all components)
        client = ChatClient(config)

        # Auto-detect model if not specified
        if not client.config.model:
            models = client.get_available_models()
            if models:
                client.set_model(models[0])
                if client.ui_manager.console:
                    client.ui_manager.show_success(f"Auto-selected model: {client.config.model}")
                else:
                    print(f"✓ Auto-selected model: {client.config.model}")
            else:
                if client.ui_manager.console:
                    client.ui_manager.show_error("No models available on server")
                else:
                    print("❌ No models available on server")
                sys.exit(1)

        # Show welcome message
        tools_count = len(client.tool_manager.mcp_tools) if hasattr(client.tool_manager, 'mcp_tools') else 0
        client.ui_manager.show_welcome(client.config.model, tools_count)

        # Setup enhanced input
        session = create_prompt_session()

        # Main chat loop using the existing client's chat method
        while True:
            try:
                user_input = session.prompt().strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    cmd = user_input[1:].lower()

                    if cmd in ['quit', 'exit', 'q']:
                        if client.ui_manager.console:
                            client.ui_manager.console.print("[yellow]👋 Goodbye![/yellow]")
                        else:
                            print("👋 Goodbye!")
                        break

                    elif cmd == 'help':
                        client.ui_manager.show_help()
                        continue

                    elif cmd == 'clear':
                        client.clear_history()
                        continue

                    elif cmd == 'history':
                        client.show_history()
                        continue

                    elif cmd == 'mcp' and client.config.enable_mcp:
                        client.show_mcp_status()
                        continue

                    else:
                        client.ui_manager.show_error(f"Unknown command: {user_input}")
                        continue

                # Send message to LLM using the existing client
                client.chat(user_input)

            except KeyboardInterrupt:
                if client.ui_manager.console:
                    client.ui_manager.console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
                else:
                    print("\nInterrupted. Type 'quit' to exit.")
                continue
            except EOFError:
                break

        if client.ui_manager.console:
            client.ui_manager.console.print("[bold blue]Chat session ended.[/bold blue]")
        else:
            print("Chat session ended.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()