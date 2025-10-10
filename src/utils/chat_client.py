#!/usr/bin/env python3
"""
Simple CLI Chat Client for vLLM Server

A lightweight chat client that connects to a running vLLM server
and maintains conversational context.
"""

import argparse
import json
import sys
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    print("Error: requests library is required. Install with: pip install requests")
    sys.exit(1)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.styles import Style
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.live import Live
    from rich.spinner import Spinner
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ChatClient:
    """Simple chat client for vLLM OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8000", model: str = None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.session = requests.Session()
        
        # Initialize rich console if available
        self.console = Console() if RICH_AVAILABLE else None
        
        # Test connection
        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to the vLLM server."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                if self.console:
                    self.console.print(f"[green]✓[/green] Connected to vLLM server at {self.base_url}")
                else:
                    print(f"✓ Connected to vLLM server at {self.base_url}")
            else:
                if self.console:
                    self.console.print(f"[yellow]⚠[/yellow] Server responded with status {response.status_code}")
                else:
                    print(f"⚠ Server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            if self.console:
                self.console.print(f"[red]❌[/red] Cannot connect to server: {e}")
                self.console.print(f"Make sure the vLLM server is running at {self.base_url}")
            else:
                print(f"❌ Cannot connect to server: {e}")
                print(f"Make sure the vLLM server is running at {self.base_url}")
            sys.exit(1)

    def _get_available_models(self) -> List[str]:
        """Get list of available models from the server."""
        try:
            response = self.session.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
            else:
                print(f"⚠ Could not fetch models: HTTP {response.status_code}")
                return []
        except Exception as e:
            print(f"⚠ Could not fetch models: {e}")
            return []

    def chat(self, message: str, temperature: float = 0.7, max_tokens: int = 512,
             stream: bool = False) -> str:
        """Send a chat message and get response."""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})

        # Prepare request
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        try:
            if self.console and not stream:
                with self.console.status("[bold green]Thinking...", spinner="dots") as status:
                    response = self.session.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        stream=stream
                    )
            else:
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=stream
                )

            if response.status_code != 200:
                error_msg = f"Server error: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text[:200]}"
                
                if self.console:
                    self.console.print(f"[red]❌ {error_msg}[/red]")
                else:
                    print(f"❌ {error_msg}")
                return error_msg

            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_regular_response(response)

        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg

    def _handle_regular_response(self, response) -> str:
        """Handle non-streaming response."""
        try:
            data = response.json()
            choice = data['choices'][0]
            assistant_message = choice['message']['content']

            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            # Display with rich formatting
            if self.console:
                # Check if response contains markdown-like content
                if any(marker in assistant_message for marker in ['**', '*', '`', '#', '-', '1.']):
                    markdown = Markdown(assistant_message)
                    self.console.print(Panel(markdown, title="[bold blue]🤖 Assistant[/bold blue]", border_style="blue"))
                else:
                    self.console.print(Panel(assistant_message, title="[bold blue]🤖 Assistant[/bold blue]", border_style="blue"))
            else:
                print(f"Assistant: {assistant_message}")

            return assistant_message

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            error_msg = f"Invalid response format: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg

    def _handle_streaming_response(self, response) -> str:
        """Handle streaming response."""
        full_content = ""
        try:
            if self.console:
                # Use rich Live for streaming display
                with Live(console=self.console, refresh_per_second=10) as live:
                    current_text = Text("", style="blue")
                    live.update(Panel(current_text, title="[bold blue]🤖 Assistant (streaming)[/bold blue]", border_style="blue"))
                    
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                data = line[6:]
                                if data == '[DONE]':
                                    break

                                try:
                                    chunk = json.loads(data)
                                    if chunk['choices'][0]['finish_reason'] is None:
                                        delta = chunk['choices'][0]['delta'].get('content', '')
                                        full_content += delta
                                        current_text.plain = full_content
                                        live.update(Panel(current_text, title="[bold blue]🤖 Assistant (streaming)[/bold blue]", border_style="blue"))
                                except json.JSONDecodeError:
                                    continue
            else:
                # Fallback for no rich
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break

                            try:
                                chunk = json.loads(data)
                                if chunk['choices'][0]['finish_reason'] is None:
                                    delta = chunk['choices'][0]['delta'].get('content', '')
                                    print(delta, end='', flush=True)
                                    full_content += delta
                            except json.JSONDecodeError:
                                continue
                print()  # New line after streaming

            # Add to history
            self.conversation_history.append({"role": "assistant", "content": full_content})
            return full_content

        except Exception as e:
            error_msg = f"Streaming error: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        if self.console:
            self.console.print("[green]🧹 Conversation history cleared[/green]")
        else:
            print("🧹 Conversation history cleared")

    def show_history(self) -> None:
        """Show conversation history."""
        if not self.conversation_history:
            if self.console:
                self.console.print("[dim]📝 No conversation history[/dim]")
            else:
                print("📝 No conversation history")
            return

        if self.console:
            table = Table(title="📝 Conversation History", show_header=True, header_style="bold magenta")
            table.add_column("Turn", style="cyan", no_wrap=True, width=4)
            table.add_column("Role", style="bold", width=10)
            table.add_column("Content", style="white", overflow="fold")
            
            for i, msg in enumerate(self.conversation_history, 1):
                role = msg['role'].title()
                content = msg['content']
                # Truncate long content for display
                if len(content) > 100:
                    content = content[:97] + "..."
                table.add_row(str(i), role, content)
            
            self.console.print(table)
        else:
            print("\n📝 Conversation History:")
            print("-" * 40)
            for i, msg in enumerate(self.conversation_history, 1):
                role = msg['role'].title()
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"{i}. {role}: {content}")
            print("-" * 40)


def create_prompt_session():
    """Create an enhanced prompt session if available."""
    if PROMPT_TOOLKIT_AVAILABLE:
        style = Style.from_dict({
            'prompt': '#00aa00 bold',
        })
        return PromptSession(
            history=InMemoryHistory(),
            style=style
        )
    else:
        return None


def show_welcome(console: Optional[Console], model: str, url: str) -> None:
    """Show welcome message."""
    if console:
        welcome_text = f"""
[bold green]🤖 vLLM Chat Client[/bold green]

[blue]Configuration:[/blue]
• Server: {url}
• Model: {model}

[dim]Commands: /help, /clear, /history, /quit[/dim]
[dim]Type your message and press Enter to chat![/dim]
        """
        console.print(Panel(welcome_text.strip(), title="Welcome", border_style="green"))
    else:
        print("🤖 vLLM Chat Client")
        print(f"Server: {url}")
        print(f"Model: {model}")
        print("Commands: /help, /clear, /history, /quit")
        print("-" * 50)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Simple CLI chat client for vLLM server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='vLLM server URL'
    )

    parser.add_argument(
        '--model',
        help='Model name (if not specified, will try to detect from server)'
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
        default=512,
        help='Maximum tokens to generate'
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming responses'
    )

    args = parser.parse_args()

    # Initialize client
    client = ChatClient(base_url=args.url, model=args.model)

    # Auto-detect model if not specified
    if not client.model:
        models = client._get_available_models()
        if models:
            client.model = models[0]
            if client.console:
                client.console.print(f"[green]✓[/green] Auto-selected model: {client.model}")
            else:
                print(f"✓ Auto-selected model: {client.model}")
        else:
            if client.console:
                client.console.print("[red]❌ No models available on server[/red]")
            else:
                print("❌ No models available on server")
            sys.exit(1)

    # Show welcome
    show_welcome(client.console, client.model, args.url)

    # Create prompt session
    session = create_prompt_session()

    try:
        while True:
            try:
                if session:
                    # Enhanced input with history
                    user_input = session.prompt("You: ").strip()
                else:
                    # Basic input
                    user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    cmd = user_input[1:].lower()
                    if cmd in ['quit', 'exit', 'q']:
                        if client.console:
                            client.console.print("[yellow]👋 Goodbye![/yellow]")
                        else:
                            print("👋 Goodbye!")
                        break
                    elif cmd == 'help':
                        if client.console:
                            help_table = Table(title="Available Commands")
                            help_table.add_column("Command", style="cyan", no_wrap=True)
                            help_table.add_column("Description", style="white")
                            help_table.add_row("/help", "Show this help message")
                            help_table.add_row("/clear", "Clear conversation history")
                            help_table.add_row("/history", "Show conversation history")
                            help_table.add_row("/quit", "Exit the chat")
                            client.console.print(help_table)
                        else:
                            print("Commands:")
                            print("  /help     - Show this help")
                            print("  /clear    - Clear conversation history")
                            print("  /history  - Show conversation history")
                            print("  /quit     - Exit the chat")
                        continue
                    elif cmd == 'clear':
                        client.clear_history()
                        continue
                    elif cmd == 'history':
                        client.show_history()
                        continue
                    else:
                        if client.console:
                            client.console.print(f"[red]Unknown command: {user_input}[/red]")
                        else:
                            print(f"Unknown command: {user_input}")
                        continue

                # Send message (don't print "Assistant:" prefix for non-streaming, it's handled in the response methods)
                if not args.stream:
                    response = client.chat(
                        user_input,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        stream=args.stream
                    )
                else:
                    # For streaming, we handle the display in the method
                    response = client.chat(
                        user_input,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        stream=args.stream
                    )

            except KeyboardInterrupt:
                if client.console:
                    client.console.print("\n[yellow]👋 Goodbye![/yellow]")
                else:
                    print("\n👋 Goodbye!")
                break
            except EOFError:
                break

    except Exception as e:
        if client.console:
            client.console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()