#!/usr/bin/env python3
"""
Enhanced CLI Chat Client for vLLM Server with MCP Tool Support

A chat client that connects to a running vLLM server and supports MCP tools
for enhanced functionality like weather queries.
"""

import argparse
import json
import sys
from typing import List, Dict, Any, Optional
import os

# Add the parent directory to the path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class ChatClient:
    """Enhanced chat client for vLLM OpenAI-compatible API with MCP tool support."""

    def __init__(self, base_url: str = "http://localhost:8000", model: str = None,
                 enable_mcp: bool = False, ollama_url: str = "http://ollama.mixwarecs-home.net:11434"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.enable_mcp = enable_mcp and MCP_AVAILABLE
        self.ollama_url = ollama_url
        self.conversation_history: List[Dict[str, str]] = []
        self.session = requests.Session()

        # Initialize rich console if available
        self.console = Console() if RICH_AVAILABLE else None

        # Initialize MCP agent if enabled
        self.agent = None
        if self.enable_mcp:
            self._setup_mcp_agent()

        # Test connection
        self._test_connection()

    def _setup_mcp_agent(self):
        """Set up the MCP-enabled LangChain agent."""
        try:
            from mcp_tools import get_available_mcp_tools

            llm = ChatOllama(
                model="gpt-oss:20b",  # Use a capable model for tool calling
                base_url=self.ollama_url,
                temperature=0.1
            )

            tools = get_available_mcp_tools()
            self.agent = create_react_agent(llm, tools)

            if self.console:
                self.console.print("[green]✓[/green] MCP tools enabled (weather queries available)")
            else:
                print("✓ MCP tools enabled (weather queries available)")

        except Exception as e:
            if self.console:
                self.console.print(f"[yellow]⚠[/yellow] Failed to setup MCP agent: {e}")
                self.console.print("[dim]Falling back to simple chat mode[/dim]")
            else:
                print(f"⚠ Failed to setup MCP agent: {e}")
                print("Falling back to simple chat mode")
            self.enable_mcp = False

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
                if self.console:
                    self.console.print(f"[yellow]⚠[/yellow] Could not fetch models: HTTP {response.status_code}")
                else:
                    print(f"⚠ Could not fetch models: HTTP {response.status_code}")
                return []
        except Exception as e:
            if self.console:
                self.console.print(f"[yellow]⚠[/yellow] Could not fetch models: {e}")
            else:
                print(f"⚠ Could not fetch models: {e}")
            return []

    def chat(self, message: str, temperature: float = 0.7, max_tokens: int = 512,
             stream: bool = False) -> str:
        """Send a chat message and get response."""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})

        # Use MCP agent if enabled, otherwise use direct API
        if self.enable_mcp and self.agent:
            return self._chat_with_mcp(message, temperature, max_tokens, stream)
        else:
            return self._chat_direct(message, temperature, max_tokens, stream)

    def _chat_with_mcp(self, message: str, temperature: float, max_tokens: int, stream: bool) -> str:
        """Chat using MCP-enabled agent."""
        try:
            if self.console and not stream:
                with self.console.status("[bold green]Agent thinking...", spinner="dots") as status:
                    result = self.agent.invoke(
                        {"messages": [{"role": "user", "content": message}]}
                    )
            else:
                result = self.agent.invoke(
                    {"messages": [{"role": "user", "content": message}]}
                )

            final_message = result["messages"][-1].content

            # Add to history
            self.conversation_history.append({"role": "assistant", "content": final_message})

            # Display with rich formatting
            if self.console:
                if any(marker in final_message.lower() for marker in ['weather', 'temperature', 'forecast']):
                    # Weather responses get special formatting
                    self.console.print(Panel(final_message, title="[bold blue]🌤️ Assistant (with tools)[/bold blue]", border_style="blue"))
                elif "**" in final_message or "* " in final_message:
                    # Markdown responses
                    markdown = Markdown(final_message)
                    self.console.print(Panel(markdown, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
                else:
                    self.console.print(Panel(final_message, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
            else:
                print(f"Assistant: {final_message}")

            return final_message

        except Exception as e:
            error_msg = f"Agent error: {str(e)}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg

    def _chat_direct(self, message: str, temperature: float, max_tokens: int, stream: bool) -> str:
        """Chat using direct vLLM API calls."""
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

    def show_mcp_status(self) -> None:
        """Show MCP integration status."""
        if self.enable_mcp and self.agent:
            if self.console:
                from mcp_tools import get_mcp_tool_names, get_mcp_servers
                tools = get_mcp_tool_names()
                servers = get_mcp_servers()

                # Create a table showing servers and their status
                table = Table(title="🔧 MCP Integration Status")
                table.add_column("Server", style="cyan", no_wrap=True)
                table.add_column("Status", style="green")
                table.add_column("Tools", style="yellow")

                for server_name, server_info in servers.items():
                    # Test connection to get tool count
                    from mcp_client import test_mcp_connection
                    status = test_mcp_connection(server_name)

                    if status['status'] == 'connected':
                        status_text = f"✓ Connected ({status['tools_count']} tools)"
                        tools_text = ", ".join(status['tools']) if status['tools'] else "None"
                    else:
                        status_text = f"❌ {status['error'][:30]}..."
                        tools_text = "N/A"

                    table.add_row(server_name, status_text, tools_text)

                self.console.print(table)
            else:
                print("🔧 MCP Integration: Active")
                from mcp_tools import get_mcp_tool_names
                print(f"Available tools: {', '.join(get_mcp_tool_names())}")
        else:
            if self.console:
                self.console.print("[yellow]🔧 MCP Integration: Disabled[/yellow]")
                if not MCP_AVAILABLE:
                    self.console.print("[dim]Install langchain-ollama, langchain-core, and langgraph to enable MCP tools[/dim]")
                else:
                    self.console.print("[dim]Use --enable-mcp to activate MCP tools[/dim]")
            else:
                print("🔧 MCP Integration: Disabled")
                if not MCP_AVAILABLE:
                    print("Install langchain-ollama, langchain-core, and langgraph to enable MCP tools")
                else:
                    print("Use --enable-mcp to activate MCP tools")


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


def show_welcome(console: Optional[Console], model: str, url: str, mcp_enabled: bool) -> None:
    """Show welcome message."""
    if console:
        mode = "with MCP tools" if mcp_enabled else "simple chat"
        welcome_text = f"""
[bold green]🤖 Enhanced vLLM Chat Client[/bold green] [dim]({mode})[/dim]

[blue]Configuration:[/blue]
• Server: {url}
• Model: {model}
• MCP Tools: {'Enabled' if mcp_enabled else 'Disabled'}
"""

        # Add MCP tools information if enabled
        if mcp_enabled:
            try:
                from mcp_tools import get_available_mcp_tools, get_mcp_servers
                tools = get_available_mcp_tools()
                servers = get_mcp_servers()

                if tools:
                    welcome_text += f"\n[green]Available MCP Tools ({len(tools)}):[/green]\n"
                    for tool in tools:
                        # Get server name from tool name (format: server_toolname)
                        server_name = tool.name.split('_')[0] if '_' in tool.name else 'unknown'
                        server_info = servers.get(server_name, {})
                        server_desc = server_info.get('description', server_name)

                        # Get tool description (first line only for brevity)
                        tool_desc = tool.description.split('\n')[0] if tool.description else 'No description'
                        if len(tool_desc) > 60:
                            tool_desc = tool_desc[:57] + "..."

                        welcome_text += f"• [cyan]{tool.name}[/cyan] - {tool_desc} ([dim]{server_desc}[/dim])\n"
                else:
                    welcome_text += "\n[yellow]⚠ No MCP tools available[/yellow]\n"
            except Exception as e:
                welcome_text += f"\n[yellow]⚠ MCP tools error: {str(e)[:50]}...[/yellow]\n"

        welcome_text += f"""
[dim]Commands: /help, /clear, /history, /mcp, /quit[/dim]
[dim]Type your message and press Enter to chat![/dim]
        """
        console.print(Panel(welcome_text.strip(), title="Welcome", border_style="green"))
    else:
        mode = "with MCP tools" if mcp_enabled else "simple chat"
        print("🤖 Enhanced vLLM Chat Client ({mode})")
        print(f"Server: {url}")
        print(f"Model: {model}")
        print(f"MCP Tools: {'Enabled' if mcp_enabled else 'Disabled'}")

        # Add MCP tools information if enabled
        if mcp_enabled:
            try:
                from mcp_tools import get_available_mcp_tools, get_mcp_servers
                tools = get_available_mcp_tools()
                servers = get_mcp_servers()

                if tools:
                    print(f"\nAvailable MCP Tools ({len(tools)}):")
                    for tool in tools:
                        server_name = tool.name.split('_')[0] if '_' in tool.name else 'unknown'
                        server_info = servers.get(server_name, {})
                        server_desc = server_info.get('description', server_name)

                        # Get tool description (first line only for brevity)
                        tool_desc = tool.description.split('\n')[0] if tool.description else 'No description'
                        if len(tool_desc) > 60:
                            tool_desc = tool_desc[:57] + "..."

                        print(f"  • {tool.name} - {tool_desc} ({server_desc})")
                else:
                    print("\n⚠ No MCP tools available")
            except Exception as e:
                print(f"\n⚠ MCP tools error: {str(e)[:50]}...")

        print("Commands: /help, /clear, /history, /mcp, /quit")
        print("-" * 50)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Enhanced CLI chat client for vLLM server with MCP tool support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='vLLM server URL'
    )

    parser.add_argument(
        '--model',
        help='Model name (optional - auto-detected from server if not specified)'
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

    parser.add_argument(
        '--enable-mcp',
        action='store_true',
        help='Enable MCP tool integration (requires LangChain dependencies)'
    )

    parser.add_argument(
        '--ollama-url',
        default='http://ollama.mixwarecs-home.net:11434',
        help='Ollama server URL for MCP agent'
    )

    args = parser.parse_args()

    # Initialize client
    client = ChatClient(
        base_url=args.url,
        model=args.model,
        enable_mcp=args.enable_mcp,
        ollama_url=args.ollama_url
    )

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
    show_welcome(client.console, client.model, args.url, client.enable_mcp)

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
                            help_table.add_row("/mcp", "Show MCP integration status")
                            help_table.add_row("/quit", "Exit the chat")
                            client.console.print(help_table)
                        else:
                            print("Commands:")
                            print("  /help     - Show this help")
                            print("  /clear    - Clear conversation history")
                            print("  /history  - Show conversation history")
                            print("  /mcp      - Show MCP integration status")
                            print("  /quit     - Exit the chat")
                        continue
                    elif cmd == 'clear':
                        client.clear_history()
                        continue
                    elif cmd == 'history':
                        client.show_history()
                        continue
                    elif cmd == 'mcp':
                        client.show_mcp_status()
                        continue
                    else:
                        if client.console:
                            client.console.print(f"[red]Unknown command: {user_input}[/red]")
                        else:
                            print(f"Unknown command: {user_input}")
                        continue

                # Send message
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