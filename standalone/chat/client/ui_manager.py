"""
UI management for displaying messages and status.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from .config import ChatConfig


class UIManager:
    """Manages user interface elements."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.console = Console() if Console is not None else None

    def show_welcome(self, model: str, tools_count: int = 0):
        """Show welcome message with clean formatting."""
        if self.console:
            from rich.panel import Panel
            from rich.text import Text

            welcome_text = Text()
            welcome_text.append("🤖 Enhanced vLLM Chat Client", style="bold blue")
            if self.config.enable_mcp:
                welcome_text.append(" (with MCP tools)", style="bold green")
            welcome_text.append("\n\n", style="")
            
            welcome_text.append("Configuration:\n", style="bold")
            welcome_text.append(f"• Server: {self.config.base_url}\n", style="")
            welcome_text.append(f"• Model: {model}\n", style="")
            if self.config.enable_mcp:
                welcome_text.append(f"• MCP Tools: Enabled ({tools_count} tools)\n", style="green")
            else:
                welcome_text.append("• MCP Tools: Disabled\n", style="")
            welcome_text.append(f"• Temperature: {self.config.temperature}\n", style="")
            welcome_text.append(f"• Max Tokens: {self.config.max_tokens}\n", style="")
            welcome_text.append(f"• Streaming: {'Enabled' if self.config.stream else 'Disabled'}\n", style="")

            welcome_text.append("\nCommands: /help, /clear, /history", style="dim")
            if self.config.enable_mcp:
                welcome_text.append(", /mcp", style="dim")
            welcome_text.append(", /quit\n", style="dim")
            welcome_text.append("Type your message and press Enter to chat!", style="italic")

            self.console.print(Panel(welcome_text, title="🚀 Welcome", border_style="blue"))

            if self.config.enable_mcp and tools_count > 0:
                # Show available tools with simple table
                from .utils.mcp_tools import get_available_mcp_tools
                tools = get_available_mcp_tools(self.config.mcp_config_path)
                if tools:
                    from rich.table import Table
                    table = Table(title="Available MCP Tools", show_header=True)
                    table.add_column("Tool", style="cyan", no_wrap=True)
                    table.add_column("Description", style="white", overflow="fold")

                    for tool in tools:
                        # Truncate long descriptions
                        desc = tool.description
                        if len(desc) > 80:
                            desc = desc[:77] + "..."
                        table.add_row(tool.name, desc)

                    self.console.print(table)
        else:
            print("🤖 Enhanced vLLM Chat Client" + (" (with MCP tools)" if self.config.enable_mcp else ""))
            print()
            print("Configuration:")
            print(f"• Server: {self.config.base_url}")
            print(f"• Model: {model}")
            if self.config.enable_mcp:
                print(f"• MCP Tools: Enabled ({tools_count} tools)")
            else:
                print("• MCP Tools: Disabled")
            print(f"• Temperature: {self.config.temperature}")
            print(f"• Max Tokens: {self.config.max_tokens}")
            print(f"• Streaming: {'Enabled' if self.config.stream else 'Disabled'}")
            print()
            print("Commands: /help, /clear, /history" + (", /mcp" if self.config.enable_mcp else "") + ", /quit")
            print("Type your message and press Enter to chat!")

    def show_status_message(self, message: str, status_type: str = "info"):
        """Show a status message with appropriate styling."""
        if self.console:
            from rich.text import Text
            
            # Choose emoji and color based on status type
            if status_type == "success":
                emoji = "✅"
                color = "green"
            elif status_type == "error":
                emoji = "❌"
                color = "red"
            elif status_type == "warning":
                emoji = "⚠️"
                color = "yellow"
            elif status_type == "info":
                emoji = "ℹ️"
                color = "blue"
            elif status_type == "tool":
                emoji = "🔧"
                color = "cyan"
            else:
                emoji = "•"
                color = "white"
            
            status_text = Text()
            status_text.append(f"{emoji} ", style=f"bold {color}")
            status_text.append(message, style=color)
            
            self.console.print(status_text)
        else:
            print(f"{message}")

    def show_thinking_indicator(self, message: str = "Thinking..."):
        """Show a thinking indicator for when the LLM is processing."""
        if self.console:
            from rich.text import Text
            
            thinking_text = Text()
            thinking_text.append("🤔 ", style="bold yellow")
            thinking_text.append(message, style="dim yellow italic")
            
            self.console.print(thinking_text)
        else:
            print(f"🤔 {message}")

    def show_tool_activity(self, tool_name: str, action: str = "calling"):
        """Show tool activity status."""
        if self.console:
            from rich.text import Text
            
            if action == "calling":
                emoji = "🔧"
                color = "cyan"
                message = f"Calling tool: {tool_name}"
            elif action == "success":
                emoji = "✅"
                color = "green"
                message = f"Tool completed: {tool_name}"
            elif action == "error":
                emoji = "❌"
                color = "red"
                message = f"Tool failed: {tool_name}"
            else:
                emoji = "•"
                color = "white"
                message = f"{action}: {tool_name}"
            
            tool_text = Text()
            tool_text.append(f"{emoji} ", style=f"bold {color}")
            tool_text.append(message, style=f"{color} italic")
            
            self.console.print(tool_text)
        else:
            print(f"🔧 {action}: {tool_name}")

    def show_mcp_status(self):
        """Show MCP integration status."""
        if self.config.enable_mcp:
            if self.console:
                from .utils.mcp_tools import get_mcp_tool_names, get_mcp_servers
                from .utils.mcp_client import test_mcp_connection
                tools = get_mcp_tool_names()
                servers = get_mcp_servers()

                # Create a table showing servers and their status
                table = Table(title="🔧 MCP Integration Status")
                table.add_column("Server", style="cyan", no_wrap=True)
                table.add_column("Status", style="green")
                table.add_column("Tools", style="yellow")

                for server_name, server_info in servers.items():
                    # Test connection to get tool count
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
                from .utils.mcp_tools import get_mcp_tool_names
                print(f"Available tools: {', '.join(get_mcp_tool_names())}")
        else:
            if self.console:
                self.console.print("[yellow]🔧 MCP Integration: Disabled[/yellow]")
                self.console.print("[dim]MCP tools require proper MCP server configuration[/dim]")
            else:
                print("🔧 MCP Integration: Disabled")
                print("MCP tools require proper MCP server configuration")

    def show_help(self):
        """Show help message."""
        if self.console:
            from rich.table import Table
            help_table = Table(title="Available Commands")
            help_table.add_column("Command", style="cyan", no_wrap=True)
            help_table.add_column("Description", style="white")
            help_table.add_row("/help", "Show this help message")
            help_table.add_row("/clear", "Clear conversation history")
            help_table.add_row("/history", "Show conversation history")
            if self.config.enable_mcp:
                help_table.add_row("/mcp", "Show MCP integration status")
            help_table.add_row("/quit", "Exit the chat")
            self.console.print(help_table)
        else:
            print("Commands:")
            print("  /help     - Show this help")
            print("  /clear    - Clear conversation history")
            print("  /history  - Show conversation history")
            if self.config.enable_mcp:
                print("  /mcp      - Show MCP integration status")
            print("  /quit     - Exit the chat")

    def show_error(self, message: str):
        """Show error message."""
        if self.console:
            self.console.print(f"[red]❌ {message}[/red]")
        else:
            print(f"❌ {message}")

    def show_success(self, message: str):
        """Show success message."""
        if self.console:
            self.console.print(f"[green]✓ {message}[/green]")
        else:
            print(f"✓ {message}")