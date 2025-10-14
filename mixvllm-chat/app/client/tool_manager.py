"""
MCP tool management and execution.
"""

from typing import Dict, Any, Optional
from rich.console import Console

from .config import ChatConfig
from app.client.utils.mcp_tools import get_available_mcp_tools


class ToolManager:
    """Manages MCP tools and their execution."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.console = Console() if Console is not None else None
        self.mcp_tools: Dict[str, Any] = {}

        if self.config.enable_mcp:
            self._setup_mcp_agent()

    def _setup_mcp_agent(self):
        """Set up the MCP-enabled tools."""
        try:
            # For now, disable the LangChain agent and use direct MCP tool calling
            # This avoids complex LangChain integration issues
            tools = get_available_mcp_tools(self.config.mcp_config_path)

            if not tools:
                raise ValueError("No MCP tools available")

            # Store tools for direct calling instead of using LangChain agent
            self.mcp_tools = {tool.name: tool for tool in tools}

            if self.console:
                self.console.print(f"[green]✓[/green] MCP tools enabled ({len(tools)} tools available)")
            else:
                print(f"✓ MCP tools enabled ({len(tools)} tools available)")

        except Exception as e:
            if self.console:
                self.console.print(f"[yellow]⚠[/yellow] Failed to setup MCP agent: {e}")
                self.console.print("[dim]Falling back to simple chat mode[/dim]")
            else:
                print(f"⚠ Failed to setup MCP agent: {e}")
                print("Falling back to simple chat mode")
            self.config.enable_mcp = False

    def execute_tool(self, tool_name: str, params: dict) -> Optional[str]:
        """Execute the specified MCP tool with the given parameters."""
        try:
            if self.console:
                self.console.print(f"[dim]🔍 Executing tool: {tool_name}[/dim]")
                self.console.print(f"[dim]🔍 Parameters: {params}[/dim]")

            if tool_name not in self.mcp_tools:
                error_msg = f"Unknown tool: {tool_name}"
                if self.console:
                    self.console.print(f"[red]❌ {error_msg}[/red]")
                return error_msg

            tool_obj = self.mcp_tools[tool_name]

            # Call the tool's execute method with the parameters
            result = tool_obj.execute(**params)

            if self.console:
                self.console.print(f"[dim]🔍 Tool result: {result[:100]}...[/dim]")

            return str(result)

        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            return error_msg

    def format_tools_for_llm(self) -> str:
        """Format all available MCP tools for inclusion in LLM prompts."""
        if not self.mcp_tools:
            return "No tools available."

        tools_info = []
        for tool_name, tool_obj in self.mcp_tools.items():
            # Get tool description
            description = tool_obj.description if hasattr(tool_obj, 'description') else "No description"

            # Get parameters from args_schema
            params_info = []
            if hasattr(tool_obj, 'args_schema') and tool_obj.args_schema:
                schema = tool_obj.args_schema
                fields = None
                if hasattr(schema, 'model_fields'):
                    fields = schema.model_fields
                elif hasattr(schema, '__fields__'):
                    fields = schema.__fields__

                if fields:
                    for field_name, field_info in fields.items():
                        # Get field type
                        field_type = str(field_info.annotation).replace('typing.', '')
                        if hasattr(field_info.annotation, '__name__'):
                            field_type = field_info.annotation.__name__

                        # Check if required
                        required = getattr(field_info, 'is_required', lambda: True)()
                        if hasattr(field_info, 'default') and field_info.default is not ...:
                            required = False

                        desc = field_info.description or ""
                        params_info.append(f"  - {field_name} ({field_type}, {'required' if required else 'optional'}): {desc}")

            params_str = "\n".join(params_info) if params_info else "  No parameters required"

            tools_info.append(f"Tool: {tool_name}\nDescription: {description}\nParameters:\n{params_str}")

        return "\n\n".join(tools_info)

    def has_tools(self) -> bool:
        """Check if MCP tools are available."""
        return bool(self.mcp_tools)