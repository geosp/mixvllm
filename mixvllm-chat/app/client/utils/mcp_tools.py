"""MCP (Model Context Protocol) tools for LangChain integration.

This module provides LangChain tools that wrap MCP server functionality,
allowing LLMs to call MCP tools during conversations.
"""

import functools
import logging
from typing import Any, Dict, List, Optional
from app.client.utils.mcp_client import MCPTool as ClientMCPTool, get_mcp_config, create_mcp_client, MCPError


# Cache for discovered tools
_discovered_tools = None
_tool_functions = {}

logger = logging.getLogger(__name__)


def _discover_mcp_tools(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Discover and cache MCP tools from configured servers."""
    global _discovered_tools
    if _discovered_tools is not None:
        return _discovered_tools

    _discovered_tools = {}
    config = get_mcp_config(config_path)
    servers = config.get_servers()

    for server_name, server_config in servers.items():
        try:
            client = create_mcp_client(server_name)
            tools = client.list_tools()

            for tool in tools:
                # Create a unique tool name combining server and tool name
                full_tool_name = f"{server_name}_{tool.name}"

                # Create tool function using a class to avoid closure issues
                class ToolExecutor:
                    def __init__(self, server, tool_name):
                        self.server = server
                        self.tool_name = tool_name
                    
                    async def __call__(self, **kwargs):
                        try:
                            client = create_mcp_client(self.server)

                            # Call the intended MCP tool
                            result = client.call_tool(self.tool_name, kwargs)

                            if result and len(result) > 0:
                                content = result[0].get("text", "")
                                return f"[{self.server}] {content}"
                            else:
                                return f"[{self.server}] Tool executed but returned no result"

                        except MCPError as e:
                            return f"[{self.server}] Error: {e}"
                        except Exception as e:
                            return f"[{self.server}] Unexpected error: {e}"

                tool_function = ToolExecutor(server_name, tool.name)

                # Store the tool info
                _discovered_tools[full_tool_name] = {
                    'function': tool_function,
                    'description': tool.description,
                    'server': server_name,
                    'tool_name': full_tool_name,
                    'input_schema': tool.input_schema
                }

        except Exception as e:
            # Skip servers that can't be reached
            logger.warning(f"Could not connect to MCP server '{server_name}': {e}")
            continue

    return _discovered_tools


class MCPTool:
    """Represents an MCP tool for LangChain integration."""

    def __init__(self, name: str, description: str, server: str, tool_name: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.server = server
        self.tool_name = tool_name
        self.input_schema = input_schema

    def execute(self, **kwargs) -> str:
        """Execute the MCP tool with given arguments."""
        try:
            client = create_mcp_client(self.server)
            result = client.call_tool(self.tool_name, kwargs)

            if result and len(result) > 0:
                content = result[0].get("text", "")
                return f"[{self.server}] {content}"
            else:
                return f"[{self.server}] Tool executed but returned no result"

        except MCPError as e:
            return f"[{self.server}] Error: {e}"
        except Exception as e:
            return f"[{self.server}] Unexpected error: {e}"

    async def _run(self, **kwargs) -> str:
        """Run the MCP tool asynchronously."""
        return self.execute(**kwargs)


# Updated debug statements to use logging.debug
def get_available_mcp_tools(config_path: Optional[str] = None) -> List[MCPTool]:
    """Get all available MCP tools."""
    global _discovered_tools
    if _discovered_tools is not None:
        return _discovered_tools

    _discovered_tools = []
    config = get_mcp_config(config_path)
    servers = config.get_servers()

    for server_name, server_config in servers.items():
        try:
            client = create_mcp_client(server_name)
            tools = client.list_tools()

            for tool in tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description,
                    server=server_name,
                    tool_name=tool.name,
                    input_schema=tool.input_schema,
                )
                _discovered_tools.append(mcp_tool)

        except Exception as e:
            logger.warning(f"Failed to load tools from server {server_name}: {e}")

    return _discovered_tools


def get_mcp_tool_names() -> List[str]:
    """Get names of all available MCP tools."""
    tools = _discover_mcp_tools()
    return list(tools.keys())


def get_mcp_servers() -> Dict[str, Any]:
    """Get information about configured MCP servers."""
    config = get_mcp_config()
    servers = config.get_servers()

    server_info = {}
    for name, server_config in servers.items():
        server_info[name] = {
            'url': server_config.url,
            'description': server_config.description,
            'enabled': server_config.enabled
        }

    return server_info


def test_mcp_connection(server_name: str) -> Dict[str, Any]:
    """Test connection to an MCP server and return status info."""
    try:
        client = create_mcp_client(server_name)
        tools = client.list_tools()
        resources = client.list_resources()

        return {
            'status': 'connected',
            'tools_count': len(tools),
            'resources_count': len(resources),
            'tools': [t.name for t in tools],
            'resources': [r.name for r in resources]
        }

    except MCPError as e:
        return {
            'status': 'error',
            'error': str(e)
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': f"Unexpected error: {e}"
        }