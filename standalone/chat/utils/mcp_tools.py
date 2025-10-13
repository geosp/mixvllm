"""MCP (Model Context Protocol) tools for standalone chat client.

This module provides MCP tool functionality for the standalone chat client,
without LangChain dependencies.
"""

from typing import Any, Dict, List, Optional
from .mcp_client import get_mcp_config, create_mcp_client, MCPError


class MCPTool:
    """Represents an MCP tool for standalone use."""
    
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


# Cache for discovered tools
_discovered_tools = None


def get_available_mcp_tools(config_path: Optional[str] = None) -> List[MCPTool]:
    """Get all available MCP tools."""
    global _discovered_tools
    if _discovered_tools is not None:
        return _discovered_tools

    _discovered_tools = []
    try:
        config = get_mcp_config(config_path)
        servers = config.get_servers()
        
        for server_name, server_config in servers.items():
            if not server_config.enabled:
                continue
                
            try:
                client = create_mcp_client(server_name)
                tools = client.list_tools()
                
                for tool in tools:
                    # Create a unique tool name combining server and tool name
                    full_tool_name = f"{server_name}_{tool.name}"
                    
                    mcp_tool = MCPTool(
                        name=full_tool_name,
                        description=tool.description,
                        server=server_name,
                        tool_name=tool.name,
                        input_schema=tool.input_schema
                    )
                    
                    _discovered_tools.append(mcp_tool)
                    
            except Exception as e:
                print(f"Warning: Could not connect to MCP server '{server_name}': {e}")
                continue
                
    except Exception as e:
        print(f"Warning: Could not load MCP configuration: {e}")
        return []
    
    return _discovered_tools