"""MCP (Model Context Protocol) tools for standalone chat client."""MCP (Model Context Protocol) tools for standalone chat client.



This module provides MCP tool functionality for the standalone chat client,This module provides MCP tool functionality for the standalone chat client,

without LangChain dependencies.without LangChain dependencies.

""""""



from typing import Any, Dict, List, Optionalfrom typing import Any, Dict, List, Optional

from .mcp_client import get_mcp_config, create_mcp_client, MCPErrorfrom .mcp_client import get_mcp_config, create_mcp_client, MCPError





class MCPTool:class MCPTool:

    """Represents an MCP tool for standalone use."""    """Represents an MCP tool for standalone use."""

        

    def __init__(self, name: str, description: str, server: str, tool_name: str, input_schema: Dict[str, Any]):    def __init__(self, name: str, description: str, server: str, tool_name: str, input_schema: Dict[str, Any]):

        self.name = name        self.name = name

        self.description = description        self.description = description

        self.server = server        self.server = server

        self.tool_name = tool_name        self.tool_name = tool_name

        self.input_schema = input_schema        self.input_schema = input_schema

        

    def execute(self, **kwargs) -> str:    def execute(self, **kwargs) -> str:

        """Execute the MCP tool with given arguments."""        """Execute the MCP tool with given arguments."""

        try:        try:

            client = create_mcp_client(self.server)            client = create_mcp_client(self.server)

            result = client.call_tool(self.tool_name, kwargs)            result = client.call_tool(self.tool_name, kwargs)

                        

            if result and len(result) > 0:            if result and len(result) > 0:

                content = result[0].get("text", "")                content = result[0].get("text", "")

                return f"[{self.server}] {content}"                return f"[{self.server}] {content}"

            else:            else:

                return f"[{self.server}] Tool executed but returned no result"                return f"[{self.server}] Tool executed but returned no result"

                                

        except MCPError as e:        except MCPError as e:

            return f"[{self.server}] Error: {e}"            return f"[{self.server}] Error: {e}"

        except Exception as e:        except Exception as e:

            return f"[{self.server}] Unexpected error: {e}"            return f"[{self.server}] Unexpected error: {e}"





# Cache for discovered tools# Cache for discovered tools

_discovered_tools = None_discovered_tools = None





# Added debug logging to verify config path and server connectivity# Added debug logging to verify config path and server connectivity

def get_available_mcp_tools(config_path: Optional[str] = None) -> List[MCPTool]:def get_available_mcp_tools(config_path: Optional[str] = None) -> List[MCPTool]:

    """Get all available MCP tools."""    """Get all available MCP tools."""

    global _discovered_tools    global _discovered_tools

    if _discovered_tools is not None:    if _discovered_tools is not None:

        return _discovered_tools        return _discovered_tools



    _discovered_tools = []    _discovered_tools = []

    config = get_mcp_config(config_path)    try:

    servers = config.get_servers()        print(f"Resolved config path: {config_path}")  # Debug log

        config = get_mcp_config(config_path)

    for server_name, server_config in servers.items():        servers = config.get_servers()

        try:

            client = create_mcp_client(server_name)        for server_name, server_config in servers.items():

            tools = client.list_tools()            if not server_config.enabled:

                continue

            for tool in tools:

                mcp_tool = MCPTool(            try:

                    name=tool.name,                print(f"Connecting to server: {server_name} at {server_config.url}")  # Debug log

                    description=tool.description,                client = create_mcp_client(server_name)

                    server=server_name,                tools = client.list_tools()

                    tool_name=tool.name,                print(f"Discovered tools from {server_name}: {tools}")  # Debug log

                    input_schema=tool.input_schema,

                )                for tool in tools:

                _discovered_tools.append(mcp_tool)                    _discovered_tools.append(

                        MCPTool(

        except Exception as e:                            name=tool.name,

            print(f"Failed to load tools from server {server_name}: {e}")                            description=tool.description,

                            server=server_name,

    return _discovered_tools                            tool_name=tool.name,
                            input_schema=tool.input_schema
                        )
                    )
            except Exception as e:
                print(f"Error connecting to server {server_name}: {e}")  # Debug log
    except Exception as e:
        print(f"Error loading MCP tools: {e}")  # Debug log

    return _discovered_tools