"""MCP (Model Context Protocol) tools for LangChain integration.

This module provides LangChain tools that wrap MCP server functionality,
allowing LLMs to call MCP tools during conversations.
"""

from typing import Any, Dict, List, Optional
from mcp_client import get_mcp_config, create_mcp_client, MCPError


# Cache for discovered tools
_discovered_tools = None
_tool_functions = {}


def _discover_mcp_tools() -> Dict[str, Any]:
    """Discover and cache MCP tools from configured servers."""
    global _discovered_tools
    if _discovered_tools is not None:
        return _discovered_tools

    _discovered_tools = {}
    config = get_mcp_config()
    servers = config.get_servers()

    for server_name, server_config in servers.items():
        try:
            client = create_mcp_client(server_name)
            tools = client.list_tools()

            for tool in tools:
                # Create a unique tool name combining server and tool name
                full_tool_name = f"{server_name}_{tool.name}"

                # Create the tool function dynamically
                async def create_tool_function(server_name=server_name, tool_name=tool.name):
                    async def tool_function(**kwargs):
                        try:
                            client = create_mcp_client(server_name)
                            result = client.call_tool(tool_name, kwargs)

                            if result and len(result) > 0:
                                content = result[0].get("text", "")
                                return f"[{server_name}] {content}"
                            else:
                                return f"[{server_name}] Tool executed but returned no result"

                        except MCPError as e:
                            return f"[{server_name}] Error: {e}"
                        except Exception as e:
                            return f"[{server_name}] Unexpected error: {e}"

                    # Set function metadata
                    tool_function.__name__ = f"{server_name}_{tool_name}"
                    tool_function.__doc__ = f"{tool.description}\n\nServer: {server_name}"

                    return tool_function

                # Create the actual tool function
                tool_function = create_tool_function()

                # Store the tool info
                _discovered_tools[full_tool_name] = {
                    'function': tool_function,
                    'description': tool.description,
                    'server': server_name,
                    'tool_name': tool.name,
                    'input_schema': tool.input_schema
                }

        except Exception as e:
            # Skip servers that can't be reached
            print(f"Warning: Could not connect to MCP server '{server_name}': {e}")
            continue

    return _discovered_tools


def get_available_mcp_tools() -> List[Any]:
    """Get all available MCP tools for LangChain integration."""
    try:
        from langchain_core.tools import BaseTool
        from langchain_core.callbacks import CallbackManagerForToolRun
        from typing import Optional, Type
        from pydantic import BaseModel, Field

        tools = _discover_mcp_tools()
        langchain_tools = []

        for tool_name, tool_info in tools.items():
            # Create input schema for the tool
            input_schema = tool_info['input_schema']
            if input_schema:
                # Create a Pydantic model from the input schema
                fields = {}
                required_fields = input_schema.get('required', [])

                for prop_name, prop_info in input_schema.get('properties', {}).items():
                    # Determine field type
                    prop_type = prop_info.get('type', 'string')
                    if prop_type == 'number':
                        field_type = float
                    elif prop_type == 'integer':
                        field_type = int
                    elif prop_type == 'boolean':
                        field_type = bool
                    else:
                        field_type = str

                    # Make it Optional if not required
                    if prop_name not in required_fields:
                        from typing import Optional
                        field_type = Optional[field_type]

                    fields[prop_name] = Field(
                        description=prop_info.get('description', ''),
                        default=... if prop_name in required_fields else None
                    )

                # Create the args schema class
                ArgsSchema = type(f"{tool_name}Args", (BaseModel,), {
                    "__annotations__": fields,
                    "model_config": {"arbitrary_types_allowed": True}
                })
            else:
                # Empty schema if no input schema provided
                ArgsSchema = type(f"{tool_name}Args", (BaseModel,), {
                    "model_config": {"arbitrary_types_allowed": True}
                })

            # Create a custom tool class
            class MCPTool(BaseTool):
                name: str = tool_name
                description: str = f"{tool_info['description']} (Server: {tool_info['server']})"
                args_schema: Type[BaseModel] = ArgsSchema

                def _run(self, **kwargs) -> str:
                    """Run the tool synchronously."""
                    try:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(tool_info['function'](**kwargs))
                        loop.close()
                        return result
                    except Exception as e:
                        return f"[{tool_info['server']}] Error: {e}"

                async def _arun(self, **kwargs) -> str:
                    """Run the tool asynchronously."""
                    try:
                        return await tool_info['function'](**kwargs)
                    except Exception as e:
                        return f"[{tool_info['server']}] Error: {e}"

            # Create instance of the tool
            langchain_tool = MCPTool()
            langchain_tools.append(langchain_tool)

        return langchain_tools

    except ImportError:
        # Return empty list if LangChain not available
        return []


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