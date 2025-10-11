"""MCP (Model Context Protocol) tools for LangChain integration.

This module provides LangChain tools that wrap MCP server functionality,
allowing LLMs to call MCP tools during conversations.
"""

import functools
from typing import Any, Dict, List, Optional
try:
    from .mcp_client import get_mcp_config, create_mcp_client, MCPError
except ImportError:
    from mcp_client import get_mcp_config, create_mcp_client, MCPError


# Cache for discovered tools
_discovered_tools = None
_tool_functions = {}


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
            print(f"Warning: Could not connect to MCP server '{server_name}': {e}")
            continue

    return _discovered_tools


def get_available_mcp_tools(config_path: Optional[str] = None) -> List[Any]:
    """Get all available MCP tools for LangChain integration."""
    try:
        from langchain_core.tools import BaseTool
        from langchain_core.callbacks import CallbackManagerForToolRun
        from typing import Optional, Type
        from pydantic import BaseModel, Field

        tools = _discover_mcp_tools(config_path)
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

            # Create a dynamic tool class
            def create_tool_class(t_name, t_info, args_schema):
                # Create a custom tool class
                class MCPTool(BaseTool):
                    name: str = t_name
                    description: str = f"{t_info['description']} (Server: {t_info['server']})"
                    args_schema: Type[BaseModel] = ArgsSchema

                    def _run(self, **kwargs) -> str:
                        """Run the tool synchronously."""
                        try:
                            import asyncio
                            import concurrent.futures
                            import threading
                            
                            # Check if we're already in an async context
                            try:
                                current_loop = asyncio.get_running_loop()
                                # If we have a running loop, we need to run in a separate thread
                                def run_in_thread():
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    try:
                                        result = new_loop.run_until_complete(t_info['function'](**kwargs))
                                        return result
                                    finally:
                                        new_loop.close()
                                
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(run_in_thread)
                                    return future.result()
                            except RuntimeError:
                                # No running loop, safe to create one
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    result = loop.run_until_complete(t_info['function'](**kwargs))
                                    return result
                                finally:
                                    loop.close()
                        except Exception as e:
                            return f"[{t_info['server']}] Error: {e}"

                    async def _arun(self, **kwargs) -> str:
                        """Run the tool asynchronously."""
                        try:
                            return await t_info['function'](**kwargs)
                        except Exception as e:
                            return f"[{t_info['server']}] Error: {e}"

                return MCPTool
            
            MCPToolClass = create_tool_class(tool_name, tool_info, ArgsSchema)
            
            # Create instance of the tool
            langchain_tool = MCPToolClass()
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