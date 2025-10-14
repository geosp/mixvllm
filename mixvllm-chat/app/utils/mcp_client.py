"""Simple MCP (Model Context Protocol) HTTP client.

This module provides a basic client for communicating with MCP servers
over HTTP, following the MCP JSON-RPC protocol.
"""

import json
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class MCPResource:
    """Represents an MCP resource."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    url: str
    auth_token: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True


class MCPClient:
    """Simple HTTP client for MCP servers following MCP protocol."""

    def __init__(self, base_url: str, auth_token: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.timeout = timeout
        self.session_id = None
        self.request_id = 0

        # Set headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json"
        })

        if auth_token:
            self.session.headers.update({
                "Authorization": f"Bearer {auth_token}"
            })

    def _next_request_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id

    def _parse_sse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response."""
        lines = response_text.split('\n')
        for line in lines:
            if line.startswith("data: "):
                data = line[6:].strip()
                if data:
                    return json.loads(data)
        # Fallback: try to parse as direct JSON
        try:
            return json.loads(response_text)
        except:
            raise MCPError("No valid response data found in SSE stream")

    def _call_method(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a JSON-RPC call to the MCP server."""
        request_id = self._next_request_id()

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        if params:
            payload["params"] = params

        # Add session ID header for non-initialize requests
        headers = dict(self.session.headers)
        if self.session_id and method != "initialize":
            headers["mcp-session-id"] = self.session_id

        try:
            response = self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Parse response based on content type
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                result = self._parse_sse_response(response.text)
            else:
                result = response.json()

            # Extract session ID from initialize response
            if method == "initialize" and "mcp-session-id" in response.headers:
                self.session_id = response.headers["mcp-session-id"]

            if "error" in result:
                raise MCPError(f"MCP error: {result['error']}")

            return result.get("result", {})

        except requests.exceptions.Timeout:
            raise MCPError(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise MCPError(f"HTTP error: {e}")
        except json.JSONDecodeError as e:
            raise MCPError(f"Invalid JSON response: {e}")

    def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP connection."""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mixvllm-client", "version": "1.0.0"}
        }
        result = self._call_method("initialize", params)

        # Send initialized notification (required by protocol)
        self._send_notification("notifications/initialized")

        return result

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Send a notification (no response expected)."""
        payload = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params:
            payload["params"] = params

        # Add session ID header
        headers = dict(self.session.headers)
        if self.session_id:
            headers["mcp-session-id"] = self.session_id

        try:
            response = self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
        except Exception as e:
            # Notifications don't expect responses, so we ignore errors
            pass

    def list_tools(self) -> List[MCPTool]:
        """List available tools from the MCP server."""
        try:
            # Ensure we're initialized
            if not self.session_id:
                self.initialize()

            result = self._call_method("tools/list")
            tools = []
            for tool_data in result.get("tools", []):
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {})
                )
                tools.append(tool)
            return tools
        except Exception:
            # If tools/list fails, return empty list
            return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        # Ensure we're initialized
        if not self.session_id:
            self.initialize()

        params = {
            "name": tool_name,
            "arguments": arguments
        }
        result = self._call_method("tools/call", params)
        return result.get("content", [])

    def list_resources(self) -> List[MCPResource]:
        """List available resources from the MCP server."""
        try:
            # Ensure we're initialized
            if not self.session_id:
                self.initialize()

            result = self._call_method("resources/list")
            resources = []
            for resource_data in result.get("resources", []):
                resource = MCPResource(
                    uri=resource_data["uri"],
                    name=resource_data.get("name", ""),
                    description=resource_data.get("description"),
                    mime_type=resource_data.get("mimeType")
                )
                resources.append(resource)
            return resources
        except Exception:
            return []

    def read_resource(self, uri: str) -> List[Dict[str, Any]]:
        """Read a resource from the MCP server."""
        # Ensure we're initialized
        if not self.session_id:
            self.initialize()

        params = {"uri": uri}
        result = self._call_method("resources/read", params)
        return result.get("contents", [])


class MCPConfig:
    """MCP configuration manager."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default path relative to the workspace root
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "mcp_servers.yaml"

        self.config_path = Path(config_path)
        self._config = None

    def load_config(self) -> Dict[str, Any]:
        """Load MCP configuration from YAML file."""
        if self._config is not None:
            return self._config

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            # Return default empty config if file doesn't exist
            self._config = {"servers": {}, "settings": {}}
        except yaml.YAMLError as e:
            raise MCPError(f"Invalid YAML in {self.config_path}: {e}")

        return self._config

    def get_servers(self) -> Dict[str, MCPServerConfig]:
        """Get all configured MCP servers."""
        config = self.load_config()
        servers = {}

        for name, server_config in config.get("servers", {}).items():
            if server_config.get("enabled", True):
                servers[name] = MCPServerConfig(
                    name=name,
                    url=server_config["url"],
                    auth_token=server_config.get("auth_token"),
                    description=server_config.get("description"),
                    enabled=server_config.get("enabled", True)
                )

        return servers

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get a specific MCP server configuration."""
        servers = self.get_servers()
        return servers.get(name)

    def get_settings(self) -> Dict[str, Any]:
        """Get global MCP settings."""
        config = self.load_config()
        return config.get("settings", {})

    def create_client(self, server_name: str) -> MCPClient:
        """Create an MCP client for a configured server."""
        server_config = self.get_server(server_name)
        if not server_config:
            raise MCPError(f"MCP server '{server_name}' not found or not enabled")

        settings = self.get_settings()
        timeout = settings.get("timeout", 30)

        return MCPClient(
            base_url=server_config.url,
            auth_token=server_config.auth_token,
            timeout=timeout
        )


class MCPError(Exception):
    """Exception raised for MCP-related errors."""
    pass


# Global config instance
_mcp_config = None

def get_mcp_config(config_path: Optional[str] = None) -> MCPConfig:
    """Get the global MCP configuration instance."""
    global _mcp_config
    if _mcp_config is None or (config_path is not None and str(_mcp_config.config_path) != config_path):
        _mcp_config = MCPConfig(config_path)
    return _mcp_config


def create_mcp_client(server_name: str) -> MCPClient:
    """Create an MCP client for a configured server."""
    return get_mcp_config().create_client(server_name)