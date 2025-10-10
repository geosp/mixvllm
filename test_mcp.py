#!/usr/bin/env python3
"""
Test script for MCP configuration and connectivity.

Test MCP server configurations and tool discovery.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.mcp_client import get_mcp_config
from utils.mcp_tools import get_mcp_servers, get_mcp_tool_names, test_mcp_connection

def test_mcp_config():
    """Test MCP configuration loading."""
    print("Testing MCP Configuration...")

    try:
        config = get_mcp_config()
        servers = config.get_servers()
        settings = config.get_settings()

        print(f"✓ Loaded configuration from: {config.config_path}")
        print(f"✓ Found {len(servers)} configured servers")
        print(f"✓ Settings: {settings}")

        for name, server in servers.items():
            print(f"  - {name}: {server.url} ({'enabled' if server.enabled else 'disabled'})")

        return True

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_mcp_servers():
    """Test connectivity to MCP servers."""
    print("\nTesting MCP Server Connectivity...")

    servers = get_mcp_servers()
    if not servers:
        print("⚠ No MCP servers configured")
        return

    for server_name, server_info in servers.items():
        print(f"\nTesting {server_name}...")
        result = test_mcp_connection(server_name)

        if result['status'] == 'connected':
            print(f"✓ Connected to {server_name}")
            print(f"  Tools: {result['tools_count']} ({', '.join(result['tools'])})")
            print(f"  Resources: {result['resources_count']} ({', '.join(result['resources'])})")
        else:
            print(f"❌ Failed to connect to {server_name}: {result['error']}")


def test_mcp_tools():
    """Test MCP tool discovery."""
    print("\nTesting MCP Tool Discovery...")

    try:
        tool_names = get_mcp_tool_names()
        print(f"✓ Discovered {len(tool_names)} MCP tools: {tool_names}")

        # Try to get LangChain tools (will fail if LangChain not installed)
        try:
            from utils.mcp_tools import get_available_mcp_tools
            tools = get_available_mcp_tools()
            print(f"✓ Created {len(tools)} LangChain tools")
        except ImportError:
            print("⚠ LangChain not available - tools will be disabled")

    except Exception as e:
        print(f"❌ Tool discovery failed: {e}")


def main():
    """Run all MCP tests."""
    print("🔧 MCP Integration Test Suite")
    print("=" * 50)

    success = test_mcp_config()
    if success:
        test_mcp_servers()
        test_mcp_tools()

    print("\n" + "=" * 50)
    if success:
        print("✅ MCP tests completed")
    else:
        print("❌ MCP tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()