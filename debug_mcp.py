#!/usr/bin/env python3
"""Debug script to test MCP server connections."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mixvllm.client.utils.mcp_tools import test_mcp_connection, get_mcp_servers

def main():
    print("=== MCP Server Connection Debug ===\n")

    # Get configured servers
    servers = get_mcp_servers()
    print(f"Configured servers: {list(servers.keys())}\n")

    # Test each server
    for server_name, server_info in servers.items():
        print(f"Testing {server_name}:")
        print(f"  URL: {server_info['url']}")
        print(f"  Description: {server_info['description']}")
        print(f"  Enabled: {server_info['enabled']}")

        if server_info['enabled']:
            print("  Testing connection...")
            result = test_mcp_connection(server_name)
            print(f"  Status: {result['status']}")

            if result['status'] == 'connected':
                print(f"  Tools: {result['tools_count']} ({', '.join(result['tools'])})")
                print(f"  Resources: {result['resources_count']} ({', '.join(result['resources'])})")
            else:
                print(f"  Error: {result['error']}")
        else:
            print("  Server disabled, skipping test")

        print()

if __name__ == "__main__":
    main()