"""
Connection management for vLLM server communication.

This module handles HTTP communication with the vLLM server including:
- Health checks and connectivity testing
- Model discovery via the OpenAI-compatible API
- Connection pooling for performance

Learning Points:
- requests.Session() enables HTTP connection pooling (reuses TCP connections)
- Connection pooling reduces latency by avoiding repeated handshakes
- Graceful error handling with user-friendly messages
- Optional Rich console for enhanced terminal output
"""

import requests
from typing import List, Optional
from rich.console import Console

from .config import ChatConfig


class ConnectionManager:
    """Manages HTTP connections to the vLLM server.

    Uses requests.Session for efficient connection pooling which:
    - Reuses TCP connections across multiple requests
    - Maintains cookies and headers automatically
    - Significantly improves performance for multiple requests

    Typical performance gains:
    - Without pooling: ~50-100ms per request (TCP handshake)
    - With pooling: ~5-10ms per request (reuse connection)
    """

    def __init__(self, config: ChatConfig):
        """Initialize the connection manager.

        Args:
            config: ChatConfig object with server connection settings
        """
        self.config = config

        # ====================================================================
        # HTTP Session Setup
        # ====================================================================
        self.session = requests.Session()
        # Session benefits:
        # 1. Connection pooling - reuse TCP connections
        # 2. Persistent cookies across requests
        # 3. Default headers applied to all requests
        # 4. Better performance than individual requests.get() calls

        # ====================================================================
        # UI Console Setup
        # ====================================================================
        self.console = Console() if Console is not None else None
        # Rich Console provides:
        # - Colored output with markup syntax: [red]text[/red]
        # - Emoji support: ✓ ❌ ⚠
        # - Advanced formatting: tables, panels, progress bars
        # Fallback to print() if Rich not available

    def test_connection(self) -> bool:
        """Test connection to the server using health endpoint or models endpoint.

        First tries the vLLM /health endpoint, then falls back to /v1/models
        for OpenAI-compatible servers like Ollama that don't have /health.

        Returns:
            bool: True if server is reachable and healthy, False otherwise
        """
        try:
            # First try vLLM health endpoint
            response = self.session.get(f"{self.config.base_url}/health")
            if response.status_code == 200:
                if self.console:
                    self.console.print(f"[green]✓[/green] Connected to vLLM server at {self.config.base_url}")
                else:
                    print(f"✓ Connected to vLLM server at {self.config.base_url}")
                return True
            elif response.status_code == 503:
                # 503 means server is starting up, which is acceptable
                if self.console:
                    self.console.print(f"[yellow]⚠[/yellow] Server is starting up at {self.config.base_url} (status 503)")
                else:
                    print(f"⚠ Server is starting up at {self.config.base_url} (status 503)")
                return True
            elif response.status_code == 404:
                # 404 on /health, try /v1/models as fallback for Ollama/other OpenAI-compatible servers
                models_response = self.session.get(f"{self.config.base_url}/v1/models")
                if models_response.status_code == 200:
                    if self.console:
                        self.console.print(f"[green]✓[/green] Connected to OpenAI-compatible server at {self.config.base_url}")
                    else:
                        print(f"✓ Connected to OpenAI-compatible server at {self.config.base_url}")
                    return True
                else:
                    if self.console:
                        self.console.print(f"[yellow]⚠[/yellow] Server responded with status {models_response.status_code}")
                    else:
                        print(f"⚠ Server responded with status {models_response.status_code}")
                    return False
            else:
                # Other non-success status codes
                if self.console:
                    self.console.print(f"[yellow]⚠[/yellow] Server responded with status {response.status_code}")
                else:
                    print(f"⚠ Server responded with status {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            if self.console:
                self.console.print(f"[red]❌[/red] Cannot connect to server: {e}")
                self.console.print(f"Make sure the server is running at {self.config.base_url}")
            else:
                print(f"❌ Cannot connect to server: {e}")
                print(f"Make sure the server is running at {self.config.base_url}")
            return False

    def get_available_models(self) -> List[str]:
        """Query vLLM server for available models via OpenAI-compatible API.

        vLLM implements the OpenAI API specification, including the /v1/models endpoint
        This endpoint returns metadata about models loaded on the server.

        API Response Format (OpenAI-compatible):
        {
            "object": "list",
            "data": [
                {
                    "id": "microsoft/Phi-3-mini-4k-instruct",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "vllm"
                }
            ]
        }

        Returns:
            List[str]: List of model IDs available on server, or empty list if error

        Use Cases:
            - Auto-detect which model to use when user doesn't specify
            - Validate that requested model is available
            - Display available models to user
        """
        try:
            # Query the OpenAI-compatible models endpoint
            response = self.session.get(f"{self.config.base_url}/v1/models")

            if response.status_code == 200:
                # Parse JSON response
                data = response.json()

                # Extract model IDs from 'data' array
                # List comprehension: [model['id'] for model in data.get('data', [])]
                # - data.get('data', []) safely gets 'data' key, defaults to empty list
                # - Iterates through each model dict
                # - Extracts the 'id' field
                return [model['id'] for model in data.get('data', [])]
            else:
                # Non-200 response: Server error or endpoint not available
                if self.console:
                    self.console.print(f"[yellow]⚠[/yellow] Could not fetch models: HTTP {response.status_code}")
                else:
                    print(f"⚠ Could not fetch models: HTTP {response.status_code}")
                return []

        except Exception as e:
            # Catch all exceptions: network errors, JSON parsing errors, etc.
            # Return empty list rather than raising exception (graceful degradation)
            if self.console:
                self.console.print(f"[yellow]⚠[/yellow] Could not fetch models: {e}")
            else:
                print(f"⚠ Could not fetch models: {e}")
            return []