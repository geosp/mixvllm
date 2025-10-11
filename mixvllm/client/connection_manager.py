"""
Connection management for vLLM server communication.
"""

import requests
from typing import List, Optional
from rich.console import Console

from .config import ChatConfig


class ConnectionManager:
    """Manages connections to the vLLM server."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.session = requests.Session()
        self.console = Console() if Console is not None else None

    def test_connection(self) -> bool:
        """Test connection to the vLLM server."""
        try:
            response = self.session.get(f"{self.config.base_url}/health")
            if response.status_code == 200:
                if self.console:
                    self.console.print(f"[green]✓[/green] Connected to vLLM server at {self.config.base_url}")
                else:
                    print(f"✓ Connected to vLLM server at {self.config.base_url}")
                return True
            else:
                if self.console:
                    self.console.print(f"[yellow]⚠[/yellow] Server responded with status {response.status_code}")
                else:
                    print(f"⚠ Server responded with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            if self.console:
                self.console.print(f"[red]❌[/red] Cannot connect to server: {e}")
                self.console.print(f"Make sure the vLLM server is running at {self.config.base_url}")
            else:
                print(f"❌ Cannot connect to server: {e}")
                print(f"Make sure the vLLM server is running at {self.config.base_url}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models from the server."""
        try:
            response = self.session.get(f"{self.config.base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
            else:
                if self.console:
                    self.console.print(f"[yellow]⚠[/yellow] Could not fetch models: HTTP {response.status_code}")
                else:
                    print(f"⚠ Could not fetch models: HTTP {response.status_code}")
                return []
        except Exception as e:
            if self.console:
                self.console.print(f"[yellow]⚠[/yellow] Could not fetch models: {e}")
            else:
                print(f"⚠ Could not fetch models: {e}")
            return []