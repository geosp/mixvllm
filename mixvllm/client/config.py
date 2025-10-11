"""
Configuration management for the chat client.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatConfig:
    """Configuration for the chat client."""
    base_url: str = "http://localhost:8000"
    model: Optional[str] = None
    enable_mcp: bool = False
    debug: bool = False
    mcp_config_path: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

    def __post_init__(self):
        """Validate and normalize configuration."""
        self.base_url = self.base_url.rstrip('/')

    @classmethod
    def from_args(cls, args) -> 'ChatConfig':
        """Create config from parsed command line arguments."""
        return cls(
            base_url=getattr(args, 'base_url', "http://localhost:8000"),
            model=getattr(args, 'model', None),
            enable_mcp=getattr(args, 'enable_mcp', False),
            debug=getattr(args, 'debug', False),
            mcp_config_path=getattr(args, 'mcp_config', None),
            temperature=getattr(args, 'temperature', 0.7),
            max_tokens=getattr(args, 'max_tokens', 512),
            stream=getattr(args, 'stream', False)
        )