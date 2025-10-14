"""
Configuration management for the chat client.

This module provides centralized configuration using Python dataclasses,
which offer type safety, default values, and validation capabilities.

Learning Points:
- Dataclasses simplify configuration management vs manual __init__
- Type hints enable IDE autocomplete and static analysis
- __post_init__ allows validation after object creation
- ClassMethod factories provide alternative construction patterns
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatConfig:
    """Configuration container for the chat client.

    Uses Python's dataclass decorator which automatically generates:
    - __init__() method from the field definitions
    - __repr__() for debugging
    - __eq__() for comparisons

    This eliminates boilerplate while maintaining type safety.
    """

    # ========================================================================
    # Server Connection Settings
    # ========================================================================
    base_url: str = "http://localhost:8000"
    # The vLLM server endpoint
    # Format: http://host:port (no trailing slash)
    # Example: "http://192.168.1.100:8000" for remote server

    model: Optional[str] = None
    # Model identifier (optional, server may have default)
    # If None, the server's default model is used
    # Example: "microsoft/Phi-3-mini-4k-instruct"

    # ========================================================================
    # Feature Flags
    # ========================================================================
    enable_mcp: bool = False
    # Enable Model Context Protocol (MCP) tool integration
    # When True, the client can call external tools via MCP servers
    # Requires mcp_config_path to be set
    # See: https://github.com/anthropics/mcp for MCP spec

    debug: bool = False
    # Enable debug logging to llm_debug.log
    # Logs all LLM prompts, responses, and tool calls
    # Useful for troubleshooting generation issues

    # ========================================================================
    # MCP Configuration
    # ========================================================================
    mcp_config_path: Optional[str] = None
    # Path to MCP servers configuration YAML file
    # Required if enable_mcp=True
    # Format: YAML file with server definitions
    # Example: "configs/mcp_servers.yaml"

    # ========================================================================
    # Generation Parameters
    # ========================================================================
    temperature: float = 0.7
    # Sampling temperature (0.0-2.0)
    # Controls randomness in generation
    # Lower = more deterministic, higher = more creative
    # Default 0.7 balances coherence and creativity

    max_tokens: int = 4096
    # Maximum tokens in response
    # Limits response length and prevents runaway generation
    # Note: 1 token ≈ 0.75 words in English
    # 4096 tokens ≈ 3000 words

    stream: bool = False
    # Enable streaming responses (token-by-token)
    # When True, responses appear incrementally
    # When False, entire response arrives at once
    # Streaming improves perceived latency for long responses

    # ========================================================================
    # Post-Initialization Validation
    # ========================================================================
    def __post_init__(self):
        """Validate and normalize configuration after initialization.

        This special method is called automatically by dataclass after __init__
        Perfect for validation and normalization tasks.

        Why normalize the URL:
        - Trailing slashes cause issues when joining paths
        - Example: "http://localhost:8000/" + "/v1/chat" = "http://localhost:8000//v1/chat"
        - rstrip('/') ensures consistent URL formatting
        """
        self.base_url = self.base_url.rstrip('/')

    # ========================================================================
    # Alternative Constructors
    # ========================================================================
    @classmethod
    def from_args(cls, args) -> 'ChatConfig':
        """Create config from parsed command line arguments.

        This is a Factory Method pattern - an alternative constructor
        that creates ChatConfig from argparse Namespace objects.

        Why use @classmethod:
        - Returns cls (the class itself) not 'self' (instance)
        - Enables alternative construction pathways
        - Common pattern for "from_X" constructor methods

        Why getattr with defaults:
        - argparse may not set attributes if not provided
        - getattr(args, 'model', None) returns None if 'model' not found
        - Prevents AttributeError exceptions
        - Gracefully handles partial argument sets

        Args:
            args: argparse.Namespace object from parser.parse_args()

        Returns:
            ChatConfig: Fully initialized configuration object
        """
        return cls(
            base_url=getattr(args, 'base_url', "http://localhost:8000"),
            model=getattr(args, 'model', None),
            enable_mcp=getattr(args, 'enable_mcp', False),
            debug=getattr(args, 'debug', False),
            mcp_config_path=getattr(args, 'mcp_config', None),
            temperature=getattr(args, 'temperature', 0.7),
            max_tokens=getattr(args, 'max_tokens', 4096),
            stream=getattr(args, 'stream', False)
        )