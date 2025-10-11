"""
Refactored Chat Client with Separation of Concerns

This package contains a modular implementation of the vLLM chat client,
separated into focused components for better maintainability.
"""

from .chat_client import ChatClient
from .config import ChatConfig
from .cli import main

__all__ = ['ChatClient', 'ChatConfig', 'main']