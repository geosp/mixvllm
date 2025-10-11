"""
Main chat client that orchestrates all components.
"""

from .config import ChatConfig
from .connection_manager import ConnectionManager
from .tool_manager import ToolManager
from .response_handler import ResponseHandler
from .history_manager import HistoryManager
from .ui_manager import UIManager
from .chat_engine import ChatEngine


class ChatClient:
    """Enhanced chat client for vLLM OpenAI-compatible API with MCP tool support."""

    def __init__(self, config: ChatConfig):
        self.config = config

        # Initialize components
        self.connection_manager = ConnectionManager(config)
        self.tool_manager = ToolManager(config)
        self.response_handler = ResponseHandler(config)
        self.history_manager = HistoryManager(config)
        self.ui_manager = UIManager(config)
        self.chat_engine = ChatEngine(config, self.tool_manager, self.response_handler, self.history_manager)

        # Test connection
        if not self.connection_manager.test_connection():
            raise ConnectionError("Failed to connect to vLLM server")

    def chat(self, message: str) -> str:
        """Send a chat message and get response."""
        return self.chat_engine.chat(message)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history_manager.clear_history()

    def show_history(self) -> None:
        """Show conversation history."""
        self.history_manager.show_history()

    def show_mcp_status(self) -> None:
        """Show MCP integration status."""
        self.ui_manager.show_mcp_status()

    def get_available_models(self):
        """Get available models from the server."""
        return self.connection_manager.get_available_models()

    def set_model(self, model: str):
        """Set the model to use."""
        self.config.model = model