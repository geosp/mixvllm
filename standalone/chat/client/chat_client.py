"""
Main chat client that orchestrates all components.

This module demonstrates the Facade Pattern - providing a simple unified interface
to a complex subsystem of multiple managers and handlers.

Learning Points:
- Facade Pattern: Simplifies complex systems with a single entry point
- Dependency Injection: Components receive dependencies explicitly
- Separation of Concerns: Each manager handles one responsibility
- Fail-Fast Principle: Test connection during initialization
- Composition over Inheritance: Build functionality by composing objects
"""

from .config import ChatConfig
from .connection_manager import ConnectionManager
from .tool_manager import ToolManager
from .response_handler import ResponseHandler
from .history_manager import HistoryManager
from .ui_manager import UIManager
from .chat_engine import ChatEngine


class ChatClient:
    """Facade class that orchestrates the chat client subsystem.

    This is the main entry point for the chat client application, implementing
    the Facade Pattern to hide complexity from the user.

    Architecture Pattern: Facade
    - Provides a simple interface to a complex subsystem
    - User only needs to interact with ChatClient, not individual managers
    - Internally coordinates between 6 different components

    Component Responsibilities:
    - ConnectionManager: HTTP communication with vLLM server
    - ToolManager: MCP tool discovery and execution
    - ResponseHandler: Response formatting and display
    - HistoryManager: Conversation context tracking
    - UIManager: Terminal UI and user feedback
    - ChatEngine: Core chat logic and LLM interaction

    Why this design:
    - Single Responsibility: Each component has one job
    - Testability: Can mock individual components in tests
    - Maintainability: Changes to one component don't affect others
    - Flexibility: Can swap implementations (e.g., different UI)
    """

    def __init__(self, config: ChatConfig):
        """Initialize the chat client and all its components.

        This constructor demonstrates Dependency Injection - passing dependencies
        explicitly rather than creating them internally or using globals.

        Args:
            config: ChatConfig object with all settings

        Raises:
            ConnectionError: If unable to connect to vLLM server during initialization

        Component Initialization Order Matters:
        1. Config stored first (needed by all components)
        2. Independent managers created (connection, tools, response, history, UI)
        3. ChatEngine created last (depends on tool_manager, response_handler, history_manager)
        4. Connection tested (fail-fast if server unreachable)
        """
        self.config = config

        # ====================================================================
        # Component Initialization (Dependency Injection Pattern)
        # ====================================================================
        # Each component receives 'config' as dependency
        # This makes dependencies explicit and enables testing

        self.connection_manager = ConnectionManager(config)
        # Handles: HTTP communication, health checks, model discovery

        self.tool_manager = ToolManager(config)
        # Handles: MCP tool loading, tool execution, tool formatting

        self.response_handler = ResponseHandler(config)
        # Handles: Response display, streaming, LaTeX rendering

        self.history_manager = HistoryManager(config)
        # Handles: Conversation context, history display, memory management

        self.ui_manager = UIManager(config)
        # Handles: Welcome screen, status messages, help text, errors

        self.chat_engine = ChatEngine(config, self.tool_manager, self.response_handler, self.history_manager)
        # Handles: Core chat logic, LLM calls, MCP integration
        # Note: Receives 3 manager dependencies - demonstrates composition

        # ====================================================================
        # Fail-Fast Connection Test
        # ====================================================================
        # Test connection immediately during initialization
        # Fail-Fast Principle: Detect errors as early as possible
        #
        # Why fail during __init__:
        # - Better UX: User knows immediately if server is down
        # - Prevents confusing errors later during chat
        # - Forces user to fix connection before proceeding
        #
        # Alternative approach: Lazy connection (connect on first message)
        # - Pro: Faster startup
        # - Con: Error happens during chat, worse UX
        if not self.connection_manager.test_connection():
            raise ConnectionError("Failed to connect to vLLM server")

    # ========================================================================
    # Public API Methods (Facade Interface)
    # ========================================================================
    # These methods provide a simple interface to complex subsystem operations
    # Each method delegates to the appropriate manager component

    def chat(self, message: str) -> str:
        """Send a chat message and get a response from the LLM.

        This is the primary method for user interaction. It delegates to the
        ChatEngine which handles all the complexity of:
        - Adding message to history
        - Deciding between direct chat vs MCP tool calling
        - Calling the LLM (via OpenAI client or HTTP)
        - Formatting and displaying the response
        - Managing conversation context

        Args:
            message: User's chat message

        Returns:
            str: Assistant's response text

        Flow:
            User message → ChatEngine.chat()
                         ↓
            If MCP enabled: → Tool selection → Tool execution → Format result
            If MCP disabled: → Direct LLM call
                         ↓
            Response displayed and returned

        Example:
            >>> client = ChatClient(config)
            >>> response = client.chat("What is 2+2?")
            >>> print(response)  # "2 + 2 equals 4"
        """
        return self.chat_engine.chat(message)

    def clear_history(self) -> None:
        """Clear the conversation history.

        Removes all messages from the conversation context. Useful when:
        - Starting a new conversation topic
        - Context window is getting full
        - Want to reset the conversation state

        Why clearing history matters:
        - LLMs have limited context windows (e.g., 4096 tokens)
        - Older messages consume context budget
        - Clearing frees up context for new conversation

        Note: This only clears local history. The LLM has no memory beyond
        the current context window - each API call is stateless.

        Example:
            >>> client.clear_history()
            🧹 Conversation history cleared
        """
        self.history_manager.clear_history()

    def show_history(self) -> None:
        """Display the conversation history in a formatted table.

        Shows all messages exchanged in the current session:
        - Turn number
        - Role (User/Assistant)
        - Message content (truncated if long)

        Useful for:
        - Reviewing what was discussed
        - Debugging conversation context issues
        - Understanding what the LLM "sees"

        Example output:
            ┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃ Turn┃ Role      ┃ Content                      ┃
            ┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │ 1   │ User      │ What is 2+2?                 │
            │ 2   │ Assistant │ 2 + 2 equals 4               │
            └─────┴───────────┴──────────────────────────────┘
        """
        self.history_manager.show_history()

    def show_mcp_status(self) -> None:
        """Display MCP (Model Context Protocol) integration status.

        Shows detailed information about MCP tool availability:
        - Which MCP servers are configured
        - Connection status of each server
        - Available tools from each server
        - Any connection errors

        MCP (Model Context Protocol):
        - Standard for connecting LLMs to external tools
        - Enables LLMs to interact with APIs, databases, files, etc.
        - Each "server" provides one or more "tools"

        Example output:
            ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
            ┃ Server    ┃ Status                   ┃ Tools           ┃
            ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
            │ weather   │ ✓ Connected (2 tools)    │ get, forecast   │
            │ filesystem│ ✓ Connected (5 tools)    │ read, write,... │
            └───────────┴──────────────────────────┴─────────────────┘
        """
        self.ui_manager.show_mcp_status()

    def get_available_models(self):
        """Query the vLLM server for available models.

        Calls the OpenAI-compatible /v1/models endpoint to discover which
        models are loaded on the server.

        Returns:
            List[str]: List of model identifiers (e.g., ["microsoft/Phi-3-mini-4k-instruct"])
                      Empty list if query fails

        Use cases:
        - Auto-detect model when user doesn't specify one
        - Validate that requested model exists
        - Display model options to user

        Why this is useful:
        - vLLM servers can load different models at different times
        - User may not know which model is currently loaded
        - Prevents errors from requesting non-existent models

        Example:
            >>> models = client.get_available_models()
            >>> print(models)
            ['microsoft/Phi-3-mini-4k-instruct', 'meta-llama/Llama-2-7b-hf']
        """
        return self.connection_manager.get_available_models()

    def set_model(self, model: str):
        """Change the model to use for subsequent chat messages.

        Updates the configuration to use a different model. The change takes
        effect immediately for the next chat() call.

        Args:
            model: Model identifier (e.g., "microsoft/Phi-3-mini-4k-instruct")

        Important notes:
        - Does NOT validate that the model exists on the server
        - Does NOT clear conversation history (consider doing this manually)
        - Model name must match exactly what the server expects

        Best practice:
            >>> # First check what models are available
            >>> available = client.get_available_models()
            >>> if "new-model" in available:
            >>>     client.set_model("new-model")
            >>>     client.clear_history()  # Start fresh with new model

        Example:
            >>> client.set_model("microsoft/Phi-3-mini-4k-instruct")
            >>> response = client.chat("Hello")  # Uses new model
        """
        self.config.model = model


# ============================================================================
# Design Pattern Summary: Facade Pattern
# ============================================================================
"""
The ChatClient class demonstrates the Facade Pattern - one of the most useful
structural design patterns in software engineering.

PROBLEM: Complex subsystems are hard to use
- 6 different managers with different interfaces
- User needs to understand all components
- Initialization order matters
- Error-prone to use directly

SOLUTION: Provide a unified, simplified interface
- ChatClient hides all complexity
- User only needs to know: chat(), clear_history(), etc.
- Internally coordinates all managers
- Handles initialization order automatically

BENEFITS:
1. Simplicity: Easy to use - just call chat()
2. Decoupling: User code doesn't depend on managers
3. Maintainability: Can change managers without affecting users
4. Testability: Can mock the facade for testing

EXAMPLE USAGE:

    # Without facade (complex):
    config = ChatConfig(...)
    conn_mgr = ConnectionManager(config)
    tool_mgr = ToolManager(config)
    resp_handler = ResponseHandler(config)
    hist_mgr = HistoryManager(config)
    ui_mgr = UIManager(config)
    engine = ChatEngine(config, tool_mgr, resp_handler, hist_mgr)

    if not conn_mgr.test_connection():
        raise ConnectionError("Failed to connect")

    hist_mgr.add_message("user", "Hello")
    response = engine.chat("Hello")
    # ... many more steps ...

    # With facade (simple):
    client = ChatClient(config)  # Handles all initialization
    response = client.chat("Hello")  # That's it!

RELATED PATTERNS:
- Dependency Injection: ChatClient injects config into all managers
- Composition: ChatClient is composed of managers (vs inheritance)
- Single Responsibility: Each manager has one job
- Fail-Fast: Connection tested in __init__, not during first use

LEARNING TAKEAWAY:
When building complex systems, provide a simple facade for common use cases.
Internal complexity is fine, but don't force users to deal with it.
"""