# mixvllm-chat: Technical Architecture & Component Overview

## Overview

`mixvllm-chat` is a modular chat client for vLLM servers, supporting both direct LLM conversations and advanced Model Context Protocol (MCP) tool integration. It features a rich CLI, streaming responses, and a highly extensible architecture.

---

## Main Components

### 1. `chat_client.py` (Facade Pattern)
- Orchestrates all subsystems: connection, tools, response, history, UI, and chat engine.
- Provides a simple API (`chat()`, `clear_history()`, etc.) for user interaction.
- Uses Dependency Injection for testability and maintainability.
- Tests server connection on startup (Fail-Fast Principle).

### 2. `config.py`
- Centralized configuration using Python dataclasses.
- Handles server URL, model, MCP settings, generation parameters, and validation.
- Factory method `from_args()` builds config from CLI arguments.

### 3. `connection_manager.py`
- Manages HTTP connections to vLLM server.
- Uses `requests.Session` for connection pooling (performance).
- Health checks and model discovery via OpenAI-compatible endpoints.
- Rich console output for status and errors.

### 4. `tool_manager.py`
- Discovers and manages MCP tools from YAML config.
- Executes tools and formats them for LLM prompts.
- Handles errors and tool validation.
- Integrates with custom MCP client and tool registry.

### 5. `chat_engine.py` (Strategy Pattern)
- Core chat logic: switches between direct and MCP-enhanced chat.
- Handles two-phase prompting for tool selection and execution.
- Supports streaming and non-streaming responses.
- Uses `openai` client and HTTP fallback.

### 6. `response_handler.py`
- Formats and displays LLM responses (Markdown, LaTeX, plain text).
- Converts LaTeX to Unicode for terminal readability.
- Handles live streaming display with `rich.live.Live`.
- Robust error handling for malformed responses.

### 7. `history_manager.py`
- Tracks conversation context as a list of message dicts.
- Displays history in formatted tables using `rich.table.Table`.
- Supports clearing and reviewing history.

### 8. `ui_manager.py`
- Manages all user interface elements and status messages.
- Uses `rich` for panels, tables, markdown, and colored output.
- Displays welcome screen, help, errors, and MCP tool status.

### 9. `cli.py`
- Implements the REPL (Read-Eval-Print Loop) CLI interface.
- Uses `prompt_toolkit` for enhanced input (history, styling).
- Parses commands (`/help`, `/clear`, `/history`, `/mcp`, `/quit`).
- Handles argument parsing, auto-detection, and error recovery.

---

## MCP Integration

- MCP tools are loaded from YAML config and managed by `tool_manager.py`.
- Tool calls follow a three-phase flow: selection, execution, result formatting.
- MCP client and tools are implemented in `app/client/utils/mcp_client.py` and `mcp_tools.py`.

---

## Libraries Used

- `openai`: Official client for vLLM API compatibility.
- `requests`: HTTP client with connection pooling.
- `rich`: Terminal UI (panels, tables, markdown, live updates).
- `prompt_toolkit`: Enhanced CLI input.
- `pyyaml`: YAML config parsing.
- `dataclasses`: Type-safe configuration.
- Custom MCP client/tools for protocol integration.

---

## Technical Architecture

- Modular design: Each manager/component has a single responsibility.
- Facade and Strategy Patterns for orchestration and extensibility.
- Dependency Injection for testability.
- Progressive enhancement: Rich UI with fallback to basic output.
- Robust error handling and debug logging.

---

## Extensibility

- Add new MCP tools by updating YAML config and tool registry.
- New commands can be added to CLI with minimal changes.
- UI and chat logic are decoupled for easy upgrades (e.g., web UI).

---

## Testing & Debugging

- Debug mode logs all prompts, responses, and tool calls.
- Each component can be tested independently.
- CLI supports interactive and automated testing.

---

Let me know if you want to include more details from other subfolders (e.g., `terminal/`, `utils/`, etc.).
