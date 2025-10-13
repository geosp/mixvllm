"""
Core chat functionality for both direct and MCP-enabled conversations.

This module implements the Strategy Pattern for chat handling, switching between
two strategies based on configuration:
1. Direct chat: Simple LLM conversation
2. MCP-enhanced chat: LLM with tool calling capabilities

Learning Points:
- Strategy Pattern: Runtime selection between algorithms
- Two-phase prompting: Tool selection → Execution → Formatting
- OpenAI API: Standard interface for LLM communication
- Streaming vs non-streaming: Different response handling strategies
- Debug logging: Essential for troubleshooting LLM interactions
- JSON parsing: Structured communication with LLMs
"""

import json
import logging
import requests
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

import openai

from .config import ChatConfig
from .tool_manager import ToolManager
from .response_handler import ResponseHandler
from .history_manager import HistoryManager


class ChatEngine:
    """Core chat engine implementing two conversation strategies.

    This class demonstrates the Strategy Pattern where the chat method
    dynamically selects between two algorithms:
    - Direct chat: Standard LLM conversation
    - MCP chat: Tool-augmented LLM conversation

    Strategy Pattern Benefits:
    - Runtime algorithm selection (no recompilation)
    - Encapsulated algorithms (_chat_direct vs _chat_with_mcp_tools)
    - Easy to add new strategies without changing existing code

    Architecture:
        User Input
            ↓
        chat() - Strategy Selector
            ↓
        ┌─────────────────────────────┐
        │ if MCP enabled & has tools: │
        │   → _chat_with_mcp_tools()  │
        │ else:                       │
        │   → _chat_direct()          │
        └─────────────────────────────┘
            ↓
        Response

    Dependencies:
    - ToolManager: MCP tool discovery and execution
    - ResponseHandler: Response formatting and display
    - HistoryManager: Conversation context tracking
    """

    def __init__(self, config: ChatConfig, tool_manager: ToolManager,
                 response_handler: ResponseHandler, history_manager: HistoryManager):
        """Initialize the chat engine with dependencies.

        Args:
            config: Configuration object with all settings
            tool_manager: Handles MCP tool operations
            response_handler: Handles response display
            history_manager: Manages conversation history

        Initialization Steps:
        1. Store all dependencies (Dependency Injection)
        2. Setup debug logging if enabled
        3. Initialize OpenAI client for LLM communication
        4. Initialize HTTP session as fallback
        """
        self.config = config
        self.tool_manager = tool_manager
        self.response_handler = response_handler
        self.history_manager = history_manager
        self.console = Console() if Console is not None else None

        # ====================================================================
        # Debug Logging Setup
        # ====================================================================
        # Set up debug logging if enabled
        # Logs all LLM prompts and responses to llm_debug.log
        if self.config.debug:
            logging.basicConfig(
                filename='llm_debug.log',  # Log file location
                level=logging.DEBUG,       # Log level
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w'  # Overwrite each run (not append)
            )
            self.logger = logging.getLogger('llm_debug')
            # Why debug logging matters:
            # - LLM behavior is hard to predict
            # - Seeing exact prompts helps troubleshoot issues
            # - Response inspection reveals truncation, formatting problems
            # - Essential for prompt engineering and debugging
        else:
            self.logger = None

        # ====================================================================
        # OpenAI Client Setup
        # ====================================================================
        # Initialize OpenAI client for LLM interactions
        # vLLM implements OpenAI's API, so we can use their official client
        if openai is not None:
            self.openai_client = openai.OpenAI(
                base_url=f"{self.config.base_url}/v1",  # vLLM endpoint
                api_key="dummy",  # vLLM doesn't require authentication
                timeout=120.0  # 2 minute timeout for long responses
            )
            # Why use OpenAI client:
            # - Higher-level API than raw HTTP
            # - Built-in streaming support
            # - Better error handling
            # - Type hints and IDE support
        else:
            self.openai_client = None

        # ====================================================================
        # HTTP Session Setup (Fallback)
        # ====================================================================
        # Initialize HTTP session as fallback when OpenAI client unavailable
        self.session = requests.Session()
        # Session benefits: connection pooling, persistent headers

    # ========================================================================
    # Public API
    # ========================================================================

    def chat(self, message: str) -> str:
        """Send a chat message and get response (Strategy Pattern).

        This is the strategy selector - it chooses which algorithm to use
        based on runtime configuration.

        Strategy Selection Logic:
        - If MCP enabled AND tools available → _chat_with_mcp_tools()
        - Otherwise → _chat_direct()

        Args:
            message: User's chat message

        Returns:
            str: Assistant's response

        Flow:
            1. Add user message to history (for context)
            2. Check if MCP tools are enabled and available
            3. Select appropriate strategy
            4. Execute strategy and return response

        Why Strategy Pattern here:
        - Behavior changes based on configuration
        - Both strategies have same interface (input: str, output: str)
        - Easy to add new strategies (e.g., _chat_with_rag)
        """
        # Step 1: Add user message to history
        # This ensures LLM has conversation context
        self.history_manager.add_message("user", message)

        # Step 2: Strategy selection
        # Use MCP tools if enabled and available, otherwise use direct API
        if self.config.enable_mcp and self.tool_manager.has_tools():
            return self._chat_with_mcp_tools(message)
        else:
            return self._chat_direct(message)

    # ========================================================================
    # Strategy Implementations (Private Methods)
    # ========================================================================

    def _chat_with_mcp_tools(self, message: str) -> str:
        """MCP-enhanced chat strategy using tool calling (Two-Phase Prompting).

        This implements a sophisticated tool calling pattern:
        Phase 1: Tool Selection - LLM decides which tool to use and extracts parameters
        Phase 2: Tool Execution - Execute the selected tool
        Phase 3: Result Formatting - LLM formats raw results nicely

        Why Two-Phase Prompting:
        - Separates tool selection from result presentation
        - Enables complex parameter extraction from natural language
        - Produces better formatted output than single-phase approaches
        - More robust than function calling APIs (works with any LLM)

        Tool Calling Flow:
            User: "What's the weather in San Francisco?"
                ↓
            LLM Phase 1: {"tool": "weather_get", "parameters": {"city": "San Francisco"}}
                ↓
            Execute: weather_get(city="San Francisco") → "72°F, sunny"
                ↓
            LLM Phase 2: "The weather in San Francisco is 72°F and sunny!"
                ↓
            Display formatted result to user

        Args:
            message: User's query (may require tool use)

        Returns:
            str: Formatted response (with or without tool results)

        Implementation Notes:
        - Uses JSON for structured LLM responses
        - Try/except handles both tool calls and regular responses
        - Lower temperature (0.3) for formatting ensures consistency
        - High max_tokens (16384) accommodates large tool results
        """
        try:
            # ================================================================
            # Step 1: Prepare Tool Information for LLM
            # ================================================================
            # Format all available tools into a human-readable string
            # Includes: tool name, description, parameters
            tools_formatted = self.tool_manager.format_tools_for_llm()

            # ================================================================
            # Step 2: Build System Prompt (Tool Calling Instructions)
            # ================================================================
            # This prompt teaches the LLM how to use tools via JSON
            system_prompt = f"""You are a helpful assistant with access to the following tools:

{tools_formatted}

When you need to use a tool to answer the user's question, respond with a JSON object in this exact format:
{{"tool": "tool_name", "parameters": {{"param1": "value1", "param2": "value2"}}}}

If you can answer the question without using any tools, just respond normally with your helpful answer.

When presenting tool results to the user, format them as a nice, human-readable report. Include:
- A header showing which tool function was called and with what parameters
- Well-formatted, easy-to-read information extracted from the tool results
- Use appropriate emojis and formatting to make the report visually appealing
- Do not show raw JSON data - always format it nicely

Do not mention the tools or JSON format in your normal responses."""
            # Prompt Engineering Notes:
            # - Clear JSON format specification reduces parsing errors
            # - "exact format" instruction improves compliance
            # - Formatting guidelines ensure good UX
            # - "Do not mention" prevents LLM from exposing internals

            # Build conversation messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history (excluding system messages)
            for hist_msg in self.history_manager.get_history()[:-1]:  # Exclude the current user message
                if hist_msg["role"] != "system":
                    messages.append(hist_msg)

            # Add the current user message
            messages.append({"role": "user", "content": message})

            # Log the prompt being sent to LLM if debug enabled
            if self.config.debug:
                self.logger.debug("=== MCP TOOLS PROMPT ===")
                for msg in messages:
                    self.logger.debug(f"{msg['role'].upper()}: {msg['content'][:500]}{'...' if len(msg['content']) > 500 else ''}")
                self.logger.debug("=== END MCP PROMPT ===")

            # Call the LLM using OpenAI client
            if self.console:
                self.console.print(f"[cyan]🔧 MCP mode active, sending to LLM with tools...[/cyan]")

            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                llm_response = response.choices[0].message.content
                if llm_response is None:
                    llm_response = ""
                else:
                    llm_response = llm_response.strip()

                # Log the LLM response if debug enabled
                if self.config.debug:
                    self.logger.debug("=== MCP TOOLS RESPONSE ===")
                    self.logger.debug(f"Response: {llm_response[:500]}{'...' if len(llm_response) > 500 else ''}")
                    self.logger.debug("=== END MCP RESPONSE ===")
            else:
                # Fallback to direct HTTP if OpenAI client not available
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                http_response = self.session.post(
                    f"{self.config.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120  # Increased timeout for long responses
                )
                if http_response.status_code == 200:
                    data = http_response.json()
                    content = data['choices'][0]['message']['content']
                    if content is None:
                        llm_response = ""
                    else:
                        llm_response = content.strip()

                    # Log the LLM response if debug enabled
                    if self.config.debug:
                        self.logger.debug("=== MCP TOOLS RESPONSE (HTTP) ===")
                        self.logger.debug(f"Response: {llm_response[:500]}{'...' if len(llm_response) > 500 else ''}")
                        self.logger.debug("=== END MCP RESPONSE ===")
                else:
                    error_msg = f"LLM API error: {http_response.status_code}"
                    if self.console:
                        self.console.print(f"[red]❌ {error_msg}[/red]")
                    return f"Error: {error_msg}"

            # Try to parse as JSON for tool call
            try:
                import json
                tool_call = json.loads(llm_response)
                if isinstance(tool_call, dict) and "tool" in tool_call and "parameters" in tool_call:
                    # This is a tool call
                    tool_name = tool_call["tool"]
                    params = tool_call["parameters"]

                    if self.console:
                        self.console.print(f"[green]🔧 Tool Call: {tool_name} with parameters: {params}[/green]")

                    # Execute the tool
                    result = self.tool_manager.execute_tool(tool_name, params)
                    if result:
                        # Now send the raw tool result to LLM for nice formatting
                        formatting_messages = [
                            {"role": "system", "content": f"""Present the tool result in a helpful way for the user.

Tool: {tool_name}
Parameters: {params}
Result: {result}"""},
                            {"role": "user", "content": message}
                        ]

                        # Call LLM for formatting
                        if self.openai_client:
                            format_response = self.openai_client.chat.completions.create(
                                model=self.config.model,
                                messages=formatting_messages,
                                temperature=0.3,  # Lower temperature for consistent formatting
                                max_tokens=16384,  # Large token limit for long tool results
                            )
                            formatted_result = format_response.choices[0].message.content
                            if formatted_result is None:
                                formatted_result = result  # Fallback to raw result
                            else:
                                formatted_result = formatted_result.strip()
                        else:
                            # Fallback HTTP call
                            format_payload = {
                                "model": self.config.model,
                                "messages": formatting_messages,
                                "temperature": 0.3,
                                "max_tokens": 16384,  # Large token limit for long tool results
                            }
                            format_http_response = self.session.post(
                                f"{self.config.base_url}/v1/chat/completions",
                                json=format_payload,
                                headers={"Content-Type": "application/json"},
                                timeout=120  # Increased timeout for long responses
                            )
                            if format_http_response.status_code == 200:
                                format_data = format_http_response.json()
                                content = format_data['choices'][0]['message']['content']
                                if content is None:
                                    formatted_result = result  # Fallback to raw result
                                else:
                                    formatted_result = content.strip()
                            else:
                                formatted_result = result  # Fallback to raw result

                        # Display the formatted result
                        self.response_handler.display_response(formatted_result, "🤖 Assistant (with tools)", is_tool_response=True)

                        # Add to history
                        self.history_manager.add_message("assistant", formatted_result)
                        return formatted_result
                    else:
                        result = "Tool execution failed."
                        self.response_handler.display_response(result, "🤖 Assistant (with tools)", is_tool_response=True)
                        self.history_manager.add_message("assistant", result)
                        return result
                else:
                    # Not a tool call, return as normal response
                    # Display the response
                    self.response_handler.display_response(llm_response, "🤖 Assistant (with tools)", is_tool_response=True)

                    # Add to history
                    self.history_manager.add_message("assistant", llm_response)
                    return llm_response
            except json.JSONDecodeError:
                # Not JSON, treat as normal response
                # Display the response
                self.response_handler.display_response(llm_response, "🤖 Assistant (with tools)", is_tool_response=True)

                # Add to history
                self.history_manager.add_message("assistant", llm_response)
                return llm_response
        except Exception as e:
            error_msg = f"MCP tools error: {str(e)}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            return f"Error: {error_msg}"

    def _chat_direct(self, message: str) -> str:
        """Chat using direct vLLM API calls."""
        try:
            # Debug logging: log the prompt being sent
            if self.config.debug:
                self.logger.debug("=== DIRECT CHAT PROMPT ===")
                self.logger.debug(f"Temperature: {self.config.temperature}, Max tokens: {self.config.max_tokens}, Stream: {self.config.stream}")
                for msg in self.history_manager.get_history():
                    self.logger.debug(f"{msg['role'].upper()}: {msg['content'][:500]}{'...' if len(msg['content']) > 500 else ''}")

            if self.console and not self.config.stream:
                with self.console.status("[bold green]Thinking...", spinner="dots") as status:
                    if self.openai_client:
                        response = self.openai_client.chat.completions.create(
                            model=self.config.model,
                            messages=self.history_manager.get_history(),
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            stream=self.config.stream
                        )
                    else:
                        # Fallback to direct HTTP
                        payload = {
                            "model": self.config.model,
                            "messages": self.history_manager.get_history(),
                            "temperature": self.config.temperature,
                            "max_tokens": self.config.max_tokens,
                            "stream": self.config.stream
                        }
                        response = self.session.post(
                            f"{self.config.base_url}/v1/chat/completions",
                            json=payload,
                            headers={"Content-Type": "application/json"},
                            stream=self.config.stream,
                            timeout=120  # Increased timeout for long responses
                        )
            else:
                if self.openai_client:
                    response = self.openai_client.chat.completions.create(
                        model=self.config.model,
                        messages=self.history_manager.get_history(),
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        stream=self.config.stream
                    )
                else:
                    # Fallback to direct HTTP
                    payload = {
                        "model": self.config.model,
                        "messages": self.history_manager.get_history(),
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "stream": self.config.stream
                    }
                    response = self.session.post(
                        f"{self.config.base_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        stream=self.config.stream,
                        timeout=120  # Increased timeout for long responses
                    )

            # Check for HTTP errors (only for HTTP responses, not OpenAI client responses)
            if not hasattr(response, 'choices') and not self.config.stream and response.status_code != 200:
                error_msg = f"Server error: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text[:200]}"

                if self.console:
                    self.console.print(f"[red]❌ {error_msg}[/red]")
                else:
                    print(f"❌ {error_msg}")
                return error_msg

            if self.config.stream:
                return self.response_handler.handle_streaming_response(response, self.history_manager.get_history())
            else:
                return self.response_handler.handle_regular_response(response, self.history_manager.get_history())

        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg


# ============================================================================
# Learning Summary: Two-Phase Prompting for Tool Calling
# ============================================================================
"""
This module demonstrates an advanced LLM pattern: Two-Phase Prompting for tool calling.

PROBLEM: How do we enable LLMs to use external tools?
- LLMs can't directly call APIs, query databases, or access files
- Need a bridge between LLM reasoning and tool execution
- Must handle parameter extraction from natural language
- Results should be formatted nicely, not raw data

SOLUTION: Two-Phase Prompting Pattern
Phase 1 - Tool Selection:
    User: "What's the weather in San Francisco?"
    ↓
    LLM + System Prompt with tool descriptions
    ↓
    LLM Output: {"tool": "weather_get", "parameters": {"city": "San Francisco"}}

Phase 2 - Tool Execution:
    Parse JSON → Extract tool_name and params
    ↓
    Execute: weather_get(city="San Francisco")
    ↓
    Result: {"temperature": 72, "condition": "sunny", "wind": "10 mph"}

Phase 3 - Result Formatting:
    Send result back to LLM with formatting instructions
    ↓
    LLM formats nicely:
    "The weather in San Francisco is currently 72°F and sunny, with light
     winds at 10 mph. Great day for outdoor activities! ☀️"

WHY THIS PATTERN WORKS:
1. JSON Parsing: Structured format easy to parse programmatically
2. Natural Language: User doesn't need to know JSON or tool syntax
3. Separation: Tool selection separate from result presentation
4. Flexibility: Works with any LLM (no special function calling API needed)
5. Better UX: Results formatted for humans, not machines

KEY IMPLEMENTATION DETAILS:

1. System Prompt Engineering:
   - Clearly specify JSON format: {"tool": "name", "parameters": {...}}
   - Provide tool descriptions and parameter schemas
   - Include formatting guidelines for results
   - Tell LLM when to use tools vs answering directly

2. Temperature Settings:
   - Phase 1 (tool selection): Use config temperature (e.g., 0.7)
     Allows some creativity in parameter interpretation
   - Phase 3 (formatting): Use low temperature (0.3)
     Ensures consistent, predictable formatting

3. Error Handling:
   - Try JSON parsing, fall back to regular response if not JSON
   - Handle tool execution failures gracefully
   - Provide error messages in formatted results

4. Context Management:
   - Include conversation history for context
   - Exclude system messages from history display
   - Current message added before processing

ALTERNATIVE APPROACHES:

1. Function Calling API (OpenAI, Claude):
   Pros: Built-in, structured, reliable
   Cons: Requires specific API support, vendor lock-in

2. ReAct Pattern (Reasoning + Acting):
   Pros: LLM can chain multiple tools
   Cons: More complex, multiple LLM calls

3. Single-Phase Prompting:
   Pros: Simpler, one LLM call
   Cons: Worse formatting, harder to parse

WHY WE CHOSE TWO-PHASE:
- Works with any OpenAI-compatible LLM (vLLM, Ollama, etc.)
- Better output quality than single-phase
- Simpler than full ReAct agent
- Easy to debug (can inspect each phase)

PERFORMANCE CONSIDERATIONS:
- Two LLM calls per tool use (selection + formatting)
- Higher latency than single-phase (~2x)
- Better UX justifies the cost
- Can cache tool descriptions to reduce prompt size

DEBUGGING TIPS:
Enable debug mode (--debug flag) to see:
- Exact system prompts sent to LLM
- LLM's raw JSON responses
- Tool execution results
- Final formatted output

This helps troubleshoot:
- Why LLM isn't using tools when expected
- JSON parsing failures
- Parameter extraction issues
- Formatting problems

EXTENDING THIS PATTERN:
To add new capabilities:
1. Add MCP server with new tools
2. Tool manager auto-discovers them
3. They appear in system prompt automatically
4. LLM can use them immediately

No code changes needed for new tools!
"""