"""
Core chat functionality for both direct and MCP-enabled conversations.
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
    """Handles core chat functionality."""

    def __init__(self, config: ChatConfig, tool_manager: ToolManager,
                 response_handler: ResponseHandler, history_manager: HistoryManager):
        self.config = config
        self.tool_manager = tool_manager
        self.response_handler = response_handler
        self.history_manager = history_manager
        self.console = Console() if Console is not None else None

        # Set up debug logging if enabled
        if self.config.debug:
            logging.basicConfig(
                filename='llm_debug.log',
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w'  # Overwrite each run
            )
            self.logger = logging.getLogger('llm_debug')
        else:
            self.logger = None

        # Initialize OpenAI client for LLM interactions
        if openai is not None:
            self.openai_client = openai.OpenAI(
                base_url=f"{self.config.base_url}/v1",
                api_key="dummy"  # vLLM doesn't require authentication
            )
        else:
            self.openai_client = None

        # Initialize HTTP session
        self.session = requests.Session()

    def chat(self, message: str) -> str:
        """Send a chat message and get response."""
        # Add user message to history
        self.history_manager.add_message("user", message)

        # Use MCP tools if enabled and available, otherwise use direct API
        if self.config.enable_mcp and self.tool_manager.has_tools():
            return self._chat_with_mcp_tools(message)
        else:
            return self._chat_direct(message)

    def _chat_with_mcp_tools(self, message: str) -> str:
        """Chat using MCP tools with a single LLM call for tool selection and parameter extraction."""
        try:
            # Format tools for the LLM
            tools_formatted = self.tool_manager.format_tools_for_llm()

            # Build the system prompt
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
                self.console.print(f"[dim]🔧 MCP mode active, sending to LLM with tools...[/dim]")

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
                    timeout=30
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
                        self.console.print(f"[green]🔧 LLM requested tool: {tool_name} with params: {params}[/green]")

                    # Execute the tool
                    result = self.tool_manager.execute_tool(tool_name, params)
                    if result:
                        # Now send the raw tool result to LLM for nice formatting
                        formatting_messages = [
                            {"role": "system", "content": f"""You are a helpful assistant. The user asked a question and you used a tool to get the answer.

Format the tool result into a natural, human-readable response. Make it conversational and easy to understand, like you're explaining the information to a friend. Include:

- Which tool function was called and what parameters were used (briefly)
- The key information from the results in natural language
- Keep it concise but informative
- Use simple formatting if needed, but avoid complex tables or charts unless absolutely necessary

Do not show raw JSON data. Provide a natural, conversational response.

Tool call details:
- Function: {tool_name}
- Parameters: {params}
- Raw result: {result}

Respond naturally as if you're answering the user's original question."""},
                            {"role": "user", "content": f"Please format this tool result naturally: {result}"}
                        ]

                        # Call LLM for formatting
                        if self.openai_client:
                            format_response = self.openai_client.chat.completions.create(
                                model=self.config.model,
                                messages=formatting_messages,
                                temperature=0.3,  # Lower temperature for consistent formatting
                                max_tokens=self.config.max_tokens,
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
                                "max_tokens": self.config.max_tokens,
                            }
                            format_http_response = self.session.post(
                                f"{self.config.base_url}/v1/chat/completions",
                                json=format_payload,
                                headers={"Content-Type": "application/json"},
                                timeout=30
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
                            stream=self.config.stream
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
                        stream=self.config.stream
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