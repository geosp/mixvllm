#!/usr/bin/env python3
"""
Enhanced CLI Chat Client for vLLM Server with MCP Tool Support

A chat client that connects to a running vLLM server and supports MCP tools.
"""

import argparse
import json
import sys
from typing import List, Dict, Any, Optional
import os
import logging
import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
import openai


class ChatClient:
    """Enhanced chat client for vLLM OpenAI-compatible API with MCP tool support."""

    def __init__(self, base_url: str = "http://localhost:8000", model: str = None,
                 enable_mcp: bool = False, debug: bool = False, mcp_config_path: str = None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.enable_mcp = enable_mcp
        self.debug = debug
        self.mcp_config_path = mcp_config_path

        # Set up debug logging if enabled
        if self.debug:
            logging.basicConfig(
                filename='llm_debug.log',
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w'  # Overwrite each run
            )
            self.logger = logging.getLogger('llm_debug')
        else:
            self.logger = None

        self.conversation_history: List[Dict[str, str]] = []
        self.session = requests.Session()

        # Initialize rich console if available
        self.console = Console() if Console is not None else None

        # Initialize OpenAI client for LLM interactions
        if openai is not None:
            self.openai_client = openai.OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="dummy"  # vLLM doesn't require authentication
            )
        else:
            self.openai_client = None

        # Initialize MCP agent if enabled
        self.agent = None
        if self.enable_mcp:
            self._setup_mcp_agent()

        # Test connection
        self._test_connection()

    def _setup_mcp_agent(self):
        """Set up the MCP-enabled LangChain agent."""
        try:
            from .mcp_tools import get_available_mcp_tools

            # For now, disable the LangChain agent and use direct MCP tool calling
            # This avoids complex LangChain integration issues
            tools = get_available_mcp_tools(self.mcp_config_path)
            
            if not tools:
                raise ValueError("No MCP tools available")
            
            # Store tools for direct calling instead of using LangChain agent
            self.mcp_tools = {tool.name: tool for tool in tools}
            self.agent = None  # Will implement direct tool calling

            if self.console:
                self.console.print(f"[green]✓[/green] MCP tools enabled ({len(tools)} tools available)")
            else:
                print(f"✓ MCP tools enabled ({len(tools)} tools available)")

        except Exception as e:
            if self.console:
                self.console.print(f"[yellow]⚠[/yellow] Failed to setup MCP agent: {e}")
                self.console.print("[dim]Falling back to simple chat mode[/dim]")
            else:
                print(f"⚠ Failed to setup MCP agent: {e}")
                print("Falling back to simple chat mode")
            self.enable_mcp = False

    def _test_connection(self) -> None:
        """Test connection to the vLLM server."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                if self.console:
                    self.console.print(f"[green]✓[/green] Connected to vLLM server at {self.base_url}")
                else:
                    print(f"✓ Connected to vLLM server at {self.base_url}")
            else:
                if self.console:
                    self.console.print(f"[yellow]⚠[/yellow] Server responded with status {response.status_code}")
                else:
                    print(f"⚠ Server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            if self.console:
                self.console.print(f"[red]❌[/red] Cannot connect to server: {e}")
                self.console.print(f"Make sure the vLLM server is running at {self.base_url}")
            else:
                print(f"❌ Cannot connect to server: {e}")
                print(f"Make sure the vLLM server is running at {self.base_url}")
            sys.exit(1)

    def _get_available_models(self) -> List[str]:
        """Get list of available models from the server."""
        try:
            response = self.session.get(f"{self.base_url}/v1/models")
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

    def chat(self, message: str, temperature: float = 0.7, max_tokens: int = 512,
             stream: bool = False) -> str:
        """Send a chat message and get response."""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})

        # Use MCP tools if enabled and available, otherwise use direct API
        if self.enable_mcp and hasattr(self, 'mcp_tools') and self.mcp_tools:
            return self._chat_with_mcp_tools(message, temperature, max_tokens, stream)
        else:
            return self._chat_direct(message, temperature, max_tokens, stream)

    def _chat_with_mcp_tools(self, message: str, temperature: float, max_tokens: int, stream: bool) -> str:
        """Chat using MCP tools with a single LLM call for tool selection and parameter extraction."""
        try:
            # Format tools for the LLM
            tools_formatted = self._format_tools_for_llm()
            
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
            for hist_msg in self.conversation_history[:-1]:  # Exclude the current user message
                if hist_msg["role"] != "system":
                    messages.append(hist_msg)
            
            # Add the current user message
            messages.append({"role": "user", "content": message})
            
            # Log the prompt being sent to LLM if debug enabled
            if self.debug:
                self.logger.debug("=== MCP TOOLS PROMPT ===")
                for msg in messages:
                    self.logger.debug(f"{msg['role'].upper()}: {msg['content'][:500]}{'...' if len(msg['content']) > 500 else ''}")
                self.logger.debug("=== END MCP PROMPT ===")
            
            # Call the LLM using OpenAI client
            if self.console:
                self.console.print(f"[dim]🔧 MCP mode active, sending to LLM with tools...[/dim]")
            
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                llm_response = response.choices[0].message.content
                if llm_response is None:
                    llm_response = ""
                else:
                    llm_response = llm_response.strip()
                
                # Log the LLM response if debug enabled
                if self.debug:
                    self.logger.debug("=== MCP TOOLS RESPONSE ===")
                    self.logger.debug(f"Response: {llm_response[:500]}{'...' if len(llm_response) > 500 else ''}")
                    self.logger.debug("=== END MCP RESPONSE ===")
            else:
                # Fallback to direct HTTP if OpenAI client not available
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                http_response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
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
                    if self.debug:
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
                    if tool_name in self.mcp_tools:
                        tool_obj = self.mcp_tools[tool_name]
                        result = self._execute_tool_direct(tool_obj, params)
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
                                    model=self.model,
                                    messages=formatting_messages,
                                    temperature=0.3,  # Lower temperature for consistent formatting
                                    max_tokens=max_tokens,
                                )
                                formatted_result = format_response.choices[0].message.content
                                if formatted_result is None:
                                    formatted_result = result  # Fallback to raw result
                                else:
                                    formatted_result = formatted_result.strip()
                            else:
                                # Fallback HTTP call
                                format_payload = {
                                    "model": self.model,
                                    "messages": formatting_messages,
                                    "temperature": 0.3,
                                    "max_tokens": max_tokens,
                                }
                                format_http_response = self.session.post(
                                    f"{self.base_url}/v1/chat/completions",
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
                            if self.console:
                                if any(marker in formatted_result for marker in ['**', '*', '`', '#', '-', '1.']):
                                    markdown = Markdown(formatted_result)
                                    self.console.print(Panel(markdown, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
                                else:
                                    self.console.print(Panel(formatted_result, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
                            else:
                                print(f"Assistant: {formatted_result}")
                            
                            # Add to history
                            self.conversation_history.append({"role": "assistant", "content": formatted_result})
                            return formatted_result
                        else:
                            result = "Tool execution failed."
                            if self.console:
                                self.console.print(Panel(result, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
                            else:
                                print(f"Assistant: {result}")
                            self.conversation_history.append({"role": "assistant", "content": result})
                            return result
                    else:
                        result = f"Unknown tool: {tool_name}"
                        if self.console:
                            self.console.print(Panel(result, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
                        else:
                            print(f"Assistant: {result}")
                        self.conversation_history.append({"role": "assistant", "content": result})
                        return result
                else:
                    # Not a tool call, return as normal response
                    # Display the response
                    if self.console:
                        if any(marker in llm_response for marker in ['**', '*', '`', '#', '-', '1.']):
                            markdown = Markdown(llm_response)
                            self.console.print(Panel(markdown, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
                        else:
                            self.console.print(Panel(llm_response, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
                    else:
                        print(f"Assistant: {llm_response}")
                    
                    # Add to history
                    self.conversation_history.append({"role": "assistant", "content": llm_response})
                    return llm_response
            except json.JSONDecodeError:
                # Not JSON, treat as normal response
                # Display the response
                if self.console:
                    if any(marker in llm_response for marker in ['**', '*', '`', '#', '-', '1.']):
                        markdown = Markdown(llm_response)
                        self.console.print(Panel(markdown, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
                    else:
                        self.console.print(Panel(llm_response, title="[bold blue]🤖 Assistant (with tools)[/bold blue]", border_style="blue"))
                else:
                    print(f"Assistant: {llm_response}")
                
                # Add to history
                self.conversation_history.append({"role": "assistant", "content": llm_response})
                return llm_response
        except Exception as e:
            error_msg = f"MCP tools error: {str(e)}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            return f"Error: {error_msg}"

    def _execute_tool_direct(self, tool_obj, params: dict) -> Optional[str]:
        """Execute a tool directly with provided parameters."""
        try:
            # Debug: Show which tool we're executing
            if self.console:
                self.console.print(f"[dim]🔍 Executing tool: {tool_obj.name if hasattr(tool_obj, 'name') else 'unknown'}[/dim]")
                self.console.print(f"[dim]🔍 Parameters: {params}[/dim]")
            
            # Call the tool's _run method with the parameters
            result = tool_obj._run(**params)
            
            if self.console:
                self.console.print(f"[dim]🔍 Tool result: {result[:100]}...[/dim]")
            
            return str(result)
            
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            return error_msg

    def _chat_direct(self, message: str, temperature: float, max_tokens: int, stream: bool) -> str:
        """Chat using direct vLLM API calls."""
        try:
            # Debug logging: log the prompt being sent
            if self.debug:
                self.logger.debug("=== DIRECT CHAT PROMPT ===")
                self.logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens}, Stream: {stream}")
                for msg in self.conversation_history:
                    self.logger.debug(f"{msg['role'].upper()}: {msg['content'][:500]}{'...' if len(msg['content']) > 500 else ''}")

            if self.console and not stream:
                with self.console.status("[bold green]Thinking...", spinner="dots") as status:
                    if self.openai_client:
                        response = self.openai_client.chat.completions.create(
                            model=self.model,
                            messages=self.conversation_history,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=stream
                        )
                    else:
                        # Fallback to direct HTTP
                        payload = {
                            "model": self.model,
                            "messages": self.conversation_history,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "stream": stream
                        }
                        response = self.session.post(
                            f"{self.base_url}/v1/chat/completions",
                            json=payload,
                            headers={"Content-Type": "application/json"},
                            stream=stream
                        )
            else:
                if self.openai_client:
                    response = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream
                    )
                else:
                    # Fallback to direct HTTP
                    payload = {
                        "model": self.model,
                        "messages": self.conversation_history,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": stream
                    }
                    response = self.session.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        stream=stream
                    )

            # Check for HTTP errors (only for HTTP responses, not OpenAI client responses)
            if not hasattr(response, 'choices') and response.status_code != 200:
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

            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_regular_response(response)

        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg

    def _handle_regular_response(self, response) -> str:
        """Handle non-streaming response."""
        try:
            # Check if this is an OpenAI client response or raw HTTP response
            if hasattr(response, 'choices'):  # OpenAI client response
                assistant_message = response.choices[0].message.content
            else:  # Raw HTTP response
                data = response.json()
                choice = data['choices'][0]
                assistant_message = choice['message']['content']
            
            # Handle None content
            if assistant_message is None:
                assistant_message = ""

            # Debug logging: log the response received
            if self.debug:
                self.logger.debug("=== DIRECT CHAT RESPONSE ===")
                self.logger.debug(f"Response: {assistant_message[:500]}{'...' if len(assistant_message) > 500 else ''}")

            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            # Display with rich formatting
            if self.console:
                # Check if response contains markdown-like content
                if any(marker in assistant_message for marker in ['**', '*', '`', '#', '-', '1.']):
                    markdown = Markdown(assistant_message)
                    self.console.print(Panel(markdown, title="[bold blue]🤖 Assistant[/bold blue]", border_style="blue"))
                else:
                    self.console.print(Panel(assistant_message, title="[bold blue]🤖 Assistant[/bold blue]", border_style="blue"))
            else:
                print(f"Assistant: {assistant_message}")

            return assistant_message

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            error_msg = f"Invalid response format: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg

    def _handle_streaming_response(self, response) -> str:
        """Handle streaming response."""
        full_content = ""
        try:
            if self.console:
                # Use rich Live for streaming display
                with Live(console=self.console, refresh_per_second=10) as live:
                    current_text = Text("", style="blue")
                    live.update(Panel(current_text, title="[bold blue]🤖 Assistant (streaming)[/bold blue]", border_style="blue"))

                    # Check if this is OpenAI client streaming or raw HTTP streaming
                    if hasattr(response, '__iter__') and hasattr(response, '__next__'):  # OpenAI client streaming
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                delta = chunk.choices[0].delta.content
                                full_content += delta
                                current_text.plain = full_content
                                live.update(Panel(current_text, title="[bold blue]🤖 Assistant (streaming)[/bold blue]", border_style="blue"))
                    else:  # Raw HTTP streaming
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    data = line[6:]
                                    if data == '[DONE]':
                                        break

                                    try:
                                        chunk = json.loads(data)
                                        if chunk['choices'][0]['finish_reason'] is None:
                                            delta = chunk['choices'][0]['delta'].get('content', '')
                                            full_content += delta
                                            current_text.plain = full_content
                                            live.update(Panel(current_text, title="[bold blue]🤖 Assistant (streaming)[/bold blue]", border_style="blue"))
                                    except json.JSONDecodeError:
                                        continue
            else:
                # Fallback for no rich - handle both OpenAI and HTTP streaming
                if hasattr(response, '__iter__') and hasattr(response, '__next__'):  # OpenAI client streaming
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            delta = chunk.choices[0].delta.content
                            print(delta, end='', flush=True)
                            full_content += delta
                else:  # Raw HTTP streaming
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                data = line[6:]
                                if data == '[DONE]':
                                    break

                                try:
                                    chunk = json.loads(data)
                                    if chunk['choices'][0]['finish_reason'] is None:
                                        delta = chunk['choices'][0]['delta'].get('content', '')
                                        print(delta, end='', flush=True)
                                        full_content += delta
                                except json.JSONDecodeError:
                                    continue
                print()  # New line after streaming

            # Debug logging: log the streaming response received
            if self.debug:
                self.logger.debug("=== DIRECT CHAT STREAMING RESPONSE ===")
                self.logger.debug(f"Response: {full_content[:500]}{'...' if len(full_content) > 500 else ''}")

            # Add to history
            self.conversation_history.append({"role": "assistant", "content": full_content})
            return full_content

        except Exception as e:
            error_msg = f"Streaming error: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        if self.console:
            self.console.print("[green]🧹 Conversation history cleared[/green]")
        else:
            print("🧹 Conversation history cleared")

    def show_history(self) -> None:
        """Show conversation history."""
        if not self.conversation_history:
            if self.console:
                self.console.print("[dim]📝 No conversation history[/dim]")
            else:
                print("📝 No conversation history")
            return

        if self.console:
            table = Table(title="📝 Conversation History", show_header=True, header_style="bold magenta")
            table.add_column("Turn", style="cyan", no_wrap=True, width=4)
            table.add_column("Role", style="bold", width=10)
            table.add_column("Content", style="white", overflow="fold")

            for i, msg in enumerate(self.conversation_history, 1):
                role = msg['role'].title()
                content = msg['content']
                # Truncate long content for display
                if len(content) > 100:
                    content = content[:97] + "..."
                table.add_row(str(i), role, content)

            self.console.print(table)
        else:
            print("\n📝 Conversation History:")
            print("-" * 40)
            for i, msg in enumerate(self.conversation_history, 1):
                role = msg['role'].title()
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"{i}. {role}: {content}")
            print("-" * 40)

    def show_mcp_status(self) -> None:
        """Show MCP integration status."""
        if self.enable_mcp and self.agent:
            if self.console:
                from mcp_tools import get_mcp_tool_names, get_mcp_servers
                tools = get_mcp_tool_names()
                servers = get_mcp_servers()

                # Create a table showing servers and their status
                table = Table(title="🔧 MCP Integration Status")
                table.add_column("Server", style="cyan", no_wrap=True)
                table.add_column("Status", style="green")
                table.add_column("Tools", style="yellow")

                for server_name, server_info in servers.items():
                    # Test connection to get tool count
                    from mcp_client import test_mcp_connection
                    status = test_mcp_connection(server_name)

                    if status['status'] == 'connected':
                        status_text = f"✓ Connected ({status['tools_count']} tools)"
                        tools_text = ", ".join(status['tools']) if status['tools'] else "None"
                    else:
                        status_text = f"❌ {status['error'][:30]}..."
                        tools_text = "N/A"

                    table.add_row(server_name, status_text, tools_text)

                self.console.print(table)
            else:
                print("🔧 MCP Integration: Active")
                from mcp_tools import get_mcp_tool_names
                print(f"Available tools: {', '.join(get_mcp_tool_names())}")
        else:
            if self.console:
                self.console.print("[yellow]🔧 MCP Integration: Disabled[/yellow]")
                self.console.print("[dim]MCP tools require proper MCP server configuration[/dim]")
            else:
                print("🔧 MCP Integration: Disabled")
                print("MCP tools require proper MCP server configuration")

    def _format_tools_for_llm(self) -> str:
        """Format all available MCP tools for inclusion in LLM prompts."""
        if not hasattr(self, 'mcp_tools') or not self.mcp_tools:
            return "No tools available."
        
        tools_info = []
        for tool_name, tool_obj in self.mcp_tools.items():
            # Get tool description
            description = tool_obj.description if hasattr(tool_obj, 'description') else "No description"
            
            # Get parameters from args_schema
            params_info = []
            if hasattr(tool_obj, 'args_schema') and tool_obj.args_schema:
                schema = tool_obj.args_schema
                fields = None
                if hasattr(schema, 'model_fields'):
                    fields = schema.model_fields
                elif hasattr(schema, '__fields__'):
                    fields = schema.__fields__
                
                if fields:
                    for field_name, field_info in fields.items():
                        # Get field type
                        field_type = str(field_info.annotation).replace('typing.', '')
                        if hasattr(field_info.annotation, '__name__'):
                            field_type = field_info.annotation.__name__
                        
                        # Check if required
                        required = getattr(field_info, 'is_required', lambda: True)()
                        if hasattr(field_info, 'default') and field_info.default is not ...:
                            required = False
                        
                        desc = field_info.description or ""
                        params_info.append(f"  - {field_name} ({field_type}, {'required' if required else 'optional'}): {desc}")
            
            params_str = "\n".join(params_info) if params_info else "  No parameters required"
            
            tools_info.append(f"Tool: {tool_name}\nDescription: {description}\nParameters:\n{params_str}")
        
        return "\n\n".join(tools_info)


def show_welcome(console, model, url, enable_mcp=False, tools_count=0):
    """Show welcome message with rich formatting."""
    if console:
        from rich.panel import Panel
        from rich.text import Text
        
        welcome_text = Text()
        welcome_text.append("🤖 Enhanced vLLM Chat Client", style="bold blue")
        if enable_mcp:
            welcome_text.append(" (with MCP tools)", style="bold green")
        welcome_text.append("\n\n", style="")
        welcome_text.append("Configuration:\n", style="bold")
        welcome_text.append(f"• Server: {url}\n", style="")
        welcome_text.append(f"• Model: {model}\n", style="")
        if enable_mcp:
            welcome_text.append(f"• MCP Tools: Enabled ({tools_count} tools)\n", style="green")
        else:
            welcome_text.append("• MCP Tools: Disabled\n", style="")
        
        welcome_text.append("\nCommands: /help, /clear, /history", style="dim")
        if enable_mcp:
            welcome_text.append(", /mcp", style="dim")
        welcome_text.append(", /quit\n", style="dim")
        welcome_text.append("Type your message and press Enter to chat!", style="italic")
        
        console.print(Panel(welcome_text, title=":rocket: Welcome", border_style="blue"))
        
        if enable_mcp and tools_count > 0:
            # Show available tools
            from .mcp_tools import get_available_mcp_tools
            tools = get_available_mcp_tools()
            if tools:
                from rich.table import Table
                table = Table(title="Available MCP Tools", show_header=False)
                table.add_column("Tool", style="cyan", no_wrap=True)
                table.add_column("Description", style="white", overflow="fold")
                
                for tool in tools:
                    desc = tool.description[:80] + "..." if len(tool.description) > 80 else tool.description
                    table.add_row(tool.name, desc)
                
                console.print(table)
    else:
        print("🤖 Enhanced vLLM Chat Client" + (" (with MCP tools)" if enable_mcp else ""))
        print()
        print("Configuration:")
        print(f"• Server: {url}")
        print(f"• Model: {model}")
        if enable_mcp:
            print(f"• MCP Tools: Enabled ({tools_count} tools)")
        else:
            print("• MCP Tools: Disabled")
        print()
        print("Commands: /help, /clear, /history" + (", /mcp" if enable_mcp else "") + ", /quit")
        print("Type your message and press Enter to chat!")


def create_prompt_session():
    """Create an enhanced prompt session if available."""
    if PromptSession is not None:
        try:
            # Custom style
            style = Style.from_dict({
                'prompt': 'bold cyan',
            })
            
            return PromptSession(
                history=InMemoryHistory(),
                style=style,
                message="You: "
            )
        except ImportError:
            pass
    return None


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Enhanced CLI chat client for vLLM server with MCP tool support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base-url',
        default='http://localhost:8000',
        help='vLLM server base URL'
    )

    parser.add_argument(
        '--model',
        help='Model name (if not specified, will try to detect from server)'
    )

    parser.add_argument(
        '--enable-mcp',
        action='store_true',
        help='Enable MCP (Model Context Protocol) tools'
    )

    parser.add_argument(
        '--mcp-config',
        help='Path to MCP servers configuration file'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate'
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming responses'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging of LLM prompts and responses'
    )

    args = parser.parse_args()

    # Initialize client
    client = ChatClient(
        base_url=args.base_url,
        model=args.model,
        enable_mcp=args.enable_mcp,
        debug=args.debug,
        mcp_config_path=args.mcp_config
    )

    # Auto-detect model if not specified
    if not client.model:
        models = client._get_available_models()
        if models:
            client.model = models[0]
            if client.console:
                client.console.print(f"[green]✓[/green] Auto-selected model: {client.model}")
            else:
                print(f"✓ Auto-selected model: {client.model}")
        else:
            if client.console:
                client.console.print("[red]❌ No models available on server[/red]")
            else:
                print("❌ No models available on server")
            sys.exit(1)

    # Get tools count for welcome message
    tools_count = len(client.mcp_tools) if hasattr(client, 'mcp_tools') and client.mcp_tools else 0

    # Show welcome
    show_welcome(client.console, client.model, args.base_url, args.enable_mcp, tools_count)

    # Create prompt session
    session = create_prompt_session()

    try:
        while True:
            try:
                if session:
                    # Enhanced input with history
                    user_input = session.prompt().strip()
                else:
                    # Basic input
                    user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    cmd = user_input[1:].lower()
                    if cmd in ['quit', 'exit', 'q']:
                        if client.console:
                            client.console.print("[yellow]👋 Goodbye![/yellow]")
                        else:
                            print("👋 Goodbye!")
                        break
                    elif cmd == 'help':
                        if client.console:
                            from rich.table import Table
                            help_table = Table(title="Available Commands")
                            help_table.add_column("Command", style="cyan", no_wrap=True)
                            help_table.add_column("Description", style="white")
                            help_table.add_row("/help", "Show this help message")
                            help_table.add_row("/clear", "Clear conversation history")
                            help_table.add_row("/history", "Show conversation history")
                            if args.enable_mcp:
                                help_table.add_row("/mcp", "Show MCP integration status")
                            help_table.add_row("/quit", "Exit the chat")
                            client.console.print(help_table)
                        else:
                            print("Commands:")
                            print("  /help     - Show this help")
                            print("  /clear    - Clear conversation history")
                            print("  /history  - Show conversation history")
                            if args.enable_mcp:
                                print("  /mcp      - Show MCP integration status")
                            print("  /quit     - Exit the chat")
                        continue
                    elif cmd == 'clear':
                        client.clear_history()
                        continue
                    elif cmd == 'history':
                        client.show_history()
                        continue
                    elif cmd == 'mcp' and args.enable_mcp:
                        client.show_mcp_status()
                        continue
                    else:
                        if client.console:
                            client.console.print(f"[red]Unknown command: {user_input}[/red]")
                        else:
                            print(f"Unknown command: {user_input}")
                        continue

                # Send message
                response = client.chat(
                    user_input,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stream=args.stream
                )

            except KeyboardInterrupt:
                if client.console:
                    client.console.print("\n[yellow]👋 Goodbye![/yellow]")
                else:
                    print("\n👋 Goodbye!")
                break
            except EOFError:
                break

    except Exception as e:
        if client.console:
            client.console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()