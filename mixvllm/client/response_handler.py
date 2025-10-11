"""
Response handling for different types of chat responses.
"""

import json
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live

from .config import ChatConfig


class ResponseHandler:
    """Handles different types of chat responses."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.console = Console() if Console is not None else None

    def display_response(self, content: str, title: str = "🤖 Assistant", is_tool_response: bool = False):
        """Display a response with appropriate formatting."""
        if self.console:
            # Check if response contains markdown-like content
            if any(marker in content for marker in ['**', '*', '`', '#', '-', '1.']):
                markdown = Markdown(content)
                panel_title = f"[bold blue]{title}[/bold blue]" if not is_tool_response else f"[bold green]{title}[/bold green]"
                self.console.print(Panel(markdown, title=panel_title, border_style="blue" if not is_tool_response else "green"))
            else:
                panel_title = f"[bold blue]{title}[/bold blue]" if not is_tool_response else f"[bold green]{title}[/bold green]"
                self.console.print(Panel(content, title=panel_title, border_style="blue" if not is_tool_response else "green"))
        else:
            print(f"{title}: {content}")

    def handle_regular_response(self, response, conversation_history: List[Dict[str, str]]) -> str:
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
            if self.config.debug:
                self.logger.debug("=== DIRECT CHAT RESPONSE ===")
                self.logger.debug(f"Response: {assistant_message[:500]}{'...' if len(assistant_message) > 500 else ''}")

            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": assistant_message})

            # Display with rich formatting
            self.display_response(assistant_message)

            return assistant_message

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            error_msg = f"Invalid response format: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg

    def handle_streaming_response(self, response, conversation_history: List[Dict[str, str]]) -> str:
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
            if self.config.debug:
                self.logger.debug("=== DIRECT CHAT STREAMING RESPONSE ===")
                self.logger.debug(f"Response: {full_content[:500]}{'...' if len(full_content) > 500 else ''}")

            # Add to history
            conversation_history.append({"role": "assistant", "content": full_content})
            return full_content

        except Exception as e:
            error_msg = f"Streaming error: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return error_msg