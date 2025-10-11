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

    def _improve_latex_display(self, content: str) -> str:
        """Convert LaTeX expressions to more readable Unicode equivalents using pylatexenc."""
        try:
            from pylatexenc.latex2text import LatexNodes2Text
            
            # Create a LatexNodes2Text converter with unicode output
            converter = LatexNodes2Text(
                keep_inline_math=False,  # Convert inline math to unicode
                keep_comments=False,     # Remove LaTeX comments
                strict_latex_spaces=False,  # Be more flexible with spaces
            )
            
            import re
            
            # Only convert specific LaTeX patterns, not the entire content
            # Handle display math blocks \[...\] and $$...$$
            def convert_display_math(match):
                latex = match.group(1)
                try:
                    return f" {converter.latex_to_text(latex)} "
                except:
                    return match.group(0)  # Return original if conversion fails
            
            content = re.sub(r'\\\[(.*?)\\\]', convert_display_math, content, flags=re.DOTALL)
            content = re.sub(r'\$\$(.*?)\$\$', convert_display_math, content, flags=re.DOTALL)
            
            # Handle inline math $...$, but be more careful to not break markdown
            def convert_inline_math(match):
                latex = match.group(1)
                # Skip if it looks like a markdown table or other formatting
                if '|' in latex or '-' in latex or latex.strip() in ['', ' ']:
                    return match.group(0)
                try:
                    return converter.latex_to_text(latex)
                except:
                    return match.group(0)  # Return original if conversion fails
            
            content = re.sub(r'\$([^$\n]+)\$', convert_inline_math, content)
            
            # Manual replacements for common symbols that might not be in math mode
            symbol_replacements = {
                r'\\mu': 'μ', r'\\nu': 'ν', r'\\Lambda': 'Λ', r'\\lambda': 'λ',
                r'\\gamma': 'γ', r'\\alpha': 'α', r'\\beta': 'β', r'\\delta': 'δ',
                r'\\Delta': 'Δ', r'\\pi': 'π', r'\\sigma': 'σ', r'\\theta': 'θ',
                r'\\phi': 'φ', r'\\psi': 'ψ', r'\\omega': 'ω', r'\\Omega': 'Ω',
            }
            
            for latex_symbol, unicode_char in symbol_replacements.items():
                content = content.replace(latex_symbol, unicode_char)
                
        except ImportError:
            # Fallback to manual replacements if pylatexenc not available
            replacements = {
                r'\\mu': 'μ', r'\\nu': 'ν', r'\\Lambda': 'Λ', r'\\lambda': 'λ',
                r'\\gamma': 'γ', r'\\alpha': 'α', r'\\beta': 'β', r'\\delta': 'δ',
                r'\\Delta': 'Δ', r'\\pi': 'π', r'\\sigma': 'σ', r'\\theta': 'θ',
                r'\\phi': 'φ', r'\\psi': 'ψ', r'\\omega': 'ω', r'\\Omega': 'Ω',
                r'\\infty': '∞', r'\\approx': '≈', r'\\neq': '≠', r'\\leq': '≤',
                r'\\geq': '≥', r'\\pm': '±', r'\\times': '×', r'\\cdot': '·',
                r'\\partial': '∂', r'\\nabla': '∇', r'\\sum': '∑', r'\\int': '∫',
                r'\\sqrt': '√', r'\\frac': '/', r'\\^{2}': '²', r'\\^{3}': '³',
            }
            
            for latex, unicode_char in replacements.items():
                content = content.replace(latex, unicode_char)
            
            # Clean up LaTeX brackets but preserve markdown
            import re
            content = re.sub(r'\\\[|\\\]', '', content)
        
        return content

    def display_response(self, content: str, title: str = "🤖 Assistant", is_tool_response: bool = False):
        """Display a response with appropriate formatting."""
        if self.console:
            # Choose colors based on response type
            border_style = "green" if is_tool_response else "blue"
            
            # For tool responses, improve LaTeX display and try markdown rendering
            if is_tool_response:
                # Improve LaTeX expressions first
                content = self._improve_latex_display(content)
                
                try:
                    # Try to render as markdown for tool responses
                    markdown_content = Markdown(content)
                    panel = Panel(
                        markdown_content,
                        title=title,
                        border_style=border_style,
                        padding=(0, 1)
                    )
                except Exception:
                    # Fallback to plain text if markdown parsing fails
                    panel = Panel(
                        content,
                        title=title,
                        border_style=border_style,
                        padding=(0, 1)
                    )
            else:
                # For regular responses, use plain text to avoid issues with LaTeX/complex content
                panel = Panel(
                    content,
                    title=title,
                    border_style=border_style,
                    padding=(0, 1)
                )
            
            self.console.print(panel)
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
                self.logger.debug(f"Response length: {len(assistant_message)} characters")
                self.logger.debug(f"Response: {assistant_message[:500]}{'...' if len(assistant_message) > 500 else ''}")
            else:
                # Always print response length for debugging truncation issues
                print(f"[DEBUG] Response length: {len(assistant_message)} characters")
                print(f"[DEBUG] Last 100 chars: {assistant_message[-100:]}")  # Show the end

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
        """Handle streaming response with simple formatting."""
        full_content = ""
        try:
            if self.console:
                # Use simple live display
                with Live(console=self.console, refresh_per_second=10) as live:
                    current_text = Text("", style="default")
                    panel = Panel(
                        current_text,
                        title="🤖 Assistant (streaming)",
                        border_style="blue",
                        padding=(0, 1)
                    )
                    live.update(panel)

                    # Check if this is OpenAI client streaming or raw HTTP streaming
                    if hasattr(response, '__iter__') and hasattr(response, '__next__'):  # OpenAI client streaming
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                delta = chunk.choices[0].delta.content
                                full_content += delta
                                current_text.plain = full_content
                                panel = Panel(
                                    current_text,
                                    title="🤖 Assistant (streaming)",
                                    border_style="blue",
                                    padding=(0, 1)
                                )
                                live.update(panel)
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
                                            panel = Panel(
                                                current_text,
                                                title="🤖 Assistant (streaming)",
                                                border_style="blue",
                                                padding=(0, 1)
                                            )
                                            live.update(panel)
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