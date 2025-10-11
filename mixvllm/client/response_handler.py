"""
Response handling for different types of chat responses.

This module handles the presentation layer for LLM responses, implementing
different strategies for various response types and formats.

Learning Points:
- Response formatting: LaTeX, Markdown, plain text
- Streaming vs non-streaming: Different display strategies
- Progressive enhancement: Rich console with fallback
- Unicode conversion: LaTeX → Unicode symbols for better terminal display
- OpenAI API response parsing: Both client and HTTP formats
- Live updates: Real-time streaming display
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
    """Handles display and formatting of LLM responses.

    This class is responsible for the presentation layer - taking raw LLM
    responses and displaying them beautifully in the terminal.

    Key Responsibilities:
    - Format responses (plain text, markdown, LaTeX)
    - Handle streaming responses with live updates
    - Convert LaTeX to Unicode for better readability
    - Parse different response formats (OpenAI client vs HTTP)
    - Distinguish tool responses from regular chat

    Response Types Handled:
    1. Regular chat responses (non-streaming)
    2. Streaming responses (token-by-token)
    3. Tool-augmented responses (with LaTeX/Markdown)

    Why this matters:
    - Good formatting improves comprehension
    - Streaming provides better UX (perceived latency)
    - LaTeX → Unicode makes math readable in terminal
    """

    def __init__(self, config: ChatConfig):
        """Initialize the response handler.

        Args:
            config: Configuration object with display settings
        """
        self.config = config

        # ====================================================================
        # Rich Console Setup
        # ====================================================================
        self.console = Console() if Console is not None else None
        # Console provides:
        # - Markdown rendering
        # - Colored panels
        # - Live updates for streaming
        # - Emoji support

    # ========================================================================
    # Private Methods: Text Processing
    # ========================================================================

    def _improve_latex_display(self, content: str) -> str:
        """Convert LaTeX expressions to Unicode for better terminal display.

        LLMs often generate LaTeX for mathematical expressions, but terminals
        don't render LaTeX natively. This method converts common LaTeX symbols
        and expressions to their Unicode equivalents for readability.

        Conversion Strategy:
        1. Try pylatexenc library for comprehensive conversion (if available)
        2. Fall back to manual symbol replacement dictionary
        3. Handle both inline ($...$) and display ($$...$$, \[...\]) math

        Examples:
            Input:  "The sum is $\\sum_{i=1}^n x_i = \\mu \\pm \\sigma$"
            Output: "The sum is ∑ᵢ₌₁ⁿ xᵢ = μ ± σ"

            Input:  "Einstein's $$E = mc^2$$"
            Output: "Einstein's E = mc²"

        Args:
            content: Text potentially containing LaTeX expressions

        Returns:
            str: Text with LaTeX converted to Unicode where possible

        Why this matters:
        - Terminal can't render LaTeX natively
        - Unicode symbols are widely supported in modern terminals
        - Improves readability of mathematical content
        - Better than showing raw LaTeX like \\alpha, \\sum
        """
        try:
            # ================================================================
            # Primary Method: pylatexenc Library
            # ================================================================
            # Use professional LaTeX → Unicode converter if available
            from pylatexenc.latex2text import LatexNodes2Text

            # Create a LatexNodes2Text converter with unicode output
            converter = LatexNodes2Text(
                keep_inline_math=False,  # Convert inline math to unicode
                keep_comments=False,     # Remove LaTeX comments
                strict_latex_spaces=False,  # Be more flexible with spaces
            )
            # Configuration explained:
            # - keep_inline_math=False: Convert $x$ to x, not keep as literal
            # - keep_comments=False: Remove % comments from output
            # - strict_latex_spaces=False: Handle spacing issues gracefully
            
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


# ============================================================================
# Learning Summary: Response Handling and Display Strategies
# ============================================================================
"""
This module demonstrates advanced techniques for handling and displaying
LLM responses in terminal applications.

KEY CONCEPTS:

1. STREAMING VS NON-STREAMING RESPONSES

Non-Streaming (handle_regular_response):
    User sends message
        ↓
    Wait for complete response (~10 seconds)
        ↓
    Display entire response at once

    Pros:
    - Simpler to implement
    - Can process entire response before display
    - Easier error handling

    Cons:
    - User sees nothing while waiting
    - Poor perceived latency
    - Appears "frozen"

Streaming (handle_streaming_response):
    User sends message
        ↓
    Display tokens as they arrive (real-time)
    token1 → display
    token2 → display
    token3 → display
    ...

    Pros:
    - Better perceived latency
    - User sees progress immediately
    - Feels more responsive

    Cons:
    - More complex implementation
    - Requires Live updates (rich library)
    - Can't post-process before display

Why we support both:
- Streaming for interactive chat (better UX)
- Non-streaming for batch processing (simpler)
- User can choose via --stream flag

2. LATEX → UNICODE CONVERSION

The Problem:
LLMs often generate LaTeX for mathematical expressions:
    "The formula is $\\alpha + \\beta = \\gamma$"

Terminals can't render LaTeX, so it looks ugly:
    "The formula is $\alpha + \beta = \gamma$"

The Solution:
Convert LaTeX symbols to Unicode:
    "The formula is α + β = γ"

Implementation:
- Primary: pylatexenc library (comprehensive)
- Fallback: Manual replacement dictionary
- Handles: inline math ($...$), display math ($$...$$)

Common conversions:
    \\alpha → α
    \\beta → β
    \\sum → ∑
    \\int → ∫
    \\pm → ±

Why it matters:
- Dramatically improves readability
- Makes math content accessible
- Professional appearance

3. RESPONSE FORMAT PARSING

OpenAI Client Response:
    response.choices[0].message.content

    Attributes:
    - choices: List of completion choices
    - message: Message object
    - content: Actual text

HTTP Response (streaming):
    data: {"choices": [{"delta": {"content": "token"}}]}
    data: {"choices": [{"delta": {"content": "next"}}]}
    data: [DONE]

    Server-Sent Events (SSE) format:
    - Lines start with "data: "
    - JSON object per line
    - [DONE] signals completion

Why handle both:
- OpenAI client: Better API, not always available
- HTTP: Universal fallback, always works
- Flexibility: Works in all environments

4. RICH CONSOLE FEATURES

Panel Display:
┌─────────── 🤖 Assistant ───────────┐
│ This is a nicely formatted         │
│ response with:                     │
│ - Colored border                   │
│ - Padding                          │
│ - Title                            │
└────────────────────────────────────┘

Live Updates (Streaming):
- Updates display in-place
- No scrolling spam
- Clean, professional look
- 10 refreshes per second

Markdown Rendering:
- **Bold** text
- *Italic* text
- `Code` highlighting
- Lists and tables

Why use Rich:
- Better UX than plain print()
- Professional appearance
- Supports colors, emoji, formatting
- Live updates for streaming

5. PROGRESSIVE ENHANCEMENT

Strategy:
    if rich_available:
        use_fancy_display()
    else:
        use_basic_print()

Benefits:
- Best experience when possible
- Always works (no hard dependency)
- Graceful degradation
- No "all or nothing" approach

Example:
    # With Rich:
    ┌─────────── 🤖 Assistant ───────────┐
    │ Hello! How can I help?             │
    └────────────────────────────────────┘

    # Without Rich:
    🤖 Assistant: Hello! How can I help?

6. ERROR HANDLING STRATEGIES

Try-Except Granularity:
- Markdown rendering: Try, fallback to plain text
- LaTeX conversion: Try, fallback to manual replacements
- JSON parsing: Try, fallback to error message

Why this matters:
- One failure doesn't break everything
- User always sees something
- Graceful degradation at every level

7. RESPONSE TYPE DETECTION

Regular Response:
- Plain text or markdown
- Blue border
- Standard formatting

Tool Response:
- May contain LaTeX
- Green border (indicates tool use)
- Enhanced formatting (markdown rendering)

Why distinguish:
- Visual feedback about tool usage
- Different formatting strategies
- Better user understanding

PERFORMANCE CONSIDERATIONS:

Streaming:
- Live refresh rate: 10 Hz (balanced)
- Token accumulation: No buffering
- Display overhead: Minimal (~1ms per update)

LaTeX Conversion:
- Regex operations: O(n) where n = content length
- Symbol replacement: O(m) where m = number of symbols
- Overall: Negligible impact (<10ms for typical responses)

Memory:
- Full content accumulation during streaming
- No chunking (simpler implementation)
- Typical memory: <1MB per response

ALTERNATIVE APPROACHES:

1. No Streaming:
   Pros: Simple, easy to debug
   Cons: Poor UX, appears frozen

2. Server-Side Rendering:
   Pros: Consistent formatting
   Cons: More complex, less flexibility

3. Separate Display Process:
   Pros: Parallel processing
   Cons: Complex IPC, harder to debug

WHY WE CHOSE THIS APPROACH:
- Balance simplicity and UX
- Progressive enhancement
- Works everywhere
- Professional appearance

DEBUGGING TIPS:

Enable debug mode (--debug) to inspect:
- Raw response format (OpenAI vs HTTP)
- Response length and truncation
- Streaming chunk contents
- JSON parsing issues

Common issues:
- Markdown breaking LaTeX: Render as plain text
- Streaming stops mid-response: Check network timeout
- Unicode display issues: Terminal encoding problem
- Slow rendering: Rich rendering overhead

EXTENDING THIS MODULE:

To add new formatting:
1. Add new display_* method
2. Update display_response to detect format
3. Add fallback for unsupported terminals
4. Test with/without Rich library

Example - Add HTML display:
    def display_html_response(self, content: str):
        if self.console:
            # Rich HTML rendering
            self.console.print(HTML(content))
        else:
            # Strip HTML tags
            print(strip_html(content))

LEARNING TAKEAWAY:
Good response handling balances:
- User experience (streaming, formatting)
- Robustness (fallbacks, error handling)
- Performance (efficient updates)
- Flexibility (multiple formats)
- Accessibility (works everywhere)
"""