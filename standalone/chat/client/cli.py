"""
CLI interface for the chat client.

This module provides the command-line interface for interacting with vLLM
servers, implementing a REPL (Read-Eval-Print Loop) for chat conversations.

Learning Points:
- REPL Pattern: Interactive command-line interface
- Argument Parsing: Using argparse for CLI arguments
- Enhanced Input: prompt_toolkit for better UX (history, styling)
- Command Pattern: Slash commands for special operations
- Error Handling: Graceful handling of interrupts and errors
- Auto-detection: Smart defaults for user convenience
"""

import argparse
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from .config import ChatConfig
from .chat_client import ChatClient


def create_prompt_session():
    """Create an enhanced prompt session with history and styling.

    Uses prompt_toolkit to provide a better terminal input experience than
    the built-in input() function.

    Enhanced Features:
    - Command history: Use ↑/↓ arrows to recall previous messages
    - In-memory history: Persists during session (not across restarts)
    - Custom styling: Colored prompt for better visual distinction
    - Multi-line support: Can span multiple lines if needed
    - Auto-completion: Framework supports it (not used here)

    Returns:
        PromptSession: Configured prompt session, or None if unavailable

    Why prompt_toolkit over input():
    - Better UX: History navigation, editing
    - More features: Syntax highlighting, auto-completion
    - Professional: Similar to ipython, ptpython
    - Cross-platform: Works on Windows, Linux, macOS

    Fallback Strategy:
    If prompt_toolkit unavailable or fails to import, returns None.
    Caller should fall back to basic input().
    """
    if PromptSession is not None:
        try:
            # ================================================================
            # Define Custom Styling
            # ================================================================
            # Style the prompt text to make it visually distinct
            style = Style.from_dict({
                'prompt': 'bold cyan',  # "You: " appears in bold cyan
            })
            # Why styling matters:
            # - Visual separation between input and output
            # - Professional appearance
            # - User knows where to type

            # ================================================================
            # Create Prompt Session
            # ================================================================
            return PromptSession(
                history=InMemoryHistory(),  # Enable command history (↑/↓)
                style=style,                 # Apply custom styling
                message="You: "              # Prompt prefix
            )
            # InMemoryHistory:
            # - Stores commands during session
            # - Accessible via up/down arrows
            # - Lost when program exits
            # Alternative: FileHistory() for persistent history

        except ImportError:
            # prompt_toolkit might be partially installed
            # or have missing dependencies
            pass
    return None  # Fallback to basic input()


def main():
    """Main CLI function implementing the REPL (Read-Eval-Print Loop) pattern.

    This function orchestrates the entire CLI application lifecycle:
    1. Parse command-line arguments
    2. Initialize chat client
    3. Auto-detect model if needed
    4. Run interactive chat loop
    5. Handle commands and errors gracefully

    REPL Pattern:
    - Read: Get user input from terminal
    - Eval: Process input (chat or command)
    - Print: Display LLM response
    - Loop: Repeat until user quits

    Exit Codes:
        0: Normal exit (user quit)
        1: Error (connection failed, no models, etc.)
    """
    # ========================================================================
    # STEP 1: Argument Parser Setup
    # ========================================================================
    # Create argument parser with automatic default value display
    parser = argparse.ArgumentParser(
        description="Enhanced CLI chat client for vLLM server with MCP tool support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        # ArgumentDefaultsHelpFormatter automatically shows default values in --help
    )

    # ------------------------------------------------------------------------
    # Server Connection Arguments
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000',
        help='vLLM server base URL'
    )
    # Why default to localhost:8000:
    # - Standard vLLM server port
    # - Most common development setup
    # - Easy to override for remote servers

    parser.add_argument(
        '--model',
        help='Model name (if not specified, will try to detect from server)'
    )
    # Optional: If not provided, auto-detection queries /v1/models
    # Smart default: User doesn't need to know model name

    # ------------------------------------------------------------------------
    # MCP (Model Context Protocol) Arguments
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--enable-mcp',
        action='store_true',  # Boolean flag: presence = True
        help='Enable MCP (Model Context Protocol) tools'
    )
    # action='store_true': No value needed, just --enable-mcp

    parser.add_argument(
        '--mcp-config',
        help='Path to MCP servers configuration file'
    )
    # Required if --enable-mcp is set
    # Points to YAML file with MCP server definitions

    # ------------------------------------------------------------------------
    # Generation Parameters
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--temperature',
        type=float,  # Validates input is a decimal number
        default=0.7,
        help='Sampling temperature'
    )
    # Default 0.7: Balanced creativity and coherence
    # User can override: --temperature 0.2 for factual, 1.5 for creative

    parser.add_argument(
        '--max-tokens',
        type=int,  # Validates input is an integer
        default=4096,
        help='Maximum tokens to generate'
    )
    # Default 4096: Reasonable for most responses
    # ~3000 words of output

    # ------------------------------------------------------------------------
    # Feature Flags
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--stream',
        action='store_true',  # Boolean flag
        help='Enable streaming responses'
    )
    # Streaming: Tokens appear as generated
    # Non-streaming: Wait for complete response
    # Streaming has better perceived latency

    parser.add_argument(
        '--debug',
        action='store_true',  # Boolean flag
        help='Enable debug logging of LLM prompts and responses'
    )
    # Debug mode: Logs to llm_debug.log
    # Essential for troubleshooting prompt issues

    # ========================================================================
    # STEP 2: Parse Arguments
    # ========================================================================
    args = parser.parse_args()
    # Parses sys.argv and returns Namespace object
    # Invalid arguments trigger automatic help message and exit

    # ========================================================================
    # STEP 3: Create Configuration
    # ========================================================================
    # Use Factory Method pattern to construct config from args
    config = ChatConfig.from_args(args)
    # from_args() extracts values from argparse Namespace

    try:
        # ====================================================================
        # STEP 4: Initialize Chat Client
        # ====================================================================
        # ChatClient handles all subsystem initialization
        # Will test connection and raise ConnectionError if server unreachable
        client = ChatClient(config)
        # Fail-fast: Errors here prevent entering chat loop

        # ====================================================================
        # STEP 5: Auto-detect Model (Smart Default)
        # ====================================================================
        # If user didn't specify --model, query server for available models
        if not client.config.model:
            # Query /v1/models endpoint
            models = client.get_available_models()

            if models:
                # Success: Use first model returned
                # vLLM typically returns only one model
                client.set_model(models[0])

                # Inform user of auto-selection
                if client.ui_manager.console:
                    client.ui_manager.show_success(f"Auto-selected model: {client.config.model}")
                else:
                    print(f"✓ Auto-selected model: {client.config.model}")

                # Why auto-detect:
                # - Better UX: User doesn't need to know model name
                # - Less typing: One less argument to specify
                # - Flexible: Works with any model server loads
            else:
                # Failure: No models available
                # This is a fatal error - can't chat without a model
                if client.ui_manager.console:
                    client.ui_manager.show_error("No models available on server")
                else:
                    print("❌ No models available on server")
                sys.exit(1)  # Exit with error code

        # ====================================================================
        # STEP 6: Prepare Welcome Message
        # ====================================================================
        # Count MCP tools for informational display
        tools_count = len(client.tool_manager.mcp_tools) if hasattr(client.tool_manager, 'mcp_tools') else 0
        # hasattr check: Safe access in case tool_manager doesn't have mcp_tools

        # Display welcome screen with configuration summary
        client.ui_manager.show_welcome(client.config.model, tools_count)
        # Shows: model name, server URL, MCP status, commands available

        # ====================================================================
        # STEP 7: Setup Enhanced Input
        # ====================================================================
        # Create prompt session for better terminal experience
        session = create_prompt_session()
        # Returns PromptSession or None (fallback to input())

        # ====================================================================
        # STEP 8: REPL Loop (Read-Eval-Print Loop)
        # ====================================================================
        # This is the heart of the CLI - infinite loop for user interaction
        try:
            while True:
                # ============================================================
                # Inner try-except: Handle per-message errors and interrupts
                # ============================================================
                try:
                    # ========================================================
                    # READ: Get user input
                    # ========================================================
                    if session:
                        # Enhanced input with history and styling
                        # Supports: ↑/↓ for history, Ctrl+R for search
                        user_input = session.prompt().strip()
                    else:
                        # Fallback: Basic input() function
                        # No history, no styling, but always works
                        user_input = input("You: ").strip()

                    # ========================================================
                    # Validate input
                    # ========================================================
                    if not user_input:
                        # Empty input: Skip to next iteration
                        # User pressed Enter without typing
                        continue

                    # ========================================================
                    # EVAL: Process input (Command Pattern)
                    # ========================================================
                    # Check if input is a command (starts with /)
                    if user_input.startswith('/'):
                        # Extract command name (remove / and lowercase)
                        cmd = user_input[1:].lower()

                        # Command dispatch
                        if cmd in ['quit', 'exit', 'q']:
                            # Exit command: Break out of REPL loop
                            if client.ui_manager.console:
                                client.ui_manager.console.print("[yellow]👋 Goodbye![/yellow]")
                            else:
                                print("👋 Goodbye!")
                            break  # Exit while loop → program ends

                        elif cmd == 'help':
                            # Help command: Show available commands
                            client.ui_manager.show_help()
                            continue  # Skip to next iteration

                        elif cmd == 'clear':
                            # Clear command: Reset conversation history
                            client.clear_history()
                            # Useful when starting new topic or context full
                            continue

                        elif cmd == 'history':
                            # History command: Display conversation so far
                            client.show_history()
                            # Helps user review what was discussed
                            continue

                        elif cmd == 'mcp' and client.config.enable_mcp:
                            # MCP status command: Show tool integration status
                            # Only available if MCP is enabled
                            client.show_mcp_status()
                            continue

                        else:
                            # Unknown command: Show error
                            client.ui_manager.show_error(f"Unknown command: {user_input}")
                            continue

                    # ========================================================
                    # Not a command: Send to LLM
                    # ========================================================
                    # This is the main chat path
                    response = client.chat(user_input)
                    # PRINT: Response displayed by response_handler
                    # (happens inside client.chat())

                # ============================================================
                # Exception Handling: Graceful error recovery
                # ============================================================
                except KeyboardInterrupt:
                    # User pressed Ctrl+C
                    # Don't crash - exit gracefully
                    if client.ui_manager.console:
                        client.ui_manager.console.print("\n[yellow]👋 Goodbye![/yellow]")
                    else:
                        print("\n👋 Goodbye!")
                    break  # Exit REPL loop

                except EOFError:
                    # User pressed Ctrl+D (Unix) or Ctrl+Z (Windows)
                    # Signals end of input stream
                    # Exit silently
                    break

        # ====================================================================
        # Outer Exception Handler: Fatal errors during REPL
        # ====================================================================
        except Exception as e:
            # Catch any unexpected errors during the REPL loop
            # These are programming errors or unexpected server issues
            client.ui_manager.show_error(f"Error: {e}")
            sys.exit(1)  # Exit with error code

    # ========================================================================
    # Initialization Exception Handler: Errors before REPL starts
    # ========================================================================
    except Exception as e:
        # Catch errors during initialization:
        # - Connection failures
        # - Configuration errors
        # - Client setup issues
        print(f"❌ Error: {e}")
        sys.exit(1)  # Exit with error code


# ============================================================================
# Script Entry Point
# ============================================================================
if __name__ == "__main__":
    # Standard Python idiom for executable scripts
    # Only runs main() when script executed directly
    # Prevents main() from running if this module is imported
    main()


# ============================================================================
# Learning Summary: REPL Pattern and CLI Best Practices
# ============================================================================
"""
This module demonstrates the REPL (Read-Eval-Print Loop) pattern and
CLI application best practices.

REPL PATTERN:
The REPL is a classic interactive program structure used by:
- Python interpreter (>>> prompt)
- Shell/Bash ($ prompt)
- SQL clients (mysql> prompt)
- Database REPLs (psql, mongo)

Structure:
    while True:
        # READ: Get user input
        user_input = input()

        # EVAL: Process the input
        result = process(user_input)

        # PRINT: Display the result
        print(result)

        # LOOP: Repeat

Benefits:
- Interactive: Immediate feedback
- Exploratory: Try things quickly
- Forgiving: Errors don't crash the program
- Stateful: Maintains context across interactions

KEY DESIGN DECISIONS:

1. Command Pattern for Special Operations
   Why slash commands (/help, /quit):
   - Clear distinction from chat messages
   - Familiar pattern (IRC, Slack, Discord)
   - Easy to parse
   - Doesn't interfere with normal conversation

2. Auto-detection for Better UX
   Model auto-detection:
   - User doesn't need to know model name
   - Works with any model server loads
   - One less argument to remember
   - Fails gracefully if no models available

3. Graceful Error Handling
   Multiple exception handlers:
   - Initialization errors: Exit immediately
   - Per-message errors: Show error, continue loop
   - KeyboardInterrupt: Clean exit on Ctrl+C
   - EOFError: Clean exit on Ctrl+D

   Why this matters:
   - Better UX: Program doesn't crash unexpectedly
   - User control: Can always exit cleanly
   - Debugging: Errors are informative

4. Enhanced Input with Fallback
   prompt_toolkit with input() fallback:
   - Best experience when available
   - Always works (no hard dependency)
   - Progressive enhancement pattern

5. Nested Try-Except Structure
   Outer try-except: Initialization errors
   Inner while loop: REPL loop
   Per-iteration try-except: Message handling

   Benefits:
   - Different error handling strategies for different phases
   - REPL continues on message errors
   - Fatal errors still exit cleanly

ALTERNATIVE APPROACHES:

1. Command-Line Arguments Only:
   Pros: Scriptable, simple
   Cons: No interactive conversation, restart for each query

2. Web Interface:
   Pros: Rich UI, accessible remotely
   Cons: More complex, requires browser

3. API Client:
   Pros: Programmatic access
   Cons: Not interactive, requires coding

WHY WE CHOSE REPL:
- Interactive conversation (chatbot nature)
- Simple to implement and use
- Familiar pattern for developers
- Good for testing and debugging
- Can be scripted with pipes if needed

EXTENSIBILITY:

Adding new commands:
1. Add new elif branch in command dispatch
2. Implement handler method
3. Update help text
4. No other changes needed

Example:
    elif cmd == 'newcmd':
        client.handle_new_command()
        continue

TESTING TIPS:

Interactive testing:
- Normal chat: "Hello, how are you?"
- Commands: /help, /clear, /history
- Edge cases: Empty input, very long input
- Interrupts: Ctrl+C, Ctrl+D
- Errors: Invalid server, network issues

Automated testing:
- Pipe input: echo "Hello" | python -m mixvllm.client.cli
- File input: python -m mixvllm.client.cli < test_input.txt
- Pytest fixtures for mocking user input

PERFORMANCE CONSIDERATIONS:

- prompt_toolkit adds ~20ms startup time
- Session history stored in memory (not disk)
- No performance impact during chat
- HTTP session reuse for efficiency

USER EXPERIENCE DETAILS:

1. Progress indicators:
   - "Thinking..." spinner during generation
   - Streaming shows tokens as generated

2. Visual feedback:
   - ✓ for success, ❌ for errors
   - Color coding for different message types
   - Emoji for personality

3. Error messages:
   - Clear descriptions
   - Actionable suggestions
   - No technical jargon

4. Smart defaults:
   - Auto-detect model
   - Reasonable temperature (0.7)
   - Standard port (8000)

LEARNING TAKEAWAY:
Good CLI applications prioritize user experience through:
- Graceful error handling
- Intuitive commands
- Smart defaults
- Progressive enhancement
- Clear feedback
"""