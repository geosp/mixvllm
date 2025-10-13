"""Web terminal server using terminado and tornado.

This module provides a web-based terminal interface that allows users to
access a shell and run CLI tools (like mixvllm-chat) from their browser.

Architecture:
- Tornado web server handling HTTP and WebSocket connections
- Terminado for PTY (pseudo-terminal) management
- xterm.js frontend (loaded via CDN in the HTML)
- Runs on separate port from the model server

Security Note:
This provides full shell access with the same permissions as the server process.
Only enable in trusted environments. Consider adding authentication for production.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import tornado.web
import tornado.ioloop
from terminado import TermSocket, UniqueTermManager

from .terminal_config import TerminalConfig

logger = logging.getLogger(__name__)


class TerminalPageHandler(tornado.web.RequestHandler):
    """Serves the HTML page with xterm.js terminal interface."""

    def initialize(self, config: TerminalConfig, model_server_url: str):
        """Initialize handler with configuration.

        Args:
            config: Terminal server configuration
            model_server_url: URL of the model server for display/connection
        """
        self.config = config
        self.model_server_url = model_server_url

    def get(self):
        """Render the terminal page."""
        # Load the HTML template
        static_dir = Path(__file__).parent / "static"
        template_path = static_dir / "terminal.html"

        if not template_path.exists():
            self.set_status(500)
            self.write("Terminal template not found. Please check installation.")
            return

        # Read and serve the template
        # In a production system, you'd use proper template rendering
        html_content = template_path.read_text()

        # Simple template variable replacement
        html_content = html_content.replace("{{MODEL_SERVER_URL}}", self.model_server_url)
        html_content = html_content.replace(
            "{{AUTO_START_CHAT}}",
            "true" if self.config.auto_start_chat else "false"
        )

        self.set_header("Content-Type", "text/html")
        self.write(html_content)


def create_terminal_app(
    config: TerminalConfig,
    model_server_url: str,
    project_root: Optional[str] = None
) -> tornado.web.Application:
    """Create the Tornado application for the terminal server.

    Args:
        config: Terminal server configuration
        model_server_url: URL of the model server (e.g., "http://localhost:8000")
        project_root: Root directory of the project (for running convenience scripts)

    Returns:
        Configured Tornado application
    """
    # Determine working directory for terminal (user's home directory)
    if project_root is None:
        # Use the user's home directory instead of project root
        import os
        terminal_cwd = os.path.expanduser("~")
    else:
        terminal_cwd = project_root

    # Terminado needs shell_command as a list, not a string
    # We'll use bash and set the working directory via cwd
    shell_command = ["bash"]

    # Set up extra environment variables for the terminal
    extra_env = {
        "TERM": "xterm-256color",  # Proper terminal type for colors
        "CLICOLOR": "1",  # Enable colors in CLI tools
    }

    # Create a custom term manager that sets the working directory
    class CustomTermManager(UniqueTermManager):
        def new_terminal(self, **kwargs):
            kwargs['cwd'] = terminal_cwd
            return super().new_terminal(**kwargs)

    # Create terminal manager
    term_manager = CustomTermManager(
        shell_command=shell_command,
        max_terminals=10,  # Allow up to 10 concurrent terminals
        extra_env=extra_env,
    )

    # If auto-start is enabled, we'll inject the command via the HTML/JS
    # (terminado doesn't support initial commands directly, so we use xterm.js)

    # Define routes
    handlers = [
        (r"/", TerminalPageHandler, {
            "config": config,
            "model_server_url": model_server_url
        }),
        (r"/terminal", TermSocket, {"term_manager": term_manager}),
    ]

    # Create application
    app = tornado.web.Application(
        handlers,
        websocket_ping_interval=10,  # Keep websocket alive
        websocket_ping_timeout=30,
    )

    return app


def start_terminal_server(
    config: TerminalConfig,
    model_server_url: str,
    project_root: Optional[str] = None
) -> None:
    """Start the terminal server (blocking call).

    This function is designed to be run in a separate thread/process.

    Args:
        config: Terminal server configuration
        model_server_url: URL of the model server
        project_root: Root directory of the project
    """
    try:
        # Create application
        app = create_terminal_app(config, model_server_url, project_root)

        # Start listening
        app.listen(config.port, address=config.host)

        logger.info(
            f"🖥️  Web terminal started at http://{config.host}:{config.port}"
        )
        print(f"🖥️  Web terminal: http://localhost:{config.port}")

        if config.auto_start_chat:
            print(f"    Auto-starting: ./mixvllm-chat --base-url {model_server_url}")

        # Start the event loop (blocking)
        tornado.ioloop.IOLoop.current().start()

    except Exception as e:
        logger.error(f"Failed to start terminal server: {e}")
        raise
    finally:
        logger.info("Terminal server stopped")


def stop_terminal_server() -> None:
    """Stop the terminal server gracefully."""
    ioloop = tornado.ioloop.IOLoop.current()
    ioloop.add_callback(ioloop.stop)
