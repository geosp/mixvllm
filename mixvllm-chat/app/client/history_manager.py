"""
Conversation history management.
"""

from typing import List, Dict
from rich.console import Console
from rich.table import Table

from .config import ChatConfig


class HistoryManager:
    """Manages conversation history."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.console = Console() if Console is not None else None
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

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

    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()