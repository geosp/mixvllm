#!/usr/bin/env python3
"""
Standalone chat client script.

This script provides a minimal chat client that only imports the necessary
dependencies for chat functionality, avoiding heavy ML dependencies.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
import requests
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.layout import Layout
from rich.align import Align
import openai

# MCP imports
try:
    from utils.mcp_tools import get_available_mcp_tools, MCPTool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


@dataclass
class ChatConfig:
    """Configuration container for the chat client."""
    base_url: str = "http://localhost:8000"
    model: Optional[str] = None
    enable_mcp: bool = False
    mcp_config_path: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    debug: bool = False

    def __post_init__(self):
        self.base_url = self.base_url.rstrip('/')

    @classmethod
    def from_args(cls, args) -> 'ChatConfig':
        return cls(
            base_url=getattr(args, 'base_url', "http://localhost:8000"),
            model=getattr(args, 'model', None),
            enable_mcp=getattr(args, 'enable_mcp', False),
            debug=getattr(args, 'debug', False),
            mcp_config_path=getattr(args, 'mcp_config', None),
            temperature=getattr(args, 'temperature', 0.7),
            max_tokens=getattr(args, 'max_tokens', 4096),
            stream=getattr(args, 'stream', False)
        )


def create_prompt_session():
    """Create an enhanced prompt session with history and styling."""
    return PromptSession(
        history=InMemoryHistory(),
        style=Style.from_dict({
            'prompt': 'bold cyan',
        })
    )


def detect_model(base_url: str) -> Optional[str]:
    """Try to detect the model from the server."""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                return data['data'][0]['id']
    except:
        pass
    return None


class ToolManager:
    """Manages MCP tools for the standalone chat client."""
    
    def __init__(self, config: ChatConfig):
        self.config = config
        self.mcp_tools: Dict[str, MCPTool] = {}
        self.console = Console()
        
        if config.enable_mcp and MCP_AVAILABLE:
            self._setup_mcp_tools()
    
    def _setup_mcp_tools(self):
        """Set up MCP tools."""
        try:
            tools = get_available_mcp_tools(self.config.mcp_config_path)
            self.mcp_tools = {tool.name: tool for tool in tools}
            
            if tools:
                self.console.print(f"[green]✓[/green] MCP tools enabled ({len(tools)} tools available)")
                for tool in tools:
                    self.console.print(f"  [dim]- {tool.name}: {tool.description}[/dim]")
            else:
                self.console.print("[yellow]⚠[/yellow] No MCP tools found")
                
        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to setup MCP tools: {e}")
            self.config.enable_mcp = False
    
    def has_tools(self) -> bool:
        """Check if any tools are available."""
        return len(self.mcp_tools) > 0
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools in OpenAI function format."""
        tools = []
        for tool in self.mcp_tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        return tools
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        if tool_name not in self.mcp_tools:
            return f"Error: Tool '{tool_name}' not found"
        
        tool = self.mcp_tools[tool_name]
        return tool.execute(**arguments)


# Enhanced formatting functions
def format_user_message(message: str, console: Console) -> None:
    """Format and display user message with nice styling."""
    user_panel = Panel(
        f"[bold cyan]{message}[/bold cyan]",
        title="[bold blue]You[/bold blue]",
        title_align="left",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(user_panel)
    console.print()

def format_assistant_message(message: str, console: Console) -> None:
    """Format and display assistant message with markdown and panel."""
    # Use Rich's markdown for formatting
    md = Markdown(message)
    assistant_panel = Panel(
        md,
        title="[bold green]Assistant[/bold green]",
        title_align="left",
        border_style="green",
        padding=(1, 2)
    )
    console.print(assistant_panel)
    console.print()

def format_tool_execution(tool_name: str, args: dict, console: Console) -> None:
    """Format tool execution display."""
    args_str = json.dumps(args, indent=2)
    tool_panel = Panel(
        f"[bold yellow]Executing:[/bold yellow] {tool_name}\n[cyan]Arguments:[/cyan]\n{args_str}",
        title="[bold magenta]🔧 Tool Call[/bold magenta]",
        title_align="left",
        border_style="magenta",
        padding=(1, 2)
    )
    console.print(tool_panel)

def format_tool_result(tool_name: str, result: str, console: Console, show_raw_json: bool = False) -> None:
    """Format tool result display with structured output."""
    try:
        # Try to parse as JSON and format nicely
        result_data = json.loads(result)
        if isinstance(result_data, dict):
            # Handle weather data specifically
            if "current_conditions" in result_data and "hourly_forecast" in result_data:
                format_weather_result(result_data, console)
                return
            
            # Generic structured data
            table = Table(show_header=True, header_style="bold blue", show_lines=True)
            
            # Add a single row with the data
            for key, value in result_data.items():
                if isinstance(value, (dict, list)):
                    table.add_column(key.replace("_", " ").title(), style="cyan")
                else:
                    table.add_column(key.replace("_", " ").title(), style="green")
            
            row_data = []
            for key, value in result_data.items():
                if isinstance(value, dict):
                    row_data.append(json.dumps(value, indent=2))
                elif isinstance(value, list):
                    row_data.append(json.dumps(value, indent=2))
                else:
                    row_data.append(str(value))
            table.add_row(*row_data)
            
            result_panel = Panel(
                table,
                title=f"[bold magenta]📊 {tool_name} Result[/bold magenta]",
                title_align="left",
                border_style="magenta",
                padding=(1, 2)
            )
        else:
            # Simple value
            result_panel = Panel(
                f"[green]{result}[/green]",
                title=f"[bold magenta]📊 {tool_name} Result[/bold magenta]",
                title_align="left",
                border_style="magenta",
                padding=(1, 2)
            )
    except (json.JSONDecodeError, TypeError):
        # Not JSON, display as formatted text
        result_panel = Panel(
            f"[green]{result}[/green]",
            title=f"[bold magenta]📊 {tool_name} Result[/bold magenta]",
                title_align="left",
                border_style="magenta",
                padding=(1, 2)
        )
    
    if show_raw_json:
        console.print(result_panel)
        console.print()

def format_weather_result(weather_data: dict, console: Console) -> None:
    """Special formatting for weather data."""
    location = weather_data.get("location", "Unknown")
    country = weather_data.get("country", "")
    current = weather_data.get("current_conditions", {})
    
    # Current weather summary
    weather_icon = "☀️" if "clear" in current.get("weather", "").lower() else "☁️" if "cloud" in current.get("weather", "").lower() else "🌧️"
    
    summary = f"""
{weather_icon} **{location}** ({country})
📅 {current.get('time', 'Now')}

🌡️ **{current.get('temperature', {}).get('value', 'N/A')}°C** (feels like {current.get('feels_like', {}).get('value', 'N/A')}°C)
💧 Humidity: {current.get('humidity', {}).get('value', 'N/A')}%
💨 Wind: {current.get('wind', {}).get('speed', 'N/A')} km/h {current.get('wind', {}).get('direction', '')}
🌤️ {current.get('weather', 'Unknown')}
"""
    
    # Hourly forecast table
    hourly = weather_data.get("hourly_forecast", [])
    if hourly:
        table = Table(show_header=True, header_style="bold blue", show_lines=False)
        table.add_column("Time", style="cyan", justify="center")
        table.add_column("Temp", style="green", justify="center")
        table.add_column("Weather", style="yellow", justify="left")
        table.add_column("Precip", style="blue", justify="center")
        
        for hour in hourly[:6]:  # Show next 6 hours
            time_str = hour.get('time', '').split('T')[1][:5] if 'T' in hour.get('time', '') else hour.get('time', '')
            temp = f"{hour.get('temperature', {}).get('value', 'N/A')}°C"
            weather = hour.get('weather', 'Unknown')
            precip = f"{hour.get('precipitation_probability', {}).get('value', 0)}%"
            
            table.add_row(time_str, temp, weather, precip)
        
        weather_panel = Panel(
            Align.center(Markdown(summary.strip())), 
            title="[bold cyan]🌤️ Weather Report[/bold cyan]",
            title_align="center",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(weather_panel)
        
        forecast_panel = Panel(
            table,
            title="[bold blue]📅 Hourly Forecast[/bold blue]",
            title_align="left",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(forecast_panel)
    else:
        weather_panel = Panel(
            Markdown(summary.strip()),
            title="[bold cyan]🌤️ Current Weather[/bold cyan]",
            title_align="center",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(weather_panel)
    
    console.print()

def format_error(message: str, console: Console) -> None:
    """Format error messages."""
    error_panel = Panel(
        f"[bold red]{message}[/bold red]",
        title="[bold red]❌ Error[/bold red]",
        title_align="left",
        border_style="red",
        padding=(1, 2)
    )
    console.print(error_panel)
    console.print()


def chat_loop(config: ChatConfig):
    """Main chat loop."""
    console = Console()
    prompt_session = create_prompt_session()

    # Initialize OpenAI client
    client = openai.OpenAI(
        base_url=f"{config.base_url}/v1",
        api_key="dummy"  # Not used by vLLM
    )

    # Detect model if not specified
    if not config.model:
        config.model = detect_model(config.base_url)
        if config.model:
            console.print(f"[dim]Auto-detected model: {config.model}[/dim]")
        else:
            console.print("[yellow]Warning: Could not detect model from server[/yellow]")

    # Initialize ToolManager
    tool_manager = ToolManager(config)

    console.print("[bold green]Chat session started. Type 'quit' or 'exit' to end.[/bold green]")
    console.print(f"[dim]Connected to: {config.base_url}[/dim]")
    if config.model:
        console.print(f"[dim]Model: {config.model}[/dim]")
    console.print()

    messages = []

    while True:
        try:
            # Get user input
            user_input = prompt_session.prompt("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Display user message with nice formatting
            format_user_message(user_input, console)

            # Get response from server
            try:
                # Prepare API call parameters
                api_params = {
                    "model": config.model or "default",
                    "messages": messages,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "stream": config.stream
                }
                
                # Add tools if MCP is enabled and tools are available
                if config.enable_mcp and tool_manager.has_tools():
                    api_params["tools"] = tool_manager.get_available_tools()
                    api_params["tool_choice"] = "auto"

                response = client.chat.completions.create(**api_params)

                # Initialize full_response
                full_response = ""

                if config.stream:
                    # Streaming response with tool handling
                    tool_calls = []
                    full_response = ""  # Initialize full_response
                    
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                        
                        # Handle tool calls in streaming
                        if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            for tool_call_delta in chunk.choices[0].delta.tool_calls:
                                if tool_call_delta.index >= len(tool_calls):
                                    tool_calls.extend([{}] * (tool_call_delta.index - len(tool_calls) + 1))
                                
                                if tool_call_delta.id:
                                    tool_calls[tool_call_delta.index]['id'] = tool_call_delta.id
                                if tool_call_delta.function:
                                    if 'function' not in tool_calls[tool_call_delta.index]:
                                        tool_calls[tool_call_delta.index]['function'] = {'name': '', 'arguments': ''}
                                    if tool_call_delta.function.name:
                                        tool_calls[tool_call_delta.index]['function']['name'] += tool_call_delta.function.name
                                    if tool_call_delta.function.arguments:
                                        tool_calls[tool_call_delta.index]['function']['arguments'] += tool_call_delta.function.arguments
                    
                    # Format the assistant response properly
                    if full_response.strip():
                        format_assistant_message(full_response, console)
                else:
                    # Non-streaming response handling
                    assistant_message = response.choices[0].message
                    full_response = assistant_message.content or ""
                    
                    # Check for tool calls in non-streaming response
                    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                        # Handle tool calls
                        for tool_call in assistant_message.tool_calls:
                            tool_name = tool_call.function.name
                            try:
                                tool_args = json.loads(tool_call.function.arguments)
                                format_tool_execution(tool_name, tool_args, console)
                                tool_result = tool_manager.execute_tool(tool_name, tool_args)
                                
                                # Add tool call and result to messages
                                messages.append({
                                    "role": "assistant",
                                    "content": full_response,
                                    "tool_calls": [{
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments
                                        }
                                    }]
                                })
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result
                                })
                                
                                # Get follow-up response
                                followup_response = client.chat.completions.create(
                                    model=config.model or "default",
                                    messages=messages,
                                    temperature=config.temperature,
                                    max_tokens=config.max_tokens,
                                    stream=False
                                )
                                followup_content = followup_response.choices[0].message.content
                                format_assistant_message(followup_content, console)
                                full_response += followup_content
                                
                            except Exception as e:
                                format_error(f"Tool execution error: {e}", console)
                    else:
                        # Check if response looks like a tool call in JSON format
                        if config.enable_mcp and tool_manager.has_tools() and full_response.strip().startswith('{') and full_response.strip().endswith('}'):
                            try:
                                # Try to parse as JSON tool arguments
                                tool_args = json.loads(full_response.strip())
                                
                                # Find a tool that matches these arguments
                                executed_tool = False
                                for tool_name, tool in tool_manager.mcp_tools.items():
                                    # Check if the JSON keys match the tool's expected parameters
                                    if hasattr(tool, 'input_schema') and 'properties' in tool.input_schema:
                                        expected_params = set(tool.input_schema['properties'].keys())
                                        provided_params = set(tool_args.keys())
                                        if provided_params.issubset(expected_params) or len(provided_params.intersection(expected_params)) > 0:
                                            # Execute the tool
                                            format_tool_execution(tool_name, tool_args, console)
                                            tool_result = tool_manager.execute_tool(tool_name, tool_args)
                                            
                                            # Add the original response and tool call to messages
                                            messages.append({"role": "assistant", "content": full_response})
                                            messages.append({
                                                "role": "tool",
                                                "tool_call_id": f"manual_{tool_name}_{len(messages)}",
                                                "content": tool_result
                                            })
                                            
                                            # Get follow-up response
                                            followup_response = client.chat.completions.create(
                                                model=config.model or "default",
                                                messages=messages,
                                                temperature=config.temperature,
                                                max_tokens=config.max_tokens,
                                                stream=False
                                            )
                                            followup_content = followup_response.choices[0].message.content
                                            format_assistant_message(followup_content, console)
                                            full_response += followup_content
                                            executed_tool = True
                                            break
                                
                                if not executed_tool:
                                    # No matching tool found, just print the response
                                    format_assistant_message(full_response, console)
                            except (json.JSONDecodeError, Exception) as e:
                                # Not valid JSON or other error, just print the response
                                format_assistant_message(full_response, console)
                        else:
                            # Simple response without tools
                            format_assistant_message(full_response, console)
                
                # Add final assistant response to history (skip if we already added tool call responses)
                if not (hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls):
                    messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                format_error(f"Error communicating with server: {e}", console)
                continue

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            continue
        except EOFError:
            break

    console.print("[bold blue]Chat session ended.[/bold blue]")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Enhanced CLI chat client for vLLM server with MCP tool support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base-url',
        default="http://localhost:8000",
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
        default=4096,
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
        help='Enable debug logging'
    )

    args = parser.parse_args()
    config = ChatConfig.from_args(args)

    try:
        chat_loop(config)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()