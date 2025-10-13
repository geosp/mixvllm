"""Terminal server configuration."""

from pydantic import BaseModel, Field


class TerminalConfig(BaseModel):
    """Configuration for the web terminal server."""

    enabled: bool = Field(
        default=False,
        description="Enable web terminal access"
    )
    host: str = Field(
        default="0.0.0.0",
        description="Terminal server host address"
    )
    port: int = Field(
        default=8888,
        ge=1,
        le=65535,
        description="Terminal server port number"
    )
    auto_start_chat: bool = Field(
        default=True,
        description="Automatically start mixvllm-chat on terminal connection"
    )