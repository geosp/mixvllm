"""Configuration models for vLLM model serving.

This module defines Pydantic models for validating and managing
configuration settings for the vLLM model server.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class ModelConfig(BaseModel):
    """Configuration for the model to be served."""

    name: str = Field(..., description="HuggingFace model name or path")
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code for custom models"
    )


class InferenceConfig(BaseModel):
    """Configuration for inference settings."""

    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of GPUs for tensor parallelism"
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization ratio"
    )
    max_model_len: int = Field(
        default=4096,
        gt=0,
        description="Maximum model context length"
    )
    dtype: str = Field(
        default="float16",
        description="Data type for model weights"
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization method (awq, gptq, or null)"
    )

    @field_validator('tensor_parallel_size')
    @classmethod
    def validate_tensor_parallel_size(cls, v: int) -> int:
        """Validate tensor parallel size against available GPUs."""
        available_gpus = torch.cuda.device_count()
        if v > available_gpus:
            raise ValueError(
                f"Requested {v} GPUs but only {available_gpus} available"
            )
        return v


class ServerConfig(BaseModel):
    """Configuration for the server settings."""

    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port number"
    )


class GenerationDefaults(BaseModel):
    """Default settings for text generation."""

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=512,
        gt=0,
        description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        description="Top-k sampling parameter"
    )
    presence_penalty: float = Field(
        default=0.0,
        description="Presence penalty"
    )
    frequency_penalty: float = Field(
        default=0.0,
        description="Frequency penalty"
    )


class ServeConfig(BaseModel):
    """Complete configuration for model serving."""

    model: ModelConfig
    inference: InferenceConfig
    server: ServerConfig
    generation_defaults: GenerationDefaults = Field(
        default_factory=GenerationDefaults
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ServeConfig':
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ServeConfig instance
        """
        from .utils import load_serve_config_from_yaml
        return load_serve_config_from_yaml(yaml_path)

    def merge_cli_args(self, **kwargs) -> 'ServeConfig':
        """Merge CLI arguments into configuration.

        Args:
            **kwargs: CLI argument values

        Returns:
            Updated ServeConfig instance
        """
        from .utils import merge_serve_config_with_cli
        return merge_serve_config_with_cli(self, kwargs)