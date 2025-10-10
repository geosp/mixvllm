"""Utility functions for vLLM model serving.

This module provides helper functions for configuration management,
YAML loading, and system information gathering.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

from .config import ServeConfig


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration data

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the YAML is empty or invalid
        yaml.YAMLError: If there's a YAML parsing error
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {yaml_path}: {e}")

    if data is None:
        raise ValueError(f"Empty or invalid YAML file: {yaml_path}")

    return data


def load_serve_config_from_yaml(yaml_path: str) -> ServeConfig:
    """Load ServeConfig from YAML file.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        ServeConfig instance
    """
    data = load_yaml_config(yaml_path)
    return ServeConfig(**data)


def merge_configs(base_config: ServeConfig, cli_args: Dict[str, Any]) -> ServeConfig:
    """Merge CLI arguments into base configuration.

    Args:
        base_config: Base ServeConfig to merge into
        cli_args: Dictionary of CLI argument values

    Returns:
        New ServeConfig with merged values
    """
    # Mapping from CLI argument names to config paths
    arg_mapping = {
        'model': 'model.name',
        'trust_remote_code': 'model.trust_remote_code',
        'gpus': 'inference.tensor_parallel_size',
        'gpu_memory': 'inference.gpu_memory_utilization',
        'max_model_len': 'inference.max_model_len',
        'dtype': 'inference.dtype',
        'quantization': 'inference.quantization',
        'host': 'server.host',
        'port': 'server.port',
        'temperature': 'generation_defaults.temperature',
        'max_tokens': 'generation_defaults.max_tokens',
        'top_p': 'generation_defaults.top_p',
        'top_k': 'generation_defaults.top_k',
        'presence_penalty': 'generation_defaults.presence_penalty',
        'frequency_penalty': 'generation_defaults.frequency_penalty',
    }

    # Start with base config as dict
    config_dict = base_config.model_dump()

    # Apply CLI overrides
    for arg_name, value in cli_args.items():
        if value is not None and arg_name in arg_mapping:
            path = arg_mapping[arg_name]
            keys = path.split('.')
            current = config_dict
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value

    # Create new config
    return ServeConfig(**config_dict)


def merge_serve_config_with_cli(base_config: ServeConfig, cli_args: Dict[str, Any]) -> ServeConfig:
    """Merge CLI arguments into ServeConfig.

    Args:
        base_config: Base ServeConfig
        cli_args: CLI arguments dictionary

    Returns:
        Merged ServeConfig
    """
    return merge_configs(base_config, cli_args)


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPUs.

    Returns:
        Dictionary with GPU information
    """
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        gpus = []

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                'id': i,
                'name': props.name,
                'total_memory_gb': round(props.total_memory / 1024**3, 1),
            })

        return {
            'gpu_count': gpu_count,
            'gpus': gpus,
            'cuda_available': torch.cuda.is_available(),
        }
    except ImportError:
        return {
            'gpu_count': 0,
            'gpus': [],
            'cuda_available': False,
        }