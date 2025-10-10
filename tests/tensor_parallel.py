"""Test vLLM with configurable tensor parallelism.

This script tests vLLM inference with 1 or 2 GPUs to validate tensor parallelism
and measure performance characteristics.
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from vllm import LLM, SamplingParams


@dataclass
class TestConfig:
    """Configuration for tensor parallelism test."""

    gpus: int
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    gpu_memory: float
    gpu_id: Optional[int] = None


class GPUMonitor:
    """Monitor GPU memory usage."""

    @staticmethod
    def get_memory_usage() -> List[tuple[int, float, float]]:
        """Get memory usage for all GPUs.

        Returns:
            List of (gpu_id, used_gb, total_gb) tuples
        """
        gpu_count = torch.cuda.device_count()
        usage = []

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3

            # Get current memory usage
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            used_gb = max(allocated, reserved)

            usage.append((i, used_gb, total_gb))

        return usage

    @staticmethod
    def print_memory_usage(label: str) -> None:
        """Print current GPU memory usage with label."""
        print(f"\n{label}:")
        usage = GPUMonitor.get_memory_usage()

        for gpu_id, used_gb, total_gb in usage:
            percentage = (used_gb / total_gb) * 100
            free_gb = total_gb - used_gb
            print(f"  GPU {gpu_id}: {used_gb:.1f} GB used / {total_gb:.1f} GB total ({percentage:.1f}% used, {free_gb:.1f} GB free)")


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str) -> None:
    """Print a section divider."""
    print(f"\n{title}")


def check_memory_availability(config: TestConfig) -> None:
    """Check if sufficient GPU memory is available.

    Args:
        config: Test configuration

    Raises:
        ValueError: If insufficient memory is available
    """
    usage = GPUMonitor.get_memory_usage()
    
    if config.gpus == 1 and config.gpu_id is not None:
        # Check specific GPU
        gpu_id = config.gpu_id
        _, used_gb, total_gb = usage[gpu_id]
        free_gb = total_gb - used_gb
        required_gb = total_gb * config.gpu_memory
        
        if free_gb < required_gb:
            raise ValueError(
                f"Insufficient memory on GPU {gpu_id}. "
                f"Free: {free_gb:.1f} GB, Required: {required_gb:.1f} GB. "
                f"Try reducing --gpu-memory (current: {config.gpu_memory})"
            )
    elif config.gpus == 2:
        # Check both GPUs
        for gpu_id in [0, 1]:
            _, used_gb, total_gb = usage[gpu_id]
            free_gb = total_gb - used_gb
            # With tensor parallelism, each GPU needs roughly half
            required_gb = (total_gb * config.gpu_memory) / 2
            
            if free_gb < required_gb:
                print(f"  Warning: GPU {gpu_id} may have insufficient memory")
                print(f"    Free: {free_gb:.1f} GB, Estimated required: {required_gb:.1f} GB")


def load_model(config: TestConfig) -> LLM:
    """Load vLLM model with specified configuration.

    Args:
        config: Test configuration

    Returns:
        Loaded LLM instance
    """
    print_section("Loading model...")
    print(f"  Model: {config.model}")
    print(f"  Tensor Parallel Size: {config.gpus}")
    print(f"  GPU Memory Utilization: {config.gpu_memory:.2f}")
    if config.gpu_id is not None:
        print(f"  Target GPU ID: {config.gpu_id}")

    # Check memory availability before loading
    try:
        check_memory_availability(config)
    except ValueError as e:
        print(f"  ✗ Memory check failed: {e}")
        raise

    start_time = time.time()

    try:
        # Build LLM configuration
        llm_kwargs = {
            "model": config.model,
            "tensor_parallel_size": config.gpus,
            "gpu_memory_utilization": config.gpu_memory,
            "trust_remote_code": True,
            "max_model_len": 4096,
        }
        
        # For single GPU, optionally specify which GPU to use via CUDA_VISIBLE_DEVICES
        # However, vLLM manages this internally, so we just let it use the configuration
        
        llm = LLM(**llm_kwargs)

        load_time = time.time() - start_time
        print(f"  ✓ Model loaded in {load_time:.2f}s")

        return llm

    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        print("\n  Troubleshooting tips:")
        print("    - Reduce --gpu-memory (try 0.70 or 0.65)")
        print("    - Use a smaller model")
        print("    - Check GPU memory with: nvidia-smi")
        print("    - Stop other GPU processes")
        raise


def run_inference(llm: LLM, config: TestConfig) -> tuple[str, float, int]:
    """Run inference and measure performance.

    Args:
        llm: Loaded LLM instance
        config: Test configuration

    Returns:
        Tuple of (generated_text, inference_time, token_count)
    """
    print_section("Running inference...")
    print(f"  Prompt: {config.prompt[:50]}{'...' if len(config.prompt) > 50 else ''}")

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=0.9,
    )

    start_time = time.time()

    try:
        outputs = llm.generate([config.prompt], sampling_params)
        inference_time = time.time() - start_time

        generated_text = outputs[0].outputs[0].text
        token_count = len(outputs[0].outputs[0].token_ids)

        print(f"  ✓ Generated {token_count} tokens in {inference_time:.2f}s")
        print(f"  ✓ Speed: {token_count / inference_time:.1f} tokens/second")

        return generated_text, inference_time, token_count

    except Exception as e:
        print(f"  ✗ Error during inference: {e}")
        raise


def print_results(config: TestConfig, generated_text: str, inference_time: float, token_count: int) -> None:
    """Print inference results.

    Args:
        config: Test configuration
        generated_text: Generated output text
        inference_time: Time taken for inference
        token_count: Number of tokens generated
    """
    print_section("Results Summary")
    print(f"  Configuration: {config.gpus} GPU(s)")
    print(f"  GPU Memory Utilization: {config.gpu_memory:.2f}")
    print(f"  Inference Time: {inference_time:.2f}s")
    print(f"  Tokens Generated: {token_count}")
    print(f"  Throughput: {token_count / inference_time:.1f} tokens/second")

    print_section("Generated Output")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)


def validate_config(config: TestConfig) -> None:
    """Validate test configuration.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    available_gpus = torch.cuda.device_count()

    if config.gpus > available_gpus:
        raise ValueError(
            f"Requested {config.gpus} GPUs but only {available_gpus} available"
        )

    if config.gpus not in [1, 2]:
        raise ValueError(f"GPU count must be 1 or 2, got {config.gpus}")

    if config.gpu_id is not None:
        if config.gpu_id < 0 or config.gpu_id >= available_gpus:
            raise ValueError(
                f"Invalid GPU ID {config.gpu_id}. Must be between 0 and {available_gpus - 1}"
            )
        if config.gpus == 2:
            print(f"  Warning: --gpu-id is ignored when using tensor parallelism (--gpus 2)")

    if config.max_tokens < 1:
        raise ValueError(f"max_tokens must be positive, got {config.max_tokens}")

    if not 0.0 <= config.temperature <= 2.0:
        raise ValueError(f"temperature must be between 0 and 2, got {config.temperature}")

    if not 0.1 <= config.gpu_memory <= 1.0:
        raise ValueError(f"gpu_memory must be between 0.1 and 1.0, got {config.gpu_memory}")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Test vLLM with configurable tensor parallelism",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--gpus",
        type=int,
        required=True,
        choices=[1, 2],
        help="Number of GPUs to use (1 or 2)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model name or path",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short poem about AI",
        help="Prompt for text generation",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 to 2.0)",
    )

    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.80,
        help="GPU memory utilization (0.1 to 1.0). Lower if you get OOM errors.",
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Specific GPU ID to use for single GPU mode (0 or 1). Ignored for tensor parallelism.",
    )

    args = parser.parse_args()

    # Create configuration
    config = TestConfig(
        gpus=args.gpus,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_memory=args.gpu_memory,
        gpu_id=args.gpu_id,
    )

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    # Print header
    print_header("vLLM Tensor Parallelism Test")

    # Show configuration
    print_section("Configuration")
    print(f"  Model: {config.model}")
    print(f"  Tensor Parallel Size: {config.gpus}")
    print(f"  GPU Memory Utilization: {config.gpu_memory:.2f}")
    print(f"  Max Tokens: {config.max_tokens}")
    print(f"  Temperature: {config.temperature}")
    if config.gpu_id is not None and config.gpus == 1:
        print(f"  Target GPU ID: {config.gpu_id}")
    print(f"  Prompt: {config.prompt[:50]}{'...' if len(config.prompt) > 50 else ''}")

    # Show initial memory usage
    GPUMonitor.print_memory_usage("Initial GPU Memory")

    try:
        # Load model
        llm = load_model(config)

        # Show memory after loading
        GPUMonitor.print_memory_usage("GPU Memory After Model Load")

        # Run inference
        generated_text, inference_time, token_count = run_inference(llm, config)

        # Show memory during inference
        GPUMonitor.print_memory_usage("GPU Memory After Inference")

        # Print results
        print_results(config, generated_text, inference_time, token_count)

        # Print comparison tips
        if config.gpus == 1:
            print_section("Next Steps")
            print("  Run with --gpus 2 to compare tensor parallelism performance:")
            print(f"    uv run python tests/test_tensor_parallel.py --gpus 2 --gpu-memory {config.gpu_memory}")
        else:
            print_section("Next Steps")
            print("  Compare with single GPU performance:")
            print(f"    uv run python tests/test_tensor_parallel.py --gpus 1 --gpu-memory {config.gpu_memory}")

        print("\n" + "=" * 60)
        print("  Test Complete!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()