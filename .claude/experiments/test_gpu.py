"""Test GPU detection and CUDA availability."""

import torch


def main() -> None:
    """Check GPU availability and print system information."""
    print("=" * 60)
    print("GPU Detection Test")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA is not available. Please check your installation.")
        return
    
    # Get CUDA version
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Get GPU count and details
    gpu_count = torch.cuda.device_count()
    print(f"\nNumber of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi Processors: {props.multi_processor_count}")
    
    # Test tensor creation on each GPU
    print("\nTesting tensor creation on each GPU...")
    for i in range(gpu_count):
        try:
            device = f"cuda:{i}"
            x = torch.randn(1000, 1000, device=device)
            print(f"  GPU {i}: ✓ Successfully created tensor")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  GPU {i}: ✗ Error - {e}")
    
    # Test tensor parallelism simulation
    print("\nTesting multi-GPU tensor operations...")
    try:
        tensors = []
        for i in range(min(gpu_count, 2)):
            device = f"cuda:{i}"
            t = torch.randn(1000, 1000, device=device)
            tensors.append(t)
        print("  ✓ Successfully created tensors on multiple GPUs")
    except Exception as e:
        print(f"  ✗ Error with multi-GPU operations - {e}")
    
    print("\n" + "=" * 60)
    print("GPU detection test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
