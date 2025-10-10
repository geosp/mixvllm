"""Test vLLM installation and basic functionality."""

import sys


def test_imports() -> bool:
    """Test if vLLM can be imported."""
    print("Testing vLLM imports...")
    try:
        import vllm
        print(f"  ✓ vLLM version: {vllm.__version__}")
        
        from vllm import LLM
        print("  ✓ LLM class imported successfully")
        
        import ray
        print(f"  ✓ Ray version: {ray.__version__}")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_gpu_detection() -> bool:
    """Test GPU detection within vLLM."""
    print("\nTesting GPU detection...")
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"  ✓ Detected {gpu_count} GPU(s)")
        
        if gpu_count < 2:
            print("  ⚠ Warning: Less than 2 GPUs detected")
            print("    Tensor parallelism requires multiple GPUs")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("vLLM Installation Test")
    print("=" * 60)
    print()
    
    tests_passed = 0
    tests_total = 2
    
    if test_imports():
        tests_passed += 1
    
    if test_gpu_detection():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✓ All tests passed! vLLM is ready to use.")
        print("\nNext steps:")
        print("  1. Try loading a small model (see .claude/experiments/)")
        print("  2. Test tensor parallelism with a larger model")
        print("  3. Check configs/example_model.yaml for configuration options")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
