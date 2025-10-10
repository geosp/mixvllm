#!/bin/bash

# vLLM Project Bootstrap Script
# Run this from your project root to set up the complete development environment
# Usage: ./create_project.sh [--dry-run] [--force]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
PROJECT_NAME="vllm-dev"

# Flags
DRY_RUN=false
FORCE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --force)
            FORCE=true
            ;;
        --help)
            echo "Usage: ./create_project.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be done without making changes"
            echo "  --force      Remove existing files and start fresh"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Running in DRY RUN mode - no changes will be made${NC}\n"
fi

if [ "$FORCE" = true ]; then
    echo -e "${YELLOW}FORCE mode enabled - will remove existing files${NC}\n"
fi

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

execute() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC} Would execute: $1"
    else
        eval "$1"
    fi
}

# Validate we're in the right place
validate_location() {
    log_info "Validating execution location..."
    
    # Check if create_project.sh exists in current directory
    if [ ! -f "create_project.sh" ]; then
        log_error "This script must be run from the project root directory"
        log_error "Expected to find 'create_project.sh' in current directory"
        log_error "Current directory: $(pwd)"
        exit 1
    fi
    
    log_success "Running from project root: $(pwd)"
    echo ""
}

# Clean up partial installations
cleanup_partial() {
    if [ "$FORCE" = true ]; then
        log_info "Cleaning up existing files (--force mode)..."
        
        execute "rm -rf .venv"
        execute "rm -f pyproject.toml"
        execute "rm -f uv.lock"
        execute "rm -f .python-version"
        
        log_success "Cleanup complete"
        echo ""
    elif [ -f "pyproject.toml" ] || [ -d ".venv" ]; then
        log_warning "Existing project files detected"
        log_warning "Use --force to remove and start fresh"
        echo ""
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. CUDA drivers may not be installed."
        exit 1
    fi
    log_success "CUDA drivers detected"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python ${PYTHON_VERSION} or higher."
        exit 1
    fi
    
    PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_success "Python ${PYTHON_VER} detected"
    
    # Check GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log_success "Detected ${GPU_COUNT} GPU(s)"
    
    if [ "$GPU_COUNT" -lt 2 ]; then
        log_warning "Less than 2 GPUs detected. Tensor parallelism requires multiple GPUs."
    fi
    
    echo ""
}

# Install uv
install_uv() {
    log_info "Checking for uv installation..."
    
    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version)
        log_success "uv already installed: ${UV_VERSION}"
    else
        log_info "Installing uv..."
        execute "curl -LsSf https://astral.sh/uv/install.sh | sh"
        
        # Add to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        if [ "$DRY_RUN" = false ]; then
            log_success "uv installed successfully"
        fi
    fi
    
    echo ""
}

# Create directory structure
create_structure() {
    log_info "Creating project directory structure..."
    
    # Create directories
    DIRS=(
        ".claude/experiments"
        ".claude/benchmarks"
        ".claude/scratch"
        "src/inference"
        "src/utils"
        "configs"
        "tests"
    )
    
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            execute "mkdir -p $dir"
            log_success "Created: $dir"
        else
            log_info "Already exists: $dir"
        fi
    done
    
    # Create __init__.py files
    INIT_FILES=(
        "src/__init__.py"
        "src/inference/__init__.py"
        "src/utils/__init__.py"
        "tests/__init__.py"
    )
    
    for init_file in "${INIT_FILES[@]}"; do
        if [ ! -f "$init_file" ] && [ "$DRY_RUN" = false ]; then
            touch "$init_file"
        fi
    done
    
    echo ""
}

# Create pyproject.toml
create_pyproject() {
    log_info "Creating pyproject.toml configuration..."
    
    if [ -f "pyproject.toml" ] && [ "$FORCE" = false ]; then
        log_warning "pyproject.toml already exists, skipping"
        echo ""
        return
    fi
    
    if [ "$DRY_RUN" = false ]; then
        cat > pyproject.toml << 'EOF'
[project]
name = "vllm-dev"
version = "0.1.0"
description = "vLLM development environment with dual-GPU tensor parallelism"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "vllm>=0.6.0",
    "ray>=2.9.0",
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "pyyaml>=6.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "black>=23.0.0",
    "ipython>=8.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
EOF
        log_success "pyproject.toml created with proper hatchling configuration"
    fi
    
    echo ""
}

# Initialize uv project
init_uv_project() {
    log_info "Initializing uv project..."
    
    if [ ! -f ".python-version" ]; then
        execute "uv python pin ${PYTHON_VERSION}"
        log_success "Python version pinned to ${PYTHON_VERSION}"
    else
        log_info "Python version already pinned"
    fi
    
    echo ""
}

# Create .gitignore
create_gitignore() {
    log_info "Creating .gitignore..."
    
    if [ -f ".gitignore" ] && [ "$FORCE" = false ]; then
        log_warning ".gitignore already exists, skipping"
        echo ""
        return
    fi
    
    if [ "$DRY_RUN" = false ]; then
        cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# uv
uv.lock
.python-version

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/

# Models (large files)
*.bin
*.safetensors
models/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Temporary files
.claude/scratch/*
!.claude/scratch/.gitkeep
EOF
        log_success ".gitignore created"
    fi
    
    echo ""
}

# Create README
create_readme() {
    log_info "Creating README.md..."
    
    if [ -f "README.md" ] && [ "$FORCE" = false ]; then
        log_warning "README.md already exists, skipping"
        echo ""
        return
    fi
    
    if [ "$DRY_RUN" = false ]; then
        cat > README.md << 'EOF'
# vLLM Development Environment

Development environment for vLLM with dual-GPU tensor parallelism support.

## Hardware

- **GPUs**: 2x NVIDIA GeForce RTX 3090 Ti (24GB VRAM each)
- **CUDA Version**: 12.8
- **Driver**: 570.172.08

## Setup

This project was initialized using `create_project.sh` at the root.

### Install Dependencies

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Or install without dev dependencies
uv sync
```

### Activate Environment

```bash
# Run commands with uv (recommended)
uv run python script.py

# Or activate the virtual environment
source .venv/bin/activate
```

## Project Structure

```
.
├── create_project.sh     # Project bootstrap script
├── .claude/              # Temporary and experimental code
│   ├── experiments/      # Model testing and prototyping
│   ├── benchmarks/       # Performance benchmarks
│   └── scratch/          # Quick tests and scratchpad
├── src/                  # Production code
│   ├── inference/        # vLLM inference wrappers
│   └── utils/            # Shared utilities
├── configs/              # Model and server configurations
└── tests/                # Test suite
```

## Quick Start

### 1. Test GPU Detection

```bash
uv run python .claude/experiments/test_gpu.py
```

### 2. Test vLLM Installation

```bash
uv run python .claude/experiments/test_vllm.py
```

### 3. Run a Model with Tensor Parallelism

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,  # Use both GPUs
    gpu_memory_utilization=0.90,
    trust_remote_code=True
)

outputs = llm.generate("Hello, my name is")
print(outputs[0].outputs[0].text)
```

## Configuration

See `configs/example_model.yaml` for a complete configuration template.

## Development

```bash
# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Auto-formatting
uv run black src/
```

## Tensor Parallelism Notes

With 2x RTX 3090 Ti (24GB each = 48GB total):
- Can run 70B models in FP16 (requires ~140GB, use quantization)
- Can run 70B models in 4-bit quantization comfortably
- Can run 34B models in FP16 easily
- Communication overhead between GPUs is minimal on PCIe 4.0

## Troubleshooting

**Out of Memory Errors:**
- Reduce `gpu_memory_utilization` (try 0.85 or 0.80)
- Use quantization (4-bit or 8-bit)
- Reduce `max_model_len`

**Slow Inference:**
- Check GPU utilization with `nvidia-smi`
- Verify both GPUs are being used
- Ensure PCIe link is running at full speed

EOF
        log_success "README.md created"
    fi
    
    echo ""
}

# Create example config
create_example_config() {
    log_info "Creating example configuration..."
    
    if [ "$DRY_RUN" = false ]; then
        cat > configs/example_model.yaml << 'EOF'
# Example vLLM model configuration for dual RTX 3090 Ti setup

model:
  name: "meta-llama/Llama-2-70b-hf"
  # For HuggingFace models, set HF_TOKEN environment variable if needed
  
inference:
  tensor_parallel_size: 2          # Use both GPUs
  max_model_len: 4096              # Maximum context length
  gpu_memory_utilization: 0.90     # Use 90% of available GPU memory
  trust_remote_code: true
  dtype: "float16"                 # or "bfloat16", "float32"
  quantization: null               # Options: "awq", "gptq", null
  
generation:
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  max_tokens: 512
  presence_penalty: 0.0
  frequency_penalty: 0.0
  
server:
  host: "0.0.0.0"
  port: 8000
  
# Example for smaller model (fits on single GPU)
# model:
#   name: "meta-llama/Llama-2-13b-hf"
# inference:
#   tensor_parallel_size: 1
EOF
        log_success "Example config created at configs/example_model.yaml"
    fi
    
    echo ""
}

# Create GPU test script
create_gpu_test() {
    log_info "Creating GPU detection test script..."
    
    if [ "$DRY_RUN" = false ]; then
        cat > .claude/experiments/test_gpu.py << 'EOF'
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
EOF
        log_success "GPU test script created at .claude/experiments/test_gpu.py"
    fi
    
    echo ""
}

# Create vLLM test script
create_vllm_test() {
    log_info "Creating vLLM basic test script..."
    
    if [ "$DRY_RUN" = false ]; then
        cat > .claude/experiments/test_vllm.py << 'EOF'
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
EOF
        log_success "vLLM test script created at .claude/experiments/test_vllm.py"
    fi
    
    echo ""
}

# Install dependencies
install_dependencies() {
    log_info "Installing project dependencies..."
    
    if [ "$DRY_RUN" = false ]; then
        log_info "This may take 5-10 minutes for vLLM, PyTorch, and dependencies..."
        uv sync --all-extras
        log_success "Dependencies installed"
    else
        echo -e "${YELLOW}[DRY RUN]${NC} Would execute: uv sync --all-extras"
    fi
    
    echo ""
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    if [ "$DRY_RUN" = false ]; then
        # Test GPU detection
        log_info "Running GPU detection test..."
        uv run python .claude/experiments/test_gpu.py
        
        echo ""
        log_info "Running vLLM installation test..."
        uv run python .claude/experiments/test_vllm.py
        
        log_success "Installation verification complete"
    fi
    
    echo ""
}

# Main execution
main() {
    echo -e "${GREEN}"
    echo "=========================================="
    echo "  vLLM Project Bootstrap"
    echo "=========================================="
    echo -e "${NC}\n"
    
    validate_location
    cleanup_partial
    check_prerequisites
    install_uv
    create_structure
    create_pyproject
    init_uv_project
    create_gitignore
    create_readme
    create_example_config
    create_gpu_test
    create_vllm_test
    install_dependencies
    verify_installation
    
    echo -e "${GREEN}"
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo -e "${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Test GPU: uv run python .claude/experiments/test_gpu.py"
    echo "  2. Test vLLM: uv run python .claude/experiments/test_vllm.py"
    echo "  3. Review README.md for usage examples"
    echo "  4. Check configs/example_model.yaml for configuration"
    echo ""
    echo "To run commands:"
    echo "  uv run python your_script.py"
    echo ""
    echo "Or activate the environment:"
    echo "  source .venv/bin/activate"
    echo ""
}

# Run main
main
