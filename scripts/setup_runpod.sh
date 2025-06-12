#!/bin/bash
# RETFound RunPod Setup Script
# ============================
# Optimized setup for RunPod instances without conda

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RETFound RunPod Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Python version
check_python() {
    print_info "Checking Python version..."
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    print_success "Python $PYTHON_VERSION detected"
    
    if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 9 ]); then
        print_error "Python 3.9+ required, found $PYTHON_VERSION"
        exit 1
    fi
}

# Check for NVIDIA GPU
check_gpu() {
    print_info "Checking for NVIDIA GPU..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
        echo "$GPU_INFO"
        print_success "NVIDIA GPU detected"
        return 0
    else
        print_error "No NVIDIA GPU detected or nvidia-smi not found"
        return 1
    fi
}

# Check CUDA version
check_cuda() {
    print_info "Checking CUDA version..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION_INSTALLED=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_success "CUDA $CUDA_VERSION_INSTALLED detected"
        
        # Extract major version
        MAJOR_VERSION=$(echo $CUDA_VERSION_INSTALLED | cut -d'.' -f1)
        if [ "$MAJOR_VERSION" -lt 11 ]; then
            print_warning "CUDA version is less than 11. PyTorch may not be fully compatible."
        fi
        
        # Set PyTorch CUDA version
        if [[ "$CUDA_VERSION_INSTALLED" == 11.8* ]]; then
            PYTORCH_CUDA="cu118"
        elif [[ "$CUDA_VERSION_INSTALLED" == 12.1* ]]; then
            PYTORCH_CUDA="cu121"
        else
            PYTORCH_CUDA="cu118"  # Default fallback
            print_warning "Using default CUDA 11.8 PyTorch build"
        fi
        
        export PYTORCH_CUDA
        return 0
    else
        print_error "CUDA not found. GPU training will not be available."
        export PYTORCH_CUDA="cpu"
        return 1
    fi
}

# Update system packages
update_system() {
    print_info "Updating system packages..."
    apt-get update -qq
    apt-get install -y -qq \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        wget \
        curl \
        git
    print_success "System packages updated"
}

# Upgrade pip and install build tools
setup_pip() {
    print_info "Setting up pip and build tools..."
    python3 -m pip install --upgrade pip setuptools wheel
    print_success "Pip and build tools updated"
}

# Install PyTorch
install_pytorch() {
    print_info "Installing PyTorch with CUDA support..."
    
    if [ "$PYTORCH_CUDA" = "cpu" ]; then
        print_warning "Installing CPU-only PyTorch"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        print_info "Installing PyTorch with $PYTORCH_CUDA support"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$PYTORCH_CUDA
    fi
    
    # Verify PyTorch installation
    python3 -c "
import torch
print(f'PyTorch {torch.__version__} installed successfully')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
    
    if [ $? -eq 0 ]; then
        print_success "PyTorch installed and verified successfully"
    else
        print_error "PyTorch installation verification failed"
        exit 1
    fi
}

# Install RETFound dependencies
install_retfound() {
    print_info "Installing RETFound and dependencies..."
    
    # Use RunPod-optimized requirements if available
    if [ -f "requirements-runpod.txt" ]; then
        print_info "Using RunPod-optimized requirements..."
        pip install -r requirements-runpod.txt
        pip install -e . --no-deps
        print_success "RETFound installed via requirements-runpod.txt"
    elif pip install -e ".[all]" 2>/dev/null; then
        print_success "RETFound installed via pyproject.toml"
    else
        print_warning "pyproject.toml installation failed, trying requirements.txt..."
        
        # Install dependencies from requirements.txt
        pip install -r requirements.txt
        
        # Install the package in development mode without extras
        pip install -e . --no-deps
        
        print_success "RETFound installed via requirements.txt"
    fi
}

# Install additional RunPod optimizations
install_runpod_extras() {
    print_info "Installing RunPod-specific optimizations..."
    
    # TensorRT (if available)
    if [ "$PYTORCH_CUDA" != "cpu" ]; then
        print_info "Attempting to install TensorRT..."
        pip install tensorrt || print_warning "TensorRT installation failed (may not be available)"
    fi
    
    # Additional performance libraries
    pip install \
        ninja \
        psutil \
        gputil
    
    print_success "RunPod optimizations installed"
}

# Setup directories
setup_directories() {
    print_info "Setting up directory structure..."
    
    # Create RunPod-specific directories
    mkdir -p /workspace/datasets
    mkdir -p /workspace/outputs/v6.1
    mkdir -p /workspace/checkpoints/v6.1
    mkdir -p /workspace/cache/v6.1
    mkdir -p /workspace/weights
    mkdir -p /workspace/logs
    mkdir -p /workspace/runs
    
    # Create local directories
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p outputs/checkpoints
    mkdir -p outputs/logs
    mkdir -p outputs/reports
    mkdir -p outputs/exports
    
    print_success "Directory structure created"
}

# Download RETFound weights
download_weights() {
    print_info "Downloading RETFound pre-trained weights..."
    
    WEIGHTS_DIR="/workspace/weights"
    
    # RETFound model URLs and filenames
    declare -A WEIGHTS_URLS=(
        ["RETFound_mae_natureCFP.pth"]="https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureCFP.pth"
        ["RETFound_mae_natureOCT.pth"]="https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureOCT.pth"
        ["RETFound_mae_meh.pth"]="https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_meh.pth"
    )
    
    for filename in "${!WEIGHTS_URLS[@]}"; do
        FILEPATH="$WEIGHTS_DIR/$filename"
        
        if [ -f "$FILEPATH" ]; then
            print_warning "$filename already exists, skipping..."
        else
            print_info "Downloading $filename..."
            wget -q --show-progress "${WEIGHTS_URLS[$filename]}" -O "$FILEPATH"
            
            if [ $? -eq 0 ]; then
                print_success "$filename downloaded successfully"
            else
                print_error "Failed to download $filename"
            fi
        fi
    done
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Test critical imports
    python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    
    import torchvision
    print(f'✓ TorchVision {torchvision.__version__}')
    
    import timm
    print(f'✓ TIMM {timm.__version__}')
    
    import albumentations
    print(f'✓ Albumentations {albumentations.__version__}')
    
    import cv2
    print(f'✓ OpenCV {cv2.__version__}')
    
    import numpy as np
    print(f'✓ NumPy {np.__version__}')
    
    import sklearn
    print(f'✓ Scikit-learn {sklearn.__version__}')
    
    # Test RETFound import
    import retfound
    print(f'✓ RETFound {retfound.__version__}')
    
    print('\n🎉 All critical packages imported successfully!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation verified successfully"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Create RunPod-optimized config
create_runpod_config() {
    print_info "Verifying RunPod configuration..."
    
    if [ -f "configs/runpod.yaml" ]; then
        print_success "RunPod configuration already exists"
    else
        print_warning "RunPod configuration not found, but this is expected"
    fi
}

# Performance test
run_performance_test() {
    print_info "Running quick performance test..."
    
    python3 -c "
import torch
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Testing on GPU: {torch.cuda.get_device_name()}')
    
    # Simple tensor operations test
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    start_time = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f'GPU performance test: {(end_time - start_time)*1000:.2f}ms for 100 matrix multiplications')
    print('✓ GPU is working correctly')
else:
    print('⚠️  GPU not available, using CPU')
"
}

# Main setup function
main() {
    echo
    print_info "Starting RETFound RunPod setup..."
    echo
    
    # System checks
    check_python
    echo
    
    GPU_AVAILABLE=false
    if check_gpu; then
        GPU_AVAILABLE=true
        check_cuda
    else
        print_warning "No GPU detected, setting up CPU-only environment"
        export PYTORCH_CUDA="cpu"
    fi
    
    echo
    update_system
    
    echo
    setup_pip
    
    echo
    install_pytorch
    
    echo
    install_retfound
    
    echo
    install_runpod_extras
    
    echo
    setup_directories
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo
        read -p "Download RETFound pre-trained weights? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            download_weights
        fi
    fi
    
    echo
    create_runpod_config
    
    echo
    verify_installation
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo
        run_performance_test
    fi
    
    # Final success message
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🎉 RunPod Setup Completed Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    echo "Environment is ready for RETFound training!"
    echo
    echo "Quick start commands:"
    echo -e "  ${YELLOW}# Test the installation${NC}"
    echo -e "  ${YELLOW}python3 -c 'import retfound; print(retfound.__version__)'${NC}"
    echo
    echo -e "  ${YELLOW}# Start training with RunPod config${NC}"
    echo -e "  ${YELLOW}retfound train --config configs/runpod.yaml${NC}"
    echo
    echo -e "  ${YELLOW}# Or use the CLI help${NC}"
    echo -e "  ${YELLOW}retfound --help${NC}"
    echo
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "GPU training is enabled and ready to use!"
    else
        echo "CPU-only training is configured."
    fi
    echo
}

# Run main function
main "$@"
