#!/bin/bash
# RETFound Environment Setup Script
# =================================
# Sets up the complete environment for RETFound training

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV_NAME="retfound"
PYTHON_VERSION="3.9"
CUDA_VERSION="11.8"
PYTORCH_VERSION="2.3.1"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RETFound Environment Setup${NC}"
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

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_warning "This script is designed for Linux. Some features may not work on other platforms."
fi

# Check for NVIDIA GPU
check_gpu() {
    print_info "Checking for NVIDIA GPU..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
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
        
        # Check if version is compatible
        MAJOR_VERSION=$(echo $CUDA_VERSION_INSTALLED | cut -d'.' -f1)
        if [ "$MAJOR_VERSION" -lt 11 ]; then
            print_warning "CUDA version is less than 11. PyTorch may not be fully compatible."
        fi
    else
        print_error "CUDA not found. Please install CUDA $CUDA_VERSION or higher"
        return 1
    fi
}

# Check conda/mamba
check_conda() {
    print_info "Checking for conda/mamba..."
    
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
        print_success "Mamba detected (faster than conda)"
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
        print_success "Conda detected"
    else
        print_error "Neither conda nor mamba found. Please install Anaconda/Miniconda/Mambaforge"
        echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
}

# Create conda environment
create_environment() {
    print_info "Creating conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
    
    # Check if environment already exists
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        print_warning "Environment '$CONDA_ENV_NAME' already exists"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n $CONDA_ENV_NAME -y
        else
            print_info "Using existing environment"
            return 0
        fi
    fi
    
    # Create new environment
    $CONDA_CMD create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    print_success "Environment created successfully"
}

# Install PyTorch
install_pytorch() {
    print_info "Installing PyTorch $PYTORCH_VERSION with CUDA $CUDA_VERSION support..."
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
    
    # Determine CUDA version for PyTorch
    if [ "$CUDA_VERSION" = "11.8" ]; then
        PYTORCH_CUDA="cu118"
    elif [ "$CUDA_VERSION" = "12.1" ]; then
        PYTORCH_CUDA="cu121"
    else
        print_warning "CUDA version $CUDA_VERSION may not have pre-built PyTorch wheels"
        PYTORCH_CUDA="cu118"  # Default to 11.8
    fi
    
    # Install PyTorch
    pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url https://download.pytorch.org/whl/$PYTORCH_CUDA
    
    # Verify installation
    python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"
    
    if [ $? -eq 0 ]; then
        print_success "PyTorch installed successfully"
    else
        print_error "PyTorch installation failed"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_info "Installing RETFound dependencies..."
    
    # Core dependencies
    pip install --upgrade pip setuptools wheel
    
    # Install from pyproject.toml if it exists
    if [ -f "pyproject.toml" ]; then
        print_info "Installing from pyproject.toml..."
        pip install -e ".[all]"
    else
        # Manual installation of key packages
        print_info "Installing core packages manually..."
        
        # Core ML packages
        pip install \
            timm==0.9.16 \
            numpy==1.24.3 \
            scikit-learn==1.3.0 \
            matplotlib==3.7.1 \
            seaborn==0.12.2 \
            tqdm==4.65.0 \
            Pillow==10.0.0 \
            opencv-python==4.8.0.74 \
            albumentations==1.3.1 \
            pyyaml==6.0.1
        
        # Monitoring and logging
        pip install \
            tensorboard==2.13.0 \
            wandb==0.15.5
        
        # Export formats
        pip install \
            onnx==1.14.0 \
            onnxruntime-gpu==1.15.1
        
        # Development tools
        pip install \
            pytest==7.4.0 \
            pytest-cov==4.1.0 \
            black==23.7.0 \
            flake8==6.1.0 \
            mypy==1.4.1 \
            isort==5.12.0
        
        # Optional optimization tools
        pip install optuna==3.2.0
    fi
    
    print_success "Dependencies installed successfully"
}

# Download RETFound weights
download_weights() {
    print_info "Downloading RETFound pre-trained weights..."
    
    WEIGHTS_DIR="weights"
    mkdir -p $WEIGHTS_DIR
    
    # RETFound model URLs
    declare -A WEIGHTS_URLS=(
        ["cfp"]="https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureCFP.pth"
        ["oct"]="https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureOCT.pth"
        ["meh"]="https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_meh.pth"
    )
    
    for model in "${!WEIGHTS_URLS[@]}"; do
        FILENAME="RETFound_${model}.pth"
        FILEPATH="$WEIGHTS_DIR/$FILENAME"
        
        if [ -f "$FILEPATH" ]; then
            print_warning "$FILENAME already exists, skipping..."
        else
            print_info "Downloading $FILENAME..."
            wget -q --show-progress "${WEIGHTS_URLS[$model]}" -O "$FILEPATH"
            
            if [ $? -eq 0 ]; then
                print_success "$FILENAME downloaded successfully"
            else
                print_error "Failed to download $FILENAME"
            fi
        fi
    done
}

# Setup directories
setup_directories() {
    print_info "Setting up directory structure..."
    
    # Create necessary directories
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p outputs/checkpoints
    mkdir -p outputs/logs
    mkdir -p outputs/reports
    mkdir -p outputs/exports
    mkdir -p configs
    mkdir -p notebooks
    
    print_success "Directory structure created"
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Test imports
    python -c "
import torch
import timm
import albumentations
import cv2
import wandb
import retfound
print('All imports successful!')
print(f'RETFound version: {retfound.__version__}')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation verified successfully"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Create example configuration
create_example_config() {
    print_info "Creating example configuration..."
    
    cat > configs/example.yaml << EOF
# RETFound Training Configuration
# Generated by setup script

# Model configuration
model_type: vit_large_patch16_224
num_classes: 22
input_size: 224

# Training configuration
batch_size: 16
epochs: 100
base_lr: 5.0e-5

# Optimization
use_sam: true
use_ema: true
use_tta: true

# Data paths
dataset_path: ./data/processed
output_path: ./outputs

# Hardware optimization
use_amp: true
use_compile: true
use_gradient_checkpointing: true

# Monitoring
use_tensorboard: true
use_wandb: false
wandb_project: retfound-training
EOF

    print_success "Example configuration created at configs/example.yaml"
}

# Main setup flow
main() {
    echo
    print_info "Starting RETFound environment setup..."
    echo
    
    # System checks
    GPU_AVAILABLE=false
    if check_gpu; then
        GPU_AVAILABLE=true
        check_cuda || print_warning "CUDA not found, will use CPU"
    else
        print_warning "No GPU detected, will set up CPU-only environment"
    fi
    
    echo
    check_conda
    
    echo
    create_environment
    
    echo
    install_pytorch
    
    echo
    install_dependencies
    
    echo
    setup_directories
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo
        read -p "Download RETFound pre-trained weights? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            download_weights
        fi
    fi
    
    echo
    create_example_config
    
    # Final instructions
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    echo "To activate the environment, run:"
    echo -e "  ${YELLOW}conda activate $CONDA_ENV_NAME${NC}"
    echo
    echo "To start training, run:"
    echo -e "  ${YELLOW}retfound train --config configs/example.yaml${NC}"
    echo
    echo "For more information, see the documentation."
    echo
}

# Run main function
main

# Activation reminder
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "${YELLOW}Note: Remember to activate the environment before use!${NC}"
fi