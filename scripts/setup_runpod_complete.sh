#!/bin/bash
# =============================================================================
# RETFound Training Setup Script for RunPod - Complete Installation
# =============================================================================
# This script sets up the complete RETFound training environment on RunPod
# including backend API, frontend monitoring, and training preparation.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Banner
echo -e "${PURPLE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                RETFound Training Setup - RunPod              ║
║                        Version 2.0.0                          ║
║                    CAASI Medical AI Team                      ║
║               Dataset v6.1 - 211,952 images                   ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if running on RunPod
if [[ -z "${RUNPOD_POD_ID}" ]]; then
    warn "RUNPOD_POD_ID not detected. This script is optimized for RunPod."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: System Information
log "=== STEP 1: System Information ==="
info "Hostname: $(hostname)"
info "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
info "Kernel: $(uname -r)"
info "CPU: $(nproc) cores"
info "RAM: $(free -h | awk '/^Mem:/ {print $2}')"

# GPU Information
if command -v nvidia-smi &> /dev/null; then
    info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | while read line; do
        info "  $line"
    done
else
    warn "nvidia-smi not found. GPU may not be available."
fi

# Step 2: Environment Setup
log "=== STEP 2: Environment Setup ==="

# Load environment variables
if [[ -f ".env.runpod" ]]; then
    log "Loading RunPod environment configuration..."
    set -a  # automatically export all variables
    source .env.runpod
    set +a
    info "Environment loaded from .env.runpod"
else
    warn ".env.runpod not found. Using default values."
    export DATASET_PATH="/workspace/datasets/DATASET_CLASSIFICATION"
    export OUTPUT_PATH="/workspace/outputs/v6.1"
    export CHECKPOINT_PATH="/workspace/checkpoints/v6.1"
    export CACHE_DIR="/workspace/cache/v6.1"
    export MONITORING_HOST="0.0.0.0"
    export MONITORING_PORT="8000"
    export FRONTEND_PORT="3000"
fi

# Create necessary directories
log "Creating directory structure..."
mkdir -p "${DATASET_PATH}" || warn "Could not create dataset directory"
mkdir -p "${OUTPUT_PATH}"
mkdir -p "${CHECKPOINT_PATH}"
mkdir -p "${CACHE_DIR}"
mkdir -p "/workspace/weights"
mkdir -p "/workspace/logs"
mkdir -p "/workspace/runs"  # For TensorBoard

info "Directory structure created successfully"

# Step 3: Python Environment Setup
log "=== STEP 3: Python Environment Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
info "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [[ ! -d "venv_retfound" ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv venv_retfound
    info "Virtual environment created"
else
    info "Virtual environment already exists"
fi

# Activate virtual environment
log "Activating virtual environment..."
source venv_retfound/bin/activate
info "Virtual environment activated"

# Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
log "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
if [[ -f "requirements-runpod.txt" ]]; then
    log "Installing RunPod-specific requirements..."
    pip install -r requirements-runpod.txt
elif [[ -f "requirements.txt" ]]; then
    log "Installing general requirements..."
    pip install -r requirements.txt
else
    error "No requirements file found!"
    exit 1
fi

# Install additional monitoring dependencies
log "Installing additional monitoring dependencies..."
pip install fastapi uvicorn websockets psutil GPUtil

info "Python dependencies installed successfully"

# Step 4: Dataset Verification
log "=== STEP 4: Dataset Verification ==="

if [[ -d "${DATASET_PATH}" ]]; then
    DATASET_SIZE=$(du -sh "${DATASET_PATH}" 2>/dev/null | cut -f1 || echo "Unknown")
    info "Dataset directory exists: ${DATASET_PATH}"
    info "Dataset size: ${DATASET_SIZE}"
    
    # Count images if possible
    if command -v find &> /dev/null; then
        IMAGE_COUNT=$(find "${DATASET_PATH}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
        info "Total images found: ${IMAGE_COUNT}"
        
        if [[ ${IMAGE_COUNT} -lt 200000 ]]; then
            warn "Expected ~211,952 images for CAASI v6.1. Found: ${IMAGE_COUNT}"
        fi
    fi
else
    error "Dataset directory not found: ${DATASET_PATH}"
    error "Please ensure the dataset is mounted correctly."
    exit 1
fi

# Step 5: RETFound Weights Setup
log "=== STEP 5: RETFound Weights Setup ==="

WEIGHTS_DIR="/workspace/weights"
WEIGHTS_URLS=(
    "https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0.0/RETFound_mae_natureCFP.pth"
    "https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0.0/RETFound_mae_natureOCT.pth"
    "https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0.0/RETFound_mae_meh.pth"
)

WEIGHTS_FILES=(
    "RETFound_mae_natureCFP.pth"
    "RETFound_mae_natureOCT.pth"
    "RETFound_mae_meh.pth"
)

for i in "${!WEIGHTS_FILES[@]}"; do
    WEIGHT_FILE="${WEIGHTS_DIR}/${WEIGHTS_FILES[$i]}"
    if [[ ! -f "${WEIGHT_FILE}" ]]; then
        log "Downloading ${WEIGHTS_FILES[$i]}..."
        wget -O "${WEIGHT_FILE}" "${WEIGHTS_URLS[$i]}" || {
            error "Failed to download ${WEIGHTS_FILES[$i]}"
            exit 1
        }
        info "Downloaded: ${WEIGHTS_FILES[$i]}"
    else
        info "Weight file already exists: ${WEIGHTS_FILES[$i]}"
    fi
done

# Step 6: Node.js and Frontend Setup
log "=== STEP 6: Frontend Setup ==="

# Check if Node.js is available
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    info "Node.js version: $NODE_VERSION"
    
    # Install frontend dependencies
    if [[ -d "retfound/monitoring/frontend" ]]; then
        log "Installing frontend dependencies..."
        cd retfound/monitoring/frontend
        
        if command -v npm &> /dev/null; then
            npm install
            info "Frontend dependencies installed"
            
            # Build frontend for production
            log "Building frontend for production..."
            npm run build
            info "Frontend built successfully"
        else
            warn "npm not found. Frontend will not be available."
        fi
        
        cd - > /dev/null
    else
        warn "Frontend directory not found"
    fi
else
    warn "Node.js not found. Installing Node.js..."
    
    # Install Node.js using NodeSource repository
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    if command -v node &> /dev/null; then
        info "Node.js installed successfully: $(node --version)"
        
        # Retry frontend setup
        if [[ -d "retfound/monitoring/frontend" ]]; then
            cd retfound/monitoring/frontend
            npm install
            npm run build
            cd - > /dev/null
            info "Frontend setup completed"
        fi
    else
        warn "Failed to install Node.js. Frontend will not be available."
    fi
fi

# Step 7: Configuration Validation
log "=== STEP 7: Configuration Validation ==="

# Validate dataset configuration
log "Validating dataset configuration..."
python3 -c "
import sys
sys.path.append('.')
from retfound.core.config import load_config
try:
    config = load_config('configs/runpod.yaml')
    print('✓ Configuration loaded successfully')
    print(f'✓ Dataset path: {config.dataset_path}')
    print(f'✓ Number of classes: {config.model.num_classes}')
    print(f'✓ Batch size: {config.training.batch_size}')
except Exception as e:
    print(f'✗ Configuration error: {e}')
    sys.exit(1)
"

# Step 8: Service Scripts Creation
log "=== STEP 8: Creating Service Scripts ==="

# Create monitoring server start script
cat > start_monitoring.sh << 'EOF'
#!/bin/bash
# Start RETFound Monitoring Server

source venv_retfound/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting RETFound Monitoring Server..."
python -m retfound.monitoring.server \
    --host 0.0.0.0 \
    --port 8000 \
    --frontend-dir retfound/monitoring/frontend/dist
EOF

chmod +x start_monitoring.sh

# Create training start script
cat > start_training.sh << 'EOF'
#!/bin/bash
# Start RETFound Training with Monitoring

source venv_retfound/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load environment
if [[ -f ".env.runpod" ]]; then
    set -a
    source .env.runpod
    set +a
fi

echo "Starting RETFound Training..."
python -m retfound.cli.main train \
    --config configs/runpod.yaml \
    --monitor-critical \
    --enable-monitoring \
    "$@"
EOF

chmod +x start_training.sh

# Create full stack start script
cat > start_full_stack.sh << 'EOF'
#!/bin/bash
# Start Full RETFound Stack (Monitoring + Training)

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up background processes..."
    jobs -p | xargs -r kill
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting RETFound Full Stack..."

# Start monitoring server in background
echo "Starting monitoring server..."
./start_monitoring.sh &
MONITORING_PID=$!

# Wait for monitoring server to start
sleep 5

# Check if monitoring server is running
if kill -0 $MONITORING_PID 2>/dev/null; then
    echo "✓ Monitoring server started (PID: $MONITORING_PID)"
    echo "✓ Dashboard available at: http://0.0.0.0:8000"
    
    # Start training
    echo "Starting training..."
    ./start_training.sh "$@"
else
    echo "✗ Failed to start monitoring server"
    exit 1
fi

# Cleanup
cleanup
EOF

chmod +x start_full_stack.sh

info "Service scripts created successfully"

# Step 9: System Optimization
log "=== STEP 9: System Optimization ==="

# Set environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Create performance optimization script
cat > optimize_system.sh << 'EOF'
#!/bin/bash
# System optimization for RETFound training

echo "Applying system optimizations..."

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Memory optimizations
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf

# Apply sysctl changes
sudo sysctl -p

echo "System optimizations applied"
EOF

chmod +x optimize_system.sh

# Step 10: Final Verification
log "=== STEP 10: Final Verification ==="

# Test imports
log "Testing Python imports..."
python3 -c "
import torch
import torchvision
import fastapi
import uvicorn
print('✓ All critical imports successful')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'✓ GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Test configuration loading
log "Testing configuration loading..."
python3 -c "
import sys
sys.path.append('.')
from retfound.core.config import load_config
config = load_config('configs/runpod.yaml')
print('✓ Configuration validation passed')
"

# Step 11: Summary and Instructions
log "=== SETUP COMPLETE ==="

echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                    SETUP COMPLETED SUCCESSFULLY              ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

info "RETFound training environment is ready!"
echo
echo -e "${CYAN}Available Commands:${NC}"
echo -e "  ${YELLOW}./start_monitoring.sh${NC}     - Start monitoring dashboard only"
echo -e "  ${YELLOW}./start_training.sh${NC}       - Start training with monitoring"
echo -e "  ${YELLOW}./start_full_stack.sh${NC}     - Start complete stack"
echo -e "  ${YELLOW}./optimize_system.sh${NC}      - Apply system optimizations"
echo
echo -e "${CYAN}Access Points:${NC}"
echo -e "  ${YELLOW}Monitoring Dashboard:${NC} http://0.0.0.0:8000"
echo -e "  ${YELLOW}API Documentation:${NC}    http://0.0.0.0:8000/docs"
echo -e "  ${YELLOW}WebSocket:${NC}            ws://0.0.0.0:8000/ws"
echo
echo -e "${CYAN}Quick Start:${NC}"
echo -e "  1. ${YELLOW}./start_full_stack.sh${NC}  # Start everything"
echo -e "  2. Open browser to monitoring dashboard"
echo -e "  3. Monitor training progress in real-time"
echo
echo -e "${CYAN}Dataset Information:${NC}"
echo -e "  ${YELLOW}Path:${NC}     ${DATASET_PATH}"
echo -e "  ${YELLOW}Classes:${NC}  28 (18 Fundus + 10 OCT)"
echo -e "  ${YELLOW}Images:${NC}   ~211,952 total"
echo
echo -e "${GREEN}Setup completed at: $(date)${NC}"

# Deactivate virtual environment
deactivate 2>/dev/null || true

log "Setup script finished successfully!"
