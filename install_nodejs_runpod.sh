#!/bin/bash
# =============================================================================
# Node.js Installation Script for RunPod
# =============================================================================
# This script installs Node.js on RunPod without requiring sudo

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                Node.js Installation for RunPod               ║
║                        Version 20.x LTS                       ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if Node.js is already installed
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    info "Node.js is already installed: $NODE_VERSION"
    
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        info "npm version: $NPM_VERSION"
    fi
    
    echo -e "${GREEN}Node.js installation is complete!${NC}"
    exit 0
fi

log "Installing Node.js on RunPod..."

# Check if we're running as root (typical on RunPod)
if [[ $EUID -eq 0 ]]; then
    log "Running as root - installing Node.js directly..."
    
    # Update package list
    log "Updating package list..."
    apt-get update
    
    # Install curl if not available
    if ! command -v curl &> /dev/null; then
        log "Installing curl..."
        apt-get install -y curl
    fi
    
    # Install Node.js using NodeSource repository
    log "Adding NodeSource repository..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    
    log "Installing Node.js..."
    apt-get install -y nodejs
    
else
    error "This script requires root privileges on RunPod"
    error "Please run as root or contact RunPod support"
    exit 1
fi

# Verify installation
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    info "✓ Node.js installed successfully: $NODE_VERSION"
    
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        info "✓ npm version: $NPM_VERSION"
    else
        warn "npm not found after Node.js installation"
    fi
    
    # Test Node.js
    log "Testing Node.js installation..."
    node -e "console.log('✓ Node.js is working correctly')"
    
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                Node.js Installation Complete!                ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    info "You can now run the frontend setup:"
    info "  cd retfound/monitoring/frontend"
    info "  npm install"
    info "  npm run build"
    
else
    error "Node.js installation failed"
    error "Please try manual installation:"
    error "  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"
    error "  apt-get install -y nodejs"
    exit 1
fi
