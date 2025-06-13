#!/bin/bash
# =============================================================================
# Quick Start Script pour RETFound Training sur RunPod
# =============================================================================
# Ce script lance rapidement l'entraînement RETFound avec monitoring complet

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Banner
echo -e "${PURPLE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                RETFound Quick Start - RunPod                 ║
║                        Version 2.0.0                          ║
║                    CAASI Medical AI Team                      ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Functions
log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: $1${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"; }
info() { echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO: $1${NC}"; }

# Check if we're on RunPod
if [[ -z "${RUNPOD_POD_ID}" ]]; then
    warn "Not running on RunPod. Some features may not work correctly."
fi

# Step 1: Quick validation
log "=== ÉTAPE 1: Validation Rapide ==="

if [[ ! -f "scripts/validate_setup.py" ]]; then
    error "Script de validation manquant. Exécutez d'abord le setup complet."
    exit 1
fi

info "Validation de la configuration..."
python3 scripts/validate_setup.py --quick 2>/dev/null || {
    warn "Validation échouée. Continuons quand même..."
}

# Step 2: Environment setup
log "=== ÉTAPE 2: Configuration Environnement ==="

# Load environment
if [[ -f ".env.runpod" ]]; then
    log "Chargement de la configuration RunPod..."
    set -a
    source .env.runpod
    set +a
else
    warn "Fichier .env.runpod non trouvé. Utilisation des valeurs par défaut."
    export DATASET_PATH="/workspace/datasets/DATASET_CLASSIFICATION"
    export MONITORING_PORT="8000"
fi

# Activate virtual environment
if [[ -d "venv_retfound" ]]; then
    log "Activation de l'environnement virtuel..."
    source venv_retfound/bin/activate
else
    error "Environnement virtuel non trouvé. Exécutez d'abord setup_runpod_complete.sh"
    exit 1
fi

# Step 3: Check dataset
log "=== ÉTAPE 3: Vérification Dataset ==="

if [[ -d "${DATASET_PATH}" ]]; then
    DATASET_SIZE=$(du -sh "${DATASET_PATH}" 2>/dev/null | cut -f1 || echo "Unknown")
    info "Dataset trouvé: ${DATASET_PATH} (${DATASET_SIZE})"
    
    # Quick count
    IMAGE_COUNT=$(find "${DATASET_PATH}" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    info "Images détectées: ${IMAGE_COUNT}"
    
    if [[ ${IMAGE_COUNT} -lt 100000 ]]; then
        warn "Nombre d'images faible. Vérifiez que le dataset est complet."
    fi
else
    error "Dataset non trouvé: ${DATASET_PATH}"
    error "Montez le dataset avant de continuer."
    exit 1
fi

# Step 4: GPU Check
log "=== ÉTAPE 4: Vérification GPU ==="

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    info "GPU détecté: ${GPU_INFO}"
    
    # Check memory
    GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
    if [[ ${GPU_MEMORY} -lt 30000 ]]; then
        warn "Mémoire GPU faible (${GPU_MEMORY}MB). Réduisez la taille de batch."
    fi
else
    error "nvidia-smi non trouvé. GPU non disponible."
    exit 1
fi

# Step 5: Start monitoring server
log "=== ÉTAPE 5: Démarrage du Serveur de Monitoring ==="

# Check if monitoring is already running
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    info "Serveur de monitoring déjà en cours d'exécution"
else
    log "Démarrage du serveur de monitoring..."
    
    # Start monitoring in background
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    nohup python -m retfound.monitoring.server \
        --host 0.0.0.0 \
        --port 8000 \
        --frontend-dir retfound/monitoring/frontend/dist \
        > /workspace/logs/monitoring.log 2>&1 &
    
    MONITORING_PID=$!
    echo $MONITORING_PID > /tmp/monitoring.pid
    
    # Wait for server to start
    log "Attente du démarrage du serveur..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            info "✓ Serveur de monitoring démarré (PID: $MONITORING_PID)"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:8000/health >/dev/null 2>&1; then
        error "Échec du démarrage du serveur de monitoring"
        exit 1
    fi
fi

# Step 6: Display access information
log "=== ÉTAPE 6: Informations d'Accès ==="

# Get RunPod IP if available
if [[ -n "${RUNPOD_PUBLIC_IP}" ]]; then
    PUBLIC_IP="${RUNPOD_PUBLIC_IP}"
elif [[ -n "${RUNPOD_TCP_PORT_8000}" ]]; then
    PUBLIC_IP="$(curl -s ifconfig.me):${RUNPOD_TCP_PORT_8000}"
else
    PUBLIC_IP="localhost:8000"
fi

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    ACCÈS AUX INTERFACES                      ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Dashboard Monitoring: http://${PUBLIC_IP}                    "
echo "║  API Documentation:   http://${PUBLIC_IP}/docs               "
echo "║  WebSocket:           ws://${PUBLIC_IP}/ws                    "
echo "║  Health Check:        http://${PUBLIC_IP}/health             "
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 7: Training options
log "=== ÉTAPE 7: Options d'Entraînement ==="

echo -e "${YELLOW}Choisissez une option:${NC}"
echo "1. Démarrer l'entraînement immédiatement"
echo "2. Démarrer l'entraînement avec validation préalable"
echo "3. Mode monitoring seulement (pas d'entraînement)"
echo "4. Test de configuration"
echo "5. Quitter"

read -p "Votre choix (1-5): " choice

case $choice in
    1)
        log "Démarrage de l'entraînement..."
        python -m retfound.cli.main train \
            --config configs/runpod.yaml \
            --enable-monitoring \
            --monitor-critical
        ;;
    2)
        log "Validation de la configuration..."
        python -m retfound.cli.main validate --config configs/runpod.yaml
        
        if [[ $? -eq 0 ]]; then
            log "Validation réussie. Démarrage de l'entraînement..."
            python -m retfound.cli.main train \
                --config configs/runpod.yaml \
                --enable-monitoring \
                --monitor-critical
        else
            error "Validation échouée. Corrigez les erreurs avant de continuer."
            exit 1
        fi
        ;;
    3)
        info "Mode monitoring activé. L'entraînement peut être démarré via l'interface web."
        info "Accédez à http://${PUBLIC_IP} pour contrôler l'entraînement."
        
        # Keep monitoring running
        echo "Appuyez sur Ctrl+C pour arrêter le monitoring..."
        tail -f /workspace/logs/monitoring.log
        ;;
    4)
        log "Test de configuration..."
        python -c "
import sys
sys.path.append('.')
from retfound.core.config import load_config
try:
    config = load_config('configs/runpod.yaml')
    print('✓ Configuration chargée avec succès')
    print(f'✓ Dataset: {config.dataset_path}')
    print(f'✓ Classes: {config.model.num_classes}')
    print(f'✓ Batch size: {config.training.batch_size}')
    print('✓ Test réussi!')
except Exception as e:
    print(f'✗ Erreur: {e}')
    sys.exit(1)
"
        ;;
    5)
        info "Arrêt du script. Le serveur de monitoring reste actif."
        exit 0
        ;;
    *)
        error "Option invalide. Relancez le script."
        exit 1
        ;;
esac

# Cleanup function
cleanup() {
    log "Nettoyage en cours..."
    
    # Stop monitoring if we started it
    if [[ -f "/tmp/monitoring.pid" ]]; then
        MONITORING_PID=$(cat /tmp/monitoring.pid)
        if kill -0 $MONITORING_PID 2>/dev/null; then
            log "Arrêt du serveur de monitoring..."
            kill $MONITORING_PID
            rm -f /tmp/monitoring.pid
        fi
    fi
    
    log "Nettoyage terminé."
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Final message
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    ENTRAÎNEMENT TERMINÉ                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

info "Logs disponibles dans /workspace/logs/"
info "Checkpoints sauvegardés dans /workspace/checkpoints/v6.1/"
info "Résultats dans /workspace/outputs/v6.1/"

# Keep script running if monitoring only
if [[ $choice -eq 3 ]]; then
    while true; do
        sleep 60
    done
fi
