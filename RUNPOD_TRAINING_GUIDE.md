# Guide Complet d'Entraînement RETFound sur RunPod

## 📋 Vue d'Ensemble

Ce guide vous accompagne dans le déploiement complet du système d'entraînement RETFound sur RunPod, incluant :
- ✅ Backend API FastAPI avec WebSocket
- ✅ Frontend React/TypeScript pour monitoring temps réel
- ✅ Système d'entraînement avec callbacks de monitoring
- ✅ Support complet des 28 classes CAASI v6.1
- ✅ Monitoring des conditions critiques

## 🚀 Installation Rapide

### 1. Connexion SSH au RunPod

```bash
ssh root@216.81.245.138 -p 17040 -i ~/.ssh/id_ed25519
```

### 2. Clonage et Configuration

```bash
# Cloner le projet
cd /workspace
git clone https://github.com/AyoubAchkef/retfound-training.git
cd retfound-training

# Lancer l'installation complète
chmod +x scripts/setup_runpod_complete.sh
./scripts/setup_runpod_complete.sh
```

### 3. Lancement du Stack Complet

```bash
# Démarrer monitoring + entraînement
./start_full_stack.sh
```

### 4. Accès aux Interfaces

- **Dashboard de Monitoring** : `http://[RUNPOD_IP]:8000`
- **API Documentation** : `http://[RUNPOD_IP]:8000/docs`
- **WebSocket** : `ws://[RUNPOD_IP]:8000/ws`

## 📁 Structure du Projet

```
retfound-training/
├── configs/
│   ├── dataset_v6.1.yaml      # Configuration dataset v6.1
│   ├── runpod.yaml             # Configuration optimisée RunPod
│   └── default.yaml            # Configuration par défaut
├── retfound/
│   ├── monitoring/             # Système de monitoring
│   │   ├── server.py           # Serveur FastAPI + WebSocket
│   │   ├── api_routes.py       # Routes API REST
│   │   ├── data_manager.py     # Gestion des données
│   │   └── frontend/           # Interface React/TypeScript
│   │       ├── src/
│   │       │   ├── components/ # Composants UI
│   │       │   ├── hooks/      # Hooks WebSocket
│   │       │   └── store/      # État global Zustand
│   │       └── dist/           # Build de production
│   ├── core/                   # Core framework
│   ├── data/                   # Gestion des données
│   ├── models/                 # Modèles RETFound
│   └── training/               # Système d'entraînement
├── scripts/
│   └── setup_runpod_complete.sh # Script d'installation
├── .env.runpod                 # Configuration environnement
└── requirements-runpod.txt     # Dépendances RunPod
```

## ⚙️ Configuration Détaillée

### Dataset v6.1 - 28 Classes

**Fundus (18 classes)** :
- Normal, DR (4 stades), Glaucome, RVO, RAO, etc.

**OCT (10 classes)** :
- Normal, DME, CNV, Dry AMD, ERM, etc.

**Classes Critiques Surveillées** :
- RAO (Urgence) - Sensibilité min: 99%
- RVO (Urgent) - Sensibilité min: 97%
- Décollement rétinien - Sensibilité min: 99%
- CNV - Sensibilité min: 98%

### Configuration RunPod Optimisée

```yaml
# configs/runpod.yaml
dataset_path: "/workspace/datasets/DATASET_CLASSIFICATION"
training:
  batch_size: 32          # A100 40GB
  gradient_accumulation: 2 # Batch effectif: 64
  epochs: 100
optimizations:
  use_amp: true
  amp_dtype: "bfloat16"   # Spécifique A100
  use_compile: true
  compile_mode: "max-autotune"
```

## 🖥️ Interface de Monitoring

### Composants Principaux

1. **Dashboard Principal**
   - Métriques temps réel (Loss, Accuracy, AUC)
   - Progression d'entraînement
   - Statistiques GPU/RAM

2. **Monitoring des Classes**
   - Performance par classe
   - Alertes conditions critiques
   - Matrice de confusion

3. **Graphiques Avancés**
   - Courbes de perte/précision
   - Performance par époque
   - Monitoring GPU temps réel

4. **Contrôles d'Entraînement**
   - Start/Pause/Stop
   - Export des métriques
   - Gestion des checkpoints

### WebSocket en Temps Réel

```typescript
// Connexion automatique avec reconnexion
const { connect, sendMessage } = useWebSocket()

// Messages supportés :
// - metrics_update : Métriques d'entraînement
// - status_update : État d'entraînement
// - system_update : Stats système
// - critical_alert : Alertes critiques
```

## 🔧 API Backend

### Endpoints Principaux

```bash
# Contrôle d'entraînement
POST /api/training/start
POST /api/training/pause  
POST /api/training/stop

# Métriques
GET /api/metrics/latest
GET /api/metrics/history
GET /api/metrics/performance/{metric}

# Époques
GET /api/epochs
GET /api/epochs/{epoch}

# Alertes
GET /api/alerts

# Système
GET /api/stats
GET /api/health
```

### WebSocket Events

```json
{
  "type": "metrics_update",
  "epoch": 15,
  "batch": 100,
  "total_batches": 500,
  "metrics": {
    "loss": {"train": 0.234, "val": 0.267},
    "accuracy": {"train": 0.892, "val": 0.876},
    "critical_conditions": {
      "RAO": {"sensitivity": 0.991, "status": "ok"},
      "RVO": {"sensitivity": 0.973, "status": "ok"}
    }
  },
  "system": {
    "gpu_usage": 95.2,
    "gpu_memory": 38.4,
    "ram_usage": 45.6
  }
}
```

## 🚀 Commandes de Lancement

### Scripts Disponibles

```bash
# Monitoring seul
./start_monitoring.sh

# Entraînement avec monitoring
./start_training.sh

# Stack complet (recommandé)
./start_full_stack.sh

# Optimisations système
./optimize_system.sh
```

### Lancement Manuel

```bash
# Terminal 1: Backend API
source venv_retfound/bin/activate
python -m retfound.monitoring.server --host 0.0.0.0 --port 8000

# Terminal 2: Entraînement
source venv_retfound/bin/activate
python -m retfound.cli.main train --config configs/runpod.yaml --enable-monitoring

# Terminal 3: Frontend (développement)
cd retfound/monitoring/frontend
npm run dev -- --host 0.0.0.0 --port 3000
```

## 📊 Monitoring des Performances

### Métriques Surveillées

- **Loss** : Train/Validation en temps réel
- **Accuracy** : Précision globale et par classe
- **AUC-ROC** : Macro et weighted
- **F1-Score** : Score F1 global
- **Sensibilité** : Spéciale pour conditions critiques
- **GPU/RAM** : Utilisation ressources

### Alertes Critiques

Le système génère automatiquement des alertes pour :
- Sensibilité insuffisante sur conditions critiques
- Surcharge GPU/mémoire
- Divergence de l'entraînement
- Erreurs système

### Export des Données

```bash
# Via API
curl -X POST http://localhost:8000/api/export \
  -H "Content-Type: application/json" \
  -d '{"format": "json", "include_history": true}'

# Via interface web
# Bouton "Export" dans le dashboard
```

## 🔍 Dépannage

### Problèmes Courants

1. **WebSocket ne se connecte pas**
   ```bash
   # Vérifier que le serveur est démarré
   curl http://localhost:8000/health
   
   # Vérifier les ports
   netstat -tlnp | grep :8000
   ```

2. **Frontend ne charge pas**
   ```bash
   # Reconstruire le frontend
   cd retfound/monitoring/frontend
   npm run build
   ```

3. **Erreur de mémoire GPU**
   ```bash
   # Réduire la taille de batch
   # Modifier configs/runpod.yaml
   training:
     batch_size: 16  # Au lieu de 32
   ```

4. **Dataset non trouvé**
   ```bash
   # Vérifier le montage
   ls -la /workspace/datasets/DATASET_CLASSIFICATION
   
   # Vérifier la configuration
   grep dataset_path configs/runpod.yaml
   ```

### Logs et Debugging

```bash
# Logs du serveur de monitoring
tail -f /workspace/logs/monitoring.log

# Logs d'entraînement
tail -f /workspace/logs/training.log

# Logs système
dmesg | tail -20
```

## 📈 Optimisations Avancées

### Configuration A100

```yaml
# Pour A100 80GB
training:
  batch_size: 64
  gradient_accumulation: 1

# Pour multi-GPU
distributed:
  enabled: true
  backend: "nccl"
```

### Monitoring Personnalisé

```python
# Ajouter des métriques personnalisées
from retfound.monitoring.server import get_server

server = get_server()
await server.update_metrics({
    'custom_metric': value,
    'epoch': epoch,
    'batch': batch
})
```

## 🎯 Résultats Attendus

### Performance Cible (Dataset v6.1)

- **Accuracy Globale** : >92%
- **AUC-ROC Macro** : >0.95
- **Sensibilité RAO** : >99%
- **Sensibilité RVO** : >97%
- **F1-Score Moyen** : >0.90

### Temps d'Entraînement

- **A100 40GB** : ~8-12h pour 100 époques
- **A100 80GB** : ~6-8h pour 100 époques
- **Multi-GPU** : Temps divisé par nombre de GPUs

## 📞 Support

Pour toute question ou problème :

1. Vérifier les logs dans `/workspace/logs/`
2. Consulter l'API documentation : `http://[IP]:8000/docs`
3. Vérifier la santé du système : `http://[IP]:8000/health`

---

**Version** : 2.0.0  
**Dernière mise à jour** : 13 Juin 2025  
**Compatibilité** : RunPod A100, Dataset CAASI v6.1
