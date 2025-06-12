# RETFound Training Monitoring System

Un dashboard web moderne et élégant pour suivre l'entraînement des modèles RETFound en temps réel.

## 🎯 Fonctionnalités

### Dashboard Principal
- **Métriques temps réel** : Loss, Accuracy, AUC-ROC, F1-Score
- **Monitoring critique** : RAO, RVO, Retinal Detachment avec seuils de sensibilité
- **Performance par classe** : Suivi des 28 classes (18 Fundus + 10 OCT)
- **Ressources système** : GPU, RAM, température en temps réel
- **Contrôles d'entraînement** : Start/Pause/Stop avec confirmations

### Visualisations Avancées
- **Courbes de loss** interactives avec zoom et export
- **Graphiques de métriques** avec smoothing ajustable
- **Matrice de confusion** 28x28 interactive
- **Timeline des epochs** avec détails complets
- **Alertes visuelles** pour conditions critiques

### Spécificités RETFound
- **Dataset v6.1** : 211,952 images, 28 classes
- **Conditions critiques** : RAO (99%), RVO (97%), Retinal Detachment (99%)
- **Seuils de sensibilité** configurables par pathologie
- **Performance optimisée** : Rolling buffers, throttling WebSocket

## 🏗️ Architecture

### Backend (FastAPI + WebSocket)
```
retfound/monitoring/
├── server.py              # Serveur FastAPI avec WebSocket
├── monitor_callback.py    # PyTorch Lightning callback
├── data_manager.py        # Gestion données et cache
├── api_routes.py          # Routes REST basiques
└── demo.py               # Démonstration complète
```

### Frontend (React + TypeScript)
```
retfound/monitoring/frontend/
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── Dashboard/
│   │   │   ├── Header.tsx         # Status, controls, timer
│   │   │   ├── MetricsGrid.tsx    # Métriques principales
│   │   │   └── ProgressBar.tsx    # Progression globale
│   │   ├── Charts/
│   │   │   ├── LossChart.tsx      # Train/Val loss curves
│   │   │   ├── MetricsChart.tsx   # Accuracy, AUC, F1
│   │   │   ├── ConfusionMatrix.tsx # Matrice 28x28
│   │   │   └── ClassPerformance.tsx # Performance par classe
│   │   └── Monitoring/
│   │       ├── CriticalAlerts.tsx  # Alertes conditions critiques
│   │       ├── GPUStats.tsx        # GPU/Memory usage
│   │       └── EpochDetails.tsx    # Détails par epoch
│   ├── hooks/
│   │   └── useWebSocket.ts
│   ├── store/
│   │   └── monitoring.ts          # Zustand store
│   └── utils/
│       └── formatters.ts
```

## 🚀 Installation et Utilisation

### Prérequis
- Python 3.8+
- Node.js 16+
- npm ou yarn

### Installation Backend
```bash
# Les dépendances sont déjà dans requirements.txt
pip install fastapi[all] pytorch-lightning websockets
```

### Installation Frontend
```bash
cd retfound/monitoring/frontend
npm install
```

### Démarrage

#### 1. Mode Démonstration (Recommandé)
```bash
# Démonstration complète avec simulation
python -m retfound.monitoring.demo
```

#### 2. Mode Développement
```bash
# Terminal 1: Backend
python -m retfound.monitoring.server

# Terminal 2: Frontend
cd retfound/monitoring/frontend
npm run dev
```

#### 3. Intégration Training
```python
from retfound.monitoring import create_monitoring_callback

# Ajouter au trainer PyTorch Lightning
callback = create_monitoring_callback(
    callback_type='retfound',
    update_frequency=10
)
trainer.callbacks.append(callback)

# Démarrer l'entraînement avec monitoring
python -m retfound.cli.main train --config configs/dataset_v6.1.yaml --enable-monitoring
```

## 📊 Interface WebSocket

### Structure des Messages
```typescript
interface MetricsUpdate {
  type: 'metrics_update'
  epoch: number
  batch: number
  total_batches: number
  metrics: {
    loss: { train: number; val?: number }
    accuracy: { train: number; val?: number }
    auc_roc: { macro: number; weighted: number }
    f1_score: number
    learning_rate: number
    critical_conditions: {
      [condition: string]: {
        sensitivity: number
        threshold: number
        status: 'ok' | 'warning' | 'critical'
      }
    }
    per_class: { [className: string]: number }
  }
  system: {
    gpu_usage: number
    gpu_memory: number
    gpu_temp: number
    ram_usage: number
    eta_seconds: number
  }
  timestamp: string
}
```

### Endpoints API
- `GET /api/status` - Status du training
- `GET /api/metrics/history` - Historique des métriques
- `GET /api/metrics/current` - Métriques actuelles
- `POST /api/training/start` - Démarrer l'entraînement
- `POST /api/training/pause` - Mettre en pause
- `POST /api/training/stop` - Arrêter l'entraînement
- `WebSocket /ws` - Communication temps réel

## 🎨 Design et Thème

### Couleurs
```css
--bg-primary: #0a0a0a;        /* Fond principal */
--bg-secondary: #1a1a1a;      /* Fond secondaire */
--accent: #6b46c1;            /* RETFound purple */
--success: #10b981;           /* Succès */
--warning: #f59e0b;           /* Avertissement */
--danger: #ef4444;            /* Danger */
--text-primary: #ffffff;      /* Texte principal */
--text-secondary: #a0a0a0;    /* Texte secondaire */
```

### Responsive Design
- **Desktop** : 3 colonnes, layout complet
- **Tablet** : 2 colonnes, composants adaptés
- **Mobile** : 1 colonne, navigation optimisée

## 🔧 Configuration

### Variables d'Environnement
```bash
# Serveur
MONITORING_HOST=localhost
MONITORING_PORT=8000
MONITORING_DEBUG=true

# Base de données
MONITORING_DB_PATH=./monitoring.db
MONITORING_CACHE_SIZE=1000

# WebSocket
WS_UPDATE_FREQUENCY=10
WS_MAX_CONNECTIONS=50
```

### Personnalisation
```python
# Configuration du callback
callback = create_monitoring_callback(
    callback_type='retfound',
    update_frequency=10,          # Updates par seconde
    critical_thresholds={
        'RAO': 0.99,
        'RVO': 0.97,
        'Retinal_Detachment': 0.99
    },
    enable_class_monitoring=True,
    enable_system_monitoring=True
)
```

## 📈 Optimisations Performance

### Backend
- **Cache SQLite** : Stockage optimisé des métriques
- **Rolling buffers** : Limitation à 1000 points par série
- **Throttling WebSocket** : Max 10 updates/sec
- **Compression** : Données compressées pour WebSocket

### Frontend
- **React.memo** : Optimisation composants lourds
- **Virtual scrolling** : Pour longues listes
- **Debouncing** : Limitation des re-renders
- **Code splitting** : Chargement lazy des composants

## 🐛 Dépannage

### Erreurs Communes

#### WebSocket Connection Failed
```bash
# Vérifier que le serveur est démarré
python -m retfound.monitoring.server

# Vérifier les ports
netstat -an | findstr :8000
```

#### TypeScript Errors
```bash
# Réinstaller les dépendances
cd retfound/monitoring/frontend
rm -rf node_modules package-lock.json
npm install
```

#### GPU Monitoring Issues
```python
# Vérifier CUDA
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

## 📝 Logs et Debug

### Activation des Logs
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Logs spécifiques monitoring
logger = logging.getLogger('retfound.monitoring')
logger.setLevel(logging.DEBUG)
```

### Fichiers de Log
- `monitoring.log` : Logs généraux
- `websocket.log` : Communications WebSocket
- `metrics.log` : Historique des métriques

## 🤝 Contribution

### Structure du Code
- **Backend** : Python avec FastAPI et asyncio
- **Frontend** : React + TypeScript + TailwindCSS
- **Tests** : pytest pour backend, Jest pour frontend
- **Documentation** : Markdown avec exemples

### Guidelines
1. Suivre les conventions de nommage
2. Ajouter des tests pour nouvelles fonctionnalités
3. Documenter les APIs
4. Optimiser les performances

## 📄 Licence

Ce projet fait partie de RETFound et suit la même licence.

## 🆘 Support

Pour toute question ou problème :
1. Vérifier cette documentation
2. Consulter les logs d'erreur
3. Tester avec le mode démonstration
4. Créer une issue avec détails complets

---

**Dashboard RETFound Monitoring** - Surveillance temps réel pour l'entraînement de modèles de vision ophtalmologique.
