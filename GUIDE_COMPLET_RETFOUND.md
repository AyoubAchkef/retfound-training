# Guide Complet RETFound - Installation et Entraînement

## 🎯 Guide Unique et Simplifié

Ce guide remplace tous les autres fichiers de documentation. Suivez ces étapes dans l'ordre.

## 📋 Prérequis

- Python 3.8+
- Node.js 18+ (pour le monitoring)
- GPU NVIDIA avec CUDA (recommandé)
- 16GB+ RAM

## 🚀 Installation Complète

### 1. Cloner le Projet
```bash
git clone https://github.com/AyoubAchkef/retfound-training.git
cd retfound-training
```

### 2. Installation Python
```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Linux/macOS:
source venv/bin/activate
# Sur Windows:
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Installation Frontend (Monitoring)
```bash
# Fixer le frontend si nécessaire
chmod +x fix_frontend_permissions.sh
./fix_frontend_permissions.sh

# Ou manuellement:
cd retfound/monitoring/frontend
npm install
npm run build
cd ../../..
```

### 4. Configuration
```bash
# Copier le fichier de configuration
cp .env.example .env

# Éditer .env avec vos paramètres
# Ou utiliser .env.runpod pour RunPod
```

## 🔧 Vérification du Setup

### Vérifier l'Installation
```bash
# Vérifier Python et les dépendances
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"

# Vérifier le CLI RETFound
python -m retfound.cli --help

# Vérifier le frontend
cd retfound/monitoring/frontend
npm run build
cd ../../..
```

### Test Rapide
```bash
# Test du modèle
python -c "from retfound.models import RETFoundModel; print('Modèle OK')"

# Test du monitoring
python retfound/monitoring/demo.py
```

## 🏋️ Entraînement

### 1. Préparer les Données
```bash
# Placer vos données dans le dossier approprié
# Structure attendue:
# data/
#   ├── train/
#   ├── val/
#   └── test/
```

### 2. Lancer l'Entraînement Simple
```bash
# Entraînement basique
python -m retfound.cli train --config configs/default.yaml

# Avec monitoring en temps réel
python -m retfound.cli train --config configs/default.yaml --monitor
```

### 3. Entraînement avec Monitoring Complet
```bash
# Terminal 1: Lancer le serveur de monitoring
python retfound/monitoring/server.py

# Terminal 2: Lancer l'entraînement
python -m retfound.cli train --config configs/default.yaml --monitor

# Ouvrir http://localhost:8000 dans votre navigateur
```

### 4. Configurations Avancées

#### Pour RunPod/GPU Puissant
```bash
python -m retfound.cli train --config configs/runpod.yaml
```

#### Pour Multi-GPU
```bash
python -m retfound.cli train --config configs/production/multi_gpu.yaml
```

#### Pour A100
```bash
python -m retfound.cli train --config configs/production/a100_optimized.yaml
```

## 📊 Monitoring en Temps Réel

### Démarrer le Dashboard
```bash
# Lancer le serveur de monitoring
python retfound/monitoring/server.py

# Dans un autre terminal, lancer l'entraînement avec monitoring
python -m retfound.cli train --config configs/default.yaml --monitor
```

### Accéder au Dashboard
- Ouvrir votre navigateur
- Aller à `http://localhost:8000`
- Voir les métriques en temps réel

## 🔍 Évaluation et Prédiction

### Évaluer un Modèle
```bash
python -m retfound.cli evaluate --model-path checkpoints/best_model.pth --data-path data/test
```

### Faire des Prédictions
```bash
python -m retfound.cli predict --model-path checkpoints/best_model.pth --image-path image.jpg
```

## 🛠️ Dépannage

### Problème Frontend
```bash
# Nettoyer et réinstaller
cd retfound/monitoring/frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Problème CUDA
```bash
# Vérifier CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Problème Dépendances
```bash
# Réinstaller les dépendances
pip install --upgrade -r requirements.txt
```

## 📁 Structure du Projet

```
retfound-training/
├── retfound/                 # Code principal
│   ├── models/              # Modèles
│   ├── training/            # Entraînement
│   ├── data/               # Gestion des données
│   ├── monitoring/         # Monitoring temps réel
│   └── cli/                # Interface ligne de commande
├── configs/                # Configurations
├── scripts/               # Scripts utilitaires
├── tests/                 # Tests
├── requirements.txt       # Dépendances Python
└── README.md             # Documentation principale
```

## 🎯 Commandes Essentielles

```bash
# Installation complète
pip install -r requirements.txt && ./fix_frontend_permissions.sh

# Entraînement simple
python -m retfound.cli train --config configs/default.yaml

# Entraînement avec monitoring
python retfound/monitoring/server.py &
python -m retfound.cli train --config configs/default.yaml --monitor

# Évaluation
python -m retfound.cli evaluate --model-path checkpoints/best_model.pth

# Prédiction
python -m retfound.cli predict --model-path checkpoints/best_model.pth --image-path image.jpg
```

## 🆘 Support

En cas de problème :
1. Vérifiez que toutes les dépendances sont installées
2. Vérifiez que CUDA fonctionne (si GPU)
3. Consultez les logs d'erreur
4. Utilisez `./fix_frontend_permissions.sh` pour les problèmes frontend

---

**Ce guide remplace tous les autres fichiers de documentation. Gardez uniquement celui-ci pour éviter la confusion.**
