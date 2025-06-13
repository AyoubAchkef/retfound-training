# 🚀 Commandes Étape par Étape - RETFound

## 📋 Installation Complète

### 1. Cloner et Préparer l'Environnement
```bash
# Cloner le projet
git clone https://github.com/AyoubAchkef/retfound-training.git
cd retfound-training

# Créer l'environnement virtuel Python
python -m venv venv

# Activer l'environnement (Linux/macOS/RunPod)
source venv/bin/activate

# Activer l'environnement (Windows)
venv\Scripts\activate
```

### 2. Installer les Dépendances Python
```bash
# Installer toutes les dépendances
pip install -r requirements.txt

# Vérifier l'installation PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Installer et Fixer le Frontend
```bash
# Donner les permissions au script de fix
chmod +x fix_frontend_permissions.sh

# Lancer le fix automatique du frontend
./fix_frontend_permissions.sh
```

### 4. Configuration
```bash
# Copier le fichier de configuration
cp .env.example .env

# Éditer .env avec vos paramètres (optionnel)
# nano .env
```

## ✅ Vérification du Setup

### Vérifier Python et PyTorch
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

### Vérifier le CLI RETFound
```bash
python -m retfound.cli --help
```

### Vérifier le Frontend
```bash
cd retfound/monitoring/frontend
npm run build
cd ../../..
```

### Test Complet
```bash
# Test du modèle
python -c "from retfound.models import RETFoundModel; print('✅ Modèle OK')"

# Test du monitoring
python retfound/monitoring/demo.py
```

## 🏋️ Lancer l'Entraînement

### Option 1 : Entraînement Simple (Sans Monitoring)
```bash
# Entraînement basique avec config par défaut
python -m retfound.cli train --config configs/default.yaml
```

### Option 2 : Entraînement avec Monitoring Temps Réel
```bash
# Terminal 1 : Lancer le serveur de monitoring
python retfound/monitoring/server.py

# Terminal 2 : Lancer l'entraînement avec monitoring
python -m retfound.cli train --config configs/default.yaml --monitor

# Ouvrir http://localhost:8000 dans votre navigateur
```

### Option 3 : Entraînement RunPod/GPU Puissant
```bash
# Pour RunPod ou GPU puissant
python -m retfound.cli train --config configs/runpod.yaml --monitor
```

### Option 4 : Entraînement Multi-GPU
```bash
# Pour plusieurs GPU
python -m retfound.cli train --config configs/production/multi_gpu.yaml --monitor
```

### Option 5 : Entraînement A100 Optimisé
```bash
# Pour GPU A100
python -m retfound.cli train --config configs/production/a100_optimized.yaml --monitor
```

## 📊 Monitoring et Évaluation

### Accéder au Dashboard
1. Lancer le serveur : `python retfound/monitoring/server.py`
2. Ouvrir : `http://localhost:8000`
3. Voir les métriques en temps réel

### Évaluer un Modèle Entraîné
```bash
python -m retfound.cli evaluate \
  --model-path checkpoints/best_model.pth \
  --data-path data/test
```

### Faire des Prédictions
```bash
python -m retfound.cli predict \
  --model-path checkpoints/best_model.pth \
  --image-path image.jpg
```

## 🛠️ Dépannage Rapide

### Problème Frontend
```bash
./fix_frontend_permissions.sh
```

### Problème Dépendances
```bash
pip install --upgrade -r requirements.txt
```

### Problème CUDA
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Nettoyer et Recommencer
```bash
# Nettoyer l'environnement
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt
./fix_frontend_permissions.sh
```

## 🎯 Commande Complète Tout-en-Un

### Installation + Vérification + Entraînement
```bash
# Installation complète
git clone https://github.com/AyoubAchkef/retfound-training.git
cd retfound-training
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
chmod +x fix_frontend_permissions.sh
./fix_frontend_permissions.sh

# Vérification
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -m retfound.cli --help

# Lancer l'entraînement avec monitoring
python retfound/monitoring/server.py &
python -m retfound.cli train --config configs/default.yaml --monitor
```

## 📝 Notes Importantes

- **Toujours activer l'environnement virtuel** avant de lancer des commandes
- **Sur RunPod** : Utilisez les commandes Linux (pas Windows)
- **Monitoring** : Ouvrir `http://localhost:8000` pour voir le dashboard
- **Logs** : Les logs d'entraînement sont sauvegardés automatiquement
- **Checkpoints** : Les modèles sont sauvegardés dans `checkpoints/`

---

**Suivez ces commandes dans l'ordre pour une installation et un entraînement réussis !**
