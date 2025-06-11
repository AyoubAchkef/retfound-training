# Installation RETFound sur RunPod

Ce guide explique comment installer et configurer RETFound sur une instance RunPod.

## 🚀 Installation Rapide (Recommandée)

### Méthode 1 : Script automatisé RunPod

```bash
# Cloner le projet
git clone <votre-repo-url>
cd retfound-training

# Exécuter le script d'installation RunPod
bash scripts/setup_runpod.sh
```

Ce script :
- ✅ Détecte automatiquement votre GPU et CUDA
- ✅ Installe PyTorch avec le bon support CUDA
- ✅ Installe toutes les dépendances via pyproject.toml
- ✅ Configure les répertoires RunPod
- ✅ Télécharge les poids pré-entraînés
- ✅ Vérifie l'installation

### Méthode 2 : Installation manuelle pip

```bash
# Mise à jour du système
apt-get update && apt-get install -y libgl1-mesa-glx

# Installation PyTorch avec CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Option A: Installation via pyproject.toml (recommandée)
pip install -e ".[all]"

# Option B: Si pyproject.toml échoue, utiliser requirements.txt
pip install -r requirements.txt
pip install -e . --no-deps
```

## 📋 Gestion des Dépendances

### Pourquoi pas de requirements.txt ?

Le projet utilise **Poetry** et **pyproject.toml** au lieu de requirements.txt pour :
- Gestion fine des versions
- Dépendances optionnelles (monitoring, export, etc.)
- Résolution automatique des conflits
- Support des extras

### Structure des dépendances

```toml
[tool.poetry.dependencies]
# Dépendances principales
torch = "^2.0.0"
torchvision = "^0.15.0"
timm = "^0.9.0"
albumentations = "^1.3.0"

# Dépendances optionnelles
tensorboard = {version = "^2.14.0", optional = true}
wandb = {version = "^0.15.0", optional = true}
onnx = {version = "^1.14.0", optional = true}
tensorrt = {version = "^8.6.0", optional = true}

[tool.poetry.extras]
monitoring = ["tensorboard", "wandb"]
export = ["onnx", "onnxruntime"]
all = ["tensorboard", "wandb", "onnx", "onnxruntime"]
```

## 🔧 Configuration RunPod

### Répertoires automatiquement créés

```
/workspace/
├── datasets/           # Vos datasets
├── outputs/v6.1/      # Résultats d'entraînement
├── checkpoints/v6.1/  # Points de contrôle
├── cache/v6.1/        # Cache des données
├── weights/           # Poids pré-entraînés
├── logs/              # Logs d'entraînement
└── runs/              # TensorBoard logs
```

### Configuration optimisée A100

Le fichier `configs/runpod.yaml` est pré-configuré pour :
- **A100 80GB** : batch_size=64, bfloat16, torch.compile
- **Optimisations CUDA** : TF32, cuDNN benchmark
- **Monitoring complet** : TensorBoard, Wandb
- **Export multi-format** : TorchScript, ONNX, TensorRT

## 🚦 Vérification de l'installation

### Test rapide
```bash
# Vérifier l'import RETFound
python3 -c "import retfound; print(retfound.__version__)"

# Vérifier PyTorch + CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Tester le CLI
retfound --help
```

### Test complet
```bash
# Lancer le script de validation
python3 scripts/validate_dataset_v61.py

# Ou tester avec un petit dataset
retfound train --config configs/runpod.yaml --epochs 1
```

## 🎯 Démarrage Rapide

### 1. Préparer vos données
```bash
# Copier votre dataset dans RunPod
cp -r /path/to/your/dataset /workspace/datasets/DATASET_CLASSIFICATION
```

### 2. Lancer l'entraînement
```bash
# Entraînement avec configuration RunPod
retfound train --config configs/runpod.yaml

# Ou avec paramètres personnalisés
retfound train \
    --config configs/runpod.yaml \
    --dataset_path /workspace/datasets/DATASET_CLASSIFICATION \
    --batch_size 32 \
    --epochs 100
```

### 3. Monitoring
```bash
# TensorBoard (dans un autre terminal)
tensorboard --logdir /workspace/runs

# Ou utiliser Wandb (configuré dans runpod.yaml)
```

## 🔍 Dépannage

### Problèmes courants

**1. Erreur CUDA/PyTorch**
```bash
# Réinstaller PyTorch avec la bonne version CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Erreur OpenCV**
```bash
# Installer les dépendances système manquantes
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**3. Mémoire GPU insuffisante**
```bash
# Réduire la taille de batch dans configs/runpod.yaml
# batch_size: 16  # au lieu de 32
# gradient_accumulation: 4  # pour maintenir l'effective batch size
```

**4. Import RETFound échoue**
```bash
# Réinstaller en mode développement
pip install -e ".[all]"
```

### Logs utiles
```bash
# Vérifier les logs système
dmesg | tail

# Vérifier l'utilisation GPU
nvidia-smi

# Vérifier l'espace disque
df -h /workspace
```

## 📊 Optimisations RunPod

### Pour A100 80GB
```yaml
# Dans configs/runpod.yaml
training:
  batch_size: 64
  gradient_accumulation: 1

optimizations:
  use_amp: true
  amp_dtype: "bfloat16"  # Spécifique A100
  use_compile: true
  compile_mode: "max-autotune"
```

### Pour A100 40GB
```yaml
training:
  batch_size: 32
  gradient_accumulation: 2  # Effective batch = 64

optimizations:
  use_gradient_checkpointing: true  # Économise la mémoire
```

## 🔗 Ressources

- [Documentation RETFound](./README.md)
- [Configuration avancée](./configs/)
- [Scripts utiles](./scripts/)
- [Tests](./tests/)

## 💡 Conseils

1. **Utilisez toujours le script setup_runpod.sh** pour l'installation initiale
2. **Configurez Wandb** pour le monitoring à distance
3. **Sauvegardez régulièrement** vos checkpoints
4. **Utilisez TensorRT** pour l'inférence en production
5. **Profilez vos performances** avec le mode profiling activé

---

**Note** : Ce guide est spécifiquement optimisé pour les instances RunPod avec GPU NVIDIA. Pour d'autres environnements, utilisez le script `setup_environment.sh` original.
