# ✅ Vérification Simple pour RunPod

## 🎯 Vous êtes sur RunPod - Pas besoin d'environnement virtuel !

Sur RunPod, Python et les dépendances de base sont déjà installées. Voici comment vérifier que tout fonctionne.

## 📍 Étape 1 : Vérifier votre Position
```bash
# Vérifier où vous êtes
pwd

# Aller dans le dossier du projet (si pas déjà fait)
cd /workspace/retfound-training
```

## 🔍 Étape 2 : Vérifications Rapides

### Vérifier Python et PyTorch
```bash
# Vérifier Python
python --version

# Vérifier PyTorch
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"

# Vérifier CUDA
python -c "import torch; print(f'✅ CUDA disponible: {torch.cuda.is_available()}')"

# Vérifier combien de GPU
python -c "import torch; print(f'✅ Nombre de GPU: {torch.cuda.device_count()}')"
```

### Vérifier les Dépendances RETFound
```bash
# Tester l'import du modèle
python -c "from retfound.models import RETFoundModel; print('✅ Modèle RETFound OK')"

# Tester le CLI
python -m retfound.cli --help
```

### Vérifier le Frontend
```bash
# Aller dans le frontend
cd retfound/monitoring/frontend

# Vérifier que npm fonctionne
npm --version

# Vérifier que le build fonctionne
npm run build

# Retourner à la racine
cd ../../..
```

## 🚀 Étape 3 : Test Complet du Système

### Test du Monitoring
```bash
# Tester le serveur de monitoring (arrêter avec Ctrl+C après quelques secondes)
python retfound/monitoring/demo.py
```

### Test du CLI RETFound
```bash
# Tester les commandes disponibles
python -m retfound.cli --help
python -m retfound.cli train --help
```

## ✅ Si Tout Fonctionne - Lancer l'Entraînement

### Option 1 : Entraînement Simple
```bash
python -m retfound.cli train --config configs/default.yaml
```

### Option 2 : Entraînement avec Monitoring
```bash
# Terminal 1 : Lancer le serveur de monitoring
python retfound/monitoring/server.py

# Terminal 2 : Lancer l'entraînement
python -m retfound.cli train --config configs/default.yaml --monitor

# Ouvrir dans votre navigateur : http://localhost:8000
```

### Option 3 : Configuration RunPod Optimisée
```bash
python -m retfound.cli train --config configs/runpod.yaml --monitor
```

## 🛠️ Si Quelque Chose Ne Fonctionne Pas

### Problème avec les Dépendances Python
```bash
pip install -r requirements.txt
```

### Problème avec le Frontend
```bash
./fix_frontend_permissions.sh
```

### Problème avec CUDA
```bash
nvidia-smi
```

## 📋 Checklist Rapide

Cochez chaque élément :

- [ ] `python --version` fonctionne
- [ ] `python -c "import torch; print(torch.__version__)"` fonctionne
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` retourne `True`
- [ ] `python -m retfound.cli --help` affiche l'aide
- [ ] `python -c "from retfound.models import RETFoundModel; print('OK')"` fonctionne
- [ ] `cd retfound/monitoring/frontend && npm run build` fonctionne
- [ ] `python retfound/monitoring/demo.py` se lance sans erreur

## 🎯 Commande de Vérification Tout-en-Un

```bash
echo "=== Vérification RETFound sur RunPod ==="
echo "1. Python:" && python --version
echo "2. PyTorch:" && python -c "import torch; print(torch.__version__)"
echo "3. CUDA:" && python -c "import torch; print('Disponible:', torch.cuda.is_available())"
echo "4. GPU Count:" && python -c "import torch; print('Nombre:', torch.cuda.device_count())"
echo "5. RETFound Model:" && python -c "from retfound.models import RETFoundModel; print('✅ OK')"
echo "6. CLI:" && python -m retfound.cli --help > /dev/null && echo "✅ CLI OK"
echo "7. Frontend:" && cd retfound/monitoring/frontend && npm run build > /dev/null && echo "✅ Frontend OK" && cd ../../..
echo "=== Vérification Terminée ==="
```

## 🚀 Si Tout est ✅ - Lancez l'Entraînement !

```bash
# Entraînement avec monitoring complet
python retfound/monitoring/server.py &
python -m retfound.cli train --config configs/runpod.yaml --monitor
```

---

**Pas d'environnement virtuel nécessaire sur RunPod - tout est déjà configuré !**
