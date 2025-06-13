# Instructions de Déploiement sur RunPod

## 🚀 Étapes pour Déployer sur RunPod

### 1. Connexion SSH au RunPod
```bash
ssh root@216.81.245.138 -p 17040 -i ~/.ssh/id_ed25519
```

### 2. Navigation vers le Workspace
```bash
cd /workspace
```

### 3. Clonage ou Mise à Jour du Repository

#### Si c'est la première fois (nouveau clone) :
```bash
git clone https://github.com/AyoubAchkef/retfound-training.git
cd retfound-training
```

#### Si le repository existe déjà (mise à jour) :
```bash
cd retfound-training
git pull origin main
```

### 4. Vérification des Nouveaux Fichiers
```bash
ls -la
# Vous devriez voir les nouveaux fichiers :
# - .env.runpod
# - RUNPOD_TRAINING_GUIDE.md
# - FRONTEND_BACKEND_INTEGRATION.md
# - SETUP_VALIDATION_SUMMARY.md
# - quick_start_runpod.sh
# - scripts/setup_runpod_complete.sh
# - scripts/validate_setup.py
# - retfound/monitoring/frontend/.env.runpod
# - retfound/monitoring/frontend/vite.config.runpod.ts
```

### 5. Rendre les Scripts Exécutables
```bash
chmod +x scripts/setup_runpod_complete.sh
chmod +x scripts/validate_setup.py
chmod +x quick_start_runpod.sh
```

### 6. Lancement de l'Installation Complète
```bash
./scripts/setup_runpod_complete.sh
```

### 7. Validation du Setup
```bash
python scripts/validate_setup.py
```

### 8. Lancement Rapide
```bash
./quick_start_runpod.sh
```

## 🎯 Accès aux Interfaces

Une fois le setup terminé, vous pourrez accéder à :

- **Dashboard de Monitoring** : `http://[RUNPOD_IP]:8000`
- **API Documentation** : `http://[RUNPOD_IP]:8000/docs`
- **WebSocket** : `ws://[RUNPOD_IP]:8000/ws`
- **Health Check** : `http://[RUNPOD_IP]:8000/health`

## 📋 Checklist de Déploiement

- [ ] Connexion SSH réussie
- [ ] Repository cloné/mis à jour
- [ ] Scripts rendus exécutables
- [ ] Installation complète exécutée
- [ ] Validation du setup réussie
- [ ] Dataset monté dans `/workspace/datasets/DATASET_CLASSIFICATION`
- [ ] Serveur de monitoring démarré
- [ ] Interface web accessible

## 🔧 Commandes Utiles

### Vérifier l'État du Repository
```bash
git status
git log --oneline -5
```

### Vérifier les Services
```bash
# Vérifier si le serveur de monitoring fonctionne
curl http://localhost:8000/health

# Vérifier les processus Python
ps aux | grep python

# Vérifier l'utilisation GPU
nvidia-smi
```

### Logs de Débogage
```bash
# Logs du serveur de monitoring
tail -f /workspace/logs/monitoring.log

# Logs d'entraînement
tail -f /workspace/logs/training.log
```

## 🎉 Prêt pour l'Entraînement !

Une fois toutes ces étapes terminées, votre environnement RETFound sera complètement configuré avec :

✅ **Backend FastAPI** avec WebSocket  
✅ **Frontend React/TypeScript** pour monitoring  
✅ **Support des 28 classes** CAASI v6.1  
✅ **Monitoring des conditions critiques**  
✅ **Configuration optimisée** pour A100  

Vous pourrez alors lancer l'entraînement avec monitoring temps réel complet !
