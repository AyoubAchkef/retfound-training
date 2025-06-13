# Guide de Récupération Node.js pour RunPod

## Problème Rencontré

Lors de l'exécution du script `setup_runpod_complete.sh`, vous avez rencontré l'erreur :
```
WARNING: Node.js not found. Installing Node.js...
./scripts/setup_runpod_complete.sh: line 242: sudo: command not found
curl: (23) Failed writing body
```

## Solution Rapide

### Option 1: Utiliser le Script Corrigé

Le script `setup_runpod_complete.sh` a été corrigé pour fonctionner sans `sudo` sur RunPod. Relancez simplement :

```bash
cd /workspace/retfound-training
./scripts/setup_runpod_complete.sh
```

### Option 2: Installation Manuelle de Node.js

Si vous préférez installer Node.js manuellement :

```bash
# 1. Installer Node.js (vous êtes déjà root sur RunPod)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# 2. Vérifier l'installation
node --version
npm --version

# 3. Installer les dépendances frontend
cd /workspace/retfound-training/retfound/monitoring/frontend
npm install
npm run build

# 4. Retourner au répertoire principal
cd /workspace/retfound-training
```

### Option 3: Utiliser le Script Dédié

```bash
cd /workspace/retfound-training
bash install_nodejs_runpod.sh
```

## Vérification de l'Installation

Après l'installation de Node.js, vérifiez que tout fonctionne :

```bash
# Vérifier Node.js
node --version  # Devrait afficher v20.x.x

# Vérifier npm
npm --version   # Devrait afficher 10.x.x

# Vérifier que le frontend est construit
ls -la retfound/monitoring/frontend/dist/
```

## Continuer l'Installation

Une fois Node.js installé, vous pouvez :

1. **Relancer le script complet** (recommandé) :
   ```bash
   ./scripts/setup_runpod_complete.sh
   ```

2. **Ou continuer manuellement** :
   ```bash
   # Activer l'environnement virtuel
   source venv_retfound/bin/activate
   
   # Démarrer le monitoring
   ./start_monitoring.sh
   
   # Dans un autre terminal, démarrer l'entraînement
   ./start_training.sh
   ```

## État de Votre Installation

D'après les logs que vous avez partagés :

✅ **Fonctionnel :**
- Dataset présent (37GB, 211,952 images)
- Poids RETFound téléchargés
- Environnement Python configuré

❌ **À Corriger :**
- Installation Node.js (résolu avec les solutions ci-dessus)

## Commandes de Démarrage Rapide

Une fois Node.js installé :

```bash
# Démarrer tout le stack (monitoring + training)
./start_full_stack.sh

# Ou démarrer seulement le monitoring
./start_monitoring.sh

# Accéder au dashboard
# http://0.0.0.0:8000
```

## Problème de Permissions TypeScript

Si vous rencontrez l'erreur `tsc: Permission denied` après l'installation de Node.js :

```bash
# Solution rapide
bash fix_frontend_permissions.sh

# Ou manuellement
cd retfound/monitoring/frontend
chmod -R 755 node_modules/
chmod +x node_modules/.bin/*
npx tsc && npx vite build
```

## Support

Si vous rencontrez encore des problèmes :

1. **Problème de permissions TypeScript** :
   ```bash
   bash fix_frontend_permissions.sh
   ```

2. **Vérifiez le répertoire** :
   ```bash
   pwd  # Devrait afficher /workspace/retfound-training
   ```

3. **Vérifiez les permissions** :
   ```bash
   whoami  # Devrait afficher 'root'
   ```

4. **Vérifiez l'espace disque** :
   ```bash
   df -h
   ```

5. **Build manuel du frontend** :
   ```bash
   cd retfound/monitoring/frontend
   npm install
   chmod -R 755 node_modules/
   npx vite build --mode production
   ```

Le script corrigé devrait maintenant fonctionner parfaitement sur RunPod !
