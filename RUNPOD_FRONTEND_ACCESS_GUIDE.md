# Guide d'Accès au Frontend RETFound sur RunPod
=====================================================

## 🌐 Comment Accéder au Frontend de Monitoring

### ❌ Problème Courant
Si vous voyez "ERR_CONNECTION_REFUSED" sur `http://localhost:5173/`, c'est normal ! 
Le frontend tourne sur RunPod, pas sur votre machine locale.

### ✅ Solution : Accès via RunPod Web Interface

#### **Méthode 1 : Via l'Interface Web RunPod (Recommandée)**

1. **Aller dans votre Pod RunPod**
   - Connectez-vous à https://runpod.io
   - Allez dans vos Pods actifs
   - Cliquez sur votre Pod RETFound

2. **Ouvrir l'Interface Web**
   - Cliquez sur "Connect" 
   - Sélectionnez "Connect via Web Terminal" ou "HTTP Service"
   - Vous verrez une interface web avec des ports disponibles

3. **Accéder au Port 5173**
   - Cherchez le port 5173 dans la liste
   - Cliquez sur le lien du port 5173
   - Ou utilisez l'URL : `https://[votre-pod-id]-5173.proxy.runpod.net`

#### **Méthode 2 : Via l'URL Directe RunPod**

L'URL sera quelque chose comme :
```
https://[pod-id]-5173.proxy.runpod.net
```

Où `[pod-id]` est l'ID unique de votre Pod RunPod.

### 🚀 Étapes Complètes pour Lancer le Frontend

```bash
# 1. Dans le terminal RunPod, aller au frontend
cd /workspace/retfound-training/retfound/monitoring/frontend

# 2. Vérifier que les dépendances sont installées
npm install

# 3. Lancer le frontend avec la config RunPod
npm run dev:runpod

# 4. Le frontend sera accessible via l'interface web RunPod sur le port 5173
```

### 📋 Vérification que le Frontend Fonctionne

Vous devriez voir dans le terminal :
```
  ➜  Local:   http://localhost:5173/
  ➜  Network: http://0.0.0.0:5173/
  ➜  press h to show help
```

### 🔧 Si le Frontend ne Démarre Pas

1. **Corriger les permissions :**
```bash
cd /workspace/retfound-training
chmod +x fix_frontend_permissions_runpod.sh
./fix_frontend_permissions_runpod.sh
```

2. **Réinstaller les dépendances :**
```bash
cd retfound/monitoring/frontend
rm -rf node_modules package-lock.json
npm install
```

3. **Utiliser npx directement :**
```bash
npx vite --config vite.config.runpod.ts --host 0.0.0.0 --port 5173
```

### 🌐 Configuration RunPod Spécifique

Le frontend est configuré pour RunPod avec :
- **Host** : `0.0.0.0` (accessible depuis l'extérieur)
- **Port** : `5173`
- **Config** : `vite.config.runpod.ts`
- **Env** : `.env.runpod`

### 📊 Fonctionnalités du Dashboard

Une fois accessible, vous verrez :
- 📈 **Métriques en temps réel** : Loss, Accuracy, AUC
- 🖥️ **Stats GPU** : Utilisation, mémoire, température
- 📊 **Graphiques d'entraînement** : Courbes de loss et accuracy
- 🎯 **Performance par classe** : Métriques détaillées
- ⚠️ **Alertes critiques** : Notifications importantes
- 📋 **Détails des époques** : Progression détaillée

### 🔍 Dépannage

#### Si le port 5173 n'apparaît pas :
1. Vérifiez que le frontend tourne bien
2. Attendez quelques secondes pour que RunPod détecte le port
3. Rafraîchissez l'interface web RunPod

#### Si vous voyez une page blanche :
1. Vérifiez la console du navigateur (F12)
2. Assurez-vous que le backend de monitoring tourne aussi
3. Vérifiez les logs du frontend dans le terminal

### 🎯 Commandes Rapides

```bash
# Lancer le frontend
cd /workspace/retfound-training/retfound/monitoring/frontend && npm run dev:runpod

# Lancer le backend (dans un autre terminal)
cd /workspace/retfound-training && python -m retfound.monitoring.server

# Lancer l'entraînement avec monitoring
python -m retfound.cli train --config configs/runpod.yaml --weights oct --modality oct --monitor
```

### ✅ Résultat Final

Vous devriez pouvoir accéder au dashboard de monitoring RETFound via l'interface web RunPod et voir les métriques d'entraînement en temps réel !

---

**Note** : Le frontend ne sera jamais accessible sur `localhost` depuis votre machine locale car il tourne sur RunPod. Utilisez toujours l'interface web RunPod pour y accéder.
