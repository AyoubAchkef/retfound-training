# Étapes pour Lancer le Frontend RETFound sur RunPod
====================================================

## 🚨 Problème : Seul le port 8888 (Jupyter) est visible

Si vous ne voyez que le port 8888 dans l'interface RunPod, c'est que le frontend n'est pas encore lancé.

## 🚀 Solution : Lancer le Frontend Étape par Étape

### **Étape 1 : Ouvrir un Terminal dans RunPod**

1. **Via Jupyter Lab (Port 8888)** :
   - Cliquez sur le port 8888 pour ouvrir Jupyter Lab
   - Dans Jupyter Lab, cliquez sur "Terminal" pour ouvrir un terminal
   - Ou allez dans File > New > Terminal

2. **Via SSH** (si configuré) :
   - Utilisez votre client SSH préféré

### **Étape 2 : Naviguer vers le Projet**

```bash
cd /workspace/retfound-training
```

### **Étape 3 : Synchroniser les Dernières Corrections**

```bash
git pull origin main
```

### **Étape 4 : Aller dans le Répertoire Frontend**

```bash
cd retfound/monitoring/frontend
```

### **Étape 5 : Installer les Dépendances**

```bash
npm install
```

### **Étape 6 : Corriger les Permissions (si nécessaire)**

```bash
cd /workspace/retfound-training
chmod +x fix_frontend_permissions_runpod.sh
./fix_frontend_permissions_runpod.sh
```

### **Étape 7 : Retourner au Frontend et Lancer**

```bash
cd retfound/monitoring/frontend
npm run dev:runpod
```

### **Étape 8 : Vérifier que le Frontend Démarre**

Vous devriez voir quelque chose comme :
```
  VITE v6.3.5  ready in 1234 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://0.0.0.0:5173/
  ➜  ready in 1234 ms.
```

### **Étape 9 : Attendre que RunPod Détecte le Port**

- Attendez 10-30 secondes
- Rafraîchissez l'interface web RunPod
- Le port 5173 devrait maintenant apparaître à côté du port 8888

### **Étape 10 : Accéder au Frontend**

- Cliquez sur le nouveau port 5173 dans l'interface RunPod
- Ou utilisez l'URL : `https://[votre-pod-id]-5173.proxy.runpod.net`

## 🔧 Si le Frontend ne Démarre Toujours Pas

### **Option A : Utiliser npx Directement**

```bash
cd /workspace/retfound-training/retfound/monitoring/frontend
npx vite --config vite.config.runpod.ts --host 0.0.0.0 --port 5173
```

### **Option B : Vérifier Node.js**

```bash
node --version
npm --version
```

Si Node.js n'est pas installé :
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### **Option C : Réinstaller Complètement**

```bash
cd /workspace/retfound-training/retfound/monitoring/frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
npm run dev:runpod
```

## 🎯 Commandes Rapides (Copier-Coller)

```bash
# Tout en une fois
cd /workspace/retfound-training && \
git pull origin main && \
cd retfound/monitoring/frontend && \
npm install && \
npm run dev:runpod
```

## 📋 Vérification Finale

1. **Terminal montre** : `Network: http://0.0.0.0:5173/`
2. **Interface RunPod** : Port 5173 apparaît
3. **Accès web** : Dashboard RETFound s'ouvre

## 🚨 Si Rien ne Fonctionne

Utilisez cette commande de diagnostic :

```bash
cd /workspace/retfound-training/retfound/monitoring/frontend
echo "=== Diagnostic Frontend ==="
pwd
ls -la
node --version
npm --version
cat package.json | grep "dev:runpod"
echo "=== Fin Diagnostic ==="
```

Envoyez-moi le résultat pour un diagnostic plus poussé.

## ✅ Résultat Attendu

Une fois le frontend lancé, vous verrez dans l'interface RunPod :
- Port 8888 : Jupyter Lab
- Port 5173 : RETFound Monitoring Dashboard

Et vous pourrez accéder au dashboard de monitoring en temps réel !
