# Commandes pour Fixer le Frontend sur RunPod

## ⚠️ Important : RunPod utilise Linux, pas Windows !

Sur RunPod, vous devez utiliser les commandes **Linux/Unix**, même si vous développez depuis Windows.

## 🚀 Commandes Rapides pour RunPod

### Option 1 : Script Automatique (Recommandé)
```bash
# Donner les permissions d'exécution au script
chmod +x fix_frontend_permissions.sh

# Lancer le script de fix automatique
./fix_frontend_permissions.sh
```

### Option 2 : Commandes Manuelles
```bash
# Aller dans le dossier frontend
cd retfound/monitoring/frontend

# Nettoyer les dépendances corrompues
rm -rf node_modules package-lock.json

# Vider le cache npm
npm cache clean --force

# Réinstaller toutes les dépendances
npm install

# Tester la compilation TypeScript
npx tsc --noEmit

# Tester le build de production
npm run build
```

## 🔧 Commandes de Développement

### Lancer le serveur de développement
```bash
cd retfound/monitoring/frontend
npm run dev
```
Le serveur sera accessible sur `http://localhost:3000/`

### Build de production
```bash
cd retfound/monitoring/frontend
npm run build
```

### Prévisualiser le build de production
```bash
cd retfound/monitoring/frontend
npm run preview
```

## 🐛 Dépannage

### Vérifier l'installation Node.js
```bash
node --version    # Doit être >= 18.0.0
npm --version     # Doit être >= 8.0.0
```

### Vérifier l'état des dépendances
```bash
cd retfound/monitoring/frontend
npm ls            # Lister les packages installés
npm audit         # Vérifier les vulnérabilités
```

### En cas de problème persistant
```bash
# Supprimer complètement node_modules et recommencer
cd retfound/monitoring/frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

## 📝 Notes Importantes

1. **Ne pas utiliser** `.\fix_frontend_permissions.ps1` sur RunPod (c'est pour Windows)
2. **Utiliser** `./fix_frontend_permissions.sh` sur RunPod (Linux)
3. Les commandes PowerShell ne fonctionnent pas sur RunPod
4. Toujours utiliser `/` (slash) au lieu de `\` (backslash) pour les chemins

## 🎯 Commande Complète pour RunPod

```bash
# Commande tout-en-un pour fixer le frontend sur RunPod
cd /workspace/retfound-training && chmod +x fix_frontend_permissions.sh && ./fix_frontend_permissions.sh
```

Cette commande :
1. Va dans le répertoire du projet
2. Donne les permissions d'exécution au script
3. Lance le script de fix automatique

## ✅ Vérification du Succès

Après avoir lancé le fix, vous devriez voir :
- ✅ Dependencies installed successfully
- ✅ TypeScript compilation successful  
- ✅ Build successful
- 🎉 Frontend fix completed successfully!
