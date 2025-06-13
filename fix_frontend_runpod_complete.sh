#!/bin/bash

echo "🔧 RÉPARATION COMPLÈTE FRONTEND RETFOUND SUR RUNPOD"
echo "=================================================="

# Fonction pour vérifier le succès d'une commande
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1"
        return 1
    fi
}

# Aller dans le répertoire du projet
echo "📁 Navigation vers le projet..."
cd /workspace/retfound-training
check_success "Navigation vers /workspace/retfound-training"

# Synchroniser avec Git
echo ""
echo "🔄 Synchronisation Git..."
git pull origin main
check_success "Git pull"

# Installer Node.js si nécessaire
echo ""
echo "🟢 Vérification Node.js..."
if ! command -v node &> /dev/null; then
    echo "📦 Installation de Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
    check_success "Installation Node.js"
else
    echo "✅ Node.js déjà installé: $(node --version)"
fi

# Aller dans le répertoire frontend
echo ""
echo "🌐 Navigation vers le frontend..."
cd retfound/monitoring/frontend
check_success "Navigation vers frontend"

# Nettoyer complètement
echo ""
echo "🧹 Nettoyage complet..."
rm -rf node_modules package-lock.json
check_success "Suppression node_modules et package-lock.json"

npm cache clean --force
check_success "Nettoyage cache npm"

# Installer les dépendances
echo ""
echo "📦 Installation des dépendances..."
npm install
check_success "npm install"

# Corriger les permissions
echo ""
echo "🔐 Correction des permissions..."
chmod -R 755 node_modules/ 2>/dev/null
chmod +x node_modules/.bin/* 2>/dev/null
check_success "Correction permissions"

# Installer vite globalement si nécessaire
echo ""
echo "⚡ Vérification Vite..."
if ! npx vite --version &> /dev/null; then
    echo "📦 Installation globale de Vite..."
    npm install -g vite
    check_success "Installation globale Vite"
else
    echo "✅ Vite fonctionne: $(npx vite --version)"
fi

# Test de la configuration
echo ""
echo "🧪 Test de la configuration..."
if [ -f "vite.config.runpod.ts" ]; then
    echo "✅ Configuration RunPod trouvée"
else
    echo "❌ Configuration RunPod manquante"
fi

if [ -f ".env.runpod" ]; then
    echo "✅ Environnement RunPod trouvé"
else
    echo "❌ Environnement RunPod manquant"
fi

# Créer un script de lancement simple
echo ""
echo "📝 Création script de lancement..."
cat > launch_frontend.sh << 'EOF'
#!/bin/bash
echo "🚀 Lancement du frontend RETFound..."
cd /workspace/retfound-training/retfound/monitoring/frontend
echo "📍 Répertoire: $(pwd)"
echo "🔧 Commande: npx vite --config vite.config.runpod.ts --host 0.0.0.0 --port 5173"
npx vite --config vite.config.runpod.ts --host 0.0.0.0 --port 5173
EOF

chmod +x launch_frontend.sh
check_success "Création script de lancement"

# Test final
echo ""
echo "🎯 Test final..."
echo "Test npx vite --version:"
npx vite --version
echo ""
echo "Test npm run dev:runpod (5 secondes):"
timeout 5s npm run dev:runpod || echo "Test terminé (normal)"

echo ""
echo "✅ RÉPARATION TERMINÉE"
echo "===================="
echo ""
echo "🚀 COMMANDES POUR LANCER LE FRONTEND:"
echo ""
echo "Option 1 (Recommandée):"
echo "cd /workspace/retfound-training/retfound/monitoring/frontend"
echo "./launch_frontend.sh"
echo ""
echo "Option 2:"
echo "cd /workspace/retfound-training/retfound/monitoring/frontend"
echo "npm run dev:runpod"
echo ""
echo "Option 3:"
echo "cd /workspace/retfound-training/retfound/monitoring/frontend"
echo "npx vite --config vite.config.runpod.ts --host 0.0.0.0 --port 5173"
echo ""
echo "📋 APRÈS LE LANCEMENT:"
echo "1. Attendez 10-30 secondes"
echo "2. Rafraîchissez l'interface RunPod"
echo "3. Le port 5173 devrait apparaître"
echo "4. Cliquez sur le port 5173 pour accéder au dashboard"
echo ""
echo "🔍 EN CAS DE PROBLÈME:"
echo "bash debug_frontend_runpod.sh"
