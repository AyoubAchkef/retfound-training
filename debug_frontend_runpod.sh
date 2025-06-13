#!/bin/bash

echo "🔍 DIAGNOSTIC FRONTEND RETFOUND SUR RUNPOD"
echo "=========================================="

# Vérifier l'environnement
echo ""
echo "📋 ENVIRONNEMENT:"
echo "Current directory: $(pwd)"
echo "User: $(whoami)"
echo "OS: $(uname -a)"

# Vérifier Node.js et npm
echo ""
echo "🟢 NODE.JS ET NPM:"
if command -v node &> /dev/null; then
    echo "✅ Node.js version: $(node --version)"
else
    echo "❌ Node.js NOT FOUND"
fi

if command -v npm &> /dev/null; then
    echo "✅ npm version: $(npm --version)"
else
    echo "❌ npm NOT FOUND"
fi

# Vérifier le projet
echo ""
echo "📁 PROJET RETFOUND:"
if [ -d "/workspace/retfound-training" ]; then
    echo "✅ Projet trouvé: /workspace/retfound-training"
    cd /workspace/retfound-training
    echo "Git status:"
    git status --porcelain
    echo "Git branch: $(git branch --show-current)"
    echo "Last commit: $(git log -1 --oneline)"
else
    echo "❌ Projet NOT FOUND: /workspace/retfound-training"
    exit 1
fi

# Vérifier le frontend
echo ""
echo "🌐 FRONTEND:"
if [ -d "retfound/monitoring/frontend" ]; then
    echo "✅ Répertoire frontend trouvé"
    cd retfound/monitoring/frontend
    echo "Contenu du répertoire:"
    ls -la
    
    if [ -f "package.json" ]; then
        echo "✅ package.json trouvé"
        echo "Scripts disponibles:"
        cat package.json | grep -A 10 '"scripts"'
    else
        echo "❌ package.json NOT FOUND"
    fi
    
    if [ -f "vite.config.runpod.ts" ]; then
        echo "✅ vite.config.runpod.ts trouvé"
    else
        echo "❌ vite.config.runpod.ts NOT FOUND"
    fi
    
    if [ -d "node_modules" ]; then
        echo "✅ node_modules trouvé"
        echo "Taille node_modules: $(du -sh node_modules 2>/dev/null || echo 'Erreur calcul taille')"
    else
        echo "❌ node_modules NOT FOUND"
    fi
    
    if [ -f "node_modules/.bin/vite" ]; then
        echo "✅ vite binary trouvé"
        echo "Permissions vite: $(ls -la node_modules/.bin/vite)"
    else
        echo "❌ vite binary NOT FOUND"
    fi
    
else
    echo "❌ Répertoire frontend NOT FOUND"
    exit 1
fi

# Test des commandes
echo ""
echo "🧪 TESTS:"

echo "Test 1: npm --version"
npm --version 2>&1

echo ""
echo "Test 2: npx --version"
npx --version 2>&1

echo ""
echo "Test 3: Test vite direct"
if [ -f "node_modules/.bin/vite" ]; then
    ./node_modules/.bin/vite --version 2>&1
else
    echo "vite binary non trouvé"
fi

echo ""
echo "Test 4: Test npx vite"
npx vite --version 2>&1

echo ""
echo "Test 5: Test npm run dev:runpod (dry run)"
npm run dev:runpod --dry-run 2>&1 || echo "Commande échouée"

# Vérifier les ports
echo ""
echo "🔌 PORTS:"
echo "Ports en écoute:"
netstat -tlnp 2>/dev/null | grep LISTEN || ss -tlnp | grep LISTEN

echo ""
echo "🔍 DIAGNOSTIC TERMINÉ"
echo "===================="
echo ""
echo "📋 ACTIONS RECOMMANDÉES:"
echo "1. Si Node.js manque: installer Node.js"
echo "2. Si node_modules manque: npm install"
echo "3. Si permissions incorrectes: chmod +x node_modules/.bin/*"
echo "4. Si vite ne fonctionne pas: npm install -g vite"
echo ""
echo "🚀 COMMANDE DE RÉPARATION AUTOMATIQUE:"
echo "bash fix_frontend_runpod_complete.sh"
