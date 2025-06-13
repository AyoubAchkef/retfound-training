#!/bin/bash

echo "🔧 Fixing frontend permissions for RunPod..."

# Navigate to frontend directory
cd /workspace/retfound-training/retfound/monitoring/frontend

# Fix permissions for node_modules
echo "📁 Fixing node_modules permissions..."
chmod -R 755 node_modules/ 2>/dev/null || true

# Fix permissions for vite binary
echo "⚡ Fixing vite binary permissions..."
chmod +x node_modules/.bin/vite 2>/dev/null || true
chmod +x node_modules/.bin/* 2>/dev/null || true

# Fix permissions for all executables
echo "🔧 Fixing all binary permissions..."
find node_modules/.bin -type f -exec chmod +x {} \; 2>/dev/null || true

# Alternative: use npx instead of direct vite call
echo "🚀 Testing vite with npx..."
npx vite --version

echo "✅ Frontend permissions fixed!"
echo ""
echo "🎯 Now you can run:"
echo "   npm run dev:runpod"
echo "   or"
echo "   npx vite --config vite.config.runpod.ts --host 0.0.0.0 --port 5173"
