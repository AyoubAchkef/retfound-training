#!/bin/bash

# RETFound Frontend Fix Script
# This script fixes common frontend build issues by cleaning and reinstalling dependencies

echo "🔧 RETFound Frontend Fix Script"
echo "================================"

# Navigate to frontend directory
FRONTEND_DIR="retfound/monitoring/frontend"

if [ ! -d "$FRONTEND_DIR" ]; then
    echo "❌ Frontend directory not found: $FRONTEND_DIR"
    exit 1
fi

cd "$FRONTEND_DIR"

echo "📁 Working in: $(pwd)"

# Step 1: Clean existing installations
echo "🧹 Cleaning existing node_modules and package-lock.json..."
rm -rf node_modules package-lock.json 2>/dev/null || true

# Step 2: Clear npm cache
echo "🗑️  Clearing npm cache..."
npm cache clean --force 2>/dev/null || true

# Step 3: Install dependencies
echo "📦 Installing dependencies..."
if npm install; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Step 4: Fix permissions (for Unix-like systems)
if [ "$(uname)" != "MINGW64_NT"* ] && [ "$(uname)" != "MSYS_NT"* ]; then
    echo "🔐 Fixing permissions..."
    chmod -R 755 node_modules/ 2>/dev/null || true
    chmod +x node_modules/.bin/* 2>/dev/null || true
fi

# Step 5: Test TypeScript compilation
echo "🔍 Testing TypeScript compilation..."
if npx tsc --noEmit; then
    echo "✅ TypeScript compilation successful"
else
    echo "❌ TypeScript compilation failed"
    exit 1
fi

# Step 6: Test build
echo "🏗️  Testing build process..."
if npm run build; then
    echo "✅ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "🎉 Frontend fix completed successfully!"
echo ""
echo "Available commands:"
echo "  npm run dev     - Start development server"
echo "  npm run build   - Build for production"
echo "  npm run preview - Preview production build"
echo ""
