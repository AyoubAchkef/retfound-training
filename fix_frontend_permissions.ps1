# RETFound Frontend Fix Script (PowerShell)
# This script fixes common frontend build issues by cleaning and reinstalling dependencies

Write-Host "🔧 RETFound Frontend Fix Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Navigate to frontend directory
$FRONTEND_DIR = "retfound/monitoring/frontend"

if (-not (Test-Path $FRONTEND_DIR)) {
    Write-Host "❌ Frontend directory not found: $FRONTEND_DIR" -ForegroundColor Red
    exit 1
}

Set-Location $FRONTEND_DIR
Write-Host "📁 Working in: $(Get-Location)" -ForegroundColor Yellow

# Step 1: Clean existing installations
Write-Host "🧹 Cleaning existing node_modules and package-lock.json..." -ForegroundColor Yellow
Remove-Item -Recurse -Force node_modules, package-lock.json -ErrorAction SilentlyContinue

# Step 2: Clear npm cache
Write-Host "🗑️  Clearing npm cache..." -ForegroundColor Yellow
try {
    npm cache clean --force
} catch {
    Write-Host "Warning: Could not clear npm cache" -ForegroundColor Yellow
}

# Step 3: Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
$installResult = npm install
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Step 4: Test TypeScript compilation
Write-Host "🔍 Testing TypeScript compilation..." -ForegroundColor Yellow
$tscResult = npx tsc --noEmit
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ TypeScript compilation successful" -ForegroundColor Green
} else {
    Write-Host "❌ TypeScript compilation failed" -ForegroundColor Red
    exit 1
}

# Step 5: Test build
Write-Host "🏗️  Testing build process..." -ForegroundColor Yellow
$buildResult = npm run build
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful" -ForegroundColor Green
} else {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 Frontend fix completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  npm run dev     - Start development server" -ForegroundColor White
Write-Host "  npm run build   - Build for production" -ForegroundColor White
Write-Host "  npm run preview - Preview production build" -ForegroundColor White
Write-Host ""
