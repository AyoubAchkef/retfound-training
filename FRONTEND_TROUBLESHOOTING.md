# Frontend Troubleshooting Guide

## Issue: Frontend Build Failures

### Problem Description
The RETFound monitoring frontend was experiencing build failures with the following errors:
- `Error: Cannot find module '../lib/tsc.js'`
- `Cannot find module '/workspace/retfound-training/retfound/monitoring/frontend/node_modules/vite/dist/node/cli.js'`
- Permission denied errors when running scripts

### Root Cause
The `node_modules` directory was corrupted or incomplete, missing critical files for TypeScript and Vite. This commonly happens when:
- Dependencies are installed with insufficient permissions
- Network interruptions during installation
- Disk space issues during installation
- Version conflicts between packages

### Solution Applied

#### 1. Complete Dependency Reinstallation
```bash
# Remove corrupted files
cd retfound/monitoring/frontend
rm -rf node_modules package-lock.json

# Clear npm cache
npm cache clean --force

# Reinstall all dependencies
npm install
```

#### 2. Verification Steps
```bash
# Test TypeScript compilation
npx tsc --noEmit

# Test production build
npm run build

# Test development server
npm run dev
```

### Results
- ✅ TypeScript compilation successful
- ✅ Production build successful (built in 3.74s)
- ✅ Development server running on http://localhost:3000/
- ✅ All 376 packages installed with 0 vulnerabilities

### Build Output Summary
```
dist/index.html                       3.30 kB │ gzip:  1.32 kB
dist/assets/index-MX6bxRy_.css        4.49 kB │ gzip:  1.33 kB
dist/assets/charts--AtWguYJ.js        0.13 kB │ gzip:  0.14 kB
dist/assets/index-B1o8q7sg.js        43.00 kB │ gzip: 12.48 kB
dist/assets/animations-D8_t_R-t.js   98.63 kB │ gzip: 33.21 kB
dist/assets/vendor-BtP0CW_r.js      141.78 kB │ gzip: 45.52 kB
```

## Automated Fix Scripts

### For Unix/Linux/macOS
```bash
./fix_frontend_permissions.sh
```

### For Windows PowerShell
```powershell
.\fix_frontend_permissions.ps1
```

Both scripts perform the same operations:
1. Clean existing node_modules and package-lock.json
2. Clear npm cache
3. Reinstall dependencies
4. Test TypeScript compilation
5. Test production build
6. Provide status feedback

## Prevention Tips

### 1. Regular Maintenance
```bash
# Periodically clean and reinstall dependencies
npm ci  # Use for production/CI environments
npm install  # Use for development
```

### 2. Check Node.js Version
```bash
node --version  # Should be >= 18.0.0
npm --version   # Should be >= 8.0.0
```

### 3. Monitor Disk Space
Ensure sufficient disk space before running `npm install`:
```bash
df -h  # Unix/Linux/macOS
dir    # Windows
```

### 4. Network Stability
For unstable networks, consider:
```bash
npm config set registry https://registry.npmjs.org/
npm config set timeout 60000
```

## Common Commands

### Development
```bash
cd retfound/monitoring/frontend
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run type-check   # Check TypeScript without emitting
npm run lint         # Run ESLint
```

### Troubleshooting
```bash
npm ls               # List installed packages
npm outdated         # Check for outdated packages
npm audit            # Check for vulnerabilities
npm audit fix        # Fix vulnerabilities
```

## Dependencies Overview

### Runtime Dependencies
- React 18.2.0 - UI framework
- TypeScript 5.2.2 - Type safety
- Vite 6.3.5 - Build tool and dev server
- Tailwind CSS 3.3.5 - Styling
- Recharts 2.8.0 - Charts and visualization
- Zustand 4.4.0 - State management
- Framer Motion 10.16.0 - Animations

### Key Features
- Real-time monitoring dashboard
- WebSocket integration for live updates
- Responsive design with Tailwind CSS
- Interactive charts with Recharts
- Type-safe development with TypeScript
- Hot module replacement in development

## Support

If you encounter issues not covered in this guide:
1. Check the console for specific error messages
2. Verify Node.js and npm versions
3. Try the automated fix scripts
4. Clear browser cache if frontend issues persist
5. Check network connectivity for dependency downloads

Last updated: December 13, 2025
