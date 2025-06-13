import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://0.0.0.0:8000',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://0.0.0.0:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts'],
          animations: ['framer-motion'],
          utils: ['zustand', 'clsx', 'tailwind-merge'],
        },
      },
    },
  },
  define: {
    // Define environment variables for RunPod
    __RUNPOD_MODE__: true,
    __API_BASE_URL__: JSON.stringify(process.env.VITE_API_URL || 'http://0.0.0.0:8000'),
    __WS_BASE_URL__: JSON.stringify(process.env.VITE_WS_URL || 'ws://0.0.0.0:8000'),
  },
})
