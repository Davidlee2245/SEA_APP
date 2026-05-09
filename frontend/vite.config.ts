import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Long timeouts for endpoints that stream or run heavy work (e.g. Cygnus pipeline).
      '/api/cygnus/run': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        timeout: 60 * 60 * 1000, // 1 hour
        proxyTimeout: 60 * 60 * 1000,
      },
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
  base: './',          // relative paths so file:// protocol works in Electron
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});


