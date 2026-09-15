import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { writeDashboardFixtureSite } from './tests/fixtures/dashboardFixture.js';

export default defineConfig(({ mode }) => ({
  base: './',
  // Dummy benchmark history is available only for explicit fixture builds/dev servers.
  publicDir: mode === 'fixtures' ? writeDashboardFixtureSite() : false,
  plugins: [react()],
  preview: {
    port: 4173,
    strictPort: true,
  },
  build: {
    outDir: mode === 'fixtures' ? '.test-dist' : 'dist',
    sourcemap: true,
    target: 'es2022',
    rollupOptions: {
      output: {
        manualChunks(moduleId) {
          if (moduleId.includes('/node_modules/zrender/')) return 'chart-renderer';
          if (moduleId.includes('/node_modules/echarts/')) return 'charts';
          if (moduleId.includes('/node_modules/@mui/') || moduleId.includes('/node_modules/@emotion/')) return 'ui';
          if (moduleId.includes('/node_modules/react')) return 'react-vendor';
          return undefined;
        },
      },
    },
  },
}));
