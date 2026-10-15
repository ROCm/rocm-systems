import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import {
  resolveDashboardDataDirectory,
  serveDashboardData,
} from './scripts/dashboard-data-dir.mjs';
import { resolveDataBaseUrl } from './scripts/dashboard-data-source.mjs';

const fixtureDirectory = fileURLToPath(new URL('./tests/fixtures/', import.meta.url));
const localDataDirectory = process.env.DASHBOARD_DATA_DIR
  ? resolveDashboardDataDirectory(process.env.DASHBOARD_DATA_DIR)
  : null;

export default defineConfig(({ mode }) => {
  // The client and the index.html preload links read this through import.meta.env.
  process.env.VITE_DASHBOARD_DATA_BASE_URL = resolveDataBaseUrl(mode);

  return {
    base: './',
    // Dummy benchmark history is available only for explicit fixture builds/dev servers.
    // A local data directory is mounted at /data/ instead of copying into publicDir.
    publicDir: mode === 'fixtures' && !localDataDirectory ? fixtureDirectory : false,
    plugins: [
      react(),
      localDataDirectory ? serveDashboardData(localDataDirectory) : null,
    ].filter(Boolean),
    preview: {
      port: mode === 'fixtures' ? 4174 : 4173,
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
  };
});
