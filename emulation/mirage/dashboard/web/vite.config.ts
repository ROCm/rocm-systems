/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5174',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  test: {
    exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
  },
})
