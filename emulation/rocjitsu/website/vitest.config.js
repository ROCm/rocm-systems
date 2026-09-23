import { defineConfig } from 'vitest/config';

// Unit tests exercise plain data modules, so they need neither the React plugin nor a DOM.
export default defineConfig({
  test: {
    include: ['tests/unit/**/*.test.js'],
    exclude: ['tests/e2e/**'],
    environment: 'node',
    restoreMocks: true,
  },
});
