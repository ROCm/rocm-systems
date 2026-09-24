import { defineConfig, devices } from '@playwright/test';

const port = Number(process.env.PLAYWRIGHT_PRODUCTION_PORT || 4175);
if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error('PLAYWRIGHT_PRODUCTION_PORT must be an integer between 1 and 65535');
}
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: './tests/production',
  testMatch: /.*\.e2e\.js$/,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  reporter: 'line',
  timeout: 120_000,
  expect: {
    timeout: 90_000,
  },
  use: {
    ...devices['Desktop Chrome'],
    baseURL,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `npm run build && npm run preview -- --host 127.0.0.1 --port ${port}`,
    url: baseURL,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
