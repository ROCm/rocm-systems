import { defineConfig, devices } from '@playwright/test';

// The web server builds before it previews, so a reused server can serve a build from other
// sources. Reuse therefore stays opt-in for local iteration and never applies in CI.
const reuseExistingServer = !process.env.CI && process.env.PLAYWRIGHT_REUSE_EXISTING_SERVER === '1';
const port = Number(process.env.PLAYWRIGHT_PORT || 4174);
if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error('PLAYWRIGHT_PORT must be an integer between 1 and 65535');
}
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: './tests/e2e',
  testMatch: /.*\.e2e\.js$/,
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  reporter: 'line',
  use: {
    baseURL,
    trace: 'retain-on-failure',
  },
  // Rendered behavior is asserted once under desktop; the mobile project only replays the
  // layout and navigation assertions that actually depend on a phone viewport.
  projects: [
    {
      name: 'desktop',
      testIgnore: /mobile\.e2e\.js$/,
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
    },
    {
      name: 'mobile',
      testMatch: /mobile\.e2e\.js$/,
      use: { ...devices['iPhone 13'], browserName: 'chromium' },
    },
  ],
  webServer: {
    command: `npm run build -- --mode fixtures && npm run preview -- --mode fixtures --host 127.0.0.1 --port ${port}`,
    url: baseURL,
    reuseExistingServer,
  },
});
