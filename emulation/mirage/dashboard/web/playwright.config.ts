import { defineConfig } from "@playwright/test";

const E2E_ROOT = "/tmp/mirage-e2e";
const DAEMON_PORT = 50051;

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: 0,
  workers: 1,
  fullyParallel: false,
  use: {
    baseURL: `http://localhost:${DAEMON_PORT}`,
    headless: true,
  },
  webServer: [
    {
      command: `bash -c 'set -e; mkdir -p ${E2E_ROOT}/cfg ${E2E_ROOT}/rt; cd ../../.. && cargo build -p mirage_daemon --quiet && ./target/debug/mirage_daemon --mock --http-addr 127.0.0.1:${DAEMON_PORT} --socket ${E2E_ROOT}/rt/mirage.sock --config-root ${E2E_ROOT}/cfg'`,
      port: DAEMON_PORT,
      reuseExistingServer: !process.env.CI,
      timeout: 240_000,
    },
  ],
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});
