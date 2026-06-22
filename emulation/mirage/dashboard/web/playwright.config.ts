import { defineConfig } from "@playwright/test";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const E2E_ROOT = "/tmp/mirage-e2e";
const DAEMON_PORT = 50051;
const REPO_ROOT = path.resolve(__dirname, "../../");
const MIRAGE_BIN = path.join(REPO_ROOT, "target", "debug", "mirage");

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  workers: 1,
  fullyParallel: false,
  use: {
    baseURL: `http://127.0.0.1:${DAEMON_PORT}`,
    headless: true,
  },
  webServer: [
    {
      command: `bash -c 'set -e; rm -rf ${E2E_ROOT}; mkdir -p ${E2E_ROOT}/cfg ${E2E_ROOT}/rt ${E2E_ROOT}/state; cd ${REPO_ROOT} && cargo build --quiet && exec ${MIRAGE_BIN} webui --addr 127.0.0.1:${DAEMON_PORT}'`,
      port: DAEMON_PORT,
      reuseExistingServer: !process.env.CI,
      timeout: 240_000,
      env: {
        XDG_CONFIG_HOME: `${E2E_ROOT}/cfg`,
        XDG_RUNTIME_DIR: `${E2E_ROOT}/rt`,
        XDG_STATE_HOME: `${E2E_ROOT}/state`,
        MIRAGE_BIN: MIRAGE_BIN,
        MIRAGE_DASHBOARD_SKIP_NPM_CI: "1",
      },
    },
  ],
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});
