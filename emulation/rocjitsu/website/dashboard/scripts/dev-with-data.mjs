#!/usr/bin/env node

// Mount a local benchmark data directory at /data/ and start Vite.
// Usage: node scripts/dev-with-data.mjs <data-directory> [--preview] [-- <vite args>]

import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  parseServeDataArgs,
  resolveDashboardDataDirectory,
} from './dashboard-data-dir.mjs';

const websiteRoot = fileURLToPath(new URL('..', import.meta.url));
const viteCli = path.join(websiteRoot, 'node_modules', 'vite', 'bin', 'vite.js');

function main() {
  const { directory, preview, viteArgs } = parseServeDataArgs(process.argv.slice(2));
  const dataDirectory = resolveDashboardDataDirectory(directory);
  const command = preview ? ['preview', ...viteArgs] : viteArgs;
  console.log(`Serving dashboard data from ${dataDirectory} at /data/`);
  const child = spawn(process.execPath, [viteCli, ...command], {
    cwd: websiteRoot,
    env: { ...process.env, DASHBOARD_DATA_DIR: dataDirectory },
    stdio: 'inherit',
  });
  child.on('exit', (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    process.exit(code ?? 1);
  });
}

main();
