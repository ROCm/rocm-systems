# Dashboard build and test

Run commands from `emulation/rocjitsu/website/dashboard`. The React/Vite app does not need a native Rocjitsu build, a GPU, a ROCm install, or a benchmark service. Node must satisfy `package.json`: `^20.19.0 || ^22.13.0 || >=24.0.0`.

```bash
npm ci
npm run test:e2e:install
```

When Chromium's system libraries are already present, `npx playwright install chromium` installs only the browser. Installation needs network access. The maintained browser tests do not.

## Commands

| Command | Purpose |
| --- | --- |
| `npm run lint` | ESLint source, tests, and configuration |
| `npm run test:unit` | Contract, loader, selectors, state, and page logic without a browser |
| `npm run test:e2e -- --list` | List desktop and mobile cases. No server, build, or browser |
| `npm run test:e2e:production -- --list` | List production-boundary cases. No build or browser |
| `npm run test:e2e` | Desktop controls and phone-specific interactions |
| `npm run test:e2e:production` | Rebuild data-free `dist/` and test the artifact, consumer, and invalid-data boundaries |
| `npm run test:e2e:chart-interactions` | Repeat the grid point, timeframe, and scope interaction three times, serially |
| `npm run test:e2e:chart-race` | Alias for `test:e2e:chart-interactions` |
| `npm test` | Unit tests, then the fixture browser suite |
| `npm run verify` | Lint, unit tests, fixture browsers, then the production suite |
| `npm run build` | Write application assets to `dist/`. Does not publish benchmark JSON |
| `npm run validate:data -- <data-directory>` | Validate every indexed run and referenced catalog |
| `npm run process:data -- <data-directory>` | Validate, then print normalized data and the unchanged source envelopes |
| `npm run process:data -- <data-directory> --output <new-file-outside-input>` | Write a local processed snapshot. Refuses overwrite and any output inside the input directory |

The fixture server writes `.test-dist/`. The production server writes `dist/`, so `verify` does not need a separate build. The three-repeat command reruns a case already in the fixture suite. Those extra repetitions are outside `verify`.

## One build at a time

Run build and browser gates one at a time in a worktree. Fixture end-to-end tests and the interaction probe share `.test-dist/`. Production build and production end-to-end tests share `dist/`. Different ports do not isolate those directories.

Defaults are fixture port 4174 and production port 4175. To leave an occupied port in use:

```bash
PLAYWRIGHT_PORT=4184 PLAYWRIGHT_PRODUCTION_PORT=4185 npm run verify
```

For an acceptance run, clear `DASHBOARD_DATA_DIR`, `VITE_DASHBOARD_DATA_BASE_URL`, and `PLAYWRIGHT_REUSE_EXISTING_SERVER`. Reuse a server only for local fixture iteration, from this worktree, against current source. CI does not reuse a server. Fixture results go to `test-results/fixtures/`; production results go to `test-results/production/`. Failures keep traces. Default workers are two for fixtures and one for production. Retries are zero.

For deployment destinations and the separate data-publication workflow, see [website deployment versus data publication](architecture-and-publication.md#website-deployment-versus-data-publication).
