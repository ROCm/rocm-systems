# Dashboard build and test

Run commands from `emulation/rocjitsu/website/dashboard`. The React/Vite app does not need a native Rocjitsu build, a GPU, a ROCm install, or a benchmark service. Node must satisfy `package.json`: `^20.19.0 || ^22.13.0 || >=24.0.0`.

```bash
npm ci
npm run test:e2e:install
```

When Chromium's system libraries are already present, `npx playwright install chromium` installs only the browser. Installation needs network access. The deterministic browser suites do not; the opt-in published-data smoke below does.

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
| `npm run process:data -- <data-directory>` | Load/adapt, then print normalized data and the unchanged source envelopes |
| `npm run process:data -- <data-directory> --output <new-file-outside-input>` | Write a local processed snapshot. Refuses overwrite and any output inside the input directory |

The fixture server writes `.test-dist/`. The production server writes `dist/`, so `verify` does not need a separate build. The three-repeat command reruns a case already in the fixture suite. Those extra repetitions are outside `verify`.

## One build at a time

Run build and browser gates one at a time in a worktree. Fixture end-to-end tests and the interaction probe share `.test-dist/`. Production build and production end-to-end tests share `dist/`. Different ports do not isolate those directories.

Defaults are fixture port 4174 and production port 4175. To leave an occupied port in use:

```bash
PLAYWRIGHT_PORT=4184 PLAYWRIGHT_PRODUCTION_PORT=4185 npm run verify
```

For an acceptance run, clear `DASHBOARD_DATA_DIR`, `VITE_DASHBOARD_DATA_BASE_URL`, and `PLAYWRIGHT_REUSE_EXISTING_SERVER`. Reuse a server only for local fixture iteration, from this worktree, against current source. CI does not reuse a server. Fixture results go to `test-results/fixtures/`; production results go to `test-results/production/`. Failures keep traces. Default workers are two for fixtures and one for production. Retries are zero.

## Published-data production smoke (opt in)

```bash
PLAYWRIGHT_PUBLISHED_DATA=1 npm run test:e2e:production -- tests/production/published-data.e2e.js
```

Clear the data/server overrides listed above first. The production Playwright config
owns a fresh data-free build and preview; do not start another server. This test
uses the default GitHub Raw `gh-pages-rocjitsu/rocjitsu-dashboard/data/` endpoint,
without interception or fictional fallback. It waits for loading to settle and
distinguishes three published states: index HTTP 404 requires the specific
missing-data banner, disabled Download JSON, and no run requests; a valid empty
`runFiles: []` index requires successful loading and an empty source JSON download,
without data-error banners or run requests; a nonempty index requires successful
run loads and a source download matching the index and run count. Unexpected
alerts, uncaught browser errors, and JavaScript dialogs fail in every state.
Malformed data, other HTTP errors, network/CORS failures, and incompatible runs
are failures, not skips.

The website workflow runs this command separately after `npm run verify`, with
the opt-in flag scoped to that step. Ordinary production and fixture suites
remain deterministic; the smoke is skipped unless explicitly enabled. Failure
traces use the existing production diagnostics directory.

This is partial consumer coverage: **the new build against currently published
data**, not a deployed-consumer gate on benchmark publication. It neither tests
nor rewrites publication storage. Legacy experimental data can fail until the
maintainer's manual release cutover; do not treat a fictional intercepted control
as a live-data pass or suppress a known incompatibility.

## Benchmark preparation and publication tests

Use a Python 3.12 environment with Node.js (as above), Bash, Git, jq, and coreutils
(including `realpath`) on `PATH`. Publication tests exercise workflow JSON/shell operations; no npm dependencies are required. From `emulation/rocjitsu/website/dashboard`:

```bash
python -m pip install -r scripts/dashboard-data/requirements-test.txt
python -m pytest scripts/dashboard-data/tests
```

The `dashboard-data` CI job (Verify benchmark preparation and publication) uses
the same requirements file. The suite exercises benchmark preparation and the
publication workflow shell, not the removed standalone publishers. Preparation
build/install/native commands are intercepted; small real child processes cover
exit-status reporting and owned-group cleanup on SIGINT/SIGTERM. Workflow contract
tests keep only the benchmark job (120 minutes), performance step (60 minutes)
and publisher job (30 minutes) limits in Actions, with no helper timers or separate
diagnostic/upload/cleanup caps. CLI tests exercise explicit manifest parsing through
Preparation inventory and alternative recipe selection; inventory tests cover path
rejection. Workflow contracts verify generation and measurement consume the same `BENCHMARK_MANIFEST`.
These tests stay under the dashboard
and are separate from `npm run verify`; they do not require npm dependencies.

For deployment destinations and the separate data-publication workflow, see [website deployment versus data publication](architecture-and-publication.md#website-deployment-versus-data-publication).
