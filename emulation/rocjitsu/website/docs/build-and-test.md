# Website build and test

The Rocjitsu simulation-performance dashboard lives in `emulation/rocjitsu/website`. It is a standalone
React + Vite source package. Real benchmark data belongs on the
`gh-pages-rocjitsu` branch under `rocjitsu-dashboard/data/`; dummy data is retained only
as test fixtures.
Building and testing the website requires no Rocjitsu native build, ROCm
installation, GPU, or benchmark service.

## Prerequisites

- Node.js matching `website/package.json`: `^20.19.0 || ^22.13.0 || >=24.0.0`.
- npm and network access to install the locked dependencies.
- Chromium and its system libraries for Playwright browser tests. The install
  command below downloads Chromium and may need administrator privileges to
  install missing OS packages on Linux.

## Install, build, and verify

Run from the `rocm-systems` repository root:

```bash
cd emulation/rocjitsu/website
node --version
npm --version
npm ci
npm run test:e2e:install
npm run verify
```

`npm run verify` runs ESLint, a production build, Vitest unit tests, Playwright
desktop and mobile browser tests, and the chart interaction race test ten times
sequentially. A successful run exits with status 0. Playwright builds the current
source with dummy fixtures into `.test-dist/`, starts its own preview server at
`http://127.0.0.1:4174`, and stops it when done. Keep port 4174 free. Browser tests
leave the data-free production build in `dist/` untouched.

`npm run test:e2e:production` separately rebuilds and previews the production
artifact on port 4175, loads the published GitHub Raw dataset, and verifies real
browser-cache reuse without request interception. CI runs this live production
smoke test after the fixture suite; it requires network access and valid
published data.

If port 4174 is occupied, select a free port without stopping other servers:
`PLAYWRIGHT_PORT=4176 npm run verify` (or use the same variable with `npm run test:e2e`).

Individual commands, all run from `website/`:

| Command | Purpose |
| --- | --- |
| `npm run build` | Build into `dist/`, loading `rocjitsu-dashboard/data/` from `gh-pages-rocjitsu` |
| `npm run dev:fixtures` | Serve the app with dummy fixture JSON at `/data/` |
| `npm run dev:data -- <data-directory>` | Serve the app with a local data directory at `/data/` |
| `npm run preview:data -- <data-directory>` | Mount local data at `/data/` while previewing `dist/` |
| `npm run lint` | Check JavaScript and React source with ESLint |
| `npm run test:unit` | Run data loader, selector, and utility tests without a browser |
| `npm run test:e2e` | Run Chromium desktop behavior and mobile layout tests |
| `npm run test:e2e:production` | Smoke test the production build, GitHub Raw hosting, and browser caching |
| `npm run test:e2e:chart-race` | Run the chart interaction race test ten times sequentially |
| `npm test` | Run the unit tests, the browser suites, and the ten-repeat chart race test |
| `npm run verify` | Run lint, build, and everything in `npm test` |

Run the two browser commands one at a time. Both rebuild the shared `.test-dist/`
fixture output before starting their server, so a concurrent run deletes files the other
one is still copying and fails the build with `ENOENT`. Assigning a different
`PLAYWRIGHT_PORT` avoids the port conflict but not this one, because the build directory
is shared regardless of port.

If Chromium's system libraries are already installed, `npx playwright install chromium`
installs just the browser without changing OS packages. Rerun browser installation
after updating Playwright if its required browser version changes.

## Run locally

```bash
# From emulation/rocjitsu/website:
npm run dev:fixtures -- --host 127.0.0.1
npm run dev:data -- /absolute/path/to/staged/data --host 127.0.0.1
```

Use the URL printed by Vite. Fixture mode serves the static JSON in
`tests/fixtures/data/` through the same application loader used by the normal
dashboard. `dev:data` mounts any local data directory at `/data/`, the same URL
used by fixture mode. The argument may be the data directory itself or a parent
that contains `data/`. Unit tests validate the same fixture data in memory.
Restart the development server after editing JSON. For a production-build
preview of local data, build with a local URL before mounting the directory:

```bash
DASHBOARD_DATA_DIR=/absolute/path/to/staged/data npm run build
npm run preview:data -- /absolute/path/to/staged/data --host 127.0.0.1
```

To inspect the production build against published data:

```bash
npm run build
npm run preview -- --host 127.0.0.1
```

Production preview uses `http://127.0.0.1:4173`. If the published URL is invalid,
unreachable, or contains no valid data, the application remains available and
displays the same data-unavailable state as a missing local `data/` directory.
The header's **Reload all data** action fetches a fresh index, then fetches
every referenced run and catalog directly from the data source without using caches.
After validation, those files become that browser's cache for later normal refreshes.
Browser tests use port 4174 and fixtures instead. For local iteration, a fixture
preview can be started with
`npm run preview -- --mode fixtures --host 127.0.0.1 --port 4174` after a fixture
build (`npm run build -- --mode fixtures`). Set
`PLAYWRIGHT_REUSE_EXISTING_SERVER=1` to reuse that server for tests; CI ignores
this option.

## Production build

`npm run build` produces only application files in `dist/`. The default build disables
Vite's public-directory copying; dummy fixtures cannot enter `dist/` through it.
The application fetches data over HTTPS from `rocjitsu-dashboard/data/` on the
`gh-pages-rocjitsu` branch of the `ROCm/rocm-systems` repository. Publish
validated JSON independently under that directory. It contains
`metadata.json`, `index.json`, `test-catalogs/`, and `runs/`; a data-only update
does not require rebuilding the application.

This source package does not provide a deployment workflow. The hosting owner
chooses how to publish the contents of `dist/` through its existing release
process.

```bash
npm run validate:data -- /absolute/path/to/staged/data
```

Do not publish if validation fails. The browser uses the same validator and fails
closed if invalid data bypasses this gate; it does not display a partial run history.
Validation follows `index.json`, so it only checks the runs listed there and the
catalogs those runs reference. Stage the directory with its updated index before
validating; see [the data contract](website-data-contract.md) for what stays unchecked.

To preview a build against the published data branch:

```bash
npm run build
npm run preview -- --host 127.0.0.1
```

Dependencies, generated fixture data, production/test build output, coverage, and Playwright reports/results
are ignored by Git. Keep the source, lockfile, tests, and test fixtures under version
control. Parent-repository automation should use `emulation/rocjitsu/website` as its
working directory and its `package-lock.json` as the npm cache key.
