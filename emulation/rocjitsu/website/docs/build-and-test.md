# Website build and test

The Rocjitsu simulation-performance dashboard lives in `emulation/rocjitsu/website`. It is a standalone
React + Vite source package. Real benchmark data belongs on the deployment branch;
dummy data is retained only as test fixtures. Building and
testing the website requires no Rocjitsu native build, ROCm installation, GPU, or
benchmark service.

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

If port 4174 is occupied, select a free port without stopping other servers:
`PLAYWRIGHT_PORT=4176 npm run verify` (or use the same variable with `npm run test:e2e`).

Individual commands, all run from `website/`:

| Command | Purpose |
| --- | --- |
| `npm run build` | Build the static site into `dist/` |
| `npm run dev:fixtures` | Serve the app with dummy fixture JSON at `/data/` |
| `npm run dev:data -- <data-directory>` | Serve the app with a local data directory at `/data/` |
| `npm run preview:data -- <data-directory>` | Preview `dist/` with a local data directory at `/data/` |
| `npm run lint` | Check JavaScript and React source with ESLint |
| `npm run test:unit` | Run data loader, selector, and utility tests without a browser |
| `npm run test:e2e` | Run Chromium desktop behavior and mobile layout tests |
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
the production site uses for `dist/data/`. The argument may be the data directory
itself or a parent that contains `data/`. Unit tests validate the same fixture
data in memory. Restart the development server after editing JSON. For a
production-build preview of local data, run `npm run build` and then
`npm run preview:data -- /absolute/path/to/staged/data --host 127.0.0.1`.
To inspect the production build without data:

```bash
npm run build
npm run preview -- --host 127.0.0.1
```

Production preview uses `http://127.0.0.1:4173`. Without separately hosted benchmark
JSON, the production site displays its data-unavailable state. Browser tests use
port 4174 and fixtures instead. For local iteration, a fixture preview can be started
with `npm run preview -- --mode fixtures --host 127.0.0.1 --port 4174` after a fixture
build (`npm run build -- --mode fixtures`). Set `PLAYWRIGHT_REUSE_EXISTING_SERVER=1`
to reuse that server for tests; CI ignores this option.

## Deployment branch handoff

`npm run build` produces only application files in `dist/`. The default build disables
Vite's public-directory copying; dummy fixtures cannot enter `dist/` through it.
The intended release flow is:

1. Build the website from the source branch.
2. Validate the complete staged benchmark directory with
   `npm run validate:data -- /absolute/path/to/pages-checkout/data`.
3. Copy the contents of `dist/` to the site root in a separate deployment-branch checkout.
4. Publish real benchmark JSON separately under that site's `data/` directory, following
   [the data contract](website-data-contract.md).
5. Commit and push the deployment branch through the release process used for GitHub Pages.

Do not deploy if validation fails. The browser uses the same validator and fails
closed if invalid data bypasses this gate; it does not display a partial run history.
Validation follows `index.json`, so it only checks the runs listed there and the
catalogs those runs reference. Stage the directory with its updated index before
validating; see [the data contract](website-data-contract.md) for what stays unchecked.

For example, from `emulation/rocjitsu/website`, after checking out the deployment
branch in a separate directory, set the destination to that checkout's site root:

```bash
npm ci
npm run build
# Replace this path with the actual deployment checkout's site root.
pages_site_dir=/absolute/path/to/pages-checkout
rsync -a dist/ "$pages_site_dir/"
```

The trailing slash copies the contents of `dist/`. This copy preserves existing
benchmark `data/` and deployment-branch configuration. It does not remove old hashed
assets; any later cleanup should target obsolete application assets while preserving
`data/` and repository metadata. Never copy `.test-data/`, `.test-dist/`, `tests/`, or the entire source
package to the deployment branch. This guide prepares the files; no deployment branch
or publishing workflow is created by the migration.

The deployed site root contains `index.html`, `assets/`, and independently published
`data/metadata.json`, `data/index.json`, `data/test-catalogs/`, and `data/runs/`.
Serve it over HTTP; the app fetches the data at runtime. Relative asset and data URLs
allow hosting under a URL subdirectory.

Dependencies, generated fixture data, production/test build output, coverage, and Playwright reports/results
are ignored by Git. Keep the source, lockfile, tests, and test fixtures under version
control. Parent-repository automation should use `emulation/rocjitsu/website` as its
working directory and its `package-lock.json` as the npm cache key.
