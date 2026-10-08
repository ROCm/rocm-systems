# Dashboard build and test

Run from `emulation/rocjitsu/website/dashboard`. The React/Vite dashboard needs no
native Rocjitsu build, GPU, ROCm installation or benchmark service. Node must match
`package.json` (`^20.19.0 || ^22.13.0 || >=24.0.0`). Install the locked npm dependencies
and Playwright Chromium when setting up a machine:

```bash
npm ci
npm run test:e2e:install
```

If Chromium's system libraries already exist, `npx playwright install chromium`
installs only the browser. Network access is needed for installation, not for the
maintained browser tests.

## Default scope

Uninitialized dashboard filters select every available target, suite and declared
execution mode. When both gfx1250 and gfx950 publish ST and MT, all four
configurations are included. Saved or URL selections, including explicit empty
selections, take precedence; empty bootstrap data never becomes a user choice.
Recent Runs uses numbered pages with first/last (`|<` / `>|`) and previous/next
arrow icons. First/last jump to the endpoints, not adjacent pages; accessible
labels and native hover titles describe each action.

## Data and test boundaries

Published data must satisfy the [data contract](website-data-contract.md),
including explicit ST/MT configurations. Invalid or unavailable data produces
**No available test data**, with Retry and disabled export.

Fictional measurements live only under `tests/fixtures/`. Fixture builds show
**Test fixtures — fictional measurements · Not production data.** The production
suite checks the data-free artifact, deterministic responses intercepted at the
configured GitHub Raw origin, and rejection of incompatible data. Tests do not
depend on live publication or use fixtures as a production fallback. Passing
them does not verify a live dataset.

## Commands

| Command | Purpose |
| --- | --- |
| `npm run lint` | ESLint source, tests and configuration |
| `npm run test:unit` | Data contract, loader, selectors, state and page logic without a browser |
| `npm run test:e2e -- --list` | Discover desktop/mobile cases; no server, build or browser launch |
| `npm run test:e2e:production -- --list` | Discover deterministic production-boundary cases without a build/browser |
| `npm run test:e2e` | Desktop controls and phone-specific interactions |
| `npm run test:e2e:production` | Rebuild data-free `dist/` and test artifact, consumer and invalid-data boundaries |
| `npm run test:e2e:chart-interactions` | Repeat the maintained grid point/timeframe/scope interaction three times serially |
| `npm run test:e2e:chart-race` | Alias for `test:e2e:chart-interactions` |
| `npm test` | Unit tests followed by the fixture browser suite |
| `npm run verify` | Lint, unit tests, fixture browsers, then the deterministic production suite |
| `npm run build` | Build application assets into `dist/`; does not publish benchmark JSON |
| `npm run validate:data -- <data-directory>` | Validate every indexed run and referenced catalog |
| `npm run process:data -- <data-directory>` | Validate and print normalized data plus unchanged source envelopes |
| `npm run process:data -- <data-directory> --output <new-file-outside-input>` | Create a local processed snapshot; refuses overwrite and output inside the input directory |

Playwright's fixture server builds `.test-dist/`; the production server builds
`dist/`, so `verify` does not need a separate build step. The optional
three-repeat interaction command repeats a case already in the fixture suite;
it is a focused race probe, not an additional test suite.

### CI coverage

The [website test workflow](../../../../../.github/workflows/rocjitsu-website-test.yml)
and the dashboard step in the
[publish workflow](../../../../../.github/workflows/rocjitsu-publish-website.yml)
run `npm run verify`. It discovers all `tests/unit/**/*.test.js`, fixture
`tests/e2e/*.e2e.js` and `tests/production/*.e2e.js` tests. Fixture cases run on
desktop Chromium except `mobile.e2e.js`, which uses Chromium with phone emulation.
Production cases run on desktop Chromium with deterministic request interception.

The optional chart-interaction command repeats an already-covered test; its
extra repetitions are not part of `verify`. Handbook checks run separately in
the handbook workflow and publication step: strict MkDocs build, Python unittest
discovery and the Node repository-statistics tests. New test files must be
committed to be available in CI.

### Browser coverage

The fixture suite runs ordinary behavior on desktop; phone tests cover only
responsive/touch concerns. Pure contract permutations remain in Vitest.

| Spec | Maintained semantic coverage |
| --- | --- |
| `dashboard-scope-defaults.e2e.js` | All published targets/suites/ST+MT selected on fresh load and reload; stored narrow choices and explicit empty URL overrides |
| `overview.e2e.js` | Four-page navigation, themes, selected-scope sum, history ranges and keyboard/anchor inspection, develop-only execution chronology, explicit empty target/suite/mode scopes and persistence |
| `recent-runs-pagination.e2e.js` | Validated fixture-derived histories of 65 and 205 attempts; 20-run pages, numbered jumps and ellipses, icon-only first/previous/next/last navigation with boundary disabling, partial/short/empty pages, refresh shrinkage, and desktop/mobile layout |
| `branch-runs.e2e.js` | Branch/PR/SHA search, empty matches, exact/fallback/manual reference, local configuration matrix, unpublished/failure exclusions, benchmark search/suites/sorting, scoped full comparison, missing identities, history, normal 1280px and 1440px **CSS zoom 2** overflow regression |
| `comparisons.e2e.js` | Searchable exact attempts, one atomic swap/history entry, swapped totals/bars/exclusions, measured-zero percentage unavailability, generic metadata differences, escaped tooltip text |
| `benchmarks.e2e.js` | Draft/apply/cancel grid picker, bounded search/eight slots, pointer removal/re-add/empty recovery, point selection across timeframe/scope changes, keyboard result inspection and complete/zero/failed/timeout/unpublished distinctions |
| `loading.e2e.js` | Partial progress, loading action gating, fatal load and failed refresh, automatic retries and user Retry, stale export removal, exact raw measurement/branch export, cache-generation/fetch-policy contract, optional storage failure, invalid-publication fail-closed behavior |
| `feedback.e2e.js` | Trend tooltip/guides and estimate anchors, default scope persistence, commit links, comparison picker layout, branch selection/collapse, delta sorting and differing metadata rows |
| `picker-inspection-feedback.e2e.js` | Short-viewport grouped picker, standalone chart details, nested result inspection with selection and focus restoration |
| `mobile.e2e.js` | Collapsible filters, touch trend inspection, all-page navigation in both themes, contained dialogs/tables, branch list/detail Back/Forward and local scroll/focus restoration |
| `tests/production/production.e2e.js` | No shipped fictional JSON, actual production URL consumer with deterministic interception, incompatible-data rejection with generic no-data UI |
| `tests/production/overview-empty-feedback.e2e.js` | Managed production URL, generic no-data feedback, neutral empty metrics and signed performance-change colors in both themes |

The CSS `zoom: 2` regression checks layout overflow, not native browser zoom.

Request interception disables Chromium's HTTP cache. The deterministic tests
assert the real consumer's `no-store`/`force-cache`/`reload` policy and persistent
reload generation, not fabricated disk-cache hits or live GitHub CORS headers.
Loading handlers stay installed through teardown, holding exactly one request
with a synchronous flag and always releasing it in `finally`.

## Build/browser ownership

Run build/browser gates **one at a time in a worktree**. Fixture E2E and the optional
interaction probe share `.test-dist/`; production build and production E2E share
`dist/`. Different ports do not isolate the output directories. Defaults are
fixture port 4174 and production port 4175. Override occupied ports without
stopping another owner's server:

```bash
PLAYWRIGHT_PORT=4184 PLAYWRIGHT_PRODUCTION_PORT=4185 npm run verify
```

Clear `DASHBOARD_DATA_DIR`, `VITE_DASHBOARD_DATA_BASE_URL` and
`PLAYWRIGHT_REUSE_EXISTING_SERVER` for acceptance gates. Server reuse is opt-in
only for local fixture iteration, never CI. Reuse only a fixture preview from this
exact worktree and current source; otherwise Playwright must build its own server.
Fixture results/traces go to `test-results/fixtures/`, production results to
`test-results/production/`; failures retain traces. Default worker counts are two
for fixtures and one for production; retries are zero.

## Local preview and processing

```bash
npm run dev:fixtures -- --host 127.0.0.1
npm run dev:data -- /absolute/path/to/staged/data --host 127.0.0.1
```

Fixture mode serves `tests/fixtures/data/` at `/data/` through the production
loader; local-data mode mounts a supplied directory without copying it into the
artifact. Restart after editing JSON. To preview a production build against a
local data directory:

```bash
DASHBOARD_DATA_DIR=/absolute/path/to/staged/data npm run build
npm run preview:data -- /absolute/path/to/staged/data --host 127.0.0.1
```

For the default production URL, use `npm run build` then
`npm run preview -- --host 127.0.0.1` (port 4173). The generic
**No available test data** banner is expected for unavailable or
incompatible data.
**Reload all data** fetches a fresh metadata/index and cache-busts indexed runs and
catalogs; after successful validation its generation is reused for later loads.
Errors disable raw export rather than retaining stale measurements.

`process:data` validates and normalizes local data; it does not run benchmarks
or publish results. It neither invents absent
configurations nor edits the input publication. Validate staged data before
processing it; only indexed runs and their referenced catalogs are checked.

## Publication separation

Production `dist/` contains application assets only (`publicDir` is disabled).
Application assets belong on `gh-pages/rocjitsu-dashboard/`; JSON remains separate
on `gh-pages-rocjitsu/rocjitsu-dashboard/data/`, fetched from GitHub Raw. Keep fixture
JSON out of both publication paths. Application verification and benchmark-data
publication are separate gates.
