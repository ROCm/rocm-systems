# Website, data, and publication

## Source map

Paths below are relative to `emulation/rocjitsu/website/dashboard/`.

```text
src/main.jsx → src/App.jsx             # start React; coordinate loading and pages
src/components/                       # pages, charts, tables, pickers
src/hooks/useDashboardState.js         # filters, selections, URL/history state
src/data/dashboardData.js             # fetch publication JSON
src/data/dashboardValidation.js       # validate and normalize it
src/data/runSchema.js                 # supported versions and schema-1 patch
src/data/selectors.js                  # derive chart/comparison data
src/config/metadata.json              # bundled settings and target run schema
scripts/dashboard-data/                # prepare and stage benchmark publications
vite.config.js                        # build assets and select the data source
```

This is a static React website. The browser fetches JSON and computes the displayed comparisons; there is no dashboard application server in production.

## Website deployment versus data publication

These are separate workflows and Git branches:

| What | Repository-root workflow | Destination |
| --- | --- | --- |
| Website HTML/JS/CSS | `.github/workflows/rocjitsu-publish-website.yml` | `gh-pages:rocjitsu-dashboard/` |
| Benchmark JSON | `.github/workflows/rocjitsu-benchmarks.yml` | `gh-pages-rocjitsu:rocjitsu-dashboard/data/` |

Website publication checks current `develop` against the last published source revision, verifies/builds changed sites, then publishes dashboard `dist/`. Manual publication also requires `develop`. Production assets do not contain benchmark fixtures or measurements. Keep fixture JSON out of both publication destinations. Updating bundled settings requires a website rebuild; publishing new measurements does not.

```text
Website URL:
https://rocm.github.io/rocm-systems/rocjitsu-dashboard/

Production JSON base URL:
https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/
```

The base URL comes from `scripts/dashboard-data-source.mjs`; `src/data/publishedDataUrls.js` constructs the index URL. Fixture/local-data modes use `./data/`: fixtures come from `tests/fixtures/data/`, or `DASHBOARD_DATA_DIR` supplies a local directory.

## What each JSON file contains

| File | Information |
| --- | --- |
| `src/config/metadata.json` | Target run `schemaVersion`, repository URL, Beta flag, canonical branch. Bundled, never fetched from the data branch. |
| `data/index.json` | Publication time and the complete list of visible run-file paths. |
| `data/test-catalogs/<id>.json` | Workload definitions and target/ST/MT membership. No measured timings. |
| `data/runs/default-branch/<id>.json` | A develop attempt: schema version, source commit, execution/machine/environment, configurations, timings, statuses, diagnostics. |
| `data/runs/side-branches/<id>.json` | The same run format for any other branch. Flat directory, not one directory per branch. |

```text
index.runFiles[] → run file
run.testCatalog → catalog file
run result.testId → catalog.tests[].id
```

## Browser loading

```text
Fetch index.json
  → fetch every listed run
  → fetch the distinct catalogs needed by included runs
  → apply legacy migration
  → validate and normalize
  → selectors → UI
```

Unindexed files are not scanned. Compatibility, exclusions, and raw-export behavior are covered in [schema migration](schema-migration.md). The mutable index bypasses cache; immutable runs/catalogs can use cache. “Reload all data” cache-busts requests.

## Preparing and publishing measurements

The benchmark workflow runs on relevant develop pushes or manual dispatch; its “nightly” name is not a scheduled trigger. Before a side-branch benchmark starts, a lightweight job snapshots `develop` and records the dispatched commit's merge-base with that immutable revision. The optional `pull_request_number` input also records an open same-repository PR when its head branch and SHA match the selected workflow branch; without that input, the run keeps its base but has no PR metadata. Develop runs reject the PR input and omit both fields.

```text
Run ST and MT benchmarks
  → upload raw result directories as a GitHub Actions artifact
  → publish job downloads that artifact
  → prepare-dashboard-data.py (once)
  → fetch latest gh-pages-rocjitsu
  → publish-dashboard-run.py
  → validate-dashboard-data.mjs
  → Git commit + push
```

- **Prepare:** validate raw ST/MT results and provenance, lowercase commit SHAs, calculate median completed timings, and generate one schema-2 run plus its catalog. Catalog IDs are content hashes.
- **Stage:** reuse/add the catalog, add the run, then update `index.json` last while preserving history.
- **Publish:** validate the staged dataset before pushing. If the branch moves, fetch its latest state and retry staging/pushing the same prepared files—not the benchmarks.

The Actions artifact only transfers raw results between jobs. The Git push makes dashboard JSON available.

## Immutable means “do not change an existing resource”

```text
Run/catalog: same path + same JSON content → reuse; no rewrite
Run/catalog: same path + different content → error
Changed catalog content                  → new catalog hash/file
New benchmark attempt                    → new run ID/file
index.json                               → mutable; updated to reference runs
```

Timing-only changes do not change the catalog. Keep old run/catalog files for cached indexes.

Examples: [schema 1](schema-1.md), [schema 2](schema-2.md), [migration](schema-migration.md). Commands: [build and test](build-and-test.md).
