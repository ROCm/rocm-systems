# Dashboard test fixtures

These fixtures are dummy measurements for tests and local development. They must
never be published as real RocJitsu results.

## Focused cases and generated history

`data/` contains 20 explicit run cases, two catalogs, metadata, and a seed index.
The cases cover failures, timeouts, sanitizer comparisons, historical reruns,
asserted baselines, and the change from five to seven catalog tests.

`dashboardFixture.js` reads those cases and deterministically generates the remaining
35 completed Vanilla runs for unoccupied days from July 17 through August 25, 2026.
Generated runs have unique IDs and commit hashes, fixed timestamps, and a small
runtime trend. The explicit regression cases keep their original values.

The combined dataset has 55 runs: 51 Vanilla runs and four sanitizer runs. This
preserves the 45-point chart window, more than 50 selector options, pagination,
and reliability coverage without storing a JSON file for each filler run.

`syntheticDataset.js` separately generates 500 smaller runs in memory for loader
concurrency, timeout, and cancellation tests. Those tests isolate request behavior
from the dashboard's richer scenario data.

## How tests and development use the data

- Unit tests call the shared factory through `publishedData.js`; no files are generated.
- Vite's explicit `fixtures` mode uses the same factory to recreate the ignored
  `.test-data/data/` directory. Development serves it directly; browser-test builds
  copy it into the separate `.test-dist/data/` output.
- The default production build never invokes fixture generation or copies fixture JSON.

`data/index.json` lists only the explicit cases. Edit those files to change a regression
scenario; edit the factory to change generated history. Do not edit `.test-data/`
or `.test-dist/`, which are disposable generated output.
After editing fixture JSON, restart the fixture development server or rebuild the
fixture preview; generation happens when Vite loads its configuration.

See the [build and test guide](../../docs/build-and-test.md) for commands and the
[data contract](../../docs/website-data-contract.md) for the published JSON format.
