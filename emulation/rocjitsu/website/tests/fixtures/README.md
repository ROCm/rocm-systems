# Dashboard test fixtures

These fixtures are dummy measurements for tests and local development. They must
never be published as real Rocjitsu results.

## Static fixture data

`data/` contains 83 explicit run cases, two catalogs, metadata, and an index.
The cases cover failures, timeouts, sanitizer comparisons, historical reruns,
asserted baselines, and the change from five to seven catalog tests.

The combined dataset has 83 runs: 79 Vanilla runs and four sanitizer runs. This
provides the same JSON loading path used by the normal dashboard while preserving
the chart window, selector options, pagination, and reliability coverage.

`syntheticDataset.js` separately generates 500 smaller runs in memory for loader
concurrency, timeout, and cancellation tests. Those tests isolate request behavior
from the dashboard's richer scenario data.

## How tests and development use the data

- Unit tests load and validate the fixture JSON through `publishedData.js`.
- Vite's explicit `fixtures` mode serves this directory directly; browser-test builds
  copy it into the separate `.test-dist/data/` output.
- The default production build never invokes fixture generation or copies fixture JSON.

`data/index.json` lists the explicit cases. Add or edit run JSON files and their index
entries to change fixture history. Do not edit `.test-dist/`, which is disposable
generated output. After editing fixture JSON, restart the fixture development server
or rebuild the fixture preview.

See the [build and test guide](../../docs/build-and-test.md) for commands and the
[data contract](../../docs/website-data-contract.md) for the published JSON format.
