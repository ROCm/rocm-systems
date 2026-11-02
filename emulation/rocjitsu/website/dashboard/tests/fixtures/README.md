# Fictional corpus-schema dashboard fixtures

Every measurement, branch, machine and commit message is fictional test input,
not a published Rocjitsu performance claim. Never publish this directory.
Production builds neither import generators nor copy fixture JSON.

`maintainedDataset.js` builds 24 develop attempts and 20 side branches.
`feedback-publication.js` adds three deterministic develop slowdowns (attempts
18, 20 and 22) to exercise speedups below 1.00× in Performance Trend. It also adds
`users/RattataKing/test-branch` and one interrupted
develop run, `fictional-unfinished-develop`. The interrupted run has a finish
timestamp, one completed result and unexecuted cases; Recent Runs shows its run
time and Fail status. Each document has corpus `schema_version: 1`,
flat rows with embedded workload definitions and literal `single`/`default`
thread policies, raw provenance and fictional publication context. The two
benchmark sets exercise added/removed workloads; actual configuration absence,
late execution/backfill, independent environments, failure/timeout and exact
branch/base/PR behavior remain covered. No per-attempt set variants are created.

`fictional-producer-run.json` is the accepted CPU harness writer artifact. Its
measurements and source checkout are fictional/disposable evidence, not an
actual performance-v1 or native/GPU benchmark. Tests enrich only publication
context in memory and preserve its original raw values.

Regenerate the complete checked-in publication deterministically:

```bash
node tests/fixtures/regenerate-fixtures.js
node tests/fixtures/regenerate-fixtures.js /path/to/disposable-data --overview-contract
```

The generator clears its destination. Never target published data or another
owner's server directory. The seven-attempt Overview scenario has three sets,
typed problems, sparse configurations, missing days and same-commit retries.
`setFixtureConfigurations()` changes selected rows without inventing set IDs.
`syntheticDataset.js` creates queue/concurrency stress data in memory only.

Fixture runs use `no-store`; production immutable runs use `force-cache`.
Reload All preserves the existing cache-generation contract. Browser tests
intercept JSON at the real consumer boundary; they do not prove hosted
publication, live CORS or a browser disk-cache hit. Source export preserves the
complete parsed `{index,runs}`, independent of current filters and selection.

Local inspection uses `npm run process:data -- <data-directory>` and optional
`--output <new-file-outside-input>`. It refuses overwrite, symbolic resources and
input-bound output; it never publishes. See [corpus input](https://github.com/ROCm/rocjitsu-test-corpus/blob/develop/benchmarks/raw-run-schema.md).
