# Schema 1 → schema 2 patch

Implementation: [runSchema.js](../src/data/runSchema.js), called by [dashboardValidation.js](../src/data/dashboardValidation.js).

## Accepted exclusion policy

Non-Vanilla instrumented runs are excluded from dashboard history and comparisons.
Original records are retained unchanged in raw exports; their measurements and
plugin identities are never relabelled as Vanilla. This is an accepted exclusion
policy, not a migration that converts instrumented measurements into ordinary
benchmark results.

This policy documents existing behavior. It does not restore reserved plugin support or retired UI,
and does not introduce a plugin selector or instrumented-run comparison surface.

## Dispatch

```js
// Simplified loader flow; validation failures stop the whole load.
if (isExcludedPluginRun(run)) {
  // Skip non-Vanilla measurements. Keep the original run in raw exports.
} else {
  const version = runSchemaVersion(run); // missing schemaVersion → 1
  const patched = version === 1
    ? migrateSchema1Run(run, catalog)
    : { run, catalog };

  // Must match metadata.json's target version; no invented upgrades.
  if (patched.run.schemaVersion !== metadata.schemaVersion) throw new Error("No migration");
  // Validate patched run + catalog before displaying measurements.
}
```

## Exact mapping

```text
schema 1                              schema 2
run.targets[].id                   →  run.configurations[].target
target.<id>.numThreads = 1         →  run.configurations[].threadingMode = "ST"
target.<id>.numThreads > 1         →  run.configurations[].threadingMode = "MT"
missing target-specific fact       →  "MT" compatibility fallback
run.targets[].results              →  run.configurations[].results (unchanged)
catalog.targets["gfx950"]          →  catalog.configurations["gfx950:<mode>"]
missing schemaVersion, or 1         →  schemaVersion: 2
comparisonId, plugin                →  absent from migrated run
```

```js
// Core transformation after legacy validation. `legacyMode` reads the exact
// target.<id>.numThreads fact, maps 1 to ST and larger positive integers to MT,
// falls back to MT when the fact is absent, and rejects duplicate or invalid facts.
const modes = Object.fromEntries(
  run.targets.map(({ id }) => [id, legacyMode(run.environment, id)]),
);
const patchedRun = {
  schemaVersion: 2,
  id: run.id,
  testCatalog: run.testCatalog,
  source: run.source,
  execution: run.execution,
  environment: run.environment,
  configurations: run.targets.map(({ id, results }) => ({
    target: id, threadingMode: modes[id], results,
  })),
};
const patchedCatalog = {
  id: catalog.id,
  tests: catalog.tests,
  configurations: Object.fromEntries(
    Object.entries(catalog.targets).map(
      ([target, ids]) => [`${target}:${modes[target]}`, ids],
    ),
  ),
};
```

- In-memory only: source files, index paths, and raw exports stay unchanged.
- Missing version means schema 1, not permission to guess missing fields. Invalid Vanilla data and unsupported versions fail; non-Vanilla files are the explicit exclusion.
- Legacy flat `runs/<id>.json` paths remain readable. New files use the schema-2 branch directories.
- Old publication-side `metadata.json` and index-level versions are ignored.
- Unchanged source values must satisfy the [current field constraints](schema-2.md#result-rules).

Complete examples: [schema 1](schema-1.md) and [schema 2](schema-2.md).
