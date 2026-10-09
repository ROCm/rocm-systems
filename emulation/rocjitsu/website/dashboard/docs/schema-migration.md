# Schema 1 → schema 2 patch

Implementation: [runSchema.js](../src/data/runSchema.js), called by [dashboardValidation.js](../src/data/dashboardValidation.js).

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
(no threading mode)                →  run.configurations[].threadingMode = "MT"
run.targets[].results              →  run.configurations[].results (unchanged)
catalog.targets["gfx950"]          →  catalog.configurations["gfx950:MT"]
missing schemaVersion, or 1         →  schemaVersion: 2
comparisonId, plugin                →  absent from migrated run
```

```js
// Core transformation after legacy validation:
const patchedRun = {
  schemaVersion: 2,
  id: run.id,
  testCatalog: run.testCatalog,
  source: run.source,
  execution: run.execution,
  environment: run.environment,
  configurations: run.targets.map(({ id, results }) => ({
    target: id, threadingMode: "MT", results,
  })),
};
const patchedCatalog = {
  id: catalog.id,
  tests: catalog.tests,
  configurations: Object.fromEntries(
    Object.entries(catalog.targets).map(([target, ids]) => [`${target}:MT`, ids]),
  ),
};
```

- In-memory only: source files, index paths, and raw exports stay unchanged.
- Missing version means schema 1, not permission to guess missing fields. Invalid Vanilla data and unsupported versions fail; non-Vanilla files are the explicit exclusion.
- Legacy flat `runs/<id>.json` paths remain readable. New files use the schema-2 branch directories.
- Old publication-side `metadata.json` and index-level versions are ignored.
- Unchanged source values must satisfy the [current field constraints](schema-2.md#result-rules).

Complete examples: [schema 1](schema-1.md) and [schema 2](schema-2.md).
