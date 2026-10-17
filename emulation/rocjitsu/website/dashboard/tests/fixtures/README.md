# Fictional schema-2 dashboard test fixtures

Every measurement, branch, machine and commit message here is **fictional test
input**, not a published Rocjitsu performance claim. Never publish this directory.
The production build does not import this generator or copy fixture JSON;
fixtures are served only through explicit test/fixtures mode (integrator verifies
that boundary in the production build gate).

## Small deterministic publication

`schema2Dataset.js` generates the base publication: 24 canonical develop
attempts, 20 original fictional non-develop branches,
two immutable catalogs and an unversioned index. Every current run declares numeric
`schemaVersion: 2`; catalogs carry no schema version. Develop runs live in
`runs/default-branch/`, and every other branch uses the flat `runs/side-branches/`.
This gives at least 20 recent canonical rows and a scrollable 20-branch picker without imposing a production
branch-count limit. The publication time is fixed, not the viewer's clock.
Site settings use the same bundled `src/config/metadata.json` as production and
local-data builds; fixtures do not supply `metadata.json` or export synthetic
site settings in raw `{index,catalogs,runs}` downloads.

The additive overlay `createFeedbackPublication()` in `feedback-publication.js` adds
`users/RattataKing/test-branch`, giving 21 branches without renaming the original
20. Its wire record is `data/runs/side-branches/fictional-rattataking-test-branch.json` and the
index includes that record. The overlay clones an existing valid test measurement
with a unique fictional full SHA and exact develop base. `publishedData.js` retains
a compatibility re-export. The regeneration command below composes the base and
overlay, validates the complete publication, then writes all test JSON through
the shared writer. No manual overlay patch is needed. The base builder and
`writeSchema2FixtureDirectory()` remain base-only APIs; unit tests compare both forms.

Cases include explicit `threadingMode: "ST"` / `"MT"` and per-configuration
workload membership; a deleted workload and two added workloads with different first-success anchors; measured
zero; failures and timeouts; a whole missing configuration; exact develop-base
and earlier-develop fallback; optional PR metadata; late execution of an older
commit; and independent machine/environment changes. Fictional PRs intentionally
omit URLs rather than inventing working external links.

Regenerate only test JSON with:

```
node tests/fixtures/regenerate-fixtures.js
```

The command clears its destination before writing. Its default destination is
`tests/fixtures/data/`; an optional directory argument allows scratch regeneration.
Never point either fixture writer at production data.

Task-specific execution policies may require wrapping this command in an evidence
recorder. `syntheticDataset.js` separately supplies small in-memory schema-2 inputs
and the retained 500-attempt queue/concurrency stress case. No stress data is
copied into static browser fixtures.

## Retired schema-1 coverage and replacements

The large historical schema-1 corpus and production-looking SHA/date assertions
are retired. No semantic coverage was removed merely to make tests green.

| Retired assumption or over-specific assertion | Replacement coverage |
| --- | --- |
| Implicit target-only legacy results | Schema-1 Vanilla migration to MT; non-Vanilla skip with raw export retention; current schema-2 modes and configuration identities |
| Every run must be develop on one machine | Canonical history separated from branch attempts; generic environment/machine differences preserved |
| Zero duration rejected or treated as missing | Measured zero retained; zero-baseline percentages unavailable; tiny nonzero measurements stay nonzero |
| Latest successful result replaces catalog anchors | First success in current catalog, independently target/mode; failures/timeouts and missing configurations remain gaps |
| Two per-target Overview trend lines | One selected-scope sum, deleted workloads removed, estimates and anchor attempt identities disclosed |
| 1D prior-day baseline; cards always oldest vs newest | First eligible represented-period baseline, coherent cards/series/endpoints; no comparison for a single eligible endpoint |
| Exact historical run counts, historical SHAs, fixed values | Lightweight generated cases for chronology, backfills, reruns, catalog exclusions and coverage denominators |

Useful loader concurrency, cache directives/generations, retry/Retry-After,
progress, timeout, cancellation, malformed/missing resource and error-envelope
tests remain. Loader checks reject unsupported run versions (not index versions)
and accept empty snapshots. Historical schema-1 target/plugin records, with an
explicit version or no version, migrate to MT; only those records may use old
`runs/<id>.json` paths. Non-Vanilla records are explicitly skipped while their raw
files remain in source exports. Unversioned current runs are rejected. Directory validation and processing tests exercise the
same validator, input symlink escape rejection, fail-before-output and preserving
the complete raw-export envelope.

## Local consumer processing

```
node scripts/validate-dashboard-data.mjs <data-directory>
node scripts/process-dashboard-data.mjs <data-directory> --output <new-file-outside-input>
```

Processing validates the full snapshot before writing, refuses to overwrite an
existing output or place it in the input directory, and emits `{data,sourceData}`.
Only `backfillRunIds` changes representation to a JSON array. Normalized `data`
is an output for inspection and downstream tools, not a supported dashboard input;
there is no normalized-JSON re-import API. The dashboard consumes published
index, run and catalog files, including supported schema-1 migration. Without
`--output`, processing prints JSON to stdout. These are local consumer tools,
not a producer or publisher.
