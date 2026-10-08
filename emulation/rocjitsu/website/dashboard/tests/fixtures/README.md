# Fictional schema-2 dashboard test fixtures

Every measurement, branch, machine and commit message here is **fictional test
input**, not a published Rocjitsu performance claim. Never publish this directory.
The production build does not import this generator or copy fixture JSON;
fixtures are served only through explicit test/fixtures mode (integrator verifies
that boundary in the production build gate).

## Small deterministic publication

`schema2Dataset.js` is the source generator for `data/`: 24 canonical develop
attempts, 20 original fictional non-develop branches,
two immutable catalogs, metadata and index. This gives at least 20 recent
canonical rows and a scrollable 20-branch picker without imposing a production
branch-count limit. The publication time is fixed, not the viewer's clock.

The additive v11 overlay `createFeedbackPublication()` in `publishedData.js` adds
`users/RattataKing/test-branch`, giving 21 branches without renaming the original
20. Its wire record is `data/runs/fictional-rattataking-test-branch.json` and the
index includes that record. The overlay clones an existing valid test measurement
with a unique fictional full SHA and exact develop base. The base generator command
below recreates only the original 20 branches; preserve/reapply this overlay and
its index entry when regenerating v11 fixtures. Unit tests compare both forms.

Cases include explicit `threadingMode: "ST"` / `"MT"` and per-configuration
workload membership; a deleted workload and two added workloads with different first-success anchors; measured
zero; failures and timeouts; a whole missing configuration; exact develop-base
and earlier-develop fallback; optional PR metadata; late execution of an older
commit; and independent machine/environment changes. Fictional PRs intentionally
omit URLs rather than inventing working external links.

Regenerate only test JSON with:

```
node tests/fixtures/schema2Dataset.js
```

Task-specific execution policies may require wrapping this command in an evidence
recorder. `syntheticDataset.js` separately supplies small in-memory schema-2 inputs
and the retained 500-attempt queue/concurrency stress case. No stress data is
copied into static browser fixtures.

## Retired schema-1 coverage and replacements

The large historical schema-1 corpus and production-looking SHA/date assertions
are retired. No semantic coverage was removed merely to make tests green.

| Retired assumption or over-specific assertion | Replacement coverage |
| --- | --- |
| Schema 1 accepted; implicit target-only results | Explicit migration failure; schema-2 modes, configuration membership and normalized identities |
| Every run must be develop on one machine | Canonical history separated from branch attempts; generic environment/machine differences preserved |
| Zero duration rejected or treated as missing | Measured zero retained; zero-baseline percentages unavailable; tiny nonzero measurements stay nonzero |
| Latest successful result replaces catalog anchors | First success in current catalog, independently target/mode; failures/timeouts and missing configurations remain gaps |
| Two per-target Overview trend lines | One selected-scope sum, deleted workloads removed, estimates and anchor attempt identities disclosed |
| 1D prior-day baseline; cards always oldest vs newest | First eligible represented-period baseline, coherent cards/series/endpoints; no comparison for a single eligible endpoint |
| Exact historical run counts, historical SHAs, fixed values | Lightweight generated cases for chronology, backfills, reruns, catalog exclusions and coverage denominators |

Useful loader concurrency, cache directives/generations, retry/Retry-After,
progress, timeout, cancellation, malformed/missing resource and error-envelope
tests remain. New loader checks reject schema 1 before immutable requests and
accept empty snapshots. Directory validation and processing tests exercise the
same validator, input symlink escape rejection, fail-before-output and preserving
the complete raw-export envelope.

## Local consumer processing

```
node scripts/validate-dashboard-data.mjs <data-directory>
node scripts/process-dashboard-data.mjs <data-directory> --output <new-file-outside-input>
```

Processing validates the full snapshot before writing, refuses to overwrite an
existing output or place it in the input directory, and emits `{data,sourceData}`.
Only `backfillRunIds` changes representation to a JSON array; `loadDashboardData`
revalidates normalized data and rebuilds its Set. Without `--output`, processing
prints JSON to stdout. These are local consumer tools, not a producer or publisher.
