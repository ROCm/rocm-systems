# Generate benchmark data for the dashboard

This guide is for authors of benchmark producer and publication workflows. The
[data contract](website-data-contract.md) defines the required fields and consumer
behavior; this guide covers preparing, validating and publishing a dataset.
Record ST/MT explicitly from actual executions. The frontend provides validation
and preview tools, not a benchmark producer or publisher.

## 1. Produce a staged wire dataset

Use an isolated data root containing:

```text
data/
  metadata.json
  index.json
  test-catalogs/<catalog-id>.json
  runs/<run-id>.json
```

Use the four complete JSON examples in contract sections 3–6 to understand the
shape. They are fictional and must not be published as real benchmark results.
Supply actual measurement and provenance values from your execution system.

- `metadata.json`: set numeric `schemaVersion: 2`, the repository URL,
  boolean `isBeta`, and `canonicalBranch: "develop"`.
- Catalog: define logical workloads (`id`, `suite`, `name`, scalar `problem`
  object) and membership for explicit `<target>:ST` / `<target>:MT` keys. Use a
  new immutable catalog ID whenever the snapshot changes. Reusing a workload
  ID across catalogs requires identical definition facts; changed definitions
  need new workload IDs.
- Run: give each attempt a unique `id` matching its filename, reference its
  immutable catalog, record `source`, `execution`,
  `environment`, and the actual measured `configurations`.
- Index: record a strict `generatedAt` timestamp and the unique
  `runs/<run-id>.json` paths for the complete intended snapshot, not just the
  latest upload. Only indexed runs and their referenced catalogs are loaded.

Read contract sections 2–6 for exact types, allowed URLs and optional-field rules.
Filenames and references are relative to the data root; do not put absolute paths or URLs in `runFiles` or `testCatalog`.

## 2. Map execution facts without inventing data

- Keep source commit time separate from execution completion time. Use a
  primitive full 40-hex SHA and enforce
  `source.committedAt <= execution.completedAt <= index.generatedAt`.
  Every occurrence of the same SHA must have the same source commit instant.
- Record the actual source branch, machine, `auto`/`manual` trigger and generic
  scalar environment facts. Optional source base and PR metadata must be
  truthful. The recorded base is not proof of a measured baseline.
- Declare `configurations[].threadingMode` exactly `ST` or `MT`; never infer it
  from workload suffixes, sample counts or functional/clocked execution. Include only
  configurations actually measured and published; all four target/mode pairs
  are not required in one file.
- For each included configuration, emit exactly one result for every workload
  in that catalog membership. `completed` requires finite nonnegative measured
  `durationSeconds` and `error: null`. `failed` and `timeout` require
  `durationSeconds: null` and `error: null` or a nonempty diagnostic string.
  All four result fields (`testId`, `status`, `durationSeconds`, `error`) are
  required. A genuine measured zero is valid; missing results are not zero.
- An absent whole configuration means not published. An omitted required
  workload inside a published configuration is invalid. There is no `missing`
  status; `exitCode` and `findings` are not allowed.
- Publish raw measurements only. Do not generate chart points, aggregate
  percentages, adjusted history, estimated anchors, UI coverage or normalized
  consumer records as wire inputs; the dashboard derives these.

## 3. Validate and inspect locally

From the repository's `emulation/rocjitsu/website/dashboard` directory, with the
Node version and installed dependencies documented in
[build and test](build-and-test.md):

```bash
npm run validate:data -- /absolute/path/to/staged/data
npm run dev:data -- /absolute/path/to/staged/data --host 127.0.0.1
```

Require validator exit status 0. One bad indexed run or referenced catalog
rejects the entire snapshot; the browser does not display partial history.
Unindexed files are not audited by this validator. Preview against your staged
input, not fixtures. Check ST/MT availability, zero/failure gaps, branch metadata,
record details and raw Download JSON. Branch-only or empty valid datasets are
accepted, but do not populate canonical develop history.

Optional local processing:

```bash
npm run process:data -- /absolute/path/to/staged/data \
  --output /absolute/path/to/local-output/processed-dashboard.json
```

The processor validates and creates a new `{data,sourceData}` envelope. It
refuses overwrite and output inside the input directory. That envelope is a
local consumer/debugging artifact, not a wire file to publish. These commands
do not run benchmarks or publish data.

## 4. Publication handoff

Data belongs on `ROCm/rocm-systems` branch `gh-pages-rocjitsu`, under
`rocjitsu-dashboard/data/`. The production consumer fetches from:

```text
https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/
```

Application assets are separate on `gh-pages/rocjitsu-dashboard/`; a website
build does not publish data. The workflow owner must implement and verify the
producer/publisher independently.

After validation, publish new immutable catalogs first, then new immutable runs,
and the mutable index last. Do not overwrite existing run/catalog filenames;
retain resources referenced by older cached indexes. Coordinate metadata changes
with the publication snapshot. Read back and validate the exact published
snapshot and verify the real consumer's fetch/reload behavior. Shape validation
cannot attest execution semantics, experiment equivalence or provenance.

Use the [publication checklist](website-data-contract.md#13-publication-checklist)
before releasing a dataset.
