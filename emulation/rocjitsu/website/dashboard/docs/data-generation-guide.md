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
  index.json
  test-catalogs/<catalog-id>.json
  runs/
    default-branch/<run-id>.json
    side-branches/<run-id>.json
```

Use the three complete JSON examples in contract sections 4–6 to understand the
shape. They are fictional and must not be published as real benchmark results.
Supply actual measurement and provenance values from your execution system.

- Catalog: define logical workloads (`id`, `suite`, `name`, scalar `problem`
  object) and membership for explicit `<target>:ST` / `<target>:MT` keys. Use a
  new immutable catalog ID whenever the snapshot changes. Reusing a workload
  ID across catalogs requires identical definition facts; changed definitions
  need new workload IDs.
- Run: set numeric `schemaVersion: 2`, give each attempt a unique `id` matching
  its filename, reference its
  immutable catalog, record `source`, `execution`,
  `environment`, and the actual measured `configurations`.
- Directory: use `runs/default-branch/` when `source.branch` is exactly `develop`,
  otherwise `runs/side-branches/`. Do not create per-branch subdirectories.
- Index: record a strict `generatedAt` timestamp and unique relative run paths
  for the complete intended snapshot, not just the latest upload. Neither index
  nor catalogs have `schemaVersion`; each run declares its own format.
  Only indexed runs and the catalogs needed by included runs are loaded.

Site settings are not publication inputs. The source-controlled
`src/config/metadata.json` selects the target run format with `schemaVersion: 2`; its settings
are exposed by `src/config/siteConfig.js` as `DASHBOARD_SITE_CONFIG`, supplying the repository URL,
Beta flag and canonical branch to production, fixture and local-data builds;
changing these settings requires a rebuild. New publications do not generate or
require publication-side `metadata.json`; any legacy copy is ignored, not
automatically deleted. Supported versions and migrations live in website code,
not in publication configuration.

Read contract sections 2–6 for exact types, allowed URLs and optional-field rules.
Filenames and references are relative to the data root; do not put absolute paths or URLs in `runFiles` or `testCatalog`.

## 2. Map execution facts without inventing data

- Keep source commit time separate from execution completion time. Use a
  primitive lowercase full 40-hex SHA and enforce
  `source.committedAt <= execution.completedAt <= index.generatedAt`.
  Every occurrence of the same SHA must have the same source commit instant.
  The preparer accepts valid uppercase/mixed-case Rocjitsu and corpus Git SHAs,
  checks expected identity case-insensitively after validating both full SHA
  strings, and emits lowercase. Malformed or different identities still fail.
  The website rejects noncanonical SHA spellings instead of changing them.
  The preparer does not currently emit optional `source.base`; other producers
  must also lowercase `source.base.commit` before publication.
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
The intentional exception is non-Vanilla plugin files, which are excluded
without being relabeled as ordinary runs. Their plugin-only catalogs are not loaded.
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
local consumer/debugging artifact, not a wire file to publish. `sourceData` contains
only `{index,catalogs,runs}`; normalized `data` retains site settings separately
using the same bundled defaults as the browser. These commands
do not run benchmarks or publish data.

## 4. Publication handoff

Data belongs on `ROCm/rocm-systems` branch `gh-pages-rocjitsu`, under
`rocjitsu-dashboard/data/`. The production consumer fetches from:

```text
https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/
```

Application assets are separate on `gh-pages/rocjitsu-dashboard/`; a website
build does not publish data. The staging scripts must be integrated and verified
by the publication workflow owner; generating files is not a live deployment.

After validation, publish new immutable catalogs first, then new immutable runs,
and the mutable index last. Do not overwrite existing run/catalog filenames;
retain resources referenced by older cached indexes. Site-setting changes belong
to a website rebuild, not the publication snapshot. Read back and validate the
exact published snapshot and verify the real consumer's fetch/reload behavior. Shape validation
cannot attest execution semantics, experiment equivalence or provenance.

Use the [publication checklist](website-data-contract.md#13-publication-checklist)
before releasing a dataset.

## 5. Bundled-site-config rollout

This is an explicit release procedure, not a record of a live deployment. No live
deployment was performed as part of the refactor.

1. Inspect the existing publication and stage a copy; a schema number declares
   format, not provenance. New runs must declare numeric `schemaVersion: 2` and
   use the two branch-class directories. Do not stamp old target groups as schema 2.
2. Historical schema-1 Vanilla runs are migrated by the website: `targets[].id`
   becomes `configurations[].target`, catalog target memberships become `<target>:MT`,
   and all legacy results retain their values and become MT. Historical runs lacking
   a version are treated as schema 1, then validated against the legacy targets/plugin envelope. Non-Vanilla
   plugin files are excluded. Old flat `runs/<id>.json` paths remain supported for
   legacy schema-1 files; new writes never use that layout.
3. Unversioned configurations-style records are not presumed to be schema 2.
   Validate their actual format before explicitly creating versioned schema-2 copies
   in the appropriate new directory and changing index references. Keep old files
   unchanged for cached clients. Unknown versions and invalid Vanilla data reject.
4. Keep publication-side legacy `metadata.json` available for cached old clients.
   The new browser, local tools and publisher neither require nor read it, and the
   publisher does not automatically delete it. The website uses its bundled
   `src/config/metadata.json` instead.
5. Run `npm run validate:data -- <staged-data-root>` and require exit status 0.
   Publish catalogs and runs before the updated unversioned index. Read back and
   validate the snapshot, then deploy the rebuilt website and verify loading,
   reload and raw export against that real publication separately from fixture tests.
