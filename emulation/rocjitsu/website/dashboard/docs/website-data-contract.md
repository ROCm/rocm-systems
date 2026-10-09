# Rocjitsu Simulation Performance Data Contract

The dashboard consumes published benchmark attempts, immutable workload catalogs
and an index manifest. Site settings are bundled separately with the website.
This contract defines the publication JSON fields, validation
rules and derived consumer behavior. Producers must record explicit ST/MT
configurations and actual measurements; the dashboard does not infer execution
mode or generate benchmark results.

The frontend provides a validator, loader, normalizer and local processing tools.
Benchmark execution and data publication are separate responsibilities. Examples
below and repository fixtures are fictional test input, not publishable results.

For workflow authors, start with the [data-generation guide](data-generation-guide.md).
This document is the authoritative field and consumer-behavior reference.

Implementation references (paths relative to this document):

- [Validator and normalizer](../src/data/dashboardValidation.js).
- [Bundled metadata](../src/config/metadata.json) and [site configuration](../src/config/siteConfig.js).
- [Supported run versions and schema-1 migration](../src/data/runSchema.js).
- [HTTP loader](../src/data/dashboardData.js) and
  [execution/commit ordering](../src/data/runOrdering.js).
- [Canonical selectors](../src/data/selectors.js),
  [branch selectors](../src/data/branchSelectors.js), and
  [generic provenance](../src/data/provenance.js).
- [Data-source resolver](../scripts/dashboard-data-source.mjs) and
  [browser URL resolver](../src/data/publishedDataUrls.js).
- [Directory validator](../scripts/validate-dashboard-data.mjs) and
  [local processor](../scripts/process-dashboard-data.mjs).

## 1. Published layout and hosting

Benchmark JSON belongs to the `gh-pages-rocjitsu` branch of `ROCm/rocm-systems`,
under `rocjitsu-dashboard/data/`. The default production data base is exactly:

```text
https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/
```

Application assets belong separately to `gh-pages/rocjitsu-dashboard/`. A website
build is not a data publication. Ordinary production builds do not bundle
benchmark JSON. Explicit fixtures mode serves fictional test data; local-data
mode mounts a supplied directory instead of copying it into production assets.
A configured local data directory changes the resolver's source, so clear local
overrides when verifying the production boundary.

The following paths are relative to the **data root**, not the repository root:

| Path | Contents | Mutability |
| --- | --- | --- |
| `index.json` | Publication timestamp and reachable run filenames; no schema version | Mutable snapshot |
| `test-catalogs/<catalog-id>.json` | Workload definitions and target/mode membership | Immutable |
| `runs/default-branch/<run-id>.json` | Versioned attempt whose source branch is `develop` | Immutable |
| `runs/side-branches/<run-id>.json` | Versioned attempt from any other source branch; flat directory | Immutable |

New run paths must match `^runs/(default-branch|side-branches)/[A-Za-z0-9._-]+\.json$`;
catalog paths must match `^test-catalogs/[A-Za-z0-9._-]+\.json$`. The branch-class
directory must agree with `source.branch`, and the filename must be `${run.id}.json`.
No per-branch subdirectories, absolute URLs, query strings, encoded traversal or
`../` traversal are allowed. Historical schema-1 runs may retain flat
`runs/<id>.json` paths; this is a read-only compatibility rule, not a new-write layout.
A catalog's `id` must equal the filename stem after `test-catalogs/` and before
`.json`. IDs are ordinary tokens, **not hashes or content-addressed IDs**. The
validator does not prescribe a hash algorithm, hash prefix or digest length for
run/catalog identity, and does not compute a catalog semantic hash.

The index is the reachability boundary: only listed runs and catalogs referenced
by those runs are loaded and validated. Orphan files and unreferenced catalogs
are not scanned, validated or reported. Publish immutable catalogs first, then
runs, and publish the new index last. Do not mutate existing filenames. Retain
previously published files even when a newer index omits them, because cached
older indexes may still reference them. Immutability/retention is a publisher
obligation; validation of one snapshot cannot prove it.

### Loading, cache and failure behavior

The browser fetches only `index.json` with `cache: 'no-store'` before loading its
reachable runs and catalogs; it never fetches `metadata.json`. Immutable run/catalog
requests normally use `cache: 'force-cache'`. Where host configuration permits,
revalidate mutable resources and give immutable resources long-lived immutable
cache headers; the website build itself does not configure hosting headers.

**Reload all data** cache-busts the index URL and every reachable immutable URL
with one `reload` token and requests immutable resources with `cache: 'reload'`.
The successful cache generation can be persisted for subsequent loads. This is
a recovery mechanism, not permission to overwrite immutable records.

The default loader uses eight concurrent immutable-resource requests, a 20-second
per-request timeout and a 60-second overall load timeout. It retries network/body
read failures and HTTP 408, 429 and 5xx responses twice, with jittered delays
based on 250/750 ms or a `Retry-After` delay bounded at five seconds. Malformed
JSON and schema errors are not retried as transient failures.

Loading fails closed: one invalid, missing or unreadable indexed run, or a bad
referenced catalog, rejects the whole snapshot. No partial history is displayed.
The explicit product-policy exception is non-Vanilla plugin files: their
measurements are excluded, never relabeled as Vanilla; plugin-only catalogs are
not requested. Raw indexed plugin files remain in the export.
Missing index, transient unavailability, and invalid published content
remain distinguishable loader errors, not measured benchmark failures. An empty
valid index and a valid branch-only snapshot are accepted; neither requires a
canonical develop run. A failed load/reload does not expose stale downloadable
measurements.

## 2. Value conventions and URL scope

Unless stated otherwise:

- **Nonempty text** means a string whose trimmed value is nonempty. Validation
  checks emptiness but does not trim or otherwise normalize the stored string.
- **Scalar** means a string (including an empty string), boolean, or finite
  number. It excludes `null`, arrays and objects. Numeric zero and boolean false
  are preserved.
- **Object** means a non-null JSON object, not an array. An empty object is allowed
  for `problem`.
- Optional fields may be omitted. Supplying `null`, an empty string or a wrong
  type is not equivalent to omitting a constrained optional field.
- Timestamps require a calendar-valid `YYYY-MM-DDTHH:mm:ss`, optional fractional
  seconds, and either `Z` or an explicit `+HH:mm`/`-HH:mm` offset. Date-only values,
  timezone-free strings, impossible dates, hour 24 and leap-second 60 are rejected.
  Offset hours/minutes are checked at 0–23/0–59. Ordering compares parsed instants.
- Source and recorded-base commit fields require a primitive lowercase string
  matching `^[0-9a-f]{40}$`, including on normalized reload. Uppercase/mixed-case
  spellings, arrays, coerced values and abbreviated SHAs are rejected. The preparer
  canonicalizes valid input SHAs before publication; the website does not silently
  lowercase wire values. Identity comparisons use these already-canonical strings;
  validation does not verify Git-object existence.

The validator is not a blanket unknown-property rejector. Extra properties are
not generally forbidden, but their presence is not a supported extension or a
promise that normalization preserves them. Result properties `exitCode` and
`findings` are explicitly rejected even if their value is `null`. Publish the
fields specified here, not UI aggregates.

| URL field | Implemented validation |
| --- | --- |
| `siteConfig.repository` | Parseable absolute HTTP or HTTPS URL with no username/password; not restricted to GitHub |
| `source.pullRequest.url` | HTTPS, hostname exactly `github.com`, no username/password, pathname ending exactly `/pull/<number>` for the accompanying PR number |

GitHub URLs are **not** required to refer to `siteConfig.repository`; there is no
owner/repository allowlist, API lookup or existence check. Query strings and
fragments are not forbidden. The check uses the URL's hostname, not a separately
restricted port. HTTP GitHub URLs, other GitHub-related hosts, credentialed URLs
and non-HTTP schemes fail these optional GitHub-link checks. Passing a URL check
proves its accepted shape, not that its destination exists or is authoritative.

## 3. Bundled site configuration

Source-controlled `src/config/metadata.json` contains:

```json
{
  "schemaVersion": 2,
  "repository": "https://github.com/ROCm/rocm-systems",
  "isBeta": true,
  "canonicalBranch": "develop"
}
```

`schemaVersion` is the source of truth for the target **run** format the website
wants to consume, not a version of metadata itself. `src/data/runSchema.js` reads
that value; supported input formats and explicit migrations remain in website code.
Changing the target does not implement a new format or relabel existing measurements.
A run without an implemented migration to the requested target fails explicitly.
`src/config/siteConfig.js` exports the site settings as `DASHBOARD_SITE_CONFIG`
without checking a metadata format version. Production, fixture and local-data builds
bundle this same JSON. Changing settings requires rebuilding the website, not
editing benchmark data. Local validation and processing use the same defaults.
A publication-side `metadata.json` is never fetched, read, generated or required;
a legacy copy is ignored, not automatically deleted. Keep it for cached old
clients during the [rollout](data-generation-guide.md#5-bundled-site-config-rollout).
Only metadata.json selects the target run version and run files declare their
actual versions; index and catalog files have no schema version.

| Field | Requirement |
| --- | --- |
| `repository` | Required URL satisfying the scope above |
| `isBeta` | Required boolean; retained in normalized data, not raw exports, even if no Beta badge is rendered |
| `canonicalBranch` | Required string exactly `develop` |

`canonicalBranch` defines long-term history, not a restriction on every run's
source branch. These settings are not wire-publication fields and do not appear
as synthetic metadata or configuration in raw downloads.

## 4. `index.json`

Illustrative index for the catalog and run examples below:

```json
{
  "generatedAt": "2026-10-03T12:00:00.000Z",
  "runFiles": [
    "runs/side-branches/illustrative-branch-01.json"
  ]
}
```

- No `schemaVersion`: the index is a version-independent manifest. Each run
  declares its own version; an old index-level version is ignored and never
  substitutes for a run's version.
- `generatedAt`: required strict timestamp for the publication snapshot.
- `runFiles`: required array of unique run paths matching the pattern above.
  An empty array is valid. Array order binds each loaded raw run to its filename;
  it is not the dashboard's execution or commit chronology.
- Every indexed attempt must have completed by `generatedAt`. Validation does not
  compare publication time to the viewer's wall clock.

## 5. Immutable test catalogs

A catalog is a complete snapshot, not a patch. This illustrative catalog belongs
at `test-catalogs/illustrative-catalog.json`:

```json
{
  "id": "illustrative-catalog",
  "tests": [
    {
      "id": "illustrative-gemm",
      "suite": "Illustrative suite",
      "name": "Fictional GEMM workload",
      "problem": { "operation": "GEMM", "size": 16 }
    },
    {
      "id": "illustrative-decode",
      "suite": "Illustrative suite",
      "name": "Fictional decode workload",
      "problem": { "tokens": 8, "cached": false }
    },
    {
      "id": "illustrative-transfer",
      "suite": "Illustrative suite",
      "name": "Fictional transfer workload",
      "problem": {}
    }
  ],
  "configurations": {
    "gfx1250:ST": [
      "illustrative-gemm",
      "illustrative-decode",
      "illustrative-transfer"
    ],
    "gfx1250:MT": ["illustrative-gemm"]
  }
}
```

| Field | Requirement |
| --- | --- |
| `id` | Required string exactly matching the catalog filename stem |
| `tests` | Required array of definitions with unique IDs |
| `tests[].id` | Required nonempty logical workload identity, independent of target and threading mode; no token/hash pattern imposed |
| `tests[].suite` | Required nonempty grouping text |
| `tests[].name` | Required nonempty display name |
| `tests[].problem` | Required object; each key is nonempty text and each value is scalar |
| `configurations` | Required nonempty object mapping configuration keys to workload-ID arrays |

A configuration key must match `^([A-Za-z0-9._-]+):(ST|MT)$`: one target token,
then a colon and the exact uppercase mode. Every membership array must be
nonempty, contain unique IDs, and reference definitions in `tests`. Catalogs
need not include all of `gfx950:ST`, `gfx950:MT`, `gfx1250:ST`, `gfx1250:MT`;
target tokens are not restricted to those architectures. Different configurations
may have different workload membership. Unreferenced definitions are currently
accepted; there is no orphan-definition rejection rule.

Across every referenced catalog, reusing a workload ID requires identical
`suite`, `name` and scalar `problem` facts (problem-key ordering is irrelevant).
Changing any of those facts requires a new workload ID, including a display-only
rename. Configuration membership may change without renaming an unchanged
workload. Publish a new catalog ID whenever the snapshot changes. The normalized
catalog is the union of definitions from referenced catalogs, not a mutable
"current catalog" file. A run's raw coverage denominator remains the membership
of its own included configurations, not the size of a later catalog.

## 6. Immutable run files

Each new run requires numeric `schemaVersion: 2`. Supported versions and explicit
migrations live in website code, not the index or external metadata.

A run is one uniquely identified published attempt for a source revision,
machine and environment, with one or more explicitly measured
configurations. It may include a subset of its catalog's configurations. It need
not represent all targets/modes in the catalog or an entire four-configuration
workflow. An absent configuration means **not published**, not success, failure
or zero.

This illustrative run belongs at `runs/side-branches/illustrative-branch-01.json`. All duration
values and diagnostics are fictional; the zero demonstrates a valid measured
value, not a real zero-runtime claim. The GitHub URLs demonstrate syntax only.

```json
{
  "schemaVersion": 2,
  "id": "illustrative-branch-01",
  "testCatalog": "test-catalogs/illustrative-catalog.json",
  "source": {
    "branch": "illustrative/optimization",
    "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "committedAt": "2026-10-02T09:00:00.000Z",
    "message": "FICTIONAL contract example",
    "base": {
      "branch": "develop",
      "commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    "pullRequest": {
      "number": 17,
      "url": "https://github.com/ROCm/rocm-systems/pull/17"
    }
  },
  "execution": {
    "completedAt": "2026-10-02T10:00:00.000Z",
    "trigger": "manual",
    "machine": "illustrative-runner"
  },
  "environment": [
    { "key": "sdk", "label": "Illustrative SDK", "value": "example-only" },
    { "key": "feature", "label": "Feature enabled", "value": false },
    { "key": "count", "label": "Illustrative count", "value": 0 },
    { "key": "note", "label": "Optional fact", "value": "" }
  ],
  "configurations": [
    {
      "target": "gfx1250",
      "threadingMode": "ST",
      "results": [
        {
          "testId": "illustrative-gemm",
          "status": "completed",
          "durationSeconds": 0,
          "error": null
        },
        {
          "testId": "illustrative-decode",
          "status": "failed",
          "durationSeconds": null,
          "error": "Fictional failure diagnostic"
        },
        {
          "testId": "illustrative-transfer",
          "status": "timeout",
          "durationSeconds": null,
          "error": null
        }
      ]
    },
    {
      "target": "gfx1250",
      "threadingMode": "MT",
      "results": [
        {
          "testId": "illustrative-gemm",
          "status": "completed",
          "durationSeconds": 1.25,
          "error": null
        }
      ]
    }
  ]
}
```

### Run identity

| Field | Requirement |
| --- | --- |
| `id` | Required globally unique primitive-string attempt ID matching `^[A-Za-z0-9._-]+$`; filename must match; not a commit SHA |
| `testCatalog` | Required relative catalog path matching the catalog pattern |

Attempt IDs must be primitive strings before the token pattern is applied.
Numbers, booleans, arrays (including single-element string arrays), and objects
are rejected rather than coerced to strings. This also applies to `runId` in
normalized reloads; valid string IDs retain their exact value for URL selection.

Each rerun receives a new run ID. Attempts are selected by this ID, not by
source SHA or a target/mode batch identity.

### `source`

| Field | Requirement |
| --- | --- |
| `branch` | Required nonempty string; develop and non-develop branches are accepted |
| `commit` | Required primitive string containing a full lowercase 40-hex source SHA |
| `committedAt` | Required strict timestamp of that source commit |
| `message` | Optional nonempty string |
| `base` | Optional object with required nonempty string `branch` and primitive string full lowercase 40-hex `commit` |
| `pullRequest` | Optional object with required positive integer `number` |
| `pullRequest.url` | Optional GitHub HTTPS URL with the matching `/pull/<number>` pathname suffix |

The base describes the recorded source base; it is not a measured baseline or
proof that the commit is published. The base branch need not be develop.
PR metadata and base metadata are independent and optional. The validator does
not resolve Git ancestry or require branch-head/PR membership. Historical reruns
retain the original source timestamp. Across the dataset the same supplied commit
SHA must have the same parsed `committedAt` instant, even on different branches.

### `execution`

| Field | Requirement |
| --- | --- |
| `completedAt` | Required strict timestamp of attempt completion |
| `trigger` | Required enum exactly `auto` or `manual` |
| `machine` | Required nonempty machine identifier |

The following ordering is enforced, with equality permitted:

```text
source.committedAt <= execution.completedAt <= index.generatedAt
```

Independent runs may use different branches, machines and environments. There
is no publication-wide machine or environment equality policy. These differences
remain available for disclosure; accepted input is not a guarantee of experimental
equivalence. Target and simulator threading mode belong in `configurations`, not
in the execution identity or generic environment as inferred labels.

### Generic `environment`

`environment` is required and may be an empty array. Each item has required
nonempty `key` and `label`, and a required scalar `value`. Keys must be unique
within a run; labels may repeat. Raw values, including `""`, zero and false, are
preserved. Display formatting may stringify scalars or show booleans as Yes/No.
There is no mandatory fixed SDK/compiler/version key list.

Environment identity is a JSON string of sorted `[key,value]` pairs, ordered by
key with `localeCompare`. Labels and item order do not affect compatibility;
scalar type and value do. Two empty environments are compatible; an empty and
populated one are not. This identity is **not a hash or attestation**.

### `configurations` and result rules

`configurations` is a required nonempty array. Each included item contains:

| Field | Requirement |
| --- | --- |
| `target` | Required nonempty string; `${target}:${threadingMode}` must match the configuration pattern and exist in the catalog |
| `threadingMode` | Required string exactly `ST` or `MT`; no type coercion |
| `results` | Required array with exactly one result for each workload in this catalog configuration |

A target/mode pair may appear at most once in a run. Configuration order and
result order are not significant for membership. No included configuration may
omit, duplicate or add a workload relative to its catalog membership. Entire
configurations may be absent; individual required result omissions are invalid.

Comparison exclusions distinguish each side independently:

- **No run selected**: there is no selected published attempt on that side.
- **Configuration not published**: the attempt does not include the exact
  target/mode pair. Check this before reporting catalog absence, even when the
  referenced catalog defines that configuration or omits the compared workload.
- **Unavailable in catalog**: the configuration is published, but the compared
  workload is absent from that configuration's catalog membership.
- **Published result status**: an existing result retains its `completed`,
  `failed` or `timeout` status.

For example, comparing an ST-only attempt against an ST+MT attempt reports
unpublished configuration for the ST-only side's MT rows, in either comparison
direction. Neither an unpublished configuration nor catalog absence is a failed
result or a zero-duration measurement.

`threadingMode` is an **explicit declaration of simulator threading**, never
inferred from workload names, `.threads1`/`.threads8` suffixes, sample counts,
warmup counts, functional/clocked execution mode or another configuration. The
validator checks its string type and exact enum before constructing the
target/mode key; arrays such as `["ST"]` or `["MT"]` are invalid even if string
coercion would resemble a supported mode. The producer must truthfully declare
the actual execution semantics; the JSON validator cannot verify them. A run
may include any measured subset of gfx950/gfx1250 × ST/MT configurations; all
four are not required in one file. This does not define the sampling protocol.

Every result has all four fields:

| Field | Requirement |
| --- | --- |
| `testId` | Required nonempty logical workload ID, matching catalog membership |
| `status` | Required enum exactly `completed`, `failed` or `timeout` |
| `durationSeconds` | Required finite nonnegative number for completed; required `null` for failed/timeout |
| `error` | Required `null` for completed; `null` or nonempty string for failed/timeout |

`completed` with `durationSeconds: 0` is a valid measured value. Negative values,
nonfinite numbers, numeric strings and a null completed duration are invalid.
Failed/timed-out results must stay in the matrix with null seconds; diagnostics
are optional in content but the `error` field itself is mandatory. There is no
published `missing` status. `exitCode` and `findings` are removed fields.

Raw result identity is `(run id, target, threadingMode, logical testId)`. Comparison
matching requires the exact target, mode and logical workload; ST and MT are
never paired implicitly.

### Legacy schema-1 migration

The website supports run schemas 1 and 2 explicitly. Historical schema-1 files
without a `schemaVersion` field are treated as schema 1 and passed to the explicit
migration. The migration still requires the legacy targets/plugin contract; it does
not guess missing fields. Unversioned configurations-style runs fail schema-1
validation, and explicitly unknown versions are errors, not implicit upgrades.

- A nonempty `plugin.id` other than `vanilla` excludes the whole file from the
  normalized dataset. Raw indexed files are retained in downloads. Such files
  cannot contribute ordinary timings or workload definitions, and plugin-only
  catalogs are not loaded.
- A Vanilla schema-1 file requires its legacy `comparisonId`, plugin name,
  target groups and matching catalog `targets` memberships. Ambiguous mixed
  target/configuration envelopes and incomplete results fail validation.
- The migration maps every `targets[].id` to `configurations[].target`
  with `threadingMode: "MT"`, and every catalog target membership to `<target>:MT`.
  Result values, diagnostics, workload definitions and provenance remain unchanged.
  No durations are guessed, discarded or rewritten. Legacy completed durations
  must remain positive as required by schema 1; schema 2 permits measured zero.
- Migration is in memory. It does not edit immutable source files or rewrite
  downloaded raw data. The ordinary current validator checks migrated results;
  malformed Vanilla measurements fail the whole snapshot.
- Version-1 and version-2 runs may coexist in one manifest. Newly prepared runs
  always use schema 2 and the branch-class directory layout.

## 7. Normalized and export envelopes

`validatePublishedManifest(index)` validates the index manifest.
`validatePublishedDashboardData({index,runs,catalogs,siteConfig?})` returns
`{ data, sourceData }`; omitted `siteConfig` defaults to `DASHBOARD_SITE_CONFIG`,
as does normalization. The HTTP loader
adds a consumer-only `cacheGeneration` field alongside those two keys. Neither
normalized data nor cache state is a new wire-publication format.

### `sourceData`: raw download

The dashboard's **Download JSON** exports this envelope, unfiltered:

```text
{
  index: <index.json object>,
  catalogs: { "test-catalogs/<id>.json": <referenced catalog object>, ... },
  runs: [<raw indexed run objects, in index order>, ...]
}
```

It preserves indexed develop/branch attempts, optional source
and execution metadata, generic environments, configurations, diagnostics and
zero durations. Schema-1 files remain unchanged here, including excluded plugin
files; exclusion is a display policy, not validation of those files' measurements.
It includes fetched referenced catalogs (not plugin-only catalogs), and no synthetic metadata
or bundled site configuration. It is not the filtered
chart, the comparison's matched subset, or `{data,sourceData}` from processing.

### `data`: normalized consumer model

| Field | Meaning |
| --- | --- |
| `schemaVersion` | Current normalized consumer schema (`2`), after migrations |
| `repository`, `isBeta`, `canonicalBranch` | Site settings retained separately from the raw source envelope; default to bundled configuration |
| `generatedAt` | Index publication timestamp |
| `catalogs` | Validated, migrated catalog envelopes keyed by relative path; required for reloading every included run |
| `testCatalog` | Union of referenced catalog definitions, keyed logically by workload ID |
| `allRuns` | All included Vanilla attempts, including non-develop branches, ascending execution order; plugin runs excluded |
| `runs` | Develop attempts only, ascending execution order |
| `targets` | Sorted unique targets present in `allRuns`, not all catalog-only targets |
| `suites` | Sorted unique suites in the union `testCatalog` |
| `modes` | Supported mode names `['ST','MT']`; not proof both were published |
| `latestRun` | Last canonical attempt in execution order, or `null` |
| `latestCommitRun` | Last canonical attempt in commit order, or `null` |
| `backfillRunIds` | Runtime `Set` of canonical attempt IDs executed after an attempt of a newer commit |

A normalized run contains:

| Field | Origin/meaning |
| --- | --- |
| `runId`, `testCatalog`, `catalogId` | Raw attempt identity, catalog path and catalog ID |
| `timestamp`, `commitTimestamp` | Completion time and source commit time, respectively |
| `trigger`, `machineId`, `branch` | Raw execution trigger/machine and source branch |
| `sourceBase`, `pullRequest` | Present only when the corresponding optional raw metadata exists |
| `environmentId` | Sorted key/value JSON identity described above |
| `provenance.rocjitsuCommitSha` | Full lowercase source commit SHA |
| `provenance.commitMessage` | Optional source message |
| `provenance.details` | Raw generic environment items |
| `targets`, `modes` | Targets actually included; included modes in ST/MT order |
| `configurations` | Included `{target,threadingMode}` descriptors, without raw result arrays |
| `tests` | Flattened definition/result records with explicit configuration identity |

A normalized test carries definition fields `id`, `suite`, `name`, `problem`,
result fields `status`, `durationSeconds`, `error`, and:

```text
logicalTestId = raw result.testId
testId        = "<target>:<mode>:<logicalTestId>"
target        = configuration.target
mode          = configuration.threadingMode
```

Its definition `id` remains the logical workload ID; normalized `testId` is the
configuration-specific matching key. Catalog definitions are authoritative:
normalization takes only `testId`, `status`, `durationSeconds` and `error` from a
raw result, then derives the configuration identity. Extra result properties
cannot override catalog `id`, `suite`, `name` or `problem`, change suite filtering,
or hide a required workload from coverage. Accepted result extensions remain in
`sourceData`/raw download, not as overrides in normalized tests.

The local processor serializes runtime `backfillRunIds` as an array. To rehydrate,
pass **`processed.data`**, not the entire processing envelope, to
`loadDashboardData`. That function chooses the normalized source array in priority
order `allRuns`, then `runs`; reconstructs raw runs; checks
normalized test identities and duplicate attempts; and routes the reconstruction
through the same wire validator. Every run's referenced authoritative catalog
envelope must be present in `data.catalogs` and pass catalog validation. Missing
catalogs are never synthesized from surviving normalized tests, the union
`testCatalog`, or result membership. An absent/empty catalog map with referenced
runs, a missing referenced envelope, or an omitted workload required by an
included catalog configuration rejects the reload; an empty snapshot references
no catalogs. Derived summaries, latest pointers, environment identity and the
backfill Set are rebuilt, not trusted as authoritative input. Retain the full
processed envelope for lossless raw-source provenance; do not treat rehydration
as byte-preserving raw export.

## 8. Chronology and population boundaries

**Execution order** is ascending parsed completion timestamp, with `runId`
`localeCompare` as the stable tie-break. "Latest execution" includes late reruns
of old commits. Index order is not used as chronology.

**Commit order** uses source commit time, then SHA text for distinct commits at
identical times. Attempts of the same SHA share one commit position and are
ordered by execution time/attempt ID within it. "Latest commit" and "latest
execution" may identify different runs. A backfill is an attempt executed after
an attempt from a newer commit; it does not move the old commit to the newest
position in history.

| Consumer | Population and order |
| --- | --- |
| Overview and long-term benchmark history | `data.runs`: develop only; commit chronology |
| Recent Runs | All canonical develop attempts by completion, newest first, paginated in groups of 20 with numbered pages and icon-only first/previous/next/last controls; selected target/suite/mode filters affect coverage and duration, not the run population; rows show coverage, duration and status without baseline comparisons or deltas |
| Benchmark result inspector | Searchable choices derived from canonical develop chart series in the selected timeframe and target/suite/mode scope, including failed, timed-out and unavailable results; no fixed-size record pages |
| General Run Comparison | Selected attempts may come from `data.allRuns`, including branches |
| Branch Runs | Published non-develop attempts, grouped by source branch |

Arbitrary Run Comparison consumes the explicit candidate/reference pair; it does not
require equal machines/environments or decide that differences are harmless.

### Retained selector APIs

The current Recent Runs table uses `selectRecentRunAttempts` and
`selectRecentRunSummaries`; the benchmark result inspector uses
`selectBenchmarkSeries` and `benchmarkResultChoices`. Their implementations are
in [selectors.js](../src/data/selectors.js) and
[benchmarkExplorer.js](../src/components/benchmarks/benchmarkExplorer.js).
Do not infer retired baseline columns or record pagination from the retained
API behavior below.

- `selectRecentRuns` retains baseline and duration-delta calculations, with a
  default limit of 20 attempts ordered newest by completion. Its baseline is
  the nearest earlier **commit position** with completed selected tests and a
  completed exact match for every selected candidate ID; latest attempt wins
  within that earlier commit.
- `selectBenchmarkRecords` retains completion-ordered record pagination, with a
  default page size of 25. Its baseline is the nearest earlier commit with a
  completed result for that exact configuration/workload.

Neither retained baseline means "the last file published." These APIs are
separate from the mounted summary table and searchable result inspector.

## 9. Scope, zero, failure and comparison calculations

Selector filters are `{ targets: string[], suites: string[], modes?: string[] }`.
The intersection is exact. Explicit empty arrays mean empty selections, not
"all". Omitting `modes` means both ST and MT, including Benchmark Explorer
availability; it never synthesizes missing mode results. `selectBenchmarkCatalog`
is the single availability authority; the explorer's `explorerCatalog` delegates
to it. A catalog workload is available when at least one result in the supplied
canonical runs matches the selected target, suite and mode, regardless of result
status. For direct callers that omit result-level suite metadata or supply null,
the selector resolves the suite from the matching catalog definition; an explicit
result suite retains precedence. It preserves catalog order and returns
`{ all, available, hiddenCount }`. Availability does not derive from `data.modes` alone.

Pairwise comparison uses the union of selected candidate and reference IDs:

- Both results completed with finite seconds: a comparable pair, including zero.
- A failed/timeout result or a one-sided missing ID: excluded, retaining the
  candidate/reference facts. Baseline-only workloads are not silently omitted.
- Matched sums include only comparable pairs. If there are no pairs, both sums
  and the aggregate percentage are `null`, not zero. Coverage/exclusions must
  accompany a partial matched sum; it is not the full selected workload total.

```text
deltaSeconds = candidateSeconds - baselineSeconds
deltaPercent = (candidateSeconds - baselineSeconds) / baselineSeconds * 100
```

Percentage change is available only for a positive finite baseline. A zero
baseline remains matched and has an absolute-seconds delta, but its percentage
and faster/slower classification are unavailable. Aggregate percentage uses the
matched sums and is likewise unavailable when the baseline sum is zero. With the
default tolerance, changes greater than +3% are slower, below -3% faster, and
inclusive -3% to +3% neutral. These are display classifications, not a statistical
confidence statement.

Raw coverage counts selected result records from the run's included catalog
configurations. Completed, failed and timeout remain distinct; overall failed
presentation may combine failed+timeout while raw counts retain both. Source
coverage does not count estimated history values as measured successes. A whole
missing configuration is unavailable and does not become a collection of failed
results or a fabricated denominator. For simple selected-run summaries, an empty
selection or any selected failure gives no full selected duration; absent
configurations are separately inspectable. Overview's adjusted sum uses the
stricter missing-configuration rule below.

Benchmark series are separated by target+mode. A completed zero is a point;
failed/timeout/absent results are gaps, with available diagnostics retained in
records. Catalog normalization is not applied to individual benchmark series or
to arbitrary selected-pair comparison totals.

## 10. Overview chronology and catalog adjustment

Overview exposes **one selected-scope sum**, named `Selected runtime`, not one
line per target. Its cards and history duration/delta use the same period
endpoints and adjusted workload. No authoritative aggregates, baselines, coverage,
chart labels or deltas are published in the wire JSON.

### Slots and endpoints

The latest-commit canonical run supplies the UTC anchor commit day. Current
workload catalogs are selected independently per target/mode as described below;
there is no single global reference catalog. Supported ranges are `1D`, `1W`,
`1M`, `3M`, `6M`, `YTD`, `ALL`:

- `1D`: latest execution per source commit on that UTC day, retaining up to the
  last 56 commits. No automatic previous-day baseline.
- `1W`: seven UTC days ending on the anchor day, up to eight latest commit
  representatives per day; missing days remain slots without runs.
- `1M`/`3M`/`6M`: 30/90/180 UTC days; `YTD`: January 1 through the anchor day;
  `ALL`: earliest canonical commit day through the anchor day. These use the
  latest attempt of the latest commit per day, **not a search for a successful
  replacement** when that representative fails.

The first finite adjusted slot is the period baseline. The last represented run
is the candidate, even if its adjusted sum is unavailable. Cards/history agree
on this candidate's value; a final failure is not skipped in favor of an older
success. There is no percentage for a single endpoint or a zero baseline.
Per-test Largest Changes uses raw completed pairs between those endpoint runs;
its workload-level comparisons are not substituted anchor measurements.

For ranges other than `ALL`, `1D`, `1W`, fewer than 75% represented calendar days
sets `history.insufficientData`. Representation counts days with a run, not just
finite sums. This is a coverage flag; the selector does **not** automatically
clear an otherwise available endpoint percentage because of this flag.

### Current-workload projection and first-success anchors

1. For each selected target/mode independently, choose the **latest canonical
   attempt in commit chronology whose referenced catalog declares that exact
   configuration key**. Within the same commit, execution time and attempt ID
   break ties. The attempt need not itself publish that configuration. Use its
   catalog's membership, filtered by selected suites, as that pair's current
   workload. A newer commit's ST-only catalog does not erase historical MT scope:
   MT retains the latest catalog that declares MT. A late execution of an older
   commit does not override a newer-commit catalog. Only a pair with no declaring
   catalog in canonical history, or no members in the selected suites, contributes
   no workload. Deleted workloads disappear according to each pair's chosen
   catalog, even if older runs measured them.
2. Scan canonical develop attempts in ascending completion time, then
   attempt-ID order. For each current-catalog workload and exact target/mode,
   record the **first completed finite measurement in that pair's chosen current
   catalog**.
   Zero qualifies. This scan covers canonical history, not just the displayed
   range; an anchor can occur after an older projected run. A later success must
   not replace that anchor. Each target/mode/workload has its own anchor.
3. For a projected run, retain its measured values for current workloads that it
   actually completed. If a current workload is absent from a different/older
   catalog's membership, add its first-success anchor value as an **estimate**.
   This applies only when that target/mode configuration itself was published.
4. Do not substitute an anchor for a failed/timeout measurement, a missing result
   that its catalog required, a whole absent configuration, a missing workload in
   that pair's same current catalog, or a workload with no successful anchor.
   These give a null adjusted sum/gap, not a smaller total. In particular, when
   historical MT remains in the selected scope but the represented current run
   publishes only ST, the full selected sum and endpoint delta are unavailable;
   MT is not dropped, estimated as a whole configuration, or replaced by zero.
5. Sum the selected projection; do not interpolate gaps or rewrite raw results.
   Anchor additions affect derived history only. Old-catalog raw success counts
   and durations remain facts from that old catalog.

The selector exposes `history.normalized`, per-slot `series[].estimated`,
`history.anchors`, `history.estimates`, and `metrics.estimatedBaseline`.
Anchor records identify `testId`, `logicalTestId`, target, mode, duration,
`runId`, `catalogId`, completion `timestamp`; applied estimates additionally name
`estimatedRunId`. Consumers must disclose estimated values and anchor identities,
not label them measured historical results. General pair comparisons and raw
exports do not receive these adjustments.

## 11. Published branches and automatic references

`selectPublishedBranches(data, { query: '', pr: 'all' })` uses publication time,
not the viewer's current date. A branch is active when its latest published
execution lies in the inclusive interval from **30 days before
`data.generatedAt` through `data.generatedAt`**. Develop and future completions
are excluded. An active group's `runs` retains **all published attempts for that
branch**, including older attempts outside the active window, newest first.
Groups are sorted newest execution first with the same attempt-ID tie-break.

Search is case-insensitive over branch names, any published attempt's SHA, and
PR numbers (an initial `#` is removed for the PR-number search). PR filtering
uses the latest run's metadata: `all`, `with-pr` or `no-pr`. Retained aliases
`with`, `without-pr`, `without` have the corresponding meaning. Returned groups
are `{branch,runs,latestRun,pullRequest}`, with `pullRequest` taken from the
latest run or `null`; this is published metadata, not live GitHub branch status.

`selectAutomaticReference(data, candidate)` selects only published develop
attempts:

1. If the recorded `sourceBase.branch` is develop and its commit is published,
   choose the **latest execution of that exact base commit**, breaking completion
   ties by attempt ID. This exact-base preference has no candidate-completion
   cutoff: its chosen rerun may complete at or after the candidate.
2. Otherwise choose the latest develop execution **strictly completed before**
   the candidate's completion. Equal completion times are ineligible for this
   fallback. This is execution chronology, not nearest commit date/ancestry and
   not publication-file order.
3. If none exists, return no reference. Do not invent a baseline, pick a branch
   attempt, substitute zero, or imply the candidate passed a comparison.

The return value is `{run,reason,description}`, with reason exactly `exact-base`,
`earlier-develop`, or `unavailable` and `run: null` when unavailable. Selection
does not require successful results or environment equivalence; exclusions and
provenance still apply. A manual reference is external UI/route state, not a
change to this selector or to the run's recorded source base.

`selectConfigurationComparison(candidate, baseline, {target,mode,suites,query})`
uses that exact pair and one exact target/mode, applying the same matched sums,
zero rules and symmetric exclusions as general comparison. Optional search
matches workload name, logical ID or suite. Comparable rows are sorted by
absolute seconds change descending, then normalized test ID. Percentage sorting,
when offered by a UI, is a presentation choice, not a different denominator.

### Branch configuration controls and partial failures

Entering Branch Runs shows every target present in the validated publication's
run configurations, not a fixed architecture list or only the selected pair.
Columns are sorted, with ST then MT rows. An explicitly selected target absent
from the current publication remains visible and unavailable; unavailable cells
are still selectable for inspection. Narrow matrices scroll internally.

Absent target/mode route fields remain uninitialized until data is available.
First use chooses the first sorted target from the candidate's configurations,
then the reference's, then another published attempt if necessary. For that target,
ST is preferred when published in the first applicable attempt; otherwise MT.
Explicit target/mode selections are preserved, including missing identities.

Each matrix card compares only workloads completed in both selected attempts for
that target/mode and the current suite/search scope. Partial failures do not hide
the target: if completed matches remain, the card shows their aggregate change,
"Less time"/"More time"/"Same measured time", and the matched count. This is not an
all-tests-passed status and may show an improvement even when other workloads fail.
Failed/timed-out or missing workloads are excluded, never counted as zero. Select
the card to see matched/excluded counts and diagnostics in Benchmark differences.
With no completed matches, it shows "Unavailable" and "No matched completed
benchmarks"; entirely unpublished configurations have their own unavailable reason.

## 12. Local validation and processing

From `emulation/rocjitsu/website/dashboard`, with the Node version required by
`package.json` and a complete staged **data root**:

```bash
npm run validate:data -- /absolute/path/to/staged/data
# Equivalent direct entry point:
node scripts/validate-dashboard-data.mjs /absolute/path/to/staged/data

# Print the validated {data,sourceData} JSON envelope to stdout:
node scripts/process-dashboard-data.mjs /absolute/path/to/staged/data

# Write a new local processed file outside the input directory:
node scripts/process-dashboard-data.mjs /absolute/path/to/staged/data \
  --output /absolute/path/to/local-output/processed-dashboard.json
```

`npm run process:data -- <data-root>` is also available as the package alias for
the direct processor entry point above. Validation succeeds with exit status 0 and reports indexed run
files and develop attempts; malformed input exits nonzero. Processing
validates the snapshot before emitting output, creates output parents as needed,
and refuses to overwrite an existing output (`wx`). Its output path must be
outside the input directory, not a published wire file. Before creating any
output directory or file, the processor canonicalizes the input root with
`realpath`, resolves the output's nearest existing ancestor with `realpath`, and
appends any nonexistent descendant path. It rejects both lexical containment in
the supplied input path and resolved containment in the canonical input root.
Existing symlink aliases of the input root, output parent or a higher output
ancestor cannot bypass this check, including when output descendants do not yet
exist. A symlinked output directory that resolves genuinely outside the input is
allowed; exclusive `wx` creation still applies. These checks handle existing
filesystem aliases, not race-proof confinement against concurrent adversarial
filesystem mutation, and do not constitute a publisher sandbox.

The directory validator resolves input resources and rejects any indexed resource
whose real path escapes the resolved data root, including an escaping symlink.
It does not scan unindexed files or forbid every symlink/root alias. Both CLI
entry points use the same schema validator as the browser. They do not execute
benchmarks, infer threading, add authoritative measurements, publish or deploy.

## 13. Publication checklist

Before releasing a dataset:

- [ ] Define and independently verify producer ST/MT semantics and measurement
  provenance. Preserve actual sample/warmup protocol; do not
  import fictional fixtures. Declare only measured/published configurations.
- [ ] Generate the fields above from real executions, including
  explicit `threadingMode`, immutable attempt IDs, truthful source/base/PR metadata,
  completion time, generic environment and machine.
- [ ] Use immutable catalog snapshots. Preserve unchanged workload identities;
  give changed definitions new IDs. Record every required workload in each
  included configuration, including failed and timed-out results with null
  seconds, and preserve genuine measured zero.
- [ ] Enforce source-to-completion-to-publication chronology and consistent
  source timestamps. Verify URL destinations and provenance independently where
  needed; passing shape validation is not attestation.
- [ ] Stage the full intended schema-2 index and every reachable run/catalog.
  Audit unindexed/orphan files separately; the CLI cannot certify them. Run
  `npm run validate:data -- <staged-data-root>` with the actual consumer validator
  and require exit status 0 before publication.
- [ ] Exercise normalization/reload/export against that staged **real** snapshot:
  canonical/branch separation, first-success anchors and disclosed
  estimates, missing-mode gaps, zero-baseline percentages, failures, exact-base
  selection and strictly earlier-completion fallback.
- [ ] Publish to `gh-pages-rocjitsu/rocjitsu-dashboard/data/`, not the application
  assets branch. Upload new catalogs and runs before the new index; preserve
  immutable historical resources for cached-index readers.
- [ ] Read back the exact published index and all referenced resources,
  validate the published snapshot, and verify fetch/cache/reload behavior through
  the consumer. A successful upload or passing fixture suite alone is not enough.
- [ ] Keep application release and data release acceptance separate. Passing
  frontend tests does not validate a live publication; verify the published
  dataset independently.
