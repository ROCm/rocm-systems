# Rocjitsu Simulation Performance Data Contract

Schema version 1 uses plain JSON, immutable test catalogs, and one run file per Rocjitsu plugin execution. A run file contains every target measured by that execution; target groups contain result records that reference the shared catalog.

## Published files

| Path | Contains | Changes when |
| --- | --- | --- |
| `data/metadata.json` | Contract version and site-level settings | The contract or site settings change |
| `data/index.json` | Publication time and run filenames | A run is added or removed |
| `data/test-catalogs/<catalog-id>.json` | An immutable test-definition snapshot and target applicability | The planned test set changes |
| `data/runs/<run-id>.json` | One plugin execution across every catalog target | A run is first published |

Production builds fetch these paths from `rocjitsu-dashboard/data/` on the
`gh-pages-rocjitsu` branch of `ROCm/rocm-systems`. The build contains no
benchmark JSON. Fixture builds and local-data servers expose the same paths
under `./data/`, using the same runtime loader and validator. This package does
not define how the application is deployed. Test fixtures are dummy records and
must not be published.

Published benchmark data will be validated and managed by a separate
data-publication workflow. Website development and CI validate application code
and fixtures only; they do not guarantee the availability or validity of
published data.

Upload a new catalog before any run that references it. Upload run files before
publishing the updated index. Existing catalogs and runs are immutable.

## Hosting requirements

The dashboard loads one HTTP request per published run. Where the hosting service
allows cache configuration, use the following recommended policies. The website
build does not configure HTTP response headers.

| URL | Recommended host behavior |
| --- | --- |
| `data/metadata.json`, `data/index.json` | Revalidate on every load, for example `Cache-Control: no-cache`. The browser also requests them with `cache: 'no-store'`. |
| `data/test-catalogs/*.json`, `data/runs/*.json` | Immutable once published; the browser requests cached copies with `cache: 'force-cache'`. Hosts should still send `Cache-Control: public, max-age=31536000, immutable`. |

The dashboard's **Reload all data** action first requests cache-busted
metadata and index URLs. It then requests every run named by that fresh index
and every catalog referenced by those runs with the same per-click cache-busting
token and `cache: 'reload'`. This bypasses both browser and shared CDN cache
entries and stores the fresh immutable files in a new per-browser cache
generation. The generation is persisted only after the complete dataset
validates, so the user's next normal refresh reuses those files without purging
or changing another user's cache. Metadata and index requests remain
`cache: 'no-store'` because they are mutable. This lets a user recover after a
publisher corrects content under an existing filename, although publishing a
new filename remains the required normal practice.

The loader retries transient network failures and HTTP 408, 429, and 5xx
responses twice. It honors a bounded `Retry-After` response or otherwise applies
jittered backoff. Other HTTP failures, malformed JSON, and validation errors
fail immediately. If the retry budget is exhausted, the dashboard fails closed
instead of displaying a partial history.

Publishing a mutated catalog or run under an existing filename is a contract violation:
cached clients can keep the old body until cache expiry. Publish a new ID instead.
Never delete a published catalog or run file: clients can temporarily retain an
older index that still references it. Remove obsolete runs from new indexes while
leaving their immutable files available.

## `metadata.json`

```json
{
  "schemaVersion": 1,
  "repository": "https://github.com/ROCm/rocm-systems",
  "isBeta": true
}
```

- `schemaVersion`: required integer; this contract requires `1`.
- `repository`: required repository URL used for source links.
- `isBeta`: required boolean controlling the header's Beta label.

Catalogs, targets, plugins, and run filenames do not belong in metadata.

## `index.json`

```json
{
  "generatedAt": "2026-09-09T04:15:15.135Z",
  "runFiles": [
    "runs/comparison-123-vanilla.json",
    "runs/comparison-123-asan.json",
    "runs/comparison-123-tsan.json"
  ]
}
```

- `generatedAt`: required ISO-8601 publication timestamp.
- `runFiles`: required array of unique paths relative to `index.json`. Each path must match `runs/<filename>.json`.

The index contains no catalog, target, plugin, or benchmark data.

Validation fails closed. An invalid, missing, or unreadable run listed in the index rejects the whole dataset instead of being skipped; a missing or invalid catalog rejects every run that references it, and therefore the dataset. Invalid dataset metadata or index structure is equally fatal, as is a dataset containing no valid Vanilla run. That last check requires a Vanilla run to exist, not to have completed every result: a valid Vanilla run whose results are failed or timed out still satisfies it.

## Test catalog

A catalog is a complete immutable snapshot, not a diff from an earlier catalog.

```json
{
  "id": "rocjitsu-core-v2",
  "tests": [
    {
      "id": "gemm-fp16-1024",
      "suite": "GEMM",
      "name": "GEMM FP16 1024³",
      "problem": {
        "operation": "GEMM",
        "dataType": "fp16",
        "m": 1024,
        "n": 1024,
        "k": 1024
      }
    }
  ],
  "targets": {
    "gfx950": [
      "gemm-fp16-1024"
    ],
    "gfx1250": [
      "gemm-fp16-1024"
    ]
  }
}
```

### Catalog fields

- `id`: required stable, unique catalog identifier matching `<catalog-id>` in the catalog filename.
- `tests`: required array of benchmark definitions.
- `targets`: required object mapping each target to the exact test IDs applicable to it.

Every test contains:

- `id`: required plugin-neutral benchmark identity.
- `suite`: required dashboard grouping.
- `name`: required display name.
- `problem`: required object of workload facts. Keys must be non-empty; values must be strings, numbers, or booleans.

The dashboard renders every `problem` entry under **Problem Details**.
Every definition in `tests` must be referenced by at least one target. Catalogs with
orphan definitions are rejected instead of silently dropping those definitions from
the dashboard-wide test catalog.

### Catalog evolution

- Never edit a published catalog.
- Add or remove tests by publishing a new catalog ID.
- Retain the same test ID only when `suite`, `name`, and `problem` are all unchanged.
- Use a new test ID for any other edit, including display-only changes to `suite` or `name`.
- Catalog versions may share any number of unchanged tests.
- Runs referencing a five-test catalog remain 5/5 after a seven-test catalog is introduced.

The browser derives the dashboard-wide picker from the union of referenced catalogs. No separate mutable current catalog exists.

Because that union is keyed by test ID, a shared test ID must carry an identical `suite`, `name`, and `problem` in every catalog that defines it. There is no authoritative winner when two catalogs disagree, so the dashboard rejects the whole dataset and names both catalogs rather than silently comparing different workloads. Renaming a published test therefore costs its history continuity; that price buys the guarantee that one test ID always means one workload.

## Run file

A run represents one Rocjitsu plugin execution on one source revision and machine, across all targets required by its catalog.

```json
{
  "id": "comparison-123-asan",
  "comparisonId": "comparison-123",
  "testCatalog": "test-catalogs/rocjitsu-core-v2.json",
  "plugin": {
    "id": "asan",
    "name": "AddressSanitizer",
    "version": "LLVM 20"
  },
  "source": {
    "branch": "develop",
    "commit": "0123456789abcdef0123456789abcdef01234567",
    "committedAt": "2026-09-08T10:42:00.000Z",
    "message": "Improve clocked-mode batching"
  },
  "execution": {
    "completedAt": "2026-09-08T12:00:00.000Z",
    "trigger": "auto",
    "machine": "sjc-rocjitsu-perf-01"
  },
  "environment": [
    {
      "key": "rocmSdkVersion",
      "label": "ROCm SDK",
      "value": "7.2.0"
    }
  ],
  "targets": [
    {
      "id": "gfx950",
      "results": [
        {
          "testId": "gemm-fp16-1024",
          "durationSeconds": 3.12,
          "status": "completed",
          "error": null
        }
      ]
    },
    {
      "id": "gfx1250",
      "results": [
        {
          "testId": "gemm-fp16-1024",
          "durationSeconds": 2.82,
          "status": "completed",
          "error": null
        }
      ]
    }
  ]
}
```

### Run and comparison identity

- `id`: required unique plugin-execution ID matching `<run-id>` in the run filename.
- `comparisonId`: required controlled-experiment ID shared by Vanilla, ASan, TSan, UBSan, or other plugin runs.
- `testCatalog`: required path matching `test-catalogs/<filename>.json`.

Do not give plugin executions the same `id`. For different plugins to share a `comparisonId`, they must have identical catalog, source, trigger, machine, environment, and target sets. Their completion times and result values may differ.

Validation enforces three rules per `comparisonId`, and a violation of any of them rejects the whole dataset:

- Every run sharing the ID must carry the same catalog path, branch, commit SHA, commit timestamp, commit message, trigger, machine, normalized environment, and target set.
- A plugin ID may appear at most once. Publish a rerun under a new `comparisonId` rather than repeating a plugin inside an existing comparison.
- A comparison containing any non-Vanilla plugin must also contain its `vanilla` run. Publish the instrumented runs together with their baseline, never on their own.

Normal performance history, Overview, Run Comparison, Benchmarks, and Failures use only the `vanilla` plugin. The Plugin Comparison list includes only compatible controlled comparisons containing Vanilla and at least one non-Vanilla plugin, and renders each target selected in the global filter without adding a second target picker.

### `plugin`

- `id`: required stable identifier, such as `vanilla`, `asan`, or `tsan`.
- `name`: required display name.
- `version`: optional plugin or toolchain version.
- `options`: optional object containing plugin-specific configuration.

The uninstrumented baseline is:

```json
{
  "id": "vanilla",
  "name": "Vanilla"
}
```

Plugin identity and options do not belong in `environment`, because plugins are the intentional comparison dimension.

### `source`

- `branch`: required and currently must be `develop`.
- `commit`: required full SHA of the tested Rocjitsu commit.
- `committedAt`: required ISO-8601 timestamp of that commit.
- `message`: optional commit message.

Historical reruns preserve the source commit's original `committedAt`.

### `execution`

- `completedAt`: required ISO-8601 timestamp for the completed plugin execution.
- `trigger`: required; exactly `auto` or `manual`.
- `machine`: required stable runner name shared by every target in the run. Publication validation requires the same machine across all runs.

Target identity is intentionally absent from `execution`; targets are grouped under `targets`.

### `environment`

`environment` is a required array of shared execution details and may be empty. Items contain:

- `key`: stable identifier, unique within the run.
- `label`: display label.
- `value`: string, number, or boolean.

Array order and labels do not affect compatibility; normalized key/value pairs do. Two empty environments are compatible. An empty and populated environment are not.

Environments may change between independent historical runs. Runs sharing a
`comparisonId` must have the same normalized environment because plugin overhead is
valid only when the compared executions use the same environment.

### `targets` and results

`targets` must contain exactly one group for every target defined by the referenced catalog. Each group contains:

- `id`: target architecture such as `gfx950` or `gfx1250`.
- `results`: exactly one result for every test ID listed under `catalog.targets[id]`.

The result identity is `(run id, target id, testId)`. Plugin comparison identity is `(comparisonId, target id, testId)`.

Every result contains:

- `testId`: required catalog test ID.
- `durationSeconds`: positive numeric seconds for a completed result; `null` for failed or timed-out results.
- `status`: required; `completed`, `failed`, or `timeout`.
- `error`: failure description or `null`.

Failed and timed-out tests must remain in `results`. Omitting a catalog result invalidates the run instead of silently changing its denominator.

## Plugin comparison calculations

Plugin comparison always occurs within one `comparisonId` and one target. Vanilla is the default baseline.

```text
duration change (%) = (plugin duration - baseline duration) / baseline duration × 100
```

Per-test changes within ±3% are presented as measurement noise rather than overhead.

The overall runtime overhead is the geometric mean of per-test duration ratios. When every selected test is comparable, the value is exact. When only some tests pass for both the plugin and baseline, the dashboard measures the geometric-mean overhead from those passed pairs and assumes the same overhead for the entire selected test set. This whole-set estimate is marked with `*` and reports how many passed pairs contributed. Failed, timed-out, missing, and baseline-incomplete results contribute no measured ratio; they receive the passed-pair overhead assumption. No estimate is shown when no passed pair exists.

## Publishing checklist

1. Choose an existing immutable catalog or publish a new catalog first.
2. Run Vanilla and each instrumented plugin against the catalog's complete target/test matrix.
3. Give each plugin execution a unique `id` and the same controlled `comparisonId`.
4. Include exactly one completed, failed, or timed-out result for every catalog test applicable to every target.
5. Validate the complete staged data directory with `npm run validate:data -- <data-directory>`.
6. Upload each new catalog and `data/runs/<run-id>.json`.
7. Add the run filenames to `data/index.json`, update `generatedAt`, and publish
   the index last.

No React or Vite build is required for a data-only GitHub Pages update.
Validation is mandatory before publication. The browser runs the same validation and
fails closed with no benchmark data if invalid files bypass the publishing gate; it
does not skip invalid runs or construct partial history.

`validate:data` covers only what the index can reach. It starts at `metadata.json` and
`index.json` in the given directory, then reads the runs listed in `runFiles` and the
catalogs those runs reference. A file that exists on disk but is unreachable from the
index is never read: an orphan run file, a catalog no run references, or a run left out
of `runFiles` is neither validated nor reported. Stage the directory exactly as it will
be published, with the updated index in place, so that validation covers everything the
browser will load.

## Values derived by the dashboard

Do not publish aggregate duration, runtime overhead, baselines, deltas, coverage percentages, reliability percentages, comparison counts, chart labels, colors, or layout. The browser derives them from catalogs and run results.

### Baseline selection

There is no single published baseline. Each surface answers a different question and
chooses its own, so the same run can show different changes in different places. Every
choice below respects the active target and suite filters, and *complete* means that
every selected test in that run completed with a finite duration.

| Surface | Candidate | Baseline |
| --- | --- | --- |
| Overview metric cards | Latest commit run | Latest complete run of the oldest commit that has one |
| Duration history and Largest Changes | Latest run shown in the selected timeframe | First run shown in that timeframe; the `1D` range instead uses the previous commit day's latest run |
| Latest Commit Results | Latest commit run | The same timeframe baseline the duration history uses, so the Overview range selector also changes this table |
| Recent Runs | Each listed run | Nearest earlier complete commit run, evaluated per row |
| Run Comparison | Selected run, defaulting to the latest execution | Nearest earlier complete commit run for a selected candidate; with no selection, the previous execution in publication order |
| Plugin Comparison | Each plugin run in the comparison | The `vanilla` run of the same `comparisonId` and target |

Baselines are ordered by commit, not by publication time, so a historical rerun
published today does not become the baseline for older commits. Largest Changes and
Run Comparison treat changes within ±3% as measurement noise.

### Workload normalization

Comparing a summed duration across catalog versions would otherwise compare different
test sets, so history is normalized to one workload before any total is computed:

- The canonical workload is the set of tests the latest commit run covers for each
  selected target and suite.
- A run published under a different catalog ID that lacks a canonical test contributes
  that test's most recent completed duration from history. When this occurs, the
  Performance Trend displays a note explaining that its results are normalized to the
  latest test catalog and that values absent from older catalogs are estimated.
- If a canonical test is present but not completed, or is missing with no earlier
  completed duration to draw on, the run contributes no value and the chart shows a gap
  instead of a smaller total.
- Daily ranges plot the latest complete run for each UTC commit date; the `1D` range
  plots the individual commits of a single UTC date.
- A fixed-length range representing fewer than 75% of its calendar days is reported as
  insufficient data, and its range-level change is withheld rather than estimated.

Normalization only fills tests absent from an older catalog. It never substitutes a
value for a test that ran and failed, so failures always reduce coverage rather than
silently inheriting an earlier duration.
