# Schema 2 — current JSON

Fictional examples. Only `metadata.json` selects the target run version; each run declares its actual version. Indexes and catalogs have no version field.

```text
website/src/config/metadata.json    # bundled with the website; never fetched

data/
  index.json
  test-catalogs/example.json
  runs/
    default-branch/run-001.json     # source.branch == "develop"
    side-branches/                  # every other branch; no branch subdirectories
```

## src/config/metadata.json

`schemaVersion` means the run schema the website wants to read, not a metadata format version.

```json
{
  "schemaVersion": 2,
  "repository": "https://github.com/ROCm/rocm-systems",
  "isBeta": true,
  "canonicalBranch": "develop"
}
```

## index.json

```json
{
  "generatedAt": "2026-10-01T12:00:00Z",
  "runFiles": ["runs/default-branch/run-001.json"]
}
```

## test-catalogs/example.json

```json
{
  "id": "example",
  "tests": [
    { "id": "gemm", "suite": "Example", "name": "GEMM", "problem": { "size": 16 } }
  ],
  "configurations": {
    "gfx950:ST": ["gemm"],
    "gfx950:MT": ["gemm"]
  }
}
```

## runs/default-branch/run-001.json

```json
{
  "schemaVersion": 2,
  "id": "run-001",
  "testCatalog": "test-catalogs/example.json",
  "source": {
    "branch": "develop",
    "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "committedAt": "2026-10-01T10:00:00Z"
  },
  "execution": {
    "completedAt": "2026-10-01T11:00:00Z",
    "trigger": "auto",
    "machine": "example-runner"
  },
  "environment": [
    { "key": "cores", "label": "CPU cores", "value": 32 }
  ],
  "configurations": [
    {
      "target": "gfx950",
      "threadingMode": "ST",
      "results": [
        { "testId": "gemm", "status": "completed", "timing_results_s": [1, 9, 2], "durationSeconds": 2, "error": null }
      ]
    },
    {
      "target": "gfx950",
      "threadingMode": "MT",
      "results": [
        { "testId": "gemm", "status": "completed", "timing_results_s": [1, 1.5], "durationSeconds": 1.25, "error": null }
      ]
    }
  ]
}
```

## Result rules

```js
// completed: positive finite samples; duration is their median; error must be null
{ testId: "gemm", status: "completed", timing_results_s: [1, 9, 2], durationSeconds: 2, error: null }

// failed / timeout: seconds must be null; error may be null or nonempty text
{ testId: "gemm", status: "failed", timing_results_s: [5], durationSeconds: null, error: "Execution failed" }
{ testId: "gemm", status: "timeout", timing_results_s: [], durationSeconds: null, error: null }
```

- The preparer publishes `timing_results_s` for every result, preserving accepted sample values and order. Normalization retains the samples. Completed results require a nonempty array and `durationSeconds` equal to its median. Failed/timeout results may retain partial samples or an empty array, but their duration remains null.
- Older publications (including migrated schema-1 runs) may omit `timing_results_s`; their existing finite, nonnegative completed durations remain valid. Missing samples are not reconstructed from a median. New run bodies must use new run IDs rather than overwrite immutable published files.
- Each included configuration has exactly its catalog's workload IDs. An absent configuration means unpublished, not zero.
- Commit/base SHAs: lowercase, 40 hex characters. Times: commit ≤ completion ≤ publication.
- `source.message`, `source.base: { branch, commit }`, and `source.pullRequest: { number, url? }` are optional.

See [preparing and publishing measurements](architecture-and-publication.md#preparing-and-publishing-measurements) for generation and upload. Related: [schema 1](schema-1.md), [migration](schema-migration.md), and [build/test commands](build-and-test.md).
