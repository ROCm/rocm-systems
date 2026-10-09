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
        { "testId": "gemm", "status": "completed", "durationSeconds": 2, "error": null }
      ]
    },
    {
      "target": "gfx950",
      "threadingMode": "MT",
      "results": [
        { "testId": "gemm", "status": "completed", "durationSeconds": 1.25, "error": null }
      ]
    }
  ]
}
```

## Result rules

```js
// completed: finite seconds >= 0; error must be null
{ testId: "gemm", status: "completed", durationSeconds: 0, error: null }

// failed / timeout: seconds must be null; error may be null or nonempty text
{ testId: "gemm", status: "failed", durationSeconds: null, error: "Execution failed" }
{ testId: "gemm", status: "timeout", durationSeconds: null, error: null }
```

- Each included configuration has exactly its catalog's workload IDs. An absent configuration means unpublished, not zero.
- Commit/base SHAs: lowercase, 40 hex characters. Times: commit ≤ completion ≤ publication.
- `source.message`, `source.base: { branch, commit }`, and `source.pullRequest: { number, url? }` are optional.

See [preparing and publishing measurements](architecture-and-publication.md#preparing-and-publishing-measurements) for generation and upload. Related: [schema 1](schema-1.md), [migration](schema-migration.md), and [build/test commands](build-and-test.md).
