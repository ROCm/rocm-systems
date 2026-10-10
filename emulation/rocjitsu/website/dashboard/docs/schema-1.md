# Schema 1 — legacy JSON

Fictional example. Missing `schemaVersion` means `1`; explicitly setting it to `1` is also accepted.

```text
data/
  index.json
  test-catalogs/example.json
  runs/legacy-run.json
```

## index.json

```json
{
  "generatedAt": "2026-10-01T12:00:00Z",
  "runFiles": ["runs/legacy-run.json"]
}
```

## test-catalogs/example.json

```json
{
  "id": "example",
  "tests": [
    { "id": "gemm", "suite": "Example", "name": "GEMM", "problem": { "size": 16 } }
  ],
  "targets": {
    "gfx950": ["gemm"]
  }
}
```

## runs/legacy-run.json

```json
{
  "id": "legacy-run",
  "comparisonId": "legacy-comparison",
  "plugin": { "id": "vanilla", "name": "Vanilla" },
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
  "environment": [],
  "targets": [
    {
      "id": "gfx950",
      "results": [
        { "testId": "gemm", "status": "completed", "durationSeconds": 1.25, "error": null }
      ]
    }
  ]
}
```

One file has one plugin identity. Completed schema-1 durations must be positive.

See [schema-1 patch](schema-migration.md) for compatibility and exclusions. Shared field constraints, including commit SHA format, follow [schema 2](schema-2.md#result-rules).
