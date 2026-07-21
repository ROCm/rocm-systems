# Knowledge retention

PerfXpert retains allowlisted performance observations in a user-local SQLite
database. This is persistence, not model training: no weights, training
datasets, embeddings, vector database, prompt injection, or provider behavior
are involved.

## Knowledge layers

- **Curated knowledge** remains the reviewed, schema-validated YAML under
  `perfxpert/knowledge/`. Runtime code never edits it.
- **Retained observations** are local prediction, trace-analysis, and
  run-comparison records produced by trusted CLI/orchestration boundaries.
- **Model-derived decisions** are labeled separately from deterministic
  evidence. Raw prompts and model prose are never retained.

## Store and project scope

The default database is `~/.perfxpert/knowledge.db`.
`PERFXPERT_KNOWLEDGE_ROOT` overrides the containing directory, not the database
filename. The database uses rollback journaling for compatibility with
network-mounted HPC home directories.

Every row belongs to a path-bound project `scope_id`. PerfXpert resolves one
scope context from:

1. an explicit project root;
2. `PERFXPERT_PROJECT_ROOT`;
3. the nearest VCS root above the source directory or current directory;
4. the canonical current directory.

Ordinary Python and MCP readers can access only the current scope.
Cross-project access is limited to the explicit `perfxpert knowledge
--scope all` administration surface and the non-MCP `perfxpert.retention`
administration API.

## Identities

Prediction and storage identities serve different purposes:

- `prediction_id` is a source-independent SHA-256 identity over effective
  prediction parameters, canonical semantic output, and predictor/catalog
  versions. `baseline_db`, source metadata, scope, timestamps, and generated
  fields are excluded.
- `record_id` identifies one retained revision. It includes project scope,
  ordered source fingerprints and roles, producer/schema versions, the
  prediction ID when applicable, and the exact policy-compliant payload.

Repeated identical records increment `seen_count`. Different payloads or
provenance produce immutable revisions; rows are never merged into hybrid
facts.

## Source fingerprints

V1 uses a metadata fingerprint—path hash, size, `mtime_ns`, SQLite header
metadata, and WAL metadata—not a whole-file content digest. Trusted
orchestration captures metadata before and after computation and refuses to
persist a torn observation when it changes.

When ATT analysis is enabled, orchestration also snapshots the ordered regular
files under the ATT directory. The raw ATT directory path is never accepted as
an observation option; only its policy-compliant file descriptors and an
`att_enabled` marker participate in retention.

A same-size rewrite with preserved metadata can alias. The store and docs do
not claim content-level source identity; a streamed digest can be added later
if production workloads justify its I/O cost.

## Privacy and durability

Stored payloads are built from allowlists. By default, full paths are replaced
with project-relative paths or basenames. Set
`PERFXPERT_KNOWLEDGE_STORE_PATHS=1` to retain full local provenance.

`predict_change_impact` keeps returning the caller-supplied `baseline_db`.
Storage sanitization does not change that public result. A durable
`explain_prediction` fallback preserves semantic prediction fields and reports
`provenance_redacted=true` when the original path was not retained.

Every explicit writer returns a `PersistenceReceipt`:

- `persisted` — the record committed successfully;
- `disabled` — retention is off and no store was created;
- `quota_exceeded` — computation succeeded but the configured size ceiling
  refused the write;
- `error` — computation succeeded and the persistence failure is visible.

Only `persisted` guarantees cross-process lookup.

## Write and read boundaries

READ_ONLY compute tools remain free of filesystem writes:

- `predict_impact.predict_change_impact`
- `agents.analysis.run_analysis`
- `trace_diff.diff_runs`

Batch `analyze`, `diff`, and `ci` orchestration records their structured
results after computation. Explicit Python callers use
`perfxpert.retention.predict_change_impact_durable`, `record_analysis`, or
`record_comparison`.

The MCP surface exposes only no-create readers:

- `knowledge_history.get_knowledge_observation`
- `knowledge_history.query_knowledge`
- `knowledge_history.knowledge_stats`

Missing-store reads return empty query/stat results and never create or migrate
the database.

## Configuration and lifecycle

Configuration fields:

- `knowledge_retention` / `PERFXPERT_KNOWLEDGE_RETENTION` (default: on)
- `knowledge_max_mb` / `PERFXPERT_KNOWLEDGE_MAX_MB` (default: 256 MiB)
- `knowledge_store_paths` / `PERFXPERT_KNOWLEDGE_STORE_PATHS` (default: off)
- `PERFXPERT_KNOWLEDGE_ROOT`
- `PERFXPERT_PROJECT_ROOT`

Administration examples:

```bash
# SKIP-SAMPLE — commands depend on the caller's local retained observations
perfxpert knowledge stats
perfxpert knowledge query --kind prediction --kernel-name my_kernel
perfxpert knowledge prune --older-than 90 --yes
perfxpert knowledge clear --scope current --yes
```

`clear` and `prune` are never exposed through MCP. Disabling retention prevents
new writes and store creation; explicit administration can still remove an
existing store's rows.
