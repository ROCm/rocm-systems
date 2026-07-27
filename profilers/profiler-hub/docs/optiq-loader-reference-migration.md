# Reference Migration: Optiq Loader onto the ProfilerHub Reader API

## Status and framing

This is a **prototype, not a proposal**. To validate that the ProfilerHub reader API (see `profiler-hub-new-reader-api-overview.md`) actually covers everything Optiq's loader needs — not just in theory, but against real query semantics — we migrated Optiq's own loader (`roc-optiq`, branch `users/avansick-amd/profiler-hub-loader`) end to end, one track type at a time, and verified each migration for fidelity against the existing SQL-based behavior.

Nothing here is committed to the Optiq repo's mainline and nothing is pushed to any shared remote. The Optiq team can adopt all of it, part of it, or none of it — the value of this exercise was proving the reader API is sufficient, not prescribing that Optiq must change its loader architecture. Where this document says "replaces," it means "replaces in this prototype branch," not "should replace in your codebase."

## What changed, at a glance

Baseline: `48e29fbd` ("Duplicate File Open Protection"). Seven commits on top, each migrating one track type or capability:

| Commit | Track type / capability | What it replaced |
|---|---|---|
| `344fef7f` | `cpu_thread` | Standalone CPU-region SQL discovery |
| `6cbdf6e5` | `gpu_queue` + `stream` | Queue-keyed kernel-dispatch SQL + the cross-cutting Stream-track SQL |
| `59a80078` | `memory` (standalone memory-allocate) | `GetRocprofMemoryAllocTrackQuery`/`LevelQuery`/`SliceQuery` |
| `722bc2d7` | `dma` (standalone memory-copy) | `GetRocprofMemoryCopyTrackQuery`/`LevelQuery`/`SliceQuery`/`TableQuery` |
| `79a22f84` + `697f532a` | `counter` (SMI PMC) | Standalone counter-track SQL; second commit made system-test assertions order/index-independent |
| `aab7609d` | dataflow (`get_flows()`) | Four `GetRocprofDataFlowQueryFor*` SQL methods, deleted entirely |
| `aa16c861` | — | Formatting only (clang-format-18, no logic change) |

Net diff vs. baseline: 10 files, +6,870/-4,985 lines. The two largest files by volume are `rocprofvis_db_query_factory.cpp` (losing most of its hand-written SQL generation) and `rocprofvis_db_rocprof.cpp` (gaining the adapter functions described below).

## The pattern, once, since it repeats seven times

Every migration in this branch follows the same two-function shape in `rocprofvis_db_rocprof.cpp`:

1. **`Reader<Type>TrackToTrackParams(...)`** — a small adapter that reads a ProfilerHub `track_info_t` (id, agent/queue/stream/thread/pmc info) and populates Optiq's internal `rocprofvis_dm_track_params_t` (identifiers, category, op-type). This is where reader fields map onto Optiq's existing topology/naming conventions — e.g. numeric `agent_info->id` feeding `TRACK_ID_AGENT` for GPU-topology nesting.
2. **`AddReader<Type>Tracks(Future* future)`** — the discovery/load function. Checks the metadata-version cache first (a cache hit skips the reader entirely and reloads from the DB's own saved track table); on a cache miss, calls `reader->get_tracks()`, filters to the relevant `track_type_t`, and for each track computes `record_count` via `get_track_stats()` and min/max timestamp/level via a single `get_interval_track()` walk. Threaded per DB instance, matching the existing threading model.

At the call site, the old block — typically a raw `ExecuteSQLQuery` wired to 3-6 hand-written `QueryFactory` methods via callbacks — collapses to a single `AddReader<Type>Tracks(future)` call.

## Track-type notes worth knowing

- **`dma`**: the reader keys these tracks by *destination agent* (not stream), which was a deliberate reader-side design choice (see `005B-3-fix-2` in the project history) made specifically so this migration wouldn't change Optiq's existing by-agent swimlane grouping. `queue_info` isn't exposed on the public `dma` track, so `queue_id` defaults to 0 in the adapter — harmless against the verification fixture where it was already 0, but worth knowing if a different fixture has non-zero queue ids on dma tracks.
- **`counter`**: this migration surfaced a topology mismatch during verification (reader's 18 real PMC tracks vs. an old SQL-path artifact of 387 vs 388 system-test assertions) that was root-caused to the *old* SQL doing index-based test selection over a track ordering that legitimately changed shape post-migration — not a data-fidelity loss. The `697f532a` companion commit makes those system-test assertions order/index-independent so this class of false-positive can't recur.
- **Dataflow (`get_flows()`)**: this is the migration with the strongest fidelity evidence — verified byte-identical against a pristine pre-migration SQL oracle (a separate frozen worktree, `roc-optiq-sql-oracle` at `48e29fbd`) for region, kernel-dispatch, and memory-copy causal edges, plus a fabricated fixture for memory-allocate edges. Four `GetRocprofDataFlowQueryFor*` SQL methods are deleted outright rather than left dormant.

## Verification evidence

Each migration commit's message documents its own fidelity check (membership verification against `rocpd-transpose.db`, or the SQL-oracle comparison for flows), plus the project-wide regression gates: `datamodel-system-tests-DB` and `datamodel-compute-tests` both pass at every step, with the one documented, understood exception (system-tests count moving 387→388 across the counter migration, explained above and locked down by the order-independence fix).

## Bottom line for the Optiq team

If you want to adopt any of this, this branch is a working reference for exactly how each track type maps onto the reader API and what each migration's fidelity check looked like. If you'd rather keep your existing SQL path for some or all track types, nothing here forces a change — the reader API itself (documented in `profiler-hub-new-reader-api-overview.md`) is the durable interface; this branch is just proof it's sufficient to build on.
