# ProfilerHub Reader API: Track-Based API Surface (Overview for Optiq)

## Why this matters to Optiq

The current reader API has a scoping bug in `get_events_for_track()`: for a GPU queue track, it returns *all* kernel dispatch events across the whole process instead of just the events belonging to that one queue.

Separately, you currently compute event nesting (level/parent relationships) yourselves in a per-row callback as your SQL query returns rows, then persist the result into your own tables for reuse — ProfilerHub instead computes this once per request and hands it back directly on each event, so there's no separate table or callback to maintain on your side.

ProfilerHub addresses both through a type-aware track model and a small family of track-oriented reader methods, described below.

## Track types

Every track carries an explicit type, via a `track_type_t` enum:

- `cpu_thread` — CPU-side HIP/API region events
- `gpu_queue` — kernel dispatch events, correctly scoped to a single queue
- `dma` — memory copy events, keyed `(nid, pid, queue_id, dst_agent_id)` — grouped by destination agent, matching the swimlanes your memory-copy view already produces. On a `dma` track, `agent_info` is the destination agent; `stream_info` is unset.
- `counter` — scalar/PMC counter samples (SMI-style, one PMC per track). `track_info_t.pmc_info` carries the full PMC metadata panel (name, symbol, description, units, block, expression, …) keyed on the real `pmc_id`, so no side lookup is needed to identify a counter track's metric. Each counter track resolves deterministically to exactly one PMC.
- `stream` — a cross-cutting track that aggregates kernel dispatch, memory copy, and memory allocate events sharing a stream, giving a unified per-stream view without needing to merge separate event sources yourself.
- `memory` — standalone memory-allocation events (`rocpd_memory_allocate`), keyed `(nid, agent_id, queue_id, pid)` to match your `GetRocprofMemoryAllocTrackQuery` grouping exactly (both `agent_id`/`queue_id` are nullable, and NULL is preserved as its own group rather than dropped). The `stream` track's memory-allocate leg is a separate, cross-cutting view and is unaffected by this.
- `kernel_dispatch_pmc` — per-kernel-dispatch PMC samples (`rocpd_pmc_event` joined to `rocpd_kernel_dispatch`), keyed `(nid, agent_id, pmc_id, pid)` with `agent_info` populated. Distinct from `counter`: `counter` tracks are periodic device/process samples, `kernel_dispatch_pmc` tracks are PMC values attached to individual kernel dispatches.
- `memory_activity` — a per-agent cumulative running-total byte count synthesized from allocations/frees, mirroring your own memory-activity view. Has no `pmc_info`, avoiding collision with the real counter/PMC id space.

`track_info_t` (returned by `get_tracks()`) carries an explicit `id` and `type`, plus optional context objects (agent/queue/stream/thread/pmc info, populated depending on track type), so a track is fully self-describing without side queries. `agent_info` carries a numeric `id`, giving direct access to the raw agent id (e.g. for topology nesting or table-cell naming) without a lookup.

`pmc_info_t` also carries an `ambiguous` flag (default `false`), set when a `pmc_id` has more than one `rocpd_pmc_event` row per `event_id` in the source data — a known producer bug on some databases. This is read-only telemetry, not a query-behavior change; it lets you flag or filter suspect PMCs instead of getting silently-wrong values.

## Methods

**`get_interval_track(track_id, filter = {})`**
Covers anything interval-shaped: `cpu_thread`, `gpu_queue`, `dma`, `stream`, `memory`, `kernel_dispatch_pmc` tracks. Scopes correctly to the specific track identity — this is the fix for the GPU queue bug. Returns `level` and `parent_id` on each event, computed by ProfilerHub in a single pass, so nesting doesn't need to be derived on your side.

**`get_scalar_track(track_id, filter = {})`**
Covers the sample/PMC event path for `counter` and `memory_activity` tracks. Returns `(opaque_id, timestamp, value)` tuples, with `value` as a double to accommodate both PMC and other scalar sources (for `memory_activity`, `value` is the running-total byte count after that event). A `counter` track's PMC identity lives on the track itself (`track_info_t.pmc_info`), not the individual sample, and the underlying queries resolve each track's PMC deterministically — a counter track returns only its own metric's samples, with no fan-out across co-sampled metrics.

**`get_flows(filter = {})`**
A whole-database (by default) API returning causal links between events — e.g., a CPU-side API call and the GPU kernel dispatch it triggered — as flow edges keyed on `stack_id`, so you don't need to derive these relationships yourselves from `correlation_id`/`stack_id` joins. This is a batch/post-hoc query rather than something attachable to individual events, since the destination event is typically discovered on a different track processed later in the stream. An optional time-window filter is supported; other filter fields are reserved but not yet honored.

Each returned `flow_edge_t` carries **typed endpoints**: `source_opaque_id`/`dest_opaque_id` plus a `source_type`/`dest_type` (an `event_type_t`) on each side. This matters because opaque ids are only unique *within* an event table — a region event and a kernel-dispatch event can share the same raw id — so the type tag lets you safely disambiguate which table an id belongs to.

`get_flows()` emits the **full stack-clique** for each causal group: region↔region edges, same-type sibling edges (kernel-dispatch↔kernel-dispatch, memory-copy↔memory-copy, memory-allocate↔memory-allocate), and region-to-leaf edges (region↔kernel-dispatch, region↔memory-copy, region↔memory-allocate) — 7 statement sets in total, each with its own base and time-windowed variant. This gives you the same causal relationships you'd otherwise compute with separate per-event-type queries, in one call.

**`get_track_stats(track_id)`**
Returns a cheap `{min_ts, max_ts, count}` for any track without materializing its events — useful for computing per-track time origins at discovery time.

## Event payload additions

Interval events (`interval_entry_t`) also carry:

- `category` — a display string such as `"rocm_hip_api"` or `"timer_sampling"` (empty when an event has no category)
- `op_kind` — for `stream` tracks only, indicates which underlying event table (kernel dispatch, memory copy, memory allocate) a given event's `opaque_id` came from, so the right detail method can be called. Unset for single-source track types, where the track's own type already disambiguates.

## Detail methods stay backward compatible

The existing `get_*_details()` methods (keyed on `timeline_event_t`) are not being removed. Overloads that accept the plain `opaque_id` (a `size_t`) directly are available alongside them, so existing call sites keep working while new code can call detail methods directly off the track APIs without reconstructing a `timeline_event_t`. Deprecating the old overloads is a separate, later decision.

## What this means for Optiq

- GPU queue dispatch tracks return correctly scoped data instead of all-process dispatch events.
- Nesting/level computation can be dropped from your SQL and read directly off `get_interval_track()` results.
- The `stream` track type gives you a unified per-stream interval view from `get_tracks()`.
- `get_flows()` — with typed endpoints and full stack-clique edges — gives you causal-link relationships without deriving them from `correlation_id`/`stack_id` joins yourselves.
- The `memory` track type surfaces standalone memory-allocation tracks (grouped by agent/queue), so your standalone Memory Allocation view no longer needs its own SQL.
- `kernel_dispatch_pmc` and `memory_activity` cover per-kernel-dispatch PMC values and cumulative memory-activity totals, both retrievable through the same `get_interval_track`/`get_scalar_track` methods used everywhere else.
- `pmc_info_t.ambiguous` flags PMCs affected by a known producer bug (duplicate rows per event), so you can choose how to handle them instead of getting silently-wrong values.
- No existing detail-method call sites need to change immediately.

## Potential future work

Today, `get_interval_track()` computes event nesting (level/parent) fresh on every call — a sort plus a single linear pass over the track's events, with nothing cached or persisted. That computation only depends on the track's own event set, not on any caller-supplied filter, which makes it a natural candidate for precomputation.

One direction under consideration: a preprocessing pass over an input database that builds summary tables and indexes ahead of time — in the same spirit as the level/parent computation, but generalized to other read-time work that's currently repeated per query. This would move "compute once, reuse many times" from being something each consumer manages for itself into something ProfilerHub does once, upstream, for everyone. Nothing here is committed or scheduled; it's a design direction worth keeping in view as the reader API matures.

## Status

This describes the ProfilerHub reader API surface as implemented and validated against ProfilerHub's own test suite, including a fidelity check confirming these APIs can reproduce, exactly, the behavior of one of your existing code paths (flow-trace lookup) when prototyped against them. That prototype was exploratory and is not a commitment to any particular change on the Optiq side — the reader API described here is the durable, intended integration point; how (or whether) any given Optiq code path adopts it is a separate decision. This has not yet gone through final review or been merged upstream.

For a worked example of what a full loader migration onto this API looks like — every Optiq track type migrated, one at a time, with fidelity verification for each — see `profiler-hub-optiq-loader-reference-migration.md`. That migration is a reference prototype, not a proposal: nothing there is committed or pushed, and adopting any, all, or none of it is entirely the Optiq team's call.
