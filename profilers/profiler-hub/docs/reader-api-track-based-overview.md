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

`track_info_t` (returned by `get_all_tracks()`) carries an explicit `id` and `type`, plus optional context objects (agent/queue/stream/thread/pmc info, populated depending on track type), so a track is fully self-describing without side queries. `agent_info` carries a numeric `id`, giving direct access to the raw agent id (e.g. for topology nesting or table-cell naming) without a lookup.

`pmc_info_t` also carries an `ambiguous` flag (default `false`), set when a `pmc_id` has more than one `rocpd_pmc_event` row per `event_id` in the source data — a known producer bug on some databases. This is read-only telemetry, not a query-behavior change; it lets you flag or filter suspect PMCs instead of getting silently-wrong values.

## Methods

**`get_interval_track(track_id, filter = {})`**
Covers anything interval-shaped: `cpu_thread`, `gpu_queue`, `dma`, `stream`, `memory`, `kernel_dispatch_pmc` tracks. Scopes correctly to the specific track identity — this is the fix for the GPU queue bug. Returns `level` and `parent_id` on each event, computed by ProfilerHub in a single pass, so nesting doesn't need to be derived on your side.

**`get_scalar_track(track_id, filter = {})`**
Covers the sample/PMC event path for `counter` and `memory_activity` tracks. Returns `(opaque_id, timestamp, value)` tuples, with `value` as a double to accommodate both PMC and other scalar sources (for `memory_activity`, `value` is the running-total byte count after that event). A `counter` track's PMC identity lives on the track itself (`track_info_t.pmc_info`), not the individual sample, and the underlying queries resolve each track's PMC deterministically — a counter track returns only its own metric's samples, with no fan-out across co-sampled metrics.

**`get_flows(filter = {})`**
A whole-database (by default) API returning causal links between events — e.g., a CPU-side API call and the GPU kernel dispatch it triggered — as flow edges keyed on `stack_id`, so you don't need to derive these relationships yourselves from `correlation_id`/`stack_id` joins. This is a batch/post-hoc query rather than something attachable to individual eve