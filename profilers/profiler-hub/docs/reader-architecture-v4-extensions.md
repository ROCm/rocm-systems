# Reader Architecture: v4/Policy-Based Extensions

> **Branch note.** This document describes the reader architecture as implemented on branch
> `users/avansick-amd/profiler-hub-interval-scalar-api`. This work extends, but has not
> replaced, the canonical `architecture-profiler-hub.md`. That document remains unchanged
> pending any future decision to propose this work upstream.

## Overview

Chapter 3 of `architecture-profiler-hub.md` describes the v3-only reader as a monolithic
`impl` struct and frames policy-based, multi-schema restructuring as intended future work.
That restructuring has now been built on this branch. The reader is no longer monolithic: it
has a version-dispatch seam, a schema-neutral abstract interface, two schema-specific
backends, and an eight-value track type system. This document is the authoritative reference
for that implemented architecture.

## The Version-Dispatch Seam

The central structural addition is `read_statements_base`, an abstract class in
`source/data_storage/read_statements_base.hpp`. It defines the complete interface that
`reader_t::impl` uses to query the database. It declares pure-virtual accessors for the
shared subset (info tables, counter/scalar/flow queries) and virtual accessors with
default-empty bodies for backend-specific statements. The two schema-specific backends
inherit from it and override the accessors they implement; each inherits empty stubs for
the accessors it does not implement, and the reader never invokes unimplemented accessors
because it guards them behind the `m_is_v4` flag.

The two concrete backends are:

- `data_storage::schema_v3::read_statements` — the v3 backend. Initializes prepared SQL
  against `rocpd_region`, `rocpd_kernel_dispatch`, `rocpd_memory_copy`,
  `rocpd_memory_allocate`, `rocpd_sample`, and `rocpd_pmc_event`. Uses multi-column
  WHERE clauses (identity tuples) for track-scoped queries. Overrides the legacy timeline
  event statement sets, detail statements, event-id resolution, correlated events, count
  and time-range queries, and all track synthesis accessors.

- `data_storage::schema_v4::read_statements` — the v4.0 backend. Initializes SQL that
  joins through the `rocpd_timestamp` spine for interval start/end and `rocpd_track` for
  topology. Uses single `track_id`-anchored WHERE clauses for track-scoped interval queries.
  Overrides the v4-specific interval track, stats, and topology accessors. Does not override
  the legacy timeline/detail/event-id surface (those return empty stubs on v4).

`reader_t::impl` holds the selected backend as `shared_ptr<read_statements_base>` in
`m_read_statements`. Selection happens once in the constructor:

```cpp
// Detect v4 by the presence of the rocpd_timestamp_{uuid} table.
// m_is_v4 is stored and used to guard v3-only call sites.
if(m_is_v4)
    m_read_statements = make_shared<schema_v4::read_statements>(m_backend, uuid);
else
    m_read_statements = make_shared<schema_v3::read_statements>(m_backend, uuid);
```

All downstream call sites use `m_read_statements->foo()` without inspecting the schema
version again. The backends prepare their SQL at construction, so only the matching
backend's SQL is ever compiled.

### Why runtime virtual dispatch instead of compile-time CRTP

The writer uses compile-time policy dispatch (a single `active_policy_t` alias) because it
writes to exactly one schema version. The reader must support all schema versions
simultaneously — it opens existing databases that may be any prior version. This rules out
a compile-time alias. Instead, `read_statements_base` is a vtable-based abstract interface:
the schema is detected at runtime and the matching concrete backend is instantiated. The
public `reader_t` interface does not change between schema versions; the schema-specific
backends map their query results into the common output types in `reader_types`.

## The Track Type System

The core addition on the API surface is `track_type_t`, an enum that classifies every track
returned by `get_tracks()`. It determines which identity fields in `track_info_t` are
populated, which query method applies (`get_interval_track` or `get_scalar_track`), and
which `get_*_details()` overload applies to `opaque_id` values drawn from that track.

```
track_type_t
├── cpu_thread          thread_info populated. Interval. region events. v3 synthesized.
├── gpu_queue           agent_info + queue_info. Interval. kernel_dispatch events.
├── dma                 agent_info populated (dst agent). Interval. memory_copy events.
│                       Keyed (nid, pid, queue_id, dst_agent_id) — NOT stream_id.
├── counter             thread_info + pmc_info + optional agent_info. Scalar.
│                       rocpd_sample rows. pmc_info carries the full PMC metadata panel.
├── stream              stream_info populated. Interval. Aggregates kernel_dispatch +
│                       memory_copy + memory_allocate sharing a stream_id (3-way UNION).
│                       Each interval_entry_t carries op_kind to select get_*_details().
├── memory              agent_info + queue_info. Interval. memory_allocate events.
│                       Keyed (nid, agent_id, queue_id, pid). Both nullable; NULL is a
│                       distinct group, not dropped.
├── kernel_dispatch_pmc agent_info populated. Interval. rocpd_pmc_event JOIN
│                       rocpd_kernel_dispatch. Keyed (nid, agent_id, pmc_id, pid).
│                       Distinct from counter (SMI-style periodic samples).
└── memory_activity     agent_info populated. Scalar. Cumulative bytes-allocated per
                        agent synthesized from rocpd_memory_allocate: ALLOC +size,
                        FREE -size (address-map recovery for NULL-agent_id FREE rows),
                        REALLOC/RECLAIM no-op. One series per (nid, pid, agent_id).
```

`track_info_t` carries an opaque `id` (stable for the reader lifetime), the `type`, and
optional identity shared pointers. `node_info` and `process_info` are always populated;
all others depend on the track type (see the doc-comments on `track_info_t` in
`reader_types.hpp`).

Two additional fields on `track_info_t` are worth noting:

- `region_kind` — v3 `cpu_thread` tracks only. A single thread can produce two synthesized
  tracks: `main` (regions with no `rocpd_sample`) and `sample` (regions that have one),
  mirroring roc-optiq's region-main / region-sample split. All other track types set
  `region_kind = none`.

- `ambiguous_classification` — v4 only. Set when a `track_id` appeared in both the counter
  (via `rocpd_sample`/`rocpd_pmc_event`) and memory-allocate discovery sets — an ambiguous
  schema state. The counter classification takes precedence; memory-allocate events would be
  silently absent. This flag is a data-integrity warning for consumers. No known real database
  triggers it today.

## Track Discovery and Synthesis

Track discovery splits by schema version.

### v3: synthesize_derived_tracks()

In v3, `rocpd_track` exists but covers only counter tracks (via `rocpd_sample.track_id`).
All other track types have no registry rows; they are synthesized by querying the event
tables for distinct identity tuples.

Each distinct-topology query lives in `read_statements_base` as a virtual accessor (e.g.
`distinct_gpu_queue_tracks()`, `distinct_region_tracks()`) and is overridden by the v3
backend. The C++ loop in `synthesize_derived_tracks()` iterates each result set, assigns a
synthetic `track_info_t::id` above the real `rocpd_track` id space (`max_track_id()` gives
the ceiling), and populates `m_track_query_info` with the routing info for that track.

The `memory_activity` track type adds a further step: after track synthesis, the impl reads
all `rocpd_memory_allocate` rows for each `(nid, pid)` group and computes per-agent running
sums entirely in C++. This is the only track type whose scalar values are not stored
directly in the database — they are derived at read time.

### v4: build_v4_tracks()

In v4, every swimlane is a real `rocpd_track` row. `build_v4_tracks()` queries
`track_info_statement()` and classifies each row by inspecting which of its identity
columns (`agent_id`, `queue_id`, `stream_id`, `tid`) are populated and whether the
`track_id` appears in the `memory_alloc_track_ids()` or `kd_pmc_track_ids()` discovery
sets (which guard against the `gpu_queue` classifier claiming those tracks). The
classification precedence is: counter → memory → kernel_dispatch_pmc → gpu_queue → dma →
stream → cpu_thread. The `ambiguous_classification` flag is set when a track_id is found in
both the counter and memory-allocate sets before classification resolves.

## Loading Strategy

The two-tier loading strategy described in Chapter 3 is preserved and extended.

**Eager.** Constructed once at startup. Includes all info tables (nodes, processes,
threads, agents, streams, queues, PMC info, code objects, kernel symbols), the string
table, and the complete track list (including derived tracks for v3). Utility maps keying
each entity type by its database id are also built at this time, so foreign-key resolution
in event queries runs in O(1) without additional queries.

**Lazy.** `get_interval_track()`, `get_scalar_track()`, `get_flows()`, and all
`get_*_details()` methods execute on-demand SQL when called. The event set is too large to
cache, and callers typically filter by track or time window. The returned vectors are owned
by the caller; the reader caches nothing from these calls.

One exception: `memory_activity` scalar values are computed in `get_scalar_track()` by
loading all raw `rocpd_memory_allocate` rows for the track's `(nid, pid)` group and
running the accumulation in C++. It is still lazy (only computed when called) but reads
more rows than a single track's worth.

## Track-Scoped Routing

`reader_t::impl` maintains a map `m_track_query_info` from `track_info_t::id` to
`track_query_info_t`. This struct captures everything needed to route a
`get_interval_track(track_id)` or `get_scalar_track(track_id)` call to the right SQL
statement with the right bind parameters.

```cpp
struct track_query_info_t {
    reader_types::track_type_t type{};
    size_t  nid{}, pid{};
    optional<size_t> tid;        // cpu_thread
    optional<size_t> agent_id;   // gpu_queue, memory, dma (dst_agent_id)
    optional<size_t> queue_id;   // gpu_queue, memory, dma
    optional<size_t> stream_id;  // stream
    optional<size_t> pmc_id;     // kernel_dispatch_pmc
    size_t  real_track_id{};     // counter: rocpd_track id for sample.track_id
    bool    region_is_sample{};  // cpu_thread: main vs. sample kind
};
```

When a caller invokes `get_interval_track(track_id)`, the impl looks up `track_query_info_t`
by id, selects the matching SQL accessor from `m_read_statements`, binds the identity tuple
from `track_query_info_t`, and returns the result set. The reader never exposes the identity
tuple to callers — `track_info_t::id` is the stable opaque handle.

## Interval Event Fields

`interval_entry_t` (returned by `get_interval_track()`) carries:

- `opaque_id` — the SQLite row id in the event's per-type table (e.g., `rocpd_region.id`
  for cpu_thread tracks, `rocpd_kernel_dispatch.id` for gpu_queue). Pass to the
  `get_*_details()` method corresponding to the track type. For `stream` tracks, use
  `op_kind` to select the method.
- `start` / `end` — nanoseconds.
- `display_name` — human-readable label, resolved from the event's name foreign key (or
  kernel symbol for kernel_dispatch tracks) within the SQL query.
- `category` — the event's category string (e.g., `"rocm_hip_api"`, `"rocm_kernel_dispatch"`),
  resolved in-SQL for v3 via `rocpd_event → rocpd_string` and for v4 via
  `rocpd_event → rocpd_info_category`. Empty string when the event carries no category.
- `op_kind` — `optional<event_type_t>`. Populated only for `stream` tracks, whose events
  aggregate three source tables (kernel_dispatch / memory_copy / memory_allocate) into a
  single UNION. `op_kind` indicates which source table a given row came from, selecting the
  correct `get_*_details()` overload.
- `level` — nesting depth, 0 = outermost. Computed in the reader in a single pass over the
  sorted event list; callers do not need to derive this themselves.
- `parent_id` — `opaque_id` of the enclosing event when truly nested; `nullopt` when the
  event overlaps but is not strictly contained.

## Flow Events

`flow_edge_t` (returned by `get_flows()`) represents a causal link between events:

```cpp
struct flow_edge_t {
    size_t       source_opaque_id{};
    event_type_t source_type{};   // disambiguates source_opaque_id across event tables
    size_t       dest_opaque_id{};
    event_type_t dest_type{};     // disambiguates dest_opaque_id across event tables
};
```

`opaque_id` values are row ids in per-type tables and collide across types (a
`rocpd_region` row and a `rocpd_kernel_dispatch` row may share the same integer id).
Both `source_type` and `dest_type` are required to resolve the correct details lookup.

The flow query emits the full stack-clique: region→region, region→{kernel_dispatch,
memory_copy, memory_allocate}, and same-type siblings (kernel_dispatch→kernel_dispatch,
memory_copy→memory_copy, memory_allocate→memory_allocate). Flows are linked via shared
non-zero `stack_id` on `rocpd_event`; the SQL joins the event tables on this field.
Both v3 and v4 backends implement the flow accessors.

## Expansion Points

**Adding a new track type.** Add an enumerator to `track_type_t` and a doc-comment
describing which identity fields it populates and which query method applies. Add a
`distinct_*_tracks()` virtual accessor to `read_statements_base` with a default-empty stub.
Override it in each backend that supports the new type. Add synthesis logic in
`synthesize_derived_tracks()` (v3) or classification logic in `build_v4_tracks()` (v4).
Add the track_query_info_t routing fields and the `get_interval_track` / `get_scalar_track`
dispatch branch in `reader_impl.cpp`. Add paired unit tests for both backends.

**Adding a new schema version.** Create a new class derived from `read_statements_base`,
overriding the accessors relevant to the new schema. Extend the schema-detection logic in
`reader_t::impl::impl()` to detect the new version and instantiate the new backend. The
public `reader_t` interface and the `reader_types` output structures do not change.
