# cpu-snap Architecture

This document is the prose orientation for a developer new to cpu-snap: how the
codebase is split into layers, how a sample travels from a CPU interrupt to a
Perfetto trace, and where the two pluggable capture backends fit. It is the
narrative companion to:

- [`../README.md`](../README.md) - the host-facing usage contract: API
  walk-throughs (in-memory and custom-sink streaming), the wire-format byte tables, build
  options, overhead numbers, and known limitations.

cpu-snap is a Linux, x86_64, in-process profiling library suite. It never opens
a file: the host owns persistence through a `sink_fn` callback. It is published
as one CMake package with three independently selectable components
(`cpu_snap::core`, `cpu_snap::sampling`, `cpu_snap::processing`); see the README
"Components" and "Install and find_package" sections.

## The two layers

cpu-snap is layered so that the on-wire byte format is the only contract shared
across components, and so that the recorder (sampling) sits
on top of one substrate.

```
+-----------------------------------------------------------------+
| Layer 2: capture + processing                                   |
|   src/sampling/  (per-thread stack sampler + offline shaping)   |
+-----------------------------------------------------------------+
                |
                v
+-----------------------------------------------------------------+
| Layer 1: core wire / IO substrate                               |
|   src/core/  (chunk framer, reader, records, perf_ring,         |
|           dl_module_watcher, proc_maps, per-tid + thread utils) |
+-----------------------------------------------------------------+
                |
                v
        Linux: perf_event_open, mmap rings, /proc, dl_iterate_phdr
        elfutils libdwfl (sampling only), addr2line (offline)
```

Layer 2 depends on Layer 1; Layer 1 has no dependency back on
it, and `core` carries no `libdw` dependency (only `sampling` does).

### Layer 1 - core: wire / IO substrate

`namespace cpu_snap::core` (utilities in `cpu_snap::core::utils`). Source under
[`../src/cpu_snap/core`](../src/cpu_snap/core).

The wire format is the heart of the library. `records.hpp` is the single source
of truth: it defines the `record_type` tag enum (codes `0x01` NEW_NODE through
`0x07` EOF; `0x08+` reserved for forward compatibility), the per-record
bodies, the `record_variant` sum type, and the pragma-packed file/chunk
header/footer framing structs. Framing structs are byte-for-byte wire layout;
record bodies are serialized field-by-field, not packed. The exact byte tables
live in the README "Wire format summary".

| Component | File | Role |
|---|---|---|
| `records` | `records.hpp` | Canonical wire record types, `record_variant`, framing structs. Root of the wire dependency tree. |
| `types` | `types.hpp` | The pipeline ABI aliases `sink_fn` (host byte sink) and `event_fn` (host record callback), plus the default chunk size. Kept thin so public headers avoid heavy includes. |
| `chunk_framer` | `chunk_framer.hpp` | Producer side: serializes `record_variant` values into ~1 MiB framed chunks (header + body + footer) and hands each completed chunk plus the 64-byte `file_header` to a `sink_fn`. Holds a 1 MiB inline buffer; do not stack-allocate. Records never span a chunk. Sink exceptions are caught and counted. |
| `reader` | `reader.hpp` + `src/reader.cpp` | Consumer side: push-based decoder that buffers fed bytes, validates each chunk header (resyncing past rejected ones), decodes each record, and invokes an `event_fn`. Rebuilds the NEW_NODE trie so `resolve_stack(leaf_id)` returns the innermost-first IP chain. `feed()` is `noexcept`: all errors are absorbed into `reader_counters`. Largest core component. |
| `perf_ring` | `perf_ring.hpp` | RAII owner of `perf_event_open` mmap-ring resources with a decoder-templated `drain()` loop. Pure lifetime + drain owner; does not arm/disable the fd. Shared by sampling `triggers::perf::kernel_drain`. |
| `dl_module_watcher` | `dl_module_watcher.hpp` + `src/dl_module_watcher.cpp` | Streaming-time loaded-module observer: walks `dl_iterate_phdr`, reads each ELF build-id, emits the initial `maps_init_record` and add/remove `maps_delta_record`s. |
| `proc_maps` | `proc_maps.hpp` | Offline `/proc/self/maps` snapshot; translates a runtime IP into (binary path + file offset) for addr2line. The offline counterpart to `dl_module_watcher`. |
| `flat_tid_map` | `flat_tid_map.hpp` | Per-tid container for the hot path: 256-slot direct-mapped table with an `unordered_map` overflow fallback. |
| `thread_comm` | `thread_comm.hpp` | `/proc/self`-backed thread/process comm-name lookup and live-tid enumeration. |
| `time` | `time.hpp` | Wall-clock helper (system_clock nanoseconds since epoch). |

### Layer 2a - sampling: capture + processing

`namespace cpu_snap::sampling` (kernel-facing and pipeline internals in
`cpu_snap::sampling::internal`; symbolizer in `cpu_snap::sampling::utils`).
Source under
[`../src/cpu_snap/sampling`](../src/cpu_snap/sampling).
This is the largest component. It divides into three concerns: the hot-path
driver and its pluggable backends, the live streaming encode pipeline, and the
offline shaping/symbolization pass.

Driver and backend seam (detailed in "The two capture backends" below):

| Component | File | Role |
|---|---|---|
| `sampler` | `sampler.hpp` + `src/sampler.cpp` | The neutral hot-path driver. Owns a `unique_ptr<i_trigger>` backend, the worker thread (wait / drain / 100 ms tid rescan), fork handling, and the streaming pipeline to the mandatory sink. Does no symbol resolution or filtering. Public entry point for capture. |
| `i_trigger` | `i_trigger.hpp` | Abstract class defining the backend-neutral strategy seam (per-tid attach/detach, wait, batch `drain_into`, pause/resume, teardown, tid bookkeeping). |
| `triggers::perf::backend` | `perf/backend.hpp` | Preferred `i_trigger`: thin adapter over `internal::kernel_drain`. |
| `triggers::signal::backend` | `signal/backend.hpp` + `src/signal/backend.cpp` | Privilege-free `i_trigger`: per-thread POSIX timer + realtime signal. |
| `triggers::perf::kernel_drain` | `perf/kernel_drain.hpp` | Per-tid `perf_event_open` fd + mmap-ring lifecycle and the PERF_RECORD_SAMPLE/LOST decoder for FP / DWARF modes. Built on `sampling::perf_ring`. |
| `triggers::signal::snapshot_ring` | `signal/snapshot.hpp` | Async-signal-safe per-thread snapshot record (regs + bounded stack) plus the lock-free SPSC ring used by `triggers::signal::backend`. |
| `dwarf_unwinder` | `dwarf_unwinder.hpp` + `src/dwarf_unwinder.cpp` | libdwfl (elfutils) DWARF stack unwinder for perf REGS_USER+STACK_USER snapshots. Pimpl keeps elfutils headers out of consumers. Used by both backends. |
| `sampler_capabilities` | `sampler_capabilities.hpp` | Probes which `perf_event_open` features the kernel + CPU accept (precise_ip, IBS). Exposed via `sampler::caps()`. |
| `unwind_mode` | `unwind_mode.hpp` | Enum of the three stack-walk strategies: `auto_best`, `frame_pointer`, `dwarf`. |
| `raw_sample` | `raw_sample.hpp` | POD for one captured stack sample (tid, timestamp, ip, innermost-first frames). The central in-memory data type. |
| `frame_limits` | `frame_limits.hpp` | Default and sanity bound for the runtime callchain depth, kept as a leaf so capture and storage share them. |

Live streaming encode pipeline:

| Component | File | Role |
|---|---|---|
| `streaming_pipeline` | `streaming_pipeline.hpp` + `src/streaming_pipeline.cpp` | Single-thread facade turning per-sample frame arrays into the v1 byte stream. Composes `temporal_diff_filter` + `frame_trie` + `core::chunk_framer` + `sampling::dl_module_watcher`. `noexcept` end-to-end; emits NEW_NODE before the SAMPLE that references the new leaf. |
| `temporal_diff_filter` | `temporal_diff_filter.hpp` | Per-tid running-stack tracker: interns only the changed suffix vs the prior sample on the same tid, so an unchanged stack costs zero trie ops. |
| `frame_trie` | `frame_trie.hpp` | Grow-only Flyweight trie keyed on `(parent_node_id, ip)` edges; collapses a recurring stack to one u32 leaf id. The compression core. |

Offline shaping / symbolization (run after `sampler::stop()`, or fed from
`reader` output):

| Component | File | Role |
|---|---|---|
| `sample_postprocessor` | `sample_postprocessor.hpp` | Offline pass over `raw_sample`s: symbolizes IPs (cached per IP), applies frame predicates, feeds survivors to a sink. Also `coalesce_frames_by_name`. |
| `slice_emitter` | `src/cpu_snap/processing/timeline/slice_emitter.hpp` | Timeline-shaping pass: converts per-tid `raw_sample` vectors into `emitted_slice` records (nested flame-graph slices). The single backend-neutral timeline owner. |
| `symbolizer` | `symbolizer.hpp` | Batched IP -> `function(args) [binary]` resolver bundling `processing::symbols::proc_maps` with `addr2line_client`; groups IPs by binary so addr2line runs once per binary. Exposes inline chains + file:line. |
| `frame_filters` | `frame_filters.hpp` | Reusable `frame_predicate` builders (main-exe-only, library-substring exclude). `sample_postprocessor` uses them behind its own `drop_library_frames` toggle; `slice_emitter` takes a ready-made predicate via its `library_frame_filter` callback field instead (it does not link `symbols`/`proc_maps` itself). |

The `emitted_slice` produced by `slice_emitter` is the boundary to output
backends. The bundled Perfetto writer
([`../examples/common/perfetto_proto_writer.hpp`](../examples/common/perfetto_proto_writer.hpp))
is a pure sink that maps `emitted_slice` to native trace records; it is demo
glue, not an installed library target. Any consumer can add another format
(pprof, folded stacks, rocpd) without touching the library.

## Data flow: capture to Perfetto

Every capture streams to a mandatory `sink_fn`. For same-process analysis the
sink is `core::memory_sink`; for long runs it is a file or socket writer. Both
converge on the same offline rendering side.

```
   workload threads
        |
        |  (1) perf IRQ / timer signal -> kernel ring or SPSC ring
        v
  capture backend (triggers::perf::backend | triggers::signal::backend)
        |  (2) worker thread drains a batch, decodes/unwinds
        v
  streaming_pipeline
  (temporal_diff_filter -> frame_trie -> chunk_framer -> sink_fn(bytes))
        |
        v
  host I/O: file / socket / core::memory_sink
        |
        v
  core::reader.feed(bytes)            (decode, rebuild trie)
  reader.resolve_stack(leaf_id)       -> innermost-first IPs
        |
        v
  per-tid raw_sample (core::sample_reconstructor)
                                |
        (3) offline shaping: sample_postprocessor / slice_emitter
            + symbolizer + frame_filters
                                |
                                v
                        emitted_slice -> output backend
                        (perfetto_proto_writer -> .pftrace)
```

Step by step:

1. The kernel takes a sample on a `perf_event_open` interrupt (perf backend) or
   a per-thread POSIX timer fires a realtime signal (signal backend). The raw
   event lands in a ring buffer.
2. The sampler's single worker thread waits, then drains a batch off the ring,
   decoding/unwinding each event into innermost-first frames and handing it to
   the one `streaming_pipeline`, which interns the changed suffix into the
   `frame_trie`, emits any NEW_NODE records before the referencing SAMPLE, and
   lets `chunk_framer` seal ~1 MiB chunks out through `sink_fn`. The host owns
   those bytes (file, socket, RAM ring) - the library never opens an fd.
3. Offline, the streaming bytes are replayed through `core::reader`, which
   decodes records, rebuilds the trie, and resolves a `leaf_node_id` back to
   its IP chain via `resolve_stack` (`core::sample_reconstructor` wraps this
   into per-tid `raw_sample`s). Those feed the offline shaping pass (`sample_postprocessor` / `slice_emitter` +
   `symbolizer` + `frame_filters`), producing `emitted_slice`s that an output
   backend (the example Perfetto writer) turns into a trace.

The README "Library usage (detailed)" and "Streaming pipeline" sections give the
full code-level walk-through; this is the structural summary.

## The two capture backends

How a stack is physically captured is isolated behind the `i_trigger`
seam, so the rest of the sampler (worker loop, fork handling, tid rescan,
streaming) is mechanism-neutral. The seam has a deliberate two-tier shape:

- Per-batch operations are virtual. The driver holds whichever backend it built
  through a `std::unique_ptr<i_trigger>`.
- The per-sample path is NOT virtual. `i_trigger::drain_into` takes the
  concrete `streaming_pipeline` by reference, so per-sample dispatch stays direct
  and inlinable inside the backend. Virtual dispatch happens only at the coarse
  per-batch granularity. (See the class doc comment in `i_trigger.hpp`.)

The interface requires per-tid `attach_tid`/`detach_tid`, blocking `wait`, batch
`drain_into`, `enable_all`/`disable_all`, `close_all`, and tid bookkeeping
(`tracked_tids`, `is_tracked`, `collect_tids`). `attach_tid` is the only
operation allowed to throw (resource acquisition); everything else is
`noexcept`. The backend is chosen at sampler construction via
`config::backend` (`backend_kind::automatic` default: perf when a real
`perf_event_open` probe succeeds, else signal; explicit `perf` throws if refused,
explicit `signal` never probes). `sampler::selected_backend()` reports the chosen
kind and the fallback reason; the library never writes to stderr.

### triggers::perf::backend (preferred)

[`perf/backend.hpp`](../src/cpu_snap/sampling/triggers/perf/backend.hpp).
A thin adapter that owns a `triggers::perf::kernel_drain`. `drain_into` forwards two
lambdas into `triggers::perf::kernel_drain::drain_all`, keeping the per-sample decode inside
`triggers::perf::kernel_drain` (inlined, no virtual / no `std::function`). It opens one
`perf_event_open` fd + mmap ring per tid (via `sampling::perf_ring`), shares an
epoll fd for level-triggered waits, and decodes the FP / DWARF record
layouts. This path needs
`perf_event_paranoid <= 2` or `CAP_PERFMON`.

### triggers::signal::backend (privilege-free)

[`signal/backend.hpp`](../src/cpu_snap/sampling/triggers/signal/backend.hpp)
+ `src/signal/backend.cpp`. The async-signal-safe-critical bodies (the handler,
timer arm, ucontext snapshot) live in the `.cpp` under `_GNU_SOURCE`. A
per-thread POSIX timer (`SIGEV_THREAD_ID`) delivers a realtime signal; one
process-wide handler async-signal-safely snapshots registers + a bounded stack
copy into a per-thread lock-free SPSC ring (`triggers::signal::snapshot_ring`). The worker
drains off-handler and unwinds via libdwfl. Its timer uses `CLOCK_MONOTONIC` (a
CPU-time clock would charge the worker, not the sampled thread). Only one signal
backend can be active at a time because the handler is process-wide; detached
rings are parked in a graveyard until destruction to close the use-after-free
window against an in-flight signal.

Both backends share `dwarf_unwinder` (driver-owned, must outlive the backend)
and drain into the same `streaming_pipeline`, so everything downstream of capture
is identical regardless of which backend ran. Unwind-mode selection (`auto_best`
-> dwarf on both backends; `frame_pointer` is opt-in)
is documented in the README "Unwind mode selection".

## Where to go next

- For API usage, the wire-format byte tables, build/install, and overhead
  numbers, read [`../README.md`](../README.md).
