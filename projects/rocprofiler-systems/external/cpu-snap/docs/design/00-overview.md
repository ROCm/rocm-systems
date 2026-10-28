# cpu-snap Design - 00: Overview and Philosophy

This is the map. It is the first file to read in the `docs/design/` set and the
single place that motivates the whole architecture top-down: what the two
layers are, why the codebase is cut exactly where it is, how one CPU sample
travels from a hardware interrupt to a Perfetto trace, the two delivery modes
and when to use each, the runtime-selected capture-backend seam, and the
cross-cutting principles every component obeys.

It is deliberately the big-picture document. It does not restate per-component
APIs or byte tables. For those, follow the links:

Deeper design docs (this set):

- [01-core.md](./01-core.md) - the `src/core/` data plane:
  records, framing, the reader, perf ring, maps, per-tid utilities.
- [02-capture-and-backends.md](./02-capture-and-backends.md) - the `sampler` driver,
  the `i_trigger` seam, `triggers::perf::backend`, `triggers::signal::backend`, unwinding.
- [03-streaming-and-wire.md](./03-streaming-and-wire.md) - live encode:
  `streaming_pipeline`, `temporal_diff_filter`, `frame_trie`.
- [04-offline-processing.md](./04-offline-processing.md) - `symbolizer`,
  `sample_postprocessor`, `frame_filters`, `slice_emitter` / `emitted_slice`.
- [06-design-decisions.md](./06-design-decisions.md) - the decision log:
  variant vs virtual interface, de-thunking, wall-time vs CPU-time, the
  attach-once fix, the signal-teardown leak fix, and the component audit.

Reference docs (one directory up, written for users not maintainers, but useful
ground truth):

- [../architecture.md](../architecture.md) - prose orientation; this design doc
  is its maintainer-facing rationale companion.
- [../components.md](../components.md) - per-component "what is each piece"
  reference (purpose, API, dependents) for all 29 components.
- [../wire-format.md](../wire-format.md) - the exact v1 byte layout.
- [../usage.md](../usage.md) - task-oriented how-to (modes, backends, unwind
  modes, symbolization).
- [../build-and-install.md](../build-and-install.md) - deps, CMake, presets,
  `perf_event_paranoid`.
- [../extending.md](../extending.md) - the two extension points (a capture
  backend, an output sink).

Repo-root artifacts:

- [../../planning/architecture-sampler-abstraction.md](../../planning/architecture-sampler-abstraction.md)
  - the design plan and locked decisions behind the backend seam. The source of
  most of the rationale captured in [06-design-decisions.md](./06-design-decisions.md).

---

## 1. What cpu-snap is

cpu-snap is a Linux, x86_64, in-process profiling library suite written in
C++20. It samples the CPU callstack of every thread in the calling process. It
turns those samples into a compact, self-describing byte stream
that a host turns into a Perfetto trace (or any other format).

Two facts shape everything else:

- It is in-process. There is no separate profiler daemon, no ptrace, no
  attaching from outside. The library runs threads inside the target and reads
  its own process.
- It never opens a file descriptor for output. The host owns persistence: the
  library hands completed byte spans to a host-supplied `sink_fn` callback, and
  the host decides whether those bytes go to a file, a socket, or a RAM ring.
  This keeps the library free of I/O policy and makes it embeddable.

It ships as one CMake package with two independently selectable components
(`cpu_snap::core`, `cpu_snap::sampling`), each its own source
tree and namespace. A host links only what it needs; `core` carries no `libdw`
dependency, only `sampling` does.

---

## 2. The two layers

cpu-snap is layered so that one thing - the on-wire byte format - is the only
contract shared across the whole suite, and so that the recorder (sampling)
sits on top of one common substrate.

```
+-----------------------------------------------------------------------+
|  Layer 2: capture + processing                                        |
|                                                                       |
|   src/sampling/  (cpu_snap::sampling)                                    |
|     capture:   sampler -> i_trigger (perf | signal)             |
|                triggers::perf::kernel_drain / triggers::signal::snapshot_ring / dwarf_unwinder|
|     streaming: streaming_pipeline -> temporal_diff_filter -> frame_trie|
|     offline:   symbolizer / sample_postprocessor / frame_filters /    |
|                slice_emitter -> emitted_slice                          |
+-----------------------------------------------------------------------+
                |  depends on (downward only)
                v
+-----------------------------------------------------------------------+
|  Layer 1: core data plane (cpu_snap::core)                            |
|                                                                       |
|   wire:  records, types, chunk_framer, reader                         |
|   os:    perf_ring, dl_module_watcher, proc_maps, thread_comm, time        |
|   hot:   flat_tid_map                                                  |
+-----------------------------------------------------------------------+
                |
                v
   Linux: perf_event_open + mmap rings, POSIX timers + realtime signals,
          /proc, dl_iterate_phdr, process_vm_readv
   elfutils libdwfl (sampling only), addr2line (offline only)
```

The arrows point one way. Layer 2 depends on Layer 1; Layer 1 never depends back
on Layer 2.

### Layer 1 - core: the backend-agnostic data plane

`src/core/` is the substrate. It knows nothing about how a sample was captured; it
knows only how to frame, seal, ship, and decode bytes, plus the small OS-facing
helpers both recorders need.

Why it exists as a distinct layer: the wire format is the heart of the suite and
the only contract the recorder shares with the rest of the substrate. Isolating
it means the format has exactly one source of truth (`records.hpp`), and it can
be tested in isolation (the producer `chunk_framer` and the consumer `reader`
are exact inverses). Its members and the byte layout are covered in
[01-core.md](./01-core.md) and
[../wire-format.md](../wire-format.md); briefly:

- `records` - the single source of truth for the v1 wire format: the
  `record_type` tag enum (`0x01` new_node through `0x07` eof; `0x08+`
  reserved), the per-record body structs, the `record_variant` sum type, and the
  pragma-packed file/chunk framing structs. Every wire component depends on it.
  It exists as a leaf so the format cannot drift between producer and consumer.
- `types` - the pipeline ABI aliases `sink_fn` (host byte sink) and `event_fn`
  (reader-side record callback) plus the default chunk size. Kept deliberately
  thin so public headers can name the ABI without pulling
  `chunk_framer`'s heavy includes.
- `chunk_framer` - the producer. Serializes `record_variant` values into ~1 MiB
  framed chunks (header + body) and hands each completed chunk plus
  the 64-byte `file_header` to a `sink_fn`. Holds a 1 MiB inline buffer (do not
  stack-allocate); records never span a chunk; sink exceptions are caught and
  counted. Exists to be the one place that knows how bytes are framed.
- `reader` - the consumer, and the exact inverse of `chunk_framer`. A
  push-parser: `feed(span)` buffers bytes, decodes each
  record, invokes `event_fn`, and rebuilds the new_node trie so
  `resolve_stack(leaf_id)` returns the innermost-first IP chain. `feed()` is
  `noexcept`; every error folds into `reader_counters`. It exists offline,
  decoupled from capture, which lets any sink (memory, file, socket) feed
  the same offline side (section 4).
- `perf_ring` - RAII owner of `perf_event_open` mmap-ring resources with a
  decoder-templated `drain()` loop. It owns lifetime and draining only; it does
  not arm or disable the fd. It exists so `triggers::perf::kernel_drain` does not
  reimplement mmap-ring bookkeeping.
- `dl_module_watcher` (streaming-time) and `proc_maps` (offline) - the loaded-module
  observers. `dl_module_watcher` walks `dl_iterate_phdr`, reads each ELF build-id, and
  emits the initial `maps_init` plus add/remove `maps_delta` records so a remote
  reader can attribute IPs to binaries. `proc_maps` is its offline counterpart:
  it snapshots `/proc/self/maps` and translates a runtime IP into (binary path +
  file offset) for addr2line. Two components for one concern because the live
  encode side and the offline symbolize side have different needs (stream deltas
  vs random-access lookup).
- `flat_tid_map` - a per-tid container tuned for the hot path: a 256-slot
  direct-mapped table with an `unordered_map` overflow fallback. It exists so the
  single-worker streaming path can find per-tid state without a hash lookup in
  the common (clustered, low-numbered tid) case.
- `thread_comm` and `time` - small `/proc/self`-backed helpers for thread/process
  comm-name lookup, live-tid enumeration, and wall-clock nanoseconds. They are
  cross-cutting utilities, not pipeline stages; isolating them keeps the
  recorders from each reimplementing `/proc` parsing.

### Layer 2a - sampling: capture and processing

`src/sampling/` is the largest component and divides cleanly into three concerns,
which is also how the deeper docs are split:

- Capture (the hot path): the `sampler` driver, the `i_trigger` seam, the
  two backends (`triggers::perf::backend`, `triggers::signal::backend`) and their machinery
  (`triggers::perf::kernel_drain`, `triggers::signal::snapshot_ring`, `dwarf_unwinder`,
  `sampler_capabilities`, `unwind_mode`), and the data they produce
  (`raw_sample`, `frame_limits`). See
  [02-capture-and-backends.md](./02-capture-and-backends.md).
- Live streaming encode: `streaming_pipeline` composing `temporal_diff_filter`
  over `frame_trie` (plus `core::chunk_framer` and `sampling::dl_module_watcher`). See
  [03-streaming-and-wire.md](./03-streaming-and-wire.md).
- Offline shaping: `symbolizer`, `sample_postprocessor`, `frame_filters`, and
  `slice_emitter` producing `emitted_slice`. See
  [04-offline-processing.md](./04-offline-processing.md).

Why the three concerns are separated: they run at three different times with
three different cost budgets. Capture runs under an interrupt or signal and must
not allocate or block. Streaming encode runs on the worker thread and must be
cheap and `noexcept`. Offline shaping runs after `stop()` and may freely
allocate, shell out to addr2line, and build maps. Keeping them in separate
components stops the slow concerns from leaking includes or costs into the fast
ones - the same reason `raw_sample` and `frame_limits` are leaf headers (below).

---

## 3. End-to-end data flow

There are two recording modes, fixed once at sampler construction by whether a
`sink_fn` is supplied. Both converge on the same offline rendering side.

```
   workload threads (the host's own threads)
        |
        |  (1) perf IRQ on a perf_event_open ring  -- OR --
        |      per-thread POSIX timer fires a realtime signal
        v
   capture backend  (triggers::perf::backend | triggers::signal::backend, chosen at runtime)
        |  (2) one worker thread waits, drains a batch, decodes/unwinds
        |      each event into innermost-first frames
        v
   streaming_pipeline  (the ONE concrete sink)
     temporal_diff_filter
       -> frame_trie (intern)
       -> chunk_framer (seal)
       -> sink_fn(bytes)
        |
        v
   host I/O: file / socket / core::memory_sink
        |
        v
   core::reader.feed(bytes)        (decode, rebuild trie)
   reader.resolve_stack(leaf)      -> innermost-first IPs
        |
        v
   per-tid raw_sample  (core::sample_reconstructor)
                                       |
        (3) offline shaping: sample_postprocessor / slice_emitter
            + symbolizer + frame_filters
                                       |
                                       v
                               emitted_slice  (backend-neutral boundary)
                                       |
                                       v
                       output backend (e.g. the example Perfetto writer
                       -> .pftrace; or pprof / folded stacks / rocpd)
```

Step by step:

1. The kernel takes a sample on a `perf_event_open` interrupt (perf backend), or
   a per-thread POSIX timer fires a realtime signal (signal backend). Either way
   the raw event lands in a ring buffer (the kernel mmap ring for perf, a
   per-thread lock-free SPSC ring for signal).
2. The sampler's single worker thread waits, drains a batch off the ring,
   decodes or unwinds each event into innermost-first frames, and hands the
   4-tuple `(tid, ts_ns, ip, span<frames>)` to the one `streaming_pipeline` (the `ip` is dropped;
   the trie stores per-node IPs instead), which interns the changed suffix into
   the `frame_trie`, emits any NEW_NODE records before the SAMPLE that
   references the new leaf, and lets `chunk_framer` seal ~1 MiB chunks out
   through `sink_fn`.
3. Offline, the streamed bytes are replayed through `core::reader`, which
   decodes records, rebuilds the trie, and resolves a `leaf_node_id` back to its
   IP chain via `resolve_stack`; `core::sample_reconstructor` wraps this into
   per-tid `raw_sample`s that feed
   the offline shaping pass (`sample_postprocessor` and/or `slice_emitter`, with
   `symbolizer` + `frame_filters`), producing `emitted_slice` records that an
   output backend turns into a trace.

The `emitted_slice` is the backend-neutral output boundary. The bundled Perfetto
writer (`examples/common/perfetto_proto_writer.hpp`) is a pure sink over it,
demo glue rather than an installed target. Any consumer can add another output
format without touching the library - this is one of the two documented
extension points ([../extending.md](../extending.md)).

---

## 4. Delivery: always streaming

Samples are encoded live by `streaming_pipeline` into the v1 chunk byte-stream
and pushed to the host `sink_fn` as ~1 MiB chunks. A sink is mandatory
(`streaming_options.sink`); the sampler throws without one. The host owns those
bytes (file, socket, RAM ring); the library never opens an fd.

Two ways to consume the stream:

- Same-process analysis: pair the bounded `core::memory_sink` with
  `core::sample_reconstructor`, which rebuilds per-tid `raw_sample`s after
  `stop()`. Simplest flow for short, contained runs.
- Long or unbounded runs, or shipping the trace off-box: write the chunks to a
  file or socket and replay them later through `core::reader`.

The trie-diff compression collapses a recurring stack to a single 16-byte SAMPLE
record, so steady-state cost is roughly one record per sample regardless of
stack depth. All `unwind_mode`s, including `dwarf`, work (see
[06-design-decisions.md](./06-design-decisions.md)).

Every sink converges on the identical offline shaping side (section 3), so
output format and filtering code is written once.

---

## 5. The capture-backend abstraction (high level)

How a stack is physically captured is the one thing that genuinely differs
between mechanisms, so it is the one thing abstracted. Everything else in the
sampler - the worker loop, the 100 ms `/proc/self/task` rescan, fork handling,
stats, pause/resume, streaming - is mechanism-neutral and
lives in the single concrete public `sampler` class.

The seam has a deliberate two-tier shape, and the tiering is the whole point:

- The per-batch operations are virtual. The driver holds whichever backend it
  built through a `std::unique_ptr<i_trigger>`, chosen at runtime from
  `config.backend`.
- The per-sample path is NOT virtual. `i_trigger::drain_into` takes the
  concrete `streaming_pipeline` by reference, so per-sample dispatch stays direct
  and inlinable inside the backend. Virtual dispatch happens only at the
  coarse, cold `drain_into` / `wait` / `attach_tid` granularity - one indirect
  call per drain batch, never per sample.

The abstract `i_trigger` class (a Strategy seam) names exactly the neutral
operations: per-tid `attach_tid`/`detach_tid`, blocking `wait`, batch
`drain_into`, `enable_all`/`disable_all`, `close_all`, and tid bookkeeping
(`tracked_tids`, `is_tracked`, `collect_tids`). `attach_tid` is the only
operation allowed to throw (resource acquisition); every other operation is
`noexcept`.

Two backends implement it:

- `triggers::perf::backend` (default) - a thin adapter over `triggers::perf::kernel_drain`
  (`perf_event_open` per tid + mmap ring + epoll wait, decoding the FP /
  DWARF record layouts). It needs `perf_event_paranoid <= 2` or `CAP_PERFMON`.
- `triggers::signal::backend` (privilege-free) - per-thread POSIX timers
  (`timer_create` + `SIGEV_THREAD_ID`) delivering a realtime signal; one
  process-wide async-signal-safe handler snapshots registers and a bounded stack
  copy into a per-thread lock-free ring, and the worker unwinds OFF-handler with
  the shared `dwarf_unwinder`. It works where perf is refused (containers,
  restricted CI).

Both backends share the driver-owned `dwarf_unwinder` and drain into the same
`streaming_pipeline`, so everything downstream of capture is identical regardless
of which backend ran. Why this exact abstraction (and why not variant or a
virtual `i_sampler`), plus the wall-time-vs-CPU-time deviation in the v1 signal
backend, are the subject of
[02-capture-and-backends.md](./02-capture-and-backends.md) and
[06-design-decisions.md](./06-design-decisions.md).

---

## 6. Cross-cutting design principles

These hold across all three layers and explain recurring choices in the code.

### Zero-overhead, de-thunked hot path

The per-sample path carries no virtual call, no `std::function`, and no
allocation. This is not incidental: an earlier design used a `std::function`
sample sink that "dominated recorder cycles," so the sink was collapsed into one
concrete type - `streaming_pipeline` - that the backend calls directly so the
compiler inlines the dispatch. The backend abstraction was then explicitly
placed at the cold per-batch granularity precisely so it could not reintroduce
that cost on the hot path. The async-signal handler in `triggers::signal::backend` is the
strictest case: it does only async-signal-safe work (read `ucontext` regs +
`clock_gettime` + `gettid` + a bounded `process_vm_readv`/memcpy stack copy + an
atomic publish), and defers all unwinding, allocation, and locking to the worker.

### Fixed-POD samples, no allocation in capture

`raw_sample` holds `tid`, `timestamp_ns`, `ip` and a `vector<uint64>` of innermost-first
frames. It is built off the hot path (the reconstructor); capture itself decodes into
buffers sized once from `callchain_depth`, so it still allocates nothing per sample,
including in async-signal context. The depth default and sanity bound live in a leaf header,
`frame_limits.hpp`, for one reason: the decode side (`triggers::perf::kernel_drain`) and the
storage side (`raw_sample`) must agree on the constant, but `raw_sample` must NOT
have to include `perf/kernel_drain.hpp` (which would drag in `<linux/perf_event.h>`,
epoll, and mman) just to read a number. Isolating the constant keeps `raw_sample`
a cheap leaf that postprocessor TUs can include without the decode machinery.

### Backend-neutral downstream

Nothing after `streaming_pipeline` knows or cares how the sample was captured.
`raw_sample`, the streaming wire format, the reader, the symbolizer, and the
slice emitter are all written against the captured 4-tuple, not against perf or
signal. This is what lets a second backend (signal) reuse the entire downstream
unchanged. The output side has its own neutral boundary, `emitted_slice`, so
output formats are equally decoupled.

### One wire format, one consumer

The chunk stream is sealed the same way regardless of producer, and a single
`reader` decodes it. The format is self-describing (typed, length-prefixed
records) and chunks carry a sequence number, so a dropped chunk is detected as a
gap and a malformed chunk header is skipped by resync rather than poisoning the
whole stream.

### Header-heavy C++20, value types, RAII

The style is header-heavy with value types, templated sinks where the hot path
needs inlining, virtual interfaces for the coarse seams, and RAII for every OS resource (perf fds,
mmap rings, POSIX timers, Dwfl sessions). `.cpp` bodies are reserved for things
that must be out-of-line: elfutils headers hidden behind a pimpl
(`dwarf_unwinder`), the async-signal-safe bodies under `_GNU_SOURCE`
(`triggers::signal::backend`), and the larger stateful decoders (`reader`,
`streaming_pipeline`, `dl_module_watcher`, `slice_emitter`).

### Downward-only dependencies

Dependencies point strictly down: Layer 2 to Layer 1, never up; `core` carries
no `libdw`. This is what makes the two CMake components independently
selectable and what keeps the wire format the single shared contract. The
audit of the architecture confirmed it holds, with only minor redundancy (for
example `dl_module_watcher` privately reimplementing the `time` helper, and two
tightly-coupled-but-justified pairs noted in
[06-design-decisions.md](./06-design-decisions.md)).

---

## 7. Where to go next

- New to the codebase and want the narrative: read [../architecture.md](../architecture.md),
  then this set in order (01 -> 06).
- Need a specific component's API and dependents: [../components.md](../components.md).
- Working on capture or adding a backend: [02-capture-and-backends.md](./02-capture-and-backends.md)
  and [../extending.md](../extending.md).
- Working on the byte format: [03-streaming-and-wire.md](./03-streaming-and-wire.md)
  and [../wire-format.md](../wire-format.md).
- Working on output formats: [04-offline-processing.md](./04-offline-processing.md).
- Want the "why did we choose X" answers: [06-design-decisions.md](./06-design-decisions.md)
  and [../../planning/architecture-sampler-abstraction.md](../../planning/architecture-sampler-abstraction.md).
