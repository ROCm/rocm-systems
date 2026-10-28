# 03 - Streaming and the wire format

This is the in-depth design document for the cpu-snap streaming pipeline and the
v1 wire format it produces. It explains WHY each piece exists, top to bottom: the
out-of-process delivery model, the trie-diff compression, the ordering invariant
that makes forward-replay decoding possible, the resync machinery (checkpoints,
LOST), and how the reader rebuilds state and resolves a stack.

It deliberately does NOT re-specify the byte layout. The authoritative
field-by-field layout lives in [../wire-format.md](../wire-format.md); this
document covers roles and rationale and points at that file for the bytes.

Audience: a developer who must understand and maintain the whole project. Read
[./00-overview.md](./00-overview.md) first for the layering, then this document
for the data plane, then [./02-capture-and-backends.md](./02-capture-and-backends.md) for how samples are
produced and [./04-offline-processing.md](./04-offline-processing.md) for how
they are rendered.

## Where this fits

```
   capture (src/sampling/, doc 01)            offline render (doc 02)
   sampler -> backend -> streaming_pipeline     symbolizer / postproc / slice_emitter
        |                                                ^
        | per-sample frames                              | record_variant + resolve_stack
        v                                                |
   +--------------------- THIS DOCUMENT ---------------------+
   | streaming_pipeline                          reader     |
   |   temporal_diff_filter -> frame_trie           ^       |
   |   chunk_framer (core) -----> sink_fn --[bytes]--+       |
   |   dl_module_watcher (core)                                  |
   +-------------------------------------------------------+
                          ^
                          | wire format (src/core/: records, types)
```

Cross-references for the components this document only touches in passing:

- Capture side (`sampler`, `triggers::perf::kernel_drain`, `triggers::perf::backend`, `triggers::signal::backend`,
  `dwarf_unwinder`, `raw_sample`, `frame_limits`,
  `i_trigger`, `triggers::signal::snapshot_ring`, `unwind_mode`,
  `sampler_capabilities`): [./02-capture-and-backends.md](./02-capture-and-backends.md).
- Offline side (`symbolizer`, `sample_postprocessor`, `frame_filters`,
  `slice_emitter`, `proc_maps`): [./04-offline-processing.md](./04-offline-processing.md).
- Design decisions index: [./06-design-decisions.md](./06-design-decisions.md).
- Per-component reference: [../components.md](../components.md). Architecture
  Byte layout: [../wire-format.md](../wire-format.md).

---

## 1. Why a streaming pipeline exists at all

cpu-snap has one output mode: every captured sample is encoded to bytes as it
arrives and pushed out through a mandatory host callback (`sink_fn`). The sampler
keeps no in-RAM sample map. For same-process analysis the host passes the bounded
`core::memory_sink` and rebuilds `raw_sample`s after `stop()` with
`core::sample_reconstructor` (doc 02); for long runs it writes the bytes to a
file or socket. The streaming pipeline is the encoder behind that.

Three forces shape it:

1. **Out-of-process / unbounded-run delivery.** A long-running or
   never-terminating process cannot buffer every sample in RAM. Streaming bounds
   memory to a single in-flight chunk plus the (grow-only but tiny) trie, so a
   24/7 service can be sampled and the bytes shipped to a file, pipe, socket, or
   ring buffer continuously. The library never opens a file descriptor; where the
   bytes go is entirely the host's decision (see the `sink_fn` contract below).
2. **Compressed delivery.** Writing every IP of every stack is enormous at high
   sampling rates. The streaming path interns stacks into a trie and emits one
   `u32` leaf id per sample in steady state, so the per-sample wire cost collapses
   to a 16-byte `SAMPLE` record (see section 3). This is only possible on the
   live path, where successive samples on a thread are temporally adjacent and
   almost always share a stack prefix.
3. **noexcept hot path.** The sample-producing worker thread must never have an
   exception escape into it, and must never block. The pipeline's `on_*` methods
   are `noexcept` end to end and absorb every internal failure into a counter (see
   section 6).

The sampler hands this pipeline to the backend as the per-sample sink at `start()` (doc 01).

### Push, not pull; bytes, not records

The producer/consumer contract is push-based on both ends, and the unit crossing
the boundary is opaque bytes, not typed records:

- The producer (`chunk_framer`) hands completed byte spans to a host `sink_fn`.
- The consumer (`reader`) takes arbitrary byte slices via `feed()` and reassembles
  records itself.

This is the key decoupling: the library serializes to a self-describing
byte container and the host owns transport and persistence. The
encode and decode sides never share a live object; they share only the wire
format. That is why `src/core/` (the wire format + reader) has no dependency on
`src/sampling/` (the encoder).

---

## 2. The data plane: `src/core/`

`src/core/` is the backend-agnostic data plane. It defines the wire format and the IO
state machines, and nothing in it knows about callstacks or perf.
Dependencies point downward only: `src/sampling/` depends on `src/core/`, never
the reverse. This section documents every `src/core/` component the streaming path
uses and why it is shaped the way it is.

### 2.1 `records` - the single source of truth for the wire

`src/cpu_snap/core/records.hpp` defines the format once: the
`record_type` tag enum, every record body struct, the `record_variant` sum type,
`type_of()`, and the two packed framing structs (`file_header` 64 B,
`chunk_header` 16 B).

Two deliberate layout decisions:

- **Framing structs are `#pragma pack(1)` and `static_assert`-sized.** For
  `file_header`/`chunk_header` the struct layout IS the wire layout,
  so they can be `memcpy`-ed in and out. The `static_assert(sizeof == N)` guards
  against an accidental field reorder silently changing the format.
- **Record bodies are NOT packed.** They are serialized field by field in
  little-endian order (see `chunk_framer`'s `encode_body` overloads), so the
  on-wire layout is independent of host struct padding. This is why the in-memory
  `sample_record` can stay naturally aligned (fast to fill on the hot path) while
  the wire stays canonical. The `static_assert(sizeof(sample_record) == 16)` on the
  fixed bodies catches an accidental field add, since several downstream tests
  assume that in-memory size.

`record_variant` is the type the framer accepts and the reader produces. It is the
one place sampling records (`new_node`, `sample`, `checkpoint`, `lost`, `eof`)
and the shared maps records (`maps_init`, `maps_delta`) all coexist.

Why a single source of truth matters: the framer's `body_size_of`/`encode_body`
and the reader's `decode_*` are two halves of the same contract. Keeping the
structs, tags, and version constants in one header means the producer and consumer
cannot drift. The version constants (`k_record_format_version_major/minor`,
currently 2.0) live here too.

### 2.2 `types` - the public pipeline ABI

`src/cpu_snap/core/types.hpp` holds the two function aliases that form
the public ABI of the whole pipeline, plus `k_default_chunk_bytes` (1 MiB):

- `sink_fn = std::function<void(std::span<const std::byte>)>` - the host's byte
  sink (output side).
- `event_fn = std::function<void(const record_variant&)>` - the host's record
  callback (reader side).

Why these are `std::function` and not concrete callables: they sit at the COLD,
coarse boundary (once per emitted chunk, once per decoded record), not on the
per-sample hot path. Type-erasing here buys a stable ABI - the host can pass any
callable without the library binding to a concrete type - at a cost paid only
per-chunk. Contrast this with the per-sample dispatch inside capture, which is
deliberately de-thunked (no `std::function`) because a `std::function` sink once
"dominated recorder cycles"; see doc 01 and `streaming_pipeline`.

`k_default_chunk_bytes` lives in `types.hpp` rather than `chunk_framer.hpp` on
purpose: the `sampling` public header defaults its config to it without
having to pull in `chunk_framer`'s heavy transitive includes. The `sink_fn`
contract is strict and worth restating: it is invoked serialized from the
collector worker thread (never reentrant), must not block, and the span is valid
only inside the call - copy out to retain.

### 2.3 `chunk_framer` - the producer

`src/cpu_snap/core/chunk_framer.hpp` is the serializer. It turns
`record_variant` values into framed chunks and hands the 64-byte file header (once)
plus each completed chunk to the `sink_fn`.

Design points and their reasons:

- **Inline 1 MiB `alignas(64)` buffer.** The framer holds one cache-line-aligned
  buffer sized at the 1 MiB max. There is one framer per pipeline per sampler, so
  one ~1 MiB object per process is negligible, and an inline buffer means zero
  per-chunk allocation on the hot path. The cost is that a `chunk_framer` must not
  be stack-allocated on a small thread stack - hence the streaming pipeline owns
  it via pimpl on the heap.
- **Records never span a chunk.** Before appending, the framer checks whether the
  padded record fits under the soft `chunk_bytes` threshold; if not it seals and
  emits the current chunk and opens a fresh one. This means a reader can validate
  and decode one chunk in isolation - a corrupt chunk costs you only the records
  inside it, never a half-record straddling a boundary.
- **4-byte inter-record alignment.** Each record is padded to a 4-byte boundary so
  the next record header is aligned; the trailing pad of the last record is NOT
  counted in `records_byte_count`, so the reader knows exactly how many body bytes
  to read.
- **Header back-filled at seal time.** The header's 16 bytes are reserved at
  offset 0 and filled in by `seal_and_emit_chunk()` once the body length is
  known.
- **Sink exceptions are caught and counted, never propagated.** `emit_to_sink`
  swallows host-callback throws into `sink_errors()`. A misbehaving sink must not
  kill the collector worker. This is one half of the pipeline's end-to-end
  noexcept guarantee (the framer is called from inside the pipeline's `try` blocks
  too, but this is belt-and-suspenders for the sink specifically).
- **Little-endian only, `static_assert`-enforced.** There is no byte-swap path;
  the producer asserts a little-endian host at compile time. cpu-snap targets
  x86_64 Linux, so this is a simplification, not a limitation.

The framer is non-reentrant and move/copy deleted (it owns a large buffer and a
sink). Variable-size records (`checkpoint`, `maps_init`, `maps_delta`)
get a `body_size_of` overload so the framer can pre-compute the
padded size and decide whether the record fits before encoding.

### 2.4 `reader` - the consumer

`src/cpu_snap/core/reader.hpp` + `src/cpu_snap/core/reader.cpp` is the symmetric
counterpart of `chunk_framer`. `feed(span)` is the sole mutating entry point and
**never throws**: parse errors, rejected chunk headers, unknown record types, and
host-callback exceptions are all folded into `reader_counters` instead of
propagating. This mirrors the producer's "a bad chunk costs only that chunk"
philosophy on the decode side - a streaming consumer must survive a corrupt input.

The implementation is three collaborators in the `.cpp`'s anonymous namespace; the
split exists so each concern is independently testable and the state machine stays
legible:

- **`chunk_assembler`** - the framing state machine
  (`expect_file_header -> expect_chunk_header -> expect_chunk_body`, then back
  to header). It accumulates exactly the next expected byte count into a pending
  buffer (header) or the body buffer, and on a complete body hands the body span
  to the walker. It buffers partial headers/bodies across
  `feed()` calls, so the host can feed bytes in any granularity (one byte at a
  time, or a megabyte) and the reader does not care.
- **`record_walker`** - walks one verified body, reading each 4-byte record
  header, bounds-checking `body_size`, decoding the body, dispatching to the host
  `event_fn`, and advancing by `align4(4 + body_size)` to the next record.
- **`trie_resolver`** - a `node_id -> {parent_id, ip}` map populated from every
  decoded `new_node`, used by `resolve_stack`. This is the reader's own rebuilt
  view of the encoder's trie (section 4).

The error policy is layered by severity, and the layering is the point:

- **Fatal** (bad file-header magic or unsupported major version): sets the fatal
  flag, bumps `parse_errors`, and silences all further emissions. The stream is
  not cpu-snap or not a version we understand; there is nothing safe to do.
- **Implausible chunk header** (bad `CH` magic/version, `records_byte_count`
  larger than the advertised chunk size, or an implausible `chunk_seq` jump):
  counted in `parse_errors` or `chunks_rejected`, then the reader forward-scans
  the pending bytes for the next plausible `CH` header (`resyncs`,
  `bytes_skipped`).
- **Per-record errors** (size mismatch, truncated body, unknown/`invalid` tag):
  bump `parse_errors` and abandon the rest of that chunk body.
- **Host callback exceptions**: caught and swallowed; a host bug is not a stream
  error.

`reader_counters` (`bytes_fed`, `records_emitted`, `parse_errors`, `gap_count`,
`last_seen_seq`, `unknown_records`, `chunks_rejected`, `resyncs`,
`bytes_skipped`) is the health telemetry. `gap_count` is derived from breaks in
the `chunk_seq` run, which is how a consumer detects dropped chunks.

### 2.5 `dl_module_watcher` - streaming-time module tracking

`src/cpu_snap/sampling/dl_module_watcher.hpp` + `.cpp` walks `dl_iterate_phdr` to
enumerate loaded shared objects (plus the main executable and the vDSO), reads each
ELF `.note.gnu.build-id`, and produces a `maps_init_record` (`snapshot()`, full
list) and `maps_delta_record`s (`poll()`, add/remove diffs).

Why it lives in the streaming path and why it matters: the stream carries raw
instruction pointers, not symbols. To turn an IP captured at time T back into a
function, the offline consumer needs to know which binary was mapped at that
address at time T. `dlopen`/`dlclose` can race the recording, so a single snapshot
at the start is not enough - the watcher emits deltas as the module set changes,
letting the offline symbolizer pin every IP to the binary that was mapped when it
was sampled. The pipeline emits the initial `maps_init` right after the file header
and then polls on a sample-count cadence (section 5). It is the streaming-time
twin of `proc_maps` (the offline `/proc/self/maps` reader used by the symbolizer,
doc 02). The `set_iterator_for_tests` seam lets tests drive deterministic
add/remove scenarios without touching the dynamic linker. The identity key is
`(base_va, path)`; a same-key size/offset/build_id change emits no delta (a v1
simplification).

### 2.6 `flat_tid_map` - per-tid state, hot-path shaped

`src/cpu_snap/sampling/flat_tid_map.hpp` is a per-tid associative container:
a 256-slot direct-mapped table keyed by `(tid mod 256)` with a short linear probe,
backed by a `std::unordered_map` overflow when the slot cluster fills.

Why not just `std::unordered_map`: the streaming pipeline touches per-tid state on
every sample (the diff filter's prior path, the pipeline's per-tid counters). A
`std::unordered_map` lookup is a bucket -> node -> {key,value} pointer chase; the
direct-mapped table absorbs almost all lookups with one indexed access. Linux
assigns tids roughly sequentially from `/proc/self/task`, so adjacent threads land
in adjacent slots and the probe runs are short. It is used by `temporal_diff_filter`
(per-tid prior path) and by the pipeline itself (per-tid counters). Not
thread-safe - it is a single-worker hot-path structure.

### 2.7 `perf_ring`, `thread_comm`, `time`

These are `src/core/` members used mainly by capture and bookkeeping, included here for
completeness:

- **`perf_ring`** (`src/cpu_snap/sampling/perf_ring.hpp`) - RAII owner of
  `perf_event_open` mmap-ring resources with a decoder-templated `drain()`. It is
  the substrate `triggers::perf::kernel_drain` builds on so the
  ring management is written once. It is upstream of this document (it feeds
  capture, not the encoder); see doc 01.
- **`thread_comm`** (`src/cpu_snap/sampling/thread_comm.hpp`) - `/proc/self`
  identity helpers (`read_thread_comm`, `read_proc_self_comm`,
  `list_self_thread_tids`). The pipeline indirectly relies on `read_proc_self_comm`
  via the framer's default process name.
- **`time`** (`src/cpu_snap/sampling/time.hpp`) - `wall_unix_ns_now()`,
  `system_clock` nanoseconds since the Unix epoch, used only for the header's
  `start_unix_ns`; `clock_now_ns(clock_kind)` stamps the `LOST`, `CHECKPOINT`
  and `EOF` `wall_ns` fields with the configured stream clock.

---

## 3. Trie-diff compression and the ordering invariant

This is the central size win, and the reason a steady-state sample costs one
16-byte `SAMPLE` record. Three components cooperate, all in
`cpu_snap::sampling::internal`.

### 3.1 `frame_trie` - intern stacks to ids

`src/cpu_snap/sampling/frame_trie.hpp` is a grow-only Flyweight
trie keyed on `(parent_node_id, ip)` edges. Each unique edge maps to one `node_id`
(`u32`). Node id 0 is the implicit root; the first real node is id 1.
`intern(parent_id, ip)` returns `{ id, was_new }`: an existing id on a cache hit, a
freshly allocated id (and `was_new == true`) otherwise.

Why a trie: callstacks are prefix-sharing by nature. `main -> run -> work -> a`
and `main -> run -> work -> b` share three frames. Interning each `(parent, ip)`
edge once means the shared prefix is stored once, and a whole stack is addressable
by its leaf id. 99%+ of samples reuse an existing leaf, so the per-sample wire cost
collapses to a single `u32`.

Implementation notes that carry rationale:

- The edge map is a `std::unordered_map` keyed on a 64-bit mix of `(parent_id, ip)`.
  The header comment records that this is a deliberate "robust now, swap to a flat
  open-addressed map if a bench flags it" choice - the projected hot-path rate is
  within reach of the stdlib hash, so it was not prematurely optimized.
- Nodes are stored in a `std::vector<node_view>` indexed by `id - 1`, so
  `node_at(id)` is O(1) and ids are dense.
- `set_max_nodes(cap)` lets the host bound the trie. On overflow `intern()` throws
  `cpu_snap::resource_error`. v1 has NO raw-IP fallback record - the divergent stack is
  simply lost and counted; a fallback is deferred to v2. The pipeline absorbs the
  throw into `internal_errors()` and `dropped_samples()` and emits a LOST record
  for the sample. The pipeline default cap is 2^22 nodes (0 = uncapped).

### 3.2 `temporal_diff_filter` - emit only the divergent suffix

`src/cpu_snap/sampling/temporal_diff_filter.hpp` sits between
capture and the trie. For each `(tid, frames)` sample it keeps, per tid, the prior
sample's root-ward path. It computes the longest common root-ward prefix against
that path and interns only the divergent suffix. `diff()` returns the `leaf_id`
plus a span of nodes newly allocated on this call.

Why "temporal": at a high sampling frequency, two consecutive samples on the same
thread almost always have the same stack, or differ only in the innermost frame or
two. So the common case is: zero changed frames -> zero trie ops, zero allocations,
zero `NEW_NODE` records, just a `SAMPLE` pointing at the unchanged leaf. This is
the dominant case and the reason the ~100 ns per-sample userspace budget is
reachable. The filter is header-only because `diff()` is small and hot and inlining
buys real cycles.

Frame order is a subtle, load-bearing detail. Frames arrive innermost-first
(`frames[0]` = leaf, `frames[n-1]` = outermost, matching `raw_sample`). The trie is
built root-ward: a node's parent is the frame closer to `main`. So the diff walks
both the new sample and the prior path outermost-to-innermost to find the common
prefix, then interns the changed suffix from the divergence point down to the leaf.
The `new_nodes` span it returns is therefore in root-ward order (parents before
children) - which is exactly the order they must be written (next section). The
span points into instance-owned scratch and is invalidated by the next `diff()`;
callers either consume it immediately (the pipeline does) or copy it out.
`forget_tid()` drops per-tid state when a thread exits so a recycled tid does not
inherit a stale prior path.

### 3.3 The NEW_NODE-before-SAMPLE ordering invariant

This is the contract that makes a forward-replay reader possible, and it is stated
in three places in the code (the filter header, the trie's `intern` doc, and the
pipeline header) because violating it silently breaks decode:

> NEW_NODE records for newly allocated nodes are ALWAYS emitted to the framer
> BEFORE the SAMPLE that references the resulting `leaf_node_id`. Within one
> sample's batch of new nodes the order is root-ward (parents before children).

The pipeline's `on_sample` enforces it directly: it loops over
`result.new_nodes` writing a `new_node_record` for each, THEN writes the
`sample_record` that names `result.leaf_id`.

Why it matters: the reader rebuilds the trie purely by forward replay. Every node
is declared before anything references it, so when the reader sees a `SAMPLE` it can
already resolve the leaf. Parents-before-children within a batch means the reader
can also chain a new leaf up to the root immediately. Without this ordering the
reader would have to buffer and reorder, defeating the streaming model.

The invariant degrades gracefully, which is the other half of the design. If a
`NEW_NODE` lived in a chunk that was dropped (visible as a `chunk_seq` gap) or
rejected, the reader will later see a `SAMPLE` naming a leaf it does not know.
`resolve_stack` returns an empty vector for that leaf rather than failing - one
sample is lost, the stream continues.

---

## 4. The reader's stack resolution

`reader::resolve_stack(leaf_id)` reconstructs a sample's IP chain from the trie the
reader rebuilt from `NEW_NODE` records. The `trie_resolver` walks parent pointers
from `leaf_id` up to the root (id 0), collecting one `ip` per node, and returns them
innermost-first - matching `raw_sample` / `SAMPLE` leaf-first order, so the offline
symbolizer sees frames in the same orientation whether they came from the wire or
an in-process `sample_reconstructor`.

If any node on the path is unknown (the common cause being a `NEW_NODE` in a
dropped or rejected chunk) it returns an empty vector. A typical consumer
subscribes via `on_event`, and for each `sample_record` calls
`resolve_stack(rec.leaf_node_id)` to get the frames, then symbolizes them offline
with the module list it learned from the `maps_init`/`maps_delta` records. The
reader's trie is its own structure (`std::unordered_map<u32, {parent,ip}>`),
independent of the encoder's `frame_trie`; the two never share memory, only the
`NEW_NODE` stream that keeps them in lockstep.

---

## 5. The pipeline facade: `streaming_pipeline`

`src/cpu_snap/sampling/streaming_pipeline.hpp` +
`src/cpu_snap/sampling/streaming_pipeline.cpp` is the single-thread encode facade. It owns
one `frame_trie`, one `temporal_diff_filter`, one `chunk_framer`, one
`dl_module_watcher`, and a `flat_tid_map` of per-tid counters; the host sees only the
`config.sink` callback and the `on_*` methods.

Why a facade with pimpl: the public header must not drag `chunk_framer`,
`dl_module_watcher`, and `flat_tid_map` (heavy includes, 1 MiB buffer) into every TU
that mentions a sampler config. The `impl` lives in the `.cpp`, so the public
header pulls in only `src/core/types.hpp`.

### Lifecycle

1. **Construct** with a non-null `sink`. The ctor validates the sink, clamps the
   poll/checkpoint intervals to at least 1, sets the trie cap, then immediately:
   - calls `chunk_framer::start()` (emits the 64-byte `file_header`), and
   - emits an initial `MAPS_INIT` from `dl_module_watcher.snapshot()`.
   So a consumer that connects gets the header and the initial module list before
   any sample.
2. **Drive** `on_sample` / `on_lost` / `on_tid_exit` from the collector worker,
   one call at a time (never reentrant - the instance is NOT thread-safe by
   design, because the hot path is single-threaded).
3. **`finalize()`** emits a closing `CHECKPOINT`, polls maps one last time, emits
   `EOF`, and flushes the framer's partial chunk. It is idempotent and is called
   from the destructor if not called explicitly, so a stream is always
   well-terminated even on an abnormal teardown path.

### `on_sample`, step by step

```
result = diff(tid, frames)            // common-prefix diff against prior path
for nv in result.new_nodes:           // root-ward: parents before children
    framer.write(new_node_record{nv}) // INVARIANT: before the SAMPLE
framer.write(sample_record{ts, tid, result.leaf_id})
++per_tid_counters[tid].samples
if samples % maps_poll_interval == 0: poll maps, write deltas
if samples % checkpoint_interval == 0: emit CHECKPOINT
```

The maps poll and checkpoint are on a sample-count cadence (default 65536) rather
than a wall-clock timer, so they cost nothing between cadence points and need no
extra thread or signal. The interval is configurable per concern
(`maps_poll_interval`, `checkpoint_interval`).

### `on_lost` and the LOST record

When the kernel reports dropped samples (a `PERF_RECORD_LOST`), the pipeline adds
to `lost_total` and emits a `LOST` record. The streaming pipeline emits LOST with
`tid == 0` - aggregate, not per-tid attributable - because the kernel's loss count
is not broken down per thread at the point the pipeline sees it. LOST is how a
downstream consumer knows the sample stream has holes that are NOT chunk loss: the
samples never reached the encoder at all, so there is no `gap_count` to find. It is
the difference between "the encoder dropped a chunk" (reader's `gap_count`) and
"the kernel dropped samples before encoding" (LOST records).

### CHECKPOINT and resync

`emit_checkpoint` writes a per-tid snapshot: for each tracked tid, its current leaf
id and cumulative samples/lost. Checkpoints exist for one reason: resync after
loss. If a consumer joins late, or a run of chunks is dropped, the per-chunk
forward-replay model has a gap - the reader may have missed `NEW_NODE`s or sample
counts. A CHECKPOINT lets the reader re-anchor: it carries the current leaf id per
tid (so the reader knows the live stack even if it missed the `NEW_NODE`s that
built it, as long as a later `NEW_NODE` re-declares the path) and the cumulative
counters (so loss accounting stays meaningful). A checkpoint is emitted on the
sample-count cadence and once at `finalize()`.

### EOF

`EOF` is always the last record in the last chunk, carrying the session-end
timestamp and the cumulative `SAMPLE` count. It lets a consumer distinguish a clean
end (saw EOF) from a truncated stream (stream stopped mid-chunk), and the total
count is a cross-check against the consumer's own tally.

---

## 6. The noexcept-and-absorb contract

Every `on_*` method and `finalize()` is `noexcept` and wraps its work in a `try`
that, on any exception (`bad_alloc`, trie-cap exhaustion, a sink throw that somehow
escaped the framer's own guard), increments `internal_errors()` and returns. After
`finalize()`, `on_*` calls are no-ops.

Why this is non-negotiable: the methods are called from the sampler's collector
worker, which is fed from the kernel ring (perf) or from a signal-driven snapshot
ring (signal backend, doc 01). An exception escaping into that worker would either
crash the sampled process or corrupt the drain loop. So failure is converted to
data: `internal_errors()` is a counter the host can read after `stop()`, and the
stream stays valid (a dropped sample is just a sample that never produced a
`SAMPLE` record). This pairs with the reader's symmetric "never throw, count
instead" policy: both ends treat the stream as best-effort and observable, never as
a transaction that aborts.

The stat accessors (`samples_emitted`, `new_nodes_emitted`, `lost_total`,
`maps_polls_done`, `checkpoints_emitted`, `internal_errors`) are the producer-side
telemetry that mirrors the reader's `reader_counters`. Together they let a test or
a host assert end-to-end: samples emitted should match samples decoded, lost should
reconcile, internal_errors should be zero.

---

## 7. End-to-end summary

```
on_sample(tid, ts, frames)
  -> temporal_diff_filter.diff           // common-prefix vs prior path on tid
  -> frame_trie.intern (suffix only)     // (parent,ip) -> u32, was_new flag
  -> chunk_framer.write(NEW_NODE...)     // parents before children, BEFORE sample
  -> chunk_framer.write(SAMPLE leaf_id)  // 16 bytes in steady state
  -> [cadence] maps poll, CHECKPOINT
        |
        v  sink_fn(bytes)  (host transport: file/pipe/socket/ring)
        |
   reader.feed(bytes)
  -> chunk_assembler  (state machine; implausible chunk headers resynced)
  -> record_walker    (decode + dispatch to event_fn)
  -> trie_resolver    (NEW_NODE -> {parent,ip})
  -> event_fn(SAMPLE) ; resolve_stack(leaf_id) -> innermost-first IPs
        |
        v  offline symbolize + shape (doc 02)
```

The whole design is one idea applied consistently: treat the stream as a lossy,
self-describing, independently-verifiable byte container; never let producer or
consumer throw; make the common case (an unchanged stack) cost almost nothing; and
make every failure observable as a counter rather than an abort. The trie-diff is
the size win, the NEW_NODE-before-SAMPLE invariant is what lets the consumer rebuild
state by forward replay, and per-chunk sequence numbers + checkpoints + LOST + EOF
are what let it recover, resync, and reconcile when the transport drops bytes.

An architectural audit of the project recommended keeping
all the load-bearing components with no drops - the redundancy that exists
(`dl_module_watcher` reimplementing `wall_unix_ns_now`, `sampler_capabilities` tightly
coupled to `sampler`, `triggers::signal::snapshot_ring` to `triggers::signal::backend`) is minor and noted as
merge candidates rather than defects. The streaming and wire layer documented here
is the part of that architecture that carries the format contract, so changes to it
ripple to both the encoder and every consumer; treat the wire format and the
ordering invariant as frozen for v1.
