# Design 01 - The src/core/ subsystem

This is the deep, top-down design reference for `src/core/`: the backend-agnostic
data plane that every other part of cpu-snap is built on. It documents EVERY
core class - what it is, its key API, and (the point of this document) WHY it
exists and why it is shaped the way it is. The audience is a developer who must
understand and maintain the whole project, so the rationale is given even where
it is "obvious in hindsight".

This document does not re-derive the byte tables or the per-component capsule
summaries that already live elsewhere. Read those first or alongside:

- [./00-overview.md](./00-overview.md) - where core sits in the whole system.
- [./02-capture-and-backends.md](./02-capture-and-backends.md) - the perf/signal
  backends, the hot path, and the capture seam built on top of core.
- [./03-streaming-and-wire.md](./03-streaming-and-wire.md) - the live encode
  path (`streaming_pipeline`, `temporal_diff_filter`, `frame_trie`) that drives
  `chunk_framer`.
- [./04-offline-processing.md](./04-offline-processing.md) - the offline render
  path (`symbolizer`, `sample_postprocessor`, `slice_emitter`).
- [./06-design-decisions.md](./06-design-decisions.md) - the cross-cutting
  decision log (variant vs virtual interface, de-thunking, layering).
- [../components.md](../components.md) - per-component "what is each piece"
  capsule reference (purpose / API / consumers). This doc is the WHY companion
  to those capsules.
- [../wire-format.md](../wire-format.md) - the authoritative byte-level wire
  spec (offsets, sizes, framing tables). This doc explains the structure; that
  doc nails the bytes.
- [../architecture.md](../architecture.md) - the layered prose orientation.
- [../../planning/architecture-sampler-abstraction.md](../../planning/architecture-sampler-abstraction.md)
  - the planning record for the capture-backend abstraction, with the locked
  decisions referenced below.

Everything in `src/core/` lives under `namespace cpu_snap::core`, with
cross-cutting helpers under `cpu_snap::core::utils`. Source is under
[../../src/cpu_snap/core](../../src/cpu_snap/core) (header)
and [../../src/cpu_snap/core](../../src/cpu_snap/core) (the two out-of-line bodies, `reader.cpp`
and `dl_module_watcher.cpp`).

---

## 1. Why src/core/ exists and why it is backend-agnostic

cpu-snap is published as one CMake package with two independently selectable
components, each its own source tree and namespace: `cpu_snap::core`,
`cpu_snap::sampling`. The dependency arrows point downward
only:

```
src/sampling/               <- Layer 2: capture + processing
     |
     |
     v
   src/core/                  <- Layer 1: wire format + IO substrate
         |
   Linux: perf_event_open, mmap rings, /proc, dl_iterate_phdr
```

The central design rule is: **the on-wire byte format is the only contract
shared across the recorder and the reader.** `src/core/` is exactly the set of pieces
that the sampling sampler needs but does not own:
the record definitions, the chunk framer/reader, the perf-ring RAII
owner, the `/proc` and `dl_iterate_phdr` observers, and a couple of leaf
utilities. Because the recorder emits a self-describing chunk stream, a single
`core::reader` decodes it independently of where the bytes were produced. That
is only possible if the wire machinery is factored out of the recorder into a
shared, recorder-neutral layer.

Two consequences of "backend-agnostic" are worth stating explicitly because
they constrain what is allowed to live here:

- **No `libdw`.** DWARF unwinding (elfutils `libdwfl`) is a `sampling`-only
  dependency. `core` links only `rt` and `Threads` (see
  [../../src/core/CMakeLists.txt](../../src/core/CMakeLists.txt)). A consumer that only
  wants to decode a stream never pays for libdw.
- **No knowledge of what a "sample" means.** `core` knows about NEW_NODE,
  SAMPLE, etc. as opaque record bodies with a wire layout. It does
  not know how a stack is captured or how a frame is
  symbolized. Those are Layer-2 concerns. This is what lets the same code serve
  a decoupled producer.

The rest of this document walks core top-down: the wire vocabulary (`records`,
`types`), the producer/consumer pair built on it (`chunk_framer`, `reader`), the OS
substrate (`perf_ring`, `proc_maps`, `dl_module_watcher`), and the leaf utilities
(`thread_comm`, `time`, `flat_tid_map`).

---

## 2. The wire vocabulary

### 2.1 records - the single source of truth

File: [../../src/cpu_snap/core/records.hpp](../../src/cpu_snap/core/records.hpp)

`records.hpp` is the root of the entire wire dependency tree and the most widely
depended-on header in the project. It defines, in one place:

- `enum class record_type : uint8_t` - the per-record type tag. Codes:
  `invalid=0x00`, `new_node=0x01`, `sample=0x02`, `checkpoint=0x03`,
  `maps_init=0x04`, `maps_delta=0x05`, `lost=0x06`, `eof=0x07`.
  `0x08..0xFF` are reserved for future record kinds.
- The fixed-size POD record bodies (16 bytes each, natural alignment):
  `new_node_record`, `sample_record`, `lost_record`, `eof_record`.
- The variable-size record bodies: `checkpoint_record` (+
  `per_tid_checkpoint_entry`), `maps_init_record` and `maps_delta_record` (+
  the shared `module_info`).
- `record_variant` - the sum type over all seven body structs.
- `type_of(const record_variant&)` - maps the active alternative back to its
  `record_type` tag.
- The pragma-packed framing structs `file_header` (64 B), `chunk_header` (16 B).

**Why a single source of truth.** Producer (`chunk_framer`) and consumer
(`reader`) must agree byte-for-byte. If each defined its own structs the two
could drift silently. Putting every wire type in one header means the framer's
encoder and the reader's decoder are written against the same declarations, and
a field reorder is caught by the `static_assert(sizeof(...) == N)` guards on
every fixed body and on the framing structs. The wire-format spec
([../wire-format.md](../wire-format.md)) is the prose mirror of this file; this
header is the code-of-record.

**Why `record_variant` (a `std::variant`).** The framer's `write()` takes one
parameter of type `record_variant` and dispatches with `std::visit`. A variant
is the right tool here because the record set is genuinely a *closed* sum type
owned by `records.hpp` itself: every alternative is declared in the same header,
so there is no extensibility tension (contrast the capture-backend case in
[./06-design-decisions.md](./06-design-decisions.md), where the set is open and
variant is rejected). The variant gives the framer a single typed entry point,
exhaustive `std::visit` dispatch (a new record kind is a compile error until all
the `encode_body` / `body_size_of` overloads exist), and lets `reader`'s
`event_fn` hand the host one uniform callback argument. `type_of()` is a
`constexpr` lookup indexed by `variant.index()` against a fixed seven-element
tag array, so the wire tag and the variant order are kept in lockstep in one
spot.

**Why bodies are NOT packed but framing structs ARE.** This is a deliberate
split:

- The framing structs (`file_header`, `chunk_header`) are under
  `#pragma pack(push, 1)`, so the struct layout *is* the wire layout and they
  can be emitted/parsed with a single `memcpy`. They are pure transport headers
  with no host-side semantics, so binding them to the wire byte layout is fine
  and cheap.
- The record bodies are normal naturally-aligned C++ structs. The framer
  serializes them *field by field* (see `chunk_framer::encode_body`), so the
  on-disk byte order is independent of struct padding. This means a host can add
  a field, reorder for ergonomics, or change alignment of the in-memory struct
  without changing the wire format, as long as the field-by-field encoder is
  updated. The `static_assert` on the fixed bodies' `sizeof` is not a wire
  guarantee - it is a tripwire that catches an accidental field reorder that
  would silently change the in-memory layout that downstream tests assume.

**Why the variable records carry their own length on the wire.** `module_info`
encodes `build_id_len` + `path_len` and the decoder requires the cursor to land
exactly on the body end; `checkpoint_record` carries an
explicit count and the reader rejects the record unless
`body_size == fixed_prefix + count * element_size`. This makes every record
self-delimiting so the reader can skip or reject a malformed one without losing
the rest of the chunk.

### 2.2 types - the deliberately thin ABI header

File: [../../src/cpu_snap/core/types.hpp](../../src/cpu_snap/core/types.hpp)

`types.hpp` defines exactly three things:

- `using sink_fn = std::function<void(std::span<const std::byte>)>;` - the
  host-supplied byte sink (output side). The producer hands completed spans to
  it.
- `using event_fn = std::function<void(const record_variant&)>;` - the
  host-supplied record callback (reader side). The reader fires it per decoded
  record.
- `inline constexpr std::uint32_t k_default_chunk_bytes = 1U << 20;` - the
  1 MiB default chunk size.

**Why this header is so thin, and why it is separate from `chunk_framer`.** The
public headers of both libraries (`sampler.hpp`, `streaming_pipeline.hpp`)
need to name `sink_fn` in their config structs and need
`k_default_chunk_bytes` as a default. If those aliases lived in
`chunk_framer.hpp`, every public header would transitively drag in the framer's
heavy include set (the 1 MiB buffer machinery, `<bit>`, the full
record encoder). By isolating just the function aliases and the one constant in
a leaf header that includes only `records.hpp`, the public ABI is decoupled from
the framer implementation. This is the same "leaf header to break a heavy
include" pattern used by `frame_limits.hpp` on the sampling side (see
[./02-capture-and-backends.md](./02-capture-and-backends.md)).

**Why `std::function` here when the hot path bans it.** The de-thunking rule
(no `std::function` on the per-sample path - see
[./06-design-decisions.md](./06-design-decisions.md)) applies to the per-sample
dispatch inside the capture backends. `sink_fn` and `event_fn` are *coarse*
boundaries: `sink_fn` is invoked once per completed chunk (about once per 1 MiB
of output), and `event_fn` once per decoded record on the consumer side, which
is offline. Type-erasing them via `std::function` is correct here precisely
because it does not bind the public ABI to a concrete callable type and the call
frequency is low enough that the indirection is free. The contract is documented
on the alias: the callback is invoked serialized from the collector worker
thread, must not block, and the span/reference is valid only inside the call.

### 2.3 Why the wire format is chunk framed

This is the central format decision and it is shared by `records`, `chunk_framer`,
and `reader`, so it is stated once here.

The stream is a 64-byte `file_header` followed by a sequence of independent
chunks, each `chunk_header | records body`. The reasons:

- **Streaming, not a finished file.** cpu-snap never opens a file descriptor;
  the host owns persistence via `sink_fn`. The producer must be able to hand off
  bounded, self-contained units as it goes, without buffering the whole session
  or seeking back to patch a length. A fixed-target chunk (1 MiB) is that unit.
- **Bounded buffering + bounded loss.** The framer holds exactly one inline
  1 MiB buffer and emits when it fills. The reader buffers at most one partial
  chunk. A dropped chunk costs at most one chunk's worth of records,
  not the stream.
- **Self-healing decode.** Each chunk carries a monotonic `chunk_seq`. Gaps in
  `chunk_seq` are counted (`gap_count`), and a chunk header that fails the
  plausibility checks (magic, version, size, sequence jump) is rejected and the
  reader forward-scans for the next plausible header (`resyncs`,
  `bytes_skipped`). One bad chunk can never abort a streaming consumer. There is
  no payload checksum; integrity relies on the transport.
- **Records never span a chunk.** Each chunk is independently decodable; you do
  not need chunk N-1 to parse the first record of chunk N. This is what makes
  a chunk a meaningful recovery boundary.

The detailed byte offsets are in [../wire-format.md](../wire-format.md); the
"why" above is the design intent behind those tables.

---

## 3. The producer/consumer pair

### 3.1 chunk_framer - the producer

File: [../../src/cpu_snap/core/chunk_framer.hpp](../../src/cpu_snap/core/chunk_framer.hpp)

The producer side of the wire format. Header-only, C++20, little-endian host
only (enforced by a `static_assert(std::endian::native == std::endian::little)`
- there is no byte-swap path by design). Construct with a `config` (chunk size,
flags, pid, start time, process name) and a `sink_fn`; call `start()` once to
emit the 64-byte `file_header` and open the first chunk; call `write(record)`
per record; `flush()` to force-emit a partial chunk. Sink exceptions are
counted in `sink_errors()`.

Key invariants (all enforced in code):

- **Records never span a chunk.** `write()` computes the padded record size; if
  it would push the chunk past the soft `chunk_bytes` threshold it seals and
  emits the current chunk and opens a fresh one before writing. A single record
  larger than the chunk capacity throws `cpu_snap::resource_error`.
- **4-byte inter-record alignment**, with `records_byte_count` in the chunk
  header equal to the *unpadded* body length (excludes the last record's
  trailing pad).
- **Header reserved up front.** The header's 16 bytes are reserved at offset 0
  of the buffer and filled in by `seal_and_emit_chunk()` when the chunk is
  sealed, so record offsets are body-relative.
- **u16 body-size limit** (`k_max_record_body_bytes = 0xFFFF`): a record body
  exceeding it throws `cpu_snap::resource_error`.

**Why a single inline `alignas(64)` 1 MiB buffer.** The framer holds its whole
chunk as an inline `std::array<std::byte, k_default_chunk_bytes>` aligned to a
cache line, so a `chunk_framer` instance is roughly 1 MiB. There is exactly one
framer per `streaming_pipeline` (i.e. one per sampler), so at the design scale
of one sampler per process the footprint is negligible, and an inline buffer
avoids a per-chunk heap round-trip on the hot output path. The cost is that you
must NOT stack-allocate a `chunk_framer` (heap or data-segment only) - the class
documents this and is move/copy deleted to keep its identity pinned. The
configured `chunk_bytes` is a *soft flush threshold* in
`[k_min_chunk_bytes, k_default_chunk_bytes]`; the buffer is always sized at the
1 MiB max.

**Why sink exceptions are caught and counted, never propagated.** The framer is
driven from the collector worker thread. If a host sink throws, propagating it
would tear down the worker and lose the rest of the session. Instead
`emit_to_sink()` swallows the exception and bumps `sink_errors()`, keeping the
collector alive. This mirrors the "absorb, count, keep going" philosophy on the
reader side.

**Why field-by-field encoding.** `write_record()` writes a 4-byte
`record_header_wire` then `std::visit`s the variant to the matching
`encode_body` overload, which writes each field little-endian through a
cursor-advancing `put<T>()`. As discussed under `records`, this decouples the
wire layout from in-memory struct padding. `encoded_body_size()` mirrors the
same overload set (`body_size_of`) so the framer knows the exact body size
*before* it writes, which is what lets it decide up front whether the record
fits in the current chunk.

### 3.2 reader - the consumer

Files: [../../src/cpu_snap/core/reader.hpp](../../src/cpu_snap/core/reader.hpp),
[../../src/cpu_snap/core/reader.cpp](../../src/cpu_snap/core/reader.cpp)

The symmetric L6 counterpart of `chunk_framer`, and the largest core component.
`reader` is a *push parser*: the host feeds arbitrary byte slices via
`feed(span)` (the sole mutating entry point, declared `noexcept`), the reader
buffers partial headers/bodies across calls, decodes each record, and fires the installed `event_fn`. It also exposes
`resolve_stack(leaf_id)`, `header()`, `counters()`, and `is_fatal()`. The public
class is a pimpl (`std::unique_ptr<impl>`), move/copy deleted.

**Why a push parser, not a pull/file parser.** The format is a stream the host
delivers in whatever granularity its transport produces (one chunk per
`sink_fn` call on the producer, but arbitrary TCP/file slices on the consumer).
A push parser that buffers exactly the next expected byte count and never
assumes record/chunk-aligned input is the natural dual of the streaming
producer. `feed()` must be `noexcept` because it sits at an IO boundary where the
host should never have to wrap it in a try/catch; instead every failure mode is
folded into `reader_counters`.

**Why the reader rebuilds its OWN trie view.** The encode side compresses stacks
into a trie and writes only NEW_NODE declarations + a leaf id per SAMPLE (see
[./03-streaming-and-wire.md](./03-streaming-and-wire.md)). The reader is a
forward replayer: it cannot ask the producer for the trie, so it reconstructs a
`node_id -> {parent_id, ip}` map (`trie_resolver`) from every NEW_NODE it
decodes, and `resolve_stack(leaf_id)` walks parent pointers from the leaf up to
the root, returning IPs innermost-first (matching `raw_sample` / SAMPLE order).
The ordering contract that NEW_NODE is always emitted before the SAMPLE that
references it is what makes this single-pass forward resolution possible. If a
leaf id is unknown - the common cause being a NEW_NODE that lived in a dropped or
rejected chunk - `resolve_stack` returns an empty vector rather than
failing, so one lost chunk degrades a few stacks instead of aborting the
consumer.

**Why the implementation is three collaborators.** `reader.cpp` splits the work
into an anonymous-namespace trio, each with one job:

- `chunk_assembler` - the framing state machine
  (`expect_file_header -> expect_chunk_header -> expect_chunk_body ->
  back`). It accumulates exactly the next expected byte count into a pending
  buffer (header) or the body buffer, and on a complete body hands it to the
  walker.
- `record_walker` - walks one verified body, reads each 4-byte record header,
  bounds-checks `body_size`, decodes the body, dispatches to the host `event_fn`
  (and feeds NEW_NODE into the trie), and advances by `align4(4 + body_size)`.
- `trie_resolver` - the `node_id -> {parent_id, ip}` map and the leaf-to-frames
  walk.

This separation keeps the byte-level state machine, the per-record decode/
dispatch, and the trie reconstruction independently testable and independently
reasoned about; the public `reader::impl` just wires them together with two
callbacks.

**Why absorb-and-count instead of throw.** The reader's contract is that a
single bad chunk does not abort streaming consumers. So:

- **Fatal only on the first 64 bytes**: bad `file_header` magic (`'CSNP'`) or an
  unsupported `version_major` sets an internal fatal flag, bumps `parse_errors`,
  and silences all further emissions (`is_fatal()`). A wrong-format stream is
  not recoverable, so it stops cleanly.
- **Implausible chunk header** (bad `'CH'` magic/version, oversize
  `records_byte_count`, or an implausible `chunk_seq` jump): bump `parse_errors`
  (bad magic/version) or `chunks_rejected` (size/sequence), then forward-scan
  the pending bytes for the next plausible `'CH'` (`resyncs`, `bytes_skipped`).
- **Per-record errors** (size mismatch, truncated body, unknown/`invalid` tag):
  bump `parse_errors` and abandon the rest of that chunk body.
- **Host callback exceptions**: caught and swallowed; they are explicitly NOT
  stream errors.

All of these surface through `reader_counters` (`bytes_fed`, `records_emitted`,
`parse_errors`, `gap_count`, `last_seen_seq`, `unknown_records`,
`chunks_rejected`, `resyncs`, `bytes_skipped`), so a host can monitor stream health without ever catching an
exception.

---

## 4. The OS substrate

### 4.1 perf_ring - the RAII perf-ring owner

File: [../../src/cpu_snap/sampling/perf_ring.hpp](../../src/cpu_snap/sampling/perf_ring.hpp)

A header-only, move-only RAII wrapper over the resources of one
`perf_event_open` mmap ring (the fd, the mmap base + bytes, and an optional
epoll registration fd) plus a single templated `drain(Decoder&&)` loop. The loop
reads `data_head` with acquire semantics, walks every record from the saved
`data_tail` to `head`, splices any record that wraps the ring end into an
internal scratch buffer so the decoder always sees a contiguous
`perf_event_header + body`, invokes the decoder once per record, and finally
publishes the new `data_tail` with release semantics.

**Why this lives in core, not in sampling.** Ring management is
identical regardless of what the *decoder* interprets. The
sampling side (`perf::kernel_drain`) walks `PERF_RECORD_SAMPLE` callchains.
Factoring the ring lifetime + drain loop into `sampling::perf_ring` and templating
the decoder de-duplicates the tricky part (the lock-free producer/consumer
protocol with the kernel) while letting the backend keep its own concrete
decoder. This is why a perf-specific type is in the "backend-agnostic" layer:
the *ring* is not backend-specific, only the record interpretation is.

**Why it owns lifetime but does NOT arm the fd.** The dtor reverses the exact
setup sequence in the correct order: `EPOLL_CTL_DEL`, then `munmap`, then
`close`. But it deliberately does NOT call `PERF_EVENT_IOC_ENABLE/DISABLE` -
arming the perf fd is the caller's job. `perf_ring` is a *memory + file-
descriptor lifetime owner*, not a controller. Keeping the policy (when to arm,
pause, resume) in the backend and only the resource ownership in core keeps the
shared type free of any backend-specific control flow. Move semantics null out
the moved-from source so the dtor is idempotent.

### 4.2 proc_maps - the offline maps snapshot

File: [../../src/cpu_snap/processing/symbols/proc_maps.hpp](../../src/cpu_snap/processing/symbols/proc_maps.hpp)

A header-only snapshot of the current process's executable VM mappings, parsed
from `/proc/self/maps`. `proc_maps` reads the executable regions (skipping
non-`x` and `[bracketed]` pseudo-mappings), records each as a `vm_mapping`
(`start`, `end`, `file_offset`, `is_main_exe`, `path`), and offers
`find(ip) -> const vm_mapping*` (linear scan) plus the static
`ip_to_file_offset(ip, mapping)`. `core::self_exe_path()` (`core/self_exe.hpp`) reads
`/proc/self/exe`.

**Why it exists and why it is the OFFLINE counterpart of `dl_module_watcher`.** To
turn a runtime IP into something `addr2line` understands you need the owning
binary plus the file offset of that IP within the binary. `find()` locates the
owning mapping; `ip_to_file_offset()` does the rebase: for the non-PIE main exe
the runtime IP equals the file offset, but for PIE binaries and shared libraries
the offset is `ip - mapping.start + mapping.file_offset`. This is the foundation
the whole symbolize/postprocess/filter side stands on (`symbolizer`,
`frame_filters`, `sample_postprocessor`, `slice_emitter` all consume it - see
[./04-offline-processing.md](./04-offline-processing.md)). It is "offline" because
it takes one snapshot of the address space, suitable for resolving after capture
in the same process; `dl_module_watcher` is the streaming-time path that tracks
changes as they happen.

### 4.3 dl_module_watcher - the streaming-time module observer

Files: [../../src/cpu_snap/sampling/dl_module_watcher.hpp](../../src/cpu_snap/sampling/dl_module_watcher.hpp),
[../../src/cpu_snap/sampling/dl_module_watcher.cpp](../../src/cpu_snap/sampling/dl_module_watcher.cpp)

The streaming-time loaded-module observer. `snapshot()` walks `dl_iterate_phdr`
(plus the main executable and the vDSO), reads each ELF's `.note.gnu.build-id`,
and returns a fully populated `maps_init_record`. `poll()` re-walks and diffs
against the last-seen set, returning add/remove `maps_delta_record`s (removes
first, then adds). `known_module_count()` reports the tracked set size; a
`set_iterator_for_tests()` seam swaps the real `dl_iterate_phdr` walker for an
injected one.

**Why a streaming observer separate from `proc_maps`.** Symbolization must pin
every IP timestamp to the binary that was mapped *at that time*, even when
`dlopen`/`dlclose` races the recording. A one-shot snapshot (which is what
`proc_maps` is) cannot capture a library that was loaded mid-session and unloaded
before capture ended. So the streaming pipeline drives `dl_module_watcher`: it emits
`maps_init` once at session start (right after the file header) and `maps_delta`
records as modules come and go, recording *when* each change was observed. An
offline consumer can then reconstruct the address-space timeline and resolve any
IP against the modules live at its timestamp. `proc_maps` stays for the simpler
offline symbolizer in-process case; `dl_module_watcher` is the wire-recorded,
race-tolerant path.

**Why core (not sampling), and why the v1 simplifications.** It produces
`maps_init_record` / `maps_delta_record`, which are core wire records, and it is
recorder-neutral module metadata, so it belongs in the shared layer. It is
documented as not thread-safe and is driven exclusively from the single
streaming worker thread, so it needs no locking. The identity key is
`(base_va, path)`; a same-key change to size/file_offset/build_id emits no delta
(a v1 simplification - the practical case that matters is load/unload), and a
same-path-but-different-base module is reported as a remove of the old plus an
add of the new. These narrow the scope to what symbolization actually needs
without a full mapping-diff engine.

---

## 5. The leaf utilities

These three are cross-cutting helpers, not pipeline stages. Two live in
`cpu_snap::core::utils`; `flat_tid_map` is a `cpu_snap::core` container.

### 5.1 thread_comm - thread/process identity

File: [../../src/cpu_snap/sampling/thread_comm.hpp](../../src/cpu_snap/sampling/thread_comm.hpp)

Header-only `/proc/self`-backed identity helpers in `cpu_snap::core::utils`:

- `read_thread_comm(pid_t tid)` - the kernel name of a thread from
  `/proc/self/task/<tid>/comm`, with trailing newline trimmed; falls back to
  `"tid-<N>"` when the file is gone (the thread exited) or empty.
- `read_proc_self_comm()` - the process name from `/proc/self/comm`.
- `list_self_thread_tids()` - every live tid under `/proc/self/task` (non-numeric
  entries skipped).

**Why it exists.** The recorder needs thread/process identity: a default
process name for the `file_header`, human-readable track names for output, and
the enumeration of which threads to attach to. Centralizing the `/proc` parsing
(with the exited-thread fallback, which matters because tids can vanish between
enumeration and naming) keeps that brittle string-handling in one tested place
that `sampler.cpp` and several examples share.

### 5.2 time - one wall-clock helper

File: [../../src/cpu_snap/sampling/time.hpp](../../src/cpu_snap/sampling/time.hpp)

The smallest component: a single inline `wall_unix_ns_now()` returning
`std::chrono::system_clock` nanoseconds since the Unix epoch.

**Why it exists as its own header.** Several producers stamp wall-clock times
(`start_unix_ns` in the file header, `wall_ns` on CHECKPOINT/LOST/EOF, the maps
timestamps). One canonical helper means the units (ns since epoch, `system_clock`
not `steady_clock`) are consistent everywhere a timestamp is written, which is
the contract the wire format documents. (Note: `dl_module_watcher.cpp` reimplements
the same logic privately - a minor known duplication.)

### 5.3 flat_tid_map - the hot-path per-tid table

File: [../../src/cpu_snap/sampling/flat_tid_map.hpp](../../src/cpu_snap/sampling/flat_tid_map.hpp)

A header-only `template <typename T>` per-tid associative container. It is a
256-slot direct-mapped table keyed on `(tid mod 256)` with an 8-step linear
probe, backed by a `std::unordered_map<pid_t, T>` overflow map for the rare case
where the slot cluster fills. It offers the usual `operator[]` (get-or-insert),
`find`, `erase`, `size`, `empty`, `overflow_size`, and `for_each`. Not
thread-safe by design (single-worker hot path).

**Why this exists instead of just using `std::unordered_map`.** This is a hot-
path optimization, and the rationale is specific to the recorder's access
pattern. The streaming encode path looks up per-tid state (the running stack in
`temporal_diff_filter`, per-tid counters in `streaming_pipeline`) on *every*
sample. `std::unordered_map` pays a bucket -> node -> {key,value} pointer chase
and a hash per lookup. The observation that justifies the flat table: most
workloads have far fewer than 256 concurrent threads, and Linux assigns tids
roughly sequentially from `/proc/self/task`, so tids cluster in a small range
and `(tid mod 256)` lands adjacent threads in adjacent slots; the short linear
probe absorbs the collision runs that pattern produces, and the value lives
inline in the slot (no pointer chase, cache-friendly). The overflow map is the
correctness safety net for the pathological case (more than ~256 tids or a bad
probe cluster) so the optimization never becomes a bug. Putting it in `core`
lets both the sampling streaming path and any other per-tid hot path reuse it.

---

## 6. How the core pieces compose

Putting it together, a streaming capture session uses core like this (the
Layer-2 producer is covered in its own design doc):

```
producer (sampling, Layer 2)
  | builds record_variant values (records.hpp)
  v
chunk_framer.write(record)              -- core
  | encode_body field-by-field, align4
  | seal at ~1 MiB: chunk_header + body
  v
sink_fn (types.hpp)  -->  host transport (file / socket / RAM)
                                |
                                v
                         reader.feed(span)       -- core
                           | chunk_assembler: reassemble chunks
                           | record_walker: decode + event_fn (types.hpp)
                           | trie_resolver: NEW_NODE -> resolve_stack(leaf)
                           v
                         host event_fn(record_variant)
```

Alongside that, the OS substrate feeds the producer: `perf_ring` owns the
kernel mmap rings the perf backend drains; `dl_module_watcher` emits the
`maps_init`/`maps_delta` module records into the same framer; `thread_comm`
supplies thread/process identity; `time` supplies the wall-clock stamps;
`flat_tid_map` holds the per-tid hot-path state; and `proc_maps` serves the
offline symbolize side. Every one of these is recorder-neutral, which is the
property that lets the wire format stay decoupled from the recorder that
produced it.

---

## 7. Audit note

A component-level audit of cpu-snap concluded keep-everything for core: every
piece here is on the live capture-to-output path or its offline counterpart, the
layering (downward-only dependencies, no `libdw` in core) holds, and the only
redundancy noted is minor (`dl_module_watcher.cpp` privately re-deriving
`wall_unix_ns_now`). The shapes documented above - variant-as-single-source-of-
truth, the thin `types` ABI header, chunk framing, the de-duplicated
`perf_ring`, and the hot-path `flat_tid_map` - are the justified core of the
data plane. The cross-cutting decisions (virtual interface vs variant for the capture
backend, hot-path de-thunking) are recorded in
[./06-design-decisions.md](./06-design-decisions.md).
