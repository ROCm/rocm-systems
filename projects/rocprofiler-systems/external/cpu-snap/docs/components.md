# cpu-snap Components Reference

What each component is for, how it works, what it depends on, and a per-file reference (location, size, purpose, implementation, public API).
Paths are relative to `src/cpu_snap/`; folders follow namespaces (`src/cpu_snap/<folder>` is `cpu_snap::<folder>`), include root is `src/`.
For the data flow and rationale see [architecture.md](architecture.md) and [design/](design/README.md); the wire format is in [wire-format.md](wire-format.md).

## Overview

The library is three CMake components plus the launcher tool:

| Component | Target | Depends on | Role |
|---|---|---|---|
| `core` | `cpu_snap::core` | `error.hpp`, Threads | Wire format, framing, reader, sample reconstruction, module map, error hierarchy. |
| `sampling` | `cpu_snap::sampling` | `core`, libdw, rt | Per-thread sampler: perf or signal backend, unwinders, streaming pipeline. |
| `processing` | `cpu_snap::processing` | `core` | Offline symbolization, frame filtering, slice shaping. |
| `tools` / `examples` | executables | all three | `cpu-snap-run` launcher, autoload library, demos. |

Dependency direction (verified from `#include`s): `sampling -> core`, `processing -> core`; `core` and `processing` never include `sampling`, and `sampling` never includes `processing`.

Typical flow: `sampling` captures and streams wire chunks -> `core` reader and `sample_reconstructor` turn them into `raw_sample` -> `processing` symbolizes, filters and shapes them into slices -> an output sink (in examples) writes Perfetto.

## core

### What it is for
`core` (`src/cpu_snap/core`, namespace `cpu_snap::core`, plus `src/cpu_snap/error.hpp`) is the capture-agnostic wire layer of cpu-snap. It defines the v1 (major 2) binary stream that sampling produces and consumers read, and offers the tools to write, buffer, decode and reconstruct that stream into per-thread samples. The library-wide exception hierarchy lives beside it so every layer shares one error vocabulary.

### Responsibilities
- Define all on-wire records, file/chunk headers and stream-clock flags (`records.hpp`).
- Serialize records into aligned, size-bounded chunks and hand them to a sink (`chunk_framer`).
- Decode a byte stream pushed at any granularity, tolerating truncation, gaps and corruption (`reader`).
- Rebuild the call-node trie so a leaf id resolves to an innermost-first frame chain (`reader::resolve_stack`).
- Keep a time-aware module map from MAPS_INIT/MAPS_DELTA records (`module_map`).
- Turn a stream into `std::unordered_map<pid_t, std::vector<raw_sample>>` plus a `module_map` (`sample_reconstructor`).
- Provide an in-memory bounded sink (`memory_sink`), shared constants (`k_default_chunk_bytes`, `k_default_callchain_depth`, `k_max_callchain_depth`) and small helpers (`self_exe_path`, `format_hex`).
- Define the `cpu_snap::error` hierarchy.

### Explicitly not responsible for
- Any capture: no perf, signals, threads or unwinding (that is `sampling`).
- Symbolization, frame filtering or trace output (that is `processing`).
- Deciding where bytes go; sinks are supplied by callers.

### Main building blocks
| Unit | Role |
|---|---|
| `records.hpp` | Record structs, `record_variant`, `file_header` (64 B), `chunk_header` (16 B), `stream_clock` helpers |
| `chunk_framer.hpp` | Header-only writer: file header, record framing, chunk sealing, `write_split` for oversized checkpoint/maps_init |
| `reader.hpp/.cpp` | Pimpl push decoder: `chunk_assembler`, `record_walker`, `trie_resolver`, `reader_counters`, resync |
| `module_map.hpp/.cpp` | Sorted module intervals with `[load, unload)` windows; `resolve(ip, ts)` |
| `sample_reconstructor.hpp/.cpp` | Wraps `reader`; builds `raw_sample` vectors per tid and a `module_map` |
| `raw_sample.hpp` | Plain decoded sample (`tid`, `timestamp_ns`, `ip`, `frames`) |
| `memory_sink.hpp` | Capped in-memory `sink_fn`-compatible functor with drop counters |
| `types.hpp`, `frame_limits.hpp` | `sink_fn`, `event_fn`, chunk-size and callchain-depth constants |
| `self_exe.hpp`, `format_hex.hpp` | Tiny helpers (`/proc/self/exe` readlink; hex formatting) |
| `error.hpp` | `error`, `config_error`, `usage_error`, `environment_error`, `resource_error`, `detail::coded_error` |

### How it works
1. Write path: `chunk_framer(config, sink)` validates the sink and `chunk_bytes` (throws `config_error`); `start()` emits the file header.
2. `write(record)` encodes the body into an inline 1 MiB aligned buffer, 4-byte aligned; when a record would overflow `chunk_bytes` the chunk (`chunk_header` + records) is sealed and passed to the sink; `flush()` emits the partial chunk.
3. Read path: `reader::feed()` appends bytes; `chunk_assembler` moves through file header, chunk header and chunk body states, checks `chunk_seq` (counting `gap_count`), and on a rejected header scans forward to resync.
4. `record_walker` decodes each record with bounds checks and fires the `event_fn`; unknown types are skipped by size, a malformed known record abandons the rest of the chunk.
5. `trie_resolver` consumes `new_node_record` (parent before child) so `resolve_stack(leaf_id)` returns the frames.
6. `sample_reconstructor::handle_event`: for `sample_record` it resolves the leaf (skips id 0, counts `unresolved_samples`) and appends a `raw_sample` under its tid; `maps_init`/`maps_delta` go to `module_map::apply`.
7. `module_map::resolve` binary-searches by base, scans back only within the max module size, and checks size and time window.

### Dependencies and users
- Depends on: the standard library, Threads (CMake), and `cpu_snap/error.hpp`.
- `sampling` (`streaming_pipeline`, samplers) uses `chunk_framer`, `memory_sink`, records and constants; it also throws the `error` types.
- `processing/symbols` (`sample_postprocessor`, `symbolizer`, `frame_filters`) uses `raw_sample` and `module_map`.
- Tests (`tests/core/test_chunk_framer.cpp`, `tests/support/stream_capture.hpp`), `bench/bench_overhead.cpp` and `examples/*` use the framer, reader and reconstructor.

### Invariants and constraints
- Little-endian only (`static_assert` in the framer); fixed records have size `static_assert`s.
- Records never span chunks; a record body is at most `k_max_record_body_bytes` (0xFFFF).
- `chunk_framer` is not reentrant and about 1 MiB in size; keep it off small stacks and call it from a single writer thread. The sink runs on that thread, must not block, and its exceptions are caught and counted in `sink_errors()`.
- Error policy: `chunk_framer` ctor/`start()`/`write` throw `config_error`, `usage_error`, `resource_error`; `reader` and `sample_reconstructor::feed` are `noexcept` and report only via `reader_counters`/`is_fatal()`. A bad magic, unsupported major or out-of-range chunk size makes the reader fatal and silent.
- Reader limits: resync seq jump capped at 1,000,000; trie capped at 2^24 nodes.
- `memory_sink` is single-threaded and once full drops all later chunks, even small ones.
- `module_map::remove_module` ignores unmatched removes silently.

### Known weak spots
- `reader.cpp` is 865 lines holding four internal classes plus per-record decoders.
- `chunk_framer.hpp` is 550 lines of header-only code, including the encoders, so every includer recompiles it.
- `core` hosts non-wire items (`raw_sample`, `self_exe`, `frame_limits`, `format_hex`) and `error.hpp` sits outside the `core/` directory, so the layer boundary is blurry.

### Files

#### cpu_snap root and core/

Root of `src/cpu_snap/` holds only `error.hpp` (library-wide exception hierarchy, namespace `cpu_snap`).
`core/` (namespace `cpu_snap::core`, static lib `cpu_snap::core`) is the wire-format layer: record types, chunk writer, push reader, and the sample/module reconstruction built on top.
Everything here is capture-agnostic (no perf/signal code); `core` links only Threads, and `error.hpp` is installed alongside its headers.

##### (root of cpu_snap/)

###### `error.hpp`
- **Location:** `src/cpu_snap/` root; namespace `cpu_snap` (not `core`), header shared by all layers.
- **Size:** 94 lines, header-only
- **Purpose:** Defines the exception hierarchy for every library-originated failure.
- **Contents:** `error`, `config_error`, `usage_error`, `detail::coded_error`, `environment_error`, `resource_error`.
- **Implementation:** All derive from `std::runtime_error` via `error`. They are thrown only from constructors, `start()` and setup on the caller's thread, never from the signal handler, worker loop, destructors, atfork handlers or per-sample path. `coded_error` adds an optional `std::error_code` (errno in `generic_category`); `environment_error::from_errno` formats `"<what>: <strerror>"`.
- **Public API:**
  - `class error : std::runtime_error` (base; catch to handle any cpu-snap failure)
  - `class config_error : error` (invalid configuration)
  - `class usage_error : error` (wrong call order / API misuse)
  - `class environment_error : detail::coded_error` (kernel/OS/privilege refusal); `static environment_error from_errno(int err, const std::string& what)`
  - `class resource_error : detail::coded_error` (limit or capacity exhausted)
  - `std::error_code detail::coded_error::code() const noexcept` (inherited by the two above; empty when no OS code)

##### core

###### `core/CMakeLists.txt`
- **Location:** `src/cpu_snap/core/`; build file for target `core` (alias `cpu_snap::core`).
- **Size:** 26 lines, CMake
- **Purpose:** Builds the `core` static library.
- **Contents:** Sources `module_map.cpp`, `reader.cpp`, `sample_reconstructor.cpp`; C++20, Threads dependency, warning flags.
- **Implementation:** Public include root is `src`; compiled with `-fno-omit-frame-pointer`. When `CPU_SNAP_INSTALL` is set it installs the target, `error.hpp`, and all `*.hpp` of the directory.
- **Public API:** none (target `cpu_snap::core`).

###### `core/types.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 53 lines, header-only
- **Purpose:** Public callback aliases and the default chunk size of the wire pipeline.
- **Contents:** `k_default_chunk_bytes`, `sink_fn`, `event_fn`.
- **Implementation:** Both callbacks are `std::function` so the public ABI does not bind to a concrete callable. `k_default_chunk_bytes` lives here so sampling headers can use it without pulling in `chunk_framer.hpp`. The sink is invoked from the single worker thread, serialized and must not block.
- **Public API:**
  - `inline constexpr std::uint32_t k_default_chunk_bytes = 1U << 20`
  - `using sink_fn = std::function<void(std::span<const std::byte>)>`
  - `using event_fn = std::function<void(const record_variant&)>`

###### `core/records.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 345 lines, header-only
- **Purpose:** Defines every on-wire record and framing structure of the v1 (major 2) format.
- **Contents:** Version constants, `record_type` tag enum, fixed 16-byte records, variable-size records, `record_variant`, packed `file_header` (64 B) and `chunk_header` (16 B), stream-clock flag helpers.
- **Implementation:** Fixed records are POD with `static_assert` on size; variable ones own vectors/strings and are serialized field-by-field by `chunk_framer`. `file_header::flags` bit 1 = frame-pointer stack, bit 2 = DWARF stack, bits 3-4 = stream clock. Reserved clock value 3 decodes as monotonic.
- **Public API:**
  - `k_record_format_version_major = 2`, `k_record_format_version_minor = 0`
  - `enum class record_type : uint8_t { invalid, new_node, sample, checkpoint, maps_init, maps_delta, lost, eof }`
  - `new_node_record {node_id, parent_id, ip}`; `sample_record {timestamp_ns, tid, leaf_node_id}`; `lost_record {wall_ns, tid, lost_count_delta}`; `eof_record {wall_ns, total_samples_emitted}`
  - `per_tid_checkpoint_entry {tid, leaf_id, samples, lost}`; `k_per_tid_checkpoint_entry_bytes = 24`; `checkpoint_record {wall_ns, entries}`
  - `module_info {base_va, size, file_offset, build_id, path}`; `maps_init_record {wall_ns, modules}`; `maps_delta_record {wall_ns, enum op{add,remove} operation, module}`
  - `using record_variant = std::variant<new_node_record, sample_record, checkpoint_record, maps_init_record, maps_delta_record, lost_record, eof_record>`
  - `constexpr record_type type_of(const record_variant&) noexcept`
  - `struct file_header {magic, version_major, version_minor, start_unix_ns, pid, chunk_bytes, flags, process_name}` (packed)
  - `struct chunk_header {magic, version, records_byte_count, chunk_seq}` (packed)
  - `enum class stream_clock : uint8_t { monotonic, boottime, wallclock }`; `k_flags_clock_shift`, `k_flags_clock_mask`
  - `constexpr uint32_t with_stream_clock(uint32_t flags, stream_clock) noexcept`
  - `constexpr stream_clock stream_clock_of(const file_header&) noexcept`

###### `core/chunk_framer.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 550 lines, header-only
- **Purpose:** Serializes records into framed chunks and passes each to a sink.
- **Contents:** `chunk_framer` class, `k_min_chunk_bytes`, `k_max_record_body_bytes`, private wire helpers (`record_header_wire`, `body_size_of`, `encode_body`, `encode_module`, `put`, `align4`).
- **Implementation:** `start()` emits the 64-byte file header, then records are appended into an inline 1 MiB `alignas(64)` buffer (so an instance is about 1 MiB; keep it off small stacks). Records never span chunks and are 4-byte aligned; when one would overflow `chunk_bytes` the chunk is sealed and emitted. `write_split` overloads split large checkpoint/maps_init records across several records under the u16 body limit. Not reentrant; sink exceptions are caught and counted in `sink_errors()` so the worker survives. Little-endian only (`static_assert`).
- **Public API:**
  - `inline constexpr uint32_t k_min_chunk_bytes`, `k_max_record_body_bytes = 0xFFFF`
  - `struct chunk_framer::config {uint32_t chunk_bytes; uint16_t flags; uint64_t pid; uint64_t start_unix_ns; std::string process_name}`
  - `chunk_framer(config cfg, sink_fn sink)` (throws `config_error` on null sink or out-of-range chunk_bytes); non-copyable, non-movable
  - `void start()` (throws `usage_error` if called twice)
  - `void write(const record_variant& rec)` (throws `usage_error` before `start()`, `resource_error` if a record is too large)
  - `void write_split(const checkpoint_record&)`, `void write_split(const maps_init_record&)`
  - `void flush()`
  - `uint64_t sink_errors() const noexcept`

###### `core/reader.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 130 lines, header-only declaration (implementation in `reader.cpp`)
- **Purpose:** Public interface of the push-based wire-stream decoder.
- **Contents:** `reader_counters` and `reader` (pimpl).
- **Implementation:** Callers `feed()` bytes at any granularity; the reader buffers partial chunks and fires the event callback per decoded record. Failures never throw: they become counters, and a rejected chunk header triggers a forward resync scan. A bad file-header magic, unsupported major version or out-of-range chunk size sets the fatal flag and silences emission.
- **Public API:**
  - `struct reader_counters {bytes_fed, records_emitted, parse_errors, gap_count, last_seen_seq, unknown_records, chunks_rejected, resyncs, bytes_skipped}` (all `uint64_t`)
  - `reader()`, `~reader()`; non-copyable, non-movable
  - `void on_event(event_fn fn)`
  - `void feed(std::span<const std::byte> data) noexcept`
  - `std::vector<uint64_t> resolve_stack(uint32_t leaf_id) const` (innermost-first; empty if unknown)
  - `const file_header& header() const noexcept`
  - `const reader_counters& counters() const noexcept`
  - `bool is_fatal() const noexcept`

###### `core/reader.cpp`
- **Location:** `core/`, namespace `cpu_snap::core` (internals in anonymous namespace).
- **Size:** 865 lines, .cpp
- **Purpose:** Implements the wire decoder behind `reader`.
- **Contents:** Implements the API of `reader.hpp`. Internal helpers: `load_le`, `align4`, `decode_*` functions per record type (bounds-checked), `trie_resolver`, `record_walker`, `chunk_assembler`, `reader::impl`.
- **Implementation:** `chunk_assembler` is a three-state machine (file header, chunk header, chunk body) that validates headers, tracks `chunk_seq` gaps, and resyncs by scanning for a plausible header (seq jump capped at 1,000,000). `record_walker` decodes each record and dispatches it; unknown types are skipped by body size, malformed known records abandon the rest of the chunk. `trie_resolver` rebuilds the NEW_NODE trie (parent must precede child, capped at 2^24 nodes) to resolve leaf ids to frame chains.

###### `core/module_map.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 93 lines, header-only declaration (implementation in `module_map.cpp`)
- **Purpose:** Time-aware lookup of loaded modules from MAPS_INIT/MAPS_DELTA records.
- **Contents:** `module_map` with nested `module_ref`.
- **Implementation:** Keeps every module with a `[load_ts, unload_ts)` window so a sample resolves against the layout valid at its timestamp, which handles dlopen/dlclose. Lets a profile be symbolized without reading `/proc/<pid>/maps`, even in another process.
- **Public API:**
  - `struct module_map::module_ref {base_va, size, file_offset, build_id, path}`
  - `void apply(const maps_init_record&)`
  - `void apply(const maps_delta_record&)`
  - `std::optional<module_ref> resolve(uint64_t ip, uint64_t timestamp_ns) const`

###### `core/module_map.cpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 97 lines, .cpp
- **Purpose:** Implements `module_map`.
- **Contents:** Implements the API of `module_map.hpp`. Private helpers: `add_module`, `remove_module`.
- **Implementation:** Intervals are a vector sorted by `base_va` (insertion order kept for equal bases). `add_module` does a sorted insert and tracks the largest module size. `remove_module` closes the most recent open interval with matching base and path (silently ignores no match). `resolve` binary-searches the ip and scans backward only while the offset is below the max module size, checking size and time window.

###### `core/raw_sample.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 44 lines, header-only
- **Purpose:** The decoded stack-sample record passed from sampling to post-processing.
- **Contents:** `struct raw_sample`; split out of `sampler.hpp` to keep post-processor TUs light.
- **Implementation:** Plain data; `frames` is innermost-first (kernel callchain order). Written by the worker thread, read after `stop()`.
- **Public API:**
  - `struct raw_sample { pid_t tid = -1; uint64_t timestamp_ns; uint64_t ip; std::vector<uint64_t> frames; }`

###### `core/sample_reconstructor.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 79 lines, header-only declaration (implementation in `sample_reconstructor.cpp`)
- **Purpose:** Adapter that turns a wire byte stream into per-tid `raw_sample` vectors plus a `module_map`.
- **Contents:** `sample_reconstructor` class wrapping a `reader`.
- **Implementation:** Replaces reader-to-sample conversion previously copy-pasted in demos. Samples arrive chronologically per stream, so no sort is needed. Samples whose leaf cannot be resolved are counted, not stored.
- **Public API:**
  - `sample_reconstructor()`, `~sample_reconstructor()`; non-copyable, non-movable
  - `void feed(std::span<const std::byte> data) noexcept`
  - `const std::unordered_map<pid_t, std::vector<raw_sample>>& samples() const noexcept`
  - `const module_map& modules() const noexcept`
  - `const reader_counters& counters() const noexcept`
  - `uint64_t unresolved_samples() const noexcept`

###### `core/sample_reconstructor.cpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 85 lines, .cpp
- **Purpose:** Implements `sample_reconstructor`.
- **Contents:** Implements the API of `sample_reconstructor.hpp`. Private helper `handle_event(const record_variant&)`.
- **Implementation:** The constructor installs `handle_event` as the reader callback. For `sample` records it skips leaf id 0, resolves the stack (counting failures in `m_unresolved_samples`), and appends a `raw_sample` (ip = innermost frame) under its tid. `maps_init`/`maps_delta` go to `module_map::apply`; other records are ignored.

###### `core/memory_sink.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 83 lines, header-only
- **Purpose:** Bounded in-memory `sink_fn` target.
- **Contents:** `memory_sink` with nested `config`.
- **Implementation:** Stop-when-full, deliberately not a ring buffer: an early chunk may hold NEW_NODE records later samples depend on, so dropping is always whole-chunk and the retained bytes stay a valid decodable prefix. Not thread-safe (only called from the single worker thread).
- **Public API:**
  - `struct memory_sink::config { std::size_t max_bytes; }`
  - `explicit memory_sink(config) noexcept`
  - `void operator()(std::span<const std::byte> chunk)`
  - `std::span<const std::byte> bytes() const noexcept`
  - `bool full() const noexcept`
  - `std::size_t dropped_chunks() const noexcept`, `std::size_t dropped_bytes() const noexcept`

###### `core/frame_limits.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 26 lines, header-only
- **Purpose:** Default and maximum callchain depth constants.
- **Contents:** Two constants, kept as a leaf header so capture and storage share them without heavy includes.
- **Implementation:** The depth in use is a runtime setting; these only give its default and the validation bound.
- **Public API:**
  - `inline constexpr uint32_t k_default_callchain_depth = 256`
  - `inline constexpr uint32_t k_max_callchain_depth = 4096`

###### `core/format_hex.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 30 lines, header-only
- **Purpose:** Hex-formats an address.
- **Contents:** One inline function using `std::to_chars` into a stack buffer.
- **Implementation:** Equivalent to `printf("0x%lx")`: lowercase, no padding, 0 gives "0x0". No allocation beyond the returned string.
- **Public API:**
  - `[[nodiscard]] inline std::string format_hex(std::uint64_t value)`

###### `core/self_exe.hpp`
- **Location:** `core/`, namespace `cpu_snap::core`.
- **Size:** 39 lines, header-only
- **Purpose:** Returns the path of the running executable.
- **Contents:** One inline function.
- **Implementation:** `readlink("/proc/self/exe")` into a 4096-byte buffer (longer paths truncated); returns an empty string on failure, for example in a chroot without `/proc`.
- **Public API:**
  - `[[nodiscard]] inline std::string self_exe_path()`

---
File sections written: 16 (error.hpp, CMakeLists.txt, types, records, chunk_framer, reader.hpp, reader.cpp, module_map.hpp/.cpp, raw_sample, sample_reconstructor.hpp/.cpp, memory_sink, frame_limits, format_hex, self_exe)

## sampling

### What it is for
`sampling` is the online recorder of cpu-snap. It samples the call stacks of every thread of the host process at a fixed frequency, unwinds them, and streams them in the compact wire format (chunk + frame-trie) to a user-supplied sink. It hides the two kernel-facing capture mechanisms (perf_event_open and per-thread timers + signal handler) and the unwind strategies behind one `sampler` class.

### Responsibilities
- Validate and resolve a `sampler::config` into an effective plan (`resolve_plan`): backend (`automatic` -> perf with fallback to signal, by probing `perf_event_open`), unwind mode (`auto_best` -> dwarf), perf ring sizing, clock (`clock_kind`).
- Own one `triggers::i_trigger` backend (`triggers::perf::backend` over `kernel_drain`, or `triggers::signal::backend`) and one `unwind::unwinder` variant; check the trigger/unwinder pairing via `can_provide(required_capture())`.
- Run the worker thread (`worker_loop`): wait, drain, periodically rescan threads.
- Track threads (`thread_tracker` over `/proc` listing; attach initial, attach new, detach exited), exclude the sampler worker tid.
- Re-arm correctly after `fork` (`fork_registry` + `sampler::on_child_fork`).
- Turn samples into the stream: `streaming_pipeline` (`temporal_diff_filter` -> `frame_trie` -> `core::chunk_framer` -> sink), module map polling (`dl_module_watcher`), checkpoints, lost/total counters.
- Provide pause/resume/stop, stats, capabilities probe (`sampler_capabilities`) and a pluggable logger (`log_sink`, `log_record`).

### Explicitly not responsible for
- Wire format definition, chunk framing and clock types: owned by `core`.
- Decoding, symbolization or Perfetto emission: owned by `processing` (no dependency on `sampling`).
- Choosing the target process or launching it (`tools/cpu_snap_run`).
- Writing the stream to a file or network: the sink is injected.

### Main building blocks
| unit | role |
|---|---|
| `sampler` (`sampler.hpp/.cpp`) | Public driver: config, start/pause/resume/stop, worker thread, stats |
| `sampler_plan` (`resolve_plan`, `make_drain_config`, `make_signal_config`, `capture_kind_for`, `streaming_flags_for`) | Stateless config validation/resolution into per-backend configs |
| `backend_selection`, `backend_kind`, `unwind_mode`, `clock_kind`, `time.hpp` | Selection enums, injectable perf probe, clock helpers |
| `sampler_capabilities` | Runtime probe of what the host allows (perf, paranoid, etc.) |
| `triggers/i_trigger.hpp` | Strategy interface (attach/detach tid, wait, drain_into, enable/disable, close_all) |
| `triggers/perf/*` (`backend`, `kernel_drain`, `perf_ring`, `perf_sample_decoder`, `ring_sizing`, `open_error`) | perf_event_open backend: mmap rings, epoll drain, record decoding |
| `triggers/signal/*` (`backend`, `ring_table`, `snapshot`) | Timer + realtime signal backend: AS-safe handler, lock-free per-tid ring table, eventfd wake |
| `unwind/*` (`unwinder` variant, `kernel_callchain_unwinder`, `fp_copy_unwinder`, `dwarf_unwinder` on libdw, `machine_snapshot`, `perf_regs`) | Closed set of unwind strategies |
| `streaming_pipeline`, `temporal_diff_filter`, `frame_trie`, `flat_tid_map` | Sample to wire-stream encoder with dedup, trie of frames, per-tid state |
| `dl_module_watcher` | Tracks loaded modules (maps / dl_iterate_phdr) and emits module events |
| `thread_tracker`, `thread_comm`, `fork_registry` | Thread set reconciliation, thread names, fork child handlers |
| `logger.hpp` | `log_sink` callback and `internal::emit` (lazy message building, stamped with the chosen clock) |

### How it works
1. `sampler(config, trigger = nullptr)`: `probe_sampler_capabilities`, `resolve_plan` (throws `config_error` / `environment_error`), `select_unwinder` (dwarf; else fp_copy for signal; else kernel_callchain for perf), build `thread_tracker`.
2. If no trigger was injected, construct `triggers::perf::backend` or `triggers::signal::backend` from `make_drain_config` / `make_signal_config`; reject an impossible pair with `config_error`; log the backend choice and configuration.
3. `start()`: emplace `streaming_pipeline` (`build_streaming_config`), `thread_tracker::attach_initial` under `m_state_mutex`, spawn `worker_loop`, register in `fork_registry`. Any failure goes through `rollback_start()` and rethrows.
4. `worker_loop` (every iteration): `m_backend->wait(10 ms)`, `drain_once()` = `m_backend->drain_into(*m_streaming, m_unwinder)`; every `rescan_period`, `rescan_threads()` (poll `/proc`, then `thread_tracker::rescan` under the mutex, failures counted in `m_rescan_failures`).
5. Inside `drain_into`: one `std::visit` on the unwinder per batch, then per sample decode (perf record or signal snapshot), unwind, `streaming_pipeline::on_sample` -> `temporal_diff_filter` -> `frame_trie` (emit new nodes) -> `chunk_framer` -> sink; `on_lost`, `on_tid_exit`, periodic maps poll and checkpoint.
6. Signal backend: per-thread CPU-time timer delivers an RT signal; `signal_handler` fills a slot from `ring_table::enter/leave` (AS-safe; `process_vm_readv` for stack copy), writes the eventfd when a ring fills; worker drains.
7. `stop()`: set `m_shutdown`, join worker, `disable_all`, final `drain_once`, record peak tracked tids, `close_all`, `finalize` the pipeline and keep its stats.
8. After fork, `on_child_fork()` detaches the worker, `close_all`, `abandon()` the pipeline, resets the mutex and the running flags.

### Dependencies and users
- Depends on `core` (types, `chunk_framer`, `sink_fn`, `stream_clock`), `cpu_snap/error.hpp`, libdw (dwarf unwinder), librt, Threads via core.
- Used by `tools/cpu_snap_run`, `examples/*` (single/multi-thread demos, `backend_comparison_demo`, `autoload`, `hip_matmul_demo`), `bench/bench_streaming.cpp`, `bench/bench_overhead.cpp`.
- Tested in `tests/sampling/*` (sampler, kernel_drain, streaming, fork child handler, clock option, flat_tid_map, dl_module_watcher, unwinders, thread_comm) and `tests/test_coverage_extras.cpp`; many perf tests skip when `perf_event_paranoid` is restrictive.

### Invariants and constraints
- Error policy: exceptions (`config_error`, `environment_error`, `resource_error`) only from the constructor, `start()`, `attach_tid` and setup; all other `i_trigger` operations are `noexcept`; the worker catches everything, sets `m_worker_failed` and logs, never reaches `std::terminate`.
- Hot path is single-worker; per-sample dispatch is non-virtual. Virtual calls and `std::visit` happen only per drain batch.
- The signal handler must be async-signal-safe: fixed buffers, lock-free `ring_table`, atomics, `process_vm_readv`, eventfd write; a single process-wide handler whose prior disposition is restored on destruction. `signal/backend.hpp` stays free of glibc signal symbols (`_GNU_SOURCE` only in the .cpp).
- Shared state (`m_streaming`, tracker, backend attach/detach) is guarded by `m_state_mutex`; `m_running`, `m_paused`, `m_shutdown` are atomics.
- `fork` child must not use the parent's worker or mutex; hence `fork_registry` and `on_child_fork`.
- The sampler worker tid is excluded from sampling when `exclude_sampler_worker` is set.
- Logger callbacks are lazy (`emit` takes a message builder) and stamped with the configured clock.

### Known weak spots
- `triggers/signal/backend.cpp` (831 lines) and `unwind/dwarf_unwinder.cpp` (596 lines) are the largest units and mix several concerns (handler, install/restore, enrollment, drain in one file).
- `sampler` is a large class (`sampler.hpp` 560 lines) with many atomics and a config struct holding perf, signal, threads, streaming options, plus `m_cfg.perf.unwind` is used even for the signal backend (unwind option lives under `perf`).
- `sampler_plan.hpp` includes the perf and signal backend headers just for their config types, coupling plan resolution to both backends.
- `on_child_fork` re-constructs a `std::mutex` with `construct_at` over a possibly-locked mutex, relying on single-threaded child state.

### Files

##### sampling

The sampling/ folder (namespace cpu_snap::sampling, with cpu_snap::sampling::internal for non-public pieces and ::utils for helpers) is the online recorder: the sampler driver, backend/clock/unwind selection, and the streaming pipeline that encodes samples to the wire format.
Capture mechanisms live in the subfolders triggers/ (perf, signal) and unwind/; this file set holds only the neutral driver and its support classes.
The hot path is single-worker: sampler worker -> trigger drain -> streaming_pipeline (temporal_diff_filter -> frame_trie -> chunk_framer -> host sink).

###### `sampling/backend_kind.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling (public config enum).
- **Size:** 24 lines, header-only.
- **Purpose:** Selector for the capture mechanism the sampler builds.
- **Contents:** enum class backend_kind.
- **Implementation:** A plain enum. `automatic` prefers perf and falls back to signal when perf_event_open is not permitted; `perf` and `signal` are explicit. Used in sampler::config::backend and by resolve_backend().
- **Public API:**
  - `enum class backend_kind : std::uint8_t { automatic, perf, signal }`

###### `sampling/backend_selection.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling.
- **Size:** 94 lines, header-only.
- **Purpose:** Resolve backend_kind::automatic to a concrete backend by probing perf_event_open.
- **Contents:** perf_probe_fn alias, backend_choice struct, resolve_backend(), describe_backend_choice().
- **Implementation:** signal short-circuits without probing. Otherwise the injected probe is called; success gives perf. On failure an explicit perf request throws environment_error (errno as code, message from describe_perf_open_failure with perf_event_paranoid), while automatic returns signal with a human-readable fallback reason. The probe is a function pointer so tests can inject it.
- **Public API:**
  - `using perf_probe_fn = perf_probe_result (*)(bool exclude_kernel)`
  - `struct backend_choice { backend_kind kind; std::string reason; }` (kind never automatic; reason non-empty only on fallback)
  - `backend_choice resolve_backend(backend_kind requested, bool exclude_kernel, perf_probe_fn probe)` (throws cpu_snap::environment_error for explicit perf when unavailable)
  - `std::string describe_backend_choice(const backend_choice&)`

###### `sampling/clock_kind.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling.
- **Size:** 27 lines, header-only.
- **Purpose:** Selector of the one clock all library timestamps use.
- **Contents:** enum class clock_kind.
- **Implementation:** Plain enum covering samples, module records, lost/checkpoint/eof records and log records. Mapped to POSIX clocks in time.hpp and to the wire-format flag bits in sampler_plan.cpp.
- **Public API:**
  - `enum class clock_kind : std::uint8_t { monotonic, boottime, wallclock }` (CLOCK_MONOTONIC / CLOCK_BOOTTIME / CLOCK_REALTIME)

###### `sampling/dl_module_watcher.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling (streaming-time module observer).
- **Size:** 151 lines, header-only declaration (implementation in dl_module_watcher.cpp).
- **Purpose:** Declare the class that tracks loaded shared objects and emits maps_init / maps_delta records.
- **Contents:** class dl_module_watcher with nested injected_module and iterator_fn (test hook).
- **Implementation:** snapshot() publishes the initial module set; poll() diffs against the last-seen set and returns add/remove deltas. Not thread-safe; owned and driven by the streaming pipeline worker. Lets offline symbolization pin each IP timestamp to the binary mapped at that time despite dlopen/dlclose.
- **Public API:**
  - `explicit dl_module_watcher(clock_kind clock = clock_kind::monotonic) noexcept`; non-copyable, non-movable
  - `core::maps_init_record snapshot()`
  - `std::vector<core::maps_delta_record> poll()` (removes first, then adds)
  - `std::size_t known_module_count() const noexcept`
  - `std::uint64_t full_walk_count() const noexcept`
  - `void set_iterator_for_tests(iterator_fn fn)`
  - `struct injected_module { base_va, size, file_offset, path }`; `using iterator_fn = std::function<void(std::function<void(const injected_module&)>)>`

###### `sampling/dl_module_watcher.cpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling.
- **Size:** 365 lines, .cpp.
- **Purpose:** Walk dl_iterate_phdr, read GNU build-ids and compute module deltas.
- **Contents:** Implements the API of dl_module_watcher.hpp. Internal helpers: read_build_id_from_note (parses NT_GNU_BUILD_ID in PT_NOTE), build_module_from_phdr (first executable PT_LOAD gives page-aligned base/size/file offset), dl_iterate_callback (labels the first nameless entry as the main exe via core::self_exe_path(), later nameless ones as "[vdso]"), read_generation (dlpi_adds/dlpi_subs).
- **Implementation:** poll() first compares the linker generation counters and returns empty without walking if unchanged. Otherwise it walks, builds hash sets keyed on (base_va, path) for O(N) diffing, emits removes then adds. Add timestamps use the previous observation time (so samples taken between dlopen and poll resolve), removes and same-base re-adds use the current time. Known limits: dlopen+dlclose between polls is invisible.

###### `sampling/flat_tid_map.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling.
- **Size:** 242 lines, header-only template.
- **Purpose:** Hot-path per-tid associative container faster than std::unordered_map.
- **Contents:** template class flat_tid_map<T>.
- **Implementation:** 256-slot direct-mapped table indexed by tid & 255 with linear probing (limit 8) and tombstones; overflow into std::unordered_map when the probe window is full. Used by temporal_diff_filter and streaming_pipeline. Not thread-safe; references stay valid until an overflow rehash.
- **Public API:**
  - `static constexpr std::size_t k_slot_count = 256`
  - `T& operator[](pid_t tid)` (get-or-insert)
  - `T* find(pid_t) noexcept` and `const T* find(pid_t) const noexcept`
  - `void erase(pid_t tid)`
  - `std::size_t size() const noexcept`, `std::size_t overflow_size() const noexcept`
  - `template <typename Fn> void for_each(Fn&&)` (const and non-const; callback `(pid_t, T&)`)

###### `sampling/fork_registry.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal::fork_registry.
- **Size:** 29 lines, header-only declarations (implementation in fork_registry.cpp).
- **Purpose:** Process-wide list of objects that must clean up child-side state after fork().
- **Contents:** child_callback alias, add(), remove().
- **Implementation:** The first add() installs a single pthread_atfork(child) handler, so it runs after handlers registered earlier (e.g. the signal backend's). The sampler registers itself on start() and unregisters in its destructor.
- **Public API:**
  - `using child_callback = void (*)(void* object) noexcept`
  - `void add(void* object, child_callback on_child)`
  - `void remove(void* object) noexcept`

###### `sampling/fork_registry.cpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal::fork_registry.
- **Size:** 75 lines, .cpp.
- **Purpose:** Implement the fork registry.
- **Contents:** Implements the API of fork_registry.hpp. Internals: function-local static mutex and unordered_map<void*, child_callback>, child_handler, ensure_atfork_registered (one-time pthread_atfork).
- **Implementation:** child_handler deliberately does NOT take the registry mutex (it may be held by a parent thread at fork time and would deadlock in the single-threaded child); it iterates the map directly, calls each callback, then clears the map. add()/remove() lock the mutex; remove() is noexcept so a poisoned mutex terminates.

###### `sampling/frame_trie.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal.
- **Size:** 191 lines, header-only.
- **Purpose:** Grow-only trie of callstack frames that is the core of wire compression.
- **Contents:** class frame_trie with intern_result and node_view structs, edge key/hash internals.
- **Implementation:** Each unique (parent_node_id, ip) edge maps to one node id (>= 1; 0 is the implicit root) through an unordered_map keyed on a 64-bit mix; nodes stored in a vector. Callers must emit a NEW_NODE record before any SAMPLE referencing a newly allocated id. A non-zero cap makes intern() throw resource_error. Single-thread only.
- **Public API:**
  - `using node_id = std::uint32_t; static constexpr node_id k_root = 0`
  - `struct intern_result { node_id id; bool was_new; }`, `struct node_view { node_id id; node_id parent_id; std::uint64_t ip; }`
  - `intern_result intern(node_id parent_id, std::uint64_t ip)` (throws cpu_snap::resource_error over the cap)
  - `node_id find(node_id parent_id, std::uint64_t ip) const noexcept`
  - `std::size_t size() const noexcept`, `node_id last_assigned_id() const noexcept`
  - `node_view node_at(node_id id) const noexcept` (1 <= id <= last_assigned_id)
  - `void set_max_nodes(std::size_t cap) noexcept` (0 = uncapped)
  - Default ctor/dtor, move-only.

###### `sampling/logger.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling (+ internal for emit helpers).
- **Size:** 90 lines, header-only.
- **Purpose:** Injected log sink; the library never writes to stderr itself.
- **Contents:** log_level, log_record, log_sink, internal::emit overloads.
- **Implementation:** An empty sink costs nothing. Records are stamped from the configured clock. The sink is called only from the sampler's control and worker threads, never from a signal handler or per-sample path; it must not call back into the sampler (worker-thread records are delivered under the internal state lock). Sink exceptions are swallowed. The lazy overload builds the message only when a sink exists.
- **Public API:**
  - `enum class log_level : std::uint8_t { debug, info, warn, error }`
  - `struct log_record { log_level level; std::uint64_t timestamp_ns; std::string_view message; }`
  - `using log_sink = std::function<void(const log_record&)>`
  - `internal::emit(const log_sink&, clock_kind, log_level, std::string_view) noexcept`
  - `internal::emit(const log_sink&, clock_kind, log_level, MessageFn&&) noexcept` (MessageFn returns std::string)

###### `sampling/perf_ring.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling.
- **Size:** 221 lines, header-only.
- **Purpose:** RAII owner of a perf_event_open mmap ring plus a decoder-templated drain loop.
- **Contents:** struct perf_ring_config, class perf_ring.
- **Implementation:** Takes over the perf fd, mmap region and optional epoll registration; the destructor does EPOLL_CTL_DEL, munmap, close. drain() reads data_head with acquire, walks records from data_tail, validates sizes (drops the rest on corruption), splices wrapped records into a scratch buffer, calls the decoder, and publishes data_tail with release even if the decoder throws. It does not enable or disable the event. Used by triggers/perf (kernel_drain). Move-only.
- **Public API:**
  - `struct perf_ring_config { int fd; void* mmap_base; std::size_t mmap_bytes; int epoll_fd_for_dereg; }`
  - `explicit perf_ring(perf_ring_config cfg) noexcept`
  - `template <typename Decoder> std::size_t drain(Decoder&&)` (decoder `(const perf_event_header*, const std::uint8_t* record)`)
  - `int fd() const`, `void* mmap_base() const`, `std::size_t data_bytes() const`, `bool valid() const` (all noexcept)
  - `void forget_epoll_registration() noexcept` (for the fork child)

###### `sampling/sampler_capabilities.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling (+ detail).
- **Size:** 141 lines, header-only.
- **Purpose:** Probe which perf_event_open features the kernel and CPU accept.
- **Contents:** sampler_capabilities, perf_probe_result, probe_perf_open(), probe_sampler_capabilities(), detail::try_open/probe_precise_ip/probe_ibs_op.
- **Implementation:** Each probe opens and closes one throw-away perf event (software CPU_CLOCK for availability, hardware cycles with precise_ip=1 for PEBS/IBS), and checks /sys/devices/ibs_op/type for AMD IBS. Reflects the real sysctl, CAP_PERFMON, seccomp and LSM policy. No global state.
- **Public API:**
  - `struct sampler_capabilities { bool frame_pointer = true; bool precise_ip; bool ibs_op; std::string to_string() const; }`
  - `struct perf_probe_result { bool available; int err; }`
  - `perf_probe_result probe_perf_open(bool exclude_kernel) noexcept`
  - `sampler_capabilities probe_sampler_capabilities() noexcept`

###### `sampling/sampler.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling (the main public entry point).
- **Size:** 560 lines, header-only declaration (implementation in sampler.cpp).
- **Purpose:** Declare the online CPU stack recorder for all threads of the current process.
- **Contents:** class sampler with nested option structs (perf_attr_options, signal_attr_options, thread_attach_options, streaming_options, config), stats accessors, and private state (trigger, unwinder, pipeline, worker thread, thread tracker, atomics, state mutex); constant k_default_sample_freq_hz = 200.
- **Implementation:** One worker thread drains the capture trigger into a mandatory streaming sink and periodically rescans threads. Hot path: no symbolization, filtering or user callbacks. Counters are relaxed atomics; m_state_mutex guards backend topology, pipeline lifetime and thread tracker between the worker rescan and caller-thread pause/resume/stats/stop. Readers of sink-owned data must call stop() first.
- **Public API:**
  - `inline constexpr std::uint32_t k_default_sample_freq_hz = 200`
  - `sampler::config { perf_attr_options perf; thread_attach_options threads; streaming_options streaming; backend_kind backend = automatic; signal_attr_options signal; log_sink logger; clock_kind clock = monotonic; }`
  - `perf_attr_options { sample_freq_hz, callchain_depth, kernel_pages_per_tid, exclude_kernel, exclude_hv, dwarf_stack_bytes = 32768, unwind }`
  - `signal_attr_options { sample_freq_hz, signal_rt_offset = 4, stack_snapshot_bytes = 8192, ring_capacity = 256, start_delay_ns }`
  - `thread_attach_options { rescan_period = 100ms, watch_new_threads, include_main_thread, exclude_sampler_worker }`
  - `streaming_options { core::sink_fn sink; chunk_bytes; maps_poll_interval; checkpoint_interval; trie_max_nodes; process_name }`
  - `explicit sampler(config cfg, std::unique_ptr<triggers::i_trigger> backend = nullptr)` (throws config_error / environment_error); non-copyable, non-movable
  - `void start()` (throws environment_error / resource_error after rollback), `void stop()`, `void pause() noexcept`, `void resume() noexcept`, `void on_child_fork() noexcept`
  - `const sampler_capabilities& caps() const`, `const backend_choice& selected_backend() const`, `unwind_mode resolved_unwind_mode() const`
  - `bool is_paused() const`, `bool is_running() const`, `bool worker_failed() const`
  - Stats: `std::size_t total_samples()`, `lost_samples()`, `tracked_threads()`; `std::uint64_t streaming_internal_errors()`, `streaming_dropped_samples()`, `unwinder_live_stack_samples()`, `rescan_failures()`

###### `sampling/sampler.cpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling.
- **Size:** 409 lines, .cpp.
- **Purpose:** Implement the sampler lifecycle, worker loop and construction wiring.
- **Contents:** Implements the API of sampler.hpp. Internals: required_capture_of, select_unwinder (emplaces dwarf_unwinder / fp_copy_unwinder / kernel_callchain_unwinder in the variant), backend_name/unwind_name/clock_name, describe_exception, build_streaming_config, excluded_tid, rescan_threads, drain_once, worker_loop, rollback_start.
- **Implementation:** The constructor calls resolve_plan, selects the unwinder, builds the perf or signal trigger, checks the trigger can provide the unwinder's capture kind, and logs the configuration (plus a hint at >=10 kHz in dwarf mode). start() creates the pipeline, attaches threads, spawns the worker, registers in fork_registry and rolls everything back on any failure. The worker loops: backend wait (10 ms), drain, periodic rescan; an escaping exception sets worker_failed. stop() joins, disables, does a final drain, finalizes the pipeline and keeps final stats. on_child_fork() detaches the worker, closes fds, abandons the pipeline and re-constructs the state mutex.

###### `sampling/sampler_plan.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal.
- **Size:** 59 lines, header-only declarations (implementation in sampler_plan.cpp).
- **Purpose:** Declare the stateless resolution of sampler::config into the effective configuration.
- **Contents:** struct resolved_plan and the resolve/convert free functions.
- **Implementation:** Keeps validation and policy out of the sampler constructor so it can be unit-tested without hardware (probe is injected).
- **Public API:**
  - `struct resolved_plan { sampler::config cfg; backend_choice choice; bool unwind_was_auto; }`
  - `resolved_plan resolve_plan(sampler::config cfg, perf_probe_fn probe, bool trigger_injected)` (throws config_error / environment_error)
  - `triggers::perf::kernel_drain::config make_drain_config(const sampler::config&) noexcept`
  - `triggers::signal::backend::config make_signal_config(const sampler::config&) noexcept`
  - `core::stream_clock stream_clock_for(clock_kind) noexcept`
  - `std::uint16_t streaming_flags_for(unwind_mode resolved, clock_kind) noexcept`
  - `unwind::capture_kind capture_kind_for(unwind_mode, backend_kind) noexcept`

###### `sampling/sampler_plan.cpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal.
- **Size:** 263 lines, .cpp.
- **Purpose:** Implement config validation, backend/unwind resolution and config converters.
- **Contents:** Implements the API of sampler_plan.hpp. Internals: validate_config (sink, callchain depth, power-of-two pages), validate_signal_config (freq, 8-byte stack, ring >= 2, signal number in [SIGRTMIN, SIGRTMAX]), trigger_supports, resolve_unwind_mode, validate_unwind_constraints, resolve_perf_ring_sizing, signal_number.
- **Implementation:** resolve_plan resolves the backend (an injected trigger turns automatic into perf without probing), copies the perf frequency to the signal config on automatic fallback, collapses auto_best to dwarf, validates the explicit mode against the trigger's static supports(), and sizes the perf ring (auto or checked to hold one sample record). streaming_flags_for builds file_header.flags: bit 1 frame-pointer, bit 2 dwarf, bits 3-4 stream clock.

###### `sampling/streaming_pipeline.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal.
- **Size:** 237 lines, header-only declaration (implementation in streaming_pipeline.cpp).
- **Purpose:** Declare the facade that turns per-sample frame arrays into the wire byte stream.
- **Contents:** class streaming_pipeline with config and stats structs, default constants k_default_pipeline_interval (65536) and k_default_trie_max_nodes (2^22).
- **Implementation:** Owns one frame_trie, temporal_diff_filter, core::chunk_framer and dl_module_watcher. on_sample/on_lost/on_tid_exit/finalize are noexcept end to end: all failures (bad_alloc, trie cap, sink throw) are absorbed and counted. NEW_NODE records always precede the SAMPLE that references them. Single-thread, non-reentrant; counters are atomics readable from any thread.
- **Public API:**
  - `struct config { sink_fn sink; chunk_bytes; flags; pid; start_unix_ns; process_name; maps_poll_interval; checkpoint_interval; trie_max_nodes; module_iterator_for_tests; std::atomic<std::size_t>* total_counter; lost_counter; clock_kind clock; }`
  - `explicit streaming_pipeline(config cfg)` (throws config_error on null sink); non-copyable, non-movable
  - `void on_sample(pid_t tid, std::uint64_t timestamp_ns, std::span<const std::uint64_t> frames) noexcept` (frames innermost-first)
  - `void on_lost(std::uint64_t lost_count) noexcept`, `void on_tid_exit(pid_t) noexcept`, `void finalize() noexcept`, `void abandon() noexcept`
  - Counters (uint64_t, noexcept): `samples_emitted()`, `new_nodes_emitted()`, `lost_total()`, `checkpoints_emitted()`, `internal_errors()`, `dropped_samples()`
  - `struct stats { internal_errors; dropped_samples; }`, `stats snapshot() const noexcept`

###### `sampling/streaming_pipeline.cpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal.
- **Size:** 364 lines, .cpp.
- **Purpose:** Implement the pipeline's record emission, error absorption and resync.
- **Contents:** Implements the API of streaming_pipeline.hpp. Internals: make_framer_cfg, record_sample, emit_pending_nodes, record_dropped_sample, note_sink_errors, resync_if_needed, emit_checkpoint, wall_ns_now.
- **Implementation:** The constructor writes the file header and an initial MAPS_INIT. record_sample runs the diff, emits pending trie nodes, then the SAMPLE record, and polls modules / emits a CHECKPOINT every N samples. A failed diff emits a LOST record for that tid. If the framer reports sink errors (a discarded chunk may have held NEW_NODEs) the whole trie is replayed with exponential backoff (capped at 65536 samples). finalize() writes a closing CHECKPOINT, a last module poll, EOF and flushes; abandon() only marks finished (fork child).

###### `sampling/temporal_diff_filter.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal.
- **Size:** 183 lines, header-only.
- **Purpose:** Per-tid running stack tracker that interns only the changed suffix into the frame_trie.
- **Contents:** class temporal_diff_filter with diff_result and per_tid_state.
- **Implementation:** Compares the new frames against the prior path of the same tid from the outermost frame inward, truncates to the common prefix and interns the divergent suffix. An identical stack costs zero trie operations and allocations. The returned new_nodes span points to internal scratch and is valid only until the next diff(). Single-thread, inline for the hot path.
- **Public API:**
  - `using node_id = frame_trie::node_id`
  - `struct diff_result { node_id leaf_id; std::span<const frame_trie::node_view> new_nodes; }`
  - `explicit temporal_diff_filter(frame_trie& trie) noexcept` (trie must outlive the filter)
  - `diff_result diff(pid_t tid, std::span<const std::uint64_t> frames)` (empty frames returns {k_root, {}}; may throw resource_error from the trie)
  - `void forget_tid(pid_t tid)`, `std::size_t tracked_tids() const noexcept`, `node_id current_leaf_for_tid(pid_t) const noexcept`

###### `sampling/thread_comm.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::utils.
- **Size:** 198 lines, header-only.
- **Purpose:** Thread and process identity helpers backed by /proc/self.
- **Contents:** comm readers, stat start-time parser, thread-id listing functions.
- **Implementation:** read_thread_start_ticks uses raw open/read (field 22 of /proc/self/task/<tid>/stat, parsed after the last ')'), giving a tid+start-time identity that survives tid recycling. Listing returns std::nullopt on any iteration error so a partial listing is never trusted. Used by thread_tracker and the sampler.
- **Public API:**
  - `std::string read_thread_comm(pid_t tid)` (falls back to "tid-<N>")
  - `std::string read_proc_self_comm() noexcept`
  - `std::optional<std::uint64_t> parse_stat_start_ticks(std::string_view stat) noexcept`
  - `std::optional<std::uint64_t> read_thread_start_ticks(pid_t tid)`
  - `std::optional<std::vector<pid_t>> try_list_thread_tids(const std::filesystem::path& task_dir)`
  - `std::optional<std::vector<pid_t>> try_list_self_thread_tids()`
  - `std::vector<pid_t> list_self_thread_tids()` (empty on failure)

###### `sampling/thread_tracker.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal.
- **Size:** 114 lines, header-only declaration (implementation in thread_tracker.cpp).
- **Purpose:** Declare the per-thread attach bookkeeping of the sampler.
- **Contents:** thread_entry, thread_source, proc_thread_source(), rescan_result, class thread_tracker.
- **Implementation:** Remembers start ticks of attached threads (to detect recycled tids), attach failures already logged and whether a listing failure was logged. Has no mutex: the owner calls attach_initial/rescan/reset under the trigger-topology lock; poll() touches only tracker state and may run outside it. The thread source is injectable for tests.
- **Public API:**
  - `struct thread_entry { pid_t tid; std::optional<std::uint64_t> start_ticks; }`
  - `using thread_source = std::function<std::optional<std::vector<thread_entry>>()>`; `thread_source proc_thread_source()`
  - `struct rescan_result { std::uint64_t failed_attaches; }`
  - `thread_tracker(thread_source, log_sink, clock_kind, bool include_main_thread)`
  - `void attach_initial(triggers::i_trigger&, pid_t exclude_tid)` (throws environment_error if listing fails)
  - `std::optional<std::vector<thread_entry>> poll()`
  - `rescan_result rescan(triggers::i_trigger&, streaming_pipeline*, pid_t exclude_tid, const std::vector<thread_entry>& current)`
  - `void reset() noexcept`

###### `sampling/thread_tracker.cpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::internal.
- **Size:** 164 lines, .cpp.
- **Purpose:** Implement thread listing, initial attach and rescan reconciliation.
- **Contents:** Implements the API of thread_tracker.hpp. Internals: try_attach (skips the excluded and already tracked tids, records start ticks), log_attach_failure (once per tid).
- **Implementation:** proc_thread_source lists /proc/self/task with start ticks. rescan builds the live set, detaches tracked tids that exited or whose start ticks changed (telling the pipeline via on_tid_exit), then attaches new ones, counting and logging (once per tid) attaches that throw. A failed listing is logged once until a listing succeeds again.

###### `sampling/time.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling::utils.
- **Size:** 55 lines, header-only.
- **Purpose:** Clock helpers shared across the pipeline.
- **Contents:** wall_unix_ns_now, clock_id_of, clock_now_ns.
- **Implementation:** clock_now_ns calls clock_gettime on the clock selected by clock_kind; it is async-signal-safe, so the signal handler can use it. wall_unix_ns_now uses std::chrono::system_clock for file_header.start_unix_ns.
- **Public API:**
  - `std::uint64_t wall_unix_ns_now() noexcept`
  - `constexpr clockid_t clock_id_of(clock_kind) noexcept`
  - `std::uint64_t clock_now_ns(clock_kind) noexcept`

###### `sampling/unwind_mode.hpp`
- **Location:** src/cpu_snap/sampling/, namespace cpu_snap::sampling.
- **Size:** 31 lines, header-only.
- **Purpose:** Selector of the stack-walking strategy.
- **Contents:** enum class unwind_mode.
- **Implementation:** frame_pointer walks rbp chains (perf kernel callchain, or fp_copy_unwinder on the signal backend); dwarf uses user regs plus a stack snapshot with libdwfl and works on any binary at higher cost; auto_best resolves to dwarf on every backend (see sampler_plan.cpp).
- **Public API:**
  - `enum class unwind_mode : std::uint8_t { auto_best, frame_pointer, dwarf }`

##### sampling/triggers
Folder `sampling/triggers/` (namespace `cpu_snap::sampling::triggers`) holds the capture-mechanism ("trigger") axis of the sampler.
It contains the backend-neutral `i_trigger` strategy interface plus two implementations: `perf/` (perf_event_open + mmap rings) and `signal/` (per-thread POSIX timers + signal handler).
The sampler driver owns one trigger via `std::unique_ptr<i_trigger>`; only coarse per-batch operations are virtual, the per-sample path stays concrete.

###### `sampling/triggers/i_trigger.hpp`
- **Location:** `sampling/triggers/`, namespace `cpu_snap::sampling::triggers`; the trigger-axis interface.
- **Size:** 139 lines, header-only.
- **Purpose:** Defines the abstract interface every capture mechanism implements.
- **Contents:** class `i_trigger` (non-copyable, non-movable, virtual dtor); forward declaration of `internal::streaming_pipeline`.
- **Implementation:** The sampler worker loop drives a trigger through these calls: attach/detach threads, wait for data, drain into the pipeline, enable/disable/close. Only `attach_tid` may throw (resource acquisition); everything else is `noexcept`. Per-sample dispatch is deliberately not virtual: `drain_into` receives a `unwind::unwinder` variant and the concrete trigger visits it once per batch.
- **Public API:**
  - `bool can_provide(unwind::capture_kind) const noexcept`
  - `void attach_tid(pid_t)` (may throw)
  - `void detach_tid(pid_t) noexcept`
  - `int wait(int timeout_ms) noexcept`
  - `void drain_into(internal::streaming_pipeline&, unwind::unwinder&) noexcept`
  - `void disable_all() / enable_all() / close_all() noexcept`
  - `std::size_t tracked_tids() const noexcept`
  - `bool is_tracked(pid_t) const noexcept`
  - `void collect_tids(std::vector<pid_t>& out) const`

##### sampling/triggers/perf
Folder `sampling/triggers/perf/` (namespace `cpu_snap::sampling::triggers::perf`) is the perf_event_open backend.
It opens one software CPU-clock event + mmap ring per thread, decodes PERF_RECORD_SAMPLE/LOST records and feeds the unwinders.
Needs perf_event_paranoid/CAP_PERFMON; ring sizing and error text helpers live here too.

###### `sampling/triggers/perf/backend.hpp`
- **Location:** `sampling/triggers/perf/`, namespace `...::triggers::perf`; thin `i_trigger` adapter.
- **Size:** 72 lines, header-only.
- **Purpose:** Declares the perf-based `i_trigger` implementation wrapping `kernel_drain`.
- **Contents:** class `backend : public i_trigger` with one member `kernel_drain m_drain`.
- **Implementation:** Pure forwarding adapter; it supports every `capture_kind` (kernel callchain and regs+stack both come from perf). Used when `config.backend` selects perf.
- **Public API:**
  - `explicit backend(kernel_drain::config cfg)`
  - `bool can_provide(unwind::capture_kind) const noexcept override`
  - `static constexpr bool supports(unwind::capture_kind) noexcept` (always true)
  - all `i_trigger` overrides (`attach_tid`, `detach_tid`, `wait`, `drain_into`, `disable_all`, `enable_all`, `close_all`, `tracked_tids`, `is_tracked`, `collect_tids`)

###### `sampling/triggers/perf/backend.cpp`
- **Location:** `sampling/triggers/perf/`, namespace `...::triggers::perf`.
- **Size:** 62 lines, .cpp
- **Purpose:** Out-of-line bodies of the perf `backend`.
- **Contents:** Forwarding of each `i_trigger` method to `kernel_drain`.
- **Implementation:** Implements the API of `backend.hpp`. `drain_into` does `std::visit` over the unwinder variant and calls `m_drain.drain_all` with two noexcept lambdas that forward to `pipeline.on_sample(tid, ts_ns, frames)` and `pipeline.on_lost(lost)`. `collect_tids` clears the output and fills it via `for_each_tid`.
- **Public API:** Implements the API of `backend.hpp`. No internal helpers beyond the lambdas.

###### `sampling/triggers/perf/kernel_drain.hpp`
- **Location:** `sampling/triggers/perf/`, namespace `...::triggers::perf`; kernel-facing state.
- **Size:** 492 lines, header-only.
- **Purpose:** Owns the per-thread perf_event fd + mmap ring lifecycle and the drain loop.
- **Contents:** class `kernel_drain` with nested `config`; an `unordered_map<pid_t, perf_ring>`, an epoll fd and a `decode_scratch`; static helpers `make_perf_attr`, `perf_event_open`, `mmap_ring_pages`.
- **Implementation:** `open_tid` creates a `PERF_TYPE_SOFTWARE/CPU_CLOCK` freq event (callchain sampling for fp mode, user regs + user stack for dwarf mode), mmaps `1+pages` pages, arms it and registers it in epoll; ESRCH (thread gone) is silently ignored, other failures throw `environment_error` with a paranoid-aware message, and a rejected non-monotonic clock throws instead of silently falling back. `drain_all` walks all rings via `perf_ring::drain`, dispatching PERF_RECORD_SAMPLE to `perf_sample_decoder` and PERF_RECORD_LOST to the lost sink. It has no worker thread and no fork handling; the sampler worker drives it. Templated on unwinder and sinks so the hot path is inlined.
- **Public API:**
  - `struct config { sample_freq_hz, callchain_depth, kernel_pages_per_tid, exclude_kernel, exclude_hv, dwarf_stack_bytes, unwind_mode unwind, clock_kind clock = monotonic }`
  - `explicit kernel_drain(config)`, `~kernel_drain()` (calls `close_all`); non-copyable/non-movable
  - `void open_tid(pid_t)` (throws `environment_error`)
  - `void close_tid(pid_t) noexcept`
  - `template<Unwinder, SampleSink, LostSink> void drain_all(Unw&, const SampleSink&, const LostSink&) noexcept`
  - `void disable_all() / enable_all() / close_all() noexcept`
  - `std::size_t tracked_tids() const noexcept`, `bool is_tracked(pid_t) const noexcept`
  - `template<Fn> void for_each_tid(Fn&&) const`
  - `int wait(int timeout_ms) noexcept` (epoll_wait; drops HUP/ERR fds)
  - `static perf_event_attr make_perf_attr(const config&) noexcept`

###### `sampling/triggers/perf/open_error.hpp`
- **Location:** `sampling/triggers/perf/`, namespace `...::triggers::perf`.
- **Size:** 68 lines, header-only.
- **Purpose:** Builds an actionable message when perf_event_open fails.
- **Contents:** Two inline functions.
- **Implementation:** Reads `/proc/sys/kernel/perf_event_paranoid` and, for EACCES/EPERM, appends the current value and remedies (lower it, CAP_PERFMON, or use the signal backend). Other errors return only the base text. Used by `kernel_drain::open_ring_for`.
- **Public API:**
  - `std::optional<int> read_perf_event_paranoid()`
  - `std::string describe_perf_open_failure(pid_t tid, int err, std::optional<int> paranoid)`

###### `sampling/triggers/perf/perf_sample_decoder.hpp`
- **Location:** `sampling/triggers/perf/`, namespace `...::triggers::perf`.
- **Size:** 244 lines, header-only.
- **Purpose:** Decodes one PERF_RECORD_SAMPLE record and runs the unwinder on it.
- **Contents:** struct `perf_sample_decoder` with nested `decode_scratch`, `sample_prefix`, static decoders `decode_sample_prefix`, `decode_frame_pointer_frames`, `decode_regs_and_stack_frames`, constants `k_sample_prefix_bytes` (24) and `k_perf_user_ip_ceiling` (0x0000800000000000).
- **Implementation:** Skips ip/pid, reads tid and time, then branches at compile time on `Unw::required_capture()`: kernel-callchain records are filtered (user IPs only, depth-bounded) into a `machine_snapshot.ready_frames`; regs+stack records are bounds-checked and handed over as regs pointer + stack span (rsp from register slot 7). Malformed/short records yield zero frames; exceptions from the unwinder are swallowed so the per-sample path never throws.
- **Public API:**
  - `struct decode_scratch { explicit decode_scratch(uint32_t callchain_depth); vector<uint64_t> frames, raw_ips; }`
  - `template<unwind::Unwinder Unw, typename SampleSink> static void process_sample(const perf_event_header*, decode_scratch&, Unw&, const SampleSink& on_sample) noexcept` (sink signature `(pid_t, uint64_t ts, span<const uint64_t> frames)`)
  - helpers (public statics): `decode_sample_prefix`, `decode_frame_pointer_frames`, `decode_regs_and_stack_frames`

###### `sampling/triggers/perf/ring_sizing.hpp`
- **Location:** `sampling/triggers/perf/`, namespace `...::triggers::perf`.
- **Size:** 100 lines, header-only.
- **Purpose:** Constexpr helpers sizing the perf mmap ring and wakeup threshold.
- **Contents:** Constants `k_max_dwarf_stack_bytes` (65528), `k_default_fp_ring_pages` (8), `k_default_ring_records` (8), `k_max_wakeup_events` (32), `k_kernel_max_callchain` (127) and four constexpr functions.
- **Implementation:** Computes the worst-case sample record size per unwind mode (dwarf adds regs + stack dump), derives a power-of-two page count that holds at least 8 records, checks one record fits, and picks a wakeup interval of about a quarter of the ring capacity, clamped to 1..32.
- **Public API:**
  - `size_t max_sample_record_bytes(unwind_mode, uint32_t dwarf_stack_bytes) noexcept`
  - `uint32_t default_ring_pages(unwind_mode, uint32_t dwarf_stack_bytes, size_t page_bytes) noexcept`
  - `bool ring_holds_a_record(unwind_mode, uint32_t dwarf_stack_bytes, uint32_t pages, size_t page_bytes) noexcept`
  - `uint32_t wakeup_events_for(unwind_mode, uint32_t dwarf_stack_bytes, uint32_t pages, size_t page_bytes) noexcept`

##### sampling/triggers/signal
Folder `sampling/triggers/signal/` (namespace `cpu_snap::sampling::triggers::signal`) is the unprivileged timer + signal-handler backend.
The handler snapshots registers and the user stack into preallocated per-thread SPSC rings; the worker drains and unwinds off the signal path.
Everything reachable from the handler is async-signal-safe (no malloc, locks or stdio).

###### `sampling/triggers/signal/backend.hpp`
- **Location:** `sampling/triggers/signal/`, namespace `...::triggers::signal`.
- **Size:** 308 lines, header-only declaration.
- **Purpose:** Declares the signal-based `i_trigger`.
- **Contents:** class `backend : public i_trigger`, nested `config`, private `thread_state` (timer, ring, forwarded_lost) and `retired_ring`; members for the wake eventfd, fork epoch, timers map, tracked-count atomic and frame buffer.
- **Implementation:** One POSIX timer per thread (SIGEV_THREAD_ID, realtime signal) fires the handler, which writes into that thread's `snapshot_ring`. Supports only `capture_kind::regs_and_stack` (no kernel callchain). Only one backend may be active per process.
- **Public API:**
  - `struct config { freq_hz, signal_no, stack_snapshot_bytes, callchain_depth, ring_capacity, start_delay_ns, bool cpu_time = false, clock_kind clock = monotonic }`
  - `explicit backend(config)`, `~backend() override`; non-copyable/non-movable
  - `static constexpr bool supports(unwind::capture_kind) noexcept` (true only for `regs_and_stack`)
  - all `i_trigger` overrides; `tracked_tids()` reads a relaxed atomic counter

###### `sampling/triggers/signal/backend.cpp`
- **Location:** `sampling/triggers/signal/`, namespace `...::triggers::signal`.
- **Size:** 831 lines, .cpp
- **Purpose:** Implements the signal backend and the async-signal-safe capture handler.
- **Contents:** Implements the API of `backend.hpp`. Internal (anonymous namespace) helpers: globals `g_ring_table`, `g_wake_fd`, `g_stack_bytes`, `g_sample_clock`; TLS `t_cached_tid`, `t_altstack`; `pack_regs`, `install_altstack_if_missing`, `errno_guard`, `capture_sample`, `signal_handler`, `make_capture_action`, `claim_signal` / `release_signal` / `flush_pending_signal`, `thread_cpu_clock`, `thread_has_exited`, `adopt_altstack` / `reap_altstacks` (orphan altstacks), `reset_after_fork`.
- **Implementation:** The handler reads the thread's ring via `ring_table`, captures the timestamp, packs 18 registers from `ucontext_t` into the perf register order, copies the user stack with `process_vm_readv` into a preallocated slot, installs a 64 KiB mmap'd alternate stack on first delivery, and writes the wake eventfd when a ring reaches its wake depth; errno is saved/restored. `attach_tid` allocates the ring, publishes it and creates/arms the timer (CPU-time or wall clock); `detach_tid` retires the ring and defers freeing until no handler is in flight (`reclaim_retired`). `wait` polls the eventfd; `drain_into` pops snapshots, unwinds with the visited unwinder and forwards samples and lost counts to the pipeline. The ctor claims the signal via sigaction and fails with `EBUSY` if the host already has a handler; the dtor/`close_all` restore the previous disposition.
- **Public API:** Implements the API of `backend.hpp`. Throws `usage_error` if a second backend is active and `environment_error` for eventfd/sigaction/timer failures.

###### `sampling/triggers/signal/ring_table.hpp`
- **Location:** `sampling/triggers/signal/`, namespace `...::triggers::signal`.
- **Size:** 260 lines, header-only.
- **Purpose:** Lock-free tid -> `snapshot_ring*` table readable from a signal handler.
- **Contents:** class `ring_table` with 1024 slots (`tid`, `ring`, `busy` atomics), constants `k_slots`, `k_max_probe` (64), `k_npos`; multiplicative hash for `home_slot`.
- **Implementation:** Open-addressed linear probing with tombstones. The worker publishes/retires entries; the handler uses `enter`/`leave` to bump a per-slot busy counter so the worker can wait for `idle(slot)` before freeing a ring. Tombstones are reclaimed backwards when the next slot is empty. Memory orders are explicit (seq_cst on the retire/enter handshake).
- **Public API:**
  - `static constexpr size_t home_slot(pid_t) noexcept`
  - `bool publish(pid_t, snapshot_ring*) noexcept` (false when the probe window is full)
  - `size_t retire(pid_t) noexcept` (returns slot or `k_npos`)
  - `snapshot_ring* lookup(pid_t) const noexcept`
  - `snapshot_ring* enter(pid_t, size_t& slot_idx) noexcept`; `void leave(size_t) noexcept`
  - `bool idle(size_t) const noexcept`
  - `void clear() noexcept`; `size_t occupied_slots() const noexcept`

###### `sampling/triggers/signal/snapshot.hpp`
- **Location:** `sampling/triggers/signal/`, namespace `...::triggers::signal`.
- **Size:** 243 lines, header-only.
- **Purpose:** The sample record and the per-thread SPSC ring that carries it from handler to worker.
- **Contents:** struct `snapshot` (`ts_ns`, `stack_base_addr`, `stack_size`, `tid`, 18-entry `regs`, preallocated `stack` buffer); class `snapshot_ring`.
- **Implementation:** Power-of-two ring with atomic head/tail; the handler is the single producer (`acquire` then `commit`), the worker the single consumer (`pop` then `consume`). A full ring bumps a lost counter instead of blocking. `commit` returns the fill depth so the handler can wake the worker at `wake_depth()` (a quarter of the capacity). The ring also stores the handler thread's alternate-stack base for later reaping. All slots and stack buffers are allocated up front, so the handler never allocates.
- **Public API:**
  - `snapshot_ring(size_t capacity, uint32_t stack_capacity)`; non-copyable/non-movable
  - `uint32_t stack_capacity() const noexcept`
  - `snapshot* acquire() noexcept` (nullptr + lost++ when full)
  - `size_t commit() noexcept`; `size_t wake_depth() const noexcept`
  - `const snapshot* pop() noexcept`; `void consume() noexcept`
  - `void set_altstack(void*) noexcept`; `void* altstack() const noexcept`
  - `uint64_t lost() const noexcept`

##### sampling/unwind
Folder `sampling/unwind/` (namespace `cpu_snap::sampling::unwind`) holds the stack unwinders and the data they consume.
Three unwinders (kernel callchain, frame-pointer copy, DWARF via libdwfl) satisfy the `Unwinder` concept and are combined in the `unwinder` variant.
Common input is `machine_snapshot`; x86-64 perf register layout lives in `perf_regs.hpp`.

###### `sampling/unwind/machine_snapshot.hpp`
- **Location:** `sampling/unwind/`, namespace `...::unwind`.
- **Size:** 71 lines, header-only.
- **Purpose:** Backend-neutral input to an unwinder plus the `Unwinder` concept.
- **Contents:** enum `capture_kind`, struct `machine_snapshot`, concept `Unwinder`.
- **Implementation:** A snapshot carries either already-resolved frames (kernel callchain) or registers + a stack copy (regs_and_stack). The concept requires `unwind(snapshot, span<uint64_t>)` and a static `required_capture()` so triggers know what to collect.
- **Public API:**
  - `enum class capture_kind : uint8_t { kernel_callchain, regs_and_stack }`
  - `struct machine_snapshot { span<const uint64_t> ready_frames; const uint64_t* regs = nullptr; uint64_t stack_addr = 0; span<const byte> stack; }`
  - `concept Unwinder`

###### `sampling/unwind/perf_regs.hpp`
- **Location:** `sampling/unwind/`, namespace `...::unwind`.
- **Size:** 52 lines, header-only.
- **Purpose:** Fixes the x86-64 perf user-register set and slot indices.
- **Contents:** Register mask, count, slot helper and constants, static_asserts.
- **Implementation:** The mask covers 18 registers (AX..R15, IP, FLAGS, SP); the slot of a register is the popcount of mask bits below it, matching perf's dump order. The signal backend packs `ucontext` into the same layout, so both backends feed identical register arrays. static_asserts guard against kernel ABI changes.
- **Public API:**
  - `k_perf_dwarf_regs_mask_x86_64`, `k_perf_dwarf_n_regs_x86_64` (18)
  - `constexpr uint32_t perf_slot_of(unsigned reg) noexcept`
  - `k_perf_slot_rbp` (6), `k_perf_slot_rip` (8), `k_perf_slot_rsp` (7)

###### `sampling/unwind/unwinder.hpp`
- **Location:** `sampling/unwind/`, namespace `...::unwind`.
- **Size:** 33 lines, header-only.
- **Purpose:** Defines the runtime-selected unwinder type.
- **Contents:** One alias and static_asserts.
- **Implementation:** `std::variant` of the three unwinders; triggers `std::visit` it once per drain batch, keeping per-sample calls direct.
- **Public API:**
  - `using unwinder = std::variant<kernel_callchain_unwinder, fp_copy_unwinder, dwarf_unwinder>`

###### `sampling/unwind/kernel_callchain_unwinder.hpp`
- **Location:** `sampling/unwind/`, namespace `...::unwind`.
- **Size:** 66 lines, header-only.
- **Purpose:** Unwinder for perf kernel-provided callchains.
- **Contents:** class `kernel_callchain_unwinder`.
- **Implementation:** Copies `ready_frames` into the output (bounded by output size) and subtracts 1 from every frame except the leaf so return addresses point at the call site for symbolization. Trivial and stateless.
- **Public API:**
  - `static constexpr capture_kind required_capture() noexcept` (`kernel_callchain`)
  - `size_t unwind(const machine_snapshot&, span<uint64_t> out) const noexcept`

###### `sampling/unwind/fp_copy_unwinder.hpp`
- **Location:** `sampling/unwind/`, namespace `...::unwind`.
- **Size:** 64 lines, header-only declaration.
- **Purpose:** Declares the frame-pointer unwinder working on a copied stack.
- **Contents:** class `fp_copy_unwinder`.
- **Implementation:** Stateless; used with the signal backend (and perf regs+stack) when no DWARF is wanted.
- **Public API:**
  - `static constexpr capture_kind required_capture() noexcept` (`regs_and_stack`)
  - `size_t unwind(const machine_snapshot&, span<uint64_t> out) const noexcept`

###### `sampling/unwind/fp_copy_unwinder.cpp`
- **Location:** `sampling/unwind/`, namespace `...::unwind`.
- **Size:** 79 lines, .cpp
- **Purpose:** Implements the rbp-chain walk over the snapshot's stack copy.
- **Contents:** Implements the API of `fp_copy_unwinder.hpp`; internal constant `k_frame_link_bytes` (16).
- **Implementation:** Emits the leaf RIP unadjusted, then follows saved-rbp/return-address pairs inside the copied stack, storing `return_addr - 1`. Stops on out-of-range frames, zero return address, non-increasing saved rbp (stack grows down) or full output. Needs no allocation or libraries, so it is cheap.
- **Public API:** Implements the API of `fp_copy_unwinder.hpp`.

###### `sampling/unwind/dwarf_unwinder.hpp`
- **Location:** `sampling/unwind/`, namespace `...::unwind`.
- **Size:** 169 lines, header-only declaration.
- **Purpose:** Public interface of the libdwfl-based DWARF unwinder, hiding elfutils behind a pimpl.
- **Contents:** struct `dwarf_snapshot`, helper `snapshot_word_offset`, class `dwarf_unwinder` with `class impl`.
- **Implementation:** The legacy `dwarf_snapshot` (regs, stack base/data/size) is converted from `machine_snapshot` in the generic `unwind` overload. `snapshot_word_offset` does overflow-safe bounds checking of an 8-byte word inside the stack copy. Not thread-safe by itself; one instance per sampler worker.
- **Public API:**
  - `struct dwarf_snapshot { const uint64_t* regs; uint64_t stack_addr; const uint8_t* stack_data; uint64_t stack_size; }`
  - `constexpr optional<uint64_t> snapshot_word_offset(const dwarf_snapshot&, uint64_t addr) noexcept`
  - `dwarf_unwinder() noexcept`, `~dwarf_unwinder()`; non-copyable/non-movable
  - `uint32_t unwind(const dwarf_snapshot&, uint64_t* out, uint32_t max_frames) noexcept`
  - `size_t unwind(const machine_snapshot&, span<uint64_t> out) noexcept`
  - `uint64_t live_stack_read_samples() const noexcept`
  - `static constexpr capture_kind required_capture() noexcept` (`regs_and_stack`)

###### `sampling/unwind/dwarf_unwinder.cpp`
- **Location:** `sampling/unwind/`, namespace `...::unwind`.
- **Size:** 596 lines, .cpp
- **Purpose:** libdwfl implementation of CFI-based stack unwinding over a stack snapshot.
- **Contents:** Implements the API of `dwarf_unwinder.hpp`. Internal: `detail::reg_pair`, `dwarf_unwinder::impl` (Dwfl handle and callbacks, `unwind_frames`, `walk`, `is_return_instruction`, `unwind_from_return`, `has_cfi`, `refresh_modules`, `thread_state`, stack/memory read callbacks, live-stack read counter).
- **Implementation:** Feeds libdwfl a callback set that reads memory from the snapshot (falling back to `process_vm_readv` on the live process, counted in `live_stack_read_samples`) and registers from the perf-ordered array. Special cases: a sample taken on a `ret` instruction is unwound from the post-return state (clang CFI quirk), and code without CFI (PLT, ld.so, asm) continues from the return address on top of the stack. The Dwfl module list is rebuilt only when the loader generation changes. Libdw/elfutils headers stay confined to this file.
- **Public API:** Implements the API of `dwarf_unwinder.hpp`.

## processing

### What it is for
Offline post-processing of `core::raw_sample` data after `sampler::stop()`. It turns raw instruction pointers into function names (optionally inline chains and file:line), filters unwanted frames, and shapes per-thread sample vectors into nested flame-graph style slices that any trace backend (e.g. Perfetto) can write. It never runs on the recording hot path.

### Responsibilities
- Read ELF symbol tables and PT_LOAD segments; translate runtime IPs to link-time addresses (`proc_maps`, `elf_symbols.hpp`).
- Resolve IPs to names via batched `addr2line`, corrected by ELF symbol sizes (`symbolizer`).
- Provide frame keep/drop predicates: main-exe-only, library exclusion, stdlib exclusion (`frame_filters.hpp`).
- Symbolize, filter and deliver samples to a sink; coalesce IPs by name (`sample_postprocessor`, `coalesce_frames_by_name`).
- Diff consecutive call stacks into `emitted_slice` records with depth and start/end timestamps (`slice_emitter`).

### Explicitly not responsible for
- Capturing samples, unwinding, or any signal/perf handling (sampling).
- Defining `raw_sample` / `module_map` (core).
- Writing the trace file format; sinks (examples' Perfetto writer) consume `emitted_slice`.
- Thread safety, async-signal-safety, or low-overhead guarantees.

### Main building blocks
| unit | role |
|---|---|
| `elf_symbols.hpp` (`open_elf64`, `read_elf_symbol_table`, `find_symbol`) | Sorted FUNC/IFUNC/NOTYPE symbols of an x86_64 ELF; binary search with size check |
| `proc_maps` / `vm_mapping` | Snapshot of executable mappings from /proc/self/maps, IP -> ELF vaddr, per-path cached symbols and load segments |
| `symbolizer` / `symbolizer_config` | IP -> name/inline chain/file:line: mapping lookup, per-IP caches, `symbol_identity`, hex fallbacks; delegates the addr2line process to `addr2line_client` |
| `addr2line_client` / `addr2line_chain` | Stateless `addr2line -ifCe` runner: quoting, command building, `popen`, chunking, parsing of inline chains (parser takes a line-source callable, testable without a process) |
| `symbol_name` | `clean` (drops ` [clone ...]` and `(anonymous namespace)::`) and `demangle` |
| `frame_filters.hpp` | `make_main_exe_only_filter`, `make_library_exclude_filter` (proc_maps and module_map variants), `is_stdlib_label`, `make_stdlib_exclude_filter` |
| `sample_postprocessor` | Per-IP name cache, filter chain, sink delivery of `processed_sample`; `coalesce_frames_by_name` |
| `slice_emitter` (hpp + cpp) | Stack diffing into nested `emitted_slice`; merged, name-based inline, or unmerged modes |

### How it works
1. Symbolization: `symbolizer::operator()(ip)` hits the cache, else `preload` maps IPs to modules (`proc_maps::find` or `core::module_map` at a timestamp) and groups them per binary.
2. Per binary, `addr2line_client::resolve` runs `addr2line` in chunks of at most `k_max_ips_per_invocation` (1000) and returns raw function names and the first frame's file:line per IP; `symbolizer::store_chain` cleans them, corrects them against the ELF symbol table and caches the result. `symbols_for` + `find_symbol` reject addresses outside any symbol (fallback `binary+0xoff`); vdso/vsyscall get hex names.
3. `sample_postprocessor::process_all` / `process` iterates samples; per frame it resolves a name (resolver exceptions give an empty name), applies the library filter (lazy main-exe-only) and `frame_predicate`, then calls the sink with a span over a scratch buffer.
4. `slice_emitter::process` per thread: sort by timestamp (copy only when needed), `filter_library_frames`, then diff consecutive stacks. Persisting frames extend one slice; others are closed/opened. Equality is by IP, by `symbol_identity` (`resolves_to_same_symbol`), or by name when `inline_resolver` is set.
5. Open slices are closed using `sample_period_ns`; `merge_samples=false` emits each sample's stack independently. `process` returns the slice count.

### Dependencies and users
- Depends on: core only (`raw_sample`, `module_map`, `format_hex`); CMake target `processing` links `cpu_snap::core`; no libdw, no sampling. Only `slice_emitter.cpp` is compiled, the rest is header-only.
- Users: examples (`multi_thread_callstack_demo`, `hip_matmul_demo`, `inline_frames_demo`, `autoload_sampler`, `common/perfetto_proto_writer.hpp`, `common/symbol_preload.hpp`, ...).
- Tests: `tests/processing/*` (elf_symbols, symbolizer_format, addr2line_client, symbol_name, frame_filters, sample_postprocessor, module_map_symbolization, stripped_symbol_size, lld_pt_load_mismatch) plus sampling tests (streaming perfetto equivalence, signal_sampler, sampler_streaming, dl_module_watcher) and `test_coverage_extras`.

### Invariants and constraints
- Offline and single-threaded; uses file I/O, allocation and `popen`, so never call from a signal handler or the hot path.
- `proc_maps` and `symbolizer` caches are mutable and not thread-safe.
- `string_view`s in `processed_frame` live until `clear_cache()`/destruction; `emitted_slice::name` is valid only during the sink call. A `proc_maps&` captured by a filter must outlive it.
- Error policy: user resolvers are caught (hex or empty name fallback), failures degrade to hex names; no exceptions are intended to reach callers.
- Unmapped IPs are kept by filters. `emitted_slice` ties at equal timestamps must be ordered by depth. ELF parsing is x86_64 little-endian 64-bit only.

### Known weak spots
- `symbolizer.hpp` (543 lines) still holds mapping lookup, the two caches and the ELF-based name correction in one header-only class; the addr2line process handling and name cleaning now live in `addr2line_client` and `symbol_name`.
- Symbolization depends on the external `addr2line` binary via `popen` and shell quoting.
- `proc_maps::find` is a linear scan; `slice_emitter.hpp` (458 lines) holds three emit variants (`emit_nested`, `emit_nested_inline`, `emit_nested_unmerged`).
- Almost everything is header-only templates, so only one .cpp is built into the library and changes recompile all consumers.

### Files

#### processing (src/cpu_snap/processing/)
Offline post-processing of captured core::raw_sample data, run after sampler::stop() and never on the hot recording path.
symbols/ turns instruction pointers into function names (ELF, /proc/self/maps, addr2line) and filters/coalesces frames.
timeline/ turns per-thread sample vectors into nested, flame-graph style slices for trace backends.

##### processing/symbols

###### `processing/symbols/elf_symbols.hpp`
- **Location:** processing/symbols/, namespace `cpu_snap::processing::symbols::detail` (internal helpers).
- **Size:** 265 lines, header-only
- **Purpose:** Reads FUNC-like symbols from an ELF's .symtab/.dynsym and finds the symbol covering a virtual address.
- **Contents:** `file_closer`, `elf64_file`, `open_elf64`, `elf_symbol`, `elf_symbol_table`, `find_symbol`, `add_symbols_from`, `read_elf_symbol_table`.
- **Implementation:** Opens the file with stdio, validates a 64-bit little-endian ELF header (x86_64 only), reads section headers, collects FUNC/GNU_IFUNC/NOTYPE sized defined symbols from SYMTAB and DYNSYM, and sorts them by `st_value`. `find_symbol` does a binary search and checks `vaddr < st_value + st_size`, so addresses outside every symbol are not misnamed after the preceding symbol (relevant for stripped libs with only .dynsym). Also records whether `.debug_info` exists. Not async-signal-safe (file I/O, allocation); offline use only.
- **Public API:**
  - `struct file_closer { void operator()(FILE*) const noexcept; }`
  - `struct elf64_file { std::unique_ptr<FILE, file_closer> file; Elf64_Ehdr ehdr; }`
  - `std::optional<elf64_file> open_elf64(const std::string& path)`
  - `struct elf_symbol { uint64_t st_value, st_size; std::string_view name; }`
  - `struct elf_symbol_table { std::vector<elf_symbol> symbols; bool has_debug_info; std::vector<char> symtab_strs, dynsym_strs; }`
  - `constexpr std::optional<elf_symbol> find_symbol(uint64_t vaddr, std::span<const elf_symbol> symbols_by_value) noexcept`
  - `void add_symbols_from(const Elf64_Shdr&, const std::vector<Elf64_Shdr>&, const ReadSectionFn&, std::vector<elf_symbol>&, std::vector<char>&)` (template)
  - `elf_symbol_table read_elf_symbol_table(const std::string& path)`

###### `processing/symbols/frame_filters.hpp`
- **Location:** processing/symbols/, namespace `cpu_snap::processing::symbols`.
- **Size:** 253 lines, header-only
- **Purpose:** Factory functions for per-frame keep/drop predicates and a stdlib-frame classifier.
- **Contents:** main-exe-only and library-exclude filters (each in a live `proc_maps` variant and a `core::module_map` variant), `is_stdlib_label`, `make_stdlib_exclude_filter`.
- **Implementation:** Every filter returns `std::function<bool(uint64_t ip, std::string_view name)>` where true means keep; unmapped IPs are kept. The proc_maps variants capture `maps` by reference (it must outlive the filter); the module_map variants resolve the IP at a fixed timestamp. `is_stdlib_label` strips the ` [binary]` suffix, scans the demangled name (tracking template depth and `operator`) and treats `std::`/`__gnu_cxx::` functions and `[libstdc++` labels as stdlib.
- **Public API:**
  - `std::function<bool(uint64_t, std::string_view)> make_main_exe_only_filter(proc_maps& maps)`
  - `... make_main_exe_only_filter(const core::module_map& modules, uint64_t timestamp_ns, std::string main_exe_path)`
  - `... make_library_exclude_filter(proc_maps& maps, std::vector<std::string> excluded_libraries)` (substring match on mapping path)
  - `... make_library_exclude_filter(const core::module_map& modules, uint64_t timestamp_ns, std::vector<std::string> excluded_libraries)`
  - `bool is_stdlib_label(std::string_view label)`
  - `template<Resolve> ... make_stdlib_exclude_filter(Resolve resolve)` (resolve: ip -> label)

###### `processing/symbols/proc_maps.hpp`
- **Location:** processing/symbols/, namespace `cpu_snap::processing::symbols` (+ `detail`).
- **Size:** 358 lines, header-only
- **Purpose:** Snapshot of the current process's executable mappings from /proc/self/maps, with IP-to-ELF-vaddr translation and cached per-file ELF data.
- **Contents:** `vm_mapping`, `proc_maps` class, `detail` constants (`k_vdso_path`, `k_vsyscall_path`, buffer sizes), `elf_load_segment`, `find_covering_vaddr`, `file_offset_to_vaddr`, `read_elf_load_segments`.
- **Implementation:** `reload()` parses /proc/self/maps with fgets/sscanf, keeps only executable mappings with a path, and separates [vdso]/[vsyscall] into a pseudo list (other `[...]` entries are skipped). `ip_to_elf_vaddr` converts IP to file offset then maps it through PT_LOAD segments (lazily read and cached per path), so PIE/shared objects get correct link-time addresses. `symbols_for` lazily caches `elf_symbol_table` per path (mutable caches, not thread-safe). Lookup is a linear scan.
- **Public API:**
  - `struct vm_mapping { uint64_t start, end, file_offset; bool is_main_exe; std::string path; }`
  - `class proc_maps`: `proc_maps()` (calls reload), `void reload()`, `const vm_mapping* find(uint64_t ip) const noexcept`, `const vm_mapping* find_pseudo(uint64_t ip) const noexcept`, `uint64_t ip_to_elf_vaddr(uint64_t ip, const vm_mapping&) const`, `const std::vector<vm_mapping>& mappings() const noexcept`, `const std::string& main_exe() const noexcept`, `const detail::elf_symbol_table& symbols_for(const std::string& path) const`
  - detail: `struct elf_load_segment`, `std::optional<uint64_t> find_covering_vaddr(uint64_t, std::span<const elf_load_segment>)`, `uint64_t file_offset_to_vaddr(...)`, `std::vector<elf_load_segment> read_elf_load_segments(const std::string&)`

###### `processing/symbols/sample_postprocessor.hpp`
- **Location:** processing/symbols/, namespace `cpu_snap::processing::symbols`.
- **Size:** 377 lines, header-only
- **Purpose:** Offline pipeline that symbolizes, filters and delivers raw samples to a sink, plus an IP-coalescing helper.
- **Contents:** `processed_frame`, `processed_sample`, `sample_postprocessor_config`, `sample_postprocessor`, `coalesce_frames_by_name`.
- **Implementation:** For each frame it resolves the name through a per-IP cache (a throwing resolver yields an empty name), applies the optional library filter (main-exe-only, built lazily from proc_maps or the config's main exe path) and the user `frame_predicate`, and calls the sink with a span over an internal scratch buffer. string_views in `processed_frame` point into the cache and are valid until the postprocessor is destroyed or `clear_cache()`. Single-threaded. `coalesce_frames_by_name` rewrites frame IPs in place so all IPs resolving to the same name share one canonical IP.
- **Public API:**
  - `struct processed_frame { uint64_t ip; std::string_view resolved; }`
  - `struct processed_sample { pid_t tid; uint64_t timestamp_ns; uint64_t ip; std::span<const processed_frame> frames; }`
  - `struct sample_postprocessor_config { std::function<std::string(uint64_t)> resolver; std::function<bool(uint64_t, std::string_view)> frame_predicate; bool drop_library_frames; const core::module_map* library_filter_modules; uint64_t library_filter_timestamp_ns; std::string main_exe_path; }`
  - `class sample_postprocessor`: `explicit sample_postprocessor(sample_postprocessor_config)`, movable, non-copyable; `template<Sink> void process(std::span<const core::raw_sample>, Sink&&)`; `template<Sink> void process_all(const std::unordered_map<pid_t, std::vector<core::raw_sample>>&, Sink&&)`; `void clear_cache() noexcept`; `size_t delivered_samples() const noexcept`; `size_t dropped_frames() const noexcept`; `size_t unique_symbols() const noexcept`
  - `template<Resolver> void coalesce_frames_by_name(std::unordered_map<pid_t, std::vector<core::raw_sample>>&, Resolver&&)`

###### `processing/symbols/symbolizer.hpp`
- **Location:** processing/symbols/, namespace `cpu_snap::processing::symbols`.
- **Size:** 543 lines, header-only
- **Purpose:** Resolves IPs to function names (and optional inline chains and file:line) by batching calls to the `addr2line` tool, corrected by ELF symbol sizes.
- **Contents:** `symbolizer_config`, `symbolizer` class; private helpers for mapping lookup, `correct_symbol_name`, `store_chain`, `basename_of`, `is_pseudo_module`, hex fallbacks.
- **Implementation:** IPs are mapped to a module (live `proc_maps`, or a `core::module_map` at a timestamp), grouped per binary, and resolved through `addr2line_client` (`addr2line -ifCe` in chunks of at most 1000 addresses); results are cached per IP. The ELF symbol table is used to reject addresses outside any symbol (falls back to `binary+0xoff` hex) and to demangle the ELF name. vdso/vsyscall get hex fallbacks. `symbol_identity` hashes (module path, symbol start) so different call sites in one function compare equal. Spawns external processes; offline only, not thread-safe (`resolve`/`file_line` are const reads of the cache).
- **Public API:**
  - `struct symbolizer_config { bool include_binary_name = true; bool inline_frames = false; }`
  - `class symbolizer`: `symbolizer()`, `explicit symbolizer(symbolizer_config) noexcept`, `explicit symbolizer(const core::module_map&, symbolizer_config = {}) noexcept`
  - `void rescan_maps()` (reloads proc_maps; no-op when a module_map is used)
  - `void preload(const std::vector<uint64_t>& ips)`; `void preload_at(const std::vector<std::pair<uint64_t, uint64_t>>& ip_timestamps)`
  - `std::string_view resolve(uint64_t ip) const noexcept` (cache only); `std::vector<std::string> resolve_inline(uint64_t ip) const`; `std::string_view file_line(uint64_t ip) const noexcept`
  - `std::string operator()(uint64_t ip)` (resolve, preloading on miss; hex on failure)
  - `std::optional<uint64_t> symbol_identity(uint64_t ip) const` and `(uint64_t ip, uint64_t timestamp_ns) const`
  - `std::string format_with_binary(std::string_view name, std::string_view base) const` ("name [base]")

###### `processing/symbols/addr2line_client.hpp`
- **Location:** processing/symbols/, namespace `cpu_snap::processing::symbols`.
- **Size:** 225 lines, header-only
- **Purpose:** Runs `addr2line` for a batch of addresses in one binary and parses its output into inline chains.
- **Contents:** `addr2line_chain`, class `addr2line_client` (static functions only), constants `detail::k_addr2line_line_buf` and `detail::k_max_ips_per_invocation`.
- **Implementation:** Builds a shell-quoted `addr2line -ifCe <path> <addr> [0xffff... <addr> ...]` command with a sentinel address between IPs, runs it with `popen` in chunks of at most 1000 addresses, and parses the two-line (function, file:line) pairs into chains. Unresolved IPs yield an empty chain; a failed `popen` yields `std::nullopt` for that chunk. No cache and no knowledge of mappings; the chain parser takes a line-source callable so tests need no process. Spawns external processes; offline only.
- **Public API:**
  - `struct addr2line_chain { std::vector<std::string> functions; std::string file_line; }` (raw names, innermost first; file:line of the first frame)
  - `using address_pair = std::pair<uint64_t, uint64_t>` (ip, ELF vaddr)
  - `static std::vector<std::optional<addr2line_chain>> resolve(std::string_view path, std::span<const address_pair>)`
  - `static std::string shell_single_quote(std::string_view)`; `static std::string build_command(std::string_view path, std::span<const address_pair>)`
  - `static bool is_sentinel_function(std::string_view) noexcept`; `static std::string normalize_file_line(std::string)`
  - `template<LineSource> static addr2line_chain read_chain(LineSource&&)`

###### `processing/symbols/symbol_name.hpp`
- **Location:** processing/symbols/, namespace `cpu_snap::processing::symbols`.
- **Size:** 61 lines, header-only
- **Purpose:** Tidies a symbol name for display.
- **Contents:** class `symbol_name` (static functions only).
- **Implementation:** `clean` strips the ` [clone ...]` suffix and every `(anonymous namespace)::`; `demangle` wraps `abi::__cxa_demangle` and returns the input unchanged on failure.
- **Public API:**
  - `static void clean(std::string&)`
  - `static std::string demangle(std::string_view mangled)`

##### processing/timeline

###### `processing/timeline/slice_emitter.hpp`
- **Location:** processing/timeline/, namespace `cpu_snap::processing::timeline`.
- **Size:** 458 lines, header-only part (templates; non-template members in slice_emitter.cpp)
- **Purpose:** Converts per-thread raw sample vectors into a stream of nested `emitted_slice` records for trace backends.
- **Contents:** `k_default_sample_period_ns`, `emitted_slice`, `slice_emitter_config`, `slice_emitter` with templated `process`, `emit_nested`, `emit_nested_inline`, `emit_nested_unmerged`.
- **Implementation:** `process` sorts each thread's samples by timestamp (copying only if unsorted or filtering is needed), optionally drops library frames, then diffs consecutive call stacks: frames that persist across samples become one slice, otherwise slices are closed/opened. Diffing is IP-based (with optional `symbol_identity` to treat different IPs in one function as the same frame) or name-based when `inline_resolver` is set; `merge_samples=false` emits each sample's full stack independently. Open frames are closed using `sample_period_ns`. Backend-agnostic, single-threaded, run after sampler::stop(); sink-received `name` views are valid only during the call.
- **Public API:**
  - `inline constexpr uint64_t k_default_sample_period_ns = 1'000'000`
  - `struct emitted_slice { pid_t tid; std::string_view name; uint64_t start_ts, end_ts; uint32_t depth; }` (ties at equal timestamps must be broken by depth)
  - `struct slice_emitter_config { bool merge_samples = true; uint64_t sample_period_ns; std::function<std::string(uint64_t)> resolver; std::function<std::vector<std::string>(uint64_t)> inline_resolver; std::function<std::optional<uint64_t>(uint64_t)> symbol_identity; std::function<bool(uint64_t, std::string_view)> library_frame_filter; }`
  - `class slice_emitter`: `explicit slice_emitter(slice_emitter_config)`; `template<Sink> size_t process(const std::unordered_map<pid_t, std::vector<core::raw_sample>>&, Sink&&)` returns number of slices emitted.

###### `processing/timeline/slice_emitter.cpp`
- **Location:** processing/timeline/, namespace `cpu_snap::processing::timeline`.
- **Size:** 85 lines, .cpp
- **Purpose:** Non-template member functions of `slice_emitter`.
- **Contents:** Constructor, `resolve_name`, `resolves_to_same_symbol`, `filter_library_frames`, anonymous-namespace `default_hex_name`.
- **Implementation:** Implements the API of slice_emitter.hpp. The constructor installs a hex-name resolver (`core::format_hex`) when none is given. `resolve_name` catches exceptions from the user resolver and falls back to hex. `resolves_to_same_symbol` is true only if both IPs yield non-null, equal `symbol_identity` keys. `filter_library_frames` compacts a sample's frames in place with `library_frame_filter` and resets `ip` to the new top frame.
- **Public API:** Implements the API of slice_emitter.hpp (private helpers: `filter_library_frames(core::raw_sample&) const`, `resolve_name(uint64_t) const`, `resolves_to_same_symbol(uint64_t, uint64_t) const`).

## tools and examples

The three library components are consumed from the outside only here (and in tests). The demos and the autoload library link `cpu_snap::sampling`, `cpu_snap::core` and `cpu_snap::processing` directly and play the whole pipeline: `sampling::sampler` records wire-format bytes into a `core::memory_sink`, `core::sample_reconstructor` decodes them, `processing` symbolizes and shapes slices, and demo-local glue (`examples/common/perfetto_proto_writer.hpp`) writes the Perfetto file. `cpu-snap-run` does not link any library: it only builds an environment and execs the target with the autoload library preloaded.

### What it is for
Shows how to use the library and gives users a no-code way to sample any program. `libcallstack_autoload.so` samples a process from its constructor to its destructor and writes a trace; `cpu-snap-run` is the command-line front end that preloads it and translates options into `CALLSTACK_*` variables. The demos are small, runnable usage examples of the public API.

### Responsibilities
- Parse `cpu-snap-run` options into a `launch_plan` (environment, command, preload path, error/help) as pure logic.
- Set `LD_PRELOAD` plus `CALLSTACK_*` and `execvp` the target, so pid, exit code and signals are preserved.
- In the preload library: configure `sampler::config` from `CALLSTACK_*`, start before `main`, stop after it, post-process and write one trace per process (`%p`, `%e` expansion in the path).
- Write no trace for processes with no samples; honour `CALLSTACK_ONLY_PID` (`--no-children`).
- Demonstrate API usage patterns (streaming, two-phase, backend comparison, inline frames, HIP).

### Explicitly not responsible for
- Sampling, unwinding, wire format (sampling/core) and symbolization, slice shaping (processing); only called from here.
- Production Perfetto writing: `perfetto_proto_writer` lives in `examples/common`, not in a library component.
- Attaching to an already running process.

### Main building blocks
| unit | role |
|---|---|
| `tools/cpu_snap_run/launch_plan.hpp` (`make_launch_plan`, `detail::parse_options`/`apply_option`, `detail::build_environment`, `usage`) | header-only, no I/O: argv to `launch_plan`; derives `CALLSTACK_PAGES` from `--stack-bytes` via `pages_for_snapshot` |
| `tools/cpu_snap_run/main.cpp` | reads `CPU_SNAP_PRELOAD`/`LD_PRELOAD`, prints help/errors (exit 2), checks the library exists, `setenv`, `execvp` (127 on failure) |
| `examples/autoload/autoload_sampler.cpp` | `callstack_autoload_init` (constructor) and `callstack_autoload_fini` (destructor); env parsing helpers `env_u32/env_str/env_bool` |
| `examples/autoload/sample_target.cpp` | workload used by the `run_autoload_demo` target |
| `examples/common/` | demo glue: `perfetto_proto_writer`, `pb_buf`/`pb_fields`, `backend_select`, `symbol_preload`, `name_filter`, `workload` (targets `perfetto_writer_demo`, `cpu_snap_workload`) |
| `examples/*_demo.cpp` | see table below |

| demo | shows |
|---|---|
| `single_thread_callstack_demo` | minimal sampler + memory_sink + reconstructor + trace |
| `multi_thread_callstack_demo` | 4 threads streamed to a bounded sink, one trace |
| `multi_thread_sample_demo` | two-phase: record raw, then `sample_postprocessor` offline |
| `deep_chain_callstack_demo` | 8-frame chain, self vs inclusive time |
| `inline_frames_demo` | symbolizer inline-frame expansion (`resolve_inline`) |
| `backend_comparison_demo` | same workload under perf and signal, comparison table |
| `hip_matmul_demo` | host stacks through HIP runtime (opt-in `CPU_SNAP_BUILD_HIP_EXAMPLES`) |

### How it works
1. `main` calls `make_launch_plan(args, getpid(), default_preload, existing_ld_preload)`; default preload is `CPU_SNAP_PRELOAD` or the compile-time `CPU_SNAP_RUN_DEFAULT_PRELOAD`.
2. `parse_options` loops `apply_option` until `--` or the first non-option word; the rest is the command. Errors are returned in `plan.error`.
3. `build_environment` emits `LD_PRELOAD` (ours first, existing appended), `CALLSTACK_FREQ_HZ/BACKEND/UNWIND/STACK_BYTES/DEPTH/TRACE/MERGE/FILTER_STDLIB`, optionally `CALLSTACK_PAGES` and `CALLSTACK_ONLY_PID`.
4. `main` sets the variables and `execvp`s the target; the dynamic loader runs `callstack_autoload_init` in every process that loads the library (children inherit the environment).
5. Init: skip if `CALLSTACK_ONLY_PID` differs; build a `memory_sink` and `sampler::config` from the environment; `make_unique<sampler>` + `start()`; on exception log to stderr and continue without sampling.
6. Fini: snapshot thread names, `stop()`, `unsetenv("LD_PRELOAD")` (so addr2line helpers do not re-enter), `sample_reconstructor::feed`, return early if zero samples; else `symbolizer` + `sample_postprocessor` + `coalesce_frames_by_name` + `slice_emitter` stream into `perfetto_proto_writer`, `finalize()`.

### Dependencies and users
- `launch_plan.hpp`: standard library only. `cpu-snap-run` target depends on `callstack_autoload` (tools/CMakeLists.txt; skipped if that target is missing).
- `callstack_autoload`: `cpu_snap::sampling`, `core`, `processing`, `perfetto_writer_demo`.
- Users: `tests/tools/test_launch_plan.cpp` (target `tools_tests`, header only); `run_autoload_demo` custom target; docs/usage.md and examples/README.md describe usage.

### Invariants and constraints
- `launch_plan.hpp` stays pure (no I/O, no env access) so it is unit testable; `main.cpp` owns all side effects.
- The preload code runs in constructor/destructor context, single-threaded env reads (`NOLINT(concurrency-mt-unsafe)`); no exception may escape `callstack_autoload_init`, errors go to stderr.
- Destruction order: `g_recorder` must be reset before `g_sink` (recorder dtor drains sink callbacks); done explicitly, not via static destructors.
- Library is built with `-fvisibility=hidden`, `-fno-omit-frame-pointer`.
- Launcher exit codes: 0 help, 2 usage/missing library, 127 exec failure.

### Known weak spots
- `autoload_sampler.cpp` (421 lines) mixes env parsing, sampler configuration and the whole post-processing/trace pipeline in two functions; the post-processing in `callstack_autoload_fini` largely duplicates what the demos do.
- Environment-variable defaults live in two places: `launch_plan.hpp` (`parsed_options`) and `autoload_sampler.cpp` constants differ (freq 1000 vs 200, unwind dwarf vs auto), so behaviour depends on which entry is used.
- `env_u32` uses `std::stoul` with a `static_cast`, so values above `uint32_t` silently truncate (documented in its comment).
- The only trace writer is demo code in `examples/common`, so the "tool" path depends on example sources.

## Appendix: project tree

```
src/cpu_snap/
  core/
    CMakeLists.txt  - Builds the `core` static library.
    chunk_framer.hpp  - Serializes records into framed chunks and passes each to a sink.
    format_hex.hpp  - Hex-formats an address.
    frame_limits.hpp  - Default and maximum callchain depth constants.
    memory_sink.hpp  - Bounded in-memory `sink_fn` target.
    module_map.cpp  - Implements `module_map`.
    module_map.hpp  - Time-aware lookup of loaded modules from MAPS_INIT/MAPS_DELTA records.
    raw_sample.hpp  - The decoded stack-sample record passed from sampling to post-processing.
    reader.cpp  - Implements the wire decoder behind `reader`.
    reader.hpp  - Public interface of the push-based wire-stream decoder.
    records.hpp  - Defines every on-wire record and framing structure of the v1 (major 2)...
    sample_reconstructor.cpp  - Implements `sample_reconstructor`.
    sample_reconstructor.hpp  - Adapter that turns a wire byte stream into per-tid `raw_sample` vector...
    self_exe.hpp  - Returns the path of the running executable.
    types.hpp  - Public callback aliases and the default chunk size of the wire pipeline.
  error.hpp  - Defines the exception hierarchy for every library-originated failure.
  processing/
    CMakeLists.txt
    symbols/
      addr2line_client.hpp  - Runs addr2line for a batch of addresses and parses its output into inline chains.
      elf_symbols.hpp  - Reads FUNC-like symbols from an ELF's .symtab/.dynsym and finds the sy...
      frame_filters.hpp  - Factory functions for per-frame keep/drop predicates and a stdlib-fram...
      proc_maps.hpp  - Snapshot of the current process's executable mappings from /proc/self/...
      sample_postprocessor.hpp  - Offline pipeline that symbolizes, filters and delivers raw samples to ...
      symbol_name.hpp  - Tidies a symbol name for display.
      symbolizer.hpp  - Resolves IPs to function names (and optional inline chains and file:li...
    timeline/
      slice_emitter.cpp  - Non-template member functions of `slice_emitter`.
      slice_emitter.hpp  - Converts per-thread raw sample vectors into a stream of nested `emitte...
  sampling/
    CMakeLists.txt
    backend_kind.hpp  - Selector for the capture mechanism the sampler builds.
    backend_selection.hpp  - Resolve backend_kind::automatic to a concrete backend by probing perf_...
    clock_kind.hpp  - Selector of the one clock all library timestamps use.
    dl_module_watcher.cpp  - Walk dl_iterate_phdr, read GNU build-ids and compute module deltas.
    dl_module_watcher.hpp  - Declare the class that tracks loaded shared objects and emits maps_ini...
    flat_tid_map.hpp  - Hot-path per-tid associative container faster than std::unordered_map.
    fork_registry.cpp  - Implement the fork registry.
    fork_registry.hpp  - Process-wide list of objects that must clean up child-side state after...
    frame_trie.hpp  - Grow-only trie of callstack frames that is the core of wire compression.
    logger.hpp  - Injected log sink; the library never writes to stderr itself.
    perf_ring.hpp  - RAII owner of a perf_event_open mmap ring plus a decoder-templated dra...
    sampler.cpp  - Implement the sampler lifecycle, worker loop and construction wiring.
    sampler.hpp  - Declare the online CPU stack recorder for all threads of the current p...
    sampler_capabilities.hpp  - Probe which perf_event_open features the kernel and CPU accept.
    sampler_plan.cpp  - Implement config validation, backend/unwind resolution and config conv...
    sampler_plan.hpp  - Declare the stateless resolution of sampler::config into the effective...
    streaming_pipeline.cpp  - Implement the pipeline's record emission, error absorption and resync.
    streaming_pipeline.hpp  - Declare the facade that turns per-sample frame arrays into the wire by...
    temporal_diff_filter.hpp  - Per-tid running stack tracker that interns only the changed suffix int...
    thread_comm.hpp  - Thread and process identity helpers backed by /proc/self.
    thread_tracker.cpp  - Implement thread listing, initial attach and rescan reconciliation.
    thread_tracker.hpp  - Declare the per-thread attach bookkeeping of the sampler.
    time.hpp  - Clock helpers shared across the pipeline.
    triggers/
      i_trigger.hpp  - Defines the abstract interface every capture mechanism implements.
      perf/
        backend.cpp  - Out-of-line bodies of the perf `backend`.
        backend.hpp  - Declares the perf-based `i_trigger` implementation wrapping `kernel_dr...
        kernel_drain.hpp  - Owns the per-thread perf_event fd + mmap ring lifecycle and the drain ...
        open_error.hpp  - Builds an actionable message when perf_event_open fails.
        perf_sample_decoder.hpp  - Decodes one PERF_RECORD_SAMPLE record and runs the unwinder on it.
        ring_sizing.hpp  - Constexpr helpers sizing the perf mmap ring and wakeup threshold.
      signal/
        backend.cpp  - Implements the signal backend and the async-signal-safe capture handler.
        backend.hpp  - Declares the signal-based `i_trigger`.
        ring_table.hpp  - Lock-free tid -> `snapshot_ring*` table readable from a signal handler.
        snapshot.hpp  - The sample record and the per-thread SPSC ring that carries it from ha...
    unwind/
      dwarf_unwinder.cpp  - libdwfl implementation of CFI-based stack unwinding over a stack snaps...
      dwarf_unwinder.hpp  - Public interface of the libdwfl-based DWARF unwinder, hiding elfutils ...
      fp_copy_unwinder.cpp  - Implements the rbp-chain walk over the snapshot's stack copy.
      fp_copy_unwinder.hpp  - Declares the frame-pointer unwinder working on a copied stack.
      kernel_callchain_unwinder.hpp  - Unwinder for perf kernel-provided callchains.
      machine_snapshot.hpp  - Backend-neutral input to an unwinder plus the `Unwinder` concept.
      perf_regs.hpp  - Fixes the x86-64 perf user-register set and slot indices.
      unwinder.hpp  - Defines the runtime-selected unwinder type.
    unwind_mode.hpp  - Selector of the stack-walking strategy.
```
