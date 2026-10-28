# Changelog

## Unreleased

### Added

- `cpu-snap-run [options] -- <command> [args...]` (`tools/`): runs any dynamically
  linked program with the sampler preloaded and writes one Perfetto trace per
  process of the tree. Options for frequency, backend, unwind mode, snapshot and
  ring size, depth, output path (`%p`, `%e`), `--no-merge`, `--filter-stdlib`
  and `--no-children`.
- `libcallstack_autoload.so`: new `CALLSTACK_BACKEND`, `CALLSTACK_UNWIND`,
  `CALLSTACK_STACK_BYTES`, `CALLSTACK_MERGE`, `CALLSTACK_FILTER_STDLIB` and
  `CALLSTACK_ONLY_PID`; `%p`/`%e` in `CALLSTACK_TRACE`; no trace file for a
  process without samples.
- `symbols::is_stdlib_label()` and `symbols::make_stdlib_exclude_filter()`.

- `slice_emitter_config::merge_samples` (default `true`). With `false`, nested
  mode no longer merges frames across consecutive samples: each sample emits its
  full stack as nested slices spanning `[its timestamp, the next sample's)`,
  so the timeline shows exactly what each sample captured. `hip_matmul_demo
  --no-merge` selects it.

### Changed

- `callchain_depth` is a runtime setting bounded only by `core::k_max_callchain_depth`
  (4096) instead of the compile-time `k_max_frames = 128`; the default is now 256
  (also `cpu-snap-run --depth` and `CALLSTACK_DEPTH`). `core::raw_sample::frames` is a
  `std::vector` holding exactly the captured frames and `n_frames` is gone (use
  `frames.size()`). `kernel_drain::process_sample` takes a `decode_scratch` instead of a
  depth. In frame-pointer mode on the perf backend the kernel still caps the chain at
  `kernel.perf_event_max_stack`.
- perf DWARF mode: `dwarf_stack_bytes` now defaults to 32768 (was 8192) so deep
  HIP/LLVM stacks are no longer truncated, and `kernel_pages_per_tid` defaults to
  0, which sizes the ring to hold at least 8 samples (128 pages for the default
  snapshot) instead of a fixed 8 pages that held about three. `wakeup_events` is
  derived from the ring capacity. The sampler constructor now throws
  `std::invalid_argument` when an explicit ring cannot hold one record, or when
  `dwarf_stack_bytes` is zero or above 65528; previously such rings silently
  produced no samples. Memory per sampled thread grows accordingly in DWARF mode.
- A denied `perf_event_open` now explains `perf_event_paranoid` and the signal
  backend instead of only reporting `Permission denied`.
- The autoload library no longer refuses to load into a process that was started
  by another preloaded process (`CALLSTACK_AUTOLOAD_ACTIVE` guard removed): every
  process of a tree samples itself and writes its own trace. Its default frame
  depth is now `core::k_default_max_frames` (128).
- Non-leaf stack frames are now call-site addresses (return address - 1) for
  every unwinder (`kernel_callchain_unwinder`, `fp_copy_unwinder`,
  `dwarf_unwinder`); the leaf frame stays exact. Symbolization and line lookup
  no longer attribute a call to the instruction after it (wrong inline scope,
  wrong line, or the next function after a `noreturn` call). Stored IPs of
  non-leaf frames differ by one from earlier captures.
- Streaming to a host-supplied sink is now the only capture path. The in-RAM
  sample map (`sampler::samples()`) is removed and `sampler::config` requires a
  sink. Use `core::memory_sink` (bounded, stops accepting when full) plus the
  reader -> `raw_sample` adapter (`core::sample_reconstructor`) to keep a
  capture in memory and analyze it after `stop()`. Memory use is now bounded
  and chosen by the host.

### Fixed

- perf backend: a thread that exited no longer makes the worker busy-spin on its
  hung-up ring; threads discovered while paused no longer sample until resume;
  `stop()` followed by `start()` works; a corrupt or truncated sample record no
  longer loops forever or reads out of bounds, and a throwing decoder no longer
  replays records.

### Known design limitations

- **Wire-format shotgun surgery for new record types.** Adding a new
  `record_type` enum value today requires editing 5 places:
  1. `core/include/cpu_snap/core/records.hpp` (enum + struct + variant + `type_of` tag table)
  2. `core/include/cpu_snap/core/chunk_framer.hpp` (`body_size_of(const X&)` overload)
  3. `core/include/cpu_snap/core/chunk_framer.hpp` (`encode_body(buf*, const X&)` overload)
  4. `core/src/reader.cpp` (`decode_<x>` free function)
  5. `core/src/reader.cpp` (`dispatch_record` switch arm)
  The comprehensive-cleanup audit (Item 28) considered factoring this
  into a `wire_traits<RecordT>` template specialization. The conclusion
  was to skip: (a) the per-record specialization for variable-size
  records (checkpoint, maps_init, maps_delta, pmc_sample) would exceed
  30 lines each because the field-by-field cursor encoding does not
  collapse cleanly into a single trait method, and (b) the dispatch
  table still needs a side-effect hook for `new_node` (which inserts
  into the trie before emitting), so the table replacement is only a
  partial unification of the switch. Given the low rate of new record
  types (the wire format is stable post-rename), the template
  machinery cost outweighs the shotgun-surgery cost. Re-evaluate if a
  3rd or 4th new record type lands.
