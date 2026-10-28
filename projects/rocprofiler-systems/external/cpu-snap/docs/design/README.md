# cpu-snap design documentation

The **deep design + rationale** docs: top-level architecture down to every
class, and *why* each piece exists. For task-oriented reference (how to build,
use, the wire-format byte layout, per-component API) see the parent
[`../`](../README.md) docs; this set explains the *design*.

## Read in order

1. [00-overview.md](./00-overview.md) - the map: three layers, end-to-end data
   flow, the streaming delivery path, the runtime-selected capture-backend seam, and the
   cross-cutting design principles. **Start here.**
2. [01-core.md](./01-core.md) - the `src/core/` data plane: records/types, chunk
   framing, the reader push-parser, and the OS substrate (perf_ring, proc_maps,
   dl_module_watcher, flat_tid_map, ...), with the why for each.
3. [02-capture-and-backends.md](./02-capture-and-backends.md) - the heart:
   the neutral `sampler` driver, the abstract `i_trigger` seam
   (`streaming_pipeline` as the sink), and both backends (`triggers::perf::backend` /
   `triggers::perf::kernel_drain` / `dwarf_unwinder` and `triggers::signal::backend` / `triggers::signal::snapshot_ring`),
   plus `raw_sample`, `unwind_mode`, `sampler_capabilities`, `frame_limits`.
4. [03-streaming-and-wire.md](./03-streaming-and-wire.md) - the streaming
   pipeline and trie-diff compression (`streaming_pipeline`,
   `temporal_diff_filter`, `frame_trie`) and how they ride the core wire format.
5. [04-offline-processing.md](./04-offline-processing.md) - turning samples into
   output: `symbolizer`, `sample_postprocessor`, `frame_filters`, and
   `slice_emitter` (the backend-neutral `emitted_slice` boundary).
6. [06-design-decisions.md](./06-design-decisions.md) - the cross-cutting
   rationale record: the de-thunked hot path, variant vs virtual interface,
   the wall-time-vs-CPU-time deviation, async-signal-safety rules, the
   dwfl-attach-once and RT-signal-leak fixes, and the audit conclusion.

## See also

- [../../planning/architecture-sampler-abstraction.md](../../planning/architecture-sampler-abstraction.md) -
  the original design plan for the sampler backend abstraction.
