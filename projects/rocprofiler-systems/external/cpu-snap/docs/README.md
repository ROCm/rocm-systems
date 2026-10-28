# cpu-snap documentation

cpu-snap is a C++20 library for low-overhead per-thread CPU callstack sampling,
producing Perfetto traces. This `docs/` tree is
the in-depth documentation; the top-level [`../README.md`](../README.md) is the
quickstart.

## Start here

| Doc | What it covers |
|---|---|
| [architecture.md](architecture.md) | The three components (core / sampling / processing), the capture -> reader -> shaping -> Perfetto data flow, and the two capture backends behind the `i_trigger` seam. |
| [components.md](components.md) | Per-component reference: what each of core / sampling / processing / tools is for, how it works, dependencies, invariants, plus a per-file reference (size, purpose, public API) and the project tree. |
| [usage.md](usage.md) | Task-oriented guide: in-memory vs file/socket sinks, selecting the perf vs signal backend, unwind modes, symbolization, and running the demos with `--backend`. |
| [build-and-install.md](build-and-install.md) | Dependencies, CMake options, presets, build/test, install + `find_package(cpu_snap)`, and `perf_event_paranoid` requirements. |
| [wire-format.md](wire-format.md) | The v1 streaming wire format: file header, 1 MiB chunks, record types, and the NEW_NODE/SAMPLE trie-diff compression. |
| [extending.md](extending.md) | The two extension points: adding a capture backend (`i_trigger`) and adding an output sink (`emitted_slice`). |

## Deep design + rationale

For the *why* behind the architecture - top-level design down to every class,
and the reasoning behind each decision - see [design/](design/README.md):

- [design/00-overview.md](design/00-overview.md) - architecture + design philosophy
- [design/01-core.md](design/01-core.md) - the core data plane
- [design/02-capture-and-backends.md](design/02-capture-and-backends.md) - sampler + perf/signal backends
- [design/03-streaming-and-wire.md](design/03-streaming-and-wire.md) - streaming pipeline + compression
- [design/04-offline-processing.md](design/04-offline-processing.md) - symbolize + shape to Perfetto
- [design/06-design-decisions.md](design/06-design-decisions.md) - cross-cutting decisions

## Also in the repo

- [`../planning/`](../planning/) - design plans (e.g. the sampler backend abstraction).

## The shape in one paragraph

`src/core/` owns the wire substrate (records, chunk framer, reader) and
is backend-agnostic. `src/sampling/` does capture (a neutral `sampler` driver
over an abstract `i_trigger` - `triggers::perf::backend` via `perf_event_open`
or `triggers::signal::backend` via per-thread timers + a signal handler) and
streaming compression (`streaming_pipeline` + `frame_trie` +
`temporal_diff_filter`). `src/processing/` does offline shaping over
already-captured data: `cpu_snap::processing::symbols` (`proc_maps`, `symbolizer`,
`sample_postprocessor`) and `cpu_snap::processing::timeline` (`slice_emitter` ->
`emitted_slice` -> a Perfetto sink). See [architecture.md](architecture.md)
for the full picture.
