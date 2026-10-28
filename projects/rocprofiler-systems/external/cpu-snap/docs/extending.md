# Extending cpu-snap

This guide is for a developer adding new functionality to cpu-snap. It covers
the two extension points the codebase is explicitly designed around:

1. Adding a **capture backend** - a new mechanism for getting CPU stack samples
   out of the kernel (perf_event_open, signal/timer, or something new).
2. Adding an **output backend** - a new sink that turns the backend-neutral
   timeline records into a concrete trace format (Perfetto, pprof, collapsed
   flamegraph, ...).

For the big picture (component inventory, the capture-to-output pipeline, and the
wire format) read these first, and treat them as the source of truth - this
document links to them rather than restating them:

- [../README.md](../README.md) - what cpu-snap is and how to build/run it.
- [../CHANGELOG.md](../CHANGELOG.md) - notable changes.

All paths below are relative to the repository root.

---

## 1. Adding a capture backend

### The shape of the seam

A capture backend owns the kernel-facing state of one sampling mechanism (file
descriptors, mmap rings, timers, signal handlers) and exposes a small,
backend-neutral surface that the `sampler` driver drives from its worker loop.
The driver does not know or care how samples are produced; it only knows the
neutral operations.

Two types make up the seam:

- `src/cpu_snap/sampling/triggers/i_trigger.hpp` - the abstract
  `i_trigger` class. This is the contract a new backend must implement; the
  driver holds one through `std::unique_ptr<i_trigger>`.
- `src/cpu_snap/sampling/streaming_pipeline.hpp` - the single,
  concrete per-sample sink the driver owns and every backend drains into.

Two backends already implement the seam and are the worked examples to copy:

- `src/cpu_snap/sampling/triggers/perf/backend.hpp` - the preferred,
  `perf_event_open` + mmap-ring backend. A thin adapter over
  `internal::kernel_drain`.
- `src/cpu_snap/sampling/triggers/signal/backend.hpp` (impl in
  `src/cpu_snap/sampling/triggers/signal/backend.cpp`) - the privilege-free, per-thread
  timer + realtime-signal backend.

### What a new backend must implement

Your backend must derive from `i_trigger`
and override these member functions (contract from `i_trigger.hpp`):

| Operation                       | Signature                                  | Notes |
|---------------------------------|--------------------------------------------|-------|
| `can_provide(kind) const`       | `bool` `noexcept`                          | Whether this trigger can hand an unwinder the given capture kind; checked once at sampler construction. |
| `attach_tid(tid)`               | `void`                                     | The ONLY operation allowed to throw (resource acquisition, e.g. `perf_event_open` / `timer_create`). |
| `detach_tid(tid)`               | `void` `noexcept`                          | Tear down per-tid state. |
| `wait(timeout_ms)`              | `int` `noexcept`                           | Block up to `timeout_ms` for data; return ready count (>0), 0 on timeout/EINTR, -1 on error. `0` polls, `-1` blocks forever. |
| `drain_into(pipeline, unwinder)` | `void` `noexcept`                          | Drain all pending samples once, dispatching each into the concrete `streaming_pipeline`. |
| `disable_all()`                 | `void` `noexcept`                          | Best-effort: stop sampling without tearing down state. |
| `enable_all()`                  | `void` `noexcept`                          | Re-arm sampling. |
| `close_all()`                   | `void` `noexcept`                          | Tear down everything without draining. |
| `tracked_tids() const`          | `std::size_t` `noexcept`                   | Count of attached tids. |
| `is_tracked(tid) const`         | `bool` `noexcept`                          | Whether `tid` is currently attached. |
| `collect_tids(out) const`       | `void`                                     | Overwrite `out` with the live tid set (clear first). Used by the driver's exited-tid GC. |

Key contract points to honour:

- **Only `attach_tid` may throw.** Every other operation is `noexcept`. This
  matches the driver's hot/cold split: enrollment is the one cold path allowed
  to fail.
- **Per-sample dispatch is NOT virtual.** `drain_into` takes the
  *concrete* `streaming_pipeline` (not a template, not a `std::function`), so the
  per-sample loop stays inlinable inside your backend. Virtual dispatch happens only
  at the per-batch granularity of `drain_into`. Do not introduce a virtual call or
  `std::function` on the per-sample path.
- **Exited threads attach as a silent no-op.** If `attach_tid` hits the
  thread-already-exited case (e.g. `ESRCH` on `perf_event_open`), swallow it
  instead of throwing - the driver rescans `/proc/self/task` and races with
  thread exit.

### How samples leave the backend: streaming_pipeline

Inside `drain_into`, decode each sample and hand it to the pipeline via its two
entry points (see `streaming_pipeline.hpp`):

- `pipeline.on_sample(tid, ts_ns, frames)` - one decoded sample. `frames` is an
  innermost-first `std::span<const std::uint64_t>` valid only for the duration
  of the call (the pipeline copies what it needs).
- `pipeline.on_lost(lost)` - a kernel-reported lost-sample count.

The pipeline encodes and streams the samples; your backend never sees that. Look at `triggers::perf::backend::drain_into` for the canonical pattern -
it forwards two lambdas into `kernel_drain::drain_all` so the decode stays inside
`kernel_drain` and the pipeline call inlines through.

### How config.backend selects your backend

The driver builds exactly one backend at construction, chosen by
`sampler::config::backend` (a `backend_kind` enum). The wiring lives in
`src/cpu_snap/sampling/sampler.cpp`:

```cpp
switch(m_cfg.backend)  // already resolved from backend_kind::automatic by resolve_backend()
{
    case backend_kind::perf:
        m_backend = std::make_unique<triggers::perf::backend>(make_drain_config());
        break;
    case backend_kind::signal:
        validate_signal_config(m_cfg);
        m_backend = std::make_unique<triggers::signal::backend>(make_signal_config(m_cfg));
        break;
    case backend_kind::automatic: break;  // already resolved to perf or signal
}
```

To add a third backend you need to:

1. Add an enumerator to `backend_kind` (and handle it in `resolve_backend()` in `backend_selection.hpp`) in
   `src/cpu_snap/sampling/sampler.hpp`.
2. If your backend needs tunables, add an options struct to `sampler::config`
   (mirror `perf_attr_options` / `signal_attr_options`) and a `make_*_config`
   translation helper in `sampler.cpp`.
3. Add a `case` to the switch above that constructs your backend with
   `std::make_unique<your_backend>(...)`. It is held by `unique_ptr`, so
   non-movable backends are fine (both existing backends are non-movable:
   `triggers::perf::backend` embeds a `kernel_drain`, `triggers::signal::backend`
   owns the signal handler registry).

For tests, `sampler`'s constructor also accepts a ready-made
`std::unique_ptr<triggers::i_trigger>`, which is used instead of the switch.

### Where to validate

Config validation belongs in the driver, before backend construction (see
`validate_config`, `validate_signal_config`, `resolve_unwind_mode`,
`validate_unwind_constraints` in `sampler.cpp`). Backends receive an
*already-validated* config struct (`triggers::perf::backend` and `triggers::signal::backend` both
document this on their ctors).

---

## 2. Adding an output backend (a sink)

### The backend-neutral record: emitted_slice

The output side is decoupled from the capture side through `slice_emitter`
(`src/cpu_snap/processing/timeline/slice_emitter.hpp` and
`src/cpu_snap/processing/timeline/slice_emitter.cpp`). The emitter is the timeline-shaping pass: it
takes per-tid `raw_sample` vectors, sorts each tid's samples by timestamp, diffs
consecutive stacks, and produces a stream of `emitted_slice` records. It does no
symbolization, filtering, or format-specific work.

An `emitted_slice` carries everything a sink needs:

```cpp
struct emitted_slice
{
    pid_t            tid;       // sampled thread
    std::string_view name;      // slice label (borrowed)
    std::uint64_t    start_ts;  // stream clock ns (CLOCK_MONOTONIC by default)
    std::uint64_t    end_ts;    // stream clock ns (CLOCK_MONOTONIC by default)
    std::uint32_t    depth;     // nesting depth, 0 = outermost
};
```

Slices are nested frame-spans (the flame-graph view). With
`slice_emitter_config::merge_samples = false`, each sample emits its own full
stack instead of merging frames across consecutive samples. With
`inline_resolver` set, inline frames are expanded and the stack diff runs on
frame names.

`depth` is the nesting level (0 = outermost). A sink that reconstructs begin/end
ordering from completed spans must break same-timestamp ties by depth (open
shallow-first, close deep-first).

### Writing a sink

A sink is **any callable taking `const emitted_slice&`**. There is no base class
or registry to implement against - you pass your callable to
`slice_emitter::process`, which invokes it once per slice:

```cpp
template <typename Sink>
std::size_t process(
    const std::unordered_map<pid_t, std::vector<raw_sample>>& samples,
    Sink&& sink);
```

Contract for a sink:

- **The string view is borrowed.** `slice.name` is owned by the emitter and
  valid ONLY inside the current call. Copy anything you need to retain (do not store the view).
- **Single-threaded, post-stop.** `process` runs on one thread and is safe to
  call after `sampler::stop()`. The sink does not need to be thread-safe.
- It is fine to drop slices your format cannot represent (e.g. a slice for a tid
  you never registered).

### The worked example: perfetto_proto_writer

`examples/common/perfetto_proto_writer.hpp` is the reference sink. It is a pure
sink: `write_slice(const emitted_slice&)` buffers each span (copying
`slice.name` into an owned string, since the view is borrowed), and
`finalize()` reconstructs the Perfetto begin/end `TrackEvent` ordering and
flushes the `.pftrace` bytes. Note that it ignores slices whose `tid` has no
matching `add_thread()` registration - dropping unrepresentable slices is a
valid sink choice.

A typical end-to-end output flow (see the demos under
`examples/`):

1. Run the `sampler`, then `stop()`.
2. Symbolize / filter via `sample_postprocessor` and `symbolizer` (these supply
   the `resolver` / `file_line` / `inline_resolver` callbacks for
   `slice_emitter_config`).
3. Construct a `slice_emitter` with your config.
4. Call `emitter.process(samples, my_sink)` with your sink callable.

To add a new format (pprof, collapsed flamegraph, JSON, ...), write a new sink
type that consumes `emitted_slice` and maps it to your native records. You do
not touch `slice_emitter` itself - the same neutral record feeds every format.

### Note on filtering vs shaping

`slice_emitter` and `sample_postprocessor` can both apply the same
`frame_filters::make_main_exe_only_filter` predicate, but they take it
differently: `sample_postprocessor_config::drop_library_frames` is a bool that
lets the postprocessor (in `cpu_snap::processing::symbols`) build and own the `proc_maps`
itself, while `slice_emitter_config::library_frame_filter` (in
`cpu_snap::processing::timeline`, which does not link `symbols` or `core::proc_maps`) takes
the already-built predicate as a `std::function`. Either way the two components
have distinct roles: the emitter does timeline shaping (opening/closing nested
frames across samples), while the postprocessor does per-frame predicate
filtering and per-survivor sink delivery. Keep new output formats in the
emitter/sink lane; keep new symbol or predicate logic in the postprocessor lane.

---

## Quick reference: files to read

Capture backend:
- `src/cpu_snap/sampling/triggers/i_trigger.hpp` (interface)
- `src/cpu_snap/sampling/streaming_pipeline.hpp` (per-sample sink)
- `src/cpu_snap/sampling/triggers/perf/backend.hpp` (example backend)
- `src/cpu_snap/sampling/triggers/signal/backend.hpp` (example backend)
- `src/cpu_snap/sampling/sampler.cpp` (backend selection switch)

Output backend:
- `src/cpu_snap/processing/timeline/slice_emitter.hpp` (`emitted_slice`, `process`)
- `examples/common/perfetto_proto_writer.hpp` (example sink)
