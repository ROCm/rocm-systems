# cpu-snap usage guide

Practical, task-oriented walkthrough of driving the cpu-snap library: how to
record CPU callstacks, pick a capture backend and unwind mode,
symbolize the results offline, shape them into a timeline, and run the bundled
demos.

This guide is deliberately scoped to "how do I use it". For the surrounding
context, see the sibling docs rather than re-reading them here:

- [`../README.md`](../README.md) - the canonical, deep reference: full config
  field tables, the streaming sink threading contract, the wire-format summary,
  overhead numbers, `find_package` / install, and the unwind-mode decision tree.
  When this guide and the README disagree on a field name, the README plus the
  headers win.
- [`../examples/README.md`](../examples/README.md) - the {mode, component} demo
  matrix and a one-line summary of every bundled demo.
The authoritative API is always the headers under
`src/cpu_snap/sampling/` and
`src/cpu_snap/core/`. Snippets below are grounded in
`src/cpu_snap/sampling/sampler.hpp` and the demos under
`examples/`.

## Prerequisites

Per-thread perf sampling needs `kernel.perf_event_paranoid <= 2` (or
`CAP_PERFMON`). Check and, if needed, lower it:

```bash
cat /proc/sys/kernel/perf_event_paranoid    # need <= 2
sudo sysctl kernel.perf_event_paranoid=1    # temporary
```

The default `automatic` backend falls back to the `signal` backend (below), which
needs no perf privilege at all, when perf is refused.

## Collecting samples

The sampler always streams: it encodes samples into chunks and hands every chunk
to a sink you provide (`config.streaming.sink`, required; the constructor throws
without one). There is no in-RAM sample map. Two ways to consume the stream:

- keep it in memory with the bounded `core::memory_sink` and rebuild `raw_sample`s
  with `core::sample_reconstructor` after `stop()` (short, contained runs);
- write the chunks to a file, socket or any other sink and decode them elsewhere
  with `core::reader` (long soaks, off-host analysis).

### In-memory capture

```cpp
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/sampling/sampler.hpp"

cpu_snap::core::memory_sink sink(
    cpu_snap::core::memory_sink::config{ .max_bytes = 256 * 1024 * 1024 });

cpu_snap::sampling::sampler rec(cpu_snap::sampling::sampler::config{
    .perf      = { .sample_freq_hz  = 1000,
                   .callchain_depth = 128,
                   .unwind          = cpu_snap::sampling::unwind_mode::auto_best },
    .threads   = { .watch_new_threads = true },
    .streaming = { .sink = std::ref(sink), .process_name = "myapp" },
});

rec.start();
do_my_workload();
rec.stop();                          // joins the worker; flushes the last chunk + EOF

cpu_snap::core::sample_reconstructor reconstructor;
reconstructor.feed(sink.bytes());
for (const auto& [tid, samples] : reconstructor.samples()) {
    for (const auto& s : samples) {
        // raw_sample{ tid, timestamp_ns, ip, frames (innermost-first) }
    }
}
```

`memory_sink` is bounded: once `max_bytes` is reached it stops accepting chunks
(`sink.dropped_chunks()` counts them); it is not a ring, because the trie
records in earlier chunks are needed to decode later ones.

Note the nested config shape: sampling knobs live under `.perf`, thread
discovery under `.threads`, sink and process name under `.streaming`. Leave
`kernel_pages_per_tid` unset: the sampler sizes the perf ring from
`dwarf_stack_bytes` (default 32 KiB) and rejects an explicit ring that cannot
hold one record.

`pause()` / `resume()` gate the sampling window without paying worker
create/join cost; both are idempotent and `noexcept`. `stop()` followed by
`start()` samples again.

Reference demos: `examples/single_thread_callstack_demo.cpp` (smallest
end-to-end), `examples/multi_thread_sample_demo.cpp` (multi-thread +
postprocess). To sample an existing program without writing code, use
`cpu-snap-run` (README section "Sampling any program").

### Custom sink

Any `cpu_snap::core::sink_fn` (`std::function<void(std::span<const std::byte>)>`)
works. Every encoded chunk is handed to it on the sampler's worker thread.

```cpp
std::vector<std::byte> buf;          // must outlive rec.stop()
auto sink = [&buf](std::span<const std::byte> chunk) {
    buf.insert(buf.end(), chunk.begin(), chunk.end());
};

cpu_snap::sampling::sampler rec(cpu_snap::sampling::sampler::config{
    .perf      = { .sample_freq_hz = 5000, .callchain_depth = 128 },
    .threads   = { .watch_new_threads = true },
    .streaming = { .sink         = sink,
                   .chunk_bytes  = 1 << 20,   // 1 MiB chunks (default)
                   .process_name = "myapp" },
});

rec.start();
do_heavy_workload();
rec.stop();
```

Sink contract (full version in the README "Sink threading contract"): it runs on
the worker thread, calls are serialized and never reentrant, the `span` is valid
only inside the call (copy out if you need it longer), and it MUST NOT block. A
throwing sink loses that chunk; the encoder then replays its trie so later
samples still decode. Blocking stalls all sampling and risks kernel ring
overflow (`lost_samples()`).

Decode the bytes with a reader:

```cpp
#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"

cpu_snap::core::reader reader;
reader.on_event([&](const cpu_snap::core::record_variant& var) {
    if (cpu_snap::core::type_of(var) != cpu_snap::core::record_type::sample) return;
    const auto& smp = std::get<cpu_snap::core::sample_record>(var);
    if (smp.leaf_node_id == 0) return;
    const auto ips = reader.resolve_stack(smp.leaf_node_id);  // innermost-first
    // smp.tid, smp.timestamp_ns, ips...
});
reader.feed(std::span<const std::byte>(buf.data(), buf.size()));
```

`reader.feed()` is `noexcept` and accepts any byte slicing (single byte, whole
file, network packet); it buffers partial chunks internally. Malformed chunks and
unknown records are absorbed into `reader.counters()` rather than thrown.
`resolve_stack(leaf_id)` returns the same innermost-first order as
`raw_sample::frames`; an empty result means that leaf arrived in a skipped chunk.

## Selecting the capture backend

`config.backend` picks the mechanism the sampler uses to capture stacks. Two
backends satisfy the same internal `i_trigger` seam:

| `backend_kind` | Mechanism | Clock | Privilege | Unwind |
|---|---|---|---|---|
| `automatic` (default) | resolves to `perf` when `perf_event_open` works, else `signal` | see the chosen backend | none required | per the chosen backend |
| `perf` | `perf_event_open` per tid; kernel takes the sample IRQ | CPU-clock, kernel-side sampling | needs `perf_event_paranoid <= 2` | FP / DWARF |
| `signal` | per-thread POSIX timer (`SIGEV_THREAD_ID`) + a realtime signal; handler snapshots regs+stack, worker unwinds off-handler | timer: `CLOCK_MONOTONIC` (wall-time); sample stamp: `config::clock` | none | libdwfl off-handler |

```cpp
cpu_snap::sampling::sampler::config cfg{ /* perf/threads/streaming as above */ };
cfg.backend = cpu_snap::sampling::backend_kind::signal;     // or ::perf, ::automatic (default)
cfg.signal  = { .sample_freq_hz = 1000 };                    // signal_attr_options
```

`automatic` is the default. At construction it probes with a real
`perf_event_open` call (not by reading `perf_event_paranoid`, which `CAP_PERFMON`,
containers and seccomp make unreliable), picks `perf` when it succeeds and falls
back to `signal` when it is refused. An explicit `perf` that cannot open throws at
construction with an actionable message; an explicit `signal` never probes.
`sampler::selected_backend()` returns the chosen `kind` and, after a fallback, a
non-empty `reason`; the library itself never writes to stderr. A fallback changes
the sample semantics (perf: CPU-clock samples with the kernel frame-pointer
callchain by default; signal: per-thread timers with DWARF unwinding by default),
so compare runs knowingly. When the fallback is taken and `signal.sample_freq_hz`
is still its default, `perf.sample_freq_hz` is used.

`perf` is the one the overhead numbers in the README are
measured against. The kernel takes the sampling interrupt and walks the
callchain; per-sample overhead is essentially kernel IRQ cost.

`signal` is the privilege-free alternative. It does NOT use `perf_event_open`, so
`--backend signal` works where perf is gated. Things to know (details in the
file-level note of
`src/cpu_snap/sampling/triggers/signal/backend.hpp`):

- Its timer runs on `CLOCK_MONOTONIC` (wall-time) by default, so a sleeping thread
  also receives ticks, and a blocked syscall in the host can return `EINTR` at
  the sample rate (`SA_RESTART` does not cover `poll`/`epoll_wait`/`nanosleep`).
  `backend::config::cpu_time = true` selects the target thread's own CPU-time
  clock instead (per-thread CPU time; sleeping threads get no ticks).
- It is host-safe: the handler preserves `errno`; the constructor throws
  `EBUSY` if the host already installed a handler (or `SIG_IGN`) for the chosen
  realtime signal and restores the previous action on destruction; an existing
  alternate signal stack of a thread is kept.
- `stop()` followed by `start()` samples again. A sampler that existed at `fork()`
  is not usable in the child; construct a new one there.
- The handler wakes the worker only when a thread's ring reaches a quarter of its
  capacity, so the worker must poll at a period well below
  `ring_capacity / sample_freq_hz` (the sampler's 10 ms poll is fine).
- Captured frames go through the off-handler libdwfl unwind and emit the full
  innermost-first chain; if the unwind yields nothing (e.g. a truncated stack
  copy) it falls back to a single leaf-IP frame so no sample is dropped.
  `stack_snapshot_bytes` defaults to 8 KiB, which cuts frames deeper than that;
  runtimes with deep frames (HIP/LLVM) need 32 KiB.
- Words the unwinder needs beyond the copied window are read from the thread's live
  memory, which may have changed since the sample. `sampler::unwinder_live_stack_samples()`
  counts the samples that did so (compare with `total_samples()`); if it is high, raise
  `signal.stack_snapshot_bytes` (or `perf.dwarf_stack_bytes` on the perf backend).

Only one `triggers::signal::backend` can be active per process at a time (the signal handler
is process-wide). The relevant tuning knobs are in `sampler::signal_attr_options`
(`sample_freq_hz`, `signal_rt_offset`, `stack_snapshot_bytes`, `ring_capacity`,
`start_delay_ns`).

Every bundled demo accepts `--backend auto|perf|signal` (default `auto`) via the shared helper
`examples/common/backend_select.hpp`, so you can A/B the two backends at the same
frequency without rebuilding (see "Running the demos" below).

## Selecting the unwind mode

`config.perf.unwind` (a `cpu_snap::sampling::unwind_mode`, from
`src/cpu_snap/sampling/unwind_mode.hpp`) chooses how the stack is
walked under the `perf` backend:

| Mode | perf sample_type | Availability | Notes |
|---|---|---|---|
| `auto_best` (default) | resolves at ctor | always | `dwarf` on both backends; correct for any binary. |
| `frame_pointer` | `PERF_SAMPLE_CALLCHAIN` | always | Misses frames in binaries built without `-fno-omit-frame-pointer`. Cheapest; explicit opt-in on both backends (perf: kernel callchain, signal: copied-stack rbp walk). |
| `dwarf` | `PERF_SAMPLE_REGS_USER + PERF_SAMPLE_STACK_USER` + libdwfl | any binary | Highest per-sample cost; works without frame pointers. |

There is no LBR call-stack mode: AMD CPUs (this project's target) do not
support LBR call-stack sampling, only branch-history LBR.

```cpp
cfg.perf.unwind = cpu_snap::sampling::unwind_mode::dwarf;
```

`auto_best` is recommended. To inspect what was actually probed/resolved at
construction:

```cpp
const auto& caps = rec.caps();             // frame_pointer / precise_ip / ibs_op
const auto  mode = rec.resolved_unwind_mode();
```

### Choosing unwind_mode

`auto_best` resolves to `dwarf` on both backends: it is correct for any binary,
and at the default 200 Hz it costs about 0.8% of one core. `frame_pointer` is
the cheaper explicit choice when the target and its libraries keep frame
pointers. Sampler worker CPU per 10 s, 4 threads, perf backend (median):

| Frequency | frame_pointer | dwarf |
|---|---|---|
| 1 kHz | 0.04 s | 0.41 s |
| 5 kHz | 0.07 s | 1.33 s |
| 10 kHz | 0.12 s | 2.63 s |
| 32 kHz | 0.33 s | 7.69 s (~77% of a core) |

At 32 kHz the perf ring loses about 0.005% of samples with `frame_pointer`
versus about 0.4% with `dwarf`. Application slowdown is nearly equal: 1.0% vs
1.2% at 5 kHz, 4.5% vs 5.5% at 32 kHz.

Caveats: one machine, a synthetic 4-thread workload built with frame pointers,
3 repetitions per cell. Binaries without frame pointers were not measured; there
`frame_pointer` loses or corrupts frames silently.

Guidance: keep the default `dwarf`; choose `frame_pointer` at >= 10 kHz when the
target and its libraries keep frame pointers (Ubuntu 24.04 libraries do, ROCm
libraries generally do not). With a logger set, the sampler logs a hint when
`dwarf` is combined with a rate >= 10 kHz.

The full decision flow, the "why DWARF is the fallback" rationale, and the
unwind-vs-symbolize debug-info distinction are in the README "Unwind mode
selection" section. The signal backend always unwinds via off-handler libdwfl and
does not use this enum.

## Selecting the timestamp clock

`sampler::config::clock` picks the one clock every timestamp the library
produces is read from:

| `clock_kind` | POSIX clock | Notes |
|---|---|---|
| `monotonic` (default) | `CLOCK_MONOTONIC` | Not stepped by time changes; stops while the system is suspended. |
| `boottime` | `CLOCK_BOOTTIME` | Like monotonic but keeps counting across suspend. |
| `wallclock` | `CLOCK_REALTIME` | Unix-epoch nanoseconds; can jump when the system time is set. |

```cpp
cfg.clock = cpu_snap::sampling::clock_kind::boottime;
```

It applies to sample timestamps (perf `attr.clockid`, or `clock_gettime` in the
signal handler), module (MAPS_INIT/MAPS_DELTA) records, the `wall_ns` of
LOST/CHECKPOINT/EOF records, and `log_record::timestamp_ns`. It does not change
`file_header.start_unix_ns` (always `CLOCK_REALTIME`), the timer scheduling
clocks, or the thread rescan period. The choice is stored in `file_header.flags`
bits 3-4; `core::stream_clock_of(reader.header())` reads it back. The header has
no conversion anchor, so use `wallclock` when you need Unix time directly.

With the perf backend a non-default clock the kernel rejects makes thread attach
throw `cpu_snap::environment_error` naming the clock; there is no silent fallback to
another clock. LOST/CHECKPOINT/EOF `wall_ns` used to be `CLOCK_REALTIME`; they
now follow the stream clock (see `docs/wire-format.md`).

## Error handling

Every failure the library raises derives from one base, `cpu_snap::error`
(`#include <cpu_snap/error.hpp>`, itself a `std::runtime_error`).

| Type | Thrown when | What the caller can do |
|------|-------------|------------------------|
| `config_error` | Invalid configuration, before any resource is acquired | Fix the config |
| `environment_error` | The kernel, privileges or OS refuse an operation (`perf_event_open`, `mmap`, `epoll`, `timer_create`, `sigaction`, `/proc` enumeration); `code()` holds the errno | Relax `perf_event_paranoid` / grant `CAP_PERFMON`, or use the signal backend |
| `resource_error` | A limit is exhausted (tid table full, trie node cap, record larger than a chunk) | Raise the limit or reduce the load |
| `usage_error` | API misuse (wrong call order, second active signal backend) | Fix the call sequence |

Exceptions are thrown only from the constructor, `start()` and setup, on the
caller's thread; `start()` rolls back before rethrowing. They are never thrown
from the signal handler, the worker loop, destructors, atfork handlers or the
per-sample path. Failures there (a thread that cannot be attached during a
rescan, a throwing sink, a full trie) are reported through counters (for example
`rescan_failures` and `internal_errors` in the stats) and the injected logger;
they do not surface as exceptions.

```cpp
try {
    cpu_snap::sampling::sampler s{cfg};
    s.start();
} catch(const cpu_snap::error& e) {
    std::fprintf(stderr, "cpu-snap: %s\n", e.what());
}
```

```cpp
try {
    s.start();
} catch(const cpu_snap::environment_error& e) {
    if(e.code() == std::errc::permission_denied) { /* fall back to the signal backend */ }
}
```

## Logging

The library never writes to stderr. To receive its messages, set
`sampler::config::logger` to a `cpu_snap::sampling::log_sink`
(`std::function<void(const log_record&)>`); an empty sink (the default) turns
logging off at no cost.

What is logged:

- constructor: the backend choice (`info`, or `warn` with the reason when
  `automatic` fell back to the signal backend) and the resolved unwind mode (`debug`)
- `start()`: `info` with backend, frequency and unwind mode
- `stop()`: `info` summary with samples, dropped, internal errors and live-stack samples
- thread rescan: `warn` when attaching to a thread fails, once per tid, and once
  when `/proc/self/task` cannot be listed

Nothing is logged per sample.

`log_record::timestamp_ns` is read from `sampler::config::clock` (default
`CLOCK_MONOTONIC`), the clock of sample and module-record timestamps, so log
lines line up with the sample timeline. See "Selecting the timestamp clock".

Threading: the sink is called from the thread that constructs, starts and stops
the sampler and from the sampler worker thread, never from a signal handler. It
must be thread-safe and must not call back into the sampler (records from the
worker thread are delivered while the sampler holds an internal lock). An
exception thrown by the sink is swallowed.

Adapting an application logger:

```cpp
cfg.logger = [](const cpu_snap::sampling::log_record& r) {
    static constexpr const char* names[] = { "debug", "info", "warn", "error" };
    my_app::log(names[static_cast<int>(r.level)], r.timestamp_ns, r.message);
};
```

## Offline symbolization

Symbol resolution is deliberately OUT of the recording hot path; do it after
`stop()`, from the decoded samples. The
`cpu_snap::processing::symbols::symbolizer`
(`src/cpu_snap/processing/symbols/symbolizer.hpp`) batches IPs by owning
binary and shells out to `addr2line` once per binary instead of once per IP.

```cpp
#include "cpu_snap/processing/symbols/symbolizer.hpp"

cpu_snap::processing::symbols::symbolizer sym(
    cpu_snap::processing::symbols::symbolizer_config{ .include_binary_name = true });

// Collect every unique IP first, then preload in ONE batch.
std::unordered_set<std::uint64_t> seen;
for (const auto& [tid, samples] : reconstructor.samples())
    for (const auto& s : samples) {
        if (s.ip) seen.insert(s.ip);
        for (std::uint32_t i = 0; i < s.frames.size(); ++i)
            if (s.frames[i]) seen.insert(s.frames[i]);
    }
sym.preload(std::vector<std::uint64_t>(seen.begin(), seen.end()));

const std::string label = std::string(sym.resolve(ip));   // "function(args) [binary]"
```

The single `preload()` is the load-bearing step: calling `resolve(ip)` cold,
per IP, forces one `addr2line` invocation per IP and dominates wall time on any
non-trivial trace. `resolve(ip)` returns the outer (non-inlined) name;
`resolve_inline(ip)` returns the full inline-call chain (innermost first) when
the symbolizer was constructed with `inline_frames = true`. On `addr2line`
failure the resolver falls back to a `<base>+0x<hex>` string.

The richer offline pass, `cpu_snap::processing::symbols::sample_postprocessor`, adds
per-IP-cached resolution plus frame predicate filtering and a
`drop_library_frames` toggle (keep only main-exe frames). See the README "Filter
out library frames" and "Optional: postprocess + frame coalescing" sections, and
`examples/multi_thread_sample_demo.cpp` for a worked example.

## Timeline shaping into Perfetto

`cpu_snap::processing::timeline::slice_emitter`
(`src/cpu_snap/processing/timeline/slice_emitter.hpp`) is the
backend-agnostic timeline-shaping pass: it turns per-tid `raw_sample` vectors
into a stream of `emitted_slice` records and hands each to a sink. It does the
open/close stack diffing across consecutive samples; it is NOT a
symbolizer/filter. You supply the IP-to-name resolver (typically the symbolizer
above), and a backend writer maps each `emitted_slice` to its native record type
(the demos use `examples/common/perfetto_proto_writer.hpp` for `.pftrace`).

```cpp
#include "cpu_snap/processing/timeline/slice_emitter.hpp"

cpu_snap::processing::timeline::slice_emitter_config scfg;
scfg.merge_samples = true;                    // flame-graph view (false -> one stack per sample)
scfg.resolver = [&sym](std::uint64_t ip) {
    const auto label = sym.resolve(ip);
    if (!label.empty()) return std::string(label);
    std::array<char, 40> hex{};
    std::snprintf(hex.data(), hex.size(), "0x%lx", static_cast<unsigned long>(ip));
    return std::string(hex.data());
};
// symbol_identity fixes the "one function, multiple call sites" case:
// without it, a caller invoked from several call sites splits into one
// slice per call site instead of staying open as a single slice.
scfg.symbol_identity = [&sym](std::uint64_t ip) { return sym.symbol_identity(ip); };

cpu_snap::processing::timeline::slice_emitter emitter(std::move(scfg));
emitter.process(reconstructor.samples(), [&](const cpu_snap::processing::timeline::emitted_slice& s) {
    writer.write_slice(s);                     // your Perfetto/pprof/collapsed backend
});
writer.finalize();
```

Key behaviors: `process()` sorts each tid's samples by timestamp before shaping
(Perfetto needs monotonic timestamps). With `merge_samples = true` (default) it emits one
slice per frame-span; `merge_samples = false` stops frames from being merged
across consecutive samples: every sample emits its own full stack over
`[its timestamp, the next sample's)`, so the timeline shows exactly what each
sample captured (one slice per frame per sample, so the output is large). An empty `resolver` falls back to `0x%lx` hex names. Set
`inline_resolver` to expand inline chains and diff on frame
names. `slice_emitter` also has its own `library_frame_filter` callback
(`std::function<bool(uint64_t, string_view)>`, unset by default) for dropping
shared-library frames before shaping; `cpu_snap::processing::timeline` does not link
`cpu_snap::processing::symbols` or `core::proc_maps` itself, so build the predicate with
`cpu_snap::processing::symbols::make_main_exe_only_filter` (or your own) and pass it in. The
`emitted_slice` string views are valid only inside the sink call - copy out
anything you keep.

## Running the demos

Build, then run any demo. Every callstack/streaming demo takes an optional output
path and the `--backend` flag (parsed by `examples/common/backend_select.hpp`):

```bash
mkdir build && cd build && cmake .. && make -j

# minimal single-thread callstack demo, default (auto) backend, default output path
./examples/single_thread_callstack_demo /tmp/demo.pftrace

# same demo on the privilege-free signal backend (no perf_event_paranoid needed)
./examples/single_thread_callstack_demo /tmp/demo.pftrace --backend signal

# multi-thread demo, sink -> sample_reconstructor -> Perfetto
./examples/multi_thread_callstack_demo /tmp/cs_combined.pftrace
```

`--backend` may appear anywhere in argv; `--backend=signal` is also accepted; an
unrecognized value falls back to `auto`. The first non-flag argument is the
output path. Open the resulting `.pftrace` in <https://ui.perfetto.dev/>.

To sample any program without writing code, use `cpu-snap-run` (README section
"Sampling any program"). For the `LD_PRELOAD` autoload flavour it is built on
(sampler attaches before `main()`, writes the trace at exit) and the env vars
that configure it, see the README "Autoload" section. The full demo matrix and per-demo notes are in
[`../examples/README.md`](../examples/README.md).
