# 02 - Capture and the backend abstraction

This is the heart of cpu-snap: how a callstack sample is actually produced. It
covers the neutral sampler driver, the capture-backend seam, and the
two backends (perf and signal) with every supporting class. For the big picture
read [00-overview.md](./00-overview.md) first; for the cross-cutting rationale
see [06-design-decisions.md](./06-design-decisions.md).

## The two-tier shape (and why)

There is exactly **one public type**, `cpu_snap::sampling::sampler`. It is a
concrete class - not a template, not a base class - and its name never changes
regardless of which backend you use. Internally it is two tiers:

1. A **neutral driver** (the `sampler` itself): the cold lifecycle that is
   identical for every backend - the worker thread, thread enrollment + rescan,
   fork handling, stats, pause/resume, and the streaming pipeline to the
   mandatory sink.
2. A **capture backend** (chosen at runtime from `config.backend`), held behind
   a `std::unique_ptr<i_trigger>`. The backend is the only thing that
   differs between perf and signal: how a thread's stack is actually captured.

Why this split: capture is the *only* axis that varies, so that is the only
thing abstracted. Everything else (threading, fork, stats, downstream delivery)
is written once in the driver. And because the backend is selected from a
runtime `config` value, the choice must be a runtime decision - hence a
virtual interface rather than a compile-time template (see `i_trigger` below and
decision (2) in [06](./06-design-decisions.md)).

```
                       sampler  (one concrete public type)
        +------------------------+---------------------------+
        | neutral driver         | unique_ptr<i_trigger>     |
        | - worker thread        |   -> triggers::perf::backend   OR   |
        | - /proc/self/task scan |   -> triggers::signal::backend      |
        | - fork (atfork)        |                           |
        | - stats, pause/resume  | streaming_pipeline (concrete sink)
        +------------------------+---------------------------+
                 every tick -> pipeline.on_sample(tid, ts, frames) -> downstream
```

## sampler - the neutral driver

`src/cpu_snap/sampling/sampler.hpp` + `src/sampler.cpp`.

Public surface (unchanged across backends): `start()`, `stop()`, `pause()`,
`resume()`, `caps()`, and stats
accessors (`total_samples()`, `lost_samples()`, `tracked_threads()`,
`streaming_internal_errors()`).

What the driver owns and does:
- A single internal **worker thread** (`worker_loop`) that repeatedly calls
  `backend.wait(timeout)` then `backend.drain_into(pipeline)`.
- **Thread enrollment**: it enumerates the process's threads via
  `/proc/self/task` (`sampling::utils::list_self_thread_tids`) and calls
  `backend.attach_tid(tid)` for each, rescanning on a cadence to pick up new
  threads and `detach_tid`-ing ones that exit. This is how the workload's
  threads get sampled without running any cpu-snap code themselves.
  Limits of the rescan: a thread that starts and exits between two rescans
  (`rescan_period`, 100 ms by default) is never sampled. A tid recycled by the
  kernel for a new thread inside one window is detected by comparing the
  thread's start time (`/proc/self/task/<tid>/stat`, field 22) with the one
  recorded at attach, then re-attached with fresh per-tid state; the cost is one
  small `/proc` read per tracked thread per rescan. A failed per-thread attach
  or a failed `/proc/self/task` listing only skips that thread/round and is
  counted in `rescan_failures()`; a listing failure never reaps tracked threads.
- **Fork handling**: a process-wide `pthread_atfork` registry so a forked child
  resets sampler state instead of inheriting a half-live recorder.
- **Stats**: `m_total` / `m_lost` / `m_peak_tracked` atomics, fed by the
  pipeline (via `config::total_counter` / `lost_counter`).
- **Delivery**: every sample is forwarded to a `streaming_pipeline` that feeds
  the mandatory host sink; the driver passes it as the drain sink
  at `start()`. Samples are read back by decoding the sink's bytes
  (`core::reader` / `core::sample_reconstructor`).

Why a single concrete type with the driver inlined here (rather than a
`sampler_driver<Backend>` template exposed publicly): callers want one stable
type name and a runtime backend choice; the template would leak the backend into
the type. See decision (2) in [06](./06-design-decisions.md).

### config

`sampler::config` carries:
- `backend` (`backend_kind { automatic, perf, signal }`, default `automatic`) - selects the backend.
  `automatic` prefers perf and falls back to signal when a real `perf_event_open` probe is
  refused (`resolve_backend()` in `backend_selection.hpp`); explicit `perf` throws at
  construction if refused, explicit `signal` never probes. `sampler::selected_backend()`
  reports the choice and the fallback reason (the library never writes to stderr). The
  fallback changes sample semantics (perf: CPU-clock + kernel FP callchain by default;
  signal: per-thread timers + DWARF by default).
- `perf` (`perf_attr_options`) - perf-only knobs (sample_freq_hz, callchain_depth,
  kernel_pages_per_tid, exclude_kernel/hv, dwarf_stack_bytes, unwind mode).
- `signal` (`signal_attr_options`) - signal-only knobs (sample_freq_hz,
  signal_rt_offset, stack_snapshot_bytes, ring_capacity, start_delay_ns).
- `threads` (`thread_attach_options`) and `streaming` (`streaming_options`) -
  backend-neutral, shared by both.

The per-backend sub-structs keep each backend's surface honest: a signal sampler
has no `kernel_pages_per_tid`, a perf sampler has no `signal_rt_offset`.

## i_trigger - the strategy seam

`src/cpu_snap/sampling/triggers/i_trigger.hpp`.

An abstract class describing what any backend must provide so the driver can
drive it uniformly (Strategy pattern). The contract is the per-tid lifecycle plus
draining, derived directly from how the driver uses kernel_drain:
`can_provide(capture_kind)`, `attach_tid(pid_t)`, `detach_tid(pid_t)`,
`wait(timeout_ms)`, `drain_into(streaming_pipeline&, unwinder&)`, `disable_all()`,
`enable_all()`, `close_all()`, `tracked_tids()`, `is_tracked(pid_t)`,
`collect_tids(vector&)`. The driver holds one through `std::unique_ptr<i_trigger>`,
built from `config.backend` (or injected by tests).

Why a virtual interface:
- **Runtime selection**: the backend comes from `config.backend`, a runtime value,
  so it cannot be a template parameter. One uniform `sampler` member must hold any
  backend.
- **Chosen over `std::variant`** (a closed set, awkward to extend with a third
  backend).
- **Cost is paid only where it does not matter**: virtual dispatch is at the
  coarse `wait`/`attach_tid`/`drain_into` granularity - one indirect call per
  drain *batch*, which is cold. `drain_into` takes a **concrete**
  `streaming_pipeline&`, not a template sink, so the per-*sample* loop inside a
  backend stays a direct, inlinable call (next section).

## streaming_pipeline - the concrete per-sample sink

`src/cpu_snap/sampling/streaming_pipeline.hpp`.

The single concrete sink the backend drains each sample into:
`on_sample(pid_t tid, uint64 ts_ns, span<const uint64> frames)` plus
`on_lost(uint64)`. The sampler constructs it at `start()` and wires its own
`m_total` / `m_lost` atomics through `config::total_counter` / `lost_counter`, so the
pipeline bumps the stats itself before encoding.

Why this exists as a fixed concrete type rather than a `std::function` or a
template sink: the per-sample path is the hot path. An earlier design used a
`std::function` sink and it "dominated recorder cycles". By fixing the sink to one
concrete type, `triggers::perf::backend::drain_into` (and `triggers::signal::backend::drain_into`) call
`pipeline.on_sample(...)` directly and the compiler inlines it - no virtual, no
type-erased call per sample. The virtual dispatch stays at `drain_into`, the pipeline
stays concrete; the hot path never pays for the abstraction. This is decision (1)
in [06](./06-design-decisions.md).

## triggers::perf::backend - the perf_event_open backend

`src/cpu_snap/sampling/triggers/perf/backend.hpp`.

A thin adapter that satisfies `i_trigger` by owning a `triggers::perf::kernel_drain`:
`attach_tid` -> `triggers::perf::kernel_drain::open_tid` (open a perf fd + mmap ring for the tid),
`detach_tid` -> `close_tid`, `wait` -> epoll on the ring fds, `drain_into` ->
`triggers::perf::kernel_drain::drain_all` with a lambda that calls the concrete pipeline per
sample. `triggers::perf::kernel_drain` is reused unchanged - `triggers::perf::backend` adds only the
`i_trigger` surface.

This is the preferred backend (chosen by `automatic` when it can be opened); it needs
`perf_event_paranoid` permissive enough to open per-thread software/hardware perf events.

## triggers::perf::kernel_drain - perf capture + decode

`src/cpu_snap/sampling/triggers/perf/kernel_drain.hpp`.

The substantial perf machinery: one `perf_event_open` fd + mmap ring buffer per
attached tid, an epoll set to wait on them, and the per-record decode of
`PERF_RECORD_SAMPLE` for each unwind mode:
- **frame_pointer**: decode `PERF_SAMPLE_CALLCHAIN`, dropping kernel IPs above the
  user-IP ceiling.
- **dwarf**: capture `PERF_SAMPLE_REGS_USER` + `PERF_SAMPLE_STACK_USER` (a regs +
  stack snapshot) and hand it to `dwarf_unwinder` for a CFI walk.

It is kept a separate class (not folded into `triggers::perf::backend`) so the perf wrapper
stays thin and the heavy decode logic is isolated and testable. Its
`decode_scratch` holds the frame buffers, sized once from `callchain_depth`.

## dwarf_unwinder - libdwfl CFI unwinding

`src/cpu_snap/sampling/unwind/dwarf_unwinder.hpp` + `src/dwarf_unwinder.cpp`.

Unwinds a saved `{registers, stack-copy}` snapshot using elfutils `libdwfl`
(`.eh_frame`/`.debug_frame` CFI), so it recovers correct stacks even from
frame-pointer-omitted code. It is shared by both backends: the perf DWARF mode
feeds it a `PERF_SAMPLE_REGS_USER`+`STACK_USER` snapshot, and the signal backend
feeds it the `ucontext` regs + `process_vm_readv` stack copy off-handler.

Design note (a lesson, see [06](./06-design-decisions.md)): `dwfl_attach_state`
may be called only **once** per `Dwfl` session. The unwinder attaches once,
lazily, with a stable thread-state member and resets per-call state before each
`dwfl_getthread_frames`. Calling attach per unwind (the original bug) silently
returned 0 frames after the first sample.

## unwind_mode - the stack-walk strategy enum

`src/cpu_snap/sampling/unwind_mode.hpp`.

`enum class unwind_mode { auto_best, frame_pointer, dwarf }`. It
selects how the perf backend walks the stack:
- `frame_pointer` - cheapest; needs frame-pointer-built code (RBP chain).
- `dwarf` - CFI walk over a regs+stack snapshot; most accurate on FP-omitted
  code, heaviest (copies a stack slab per sample). Rejected in streaming mode
  (the snapshot does not belong on the wire).
- `auto_best` - resolved once at construction to `dwarf` on both backends
  (correct for any binary). `frame_pointer` can be requested explicitly.

There is no LBR call-stack mode: AMD CPUs (this project's target) have LBR
branch-history hardware but no LBR call-stack mode, and `perf record
--call-graph lbr` is rejected by the kernel on AMD. A prior implementation is
preserved on the `archive/lbr-implementation` branch.

Why a separate strategy enum: unwinding is a real choice with very different
cost/accuracy tradeoffs, and `auto_best` lets callers get a sensible default
without probing themselves. It is perf-specific (the signal backend always uses
the snapshot+DWARF path), which is why it lives under `config.perf`.

## sampler_capabilities - runtime capability probe

`src/cpu_snap/sampling/sampler_capabilities.hpp`.

A small struct + probe that opens throwaway perf events to detect what the
running CPU/kernel actually support: `frame_pointer`,
`precise_ip` (PEBS/IBS-precise sampling), `ibs_op` (AMD IBS). The sampler ctor
stores it and exposes it via `caps()`.

Why a separate struct: capability detection is a distinct concern (it opens and
closes probe events), and surfacing it via `sampler::caps()` lets callers query
support up front. (The audit flags it as a possible merge into `sampler` since it
is single-owner; it is kept separate as a cohesive probe - see
[06](./06-design-decisions.md).)

## raw_sample and frame_limits - the sample POD

`raw_sample.hpp`, `frame_limits.hpp`.

`raw_sample` is `{ pid_t tid; uint64 timestamp_ns; uint64 ip;
vector<uint64> frames; }`; `frames` holds exactly the captured frames.

Capture does not build `raw_sample`: the perf decode path and the signal backend unwind
into reusable buffers sized once from `callchain_depth` (a runtime setting, default 256,
bounded by `k_max_callchain_depth` = 4096), so the hot path allocates nothing per sample.
The signal handler itself only copies registers and stack bytes; the worker unwinds. In
frame-pointer mode on the perf backend the kernel additionally caps the chain at
`kernel.perf_event_max_stack`.

`frame_limits` is a small leaf header holding the default and the bound, so `sampler.hpp`
and the examples get them without including `perf/kernel_drain.hpp` (which would drag
`<linux/perf_event.h>`, epoll, and mman into every consumer).

## triggers::signal::backend - the timer + signal backend

`src/cpu_snap/sampling/triggers/signal/backend.hpp` + `src/signal/backend.cpp`.

Captures stacks without `perf_event_open`, so it works where perf is blocked
(restricted `perf_event_paranoid`, containers). Mechanism (see also
[06](./06-design-decisions.md) and the project's `01...` POSIX-signal notes):

1. **One process-wide handler**, installed once via `sigaction` for a realtime
   signal (`SIGRTMIN + offset`, not `SIGPROF`, to avoid collisions). Signal
   dispositions are process-wide; you cannot attach a handler per thread.
2. **Per-thread timers**: `attach_tid` calls `timer_create(CLOCK_MONOTONIC, ...)`
   with `sigevent{ sigev_notify = SIGEV_THREAD_ID, _tid = tid }` so the kernel
   delivers the timer's signal to that *specific* thread. The driver's worker
   creates these timers for the workload's threads.
3. **In-handler capture (async-signal-safe only)**: when a timer fires the kernel
   runs the shared handler *on the target thread*. The handler installs that
   thread's `sigaltstack` lazily on first delivery, reads `ts` (clock_gettime),
   `tid`, and the interrupted `ucontext` registers, copies a bounded slab of the
   thread's stack via `process_vm_readv` (short-reads instead of faulting near
   the stack top), and publishes a `triggers::signal::snapshot_ring` into that thread's lock-free
   ring, then wakes the worker via an eventfd. No malloc, locks, or unwinding in
   the handler.
4. **Off-handler unwind**: the worker's `drain_into` pops each snapshot and runs
   `dwarf_unwinder` over `{regs, stack-copy}` to produce the full frame list,
   then calls the concrete `streaming_pipeline` (falling back to the leaf IP if the
   unwind yields nothing). The expensive work happens off the sampled thread.

Two design points worth knowing:
- **Wall-time by default, CPU-time opt-in**: the default timer is
  `CLOCK_MONOTONIC` (wall-time), which the worker can create for any target tid and
  which matches the pipeline's monotonic timestamps; sleeping threads also get
  ticks. A `CLOCK_THREAD_CPUTIME_ID` timer would be bound to the creating thread,
  so it cannot be used from the worker, but a timer on the *target thread's own*
  CPU clock can be created cross-thread. `backend::config::cpu_time` selects it.
  See decision (3) in [06](./06-design-decisions.md).
- **Teardown flushes pending signals**: a `SIGEV_THREAD_ID` expiration queues into
  the target thread's private pending set; if that thread blocks the RT signal the
  instance is never delivered and `timer_delete` does not reclaim it, so they
  accumulate across sampler lifecycles and exhaust `RLIMIT_SIGPENDING`. `close_all`
  disarms+deletes every timer then flushes queued instances by momentarily setting
  the signal to `SIG_IGN` (the kernel discards pending instances in every thread's
  queue) and reinstating the handler. See decision (6) in
  [06](./06-design-decisions.md).

## triggers::signal::snapshot_ring - the async-signal-safe snapshot ring

`src/cpu_snap/sampling/triggers/signal/snapshot.hpp`.

A POD `triggers::signal::snapshot_ring` record (`{ ts_ns, tid, regs[], stack_copy, stack_base,
stack_size, leaf_ip }` - everything the off-handler DWARF unwind needs) plus a
preallocated per-thread single-producer/single-consumer lock-free ring
(`triggers::signal::snapshot_ring`). The handler (sole producer on its thread) `acquire`s a
slot, fills it, and `commit`s with a release store; the worker (consumer) drains.
Ring capacity and stack-copy size are preallocated in `attach_tid`, never in the
handler.

Why a preallocated lock-free per-thread ring: the handler must be async-signal-
safe - no allocation, no locks. A per-thread SPSC ring gives the producer
(handler) a wait-free publish and the consumer (worker) a clean drain, with
overflow counted as lost (mirroring perf `PERF_RECORD_LOST`). It is owned solely
by `triggers::signal::backend`.

## perf vs signal - the practical tradeoff

| | triggers::perf::backend | triggers::signal::backend |
|---|---|---|
| Capture | kernel-side, off the target thread | in the target thread's signal handler |
| Privilege | needs `perf_event_paranoid` | none |
| Clock | perf sample period | `CLOCK_MONOTONIC` wall-time (v1) |
| Kernel stacks | possible | no (user only) |
| Overhead | ~1-2% flat (1-10 kHz) | +1.4% / +4.5% / +8.6% at 1/5/10 kHz |

perf is cheaper and flat because the kernel does the capture; the signal backend
steals target-thread CPU in the handler, so its cost grows with sample rate. The
signal backend is the portable fallback. See [00-overview.md](./00-overview.md)
for where each fits and [06-design-decisions.md](./06-design-decisions.md) for
the full rationale.
