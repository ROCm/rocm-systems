# 06 - Design Decisions and Rationale

This is the "why we did it this way" record for cpu-snap: the cross-cutting
design decisions, the component inventory with the justification for every piece,
and the tradeoffs taken. It is the document to read when you need to change
something and want to know what invariant you might be about to break.

It is the last of the design-doc set. Read the earlier ones first for the shapes
this document explains the reasons for:

- [00-overview.md](./00-overview.md) - what cpu-snap is and the end-to-end flow.
- [01-core.md](./01-core.md) - the core / sampling split.
- [02-capture-and-backends.md](./02-capture-and-backends.md) - the sampler driver and the two backends.
- [03-streaming-and-wire.md](./03-streaming-and-wire.md) - the live wire-encode path.
- [04-offline-processing.md](./04-offline-processing.md) - the postprocess / shape / symbolize path.

Reference docs that carry the concrete API and byte-level detail (this document
links to them rather than restating them):

- [../components.md](../components.md) - per-component API, dependents, notes.
- [../architecture.md](../architecture.md) - the three layers and data flow.
- [../wire-format.md](../wire-format.md) - the v1 byte layout.
- [../extending.md](../extending.md) - adding a backend or an output sink.
- [../../planning/architecture-sampler-abstraction.md](../../planning/architecture-sampler-abstraction.md) - the locked plan and alternatives for the backend seam and the signal sampler.

A note on the source as the source of truth: the rationale below is taken from
the code comments, which carry most of it. Where a header doc comment has drifted
from the implementation (the signal backend headers still describe the abandoned
"CPU-time, leaf-IP only" v1 plan), this document follows the `.cpp` and the
locked planning decisions, and calls the drift out explicitly.

---

## 1. Top-down: the shape and why it has these seams

cpu-snap captures per-thread CPU callstacks, and turns them into a Perfetto
trace. The whole library is organized
around one observation: the work splits cleanly into a tiny, fixed-cost hot path
that must run while the target is executing, and a much larger amount of cold
work (symbolization, filtering, timeline shaping, wire encoding) that can run
after the fact or on a dedicated worker. Almost every decision in the codebase
falls out of keeping those two worlds apart.

The three components and the direction of their dependencies:

- `src/core/` (`cpu_snap::core`) is the backend-agnostic data plane: the wire format
  (`records`, `types`), the producer/consumer of that format (`chunk_framer`,
  `reader`), and the OS substrate (`perf_ring`, `dl_module_watcher`,
  `thread_comm`, `time`, `flat_tid_map`). It knows nothing about callstacks.
- `src/sampling/` is capture: the neutral `sampler` driver over an abstract
  `i_trigger` (`triggers::perf::backend` or `triggers::signal::backend`) and
  the live streaming encode (`streaming_pipeline` + `frame_trie` +
  `temporal_diff_filter`).
- `src/processing/` is the offline render over already-captured data:
  `proc_maps`, `symbolizer`, `sample_postprocessor`, `frame_filters`
  (namespace `cpu_snap::processing::symbols`) and `slice_emitter` (namespace
  `cpu_snap::processing::timeline`).

Dependencies point downward only: `sampling` and `processing` each depend on
`core`, never the reverse; within `sampling`, capture depends on the leaf data
types (`raw_sample`, `frame_limits`), never the reverse. This is what lets
`core` be shipped and tested as an independent CMake component, and what lets a
postprocessor translation unit avoid pulling `<linux/perf_event.h>` and the whole
decode machinery just to read a `raw_sample`.

The rest of this document is the per-decision justification.

---

## 2. The de-thunked hot path: a concrete `streaming_pipeline`, not `std::function`

### Problem

At the configured sampling frequency, the per-sample path runs once per captured
stack across every tracked thread. Anything on that path that the compiler cannot
see through - a virtual call, a `std::function` invocation - is an indirect call
plus a lost inlining opportunity, repeated millions of times. An earlier design
routed each decoded sample through a `std::function` sink; profiling showed that
thunk "dominated recorder cycles" (the note preserved in `sampler.hpp` history
and quoted in
[the planning doc](../../planning/architecture-sampler-abstraction.md)).

### Options

1. Keep a `std::function` per-sample sink. Simplest, most flexible, but it is the
   exact thing that was measured to dominate.
2. Make the per-sample sink a template parameter of the drain loop. Zero
   overhead, fully inlinable, but it makes the sink type leak into every caller
   of the drain and forces the backend abstraction to be templated too (which
   defeats runtime backend selection - see section 3).
3. Make the per-sample sink ONE concrete, named type that the driver owns and
   every backend calls directly.

### Choice

Option 3. `sampling::streaming_pipeline`
([streaming_pipeline.hpp](../../src/cpu_snap/sampling/streaming_pipeline.hpp))
is a plain class with `on_sample(pid_t tid, uint64 ts_ns, span<const uint64>
frames)` and `on_lost(uint64)`. It is deliberately not a template and not a
`std::function`. Each backend's per-sample loop calls it by concrete type, so the
compiler inlines the dispatch. The pipeline bumps the driver's received/lost
counters (non-owning pointers in its `config`) before encoding.

The driver guarantees the counters outlive the last drain: they are sampler
members, and the worker thread (the only caller) is joined before the pipeline is
torn down.

### Tradeoff

The sink's behaviour is fixed (encode into the streaming pipeline) rather than
being an arbitrary caller-supplied callback. That is a deliberate narrowing: the
hot path gets exactly one well-understood shape, and anything more exotic belongs
in the cold postprocess/shape passes, not on the per-sample path. The flexibility
that `std::function` would have bought is recovered downstream where it costs
nothing.

---

## 3. The capture-backend abstraction: a virtual `i_trigger` at batch granularity

### Problem

There are two capture mechanisms - `perf_event_open` and a signal/timer sampler -
and the choice between them must be made at runtime (a container with
`perf_event_paranoid` locked down cannot use perf; a privilege-free signal
sampler can). They share a large neutral driver (worker thread, `/proc/self/task`
rescan, fork handling, stats, pause/resume, streaming delivery) and
differ only in the capture mechanism. The abstraction has to sit exactly where
they differ, must allow runtime selection, and must NOT reintroduce a thunk on
the per-sample path (section 2).

### Options (from the locked plan)

- A runtime-virtual `i_sampler` base with the per-sample path kept internal.
  Open to future backends and trivially mockable, but it is heap + vtable + a
  base type mirroring every neutral accessor; that extensibility is unused weight
  for a closed two-backend set, and it is intrusive.
- `std::variant<triggers::perf::backend, triggers::signal::backend>` plus a thin facade. Zero
  heap/vtable, closed known set, smallest call-site churn - but the set is closed:
  adding a third backend recompiles the variant and everything that visits it,
  and `std::visit` is awkward to extend.
- Compile-time `sampler<Backend>` (policy / CRTP). Zero overhead, but the backend
  leaks into the sampler type, so runtime selection collapses back to one of the
  above at the choice point.

### Choice

Abstract the BACKEND, not the sampler. There is exactly one concrete public
`sampler` class (name unchanged, so the ~10 call sites kept their type). It owns
the neutral driver plus a `std::unique_ptr<triggers::i_trigger>`
([i_trigger.hpp](../../src/cpu_snap/sampling/triggers/i_trigger.hpp))
pointing at either a `triggers::perf::backend` or a `triggers::signal::backend`.
The backend is selected at construction from `config.backend` (tests can inject
a fake trigger through the constructor). This is the Strategy pattern;
`i_trigger` is the strategy seam.

The crucial property: virtual dispatch happens ONLY at the coarse, per-batch
granularity - `attach_tid`, `detach_tid`, `wait`, `drain_into`,
`disable_all`/`enable_all`, `close_all`, tid bookkeeping. There is one indirect
call per drain BATCH (cold). The per-sample loop lives inside the concrete
backend and calls the concrete `streaming_pipeline` directly (section 2), so
per-sample dispatch is direct and inlinable. That is why the per-sample path is
deliberately NOT virtual: `drain_into` takes the concrete `streaming_pipeline` by
reference, not a generic callback.

### Tradeoff

A virtual interface is open at the class boundary (a third backend - eBPF, AMD
IBS - plugs in by implementing `i_trigger`, with no closed sum type to recompile
and no change to the public `sampler` type or its call sites), at the cost of a
heap allocation and a vtable per backend, both irrelevant at batch granularity.
`attach_tid` is the only operation allowed to throw (resource acquisition);
everything else is `noexcept`, matching the driver's hot/cold split where only
enrollment is allowed to fail.

See [../extending.md](../extending.md) for the step-by-step of adding a backend.

---

## 4. The wall-time deviation in v1 signal sampling

This is the single most important "read before you change it" decision in the
signal backend, and the headers still describe the abandoned plan, so it is worth
stating precisely. The authoritative source is the comment block in
[triggers::signal::backend.cpp `attach_tid`](../../src/cpu_snap/sampling/triggers/signal/backend.cpp)
(around the `timer_create` call).

### Problem

The original plan (and the still-present header doc comments in
[signal/backend.hpp](../../src/cpu_snap/sampling/triggers/signal/backend.hpp)
and the `signal_attr_options` doc in
[sampler.hpp](../../src/cpu_snap/sampling/sampler.hpp)) called for
per-thread CPU-time sampling via `timer_create(CLOCK_THREAD_CPUTIME_ID,
SIGEV_THREAD_ID)`. CPU-time attribution is what a profiler usually wants: a thread
that is asleep should not accrue samples.

### Why `CLOCK_THREAD_CPUTIME_ID` does not work in this design

The driver's worker thread creates every per-thread timer (`attach_tid` runs on
the worker, for arbitrary target tids discovered by the `/proc/self/task`
rescan). Two kernel facts collide with that:

1. A `CLOCK_THREAD_CPUTIME_ID` timer is bound by the kernel to the thread that
   CREATES it, not to the `SIGEV_THREAD_ID` target. A worker-created CPU-time
   timer therefore charges the worker's CPU time and never fires against the
   sampled thread. This was verified empirically: zero ticks against the target.
2. `timer_create` is not async-signal-safe, so the target thread cannot
   self-create its own CPU-time timer from inside the signal handler either.

### Choice

v1 uses `timer_create(CLOCK_MONOTONIC, SIGEV_THREAD_ID)` - wall-time, per-thread.
A `CLOCK_MONOTONIC` timer is worker-creatable, fires correctly against the
`SIGEV_THREAD_ID` target, and produces `CLOCK_MONOTONIC` timestamps that match
the perf path and the rest of the pipeline contract (the streaming
`sample_record` and `emitted_slice` are all monotonic ns). The handler timestamps
each snapshot with `clock_gettime(CLOCK_MONOTONIC)`.

### Update: CPU-time through the target's own clock (opt-in)

The kernel facts above apply to `CLOCK_THREAD_CPUTIME_ID`. A timer created on the
*target thread's own CPU clock id* (the clock id derived from its tid) is
created from the worker and does fire against the target. It is available as
`backend::config::cpu_time` (default off: wall-time stays the default and
sleeping threads keep receiving ticks). The paths below are what a
`pthread_create` interposition would add on top.

### What other CPU-time approaches would require

Two viable paths, both deferred:

- Process-wide `setitimer(ITIMER_PROF)` - one timer, but it loses per-thread
  attribution and collides with any other `ITIMER_PROF` user.
- Per-thread self-creation: each thread creates its own
  `CLOCK_THREAD_CPUTIME_ID` timer at a safe point. Achieving that for arbitrary
  threads of an arbitrary target requires intercepting thread creation - a
  `pthread_create` interposition so each new thread arms its own timer. This is
  exactly what rocprof-sys / timemory do, and it is why those tools can offer
  true per-thread CPU-time while cpu-snap v1 cannot: they own the thread-creation
  path; cpu-snap, attaching from a worker after the fact, does not.

### Tradeoff

v1 over-counts threads that are blocked or sleeping (they still receive
wall-clock ticks). For the primary use - finding where CPU time goes in a
busy workload - wall-time on the threads that are actually running is close
enough, and it is honest about the limitation rather than silently producing a
CPU-time timer that never fires. The perf backend remains the default and gives
the kernel's own attribution; the signal backend is the privilege-free fallback.

---

## 5. Async-signal-safety: snapshot in-handler, unwind off-handler

### Problem

The signal handler runs in async-signal context on the sampled thread, interrupting
arbitrary code - possibly mid-`malloc`, mid-lock, mid-anything. Almost nothing is
safe to call there. But a useful sample needs a DWARF unwind, and libdwfl is
emphatically not async-signal-safe (it allocates, it locks, it walks ELF).

### Choice: the hybrid split

Capture the minimum async-signal-safe snapshot in the handler; defer the heavy
unwind to the worker. The full audit lives at the top of
[signal/backend.cpp](../../src/cpu_snap/sampling/triggers/signal/backend.cpp); the handler does
exactly:

- `clock_gettime(CLOCK_MONOTONIC)` (vDSO/syscall, AS-safe) for the timestamp.
- `syscall(SYS_gettid)` once, cached in a `thread_local`.
- atomic loads on the published tid -> ring table, atomic ops on the per-thread
  ring.
- plain memory reads of the `ucontext_t` GPRs, packed into the snapshot in
  `PERF_REG_X86_*` slot order so the snapshot feeds `dwarf_unwinder` directly with
  no repacking.
- `process_vm_readv` of a bounded slab of the user stack into a PREALLOCATED ring
  slot. This is used instead of a raw `memcpy` deliberately: near the top of the
  stack the fixed-size slab can extend past the stack mapping, and a `memcpy`
  would `SIGSEGV` inside the handler; `process_vm_readv` is a plain syscall
  (AS-safe) that returns a short read on `EFAULT` instead of faulting.
- a one-time `sigaltstack` install on the first delivery per thread (TLS-guarded),
  so subsequent signals on that thread run on a dedicated stack via `SA_ONSTACK`.
- an `eventfd` write of 1 to wake the worker.

It performs NO `malloc`/`new`, NO locks, NO `std::vector` growth, NO
`std::function`, NO libdw, NO stdio, and never throws. Everything else - the
libdwfl walk, the per-tid map iteration, the sink delivery - happens off-handler
in `drain_into` on the worker thread, which is allowed to allocate and call libdw.

Two supporting decisions keep the in-handler lookup safe and cheap:

- The handler resolves its per-thread ring through a file-scope, fixed-size,
  lock-free, open-addressed `tid -> ring*` table published by the worker with
  atomic stores. The handler reads it with atomic loads (hash + ~1 probe). The
  ring pointer is deliberately NOT cached in TLS: a thread can be sampled by one
  backend, see it destroyed, then be sampled by a later backend, and a cached
  `ring*` would dangle across that boundary. The global table is the single
  source of truth, retired on detach.
- The unwinder's snapshot input is the same regs+stack shape the perf DWARF path
  already produces, so a single `dwarf_unwinder` serves both backends (proven by
  `tests/sampling/test_dwarf_unwinder.cpp`). The signal backend gained full
  off-handler DWARF unwind in `989eb46`; when the unwind yields zero frames (for
  example a truncated stack copy) it falls back to the single leaf IP so the
  sample is never lost. Note this contradicts the stale header comment claiming
  "v1 emits a single leaf-IP frame"; the `.cpp` does the full chain.

### Why not unwind in the handler

The two rejected alternatives, from the plan: an in-handler frame-pointer walk
(lightest, but unreliable - libc/libdl/JIT omit frame pointers and truncate
through interposition), and in-handler libunwind `unw_step` (full resolution but
a new multiarch dependency, and libunwind is only signal-safe under constraints).
The hybrid adds zero new dependencies (it reuses the existing `dwarf_unwinder`),
is DWARF-accurate on frame-pointer-omitted code, and mirrors the perf-DWARF
design. An opt-in frame-pointer fast path is a noted future sub-mode.

### Tradeoff

The hybrid pays a per-sample stack copy (a `process_vm_readv` of up to
`stack_snapshot_bytes`, 8 KiB by default) and a one-batch-latency delay before
frames are resolved. In exchange the handler is provably bounded and safe, and
the expensive, fallible work runs where it can fail gracefully.

---

## 6. The `dwfl_attach_state`-once lesson

### Problem (a real bug fixed this cycle, commit `0f39522`)

`dwfl_attach_state` binds thread state to a `Dwfl` session and may be called only
ONCE per session; a second call fails. The unwinder was calling it per unwind.
The first sample unwound correctly; every subsequent sample silently degraded to
leaf-only (zero frames from the failed re-attach). The failure was invisible
because the unwinder returns 0 frames on error, which the callers treat as a
benign truncation.

### Choice

Attach exactly once with a STABLE argument (`&m_ts`, a member `thread_state`), set
`m_attached`, and reuse it. Each `unwind()` swaps the per-call snapshot and output
pointers into `m_ts` and reruns `dwfl_getthread_frames`. See the comment in
[dwarf_unwinder.cpp `impl::unwind`](../../src/cpu_snap/sampling/unwind/dwarf_unwinder.cpp). This
also keeps the reported modules and cached `.eh_frame` warm across the many
samples a single worker drains, so repeated unwinds are cheaper, not just correct.

### Tradeoff / invariant

One `dwarf_unwinder` instance per worker (it is not thread-safe; the attached
state is shared mutable). The perf backend's `triggers::perf::kernel_drain` and the signal
backend each reference one driver-owned unwinder. This is a non-obvious
correctness invariant: if you ever share an unwinder across threads, or
re-introduce a per-call attach, you reintroduce this bug.

Related historical note in the same component: the unwinder uses elfutils libdwfl
rather than libunwind because the earlier libunwind custom-accessor approach
aborted inside `tdep_find_proc_info` (it requires `identity ==
unw_local_addr_space`, which a remote-snapshot accessor is not). libdwfl exposes a
proper remote-thread API. libunwind is consequently NOT a dependency.

---

## 7. The realtime-signal-pending leak and the `SIG_IGN` flush teardown

### Problem (a real bug fixed this cycle, commit `bf6ea30`)

A timer armed with `SIGEV_THREAD_ID` queues its expiration into the TARGET
thread's PRIVATE pending signal set. If that thread has the realtime signal
blocked (a host app may legitimately block it in some threads), the instance is
never delivered, and `timer_delete` does NOT reclaim it (`timer_delete(2)`: "the
treatment of any pending signal generated by the deleted timer is unspecified").
Such instances accumulate across sampler lifecycles and eventually exhaust
`RLIMIT_SIGPENDING`, after which `timer_create` fails with `EAGAIN` - the signal
backend becomes effectively one-shot-per-process.

### Choice

In `close_all`, after disarming and deleting every timer and retiring every ring,
call `flush_pending_signal`
([signal/backend.cpp](../../src/cpu_snap/sampling/triggers/signal/backend.cpp)): momentarily set
the signal's disposition to `SIG_IGN`, then restore the capture handler. Signal
disposition is a per-PROCESS attribute, so setting `SIG_IGN` on a pending signal
makes the kernel discard every matching pending instance - from the shared queue
AND from every thread's private pending set, blocked or not. This is the one
cross-thread lever that clears them. It runs only at teardown, after every timer
is deleted and the worker is joined, so no expiration is in flight and the brief
ignore window loses no sample.

### Related teardown decision: the ring graveyard

A complementary use-after-free guard. When a tid is detached mid-run, its timer
is deleted and its table entry retired immediately, but a signal already
executing on the target thread may still be writing to that ring. So the ring's
storage is NOT freed on detach; it is parked in `m_retired_rings` (the
graveyard) and freed only in the destructor, by which point the worker is joined
and no signal can be in flight. See the `m_retired_rings` comment in
[signal/backend.hpp](../../src/cpu_snap/sampling/triggers/signal/backend.hpp).

A single-active-backend guard (`g_backend_active`) backs both: the process-wide
handler routes into one backend's rings at a time via the global table, so two
concurrent signal backends would alias it. Constructing a second throws.

### Tradeoff

Teardown is heavier and more careful than a naive "delete the timers" - a
momentary disposition swap plus deferred ring frees. The payoff is a signal
backend that survives repeated start/stop cycles in a long-lived process (CI,
servers) instead of silently dying after the first.

---

## 8. `frame_limits` decoupling

### Problem

`raw_sample`
([raw_sample.hpp](../../src/cpu_snap/core/raw_sample.hpp)) is
the central POD flowing from capture to every consumer (postprocessor, slice
emitter, streaming pipeline, the reader). It carries a
`std::vector<uint64_t> frames`. The callchain depth is a runtime setting
(`sampler::config::perf.callchain_depth`); its default and sanity bound
(`k_default_callchain_depth`, `k_max_callchain_depth`) are needed by the sampler and
the examples. The naive place to define them is in `perf/kernel_drain.hpp` - but that header drags in
`<linux/perf_event.h>`, epoll, mman, and the whole decode machinery. Making
`raw_sample.hpp` include `perf/kernel_drain.hpp` merely to read one integer would force
every postprocessor translation unit to compile all of that.

### Choice

Isolate the single constant in a leaf header,
[frame_limits.hpp](../../src/cpu_snap/core/frame_limits.hpp),
which holds those two constants and nothing else. Consumers include the leaf; none
pulls the others in to share them.

This is the same instinct as `core::types` (the thin function-alias header that
keeps `chunk_framer`'s heavy includes out of public headers) and the reason
`raw_sample` itself was split out of `sampler.hpp`: keep the foundational data
types in leaf headers so the wide set of consumers does not transitively compile
the capture machinery.

### Tradeoff

One more tiny header. The cost is trivial; the benefit is that `raw_sample.hpp`
has 20+ consumers and none of them pay for the perf/epoll/mman includes. The fixed
cap (32) itself is a deliberate choice: a fixed POD array means a sample can be
filled with no allocation, including in the async-signal context of the signal
handler, which a growable container could not guarantee.

---

## 9. The streaming wire format and trie-diff compression

The streaming path is the live encode that turns per-sample frame arrays into the
v1 byte stream while the target runs. Its design is driven by one number: the
~100 ns userspace per-sample budget needed to keep up at high frequency. The
detail is in [../wire-format.md](../wire-format.md) and
[03-streaming-and-wire.md](./03-streaming-and-wire.md); the decisions and their reasons:

- File header + 1 MiB chunks, records never spanning a chunk,
  with unknown record codes skippable by `body_size`
  ([records.hpp](../../src/cpu_snap/core/records.hpp)). The chunk framing
  (`file_header` 64 B, `chunk_header` 16 B) is pragma-packed
  for byte-exact wire layout; the record bodies are NOT packed and are serialized
  field-by-field, so on-disk layout is independent of struct padding. The
  `static_assert(sizeof(...) == 16)` on each fixed body guards against a silent
  field reorder.
- `frame_trie`
  ([frame_trie.hpp](../../src/cpu_snap/sampling/frame_trie.hpp)) is
  a grow-only Flyweight trie keyed on `(parent_node_id, ip)` edges; each unique
  edge interns to one `u32` node id. This is the compression heart: 99%+ of
  samples reuse an existing leaf, so the per-sample wire cost collapses to a
  single `u32`. It throws on a configured `max_nodes` cap rather than falling back
  to raw IPs - a v1 simplification, with the raw-IP fallback record type deferred
  to v2.
- `temporal_diff_filter`
  ([temporal_diff_filter.hpp](../../src/cpu_snap/sampling/temporal_diff_filter.hpp))
  sits between capture and the trie. It tracks the running stack per tid, finds
  the longest common prefix against the previous sample on that tid, and interns
  only the divergent suffix. A sample identical to its predecessor costs zero trie
  ops and zero allocations - the dominant case at high frequency, and the reason
  the per-sample budget is reachable. Its per-tid state lives in
  `sampling::flat_tid_map` (a 256-slot direct-mapped table tuned for clustered tids).
- The ordering contract: a `NEW_NODE` record is always emitted BEFORE the `SAMPLE`
  that references it, and only for the divergent suffix. Readers rebuild the trie
  by forward replay and reject a `SAMPLE` naming an unknown node, so parents must
  precede children on the wire.
- `core::reader`
  ([reader.hpp](../../src/cpu_snap/core/reader.hpp)) is a push parser
  that rebuilds its OWN internal trie view from the `NEW_NODE` stream so
  `resolve_stack(leaf_id)` returns the full IP chain. It is noexcept end to end:
  parse errors, rejected chunk headers, unknown record types, and host-callback
  exceptions are absorbed into counters; it resyncs on a bad chunk header and
  goes fatal only on bad file-header magic or an unsupported major version.

Why a separate live encode at all, distinct from the offline render: the sampler
keeps no in-RAM sample map (it would be unbounded for a long run); streaming to a
mandatory sink is the only mode. Streaming works
with every `unwind_mode`, including `dwarf`: the perf backend unwinds on the
worker thread before handing frames to `streaming_pipeline`, same as the always-
DWARF signal backend already does; there is no separate live-encode hazard tied
to the unwind mode.

---

## 10. The offline render: postprocess, shape, symbolize

After `stop()`, the `raw_sample`s rebuilt from the streamed bytes
(`core::sample_reconstructor`) are turned into output. This is the
cold side, and it is split into three single-purpose passes plus a shared filter,
detailed in [04-offline-processing.md](./04-offline-processing.md) and
[../components.md](../components.md):

- `symbolizer`
  ([symbolizer.hpp](../../src/cpu_snap/processing/symbols/symbolizer.hpp)) -
  IP -> `function [binary]`, bundling `proc_maps` with `addr2line`. `preload()`
  groups IPs by owning binary so `addr2line` runs once per binary instead of once
  per IP, and it exposes inline-call chains and file:line. The batching is the
  point: per-IP `addr2line` would be hopeless for a large trace.
- `sample_postprocessor`
  ([sample_postprocessor.hpp](../../src/cpu_snap/processing/symbols/sample_postprocessor.hpp)) -
  the per-frame resolve + predicate + sink pass. Symbolization is cached at most
  once per unique IP; resolver exceptions are swallowed (anonymous frame). It
  emits `processed_sample`/`processed_frame` with string views into the symbol
  cache. It moves symbol resolution and filtering OUT of the recording hot path -
  the whole reason it exists as a separate pass.
- `slice_emitter`
  ([slice_emitter.hpp](../../src/cpu_snap/processing/timeline/slice_emitter.hpp)) -
  the timeline-shaping pass that converts per-tid `raw_sample` vectors into
  `emitted_slice` records (nested flame-graph view or leaf-per-sample view) and
  hands each to a sink. `emitted_slice` is the backend-neutral output boundary:
  the Perfetto writer (and any pprof/collapsed sink) maps `emitted_slice` to its
  native records. This is the single extension point for a new output format -
  see [../extending.md](../extending.md).

Why postprocessor and slice_emitter are separate passes, not one: they have
distinct jobs (predicate/symbol/sink delivery vs timeline/stack-diff shaping) and
distinct outputs. They DO share one thing - library-frame dropping - and that is
factored into `frame_filters`
([frame_filters.hpp](../../src/cpu_snap/processing/symbols/frame_filters.hpp)),
a set of `frame_predicate` builders (`make_main_exe_only_filter`,
`make_library_exclude_filter`) used by both. The shared filter is the reason
`frame_filters` exists as its own component rather than living inside either
consumer: it is the genuine common denominator, and duplicating it in two places
was the redundancy the audit (section 11) flagged and resolved by extraction.
`coalesce_frames_by_name` (a free function alongside the postprocessor) rewrites
IPs to one canonical IP per function name for callers that want name-level rather
than address-level grouping.

---

## 11. Why every component exists: the audit conclusion

The component set was audited for redundancy. The result: a small number of
merge candidates and no drops. The merge candidates
are single-owner leaf types (`sampler_capabilities` into `sampler`,
`triggers::signal::snapshot_ring` into `triggers::signal::backend`, `frame_limits` into `raw_sample`) kept
separate as cohesive units; the other investigate items were the offline
frame-filter sharing (already largely unified via `frame_filters`) and
`src/core/log` (since removed). The architecture is justified - the redundancy that
exists is minor and was addressed by extraction, not deletion. The per-component
purpose is in
[../components.md](../components.md); the justification summary, grouped by layer:

`src/core/` (the data plane, no `libdw`):

- `records` - single source of truth for the wire format. 21 consumers; cannot be
  dropped.
- `types` - thin function-alias ABI (`sink_fn`, `event_fn`) that keeps heavy
  includes out of public headers. Exists precisely to decouple.
- `chunk_framer` / `reader` - symmetric producer/consumer of the format.
- `perf_ring` - RAII owner of the `perf_event_open` mmap-ring, the substrate
  `triggers::perf::kernel_drain` builds on. Exists to avoid duplicating the ring drain
  logic.
- `proc_maps` (offline) and `dl_module_watcher` (streaming-time) - the two
  module-enumeration paths; distinct because one runs after the fact over
  `/proc/self/maps` and the other observes live via `dl_iterate_phdr`.
- `flat_tid_map` - the per-tid hot-path container; exists for the streaming
  budget.
- `thread_comm` / `time` - small cross-cutting `/proc` and clock helpers.

`src/sampling/` capture and its seam:

- `sampler` - the one public driver; the entry point.
- `i_trigger` (abstract class) + `streaming_pipeline` (concrete sink) - the seam
  from sections 2 and 3. Each carries a distinct role: the strategy contract
  and the hot-path sink.
- `triggers::perf::backend` / `triggers::signal::backend` - the two strategies.
- `triggers::perf::kernel_drain` - the perf kernel-facing state and decoder, kept separate from
  the driver so the backend wrapper is thin.
- `triggers::signal::snapshot_ring` - the AS-safe snapshot record + SPSC ring; tightly coupled to
  `triggers::signal::backend` (a noted merge candidate) but a distinct concern (the
  lock-free queue).
- `dwarf_unwinder` - shared by both backends; the single unwinder per worker.
- `raw_sample` / `frame_limits` - the leaf data type and the decoupling constant
  from section 8.
- `unwind_mode` / `sampler_capabilities` - the stack-walk strategy enum and the
  kernel/CPU probe. `sampler_capabilities` is tightly coupled to `sampler`
  (a noted merge candidate) but currently distinct.

`src/sampling/` streaming (live encode):

- `streaming_pipeline` / `temporal_diff_filter` / `frame_trie` - the live encode
  and its compression (section 9).

`src/processing/` offline render:

- `symbolizer` / `sample_postprocessor` / `slice_emitter` / `frame_filters`
  (+ `emitted_slice`, `coalesce_frames_by_name`) - the offline render (section
  10). `frame_filters` is the extracted shared filter, the audit's one real
  de-duplication.

The tightly-coupled-but-kept components (`triggers::signal::snapshot_ring` into `triggers::signal::backend`,
`sampler_capabilities` into `sampler`) are flagged as future merge candidates, not
drops: they are real, distinct concerns today, and merging them is a cleanup with
no behaviour change to defer until it pays for itself.

---

## 12. Cross-cutting invariants to preserve

A checklist of the non-obvious things a maintainer must not break:

- The per-sample path stays concrete. Do not put a `std::function` or a virtual
  call on `streaming_pipeline::on_sample` or inside a backend's decode loop
  (section 2).
- Virtual dispatch stays per-batch. Keep the per-sample sink out of the
  `i_trigger` interface; `drain_into` takes the concrete `streaming_pipeline` (section 3).
- One `dwarf_unwinder` per worker, attached once. Do not share it across threads
  or re-attach per call (section 6).
- The signal handler stays async-signal-safe. Anything new in it must be on the
  audited AS-safe list; defer everything else to the worker (section 5).
- Signal-backend teardown must flush pending RT signals and defer ring frees to
  the destructor (section 7).
- On the wire, `NEW_NODE` precedes the `SAMPLE` that references it; fixed record
  bodies keep their `static_assert`ed sizes (section 9).
- Leaf data headers (`raw_sample`, `frame_limits`, `core::types`) stay free of
  the capture/decode machinery so their many consumers do not transitively
  compile it (section 8).
- The sampler always streams to a mandatory sink; streaming supports every
  `unwind_mode`, including `dwarf` (section 9).
