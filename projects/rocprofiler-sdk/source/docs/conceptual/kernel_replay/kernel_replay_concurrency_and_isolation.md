(kernel-replay-concurrency)=
# Kernel Replay — Concurrency and Isolation

Kernel replay re-executes a single kernel dispatch several times and restores device memory between
executions so that every pass observes identical inputs. That only works if nothing else mutates the
agent's device memory between the moment the snapshot is taken and the moment the last pass
finishes. This page describes how that window is isolated, what the isolation deliberately does not
cover, and why the waits inside the window abort instead of hanging.

The window itself is `run_replay_window()` in
`source/lib/rocprofiler-sdk/kernel_replay/queue_hooks.cpp`, driven from the HSA `WriteInterceptor` in
`source/lib/rocprofiler-sdk/hsa/queue.cpp`. The interceptor decides *whether* a dispatch is eligible
and supplies two callables — submit this dispatch once, and put a barrier on the queue — and the
window does everything else. Replay runs synchronously on the thread that submitted the dispatch:
there is no replay worker thread, and the interceptor does not return until the last pass has
completed.

The per-agent lock and the reentrancy marker live together in
`source/lib/rocprofiler-sdk/kernel_replay/replay_diagnostics.cpp`, because both dispatch paths — the
replay window and an ordinary dispatch — consult them, and neither is meaningful without the other.

## The replay window

A replayed dispatch expands into the following sequence, all on the submitting thread:

```text
take per-agent WRITER lock
  capture and suppress the application's completion signal
  submit a barrier packet on this queue and wait on it   (queue drain)
  poll every queue on this agent until no async handler is in flight (agent-wide drain)
  snap()                                                 (device -> host)
  install the localized-context-control guard
  for each pass:
      PASS PHASE_ENTER
      submit the dispatch
      drain this pass's async completion handler
      PASS PHASE_EXIT
      ask the tool whether to continue; break if not
      restore()                                          (host -> device)
  fire the application's completion signal exactly once
release per-agent WRITER lock
```

`restore()` runs only when another pass follows. The last executed pass deliberately leaves device
memory in the state the application expects, so no restore follows the loop break.

## Isolation model

Isolation has three independent layers. None of them is sufficient alone.

### 1. Per-agent reader/writer serialization

The gate is a `std::shared_mutex` per agent, obtained from `agent_replay_mutex()` and keyed on
`rocprofiler_agent_id_t::handle`.

| Participant | Lock taken | Held across |
|---|---|---|
| A replayed dispatch | **unique** (writer) | the entire drain → snap → passes → restore window |
| A non-replay dispatch, while any replay service is active | **shared** (reader) | its own submit |
| A non-replay dispatch, with no replay service configured | none | — |

The writer lock excludes both other replays on the same agent and ordinary dispatches on the same
agent. The reader lock is what makes the second half of that true: without it, a normal dispatch
could submit into the middle of a replay window and have its device writes reverted by the next
`restore()`. Because ordinary dispatches do not conflict with one another, they share the reader
lock and still run concurrently; a pending replay writer simply waits for the in-flight submits to
finish and blocks new ones from entering the window.

The reader side is gated on `kernel_replay::has_active_replay_contexts()`, so a run with no replay
service configured takes no lock at all. That check itself is fronted by a process-global atomic
flag set when a tool configures the service, so the common case is one relaxed atomic load rather
than a walk of the active contexts.

The reader lock bounds *submission* only. It says nothing about GPU work that was already submitted
and is still executing — that is what the drains below are for.

### 2. Agent-wide drain

Two drains run before `snap()`:

- **Queue drain.** A barrier packet is submitted on the replaying queue and waited on, fencing the
  CPU against all prior GPU work on that queue.
- **Agent-wide drain.** `replay_drain_agent_or_decline()` waits until no queue on the agent has an
  async completion handler in flight. Sibling queues (other HIP streams) can have kernels in flight
  that would mutate device memory during snapshot and restore. The writer lock stops other threads
  from *starting* a replayed dispatch, because every kernel dispatch passes through that gate, but it
  cannot un-submit work that is already on a sibling queue.

The agent-wide drain deliberately does not hold the queue-map lock across its wait.
`QueueController::iterate_queues` holds that lock for the duration of its callback, so a blocking
per-sibling drain inside the callback would stall stream creation and destruction for the whole
wait. Instead the drain polls each queue's in-flight async count under a brief read lock and sleeps
between polls, so the map lock is held only for the duration of the poll itself. This is also safe
against concurrent queue destruction: a `Queue` is only dereferenced while the read lock is held
(`destroy_queue` erases under the write lock), and the live set is re-read on every poll. Because
the writer lock blocks new dispatches on the agent, in-flight work only decreases and the poll
converges.

### 3. Agent-scoped snapshots

`memory_snapshot::snap(agent)` captures only the allocations owned by the replaying agent. The
memory tracker tags each allocation with its owning agent at allocation time (from
`hsa_amd_pointer_info::agentOwner`), and `snap_inventory(agent)` filters on that tag.

Combined with the per-agent lock, this makes multi-GPU replay genuinely concurrent: replays on
different agents take different mutexes, snapshot disjoint memory, and proceed at the same time.

## Async completion handler drain

Each pass drains its async completion handler before PASS `PHASE_EXIT`, before the tool's
continue-decision, before `restore()`, and before the next submit.

The handler runs on a separate HSA thread. It reads hardware counters, emits records, releases
signals back to the pool, and drops correlation-id references. Proceeding while it is still running
would race its record delivery and reuse buffers and signals it still holds. Exactly one handler is
in flight per pass — the loop drains before each submit under the agent writer lock — and that
invariant is asserted rather than assumed. Draining the handler also implies the GPU work has
completed, so the loop needs no separate per-pass GPU fence.

## Bounded waits and what happens when they expire

Every drain in the replay window is bounded. Exceeding the bound declines the replay: the dispatch
runs once, with its original completion signal, and the reason is reported. It does not terminate the
process.

That is a change from the original beta behavior, which aborted. The reasoning behind aborting was
that a stuck drain indicates a state too questionable to proceed from — which is right about the
*replay* and wrong about the *process*. A drain that does not converge is not necessarily a bug at
all: the HSA full profile requires an agent to make forward progress on several queues concurrently,
which is precisely what licenses an application to keep two co-dependent kernels resident on one
agent. A persistent kernel, a spin-waiting cooperative kernel, and a collective whose peer has not
reached the same point all sit in the drain indefinitely while behaving exactly as specified. Killing
a multi-hour job under a scheduler because one of its dispatches could not be profiled is a worse
outcome than not profiling that dispatch.

| Wait | Bound | On expiry |
|---|---|---|
| `replay_drain_or_decline()` — per-pass async handler drain | up to 12 slices of `Queue::sync()`, roughly 60 s total | break the pass loop, report `pass_drain_stuck` |
| `replay_drain_agent_or_decline()` — agent-wide drain | 60 s deadline, polled every ~2 ms outside the queue-map lock | decline, report `agent_drain_stuck` |
| Queue drain barrier — fences prior GPU work on the replaying queue | 12 slices of a 5 s HSA signal wait | decline, report `queue_drain_stuck` |
| `Queue::sync()` — one drain slice | 5 s HSA signal timeout hint | returns `false` and warns; `replay_drain_or_decline()` takes another slice, teardown callers proceed |
| Queue profiling setup signal waits (adjacent to, not inside, the replay window) | 1 s timeout hint — three attempts in one path, a single attempt in the other | `ROCP_FATAL` |

Each expired `Queue::sync()` slice logs its own timeout warning naming the number of kernels still
active, so a slow drain leaves a trail before the 60 s bound is reached.

The contrast with `Queue::sync()` is still the point. `Queue::sync()` is also used at teardown, where
warning once and proceeding is right; a replay pass must not proceed on a handler that has not
finished. `replay_drain_or_decline()` therefore layers a retry loop over `Queue::sync()` to extend the
bound, and then stops the replay rather than accepting `sync()`'s warn-and-continue result.

On the queue-drain path the barrier packet is still queued when the wait expires, and it holds a
reference to the drain signal. That signal is deliberately leaked rather than destroyed: the packet
processor would otherwise decrement freed memory. Leaking one signal on a path already reporting a
stuck queue is the lesser fault.

One failure is still fatal: a `restore()` that fails partway through. There is no correct
continuation from a partial restore — some regions hold pre-kernel bytes and others post-kernel
bytes, the mix is unknown, and nothing can reconstruct it. Handing that state to the application
would corrupt its results silently, which is worse than terminating. Reaching it requires a
host→device DMA on a live allocation to fail, which is a device-level fault rather than a policy
decision.

## Reentrancy: a tool callback that launches a kernel

The per-agent lock is a `std::shared_mutex`, which is not recursive. A tool callback invoked inside
the replay window — CONFIG, PASS, `pass_count_cb`, or `replay_continue_cb` — that launches GPU work
submits a dispatch from the one thread already holding the writer lock. Acquiring the reader side
then blocks forever, and unlike the drains there is no bound: the process ends up parked in a
blocking lock acquisition.

The dispatch path therefore asks `dispatch_lock_for()` which lock to take rather than taking one
directly, and it answers "none" — consulting a thread-scoped marker — when the submitting thread is
inside its own replay window on that agent. The nested dispatch's writes land inside the
snapshot window, so the counters for the surrounding replayed dispatch are not trustworthy — that is
recorded on the outcome (`reentrancy=1`) and logged once per process with an explanation. Reporting
an untrustworthy measurement is a better outcome than a process that cannot be interrupted.

A *nested replay request* is the other half of the same problem, and it cannot be handled by skipping
a lock: a window inside a window on the same agent has no meaning, and the writer side has no
skippable path. Such a dispatch is declined with `reentrant_dispatch` and run once, and the enclosing
window's outcome is marked.

The marker carries two scopes, and both are load-bearing.

**Thread scope.** A replay window on one thread must not cause an unrelated dispatch on another
thread to skip the lock, which would drop exactly the isolation the lock exists to provide.

**Agent scope.** A callback that launches work on a *different* GPU contends for a different mutex,
which this thread does not hold, so that dispatch keeps its reader lock. An agent-blind marker would
skip it and silently break isolation against a concurrent replay on that second agent.

Agent scope is also why the marker is a stack rather than a single value. Because a replay request on
a second agent is *not* declined, a callback running inside agent A's window can legitimately open a
window on agent B from the same thread. With one slot, entering B's window would overwrite the record
of A's and leaving B's would clear it, after which the thread no longer knows it is inside A's window
and the next dispatch on A blocks forever on the reader lock — a multi-GPU-only failure. Each window
gets its own frame, so `in_replay_window()` stays true for every enclosing window and each frame
carries its own `reentrancy` flag. Nesting is bounded by the agent count, since a second window on an
agent already on the stack is declined before its lock is taken.

## Admission control

Two questions are answered before any device→host traffic is issued, both from the tracker alone.

**Would the snapshot actually cover the application's data?** `untracked_device_memory(agent)`
reports live virtual-memory mappings and GPU-resident allocations that are not snapshottable. A live
virtual-memory mapping declines by default: nothing maps virtual memory unless the application asked
for it, so it is unambiguous evidence that data is outside the snapshot. Non-coarse GPU-resident pool
allocations only warn, because runtime-internal allocations land in the same bucket.

**Would it fit and finish?** `estimate_footprint(agent)` reports what a snapshot would cost without
copying anything. Over the configured budget declines with a diagnostic, rather than surfacing later
as a `std::bad_alloc` partway through the capture. Merely expensive warns with the projected transfer
time, so a job that appears to hang is explained instead of mysterious.

Both are reported per dispatch on a single `[kernel-replay]` line, together with the outcome, the
decline reason, the footprint, and the snap/restore timing.

## What is not isolated

Two gaps are known and marked as follow-up work in the source rather than papered over.

**Async SDMA copies.** `hsa_amd_memory_async_copy` and its variants are not kernel dispatches, so
they never reach the `WriteInterceptor` and never pass through the per-agent replay gate. The
agent-wide drain closes the *kernel* half of the race, but a thread can still run an SDMA copy
against shared device memory inside another thread's replay window. Serializing those is tracked as
a separate change.

**HIP graphs.** Graph launches are not replayed at all; see
[Memory snapshot and restore](kernel_replay_memory_snapshot.md#hip-graphs) for the two-tier warn
and abort behavior.

## Localized context control and thread scope

When a tool toggles contexts per pass, the decisions are recorded in a thread-local override map
that lives only for the duration of the replay loop; global context state is never modified. Two
nested thread-local scopes are involved, both managed by the SDK:

- **Loop scope** (`scoped_local_context_control`) owns the override map for the whole loop, which is
  what gives toggles their sticky-across-passes semantics.
- **Arm window** (`set_toggles_armed`) makes the tool-facing start/stop callbacks legal only while
  the tool's PASS `PHASE_ENTER` callback is running. It is armed and disarmed through a scope guard,
  so a throwing tool callback cannot leak the armed state.

Because the map is thread-local and replays on an agent are serialized by the per-agent lock, a loop
never nests on a thread and only the replaying thread's dispatches observe the overrides. Service
consumers query `local_context_override()` at dispatch time, fronted by
`local_context_has_overrides()` so an ordinary dispatch pays a single thread-local read.

See [Callback API](kernel_replay_callback_api.md#localized-context-control) for the tool-facing
contract.

## Source reference

All paths are relative to `projects/rocprofiler-sdk/`.

| Component | File | Symbol |
|---|---|---|
| The replay window | `source/lib/rocprofiler-sdk/kernel_replay/queue_hooks.cpp` | `run_replay_window()` |
| What the window needs from the interceptor | `source/lib/rocprofiler-sdk/kernel_replay/queue_hooks.hpp` | `replay_dispatch_t` |
| Replay eligibility gate | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `has_kernel_replay` branch in `WriteInterceptor` |
| Per-agent reader/writer lock | `source/lib/rocprofiler-sdk/kernel_replay/replay_diagnostics.cpp` | `agent_replay_mutex()` |
| Writer lock acquisition | `source/lib/rocprofiler-sdk/kernel_replay/queue_hooks.cpp` | `replay_guard` in `run_replay_window()` |
| Lock decision on the non-replay path | `source/lib/rocprofiler-sdk/kernel_replay/replay_diagnostics.cpp` | `dispatch_lock_for()` |
| Replay activity check | `source/lib/rocprofiler-sdk/kernel_replay/replay_callbacks.cpp` | `has_active_replay_contexts()` |
| Per-pass handler drain | `source/lib/rocprofiler-sdk/kernel_replay/queue_hooks.cpp` | `replay_drain_or_decline()` |
| Agent-wide drain | `source/lib/rocprofiler-sdk/kernel_replay/queue_hooks.cpp` | `replay_drain_agent_or_decline()` |
| One drain slice | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `Queue::sync()` |
| Reentrancy marker | `source/lib/rocprofiler-sdk/kernel_replay/replay_diagnostics.cpp` | `enter_replay_window()`, `in_replay_window()` |
| Restore identity gate | `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp` | `region_is_restorable()` |
| Admission control and outcome reporting | `source/lib/rocprofiler-sdk/kernel_replay/replay_diagnostics.cpp` | `check_untracked()`, `check_admission()`, `log_replay_outcome()` |
| Agent-scoped inventory | `source/lib/rocprofiler-sdk/kernel_replay/memory_tracker.cpp` | `snap_inventory()` |
| Untracked-memory accounting | `source/lib/rocprofiler-sdk/kernel_replay/memory_tracker.cpp` | `untracked_device_memory()` |
| Localized context scopes | `source/lib/rocprofiler-sdk/kernel_replay/local_context.hpp` | `scoped_local_context_control`, `set_toggles_armed()` |
| Localized context consumer | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `local_context_has_overrides()` call in `process_packet_batch` |
