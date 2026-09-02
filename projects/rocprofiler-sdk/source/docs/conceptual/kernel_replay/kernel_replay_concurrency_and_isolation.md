(kernel-replay-concurrency)=
# Kernel Replay — Concurrency and Isolation

Kernel replay re-executes a single kernel dispatch several times and restores device memory between
executions so that every pass observes identical inputs. That only works if nothing else mutates the
agent's device memory between the moment the snapshot is taken and the moment the last pass
finishes. This page describes how that window is isolated, what the isolation deliberately does not
cover, and why the waits inside the window abort instead of hanging.

Everything here is driven from the HSA `WriteInterceptor` in
`source/lib/rocprofiler-sdk/hsa/queue.cpp`, using the shared window primitives in
`source/lib/rocprofiler-sdk/hsa/replay_window.cpp`. Replay runs synchronously on the thread that
submitted the dispatch — there is no replay worker thread.

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
- **Agent-wide drain.** `replay_drain_agent_or_fatal()` waits until no queue on the agent has an
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

## Bounded waits and the abort convention

Both drains in the replay window are bounded, and exceeding the bound is fatal rather than a warning.
This is a deliberate choice for a beta feature: a stuck handler or a stuck queue should fail loudly
and immediately, rather than hanging the application indefinitely or — worse — silently proceeding to
snapshot or restore memory that is still being mutated.

| Wait | Bound | On expiry |
|---|---|---|
| `replay_drain_or_fatal()` — per-pass async handler drain | up to 12 slices of `Queue::sync()`, roughly 60 s total | `ROCP_FATAL` |
| `replay_drain_agent_or_fatal()` — agent-wide drain | 60 s deadline, polled every ~2 ms outside the queue-map lock | `ROCP_FATAL` |
| `Queue::sync()` — one drain slice | 5 s HSA signal timeout hint | returns `false` and warns; `replay_drain_or_fatal()` takes another slice, teardown callers proceed |
| Queue profiling setup signal waits (adjacent to, not inside, the replay window) | 1 s timeout hint — three attempts in one path, a single attempt in the other | `ROCP_FATAL` |

Each expired `Queue::sync()` slice logs its own timeout warning naming the number of kernels still
active, so a slow drain leaves a trail before the 60 s bound is reached.

The contrast with `Queue::sync()` is the point. `Queue::sync()` is also used at teardown, where
warning once and proceeding is the right behavior; a replay pass must not proceed on a handler that
has not finished. `replay_drain_or_fatal()` therefore layers a retry loop over `Queue::sync()` to
extend the bound and then aborts, instead of accepting `sync()`'s warn-and-continue result.

The drain barrier on the replaying queue is the one wait that is unbounded (`UINT64_MAX` timeout).
It fences work the application itself submitted on this queue, which the runtime is expected to
complete.

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
| Per-agent reader/writer lock | `source/lib/rocprofiler-sdk/hsa/replay_window.cpp` | `agent_replay_mutex()` |
| Writer lock acquisition | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `replay_guard` in `WriteInterceptor` |
| Reader lock on the non-replay path | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `replay_reader_guard` in `WriteInterceptor` |
| Replay activity check | `source/lib/rocprofiler-sdk/kernel_replay/replay_callbacks.cpp` | `has_active_replay_contexts()` |
| Per-pass handler drain | `source/lib/rocprofiler-sdk/hsa/replay_window.cpp` | `replay_drain_or_fatal()` |
| Agent-wide drain | `source/lib/rocprofiler-sdk/hsa/replay_window.cpp` | `replay_drain_agent_or_fatal()` |
| One drain slice | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `Queue::sync()` |
| Agent-scoped inventory | `source/lib/rocprofiler-sdk/kernel_replay/memory_tracker.cpp` | `snap_inventory()` |
| Localized context scopes | `source/lib/rocprofiler-sdk/kernel_replay/local_context.hpp` | `scoped_local_context_control`, `set_toggles_armed()` |
| Localized context consumer | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `local_context_has_overrides()` call in `process_packet_batch` |
