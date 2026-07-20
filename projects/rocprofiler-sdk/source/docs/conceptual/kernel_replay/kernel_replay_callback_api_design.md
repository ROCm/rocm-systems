# Kernel Replay Callback Tracing API Design

## Overview

Kernel replay is redesigned as a **standalone callback tracing service** under a new
domain `ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY`. This decouples replay from
hardware counter collection, allowing tools to use replay for arbitrary purposes
(counter collection, kernel timing statistics, PC sampling, ATT, etc.).

Tools configure replay through the existing `rocprofiler_configure_callback_tracing_service()`
infrastructure -- no new `rocprofiler_configure_*` function is needed.

## Motivation

The previous API (`rocprofiler_configure_kernel_replay_counting_service()`) was:
- Tightly coupled to dispatch counter collection
- Mutually exclusive with regular dispatch counting on the same context
- Limited to fixed pass counts (no indefinite loop / early exit)
- Unable to give tools per-pass control over which services are active

## API Surface

### Domain

```c
ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY  // added to rocprofiler_callback_tracing_kind_t
```

### Operations

```c
typedef enum rocprofiler_kernel_replay_operation_t
{
    ROCPROFILER_KERNEL_REPLAY_NONE   = 0,
    ROCPROFILER_KERNEL_REPLAY_CONFIG = 1,  ///< Pass-count / loop configuration
    ROCPROFILER_KERNEL_REPLAY_PASS,        ///< Per-pass begin/end notification
    ROCPROFILER_KERNEL_REPLAY_LAST,
} rocprofiler_kernel_replay_operation_t;
```

### Payload Struct

Flat struct (no unions). Which members are meaningful depends on the operation.

```c
typedef struct rocprofiler_callback_tracing_kernel_replay_data_t
{
    uint64_t                           size;
    rocprofiler_kernel_dispatch_info_t dispatch_info;

    // CONFIG fields (tool-provided callbacks the SDK invokes). The tool sets these
    // during CONFIG PHASE_ENTER. pass_count_cb: SDK calls it (if non-null) to get the
    // pass count; NULL => dispatch is not replayed. replay_continue_cb: optional
    // early-exit check called after each pass. Both receive dispatch_info + user_data.
    uint64_t (*pass_count_cb)(rocprofiler_kernel_dispatch_info_t dispatch_info,
                              rocprofiler_user_data_t            user_data);
    int (*replay_continue_cb)(rocprofiler_kernel_dispatch_info_t dispatch_info,
                              uint64_t                           current_pass,
                              uint64_t                           total_passes,
                              rocprofiler_user_data_t            user_data);

    // PASS fields (read-only, populated by SDK during PASS operation)
    uint64_t current_pass;
    uint64_t total_passes;

    // PASS fields: localized context control (see "Localized Context Control").
    // SDK-provided; the tool calls these during PASS PHASE_ENTER in lieu of the global
    // rocprofiler_start_context / rocprofiler_stop_context.
    rocprofiler_status_t (*replay_local_start_context_cb)(rocprofiler_context_id_t context_id);
    rocprofiler_status_t (*replay_local_stop_context_cb)(rocprofiler_context_id_t context_id);
} rocprofiler_callback_tracing_kernel_replay_data_t;
```

### Configuration

```c
rocprofiler_configure_callback_tracing_service(
    context_id,
    ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
    NULL, 0,
    tool_kernel_replay_callback,
    tool_data);
```

## Callback Flow

```
CONFIG PHASE_ENTER
  tool sets: pass_count_cb (tool-provided), optionally replay_continue_cb
  SDK calls pass_count_cb (if set) to get N
    - pass_count_cb left null -> dispatch runs once, no replay (opt-out)
  SDK validates: N==0 && replay_continue_cb==NULL -> error

  drain queue
  snapshot device memory

  loop (i = 0..N, or indefinitely if N==0):
    PASS PHASE_ENTER  (current_pass=i, total_passes=N)
    submit kernel
    wait for completion
    PASS PHASE_EXIT
    if replay_continue_cb provided and returns 0 -> break
    if not last pass -> restore device memory

CONFIG PHASE_EXIT
fire application's original completion signal
```

## Pass Count Semantics

| pass_count_cb  | replay_continue_cb | Behavior |
|----------------|--------------------|----------|
| NULL (not set) | (n/a)              | Not replayed: dispatch runs once, execution continues as usual (no snapshot) |
| returns N > 0  | NULL               | Fixed loop of exactly N passes |
| returns N > 0  | fn                 | Up to N passes; fn can break early |
| returns 0      | fn (required)      | Indefinite loop until fn returns 0 |
| returns 0      | NULL               | Error: ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT |

## Interaction with Other Services

For now: Kernel replay is independent of all other services. During a replay pass,
other services (counter collection, kernel tracing, PC sampling, ATT) fire
their callbacks as usual. The tool coordinates which service runs on each pass --
both through its own internal state and, where it wants a service active only on
some passes, through the localized context control described below.

## Localized Context Control

Tools frequently want different services active on different passes -- e.g.
collect kernel timing / PC sampling / ATT only once, but hardware counters on
every pass. Rather than calling the global `rocprofiler_start_context` /
`rocprofiler_stop_context` (which would leak into other, non-replayed
dispatches), the tool uses the localized equivalents provided in the PASS
payload:

```c
rocprofiler_status_t (*replay_local_start_context_cb)(rocprofiler_context_id_t context_id);
rocprofiler_status_t (*replay_local_stop_context_cb)(rocprofiler_context_id_t context_id);
```

These mirror the global API signatures. The tool calls them from within PASS
PHASE_ENTER.

Semantics:

- **Scoped to the replay loop.** Each context's pre-replay active/inactive state
  is restored once the loop completes; global context state is never modified.
- **Sticky across passes.** A context disabled in one pass stays disabled until
  it is re-enabled within the same loop. This avoids redundant work -- notably,
  PC sampling requires reprogramming hardware, so a non-sticky design would
  toggle the hardware on/off every pass (e.g. 16 passes where only 1 collects PC
  sampling would otherwise reprogram the hardware 15 extra times).

### Implementation notes

- Routing applies only to the context-control downcalls
  (`replay_local_start_context_cb` / `replay_local_stop_context_cb`): because their
  signatures mirror the global functions (context id only), the SDK routes each call
  to the in-flight replay loop via an internal, thread-scoped pointer set around each
  PASS callback. This is SDK-internal and not exposed to tools. If a tool-facing
  handle parameter proves cleaner in practice, the signature may gain one -- this is
  the one shape decision still open. (`pass_count_cb` and `replay_continue_cb` are
  plain SDK->tool upcalls and need no such routing.)
- Kernel dispatch tracing (timestamps): the interceptor already queries the
  active contexts and filters them, so disabling is trivial -- copy the
  active-context array for the pass and clear the entries to disable.
- Counter collection, PC sampling, and ATT: these services are already
  agent-specific, and because kernel replay serializes execution on the agent,
  the SDK can disable a context for that agent via internal calls during the
  replay loop. Localized toggling is therefore feasible with the current design.
- Kernel replay is NOT gated on removing the queue callback registration
  mechanism. That removal (a separate, older PR) would make per-pass enable/
  disable cleaner and is a planned future improvement, but the feature works
  without it.

The per-pass toggling is performed via the `replay_local_start_context_cb` /
`replay_local_stop_context_cb` function pointers already present in
`rocprofiler_callback_tracing_kernel_replay_data_t`; wiring them into the replay
loop lands in upcoming commits.

Upcoming commits: Services will be locally-toggleable with additional function 
callbacks tht are added to the `rocprofiler_callback_tracing_kernel_replay_data_t` structure. 

## Future Work

- **Memory snapshot/restore operations**: `ROCPROFILER_KERNEL_REPLAY_SNAPSHOT`
  and `ROCPROFILER_KERNEL_REPLAY_RESTORE` for tool visibility into
  snapshot/restore phases.
- **Pass info delivery to service callbacks**: mechanism for other service
  callbacks to know which replay pass they are in without tool-side TLS.

## Concurrency hardening

The replay loop originally matched the #7960 prototype (single agent, single
thread). The following have since been implemented:

1. **Per-agent replay lock (done).** A per-`hsa_agent_t` mutex spans the whole
   `drain -> snap -> passes -> restore -> completion` window, so two dispatches
   replaying on the same agent never interleave their snapshot/restore. Different
   agents use different mutexes and replay concurrently. See `agent_replay_mutex()`
   in `hsa/queue.cpp`.
2. **Per-agent snapshot scoping (done).** The memory tracker tags each allocation
   with its owning agent (`hsa_amd_pointer_info::agentOwner`) at allocation time;
   `memory_snapshot::snap(agent)` captures only that agent's allocations, so one
   GPU's replay never touches another GPU's memory.
3. **Pool-type filter (done).** Tracking is restricted to coarse-grained device
   (VRAM) allocations. Kernarg is excluded -- its pointer arguments are what fault
   when a torn/stale restore lands on them -- as are host / fine-grained pools
   (out of scope, precarious to restore). Classified from
   `hsa_amd_pointer_info::global_flags` (`KERNARG_INIT`, `COARSE_GRAINED`).
4. **Teardown finalization guard (done).** The HSA alloc/free wrappers outlive the
   tracker's static state, so HIP's own `__cxa_finalize` can call the free wrapper
   after rocprofiler's `destroy_static_objects()` has destroyed the inventory
   static object. `record_alloc` / `record_free` / `snap_inventory` early-out on
   `registration::get_fini_status() > 0`; otherwise they locked a freed mutex,
   which threw `std::system_error` into HIP's noexcept teardown and aborted (the
   abort then deadlocked the tool's signal handler, presenting as a hang).
5. **Agent-wide drain (done).** Before snapshotting, *every* queue on the replay
   agent is drained -- not just the replaying queue -- via
   `QueueController::iterate_queues` filtered by `Queue::get_agent`, calling
   `Queue::sync()` (waits on that queue's in-flight kernel count) on each. This
   fences kernels already in flight on *sibling* queues that would otherwise
   mutate device memory during snapshot/restore, and the per-agent lock blocks new
   replayed dispatches for the window. Verified: 2-thread same-agent transpose
   replay now runs to completion and emits output (previously it faulted mid-run).

### Remaining: async-copy race

The agent-wide drain closes the *kernel* half of the race. The residual is the
**async-copy path**: `hsa_amd_memory_async_copy` is not a kernel dispatch (so the
per-agent replay lock never blocks it) and is not intercepted at all unless
mem-copy tracing is enabled, so a thread can run an SDMA copy against the shared
device memory during another thread's replay window. On the 2-thread transpose
this now surfaces as a GPU fault (`address (nil)`) during *finalization* -- the
replay itself completes and produces correct output -- the same multi-thread
teardown fault previously noted, now isolated to the copy path. Not a regression
(#7960 faults identically); matches the design review ("per-agent locking solves
~99%; async copies are the remaining exception").

### Planned fix (remaining): async-copy serialization

Gate async copies on an in-progress replay, scoped so normal runs are unaffected.
Extend the per-agent replay state (currently just a mutex) with an
`atomic<bool> replay_active` flag set for the replay window; the
`hsa_amd_memory_async_copy` wrapper checks it: fast-path no-op when false; when
true it serializes the copy against the replay. Because async copies are not
wrapped unless mem-copy tracing is on, this also requires installing the wrapper
whenever replay is active.

Care points: the flag check and the act must be atomic w.r.t. replay start (a
per-agent reader/writer scheme -- copies as readers, replay as writer -- avoids
the check-then-race), and because an async copy is not complete when its wrapper
returns, correctness requires waiting on its completion signal rather than just
serializing the submit. This is a focused serializer / async-copy extension,
intended as its own change. Alternatively, fold the whole quiesce into the profiler
serializer's exclusive slot (also gated on `replay_active`).

### Related: multi-thread teardown

A prior multi-thread teardown SIGSEGV (exit 139) was noted; the finalization guard
above removes the tracker-side teardown fault. Any residual teardown crash under
concurrency is expected to be a facet of the multi-queue race above (a GPU fault
mid-run surfacing at teardown), not a separate bug.

### Still unwired

- **Localized per-pass context control.** The `replay_local_start/stop_context_cb`
  fields exist in the payload but are not yet wired (see "Localized Context
  Control" above); the free-function + TLS design is the intended shape.
