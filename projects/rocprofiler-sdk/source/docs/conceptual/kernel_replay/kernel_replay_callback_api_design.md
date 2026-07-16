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
