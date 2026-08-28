(kernel-replay-design)=
# Kernel Replay Callback Tracing API Design

> **Current behavior** is documented in
> [Callback API](kernel_replay_callback_api.md),
> [Concurrency and isolation](kernel_replay_concurrency_and_isolation.md), and
> [Memory snapshot](kernel_replay_memory_snapshot.md).
> This page is the design rationale: why replay was decoupled from counter collection, which
> prototype choices were dropped, and what remains open.

## Overview

Kernel replay is a **standalone callback tracing service** under
`ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY`. Tools configure it through
`rocprofiler_configure_callback_tracing_service()` — no new `rocprofiler_configure_*` function is
needed.

That decouples replay from hardware counter collection, so a tool can use replay for counters,
kernel timing, PC sampling, ATT, or anything else, and can enable or disable other contexts per
pass through localized context control.

## Motivation

The previous API (`rocprofiler_configure_kernel_replay_counting_service()`, the counting-service
prototype) was:

- Tightly coupled to dispatch counter collection
- Mutually exclusive with regular dispatch counting on the same context
- Limited to fixed pass counts (no indefinite loop / early exit)
- Unable to give tools per-pass control over which services are active
- Used file-backed dirty-page hashing that this design does not ship

## API surface (as designed)

The payload, operations, and pass-count table in
[Callback API](kernel_replay_callback_api.md) are the contract. Shape decisions that were
deliberate:

- **One flat struct, no unions.** CONFIG and PASS share
  `rocprofiler_callback_tracing_kernel_replay_data_t`; unused fields are zero.
- **Tool-provided `pass_count_cb` during CONFIG `PHASE_ENTER`.** NULL means opt out of replay for
  that dispatch. Returning 0 requires `replay_continue_cb` (indefinite loop). Returning 1 skips
  snapshot because a single pass is the ordinary path.
- **Localized start/stop as function pointers on the PASS payload**, mirroring
  `rocprofiler_start_context` / `rocprofiler_stop_context`, rather than a new public API.
  Contexts are configured and started globally before replay; the toggles only mask which
  already-active contexts participate in each pass.
- **No pass-count environment variable.** A tool (including the stacked `rocprofv3` integration)
  derives N itself — for example from `--pmc` groups per agent.

`ROCPROFILER_KERNEL_REPLAY_SNAPSHOT` and `ROCPROFILER_KERNEL_REPLAY_RESTORE` are TODOs in
`fwd.h` for tool visibility into those phases; they are not implemented.

## Localized context control

The pointers are **wired**. During PASS `PHASE_ENTER` the SDK populates
`replay_local_enable_context_cb` / `replay_local_disable_context_cb`. Semantics:

- Only legal during PASS `PHASE_ENTER`.
- Sticky across passes (avoids reprogramming PC sampling hardware on every pass).
- Scoped to the replay loop; global context state is never modified.

Routing of the downcalls uses a thread-scoped override map (`scoped_local_context_control` +
`set_toggles_armed`) installed around the replay loop. That is SDK-internal. If a tool-facing handle
parameter proves cleaner, the signature may gain one — that is the one shape decision still open.
(`pass_count_cb` and `replay_continue_cb` are SDK→tool upcalls and need no such routing.)

Counter collection, PC sampling, SPM, and ATT consult the override at dispatch time. Kernel dispatch
tracing drops disabled contexts from the pass's tracing data.

Kernel replay is **not** gated on removing the queue callback registration mechanism. That removal
would make per-pass enable/disable cleaner and is a planned improvement, but the feature works
without it.

## Callback flow (as implemented)

```
CONFIG PHASE_ENTER
  tool sets: pass_count_cb (tool-provided), optionally replay_continue_cb
  SDK calls pass_count_cb (if set) to get N
    - pass_count_cb left null -> dispatch runs once, no replay (opt-out)
    - N == 1 -> ordinary path (no snapshot)
  SDK validates: N==0 && replay_continue_cb==NULL -> error

  take per-agent writer lock
  drain queue; agent-wide sibling drain
  snapshot device memory (full in-memory copy; hashing is not used)

  loop (i = 0..N, or indefinitely if N==0):
    PASS PHASE_ENTER  (current_pass=i, total_passes=N; local start/stop armed)
    submit kernel
    drain async completion handler
    PASS PHASE_EXIT
    if replay_continue_cb provided and returns 0 -> break
    if not last pass -> restore device memory

CONFIG PHASE_EXIT
fire application's original completion signal
release writer lock
```

Replay serializes dispatches **on the agent** through the per-agent reader/writer lock described in
[Concurrency and isolation](kernel_replay_concurrency_and_isolation.md). It does **not** call the
process-wide `QueueController::enable_serialization()` / `batch_packets` path used by counters, SPM,
and thread trace. Other agents are not blocked.

## Snapshot design choice

An earlier prototype hashed dirty pages and stored snapshots on disk. This design copies every
tracked region into host RAM and writes it all back. Host-side and/or device-side hashing of dirty
regions is expected in a future version so restore cost tracks bytes mutated rather than the whole
footprint. See [Memory snapshot](kernel_replay_memory_snapshot.md).

## Concurrency hardening (implemented)

The replay loop originally matched the prototype (single agent, single thread). The following are
implemented; details are on
[Concurrency and isolation](kernel_replay_concurrency_and_isolation.md):

1. Per-agent reader/writer lock for the drain → snap → passes → restore window.
2. Per-agent snapshot scoping (`hsa_amd_pointer_info::agentOwner`).
3. Pool-type filter: coarse-grained device VRAM only (kernarg, host, fine-grained, executable
   excluded).
4. Teardown finalization guard on the alloc/free wrappers.
5. Agent-wide drain of sibling queues before snapshot.
6. Per-pass async completion handler drain (`replay_drain_or_fatal`).
7. HIP graph warn-once vs fatal at the replay gate.
8. Incomplete snapshot declines replay.

### Remaining: async-copy race

`hsa_amd_memory_async_copy` is not a kernel dispatch, so the per-agent replay lock never blocks it,
and it is not intercepted unless mem-copy tracing is enabled. Serializing async copies against an
in-progress replay (including waiting on the copy's completion signal, not just the submit) is
follow-up work.

## Future work

- **Host-side and/or device-side hashing of dirty regions**, to cut bytes moved and the host-RAM
  duplication. Not in this design.
- **Replay-scoped per-agent quiesce** for async SDMA copies.
- **`ROCPROFILER_KERNEL_REPLAY_SNAPSHOT` / `RESTORE` operations** for tool visibility.
- **Pass info delivery to other service callbacks** without tool-side TLS.
- HIP graph replay.
- Multi-packet / multi-dispatch batches.
- Inlining `process_packet_batch` on the replay path (noted as a TODO in the loop).
