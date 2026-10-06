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
kernel timing, PC sampling, ATT, or anything else, and decides per pass, in each service's own
dispatch callback, what that service collects.

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
- **Tool-provided `replay_pass_count` during CONFIG `PHASE_ENTER`.** NULL means opt out of replay for
  that dispatch. Returning 0 requires `replay_continue` (indefinite loop). Returning 1 skips
  snapshot because a single pass is the ordinary path.
- **No per-pass service API on the payload.** Every context a tool wants on any pass is
  configured and started globally, before replay, exactly as it would be without replay, and each
  dispatch-scoped service's own dispatch callback decides per pass whether it collects. See
  [Per-pass service selection](#per-pass-service-selection).
- **No pass-count environment variable.** A tool derives N itself — `rocprofv3`, for example, from
  its `--pmc` groups per agent.

`ROCPROFILER_KERNEL_REPLAY_SNAPSHOT` and `ROCPROFILER_KERNEL_REPLAY_RESTORE` are TODOs in
`fwd.h` for tool visibility into those phases; they are not implemented.

## Per-pass service selection

A pass is a separate submit of the dispatch through the queue interceptor, on the replaying thread.
With the per-queue callback registry gone, every service is driven by an explicit enter hook at that
submit, and each dispatch-scoped hook calls the tool's dispatch callback for the service: dispatch
counting and SPM ask for a counter configuration, dispatch thread trace asks for control flags. So
the tool is consulted once per pass, per service, on the thread that ran `PASS` `PHASE_ENTER`, and
answering "nothing on this pass" there is the per-pass switch. The tool learns the pass from the
`PASS` callback and keeps it in thread-local state; no SDK state is involved.

The first version of this API also carried localized context toggles on the PASS payload
(`replay_start_context` / `replay_stop_context`): a thread-scoped override map, armed during PASS
`PHASE_ENTER` and consulted inside each service's dispatch path, that masked already-active contexts
per pass without touching global state. It was an interim layer for the per-queue callback
registry, which offered no per-dispatch decision point of its own, and it is removed now that the
explicit hooks provide one. Three properties of the toggles made the dispatch callback the better
home for the decision:

- **They duplicated a decision the tool already makes.** A masked counters context and a dispatch
  callback that returns no configuration produce the same instrumentation; the mask just made the
  SDK skip the question.
- **Their coverage could not be uniform.** PC sampling and device counting are agent-wide, so a
  toggle naming one of their contexts reported success and changed nothing. A dispatch callback
  exists exactly for the services that can be scoped to a pass.
- **Every consumer had to opt in.** Each service's dispatch path carried an override check, and the
  checks diverged (counters and SPM honored both directions, thread trace only a stop). Without the
  layer there is nothing to keep consistent.

Kernel dispatch tracing has no per-dispatch tool callback and reports every pass; tools that want a
single record per dispatch keep one per `dispatch_id` themselves.

## Callback flow (as implemented)

```
CONFIG PHASE_ENTER
  tool sets: replay_pass_count (tool-provided), optionally replay_continue
  SDK calls replay_pass_count (if set) to get N
    - replay_pass_count left null -> dispatch runs once, no replay (opt-out)
    - N == 1 -> ordinary path (no snapshot)
  SDK validates: N==0 && replay_continue==NULL -> error

  take per-agent writer lock
  drain queue; agent-wide sibling drain
  snapshot device memory (full in-memory copy; hashing is not used)

  loop (i = 0..N, or indefinitely if N==0):
    PASS PHASE_ENTER  (current_pass=i, total_passes=N)
    submit kernel     (each service's enter hook calls the tool's dispatch callback)
    drain async completion handler
    PASS PHASE_EXIT
    if replay_continue provided and returns 0 -> break
    if not last pass -> restore device memory

CONFIG PHASE_EXIT
fire application's original completion signal
release writer lock
```

Replay serializes dispatches **on the agent** through the per-agent reader/writer lock described in
[Concurrency and isolation](kernel_replay_concurrency_and_isolation.md). It does **not** call the
process-wide `QueueController::enable_serialization()` / `batch_packets` path used by counters, SPM,
and thread trace. Other agents are not blocked.

Passes are serialized within the loop as well: each pass drains its async completion handler before
the next `PASS` `PHASE_ENTER`, so two passes of the same dispatch never overlap. That is what makes
the between-pass restore safe, and it is why a replayed dispatch costs roughly N times a normal one
plus the snapshot and restore copies.

## Snapshot design choice

This design copies every tracked region into host RAM and writes it all back. It does not hash dirty
pages and does not spill snapshots to disk. Host-side and/or device-side hashing of dirty regions is
expected in a future version so restore cost tracks bytes mutated rather than the whole footprint.
See [Memory snapshot](kernel_replay_memory_snapshot.md).

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
