(range-replay-conceptual)=
# Range Replay (Experimental)

[Kernel replay](../kernel_replay/index.md) re-executes a **single dispatch** several times, restoring
device memory between passes so each pass sees identical inputs. That granularity fits a tool that
needs more counters than fit in one pass for one kernel. It does not fit a tool that needs several
passes over a *phase* — a training iteration, a transformer layer, a collective step — because a
phase's kernels feed each other. Re-running one kernel of a phase in isolation measures a kernel
whose inputs the rest of the phase has already changed, so the passes are not comparable to each
other and none of them corresponds to what the phase actually did.

**Range replay** on this line covers a sequence by re-running the application's own code. The
application marks the phase with `sigsetjmp`. After each pass the SDK restores device memory and
the host regions the application registered, then `siglongjmp`s back to that mark. The next pass
is the same instructions again, including host calls such as `hipDeviceSynchronize`. Recorded
packets are not re-submitted. A range whose tool does not set `host_replay` runs once and closes
with `PACKET_REPLAY_UNSUPPORTED`.

| | Kernel replay | Host range replay |
|---|---|---|
| Unit of repetition | one dispatch | the application's code between begin and end |
| Who drives pass 0 | the SDK | the application |
| Who drives later passes | the SDK, in place | the application, re-entered by `siglongjmp` |
| When later passes run | inside the dispatch, before the application sees it complete | before `end` returns |
| What is restored | that dispatch's device inputs | every GPU agent's tracked device memory, plus registered host memory |

The control flow, the stack at the jump, and the locks around the restore are in
[Control flow](range_replay_control_flow.md).

Range replay is **experimental**. The public header is
`rocprofiler-sdk/experimental/range_replay.h`. It shares kernel replay's device-memory snapshot,
its per-agent replay window, and its pass-count and localized-context-toggle semantics, so a tool
that already drives kernel replay reuses most of its pass logic unchanged.

## Host replay is the replay

The tool sets `host_replay` on the CONFIG record. The application opens the range with
`ROCPROFILER_RANGE_REPLAY_HOST_BEGIN`, which takes a `sigsetjmp` in the caller's frame.
`rocprofiler_range_replay_end` fences every intercepted compute queue, restores device memory on
every GPU agent plus the host regions registered with
`rocprofiler_range_replay_host_track_memory`, and `siglongjmp`s back while passes remain. Copies,
memsets, multiple streams, graph launches, balanced allocations, and host calls are part of each
pass because the application issues them again.

Pass 0 raises no `PASS` callback. A tool that wants pass 0 measured configures its services
normally and uses the `PASS` toggles only on the re-entered passes.

See [Control flow](range_replay_control_flow.md) for the sequence, and
[Host range replay](range_replay_host.md) for the application's contract, what is restored, and
what still declines.

## Soundness is checked, not enforced

Replaying a range is only meaningful if re-running the application's code from the restored
snapshot reproduces what pass 0 did. That holds when the range is **self-contained under the
snapshot's coverage**: everything the kernels read is either captured by the snapshot or unchanged
across passes.

The SDK cannot enforce that property — it would have to constrain the application. So it checks the
conditions it can observe and *declines* a range that fails any of them. A declined range is not
an error: the application's own execution is forwarded unmodified either way, and a decline only
means the extra passes are skipped. The reason reaches the tool as the `CLOSE` callback's `status`.

See [Soundness and declining](range_replay_soundness.md) for the full decision table, the
cross-thread decline channel, and the optional divergence check.

## How it fits together

```text
experimental/range_replay.h           public payload struct, status enum, begin/end
        |
        +-- callback_tracing.cpp      subscription; switches on the allocation tracker
        |
        +-- range_replay/
        |     api.cpp                 begin/end. end() host-replays, or closes without
        |                             re-submitting packets
        |     range_state.cpp         eligibility bookkeeping, decline channels
        |     replay_callbacks.cpp    CONFIG / PASS / CLOSE delivery, pass-count decisions
        |     host_replay.cpp         fence, snapshot, restore, the jump back into the caller
        |     digest.cpp              region hashing and divergence counting
        |
        +-- hsa/replay_window.cpp     per-agent lock, drains, ring submit, passthrough gate
        +-- hsa/queue.cpp             recording hook, foreign-dispatch detection
        +-- hsa/async_copy.cpp        device-write detection
        |
        +-- kernel_replay/            reused unchanged: memory_tracker, memory_snapshot,
                                      local_context
```

`range_state.cpp` and `digest.cpp` are deliberately free of GPU dependencies. The eligibility
decision table and the divergence hashing are the two parts most likely to be wrong in a way that
silently produces misleading measurements, so both are unit tested directly rather than only through
hardware integration tests; host replay consumes an already-decided record.

## Carve-outs

Packet range replay is scoped to what the existing snapshot can restore correctly. Host replay
lifts the rows marked *(packet only)*; see [Host range replay](range_replay_host.md).

| Not supported in v1 | Consequence |
|---|---|
| More than one queue or agent per range *(packet only)* | declined (`MULTI_QUEUE` / `MULTI_AGENT`) |
| Unified / managed memory, host memory, fine-grained memory, VM-mapped allocations | not restored between passes; not detected — see the divergence check |
| Host state *(packet only)* | never rewound; a range whose kernels consume values the host recomputed inside it is not a replay candidate |
| HIP graph launches inside a range *(packet only)* | declined (`GRAPH_LAUNCH`) |
| Async copies into device memory inside a range *(packet only)* | declined (`MEMORY_COPY_IN_RANGE`) |
| Code objects loaded or unloaded inside a range (lazy module loading, JIT, `hipModuleUnload`) | declined (`CODE_OBJECT_CHANGED_IN_RANGE`); run a range once before measuring it |
| Kernels that call `printf` | replayed output is printed again (hostcall `printf`, HIP's default) or lost (`-mprintf-kind=buffered`); see [Soundness and declining](range_replay_soundness.md) |
| Concurrency within a pass | host replay re-runs the application's own submission order, including concurrent streams. The SDK does not rewrite the queue |

## Multi-GPU and collectives

Multi-GPU ranges — the case that matters for RCCL and MPI, where a range's dispatches on one agent
are causally tied to dispatches and transfers on another — are follow-on work. Two parts of the
existing design already point the right way, and one piece is missing.

What works in favor: snapshots are agent-scoped and the replay lock is per-agent, so per-rank
snapshots *compose*. A collective's kernels write into peer device memory, and those writes land in
the destination rank's own allocations — so if every rank snapshots its own agent and every rank
restores before every pass, the union covers the peer writes too. Nothing about the snapshot needs
to become distributed.

What is missing is agreement. A collective's kernels rendezvous: they spin on flags in peer memory.
If one rank replays a collective and another does not, the replaying rank's kernel waits for a peer
that never arrives. So participating ranks must replay in lockstep — same pass count, entering each
pass together — and, critically, **a range declined on one rank must be abandoned on all of them**.

A tool can arrange the lockstep part with the existing API, because `pass_count_cb` and
`replay_continue_cb` are tool callbacks: it can allreduce the pass count and the continue-decision
inside them. It cannot arrange the abandonment part. Eligibility is not known at `CONFIG` (the range
has not run yet), and by `CLOSE` the passes have already happened. There is no point in the current
API where a tool learns that its range is replayable and can still veto it.

That argues for one addition before the API stabilizes: a **vetoable pre-replay notification**,
delivered from `rocprofiler_range_replay_end()` after eligibility is fully determined — after the
drain, the allocation-stability check, and kernarg staging — but before the first pass rewinds device
memory. A tool would allreduce the status across the communicator and veto unless every rank is
eligible. It is a small change (the executor already computes everything in that order; the callback
slots in before the pass loop) and it is the enabler for the entire distributed case, so it is worth
doing deliberately rather than discovering later that the API cannot express it.

Host-side and network state is the harder limit. Intra-node collectives over xGMI are
peer-to-peer device writes, which the snapshot covers. Inter-node collectives involve a host proxy
thread and NIC queue-pair state that no device snapshot rewinds, so those ranges are outside what
this mechanism can reproduce. For GPU-aware MPI the practical guidance is to bracket the *compute*
phases between communication rather than the communication itself — which is also where the
interesting counters are.

## Documentation in this section

- **[Callback API and tool configuration](range_replay_callback_api.md)** — the
  `ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY` domain, its three operations, the payload struct,
  pass-count semantics, the localized context toggles, and how a tool configures range replay.
- **[Soundness and declining](range_replay_soundness.md)** — what makes a range replayable, the
  decline decision table and why each condition is checked, the cross-thread decline channel, the
  kernarg-recycling problem, the interceptor passthrough gate, and the optional divergence check.
- **[Host range replay](range_replay_host.md)** — the `sigsetjmp` / `siglongjmp` path that re-runs
  the application's own code per pass: the begin macro, registered host memory, the application's
  contract, and the time-stepping integration test.
- **[Performance](range_replay_performance.md)** — the two-term cost model, why the fixed window
  cost is paid once per range rather than once per dispatch, what the two nightly perf tests
  measure, and why a declined range must never be timed.
