(range-replay-conceptual)=
# Range Replay (Experimental)

[Kernel replay](../kernel_replay/index.md) re-executes a **single dispatch** several times, restoring
device memory between passes so each pass sees identical inputs. That granularity fits a tool that
needs more counters than fit in one pass for one kernel. It does not fit a tool that needs several
passes over a *phase* — a training iteration, a transformer layer, a collective step — because a
phase's kernels feed each other. Re-running one kernel of a phase in isolation measures a kernel
whose inputs the rest of the phase has already changed, so the passes are not comparable to each
other and none of them corresponds to what the phase actually did.

**Range replay** covers a sequence. A tool brackets the phase; the SDK records the dispatches inside
it, and after the range closes it restores the range-entry device memory and re-submits the whole
recording once per extra pass.

| | Kernel replay | Range replay |
|---|---|---|
| Unit of repetition | one dispatch | a recorded sequence of dispatches |
| Who drives pass 0 | the SDK | the application (the SDK only observes) |
| When passes run | inside the dispatch, before the application sees it complete | after the range closes |
| Application observability | none: the extra passes are invisible | none for correctness, but the range takes longer to close |
| Ordering within a pass | n/a (one dispatch) | serialized dispatch-by-dispatch |

Range replay is **experimental**. The public header is
`rocprofiler-sdk/experimental/range_replay.h`. It shares kernel replay's device-memory snapshot,
its per-agent replay window, and its pass-count and localized-context-toggle semantics, so a tool
that already drives kernel replay reuses most of its pass logic unchanged.

## Record live, then replay

The essential difference from kernel replay is *when* the SDK gets control. Kernel replay intercepts
a dispatch and expands it into `N` passes before returning. A range has already run by the time the
tool closes it, so the SDK cannot drive its first execution — it records it instead:

```text
rocprofiler_range_replay_begin(id)
    CONFIG callback (ENTER then EXIT)  ->  tool returns the pass count for this range

    <the application's dispatches>     ->  recorded from WriteInterceptor; kernarg bytes copied
                                           out; device-memory snapshot taken before the first
                                           one executes.  This is PASS 0 -- the application's
                                           own execution, forwarded unmodified.

rocprofiler_range_replay_end()
    take the per-agent writer lock, drain the agent
    for pass = 1 .. N-1:
        restore the entry snapshot
        PASS PHASE_ENTER          ->  tool may toggle its contexts for this pass
        re-submit the recorded dispatches, serialized
        PASS PHASE_EXIT
        ask the tool whether to continue
    CLOSE callback                ->  status, dispatch count, divergence count
```

Because pass 0 is the application's own run, it raises no `PASS` callback: there is nothing for the
tool to configure, since the dispatches have already been submitted by the time the SDK sees them.
A tool that wants pass 0 measured configures its services normally and uses the `PASS` toggles only
to change what is collected on the *re-executed* passes.

## Soundness is checked, not enforced

Replaying a range is only meaningful if re-running the recorded dispatches from the restored
snapshot reproduces what the application's own execution did. That holds when the range is
**self-contained under the snapshot's coverage**: everything the recorded dispatches read is either
captured by the snapshot or unchanged across passes.

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
        |     api.cpp                 begin/end, context validation
        |     range_state.cpp         eligibility bookkeeping, the recorder, decline channels
        |     replay_callbacks.cpp    CONFIG / PASS / CLOSE delivery, pass-count decisions
        |     executor.cpp            snapshot, kernarg staging, the pass loop, restore
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
hardware integration tests; the executor consumes an already-decided record.

## Carve-outs

Range replay v1 is scoped to what the existing snapshot can restore correctly.

| Not supported in v1 | Consequence |
|---|---|
| More than one queue or agent per range | declined (`MULTI_QUEUE` / `MULTI_AGENT`) |
| Unified / managed memory, host memory, fine-grained memory, VM-mapped allocations | not restored between passes; not detected — see the divergence check |
| Host state | never rewound; a range whose kernels consume values the host recomputed inside it is not a replay candidate |
| HIP graph launches inside a range | declined (`GRAPH_LAUNCH`) |
| Concurrency within a pass | passes are serialized dispatch-by-dispatch, so concurrency-sensitive measurements differ from pass 0 |

Multi-GPU ranges — the case that matters for RCCL and MPI, where a range's dispatches on one agent
are causally tied to dispatches and transfers on another — are follow-on work. The per-agent lock and
the agent-scoped snapshot are already the right shape for it, but coordinating the *entry* to a
replay window across agents (and across ranks) is a separate problem: a collective's participants
must all restore and re-execute together or none of them can.

## Documentation in this section

- **[Callback API and tool configuration](range_replay_callback_api.md)** — the
  `ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY` domain, its three operations, the payload struct,
  pass-count semantics, the localized context toggles, and how a tool configures range replay.
- **[Soundness and declining](range_replay_soundness.md)** — what makes a range replayable, the
  decline decision table and why each condition is checked, the cross-thread decline channel, the
  kernarg-recycling problem, the interceptor passthrough gate, and the optional divergence check.
