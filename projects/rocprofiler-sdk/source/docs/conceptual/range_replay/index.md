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

rocprofv3 drives it the same way it drives kernel replay. ``--replay-mode range``
(with ``--range-replay-beta-enabled``) collects one ``--pmc`` group per pass of a range the
application marks, in a single run. ``--range-replay-host`` selects host re-execution. ``kernel``
remains the per-dispatch strategy, and the two are not combined. See the rocprofv3 how-to.

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
        re-submit the recorded dispatches, serialized, on the SDK's replay queue
        PASS PHASE_EXIT
        ask the tool whether to continue
    CLOSE callback                ->  status, dispatch count, divergence count
```

Because pass 0 is the application's own run, it raises no `PASS` callback: there is nothing for the
tool to configure, since the dispatches have already been submitted by the time the SDK sees them.
A tool that wants pass 0 measured configures its services normally and uses the `PASS` toggles only
to change what is collected on the *re-executed* passes.

## Host replay (`sigsetjmp` / `siglongjmp`)

Packet replay re-submits the recorded packets and never runs the host code inside the range again,
which is why it declines ranges with copies, several streams, graphs, or host computation that feeds
the kernels. Host replay is the alternate path for those ranges. The tool sets `host_replay` on the
CONFIG record. The application opens the range with `ROCPROFILER_RANGE_REPLAY_HOST_BEGIN`, which
takes a `sigsetjmp` in the caller's frame. `rocprofiler_range_replay_end` then fences every GPU queue,
restores device memory on every GPU agent plus the host regions registered with
`rocprofiler_range_replay_host_track_memory`, and `siglongjmp`s back. The application's own code runs
again for every pass, so copies, memsets, multiple streams, graph launches, balanced allocations and
host calls are all part of each pass.

See [Host range replay](range_replay_host.md) for the protocol, the application's contract, what is
restored, and what still declines.

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
        |     host_replay.cpp         host replay: all-agent fence/snapshot/restore, host regions
        |     retained_kernarg.cpp    the kernarg block kept for each agent's next range
        |     digest.cpp              region hashing and divergence counting
        |
        +-- hsa/replay_window.cpp     per-agent lock, drains, copy tickets, replay queue
        +-- hsa/queue.cpp             recording hook, foreign-dispatch detection
        +-- hsa/async_copy.cpp        device-write detection, copy lock and tracking
        |
        +-- kernel_replay/            reused unchanged: memory_tracker, memory_snapshot,
                                      local_context
```

`range_state.cpp` and `digest.cpp` are deliberately free of GPU dependencies. The eligibility
decision table and the divergence hashing are the two parts most likely to be wrong in a way that
silently produces misleading measurements, so both are unit tested directly rather than only through
hardware integration tests; the executor consumes an already-decided record.

## Carve-outs

Packet range replay is scoped to what the existing snapshot can restore correctly. Host replay
lifts the rows marked *(packet only)*; see [Host range replay](range_replay_host.md).

| Not supported in v1 | Consequence |
|---|---|
| More than one queue or agent per range *(packet only)* | declined (`MULTI_QUEUE` / `MULTI_AGENT`) |
| Unified / managed memory, host memory, fine-grained memory, VM-mapped allocations | not restored between passes; not detected — see the divergence check |
| Host state *(packet only)* | never rewound; a range whose kernels consume values the host recomputed inside it is not a replay candidate |
| HIP graph launches inside a range *(packet only)* | declined (`GRAPH_LAUNCH`) |
| Async copies into device memory inside a range, including HIP's blit-kernel copies *(packet only)* | declined (`MEMORY_COPY_IN_RANGE`) |
| RCCL collectives or point-to-point operations inside a range | declined (`COLLECTIVE_IN_RANGE`) |
| Device-initiated writes to another process's memory (rocSHMEM, IPC kernels) | **not handled**: repeated on every pass; warned when peer memory is mapped |
| Code objects loaded or unloaded inside a range (lazy module loading, JIT, `hipModuleUnload`) | declined (`CODE_OBJECT_CHANGED_IN_RANGE`); run a range once before measuring it |
| Kernels that call `printf` | replayed output is printed again (hostcall `printf`, HIP's default) or lost (`-mprintf-kind=buffered`); see [Soundness and declining](range_replay_soundness.md) |
| Concurrency within a pass | passes are serialized dispatch-by-dispatch, so concurrency-sensitive measurements differ from pass 0 |

## Multi-GPU and collectives

Each MPI rank runs its own range in its own process. Snapshots are agent-scoped and per-process,
so ranks replay independently; what they have to share is the decision.

**Agreement.** A rank that declines runs its range once, while one that replays runs it again.
Anything in the range that talks to other ranks -- a host `MPI_Allreduce` in a host-replay range,
for example -- then runs a different number of times on different ranks. `replay_commit_cb` closes
that gap: it is delivered once per range on every rank, declined or not, after eligibility is final
and before anything is rewound, so a tool can allreduce the status and veto unless every rank can
replay. Vetoed ranges close with `VETOED`. See
[Committing to a replay](range_replay_callback_api.md#committing-to-a-replay).

**What agreement makes replayable.** Host-side communication on host buffers inside a host-replay
range (staged D2H copy, `MPI_Allreduce`, H2D copy), because every pass re-runs it on every rank in
the same order. Pass counts must also agree; a fixed `pass_count_cb` does that, and an open-ended
`replay_continue_cb` must reduce its decision across ranks the same way.

**What it does not.** Device-side communication. RCCL kernels keep protocol state in device memory
the snapshot rewinds and read flags in peer and fine-grained memory it does not, so replaying them is
unsound on every rank, and ranges that dispatch them decline with `COLLECTIVE_IN_RANGE`. User
kernels that write another process's memory (rocSHMEM puts and atomics) are not recognizable and
are re-executed by every pass; the SDK warns once when the process maps peer memory. See
[Communication kernels](range_replay_soundness.md#communication-kernels).

**What to bracket.** The compute between communication steps. A range around Jacobi sweeps and a
matmul in an RCCL or MPI job replays on every rank, and the communicator keeps working afterwards.
Inter-node collectives add host proxy threads and NIC state that no device snapshot rewinds, which
is another reason to keep communication outside the range.

**Measured** on one node with eight MI325X GPUs, Open MPI 4.1.2 (not GPU-aware) and RCCL, one GPU
per rank, four passes with `SQ_WAVES`, verification on, and five full iterations after the range to
check that the communicator still works:

| Range contents | Mode | All ranks eligible | One rank declines, no agreement | One rank declines, with agreement |
|---|---|---|---|---|
| Jacobi + matmul, `MPI_Allreduce` after the range | packet | `REPLAYED` | others `REPLAYED` | others `VETOED` |
| same | host | `REPLAYED` | others `REPLAYED` | others `VETOED` |
| + D2H copy, host `MPI_Allreduce`, H2D copy | packet | `MEMORY_COPY_IN_RANGE` | `MEMORY_COPY_IN_RANGE` | `MEMORY_COPY_IN_RANGE` |
| same | host | `REPLAYED` | `MPI_ERR_TRUNCATE` abort | others `VETOED` |
| + `ncclAllReduce` on device buffers | packet, host | `COLLECTIVE_IN_RANGE` | others `COLLECTIVE_IN_RANGE` | others `COLLECTIVE_IN_RANGE` |
| Jacobi + matmul, `ncclAllReduce` right after the range | packet | `REPLAYED` | others `REPLAYED` | others `VETOED` |
| same | host | `REPLAYED` | others `REPLAYED` | others `VETOED` |

Every run produced the application's own result on every rank, on 2, 4 and 8 ranks, except the
one marked as an abort. The declining rank reports its own reason. RCCL runs a different kernel on
eight ranks (`meta::comms::ddaAllReduceFlatIpc`) than on four (`ncclDevKernel_Generic`); both
decline. A rocSHMEM range whose kernel adds to a neighbour's counter replays in both
modes and leaves every counter at 4 instead of 1; that is the case the warning is for.

## Documentation in this section

- **[Callback API and tool configuration](range_replay_callback_api.md)** — the
  `ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY` domain, its three operations, the payload struct,
  pass-count semantics, the localized context toggles, and how a tool configures range replay.
- **[Soundness and declining](range_replay_soundness.md)** — what makes a range replayable, the
  decline decision table and why each condition is checked, the cross-thread decline channel, the
  kernarg-recycling problem, the replay queue, how other threads' work is kept out of the replay
  window, and the optional divergence check.
- **[Host range replay](range_replay_host.md)** — the `sigsetjmp` / `siglongjmp` path that re-runs
  the application's own code per pass: the begin macro, registered host memory, the application's
  contract, and the time-stepping integration test.
- **[Performance](range_replay_performance.md)** — the two-term cost model, why the fixed window
  cost is paid once per range rather than once per dispatch, what the two nightly perf tests
  measure, and why a declined range must never be timed.
