(range-replay-soundness)=
# Range Replay — Soundness and Declining

Re-executing a recorded range is only meaningful if the passes are comparable to each other and to
the application's own execution. That requires the range to be **self-contained under the snapshot's
coverage**: everything the recorded dispatches read is either captured by the device-memory snapshot
taken at range entry, or unchanged across passes.

The SDK cannot make that true — it would have to constrain the application. What it can do is check
the conditions it is able to observe, and *decline* a range that fails one. This page covers what is
checked and why, what the checks cannot see, and the optional verification that catches the
remainder.

A decline is not an error and not a warning the tool has to hunt for. The application's own
execution of the range is forwarded unmodified in every case; a decline only means the extra passes
are skipped, and the reason arrives as the `CLOSE` callback's `status`.

## What has to hold

| Requirement | Why | How it is handled |
|---|---|---|
| One queue, one agent | The snapshot is agent-scoped, and the replay window's lock is per-agent. A range spanning two agents would restore half its state | declined |
| No device write from outside the recording | A copy or a foreign kernel writes state no recorded dispatch produces, so a replayed pass runs without it | declined |
| Allocation set unchanged | The snapshot names specific base pointers. If one was freed and its address reused, restoring writes into memory the application has repurposed | declined |
| Kernarg bytes available at replay time | HIP recycles a kernarg block as soon as its kernel completes | bytes copied at record time |
| Recorded code stays loaded, and its module variables are in the snapshot | An unload frees code a recorded packet points at; a code object loaded after the entry snapshot has `__device__` / `__constant__` globals the snapshot never saw | declined |
| Dispatch packets re-submittable | The recorded packets must reach the GPU through the same path the application used | requires the interception path; queue interposition declines |
| Everything read is in the snapshot | Unified memory, host memory, and host-side state are not captured | **not checkable** — see the divergence check |
| No kernel writes another process's memory | A device-initiated put or atomic (rocSHMEM) lands in a peer's fine-grained heap, which no snapshot covers and the peer does not rewind | **not checkable** — warned when the process maps peer memory over IPC |

## The decline decision table

Eligibility lives in `range_record_t` (`range_replay/range_state.cpp`), which is deliberately free of
GPU dependencies so the table below can be unit tested directly.

| Condition | Status | Detected at |
|---|---|---|
| Dispatch to a second queue on the bound agent | `MULTI_QUEUE` | record time |
| Dispatch to a second agent | `MULTI_AGENT` | record time |
| HIP graph launch inside the range | `GRAPH_LAUNCH` | record time |
| Kernarg segment size not resolvable for a kernel | `UNKNOWN_KERNARG_SIZE` | record time |
| A kernel from a code object loaded after the range bound | `CODE_OBJECT_CHANGED_IN_RANGE` | record time |
| Recorded dispatches exceed the budget (4096) | `PROGRAM_TOO_LARGE` | record time |
| Another thread dispatches to the bound agent | `CONCURRENT_DISPATCH` | cross-thread, folded in |
| Async copy writes memory owned by the bound agent | `MEMORY_COPY_IN_RANGE` | cross-thread, folded in |
| A HIP runtime copy or stream-memory kernel (`__amd_rocclr_copy*`, `__amd_rocclr_streamOps*`, `__amd_rocclr_batchMemOp*`) is recorded | `MEMORY_COPY_IN_RANGE` | record time |
| An RCCL kernel (`ncclDevKernel*`, `ncclSymkDevKernel*`, `ncclKernel*`, `mscclKernel*`, `meta::comms::*`) is dispatched, in either replay mode | `COLLECTIVE_IN_RANGE` | record time |
| The tool's `replay_commit_cb` returned zero for a replayable range | `VETOED` | before the first extra pass |
| Any code object is unloaded while the range is bound | `CODE_OBJECT_CHANGED_IN_RANGE` | cross-thread, folded in |
| Device allocation created or freed inside the range | `ALLOCATION_CHANGED_IN_RANGE` | close time |
| Snapshot capture or a later snapshot failed | `SNAPSHOT_FAILED` | snapshot time |
| Kernarg staging allocation failed | `STAGING_FAILED` | close time |
| Queue interposition is enabled | `UNSUPPORTED_QUEUE_PATH` | `begin` |
| Nothing was recorded | `NO_DISPATCH` | close time |
| Tool asked for one pass or none | `NO_PASS_COUNT` | close time |

**First reason wins.** A declined range stops recording, so a range that ran past its budget does not
keep copying kernargs, and a later observation cannot mask the original cause. `MULTI_AGENT` is
checked before `MULTI_QUEUE` when both differ, because the agent is the more informative reason: the
snapshot is agent-scoped.

Declining releases the recorded packets immediately — each holds a copy of its kernarg segment — but
the observed dispatch count is kept, because "how many dispatches were in this phase" is still what
the tool wants to report.

## Cross-thread declines

Two of the conditions above are observed by a thread that does not own the range: a foreign dispatch
on the bound agent, and an async copy writing memory that agent owns. Neither may touch the owning
thread's `range_record_t` — it is single-threaded by construction, which is what keeps the recording
path cheap.

Instead each open range publishes a small `external_decline_t` channel (one atomic) into a
process-wide registry keyed by agent. A foreign observer compare-exchanges its reason into that
channel, and the owning thread folds it into the record the next time it records and again at close.
The compare-exchange is what gives "first reason wins" across threads for free.

The registry entry's agent key is zero until the range binds to its first recorded dispatch, and an
unbound range cannot be interfered with, because it has recorded nothing yet. Unregistration happens
*before* the final fold at close, so no reason can be published into a channel nobody will read.

The whole mechanism is fronted by a process-global count of open ranges, read as a relaxed atomic.
A run in which no thread has a range open pays exactly one load on the dispatch path and one on the
copy path.

## Two implementation hazards

These are the two places where the obvious implementation is wrong, so they are worth stating
plainly.

### Kernarg blocks are recycled

HIP hands a kernel its arguments in a block allocated from the agent's kernarg pool, and reclaims
that block as soon as the kernel completes. A range holds its recording until `end`, by which point
a later launch in the same range may have been given the same block and overwritten an earlier
dispatch's arguments. Recording the packet alone would therefore replay some dispatches with the
wrong arguments — and there is nothing about the packet that would reveal it.

So the recorder copies each dispatch's kernarg segment out at record time, while the block still
holds that launch's arguments, and the executor stages them in a kernarg block of its own, laying
the recorded bytes out back to back with each kernel's 256-byte alignment honored and patching each
packet to point at its slot. The block is refilled before every pass, so a pass never observes
another pass's kernarg contents. It comes from the agent's kernarg pool, or is the block the agent's
previous range used when that one fits; a block is kept for the next range only after the last pass
that read it has drained.

Each recorded dispatch's completion signal is also cleared. The application has already consumed
(and may have destroyed or reused) the signal a recorded packet carried; re-firing it would corrupt
the application's own synchronization. Replayed passes get their signals from the interceptor.

### The interceptor must not transform twice

A replay pass submits its packets by handing them to `Queue::invoke_write_interceptor`, which
transforms them (minting completion signals, creating records) and then calls a writer to put them on
a ring. Writing them to the application's queue would re-enter the interception path and transform
each packet a second time, producing two completion signals and two dispatch records per dispatch.
It would also deadlock: ROCr holds a queue's intercept lock while it runs the SDK's interceptor, and
the interceptor takes the per-agent replay lock as a reader, so a thread holding that lock as a
writer must never ring an intercepted doorbell.

So the writer puts the packets on a replay queue the SDK owns, one per agent, created lazily through
the unwrapped HSA table. ROCr does not intercept it, so the packets are transformed exactly once and
the doorbell takes no lock. The queue lives for the rest of the process. Kernels replayed on it do not
inherit the application queue's CU mask or priority; a cooperative launch replays correctly, since
the grid barrier lives in memory.

The same thread-local reasoning applies to the per-agent replay lock. A replaying thread holds it as
a writer for the whole window, so the reader-lock acquisition on the ordinary dispatch and copy paths
is skipped when `this_thread_replaying()` is true — otherwise the thread would deadlock against
itself.

## The replay window

`rocprofiler_range_replay_end()` runs the following on the calling thread, which blocks for the
duration:

```text
take per-agent WRITER lock (before the range is taken, so no dispatch or copy can slip in)
  drain this queue's async handlers, then every queue on the agent and its device-writing copies
  verify the tracked allocation set still matches the entry snapshot's
  snap()  -> the EXIT snapshot: the state the application must resume with
  reserve the kernarg staging block (the agent's retained block when it fits)
  install the localized-context-control guard; mark this thread replaying
  for pass = 1 ..:
      restore(entry snapshot)
      PASS PHASE_ENTER
      fill staging, submit the recorded packets (serialized) to the agent's replay queue
      drain this pass's async completion handlers
      PASS PHASE_EXIT
      ask the tool whether to continue; break if not
  optionally: snap() and compare digests against the exit snapshot
  restore(exit snapshot)
  retain the kernarg staging block for the agent's next range
release per-agent WRITER lock
```

Two snapshots are involved, and the second is not optional. The **entry** snapshot is what each pass
rewinds to. The **exit** snapshot captures what the application's own execution produced, and is
restored after the last pass so the application resumes with its own results — not with whatever the
final replayed pass left behind. Skipping it would silently hand the application the replay's
output.

A failed `restore()` is fatal rather than a decline. A partial host-to-device copy leaves device
memory in a state that is neither the application's nor the snapshot's, and continuing would hand
that corruption to the application. This matches kernel replay's convention for the same situation.

Each dispatch in a pass has its AQL barrier bit forced on, so the passes are serialized
dispatch-by-dispatch. This is stricter than the application, whose packets may have been free to
overlap, and it is the reason replayed passes are not a faithful source of concurrency-sensitive
timings.

## Other threads during the window

While a window runs, other threads keep running. A dispatch or a device-writing copy from one of them
must either land before the window's drain, and so be part of the state the exit snapshot captures,
or wait until the window ends. Otherwise the final restore overwrites its result. Three rules hold
this together:

- Every dispatch on the instrumented path takes the per-agent replay lock as a reader, and re-reports
  itself to any open range on that agent once it holds the lock. A dispatch that waited out a window
  therefore still declines the next range it overlaps.
- An async copy that writes device memory takes the reader lock of every agent it writes, in agent-id
  order (the order host replay takes the writer locks), from reporting itself until it is submitted.
  It also registers a ticket that its completion handler retires. The agent drain waits for every
  ticket issued before the drain started, so a steady stream of new copies cannot stall it. The
  application's completion signal and its timestamps are forwarded from the SDK's own signal.
- Host replay fences every queue before taking the writer locks, because the fence goes through the
  intercepted doorbells, and drains under the locks, where the drain is bounded because new work is
  blocked.

The lock order these rules guarantee, and the absence of lost writes, are checked by a TLA+ model of
the window, the dispatch path, and the copy path in both modes, with application threads that share a
HIP queue or have their own. The `test-range-replay-concurrency-*` tests exercise the same races on
hardware: a second thread writes, by copy or by kernel, into memory a range covers shortly after each
`end()`, and every write must survive.

Tracking costs every device-writing copy one completion-handler hop whenever a tool has configured
range replay, open range or not. On MI300-class hardware that is about 0.2–0.5 µs per copy.

## The divergence check

The checks above cover interference the SDK can see. They cannot cover a range that reads state the
snapshot does not capture — unified or managed memory, fine-grained or host memory,
virtual-memory-mapped allocations, or a value the host recomputed inside the range. Such a range
passes every check and is replayed, and its passes quietly describe different inputs.

Setting `ROCPROF_RANGE_REPLAY_VERIFY` turns on a check for exactly that residue. The executor hashes
each snapshot region (FNV-1a) after the application's execution and again after the final replayed
pass, and reports how many regions differ as the `CLOSE` callback's `divergence_count`.

Regions are ordered by device address before hashing, because the allocation inventory itself is
unordered and the two snapshots must compare positionally. A region-count mismatch is reported as
fully divergent rather than as a partial comparison.

A non-zero `divergence_count` means the final pass did not reproduce the application's result from
the same starting memory, so the range is not self-contained under the snapshot's coverage and its
per-pass measurements describe different inputs. It is a diagnostic, not a decline: by the time it is
known, the passes have already run. The intended use is to validate a range once during tool
development, then leave the check off.

It is deliberately one-sided. Zero divergence does not prove the range was self-contained — a range
could read uncaptured state that happens not to change the captured regions. Non-zero divergence
proves it was not only when the range's kernels are bitwise deterministic. The comparison is
byte-exact, and some kernels produce different bytes from identical inputs: floating-point atomics
(including the hardware atomics `-munsafe-fp-atomics` selects), and reductions or split-K GEMMs whose
summation order follows wave scheduling. For such a range, divergence is expected even when the
range is self-contained, so validate with deterministic kernels, or with the non-deterministic ones
configured to be deterministic, before trusting the check.

### Runtime copy kernels

HIP performs small and host-staged copies with its own blit kernels on the application's queue
rather than on an SDMA engine, so they arrive at the recorder as ordinary dispatches. Re-submitting
one is not the same as re-running the copy: a device-to-host blit overwrites pinned host memory the
application has since modified, and a host-to-device blit reads whatever that host memory holds by
then. A staged `hipMemcpyAsync` D2H, host `MPI_Allreduce`, H2D sequence -- the usual pattern for
MPI without GPU awareness -- replayed with `divergence_count` 2 before these kernels declined. The
decline matches the SDMA path's. Fill kernels are kept: they write a value carried in their
kernarg bytes, so they replay like any recorded kernel.

### Communication kernels

RCCL keeps each connection's step counters in device memory, which the snapshot rewinds, and
compares them against flags its peers write into fine-grained and peer-mapped memory, which it does
not. A replayed collective then sees flags that already satisfy its waits and completes against
stale buffers without its peers, or waits for a peer that has moved on. Both were observed on four
MI325X ranks: three ranks finished three replayed `ncclAllReduce` passes before the fourth had
started its first, and the fourth then hung. Agreement between ranks does not help, since the state
is inconsistent within each rank, so a range that dispatches an RCCL kernel declines in both replay
modes.

RCCL chooses the kernel per call, so the match covers every cross-rank family it ships: the generic
`ncclDevKernel` and symmetric-memory `ncclSymkDevKernel` kernels, MSCCL's `mscclKernel`, and the IPC
kernels in the `meta::comms` namespace. The last matter in practice: on a full node of eight MI325X
GPUs a small `ncclAllReduce` runs as `meta::comms::ddaAllReduceFlatIpc`, which spins on barriers in
its peers' memory. Kernel names are matched on their leading qualified name after demangling the
`_Z`, `_ZL` and `_ZN...E` prefixes. RCCL loads its code objects from memory, so the code-object URI
does not identify them. Its single-rank kernels (`oneRankReduce`) touch only local memory and are
not matched.

User kernels that communicate on the device -- rocSHMEM puts and atomics, or hand-written kernels on
IPC-mapped buffers -- cannot be recognized by name. Their writes land in another process's memory,
so a replayed pass repeats them and nothing rewinds them: in a four-PE test where each kernel adds 1
to its neighbour's symmetric-heap counter, both replay modes reported `REPLAYED` with
`divergence_count` 0 and every counter ended at 4 instead of 1. When a process has peer memory
mapped through HSA IPC, the SDK warns once on the first replayed range. It does not decline: every
RCCL application maps peer memory, and the compute phases between its collectives replay soundly.

## What the compiler and runtime add

A kernel touches more state than the buffers its source names. The compiler and the HIP runtime add
state of their own, and each piece is covered, declined, or outside what range replay can see:

| State | Where it lives | How it is handled |
|---|---|---|
| Module variables (`__device__`, `__constant__`) | The loaded executable's data segment, not a tracked allocation | Captured: the snapshot enumerates every loaded executable's variables when it is taken |
| Code objects loaded inside the range | A new executable. HIP loads a module lazily on its first launch, and JITs (hipRTC, Triton) load at run time | A dispatch from one declines the range (`CODE_OBJECT_CHANGED_IN_RANGE`), as does unloading any code object |
| Hidden (implicit) kernel arguments | The tail of the kernarg segment | Copied: the copy is sized by the code object's `.kernarg_segment_size`, which includes them. A kernel without that metadata, such as hand-written assembly, declines with `UNKNOWN_KERNARG_SIZE` |
| Kernarg preloading (gfx94x and later) | SGPRs the hardware fills from the kernarg segment at wave launch | Unchanged: the patched packet points at the staged copy, so the preload reads the recorded bytes |
| `printf` | Host memory the runtime names in a hidden argument. The compiler picks the path: with `-mprintf-kind=hostcall`, HIP's default, a host thread prints each call while the kernel runs; with `-mprintf-kind=buffered`, the runtime clears a host buffer before the launch and prints it after the kernel completes | **Not handled.** Replayed passes execute the calls again. With hostcall `printf`, every pass prints again. With buffered `printf`, the replayed output lands after the runtime has printed the application's, and the next `printf` launch clears it, so it is never printed. Kernel replay differs: its passes run before that drain, so buffered output appears once per pass |

In practice: run a range once before measuring it, so lazily loaded modules are already resident when
it binds, and keep kernels that call `printf` out of ranges.

Replay passes run inside `end()`, after the application's own execution of the range, so GPU event
timings the application records inside the range are unaffected. Kernel replay differs here: its
passes run before a dispatch's completion signal fires, which stretches event-based timings, and
those are what autotuners (Triton's autotuner, PyTorch TunableOp, MIOpen's find mode) measure. Host
timers that span `end()` do include the replay window.

## Other tools that intercept queues

rocprofiler-sdk installs its queue wrappers before it hands the HSA API table to tools registered
with `rocprofiler_at_intercept_table_registration`. A tool that then replaces `hsa_queue_create` or
`hsa_amd_queue_create` with its own `hsa_amd_queue_intercept_create` call, instead of calling the
entry it replaced, takes those queues away from the SDK. kerncap does this to capture dispatches. No
queue-based service, range replay included, sees dispatches on those queues, so every range closes
with `NO_DISPATCH`. Run such a tool in a separate process from range replay.

## Source reference

All paths are relative to `projects/rocprofiler-sdk/`.

| Component | File | Symbol |
|---|---|---|
| Eligibility table | `source/lib/rocprofiler-sdk/range_replay/range_state.cpp` | `range_record_t::decline()`, `bind()`, `add_dispatch()` |
| Recording hook | `source/lib/rocprofiler-sdk/range_replay/range_state.cpp` | `note_submission()` |
| Cross-thread declines | `source/lib/rocprofiler-sdk/range_replay/range_state.cpp` | `note_foreign_dispatch()`, `note_device_write()`, `note_code_object_unload()`, `fold_external_decline()` |
| Code object watermark | `source/lib/rocprofiler-sdk/range_replay/range_state.cpp` | `range_record_t::admit_code_object()` |
| Unload hook | `source/lib/rocprofiler-sdk/code_object/code_object.cpp` | `executable_destroy_internal()` |
| Open-range fast gate | `source/lib/rocprofiler-sdk/range_replay/range_state.cpp` | `any_range_open()` |
| Replay window and pass loop | `source/lib/rocprofiler-sdk/range_replay/executor.cpp` | `lock_for_window()`, `execute_range()` |
| Entry snapshot | `source/lib/rocprofiler-sdk/range_replay/executor.cpp` | `ensure_entry_snapshot()` |
| Kernarg staging | `source/lib/rocprofiler-sdk/range_replay/executor.cpp` | `kernarg_staging` |
| Kernarg block retention | `source/lib/rocprofiler-sdk/range_replay/retained_kernarg.cpp` | `kernarg_block_fits()`, `take_retained_kernarg_block()`, `retain_kernarg_block()` |
| Divergence hashing | `source/lib/rocprofiler-sdk/range_replay/digest.cpp` | `hash_bytes()`, `count_divergent()` |
| Pass submission | `source/lib/rocprofiler-sdk/hsa/replay_window.cpp` | `replay_private_submit()` |
| Agent drain and copy tickets | `source/lib/rocprofiler-sdk/hsa/replay_window.cpp` | `replay_drain_agent_or_fatal()`, `replay_copy_submitted()`, `replay_copy_retired()` |
| Copy lock and tracking | `source/lib/rocprofiler-sdk/hsa/async_copy.cpp` | `async_copy_impl()`, `track_replay_copy()` |
| Snapshot and restore | `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp` | `snap()`, `restore()` |
