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
| Dispatch packets re-submittable | The recorded packets must reach the GPU through the same path the application used | requires the interception path; queue interposition declines |
| Everything read is in the snapshot | Unified memory, host memory, and host-side state are not captured | **not checkable** — see the divergence check |

## The decline decision table

Eligibility lives in `range_record_t` (`range_replay/range_state.cpp`), which is deliberately free of
GPU dependencies so the table below can be unit tested directly.

| Condition | Status | Detected at |
|---|---|---|
| Dispatch to a second queue on the bound agent | `MULTI_QUEUE` | record time |
| Dispatch to a second agent | `MULTI_AGENT` | record time |
| HIP graph launch inside the range | `GRAPH_LAUNCH` | record time |
| Kernarg segment size not resolvable for a kernel | `UNKNOWN_KERNARG_SIZE` | record time |
| Recorded dispatches exceed the budget (4096) | `PROGRAM_TOO_LARGE` | record time |
| Another thread dispatches to the bound agent | `CONCURRENT_DISPATCH` | cross-thread, folded in |
| Async copy writes memory owned by the bound agent | `MEMORY_COPY_IN_RANGE` | cross-thread, folded in |
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
holds that launch's arguments, and the executor allocates its own staging block for the replay,
laying the recorded bytes out back to back with each kernel's 256-byte alignment honored and
patching each packet to point at its slot. The block is refilled before every pass, so a pass never
observes another pass's kernarg contents.

Each recorded dispatch's completion signal is also cleared. The application has already consumed
(and may have destroyed or reused) the signal a recorded packet carried; re-firing it would corrupt
the application's own synchronization. Replayed passes get their signals from the interceptor.

### The interceptor must not transform twice

A replay pass submits its packets by handing them to `Queue::invoke_write_interceptor`, which
transforms them (minting completion signals, creating records) and then calls a writer to put them on
the ring. Writing to the ring re-enters the interception path — on the same thread. Without a guard,
each packet is transformed a second time, producing two completion signals and two dispatch records
per dispatch.

The guard is a thread-local passthrough flag, set around the ring write and checked at the top of
`WriteInterceptor`, which forwards packets untouched while it is set.

The same thread-local reasoning applies to the per-agent replay lock. A replaying thread holds it as
a writer for the whole window, so the reader-lock acquisition on the ordinary dispatch path is
skipped when `this_thread_replaying()` is true — otherwise the thread would deadlock against itself.

## The replay window

`rocprofiler_range_replay_end()` runs the following on the calling thread, which blocks for the
duration:

```text
take per-agent WRITER lock
  drain this queue's async handlers, then every queue on the agent
  verify the tracked allocation set still matches the entry snapshot's
  snap()  -> the EXIT snapshot: the state the application must resume with
  reserve the kernarg staging block
  install the localized-context-control guard; mark this thread replaying
  for pass = 1 ..:
      restore(entry snapshot)
      PASS PHASE_ENTER
      fill staging, submit the recorded packets (serialized)
      drain this pass's async completion handlers
      PASS PHASE_EXIT
      ask the tool whether to continue; break if not
  optionally: snap() and compare digests against the exit snapshot
  restore(exit snapshot)
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
could read uncaptured state that happens not to change the captured regions — but non-zero divergence
does prove it was not.

## Source reference

All paths are relative to `projects/rocprofiler-sdk/`.

| Component | File | Symbol |
|---|---|---|
| Eligibility table | `source/lib/rocprofiler-sdk/range_replay/range_state.cpp` | `range_record_t::decline()`, `bind()`, `add_dispatch()` |
| Recording hook | `source/lib/rocprofiler-sdk/range_replay/range_state.cpp` | `note_submission()` |
| Cross-thread declines | `source/lib/rocprofiler-sdk/range_replay/range_state.cpp` | `note_foreign_dispatch()`, `note_device_write()`, `fold_external_decline()` |
| Open-range fast gate | `source/lib/rocprofiler-sdk/range_replay/range_state.cpp` | `any_range_open()` |
| Replay window and pass loop | `source/lib/rocprofiler-sdk/range_replay/executor.cpp` | `execute_range()` |
| Entry snapshot | `source/lib/rocprofiler-sdk/range_replay/executor.cpp` | `ensure_entry_snapshot()` |
| Kernarg staging | `source/lib/rocprofiler-sdk/range_replay/executor.cpp` | `kernarg_staging` |
| Divergence hashing | `source/lib/rocprofiler-sdk/range_replay/digest.cpp` | `hash_bytes()`, `count_divergent()` |
| Passthrough gate | `source/lib/rocprofiler-sdk/hsa/replay_window.cpp` | `set_interceptor_passthrough()` |
| Ring re-submission | `source/lib/rocprofiler-sdk/hsa/replay_window.cpp` | `replay_ring_submit()` |
| Snapshot and restore | `source/lib/rocprofiler-sdk/kernel_replay/memory_snapshot.cpp` | `snap()`, `restore()` |
