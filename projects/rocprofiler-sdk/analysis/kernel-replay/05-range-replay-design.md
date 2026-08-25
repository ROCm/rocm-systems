# 5. Range replay for ROCprofiler-SDK: a design, and a cheaper alternative

Kernel replay's limits in §§2–3 all have the same shape: the unit of repetition is one dispatch, but
the unit of *meaning* in real applications is a group of dispatches — a fused attention block, a
halo exchange plus interior update, a graph-captured decode step, a collective. Every mature
profiler eventually adds a range-level replay mode for exactly this reason. This section lays out the
design space for ROCm, in order of increasing cost, and argues for a specific ordering of the work.

## 5.1 Four designs, and what each one can profile

| | Unit | Who re-executes | State needed | Preserves concurrency | Handles collectives | Cost to build |
|---|---|---|---|---|---|---|
| **A. Graph replay** | one `hipGraph` launch | the SDK, by relaunching the graph | device snapshot (as today) | yes, within the graph | no | **low** |
| **B. User-driven pass loop** | whatever the app brackets | the application | none beyond what the app already does | yes | **yes** | **low** |
| **C. API-capture range replay** | a marked API range | the SDK, by re-issuing recorded API calls | device snapshot + host memory + API log + pointer translation | yes | no | high |
| **D. App-range replay** | a marked range | a fresh process per pass | none | yes | yes | low, but it *is* application replay |

### A. Graph replay — invert the current stance

Today a HIP graph launch is explicitly excluded, with a one-shot warning. That is exactly backwards
from a replayability standpoint: **a graph is the best range-replay unit that exists on the
platform, because it is already a captured, side-effect-free, relaunchable description of a block of
GPU work.** Everything that makes API-capture range replay hard — recording the API calls,
translating pointers, deciding what is idempotent — the application already did when it captured the
graph. Relaunching it is a single `hipGraphLaunch`.

Concretely, graph replay is the current kernel-replay loop with the dispatch resubmission replaced
by a graph relaunch: take the writer lock, drain, snap, then for each pass launch the graph, drain,
read counters, restore. The pass-count callback, the local-context toggles, the reserved-id
threading and the exactly-once completion mechanics all carry over unchanged. Counters are then
range-aggregated across the graph's nodes (or per-node, if per-dispatch counter collection is left
enabled inside the graph — which is the more useful output and is already how graph dispatches are
traced).

This matters most for the workloads where kernel replay is otherwise nearly useless. vLLM-style
decode and `torch.compile` with graph capture put essentially all steady-state GPU work inside graph
launches, so under the current gate almost nothing in those processes is replayable at all. Graph
replay converts "this feature does not apply to modern inference serving" into "this feature applies
to modern inference serving better than to anything else", because a decode graph is a fixed-shape,
self-contained, repeatable unit.

Caveats to state up front: graphs containing memory-allocation nodes
(`hipGraphAddMemAllocNode`/free nodes) are not idempotent and must be declined; graphs containing
collectives inherit §3.2 wholesale; and a graph's tracked-footprint write set is larger than a
single kernel's, so §1.3's cost model applies with a bigger $F_{\text{write}}$ — which is an argument
for write-set-only restore (§9), not against graph replay.

### B. User-driven pass loop — the highest value per unit of work

This is CUPTI's `CUPTI_UserReplay` model and it is startlingly cheap to build on what already
exists. The SDK exposes:

```c
// Tool/application-facing, called on the application's thread.
rocprofiler_status_t rocprofiler_replay_range_begin(rocprofiler_context_id_t ctx,
                                                   const char*              range_name);
rocprofiler_status_t rocprofiler_replay_range_end(rocprofiler_context_id_t ctx,
                                                  int*                     needs_another_pass);
```

and the application writes:

```c
do {
    rocprofiler_replay_range_begin(ctx, "timestep");
    /* the real work: many kernels, many streams, MPI, collectives, host code */
    rocprofiler_replay_range_end(ctx, &again);
} while (again);
```

The SDK's job per pass is only: rotate the counter group (the existing per-pass local-context
mechanism), and optionally snapshot/restore the tracked footprint at the range boundary (the
existing `snap`/`restore`). It does **not** need to intercept, record, or re-issue anything, and it
does not need to reason about idempotence, because the application has taken responsibility for
running the range again. Everything hard becomes the application's problem — and for the codes that
matter it is not a problem at all, because they already have a loop: HPC codes have a timestep loop,
inference servers have a request loop, MD codes have an MD step, training has a training step.

Three properties make this the right first investment:

1. **It is the only design in this table that handles collectives and MPI.** All ranks execute the
   range again, so peer progress happens naturally, and the application's own barrier provides the
   cross-rank rendezvous that §3.3 otherwise needs a bespoke subsystem for.
2. **It preserves the real execution regime.** Concurrency, stream overlap, cache behaviour and
   clock state are those of an actual application iteration, not of a serialized, drained,
   snapshot-perturbed dispatch (§1.4).
3. **Snapshot/restore becomes optional.** If the range is naturally repeatable — a timestep over the
   same input, an inference request replayed from the same prompt, an MD step from a checkpoint —
   the application can assert that and skip snapshot entirely, which removes the entire $P \cdot F$
   host-link cost that dominates the cost model. This is the single biggest performance lever
   available, and it is only reachable by moving the loop into the application.

The obvious cost is that it requires source annotation. That is the same trade every range-based tool
makes (NVTX, ITT, roctx), and for the specific case of "I need a full `rocprof-compute`-style profile
of one timestep of my production code" it is a two-line change to the application.

### C. API-capture range replay — the expensive one, and what it really requires

This is Nsight Compute's range replay: capture the API calls in the range on the first pass, then
replay the captured range P times without re-executing application host code. It is the only design
that profiles a multi-kernel range *without* application cooperation, which is why it exists — and
it is a large subsystem. For ROCm the required pieces are:

* **API recording** for everything inside the range that has device-visible effect: kernel launches
  (already intercepted), `hipMemcpy*`/`hsa_amd_memory_async_copy`, `hipMemset*`, allocations and
  frees, stream and event creation/record/wait, graph launches, and module/global writes
  (`hipMemcpyToSymbol`).
* **Host-memory save/restore.** A range whose kernels write host-visible memory, or whose replay
  re-executes a D2H copy, must save and restore the host side too. Nsight Compute does exactly this
  and exposes `--disable-host-save`/`--disable-host-restore` because it is both expensive and
  occasionally wrong; that pair of options is a good indicator of how much residual risk this piece
  carries.
* **Pointer translation.** Replayed allocations may not land at the original addresses, so recorded
  kernargs and any device-resident pointers must be translated. AMD already has this problem solved
  in two places worth reusing rather than reinventing: the HIP Record & Replay work in the ROCm
  runtime, and Kerncap's VA-faithful capture (which sidesteps translation by restoring allocations at
  their original virtual addresses — the more robust approach, and the one to copy).
* **Non-replayable API detection.** IPC handles, peer enablement, RDMA registration, graph
  instantiation, and anything with cross-process effect must abort the capture rather than be
  silently replayed.
* **Idempotence and ordering policy.** Whether replay re-issues on the same streams (preserving
  concurrency, which is the whole point) or serializes; and whether dependent-kernel chains that
  extend beyond the range boundary are detected (Nsight Compute has an explicit
  `--disable-dependent-kernel-detection` option, which tells you the problem is real).

The honest assessment: this is the most capable design and the least appropriate *first* one. It
duplicates capabilities that CUPTI provides natively on NVIDIA (checkpoint + range profiling APIs),
and on ROCm it would be built from scratch. It should be scoped only after (A) and (B) are shipped,
and it should be built on top of an HRR-style capture layer rather than inside the profiler.

### D. App-range replay — worth naming so it is not confused with the others

Re-run the whole process once per counter group, but only profile the marked range. This is
application replay plus filtering; `rocprofv3` can already approximate it with roctx range filtering
plus one run per `--pmc` group. It requires deterministic application behaviour across runs (so the
range and its dispatches match up), and it pays $P \cdot T_{app}$. It is the baseline that kernel
replay must beat, not a competitor to build.

## 5.2 Recommended sequence

1. **Graph replay (A).** Smallest change to the existing loop, and it unlocks the entire
   graph-captured inference and `torch.compile` world that the current gate excludes. Start with
   graphs containing no allocation nodes and no collective kernels.
2. **User-driven pass loop (B).** The only path to MPI, collectives, and realistic execution
   conditions; and the only one that can eliminate the snapshot cost entirely. Reuses the existing
   pass/local-context machinery almost unchanged.
3. **Write-set restore and device-resident snapshots (§9).** These are prerequisites for range
   replay of any size, because a range's footprint is larger than a kernel's.
4. **API-capture range replay (C).** Only after 1–3, and only on top of a separate capture layer.

## 5.3 API shape that keeps all four consistent

All four designs want the same three tool-facing concepts, which the existing kernel-replay domain
already has in embryo. Generalizing rather than duplicating them is worth doing before a second
replay mode is added:

* **A replay scope**: `{dispatch, graph, api_range, app_range}` — the existing `CONFIG` operation
  carries the scope, and `dispatch_info` becomes a union with a range descriptor.
* **A pass plan**: `pass_count_cb` + `replay_continue_cb` are already scope-independent, and the
  per-agent pass count is already the right granularity.
* **Per-pass service control**: `replay_local_start_context_cb`/`replay_local_stop_context_cb` are
  already scope-independent; their documented coverage gaps (PC sampling and device counting cannot
  be toggled per pass because they are not dispatch-scoped) apply identically to ranges, and are
  actually *easier* to fix at range scope, because a range boundary is a legitimate place to
  start and stop an agent-wide service.

The one genuinely new concept a range needs is a **capture manifest** — what state the SDK snapshot
covers for this scope, and what it explicitly does not — surfaced to the tool so it can be recorded
in the output. Given how much of this study is about the gap between "device state a kernel can
write" and "device state the profiler restores", making that gap machine-readable per replayed
region is the single most useful piece of API to add.
