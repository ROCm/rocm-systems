# 1. What the mechanism actually is, and what it costs

Everything in this study is derived from the code on `users/vkale/experimental-kernel-replay`
(`247437ab96`) and `users/vkale/kernel-replay-rocprofv3` (`2dcc24399c`), not from the PR
descriptions. This section fixes the vocabulary and derives the cost model that every later
feasibility verdict depends on.

## 1.1 The mechanism, stated exactly

The replay decision is made inside the AQL packet write interceptor. The gate is a single
conjunction:

```923:923:projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp
    if(has_kernel_replay && pkt_count == 1 && num_dispatch_packets == 1 && !graph_launch_active)
```

so a dispatch is a replay candidate only when (a) some context subscribes to
`ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY`, (b) the submission the runtime handed to the
interceptor contains **exactly one packet**, (c) that packet is a kernel dispatch (plain or
ext-dispatch), and (d) the submission is not part of a HIP graph launch. Cases (b) and (d) warn
once and fall through to the ordinary single-execution path.

If the gate passes, the tool is asked for a pass count, and if it returns `N > 1` (or `0` with a
continue callback) the interceptor executes this sequence **synchronously on the application's
submitting thread**:

| Step | Code | Consequence that matters later |
|---|---|---|
| Reserve dispatch id | `replay_dispatch_id = ++sequence_counter` | all passes and all records share one dispatch id |
| `CONFIG` enter → `pass_count_cb` | `replay_callbacks.cpp:207` | pass count is chosen *per dispatch, per agent* |
| Take per-agent **writer** lock | `std::unique_lock<std::shared_mutex>` on `agent_replay_mutex(agent_id)` | the whole agent is excluded for the entire window; other agents are not |
| Queue drain | barrier packet + `hsa_signal_wait_scacquire`, 12 × 5 s slices then `ROCP_FATAL` | up to 60 s, then the process dies |
| Agent-wide drain | poll `active_async_packets()` on every sibling queue, 60 s cap then `ROCP_FATAL` | only sees work that went through an *intercepted AQL queue* |
| `snap(agent)` | full `hsa_memory_copy` device→host of every tracked region | cost is proportional to the agent's whole tracked footprint |
| For each pass | `PASS` enter → resubmit packet with app completion signal suppressed → `queue.sync()` drain → `PASS` exit → `should_continue_replay` → `restore()` | `restore()` is a full, unconditional host→device copy of every captured region |
| After the loop | trailing barrier packet carrying the app's original completion signal | the application observes exactly one completion |

Two properties of this design drive most of the analysis:

1. **The snapshot unit is the agent, not the kernel.** `snap()` captures every tracked coarse-grained
   VRAM allocation owned by the agent (`memory_tracker::snap_inventory(agent)`) plus every
   module-scope `__device__`/`__constant__` variable found by iterating the loaded executables'
   agent symbols. It does not consult the dispatch's kernarg segment, so a kernel touching 4 MB in a
   process that has 140 GB resident still pays 140 GB per copy.
2. **The restore unit is the whole snapshot, not the write set.** `restore()` walks every block and
   copies it back. There is no dirty tracking, no page granularity, no hashing (the design doc says
   so explicitly, and lists hashing as future work).

## 1.2 What is inside the snapshot boundary, precisely

`query_alloc()` is the whole policy:

```52:55:projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/kernel_replay/utils.cpp
    const bool is_kernarg = (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0;
    const bool is_coarse =
        (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) != 0;
    q.trackable = is_coarse && !is_kernarg;
```

Tracked, therefore restored:

* memory obtained through `hsa_amd_memory_pool_allocate` or `hsa_memory_allocate` that
  `hsa_amd_pointer_info` reports as coarse-grained and not kernarg-init, attributed to
  `info.agentOwner`;
* module-scope device and constant variables, discovered at snap time (not at load time, because
  constant memory may not be populated at load).

Not tracked, therefore **accumulating across passes**:

* anything allocated through a path that does not bottom out in those two HSA entry points —
  stream-ordered pools (`hipMallocAsync`), managed/unified memory (`hipMallocManaged`), and
  virtual-memory-management mappings (`hipMemMap` / `hsa_amd_vmem_*`), which is what PyTorch's
  `expandable_segments` uses;
* host memory, fine-grained memory, and kernarg memory (excluded by the predicate above);
* coarse-grained VRAM carrying `HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG`, which is diverted to a
  side inventory (`unsupported_executable_inventory`) and deliberately omitted, because the HIP
  runtime keeps such allocations live and declining every replay would disable the feature;
* memory owned by a *different* agent, including peer memory the replaying agent can write over
  XGMI, and IPC-attached memory owned by another process;
* GPU caches, and every other piece of hardware state.

The set difference between "device state a kernel can write" and "device state this mechanism
restores" is the single largest source of the feasibility limits in sections 2–4.

## 1.3 The cost model

Let

* $F$ = tracked footprint on the replaying agent at snap time (bytes),
* $P$ = pass count for the dispatch (for `rocprofv3`, the number of counter groups for that agent),
* $B_\downarrow, B_\uparrow$ = achieved device→host and host→device copy bandwidth for the snapshot
  path,
* $t_k$ = kernel execution time, $t_h$ = per-pass handler drain plus counter readout and callback
  cost,
* $D$ = agent-wide drain time, $L$ = writer-lock acquisition wait.

The replay window adds

$$\Delta T(\text{dispatch}) = L + D + \underbrace{\frac{F}{B_\downarrow}}_{\text{1 snap}} + \underbrace{(P-1)\frac{F}{B_\uparrow}}_{\text{restores}} + (P-1)\,t_k + P\,t_h$$

so **bytes over the host link per replayed dispatch is $P \cdot F$** — one snapshot plus $P-1$
restores — and **host RAM held for the window is $F$**, duplicated per concurrently replaying agent.
This is the model the in-tree perf regression test encodes; it calls it the "Figure 5 cost model
(P × footprint bytes over the host link)" and asserts a 4 GB/s floor.

### 1.3.1 The bandwidth term is worse than PCIe

`snap()` copies into `std::vector<char> host_copy` (`memory_snapshot.hpp:47`) via
`hsa_memory_copy`. That destination is ordinary, **unpinned** host memory. A DMA engine cannot
target unpinned pages, so the runtime must either stage through a pinned bounce buffer or fall back
to a CPU copy; either way the achieved rate is well below the link's pinned-transfer rate. The
in-tree test's 4 GB/s floor and its "conservative host-link floor … intentionally low" comment are
consistent with that. The practical consequence: the cost model should be evaluated at single-digit
GB/s, not at the 25–50 GB/s a pinned PCIe Gen4/Gen5 x16 transfer achieves.

Three separate inefficiencies compound here, all fixable and all quantified in section 9:

1. unpinned destination (staging or CPU copy instead of DMA);
2. one blocking `hsa_memory_copy` per region, issued serially, so nothing overlaps and every region
   pays full latency — the test suite already observes a "fixed HIP-runtime inventory … constant
   offset";
3. the transfer crosses the host link at all, when a device-resident snapshot would run at HBM
   bandwidth (Nsight Compute explicitly prefers device memory for its save/restore buffer and only
   spills to host).

### 1.3.2 Break-even against application replay

The status quo for multiple counter groups is application replay: run the program once per group.
With startup and steady state folded into a single application runtime $T_{app}$, and $K$ replayed
dispatches:

$$T_{\text{app-replay}} \approx P \cdot T_{app}, \qquad T_{\text{kernel-replay}} \approx T_{app} + K \cdot \Delta T$$

Kernel replay wins iff $K \cdot \Delta T < (P-1)T_{app}$. Dropping the small terms and taking
$B_\downarrow \approx B_\uparrow = B$:

$$\boxed{K \cdot F \;\lesssim\; B \cdot T_{app} \cdot \frac{P-1}{P}}$$

**The bytes you are about to move over the host link must be less than the bandwidth-time product
of one application run.** That single inequality explains every feasibility verdict in this study,
and it has three immediate corollaries:

* **Footprint, not kernel count, is the binding constraint for AI.** At $F$ = 140 GB and
  $B$ = 6 GB/s, one replayed dispatch with $P$ = 8 costs $8 \times 140/6 \approx 187$ s. Against a
  600 s training-step benchmark, application replay costs $7 \times 600 = 4200$ s, so kernel replay
  wins only for $K \lesssim 22$ dispatches. Kernel filtering is not an optimization here, it is a
  precondition.
* **Dispatch count, not footprint, is the binding constraint for HPC.** At $F$ = 200 MB and
  $B$ = 6 GB/s, one dispatch with $P$ = 8 costs ~0.27 s. That is cheap — until you notice that an
  iterative code launches the same kernel 10⁴–10⁶ times. Unfiltered, $K = 10^4$ costs 2670 s
  against an application replay cost of $7 T_{app}$; for a 60 s benchmark, kernel replay loses by
  6×.
* **There is a window where it wins decisively, and it is worth naming**: expensive-to-reach
  dispatches in long-running processes with modest live footprints — a specific GEMM inside a
  10-minute inference server warm-up, a specific fused attention kernel after 2000 steps, one
  ligand-scoring kernel after a 5-minute setup. That is precisely the case application replay
  handles worst, because it pays the entire setup cost $P$ times.

### 1.3.3 Host RAM is a hard ceiling, not a cost

$F$ bytes of host RAM are held for the window, per replaying agent. On an 8×MI300X node
(192 GB HBM each), a process whose per-GPU footprint is 150 GB and whose eight agents replay
concurrently — which the per-agent lock design explicitly permits — demands 1.2 TB of host RAM for
snapshots alone. `snap()` degrades safely (a `std::bad_alloc` on `host_copy.resize` returns
`ok = false`, the warning fires, and the dispatch runs once), so the failure mode is silent
loss of profiling data rather than a crash. That is the right choice, but it means **large-footprint
runs can produce empty counter output with only a warning in the log**, which the tool layer should
surface as an explicit "replay declined, footprint too large" diagnostic rather than a missing row.

## 1.4 Measurement validity: replay changes what you measure

This is separate from correctness, and it is the point most easily lost. Even a *perfectly correct*
replay does not reproduce the hardware conditions of the original execution:

* **Caches and TLBs are not restored, and not flushed.** Pass 0 runs after a multi-second stall and
  a full-footprint device→host read; passes 1..P−1 run immediately after a full host→device write of
  the same footprint. L2/MALL and TLB state at kernel entry therefore differs between passes and
  differs from the un-profiled execution. Any cache-hit-rate, memory-latency, or duration metric is
  affected. Nsight Compute treats this as a first-class problem and flushes caches between passes by
  default (`--cache-control`) precisely so that passes are comparable; this mechanism does neither.
  Concretely: `TCC_HIT/TCC_MISS`-derived metrics collected in pass *i* are not comparable to those
  collected in pass *j*, and derived metrics that combine counters from different groups are
  computed from inconsistent cache conditions.
* **Clock and power state drift.** The window stalls the agent for seconds (drain + snapshot),
  which lets clocks fall, then hammers the same kernel P times back to back, which heats and can
  trigger throttling. Nsight Compute exposes `--clock-control` for the same reason.
* **The agent is serialized.** The writer lock plus agent-wide drain means the replayed dispatch
  runs with *no* concurrent work on the GPU, even in an application that normally runs several
  streams concurrently. The measured kernel is therefore not the kernel as it runs in production:
  it has the whole L2, the whole memory system, and all CUs to itself. For kernels whose interesting
  behavior *is* the contention (small kernels co-scheduled with a big GEMM, communication overlapped
  with compute) replay measures a counterfactual.

None of this makes the feature wrong; it makes the feature's output conditional. The honest framing
is: **kernel replay measures an isolated, cold-ish, serialized execution of the kernel on
snapshot-identical inputs.** That is exactly what you want for kernel-level optimization work and
exactly what you do not want for whole-application contention analysis. Section 9 recommends
recording the conditions in the output so the distinction is visible to whoever reads the CSV.

## 1.5 Where the pass count comes from in `rocprofv3`

`--kernel-replay-beta-enabled` sets `ROCPROF_KERNEL_REPLAY=1`, requires `--pmc`, and merges CLI and
input-file counter groups while disabling the CLI multipass (one-run-per-group) path. In the tool,
`kernel_replay_pass_count_callback` returns the number of counter groups for the dispatch's agent,
and the tool publishes `tl_current_replay_pass` so each buffered counter record carries the
`replay_pass` it was collected in. Two consequences worth stating:

* $P$ is chosen per agent, so a heterogeneous node whose agents need different group counts works
  without a global pass count. That is a genuinely good design decision and it is the main thing
  the callback-tracing redesign bought.
* $P$ is exactly the number of groups the user asked for. A user who writes a large `--pmc` list
  gets a large $P$, and by §1.3.2 the cost is linear in $P \cdot F$. `rocprof-compute`-style full
  profiles imply large $P$; combining them with a large-footprint application is the worst case for
  this feature and should be refused rather than attempted.
