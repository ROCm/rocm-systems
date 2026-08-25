# 10. Portability

Two distinct questions get conflated here. (a) Would this *design* work on other stacks? (b) Are the
*ideas* portable, independent of the implementation? The answer to (a) is largely no, and the answer
to (b) is largely yes — the ideas are portable precisely where the implementation is not.

## 10.1 What is AMD-specific, and why

Three load-bearing properties of the design have no counterpart on any other accelerator stack.

**Re-submission of a captured AQL packet.** The mechanism keeps a copy of the 64-byte dispatch packet
and rewrites it into the ring N times, editing `completion_signal` to suppress all but the last. This
requires (i) a user-visible command ring in coherent memory, (ii) a packet format the tool may
author, and (iii) a doorbell the tool may ring. HSA gives all three; nothing else does. CUDA's
launch path terminates in an opaque driver ioctl. Level Zero command lists are opaque handles.
Vulkan/SYCL are further removed still. On every other stack, "replay a dispatch" must mean *re-issue
the launch through the runtime API*, which changes what is being measured: you get a new launch, new
kernarg staging, and possibly a new stream assignment.

**Suppressing completion by editing the packet.** The exactly-once completion property (§6, O4) falls
out of the packet being a mutable data structure. Elsewhere it must be reconstructed from
runtime-level primitives — CUPTI does it by owning the launch callback and not returning to the
application until the final pass.

**Agent-scoped memory-pool enumeration.** The snapshot is defined over "coarse-grained VRAM pools of
this agent." No other stack has this taxonomy. CUDA has no coarse/fine-grained distinction at all;
the closest analogue is device vs. managed vs. pinned, which is a different partition of a different
space. Level Zero has device/host/shared USM. So the *carveout list* — the specific set of things
excluded — is not portable even as a concept, because the categories don't exist elsewhere.

## 10.2 What is portable

| Idea | Portability | Notes |
|---|---|---|
| Pass count is the tool's decision, per dispatch | High | Directly transfers to CUPTI's `numReplayPasses`; L0 tells the tool the count instead and expects the tool to drive |
| Callback at the launch boundary, not inside it | High | CUPTI's `KERNEL_LAUNCH_BEGIN/END` is exactly this shape |
| Save/restore around the passes | High | Universal; the storage and scope policy is where stacks differ |
| Reconfigure counters between passes | High | Every stack supports it |
| Exactly-once completion contract | High | The requirement is portable; the packet-edit implementation is not |
| Graceful decline on unsupported constructs | High | And the most under-implemented idea in this design (§9 R2) |
| Restore only the write set | High | NVIDIA already ships it; AMD does not (§9 R13) |
| Tiered storage: device → host → filesystem | High | NVIDIA ships three tiers; AMD has one, unpinned host |
| Per-agent writer lock | **Low** | Unsound under P2P and under CPX partitioning (§3) |
| Agent-wide drain as the isolation mechanism | **Low** | Cannot see SDMA, RDMA, or display-engine writers |

## 10.3 NVIDIA / CUDA

Porting is technically straightforward and strategically pointless, because NVIDIA already ships a
strictly more capable version.

The building blocks exist and compose: `CUPTI_ACTIVITY_KIND_KERNEL` callbacks for the launch
boundary, `cuptiProfilerReplayInProcess`/`cuptiProfilerBeginPass`/`EndPass` for the pass loop, and
the **CUPTI Checkpoint API** (`cuptiCheckpointSave`/`cuptiCheckpointRestore`) for the snapshot. The
Checkpoint API is the direct analogue of `memory_snapshot`, and comparing them is instructive
because it is the same idea built with more options:

* Scope is a CUDA context, not an agent.
* `reserveDeviceMB` / `reserveHostMB` let the tool place the snapshot in device memory first, then
  host, then filesystem. AMD has exactly one tier, and it is unpinned host (§1.3.1).
* `CUPTI_CHECKPOINT_OPT_TRANSFER` enables dirty-block detection so only modified regions are
  restored. This is §9 R13, already shipped.
* `allowOverwrite` makes the destructive-restore policy explicit rather than implicit.

Nsight Compute layers four replay modes on top: `kernel` (per-launch, the AMD design),
`application` (rerun the process — what `rocprofv3 --pmc --pmc` does), `range` (a marked API range,
preserving concurrency inside it), and `application-range`. It also does the things AMD's design
defers: it flushes L1/L2 before every pass by default (`--cache-control all`) and can lock clocks
(`--clock-control base`), so measurement conditions are equalized rather than merely hoped for
(§9 R10).

Two features are direct prior art for gaps identified in §3:

* **`--lockstep-kernel-launch`** (Nsight Compute 2025.3+) synchronizes replay across processes using
  a TCP or shared-memory communicator, added explicitly for NCCL-style collectives. This is the
  cross-rank rendezvous §3.3 says does not exist on AMD. It exists; it is just not AMD's.
* **Metric Distributor** splits a metric set across N processes or GPUs, collecting different
  counters on different ranks of the same run. This is the *dual* of replay — zero extra passes, at
  the cost of assuming ranks are statistically comparable — and it is a strictly better answer for
  multi-rank MPI than replay can ever be.

Nsight Compute's own documentation names the same hazards this study derives independently:
host-allocation instability under range replay, and undefined behaviour for kernels that depend on
host or peer state.

## 10.4 Intel / Level Zero

The architecture inverts the responsibility, which makes AMD's design *unportable in the direction
that matters*.

`zetMetricGroupCalculateMultipleMetricValuesExp` and the concurrent-metric-groups query tell the tool
up front how many groups can be collected together, and therefore how many passes are required. The
tool then drives the loop — usually by asking the application to iterate. Level Zero has no
mechanism for a tool to re-submit a command list transparently: command lists are opaque handles,
immediate command lists submit on append, and there is no writable packet. So the AMD design's core
move is simply unavailable, and the Intel model is closer to §5's "user-driven pass loop" (design B)
than to kernel replay.

That is a useful signal for the roadmap: the design that ports cleanly to Intel is the one AMD has
not built yet.

## 10.5 CPUs, DPUs, FPGAs, NPUs

There is no kernel-replay equivalent anywhere outside GPUs, and the reason is uniform: replay exists
to work around a *fixed, small number of counter registers multiplexed across a re-executable unit of
work*. Remove either half and the technique has no purpose.

* **CPUs** solve the counter shortage with time multiplexing. `perf` rotates event groups on a timer
  and scales results by the fraction of time each group was scheduled, accepting sampling error
  instead of re-execution. There is no unit of work to replay, and no way to roll back memory
  cheaply. Intel's Top-Down Microarchitecture Analysis is explicitly designed so that each level
  fits available counters, avoiding multiplexing rather than replaying.
* **DPUs** (BlueField, Pensando) expose system-level telemetry — DOCA Telemetry, packet counters,
  ARM PMU — for a continuously running dataplane. There is no dispatch boundary. Replaying a
  processed packet is meaningless because the state is the network, not a buffer.
* **FPGAs** instrument the pipeline instead: Vitis Analyzer and Intel Signal Tap insert
  hardware probes at build time. If you need different counters you rebuild, which is expensive but
  needs no replay. Emulation modes give unlimited observability with no counter limit at all.
* **NPUs/TPUs** (XDNA, TPU, Habana) currently expose fixed hardware trace and profile modes rather
  than user-selectable counter sets. Google's TPU profiler captures a trace window, not counter
  groups. Nothing to multiplex, therefore nothing to replay. This may change as NPU counter
  architectures mature.

The transferable conclusion: kernel replay is a GPU-specific answer to a GPU-specific constraint, and
the *cross-platform* techniques for the same problem are multiplexing (CPU), rebuild-time
instrumentation (FPGA), and distributing counters across ranks (NVIDIA's Metric Distributor). All
three are cheaper than replay and all three trade something replay refuses to trade.

## 10.6 Implications for the AMD design

1. **The exclusion list should be expressed in portable terms.** "Coarse-grained VRAM" is an HSA
   concept. "The set of allocations the tool can prove no other agent may write" is portable and is
   the property actually needed.
2. **A cross-rank rendezvous is a solved problem elsewhere.** If multi-rank support is ever in scope,
   `--lockstep-kernel-launch` is the reference design.
3. **Metric distribution may dominate replay for MPI.** For a 64-rank job, collecting one counter
   group per rank costs zero passes. It requires ranks to be comparable, which for SPMD bulk-synchronous
   codes they usually are. This deserves a line in the roadmap next to range replay.
4. **The tool-driven loop is the portable design.** Design B in §5 is the only one of the four that
   maps onto Level Zero, and it is also the only one that works for collectives. That is not a
   coincidence: both properties come from letting the application define the replayable unit.
