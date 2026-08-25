# 11. Prior art

The relevant question is not whether kernel replay is novel — it is not, NVIDIA has shipped it since
Nsight Compute 2019.x — but *where in the established design space this implementation sits*. The
answer is consistent across every axis: **at the most conservative and most expensive end.**

## 11.1 The comparison

| Axis | Nsight Compute / CUPTI | AMD kernel replay | Gap |
|---|---|---|---|
| Replay granularity | kernel, range, application, application-range | kernel only | §5 |
| What is restored | dirty blocks only (`CUPTI_CHECKPOINT_OPT_TRANSFER`) | entire tracked footprint, unconditionally | §9 R13 |
| Snapshot residence | device → host → filesystem, tool-tunable | unpinned host `std::vector` only | §9 R14 |
| Host/pinned memory | saved and restored | excluded | carveout |
| Cache state | flushed before each pass by default | untouched | §9 R10 |
| Clocks | lockable (`--clock-control base`) | untouched | §9 R10 |
| Cross-process replay | `--lockstep-kernel-launch`, TCP/shmem communicator | none | §3.3 |
| Counter distribution across ranks | Metric Distributor | none | §10.5 |
| Failure on unsupported input | documented undefined behaviour, warnings | `ROCP_FATAL` abort after 60 s | §9 R2 |
| Restore correctness self-check | none published | none | §9 R9 |

Nsight Compute's documentation is candid about the same hazards this study derives from first
principles — host-allocation instability under range replay, undefined behaviour for kernels
depending on host or peer state — which is corroborating evidence that §6's proof obligations are the
right ones rather than an artifact of reading one implementation.

## 11.2 AMD's own prior art, which is closer than the NVIDIA comparison

Two in-house precedents matter more than they appear to have been used.

**HIP Record & Replay (HRR)**, in `clr`, records HIP API calls and replays them, and — crucially —
already contains a *validation* mechanism: it copies device buffers back to the host after replay and
compares against recorded values with a tolerance. That is §9 R9's verify mode, already implemented
once in the same organization for an adjacent problem. Whatever its limitations, the design question
"how do you know restore worked?" has a local answer that was not reused.

**Kerncap** is the closest analogue: HSA-level dispatch interception with VA-faithful capture — it
replays a single kernel into the *same virtual addresses*, which sidesteps the address-remapping
component of the refinement map in §6 entirely. It also supplies the only public AMD measurements of
snapshot cost on real workloads: on the order of 1.3–1.7 GB/s effective, with an 8.4 GB LAMMPS
snapshot taking ~1.5 s and a vLLM capture around 30 GB. **Those numbers are below the ~4 GB/s floor
§1.3.1 derives from the in-tree microbenchmark, which means the cost model in §1 is optimistic, not
pessimistic.** Any experiment should measure B rather than assume it (§7.0.2).

**`rocprofv3` multipass** is the incumbent baseline and it is application replay: "Each pass runs the
application from start to finish," incompatible with `--pid` and `--collection-period`. Every claim
about kernel replay's value is implicitly a claim relative to this.

**`rocprof-compute --iteration-multiplexing`** is the incumbent *competitor*, and it is the more
serious one. It divides counters across successive dispatches of the same kernel, requires no
snapshot, no drain, and no restore, and works under MPI — where kernel replay does not. AMD's own
documentation states the terms honestly: roughly 15 counter subsets, ~50 dispatches per kernel
recommended for coverage, kernel identity keyed by name (optionally plus launch parameters), and
values **imputed** rather than measured where coverage is short. So the real comparison is:

| | Multiplexing | Kernel replay |
|---|---|---|
| Requires many dispatches of the same kernel | yes (~50) | no |
| Counters from the *same* dispatch | no | yes |
| Works under MPI / collectives | yes | no |
| Memory cost | none | P × footprint |
| Correct for one-shot kernels | no | yes |
| Sound under managed/async allocators | yes | **no** |

Kernel replay's defensible niche is therefore narrow and specific: **kernels that execute few times
(so multiplexing has no samples), on tracked allocations, in a single-process context, where
same-dispatch counter correlation matters.** That niche is real — it is exactly §4.8's long-setup
short-kernel case — but it is much smaller than "profiling on AMD GPUs," and the study should say so.

## 11.3 Academic precedent worth borrowing

* **Record-and-replay for GPUs** (GPUReplay, RTR, and the CUDA-level record/replay literature) has
  converged on capturing at the *driver command* level rather than the API level, for the same reason
  this design intercepts AQL packets: it is the only place where the semantics are unambiguous.
* **Deterministic replay under nondeterminism** distinguishes value determinism from execution
  determinism. §4.4's OpenMM case is exactly this: replay guarantees identical inputs, not identical
  execution, and conflating the two produces false alarms in validation. The canary counter (§9 R8)
  is the standard resolution.
* **Checkpoint/restart** (BLCR, DMTCP, CRIU, and GPU-aware CRAC/CRUM) has established that
  incremental and dirty-page checkpointing is not an optimization but the difference between viable
  and non-viable at scale. Every mature system in this lineage does write-set tracking. This design
  does not, yet.
* **Linux `perf` event multiplexing** is the CPU world's settled answer and it chose *scaling with
  error bars* over re-execution. Notably, `perf` reports the multiplexing fraction so the user can
  judge the error. Neither AMD tool reports an analogous confidence signal for imputed multiplexed
  values, which is a smaller and cheaper gap than any replay feature.

## 11.4 The uncomfortable synthesis

Three facts, taken together, define the strategic position:

1. Every capability AMD's design lacks (write-set restore, tiered storage, cache/clock control,
   cross-process lockstep, range replay) is **already shipping** in the NVIDIA stack, so none of them
   are research problems.
2. The incumbent AMD alternative (iteration multiplexing) is cheaper, MPI-safe, and — because it
   never touches memory — **sound under exactly the allocators that make kernel replay silently
   wrong** (§4.1, §4.3).
3. The workloads where replay is uniquely valuable are dominated by graph-captured launches
   (ineligible), collectives (fatal), or preallocating allocators (unaffordable).

None of that argues against the feature. It argues that its value is concentrated in a narrow,
nameable niche, and that the highest-return next steps are the ones that widen the niche
(write-set restore, graph replay, decline-not-abort) rather than the ones that deepen the existing
mechanism.
