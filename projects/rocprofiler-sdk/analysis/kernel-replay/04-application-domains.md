# 4. Per-domain verdicts

Applying §2's decision procedure to named production codes. Verdicts are grounded in each code's
*allocator* behaviour, because §2.2 establishes that provenance, not kernel semantics, decides
soundness.

## 4.1 The finding that reorders everything: Kokkos on ROCm 6.x is silently unsound

`Kokkos::HIPSpace::impl_allocate` uses `hipMallocAsync` when
`KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC` is defined, and the CMake default is

```cmake
if(hip_VERSION VERSION_GREATER_EQUAL 7.0.0)
  set(HIP_MALLOC_ASYNC_DEFAULT OFF)
else()
  set(HIP_MALLOC_ASYNC_DEFAULT ${KOKKOS_ENABLE_HIP})
endif()
```

([`kokkos_enable_options.cmake`](https://github.com/kokkos/kokkos/blob/develop/cmake/kokkos_enable_options.cmake),
[`Kokkos_HIP_Space.cpp`](https://github.com/kokkos/kokkos/blob/develop/core/src/HIP/Kokkos_HIP_Space.cpp)).
So **on HIP < 7.0.0, every `Kokkos::View` in `HIPSpace` is a stream-ordered allocation by default.**
In ROCm's runtime, `hipMallocAsync` with the default pool goes through `VmHeap::Alloc` →
`hsa_amd_vmem_handle_create` + `hsa_amd_vmem_map`, because `HIP_MEM_POOL_USE_VM` defaults to true
([`flags.hpp`](https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/rocclr/utils/flags.hpp),
[`hip_mempool_impl.cpp`](https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/hipamd/src/hip_mempool_impl.cpp)).
The tracker wraps neither of those entry points.

Consequence: for a Kokkos application built against ROCm 6.x defaults, `snap()` returns
`ok = true` having captured almost nothing of consequence, replay proceeds, and every pass after the
first runs on mutated inputs. No warning. This hits LAMMPS KOKKOS, Kokkos LULESH, ExaMiniMD,
Kokkos miniFE, and E3SM's Omega — i.e. most of the HPC portfolio the feature was justified by.

It is also trivially detectable: Kokkos prints the macro state in its own configuration dump. Any
experiment must check it first, and the SDK should decline replay when any VMM mapping is live
(§9 R-new-1).

Two ironies worth recording. First, Kokkos's constant-memory launch path is *accidentally* correct:
the functor is staged via `hipMemcpyToSymbol` into `kokkos_impl_hip_constant_memory_buffer`, a
32 KiB `__constant__` array that the module-variable walk does capture, and because the snapshot is
taken after the agent drain it holds exactly this dispatch's functor. Second, `parallel_reduce`'s
scratch flags are zero at snap time because Kokkos requires kernels to reset them, so restore is a
no-op on them. Both work for non-obvious reasons that a future refactor could break.

## 4.2 HPC

| Code | 1 GPU, 1 rank | ≥2 ranks, GPU-aware MPI | Binding reason |
|---|---|---|---|
| Kokkos (HIP ≥ 7) | **A** | **C** | halo pack buffers are RDMA targets |
| Kokkos (HIP < 7, default) | **C** | **C** | `hipMallocAsync` untracked (§4.1) |
| LAMMPS KOKKOS pair/neighbor | **A** | **C** | `comm device` buffers |
| LAMMPS GPU package | **A**, unrepresentative | **C** | per-timestep H2D/D2H overlap destroyed by the drain; async copies unfenced |
| LULESH / Kokkos LULESH | **A** | **C** | economics at realistic `-s` (below) |
| AMReX, default arena | **D** (host OOM) | **D** | `The_Arena()` preallocates 3/4 of device memory |
| AMReX, managed arena | **C** | **C** | `hipMallocManaged` untracked |
| GROMACS 2025 HIP NBNxM, 1 rank | **A** | **C by default** | direct GPU communication is on by default in 2025 |
| GROMACS with graph scheduling | **D** | **D** | graph decline |
| nekRS advection/Helmholtz | **A** | **C** | gs/gslib + `MPI_Allreduce` |
| QUDA / MILC dslash | **A**, but pollutes autotuning | **C** | policy autotuner "directly write[s] the halo buffer to neighboring GPUs" |
| HPL dgemm | **A** | **C** | look-ahead broadcast |
| HPCG SpMV/SymGS | **A**, hopeless economics | **C** | thousands of small dispatches per iteration |
| Quicksilver | **C** | **C** | "Unified memory is assumed" |
| MPAS-Ocean (OpenACC) | **C** | **C** | half-ported: constant CPU↔GPU transfers, small kernels, non-GPU-aware MPI |

LULESH footprints derived from the `Domain` field list in
[`lulesh.h`](https://github.com/LLNL/LULESH/blob/master/lulesh.h) — 104 B/node + ~268 B/element:
34 MB at `-s 45`, 274 MB at `-s 90`, 4.0 GB at `-s 220`. At `-s 90` with 8 groups that is ~55 ms per
replayed dispatch, which sounds cheap until you count dispatches: ~15 kernel classes × ~900 cycles
≈ 13,500 dispatches, or ~12 minutes of pure snapshot traffic for a run that takes seconds. **Even
the cheapest realistic HPC case loses to application replay unless the user restricts replay to a
handful of dispatches.** Filtering is a precondition, not an optimization.

AMReX is the catastrophic case and it is catastrophic by default: `The_Arena()` is preallocated to
3/4 of device memory ([GPU.rst](https://github.com/AMReX-Codes/amrex/blob/development/Docs/sphinx_documentation/source/GPU.rst)),
so 144 GB on an MI300X. On a normal host the `std::vector::resize` throws, `snap()` returns
`ok = false`, and replay declines — safe and useless. On a fat-memory host it proceeds, at ~29 s per
replayed dispatch. Note that `snap()`/`restore()` are *not* covered by the 60 s bounded drains, so a
giant snapshot does not abort; it just runs, apparently hung.

## 4.3 AI

PyTorch's default caching allocator obtains memory through `hipMalloc`, which
`Device::deviceLocalAlloc` routes to the coarse-grained `gpuvm_segment_` pool — **tracked**. That is
the good case, and it is the only good case:

* `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` uses HIP virtual memory management
  (`hipMemAddressReserve`/`hipMemCreate`/`hipMemMap` → `hsa_amd_vmem_*`) → **untracked**. This is the
  option PyTorch's own OOM message tells users to set.
* `backend:hipMallocAsync` → VM heap → **untracked** (§4.1).
* `hipMallocManaged` → fine-grained host → **untracked**.

So the soundness of profiling a PyTorch job depends on an environment variable, and the failure is
silent. The worst case is an optimizer step: `exp_avg = β₁·exp_avg + (1−β₁)·g` is read-modify-write
in place, so with an untracked allocator the update is applied P times in one logical step — an
effective P× learning rate, no error, no warning, and a loss curve that diverges hours later.

**Eligibility is near zero under graph capture, and graph capture is the default for inference.**
A HIP graph launch is multi-packet *by construction*: `hipGraphLaunch` funnels through one
`amd::AccumulateCommand` whose submission is `VirtualGPU::dispatchAqlPacketBatchFlat`, which reserves
N ring slots in a single `queue_add_write_index_screlease` and rings the doorbell once per chunk
([`rocvirtual.cpp`](https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/rocclr/device/rocm/rocvirtual.cpp),
[`hip_graph_internal.cpp`](https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/hipamd/src/hip_graph_internal.cpp)).
It would fail the single-packet gate even without the graph flag. vLLM V1 defaults to
`cudagraph_mode = FULL_AND_PIECEWISE` on ROCm
([AMD vLLM V1 tuning](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html)),
so decode — where the overwhelming majority of dispatches live — is 0% eligible; prefill under
`PIECEWISE` leaves only the attention kernels eligible. `torch.compile(mode="reduce-overhead")` uses
CUDAGraph Trees and is likewise ~0% eligible in the compiled region. Eager training and plain
`torch.compile` are ~100% eligible.

**RCCL kernels pass the gate, which is the dangerous outcome.** RCCL launches
`ncclDevKernel_Generic_*` through ordinary launch APIs, not graphs, so a collective is a
single-packet single-dispatch submission and the gate provides no protection at all. Three failure
modes, all fatal: (1) the *agent-wide drain* cannot converge because a collective is in flight on a
sibling stream waiting for peers — so replaying an *unrelated GEMM* aborts after 60 s; (2) replaying
the collective itself leaves pass 1 spinning on flags no peer will set again, aborting after 60 s;
(3) even absent the abort, RCCL's LL/LL128 flags live in fine-grained and peer-mapped memory, so a
pass would see reverted data buffers and un-reverted flags — a state the kernel never legitimately
sees. Note the fuse ordering: the SDK's 60 s bound fires an order of magnitude before PyTorch's
`ProcessGroupNCCL` default 600 s watchdog, so the profiled rank aborts with a *drain timeout*
message and the other ranks die ten minutes later on a watchdog. That is a miserable thing to debug.

RNG is the one place the design is better than expected. Outside graph capture,
`philox_cuda_state()` returns the seed and offset as host scalars baked into the kernarg buffer, and
replay resubmits the same packet with the same `kernarg_address`. Dropout is therefore bit-identical
across passes and the host-side generator advanced exactly once. Kernarg memory being *excluded*
from the snapshot is load-bearing here.

Footprints: 82 GiB for Llama 3.1 8B FSDP on 8 GPUs, 67–80 GiB for 70B/405B at scale
([TorchTitan](https://arxiv.org/html/2410.06511v1)), ~177 GB for vLLM at its default 0.92 utilization
of an MI300X. Note that the caching allocator does not return memory to the driver, so the tracked
quantity is *reserved*, not live — one public reference log shows 51 GB allocated against 217 GB
reserved. At 8 groups and ~20 GB/s that is 8×140/20 ≈ 56 s **per replayed dispatch**, and the
break-even from §1.3.2 allows roughly 20–80 profiled dispatches against a 10-minute benchmark.
Aggressive kernel filtering is mandatory.

Host RAM is the other ceiling: the standard deployment is one process per GPU, so eight concurrent
windows on an 8-GPU node hold 8 × footprint. At 170 GB each that is 1.36 TB against a 2 TB node
minimum — and because Linux overcommits, the failure is more likely to be the OOM killer than the
`bad_alloc` that `snap()` handles gracefully.

## 4.4 Drug discovery

**AutoDock-GPU is the best candidate in the domain and also contains the cleanest real-world
example of the silent-corruption failure.** Populations, energies, grid maps and — critically — the
per-thread PRNG state are plain device allocations (`cudaMalloc(&tData.pMem_prng_states, …)` in
[`performdocking.cpp`](https://github.com/ccsb-scripps/AutoDock-GPU)), so restore rewinds the
generator and every pass is bit-deterministic. But in a `MAPPED_COPY` build,
`pMem_gpu_evals_of_runs` — the score-evaluation counter that drives the `--nev` termination
criterion — is `cudaMallocManaged`. Replaying the generate-and-evaluate kernel P times multiplies
the evaluation count by P, the genetic algorithm terminates early, and the docking result changes,
with no warning. One `#define` separates a correct replay from a corrupted scientific result.

Footprint is trivial (tens of MB); the problem is dispatch count, on the order of 10⁴–10⁵ per ligand
at default `--nrun 20 --ngen 42000`.

OpenMM's HIP platform is a clean **A** for nonbonded, PME, bonded and integrator kernels, with a
public one-line benchmark (`python benchmark.py --platform=HIP --test=apoa1pme`). Note
`DeterministicForces` defaults to false, so replay gives identical *inputs* but not identical
*execution* — the canary of §7.0 is how you tell those apart. GROMACS 2025 mainline offloads only
NBNxM kernels on HIP; the feature-complete HIP work is on an AMD-maintained branch. AlphaFold3 and
Boltz-2 have ideal kernel shape (few large fused dispatches) and ruinous footprint, because JAX
preallocates 75% of device memory by default.

## 4.5 Finance

Monte Carlo path kernels are **the best semantic fit in the entire study**: large read-only market
data, per-thread rocRAND/QMC state in `hipMalloc`'d memory (tens to hundreds of MB — a `philox4x32_10`
state is ~48 B, so 16 M threads is ~770 MB), small accumulation buffers, no host interaction
mid-kernel, single dispatch, idempotent under restore. Restoring the generator state is exactly what
makes group-to-group counter deltas meaningful.

Two caveats. Move the accumulator or the generator state to a stream-ordered pool and the printed
price is wrong by roughly a factor of P, silently — which makes FinanceBench's Monte-Carlo app a
five-minute, one-line reproduction of the mechanism's central blind spot, worth attaching to the PR.
And production XVA kernels can be long single dispatches; anything exceeding the 60 s drain bound
aborts the process, which no application-side change can mitigate. Never in a live pricing path
regardless: STAC-A2's warm baseline Greeks time is 7.40 ms, and a replay window is two to three
orders of magnitude larger than the entire measured task.

## 4.6 Robotics and real time

**Offline only, and for a structural reason rather than a policy one.** Robotics pipelines violate
both premises of the isolation argument. Sensor ingest writes device memory without touching an AQL
queue: NVIDIA's Holoscan Sensor Bridge places RoCE v2 RDMA traffic directly into GPU memory with the
CPU bypassed, and the AMD equivalent is documented — "the AMD kernel driver exposes remote direct
memory access (RDMA) through PeerDirect interfaces. This allows network interface cards (NICs) to
directly read and write to RDMA-capable GPU device memory"
([ROCm GPU-enabled MPI](https://rocm.docs.amd.com/en/develop/how-to/gpu-enabled-mpi.html)). Hardware
video decode is the same shape: rocDecode's `OUT_SURFACE_MEM_DEV_INTERNAL` hands back a device
pointer to a surface allocated by the VA-API driver and written by VCN
([rocDecode HW decoder API](https://rocm.docs.amd.com/projects/rocDecode/en/latest/reference/rocDecode-hw-decoder.html)).
Those surfaces are neither in the inventory nor covered by the drain, so each pass reads different —
possibly torn mid-frame — data. **Outcome C, not merely noisy.**

The deadline arithmetic settles the rest. A 4 GB perception footprint at 4 groups is ~0.8 s of copy
traffic per replayed dispatch, against a 33 ms camera period and a 10 ms control period: roughly 24
dropped frames or 80 missed control cycles, DDS deadline/liveliness violations, sensor ring
overflow, and watchdog resets. And a drain that does not converge in 60 s kills the perception
process mid-motion.

The one genuinely good robotics candidate is MPC/MPPI rollout kernels — thousands of independent
rollouts, per-thread RNG, device-resident cost buffers — which are structurally identical to the
finance Monte Carlo case.

## 4.7 Other domains

**Quantum circuit simulation has the best semantics and the worst footprint.** An apply-gate kernel
is one dispatch doing in-place read-modify-write of exactly one device buffer, with no untracked
state and no host interaction — the purest case in this study. The buffer is also the largest object
on the device: complex128 state vectors are 4.3 GB at 28 qubits, 17.2 GB at 30, 68.7 GB at 32. Profile
at n ≤ 26–28 and extrapolate.

**Data analytics is a carveout for two opposite reasons.** RMM's `pool_memory_resource` suballocates
from `cudaMalloc`/`hipMalloc`, so it *would* be tracked — meaning the snapshot is the entire pool
rather than the live columns, and Spark RAPIDS logs pools in the tens of GB for microsecond kernels.
Meanwhile `cuda_async_memory_resource` would *not* be tracked, and ASYNC is the documented Spark
RAPIDS default. AMD ships the ports (hipDF and hipMM reached production in AMD Data Science 25.10),
so this is direct, not analogical.

**JAX/XLA flips from unusable to excellent with two environment variables.** XLA fuses aggressively
into few large kernels — ideal eligibility — but preallocates 75% of device memory by default
([JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)), so the
snapshot is ~144 GB of one region on an MI300X. `XLA_PYTHON_CLIENT_PREALLOCATE=false` makes it the
live working set. This belongs in the kernel-replay how-to.

**Genomics** is footprint-bound: Parabricks `fq2bam` documents a 38–40 GB per-GPU requirement, which
is 8 s per replayed dispatch at 8 groups before anything else. The ROCm-relevant code is
[mm2-gb](https://github.com/Minimap2onGPU/mm2-gb) (minimap2 with GPU chaining, HIP and CUDA,
validated on MI210), whose PAF output is diffable against CPU minimap2 — a rare case with a built-in
oracle. **Graph analytics** (Gunrock has a native AMD backend, `-DESSENTIALS_AMD_BACKEND=ON`) is a
clean **A**: CSR arrays and frontier buffers are ordinary allocations, atomics are irrelevant under
restore, footprint is the graph. **Commercial CFD**: Fluent documents ~1 GB of GPU memory per
million cells (+50% for double precision), so a 20 M-cell case is 20–30 GB; STAR-CCM+ 2406 added
MI300X support and is the one commercial CFD path testable on ROCm. **EDA** (PrimeSim SPICE) has
ideal kernel shape and is closed-source CUDA-only, hence untestable.

## 4.8 The ranked answer

**Best candidates**, ordered by semantic fit × testability × cost:

1. Monte Carlo path-and-reduce kernels with `hipMalloc`-backed RNG state (FinanceBench, rocRAND
   device API) — and MPC/MPPI rollouts, which are the same kernel shape.
2. Quantum state-vector apply-gate at 24–28 qubits.
3. AutoDock-GPU `kernel4`/`kernel_ad` in a non-`MAPPED_COPY` build.
4. OpenMM HIP nonbonded and PME.
5. Gunrock AMD-backend frontier-advance and SpMV; mm2-gb GPU chaining.
6. Any long-setup, short-kernel-of-interest pipeline — this is where the feature beats application
   replay decisively, and the ratio is the headline number worth measuring (§7.3, 2.5).

**Hard carveouts**, each with the clause it violates:

1. Anything fed by PeerDirect/RDMA ingest or a hardware video decoder — violates both the inventory
   premise and the isolation premise.
2. Anything on `hipMallocAsync`, VMM/expandable segments, or managed memory — including Kokkos on
   ROCm 6.x, the default Spark RAPIDS configuration, and Quicksilver.
3. Collectives and GPU-aware-MPI halo exchange — peer corruption, then a job-killing abort.
4. Preallocating framework allocators (XLA 75%, RMM pools, AMReX arena) — snapshot becomes the pool.
5. Live closed-loop and deadline-bound systems.
