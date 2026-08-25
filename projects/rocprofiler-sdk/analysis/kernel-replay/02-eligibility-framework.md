# 2. A decision procedure for "can this kernel be replayed?"

Per-application verdicts (§4) are only useful if they come from a stated procedure. This section
gives one: five questions, asked in order, each with a defined outcome. The point of the ordering is
that the *safe* failures come first and the *silent* failures come later, which is the opposite of
how a user encounters them.

## 2.1 The five questions

**Q1 — Submission shape.** Is the submission exactly one packet, containing exactly one kernel
dispatch, outside a HIP graph launch?
*If no:* the dispatch runs once, a one-shot warning is logged, and no counters are collected for it.
**Outcome D (declined, safe).** No correctness risk; the cost is silently missing data, which is why
the tool layer needs to report declines explicitly (R11).

**Q2 — Allocator provenance of the write set.** Is every byte the kernel writes inside coarse-grained
device VRAM that was allocated through `hsa_amd_memory_pool_allocate` / `hsa_memory_allocate`,
attributed to the replaying agent, and not carrying the executable flag — or a module-scope
`__device__`/`__constant__` variable?
*If no:* the untracked part of the write set accumulates over P passes. **Outcome C (unsound,
silent).** Note the asymmetry that is easy to get backwards: *accumulating into a tracked buffer is
fine* (restore reverts the accumulator, so each pass starts from the original value); it is
accumulating into an **untracked** buffer that is wrong. The distinction is entirely about
provenance, not about the kernel's arithmetic.

**Q3 — Isolation of the write set.** During the window, is any non-AQL actor reading or writing those
buffers — an SDMA/blit copy, a peer agent over XGMI, another process through IPC, the host CPU
through a mapped pointer, a NIC doing RDMA, or a video/display engine?
*If yes:* **Outcome C (unsound, silent)** for restore-clobbers-their-write, and a data race for
their-write-clobbers-restore. Nothing detects this today.

**Q4 — Progress dependency.** Can the kernel complete without any external agent making progress
concurrently?
*If no* — a collective spinning on a peer's flags, a persistent kernel waiting for host-side work, a
producer/consumer pair split across streams — the pass hangs, the bounded drains expire, and
`ROCP_FATAL` terminates the process. **Outcome E (fatal).** This is the only outcome that is worse
than not profiling at all, and it is why R2 and R4 are P0.

**Q5 — Economics.** Is $P \cdot F$ over the host link acceptable, and does $F$ fit in host RAM
(§1.3)?
*If no:* either the run becomes unusably slow, or `snap()` fails on host allocation and the dispatch
silently runs unprofiled. **Outcome B (sound but uneconomic)** or **D**.

A kernel that passes all five is **Outcome A**: soundly and cheaply replayable.

## 2.2 Allocator provenance is the load-bearing question

Q2 decides most real verdicts, and the answer is a property of the *framework*, not of the kernel.
The mapping that matters:

| Allocation path | Tracked? | Consequence |
|---|---|---|
| `hipMalloc`, default flags | **yes** | the good case; restore works |
| `hipExtMallocWithFlags(hipDeviceMallocFinegrained)` | no | routes to a fine-grained *device* pool |
| `hsa_amd_memory_pool_allocate` directly (HSA apps, RCCL, some libraries) | **yes**, unless the executable flag is set | good case |
| `hipHostMalloc` / pinned host | no | writes accumulate |
| `hipMallocManaged` / unified memory (XNACK) | no | writes accumulate |
| `hipMallocAsync` / stream-ordered memory pools | no | writes accumulate; **Kokkos default on ROCm 6.x** |
| `hipMemAddressReserve` + `hipMemMap` (VMM) | no | writes accumulate; PyTorch `expandable_segments:True` |
| coarse VRAM with `HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG` | no (side inventory) | writes accumulate |
| memory owned by another agent (peer, P2P) | no | writes accumulate; also breaks isolation |
| IPC-attached memory from another process | no | writes accumulate; also cross-process corruption |
| fine-grained device memory | no | writes accumulate |
| MI300A APU shared HBM, host-coherent | effectively no | the tracked/untracked distinction largely dissolves |

The first row deserves a caveat, because "`hipMalloc` is always coarse-grained" is false.
`Device::deviceLocalAlloc` selects `gpu_ext_fine_grained_segment_` when `pseudo_fine_grain_` is set,
`gpu_fine_grained_segment_` when `atomics_` is set, and only otherwise `gpuvm_segment_`
([`rocdevice.cpp`](https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/rocclr/device/rocm/rocdevice.cpp)).
The dominant path is the coarse-grained one, but the flag-dependent exits are real.

Symmetrically, the `hipMallocAsync` row is worse than "stream-ordered pools are untracked" suggests:
`HIP_MEM_POOL_USE_VM` defaults to true in
[`flags.hpp`](https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/rocclr/utils/flags.hpp),
so the default pool is backed by `VmHeap` → `hsa_amd_vmem_handle_create`/`hsa_amd_vmem_map`. A
stream-ordered allocation is therefore untracked *twice over*, and neither entry point is wrapped.
(Note also that the public docs give `HIP_MEM_POOL_SUPPORT` a default of 0 while the source says
true — query `hipDeviceAttributeMemoryPoolsSupported` at runtime rather than trusting either.)

Three implications worth stating plainly:

1. **A framework's default allocator decides whether its kernels are replayable, and users can change
   that default with an environment variable.** A workload that is Outcome A today becomes Outcome C
   tomorrow because someone set an allocator-configuration variable to reduce fragmentation. This is
   not a hypothetical: expandable-segment and async-pool allocator modes exist precisely because
   they are better for large models, and both move memory out of the tracked set. A profiling feature
   whose soundness depends on an environment variable the user has never heard of needs to *detect*
   the condition, not document it.
2. **Coverage is measurable, so it should be measured.** At snap time the mechanism knows the tracked
   footprint. It can also enumerate what it declined to track: the executable-flag side inventory
   already exists, and wrapping the managed/async/VMM entry points to count untracked bytes is the
   same pattern. A per-run "tracked N bytes across M regions; observed K bytes of untracked
   device-visible allocations" line converts the central unverified assumption into a number the user
   can act on.
3. **The soundness of a given application can change with the ROCm version, in the unsafe direction
   for older versions.** Kokkos enables `hipMallocAsync` by default for HIP < 7.0.0 and disables it
   at ≥ 7.0.0 (§4.1). So the same LAMMPS binary, same input deck, same GPU, is Outcome A on ROCm 7
   and Outcome C on ROCm 6.4 — and the tool cannot tell the difference from anything it currently
   inspects. This is the strongest available argument for R-new-1 (decline when live VMM mappings or
   stream-ordered pools are detected) over any amount of documentation.

## 2.3 Outcome classes, and what each one deserves from the tool

| Class | Meaning | What the tool should do |
|---|---|---|
| **A** | sound and cheap | replay, record conditions (R11) |
| **B** | sound, uneconomic | decline by default with a diagnostic naming footprint and projected cost; allow an explicit override (R6) |
| **C** | unsound, silent | detect what is detectable (untracked coverage ratio, peer pointers, async-copy overlap) and decline; where undetectable, surface via verify mode (R9) |
| **D** | declined, safe | record the decline and the reason so missing rows are explained (R11) |
| **E** | fatal | detect and decline (R4); never abort (R2) |

The current implementation has correct behaviour for D, correct-but-brutal behaviour for E, no
behaviour at all for B and C, and no reporting for any of them. That gap — not the core snap/restore
logic, which is sound for what it covers — is what limits deployability.

## 2.4 Kernel-level patterns, independent of application

Certain kernel shapes recur across every domain in §4, and their verdicts are structural:

| Pattern | Verdict | Reason |
|---|---|---|
| Pure BLAS-like: reads A, B from tracked VRAM, writes C in tracked VRAM | **A** | textbook case; the entire write set is tracked |
| Elementwise / activation / normalization on tracked tensors | **A** | same |
| Stencil / interior update on tracked arrays | **A** | same |
| Reduction with FP atomics into a tracked buffer | **A for correctness, caveat for measurement** | restore reverts the accumulator; but the kernel is nondeterministic, so counters legitimately vary across passes (§6.3 O6) — exactly the case R8's canary is for |
| RNG-consuming kernel with generator state in tracked VRAM | **A**, and unusually good | restore rewinds the generator, so every pass draws the *same* numbers — better reproducibility than the application itself has |
| RNG-consuming kernel with counter/offset held in host memory or advanced host-side | **C** | the offset is untracked, so passes diverge |
| Kernel writing an output buffer obtained from a stream-ordered or managed allocator | **C** | provenance, not arithmetic |
| Kernel whose grid or work assignment is read from device memory (MoE routing, sorted indices, work queues) | **A if the index buffer is tracked** | restore rewinds the routing decision too, which is what makes it reproducible |
| Persistent / cooperative kernel with host or peer handshake | **E** | cannot complete in isolation |
| Collective (ring/tree allreduce, all-gather) | **E**, plus peer corruption | §3.2 |
| Kernel that consumes a buffer being filled concurrently by SDMA (pipelined H2D + compute) | **C** | the copy is invisible to the window |
| Graph-captured block of work | **D today**, and the best **A** candidate after R15 | §5.1A |

Read together with §2.2, this table is most of the per-application answer already: **the question is
almost never "is this kernel replayable", it is "does this application allocate the kernel's outputs
in a way the tracker sees, and does anything else touch them while the window is open".**
