# Kernel replay: feasibility, portability and correctness study

A GPU block has a handful of counter registers — 8 in the sequencer, 4 in the L2, 2 in the texture
units — and that budget has not changed across three CDNA generations, while almost every metric worth
reporting is a ratio assembled from several raw events. A tool therefore needs more observations of a
kernel than the hardware will give it per execution, and the only way to close that gap is to run
something more than once.

*Which* thing gets run again is the whole design space. Re-running the application is trivially
correct and costs the application, once per counter group. Rotating counter groups across dispatches
costs almost nothing but never observes two groups on the same execution, so it cannot support a
derived metric about a single event. **Kernel replay** re-executes one dispatch in place, restoring
the memory it reads before each execution, and is the only option that yields several counter groups
from the same execution of the same kernel on the same data.

It buys that by taking on an obligation the other two do not have: it must reconstruct a dispatch's
inputs, and it can be wrong about it. It cannot see which memory a kernel reads — that would require
analyzing its machine code — so it restores a superset it can enumerate and depends on the kernel's
footprint fitting inside. Memory inside the superset but outside the footprint costs time. Memory
inside the footprint but outside the superset produces wrong answers, silently, and *which side an
allocation falls on is decided by the application's choice of allocator rather than by anything about
the kernel*.

This study establishes where that boundary actually falls for real production code, what replay costs
when the terms are measured rather than assumed, which correctness obligations the implementation
discharges and under what assumptions, and whether replay is the right mechanism at all given what
else exists. **[§0](00-introduction.md) is the conceptual introduction** — start there; it develops
the argument above properly and states the cost model.

The short version of the conclusion: the mechanism is sound and the implementation is careful, but its
domain of applicability is narrower than the feature's framing suggests, and the boundary of that
domain is currently invisible to users. The highest-value work is not making replay faster or more
general — it is making the boundary visible, so an application outside it is told so rather than
handed numbers.

Subject of the analysis: [#7960](https://github.com/ROCm/rocm-systems/pull/7960) (SDK service),
[#10439](https://github.com/ROCm/rocm-systems/pull/10439) (`rocprofv3` integration),
[#10193](https://github.com/ROCm/rocm-systems/pull/10193) (integration preview). This directory is
analysis, not product documentation; user-facing documentation lives in
`projects/rocprofiler-sdk/source/docs/` (`how-to/using-kernel-replay.rst` and
`conceptual/kernel_replay/`).

## Contents

| File | What it covers |
|---|---|
| `00-introduction.md` | conceptual introduction: the register-scarcity constraint, the three ways to close the gap, the equivalence claim replay has to make, the mechanism, and the cost model |
| `01-mechanism-and-cost-model.md` | what the code actually does, the snapshot boundary, and the cost model with its break-even inequality |
| `02-eligibility-framework.md` | the five-question decision procedure, allocator provenance table, and outcome classes A–E |
| `03-scale-multi-gpu-and-mpi.md` | per-agent locking at scale, peer access, device partitioning, collectives, MPI |
| `04-application-domains.md` | per-code verdicts for HPC, AI, drug discovery, finance, robotics and five other domains, with allocator evidence |
| `05-range-replay-design.md` | four range-replay designs, why graph launches are the best replay unit, recommended sequence |
| `06-formal-correctness.md` | state model, six proof obligations, liveness hazards, TLA+ specification sketch, prioritized test matrix |
| `07-experiment-matrix.md` | runnable experiments in four tiers with explicit pass/fail oracles, including the canary-counter method |
| `08-design-evaluation-real-world-use-cases.md` | design-document section: use-case classes, research questions, design questions and trade-offs, lessons learned, future work in the basic case |
| `09-recommendations.md` | prioritized recommendations P0–P3 with code locations |
| `10-portability.md` | what is AMD-specific and what transfers; CUDA/CUPTI, Level Zero, CPUs, DPUs, FPGAs, NPUs |
| `11-prior-art.md` | Nsight Compute and CUPTI Checkpoint comparison, AMD's own HRR and Kerncap precedents, iteration multiplexing as the real competitor |

## Confluence export

`tools/md_to_confluence.py` converts any of these files to Confluence storage-format XHTML for
pasting into the internal design document:

```bash
python3 tools/md_to_confluence.py 08-design-evaluation-real-world-use-cases.md > out.html
```

`confluence/section-real-world-use-case-evaluation.html` is the pre-generated export of §8.

## Reading order

Read **§0** first for the conceptual framing; everything else is an elaboration of one of its
sections.

For a reviewer with limited time after that: **§4.1** (the finding that reorders the rest — Kokkos on
ROCm 6.x is silently unsound by default), §1.3 (the cost model and its break-even inequality), §2.1
(the decision procedure), §11.2 (iteration multiplexing is the real competitor and its terms are now
known), §8.6–§8.7 (lessons learned and what is still missing even in the friendliest case), and §9's
closing items.

## The four claims that most affect the roadmap

1. **§4.1** — Kokkos enables `hipMallocAsync` by default for HIP < 7.0.0, and ROCm's stream-ordered
   pool is VM-heap backed by default, so most of the HPC portfolio replays on untracked memory and
   returns wrong numbers with no diagnostic. Detection is cheap; this is recommendation R0.
2. **§3.1b** — The HIP runtime calls `hsa_amd_agents_allow_access` over all peers for ordinary
   `hipMalloc`, so on a multi-GPU node peer exposure is the default and the per-agent lock's
   isolation argument rests on application behaviour rather than on anything the tool enforces.
3. **§11.2** — `rocprof-compute --iteration-multiplexing` is cheaper, MPI-safe, and sound under the
   allocators that break replay. Replay's defensible niche is kernels that execute few times, on
   tracked allocations, in a single process, where same-dispatch correlation matters.
4. **§1.3.1 / §11.2** — Kerncap's public measurements of the same operation are 1.3–1.7 GB/s, below
   the 4 GB/s the cost model assumes, so every cost estimate here is optimistic. Measure $B$ first
   (§7.0b).
