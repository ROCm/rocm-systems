# Kernel replay: feasibility, portability and correctness study

Analysis of the experimental kernel replay feature
([#7960](https://github.com/ROCm/rocm-systems/pull/7960) SDK service,
[#10439](https://github.com/ROCm/rocm-systems/pull/10439) `rocprofv3` integration,
[#10193](https://github.com/ROCm/rocm-systems/pull/10193) integration preview) against real
production workloads, other vendors' equivalents, and the correctness obligations the mechanism
depends on.

This directory is analysis, not product documentation. User-facing documentation lives in
`projects/rocprofiler-sdk/source/docs/` (`how-to/using-kernel-replay.rst` and
`conceptual/kernel_replay/`).

## Contents

| File | What it covers |
|---|---|
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

For a reviewer with limited time: **§4.1** (the finding that reorders the rest — Kokkos on ROCm 6.x
is silently unsound by default), §1.3 (the cost model and its break-even inequality), §2.1 (the
decision procedure), §11.2 (iteration multiplexing is the real competitor and its terms are now
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
