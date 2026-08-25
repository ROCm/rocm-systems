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
| `05-range-replay-design.md` | four range-replay designs, why graph launches are the best replay unit, recommended sequence |
| `06-formal-correctness.md` | state model, six proof obligations, liveness hazards, TLA+ specification sketch, prioritized test matrix |
| `07-experiment-matrix.md` | runnable experiments in four tiers with explicit pass/fail oracles, including the canary-counter method |
| `08-design-evaluation-real-world-use-cases.md` | design-document section: use-case classes, research questions, design questions and trade-offs, lessons learned, future work in the basic case |
| `09-recommendations.md` | prioritized recommendations P0–P3 with code locations |

## Confluence export

`tools/md_to_confluence.py` converts any of these files to Confluence storage-format XHTML for
pasting into the internal design document:

```bash
python3 tools/md_to_confluence.py 08-design-evaluation-real-world-use-cases.md > out.html
```

`confluence/section-real-world-use-case-evaluation.html` is the pre-generated export of §8.

## Reading order

For a reviewer with limited time: §1.3 (the cost model and its break-even inequality), §2.1 (the
decision procedure), §8.6–§8.7 (lessons learned and what is still missing even in the friendliest
case), and §9's closing three items.
