---
name: rocprof-compute-roofline
description: Collects and reads rocprof-compute roofline data to place an AMD GPU kernel against the hardware's peak compute and peak bandwidth ceilings. Use when the user asks whether a kernel is compute-bound or memory-bound, asks about arithmetic intensity or FLOPs per byte, asks how much performance is left on the table, or asks for a roofline. Not for CUDA tools, Windows, or per-instruction analysis.
---

# Roofline analysis

A roofline places a kernel's arithmetic intensity against two measured
ceilings, peak compute and peak bandwidth, and answers one question: which of
the two is the limit.

Run `rocprof-compute profile --help` and `rocprof-compute analyze --help`
before choosing flags. Do not open the generated HTML chart. The analyze table
carries the same numbers in readable form.

## 1. Collect

Roofline is collected by default, so an ordinary profile already has it:

```bash
rocprof-compute profile --name <workload_name> -- <application> <args>
```

Collecting it runs microbenchmarks in addition to profiling the application,
which takes extra time. `--no-roof` skips them when the user only wants
counters.

MI100 (gfx908) does not support the roofline microbenchmarks. Every other
supported architecture does. On MI100 the benchmark is skipped and the profile
still succeeds, so a missing roofline there is not a mistake in how the profile
was run. Never tell an MI100 user to re-profile without `--no-roof`; use
Speed-of-Light and the `rocprof-compute-memory` skill instead. Confirm support
for a new architecture against
[compatible-accelerators.rst](../../docs/reference/compatible-accelerators.rst).

`--device` selects the GPU for the roofline microbenchmarks. It does not
choose which GPU the application runs on; use `HIP_VISIBLE_DEVICES` for that.

See [standalone roofline](../../docs/how-to/profile/mode.rst) for `--roof-only`
and `--bench-only`, which run the microbenchmarks without profiling an
application.

## 2. Read the table

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-stats
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b roof
```

Always pass one kernel id. The table gives the kernel's arithmetic intensity in
FLOPs per byte alongside the empirical peak FLOPs and peak bandwidth measured
on this machine. Compare against those printed numbers; never compute a peak
yourself.

Report two things first, before any tuning advice:

1. **Which ceiling limits the kernel**, memory or compute.
2. **How far the kernel sits below that ceiling**, as a percentage of it.

Read the verdict from the table as:

- Low arithmetic intensity, close to the bandwidth ceiling: memory-bound. Work
  on traffic, reuse, tiling, and locality. Continue with the
  `rocprof-compute-memory` skill.
- High arithmetic intensity, close to the compute ceiling: compute-bound. Work
  on instruction mix, matrix instruction use, and divergence.
- Well below both ceilings: neither is the limit. The kernel is held back by
  occupancy, launch configuration, or stalls. Go back to the
  `rocprof-compute-kernel-bottleneck` skill.

A kernel sitting below both ceilings is the common case and the most
misread one. Never report it as compute-bound merely because its arithmetic
intensity is high.

The distance to the ceiling is what makes the verdict actionable. "Memory-bound
at 45% of peak bandwidth" tells the user there is headroom; "memory-bound at
94%" tells them to reduce traffic instead of chasing it.

## 3. Pick the right precision and memory level

The table lists a peak for every precision and an arithmetic intensity for
every memory level. Compare against the peak row for the precision the kernel
actually issues, and the intensity row for the memory level in question.

A mixed-precision or matrix kernel compared against the FP32 peak will look far
worse than it is. Read the precision from the instruction mix with `-b cu_ins`
rather than assuming it. Ask the user which precision the kernel is meant to
use only when the mix is genuinely mixed and no single type dominates; they
know the intent, the counters know what ran.

HBM answers "am I limited by main memory", while L2 and vL1D show whether
caches are absorbing the traffic.

`--view table` prints one table for all kernels together instead of one per
kernel. To compare many kernels, write the
[analysis database](../../docs/how-to/analyze/cli.rst) with
`--output-format db` and query it with SQL.

## 4. When roofline data is missing

| Cause | What to do |
|---|---|
| Profiled with `--no-roof` | re-profile without it |
| Architecture has no microbenchmark support | use Speed-of-Light and the `rocprof-compute-memory` skill instead |
| `roofline.csv` absent from the workload directory | the benchmark did not complete; check the profile log |
| `roofline.csv` fails validation (analyze warns with the error) | the benchmark output is incomplete or corrupt; re-profile to regenerate it |

The concepts behind the ceilings are in the
[performance model](../../docs/conceptual/performance-model.rst).
