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

Check the architecture before collecting. Navi48 (gfx1201) GPU metrics and the
Memory Chart start with ROCm Compute Profiler 3.11 (ROCm 10.3), but roofline and
benchmarking are outside its support scope in this release. ROCm 10.2 and 10.1
shipped without gfx1201 support. For Navi48, use the available Speed-of-Light
and memory metrics, following the `STABLE_STD` prerequisite and interpretation
limits in the [Navi48 guide](../../docs/conceptual/rdna/navi48.rst). Explain that
these metrics do not supply a measured roofline ceiling; do not retry roofline
collection or fabricate arithmetic intensity and peak data.

Collect only the roofline block:

```bash
rocprof-compute profile --name <workload_name> -b roof -- <application> <args>
```

Collecting it runs microbenchmarks in addition to profiling the application,
which takes extra time. An existing profile already has roofline unless it was
collected with `--no-roof`, or with a `-b` list that left out `roof`.

MI100 (gfx908) also does not support the roofline microbenchmarks. On MI100 the
benchmark is skipped and the profile
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

### Collect only what the question needs

Never run a full profile. It replays the application once per counter pass and
also runs roofline benchmarks on architectures that support them.

- List block aliases with `rocprof-compute --list-metrics-gfx-id <gfx_arch>`
  and metrics with
  `rocprof-compute --list-metrics-gfx-id <gfx_arch> --list-metrics all`.
  Pass `-b` only the block aliases, block ids, or metric ids (for example
  `2.1.0`) the question needs. Aliases differ by architecture.
- `--set <set_name>` collects a predefined topic in one pass.
  `rocprof-compute profile --list-sets` lists the sets.
- Add `-k <kernel_name>` when the user names the kernels that matter.
- Add `--no-roof` unless the question is about roofline. A `-b` list without
  block 4 (`roof`) already skips it.
- `--iteration-multiplexing kernel` spreads counters over repeated dispatches of
  each kernel, so it only helps when every profiled kernel is dispatched many
  times. Pair it with `-k`. See Iteration multiplexing in
  [profile mode](../../docs/how-to/profile/mode.rst).
- Tell the user which analysis panels the narrower profile leaves empty.

In ROCm 10.2 and 10.1, use `--list-blocks <gfx_arch>` to list blocks and
`--list-metrics <gfx_arch>` to list metrics. The architecture selector changed
in ROCm 10.3.

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

The concepts behind the ceilings are in the
[performance model](../../docs/conceptual/performance-model.rst).
