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

## Before you start

If `command -v rocprof-compute` finds the tool and
`rocprof-compute analyze --debug-verify-deps` passes, skip this section.
On ROCm 10.2, use `--verify-deps` instead of `--debug-verify-deps`.

Otherwise, check that ROCm and a GPU are present:

```bash
rocminfo
amd-smi list
```

If either command fails, ROCm is not installed or no GPU is visible. Stop and
point the user to "Install the ROCm Core SDK" in the
[install guide](../../docs/install/core-install.rst). Installing rocprof-compute
alone does not help.

Check the GPU against
[compatible accelerators](../../docs/reference/compatible-accelerators.rst).
The table lists product families, so match the `Marketing Name` that `rocminfo`
prints, and use the `gfx` name to confirm. If the GPU is not supported, stop and
tell the user.

Then follow "Install the analyze mode dependencies" in the
[install guide](../../docs/install/core-install.rst), which installs
rocprof-compute in a virtual environment of its own. Use the nightly index so the
version matches these skills. Activate that environment with
`source <venv>/bin/activate` before you run rocprof-compute, and confirm with
`rocprof-compute analyze --debug-verify-deps`. If the application is a Python
script, note the full path of its Python before you activate, and run the
script with that path.

## 1. Collect

On Navi48 (gfx1201), read the
[Navi48 (gfx1201)](../rocprof-compute-speed-of-light/SKILL.md#navi48-gfx1201)
section of the `rocprof-compute-speed-of-light` skill for the release and the
prerequisites. The roofline plots the `VRAM`, `L0`, and `LDS` bandwidth roofs.
`roofline.csv` has 0 in `L1Bw`, `L2Bw`, and `MALLBw`, because gfx1201 measures no
L1, L2, or Infinity Cache (`MALL`) roof.

The `AI VRAM` point counts GL2C requests destined for memory, which the
Infinity Cache can also serve, so the plot marks it once, at `VRAM`. A point
above the `VRAM` roof can mean the Infinity Cache is absorbing the traffic, so
never call that kernel VRAM-limited from the plot alone. "Limited by" and the
hover text consider `VRAM` only. Under `STABLE_STD` the measured roofs sit well
below the interface peak.

The VALU roofline metrics subtract WMMA issues from all VALU issues and multiply
by the wave size. They are estimates, most reliable for wave32 FP16 16x16x16 WMMA kernels.
Integer and floating-point instructions both count and an FMA is one issue, so they read low for FMA-only loops and high when index
arithmetic dominates. The WMMA metric is compared with the FP16 WMMA peak. WMMA
points need an SDK that provides the `SQ_WMMA_VALU_INSTS` counter, so check the
profiling warnings. See the
[RDNA WGP page](../../docs/conceptual/rdna/wgp.rst) for the roofline details.

Collect only the roofline block:

```bash
rocprof-compute profile --name <workload_name> --roofline -- <application> <args>
```

On ROCm 10.2 and earlier, use `-b roof` instead of `--roofline`.

Collecting it runs microbenchmarks in addition to profiling the application,
which takes extra time. They are skipped when `-b` or `--set` selects blocks
that leave out roofline (block 4), or with the deprecated `--no-roof`, so an
existing profile collected that way has no roofline data.

MI100 (gfx908) does not support the roofline microbenchmarks. Every other
supported architecture does. On MI100 a full profile skips the benchmark and
still succeeds, and selecting roofline fails, so a missing roofline there is not
a mistake in how the profile was run. Never tell an MI100 user to re-profile to get roofline; use
Speed-of-Light and the `rocprof-compute-memory` skill instead. Confirm support
for a new architecture against
[compatible-accelerators.rst](../../docs/reference/compatible-accelerators.rst).

`--roofline-device` selects the GPU for the roofline microbenchmarks. It does not
choose which GPU the application runs on; use `HIP_VISIBLE_DEVICES` for that.

See [standalone roofline](../../docs/how-to/profile/mode.rst) for two more options:

- `--roofline` selects the roofline block; used alone, it collects only roofline data.
- `--roofline-bench-only` runs only the microbenchmarks, without an application.

On ROCm 10.2 and earlier, these options are named `--device`, `--roof-only` and
`--bench-only`.

### Collect only what the question needs

Never run a full profile. It replays the application once per counter pass and
also runs the roofline benchmarks.

- List what this GPU can collect with `rocprof-compute --list-metrics`, and a
  block's metrics with `rocprof-compute --list-metrics <alias>`. On ROCm 10.2
  and earlier, use `--list-blocks <gfx_arch>` and `--list-metrics <gfx_arch>`.
  Pass `-b` only the block aliases, block ids, or metric ids (for example
  `2.1.0`) the question needs. Aliases differ by architecture.
- `--set <set_name>` collects a predefined topic in one pass.
  `rocprof-compute profile --list-sets` lists the sets.
- Add `-k <kernel_name>` when the user names the kernels that matter.
- Unless the question is about roofline, select blocks with `-b` and leave out
  block 4 (`roof`), so the roofline benchmark is skipped.
- `--iteration-multiplexing` spreads counters over repeated dispatches of each
  kernel with the same launch parameters, so it only helps when each profiled
  kernel runs many times with the same launch parameters. Pair it with `-k`.
  See Iteration multiplexing in
  [profile mode](../../docs/how-to/profile/mode.rst).
- Tell the user which analysis panels the narrower profile leaves empty.

## 2. Read the table

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-stats
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> --roofline
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

To compare many kernels, write the
[analysis database](../../docs/how-to/analyze/cli.rst) with
`--output-format db` and query it with SQL.

## 4. When roofline data is missing

| Cause | What to do |
|---|---|
| Profiled with `--no-roof`, or `-b`/`--set` without block 4 | re-profile without `--no-roof`; add `--roofline` to `-b`, or replace `--set` with `-b` blocks plus `--roofline` (no set includes block 4) |
| Architecture has no microbenchmark support | use Speed-of-Light and the `rocprof-compute-memory` skill instead |
| `roofline.csv` absent from the workload directory | the benchmark did not complete; check the profile log |

The concepts behind the ceilings are in the
[performance model](../../docs/conceptual/performance-model.rst).
