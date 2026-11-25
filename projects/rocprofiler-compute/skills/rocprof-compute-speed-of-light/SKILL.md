---
name: rocprof-compute-speed-of-light
description: Reads the rocprof-compute System Speed-of-Light panel to show how close an AMD GPU kernel runs to hardware peak for compute throughput, memory bandwidth, and utilization. Use when the user asks how close to peak a kernel is, what percent of peak it reaches, whether a measured percentage is good enough or normal, whether a utilization number should be higher, or asks for an overall efficiency summary. Not for CUDA tools, Windows, per-instruction analysis, or a datasheet or theoretical-peak question when there is no profile to read.
---

# Read System Speed-of-Light

Speed-of-Light is the first panel to read after profiling. It puts headline
metrics next to hardware peaks where the architecture provides a meaningful
peak, so one table shows which measured limits a kernel approaches.

Run `rocprof-compute analyze --help` before choosing flags.

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

## 1. Get the panel

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-stats
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> --speed-of-light
```

On ROCm 10.2 and earlier, use `-b sol` instead of `--speed-of-light`.

`--path` points at the directory containing `profiling_config.yaml`. Always
pass `-k` with one kernel id from `--list-stats`; a Speed-of-Light table
averaged over unrelated kernels means nothing.

If there is no workload yet, or the panel is missing or empty, collect
Speed-of-Light alone:

```bash
rocprof-compute profile --name <workload_name> --speed-of-light -- <application> <args>
```

### Navi48 (gfx1201)

This section covers Navi48 for Speed-of-Light. The memory skill has its own Navi48
section, and the roofline skill covers Navi48 in its Collect step. Navi48 boards
are gfx1201, for example the Radeon RX 9070, RX 9070 XT, and Radeon AI PRO R9700.

Linux GPU metrics, the Memory Chart, and roofline start with ROCm Compute
Profiler 3.11 (ROCm 10.3); ROCm 10.2 and 10.1 shipped without gfx1201 support.
gfx1201 support is experimental: add `--experimental` to every `profile` and
`analyze` command, for example
`rocprof-compute --experimental profile -n <name> -- <app>`. PC sampling is not
supported on gfx1201. Follow the `STABLE_STD` prerequisite and counter limits in
the
[RDNA System Speed-of-Light](../../docs/conceptual/rdna/system-speed-of-light.rst),
and compare runs only within one performance state. List the Navi48 aliases
with
`rocprof-compute --experimental --debug-list-metrics-gfx-id gfx1201 --list-metrics`:
the cache blocks are `gl0`, `gl1`, and `gl2`, and LDS metrics belong to the WGP
block, `wgp`.

`VALU Issue Rate` and `FP64 VALU Issue Rate` use `GInstructions/s`, with no FLOP
peak or percentage of a floating-point peak. Do not turn those instruction rates
into FLOPs or use them to judge distance from a FLOP ceiling.
`WMMA FLOPs (F16->F32 16x16x16)` assumes that precision and shape. The counter
counts issues per vec32 and cannot distinguish FP16, BF16, FP8, or BF8, so
present that comparison as approximate. Do not subtract WMMA issues to infer
FP32 FLOPs from these metrics, and do not combine WMMA and VALU into an
independent total. WGP `SQ_INST_CYCLES_VALU` counts the cycles needed to execute
VALU operations. It has no peak and is not elapsed time, FLOPs, or utilization.

The gfx1201 Speed-of-Light panel has no `GL0 Cache BW`, `Instruction Cache BW`,
or `GL1 Cache Hit Rate` metric, and there is no wave-wait metric or WGP stall
table. Do not infer them from other metrics.

`Wavefront Occupancy`, the Memory Chart occupancy, and WGP `Wave Life` use
`SQ_WAVE_CYCLES`, which follows the live waves, starting with ROCm Compute
Profiler 3.12 (ROCm 10.4). In ROCm Compute Profiler 3.11 (ROCm 10.3) they used
`SQG_WAVE_CYCLES`, which did not track the true wave count on an RX 9070, so do
not trust gfx1201 occupancy from that release. The occupancy peak and the
wave-count normalization are still being validated, so treat these values as
estimates when inferring occupancy limits or quantified scheduler stalls.

If a metric has no usable peak, report its measured rate and compare like
workloads in the same performance state instead of inferring a percent of peak
or a throughput bottleneck.

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

## 2. Read it

The table has `Metric`, `Avg`, `Unit`, `Peak`, and `Percent of Peak` columns.
Where a peak is available, it is computed for the detected architecture.
Compare `Avg` against the `Peak` in the same row and never calculate a peak
yourself. A missing peak does not establish that the metric is far below a limit.

`Percent of Peak` is `N/A` for metrics that are already percentages, such as
the utilization rows. Read those as the percentage they are.

What the shape of the table tells you:

| Reading | Meaning | Next step |
|---|---|---|
| A memory row near peak | bandwidth-limited | the `rocprof-compute-memory` skill |
| An MFMA or VALU row near peak | compute-limited | the `rocprof-compute-roofline` skill |
| Every row far below peak | not limited by throughput | occupancy and launch limits, via the `rocprof-compute-kernel-bottleneck` skill |
| MFMA at zero on a GEMM | matrix pipeline unused | check the kernel's instruction mix with `-b cu_ins` |

A row at zero is a real reading only if the kernel was expected to use that
unit. MFMA at zero is normal for a kernel with no matrix instructions.

Do not apply remembered thresholds. "Above 70% is saturated" is not a rule the
tool encodes, and the correct comparison is always against the printed peak,
the neighbouring kernels, and a known-good baseline for the same workload.

## 3. Understand the metrics

Each metric's definition and the counters behind it are in
[system-speed-of-light.rst](../../docs/conceptual/cdna/system-speed-of-light.rst)
for CDNA accelerators and
[the RDNA equivalent](../../docs/conceptual/rdna/system-speed-of-light.rst)
for Ryzen APUs and Radeon GPUs. Select the architecture tab on each table. The whole metric hierarchy is in the
[performance model](../../docs/conceptual/performance-model.rst).

## 4. Compute Speed-of-Light

The compute pipeline has its own Speed-of-Light table with the same shape:

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b cu_pipe
```

Read it when the system-level table shows compute near peak and you need to
know which pipeline is responsible.

## 5. Normalization

`-n` changes the unit for the entire report, not one metric. Leave it at
`per_kernel` for Speed-of-Light. The peaks are expressed to match the default,
and switching units makes the peak comparison harder to read, not easier.
