---
name: rocprof-compute-speed-of-light
description: Reads the rocprof-compute System Speed-of-Light panel to show how close an AMD GPU kernel runs to hardware peak for compute throughput, memory bandwidth, and utilization. Use when the user asks how close to peak a kernel is, what percent of peak it reaches, whether a measured percentage is good enough or normal, whether a utilization number should be higher, or asks for an overall efficiency summary. Not for CUDA tools, Windows, per-instruction analysis, or a datasheet or theoretical-peak question when there is no profile to read.
---

# Read System Speed-of-Light

Speed-of-Light is the first panel to read after profiling. It puts headline
metrics next to hardware peaks where the architecture provides a meaningful
peak, so one table shows which measured limits a kernel approaches.

Run `rocprof-compute analyze --help` before choosing flags.

## 1. Get the panel

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-stats
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b sol
```

`--path` points at the directory containing `profiling_config.yaml`. Always
pass `-k` with one kernel id from `--list-stats`; a Speed-of-Light table
averaged over unrelated kernels means nothing.

If there is no workload yet, or the panel is missing or empty, collect
Speed-of-Light alone:

```bash
rocprof-compute profile --name <workload_name> -b sol -- <application> <args>
```

### Navi48 (gfx1201)

Linux GPU metrics start with ROCm Compute Profiler 3.11 (ROCm 10.3); ROCm 10.2
and 10.1 shipped without gfx1201 support. Follow the `STABLE_STD` prerequisite
and counter limits in the [Navi48 guide](../../docs/conceptual/rdna/navi48.rst).

`VALU Issue Rate` and `FP64 VALU Issue Rate` use `GInstr/s`, with no FLOP peak
or percentage of a floating-point peak. Do not turn those instruction rates
into FLOPs. `WMMA FLOPs (F16->F32 16x16x16)` assumes that precision and shape;
the raw counter counts issues per vec32 and cannot distinguish FP16, BF16, or
FP8. Its FLOP conversion and peak need validation for the kernel's precision
and wave size. Present that comparison as approximate. Do not subtract WMMA
issues to infer FP32 FLOPs or combine WMMA and VALU into an independent total.
A fixed wave32 FP16-to-FP32 16x16x16 diagnostic supports WMMA inclusion in
VALU for that instruction; counted population and other variants remain
unqualified.

`Wave Dependency Wait` and `Wave Issue Wait` are percentages of wave cycles,
from `SQ_WAIT_ANY` and `SQ_WAIT_INST_ANY` over `SQG_WAVE_CYCLES`. The
`wave_wait` shortcut collects those two rows. The WGP Instruction Issue Stalls
table, including `Wait for Any`, stays unavailable. `Wavefront Occupancy` and
WGP `Wave Life` remain provisional while living-wave counter units, reduction,
and wave-count normalization are validated. Treat these values as unknown when
inferring occupancy limits or quantified scheduler stalls. Restoration is
tracked under AIPROFCOMP-907.

`GL2 Cache BW` reports a percentage of the L2 interface peak. `GL2-Fabric Read BW`
and `GL2-Fabric Write BW` include traffic served or absorbed by Infinity Cache.
They have no GDDR6 percentage of peak. Use the Memory Chart
and available cache metrics for memory questions; roofline benchmarks are
outside Navi48 support in this release. If the relevant row has no usable peak,
report its measured rate and compare like workloads in the same performance
state instead of inferring a percent of peak or a throughput bottleneck.

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
for client APUs. The whole metric hierarchy is in the
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
