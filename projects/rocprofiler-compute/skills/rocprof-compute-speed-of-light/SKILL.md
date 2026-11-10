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

This section is the one Navi48 reference for the rocprof-compute skills; the
memory, kernel-bottleneck, and roofline skills point here.

Linux GPU metrics, the Memory Chart, and roofline start with ROCm Compute
Profiler 3.11 (ROCm 10.3); ROCm 10.2 and 10.1 shipped without gfx1201 support.
PC sampling is outside Navi48 support in this release. Follow the `STABLE_STD`
prerequisite and counter limits in the
[RDNA System Speed-of-Light](../../docs/conceptual/rdna/system-speed-of-light.rst),
and compare runs only within one performance state. List the Navi48 aliases
with `rocprof-compute --list-metrics-gfx-id gfx1201`: the cache blocks are
`gl0`, `gl1`, and `gl2`, and LDS metrics belong to the WGP block, `wgp`.

`VALU Issue Rate` and `FP64 VALU Issue Rate` use `GInstr/s`, with no FLOP peak
or percentage of a floating-point peak. Do not turn those instruction rates
into FLOPs or use them to judge distance from a FLOP ceiling.
`WMMA FLOPs (F16->F32 16x16x16)` assumes that precision and shape; the raw
counter counts issues per vec32 and cannot distinguish FP16, BF16, FP8, or BF8.
Present that comparison as approximate. Do not subtract WMMA issues to infer
FP32 FLOPs from these rows or combine WMMA and VALU into an independent total.
A fixed wave32 FP16-to-FP32 16x16x16 diagnostic supports WMMA inclusion in
VALU for that instruction; counted population and other variants remain
unqualified. WGP `SQ_INST_CYCLES_VALU` is a raw SIMD execution-cycle count with
no peak; it is not elapsed time, FLOPs, or utilization.

`Wave Dependency Wait` and `Wave Issue Wait` are percentages of wave cycles,
from `SQ_WAIT_ANY` and `SQ_WAIT_INST_ANY` over `SQG_WAVE_CYCLES`. The
`wave_wait` shortcut collects those two rows. The WGP Instruction Issue Stalls
table, including `Wait for Any`, stays unavailable. `Wavefront Occupancy` and
WGP `Wave Life` remain provisional while living-wave counter units, reduction,
and wave-count normalization are validated. Treat these values as unknown when
inferring occupancy limits or quantified scheduler stalls. Restoration is
tracked under AIPROFCOMP-907.

For memory, read `memchart` first. GL1 is a write-combining stage: read its
utilization and stalls, and do not invent a hit rate when the report prints
`N/A`. `GL2 Cache BW` reports a percentage of the L2 interface peak; the
`GL2 Cache Bandwidth` table counts size-weighted requests at the GL2 output
FIFOs. `GL2-Fabric Read BW`, `GL2-Fabric Write BW`, and the Memory Chart
`DRAM Read Bandwidth` and `DRAM Write Bandwidth` include traffic served or
absorbed by the Infinity Cache, so they do not measure physical GDDR6 bytes
and have no GDDR6 peak. `Infinity Cache Hit Rate` and GDDR6 bandwidth are
`N/A`; treat them as unknown, not zero. The LDS byte-access rate normalization
is provisional.

On the roofline, the device-memory roof is `DRAM` and the `MALL` roof is the
64 MB Infinity Cache. The `AI DRAM` point counts GL2C requests destined for
memory, which the Infinity Cache can also serve, so the plot marks it once, at
`DRAM`. Compare it with both roofs yourself, because "Limited by" and the hover
text never name `MALL`. A point above `DRAM` and below `MALL` means the
Infinity Cache is absorbing the traffic; never call that kernel DRAM-limited.
Under `STABLE_STD` the measured roofs sit well below the interface peak. The
VALU roofline rows subtract WMMA issues from all VALU issues and multiply by
the wave size, validated only for one wave32 FP16 16x16x16 case. They are
estimates: integer and floating-point instructions both count and an FMA is
one issue, so they read low for FMA-only loops and high when index arithmetic
dominates. The WMMA row is compared with the FP16 WMMA peak. WMMA points need
an SDK that supports SQ event 233; check the profiling warnings.

If a row has no usable peak, report its measured rate and compare like
workloads in the same performance state instead of inferring a percent of peak
or a throughput bottleneck.

### Collect only what the question needs

Never run a full profile. It replays the application once per counter pass and
also runs roofline benchmarks on architectures that support them.

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
- `--iteration-multiplexing kernel` spreads counters over repeated dispatches of
  each kernel, so it only helps when every profiled kernel is dispatched many
  times. Pair it with `-k`. See Iteration multiplexing in
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
for client APUs, and
[the RDNA4 equivalent](../../docs/conceptual/rdna4/system-speed-of-light.rst)
for RDNA4 Radeon GPUs. The whole metric hierarchy is in the
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
