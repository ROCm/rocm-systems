---
name: speed-of-light
description: Reads the rocprof-compute System Speed-of-Light panel to show how close an AMD GPU kernel runs to hardware peak for compute throughput, memory bandwidth, and utilization. Use when the user asks how close to peak a kernel is, what percent of peak it reaches, whether a measured percentage is good enough or normal, whether a utilization number should be higher, or asks for an overall efficiency summary. Not for CUDA tools, Windows, per-instruction analysis, or looking up a datasheet specification when there is no profile to read.
---

# Read System Speed-of-Light

Speed-of-Light is the first panel to read after profiling. It puts every
headline metric next to the hardware peak for the current GPU, so one table
says whether a kernel is near any limit at all.

Run `rocprof-compute analyze --help` before choosing flags.
Never use the GUI or TUI.

## 1. Get the panel

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-stats
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b sol
```

`--path` points at the directory containing `profiling_config.yaml`. Always
pass `-k` with one kernel id from `--list-stats`; a Speed-of-Light table
averaged over unrelated kernels means nothing.

If the panel is missing or empty, those counters were not collected. Collect
them with the `kernel-bottleneck` skill, using `-b sol` at profile time for a
cheap single-topic run.

## 2. Read it

The table has `Metric`, `Avg`, `Unit`, `Peak`, and `Percent of Peak` columns.
The peak is already computed for the detected architecture. Compare `Avg`
against the `Peak` in the same row and never calculate a peak yourself.

`Percent of Peak` is `N/A` for metrics that are already percentages, such as
the utilization rows. Read those as the percentage they are.

What the shape of the table tells you:

| Reading | Meaning | Next step |
|---|---|---|
| A memory row near peak | bandwidth-limited | the `memory` skill |
| An MFMA or VALU row near peak | compute-limited | the `roofline` skill |
| Every row far below peak | not limited by throughput | occupancy and launch limits, via the `kernel-bottleneck` skill |
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
