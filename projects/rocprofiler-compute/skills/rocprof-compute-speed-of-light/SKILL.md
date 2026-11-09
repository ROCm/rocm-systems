---
name: rocprof-compute-speed-of-light
description: Reads the rocprof-compute System Speed-of-Light panel to show how close an AMD GPU kernel runs to hardware peak for compute throughput, memory bandwidth, and utilization. Use when the user asks how close to peak a kernel is, what percent of peak it reaches, whether a measured percentage is good enough or normal, whether a utilization number should be higher, or asks for an overall efficiency summary. Not for CUDA tools, Windows, per-instruction analysis, or a datasheet or theoretical-peak question when there is no profile to read.
---

# Read System Speed-of-Light

Speed-of-Light is the first panel to read after profiling. It puts every
headline metric next to the hardware peak for the current GPU, so one table
says whether a kernel is near any limit at all.

Run `rocprof-compute analyze --help` before choosing flags.

## Before you start

If `command -v rocprof-compute` finds the tool, skip this section.

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
version matches these skills, and confirm with
`rocprof-compute analyze --verify-deps`. Call that environment's
`rocprof-compute` by its full path, so the application keeps running with its own
Python.

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

### Collect only what the question needs

Never run a full profile. It replays the application once per counter pass and
also runs the roofline benchmarks.

- List what this GPU can collect with `rocprof-compute --list-blocks <gfx_arch>`
  and `rocprof-compute --list-metrics <gfx_arch>`. Pass `-b` only the block
  aliases, block ids, or metric ids (for example `2.1.0`) the question needs.
  Aliases differ by architecture.
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

## 2. Read it

The table has `Metric`, `Avg`, `Unit`, `Peak`, and `Percent of Peak` columns.
The peak is already computed for the detected architecture. Compare `Avg`
against the `Peak` in the same row and never calculate a peak yourself.

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
