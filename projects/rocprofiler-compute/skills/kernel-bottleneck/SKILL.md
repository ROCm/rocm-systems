---
name: kernel-bottleneck
description: Profiles an AMD GPU application with rocprof-compute and finds which kernel is slow and why, using performance metrics and PC sampling. Collects counters and selects kernels and dispatches, then analyzes GPU memory throughput, compute efficiency, wavefront occupancy and wavefront limits, scheduler stalls, etc. Use when the user asks to profile, benchmark, or speed up a HIP/ROCm kernel or application, or asks where a kernel's bottleneck is. Not for CUDA tools, Windows, system-wide CPU/MPI tracing, or writing kernel source.
---

# Profile an AMD GPU application and find the kernel bottleneck

Start here for any "profile this" or "why is this kernel slow" request. This
skill collects counters, narrows a workload down to one kernel, and identifies
what limits it. Hand off to a focused skill once the limit is clear.

Run `rocprof-compute profile --help` and `rocprof-compute analyze --help`
before choosing flags. Options change between releases; do not guess one.

Never use the GUI or TUI. This is a command-line workflow.

## Two sources of performance data

rocprof-compute collects from two independent sources, and they answer
different questions. Choose by what the user is asking, not by difficulty.

| Source | Answers | Skill |
|---|---|---|
| Perfmon counters | How the architecture behaves: bandwidth, cache, occupancy, pipeline utilization | this skill, then `speed-of-light`, `memory`, `roofline` |
| PC sampling | How the source code behaves: which instruction or line is hot and why it stalls | `pc-sampling` |

Counters are the default starting point because they cover the whole kernel
cheaply. Go straight to `pc-sampling` when the user asks about instructions,
source lines, or stall reasons, and use both when an architectural limit needs
to be traced back to the code that causes it.

## 1. Check the environment

In this order, because each command depends on the one before it:

```bash
amd-smi static
amd-smi list
rocminfo
rocprof-compute --specs
```

`amd-smi` and `rocminfo` report the GPU and ROCm install.
`rocprof-compute --specs` reports what the profiler itself detected, including
`ROCm Version`, `GPU Model`, and `GPU Arch`. It is built on the first two, so
it cannot succeed when they fail.

Supported accelerators are listed in
[compatible-accelerators.rst](../../docs/reference/compatible-accelerators.rst).
Not every feature works on every architecture. Check there before promising a
result, and let the tool's own error message stand when a feature is
unavailable.

## 2. Collect what you need before profiling

Ask once:

1. Launch command and arguments
2. A short workload name, used as `--name`
3. Which kernels matter, if not all

Profiling replays the application once per counter pass, so the command must be
safe to run repeatedly and short enough to run several times. Ask for a reduced
iteration count or problem size rather than profiling a long run. Do not invent
a timeout; profile mode has no such option.

## 3. Profile

```bash
rocprof-compute profile --name <workload_name> -- <application> <args>
```

Output lands in `./workloads/<workload_name>/<gpu_model>/`, or `<rank>/` under
MPI. Change it with `--output-directory`, which replaces `--name` for building
the path. Re-profiling into a non-empty directory needs `--overwrite`; prefer a
fresh directory.

To restrict which GPU the application uses, set `HIP_VISIBLE_DEVICES`. Under
Slurm, derive it from `ROCR_VISIBLE_DEVICES`, not from `SLURM_STEP_GPUS`; the
indices differ.

See [profile mode](../../docs/how-to/profile/mode.rst) for output layout,
filtering, and the full option list. To profile a process that is already
running, see
[live attach and detach](../../docs/how-to/live_attach_detach.rst).

### Collect less when a full replay is too expensive

A single-pass, topic-focused collection:

```bash
rocprof-compute profile --list-sets
rocprof-compute profile --name <name> --set <set_name> -- <application>
```

Or collect only named blocks:

```bash
rocprof-compute profile --name <name> -b sol wavefront -- <application>
```

Iteration multiplexing spreads counters across kernel launches so a single pass
collects more:

```bash
rocprof-compute profile --name <name> --iteration-multiplexing kernel -- <application>
```

Whenever you collect a subset, tell the user which analysis panels will be
empty and that answering those questions needs another profile.

## 4. Find the hot kernel

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-stats
```

`--path` must point at the directory holding `profiling_config.yaml`. Analyze
does not search for it in nearby directories. If the path you have does not
contain that file, look one level down for the GPU model or rank directory.

`--list-stats` prints `Detected Kernels (sorted descending by duration)` and a
`Dispatch list`. The kernel index is what `-k` takes. Dispatch ids are 1-based
and go to `-d`.

Work on one kernel at a time:

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k 0
```

`-k` at profile time is different: it filters by kernel name and is used as a
regex, so `-k gemm` matches every kernel whose name contains `gemm`.

## 5. Narrow to a reason

Get the overview, then open exactly one detailed block:

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k 0 -b sol
```

Read the `Percent of Peak` column. The report prints the peak for each metric,
so compare against that rather than against a remembered threshold, and never
compute a peak yourself.

Resolve block names with aliases, which are stable across releases, instead of
numeric ids, which are architecture-specific:

```bash
rocprof-compute --list-blocks <arch>
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-available-metrics
```

Where to go next:

| Symptom in Speed-of-Light | Next step |
|---|---|
| Memory throughput near peak, low compute | the `memory` skill |
| Compute near peak, or unclear compute vs memory | the `roofline` skill |
| Everything far below peak | `-b wavefront` for occupancy, then `-b spi` for launch limits, both below |
| Question is about instructions, source lines, or stall reasons | the `pc-sampling` skill |
| Kernels come from PyTorch | the `torch-trace` skill |

### Occupancy and wavefront limits

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k 0 -b wavefront
```

Compare active waves per CU against the theoretical occupancy in the same
table. When theoretical occupancy is itself low, the limit is VGPR, SGPR, or
LDS allocation. Scratch traffic points at register spilling. Low occupancy is
not a defect when the kernel already saturates its limiting resource. Worked
examples are in
[occupancy-limiters-example.rst](../../docs/tutorial/includes/occupancy-limiters-example.rst).

### Scheduler and pipeline limits

```bash
# Workgroup launch rate and resource allocation limits
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k 0 -b spi

# Pipeline utilization and instruction mix
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k 0 -b cu_pipe
```

There is no single stall panel. Stall counters sit in the block that owns the
hardware: LDS stalls in `lds`, cache stalls in `vl1d` and `l2`, fabric stalls
in `l2_per_channel`. Open the block the Speed-of-Light result points at, and
read what each counter means in the
[performance model](../../docs/conceptual/performance-model.rst). Ground every
recommendation in a number you read from the report.

## 6. Normalization

`-n` / `--normal-unit` applies to every metric in the report at once:
`per_kernel` (default), `per_wave`, `per_cycle`, `per_second`. Keep the default
for a mixed report. Change it only when the user asks about one metric family
and the unit still makes sense for it. Bandwidth per cycle does not.

## 7. Compare two runs

Pass `--path` twice to put two workloads side by side. Use this for:

1. Measuring a software optimization against its baseline on the same GPU.
2. Checking that the same code performs consistently across nodes with the
   same GPU architecture.
3. Comparing the same code across different GPU architectures.

```bash
rocprof-compute analyze \
    --path ./workloads/baseline/<gpu_model> \
    --path ./workloads/optimized/<gpu_model>
```

Cases 1 and 2 expect the numbers to match except where the change intended
otherwise, so an unexplained difference is the finding. Case 3 compares
different hardware, so compare percent of peak rather than absolute values.

Never present a comparison across different GPUs, partition modes, or clock
settings without stating that those differ.

## 8. Save a report

Terminal output is the default. Only choose a format when the user asks to save
or export. `--output-format` takes `txt`, `csv`, or `db`, each of which
disables terminal output, and `--output-name` sets the file name. See
[analysis output format](../../docs/how-to/analyze/cli.rst).

## 9. When something looks wrong

| Symptom | Likely cause |
|---|---|
| No workload data | `--path` is not the directory with `profiling_config.yaml` |
| Empty or missing panels | those counters were not collected; re-profile |
| All metrics zero | kernel too short, or no dispatches were captured |
| No kernel ids | the application launched no GPU kernels |
| Roofline missing | not collected, or unsupported on this architecture |

Stop and report permission, firmware, and ROCm version failures instead of
working around them. More in the [FAQ](../../docs/reference/faq.rst).
