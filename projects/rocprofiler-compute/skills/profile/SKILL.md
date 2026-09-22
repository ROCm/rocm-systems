---
name: rocprofiler-compute-profile
description: Profiles an AMD GPU workload with rocprofiler-compute (rocprof-compute profile). Use when the user asks to profile, benchmark, measure, or collect hardware performance counters for a HIP/ROCm kernel or application on AMD Instinct GPUs or supported client APUs — including ROCTX-annotated apps, roofline collection, and experimental PC sampling.
---

# Skill: Profile an AMD GPU Workload with rocprofiler-compute

Follow this skill when a user asks to profile, benchmark, or collect GPU
hardware performance counters for an AMD GPU workload. After a successful
profile, hand off to `skills/analyze/SKILL.md`.

## 1. Understand rocprofiler-compute

**rocprofiler-compute** (formerly Omniperf) is AMD's system-level GPU
performance analysis tool. It collects hardware performance counters through
multiple profiling passes and writes a workload directory that analyze mode
reads.

| Concept | Detail |
|---|---|
| CLI | `rocprof-compute` |
| Profile backends | **Perfmon counters** (multi-pass, architecture-level); **PC sampling** (experimental, instruction-level stall samples) |
| Multi-pass | Replays the workload, collecting different counter groups each pass |
| Output format | Profile mode **always uses rocpd**. There is no `--format-rocprof-output`. Analyze consumes the workload directory (converted CSVs plus optional retained `.db`) |
| Supported HW | Instinct MI100, MI200 (MI210/MI250/MI250X), MI300A/MI300X/MI325X, MI350/MI355X, MI455X; client APUs (Strix, Strix Halo, Krackan, Gorgon) |
| ROCm | rocprofiler-compute v3.x targets current ROCm; native counter tool requires ROCm 7.x+ |

Do not invent flags. If unsure, run `rocprof-compute profile --help`.

## 2. Collect inputs before profiling

Ask **once** before running anything:

1. Application binary / launch command
2. Application arguments (pass-through after `--`)
3. Workload name (short identifier, no spaces; used as `-n` / `--name`)
4. Target GPU (default: whatever the process sees; list with `amd-smi`)
5. Which kernel(s) to profile (default: all; or `-k` name filter)
6. Roofline collection? (included by default; `--no-roof` to skip)
7. Output directory (default: `./workloads/<name>/<gpu-model>/`)

## 3. Verify environment before profiling

Do not proceed if these fail.

```bash
# 1. GPU is visible
amd-smi

# 2. rocprofiler-compute is on PATH
rocprof-compute --version

# 3. ROCm version
cat /opt/rocm/.info/version

# 4. Target application runs without the profiler
<your_app_command>
```

## 4. Run the profiler

Profile mode requires `--name` (unless `--output-directory` fully specifies
the destination) and the application after `--`.

Default output directory is `./workloads`. The workload is written under
`./workloads/<name>/%gpumodel%` (or `%rank%` with MPI).

### 4a. Basic profiling (all kernels, roofline included)

```bash
rocprof-compute profile \
    --name <workload_name> \
    -- <your_app_command_and_args>
```

Example:

```bash
rocprof-compute profile \
    --name my_gemm_bench \
    -- ./hip_gemm --m 4096 --n 4096 --k 4096
```

Override the destination with `-d` / `--output-directory` (this ignores
`--name` for path construction). Re-profile into a non-empty directory only
with `--overwrite`.

```bash
rocprof-compute profile \
    --name my_gemm_bench \
    -d ./workloads \
    -- ./hip_gemm --m 4096 --n 4096 --k 4096
```

### 4b. Skip roofline (faster)

Roofline is collected by default. Use `--no-roof` when you only need
counters:

```bash
rocprof-compute profile \
    --name <workload_name> \
    --no-roof \
    -- <your_app_command_and_args>
```

### 4c. Target a specific GPU

`--device` is **not** a general "profile this GPU" switch. It selects the
GPU used for **standalone roofline microbenchmarks** (`--roof-only` /
`--bench-only`).

To restrict which GPU the application (and therefore profiling) sees:

```bash
HIP_VISIBLE_DEVICES=<GPU_INDEX> rocprof-compute profile \
    --name <workload_name> \
    -- <your_app_command_and_args>
```

Find `<GPU_INDEX>` from `amd-smi`.

### 4d. Profile only specific kernels

`-k` / `--kernel` filters by kernel name (space-separated names; not a
regex). Example from the CLI:

```bash
rocprof-compute profile \
    --name gemm_only \
    -k vecCopy \
    -- ./my_app
```

Limit iterations with `--kernel-iteration-range` (1-based, e.g. `1` or
`3:5`).

### 4e. Single-pass topic sets (not `--single-pass`)

There is no `--single-pass` flag. For a faster, topic-focused collection,
use metric sets:

```bash
rocprof-compute profile --list-sets

rocprof-compute profile \
    --name <workload_name> \
    --set <set_name> \
    -- <your_app_command_and_args>
```

`--set` cannot be combined with `--block`, `--roof-only`, or `--bench-only`.
`--block` / `-b` filters by metric id, block id, or alias (see
`--list-available-metrics` and `--list-blocks`).

### 4f. PC sampling (experimental)

PC sampling is experimental. Pass `--experimental` **and** `--pc-sampling`.
Stochastic sampling requires MI300+ (gfx940+). Older GPUs: `host_trap`.

```bash
# Stochastic (MI300+). Interval is cycles, power of two. Default 1048576.
rocprof-compute profile \
    --experimental \
    --name <workload_name> \
    --pc-sampling \
    --pc-sampling-method stochastic \
    --pc-sampling-interval 1048576 \
    -- <your_app_command_and_args>

# host_trap. Interval is microseconds. Default 512.
rocprof-compute profile \
    --experimental \
    --name <workload_name> \
    --pc-sampling \
    --pc-sampling-method host_trap \
    --pc-sampling-interval 512 \
    -- <your_app_command_and_args>
```

Query valid intervals with `rocprofv3-avail info --pc-sampling` when
available. Analyze the samples with the analyze skill
(`--pc-sampling-sorting-type`, `--pc-sampling-rows`).

### 4g. ROCTX annotation (range-level attribution)

ROCTX markers/ranges are captured automatically when the application links
`libroctx64.so`. No extra profile flag is required.

**Annotate HIP/C++:**

```cpp
#include <roctx.h>

roctxMark("checkpoint_start");

roctxRangePush("gemm_compute_phase");
hipLaunchKernelGGL(my_gemm_kernel, grid, block, 0, stream, ...);
hipDeviceSynchronize();
roctxRangePop();
```

**Profile as usual:**

```bash
rocprof-compute profile \
    --name <workload_name> \
    -- <your_annotated_app>
```

Use ROCTX when the same kernel runs in different logical phases (for
example forward vs backward). Analyze by listing dispatches with
`--list-stats` and filtering `-d` / `--dispatch` (1-based) for the range.

Optional experimental traces (`--experimental --torch-trace`,
`--triton-trace`, `--ml-api-trace`) correlate framework operators to
counters. They are not required for ROCTX.

### 4h. Produce / retain rocpd output

Profile mode **always** captures through rocpd. Converted
`results_*.csv` / `pmc_perf.csv` land in the workload directory for analyze.

To keep the raw rocpd SQLite database in the workload directory (large;
flag is deprecated but still the way to retain `.db` today):

```bash
rocprof-compute profile \
    --name <workload_name> \
    --retain-rocpd-output \
    -- <your_app_command_and_args>
```

Analyze reports in `csv` or `db` format **require** a rocpd-collected
workload (the current default). See `skills/analyze/SKILL.md` for
`--output-format db|csv|txt`.

## 5. Validate profiling output

```bash
ls -lh ./workloads/<workload_name>/

# Typical layout (gpu model subdirectory may vary):
# workloads/<workload_name>/<gpu_model>/
#   profiling_config.yaml   (includes format_rocprof_output: rocpd)
#   pmc_perf.csv            (or .csv.gz — pivoted counters)
#   results_*.csv           (long-form rocpd conversion intermediates)
#   timestamps / sysinfo    as produced by this version
#   empirRoof_gpu-<id>_<dtype>.html   if roofline was collected
#   <pass>_<pid>.db         only if --retain-rocpd-output was passed
```

Confirm `profiling_config.yaml` exists and `pmc_perf.csv` (or gzipped
equivalent) has a header plus at least one kernel row. Empty output usually
means the app dispatched no GPU kernels.

Point `--path` in analyze at the workload directory (the folder that
contains `profiling_config.yaml`).

## 6. Common profiling issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `rocprof-compute: command not found` | Not on PATH | `export PATH=/opt/rocm/bin:$PATH` |
| `HSA_STATUS_ERROR_OUT_OF_RESOURCES` | GPU OOM during replay | Filter with `-k`; shrink the workload |
| Empty / missing `pmc_perf.csv` | No kernel dispatches | Run the app standalone first |
| Wrong GPU | Multi-GPU process visibility | `HIP_VISIBLE_DEVICES=<N>` (not `--device`) |
| Unknown flag `--path` / `--kernel-names` / `--single-pass` | Stale docs | Use `-d`, `-k`, `--set` |
| PC sampling rejected | Missing `--experimental` | Add `--experimental --pc-sampling` |
| Version skew | ROCm vs tool mismatch | Compare `rocprof-compute --version` with `/opt/rocm/.info/version` |
| Slow profiling | Many counter groups | `-k`, `--set`, or `--no-roof` |

## 7. After profiling

Use `skills/analyze/SKILL.md`. Quick preview:

```bash
rocprof-compute analyze --path ./workloads/<workload_name> --list-stats
rocprof-compute analyze --path ./workloads/<workload_name>
```

## 8. Useful reference commands

```bash
rocprof-compute profile --help
rocprof-compute profile --list-available-metrics
rocprof-compute profile --list-sets
rocprof-compute --list-metrics <arch>
rocprof-compute --list-blocks <arch>
amd-smi
rocprof-compute profile -V --name debug_run -- <your_app>
```

There is no `profile --list-devices` and no profile `--timeout`.

## 9. Related skill

- `skills/analyze/SKILL.md` — interpret counters, ROCTX ranges, stalls, and emit txt/csv/db reports
