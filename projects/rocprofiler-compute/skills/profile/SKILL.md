---
name: rocprofiler-compute-profile
description: Profiles an AMD GPU workload with rocprofiler-compute (rocprof-compute profile). Use when the user asks to profile, benchmark, measure, or collect hardware performance counters for a HIP/ROCm kernel or application on AMD Instinct GPUs or supported client APUs — including roofline collection, experimental PyTorch operator tracing, and experimental PC sampling.
---

# Skill: Profile an AMD GPU Workload with rocprofiler-compute

Follow this skill when a user asks to profile, benchmark, or collect GPU
hardware performance counters for an AMD GPU workload. After a successful
profile, hand off to `skills/analyze/SKILL.md`.

**Use this skill for:** Linux HIP/ROCm applications whose launch command is
available and whose execution can be replayed safely.

**Do not use this skill for:** CUDA tools, Windows, GUI/IDE workflows,
offline setup, or kernel generation. Use the
rocprofiler-sdk PC-sampling skill when the user specifically needs standalone
`rocprofv3` sampling rather than rocprofiler-compute.

**Recommended path:** verify the environment, run a full counter profile with
roofline, validate the workload, then invoke the analyze skill.

**Fallback — focused metric set:** when replay cost is too high, list metric
sets and collect the smallest relevant set. State that uncollected panels will
not be available during analysis.

## 1. Understand rocprofiler-compute

**rocprofiler-compute** is AMD's GPU kernel-level performance analysis tool.
Use it to fine-tune individual HIP kernels and maximize hardware utilization.

| Concept | Detail |
|---|---|
| CLI | `rocprof-compute` |
| Profile backends | **Perfmon counters** (multi-pass, architecture-level); **PC sampling** (experimental, instruction-level stall samples) |
| Multi-pass | Replays the workload, collecting different counter groups each pass |
| Output | Use profile mode's default output. Discuss format details only when the user requests a specific downstream format or raw data |
| Supported HW | Instinct MI100, MI200 (MI210/MI250/MI250X), MI300A/MI300X/MI325X, MI350/MI355X; client APUs (Strix, Strix Halo, Krackan, Gorgon) |
| ROCm | Use the tool paired with the installed ROCm release; ROCm 10.2 uses rocprofiler-compute 3.10 |

Do not invent flags. If unsure, run `rocprof-compute profile --help`.
For the architecture/OS/API matrix, dependencies, exclusions, and known
issues, load `references/support-and-operations.md`.

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
skills/profile/scripts/check-environment.sh
<your_app_command>
```

The helper only reports readiness; it does not install or change anything.
Also confirm the application run is representative and safe to replay.

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

### 4e. Focused topic sets

For a faster, topic-focused collection, use the named fallback:

```bash
rocprof-compute profile --list-sets
rocprof-compute profile --name <name> --set <set_name> -- <application>
```

Load `references/support-and-operations.md` before choosing a set or block.

### 4f. PC sampling (experimental)

PC sampling is experimental and hardware-dependent. Load
`references/support-and-operations.md`, query the installed device's valid
configuration, and avoid hardcoding an interval. Prefer stochastic when
supported; use `host_trap` as the named sampling fallback.

### 4g. PyTorch operator tracing

rocprofiler-compute supports ROCTx marker attribution through its experimental
PyTorch tracing feature. Do not describe arbitrary user-authored ROCTx ranges
as supported.

```bash
rocprof-compute profile \
    --experimental \
    --torch-trace \
    --name <workload_name> \
    -- python <script.py>
```

Before using this option, read the
[Torch operator mapping documentation](../../docs/how-to/profile/mode.rst).
It defines supported PyTorch versions, installation requirements, output,
analysis commands, overhead, and current limitations.

### 4h. Output handling

Use the default profile output and pass its workload directory directly to
analyze. Do not add or enumerate format options unless the user requests a
specific export or raw database. For such a request, load the output details
in `references/support-and-operations.md` or the analyze skill.

## 5. Validate profiling output

```bash
skills/analyze/scripts/inspect-workload.sh \
    ./workloads/<workload_name>/<gpu_model>
```

Point `--path` in analyze at that directory. It is the folder that contains
`profiling_config.yaml`. For MPI, the last component is the rank instead of
the GPU model. `analyze` does not search a parent directory for that file.

## 6. Common profiling issues

Load `references/support-and-operations.md`. Stop on permission, firmware,
version, standalone-application, or zero-dispatch failures instead of
guessing parameters.

## 7. After profiling

Use `skills/analyze/SKILL.md`. Quick preview:

```bash
rocprof-compute analyze --path ./workloads/<workload_name>/<gpu_model> --list-stats
rocprof-compute analyze --path ./workloads/<workload_name>/<gpu_model>
```

## 8. Useful reference commands

```bash
rocprof-compute profile --help
rocprof-compute profile --list-available-metrics
rocprof-compute profile --list-sets
rocprof-compute --list-blocks <arch>
```

## 9. Related skill

- `skills/analyze/SKILL.md` — interpret counters, PyTorch operators, stalls, and emit txt/csv/db reports
