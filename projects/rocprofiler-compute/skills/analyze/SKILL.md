---
name: rocprofiler-compute-analyze
description: Analyzes AMD GPU profiling results collected by rocprofiler-compute (rocprof-compute analyze). Use when the user asks to interpret, inspect, view, or investigate rocprofiler-compute output — including roofline, occupancy, memory bandwidth, instruction stalls, ROCTX ranges, bottleneck identification, and writing txt/csv/db reports from rocpd-collected workloads.
---

# Skill: Analyze AMD GPU Profiling Results with rocprofiler-compute

Follow this skill when a user wants to understand profiler data and where
the GPU bottleneck is. If no workload directory exists, collect one first
with `skills/profile/SKILL.md`.

## 1. Understand the analysis system

| Mode | What it measures | When to use |
|---|---|---|
| Perfmon counter analysis | Architecture-level counters (memory BW, occupancy, MFMA, stall types) | Overall kernel efficiency; memory vs compute |
| PC sampling analysis | Stochastic / host-trap samples with stall reasons at ISA offsets | After perfmon, which instructions stall and why |

Primary workflow is CLI (stdout, or txt/csv/db files). Interactive GUI:
Optiq. Do not invent flags; use `rocprof-compute analyze --help`.

Profile mode always captures via **rocpd**. Analyze `--output-format csv`
and `db` require a rocpd-collected workload (current default).

## 2. Prerequisites

```bash
ls -lh ./workloads/<workload_name>/
# Expect profiling_config.yaml and pmc_perf.csv (possibly under a GPU-model
# subdirectory). Optional retained rocpd *.db if profile used
# --retain-rocpd-output.

rocprof-compute analyze --help
```

Point `--path` / `-p` at the directory that contains `profiling_config.yaml`
(the workload root or the GPU-model folder, depending on how it was
collected).

## 3. Terminal analysis

**Strategy**

- Target **one kernel at a time**. Identify the hottest kernel with
  `--list-stats`, then use `-k` / `--kernel` with that integer id.
- Top-down order:
  1. System Speed-of-Light
  2. Memory Chart
  3. Roofline

  Then open the matching detailed block (Wavefront, Compute, Scheduler, or
  Memory). Verify block ids with `--list-available-metrics`; they are
  architecture-specific.

### 3a. List kernels and dispatches (always first)

```bash
rocprof-compute analyze \
    --path ./workloads/<workload_name> \
    --list-stats
```

Note integer **kernel IDs** for `-k` and **1-based dispatch IDs** for `-d` /
`--dispatch`. ROCTX range names appear alongside dispatches when the app was
annotated.

### 3b. Full report

```bash
rocprof-compute analyze --path ./workloads/<workload_name>
rocprof-compute analyze --path ./workloads/<workload_name> | less -R
```

### 3c. Produce reports (including rocpd-backed csv/db)

`--output-format` is one of `stdout` (default), `txt`, `csv`, `db`.
`txt`/`csv`/`db` disable terminal output. Default file name is
`rocprof_compute_<uuid>`; override with `--output-name`.

```bash
# Text report
rocprof-compute analyze \
    --path ./workloads/<workload_name> \
    --output-format txt \
    --output-name my_analysis

# One CSV per analysis view (requires rocpd-collected profile)
rocprof-compute analyze \
    --path ./workloads/<workload_name> \
    --output-format csv \
    --output-name my_analysis

# SQLite analysis database (requires rocpd-collected profile)
rocprof-compute analyze \
    --path ./workloads/<workload_name> \
    --output-format db \
    --output-name my_analysis
```

If csv/db fail because the workload is not rocpd, re-profile with current
`rocprof-compute profile` (rocpd is the only backend). Optionally keep the
raw capture with `--retain-rocpd-output` on profile.

### 3d. Filter metric blocks

```bash
rocprof-compute analyze \
    --path ./workloads/<workload_name> \
    --list-available-metrics

rocprof-compute analyze \
    --path ./workloads/<workload_name> \
    -b 1 2
```

Use metric ids from `--list-metrics` / `--list-available-metrics`, or block
ids/aliases from `--list-blocks`.

### 3e. Kernels, dispatches, ROCTX ranges

```bash
# Kernel id from --list-stats
rocprof-compute analyze --path ./workloads/<workload_name> -k 0

# Dispatch ids are 1-based
rocprof-compute analyze --path ./workloads/<workload_name> -d 12 34 --decimal 3

# Kernel + block
rocprof-compute analyze --path ./workloads/<workload_name> -k 0 -b <block_id>
```

**ROCTX:** `--list-stats` shows range markers. Note dispatch IDs inside the
range, then pass those IDs to `-d`. That is how range-level metrics are
isolated without a separate ROCTX analyze flag.

### 3f. Top-down workflow

```bash
# Hottest kernel
rocprof-compute analyze --path ./workloads/<workload_name> --list-stats

# Speed-of-Light / Memory Chart — confirm ids with --list-available-metrics
rocprof-compute analyze --path ./workloads/<workload_name> -k 0 -b 0
rocprof-compute analyze --path ./workloads/<workload_name> -k 0 -b 1

# Roofline HTML from profile (if not --no-roof):
# ./workloads/<workload_name>/<gpu>/empirRoof_gpu-0_FP32.html
# Left of ridge = memory-bound; right = compute-bound.

rocprof-compute analyze \
    --path ./workloads/<workload_name> \
    -k 0 \
    -R FP16 BF16 FP32
```

Then drill into L2/vL1D/LDS (memory), Compute/MFMA/VALU (compute),
Wavefront (occupancy / VGPR-LDS), or Scheduler (VMEM/LDS/barrier stalls).

### 3g. Normalize metrics

`-n` / `--normal-unit`: `per_kernel` (default), `per_wave`, `per_cycle`,
`per_second`.

```bash
rocprof-compute analyze --path ./workloads/<workload_name> -n per_wave
```

Use a unit that matches the metric: bandwidth as `per_second` or
`per_kernel`; occupancy as `per_wave` or `per_kernel`; IPC as `per_cycle`
or `per_wave`. When unsure, keep `per_kernel`.

### 3h. PC sampling

Requires a profile collected with `--experimental --pc-sampling`. Default
sort is **count** (hottest first).

```bash
rocprof-compute analyze \
    --path ./workloads/<workload_name> \
    --pc-sampling-sorting-type count \
    --pc-sampling-rows 10 \
    -k <kernel_id>
```

| Column | Meaning |
|---|---|
| PC offset | Instruction offset in the kernel ISA |
| Sample count | How often this PC was sampled |
| Stall reason | Why the wavefront stalled (VMEM, LDS, barrier, VALU, …) |

High sample count + VMEM stall → that instruction is waiting on global
memory. Analyze already correlates samples to ISA/source when symbols exist.

`--output-format csv` writes per-kernel disassembly under
`per_kernel_pc_sampling/`. `--output-format db` stores the same in the
analysis database.

## 4. Interpret key metric blocks

Ground every recommendation in numbers from the report. Do not guess.

### 4a. Roofline

Achieved FLOPS vs arithmetic intensity against peak compute and peak HBM.

- Left of ridge → **memory-bound**: reuse, tiling, LDS, less global traffic.
- Right of ridge → **compute-bound**: instruction mix, MFMA, less divergence, ILP.
- Far below both ceilings → occupancy or scheduling problem.

### 4b. Memory bandwidth

| Metric | Good | Warning | Action |
|---|---|---|---|
| HBM BW utilization | > 70% of peak | < 30% | Increase reuse / tile size |
| L2 hit rate | > 80% | < 50% | Spatial/temporal locality |
| L1 (vL1D) hit rate | > 90% | < 70% | Shrink working set, use LDS |
| LDS bank conflicts | 0 | > 0 | Pad arrays; change access pattern |

### 4c. Occupancy

| Metric | Meaning | If low |
|---|---|---|
| Active waves / CU | In-flight wavefronts per CU | Reduce VGPR/SGPR or LDS per workgroup |
| Theoretical occupancy | Max waves from register/LDS alloc | Same |
| Wave64 utilization | Full 64-lane waves | Check launch bounds |

VGPRs are the usual occupancy limiter.

### 4d. Compute stalls

| Stall | Meaning | Fix |
|---|---|---|
| VMEM | Waiting on global/scratch | Prefetch; raise arithmetic intensity |
| LDS | Waiting on LDS | Fewer bank conflicts; pipeline accesses |
| Barrier | Waiting at `s_barrier` | Less sync; reorder work |
| SALU | Scalar/branch bottleneck | Less divergence |
| VALU idle | Vector ALUs underused | More work per thread; wider vectors |

### 4e. MFMA

For GEMM/convolutions, MFMA utilization should be high (rule of thumb:
> 80% for a tuned GEMM). Low MFMA + high VMEM → data-starved, not
compute-bound.

## 5. Bottleneck decision tree

```
START
  ├─ HBM BW utilization > 80%?
  │     YES → Memory-bound (HBM). Raise AI; LDS tiling; consider lower precision.
  ├─ L2 hit rate < 50%?
  │     YES → Poor reuse. Layout (AoS→SoA); smaller working set; explicit LDS.
  ├─ Occupancy (active waves/CU) < 50% of theoretical?
  │     YES → Occupancy-limited. Cut VGPR/LDS; avoid scratch spills.
  ├─ MFMA utilization < 60% (GEMM)?
  │     YES → Data-starved MFMA. Double-buffer; larger tiles.
  ├─ VMEM stall > 40% of cycles?
  │     YES → Memory-latency bound. Prefetch; more independent loads.
  └─ All utilization < 30%?
        → Launch config or serialization. Check grid/block size; atomics; sync.
```

## 6. Visual post-analysis with Optiq

Optiq reads the workload directory:
https://rocm.docs.amd.com/projects/optiq/

## 7. Compare two workloads

```bash
rocprof-compute profile --name baseline -- ./app_v1
rocprof-compute profile --name optimized -- ./app_v2

rocprof-compute analyze \
    --path ./workloads/baseline \
    --path ./workloads/optimized

rocprof-compute analyze \
    --path ./workloads/baseline \
    --path ./workloads/optimized \
    --output-format csv \
    --output-name comparison
```

## 8. Common analysis issues

| Symptom | Likely cause | Fix |
|---|---|---|
| No workload data | Wrong `--path` or failed profile | Find `profiling_config.yaml` / `pmc_perf.csv` |
| Roofline missing | Profiled with `--no-roof` | Re-profile without `--no-roof` |
| All metrics 0 | Kernel too short | More iterations |
| Empty / missing panels | Counter group not collected | Re-profile without a narrow `--set` / `-b` |
| csv/db refused | Workload not rocpd | Re-profile with current profile mode |
| No kernel IDs | No dispatches | Verify the app actually launched kernels |

## 9. Verified commands

```bash
rocprof-compute analyze --path ./workloads/<name> --list-stats
rocprof-compute analyze --path ./workloads/<name> --list-available-metrics
rocprof-compute analyze --path ./workloads/<name>
rocprof-compute analyze --path ./workloads/<name> --output-format txt --output-name report
rocprof-compute analyze --path ./workloads/<name> --output-format db --output-name report
rocprof-compute analyze --path ./workloads/<name> -k 0
rocprof-compute analyze --path ./workloads/<name> -d 12 34 --decimal 3
rocprof-compute analyze --path ./workloads/<name> -b 1 2 3
rocprof-compute analyze --path ./workloads/<name> -n per_wave
rocprof-compute analyze --path ./workloads/baseline --path ./workloads/opt
rocprof-compute analyze --help
```

## 10. Related skill

- `skills/profile/SKILL.md` — collect counters via rocpd, ROCTX, PC sampling
