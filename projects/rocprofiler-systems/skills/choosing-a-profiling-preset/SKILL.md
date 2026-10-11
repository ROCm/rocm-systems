---
name: choosing-a-profiling-preset
description: Recommends which rocprofiler-systems runtime profiling preset (balanced, trace-gpu, trace-hpc, sys-trace, workload-trace, etc.) to pass to `rocprof-sys-run --preset=<name>` or `rocprof-sys-sample --preset=<name>` based on the user's workload and profiling goal, and returns the ready-to-run command. Use when the user asks "which preset should I use", describes a workload (GPU kernel, MPI, OpenMP, AI/ML training, production overview) and wants profiling guidance, or invokes `rocprof-sys-run`/`rocprof-sys-sample` without picking a preset. Do NOT use for CMake build presets (`cmake --preset debug|release|ci`) used to configure/compile rocprofiler-systems itself — that is a build-configuration concern, not a profiling-run concern.
---

# Choosing a Profiling Preset

Recommends one of rocprofiler-systems' built-in runtime profiling presets for a
described workload, and returns the exact command to run it.

## Prerequisites

None to make the recommendation — this skill only reasons over static preset
metadata. Running the recommended command requires an installed or built
`rocprof-sys-run` / `rocprof-sys-sample` binary.

## Terminology

"Trace" and "tracing" mean two different things here. Keep them apart in every answer.

| Term | Meaning | Setting |
| --- | --- | --- |
| Perfetto trace output | Writes a timeline file in Perfetto format | `ROCPROFSYS_TRACE` |
| rocPD output | Writes a rocPD database | `ROCPROFSYS_USE_ROCPD` |
| Profile output | Writes timemory profile files (flat or call-stack based) | `ROCPROFSYS_PROFILE` |
| Call-stack sampling | Samples the CPU call stack at a fixed frequency | `ROCPROFSYS_USE_SAMPLING` |
| API tracing | Intercepts and records the calls of one API | see below |

API tracing is chosen per API: ROCm (`ROCPROFSYS_ROCM_DOMAINS`), MPI
(`ROCPROFSYS_USE_MPIP`), RCCL (`ROCPROFSYS_USE_RCCLP`), OpenMP (`ROCPROFSYS_USE_OMPT`)
and Kokkos (`ROCPROFSYS_USE_KOKKOSP`). ROCm API tracing is on in every built-in preset,
by default for the HIP runtime API, ROCTx markers, kernel dispatches, memory copies and
scratch memory. A preset may narrow or widen that set, and `--rocm=<domains>` overrides it.

API tracing is separate from output selection. A preset's `tracing.enabled: false` (or
`ROCPROFSYS_TRACE=false`) turns Perfetto trace output off, not API tracing. The collected
data goes to whichever outputs are on: Perfetto, rocPD or profile files. Never say a
preset "has tracing off" without saying which of these you mean.

## Preset reference

| Preset | Category | Use case |
| --- | --- | --- |
| `balanced` | general | Recommended default: moderate overhead; profile output, 50 Hz call-stack sampling, GPU metrics and rocPD output on, Perfetto trace output off |
| `profile-only` | general | Lowest overhead flat profile (production, minimal impact); Perfetto trace output, rocPD output and call-stack sampling are off, ROCm API tracing stays on |
| `detailed` | general | Deep bottleneck analysis: Perfetto trace output, profile output, call-stack sampling and GPU metrics all on |
| `trace-gpu` | gpu | GPU activity timeline: Perfetto trace output with ROCm API tracing (HIP runtime, kernel dispatch, memory copy) and GPU metrics; profile output off |
| `trace-hw-counters` | gpu | GPU hardware counters (Occupancy, VALUUtilization) |
| `workload-trace` | gpu | AI/ML training, long-running GPU/HPC workloads: Perfetto trace output (2GB buffer) and rocPD output on, MPI and RCCL API tracing |
| `trace-hpc` | hpc | Compute-intensive HPC apps: Perfetto trace output, MPI/OpenMP/Kokkos/RCCL API tracing, PAPI counter events set |
| `trace-openmp` | hpc | OpenMP GPU target-offload apps: Perfetto trace output with OpenMP API tracing and ROCm API tracing (HSA API excluded) |
| `profile-mpi` | hpc | MPI communication latency: flat profile with MPI API tracing and wall-clock per rank; no Perfetto trace or rocPD output |
| `sys-trace` | tracing | Perfetto trace output with the widest ROCm API tracing (HIP + HSA + ROCTx + RCCL) for debugging runtime-layer interactions |
| `runtime-trace` | tracing | Perfetto trace output with ROCm runtime API tracing only (excludes HSA/compiler APIs) |

## How to choose

Resolve these in order; the first question that narrows to a single preset wins.

1. **Is this a debugging/troubleshooting question about API interactions**
   (e.g. "why is my HIP call behaving oddly", "what's HSA doing under the
   hood")? → `sys-trace` (full API visibility) or `runtime-trace` (if HSA/
   compiler-level noise should be excluded).

2. **Does the workload have a dominant parallel runtime?**
   - MPI communication latency is the *only* concern, no Perfetto trace file needed →
     `profile-mpi`.
   - OpenMP GPU target-offload kernels → `trace-openmp`.
   - MPI/OpenMP/Kokkos/RCCL HPC application, compute-intensive → `trace-hpc`.
     It writes a Perfetto trace, traces those runtimes' APIs and sets CPU
     hardware-counter events (PAPI); if you want neither the trace file nor the
     counters, `profile-mpi` or `balanced` write no Perfetto trace.
   - AI/ML training or a long-running GPU-accelerated HPC job needing MPI+RCCL
     API tracing and a large Perfetto trace buffer → `workload-trace`.

3. **Is the focus purely the GPU device itself** (not a parallel runtime)?
   - GPU activity timeline (kernel dispatch, memory copies) → `trace-gpu`.
   - Specific hardware counters (occupancy, VALU utilization) → `trace-hw-counters`.

4. **Otherwise, general-purpose application profiling** — pick by overhead
   tolerance and depth:
   - Lowest overhead, quick flat profile (e.g. production) → `profile-only`; it turns
     off Perfetto trace output, rocPD output and call-stack sampling, but ROCm API calls
     and kernels still appear in the flat profile.
   - Good default for general profiling: profile output, rocPD output, 50 Hz call-stack
     sampling and GPU metrics, no Perfetto trace → `balanced`.
   - Maximum depth, willing to accept high overhead → `detailed`.

If the user's need doesn't fit neatly, default to `balanced` and mention that
domain flags can extend it (see below).

## Output

Return the exact invocation, e.g.:

```bash
rocprof-sys-run --preset=<name> -- ./myapp
# or, for sampling-based profiling:
rocprof-sys-sample --preset=<name> -- ./myapp
```

State in one line what the recommended preset turns on and what it leaves off, so the
user knows what they gain and give up. Keep the terms apart: the Perfetto trace output,
the rocPD output and call-stack sampling are separate from ROCm API tracing. For example,
`profile-only` writes a flat profile with no Perfetto trace file, no rocPD database and no
call-stack sampling, while ROCm API calls and kernels still appear in the profile.

Mention, if relevant:

- Domain flags (`--gpu`, `--rocm`, `--cpu`, `--parallel`) can layer on top of
  any preset to add/override specific metrics, e.g.
  `rocprof-sys-run --preset=balanced --gpu=temp,power -- ./myapp`.
- `--explain=<name>` and `--list-presets` let the user self-serve verify the
  choice before running.
- `--export-config=<file>.json` freezes the resolved preset+overrides into a
  reusable JSON config — this is the supported way to customize a preset, not
  hand-editing the shipped JSON files.
- `rocprof-sys-run --explain=<name>` prints exactly which `ROCPROFSYS_*`
  settings a preset applies. The full file format for custom presets is the JSON
  schema installed at `<install-prefix>/share/rocprofiler-systems/presets/schema.json`.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Confusing this with the CMake `--preset debug\|release\|ci` used to *build* rocprofiler-systems | Build presets configure compilation; profiling presets configure a profiling *run*. This skill only covers the latter. |
| Looking for the built-in preset JSON files to edit | The built-in presets are compiled into the binary and are not installed as files. Use `--explain=<name>` to inspect one and `--export-config` to produce an editable copy. |
| Picking `detailed` by default for "just get me some data" | Default to `balanced` unless the user explicitly wants maximum depth and can tolerate higher overhead. |
