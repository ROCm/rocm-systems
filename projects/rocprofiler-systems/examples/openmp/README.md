# OpenMP

## Overview

This directory contains NAS Parallel Benchmarks (NPB) implemented with OpenMP threading. The Conjugate Gradient (CG) benchmark solves a sparse symmetric positive definite system using the CG method, exercising sparse matrix-vector multiplication and dot products across threads. The LU benchmark solves a 3D coupled PDE system using Lower-Upper Gauss-Seidel decomposition with block-tridiagonal operations. Both benchmarks are useful for profiling OpenMP parallel region overhead, thread synchronization, and cache behavior in scientific computing workloads.

## Source Files

- `CG/cg.cpp` - NAS CG benchmark with sparse matrix operations, configurable problem size parameters, and unrolled sparse matrix-vector multiply variants.
- `LU/lu.cpp` - NAS LU benchmark with 3D grid PDE solver, block-tridiagonal operations, and padded arrays for cache optimization.
- `common/` - Shared utilities: `c_print_results.cpp` (result reporting), `c_randdp.cpp` (random number generation), `c_timers.cpp` (timer infrastructure), `npb-CPP.hpp` (common definitions), `wtime.cpp`/`wtime.hpp` (wall-clock timing).

### OpenMP Target Offloading

- `target/main.cpp` - OpenMP target offloading example demonstrating GPU computation via `#pragma omp target`.
- `target/library.cpp` - Companion library for target offloading tests.

### OpenMP Fortran Host and Offload Programs

- `fortran/host.f90` - OpenMP host example that uses two CPU threads to update an integer array in round-robin order.
- `fortran/offload.f90` - OpenMP target offload example that launches a GPU target region to transpose a matrix.

## Prerequisites

- CMake 3.25+
- ROCm install providing `amdclang++` and `amdflang` (resolved from `ROCM_PATH` or `/opt/rocm`)
  - `amdclang++` is **required** for the C++ examples (`openmp-cg`, `openmp-lu`, and the `openmp-common` shared utility objects).
  - `amdflang` (version 20 or newer) is **required** for the Fortran examples (`fortran/host.f90`, `fortran/offload.f90`).

If a required compiler for an **enabled** target is missing, CMake configuration fails with a `FATAL_ERROR` (append `openmp` to `ROCPROFSYS_DISABLE_EXAMPLES` to skip building these examples).

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/openmp
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/
cmake --build <build_dir> --target openmp-cg openmp-lu
```

**Targets:**

| Target | Description |
| -------- | ------------- |
| `openmp-cg` | NAS Conjugate Gradient benchmark |
| `openmp-lu` | NAS LU Gauss-Seidel benchmark |
| `openmp-target` | OpenMP GPU target offloading example |
| `openmp-fortran-host` | Fortran OpenMP host example |
| `openmp-fortran-offload` | Fortran OpenMP target offload example |

## Running

```bash
# Set thread count
export OMP_NUM_THREADS=4

# Run CG benchmark
./openmp-cg

# Run LU benchmark
./openmp-lu
```

**Recommended OpenMP settings:**

| Variable | Value | Purpose |
| ---------- | ------- | --------- |
| `OMP_NUM_THREADS` | `2`-`N` | Number of OpenMP threads |
| `OMP_PROC_BIND` | `spread` | Thread placement policy |
| `OMP_PLACES` | `threads` | Thread affinity granularity |

## Profiling with rocprofiler-systems

```bash
OMP_NUM_THREADS=4 rocprof-sys-run -- ./openmp-cg
```

### Recommended Configuration

| Variable | Value | Purpose |
| ---------- | ------- | --------- |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace with OpenMP regions |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable statistical sampling |
| `ROCPROFSYS_SAMPLING_FREQ` | `50` | Sampling frequency (interrupts/sec) |

```bash
OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=threads \
    rocprof-sys-run \
    -e ROCPROFSYS_TRACE=true \
    -e ROCPROFSYS_USE_SAMPLING=ON \
    -- ./openmp-cg
```
