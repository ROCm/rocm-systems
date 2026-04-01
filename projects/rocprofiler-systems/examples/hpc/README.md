# HPC Training Examples

## Overview

This directory contains six HPC training examples covering GPU-accelerated scientific computing patterns: Jacobi iterative solvers (HIP, Fortran+OpenMP), matrix exponentiation with GPU streams, and hardware queue stream overlap. These examples demonstrate key HPC concepts including iterative PDE solving, halo exchange communication, GPU stream concurrency, and compute-communication overlap.

## Sub-Examples

### jacobi-hip

GPU-accelerated 2D Jacobi iterative solver using HIP. Performs halo exchange between subdomains, launches Jacobi iteration and Laplacian norm kernels, and demonstrates multi-GPU domain decomposition patterns.

**Source files:** `JacobiMain.hip`, `JacobiIteration.hip`, `JacobiSetup.hip`, `JacobiRun.hip`, `HaloExchange.hip`, `Laplacian.hip`, `Norm.hip`, `Input.hip`, `defines.hpp`, `Jacobi.hpp`, `markers.h`

**Dependencies:** HIP, roctracer (for markers)

### jacobi-fortran-targetdata-markers

Fortran Jacobi solver using OpenMP target data directives with profiling markers for GPU offloading analysis.

**Source files:** `main.f90`, `jacobi.f90`, `laplacian.f90`, `boundary.f90`, `mesh.f90`, `norm.f90`, `update.f90`, `input.f90`, `kind.f90`

**Dependencies:** Fortran compiler with OpenMP target support

### jacobi-fortran-usm

Fortran Jacobi solver using Unified Shared Memory (USM) for automatic data migration between host and device. Requires a GPU with XNACK support — see [OpenMP USM documentation](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/openmp.html#unified-shared-memory) for details.

**Source files:** Same structure as `jacobi-fortran-targetdata-markers` with USM modifications.

**Dependencies:** Fortran compiler with OpenMP USM support, GPU with XNACK support (`HSA_XNACK=1`)

### matrix-exponential-streams-sync-hip

Computes a matrix exponential via truncated Taylor series using HIP and rocBLAS, with one GPU stream per OpenMP thread for concurrent DGEMM operations. Demonstrates GPU stream-level parallelism.

**Source files:** `mat_exp.cpp`

**Dependencies:** HIP, rocBLAS, OpenMP, rocprofiler-sdk-roctx (for markers)

**Parameters:** N=20 truncation terms, t=0.5 evaluation point, 4 OpenMP threads (= 4 HIP streams)

### split-copy-compute-hw-queues

Demonstrates compute-communication overlap using separate hardware queues for data transfers and kernel execution, maximizing GPU utilization through pipelined operations.

**Source files:** `compute_comm_overlap.hip`

**Dependencies:** HIP

### julia

Julia language vector addition example demonstrating GPU profiling of Julia workloads.

**Source files:** `vecadd.jl`

**Dependencies:** Julia runtime, AMDGPU package (`julia -e 'import Pkg; Pkg.add("AMDGPU")'`)

## Prerequisites

- CMake 3.21+
- HIP runtime and `hipcc` compiler (for HIP examples)
- rocBLAS (for matrix exponential)
- Fortran compiler with OpenMP target support (for Fortran examples)
- Julia runtime (for Julia example)

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/hpc -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> <target_name>
```

**NOTE**: Individual sub-examples can be built by target name as defined in their respective `CMakeLists.txt` files.

## Profiling with rocprofiler-systems

```bash
# Profile Jacobi HIP solver
rocprof-sys-run -- ./jacobi-hip

# Profile matrix exponential
rocprof-sys-run -- ./matrix-exponential-streams-sync-hip
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy` | Trace HIP API and GPU operations |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace for stream overlap analysis |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy \
    -e ROCPROFSYS_TRACE=true \
    -- ./jacobi-hip
```
