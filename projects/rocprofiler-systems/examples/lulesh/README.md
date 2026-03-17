# LULESH

## Overview

LULESH (Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics) is a widely-used shock physics proxy application that solves the Sedov blast wave problem on an unstructured hexahedral mesh. This implementation uses Kokkos for performance portability across CPU and GPU backends, with optional MPI for distributed domain decomposition. Adaptive time stepping uses the Courant stability condition, and `CAUSAL_PROGRESS_NAMED` annotations are embedded for causal profiling analysis. LULESH is a representative HPC mini-app useful for profiling GPU kernel performance, memory access patterns, and multi-physics computation pipelines.

## Source Files

- `lulesh.cc` - Main simulation driver with adaptive time stepping (`TimeIncrement()`), Kokkos view buffer management, and aligned memory allocation.
- `lulesh.h` - Data structures and constants for the simulation domain.
- `lulesh_tuple.h` - Tuple helpers for Kokkos parallel dispatch.
- `lulesh-comm.cc` - MPI halo exchange communication routines.
- `lulesh-init.cc` - Domain initialization and mesh setup.
- `lulesh-util.cc` - Utility functions and timing infrastructure.
- `lulesh-viz.cc` - Visualization output routines.
- `external/kokkos/` - Kokkos submodule (can be built from source or use system installation).

## Prerequisites

- CMake 3.21+
- C++17 compiler
- Kokkos (built from submodule or system installation)
- (Optional) MPI for distributed execution
- (Optional) HIP or CUDA backend for GPU execution

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/lulesh -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target lulesh
```

**Targets:**

| Target | Description |
|--------|-------------|
| `lulesh` | Standard build with debug symbols |
| `lulesh-rocprofsys` | Linked with rocprofiler-systems user library |
| `lulesh-coz` | Linked with COZ profiler (if available) |

To build Kokkos from the included submodule:

```bash
cmake -B build -DROCPROFSYS_BUILD_KOKKOS=ON lulesh/
```

## Running

```bash
# Default problem size
./lulesh

# Custom: 35 iterations, 50^3 mesh, print progress
./lulesh -i 35 -s 50 -p

# With MPI
mpirun -np 8 ./lulesh -i 35 -s 30 -p
```

**Arguments:**

| Flag | Description |
|------|-------------|
| `-i` | Number of iterations |
| `-s` | Problem size (elements per edge) |
| `-p` | Print progress |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./lulesh -i 35 -s 50 -p
```

For causal profiling:

```bash
rocprof-sys-causal -- ./lulesh-rocprofsys -i 35 -s 50 -p
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable statistical sampling |
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy` | Trace GPU operations (when using GPU backend) |

For MPI runs:

```bash
mpirun -np 8 rocprof-sys-run \
    -e ROCPROFSYS_USE_MPIP=ON \
    -e ROCPROFSYS_TRACE=true \
    -- ./lulesh -i 35 -s 30 -p
```
