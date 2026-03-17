# Transpose

## Overview

This example performs a tiled 2D matrix transpose on the GPU using HIP. Each thread block uses a 32x32 shared memory tile to coalesce global memory accesses, and the host launches the kernel from multiple CPU threads - each with its own HIP stream - to exercise concurrent GPU execution. The workload is useful for profiling kernel dispatch latency, multi-stream overlap, and memory throughput under sustained GPU load. Optional MPI support adds an `MPI_Alltoall` phase to test distributed profiling.

## Source Files

- `transpose.cpp` - Defines the `transpose_a` kernel (shared-memory tiled transpose), the `run()` worker that allocates device memory and iterates kernel launches, and `verify()` for correctness checking.

## Prerequisites

- CMake 3.21+
- HIP runtime and `hipcc` compiler
- (Optional) MPI for distributed transpose

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/transpose -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target transpose
```

To enable MPI support:

```bash
cmake -B build -DTRANSPOSE_USE_MPI=ON -DCMAKE_PREFIX_PATH=/opt/rocm transpose/
cmake --build build
```

## Running

```bash
# Default: 2 threads, 500 iterations, sync every 10
./transpose

# Custom: 4 threads, 100 iterations, sync every 5
./transpose 4 100 5
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Number of CPU threads (each gets a HIP stream) | 2 |
| 2 | Number of kernel iterations | 500 |
| 3 | Synchronize every N iterations | 10 |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./transpose 4 200 10
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy` | Trace HIP API calls, kernel launches, and memory transfers |
| `ROCPROFSYS_ROCM_EVENTS` | `SQ_WAVES,GRBM_COUNT` | Sample GPU hardware counters |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy \
    -e ROCPROFSYS_TRACE=true \
    -- ./transpose 4 200 10
```
