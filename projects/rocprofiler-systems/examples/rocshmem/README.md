# rocSHMEM Host-Stream API Tracing Example

## Overview

This example exercises all nine rocSHMEM host-stream APIs in a minimal two-PE
program. Every PE calls every API so that rocprofiler-systems tracing captures
all nine `rocm_rocshmem_api` spans:

| API | Description |
| --- | --- |
| `barrier_all_on_stream` | Global barrier across all PEs |
| `sync_all_on_stream` | Lightweight global synchronization |
| `putmem_on_stream` | One-sided write to a remote PE |
| `getmem_on_stream` | One-sided read from a remote PE |
| `putmem_signal_on_stream` | Write data and set a remote signal atomically |
| `signal_wait_until_on_stream` | Wait until a local signal word satisfies a condition |
| `broadcastmem_on_stream` | PE 0 broadcasts a buffer to all PEs |
| `alltoallmem_on_stream` | Each PE sends a buffer to every other PE |
| `quiet_on_stream` | Ensure all outstanding remote operations are complete |

## Source Files

- `rocshmem.cpp` — Minimal two-PE program that calls all nine host-stream APIs
  in sequence using symmetric memory buffers on a single HIP stream.

## Prerequisites

- CMake 3.25+
- ROCm with HIP runtime
- rocSHMEM ≥ 3.6.0 (included in ROCm 7.15)
- MPI runtime (e.g. Open MPI)

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/rocshmem \
    -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/
cmake --build <build_dir> --target rocshmem
```

## Running

```bash
mpirun -np 2 ./rocshmem
```

Expected output:

```text
[PE 0/2] rocshmem: all 9 host-stream APIs completed
[PE 1/2] rocshmem: all 9 host-stream APIs completed
```

## Profiling with rocprofiler-systems

```bash
mpirun -np 2 rocprof-sys-run -- ./rocshmem
```

### Recommended Configuration

| Variable | Value | Purpose |
| --- | --- | --- |
| `ROCPROFSYS_ROCM_DOMAINS` | `rocshmem_api` | Enable rocSHMEM host-stream API tracing |
| `ROCPROFSYS_USE_ROCPD` | `ON` | Generate rocpd database |
| `ROCPROFSYS_USE_SAMPLING` | `OFF` | Disable statistical sampling (use instrumentation only) |

```bash
mpirun -np 2 rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,rocshmem_api \
    -e ROCPROFSYS_USE_ROCPD=ON \
    -- ./rocshmem
```

The resulting rocpd database will contain `rocm_rocshmem_api` spans for each of
the nine host-stream API calls made by each PE. It can be viewed with
[ROCm Optiq](https://github.com/ROCm/roc-optiq) visualizer.
