# SDMA Test

## Overview

This example benchmarks SDMA (System DMA) engine performance by measuring bandwidth for Host-to-Device (H2D), Device-to-Device (D2D), and Device-to-Host (D2H) memory transfers. It runs configurable iterations of async copy operations using pinned host memory and HIP streams, reports per-transfer-type bandwidth in GB/s, and verifies data integrity via `memcmp`. An infinite-loop mode allows sustained profiling of SDMA utilization over extended periods.

## Source Files

- `sdma_test.cpp` - Implements transfer benchmarking with `hipMemcpyAsync`, CLI argument parsing with size suffix support (K/M/G), bandwidth calculation, and signal handling for graceful shutdown in infinite mode.

## Prerequisites

- CMake 3.21+
- HIP runtime and `hipcc` compiler

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/sdma_test -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target sdma_test
```

## Running

```bash
# Default: 512 MB, 10 iterations, 10 copies/iteration, device 0
./sdma_test

# Custom: 1 GB transfers, 5 iterations
./sdma_test -s 1024 -n 5

# Infinite mode (Ctrl+C to stop)
./sdma_test -s 256 -n 0
```

**Flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `-s, --size` | Transfer size in MB (supports K/M/G suffixes) | 512 |
| `-n, --iterations` | Number of iterations (0 = infinite) | 10 |
| `-c, --copies` | Number of copies per iteration | 10 |
| `-d, --device` | GPU device ID | 0 |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./sdma_test -s 256 -n 5
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy` | Trace HIP API and memory copy operations |
| `ROCPROFSYS_ROCM_EVENTS` | `SQ_WAVES,GRBM_COUNT` | Sample GPU hardware counters |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace for timeline analysis |

To capture SDMA-specific metrics:

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,memory_copy \
    -e ROCPROFSYS_TRACE=true \
    -- ./sdma_test -s 512 -n 5
```
