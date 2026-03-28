# TransferBench

## Overview

TransferBench is a comprehensive all-to-all transfer benchmark that measures data movement performance across CPU, GPU compute (GFX), GPU SDMA, and NIC RDMA executors. It supports configurable transfer patterns, block sizes, and hardware queue limits, making it useful for profiling interconnect bandwidth, PCIe throughput, and multi-device communication topologies. The benchmark validates transfer correctness and reports per-executor-type bandwidth.

## Source Files

- `AllToAll.cpp` - Main benchmark driver with environment variable configuration, transfer pattern setup, and performance measurement loop.
- `TransferBench.hpp` - Data structures, executor type definitions (`EXE_CPU`, `EXE_GPU_GFX`, `EXE_GPU_DMA`, `EXE_NIC`), and configuration parameters.

## Prerequisites

- CMake 3.21+
- HIP runtime and `hipcc` compiler
- HSA runtime library (headers and library)
- AMD Instinct GPU (APU-only targets are skipped)

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/transferBench -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target transferBench
```

## Running

```bash
./transferBench
```

TransferBench is primarily configured through environment variables. Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `NUM_ITERATIONS` | Number of benchmark iterations | varies |
| `BLOCK_BYTES` | Transfer block size in bytes | 256 |
| `GFX_BLOCK_SIZE` | GPU compute block size | 256 |
| `USE_HIP_EVENTS` | Use HIP events for timing | varies |
| `GPU_MAX_HW_QUEUES` | Maximum GPU hardware queues | 4 |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./transferBench
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation` | Trace all GPU memory operations |
| `ROCPROFSYS_ROCM_EVENTS` | `SQ_WAVES,GRBM_COUNT` | Sample GPU hardware counters |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy \
    -e ROCPROFSYS_TRACE=true \
    -- ./transferBench
```
