# RCCL

## Overview

This example builds the rccl-tests performance test suite, which benchmarks RCCL (ROCm Communication Collectives Library) operations across multiple GPUs. It includes performance tests for all major collectives - AllGather, AllReduce, AllToAll, Broadcast, Gather, Reduce, ReduceScatter, Scatter, and SendRecv - measuring bandwidth and latency for inter-GPU communication. This is useful for profiling GPU-to-GPU collective communication patterns and PCIe/Infinity Fabric interconnect performance.

## Source Files

- `rccl-tests/` - Git submodule containing the RCCL performance test suite with source under `src/` and verification utilities under `verifiable/`.

## Prerequisites

- CMake 3.21+
- HIP runtime and `hipcc` compiler
- RCCL library
- Multiple AMD GPUs (tests target supported GPU architectures)

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/rccl -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target all_reduce_perf
```

**Targets:**

| Target | Description |
|--------|-------------|
| `all_gather_perf` | AllGather performance test |
| `all_reduce_perf` | AllReduce performance test |
| `alltoall_perf` | AllToAll performance test |
| `alltoallv_perf` | AllToAllV performance test |
| `broadcast_perf` | Broadcast performance test |
| `gather_perf` | Gather performance test |
| `reduce_perf` | Reduce performance test |
| `reduce_scatter_perf` | ReduceScatter performance test |
| `scatter_perf` | Scatter performance test |
| `sendrecv_perf` | SendRecv performance test |

## Running

```bash
# Run AllReduce across all GPUs
./all_reduce_perf -b 8 -e 128M -f 2 -g <num_gpus>

# Run AllToAll
./alltoall_perf -b 1K -e 64M -g <num_gpus>
```

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./all_reduce_perf -b 8 -e 128M -f 2 -g 2
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy` | Trace HIP API and GPU operations |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace for timeline analysis |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy \
    -e ROCPROFSYS_TRACE=true \
    -- ./all_reduce_perf -b 8 -e 128M -f 2 -g 2
```
