# RCCL

## Overview

This example builds the rccl-tests performance test suite, which benchmarks RCCL (ROCm Communication Collectives Library) operations across multiple GPUs. It includes performance tests for the major collectives - AllGather, AllReduce, AllToAll, AllToAllV, Broadcast, Reduce, and ReduceScatter - measuring bandwidth and latency for inter-GPU communication. This is useful for profiling GPU-to-GPU collective communication patterns and PCIe/Infinity Fabric interconnect performance.

## Source Files

- `rccl-tests/` - Vendored copy of the RCCL performance test suite, with source under `src/` and verification utilities under `verifiable/`.

## Prerequisites

- CMake 3.25+
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
| -------- | ------------- |
| `all_gather_perf` | AllGather performance test (`ncclAllGather`) |
| `all_reduce_perf` | AllReduce performance test (`ncclAllReduce`) |
| `alltoall_perf` | AllToAll performance test (`ncclAllToAll`) |
| `alltoallv_perf` | AllToAllV performance test (`ncclAllToAllv`, `ncclSend`/`ncclRecv`) |
| `broadcast_perf` | Broadcast performance test (`ncclBroadcast`, `ncclBcast`) |
| `reduce_perf` | Reduce performance test (`ncclReduce`) |
| `reduce_scatter_perf` | ReduceScatter performance test (`ncclReduceScatter`) |

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
| ---------- | ------- | --------- |
| `ROCPROFSYS_ROCM_DOMAINS` | `rccl_api, hip_runtime_api,kernel_dispatch,memory_copy` | RCCL collective API calls, Trace HIP API and GPU operations |
| `ROCPROFSYS_USE_RCCLP` | `true` | Alternative way to enable RCCL API tracing (implicitly adds the `rccl_api` domain) |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace for timeline analysis |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |

> **Note:** RCCL collective calls (`ncclAllReduce`, `ncclBroadcast`, ...) are only captured when the `rccl_api` domain is enabled. Add `rccl_api` to `ROCPROFSYS_ROCM_DOMAINS`, or set `ROCPROFSYS_USE_RCCLP=true` (which enables the `rccl_api` domain for you).

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=rccl_api,hip_runtime_api,kernel_dispatch,memory_copy \
    -e ROCPROFSYS_TRACE=true \
    -- ./all_reduce_perf -b 8 -e 128M -f 2 -g 2
```
