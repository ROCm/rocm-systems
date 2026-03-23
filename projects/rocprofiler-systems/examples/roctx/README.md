# ROCTx

## Overview

This example demonstrates the ROCTx tracing API for annotating GPU workloads with human-readable profiling markers. It shows range-based profiling using both start/stop and push/pop semantics, names OS threads, HIP devices, HIP streams, and HSA agents for clearer trace visualization, and exercises the pause/resume API for selective profiling windows. A multi-threaded workload launches a simple GPU kernel to show how ROCTx annotations appear alongside HIP API and kernel dispatch events in Perfetto traces.

## Source Files

- `roctx.cpp` - Demonstrates `roctxRangeStart()`/`roctxRangeStop()` for timing ranges, `roctxRangePush()`/`roctxRangePop()` for nested ranges, `roctxMark()` for instant markers, `roctxNameOsThread()`/`roctxNameHipDevice()`/`roctxNameHipStream()`/`roctxNameHsaAgent()` for entity labeling, and `roctxProfilerPause()`/`roctxProfilerResume()` for selective profiling.

## Prerequisites

- CMake 3.21+
- HIP runtime and `hipcc` compiler
- rocprofiler-sdk-roctx library

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/roctx -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target roctx
```

## Running

```bash
./roctx
```

No command-line arguments. The program runs a fixed workflow: names the device and stream, launches a GPU kernel with nested range annotations, creates a worker thread with its own markers, and demonstrates pause/resume.

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./roctx
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,marker_api,kernel_dispatch` | Trace HIP API, ROCTx markers, and kernel launches |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace with ROCTx annotations |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch \
    -e ROCPROFSYS_TRACE=true \
    -- ./roctx
```
