# ROCTx

## Overview

This example suite demonstrates the ROCTx tracing API for annotating GPU workloads with human-readable profiling markers. It covers range-based profiling using both start/stop and push/pop semantics, entity naming for clearer trace visualization, the pause/resume API for selective profiling windows, and the `ROCPROFSYS_TRACE_REGION` feature for filtering traces to specific roctx regions.

## Source Files

- `roctx_example_kernels.hpp` - Shared header providing GPU kernel definitions (`DEFINE_KERNEL`, `LAUNCH_KERNEL`, `LAUNCH_KERNEL_STREAM` macros), a `gpu_buffer` RAII wrapper, a `thread_barrier` for multi-threaded synchronization, and `run_on_threads()` for spawning worker threads with per-thread HIP streams.
- `roctx.cpp` - Demonstrates `roctxRangeStart()`/`roctxRangeStop()` for timing ranges, `roctxRangePush()`/`roctxRangePop()` for nested ranges, `roctxMark()` for instant markers, `roctxNameOsThread()`/`roctxNameHipDevice()`/`roctxNameHipStream()`/`roctxNameHsaAgent()` for entity labeling, and `roctxProfilerPause()`/`roctxProfilerResume()` for selective profiling.
- `pause_resume.cpp` - Demonstrates pause/resume for selective profiling. Launches kernels `CodeBlock_Z`, `A`, `B`, `C`, `D` with a pause around `B`. Expected profiled: `{Z, A, C, D}`.
- `selective_region.cpp` - Demonstrates `ROCPROFSYS_TRACE_REGION` for filtering traces to named regions. Multi-threaded workload with nested regions (`Region1`, `Region2`, `Region3`) and kernels `CodeBlock_A` through `G`. When filtered to `Region1`, only `{B, C, D, F}` are profiled.
- `selective_region_pause_1.cpp` - Pause and resume both occur **inside** the target region. Expected profiled with `Region1` filter: `{CodeBlock_A, CodeBlock_C}`.
- `selective_region_pause_2.cpp` - Pause occurs **before** the target region (ignored by region filtering). Expected profiled with `Region1` filter: `{CodeBlock_A, CodeBlock_B, CodeBlock_C}`.
- `selective_region_pause_3.cpp` - Pause occurs **inside** the region, resume occurs **outside** after region stop. Expected profiled with `Region1` filter: `{CodeBlock_A}`.

## Prerequisites

- CMake 3.25+
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
./roctx             # ROCTx API demo
./pause_resume      # Pause/resume selective profiling
./selective_region  # Region-filtered tracing
```

No command-line arguments. Each program runs a fixed workflow of GPU kernel launches with ROCTx annotations.

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
| `ROCPROFSYS_TRACE_REGION` | `Region1` | Filter tracing to only the named region(s) |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch \
    -e ROCPROFSYS_TRACE=true \
    -- ./roctx
```

### Selective Region Example

```bash
ROCPROFSYS_TRACE_REGION="Region1" rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch \
    -- ./selective_region
```
