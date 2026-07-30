# hipFILE Example

## Overview

This example demonstrates ROCm Systems Profiler tracing of the hipFILE (GPU-direct storage) API. Unlike the decode benchmarks, it is a deliberately minimal workload whose sole purpose is to exercise the hipFILE API so the calls are captured in the trace under the `rocm_hipfile_api` category. It invokes only host-side hipFILE driver APIs — version query, error-string lookup, parameter get/set, and driver open/close — so it runs on any system with the hipFILE runtime, without needing a valid GPU file mapping or storage device.

## Source Files

- `hipfile_trace.cpp` - Minimal hipFILE workload that calls host-side hipFILE driver APIs (`hipFileGetVersion`, `hipFileGetOpErrorString`, `hipFileGetParameter*` / `hipFileSetParameter*`, `hipFileDriverOpen` / `hipFileDriverGetProperties` / `hipFileDriverClose`) to generate hipFILE API trace records.

## Prerequisites

- CMake 3.25+
- HIP runtime
- hipFILE runtime library (`hipfile.h` and `libhipfile`)
- ROCProfiler-SDK 1.3.5 or later — required for hipFILE API tracing. The example builds and runs without it, but the `hipfile_api` tracing domain is only available when profiling against ROCProfiler-SDK 1.3.5 or newer.

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/hipfile -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target hipfile-trace
```

## Running

```bash
# Run the minimal hipFILE workload (no arguments required)
./hipfile-trace
```

The program prints the detected hipFILE version and exercises the host-side hipFILE driver APIs:

```text
hipFILE version: X.Y.Z
```

## Profiling with rocprofiler-systems

Add `hipfile_api` to `ROCPROFSYS_ROCM_DOMAINS` (shorthand: `hipfile`) to capture the hipFILE API:

```bash
rocprof-sys-run -e ROCPROFSYS_ROCM_DOMAINS=hipfile_api -- ./hipfile-trace
```

The captured hipFILE API calls appear in the Perfetto trace and the RocPD database under the `rocm_hipfile_api` category.

### Recommended Configuration

| Variable | Value | Purpose |
| ---------- | ------- | --------- |
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy,hipfile_api` | Trace HIP API, GPU operations, and the hipFILE API (shorthand: `hipfile`) |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace for timeline analysis |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,hipfile_api \
    -e ROCPROFSYS_TRACE=true \
    -- ./hipfile-trace
```

> **Note:** Requesting the `hipfile_api` domain against a ROCProfiler-SDK older than 1.3.5 (which does not expose the hipFILE tracing domain) results in an `unsupported ROCPROFSYS_ROCM_DOMAINS value: hipfile_api` error, the same as any other unavailable domain.
