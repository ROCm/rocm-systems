# User API

## Overview

This example demonstrates the rocprofiler-systems user API for instrumenting application code with named regions and custom annotations. It computes Fibonacci numbers across multiple threads while marking execution phases - initialization, thread creation, computation, and waiting - as named profiling regions. A custom callback adds `errno` annotations to each region push, showing how to attach application-specific metadata to trace events. Selective per-thread tracing is also demonstrated.

## Source Files

- `user-api.cpp` - Calls `rocprofsys_user_configure()` to register custom callbacks, uses `rocprofsys_user_push_region()` / `rocprofsys_user_pop_region()` to delimit named regions, and `rocprofsys_user_push_annotated_region()` to attach annotations. Also demonstrates `rocprofsys_user_stop_thread_trace()` / `rocprofsys_user_start_thread_trace()` for selective thread tracing.

## Prerequisites

- CMake 3.21+
- C++17 compiler
- rocprofiler-systems user library (`rocprofiler-systems::rocprofiler-systems-user-library`)

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/user-api -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target user-api
```

## Running

```bash
# Default: fib(10), min(16, ncpus) threads, 50000 iterations
./user-api

# Custom: fib(15), 8 threads, 10000 iterations
./user-api 15 8 10000
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Fibonacci number to compute | 10 |
| 2 | Number of threads | min(16, hardware concurrency) |
| 3 | Number of iterations per thread | 50000 |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./user-api 10 4 1000
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace with user regions |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable statistical sampling |

For binary rewrite instrumentation:

```bash
rocprof-sys-instrument -l --min-instructions=8 -E custom_push_region \
    -o user-api.inst -- ./user-api
./user-api.inst 10 4 1000
```

| Option | Purpose |
|--------|---------|
| `-l` | Instrument at the loop level so the Fibonacci loop in `run()` is profiled |
| `--min-instructions=8` | Lower the default threshold (1024) so small functions like `run()` are included |
| `-E custom_push_region` | Exclude the user callback from instrumentation to avoid recursion and trace noise (the callback runs on every region push; instrumenting it would profile the profiler rather than the workload) |

### IMPORTANT NOTE

 The `rocprof-sys-user` library will be removed in a future release. Please use `rocprofiler-sdk-roctx` instead!
