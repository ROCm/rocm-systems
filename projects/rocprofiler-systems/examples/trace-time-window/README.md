# Trace Time Window

## Overview

This example creates a workload with mixed execution patterns to demonstrate time-windowed trace analysis. Five outer functions (`outer_a` through `outer_e`) each call an `inner()` function that alternates between CPU-bound busy-wait loops and sleep-based waiting - every 5th call does a CPU spin instead of sleeping. This creates distinct timing signatures in the trace, making it useful for testing profiling tools' ability to capture and distinguish between CPU-active and idle time windows within the same call graph.

## Source Files

- `trace-time-window.cpp` - Defines `inner()` with conditional CPU-bound vs. sleep behavior, and five `outer_*()` wrapper functions (generated via macro) that each call `inner(500)` with a 500 ms duration. All functions are marked `noinline`.

## Prerequisites

- CMake 3.21+
- C++ compiler

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/trace-time-window
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/
cmake --build <build_dir> --target trace-time-window
```

**NOTE**: The target is built in Debug mode to preserve the full call chain.

## Running

```bash
# Default: 1 iteration (5 outer calls)
./trace-time-window

# Custom: 3 iterations (15 outer calls)
./trace-time-window 3
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Number of repeat iterations | 1 |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./trace-time-window 2
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace for timeline visualization |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_VERBOSE` | `2` | Verbose output for debugging trace windows |
| `ROCPROFSYS_LOG_LEVEL` | `trace` | Detailed log output |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_TRACE=true \
    -e ROCPROFSYS_VERBOSE=2 \
    -- ./trace-time-window 2
```
