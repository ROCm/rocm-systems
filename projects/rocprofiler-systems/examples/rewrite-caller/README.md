# Rewrite Caller

## Overview

This example provides a minimal function call chain - `main()` → `outer()` → `inner()` - designed as a test target for binary rewrite instrumentation. The functions are marked `noinline` to ensure they appear as distinct entries in the call graph after instrumentation. By counting the total calls through the chain, you can verify that binary rewriting correctly intercepts and records every function entry and exit. This is useful for validating instrumentation overhead and call graph accuracy.

## Source Files

- `rewrite-caller.cpp` - Defines `inner()` (increments a counter), `outer()` (calls `inner()`), and `main()` (calls `outer()` N times).

## Prerequisites

- CMake 3.21+
- C++ compiler

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/rewrite-caller
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/
cmake --build <build_dir> --target rewrite-caller
```

**NOTE**: The target is built in Debug mode to preserve the full call chain.

## Running

```bash
# Default: 10 calls
./rewrite-caller

# Custom: 1000 calls
./rewrite-caller 1000
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Number of calls to outer() | 10 |

## Profiling with rocprofiler-systems

This example is best used with binary rewrite instrumentation:

```bash
# Rewrite the binary
rocprof-sys-instrument -o rewrite-caller.inst -- ./rewrite-caller

# Run the instrumented binary
./rewrite-caller.inst 100
```

For runtime instrumentation:

```bash
rocprof-sys-instrument -- ./rewrite-caller 100
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace showing call chain |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile with call counts |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable sampling alongside instrumentation |
