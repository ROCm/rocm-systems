# Code Coverage

## Overview

This example demonstrates code coverage analysis by providing two functionally identical code paths - `run_real()` and `run_fake()` - that can be switched at runtime via the `CODE_COVERAGE_USE_FAKE` environment variable. Both functions perform multi-threaded Fibonacci computation with atomic result accumulation. By toggling which path executes, you can test profiler and coverage tool capabilities to detect exercised vs. unexercised code branches and measure instrumentation completeness across alternate execution paths.

## Source Files

- `code-coverage.cpp` - Defines `fib()` (recursive Fibonacci), `run_real()` and `run_fake()` (identical Fibonacci computation with atomic accumulation), and a function-pointer-based dispatch selected by environment variable.
- `code-coverage.py` - Python companion script for coverage analysis.

## Prerequisites

- CMake 3.21+
- C++ compiler with pthreads support

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/code-coverage
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/
cmake --build <build_dir> --target code-coverage
```

## Running

```bash
# Default: run_real path, fib(10), min(16, ncpus) threads, 5000 iterations
./code-coverage

# Use alternate code path
CODE_COVERAGE_USE_FAKE=1 ./code-coverage

# Custom: fib(12), 4 threads, 2000 iterations
./code-coverage 12 4 2000
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Fibonacci number to compute | 10 |
| 2 | Number of threads | min(16, hardware concurrency) |
| 3 | Number of iterations per thread | 5000 |

**Environment:**

| Variable | Description |
|----------|-------------|
| `CODE_COVERAGE_USE_FAKE` | If set, switches to `run_fake()` code path |

## Profiling with rocprofiler-systems

```bash
# Profile the default path
rocprof-sys-run -- ./code-coverage 10 4 2000

# Profile the alternate path
rocprof-sys-run -e CODE_COVERAGE_USE_FAKE=1 -- ./code-coverage 10 4 2000
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile showing exercised functions |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable sampling for coverage correlation |
