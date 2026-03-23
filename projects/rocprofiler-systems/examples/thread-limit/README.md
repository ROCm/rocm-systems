# Thread Limit

## Overview

This example stress-tests the threading subsystem by creating a large number of threads (up to 8000 by default) that each perform recursive Fibonacci computations. Threads are launched in batches matching the hardware concurrency level, with each batch completing before the next starts. The program reports average execution time per thread and total throughput, making it useful for profiling thread creation/destruction overhead, scheduler behavior, and profiler scalability under high thread counts.

## Source Files

- `thread-limit.cpp` - Implements batched thread creation using `std::thread`, each computing `fib(n)` recursively. Uses `std::mutex` to accumulate per-thread timing into a global total. `MAX_THREADS` is configurable at compile time (default 4000).

## Prerequisites

- CMake 3.21+
- C++ compiler with pthreads support

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/thread-limit
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/
cmake --build <build_dir> --target thread-limit
```

## Running

```bash
# Default: fib(35), concurrency=ncpus, 8000 total threads
./thread-limit

# Custom: fib(30), 8 threads per batch, 2000 total
./thread-limit 30 8 2000
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Fibonacci number to compute | 35 |
| 2 | Concurrency (threads per batch) | hardware_concurrency() |
| 3 | Total number of threads to launch | 2 * MAX_THREADS (8000) |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./thread-limit 30 4 500
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace showing thread lifecycle |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable statistical sampling of thread stacks |
| `ROCPROFSYS_SAMPLING_FREQ` | `50` | Sampling frequency (interrupts/sec) |
