# Parallel Overhead

## Overview

This example measures the overhead of parallel synchronization by comparing two approaches to shared state access: mutex-based locking and lock-free atomic operations. Multiple threads compute Fibonacci numbers with randomized inputs and accumulate results through either a mutex-protected shared variable or an atomic counter. By comparing the two builds, you can quantify the profiling overhead introduced by different synchronization primitives. Thread synchronization uses a `pthread_barrier_t` to ensure all threads start simultaneously.

## Source Files

- `parallel-overhead.cpp` - Defines `fib()` (recursive Fibonacci, `noinline`), `run()` (per-thread worker with random input variance), and barrier-synchronized thread management. The `USE_LOCKS` compile-time macro switches between mutex and atomic modes.

## Prerequisites

- CMake 3.21+
- C++ compiler with pthreads support

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/parallel-overhead
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/
cmake --build <build_dir> --target parallel-overhead parallel-overhead-locks
```

| Target | Description |
|--------|-------------|
| `parallel-overhead` | Lock-free atomic accumulation |
| `parallel-overhead-locks` | Mutex-based accumulation (`USE_LOCKS=1`) |

## Running

```bash
# Default: fib(10), min(16, ncpus) threads, 50000 iterations
./parallel-overhead

# Custom: fib(12), 8 threads, 20000 iterations
./parallel-overhead 12 8 20000

# Mutex variant
./parallel-overhead-locks 12 8 20000
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Fibonacci number to compute | 10 |
| 2 | Number of threads | min(16, hardware concurrency) |
| 3 | Number of iterations per thread | 50000 |

## Profiling with rocprofiler-systems

Compare overhead between variants:

```bash
# Atomic version
rocprof-sys-run -- ./parallel-overhead 10 8 10000

# Mutex version
rocprof-sys-run -- ./parallel-overhead-locks 10 8 10000
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace for timeline analysis |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable statistical sampling |
| `ROCPROFSYS_SAMPLING_FREQ` | `50` | Sampling frequency (interrupts/sec) |
