# Causal Profiling

## Overview

This example implements a causal profiling workload that runs slow and fast functions in parallel threads to evaluate causal profiling accuracy. Two implementation strategies are provided: an RNG-based timing approach using the Mersenne Twister random number generator, and a CPU-bound timing approach using `CLOCK_THREAD_CPUTIME_ID`. The program measures the actual execution time ratio between threads and compares it against the expected ratio to validate that causal profiling correctly identifies optimization opportunities. Progress points (`CAUSAL_PROGRESS_NAMED`) mark iteration boundaries for the causal analysis.

## Source Files

- `causal.cpp` - Sets up the threading workload with synchronization barriers and measures execution ratios.
- `impl.cpp` - Implements `rng_slow_func()`/`rng_fast_func()` (RNG-based timing) and `cpu_slow_func()`/`cpu_fast_func()` (CPU clock-based timing), with `CAUSAL_PROGRESS_NAMED` annotations.
- `causal.hpp` - Declarations for slow/fast function variants.
- `impl.hpp` - Implementation template declarations and timing utilities.

## Prerequisites

- CMake 3.21+
- C++17 compiler
- rocprofiler-systems user library (`rocprofiler-systems::rocprofiler-systems-user-library`)

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/casual -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -DCMAKE_PREFIX_PATH=/opt/rocm <project_root>/examples/
cmake --build <build_dir> --target causal-both causal-rng causal-cpu
```

The build generates multiple variants of each executable:

**Targets:**

| Target | Description |
|--------|-------------|
| `causal-both` | Both RNG and CPU workloads |
| `causal-rng` | RNG-based workload only |
| `causal-cpu` | CPU-based workload only |
| `causal-*-rocprofsys` | Linked with rocprofiler-systems user library |
| `causal-*-coz` | Linked with COZ profiler (if available) |

## Running

```bash
# Default: 70% work ratio, 50 iterations
./causal-both

# Custom: 80% ratio, 20 iterations, seed 12345, slow value 1000000000
./causal-cpu 80 20 12345 1000000000
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Work fraction (percentage ratio fast/slow) | 70 |
| 2 | Number of iterations | 50 |
| 3 | Random seed | random |
| 4 | Slow value (CPU cycles/work units) | 200000000 |
| 5 | Sync points or fast value | 1 |

## Profiling with rocprofiler-systems

Causal profiling uses the dedicated `rocprof-sys-causal` command or the `--causal` flag:

```bash
# Function-level causal profiling
rocprof-sys-causal -n 2 -w 1 -d 3 -- ./causal-cpu-rocprofsys 70 10 432525 1000000000

# Line-level causal profiling
rocprof-sys-causal --mode line -- ./causal-cpu-rocprofsys 70 10 432525 1000000000
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_CAUSAL_RANDOM_SEED` | `1342342` | Fixed seed for reproducible causal experiments |
| `ROCPROFSYS_TIME_OUTPUT` | `OFF` | Disable timestamped output directories |
| `ROCPROFSYS_FILE_OUTPUT` | `ON` | Enable file output |

**Causal CLI flags:**

| Flag | Description |
|------|-------------|
| `-n` | Number of causal experiment iterations |
| `-w` | Number of warmup iterations |
| `-d` | Virtual speedup delta (percentage steps) |
| `-b timer` | Use timer-based progress tracking |
| `-v 3` | Verbosity level |
