# SHMEM

## Overview

This directory contains two OpenSHMEM examples. `shmem_hello` demonstrates basic SHMEM initialization, symmetric memory allocation, and remote read operations between processing elements (PEs). `shmem_pingpong` implements a latency microbenchmark that measures round-trip message passing time between two PEs using various SHMEM communication strategies (polling, `shmem_wait_until`, `shmem_fence`, `shmem_quiet`). These examples are useful for profiling one-sided communication patterns in PGAS (Partitioned Global Address Space) programming models.

## Source Files

- `shmem_hello.c` - Initializes SHMEM, allocates symmetric memory with `shmem_malloc()`, performs `shmem_int_get()` from the next PE (circular), and prints the result.
- `shmem_pingpong.c` - Ping-pong latency benchmark with configurable message size, iteration count, and communication mode. Supports polling, `shmem_wait_until()`, `shmem_fence()`, `shmem_quiet()`, and global vs. heap data placement.

## Prerequisites

- CMake 3.21+
- OpenSHMEM implementation with `oshcc` compiler wrapper

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/shmem
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/
cmake --build <build_dir> --target shmem_hello shmem_pingpong
```

**Targets:**

| Target | Description |
|--------|-------------|
| `shmem_hello` | Basic SHMEM hello world |
| `shmem_pingpong` | Ping-pong latency benchmark |

## Running

```bash
# Hello world (any number of PEs)
oshrun -np 4 ./shmem_hello

# Ping-pong (requires exactly 2 PEs)
oshrun -np 2 ./shmem_pingpong
```

**shmem_pingpong flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Number of iterations | 10000 |
| `-s` | Message size in bytes | 4 |
| `-w` | Use `shmem_wait_until()` instead of polling | off |
| `-f` | Send data and flag separately with `shmem_fence()` | off |
| `-g` | Use global data instead of heap | off |
| `-q` | Call `shmem_quiet()` after each put | off |

## Profiling with rocprofiler-systems

```bash
oshrun -np 2 rocprof-sys-run -- ./shmem_pingpong -n 1000
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable statistical sampling |
| `ROCPROFSYS_SAMPLING_FREQ` | `50` | Sampling frequency (interrupts/sec) |
