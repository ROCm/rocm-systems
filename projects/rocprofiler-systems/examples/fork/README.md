# Fork

## Overview

This example demonstrates profiling across process boundaries by forking child processes from worker threads. Each worker thread creates a child process via `fork()`, and each child spawns its own thread that sleeps for a variable duration before exiting. The parent process waits for all children using `waitpid()` with detailed exit status reporting. The entire workflow is annotated with rocprofiler-systems user API regions (`launch_child`, `child_process`, `child_process_child_thread`, `wait_for_children`) to show how profiling data is captured across `fork()` boundaries.

## Source Files

- `fork.cpp` - Implements barrier-synchronized thread-to-fork workflow with user API region annotations, child process management with `waitpid()`, and exit status decoding (`WIFEXITED`, `WIFSIGNALED`, etc.).
- `hipMallocConcurrencyMproc.cpp` - (Optional) HIP GPU memory allocation test across forked processes, built only when `hipcc` is available.

## Prerequisites

- CMake 3.21+
- C++ compiler with pthreads support
- rocprofiler-systems user library
- (Optional) HIP runtime for `hipMallocConcurrencyMproc`

## Building

**Standalone build:**

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build
```

**As part of the examples suite:**

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/opt/rocm examples/
cmake --build build --target fork-example
```

## Running

```bash
# Default: 4 child processes, 1 cycle
./fork-example

# Custom: 8 children, 3 cycles
./fork-example 8 3
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Number of child processes to fork | 4 |
| 2 | Number of fork/wait cycles to repeat | 1 |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./fork-example 4 2
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace with cross-process events |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Sample call stacks in parent and child processes |
| `ROCPROFSYS_USE_PROCESS_SAMPLING` | `ON` | Enable process-level resource sampling |
