# HIP Graph Bubbles

## Overview

This example builds a HIP graph that captures many sequential kernel launches, instantiates it once, and then executes the graph repeatedly in a loop, synchronizing with the stream once after the final iteration. The workload is useful for profiling HIP graph capture and replay, kernel dispatch from graphs, and the timing behavior of graphs when many kernels are bundled into a single graph launch. It mirrors the upstream rocprofiler-sdk `hip-graph-bubbles` example (`projects/rocprofiler-sdk/tests/bin/hip-graph-bubbles/hip-graph-bubbles.cpp`).

## Source Files

- `hip-graph-bubbles.cpp` - Captures `num_kernels` `simpleKernel` launches into a HIP graph, instantiates the graph, and runs `num_iterations` `hipGraphLaunch` calls with progress output.

## Prerequisites

- CMake 3.25+
- HIP runtime and `hipcc` compiler

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/hip-graph-bubbles -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target hip-graph-bubbles
```

## Running

```bash
# Default: 2000 kernels per graph, 200 graph executions
./hip-graph-bubbles

# Custom: 64 kernels per graph, 6 graph executions (matches CI test defaults)
./hip-graph-bubbles 64 6
```

**Arguments:**

| Position | Description | Default |
| ---------- | ------------- | --------- |
| 1 | Number of kernels captured into the graph (`num_kernels`) | 2000 |
| 2 | Number of times to launch the graph (`num_iterations`) | 200 |
| 3 | Per-kernel data array size (`array_size`, must be >= 256) | 256 |
| 4 | Iterations between progress log lines (`progress_interval`) | 50 |

Total kernel dispatches reported at exit equals `num_kernels * num_iterations`.

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./hip-graph-bubbles 64 6
```

### Recommended Configuration

| Variable | Value | Purpose |
| ---------- | ------- | --------- |
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch` | Trace HIP API and kernel launches |
| `ROCPROFSYS_ROCM_EVENTS` | `GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU` (Instinct) or `SQ_WAVES` (e.g. Navi) | Sample GPU hardware counters |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch \
    -e ROCPROFSYS_ROCM_EVENTS=GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU \
    -- ./hip-graph-bubbles 64 6
```
