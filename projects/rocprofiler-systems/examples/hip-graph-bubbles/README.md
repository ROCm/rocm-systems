# HIP Graph Bubbles

## Overview

This example builds a HIP graph that captures many sequential kernel launches, instantiates it once, and then executes the graph repeatedly in a loop. Each iteration is wrapped in a `roctxRangePush`/`roctxRangePop` pair named `graph_launch`, which produces distinct marker regions in traces and helps validate graph execution under rocprofiler-systems. The workload is useful for profiling HIP graph capture and replay, kernel dispatch from graphs, and marker API correlation when many kernels are bundled into a single graph launch.

## Source Files

- `hip-graph-bubbles.cpp` - Captures `NUM_KERNELS` `simpleKernel` launches into a HIP graph, instantiates the graph, and runs `NUM_ITERATIONS` `hipGraphLaunch` calls with rocTX ranges and optional progress output.

## Prerequisites

- CMake 3.25+
- HIP runtime and `hipcc` compiler
- rocprofiler-sdk-roctx library

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
| 1 | Number of kernels captured into the graph | 2000 |
| 2 | Number of times to launch the graph | 200 |

Total kernel dispatches reported at exit equals `num_kernels * num_iterations`.

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./hip-graph-bubbles 64 6
```

### Recommended Configuration

| Variable | Value | Purpose |
| ---------- | ------- | --------- |
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,marker_api` | Trace HIP API, kernel launches, and rocTX markers |
| `ROCPROFSYS_ROCM_EVENTS` | `GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU` (Instinct) or `SQ_WAVES` (e.g. Navi) | Sample GPU hardware counters |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,marker_api \
    -e ROCPROFSYS_ROCM_EVENTS=GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU \
    -- ./hip-graph-bubbles 64 6
```
