# Scratch Memory

## Overview

This example stress-tests GPU scratch memory allocation by launching kernels with varying per-thread local memory requirements - small (8 bytes), medium (~700 bytes), and large (~16 KB). It exercises transitions between primary scratch slots and secondary/overflow (USO) allocations by filling primary slots with medium kernels and then launching large kernels that require additional scratch. The program also scales grid dimensions to trigger scratch memory pressure. This is useful for profiling scratch memory allocation events and understanding GPU resource management under memory contention.

## Source Files

- `scratch-memory.cpp` - Defines three kernels (`test_kern_small`, `test_kern_medium`, `test_kern_large`) with different local array sizes, plus test orchestration functions (`test_primary_then_uso`, `test_gridx`, `test_scratch`) that drive scratch memory allocation scenarios.

## Prerequisites

- CMake 3.21+
- HIP runtime and `hipcc` compiler
- HSA runtime library

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/scratch_memory -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target scratch-memory
```

## Running

```bash
./scratch-memory
```

No command-line arguments. The program iterates through all detected GPU agents and runs a fixed set of scratch memory test scenarios.

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./scratch-memory
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_api,hsa_api,kernel_dispatch,scratch_memory` | Trace HIP/HSA API and scratch memory events |
| `ROCPROFSYS_ROCM_EVENTS` | `SQ_WAVES` | Sample GPU wave occupancy |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_api,hsa_api,kernel_dispatch,scratch_memory \
    -e ROCPROFSYS_TRACE=true \
    -- ./scratch-memory
```
