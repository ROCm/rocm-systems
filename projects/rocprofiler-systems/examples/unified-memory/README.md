# Unified Memory

## Overview

This example exercises HIP managed-memory (`hipMallocManaged`) access patterns
that intentionally trigger KFD page-fault and page-migration events. It runs a
sequence of workloads -- single-allocation host/device alternation, ping-pong
access, prefetch-driven migration, memory-pressure scenarios, and
concurrent-kernel access -- chosen to populate every direction bucket
(host-to-device, device-to-host) and every fault trigger (`gpu_page_fault`,
`cpu_page_fault`, `prefetch`) in the unified-memory report. It is the reference
workload for validating `ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING` in the
pytest/CTest validation suite.

## Source Files

- `unified-memory.cpp` - Implements 16 managed-memory test scenarios (basic
  faulting, ping-pong, prefetch, pressure, concurrent kernels, fine-grain
  access), CLI argument parsing, and a configurable subset runner (defaults to
  the first 6 tests, which is what the e2e validator expects).

## Prerequisites

- CMake 3.25+
- HIP runtime and `hipcc` compiler
- XNACK-capable AMD GPU (for example, MI200/MI300 or newer Instinct accelerators) with `HSA_XNACK=1`
- ROCProfiler-SDK 1.2.2 or later (only required when running under `rocprof-sys-run`)

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/unified-memory -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target unified-memory
```

## Running

```bash
# Default: 64 MB per allocation, 512 MB pressure pool, 8 ping-pong iterations, first 6 tests
./unified-memory

# Smaller workload matching the e2e validation path
./unified-memory -s 32 -p 256 -i 4

# Run the full set of 16 tests
./unified-memory -a
```

**Flags:**

| Flag | Description | Default |
| ------ | ------------- | --------- |
| `-s, --size <MB>` | Per-allocation size in MB | 64 |
| `-p, --pressure <MB>` | Managed memory for the pressure test | 512 |
| `-d, --device <ID>` | GPU device ID | 0 |
| `-i, --iterations <N>` | Ping-pong iterations | 8 |
| `-a, --all` | Run all 16 tests (default: first 6 only) | off |

Running on a host without XNACK support, or without `HSA_XNACK=1`, will not
crash, but no KFD page-fault / page-migration events will be captured under
`rocprof-sys-run`, so the unified-memory report will be empty.

## Profiling with rocprofiler-systems

```bash
HSA_XNACK=1 \
ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON \
rocprof-sys-run -- ./unified-memory -s 32 -p 256 -i 4
```

Output files (written to the standard rocprofiler-systems output directory):

| File | Contents |
| ------ | ---------- |
| `unified_memory.txt` | Human-readable per-GPU summary: total page faults, trigger breakdown (`gpu_page_fault`, `cpu_page_fault`, `prefetch`), and Host-to-Device / Device-to-Host migration bandwidth. |
| `unified_memory.json` | Machine-readable equivalent with the same fields, plus an `xnack_enabled` flag and an always-present `device_to_device` bucket for schema stability. |
| `perfetto-trace.proto` | Standard Perfetto trace, including KFD page-fault and migration tracks. |
| `rocpd.db` *(optional)* | Standard ROCpd database; KFD events are recorded in the `kfd_page_fault` / `kfd_page_migrate` tables. |

### Recommended Configuration

| Variable | Value | Purpose |
| ---------- | ------- | --------- |
| `HSA_XNACK` | `1` | Required runtime prerequisite for any KFD page-fault / migration events. |
| `ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING` | `ON` | Enable the aggregate `unified_memory.{txt,json}` reports and automatically opt-in to the required KFD tracing domains. |
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,kfd_events` | Capture HIP API calls and kernel dispatches alongside the KFD events. `kfd_events` is auto-enabled by the setting above; listing it explicitly is harmless. |
| `ROCPROFSYS_TRACE` | `true` | Generate the Perfetto trace alongside the unified-memory reports. |

```bash
HSA_XNACK=1 \
ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON \
ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,kfd_events \
ROCPROFSYS_TRACE=true \
rocprof-sys-run -- ./unified-memory -s 32 -p 256 -i 4
```

## See Also

- [`examples/README.md`](../README.md) -- the unified-memory entry in the top-level examples catalog (GPU Compute table).
- `docs/how-to/configuring-runtime-options.rst` -- the `ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING` reference entry.
- `tests/pytest/test_unified_memory.py` -- the pytest harness that drives unified-memory output validation against this example.
