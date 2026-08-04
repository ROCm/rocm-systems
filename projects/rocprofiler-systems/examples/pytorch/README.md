# PyTorch

## Overview

This example trains a small multi-layer perceptron on the GPU to reproduce `sin(x)/x`, scoring it on points it never trained on so that a run which merely memorizes its training data fails. Every operation is a general matrix multiply (GEMM), so the GPU work goes through rocBLAS.

## Source Files

- `pytorch_function_fit.py` - Builds the network, trains it, prints an ASCII plot of the target against the fit, and reports a pass or fail verdict. The target function, activation and domain are module-level constants intended to be edited.

## Prerequisites

- Python 3
- An AMD GPU and a ROCm PyTorch wheel; see [Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

## Building

There is nothing to compile. CMake copies the script to the top level of the build directory, `<build_dir>/pytorch_function_fit.py`.

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/pytorch
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
```

Installing places it with every other example, under `share/rocprofiler-systems/examples/`:

```bash
cmake --install <build_dir>
```

## Running

The script runs untraced under any interpreter that has torch installed:

```bash
./pytorch_function_fit.py
```

**Common arguments:**

| Flag | Description | Default |
| ------ | ------------- | --------- |
| `-e, --epochs` | Passes over the training set | 200 |
| `-r, --report-every` | Print the training loss every N epochs | 20 |

It prints an ASCII plot of the target against the fit, then a single machine-readable `RESULT: status=pass|fail ...` line, and exits non-zero when the fit misses the threshold.

## Profiling with rocprofiler-systems

```bash
rocprof-sys-python -- ./pytorch_function_fit.py
```

### Recommended Configuration

| Variable | Value | Purpose |
| ---------- | ------- | --------- |
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation` | Trace HIP API, GPU kernels, copies and allocations |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace with Python call stacks |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |

```bash
ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation \
    rocprof-sys-python -- ./pytorch_function_fit.py
```

### Narrowing the Python side

The commands above trace torch's own Python frames alongside the example's. To reduce the trace, filter the Python side rather than dropping instrumentation wholesale:

- `-E '^_call_impl$'` drops the largest single contributor, `nn.Module.__call__`, which fires once per layer per batch and accounts for roughly 28,800 entries in a default run.
- `-MR pytorch_function_fit.py` restricts instrumentation to the example's own module, removing torch's Python frames entirely.

Neither affects the GPU side. Kernel dispatches, HIP API calls and memory copies come from `ROCPROFSYS_ROCM_DOMAINS` and are identical with or without the Python filters.
