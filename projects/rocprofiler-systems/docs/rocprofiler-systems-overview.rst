# ROCm Systems Profiler (rocprofiler-systems) - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Available Tools](#available-tools)
5. [Data Collection Modes](#data-collection-modes)
6. [Profiling Modes](#profiling-modes)
7. [Workload-Specific Presets](#workload-specific-presets)
8. [Command Reference](#command-reference)
9. [Configuration](#configuration)
10. [Output Formats](#output-formats)
11. [User API](#user-api)
12. [Parallelism Framework Support](#parallelism-framework-support)
13. [GPU and CPU Metrics](#gpu-and-cpu-metrics)
14. [Best Practices](#best-practices)
15. [Examples](#examples)
16. [Troubleshooting](#troubleshooting)

---

## Overview

ROCm Systems Profiler (rocprofiler-systems), formerly known as Omnitrace, is a comprehensive profiling and tracing tool for parallel applications written in **C, C++, Fortran, HIP, OpenCL, and Python** that execute on **CPU or CPU+GPU** systems.

### What It Does

- **Gathers performance information** through binary instrumentation, call-stack sampling, user-defined regions, and Python interpreter hooks
- **Supports interactive visualization** of comprehensive traces in web browsers via [Perfetto](https://ui.perfetto.dev)
- **Provides high-level summary profiles** with mean/min/max/stddev statistics
- **Collects system-level metrics** such as CPU frequency, GPU temperature, GPU utilization, memory usage, page-faults, and hardware counters

### Key Philosophy

ROCm Systems Profiler is designed to analyze the **entire application and its performance** rather than making assumptions about bottlenecks. It characterizes where optimization would have the greatest impact on end-to-end runtime.

> **Important**: When GPUs are involved, don't assume kernel optimization is the priority. If your application never waits for GPU kernels to complete, faster kernels won't improve overall performance. Use ROCm Systems Profiler to get the broad picture first.

---

## Key Features

### Data Collection Modes

| Mode | Description |
|------|-------------|
| **Dynamic Instrumentation** | Runtime instrumentation of executables and shared libraries, or binary rewriting to generate new instrumented binaries |
| **Statistical Sampling** | Periodic software interrupts per-thread for low-overhead profiling |
| **Process-level Sampling** | Background thread records process-, system-, and device-level metrics |
| **Causal Profiling** | Quantifies potential impact of optimizations in parallel code |

### Data Analysis Capabilities

- **High-level summary profiles** with mean, min, max, and standard deviation statistics (low overhead, memory efficient, ideal for scale)
- **Comprehensive traces** for every individual event/measurement
- **Application speedup predictions** from causal profiling

### Parallelism API Support

- HIP (GPU compute)
- HSA (Heterogeneous System Architecture)
- Pthreads
- MPI (Message Passing Interface)
- Kokkos-Tools (KokkosP)
- OpenMP-Tools (OMPT)

---

## Installation

### Quick Setup

**Option 1: Source the setup script (Recommended)**

```bash
source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh
```

**Option 2: Load modulefile**

```bash
module use /opt/rocprofiler-systems/share/modulefiles
module load rocprofiler-systems
```

**Option 3: Manual PATH setup**

```bash
export PATH=/opt/rocprofiler-systems/bin:${PATH}
export LD_LIBRARY_PATH=/opt/rocprofiler-systems/lib:${LD_LIBRARY_PATH}
```

### Verification

```bash
rocprof-sys-sample --version
```

### Building from Source

```bash
cmake -B build -S . \
    -DCMAKE_INSTALL_PREFIX=/opt/rocprofiler-systems \
    -DROCPROFSYS_USE_PYTHON=ON \
    -DROCPROFSYS_BUILD_DYNINST=ON \
    -DROCPROFSYS_BUILD_TBB=ON \
    -DROCPROFSYS_BUILD_BOOST=ON \
    -DROCPROFSYS_BUILD_ELFUTILS=ON \
    -DROCPROFSYS_BUILD_LIBIBERTY=ON

cmake --build build --target all --parallel 8
cmake --build build --target install
```

---

## Available Tools

ROCm Systems Profiler provides the following command-line tools:

### 1. `rocprof-sys-sample` - Call-Stack Sampling

Execute call-stack sampling on target applications **without binary instrumentation**.

```bash
rocprof-sys-sample [options] -- <exe> <exe-options>
```

**Features:**
- No instrumentation required
- Lowest overhead profiling method
- MPI compatible (works with `mpirun`)
- Supports workload-specific presets

**Common Options:**
| Option | Description |
|--------|-------------|
| `-f, --freq <N>` | Sampling frequency (interrupts per second) |
| `-T, --trace` | Generate detailed trace (Perfetto output) |
| `-P, --profile` | Generate call-stack-based profile |
| `-F, --flat-profile` | Generate flat profile (aggregated) |
| `-H, --host` | Enable host-based metrics (CPU frequency, memory, etc.) |
| `-D, --device` | Enable device-based metrics (GPU temperature, memory, etc.) |
| `-o, --output` | Output path and prefix |
| `--quick` | Fast profiling with sensible defaults |
| `--trace-hpc` | Optimized for HPC/MPI/OpenMP applications |
| `--trace-ai` | Optimized for AI/ML/GPU workloads |

**Examples:**

```bash
# Quick profiling
rocprof-sys-sample --quick -- ./myapp

# HIP application with GPU tracing
rocprof-sys-sample --quick --hip-trace -- ./hip_app

# HPC application with MPI
mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app

# AI/ML workload
rocprof-sys-sample --trace-ai -- python train.py

# Custom sampling frequency
rocprof-sys-sample -f 100 -- ./myapp
```

---

### 2. `rocprof-sys-instrument` - Binary Instrumentation

Instrument existing binaries for detailed tracing.

```bash
rocprof-sys-instrument [options] -- <exe-or-library> <exe-options>
```

**Modes:**

| Mode | Description | Command |
|------|-------------|---------|
| **Runtime Instrumentation** | Instruments at runtime (like `gdb --args`) | `rocprof-sys-instrument -- <exe>` |
| **Binary Rewrite** | Generates new instrumented binary | `rocprof-sys-instrument -o <output> -- <exe>` |
| **Attach to Process** | Connects to running process (alpha) | `rocprof-sys-instrument -p <PID> -- <exe-name>` |

**Key Options:**
| Option | Description |
|--------|-------------|
| `-o, --output` | Enable binary rewrite, output new executable |
| `-M, --mode` | Instrumentation mode: `trace`, `sampling`, `coverage` |
| `-I, --function-include` | Regex for including functions |
| `-E, --function-exclude` | Regex for excluding functions |
| `-R, --function-restrict` | Regex for restricting functions |
| `-MI, --module-include` | Regex for including modules/libraries |
| `-ME, --module-exclude` | Regex for excluding modules/libraries |
| `-i, --min-instructions` | Minimum instructions to instrument (default: 1024) |
| `--env` | Embed environment variable defaults |
| `--simulate` | Simulate without executing |

**Examples:**

```bash
# Binary rewrite
rocprof-sys-instrument -o app.inst -- ./app
rocprof-sys-run -- ./app.inst

# Runtime instrumentation (quick testing)
rocprof-sys-instrument -- ./app

# Instrument specific functions
rocprof-sys-instrument -R '^compute_' -o app.inst -- ./app

# Exclude libraries
rocprof-sys-instrument -ME '^(libhsa-runtime64|libz\\.so)' -- ./app

# Embed default settings
rocprof-sys-instrument -o app.inst --env ROCPROFSYS_USE_SAMPLING=ON -- ./app
```

---

### 3. `rocprof-sys-run` - Execute Instrumented Binaries

Execute instrumented binaries or rewritten applications.

```bash
rocprof-sys-run [options] -- <instrumented-exe> <args>
```

**Options:**
| Option | Description |
|--------|-------------|
| `--trace` | Enable tracing |
| `--profile` | Enable profiling |
| `--sample` | Enable sampling |
| `--hip-trace` | Enable HIP API/kernel tracing |
| `--rocm-events` | Specify GPU hardware counter events |
| `--perfetto-backend` | Backend: `inprocess` or `system` |
| `--trace-buffer-size` | Perfetto buffer size in KB |

**Examples:**

```bash
# Run with tracing
rocprof-sys-run --trace -- ./app.inst

# Run with HIP tracing and hardware counters
rocprof-sys-run --hip-trace --rocm-events=SQ_WAVES,SQ_INSTS_VALU -- ./app.inst

# Multi-process with system backend
rocprof-sys-run --perfetto-backend=system -- ./app.inst
```

---

### 4. `rocprof-sys-avail` - Configuration Query Tool

Query available settings, components, and hardware counters.

```bash
rocprof-sys-avail [options]
```

**Options:**
| Option | Description |
|--------|-------------|
| `-S, --settings` | Display runtime settings |
| `-C, --components` | Display components data |
| `-H, --hw-counters` | Display hardware counters |
| `-a, --all` | Print all available info |
| `-G, --generate-config` | Generate configuration file |
| `-d, --description` | Display descriptions |
| `--list-categories` | List available categories |

**Examples:**

```bash
# Generate configuration file
rocprof-sys-avail -G rocprof-sys.cfg

# Generate verbose config with descriptions
rocprof-sys-avail -G rocprof-sys.cfg --all

# List hardware counters
rocprof-sys-avail -H

# List settings with descriptions
rocprof-sys-avail -S -d

# Filter settings
rocprof-sys-avail -S -r "SAMPLING"
```

---

### 5. `rocprof-sys-causal` - Causal Profiling

Quantify potential optimization impact by determining: *"If you speed up a block of code by X%, the application runs Y% faster."*

```bash
rocprof-sys-causal [options] -- <exe> <args>
```

**Options:**
| Option | Description |
|--------|-------------|
| `-n, --iterations` | Number of iterations |
| `-m, --mode` | Mode: `function` or `line` |
| `-s, --speedups` | Pool of virtual speedups to sample |
| `-e, --end-to-end` | Single experiment for entire runtime |
| `-F, --function-scope` | Restrict to matching functions |
| `-S, --source-scope` | Restrict to matching source files |
| `-B, --binary-scope` | Restrict to matching binaries |
| `-o, --output-name` | Output filename |

**Examples:**

```bash
# Run 5 iterations in function mode
rocprof-sys-causal -n 5 -m function -- ./app

# Target specific functions
rocprof-sys-causal -n 10 -F "cpu_slow_func" "cpu_fast_func" -- ./app

# Line-level analysis
rocprof-sys-causal -n 10 -m line -S "main\\.cpp:100" -- ./app

# With MPI launcher
rocprof-sys-causal -l foo -n 3 -- mpirun -n 4 foo
```

---

### 6. `rocprof-sys-python` - Python Profiling

Profile Python scripts with interpreter function call tracking.

```bash
rocprof-sys-python [options] -- <script.py> <script-args>
```

**Features:**
- Profiles Python interpreter function calls
- Supports `@profile` and `@noprofile` decorators for selective profiling
- Version-specific interpreters: `rocprof-sys-python-3.8`, `rocprof-sys-python-3.10`, etc.

**Examples:**

```bash
# Basic Python profiling
rocprof-sys-python -- ./script.py

# With built-in decorator support
rocprof-sys-python -b -- ./script.py

# Specific Python version
rocprof-sys-python-3.10 -- ./script.py

# Custom Python interpreter
PYTHON_EXECUTABLE=/opt/conda/bin/python rocprof-sys-python -- ./script.py
```

**Python Decorator Usage:**

```python
@profile
def important_function():
    # This function will be profiled
    pass

@noprofile
def excluded_function():
    # This function and its callees are excluded
    pass
```

---

## Data Collection Modes

### Binary Instrumentation

Records **deterministic measurements for every invocation** of instrumented functions.

**Advantages:**
- Complete call tree reconstruction
- Precise timing for every call
- Useful for anomaly detection (min/max vs average)

**Disadvantages:**
- Higher overhead for small functions
- Instrumentation adds ~1024 instructions per function entry/exit

**Best For:** Functions with significant computation (≥1024 instructions)

### Statistical Sampling

Periodically interrupts application at regular intervals using OS interrupts.

**Advantages:**
- Low overhead
- Nearly full-speed execution
- Immune to over-evaluating small, frequently-called functions
- Better representative picture of actual execution

**Disadvantages:**
- Statistical approximation, not exact data
- May miss short-lived functions

**Best For:** Initial profiling, production environments, MPI applications

### Comparison Example

For a recursive Fibonacci function:
- **Instrumentation**: Records every call (millions of entries for `fib(30)`)
- **Sampling**: Captures statistical distribution without massive overhead

---

## Profiling Modes

### Trace Mode (Default)

Generates comprehensive, deterministic traces of every event.

```bash
export ROCPROFSYS_TRACE=true
# or
export ROCPROFSYS_MODE=trace
```

**Configuration Options:**
| Variable | Description |
|----------|-------------|
| `ROCPROFSYS_TRACE_DELAY` | Delay before starting trace (seconds) |
| `ROCPROFSYS_TRACE_DURATION` | Duration of trace collection (seconds) |
| `ROCPROFSYS_TRACE_PERIODS` | Multiple delay/duration periods |

### Profile Mode

Generates high-level summary profiles with statistical aggregations.

```bash
export ROCPROFSYS_PROFILE=true
```

**Profile Types:**
- **Flat Profile** (`--flat-profile`): Aggregated metrics per function
- **Hierarchical Profile** (`--profile`): Metrics by call-stack context

### Sampling Mode

Statistical call-stack sampling through periodic software interrupts.

**Sampling Types:**

1. **CPU-Time Sampling** (default)
   ```bash
   export ROCPROFSYS_SAMPLING_CPUTIME=ON
   export ROCPROFSYS_SAMPLING_CPUTIME_FREQ=100
   ```

2. **Real-Time Sampling**
   ```bash
   export ROCPROFSYS_SAMPLING_REALTIME=ON
   export ROCPROFSYS_SAMPLING_REALTIME_FREQ=100
   ```

3. **Overflow Sampling** (requires Linux perf)
   ```bash
   export ROCPROFSYS_SAMPLING_OVERFLOW=ON
   export ROCPROFSYS_SAMPLING_OVERFLOW_EVENT=PAPI_TOT_CYC
   ```

4. **Process Sampling** (background metrics)
   ```bash
   export ROCPROFSYS_USE_PROCESS_SAMPLING=ON
   export ROCPROFSYS_PROCESS_SAMPLING_FREQ=10
   ```

### Causal Mode

Predicts optimization impact by virtual speedup experiments.

```bash
export ROCPROFSYS_USE_CAUSAL=true
# or
export ROCPROFSYS_MODE=causal
```

### Coverage Mode

Tracks code execution for coverage analysis.

```bash
rocprof-sys-instrument -M coverage -o app.inst -- ./app
```

**Granularity:**
- `--coverage=function`: Function-level coverage
- `--coverage=basic_block`: Basic block-level coverage

---

## Workload-Specific Presets

### `--quick` - General Profiling

Fast profiling with sensible defaults for immediate insights.

```bash
rocprof-sys-sample --quick -- ./app
```

**Overhead:** Very Low (2-5%)

### `--simple` - Flat Profile Only

Minimal overhead flat profile.

```bash
rocprof-sys-sample --simple -- ./app
```

### `--detailed` - Full Trace with Hardware Counters

Comprehensive trace with all available metrics.

```bash
rocprof-sys-sample --detailed -- ./app
```

### `--trace-hpc` - HPC Applications

Optimized for MPI, OpenMP, and compute-intensive workloads.

```bash
mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app
```

**Configures:**
- OpenMP (OMPT) and Kokkos instrumentation
- MPI profiling (MPIP)
- CPU hardware counters: Instructions, Cycles, L3 Cache Misses
- GPU metrics: Utilization, Temperature, Power, Memory Usage
- ROCm API tracing

### `--trace-ai` - AI/ML Workloads

Optimized for PyTorch, TensorFlow, JAX and GPU-accelerated ML.

```bash
rocprof-sys-sample --trace-ai -- python train.py
```

**Configures:**
- GPU-focused monitoring (no CPU sampling overhead)
- HIP API calls, kernel dispatch, memory operations
- RCCL for distributed training
- ROCPD for kernel analysis
- 2GB Perfetto buffer for long traces

---

## Command Reference

### Environment Variables

#### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFSYS_TRACE` | Enable tracing | `true` |
| `ROCPROFSYS_PROFILE` | Enable profiling | `false` |
| `ROCPROFSYS_USE_SAMPLING` | Enable call-stack sampling | `false` |
| `ROCPROFSYS_USE_PROCESS_SAMPLING` | Enable process-level sampling | `true` |
| `ROCPROFSYS_MODE` | Active mode: `trace`, `sampling`, `causal`, `coverage` | `trace` |

#### Output Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFSYS_OUTPUT_PATH` | Output directory | `rocprof-sys-output` |
| `ROCPROFSYS_OUTPUT_PREFIX` | Output prefix | `%tag%` |
| `ROCPROFSYS_TIME_OUTPUT` | Timestamp subdirectories | `true` |
| `ROCPROFSYS_CONFIG_FILE` | Configuration file path | `~/.rocprof-sys.cfg` |

#### Sampling Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFSYS_SAMPLING_FREQ` | Sampling frequency (Hz) | `50` |
| `ROCPROFSYS_SAMPLING_CPUS` | CPUs to sample | `all` |
| `ROCPROFSYS_SAMPLING_GPUS` | GPUs to sample | `$env:HIP_VISIBLE_DEVICES` |
| `ROCPROFSYS_SAMPLING_CPUTIME` | CPU-time sampling | `true` |
| `ROCPROFSYS_SAMPLING_REALTIME` | Real-time sampling | `false` |

#### Tracing Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB` | Perfetto buffer size | `1024000` (1GB) |
| `ROCPROFSYS_TRACE_DELAY` | Delay before tracing (seconds) | `0` |
| `ROCPROFSYS_TRACE_DURATION` | Trace duration (seconds) | `0` (unlimited) |

#### Backend Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFSYS_USE_ROCM` | Enable ROCm support | `ON` |
| `ROCPROFSYS_USE_OMPT` | Enable OpenMP tools | `ON` |
| `ROCPROFSYS_USE_MPIP` | Enable MPI profiling | `OFF` |
| `ROCPROFSYS_USE_KOKKOSP` | Enable Kokkos tools | `OFF` |
| `ROCPROFSYS_USE_RCCLP` | Enable RCCL profiling | `OFF` |
| `ROCPROFSYS_USE_AMD_SMI` | Enable AMD SMI metrics | `ON` |

#### Hardware Counter Settings

| Variable | Description |
|----------|-------------|
| `ROCPROFSYS_PAPI_EVENTS` | CPU hardware counters (comma-separated) |
| `ROCPROFSYS_ROCM_EVENTS` | GPU hardware counters (comma-separated) |
| `ROCPROFSYS_AMD_SMI_METRICS` | AMD SMI metrics: `busy`, `temp`, `power`, `mem_usage` |

---

## Configuration

### Configuration File

Generate a configuration file:

```bash
rocprof-sys-avail -G rocprof-sys.cfg --all
```

**Example Configuration (`~/.rocprof-sys.cfg`):**

```ini
# Core settings
ROCPROFSYS_TRACE                = true
ROCPROFSYS_PROFILE              = true
ROCPROFSYS_USE_SAMPLING         = true
ROCPROFSYS_USE_PROCESS_SAMPLING = true

# Sampling configuration
ROCPROFSYS_SAMPLING_FREQ        = 50
ROCPROFSYS_SAMPLING_CPUS        = all
ROCPROFSYS_SAMPLING_GPUS        = $env:HIP_VISIBLE_DEVICES

# Output configuration
ROCPROFSYS_OUTPUT_PATH          = ./rocprof-sys-output
ROCPROFSYS_TIME_OUTPUT          = true

# Perfetto trace configuration
ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB = 1024000

# Backend settings
ROCPROFSYS_USE_ROCM             = true
ROCPROFSYS_USE_OMPT             = true
```

### Configuration Precedence

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Configuration file** (`ROCPROFSYS_CONFIG_FILE`)
4. **Default configuration** (`~/.rocprof-sys.cfg`)

---

## Output Formats

### Perfetto Traces (`.proto`)

Comprehensive visual traces viewable at [ui.perfetto.dev](https://ui.perfetto.dev).

**Features:**
- Timeline of function execution
- Thread activity and parallelism
- GPU kernel execution
- System metrics over time
- Zoom, search, and filtering

### Text Profiles (`wall_clock.txt`)

Human-readable hierarchical profiles.

**Columns:**
| Column | Description |
|--------|-------------|
| LABEL | Function/region name with call hierarchy |
| COUNT | Number of invocations |
| DEPTH | Call stack depth |
| SUM | Total time spent |
| MEAN | Average time per call |
| MIN/MAX | Minimum/maximum time per call |
| VAR/STDDEV | Variance and standard deviation |
| % SELF | Percentage of time in self (excluding callees) |

### JSON Profiles (`wall_clock.json`)

Machine-readable profiles compatible with [Hatchet](https://github.com/hatchet/hatchet).

### SQLite Database (`rocpd-*.db`)

ROCm Performance Data database for structured queries.

### Causal Output (`.coz`, `.json`)

Causal profiling results viewable at [plasma-umass.org/coz](https://plasma-umass.org/coz/).

---

## User API

Include the ROCm Systems Profiler API for custom instrumentation:

```cpp
#include <rocprofiler-systems/user.h>
#include <rocprofiler-systems/types.h>
#include <rocprofiler-systems/categories.h>
```

### Region Marking

```cpp
// Push/pop named regions
rocprofsys_user_push_region("my_computation");
// ... code to profile ...
rocprofsys_user_pop_region("my_computation");

// Annotated regions with metadata
rocprofsys_annotation_t annotations[] = {
    { "iteration", ROCPROFSYS_INT32, &iter },
    { "size", ROCPROFSYS_INT64, &size }
};
rocprofsys_user_push_annotated_region("loop", annotations, 2);
```

### Trace Control

```cpp
// Control tracing for current thread
rocprofsys_user_start_thread_trace();
rocprofsys_user_stop_thread_trace();

// Control tracing globally
rocprofsys_user_start_trace();
rocprofsys_user_stop_trace();
```

### Causal Progress Points

```cpp
// Mark progress for causal profiling
ROCPROFSYS_CAUSAL_PROGRESS;
```

### CMake Integration

```cmake
find_package(rocprofiler-systems REQUIRED COMPONENTS user)
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE rocprofiler-systems::rocprofiler-systems-user-library)
```

### Manual Compilation

```bash
g++ -g -I/opt/rocprofsys/include -L/opt/rocprofsys/lib foo.cpp -o foo -lrocprof-sys-user
```

---

## Parallelism Framework Support

### HIP/ROCm

```bash
# Enable HIP tracing
rocprof-sys-sample --hip-trace -- ./hip_app

# With GPU hardware counters
export ROCPROFSYS_ROCM_EVENTS="SQ_WAVES,SQ_INSTS_VALU"
rocprof-sys-run --hip-trace -- ./app.inst
```

### MPI

```bash
# Sampling (recommended)
mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app

# Instrumented (requires binary rewrite)
rocprof-sys-instrument -o mpi_app.inst -- ./mpi_app
mpirun -n 4 rocprof-sys-run -- ./mpi_app.inst
```

**Enable MPI Testing:**
```bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

### OpenMP

```bash
# Enable OpenMP tools
export ROCPROFSYS_USE_OMPT=ON
rocprof-sys-sample -- ./openmp_app
```

### Kokkos

```bash
# Enable Kokkos tools
export ROCPROFSYS_USE_KOKKOSP=ON
export KOKKOS_TOOLS_LIBS=/opt/rocprofiler-systems/lib/librocprof-sys.so
rocprof-sys-sample -- ./kokkos_app
```

### RCCL (Collective Communications)

```bash
# Enable RCCL profiling
export ROCPROFSYS_USE_RCCLP=ON
rocprof-sys-sample --trace-ai -- python distributed_training.py
```

---

## GPU and CPU Metrics

### GPU Metrics (AMD SMI)

| Metric | Description |
|--------|-------------|
| `busy` | GPU utilization percentage |
| `temp` | GPU temperature |
| `power` | Power consumption |
| `mem_usage` | Memory usage |
| VCN | Video Core Next utilization |
| JPEG | JPEG engine utilization |
| XGMI | Interconnect metrics (link width, speed, read/write) |
| PCIe | PCIe metrics (link width, speed, bandwidth) |

```bash
export ROCPROFSYS_AMD_SMI_METRICS="busy,temp,power,mem_usage"
export ROCPROFSYS_SAMPLING_GPUS=0,1
```

### GPU Hardware Counters

```bash
# List available GPU counters
rocprof-sys-avail -H -c GPU

# Use specific counters
export ROCPROFSYS_ROCM_EVENTS="SQ_WAVES,SQ_INSTS_VALU,TCC_HIT,TCC_MISS"
```

### CPU Metrics

| Category | Metrics |
|----------|---------|
| **Timing** | Wall time, CPU time, user/kernel time, CPU utilization |
| **Memory** | Peak RSS, page allocation, virtual memory usage |
| **System** | Network statistics, I/O metrics |
| **Hardware Counters** | Instructions, cycles, cache misses, etc. |

```bash
# CPU hardware counters
export ROCPROFSYS_PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_L3_TCM"
```

---

## Best Practices

### 1. Start with Sampling

Begin with low-overhead sampling before instrumentation:

```bash
rocprof-sys-sample -F -- ./app
head -30 rocprof-sys-output/sampling_wall_clock.txt
```

### 2. Use Flat Profiles for Hotspot Identification

```bash
rocprof-sys-sample -F -- ./app
# Look for high "% of total" functions
```

### 3. Profile Representative Workloads

Ensure your profiling run represents actual usage patterns.

### 4. Iterative Refinement

1. **Identify hotspots** with sampling
2. **Detailed trace** of hotspot functions
3. **Hardware counter analysis** for specific issues
4. **Verify improvements** with re-profiling

### 5. Control Overhead

```bash
# Lower sampling frequency for less overhead
rocprof-sys-sample -f 20 -- ./app

# Profile only specific functions
rocprof-sys-instrument -R '^critical_' -o app.inst -- ./app
```

### 6. Understand the Output

**Look for:**
- High `% of total` - Focus optimization here
- High `COUNT` with low individual time - Consider inlining
- High `STDDEV` - Investigate inconsistent performance
- Gaps in traces - Idle time (optimization opportunity)

---

## Examples

### Basic Workflow

```bash
# Step 1: Quick sampling to find hotspots
rocprof-sys-sample --quick -- ./app

# Step 2: View results
cat rocprof-sys-output/wall_clock.txt

# Step 3: Visualize trace
# Open rocprof-sys-output/perfetto-trace.proto at ui.perfetto.dev
```

### HIP Application Profiling

```bash
# Sample with GPU tracing
rocprof-sys-sample --quick --hip-trace -- ./hip_app

# Detailed instrumentation
rocprof-sys-instrument -o hip_app.inst -- ./hip_app
rocprof-sys-run --trace --hip-trace --rocm-events=SQ_WAVES -- ./hip_app.inst
```

### MPI Application Profiling

```bash
# Sampling (easiest)
mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app

# Instrumentation
rocprof-sys-instrument -o mpi_app.inst -- ./mpi_app
mpirun -n 4 rocprof-sys-run -- ./mpi_app.inst
```

### Python Application Profiling

```bash
# Basic profiling
rocprof-sys-python -- ./train.py

# AI/ML optimized
rocprof-sys-sample --trace-ai -- python train.py
```

### Causal Profiling Workflow

```bash
# Function-level analysis
rocprof-sys-causal -n 10 -m function -o results -- ./app

# Line-level for specific function
rocprof-sys-causal -n 10 -m line -F "hot_function" -o results -- ./app

# View results at plasma-umass.org/coz
```

---

## Troubleshooting

### No Output Generated

1. **Verify installation:**
   ```bash
   which rocprof-sys-sample
   rocprof-sys-sample --version
   ```

2. **Check environment:**
   ```bash
   echo $LD_LIBRARY_PATH | grep rocprof
   ```

3. **Verify permissions for hardware counters:**
   ```bash
   cat /proc/sys/kernel/perf_event_paranoid  # Should be <= 2
   ```

### Instrumentation Hangs on Large Binaries

Use sampling instead:
```bash
rocprof-sys-sample -- ./large_app
```

Or limit instrumentation scope:
```bash
rocprof-sys-instrument -R '^main$' -o app.inst -- ./app
```

### MPI Compatibility Issues

Use binary rewrite instead of runtime instrumentation:
```bash
# Don't do this:
# mpirun -n 4 rocprof-sys-instrument -- ./mpi_app

# Do this:
rocprof-sys-instrument -o mpi_app.inst -- ./mpi_app
mpirun -n 4 rocprof-sys-run -- ./mpi_app.inst
```

### Library Resolution Issues (RPATH)

Check for embedded RPATH:
```bash
objdump -p <exe> | egrep 'RPATH|RUNPATH'
```

Modify RPATH if needed:
```bash
patchelf --remove-rpath <exe>
patchelf --set-rpath '/path/to/instrumented/libs' <exe>
```

### Perfetto Visualization Issues

For ROCm versions prior to 6.3.1, use [Perfetto UI v46.0](https://ui.perfetto.dev/v46.0-35b3d9845/#!/).

### Application Behavior Changes When Profiled

Reduce overhead:
```bash
# Lower sampling frequency
rocprof-sys-sample -f 10 -- ./app

# Use sampling instead of instrumentation
rocprof-sys-sample -- ./app
```

---

## Additional Resources

- **Full Documentation**: [rocm.docs.amd.com/projects/rocprofiler-systems](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/)
- **GitHub Repository**: [github.com/ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)
- **Perfetto Visualization**: [ui.perfetto.dev](https://ui.perfetto.dev)
- **Causal Profiling Visualization**: [plasma-umass.org/coz](https://plasma-umass.org/coz/)
- **Hatchet Analysis**: [github.com/hatchet/hatchet](https://github.com/hatchet/hatchet)

---

## License

ROCm Systems Profiler is released under the MIT License.

Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
