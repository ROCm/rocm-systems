# ROCm Systems Profiler (rocprofiler-systems) - Complete Tools Reference

This document provides comprehensive, detailed documentation of all available tools in the ROCm Systems Profiler suite, including their internal functions, command-line options, and usage patterns.

---

## Table of Contents

1. [Tool Overview](#tool-overview)
2. [rocprof-sys-sample](#rocprof-sys-sample)
3. [rocprof-sys-instrument](#rocprof-sys-instrument)
4. [rocprof-sys-run](#rocprof-sys-run)
5. [rocprof-sys-avail](#rocprof-sys-avail)
6. [rocprof-sys-causal](#rocprof-sys-causal)
7. [User API Reference](#user-api-reference)
8. [Configuration System](#configuration-system)
9. [Environment Variables Reference](#environment-variables-reference)

---

## Tool Overview

The ROCm Systems Profiler provides six primary command-line tools:

| Tool | Purpose | Primary Use Case |
|------|---------|------------------|
| `rocprof-sys-sample` | Statistical call-stack sampling | Low-overhead profiling without binary modification |
| `rocprof-sys-instrument` | Binary instrumentation (runtime/rewrite) | Detailed function-level tracing |
| `rocprof-sys-run` | Execute instrumented binaries | Running pre-instrumented applications |
| `rocprof-sys-avail` | Query available settings/counters | Configuration and capability discovery |
| `rocprof-sys-causal` | Causal profiling experiments | Optimization impact prediction |
| `rocprof-sys-python` | Python profiling | Python script performance analysis |

---

## rocprof-sys-sample

### Description

Call-stack sampling profiler for applications without binary instrumentation. Uses timer-based interrupts to periodically sample the call stack, providing statistical profiles with minimal overhead.

### Core Mechanism

The tool works by:
1. Setting up the environment with the profiler's dynamic library (`librocprof-sys-dl.so`)
2. Configuring sampling parameters via environment variables
3. Executing the target application via `execvpe()`
4. The preloaded library intercepts and samples at configured intervals

### Command-Line Synopsis

```bash
rocprof-sys-sample [OPTIONS] -- <command> [args...]
```

### Preset Modes

| Preset | Description | Key Settings |
|--------|-------------|--------------|
| `--quick` | Fast profiling with sensible defaults | Trace + Profile ON, Sampling @ 50Hz, Process sampling ON |
| `--simple` | Flat profile only, minimal overhead | Trace OFF, Profile ON, Flat profile ON, Sampling @ 100Hz |
| `--detailed` | Full trace, profile, hardware counters | Trace + Profile ON, Sampling @ 100Hz, Hardware counters enabled |
| `--trace-hpc` | Optimized for MPI/OpenMP/HPC workloads | MPI/OpenMP/Kokkos support, HW counters (PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L3_TCM) |
| `--trace-ai` | Optimized for PyTorch/TensorFlow/JAX | GPU tracing, RCCL support, Large buffer (2GB), HIP API tracing |

### General Options

| Option | Description |
|--------|-------------|
| `-c, --config <file>` | Load configuration from file |
| `-o, --output <path> [prefix]` | Set output path and optional prefix |
| `-T, --trace` | Generate Perfetto trace output |
| `-P, --profile` | Generate call-stack-based profile |
| `-F, --flat-profile` | Generate flat profile (no call hierarchy) |
| `-H, --host` | Enable host-based process sampling (CPU freq, memory) |
| `-D, --device` | Enable device-based process sampling (GPU temp, power) |
| `-w, --wait <seconds>` | Delay before starting data collection |
| `-d, --duration <seconds>` | Duration of data collection |

### Sampling Options

| Option | Description |
|--------|-------------|
| `-f, --freq <Hz>` | Sampling frequency (interrupts per second) |
| `--sampling-wait <seconds>` | Delay before first sample |
| `--sampling-duration <seconds>` | Duration of sampling |
| `-t, --tids <ids>` | Thread IDs to sample (0 = main thread) |
| `--cputime [freq] [delay] [tids...]` | CPU-clock timer sampling |
| `--realtime [freq] [delay] [tids...]` | Real-clock timer sampling |

#### CPU-Time vs Real-Time Sampling

- **CPU-Time Sampling**: Samples based on CPU clock time consumed by the thread. Idle threads don't trigger samples.
- **Real-Time Sampling**: Samples based on wall-clock time. Can cause overhead for idle threads.

### Tracing Options

| Option | Description |
|--------|-------------|
| `--trace-file <filename>` | Specify trace output filename |
| `--trace-buffer-size <KB>` | Size limit for trace output |
| `--trace-fill-policy` | `discard` (drop new data) or `ring_buffer` (overwrite oldest) |
| `--trace-wait <seconds>` | Delay before tracing |
| `--trace-duration <seconds>` | Duration of tracing |
| `--trace-periods` | Complex timing: `<DELAY>:<DURATION>[:<REPEAT>[:<CLOCK_ID>]]` |

### Backend Options

| Option | Description |
|--------|-------------|
| `-I, --include <backends>` | Include specific backends |
| `-E, --exclude <backends>` | Exclude specific backends |

Available backends: `all`, `kokkosp`, `mpip`, `ompt`, `rcclp`, `amd-smi`, `mutex-locks`, `spin-locks`, `rw-locks`, `rocm`

### Hardware Counter Options

| Option | Description |
|--------|-------------|
| `-C, --cpu-events <events>` | CPU hardware counter events (PAPI) |

### Clock ID Choices

| Value | Name | Description |
|-------|------|-------------|
| 0 | `CLOCK_REALTIME` | System-wide real-time clock |
| 1 | `CLOCK_MONOTONIC` | Monotonic clock |
| 2 | `CLOCK_PROCESS_CPUTIME_ID` | Process CPU time |
| 4 | `CLOCK_MONOTONIC_RAW` | Raw monotonic clock |
| 5 | `CLOCK_REALTIME_COARSE` | Coarse real-time clock |
| 6 | `CLOCK_MONOTONIC_COARSE` | Coarse monotonic clock |
| 7 | `CLOCK_BOOTTIME` | Boot time clock |

### Examples

```bash
# Quick start with default settings
rocprof-sys-sample --quick -- ./myapp

# High-frequency sampling with trace
rocprof-sys-sample -f 500 --trace -- ./myapp

# HPC workload with MPI
mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app

# AI/ML workload
rocprof-sys-sample --trace-ai -- python train.py

# Custom CPU-time sampling at 100Hz, starting after 2 seconds
rocprof-sys-sample --cputime 100 2.0 -- ./myapp

# Include specific backends
rocprof-sys-sample --quick -I ompt mpip -- ./hybrid_app
```

---

## rocprof-sys-instrument

### Description

Binary instrumentation tool using Dyninst for function-level tracing. Supports three modes of operation:

1. **Runtime Instrumentation**: Instruments at launch time
2. **Attach to Process**: Instruments a running process
3. **Binary Rewrite**: Creates a new instrumented binary

### Core Mechanism

The tool uses Dyninst to:
1. Parse the target binary's symbol table and debug information
2. Identify instrumentable functions
3. Insert tracing callbacks at function entry/exit points
4. Either execute immediately (runtime) or write a new binary (rewrite)

### Command-Line Synopsis

```bash
# Runtime instrumentation
rocprof-sys-instrument [OPTIONS] -- <command> [args...]

# Binary rewrite
rocprof-sys-instrument -o <output> [OPTIONS] -- <input>

# Attach to process
rocprof-sys-instrument -p <pid> [OPTIONS]
```

### Instrumentation Modes

| Mode | Flag | Description |
|------|------|-------------|
| `trace` | `-M trace` | Full function tracing (default) |
| `sampling` | `-M sampling` | Call-stack sampling only |
| `coverage` | `--coverage` | Code coverage collection |
| `causal` | `-M causal` | Causal profiling mode |

### Key Options

| Option | Description |
|--------|-------------|
| `-o, --output <file>` | Output file for binary rewrite |
| `-p, --pid <pid>` | Attach to running process |
| `-M, --mode <mode>` | Instrumentation mode |
| `-c, --config <file>` | Configuration file |
| `-l, --instrument-libraries` | Instrument linked libraries |
| `-s, --simulate` | Print instrumentation plan without executing |

### Function Selection

| Option | Description |
|--------|-------------|
| `-I, --function-include <regex>` | Include functions matching regex |
| `-E, --function-exclude <regex>` | Exclude functions matching regex |
| `--module-include <regex>` | Include modules (files) matching regex |
| `--module-exclude <regex>` | Exclude modules matching regex |
| `--min-instructions <N>` | Minimum instructions for instrumentation |
| `--min-address-range <N>` | Minimum address range for instrumentation |

### Advanced Options

| Option | Description |
|--------|-------------|
| `--allow-traps` | Allow instrumenting functions that use traps |
| `--allow-overlapping` | Allow instrumenting overlapping functions |
| `--dynamic-callsites` | Instrument dynamic call sites |
| `--trampoline-size <bytes>` | Size of trampoline code |
| `--loop-tracing` | Enable loop-level instrumentation |

### Binary Rewrite Workflow

```bash
# Step 1: Instrument the binary
rocprof-sys-instrument -o ./app.inst -- ./app

# Step 2: Run the instrumented binary
rocprof-sys-run --quick -- ./app.inst

# Or run directly
./app.inst
```

### Library Instrumentation

```bash
# Instrument a shared library
rocprof-sys-instrument -o ./instrumented/libfoo.so -- ./libfoo.so

# Run with instrumented library
LD_LIBRARY_PATH=./instrumented:$LD_LIBRARY_PATH ./app
```

### Examples

```bash
# Basic runtime instrumentation
rocprof-sys-instrument -- ./myapp arg1 arg2

# Binary rewrite
rocprof-sys-instrument -o ./myapp.inst -- ./myapp

# Instrument only specific functions
rocprof-sys-instrument -I 'compute.*' -E 'helper.*' -- ./myapp

# Sampling mode (less overhead)
rocprof-sys-instrument -M sampling -- ./myapp

# Simulate and show what would be instrumented
rocprof-sys-instrument -s -- ./myapp
```

---

## rocprof-sys-run

### Description

Executes pre-instrumented binaries with the ROCm Systems Profiler runtime configuration. Used to run binaries that were created via `rocprof-sys-instrument -o`.

### Core Mechanism

The tool:
1. Parses command-line arguments for configuration
2. Sets up the environment (LD_PRELOAD, LD_LIBRARY_PATH)
3. Optionally forks a child process
4. Executes the instrumented binary via `execvpe()`

### Command-Line Synopsis

```bash
rocprof-sys-run [OPTIONS] -- <instrumented-binary> [args...]
```

### Preset Modes

Same presets as `rocprof-sys-sample`:
- `--quick`: Fast profiling with defaults
- `--simple`: Flat profile, minimal overhead
- `--detailed`: Full trace and hardware counters
- `--trace-hpc`: MPI/OpenMP optimization
- `--trace-ai`: GPU/ML workload optimization

### Execution Options

| Option | Description |
|--------|-------------|
| `--fork` | Execute via fork + execvpe (for debugging) |
| `--launcher <regex>` | Regex to identify launcher command (for MPI, etc.) |

### Configuration Options

All configuration options from `rocprof-sys-sample` are available, including:
- Output path/prefix
- Trace settings
- Profile settings
- Hardware counters
- Backend selection

### Environment Setup

The tool automatically configures:
- `LD_PRELOAD`: Adds `librocprof-sys-dl.so`
- `LD_LIBRARY_PATH`: Adds profiler library path
- `ROCPROFSYS_*`: Various runtime settings

### Examples

```bash
# Run with quick preset
rocprof-sys-run --quick -- ./app.inst

# Custom output location
rocprof-sys-run -o /tmp/results -- ./app.inst

# With MPI launcher
mpirun -n 4 rocprof-sys-run --trace-hpc -- ./mpi_app.inst

# Verbose output
rocprof-sys-run -v 2 -- ./app.inst
```

---

## rocprof-sys-avail

### Description

Query and display available runtime settings, components, and hardware counters. Can also generate configuration files.

### Core Mechanism

The tool:
1. Initializes the timemory and ROCm subsystems
2. Queries available settings from the settings registry
3. Queries hardware counters from PAPI and ROCm
4. Displays or outputs the information in various formats

### Command-Line Synopsis

```bash
rocprof-sys-avail [OPTIONS] [REGEX_FILTER]
```

### Information Display Options

| Option | Description |
|--------|-------------|
| `-S, --settings` | Display runtime settings |
| `-C, --components` | Display available components |
| `-H, --hw-counters` | Display hardware counters |
| `-a, --all` | Display all information |
| `-d, --description` | Show descriptions |
| `-A, --available` | Only show available items |

### Column Options

| Option | Description |
|--------|-------------|
| `-b, --brief` | Suppress availability/value info |
| `-s, --string` | Show string identifiers |
| `-v, --value` | Show value types |
| `-f, --filename` | Show output filenames |
| `-c, --categories` | Show categories |

### Filter Options

| Option | Description |
|--------|-------------|
| `-r, --filter <regex>` | Filter output by regex |
| `-R, --category-filter <regex>` | Filter by category |
| `-i, --ignore-case` | Case-insensitive filtering |
| `--alphabetical` | Sort alphabetically |

### Output Options

| Option | Description |
|--------|-------------|
| `-G, --generate-config <file>` | Generate configuration file |
| `-F, --config-format <fmt>` | Format: `txt`, `json`, `xml` |
| `-O, --output <file>` | Write output to file |
| `-M, --markdown` | Output in markdown format |
| `--csv` | Output in CSV format |
| `--force` | Overwrite existing config file |

### Display Options

| Option | Description |
|--------|-------------|
| `-w, --column-width <N>` | Maximum column width |
| `-W, --max-total-width <N>` | Maximum total width |
| `--list-categories` | List available categories |
| `--list-keys` | List output keys |
| `--expand-keys` | Expand keys to current values |
| `--advanced` | Show advanced settings |

### Examples

```bash
# View all settings with descriptions
rocprof-sys-avail -S -d

# Generate default configuration file
rocprof-sys-avail -G ~/.rocprof-sys.cfg

# Generate verbose configuration with all info
rocprof-sys-avail -G config.cfg --all

# View available CPU hardware counters
rocprof-sys-avail -H -c CPU -A

# View GPU hardware counters
rocprof-sys-avail -H -c GPU

# Filter settings by regex
rocprof-sys-avail -S -r "SAMPLING"

# Generate JSON and XML configs
rocprof-sys-avail -G myconfig -F txt json xml

# View components with string identifiers
rocprof-sys-avail -C -s -b

# List categories
rocprof-sys-avail --list-categories
```

### Hardware Counter Categories

| Category | Description |
|----------|-------------|
| `CPU` | PAPI-based CPU hardware counters |
| `GPU` | ROCm-based GPU hardware counters |

---

## rocprof-sys-causal

### Description

Performs causal profiling experiments to predict the impact of optimizing specific code regions. Runs the target application multiple times with virtual speedups applied to selected regions.

### Core Mechanism

Causal profiling works by:
1. Identifying "progress points" that indicate throughput or latency
2. Running the application multiple times
3. Virtually "speeding up" selected code regions by slowing down all other threads
4. Measuring the effect on overall progress
5. Predicting the impact of actual optimizations

### Command-Line Synopsis

```bash
rocprof-sys-causal [OPTIONS] -- <command> [args...]
```

### Key Concepts

#### Progress Points

Progress points mark where "work" is completed:

| Type | Purpose | Example |
|------|---------|---------|
| **Throughput** | How much work completed | Loop iterations, requests processed |
| **Latency** | How long work takes | Function execution time |

#### Backends

| Backend | Description | Requirement |
|---------|-------------|-------------|
| `perf` | Uses Linux perf events | `perf_event_paranoid <= 2` |
| `timer` | Uses timer-based sampling | Always available |

### Options

| Option | Description |
|--------|-------------|
| `--mode <mode>` | `function` or `line` mode |
| `--backend <backend>` | `perf` or `timer` |
| `--scope <scope>` | Instrumentation scope |
| `--speedup <range>` | Virtual speedup percentages to test |
| `--end-to-end` | Run end-to-end experiments |

### Using Progress Points in Code

```c
#include <rocprofiler-systems/causal.h>

void process_batch(data_t* data, int count) {
    for (int i = 0; i < count; i++) {
        process_item(&data[i]);
        ROCPROFSYS_CAUSAL_PROGRESS  // Throughput point
    }
}

void compute_result(input_t* in, output_t* out) {
    ROCPROFSYS_CAUSAL_BEGIN("compute")
    // ... computation ...
    ROCPROFSYS_CAUSAL_END("compute")
}
```

### Examples

```bash
# Basic causal profiling
rocprof-sys-causal -- ./myapp

# With specific backend
rocprof-sys-causal --backend perf -- ./myapp

# Test specific speedup range
rocprof-sys-causal --speedup 0,10,20,30,40,50 -- ./myapp

# Function-level analysis
rocprof-sys-causal --mode function -- ./myapp
```

---

## User API Reference

### Overview

The User API allows applications to define custom regions for profiling and control tracing behavior programmatically.

### Header Files

| Header | Purpose |
|--------|---------|
| `rocprofiler-systems/user.h` | Main user API functions |
| `rocprofiler-systems/types.h` | Type definitions and callbacks |
| `rocprofiler-systems/categories.h` | Category and annotation definitions |
| `rocprofiler-systems/causal.h` | Causal profiling macros |

### Core Functions

#### Trace Control

```c
// Enable/disable tracing globally (affects all threads)
int rocprofsys_user_start_trace(void);
int rocprofsys_user_stop_trace(void);

// Enable/disable tracing for current thread only
int rocprofsys_user_start_thread_trace(void);
int rocprofsys_user_stop_thread_trace(void);
```

#### Region Marking

```c
// Start a named region
int rocprofsys_user_push_region(const char* name);

// End a named region (FILO order recommended)
int rocprofsys_user_pop_region(const char* name);
```

#### Annotated Regions

```c
// Region with annotations (visible in Perfetto trace)
int rocprofsys_user_push_annotated_region(
    const char* name,
    rocprofsys_annotation_t* annotations,
    size_t num_annotations
);

int rocprofsys_user_pop_annotated_region(
    const char* name,
    rocprofsys_annotation_t* annotations,
    size_t num_annotations
);
```

#### Progress Points (Causal Profiling)

```c
// Mark throughput progress
int rocprofsys_user_progress(const char* name);

// Progress with annotations
int rocprofsys_user_annotated_progress(
    const char* name,
    rocprofsys_annotation_t* annotations,
    size_t num_annotations
);
```

#### Callback Configuration

```c
// Configure callback functions
int rocprofsys_user_configure(
    rocprofsys_user_configure_mode_t mode,
    rocprofsys_user_callbacks_t inp,
    rocprofsys_user_callbacks_t* out
);

// Get current callbacks
int rocprofsys_user_get_callbacks(rocprofsys_user_callbacks_t*);

// Get error description
const char* rocprofsys_user_error_string(int error_category);
```

### Data Types

#### Annotation Structure

```c
typedef struct rocprofsys_annotation {
    const char* name;     // Annotation label
    uintptr_t type;       // Data type (ROCPROFSYS_VALUE_*)
    void* value;          // Pointer to data
} rocprofsys_annotation_t;
```

#### Annotation Types

| Type | Description |
|------|-------------|
| `ROCPROFSYS_VALUE_CSTR` | C string (const char*) |
| `ROCPROFSYS_VALUE_SIZE_T` | size_t |
| `ROCPROFSYS_VALUE_INT64` | int64_t |
| `ROCPROFSYS_VALUE_UINT64` | uint64_t |
| `ROCPROFSYS_VALUE_FLOAT64` | double |
| `ROCPROFSYS_VALUE_VOID_P` | void pointer |
| `ROCPROFSYS_VALUE_INT32` | int32_t |
| `ROCPROFSYS_VALUE_UINT32` | uint32_t |
| `ROCPROFSYS_VALUE_FLOAT32` | float |

#### Configure Modes

| Mode | Description |
|------|-------------|
| `ROCPROFSYS_USER_UNION_CONFIG` | Replace non-null callbacks only |
| `ROCPROFSYS_USER_REPLACE_CONFIG` | Replace entire configuration |
| `ROCPROFSYS_USER_INTERSECT_CONFIG` | Keep only matching callbacks |

#### Error Codes

| Code | Description |
|------|-------------|
| `ROCPROFSYS_USER_SUCCESS` | No error |
| `ROCPROFSYS_USER_ERROR_NO_BINDING` | Function pointer not assigned |
| `ROCPROFSYS_USER_ERROR_BAD_VALUE` | Invalid value provided |
| `ROCPROFSYS_USER_ERROR_INVALID_CATEGORY` | Invalid category |
| `ROCPROFSYS_USER_ERROR_INTERNAL` | Internal library error |

### Callback Structure

```c
typedef struct rocprofsys_user_callbacks {
    rocprofsys_trace_func_t start_trace;
    rocprofsys_trace_func_t stop_trace;
    rocprofsys_trace_func_t start_thread_trace;
    rocprofsys_trace_func_t stop_thread_trace;
    rocprofsys_region_func_t push_region;
    rocprofsys_region_func_t pop_region;
    rocprofsys_region_func_t progress;
    rocprofsys_annotated_region_func_t push_annotated_region;
    rocprofsys_annotated_region_func_t pop_annotated_region;
    rocprofsys_annotated_region_func_t annotated_progress;
} rocprofsys_user_callbacks_t;
```

### Categories

The profiler tracks data in various categories:

| Category | Description |
|----------|-------------|
| `ROCPROFSYS_CATEGORY_USER` | User-defined regions |
| `ROCPROFSYS_CATEGORY_HOST` | Host-side operations |
| `ROCPROFSYS_CATEGORY_ROCM` | General ROCm operations |
| `ROCPROFSYS_CATEGORY_ROCM_HIP_API` | HIP API calls |
| `ROCPROFSYS_CATEGORY_ROCM_HSA_API` | HSA API calls |
| `ROCPROFSYS_CATEGORY_ROCM_KERNEL_DISPATCH` | Kernel dispatches |
| `ROCPROFSYS_CATEGORY_ROCM_MEMORY_COPY` | Memory copies |
| `ROCPROFSYS_CATEGORY_SAMPLING` | Call-stack samples |
| `ROCPROFSYS_CATEGORY_PTHREAD` | Pthread operations |
| `ROCPROFSYS_CATEGORY_KOKKOS` | Kokkos regions |
| `ROCPROFSYS_CATEGORY_MPI` | MPI operations |
| `ROCPROFSYS_CATEGORY_CAUSAL` | Causal profiling |
| `ROCPROFSYS_CATEGORY_AMD_SMI` | AMD SMI metrics |

### Causal Profiling Macros

```c
#include <rocprofiler-systems/causal.h>

// Throughput progress point with auto-generated label
ROCPROFSYS_CAUSAL_PROGRESS

// Throughput progress point with custom label
ROCPROFSYS_CAUSAL_PROGRESS_NAMED("my_progress_point")

// Latency region
ROCPROFSYS_CAUSAL_BEGIN("region_name")
// ... code ...
ROCPROFSYS_CAUSAL_END("region_name")
```

### Example: Complete API Usage

```cpp
#include <rocprofiler-systems/user.h>

void compute(size_t n, double* data) {
    // Create annotations
    rocprofsys_annotation_t annotations[] = {
        { "size", ROCPROFSYS_VALUE_SIZE_T, &n },
        { "data_ptr", ROCPROFSYS_VALUE_VOID_P, data }
    };

    // Start annotated region
    rocprofsys_user_push_annotated_region(
        "compute", annotations, 2);

    // Disable tracing for this thread during hot loop
    rocprofsys_user_stop_thread_trace();

    for (size_t i = 0; i < n; i++) {
        // Hot computation
        data[i] = process(data[i]);
    }

    // Re-enable tracing
    rocprofsys_user_start_thread_trace();

    // End region
    rocprofsys_user_pop_annotated_region(
        "compute", annotations, 2);
}
```

### Linking the User Library

#### CMake

```cmake
find_package(rocprofiler-systems REQUIRED COMPONENTS user)
target_link_libraries(myapp PRIVATE rocprofiler-systems::user)
```

#### Manual Compilation

```bash
g++ -o myapp myapp.cpp \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lrocprofiler-systems-user
```

---

## Configuration System

### Configuration File Locations

Default search paths (in order):
1. `${HOME}/.rocprof-sys.cfg`
2. `${HOME}/.rocprof-sys.json`
3. Path specified in `ROCPROFSYS_CONFIG_FILE`

### Supported Formats

| Format | Extension | Features |
|--------|-----------|----------|
| Text | `.cfg` | Simple key=value, variables |
| JSON | `.json` | Full schema with metadata |
| XML | `.xml` | Full schema with metadata |

### Text Configuration Syntax

```bash
# Variables (start with $)
$ENABLE = ON
$SAMPLE_RATE = 100

# Settings
ROCPROFSYS_TRACE = $ENABLE
ROCPROFSYS_SAMPLING_FREQ = $SAMPLE_RATE

# Environment variable reference
ROCPROFSYS_SAMPLING_GPUS = $env:HIP_VISIBLE_DEVICES

# Export to environment (won't override existing)
$env:MY_CUSTOM_VAR = some_value
```

### Generating Configuration Files

```bash
# Default text format
rocprof-sys-avail -G ~/.rocprof-sys.cfg

# All formats
rocprof-sys-avail -G myconfig -F txt json xml

# With full descriptions
rocprof-sys-avail -G myconfig.cfg --all
```

---

## Environment Variables Reference

### Core Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ROCPROFSYS_TRACE` | bool | true | Enable Perfetto tracing |
| `ROCPROFSYS_PROFILE` | bool | false | Enable timemory profiling |
| `ROCPROFSYS_USE_SAMPLING` | bool | false | Enable call-stack sampling |
| `ROCPROFSYS_USE_PROCESS_SAMPLING` | bool | true | Enable process-level metrics |
| `ROCPROFSYS_MODE` | string | trace | Mode: trace, sampling, causal |

### Output Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ROCPROFSYS_OUTPUT_PATH` | string | rocprof-sys-%tag%-output | Output directory |
| `ROCPROFSYS_OUTPUT_PREFIX` | string | | Output file prefix |
| `ROCPROFSYS_USE_PID` | bool | true | Include PID in filenames |
| `ROCPROFSYS_TIME_OUTPUT` | bool | true | Include timestamp in output |

### Sampling Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ROCPROFSYS_SAMPLING_FREQ` | int | 10 | Samples per second |
| `ROCPROFSYS_SAMPLING_DELAY` | float | 0.5 | Delay before first sample (seconds) |
| `ROCPROFSYS_SAMPLING_CPUS` | string | | CPU IDs for frequency sampling |
| `ROCPROFSYS_SAMPLING_GPUS` | string | all | GPU IDs for SMI queries |
| `ROCPROFSYS_SAMPLING_TIDS` | string | | Thread IDs to sample |
| `ROCPROFSYS_SAMPLING_CPUTIME` | bool | true | Enable CPU-time sampling |
| `ROCPROFSYS_SAMPLING_REALTIME` | bool | false | Enable real-time sampling |

### Perfetto Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ROCPROFSYS_PERFETTO_BACKEND` | string | inprocess | Backend: inprocess, system |
| `ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB` | int | 1024000 | Trace buffer size in KB |
| `ROCPROFSYS_PERFETTO_FILE` | string | perfetto-trace.proto | Output filename |
| `ROCPROFSYS_PERFETTO_FILL_POLICY` | string | discard | Buffer policy: discard, ring_buffer |

### ROCm Settings

| Variable | Type | Description |
|----------|------|-------------|
| `ROCPROFSYS_USE_ROCM` | bool | Enable ROCm tracing |
| `ROCPROFSYS_USE_AMD_SMI` | bool | Enable AMD SMI metrics |
| `ROCPROFSYS_ROCM_DOMAINS` | string | ROCm domains to trace |
| `ROCPROFSYS_ROCM_EVENTS` | string | GPU hardware counters |
| `ROCPROFSYS_AMD_SMI_METRICS` | string | AMD SMI metrics to collect |

### AMD SMI Metrics

Values for `ROCPROFSYS_AMD_SMI_METRICS`:
- `busy` - GPU utilization
- `temp` - Temperature
- `power` - Power consumption
- `mem_usage` - Memory usage
- `vcn_activity` - VCN activity
- `jpeg_activity` - JPEG activity
- `xgmi` - XGMI link metrics
- `pcie` - PCIe metrics

### Parallelism API Settings

| Variable | Type | Description |
|----------|------|-------------|
| `ROCPROFSYS_USE_MPIP` | bool | Enable MPI profiling |
| `ROCPROFSYS_USE_OMPT` | bool | Enable OpenMP Tools |
| `ROCPROFSYS_USE_KOKKOSP` | bool | Enable Kokkos Tools |
| `ROCPROFSYS_USE_RCCLP` | bool | Enable RCCL profiling |

### Hardware Counter Settings

| Variable | Type | Description |
|----------|------|-------------|
| `ROCPROFSYS_PAPI_EVENTS` | string | PAPI hardware counter events |
| `ROCPROFSYS_PAPI_MULTIPLEXING` | bool | Enable PAPI multiplexing |
| `ROCPROFSYS_PAPI_OVERFLOW` | int | Overflow threshold |

### Timemory Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ROCPROFSYS_TIMEMORY_COMPONENTS` | string | wall_clock | Components to collect |
| `ROCPROFSYS_FLAT_PROFILE` | bool | false | Flat vs. hierarchical profile |
| `ROCPROFSYS_TIMELINE_PROFILE` | bool | false | Timeline profiling mode |
| `ROCPROFSYS_MAX_DEPTH` | int | 65535 | Maximum call stack depth |

### Debug Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ROCPROFSYS_DEBUG` | bool | false | Enable debug output |
| `ROCPROFSYS_VERBOSE` | int | 0 | Verbosity level (0-3) |
| `ROCPROFSYS_DL_VERBOSE` | int | 0 | Dynamic loader verbosity |

---

## Output Files

### Generated Files

| File | Description |
|------|-------------|
| `perfetto-trace.proto` | Perfetto trace (visualize at ui.perfetto.dev) |
| `wall_clock.txt` | Text profile summary |
| `wall_clock.json` | JSON profile data |
| `metadata.json` | Run metadata and configuration |
| `call-stack.txt` | Call-stack samples (if sampling enabled) |

### Visualizing Results

1. **Perfetto UI**: Open `perfetto-trace.proto` at https://ui.perfetto.dev
2. **Text Profiles**: View `wall_clock.txt` directly
3. **JSON Processing**: Parse `wall_clock.json` programmatically

---

## Common Workflows

### Quick Performance Overview

```bash
rocprof-sys-sample --quick -- ./myapp
cat rocprof-sys-output/wall_clock.txt
```

### Detailed GPU Analysis

```bash
rocprof-sys-sample --trace-ai -- ./gpu_app
# Open perfetto-trace.proto in ui.perfetto.dev
```

### MPI Application Profiling

```bash
mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app
```

### Binary Instrumentation Workflow

```bash
# Step 1: Instrument
rocprof-sys-instrument -o ./app.inst -- ./app

# Step 2: Profile
rocprof-sys-run --quick -- ./app.inst

# Step 3: Analyze
cat rocprof-sys-output/wall_clock.txt
```

### Causal Profiling Workflow

```bash
# Add progress points to code, recompile
# Run causal profiling
rocprof-sys-causal -- ./myapp

# Analyze causal_*.json output
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| No output generated | Check `ROCPROFSYS_OUTPUT_PATH` and permissions |
| Empty trace | Ensure `ROCPROFSYS_TRACE=true` |
| Missing GPU data | Set `ROCPROFSYS_USE_ROCM=true` |
| High overhead | Use `--simple` preset or reduce sampling frequency |
| Symbol resolution fails | Compile with `-g` for debug symbols |

### Debug Mode

```bash
# Enable verbose output
ROCPROFSYS_VERBOSE=3 ROCPROFSYS_DEBUG=true rocprof-sys-sample -- ./app
```

---

## Version Information

This documentation is for ROCm Systems Profiler version 2.0.0+.

For the latest information, see:
- GitHub: https://github.com/ROCm/rocm-systems
- Documentation: https://rocm.docs.amd.com/projects/rocprofiler-systems/
