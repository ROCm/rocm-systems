# ROCm Systems Profiler Examples

This directory contains examples demonstrating how to use ROCm Systems Profiler to profile and analyze applications. Examples range from simple introductory programs to complex real-world scenarios.

## Getting Started

New to ROCm Systems Profiler? Start here:

1. **Read the [Quickstart Guide](../docs/tutorials/quickstart.rst)**
2. **Try the [HIP Quickstart Examples](hip-quickstart/)** - Simple, well-documented HIP examples perfect for learning
3. **Profile your first application** - Use the knowledge from examples to profile your own code

## Example Categories

### Beginner-Friendly Examples

These examples are designed for learning profiling basics:

- **[hip-quickstart/](hip-quickstart/)** - Simple HIP examples with profiling guidance
  - `vector_add` - Basic memory-bound kernel
  - `matrix_multiply` - Optimization comparison
  - `streams` - Concurrent kernel execution

### Code Instrumentation Examples

Learn different ways to instrument code:

- **[user-api/](user-api/)** - Using ROCm Systems Profiler API to mark custom regions
- **[roctx/](roctx/)** - Using ROCTX API for GPU tracing annotations
- **[python/](python/)** - Profiling Python applications

### Performance Analysis Examples

Examples demonstrating performance optimization:

- **[transpose/](transpose/)** - Memory access pattern optimization for matrix transpose
- **[parallel-overhead/](parallel-overhead/)** - Measuring parallelization overhead
- **[lulesh/](lulesh/)** - Real-world application profiling (Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics)

### Advanced Profiling Techniques

Complex profiling scenarios:

- **[causal/](causal/)** - Causal profiling to identify optimization opportunities
- **[mpi/](mpi/)** - Profiling MPI applications
- **[rccl/](rccl/)** - Profiling collective communication with RCCL
- **[openmp/](openmp/)** - Profiling OpenMP applications

### Special Features

Examples showcasing specific ROCm Systems Profiler features:

- **[code-coverage/](code-coverage/)** - Code coverage analysis
- **[trace-time-window/](trace-time-window/)** - Profiling specific time windows
- **[videodecode/](videodecode/)** - Profiling ROCm video decode operations
- **[jpegdecode/](jpegdecode/)** - Profiling JPEG decode operations
- **[fork/](fork/)** - Profiling applications that fork
- **[thread-limit/](thread-limit/)** - Thread count limitations

### Binary Instrumentation Examples

- **[rewrite-caller/](rewrite-caller/)** - Binary rewrite and instrumentation examples

## Quick Reference

### Basic Profiling Commands

```bash
# Sampling (no instrumentation needed)
rocprof-sys-sample --quick -- ./your_app

# Binary instrumentation
rocprof-sys-instrument -o app.inst -- ./your_app
rocprof-sys-run -- ./app.inst

# Python profiling
rocprof-sys-python -- ./script.py
```

### Building Examples

#### Build all examples

```bash
cd /path/to/rocprofiler-systems
cmake -B build -DROCPROFSYS_BUILD_EXAMPLES=ON
cmake --build build
```

#### Build specific example

```bash
cd examples/hip-quickstart
cmake -B build
cmake --build build
```

## Example Workflow

### 1. Identify Hotspots (Sampling)

Use sampling to quickly find performance bottlenecks without instrumentation:

```bash
rocprof-sys-sample -F -- ./your_app
cat rocprof-sys-output/sampling_wall_clock.txt
```

Look for functions with high `% of total` - these are your hotspots.

### 2. Detailed Analysis (Instrumentation)

Instrument the application for detailed tracing:

```bash
rocprof-sys-instrument -o your_app.inst -- ./your_app
rocprof-sys-run --trace --hip-trace -- ./your_app.inst
```

View trace in Perfetto: [ui.perfetto.dev](https://ui.perfetto.dev)

### 3. Targeted Optimization (Custom Regions)

Use the User API to profile specific code sections:

```cpp
#include <rocprof-sys/user.h>

rocprofsys_user_push_region("critical_section");
// ... code to profile ...
rocprofsys_user_pop_region("critical_section");
```

See [user-api/](user-api/) for complete examples.

### 4. Verify Improvements

After optimization, re-profile to verify improvements:

```bash
# Before optimization
rocprof-sys-sample --quick -- ./app_v1 > profile_v1.txt

# After optimization
rocprof-sys-sample --quick -- ./app_v2 > profile_v2.txt

# Compare results
diff profile_v1.txt profile_v2.txt
```

## Prerequisites

### Required

- ROCm Systems Profiler installed
- C/C++ compiler (GCC 9+ or Clang 10+)
- CMake 3.21+

### Optional (for specific examples)

- **HIP examples**: ROCm with HIP
- **MPI examples**: MPI implementation (OpenMPI, MPICH)
- **Python examples**: Python 3.7+
- **OpenMP examples**: OpenMP-enabled compiler

## Environment Setup

Before running examples, set up the environment:

```bash
# Option 1: Source setup script (recommended)
source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh

# Option 2: Manual setup
export PATH=/opt/rocprofiler-systems/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocprofiler-systems/lib:$LD_LIBRARY_PATH
```

Verify installation:

```bash
rocprof-sys-sample --version
```

## Common Profiling Patterns

### Pattern 1: Compare Implementations

Profile multiple versions to find the fastest:

```bash
# Baseline
rocprof-sys-sample --quick -o ./baseline run1 -- ./app_baseline

# Optimized version 1
rocprof-sys-sample --quick -o ./opt1 run1 -- ./app_opt1

# Optimized version 2
rocprof-sys-sample --quick -o ./opt2 run1 -- ./app_opt2

# Compare execution times
```

### Pattern 2: Scaling Analysis

Test performance across different problem sizes:

```bash
for size in 1000 10000 100000 1000000; do
    echo "Testing size: $size"
    rocprof-sys-sample -- ./app $size
done
```

### Pattern 3: Hardware Counter Analysis

Collect specific hardware metrics:

```bash
# CPU cache analysis
export ROCPROFSYS_PAPI_EVENTS="PAPI_L1_DCM PAPI_L2_DCM PAPI_L3_TCM"
rocprof-sys-sample -- ./app

# GPU memory analysis
export ROCPROFSYS_USE_ROCPROFILER=ON
export ROCPROFSYS_ROCM_EVENTS="TCC_HIT,TCC_MISS"
rocprof-sys-run --hip-trace -- ./app.inst
```

## Interpreting Results

### Text Output

Look at `rocprof-sys-output/wall_clock.txt`:

- **High COUNT + Low individual time**: Consider inlining
- **High % of total**: Focus optimization here
- **High STDDEV**: Inconsistent performance - investigate why

### Perfetto Traces

Open `.proto` files in [ui.perfetto.dev](https://ui.perfetto.dev):

- **Look for gaps**: Idle time (optimization opportunity)
- **Check thread activity**: Is parallelism effective?
- **Identify bottlenecks**: Which operations dominate?
- **Memory transfers**: Are they overlapping with compute?

## Tips for Effective Profiling

1. **Start simple**: Use sampling before instrumentation
2. **Profile representative runs**: Ensure workload is realistic
3. **Minimize external factors**: Run on quiet system
4. **Profile multiple times**: Average results for consistency
5. **Focus on hotspots**: Don't optimize everything
6. **Measure improvements**: Always verify optimizations help

## Troubleshooting

### Examples fail to build

**Check ROCm installation:**

```bash
hipconfig
rocminfo
```

**Verify CMake version:**

```bash
cmake --version  # Should be 3.21+
```

### Profiling produces no output

**Check environment:**

```bash
which rocprof-sys-sample
rocprof-sys-sample --version
```

**Verify permissions:**

```bash
# For CPU hardware counters
cat /proc/sys/kernel/perf_event_paranoid  # Should be <= 2
```

### Application behavior changes when profiled

**Reduce overhead:**

```bash
# Use sampling instead of instrumentation
rocprof-sys-sample -f 20 -- ./app  # Lower frequency

# Or profile only specific functions
rocprof-sys-instrument -R '^main$' -- ./app
```

## Additional Resources

- **Documentation**: [ROCm Systems Profiler Docs](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/)
- **Tutorials**: [docs/tutorials/](../docs/tutorials/)
- **API Reference**: [docs/reference/](../docs/reference/)
- **Video Tutorials**: [docs/tutorials/video-tutorials.rst](../docs/tutorials/video-tutorials.rst)
- **GitHub**: [github.com/ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)
