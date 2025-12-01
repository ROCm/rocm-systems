# HIP Quickstart Examples for ROCm Systems Profiler

This directory contains simple, well-documented HIP examples specifically designed for learning how to profile GPU applications with ROCm Systems Profiler.

## Examples

### 1. Vector Addition (`vector_add.cpp`)

**Difficulty:** Beginner
**Concepts:** Basic HIP kernel, memory transfers, memory-bound kernels

A simple vector addition example that demonstrates:
- Basic HIP API usage (memory allocation, transfers, kernel launch)
- Memory-bound kernel characteristics
- How to identify data transfer bottlenecks

**Profile it:**
```bash
# Quick profiling
rocprof-sys-sample --quick --hip-trace -- ./vector_add

# View in Perfetto
# Open rocprof-sys-output/perfetto-trace.proto in ui.perfetto.dev
```

**What to look for:**
- Ratio of kernel execution time to memory transfer time
- Memory bandwidth utilization
- GPU busy time vs. idle time

### 2. Matrix Multiplication (`matrix_multiply.cpp`)

**Difficulty:** Intermediate
**Concepts:** Kernel optimization, shared memory, performance comparison

Implements two versions of matrix multiplication:
- **Naive**: Straightforward implementation with poor memory locality
- **Tiled**: Optimized version using shared memory for better cache utilization

**Profile it:**
```bash
# Compare kernel performance
rocprof-sys-sample --quick --hip-trace -- ./matrix_multiply

# With hardware counters
rocprof-sys-instrument -o matrix_multiply.inst -- ./matrix_multiply
rocprof-sys-run --hip-trace --rocm-events=MemUnitBusy,TCC_HIT,TCC_MISS -- ./matrix_multiply.inst
```

**What to look for:**
- Performance difference between naive and tiled kernels
- Memory access patterns (naive has more cache misses)
- Arithmetic intensity differences
- L2 cache hit rates

**Expected speedup:** 2-5x for tiled version (depends on matrix size and hardware)

### 3. Concurrent Execution with Streams (`streams.cpp`)

**Difficulty:** Advanced
**Concepts:** Async operations, concurrent kernel execution, stream management

Demonstrates three execution modes:
- **Sequential**: All kernels execute one after another
- **Concurrent**: Multiple kernels execute simultaneously using HIP streams
- **Async Memory**: Overlapping memory transfers with computation

**Profile it:**
```bash
# View concurrent execution timeline
rocprof-sys-sample --quick --hip-trace -- ./streams

# Visualize in Perfetto to see kernel overlap
```

**What to look for:**
- In Perfetto timeline: overlapping kernel executions in concurrent mode
- GPU utilization % (higher with concurrent execution)
- Gaps between kernels in sequential mode
- Memory transfer overlap in async mode

**Expected speedup:** Up to Nx where N is the number of streams (in ideal conditions)

## Building the Examples

### Prerequisites

- ROCm (with HIP) installed
- CMake 3.21 or later
- C++17 compatible compiler

### Build Instructions

#### As part of rocprofiler-systems build:

```bash
cd /path/to/rocprofiler-systems
cmake -B build -DROCPROFSYS_BUILD_EXAMPLES=ON
cmake --build build --target vector_add matrix_multiply streams
```

#### Standalone build:

```bash
cd examples/hip-quickstart
cmake -B build
cmake --build build
```

The executables will be in the `build` directory.

## Running the Examples

### Basic Usage

```bash
# Run without profiling (to verify they work)
./vector_add
./matrix_multiply
./streams

# With custom parameters
./vector_add 10000000              # 10M element vectors
./matrix_multiply 2048 2048 2048   # 2048x2048 matrices
./streams 1000000 8 5000           # 1M elements, 8 streams, 5000 iterations
```

### Profiling Workflows

#### Beginner Workflow (Sampling)

1. **Quick profile:**
   ```bash
   rocprof-sys-sample --quick --hip-trace -- ./vector_add
   ```

2. **View results:**
   - Text summary: `cat rocprof-sys-output/wall_clock.txt`
   - Visual trace: Open `rocprof-sys-output/perfetto-trace.proto` in [ui.perfetto.dev](https://ui.perfetto.dev)

#### Intermediate Workflow (Instrumentation)

1. **Instrument the binary:**
   ```bash
   rocprof-sys-instrument -o vector_add.inst -- ./vector_add
   ```

2. **Run with profiling:**
   ```bash
   rocprof-sys-run --hip-trace --trace -- ./vector_add.inst
   ```

3. **Analyze:**
   - Look at kernel execution times
   - Check memory transfer costs
   - Identify optimization opportunities

#### Advanced Workflow (Hardware Counters)

1. **Profile with specific counters:**
   ```bash
   export ROCPROFSYS_USE_ROCPROFILER=ON
   export ROCPROFSYS_ROCM_EVENTS="TCC_HIT,TCC_MISS,MemUnitBusy,GPUBusy"
   rocprof-sys-run --hip-trace -- ./matrix_multiply.inst
   ```

2. **Analyze metrics:**
   - Cache hit rates
   - Memory unit utilization
   - GPU busy percentage

## Understanding the Output

### Text Profiles

The text output shows function-level statistics:

```
|------------------|-------|---------|---------|
| FUNCTION         | COUNT | SUM(ms) | % TOTAL |
|------------------|-------|---------|---------|
| hipMemcpy        | 3     | 125.4   | 35.2%   |
| vector_add_kernel| 1     | 230.6   | 64.8%   |
|------------------|-------|---------|---------|
```

- **High % for hipMemcpy**: Memory transfer bottleneck → Consider reducing transfers
- **High % for kernel**: Compute-bound → Optimize kernel code

### Perfetto Traces

Open in [ui.perfetto.dev](https://ui.perfetto.dev) to see:
- **Timeline view**: When each operation occurred
- **Thread activity**: CPU and GPU thread execution
- **HIP API calls**: Memory operations, kernel launches
- **Kernel execution**: Individual kernel instances

**Navigation tips:**
- Press `W` to zoom in, `S` to zoom out
- Use `A` and `D` to pan left and right
- Click on events to see details
- Look for gaps (idle time) and overlaps (concurrency)

## Common Profiling Questions

### Q: Why is my kernel so slow?

**Check:**
1. Memory bandwidth utilization (is it memory-bound?)
2. Kernel occupancy (are enough threads running?)
3. Arithmetic intensity (FLOPs per byte transferred)
4. Cache hit rates (is data reused effectively?)

**Profile with:**
```bash
rocprof-sys-run --rocm-events=MemUnitBusy,VALUBusy,TCC_HIT,TCC_MISS -- ./app.inst
```

### Q: How do I reduce memory transfer overhead?

**Strategies:**
1. Batch operations to amortize transfer cost
2. Use async memory operations (`hipMemcpyAsync`)
3. Keep data on device between kernels
4. Use unified memory when appropriate

**Profile with:**
```bash
rocprof-sys-sample --hip-trace -- ./app
# Look for hipMemcpy time in trace
```

### Q: How do I improve GPU utilization?

**Strategies:**
1. Use streams for concurrent kernel execution
2. Overlap memory transfers with computation
3. Launch multiple smaller kernels instead of sequential large ones
4. Ensure kernel execution time > launch overhead

**Profile with:**
```bash
rocprof-sys-sample --quick --hip-trace -- ./app
# Check GPUBusy percentage and timeline gaps
```

## Additional Resources

- [ROCm Systems Profiler Documentation](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/)
- [Quickstart Guide](../../docs/tutorials/quickstart.rst)
- [Hardware Counters Reference](../../docs/reference/hardware-counters-reference.rst)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)

## Troubleshooting

### Examples don't build

**Problem:** HIP not found

**Solution:**
```bash
# Ensure ROCm is in your path
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Verify HIP is available
hipconfig
```

### Profiling produces no output

**Problem:** ROCm Systems Profiler not properly installed

**Solution:**
```bash
# Source the setup script
source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh

# Verify installation
rocprof-sys-sample --version
```

### Perfetto trace won't open

**Problem:** Trace file too large

**Solution:**
```bash
# Reduce trace size by profiling shorter runs
./vector_add 100000  # Smaller problem size

# Or reduce buffer size
export ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB=512000
```
