# Matrix Transpose Example

This example demonstrates memory access pattern optimization through matrix transpose operations.

## What This Example Demonstrates

- **Memory coalescing**: How memory access patterns affect performance
- **Cache utilization**: Impact of spatial locality on performance
- **Optimization techniques**: Multiple transpose implementations with varying performance

## Building

```bash
# As part of rocprofiler-systems build
cmake -B build -DROCPROFSYS_BUILD_EXAMPLES=ON
cmake --build build --target transpose

# The executables will be in the build directory
```

## Profiling This Example

### Quick Profile

```bash
# Profile with sampling
rocprof-sys-sample --quick --hip-trace -- ./transpose

# View results
cat rocprof-sys-output/wall_clock.txt
```

### Detailed Analysis

```bash
# Instrument and trace
rocprof-sys-instrument -o transpose.inst -- ./transpose
rocprof-sys-run --hip-trace --trace -- ./transpose.inst

# Open perfetto-trace.proto in ui.perfetto.dev
```

### With Hardware Counters

```bash
# Check cache behavior
export ROCPROFSYS_USE_ROCPROFILER=ON
export ROCPROFSYS_ROCM_EVENTS="TCC_HIT,TCC_MISS,TCC_EA_RDREQ,TCC_EA_WRREQ"
rocprof-sys-run --hip-trace -- ./transpose.inst
```

## What to Look For

1. **Kernel Execution Times**: Compare different transpose implementations
2. **Memory Bandwidth**: Which version achieves higher bandwidth?
3. **Cache Hit Rates**: Better spatial locality → higher hit rates
4. **Performance Scaling**: How do different sizes affect each implementation?

## Understanding the Results

**Good transpose implementation characteristics:**
- High memory bandwidth utilization (close to theoretical peak)
- High L2 cache hit rate
- Minimal warp divergence
- Efficient use of shared memory

**Poor transpose implementation characteristics:**
- Uncoalesced global memory accesses
- Many cache misses
- Bank conflicts in shared memory

## Related Examples

- [hip-quickstart/matrix_multiply](../hip-quickstart/) - Another memory optimization example
- [parallel-overhead](../parallel-overhead/) - Understanding parallelization costs

## Additional Resources

- [Metrics Glossary](../../docs/reference/metrics-glossary.rst) - Understanding performance metrics
- [Hardware Counters Reference](../../docs/reference/hardware-counters-reference.rst) - GPU counter details

