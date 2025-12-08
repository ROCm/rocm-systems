# ROCTX API Example

This example demonstrates using the ROCTX (ROCm Tracer Extension) API to add custom annotations to GPU traces.

## What This Example Demonstrates

- **Custom region marking**: Adding named regions to traces
- **Nested regions**: Creating hierarchical trace annotations
- **Message annotations**: Adding contextual information to traces
- **Marker placement**: Strategic annotation for trace clarity

## ROCTX API Basics

ROCTX provides API calls to annotate GPU traces with custom information:

```cpp
#include <roctracer/roctx.h>

// Mark a region
roctxRangePush("my_region");
// ... code to annotate ...
roctxRangePop();

// Add a marker
roctxMark("checkpoint_reached");
```

## Building

```bash
# As part of rocprofiler-systems build
cmake -B build -DROCPROFSYS_BUILD_EXAMPLES=ON
cmake --build build --target roctx

# The executable will be in the build directory
```

## Profiling This Example

### Basic Profiling

```bash
# Profile with trace enabled
rocprof-sys-sample --quick --hip-trace -- ./roctx

# View in Perfetto
# Open rocprof-sys-output/perfetto-trace.proto in ui.perfetto.dev
```

### With Instrumentation

```bash
# Instrument the binary
rocprof-sys-instrument -o roctx.inst -- ./roctx

# Run with tracing
rocprof-sys-run --trace --hip-trace -- ./roctx.inst
```

## What to Look For in Traces

1. **Custom Regions**: Your annotated regions should appear in the Perfetto timeline
2. **Nested Structure**: Check if region hierarchy is preserved
3. **Markers**: Look for marker events at specific points
4. **Correlation**: How custom regions align with HIP API calls

## Use Cases for ROCTX

### 1. Algorithm Phases

```cpp
roctxRangePush("data_preparation");
// prepare data
roctxRangePop();

roctxRangePush("computation");
// run computation
roctxRangePop();

roctxRangePush("result_collection");
// collect results
roctxRangePop();
```

### 2. Iterative Algorithms

```cpp
for(int iter = 0; iter < max_iters; iter++)
{
    char label[64];
    snprintf(label, sizeof(label), "iteration_%d", iter);
    roctxRangePush(label);

    // iteration work

    roctxRangePop();
}
```

### 3. Debugging and Checkpoints

```cpp
roctxMark("before_critical_section");
// critical code
roctxMark("after_critical_section");
```

## Best Practices

1. **Use descriptive names**: Make regions easy to identify in traces
2. **Balance granularity**: Too many regions clutters traces, too few lacks detail
3. **Match push/pop**: Always balance `roctxRangePush` with `roctxRangePop`
4. **Use for algorithms, not every function**: Focus on logical phases
5. **Combine with other profiling**: ROCTX enhances, doesn't replace, standard profiling

## Comparison with ROCm Systems Profiler User API

| Feature | ROCTX | ROCm Systems Profiler User API |
|---------|-------|--------------------------------|
| Purpose | GPU trace annotation | General profiling regions |
| Overhead | Very low | Low |
| Visibility | GPU traces only | All profiling output |
| Use case | GPU-focused apps | All applications |

Both can be used together for comprehensive profiling!

## Related Examples

- [user-api](../user-api/) - Using ROCm Systems Profiler's User API
- [hip-quickstart](../hip-quickstart/) - Basic HIP profiling examples

## Additional Resources

- [ROCm Systems Profiler User API Guide](../../docs/how-to/using-rocprof-sys-api.rst)
- [Understanding Output](../../docs/how-to/understanding-rocprof-sys-output.rst)
- [ROCTracer Documentation](https://rocm.docs.amd.com/projects/roctracer/en/latest/)
