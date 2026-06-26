# Example 6: Memory Coalescing Analysis

## Objective

Analyze and optimize memory access patterns for better performance.

## Key Concepts

- Coalesced vs. strided access
- Memory bandwidth utilization
- Access pattern optimization
- Performance profiling

## Files

- `src/strided_access.cpp` - Uncoalesced (slow)
- `src/coalesced_access.cpp` - Coalesced (fast)
- `Makefile`

## Memory Access Patterns

### Uncoalesced (Bad)
```cpp
__global__ void strided(float *out, float *in, int stride) {
  int i = threadIdx.x;
  out[i] = in[i * stride];  // Non-contiguous access
}
```

### Coalesced (Good)
```cpp
__global__ void coalesced(float *out, float *in) {
  int i = threadIdx.x;
  out[i] = in[i];  // Contiguous access
}
```

## Running

```bash
make
make run-profile  # Profile both versions
```

Profile with:
```bash
RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1 make run
```