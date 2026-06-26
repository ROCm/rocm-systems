# Example 5: Global Memory Race Detection

## Objective

Detect races in global memory across multiple thread blocks.

## Key Concepts

- Global memory atomics
- Inter-block synchronization
- Bank conflicts
- Reduction patterns

## Files

- `src/global_race.cpp` - Multi-block race condition
- `src/global_fixed.cpp` - Fixed with atomics
- `Makefile`

## Quick Start

```bash
make
make run-race     # See races detected
make run-fixed    # No races
```

## The Pattern

Multiple blocks writing to shared global counter:

```cpp
// Buggy: Race between blocks
__global__ void count_total(int *global_counter, int *data, int N) {
  int sum = 0;
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    sum += data[i];
  }
  *global_counter += sum;  // RACE!
}

// Fixed: Atomic update
__global__ void count_total_safe(int *global_counter, int *data, int N) {
  int sum = 0;
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    sum += data[i];
  }
  atomicAdd(global_counter, sum);  // SAFE
}
```

## Running

Enable race detector:
```bash
RJ_SINKS=race_detector make run-race
```