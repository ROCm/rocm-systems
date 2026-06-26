# Example 2: Memory Bounds Error Detection

## Objective

Learn to detect and fix out-of-bounds memory access in GPU kernels:
- Identify buffer overruns
- Debug array indexing errors
- Understand bounds checking
- Fix off-by-one errors

## What This Example Demonstrates

1. **Out-of-bounds access** - Reading/writing beyond array limits
2. **Index calculation errors** - Wrong grid/block math
3. **Bounds checking** - Proper safety checks
4. **Memory corruption** - Effects of invalid access

## Key Debugging Points

### The Bug

```cpp
__global__ void process_array(float *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // BUG: Missing bounds check!
  data[i] = i * 2.0f;  // May access beyond array
}
```

### The Fix

```cpp
__global__ void process_array(float *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // FIXED: Add bounds check
  if (i < N) {
    data[i] = i * 2.0f;
  }
}
```

## Files

- `src/bounds_error.cpp` - Buggy version with out-of-bounds access
- `src/bounds_fixed.cpp` - Fixed version with proper checks
- `Makefile` - Build and run commands

## Building

```bash
make
```

## Running

### Run Buggy Version
```bash
make run-buggy
```

### Run Fixed Version
```bash
make run-fixed
```

### Compare Both
```bash
make compare
```

## Expected Output

### Buggy Version
```
Memory Bounds Error Example - BUGGY VERSION
  Array size: 1000 elements
  Grid: 16 blocks, Block: 64 threads
  Total threads: 1024

WARNING: Grid size (1024) > array size (1000)
         24 threads will access invalid memory!

Launching kernel...
ERROR: Memory corruption detected!
  data[1000] = 2000.0 (should be uninitialized)
  data[1023] = 2046.0 (out of bounds!)

MEMORY BOUNDS ERROR - Kernel accessed invalid memory
```

### Fixed Version
```
Memory Bounds Error Example - FIXED VERSION
  Array size: 1000 elements
  Grid: 16 blocks, Block: 64 threads
  Total threads: 1024

Bounds check enabled: threads >= 1000 will skip write

Launching kernel...
Verification: PASSED
  All 1000 elements correct
  No out-of-bounds access detected

SUCCESS - Proper bounds checking prevented invalid access
```

## Common Patterns

### Pattern 1: Grid Larger Than Data
```cpp
// Problem: Grid doesn't match data size exactly
int threads = 1024;
int blocks = 10;
// Total: 10240 threads for 10000 elements = 240 extra

// Fix: Add bounds check OR calculate grid carefully
int blocks = (N + threads - 1) / threads;  // Exact coverage
```

### Pattern 2: Off-by-One Errors
```cpp
// Wrong: <= allows accessing element N (out of bounds)
if (i <= N) {
  data[i] = ...;  // BUG when i == N
}

// Correct: < ensures i is in [0, N-1]
if (i < N) {
  data[i] = ...;
}
```

### Pattern 3: 2D Index Calculation
```cpp
// Calculate 2D index from thread coordinates
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

// Wrong: No bounds check
int idx = y * width + x;
data[idx] = ...;  // May overflow if x >= width or y >= height

// Correct: Check both dimensions
if (x < width && y < height) {
  int idx = y * width + x;
  data[idx] = ...;
}
```

## Debugging Workflow

1. **Identify the symptoms**
   - Incorrect results in part of array
   - Memory corruption
   - Segmentation faults

2. **Check grid/block sizes**
   ```bash
   RJ_LOG=1 make run-buggy
   # Look for: Grid: (X, Y, Z), Block: (X, Y, Z)
   # Calculate: total_threads = gridX * gridY * gridZ * blockX * blockY * blockZ
   ```

3. **Add bounds checking**
   - Always check `if (i < N)` for 1D
   - Check all dimensions for 2D/3D

4. **Verify the fix**
   ```bash
   make run-fixed
   ```

## Key Takeaways

- Always add bounds checks in GPU kernels
- Calculate grid size carefully: `(N + block_size - 1) / block_size`
- Use `<` not `<=` for bounds checking
- Test with non-power-of-2 sizes to catch edge cases

## Next Steps

- [Example 3: Kernel Crash Debug](../03-kernel-crash-debug/)
- [Example 1: Vector Add Basic](../01-vector-add-basic/)