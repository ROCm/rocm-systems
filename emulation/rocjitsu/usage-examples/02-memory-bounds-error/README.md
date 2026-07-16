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
4. **Host-side detection** - Sentinel buffer checks after the kernel (rocjitsu does not flag OOB automatically)

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
ERROR: Out-of-bounds write at index 1000 = 2000
ERROR: Out-of-bounds write at index 1001 = 2002
ERROR: Out-of-bounds write at index 1002 = 2004
ERROR: Out-of-bounds write at index 1003 = 2006
ERROR: Out-of-bounds write at index 1004 = 2008

MEMORY BOUNDS ERROR - 24 out-of-bounds writes detected!
This is expected - the kernel has no bounds checking.
```

With optional `RJ_LOG=1`, stderr also shows kernel dispatch metadata (HIP runtime
kernels may appear before/after yours):

```
[rocjitsu] Logging enabled (RJ_LOG)

[rocjitsu] Kernel #2 dispatch
  entry_pc=0x...  grid=[1024,1,1]  wg=[64,1,1]
  wgs=16  wfs/wg=1  sgprs=...  vgprs=...
```

Here `grid=[1024,1,1]` is the **global work-item count** (total threads), `wg=[64,1,1]`
is the block size, and `wgs=16` is the number of blocks. See **Debugging Workflow**.

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
   The app prints launch geometry on stdout (`Grid: 16 blocks, Block: 64 threads`,
   `Total threads: 1024`). Compare to `N=1000`.

   Optionally confirm the kernel dispatch under rocjitsu:
   ```bash
   RJ_LOG=1 make run-buggy
   # stderr shows dispatch metadata, e.g.:
   #   grid=[1024,1,1]  wg=[64,1,1]  wgs=16
   #
   # grid = global work-items (total threads) per dimension
   # wg   = workgroup (block) size
   # wgs  = number of workgroups (blocks)
   #
   # total_threads = grid[0] * grid[1] * grid[2]   (do NOT multiply grid × wg again)
   #              or wgs * wg[0] * wg[1] * wg[2]
   #
   # Pick the user kernel (often Kernel #2); #1/#3 may be HIP runtime memcpy kernels.
   ```

   **Note:** rocjitsu does not report out-of-bounds writes. Detection is the host
   sentinel loop in `bounds_error.cpp` after `hipMemcpy`.

3. **Add bounds checking**
   - Always check `if (i < N)` for 1D
   - Check all dimensions for 2D/3D

4. **Verify the fix**
   ```bash
   make run-fixed
   ```

## Key Takeaways

- Always add bounds checks in GPU kernels (`if (i < N)`)
- Calculate grid size carefully: `(N + block_size - 1) / block_size`
- Use `<` not `<=` for bounds checking
- Test with non-power-of-2 sizes to catch edge cases
- OOB in this example is detected by **host code**, not by `RJ_LOG=1`
- `RJ_LOG=1` only confirms dispatch geometry (`grid` / `wg` / `wgs` on stderr)

## Next Steps

- [Example 3: Kernel Crash Debug](../03-kernel-crash-debug/)
- [Example 1: Vector Add Basic](../01-vector-add-basic/)