# Example 1: Basic Vector Addition Debugging

## Objective

Learn to debug a simple HIP vector addition kernel using rocjitsu:
- Verify kernel launches successfully
- Check memory transfers (host ↔ device)
- Validate computation correctness
- Enable basic logging and tracing

## What This Example Demonstrates

1. **Basic HIP kernel structure** - Simple `__global__` kernel
2. **Memory management** - `hipMalloc`, `hipMemcpy`, `hipFree`
3. **Kernel launch** - Grid/block dimensions
4. **Verification** - Host-side result checking
5. **rocjitsu simulation** - Running without physical GPU

## Key Debugging Points

### 1. Kernel Launch Verification
- Are grid/block dimensions correct?
- Is the kernel dispatched successfully?
- Enable `RJ_LOG=1` to see kernel dispatch metadata on stderr (`grid`, `wg`, `wgs`)

### 2. Memory Transfer Debugging
- Host-to-device copy before kernel
- Device-to-host copy after kernel
- Check buffer sizes match data size

### 3. Correctness Checking
- Compare GPU results with CPU reference
- Identify numerical errors
- Check boundary conditions

## Files

- `src/vector_add.cpp` - Complete HIP application
- `Makefile` - Build and run commands
- `expected-output/output.txt` - Expected simulation output

## Building

### Prerequisites

```bash
# Ensure rocjitsu is built
cd ../../
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
export PATH=$PWD/build/tools/rocjitsu:$PATH

# Ensure hipcc is available
which hipcc
```

### Compile

```bash
cd usage-examples/01-vector-add-basic
make
```

This produces: `build/vector_add`

## Running

### Basic Execution
```bash
make run
```

Equivalent to:
```bash
rocjitsu --config ../../configs/amdgpu_cdna4_kmd.json -- ./build/vector_add
```

### With Logging Enabled
```bash
make run-verbose
```

Equivalent to:
```bash
RJ_LOG=1 rocjitsu --config ../../configs/amdgpu_cdna4_kmd.json -- ./build/vector_add
```

### Daemon Mode
```bash
make run-daemon
```

Equivalent to:
```bash
rocjitsu --daemon --config ../../configs/amdgpu_cdna4_kmd.json -- ./build/vector_add
```

## Expected Output

### Successful Run
```
Vector addition: C = A + B
  Vector size: 1024 elements (4096 bytes)
  
Allocating device memory...
Copying input data to device...
Launching kernel: grid(16, 1, 1), block(64, 1, 1)
Copying results back to host...
  
Verification: PASSED
  All 1024 elements correct!
  Sample: C[0] = 0 (A[0]=0 + B[0]=0)
  Sample: C[512] = 153.6 (A[512]=51.2 + B[512]=102.4)
  
Cleanup complete.
```

### With Verbose Logging (RJ_LOG=1)

`RJ_LOG=1` enables the kernel logging plugin only. It does **not** trace HIP API
calls (`hipMalloc`, `hipMemcpy`, etc.) or print kernel names.

```
[rocjitsu] Logging enabled (RJ_LOG)
Vector addition: C = A + B
  Vector size: 1024 elements (4096 bytes)

Allocating device memory...
Copying input data to device...
Launching kernel: grid(16, 1, 1), block(64, 1, 1)
Copying results back to host...

[rocjitsu] Kernel #2 dispatch
  entry_pc=0x...  grid=[1024,1,1]  wg=[64,1,1]
  wgs=16  wfs/wg=1  sgprs=...  vgprs=...

Verification: PASSED
  All 1024 elements correct!
  Sample: C[0] = 0 (A[0]=0 + B[0]=0)
  Sample: C[512] = 153.6 (A[512]=51.2 + B[512]=102.4)

Cleanup complete.
```

**Reading dispatch lines:** `grid` is the global work-item count (here 1024 total
threads). `wg` is the block (workgroup) size. `wgs` is the number of blocks.
Kernel #1 and #3 (if present) are usually HIP runtime setup kernels, not your
`vector_add_kernel`.

## Common Issues and Debugging

### Issue 1: Incorrect Results

**Symptom**: Verification fails, some elements differ

**Debug Steps**:
1. Enable logging: `RJ_LOG=1 make run`
2. Check app stdout for `Launching kernel: grid(...), block(...)`
3. On stderr, find your kernel dispatch (`grid=[...] wg=[...] wgs=...`)
4. Verify array indexing: `i = blockIdx.x * blockDim.x + threadIdx.x`
5. Check boundary condition: `if (i < N)`

**Example Fix**:
```cpp
// Wrong: missing bounds check
__global__ void vector_add(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];  // May access beyond array!
}

// Correct: with bounds check
__global__ void vector_add(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {  // Safety check
        C[i] = A[i] + B[i];
    }
}
```

### Issue 2: Kernel Launch Failure

**Symptom**: hipLaunchKernel returns error

**Debug Steps**:
1. Check grid/block dimensions are valid
2. Verify block size ≤ 1024 for most GPUs
3. Ensure kernel code compiles correctly

### Issue 3: Memory Transfer Errors

**Symptom**: hipMemcpy fails or returns garbage

**Debug Steps**:
1. Verify `hipMalloc` succeeded before `hipMemcpy`
2. Check pointer is not NULL
3. Confirm size parameter matches allocation
4. Verify direction flag (H2D vs D2H)

## Exercises

1. **Modify vector size** - Change `N` to 10000, observe scalability
2. **Change block size** - Try 128, 256, 512 threads per block
3. **Add more operations** - Implement vector subtraction or multiplication
4. **Break it intentionally** - Remove bounds check, observe crash

## Key Takeaways

- rocjitsu simulates GPU execution without physical hardware
- `RJ_LOG=1` prints kernel dispatch metadata on stderr (`grid`, `wg`, `wgs`)
- Always check bounds in GPU kernels
- Verify results against CPU reference implementation
- Use daemon mode for complex applications (PyTorch, etc.)

## Next Steps

- [Example 2: Memory Bounds Error](../02-memory-bounds-error/) - Detect buffer overruns
- [Example 4: Data Race Detection](../04-data-race-simple/) - Find race conditions
- [Example 9: PyTorch Debugging](../09-pytorch-model-debug/) - Debug ML models
