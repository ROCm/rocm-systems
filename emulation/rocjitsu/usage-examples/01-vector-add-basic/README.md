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
- Enable `RJ_LOG=1` to see dispatch events

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
  Sample: C[0] = 0.3 (A[0]=0.1 + B[0]=0.2)
  Sample: C[512] = 153.9 (A[512]=51.3 + B[512]=102.6)
  
Cleanup complete.
```

### With Verbose Logging (RJ_LOG=1)
```
[rocjitsu] Initializing KMD interposer
[rocjitsu] Config: amdgpu_cdna4_kmd.json
[rocjitsu] Creating GPU agent: gfx950
Vector addition: C = A + B
  Vector size: 1024 elements (4096 bytes)
  
[rocjitsu] hipMalloc: 4096 bytes at 0x7f8a40000000
[rocjitsu] hipMalloc: 4096 bytes at 0x7f8a40001000
[rocjitsu] hipMalloc: 4096 bytes at 0x7f8a40002000
Allocating device memory...
[rocjitsu] hipMemcpy: H2D, 4096 bytes
[rocjitsu] hipMemcpy: H2D, 4096 bytes
Copying input data to device...
[rocjitsu] Kernel dispatch: vector_add
[rocjitsu]   Grid: (16, 1, 1), Block: (64, 1, 1)
[rocjitsu]   Executing 1024 workitems
Launching kernel: grid(16, 1, 1), block(64, 1, 1)
[rocjitsu] Kernel completed: 1024 workitems executed
[rocjitsu] hipMemcpy: D2H, 4096 bytes
Copying results back to host...
  
Verification: PASSED
  All 1024 elements correct!
  Sample: C[0] = 0.3 (A[0]=0.1 + B[0]=0.2)
  Sample: C[512] = 153.9 (A[512]=51.3 + B[512]=102.6)
  
[rocjitsu] hipFree: 0x7f8a40000000
[rocjitsu] hipFree: 0x7f8a40001000
[rocjitsu] hipFree: 0x7f8a40002000
Cleanup complete.
```

## Common Issues and Debugging

### Issue 1: Incorrect Results

**Symptom**: Verification fails, some elements differ

**Debug Steps**:
1. Enable verbose logging: `RJ_LOG=1 make run`
2. Check kernel launch parameters
3. Verify array indexing: `i = blockIdx.x * blockDim.x + threadIdx.x`
4. Check boundary condition: `if (i < N)`

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
- `RJ_LOG=1` enables detailed execution tracing
- Always check bounds in GPU kernels
- Verify results against CPU reference implementation
- Use daemon mode for complex applications (PyTorch, etc.)

## Next Steps

- [Example 2: Memory Bounds Error](../02-memory-bounds-error/) - Detect buffer overruns
- [Example 4: Data Race Detection](../04-data-race-simple/) - Find race conditions
- [Example 9: PyTorch Debugging](../09-pytorch-model-debug/) - Debug ML models
