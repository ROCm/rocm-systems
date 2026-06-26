# Example 3: Kernel Crash Debugging

## Objective

Debug kernel crashes and segmentation faults:
- Identify NULL pointer dereferences
- Debug invalid memory access
- Trace kernel execution
- Fix crash-causing bugs

## Key Concepts

- NULL pointer checks
- Memory validation
- Error propagation
- Execution tracing with `RJ_LOG=1`

## Files

- `src/crash_example.cpp` - Kernel with NULL pointer bug
- `src/crash_fixed.cpp` - Fixed with proper validation
- `Makefile`

## Quick Start

```bash
make
make run-crash    # See the crash
make run-fixed    # See the fix
```

## Common Crashes

### NULL Pointer Dereference
```cpp
// Bug
__global__ void kernel(float *ptr) {
  *ptr = 1.0f;  // Crashes if ptr is NULL!
}

// Fix
__global__ void kernel(float *ptr) {
  if (ptr != nullptr) {
    *ptr = 1.0f;
  }
}
```

### Uninitialized Pointers
```cpp
// Bug
float *d_data;  // Uninitialized!
kernel<<<1, 1>>>(d_data);  // Crash

// Fix
float *d_data = nullptr;
hipMalloc(&d_data, size);
kernel<<<1, 1>>>(d_data);
```

## Debugging Steps

1. Enable logging: `RJ_LOG=1 make run-crash`
2. Identify crash location
3. Add validation
4. Re-test

## Next Steps

- [Example 2: Memory Bounds](../02-memory-bounds-error/)
- [Example 4: Race Detection](../04-data-race-simple/)