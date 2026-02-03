# Device Linker Changes Summary - February 3, 2026

## Changes Made This Session

### 1. Fixed NCCL_CUDA_ARCH for AMD GPUs (`src/include/device.h`)

**Problem**: `NCCL_CUDA_ARCH` was evaluating to 0 on AMD HIP device compilation because `__CUDA_ARCH__` is a CUDA-specific macro.

**Impact**: This caused `ncclCollUnroll` to return 4 instead of 8 for gfx950, leading to incorrect LDS size calculations:
- Device code expected: 40800 bytes (NCCL_CUDA_ARCH=0)
- Host code expected: 73568 bytes (cudaArch=950)

**Fix**:
```cpp
#ifdef __CUDA_ARCH__
  #define NCCL_CUDA_ARCH __CUDA_ARCH__
#elif defined(__gfx950__)
  #define NCCL_CUDA_ARCH 950
#elif defined(__gfx942__)
  #define NCCL_CUDA_ARCH 942
#elif defined(__GFX9__)
  #define NCCL_CUDA_ARCH 900
#else
  #define NCCL_CUDA_ARCH 0
#endif
```

### 2. Added ENABLE_WARP_SPEED to Specialized Kernel Build (`CMakeLists.txt`)

**Problem**: Specialized kernels were compiled without `ENABLE_WARP_SPEED`, causing structure layout mismatch with host code.

**Fix**: Added conditional `ENABLE_WARP_SPEED` to `SPEC_KERNEL_DEFS`:
```cmake
if(ENABLE_WARP_SPEED)
  list(APPEND SPEC_KERNEL_DEFS -DENABLE_WARP_SPEED)
endif()
```

### 3. Added ENABLE_LL128 to All Device Linker Builds

**Problem**: `ENABLE_LL128` was missing from dispatcher and specialized kernel compilation.

**Files Changed**:
- `tools/device_linker/CMakeLists.txt`: Added `ENABLE_LL128` to `DISPATCHER_DEFS`
- `tools/device_linker/build_with_device_linker.sh`: Added `-DENABLE_LL128`

### 4. Expanded WARP_SPEED_SUPPORTED_ARCHS (`CMakeLists.txt`)

**Problem**: `sramecc+` variants (used by device linker) weren't in the supported list.

**Fix**: Added all variants including sramecc combinations:
```cmake
set(WARP_SPEED_SUPPORTED_ARCHS 
    "gfx942" "gfx942:xnack-" "gfx942:xnack+" 
    "gfx942:xnack-:sramecc+" "gfx942:xnack+:sramecc+" ...
    "gfx950" "gfx950:xnack-" "gfx950:xnack+" 
    "gfx950:xnack-:sramecc+" "gfx950:xnack+:sramecc+" ...)
```

## Compilation Flags Audit

All compilation units MUST have consistent defines. Current required defines:

| Define | Purpose | Affects Structure Layout |
|--------|---------|-------------------------|
| `DEVICE_LINKER` | Enables function table dispatch | No |
| `ENABLE_FAULT_INJECTION` | Adds faults field | Yes |
| `ENABLE_WARP_SPEED` | Adds warpComm, warpChannel fields | Yes |
| `ENABLE_LL128` | Affects protocol selection | Possibly |

### Current Define Status by Compilation Unit

| Unit | DEVICE_LINKER | ENABLE_FAULT_INJECTION | ENABLE_WARP_SPEED | ENABLE_LL128 |
|------|---------------|------------------------|-------------------|--------------|
| Host code (rccl.so) | ✓ | ✓ | ✓ | ✓ |
| Specialized kernels | ✓ | ✓ | ✓ | ✓ (conditional) |
| Dispatcher (CMake) | ✓ | ✓ | ✓ | ✓ (after fix) |
| Dispatcher (script) | ✓ | ✓ | ✓ | ✓ (after fix) |

## Current Status

### Working
- Single-GPU operations (AllReduce, etc.)
- All structure layouts are now consistent (LDS = 73568 bytes)
- Function table dispatch mechanism
- PC-relative addressing in merged ELF

### Not Working
- **Multi-GPU operations crash with Memory Fault Error**
- Fault occurs in specialized kernels (both TREE and RING variants)
- Crash location: `flat_load_dwordx2` instructions reading from computed addresses
- Root cause appears to be NULL or invalid connection pointers in `ncclShmemGroup`

## Crash Analysis

### Fault Location
Both TREE and RING algorithms crash when accessing `ncclConnInfo` pointers:

**TREE crash at `ncclDevFunc_AllReduce_TREE_LL_Sum_f32_0_0_1()+5680`:**
- v[6:7] = 0x8 (computed pointer)
- v[8:9] = 0x0 (base should be valid pointer)
- v[10:11] = 0x1 (count/index)
- Result: tries to load from address 0x8 (invalid)

**RING crash at `ncclDevFunc_AllReduce_RING_LL_Sum_f32_0_0_1()+808`:**
- Similar pattern with NULL pointer dereference

### Theory
The specialized kernels expect `ncclShmem.groups[].recvConns[]` and `sendConns[]` to contain valid pointers to `ncclConnInfo` structures. In the device linker build, these are NULL or garbage.

This could be caused by:
1. Structure layout mismatch (partially fixed, but may still exist)
2. Different code path between single-GPU and multi-GPU
3. Connection setup issue specific to device linker build
4. Missing initialization in kernel launch path

## Next Steps

1. Verify all structure sizes match between dispatcher and specialized kernels
2. Check if connection pointers are populated correctly for multi-GPU
3. Compare device linker vs system RCCL memory layout at runtime
4. Investigate if ENABLE_LL128 actually affects structure layout

## File Changes Summary

```
projects/rccl/CMakeLists.txt                       |  17 ++-
projects/rccl/src/include/device.h                 |   8 ++
projects/rccl/tools/device_linker/CMakeLists.txt   |   9 ++
projects/rccl/tools/device_linker/build_with_device_linker.sh | 8 +-
projects/rccl/tools/device_linker/smoke_test/test_two_gpu.cpp | 36 ++++--
```
