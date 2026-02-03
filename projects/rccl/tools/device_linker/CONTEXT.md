# Device Linker Context - February 3, 2026

## Environment

**Machine:** MI350X (cv350-zts-gtu-e11-18)  
**GPU Target:** gfx950  
**GPUs Available:** 8  
**ROCm Version:** 7.0  
**Build Path:** /work/lmeadows/rocm-systems/projects/rccl/build/release

## Current Status

| Test | Status |
|------|--------|
| Single-GPU | WORKING |
| Multi-GPU | CRASHING (Memory Fault Error) |
| System RCCL Multi-GPU | WORKING |
| LDS Size | 73568 bytes (correct) |

## Changes Made This Session (Feb 3, 2026)

### 1. Fixed NCCL_CUDA_ARCH for AMD GPUs

**File:** `src/include/device.h`

**Problem:** `NCCL_CUDA_ARCH` evaluated to 0 on AMD device compilation because `__CUDA_ARCH__` is CUDA-specific.

**Impact:** Caused `ncclCollUnroll()` to return wrong values, leading to incorrect LDS size calculations.

**Fix:**
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

### 2. Added ENABLE_WARP_SPEED to Specialized Kernels

**File:** `CMakeLists.txt`

**Problem:** Specialized kernels compiled without `ENABLE_WARP_SPEED`, causing structure layout mismatch.

**Fix:**
```cmake
if(ENABLE_WARP_SPEED)
  list(APPEND SPEC_KERNEL_DEFS -DENABLE_WARP_SPEED)
endif()
```

### 3. Added ENABLE_LL128 to Dispatcher Builds

**Files:** 
- `tools/device_linker/CMakeLists.txt`
- `tools/device_linker/build_with_device_linker.sh`

**Problem:** `ENABLE_LL128` was missing from dispatcher compilation.

**Fix:** Added `-DENABLE_LL128` to `DISPATCHER_DEFS` and build script.

### 4. Expanded WARP_SPEED_SUPPORTED_ARCHS

**File:** `CMakeLists.txt`

**Problem:** `sramecc+` variants used by device linker weren't in the supported list.

**Fix:** Added all variants including sramecc combinations.

## Compilation Flags (Single Source of Truth)

All compilation units MUST have these flags for consistent structure layouts:

| Flag | Purpose | Affects Layout |
|------|---------|----------------|
| `DEVICE_LINKER` | Function table dispatch | No |
| `ENABLE_FAULT_INJECTION` | Adds faults field to ncclShmemData | Yes |
| `ENABLE_WARP_SPEED` | Adds warpComm/warpChannel fields | Yes |
| `ENABLE_LL128` | Protocol selection | Possibly |

### Current Flag Status by Unit

| Unit | DEVICE_LINKER | ENABLE_FAULT_INJECTION | ENABLE_WARP_SPEED | ENABLE_LL128 |
|------|---------------|------------------------|-------------------|--------------|
| Host code (librccl.so) | ✓ | ✓ | ✓ | ✓ |
| Specialized kernels | ✓ | ✓ | ✓ | ✓ |
| Dispatcher (CMake) | ✓ | ✓ | ✓ | ✓ |
| Dispatcher (script) | ✓ | ✓ | ✓ | ✓ |

## Current Issue: Multi-GPU Memory Fault

### Symptoms
- Single-GPU AllReduce works correctly
- Multi-GPU AllReduce crashes with "Memory Fault Error"
- Both TREE and RING algorithms crash
- Both kernels launch successfully, then crash during execution

### Crash Details

**Location:** Specialized kernel functions (e.g., `ncclDevFunc_AllReduce_TREE_LL_Sum_f32_0_0_1`)

**Faulting Instruction:** `flat_load_dwordx2` trying to load from computed address

**Register State at Crash (TREE example, offset +5680):**
```
v[6:7] = 0x8        (computed address - invalid!)
v[8:9] = 0x0        (base pointer - NULL)
v[10:11] = 0x1      (index/count)
```

**Root Cause:** NULL connection pointers in `ncclShmemGroup`:
- Specialized kernels expect `ncclShmem.groups[].recvConns[]` and `sendConns[]` to be valid
- In device linker build, these are NULL during multi-GPU execution
- Single-GPU works because no inter-GPU communication is needed

### Investigation Status

1. ✓ All compilation flags are now consistent
2. ✓ LDS size is correct (73568 bytes)
3. ✓ Function table dispatch works correctly
4. ✗ Connection pointers are NULL during multi-GPU ops

## Previously Fixed Issues

1. **COMGR "Cannot Find Global Var Sizes"** - Fixed by adding `__clang_gpu_used_external` symbol
2. **PC-relative addressing** - Fixed by preserving .rodata layout for function tables
3. **Helper function extraction** - Device linker now extracts complete code blocks
4. **ENABLE_WARP_SPEED mismatch** - All units now compile with same flag
5. **Debug info merging** - Added .debug_ranges section handling

## Build Commands

```bash
# Full rebuild
cd /work/lmeadows/rocm-systems/projects/rccl/build/release
rm -rf specialized_objs device_linker_output
mkdir -p specialized_objs device_linker_output
make -j

# Test single-GPU (should pass)
cd tools/device_linker/smoke_test
LD_LIBRARY_PATH=/work/lmeadows/rocm-systems/projects/rccl/build/release ./test_single

# Test multi-GPU (currently crashes)
LD_LIBRARY_PATH=/work/lmeadows/rocm-systems/projects/rccl/build/release ./test_simple
```

## Next Steps

1. Compare ncclShmemGroup initialization between system RCCL and device linker build
2. Verify connection pointer setup in host code
3. Check if there's a code path difference for multi-GPU that device linker misses
4. Consider adding debug prints to track connection pointer population

## Key Documents

- `CHANGES_SUMMARY.md` - Detailed change log for this session
- `DEVICE_LINKER_DESIGN.md` - Full design and implementation details
- `DEVICE_LINKER_REDESIGN.md` - Original redesign document
