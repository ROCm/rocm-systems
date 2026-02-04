# Device Linker Context - February 3, 2026

## Environment

**Machine:** MI300A (ringo)  
**GPU Target:** gfx942  
**GPUs Available:** 4  
**ROCm Version:** 7.0  
**Build Path:** /work/lmeadows/rocm-systems/projects/rccl/build/release

## Current Status

| Test | Status |
|------|--------|
| Single-GPU | WORKING |
| Multi-GPU | HANGING (kernel never completes) |
| System RCCL Multi-GPU | WORKING |
| Shared Memory | FIXED (see fix #5 below) |

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

### 5. Fixed Shared Memory Declaration for Dynamic Scratch

**File:** `src/device/common.h` (and hipified version)

**Problem:** In DEVICE_LINKER mode, `ncclShmemPerWarp` was declared as `__shared__` (static), 
consuming 37776 bytes of the device's 65536 byte shared memory limit. This left only 27760 
bytes for dynamic allocation, but the host code expected 32832 bytes of dynamic shared memory.

**Error:** `cudaArch 940 ncclMaxSharedMem 32832 exceeds device/fn maxSharedMem 27760`

**Fix:** Changed `ncclShmemPerWarp` to use `extern __shared__` (dynamic allocation) for 
CUDA arch >= 700, matching the standard build behavior:

```cpp
#if __CUDA_ARCH__ >= 700 || NCCL_CUDA_ARCH >= 700
  // Dynamic scratch - use extern __shared__ with incomplete size for runtime allocation
  extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
  // Static scratch for older archs
  NCCL_SHMEM_DECL ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif
```

**Also updated:** Device linker now calculates and logs required LDS size based on GPU arch.

### 6. Fixed DWARF5 Debug Line String Offsets

**File:** `tools/device_linker/device_linker.cpp`

**Problem:** When merging specialized kernel debug info, the device linker was concatenating 
`.debug_line_str` sections but not adjusting the string offset references in `.debug_line`. 
DWARF5 uses `DW_FORM_line_strp` (4-byte offsets into `.debug_line_str`) for file/directory names.
After merging, these offsets pointed to wrong strings, causing corrupted file names in debugger output.

**Symptoms:** `llvm-dwarfdump --debug-line` showed corrupted file names like:
```
name: "m-systems/projects/rccl/build/..."  (truncated)
name: "se/hipify/gensrc/specialized/..."   (wrong offset)
```

**Fix:** Added `patchDwarf5StringOffsets()` function that:
1. Parses DWARF5 line table prologues to find directory and file name entries
2. Identifies entries using `DW_FORM_line_strp` form
3. Adjusts each string offset by `str_offset_base` (cumulative offset from earlier kernels)
4. Properly handles all DWARF forms including `DW_FORM_data16` (MD5 checksums)

### 7. Fixed Specialized Kernel Symbol Addresses

**File:** `tools/device_linker/device_linker.cpp`

**Problem:** Symbol table entries for specialized kernels (`ncclDevFunc_*`) pointed to the 
start of the extracted code block rather than the actual function. Each extracted block 
includes helper functions (like `__ockl_fprintf_append_string_n`, `__assert_fail`, `runRing`) 
before the `ncclDevFunc_` entry point.

**Impact:** Debuggers (like `rocgdb`) would show incorrect addresses and couldn't properly 
correlate source lines with the actual function code.

**Fix:** Changed symbol value calculation from:
```cpp
sym.st_value = text_addr_ + text_off;                    // Wrong: code block start
sym.st_size = kern->code.size();                         // Wrong: whole block size
```
to:
```cpp
sym.st_value = text_addr_ + text_off + kern->func_offset; // Correct: actual function
sym.st_size = kern->code.size() - kern->func_offset;      // Correct: function size only
```

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

## Current Issue: Multi-GPU Kernel Hang

### Symptoms
- Single-GPU AllReduce works correctly
- Multi-GPU AllReduce hangs indefinitely (kernel never completes)
- Initialization succeeds, channels are connected, kernel is launched
- Hang occurs during `hipStreamSynchronize()`

### Progress (This Session)
- ✓ Fixed shared memory issue (see fix #5)
- ✓ Initialization now completes successfully
- ✓ Channels connect via P2P/direct pointer
- ✓ Kernel launches (no longer crashes on launch)
- ✗ Kernel hangs during execution (never completes)

### Likely Root Cause
The kernel hangs because connection pointers accessed from `ncclShmem.groups[].recvConns[]` 
and `sendConns[]` are NULL or invalid. These pointers are populated in the Primitives 
constructor from `channel->peers[peer]`, which comes from `ncclShmem.channel.peers`.

**Key code path (prims_simple.h):**
```cpp
// In Primitives constructor:
if (flags & (RoleWaitRecv|RolePostRecv)) loadRecvConn(channel->peers[peer], ...);
if (flags & (RoleWaitSend|RolePostSend)) loadSendConn(channel->peers[peer], ...);
```

The `peers` array is copied from host memory during kernel initialization, but may not 
contain valid pointers in the device linker build.

### Debug Instrumentation Added
Added `DEBUG_PEER_POINTERS` compile flag to `prims_simple.h` and `prims_ll.h` that prints:
- `channel->peers` pointer value
- Peer index being accessed
- `channel->peers[peer]` value
- Connection info fields

**Note:** This debug output increases shared memory usage significantly and may not work 
without further LDS adjustments.

## Previously Fixed Issues

1. **COMGR "Cannot Find Global Var Sizes"** - Fixed by adding `__clang_gpu_used_external` symbol
2. **PC-relative addressing** - Fixed by preserving .rodata layout for function tables
3. **Helper function extraction** - Device linker now extracts complete code blocks
4. **ENABLE_WARP_SPEED mismatch** - All units now compile with same flag
5. **Debug info merging** - Added .debug_ranges section handling
6. **Shared memory limit exceeded** - Fixed `ncclShmemPerWarp` to use dynamic allocation
7. **DWARF5 debug line string offsets** - Fixed merging of `.debug_line_str` sections (see fix #6 below)
8. **Specialized kernel symbol addresses** - Fixed to point to actual function, not code block start (see fix #7 below)

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
