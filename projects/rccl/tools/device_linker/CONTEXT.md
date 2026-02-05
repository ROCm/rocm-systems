# Device Linker Context

## Environment

| | |
|---|---|
| **Machine** | MI300A (ringo) |
| **GPU Target** | gfx942 |
| **GPUs Available** | 4 |
| **ROCm Version** | 7.0 |
| **Build Path** | `/work/lmeadows/rocm-systems/projects/rccl/build/release` |

## Current Status

| Test | Status |
|------|--------|
| Single-GPU | WORKING |
| Multi-GPU | **MEMORY FAULT** (illegal memory access in kernel) |
| System RCCL Multi-GPU | WORKING |

## Current Problem: Multi-GPU Memory Fault

**Symptoms:**
- Single-GPU AllReduce works correctly
- Multi-GPU AllReduce crashes with "illegal memory access" during kernel execution
- Error reported at `hipStreamSynchronize()` but fault occurs in the AllReduce kernel
- This is PROGRESS from previous "infinite hang" - now the function dispatch appears to be working

**Likely Root Cause:**
Connection pointers in `ncclShmem.groups[].recvConns[]` and `sendConns[]` are NULL or invalid.
These are populated from `channel->peers[peer]` in the Primitives constructor.

**Key code path (`prims_simple.h`):**
```cpp
if (flags & (RoleWaitRecv|RolePostRecv)) loadRecvConn(channel->peers[peer], ...);
if (flags & (RoleWaitSend|RolePostSend)) loadSendConn(channel->peers[peer], ...);
```

**Confirmed crash location (from GDB):**
```
Thread 19 "ncclDevKern-c14f" received signal SIGSEGV, Segmentation fault.
0x00007fbfa4763eec in ?? () at hipify/src/device/prims_ll.h:670
670	      loadRecvConn(&channel->peers[recvPeers[nrecv]]->recv[connIndexRecv], nrecv);
```

**Next debugging step:**
Rebuild with the new DWARF merging code, then use `rocgdb` to inspect the actual pointer
values at the crash site. The new debug info should show function names (not just `??`).

## Build & Test

```bash
# Full rebuild
cd /work/lmeadows/rocm-systems/projects/rccl
rm -rf build
./install.sh --amdgpu_targets=gfx942 --device-linker

# Test single-GPU (should pass)
cd tools/device_linker/smoke_test
./run_tests.sh test_single_gpu

# Test multi-GPU (currently crashes with memory fault)
./run_tests.sh test_two_gpu_simple
```

## Critical Constraint: Compilation Flags

All compilation units MUST have identical flags for consistent structure layouts:

| Flag | Purpose | Affects Layout |
|------|---------|----------------|
| `DEVICE_LINKER` | Function table dispatch | No |
| `ENABLE_FAULT_INJECTION` | Adds faults field to ncclShmemData | **Yes** |
| `ENABLE_WARP_SPEED` | Adds warpComm/warpChannel fields | **Yes** |
| `ENABLE_LL128` | Protocol selection | Possibly |

If structure layouts mismatch between host, dispatcher, and specialized kernels, 
the kernel will access wrong memory offsets and crash or hang.

## Key Documents

| Document | Purpose |
|----------|---------|
| `DEVICE_LINKER_REDESIGN.md` | **Primary design doc** - architecture, ELF layout, implementation phases, lessons learned, common mistakes |
| `LDS_LAYOUT.md` | Shared memory layout reference - actual offsets from disassembly analysis |
| `IFC_VS_DEVICE_LINKER_COMPARISON.md` | Detailed ELF comparison with production build |
| `BUILD_PROCESS.md` | Build pipeline diagrams and data flow |

## Key Source Files

| File | Purpose |
|------|---------|
| `tools/device_linker/device_linker.cpp` | The device linker tool - merges ELFs, patches debug info |
| `tools/device_linker/func_ptr_test.hip` | Simple test showing how function pointer tables work in GPU code |
| `src/device/common.cu` | Dispatcher kernels (`ncclDevKernel_Generic_*`) |
| `src/device/common.h` | Function table declarations, dispatch logic |
| `src/device/prims_simple.h` | Primitives constructor - where connection pointers are loaded |
| `src/device/prims_ll.h` | LL protocol primitives |
| `src/device/generate_specialized.py` | Generates specialized kernel source files |

## Recently Fixed Issues (Feb 5, 2026)

### 1. Device Linker Not Auto-Built
**Problem:** The `device_linker` tool wasn't being built automatically, causing builds to
fail silently or use stale binaries.
**Fix:** Modified `CMakeLists.txt` to add an `add_executable` for `device_linker_tool`:
```cmake
add_executable(device_linker_tool ${DEVICE_LINKER_DIR}/device_linker.cpp)
target_compile_features(device_linker_tool PRIVATE cxx_std_17)
set_target_properties(device_linker_tool PROPERTIES
  OUTPUT_NAME device_linker
  RUNTIME_OUTPUT_DIRECTORY ${DEVICE_LINKER_DIR}
)
```
**Result:** The device linker is now built automatically as part of the main build.

### 2. Proper DWARF Merging (Replacing Synthetic Debug Info)
**Problem:** GDB showed line numbers but not function names ("No function contains program counter").
The previous `buildSyntheticDebugInfo()` created minimal compile units that only had address
ranges and `stmt_list` pointers, but no `DW_TAG_subprogram` DIEs for function names.
**Root cause:** The synthetic approach discarded the actual DWARF from each kernel, which
contains `DW_TAG_subprogram` and `DW_TAG_inlined_subroutine` DIEs with function names.
**Fix:** Replaced `buildSyntheticDebugInfo()` with `mergeDebugInfo()` that does proper
linker-style DWARF merging:
1. Merges `.debug_abbrev`, `.debug_str`, `.debug_str_offsets`, `.debug_addr`, 
   `.debug_rnglists`, `.debug_info` from each kernel
2. Patches addresses in `.debug_addr` based on code relocation delta
3. Patches CU header `debug_abbrev_offset` in `.debug_info`
4. Tracks base offsets for cross-section references

**Key data structure:**
```cpp
struct DebugInfoChunk {
    size_t merged_offset;      // Offset in merged .debug_info
    size_t size;               // Size of this chunk
    uint64_t orig_text_addr;   // Original .text address
    uint64_t new_text_offset;  // New offset in merged .text
    size_t abbrev_base;        // Base offset in merged .debug_abbrev
    size_t str_base;           // Base offset in merged .debug_str
    size_t addr_base;          // Base offset in merged .debug_addr
    size_t line_base;          // Base offset in merged .debug_line
    // ... etc
};
```

**Result:** GDB should now resolve function names like `loadRecvConn`, `Primitives`, etc.,
and show the full inline call chain.

## Fixed Issues (Feb 4-5, 2026)

### 3. Debug Line Numbers for Specialized Kernels
**Problem:** `llvm-symbolizer` and GDB showed `??:0:0` for specialized kernel addresses,
even though `.debug_line` contained correct line info.
**Root cause:** The synthetic `.debug_info` was missing or had wrong address ranges.
The device linker was calling `buildSyntheticDebugInfo()` before `text_addr_` was set,
resulting in compile units with addresses starting at 0x0 instead of the actual text address.
Also missing `DW_AT_comp_dir` attribute for GDB source file lookup.
**Fix:** In `device_linker.cpp`:
1. Move `buildSyntheticDebugInfo()` call to after `computeLayout()` in `link()`
2. Add `DW_AT_comp_dir` attribute with absolute path (via `realpath()`) to each compile unit
3. Each CU covers a specific code range and points to its correct `.debug_line` offset

**Result:** Line numbers now resolve correctly:
```
$ echo "0xa361ac" | llvm-symbolizer -e build/release/device_linker_output/merged_device.elf
ncclDevFunc_AllReduce_RING_LL_Sum_f32_0_0_2()
/work1/.../build/release/hipify/gensrc/specialized/specialized_all_reduce_ring_ll_sum_f32_unroll2.cpp:13:0
```

### 4. LDS Allocation for ncclShmemPerWarp
**Problem:** Static LDS allocation was 37776 bytes, exceeding available dynamic LDS.
**Root cause:** `#if __CUDA_ARCH__ >= 700` was false on AMD GPUs, causing static allocation.
**Fix:** Modified `src/device/common.h` to use `extern __shared__` for dynamic allocation
when `DEVICE_LINKER` is defined:
```cpp
#if __CUDA_ARCH__ >= 700 || defined(DEVICE_LINKER)
  extern __shared__ ulong2 ncclShmemPerWarp[];
#else
  // static allocation fallback
#endif
```
**Result:** Static LDS reduced from 37776 to 4944 bytes, dynamic allocation works.

### 5. Malformed RELRO Segment
**Problem:** Third LOAD segment had offset=0, vaddr=0, size=0x79ad0b0 (overlapping everything).
**Root cause:** `.data.rel.ro` section doesn't exist, so `relro_start=0`.
**Fix:** In `device_linker.cpp`, use `.dynamic` address if no `.data.rel.ro`:
```cpp
uint64_t relro_start = data_rel_ro_start ? data_rel_ro_start : dyn_sec_addr;
```

### 6. Function Table Addend Verification
**Investigation:** Suspected wrong addends in R_AMDGPU_RELATIVE64 relocations.
**Finding:** Addends are CORRECT - they match function symbol addresses.
For example, funcId 622 uses the unroll=2 variant:
- `_Z43ncclDevFunc_AllReduce_RING_LL_Sum_f32_0_0_2v` at 0xa361ac
- Relocation addend: 0xa361ac ✓
**Conclusion:** The function dispatch mechanism is working correctly.

## Previously Fixed Issues

These are documented in detail in `DEVICE_LINKER_REDESIGN.md`:

1. COMGR "Cannot Find Global Var Sizes" - `__clang_gpu_used_external` symbol
2. PC-relative addressing - preserved .rodata layout
3. Helper function extraction - complete code blocks
4. ENABLE_WARP_SPEED mismatch - consistent flags
5. Shared memory limit - `extern __shared__` for dynamic allocation
6. DWARF5 debug line string offsets - `patchDwarf5StringOffsets()`
7. Specialized kernel symbol addresses - `func_offset` correction
8. ~~Synthetic `.debug_info`~~ - **Replaced** by proper DWARF merging (see issue #2 above)

## Understanding Function Pointer Tables

The file `func_ptr_test.hip` demonstrates how GPU function pointer tables work:

1. **PC-relative addressing** finds the table:
```asm
s_getpc_b64 s[8:9]              # Get current PC
s_add_u32 s8, s8, <offset>      # Add offset to table
```

2. **Table contents** filled by R_AMDGPU_RELATIVE64 relocations at load time

3. **Indirect call** via `s_swappc_b64`:
```asm
s_load_dwordx2 s[18:19], s[6:7], 0x0   # Load function pointer
s_swappc_b64 s[30:31], s[18:19]        # Call function
```

The device linker replicates this pattern correctly - verified by comparing
with the simple test ELF.
