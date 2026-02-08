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
| Multi-GPU | WORKING |
| System RCCL Multi-GPU | WORKING |
| Smoke Test | PASSED |
| Multi-GPU Unit Tests | IN PROGRESS - some failures |

## Key Lesson Learned

**Most problems were caused by the dispatcher not having the same defines as the specialized 
kernels.** Structure layout mismatches (from missing `ENABLE_COLLTRACE`, `ENABLE_PROFILING`, 
etc.) caused crashes, hangs, and incorrect memory access. See "Critical Constraint: 
Compilation Flags" section below.

## Phase 3 (DWARF emission) — Partially implemented

Phase 3 of the DWARF rewrite (see `tools/device_linker/DWARF_LLVM_REWRITE_PLAN.md`) will switch to **emission** for merged `.debug_*` sections so line_strp and addresses are correct by construction.

- **Phase 3(a) — Implemented:** When `kUsePhase3Emission` is `true` (in `device_linker.cpp`, default `false`), `mergeDebugSectionsViaLLVMEmission()` re-emits each `.debug_line` chunk with `line_strp` and `DW_LNE_set_address` patched into the output buffer instead of concatenating raw bytes and patching in place. This removes the format-driven prologue walk and avoids desync issues (e.g. `DW_FORM_flag_present`). Entry point: `mergeDebugSectionsViaLLVMEmission()` rebuilds `.debug_line` via `emitDebugLineChunk()` per chunk, then calls `mergeDebugInfo()`; `patchDebugLine()` is skipped when Phase 3 is enabled.
- **(b) .debug_info:** clone+emit or patch-from-LLVM — not yet done.
- **(c) Optional DWARFLinker** — not yet done.

## Device-Only Binaries (Feb 2026)

Specialized kernels and the device linker now use **device-only** objects instead of host fat binaries:

- **Specialized kernels:** Built with `compile_specialized_device.sh` (clang -cc1 -fcuda-is-device -emit-obj) → output `*.device.o` (raw amdgpu ELF). No host code, no fat binary.
- **Main CMakeLists.txt:** Step 1 compiles each specialized kernel via the script → `specialized_objs/${name}.device.o`.
- **Device linker:** `--input-dir` accepts only `*.device.o`; reads each file as the device ELF directly (no extraction from host .o via llvm-objcopy/clang-offload-bundler).
- **Standalone build:** `tools/device_linker/CMakeLists.txt` also builds specialized kernels as `.device.o` via the script.

Dispatcher is already built as device ELF in `build_with_device_linker.sh` (Steps 1–2); no change there.

## LLVM/DWARF Debugger Messages and Fixes (Feb 2026)

When parsing DWARF (e.g. during kernel parsing or merge), LLVM can emit many non-fatal errors/warnings. The device linker now **exits on any such message** so the build fails instead of continuing.

### Messages seen (from LLVM)

- **`error: invalid reference to or invalid content in .debug_str_offsets[.dwo]: insufficient space for 32 bit header prefix`**  
  Caused when LLVM parses DWARF and finds `DW_AT_str_offsets_base` but the `.debug_str_offsets` section is missing or smaller than 8 bytes (DWARF5 header). Fix: (1) **Minimal ELF:** `createMinimalElfForDwarf()` now includes a `.debug_str_offsets` section with an 8-byte valid DWARF5 header so LLVM can parse without error. (2) **Merging:** When appending dispatcher or kernel `.debug_str_offsets`, if size &lt; 8 use the same 8-byte header; if the chunk has no section at all, append 8 bytes so the merged section stays parseable.

- **`error: invalid reference to or invalid content in .debug_str_offsets[.dwo]: section offset exceeds section size`**  
  Occurs when LLVM parses a minimal ELF that has `.debug_info` with `DW_AT_str_offsets_base` but no (or undersized) `.debug_str_offsets`; e.g. 807× during kernel parsing. The device linker uses minimal ELFs (`.debug_info` + `.debug_abbrev` only) for attribute finding; LLVM still validates cross-section references and reports this.

### Fixes applied

1. **Exit on LLVM DWARF errors/warnings:** Custom handlers `llvmDwarfErrorHandler(llvm::Error)` and `llvmDwarfWarningHandler(llvm::Error)` (signature required by `DWARFContext::create`). They set `g_llvm_dwarf_error` and log the error; after kernel parsing we check the flag and exit with status 1 so the build fails.
2. **`.debug_str_offsets` padding:** When merging, if a chunk’s size is &lt; 8 bytes, get an 8-byte valid header (minimal ELF also includes .debug_str_offsets for parsing).
3. **Phase 3 off:** `kUsePhase3Emission = false` to avoid the re-emission path that was triggering `.debug_str_offsets` issues.

## Known Issues (Feb 8, 2026)

### DWARF Info Still Has Issues
There is probably still something wrong with the DWARF debug info. For example, when 
trying to print the address of `ncclShmem` in rocgdb, errors occur. The DWARF sections
pass `llvm-dwarfdump --verify` but rocgdb may have stricter requirements.

**Recent fix attempt:** Changed dispatcher compilation to use `-dwarf-version=5` to ensure
consistent DWARF5 across all compilation units (dispatcher was previously DWARF4, specialized
kernels were DWARF5). This resolved the `DW_FORM_line_strp pointing outside of .debug_line_str`
error from llvm-dwarfdump, but rocgdb issues may persist.

## Recent Fix: Multi-GPU Memory Fault (Feb 5, 2026)

The multi-GPU memory fault was caused by an `ncclShmemData` structure layout mismatch.

**Root Cause:**
The `COLLTRACE` CMake option is `ON` by default, which defines `ENABLE_COLLTRACE` for the
host library build. This adds two pointer fields (`collTrace`, `collTraceTail`) to 
`ncclShmemData` - 16 bytes total. However, these flags were NOT being propagated to:
1. The dispatcher compilation (`build_with_device_linker.sh`)
2. The specialized kernel compilation

This caused all field offsets after `devicePlugin` in `ncclShmemData` to be wrong by 16 bytes,
leading to NULL pointer dereferences when accessing `channel->peers[peer]`.

**Fix:**
Added `ENABLE_COLLTRACE` and `ENABLE_PROFILING` propagation to all device compilation units:
- `CMakeLists.txt`: Added to `SPEC_KERNEL_DEFS` and environment variables for dispatcher build
- `build_with_device_linker.sh`: Added handling for these environment variables
- `build_skeleton_dispatcher.sh`: Same
- `tools/device_linker/CMakeLists.txt`: Added corresponding options

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
| `ENABLE_FAULT_INJECTION` | Adds `faults` field to ncclShmemData | **Yes** |
| `ENABLE_WARP_SPEED` | Adds `warpComm`/`warpChannel` fields | **Yes** |
| `ENABLE_LL128` | Protocol selection | Possibly |
| `ENABLE_COLLTRACE` | Adds `collTrace`/`collTraceTail` fields | **Yes** |
| `ENABLE_PROFILING` | Adds `prof` field to ncclShmemData | **Yes** |

If structure layouts mismatch between host, dispatcher, and specialized kernels, 
the kernel will access wrong memory offsets and crash or hang.

These flags are now properly propagated via CMake and environment variables to ensure
consistency across all compilation units.

## Key Documents

| Document | Purpose |
|----------|---------|
| `DEVICE_LINKER_REDESIGN.md` | **Primary design doc** - architecture, ELF layout, implementation phases, lessons learned, common mistakes |
| `LDS_LAYOUT.md` | Shared memory layout reference - actual offsets from disassembly analysis |
| `IFC_VS_DEVICE_LINKER_COMPARISON.md` | Detailed ELF comparison with production build |
| `BUILD_PROCESS.md` | Build pipeline diagrams and data flow |
| `DWARF_FIX_PROGRESS.md` | **IN PROGRESS** - DWARF5 line table patching for rocgdb compatibility |

## Key Source Files

| File | Purpose |
|------|---------|
| `tools/device_linker/device_linker.cpp` | The device linker tool - merges ELFs, patches debug info |
| `tools/device_linker/compile_specialized_device.sh` | Compiles one specialized kernel source to device-only `.device.o` (no host fat binary) |
| `tools/device_linker/func_ptr_test.hip` | Simple test showing how function pointer tables work in GPU code |
| `src/device/common.cu` | Dispatcher kernels (`ncclDevKernel_Generic_*`) |
| `src/device/common.h` | Function table declarations, dispatch logic |
| `src/device/prims_simple.h` | Primitives constructor - where connection pointers are loaded |
| `src/device/prims_ll.h` | LL protocol primitives |
| `src/device/generate_specialized.py` | Generates specialized kernel source files |

## Recently Fixed Issues (Feb 5, 2026)

### 1. ENABLE_COLLTRACE/ENABLE_PROFILING Not Propagated to Device Code
**Problem:** Multi-GPU AllReduce crashed with "illegal memory access" (SIGSEGV) at 
`channel->peers[peer]` in `prims_ll.h:670`.
**Root cause:** The `COLLTRACE` CMake option (ON by default) defines `ENABLE_COLLTRACE`
for the host library, adding 16 bytes (`collTrace`, `collTraceTail` pointers) to
`ncclShmemData`. These flags were NOT passed to dispatcher or specialized kernel builds,
causing a structure layout mismatch - all fields after `devicePlugin` had wrong offsets.
**Fix:** 
- Added `ENABLE_COLLTRACE` and `ENABLE_PROFILING` to `SPEC_KERNEL_DEFS` in main `CMakeLists.txt`
- Added environment variable propagation to `build_with_device_linker.sh`
- Updated `build_skeleton_dispatcher.sh` and `tools/device_linker/CMakeLists.txt`
**Result:** Multi-GPU AllReduce now works correctly.

### 2. Device Linker Not Auto-Built
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

### 3. Proper DWARF Merging (Replacing Synthetic Debug Info)
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

### 4. Debug Line Numbers for Specialized Kernels
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

### 5. LDS Allocation for ncclShmemPerWarp
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

### 6. Malformed RELRO Segment
**Problem:** Third LOAD segment had offset=0, vaddr=0, size=0x79ad0b0 (overlapping everything).
**Root cause:** `.data.rel.ro` section doesn't exist, so `relro_start=0`.
**Fix:** In `device_linker.cpp`, use `.dynamic` address if no `.data.rel.ro`:
```cpp
uint64_t relro_start = data_rel_ro_start ? data_rel_ro_start : dyn_sec_addr;
```

### 7. Function Table Addend Verification
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
8. ~~Synthetic `.debug_info`~~ - **Replaced** by proper DWARF merging (see issue #3 above)

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
