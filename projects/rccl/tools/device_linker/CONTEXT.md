# Device Linker Context

## Environment

| | |
|---|---|
| **Machine** | MI300A (ringo) |
| **GPU Target** | gfx942 |
| **GPUs Available** | 4 |
| **ROCm Version** | 7.0 |
| **Build Path** | `/work2/lmeadows/rocm-systems/projects/rccl/build/release` |

## Current Status

| Test | Status |
|------|--------|
| Single-GPU | WORKING |
| Multi-GPU | WORKING |
| System RCCL Multi-GPU | WORKING |
| Smoke Test | PASSED |

## Key Lesson Learned

**Most problems were caused by the dispatcher not having the same defines as the specialized 
kernels.** Structure layout mismatches (from missing `ENABLE_COLLTRACE`, `ENABLE_PROFILING`, 
etc.) caused crashes, hangs, and incorrect memory access. See "Critical Constraint: 
Compilation Flags" section below.

## DWARFLinker Integration (Current)

The device linker now uses **LLVM's DWARFLinker** (`llvm::dwarf_linker::classic::DWARFLinker`) to merge DWARF debug information from dispatcher and specialized kernels. This replaces the previous manual patching approach and ensures correct DWARF5 format compliance.

- **Implementation:** `mergeDebugInfoWithDWARFLinker()` uses `CustomStreamer` to capture merged DWARF sections
- **Status:** DWARF sections are merged correctly, but some issues remain (see Known Issues below)

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

### DWARF Debug Info
- DWARF sections pass `llvm-dwarfdump --verify` but some issues may remain with rocgdb compatibility
- `abbr_offset = 0x0000` problem in DWARF CU headers - investigation ongoing

### Symbol-Based Relocations (In Progress)
- Converting relocations from absolute addresses to symbol indices (`R_AMDGPU_ABS64` with symbol index)
- Adding `ncclDevFunc_*` and oneRankReduce symbols to `.dynsym` as LOCAL HIDDEN (matching specialized kernel `.o` files)
- Address calculation verification needed to ensure symbol addresses match function table addresses


## Build & Test

```bash
# Full rebuild
cd /work2/lmeadows/rocm-systems/projects/rccl
rm -rf build
./install.sh -l --device-linker

# Rebuild device linker only (after code changes)
cd tools/device_linker
./build.sh

# Test single-GPU (should pass)
cd smoke_test
./run_tests.sh test_single_gpu

# Test multi-GPU
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
| `DWARF_FIX_PROGRESS.md` | DWARF5 line table patching for rocgdb compatibility |
| `DWARFLINKER_STATUS.md` | Status of DWARFLinker integration |

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

## Fixed Issues

### Structure Layout Mismatches
- **ENABLE_COLLTRACE/ENABLE_PROFILING propagation:** Fixed multi-GPU crashes caused by `ncclShmemData` layout mismatches. Flags now properly propagated to all compilation units.
- **ENABLE_WARP_SPEED mismatch:** Ensured consistent flags across dispatcher and specialized kernels.

### DWARF Debug Information
- **DWARFLinker integration:** Replaced manual DWARF patching with LLVM's DWARFLinker for proper DWARF5 merging
- **Debug line numbers:** Fixed line number resolution for specialized kernels
- **Function names:** Proper DWARF merging ensures function names resolve in GDB

### ELF Structure
- **LDS allocation:** Fixed static LDS allocation exceeding limits by using dynamic allocation
- **RELRO segment:** Fixed malformed RELRO segment when `.data.rel.ro` doesn't exist
- **Function tables:** Verified function pointer table relocations are correct

### Build System
- **Device linker build:** Device linker must be built manually via `tools/device_linker/build.sh` before running `install.sh --device-linker`

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

---

## Current Work Status (Feb 8, 2026)

### Symbol-Based Relocations
**Goal:** Convert relocations from absolute addresses to symbol indices for better compatibility and correctness.

**Recent Changes:**
- Added `ncclDevFunc_*` symbols to `.dynsym` as LOCAL HIDDEN (matching specialized kernel `.o` files)
- Added oneRankReduce symbols to `.dynsym` as LOCAL HIDDEN
- Updated `sh_info` calculation to correctly mark LOCAL/GLOBAL boundary
- Enhanced diagnostics to show which symbols are missing from `.dynsym`

**Current Issue:**
- Address mismatch: Symbols added to `.dynsym` have different addresses than those referenced in function tables
- Need to verify address calculation: `text_addr_ + text_off + kern->func_offset` vs `text_addr_ + table_2_[i]`

**Next Steps:**
- Verify address calculation consistency
- Ensure all function table entries have corresponding symbols in `.dynsym`
- Test symbol-based relocations with `R_AMDGPU_ABS64` type

---

## Session Summary (Feb 11-12, 2026)

### Code Cleanup: Dead DWARF Code Removal

Removed **~1071 lines** (~21%) of dead code from `device_linker.cpp`. This code was related to manual DWARF parsing and debug section merging that was superseded by the DWARFLinker integration.

**Removed:**
- `findLineStrpPositionsInDebugInfo()` - never called
- `findDwarfAttrPositionsFromElf()` and `findDwarfAttrPositionsUsingLLVM()` - populated data that was never used
- `DwarfAttrPositions` struct - data never consumed
- `createMinimalElfForDwarf()` and `createMinimalElfForLineTable()` - only called by dead code
- `patchStrOffsetsBaseToZero()` - only called by dead code
- `parseAbbrevTableWithAttrs()` and `parseAbbrevTable()` - only used by dead code
- `findLineStrpInChunkManual()` - only used by dead code
- `getFormFixedSize()`, `skipVariableForm()` - DWARF form parsing helpers, now dead
- `appendULEB128()`, `decodeULEB128()`, `decodeSLEB128()` - ULEB/SLEB helpers, now dead
- `patchDebugLine()` - was a no-op when DWARFLinker enabled
- Large block of manual debug section merging in `collectSections()` - superseded by DWARFLinker

**Simplified:**
- `DebugInfoChunk` struct now only contains `orig_text_addr` and `new_text_offset` (essential for DWARFLinker address remapping)

Build verified successful after cleanup.

### Test Investigation: AllReduce Verification Failure

Investigated `test_smoke_allreduce_batched` which was failing. This led to discovering a **fundamental issue**: AllReduce operations in the device linker build launch successfully but don't actually compute results (output buffers remain unchanged).

**Key Finding:** `test_minimal_allreduce` does NOT verify results - it only checks that the operation doesn't crash or hang. This violates the completion criteria requirement that tests should verify correctness.

**Symptoms:**
- Kernel launches with correct funcId (e.g., 368 for AllReduce_TREE_LL_Sum_f32)
- Kernel "completes" (hipStreamSynchronize returns)
- Output buffers contain 0 instead of expected sum
- System RCCL (non-device-linker) works correctly with same test

**Ruled Out:**
- Define mismatches (CMake generates correct flags for both dispatcher and specialized kernels)
- LDS allocation (matches between dispatcher and specialized functions)
- Function code existence (17KB of real code with LDS operations after s_trap patching)
- Algorithm/protocol specifics (Ring, Tree, LL, LL128, Simple all fail)

**Root Cause (identified by user via rocgdb):** Missing noinline problem - related to function inlining behavior in specialized kernels.

### Action Items for Next Session

1. **Fix noinline issue** - User has identified the specific problem through rocgdb debugging
2. **Add result verification to test_minimal_allreduce** - Current test is not a valid completion test
3. **Consider moving dispatcher build steps 3-5 from shell script to CMake** - Would ensure flag consistency automatically (though the immediate issue was not a flag mismatch)
