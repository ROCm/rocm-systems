# Device Linker Context - February 2, 2026

## Environment

**Machine:** MI350X (cv350-zts-gtu-e11-18)
**GPU Target:** gfx950
**GPUs Available:** 8

## Current Status

**Single-GPU:** WORKING  
**Multi-GPU:** HANGING (no crash, just never completes)  
**Original Issue:** COMGR error "Cannot Find Global Var Sizes" - FIXED  
**PC-relative fix:** IMPLEMENTED but multi-GPU still broken

## Progress Summary

### What's Fixed
1. **`__clang_gpu_used_external` symbol** - Now correctly generated
   - Symbol present at 0x517000, size 96 bytes (12 × 8-byte pointers)
   - 12 R_AMDGPU_RELATIVE64 relocations pointing to oneRankReduce kernels
   - Total 2589 relocations (2577 function table + 12 oneRankReduce)

2. **oneRankReduce kernel detection** - Device linker now finds all 12 kernels
   - Pattern matching on FUNC symbols with "oneRankReduce" in name
   - Excludes `.kd` kernel descriptor symbols and ABS metadata symbols

3. **Original COMGR error** - No longer occurs (was due to missing symbol)

### What's Still Broken - ROOT CAUSE FOUND
**Multi-GPU crashes with "illegal memory access"**

The crash occurs at PC=0x1a4400, which is the **unrelocated** ELF address of 
`ncclDevFunc_AllReduce_TREE_LL_Sum_f32_0_0_1v`. This happens because the 
dispatcher kernel reads a garbage function pointer and jumps to it.

## Root Cause Analysis

### Original Issue (FIXED)
Missing `__clang_gpu_used_external` symbol caused COMGR to fail with "Cannot Find Global Var Sizes".

### Current Issue - PC-RELATIVE ADDRESSING BROKEN

**Discovery:** The dispatcher kernel (`ncclDevKernel_Generic_1`) uses **PC-relative 
addressing** to find `ncclDevFuncTable_1`:

```asm
s_getpc_b64 s[0:1]                    ; Get current PC
s_add_u32 s0, s0, 0xffffa3bc          ; Add hardcoded offset (-23620)
s_addc_u32 s1, s1, -1                 ; Sign extend to 64-bit
...
s_load_dwordx2 s[0:1], s[0:1], s2     ; Load function pointer from table
...
s_swappc_b64 s[30:31], s[0:1]         ; Call the function
```

**The Problem:**
1. The offset `-23620` was calculated at **compile time** based on the original layout
2. The device linker **changed the relative positions** of:
   - Dispatcher code (ncclDevKernel_Generic_1)
   - Function table (ncclDevFuncTable_1)
3. The hardcoded offset now points to **wrong memory** (code instead of data)
4. GPU reads garbage, interprets it as pointer `0x1a4400`, and crashes

**Evidence from rocgdb:**
```
At s_swappc instruction:
  s0 = 0x1a4400   ; This is garbage read from wrong address!
  s1 = 0x0
  => s_swappc_b64 s[30:31], s[0:1]  ; Jump to 0x1a4400 - CRASH!
```

**Layout Comparison:**
```
IFC Layout:
  Dispatcher code at: 0xcc300
  Function table at:  0x68cfa90
  Distance: ~104 MB

Device Linker Layout:
  Dispatcher code at: 0x1ae00  
  Function table at:  0x511000
  Distance: ~5 MB
  
But code uses offset: -23620 bytes (pointing BACKWARDS!)
```

The PC-relative offset is baked into the machine code and doesn't match either layout
correctly. It appears to be pointing to some intermediate structure in the original
compilation that no longer exists in the merged ELF.

## Changes Made (Feb 2, 2026)

### device_linker.cpp
1. **Added `.data` copy logic** - Copies dispatcher's `.data` if present, else creates 0x60 byte section
2. **Added oneRankReduce collection** - Scans symbols for FUNC type in `.text` with "oneRankReduce" in name
3. **Added debug output** - Shows collection progress and matches found

### Key Code Changes
```cpp
// Collect oneRankReduce kernels (around line 745)
if (ELF64_ST_TYPE(syms[i].st_info) == STT_FUNC &&
    syms[i].st_shndx == text_idx &&
    strstr(name, "oneRankReduce") != nullptr &&
    strstr(name, ".kd") == nullptr) {
    onerank_text_offsets_.push_back(syms[i].st_value - disp_text_for_onerank->addr);
}

// Add __clang_gpu_used_external symbol (around line 1788)
if (!onerank_text_offsets_.empty()) {
    // Create symbol in .data pointing to array of kernel addresses
    sym.st_value = data_addr_;
    sym.st_size = onerank_text_offsets_.size() * 8;  // 8 bytes per pointer
}

// Add relocations (around line 2046)
for (size_t i = 0; i < onerank_text_offsets_.size(); i++) {
    addReloc(data_addr_ + i * 8, text_addr_ + onerank_text_offsets_[i]);
}
```

## Debugging Session (Feb 2, 2026)

### How the Root Cause Was Found

1. **Initial observation:** Multi-GPU test crashed with "illegal memory access"
2. **rocgdb investigation:** Crash at PC=0x1a4400 (unrelocated ELF address)
3. **Examined function table:** Host view showed correctly relocated pointers
4. **Set breakpoint at dispatcher:** Found `s_swappc_b64` instruction at offset +2900
5. **Examined registers at swappc:** s[0:1] = 0x1a4400 (garbage, not a valid pointer)
6. **Traced backwards:** Found PC-relative load from wrong address
7. **Compared layouts:** Device linker changed code-to-table distance, but offset is hardcoded

### Key rocgdb Commands Used
```bash
# Break at dispatcher kernel entry
break ncclDevKernel_Generic_1

# Find the indirect call (swappc)
disas /r $pc, $pc+4000 | grep swappc

# Break just before the call
break *((char*)ncclDevKernel_Generic_1 + 2900)

# Check what value is about to be jumped to
info registers s0 s1
```

## Verification

### Symbol Check (PASSING)
```bash
$ llvm-readelf --symbols /tmp/librccl_device.elf | grep clang_gpu
  2734: 0000000000517000    96 OBJECT  LOCAL  HIDDEN     12 __clang_gpu_used_external
```

### Relocation Check (PASSING)
```bash
$ llvm-readelf -r /tmp/librccl_device.elf | tail -15
# Shows 12 relocations at 0x517000-0x517058 pointing to oneRankReduce kernels
0000000000517000  R_AMDGPU_RELATIVE64  1e800
0000000000517008  R_AMDGPU_RELATIVE64  1f600
...
0000000000517058  R_AMDGPU_RELATIVE64  28300
```

### Single-GPU Test (PASSING)
```bash
$ ./bin/test_single_gpu
=== Simple RCCL Test ===
Found 8 HIP device(s)
Using device: AMD Instinct MI350X
NCCL communicator created
Running AllReduce...
AllReduce completed!
=== TEST PASSED ===
```

### Multi-GPU Test (HANGING)
```bash
$ NCCL_DEBUG=INFO timeout 120 ./bin/test_two_gpu
# Gets through all ring connections, hangs after "Connected all rings"
```

## Build Commands

```bash
# Rebuild device_linker tool
cd /work/lmeadows/rocm-systems/projects/rccl/tools/device_linker
g++ -O2 -std=c++17 device_linker.cpp -o device_linker -lpthread

# Full RCCL rebuild
cd /work/lmeadows/rocm-systems/projects/rccl
./install.sh --amdgpu_targets gfx950 --device-linker

# Run tests
cd tools/device_linker/smoke_test
./bin/test_single_gpu      # Should pass
./bin/test_two_gpu         # Currently hangs
```

## Fundamental Problem - No Easy Fix

The device linker approach has hit a **fundamental architectural limitation**:

### Why There's No Relocation to Patch

1. **Original dispatcher compilation:**
   - Function tables (`ncclDevFuncTable_1/2/4`) are in `.rodata` at addresses **before** `.text`
   - PC-relative references from code to tables are resolved at compile time
   - No relocations emitted because references are "internal" to the compilation unit

2. **What the device linker does:**
   - Creates new `.data.rel.ro` section for function tables
   - Places it **after** `.text` (standard ELF layout)
   - PC-relative offsets in code now point to wrong location

3. **Why it can't be fixed easily:**
   - No relocations exist to tell device linker what code to patch
   - Would need to parse machine code looking for `s_getpc` + `s_add` patterns
   - This is fragile and architecture-specific

### Original Dispatcher Layout
```
Section         Address    Contents
.rodata         0x6600     Function tables (ncclDevFuncTable_1 at 0x69c0)
.text           0xcb00     Dispatcher kernels
                           PC-relative offset: -24896 bytes (points backwards to .rodata)
```

### Device Linker Output Layout  
```
Section         Address    Contents
.text           0x1ae00    Dispatcher + specialized kernels
.data.rel.ro    0x511000   Function tables (moved here)
                           PC-relative offset still: ~-23620 bytes (points to garbage in .text)
```

### Possible Solutions (All Non-Trivial)

1. **Recompile dispatcher differently** - Emit relocations for internal references
   - Requires changes to RCCL build system
   - May not be possible with current compiler

2. **Preserve original section order** - Put `.rodata` before `.text`
   - Breaks standard ELF conventions
   - May confuse loaders/debuggers

3. **Scan and patch machine code** - Find `s_getpc` patterns manually
   - Fragile, architecture-specific
   - Need to understand all addressing patterns used

4. **Abandon device linker approach** - Use IFC (Incremental Fat binary Compilation) instead
   - Already works
   - Longer build times but correct results

## PC-Relative Fix Attempt (Feb 2, 2026)

### Approach
Changed from creating separate `.data.rel.ro` for function tables to keeping them in `.rodata`:

1. **Removed `.data.rel.ro` section creation** - Tables stay in `.rodata` at original offsets
2. **Made `.rodata` writable** (`SHF_ALLOC | SHF_WRITE`) so relocations can fill the tables
3. **Updated relocation targets** to `rodata_addr_ + rodata_table_*_off_`
4. **PC-relative patching** already works - verified bytes in ELF match expected values

### Current Layout
```
.rodata         0x15900   Function tables at offset 0x3c0 (so at 0x15cc0)
.text           0x1b000   Dispatcher kernel at start
```

PC-relative offset from getpc (at 0x1bb04) to table (at 0x15cc0):
- Expected: -0x5e44 = 0xffffa1bc
- Actual in ELF: 0xffffa1bc ✓ CORRECT

### Test Results
- **Single-GPU:** PASSES (both IFC and device linker)
- **Multi-GPU proper test:** IFC PASSES, device linker HANGS

### Flawed Test Warning
The `test_one_at_time.cpp` test was misleading - it called ncclAllReduce on just one GPU from a 2-GPU communicator, which hangs because collectives require all participants. Use `test_multi_gpu_proper.cpp` instead.

### Next Steps for Debugging
The hang (not crash) suggests something different than the original PC-relative issue. Possible causes:
- Synchronization/barrier issue between GPUs
- Different code paths for multi-GPU vs single-GPU
- Resource allocation differences

## Key Documents

- `DEVICE_LINKER_DESIGN.md` - Full design and implementation details
- `DEVICE_LINKER_REDESIGN.md` - Original redesign document  
- `IFC_VS_DEVICE_LINKER_COMPARISON.md` - Detailed ELF comparison
