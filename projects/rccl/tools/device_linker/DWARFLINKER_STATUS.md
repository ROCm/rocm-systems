# DWARFLinker Integration Status

**Last Updated:** February 8, 2026  
**Status:** Debugging persistent "invalid abbreviation" error during input DWARF processing

## Overview

We are integrating LLVM's `DWARFLinker` library into the `device_linker` tool to replace error-prone manual DWARF patching. The goal is to properly merge DWARF debug information from multiple object files (dispatcher + 859 specialized kernels) with correct address relocation.

## Current State

### What's Implemented

1. **Custom `DwarfEmitter` Implementation** (`DeviceLinkerDwarfEmitter`):
   - Implements `emitCompileUnitHeader()` for manual CU header serialization
   - Implements `emitDIE()` for manual DIE serialization with abbreviation renumbering
   - Implements `emitAbbrevs()` (currently a no-op, relies on `emitSectionContents`)
   - Implements `emitSectionContents()` for other DWARF sections (strings, line tables, etc.)
   - Maintains `abbrev_content_to_number_` map for abbreviation code renumbering

2. **AddressesMap Implementation** (`DeviceLinkerAddressesMap`):
   - Provides address relocation adjustments for each input object
   - Tracks original and new addresses for code sections

3. **DWARFLinker Setup**:
   - Creates `DWARFLinker` instance with error/warning handlers
   - Sets custom emitter via `setOutputDWARFEmitter()`
   - Dynamically tracks maximum DWARF version from input CUs (like `dsymutil` does)
   - Sets target DWARF version *after* all objects are added
   - Uses `ProcessDebugRelocations::Process` (aligned with `dsymutil`)

4. **Input File Preparation**:
   - Creates minimal ELF files from kernel debug sections for `DWARFContext`
   - Extracts debug sections from original ELF files
   - Creates `DWARFFile` objects with `DWARFContext` and `AddressesMap`

### Current Problem

**Error:** `DWARF unit at offset 0x00000000 contains invalid abbreviation 1303 at offset 0x0000000e, valid abbreviations are 1-12`

**Key Observations:**
1. **Emitter methods are NOT being called**: Despite extensive debug prints, `emitCompileUnitHeader()` and `emitDIE()` never execute. The `debug_info_out_` buffer remains size 0 throughout.
2. **Error occurs during input processing**: The error happens inside `DWARFLinker::link()` when processing *input* DWARF, not during emission.
3. **Input data is correct**: We verified `kernel->debug_info[14] = 0x21` (33 decimal), not 1303. The original ELF files have valid abbreviation codes.
4. **"Valid abbreviations are 1-12"**: This suggests DWARFLinker is validating input DWARF against a merged abbreviation table (codes 1-12), which shouldn't happen during input processing.

**Error Location:**
- Source: `llvm/lib/DebugInfo/DWARF/DWARFDebugInfoEntry.cpp:54-62`
- Function: `DWARFDebugInfoEntry::extract()` → `AbbrevSet->getAbbreviationDeclaration(AbbrCode)`
- Called from: `DWARFUnit::tryExtractDIEsIfNeeded()` → `extractDIEsToVector()`
- Triggered during: `DWARFLinker::link()` → input DWARF processing loop

## What We've Tried

### 1. Aligning with `dsymutil` Pattern
- **Removed** premature `setTargetDWARFVersion(5)` call
- **Added** dynamic DWARF version tracking via `OnCUDieLoaded` callback
- **Moved** `setTargetDWARFVersion()` call to *after* all objects are added
- **Changed** `ProcessDebugRelocations::Ignore` → `ProcessDebugRelocations::Process`
- **Result:** Error persists, same behavior

### 2. Debug Output
- Added unbuffered `write(2, ...)` calls in `emitCompileUnitHeader()` and `emitDIE()`
- Added debug prints showing `debug_info_out_` size before/after `linker.link()`
- Verified input data correctness (`kernel->debug_info[14] = 0x21`)
- **Result:** Confirmed emitter methods are never called; input data is correct

### 3. GDB Debugging
- Created gdb scripts to catch the error location
- Confirmed error originates in `DWARFUnit::tryExtractDIEsIfNeeded`
- Stack trace shows error occurs during input DWARF parsing, not emission
- **Result:** Error confirmed to occur before emission phase

### 4. Abbreviation Handling
- Implemented `abbrev_content_to_number_` map for renumbering
- Added fallback lookup in `emitDIE()` for invalid abbreviation codes
- Made `emitAbbrevs()` a no-op (relying on `emitSectionContents`)
- **Result:** Not relevant since `emitDIE()` is never called

## Technical Details

### DWARFLinker Flow (Expected)

1. **Add Objects**: `linker.addObjectFile(dwarf_file, loader, on_cu_loaded)`
   - Creates `DWARFContext` from input ELF
   - Calls `on_cu_loaded` callback for each CU to track max DWARF version
   - Stores `DWARFFile` in internal list

2. **Set Target Version**: `linker.setTargetDWARFVersion(max_version)`
   - Must be called *after* all objects are added (per `dsymutil` pattern)

3. **Link**: `linker.link()`
   - **Phase 1 - Process Input**: Reads input DWARF, clones DIEs, builds merged abbreviation table
   - **Phase 2 - Emit**: Calls `emitter.emitAbbrevs()`, `emitter.emitCompileUnitHeader()`, `emitter.emitDIE()` for each CU

### Current Behavior

- Phase 1 (input processing) crashes with "invalid abbreviation" error
- Phase 2 (emission) never executes
- Error suggests DWARFLinker is validating input DWARF against merged abbreviation table

### Hypothesis

The "valid abbreviations are 1-12" message suggests DWARFLinker has already created a merged abbreviation table with codes 1-12, but is somehow trying to validate *input* DWARF against this merged table. This could indicate:

1. **Context mixing**: DWARFLinker is reading from the wrong `DWARFContext` or abbreviation set
2. **Internal bug**: DWARFLinker may have a bug when processing multiple input files
3. **Minimal ELF issue**: The minimal ELF we create might be malformed in a way that confuses DWARFLinker
4. **Abbreviation table corruption**: The input abbreviation tables might be getting corrupted during `DWARFContext` creation

## Code References

### Key Files
- `tools/device_linker/device_linker.cpp`: Main implementation
  - `DeviceLinkerDwarfEmitter` class (lines ~1800-2200)
  - `DeviceLinkerAddressesMap` class (lines ~1650-1800)
  - `mergeDebugInfoWithDWARFLinker()` function (lines ~4879-5200)

### LLVM Source References
- `/work2/lmeadows/llvm-project/llvm/tools/dsymutil/DwarfLinkerForBinary.cpp`: Reference implementation
- `/work2/lmeadows/llvm-project/llvm/lib/DebugInfo/DWARF/DWARFDebugInfoEntry.cpp:54-62`: Error location
- `/work2/lmeadows/DWARFLinker/Classic/DWARFLinker.cpp`: DWARFLinker implementation

## Next Steps

### Immediate
1. **Verify minimal ELF correctness**: Check if the minimal ELF we create preserves debug_info data correctly
2. **Check DWARFContext creation**: Verify that `DWARFContext::create()` is reading the correct data
3. **Compare with dsymutil**: Try running `dsymutil` on a single input file to see if it works (though dsymutil is Mach-O only)

### Potential Solutions
1. **Skip problematic kernels**: If only some kernels cause issues, skip them temporarily
2. **Use smaller test set**: Continue testing with `MAX_KERNELS_FOR_TEST=5` to isolate the issue
3. **Check LLVM version compatibility**: Verify that our LLVM version matches what `dsymutil` uses
4. **Alternative approach**: Consider using LLVM's DWARF APIs directly instead of DWARFLinker if the issue persists

### Long-term
1. **Complete emission implementation**: Once input processing works, ensure all emitter methods are correctly implemented
2. **Validate merged output**: Use `llvm-dwarfdump --validate` on merged output
3. **Test with rocgdb**: Verify that merged DWARF works with actual debugger

## Environment

- **LLVM Path**: `/work2/lmeadows/rocm/aomp_23.0-0/llvm` (AOMP 23.0)
- **LLVM Clone**: `/work2/lmeadows/llvm-project` (for reference, trunk version)
- **Test Command**: `MAX_KERNELS_FOR_TEST=5 tools/device_linker/device_linker -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/`

## Related Documents

- `DWARFLINKER_CONTEXT.md`: Initial context and plan for DWARFLinker integration
- `DWARFLINKER_PLAN.md`: Step-by-step implementation plan
- `DWARF_LLVM_REWRITE_PLAN.md`: Overall DWARF rewrite strategy
- `CONTEXT.md`: Main device linker context document
