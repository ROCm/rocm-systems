# DWARFLinker Implementation Context

**Date**: February 8, 2026  
**Status**: Planning complete, ready to implement  
**Goal**: Replace manual DWARF patching with DWARFLinker to fix 48,339 DWARF verification errors

## Current Problem Summary

### Verification Errors
- **36,480 errors**: "DIE address ranges are not contained by parent ranges"
- **11,856 errors**: "DIEs have overlapping address ranges"  
- **Total**: 48,339 errors reported by `llvm-dwarfdump --verify`

### Root Causes Identified
1. **Range list containment**: Child DIEs have ranges that should be contained in parent ranges, but verification fails
2. **CU low_pc patching**: CU's `DW_AT_low_pc` is patched to 0x2c604, but range lists expect base address 0x2cb50
3. **Manual patching complexity**: Current code manually patches `.debug_addr` and `.debug_rnglists` by:
   - Finding addresses via LLVM parsing
   - Manually patching raw bytes with address deltas
   - Handling different range list entry types (DW_RLE_startx_endx, DW_RLE_offset_pair, etc.)
4. **DW_RLE_offset_pair handling**: Range lists use `DW_RLE_offset_pair` (kind 0x04) which are relative to a base address. The base address mechanism isn't being handled correctly.

### Specific Example
- **Parent DIE** (0x00006080 "Primitives"): ranges `[0x2d374, 0x2d89c)`, `[0x2d270, 0x2d278)`, `[0x2dbc4, 0x2dbd0)`
- **Child DIE** (0x000060a6 "patBarrier"): ranges `[0x2d620, 0x2d874)`, `[0x2d894, 0x2d89c)`
- **Mathematically**: Child ranges ARE contained in parent's `[0x2d374, 0x2d89c)` range
- **But**: `llvm-dwarfdump --verify` reports containment error
- **Root cause**: Range lists use `DW_RLE_offset_pair` entries relative to CU's `DW_AT_low_pc`, but the base address calculation is incorrect

## Solution: Adopt DWARFLinker

### Why DWARFLinker?
- **Purpose-built for linking**: Designed to merge DWARF from multiple objects with address relocation
- **Handles all DWARF5 features**: Properly handles `.debug_rnglists`, `.debug_addr`, range containment
- **Address mapping**: Built-in support for per-object address relocation (exactly what we need)
- **Valid output**: Produces valid DWARF by construction, eliminating patching errors

### Availability
✅ **DWARFLinker is available**:
- **Headers**: `/COD/LATEST/aomp/llvm/include/llvm/DWARFLinker/`
- **Libraries**: `libLLVMDWARFLinker.a`, `libLLVMDWARFLinkerClassic.a`
- **Build system**: Already configured in `CMakeLists.txt`:
  - Line 1305: `set(AOMP_LLVM_DIR "/COD/LATEST/aomp/llvm")`
  - Line 1309: `target_include_directories(device_linker_tool PRIVATE ${AOMP_LLVM_DIR}/include)`
  - Line 1318: `target_link_libraries(device_linker_tool PRIVATE LLVM)`

## API Structure

### Key Components

1. **DWARFLinker**: `llvm::dwarf_linker::classic::DWARFLinker`
   - Location: `DWARFLinker/Classic/DWARFLinker.h`
   - Usage: `linker.addObjectFile(dwarfFile)`, then `linker.link()`

2. **DwarfEmitter**: Abstract base class we need to implement
   - Location: `DWARFLinker/Classic/DWARFLinker.h`
   - Key methods:
     - `emitSectionContents(StringRef SecData, DebugSectionKind SecKind)`
     - `emitAbbrevs(const std::vector<std::unique_ptr<DIEAbbrev>>& Abbrevs, unsigned DwarfVersion)`
     - `emitStrings(const NonRelocatableStringpool& Pool)`
     - `emitStringOffsets(const SmallVector<uint64_t>& StringOffsets, uint16_t TargetDWARFVersion)`
     - `emitLineStrings(const NonRelocatableStringpool& Pool)`
     - `emitCompileUnitHeader(CompileUnit& Unit, unsigned DwarfVersion)`
     - `emitDIE(DIE& Die)`
     - And more...

3. **AddressesMap**: Abstract interface for address mapping
   - Location: `DWARFLinker/AddressesMap.h`
   - Key methods:
     - `getSubprogramRelocAdjustment(const DWARFDie& DIE, bool Verbose) -> optional<int64_t>`
     - `getExprOpAddressRelocAdjustment(...) -> optional<int64_t>`
     - `hasValidRelocs() -> bool`
     - `applyValidRelocs(...) -> bool`

4. **DWARFFile**: Container for input objects
   - Location: `DWARFLinker/DWARFFile.h`
   - Contains: `DWARFContext` + `AddressesMap` per input

## Implementation Plan

See `DWARFLINKER_PLAN.md` for detailed plan. Summary:

### Phase 1: Research (1-2 days) ✅ DONE
- [x] Verify DWARFLinker availability
- [x] Understand API structure
- [x] Document interface requirements

### Phase 2: Custom DwarfEmitter (2-3 days)
- [ ] Implement `DeviceLinkerDwarfEmitter` class
- [ ] Write to device linker's section buffers
- [ ] Implement all required emitter methods

### Phase 3: Integration (3-4 days)
- [ ] Create `mergeDebugInfoWithDWARFLinker()` function
- [ ] Create `DWARFFile` for each input (dispatcher + kernels)
- [ ] Wire up `DWARFLinker` with custom emitter and address maps
- [ ] Replace `mergeDebugInfo()` call

### Phase 4: Custom AddressesMap (1-2 days)
- [ ] Implement `DeviceLinkerAddressesMap` class
- [ ] Map addresses: `new_addr = text_addr_ + chunk.new_text_offset + (orig_addr - chunk.orig_text_addr)`

### Phase 5: Testing (2-3 days)
- [ ] Verify zero `llvm-dwarfdump --verify` errors
- [ ] Test with rocgdb, llvm-symbolizer
- [ ] Regression testing

### Phase 6: Cleanup (1 day)
- [ ] Remove old manual patching code
- [ ] Update documentation

**Total estimate**: 10-15 days

## Current Code State

### Files Modified
- `tools/device_linker/device_linker.cpp`: Main implementation
  - `mergeDebugInfo()`: Current merge logic (lines ~3961+)
  - `patchAddressesUsingLLVM()`: Current address patching (lines ~3611+)
  - `createTempElfForMergedDebugSections()`: Helper for LLVM parsing (lines ~3446+)

### Current Approach
1. Concatenate debug sections from all inputs
2. Patch CU headers (abbrev_offset, etc.)
3. Use LLVM to parse merged sections
4. Manually patch addresses in `.debug_addr` and `.debug_rnglists`
5. **Problem**: Manual patching introduces errors

### What Works
- ✅ LLVM reading APIs solve many problems
- ✅ Relocation application works correctly
- ✅ CU header patching works
- ✅ Section concatenation works

### What Doesn't Work
- ❌ Range list containment (36,480 errors)
- ❌ Overlapping ranges (11,856 errors)
- ❌ CU low_pc vs range list base address mismatch

## Key Technical Details

### Range List Structure
- Range lists use `DW_RLE_offset_pair` (kind 0x04) entries
- Offsets are relative to a base address (CU's `DW_AT_low_pc` or explicit `DW_RLE_base_address`)
- Current CU `DW_AT_low_pc` = 0x2c604
- Range lists decode correctly with base = 0x2cb50 (54 bytes difference)
- This mismatch causes containment verification failures

### Address Mapping
- Each kernel object has: `orig_text_addr` (original .text address)
- After merging: `new_text_addr = text_addr_ + chunk.new_text_offset`
- Address delta: `addr_delta = new_text_addr - orig_text_addr`
- All addresses in that object need: `new_addr = orig_addr + addr_delta`

### DWARF5 Sections Involved
- `.debug_info`: Compilation units and DIEs
- `.debug_abbrev`: Abbreviation tables
- `.debug_str`: String table
- `.debug_str_offsets`: String offset table (DWARF5)
- `.debug_addr`: Address table (DWARF5)
- `.debug_rnglists`: Range lists (DWARF5, replaces .debug_ranges)
- `.debug_line`: Line tables
- `.debug_line_str`: Line string table (DWARF5)

## Next Steps

1. **Test compilation**: Add `#include "llvm/DWARFLinker/Classic/DWARFLinker.h"` to verify headers compile
2. **Start Phase 2**: Implement `DeviceLinkerDwarfEmitter` class
3. **Start Phase 4**: Implement `DeviceLinkerAddressesMap` class  
4. **Phase 3**: Wire everything together
5. **Phase 5**: Test and validate

## References

- **Plan document**: `DWARFLINKER_PLAN.md` (detailed implementation plan)
- **Previous plan**: `DWARF_LLVM_REWRITE_PLAN.md` (original Phase 3 plan)
- **Current code**: `device_linker.cpp` (lines ~3611-4023 for current patching logic)
- **LLVM headers**: `/COD/LATEST/aomp/llvm/include/llvm/DWARFLinker/`

## Success Criteria

1. ✅ Zero verification errors: `llvm-dwarfdump --verify` reports no errors
2. ✅ Functional parity: All existing functionality works (rocgdb, llvm-symbolizer)
3. ✅ Code simplification: Removed ~500+ lines of manual patching code
4. ✅ Maintainability: Future DWARF changes handled by LLVM, not manual code

## Notes

- DWARFLinker handles all the complexity of merging DWARF correctly
- We just need to implement the emitter (writes to our buffers) and address map (tells linker how to relocate)
- The linker will handle range list containment, base addresses, and all DWARF5 features automatically
- This should eliminate all 48,339 verification errors
