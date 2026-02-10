# DWARFLinker Investigation Summary

**Date**: February 8, 2026  
**Status**: Investigation complete, ready for testing

## Problem

**Error**: `DWARF unit at offset 0x00000000 contains invalid abbreviation 97 at offset 0x0000000d, valid abbreviations are 1-12`

**Root Cause Hypothesis**: DWARFLinker is validating input DWARF files against a merged abbreviation table (codes 1-12) instead of each input's own abbreviation table.

## Investigation Results

### ✅ Verified: Minimal ELF Creation

1. **`.debug_abbrev` preservation**: The section is copied verbatim (line 358)
2. **`abbrev_offset` correctness**: Original files have `abbrev_offset=0` (relative to section start), which is preserved
3. **ELF structure**: Section headers and offsets are calculated correctly
4. **Section sizes**: All sections are copied with correct sizes

### ✅ Critical Fix Applied

**`DW_AT_str_offsets_base` patching**: Before creating the minimal ELF for DWARFLinker, we now patch `DW_AT_str_offsets_base` to point to offset 8 in the new `.debug_str_offsets` section. This ensures the CU header correctly references the minimal ELF's string offsets table.

**Location**: Lines 5044-5045 in `device_linker.cpp`

### ✅ Debug Enhancements Added

1. **Section verification**: Checks that `.debug_info` and `.debug_abbrev` sections are found and have correct sizes
2. **Abbrev offset validation**: Verifies `abbrev_offset` is within bounds
3. **Abbreviation table parsing**: Parses and displays all abbreviation codes in each input file's table (shows codes 1-9 for test file)
4. **DWARFContext testing**: Attempts to parse the first CU DIE before passing to DWARFLinker
5. **Abbreviation set inspection**: Logs the abbreviation codes available in each CU's abbreviation set (from LLVM's perspective via `getCodeRange()`)
6. **DWARFFile tracking**: Logs when DWARFFile objects are created and added to the linker
7. **Enhanced error handling**: Captures detailed error information during `link()`

## Key Findings

1. **Minimal ELF creation is correct**: Verified that `.debug_abbrev` is copied correctly and `abbrev_offset=0` is preserved
2. **Each input has isolated DWARFContext**: Like dsymutil, we create separate `DWARFContext` for each input file
3. **Error occurs during `link()`**: The error happens when `CU->getUnitDIE()` is called, which needs to parse DIEs using the abbreviation table
4. **Test file has codes 1-9**: The test file we examined has abbreviation codes 1, 2, 4, 5, 6, 7, 8, 9 (code 3 is missing, which is normal)
5. **Error mentions code 97**: This suggests a different input file has code 97, but it's being validated against an abbreviation set with codes 1-12

## Code Changes

### Modified Files
- `tools/device_linker/device_linker.cpp`:
  - Added `DW_AT_str_offsets_base` patching before creating minimal ELF (lines 5044-5048)
  - Added comprehensive debug output throughout the DWARFLinker integration
  - Added abbreviation table parsing to show all codes in each input file

### Debug Output Locations
- Lines 5047-5048: Patched debug_info size logging
- Lines 5078-5116: Section verification and size checking
- Lines 5120-5216: CU header and abbreviation table inspection
- Lines 5168-5270: DWARFContext creation and CU parsing verification
- Lines 5193-5203: DWARFFile creation and addition tracking
- Lines 5234-5250: Enhanced error handling during `link()`

## Next Steps

1. **Rebuild the device linker** with the new debug output
2. **Run with `MAX_KERNELS_FOR_TEST=5`** to test with a small subset
3. **Analyze the debug output** to:
   - Identify which input file triggers the error (if it still occurs)
   - Compare abbreviation codes from our parsing vs LLVM's `getCodeRange()`
   - Verify if the `DW_AT_str_offsets_base` patching fixed the issue
   - Determine if the error occurs during initial parsing or during DWARFLinker processing

## Expected Debug Output

When running, you should see for each kernel:
```
DEBUG: Patched debug_info (original size=X, patched size=X)
DEBUG: Created ObjectFile, checking sections...
DEBUG: .debug_info section size: X
DEBUG: .debug_abbrev section size: X (expected X)
DEBUG: First byte matches: yes/no
DEBUG: CU header abbrev_offset=0 (0x0), debug_abbrev size=X
DEBUG: First abbrev code at offset 0: 0x01
DEBUG: Abbrev code 1 at table offset 0
DEBUG: Abbrev code 2 at table offset X
...
DEBUG: Found N abbreviation codes: 1-N
DEBUG: Code range: 1 to N
DEBUG: Creating DWARFContext for kernel X (minimal ELF size=Y)
DEBUG: Successfully got DebugAbbrev for kernel X
DEBUG: Found 1 compile unit(s), testing first CU...
DEBUG: CU abbreviation set has codes: 1-N
DEBUG: CU abbrev_offset=0
DEBUG: Successfully parsed first CU DIE for kernel X
```

## References

- **Error location**: `DWARFDebugInfoEntry::extract()` at line 54-62 in `DWARFDebugInfoEntry.cpp`
- **Minimal ELF creation**: `createMinimalElfForDwarf()` at line 129
- **DWARFLinker usage**: `mergeDebugInfoWithDWARFLinker()` at line 4879
- **Abbreviation set lookup**: `DWARFUnit::getAbbreviations()` → `DWARFDebugAbbrev::getAbbreviationDeclarationSet()`
