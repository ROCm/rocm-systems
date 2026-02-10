# DWARFLinker Debug Plan

**Date**: February 8, 2026  
**Status**: Investigating abbreviation table validation issue

## Problem Summary

**Error**: `DWARF unit at offset 0x00000000 contains invalid abbreviation 97 at offset 0x0000000d, valid abbreviations are 1-12`

**Root Cause Hypothesis**: DWARFLinker is validating input DWARF files against a merged abbreviation table (codes 1-12) instead of each input's own abbreviation table.

## Verification Needed

### 1. Minimal ELF Creation - Abbreviation Table Preservation ✅

**Status**: VERIFIED - Minimal ELF creation preserves `.debug_abbrev` correctly:
- Line 358: Copies `debug_abbrev` data as-is
- Line 304: Sets correct section offset
- Original files have `Abbrev Offset: 0x0` (verified with readelf)

**Conclusion**: The minimal ELF creation is correct. The `.debug_abbrev` section is copied verbatim, and since `abbrev_offset` in the CU header is relative to the start of `.debug_abbrev`, it remains valid.

### 2. CU Header `abbrev_offset` Patching

**Current State**: 
- `createMinimalElfForDwarf()` copies `.debug_info` as-is (line 357)
- Original CU headers have `abbrev_offset=0` (verified)
- No patching needed if offset is 0

**Potential Issue**: If any input file has a non-zero `abbrev_offset`, we might need to patch it to 0 in the minimal ELF.

**Action**: Add verification to ensure all inputs have `abbrev_offset=0`, or add patching logic if needed.

### 3. DWARFLinker Input Processing

**The Real Problem**: The error message "valid abbreviations are 1-12" suggests DWARFLinker has already created a merged abbreviation table and is using it to validate inputs.

**Hypothesis**: DWARFLinker might be:
1. Processing inputs in the wrong order
2. Sharing abbreviation tables between inputs
3. Validating inputs against the output abbreviation table

**Investigation Needed**:
- Check how dsymutil creates `DWARFContext` - does it create separate contexts per input?
- Verify that each `DWARFFile` has an isolated `DWARFContext`
- Check if `setTargetDWARFVersion()` is being called before inputs are processed

### 4. Memory Usage (200+ GB)

**Issue**: DWARF debug information is using 200+ GB of working set size, which may cause issues with gdb.

**Potential Causes**:
1. Loading all 859 kernel DWARF contexts simultaneously
2. DWARFLinker keeping all input DWARF in memory
3. Minimal ELF creation creating large temporary buffers

**Investigation**:
- Check if we can process inputs in batches
- Verify if DWARFLinker has options to reduce memory usage
- Consider processing fewer kernels at a time for testing

## Next Steps

1. ✅ **Verify `abbrev_offset` patching**: VERIFIED - Minimal ELF creation preserves `.debug_abbrev` correctly, `abbrev_offset=0` is correct
2. **Investigate DWARFLinker input processing**: The error "invalid abbreviation 97... valid abbreviations are 1-12" suggests DWARFLinker is using a merged abbreviation table (codes 1-12) to validate inputs that have their own abbreviation codes (like 97)
3. **Root cause hypothesis**: When `CU->getUnitDIE()` is called during `link()`, it calls `getAbbreviations()` which should use the input's own `DWARFDebugAbbrev`, but somehow it's using the wrong one
4. **Added debug output**: Added logging to verify minimal ELF structure and abbreviation table access
5. **Test with fewer kernels**: Use `MAX_KERNELS_FOR_TEST` to test with a small subset and see which input triggers the error

## Key Findings

1. **Minimal ELF creation is correct**: Verified that `.debug_abbrev` is copied correctly and `abbrev_offset=0` is preserved
2. **Each input has isolated DWARFContext**: Like dsymutil, we create separate `DWARFContext` for each input file
3. **Error occurs during `link()`**: The error happens when `CU->getUnitDIE()` is called, which needs to parse DIEs using the abbreviation table
4. **Added comprehensive debug output**: 
   - Verify `abbrev_offset` values
   - Check section presence in ObjectFile
   - Verify DebugAbbrev access
   - Enhanced error handling during `link()`

## Debug Output Added

1. **Section verification**: Checks that `.debug_info` and `.debug_abbrev` sections are found and have correct sizes
2. **Abbrev offset validation**: Verifies `abbrev_offset` is within bounds and logs the first abbreviation code
3. **DWARFContext testing**: Attempts to parse the first CU DIE before passing to DWARFLinker
4. **Abbreviation set inspection**: Logs the abbreviation codes available in each CU's abbreviation set
5. **DWARFFile tracking**: Logs when DWARFFile objects are created and added to the linker
6. **Enhanced error handling**: Captures detailed error information during `link()`

## Critical Fix Applied

**Added `DW_AT_str_offsets_base` patching**: Before creating the minimal ELF for DWARFLinker, we now patch `DW_AT_str_offsets_base` to point to offset 8 in the new `.debug_str_offsets` section. This ensures the CU header correctly references the minimal ELF's string offsets table.

## Debug Output Added

1. **Section verification**: Checks that `.debug_info` and `.debug_abbrev` sections are found and have correct sizes
2. **Abbrev offset validation**: Verifies `abbrev_offset` is within bounds and logs the first abbreviation code
3. **Abbreviation table parsing**: Parses and displays all abbreviation codes in each input file's table
4. **DWARFContext testing**: Attempts to parse the first CU DIE before passing to DWARFLinker
5. **Abbreviation set inspection**: Logs the abbreviation codes available in each CU's abbreviation set (from LLVM's perspective)
6. **DWARFFile tracking**: Logs when DWARFFile objects are created and added to the linker
7. **Enhanced error handling**: Captures detailed error information during `link()`

## Next Action

Run the device linker with `MAX_KERNELS_FOR_TEST=5` to:
1. See which specific input triggers the error (if it still occurs)
2. Verify the debug output shows correct `abbrev_offset` values
3. Check if the error occurs during initial parsing or during DWARFLinker processing
4. See what abbreviation codes each input file has (both from our parsing and from LLVM)
5. Compare the abbreviation codes to identify if there's a mismatch
6. Determine if the `DW_AT_str_offsets_base` patching fixed the issue

## References

- **Current error**: "invalid abbreviation 97... valid abbreviations are 1-12"
- **Minimal ELF creation**: `createMinimalElfForDwarf()` at line 129
- **DWARFLinker usage**: `mergeDebugInfoWithDWARFLinker()` at line 4879
- **Error location**: `DWARFDebugInfoEntry::extract()` calls `getAbbreviations()` which uses `getAbbreviationsOffset()` from CU header
