# DWARFLinker Fix Plan

**Date**: February 8, 2026  
**Status**: Analysis complete, ready to implement fixes

## Key Findings

### 1. CMakeLists.txt Updated ✅
- Changed LLVM path from `/COD/LATEST/aomp/llvm` to `/work2/lmeadows/llvm`
- This provides debug symbols and source code for debugging

### 2. Current Problem

**Error**: `DWARF unit at offset 0x00000000 contains invalid abbreviation 1303 at offset 0x0000000e, valid abbreviations are 1-12`

**Root Cause Analysis**:
- The error occurs during **input DWARF processing**, not emission
- DWARFLinker is validating input DWARF against a merged abbreviation table (codes 1-12)
- This suggests input objects are being parsed incorrectly or abbreviation tables are being mixed

### 3. Comparison with dsymutil

**dsymutil approach**:
- Creates `DWARFContext` directly from object files (not minimal ELFs)
- Uses `ProcessDebugRelocations::Process` (same as device linker ✅)
- Each `DWARFFile` contains its own `DWARFContext` with its own abbreviation table
- DWARFLinker processes each input separately before merging

**Device linker approach**:
- Creates minimal ELFs from debug sections
- Creates `DWARFContext` from minimal ELFs
- **Potential issue**: Minimal ELFs might not preserve abbreviation table relationships correctly

### 4. Stub Methods That Need Implementation

Critical stubs that are currently empty but DWARFLinker may call:

1. **`emitDwarfDebugRangeListHeader()`** - Returns nullptr, should return symbol
2. **`emitDwarfDebugRangeListFragment()`** - Empty stub, needs to emit range list entries
3. **`emitDwarfDebugRangeListFooter()`** - Empty stub, needs to emit footer
4. **`emitDwarfDebugAddrsHeader()`** - Returns nullptr, should return symbol  
5. **`emitDwarfDebugAddrs()`** - Empty stub, needs to emit addresses
6. **`emitDwarfDebugAddrsFooter()`** - Empty stub, needs to emit footer
7. **`emitLineTableForUnit()`** - Empty stub, needs to emit line table

### 5. Manual Serialization Issues

The device linker manually serializes DIEs without MC classes. Key issues:

- **Abbreviation code handling**: Complex fallback logic suggests `setAbbrevNumber()` might not be called correctly
- **Form value serialization**: Manual serialization might not match MC layer exactly
- **Size calculations**: `Die.getSize()` might not match manual serialization

## Implementation Strategy

### Phase 1: Fix Input DWARF Processing (CRITICAL)

**Hypothesis**: The minimal ELF creation might be corrupting abbreviation table references.

**Actions**:
1. Verify minimal ELF includes `.debug_abbrev` correctly
2. Check if `DWARFContext::create()` is parsing abbreviation tables correctly
3. Consider creating `DWARFContext` directly from original object files instead of minimal ELFs
4. Add debug output to verify each input's abbreviation table is parsed separately

**Files to modify**:
- `device_linker.cpp::mergeDebugInfoWithDWARFLinker()` - Input file preparation

### Phase 2: Implement Critical Stub Methods

Based on dsymutil's `DwarfStreamer` implementation:

1. **Range Lists** (`emitDwarfDebugRngListsTableFragment`):
   - Emit `DW_RLE_base_addressx` or `DW_RLE_base_address` for first range
   - Emit `DW_RLE_offset_pair` entries for each range
   - Emit `DW_RLE_end_of_list` terminator
   - Write raw bytes to `debug_rnglists_out_` buffer

2. **Address Table** (`emitDwarfDebugAddrsHeader`, `emitDwarfDebugAddrs`, `emitDwarfDebugAddrsFooter`):
   - Header: length (4 bytes), version (2 bytes), addr_size (1 byte), segment_size (1 byte)
   - Addresses: Write each address as `AddrSize` bytes
   - Footer: Emit end label (if needed)

3. **Line Table** (`emitLineTableForUnit`):
   - Emit prologue (version, address_size, header_length, etc.)
   - Emit file table
   - Emit line table rows
   - Write raw bytes to `debug_line_out_` buffer

**Files to modify**:
- `device_linker.cpp::DeviceLinkerDwarfEmitter` - Implement stub methods

### Phase 3: Fix Manual Serialization

1. Verify `Die.getAbbrevNumber()` is set correctly before `emitDIE()`
2. Ensure form value serialization matches MC layer exactly
3. Fix size calculations to match `Die.getSize()`

**Files to modify**:
- `device_linker.cpp::DeviceLinkerDwarfEmitter::emitDIE()`
- `device_linker.cpp::DeviceLinkerDwarfEmitter::serializeFormValue()`

## Testing Strategy

1. **Minimal test**: Run device linker on dispatcher + 1 specialized kernel
2. **Verify**: Check that `emitCompileUnitHeader()` and `emitDIE()` are called
3. **Validate**: Run `llvm-dwarfdump --verify` on output
4. **Debug**: Use GDB with new LLVM build to step through DWARFLinker code

## Next Steps

1. ✅ Update CMakeLists.txt (DONE)
2. 🔄 Fix input DWARF processing (in progress)
3. ⏳ Implement stub methods
4. ⏳ Fix manual serialization
5. ⏳ Test and validate

## References

- **dsymutil source**: `/work2/lmeadows/llvm-project/llvm/tools/dsymutil/`
- **DWARFStreamer**: `/work2/lmeadows/llvm-project/llvm/lib/DWARFLinker/Classic/DWARFStreamer.cpp`
- **Current status**: `tools/device_linker/DWARFLINKER_STATUS.md`
