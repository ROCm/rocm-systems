# DWARF Fix Progress

## Goal
Fix `rocgdb` crashes/errors when loading device linker output ELF due to malformed DWARF debug info.

## Current Status
- `rocgdb` still reports: "DW_FORM_line_strp pointing outside of .debug_line_str section"
- `llvm-dwarfdump --verify` passes (no section offset errors)
- Some function lookups work (`info functions ncclKernel`), others fail (`info functions reduce`)

## Completed Fixes

### 1. DWARF Attribute Position Finding (LLVM API)
- Created `findDwarfAttrPositions()` using LLVM's `DWARFContext` API
- Properly parses abbreviation table and DIE attributes
- Finds positions of: `DW_AT_ranges`, `DW_AT_str_offsets_base`, `DW_AT_addr_base`, `DW_AT_rnglists_base`, `DW_AT_stmt_list`
- Location: lines 125-407 in `device_linker.cpp`

### 2. Relocation Handling for .debug_line
- Added code to apply `.rela.debug_line` relocations when parsing kernels
- Handles `R_X86_64_32` (type 10) for string offsets into `.debug_line_str`
- Handles `R_X86_64_64` (type 1) for `.text` addresses
- Location: lines 738-770 in `device_linker.cpp` (in `parseKernel()`)

### 3. DWARF5 String Offset Patching (Manual - BUGGY)
- `patchDwarf5StringOffsets()` manually parses DWARF5 line table prologues
- Finds `DW_FORM_line_strp` positions and adds `str_offset_base`
- Reports patching 52,764 offsets across 2,578 kernels
- Location: lines 2903-3065 in `device_linker.cpp`

## Remaining Issue
The manual DWARF5 line table parsing in `patchDwarf5StringOffsets()` is likely buggy. It doesn't use LLVM API and may not handle all edge cases correctly.

## Next Step: Use LLVM DWARFDebugLine API
Replace manual prologue parsing with LLVM's `DWARFDebugLine` class:

```cpp
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"

// Use DWARFDebugLine::Prologue to properly parse line table headers
// and find exact positions of DW_FORM_line_strp values
```

This would:
1. Create minimal ELF with `.debug_line` and `.debug_line_str` sections
2. Use `DWARFDebugLine::parse()` to get proper prologue info
3. Use prologue's `FileNames` and `IncludeDirectories` to find string offset positions
4. Patch those positions with correct merged offsets

## Files Modified
- `/work2/lmeadows/rocm-systems/projects/rccl/tools/device_linker/device_linker.cpp`

## Build Command
```bash
cd /work2/lmeadows/rocm-systems/projects/rccl
rm -rf build
./install.sh --amdgpu_targets=gfx942 --device-linker
```

## Test Commands
```bash
# Should work:
rocgdb -batch -ex "file build/release/device_linker_output/merged_device.elf" -ex "info functions ncclKernel"

# Currently fails:
rocgdb -batch -ex "file build/release/device_linker_output/merged_device.elf" -ex "info functions reduce"

# Verify DWARF:
llvm-dwarfdump --verify build/release/device_linker_output/merged_device.elf
```
