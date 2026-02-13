# DWARF Line Table Debug Info Mismatch - Investigation Findings

## Empirical Data (from debug run)

**Device linker AddressesMap for `specialized_all_reduce_ring_ll_sum_f32_unroll2.cpp.o`:**
- `orig_text` = 0x1a00
- `new_text` = 0x6e0700
- `size` = 0x5970
- `adjustment` = 0x6DFD00 (7204096)

**Correct mapping:** First line at 0x1b98 → should map to 0x6e0700 + 0x198 = **0x6e0898**

**Merged .debug_line for our kernel (addresses 0x6e07xx):**
- Function start: 0x6e0700
- First line (prologue_end): **0x6e0764** → offset = 0x64 = **100** from function start
- **Correct** would be 0x6e0700 + 0x198 = **0x6e0898** (offset 408)

**Bug:** The merged .debug_line has the first line at offset 100 instead of 408. The emitted address 0x6e0764 = 0x1a64 + adjustment, but the correct source address is 0x1b98. So the DWARFLinker is emitting 0x1a64 instead of 0x1b98 for the first line row—a 308-byte error in the line table addresses.

## Summary

When debugging `_Z43ncclDevFunc_AllReduce_RING_LL_Sum_f32_0_0_2v` with rocgdb, the first source line is incorrectly attributed to an address **~300 bytes too early** in the function.

- **Correct** (llvm-objdump -d -l on original .o): First line at offset `0x1B98` in .text
- **Wrong** (rocgdb on merged librccl.so): First line at offset **108 (0x6C)** from function start
- The instruction at that wrong offset: `scratch_store_dword` at `0x1A6C`

## Numeric Analysis

If the function starts at `0x1A00` and the first line is at `0x1B98`:
- **Correct offset** from function start: `0x1B98 - 0x1A00 = 0x198 = 408` bytes
- **rocgdb reports**: 108 bytes (0x6C)
- **Error**: 300 bytes (0x12C) – line info is mapped too early

## Relevant Code Paths

### 1. Device Linker Address Mapping

`DeviceLinkerAddressesMap` in `device_linker.cpp` (lines 776-888):
- Maps `[orig_text_start, orig_text_start + text_size)` → add `adjustment`
- `adjustment = new_text_start - orig_text_start`
- Used by: `getSubprogramRelocAdjustment()`, `getExprOpAddressRelocAdjustment()`

### 2. Parameters Passed to AddressesMap

```cpp
// For each specialized kernel (device_linker.cpp ~line 3034):
kernel_orig_text_addr = chunk.orig_text_addr;   // = text->addr (section start)
kernel_new_text_addr = text_addr_ + chunk.new_text_offset;
kernel_text_size = kernel->code.size();         // = func_end - text->addr
```

### 3. DWARFLinker Flow

1. `shouldKeepSubprogramDIE` → `getSubprogramRelocAdjustment` → `Unit.addFunctionRange(LowPc, HighPc, AddrAdjust)`
2. DWARFLinker clones the line table and applies `Unit.getRanges()` to adjust row addresses
3. `CustomStreamer::emitLineTableRows` emits `Row.Address.Address` directly (no further adjustment)

### 4. Potential Root Causes

| Hypothesis | Description |
|-----------|-------------|
| **A. LLVM DWARFLinker line table clone** | The DWARFLinker's line table cloning may use a different address base (e.g. subprogram low_pc) than the device linker's section-based mapping, or may mishandle AMDGPU address layout. |
| **B. kernel_text_size vs. .text size** | We use `kernel->code.size()` (extracted from func_start to func_end). The DWARF line table may cover a different range (e.g. full .text with padding). |
| **C. func_offset not accounted for** | The extracted code may have a leading offset (func_offset = func_start - text->addr). If the line table's first address is before the first extracted byte, the range lookup could fail or use wrong bounds. |
| **D. AMDGPU segment/addressing** | AMDGPU code objects may use segment-relative or other addressing that doesn't match the flat section-relative mapping we use. |

## Debugging Tools Created

- **`debug_dwarf_line_table.py`**: Analyzes a specialized .o file's DWARF line table, symbol addresses, and .text layout.
- **`compare_debug_line.sh`**: Compares .debug_line between original .o and merged ELF. Output in `dwarf_comparison_output/`.

## Investigation Completed

1. Debug script confirmed orig .text 0x1a00, func 0x1a00-0x7390, first line at 0x1b98.
2. Device linker AddressesMap verified correct (orig=0x1a00, new=0x6e0700, size=0x5970).
3. **Root cause found (fixed):** CustomStreamer::emitLineTableRows was passing AddressDelta in *instruction units* to MCDwarfLineAddr::encode(), but that LLVM function expects *bytes* (it internally divides by MinInstLength via ScaleAddrDelta). We were effectively dividing twice, yielding ~4x too small deltas. Fix: pass byte delta to encode(), use instruction units only for DW_LNS_advance_pc (EndSequence path).
4. **Duplicate .debug_line:** buildDebugLine was adding unpatched .debug_line/.debug_line_str; DWARFLinker added correct versions. Consumers use the first section, so we now skip emitting buildDebugLine's .debug_line when DWARFLinker is used.
