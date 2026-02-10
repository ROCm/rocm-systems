# Plan: Full Conversion to DWARFLinker (No Manual DWARF)

**Goal:** Make DWARFLinker the only path for merging debug info. Remove all manual concatenation, manual parsing, and manual patching of DWARF. Use LLVM’s `DwarfStreamer` (or equivalent) so that merged DWARF is produced entirely by the linker and is valid by construction.

**Reference:** `llvm-project/llvm/lib/DWARFLinker/Classic/DWARFStreamer.cpp` and `llvm-dwarfutil` in `llvm-project/llvm/tools/llvm-dwarfutil/DebugInfoLinker.cpp`.

---

## 1. High-Level Strategy

- **Single path:** Only DWARFLinker. Remove `kUseDWARFLinker` and the entire `mergeDebugInfo()` path (and any `mergeDebugSectionsViaLLVMEmission()` path).
- **Use the same pattern as llvm-dwarfutil:** Create `DWARFContext` from each **real object file** (dispatcher + each kernel `.o`) with **`ProcessDebugRelocations::Process`** so LLVM applies debug relocations when reading. One `DWARFFile` per input (name + `DWARFContext` + `AddressesMap`). Call `addObjectFile()` for each, then `setTargetDWARFVersion()`, then `link()`.
- **Emitter:** Either (A) use LLVM’s **`DwarfStreamer`** writing to a **buffer** (e.g. `raw_pwrite_stream` over `SmallVector`/`std::vector`), then parse that object file to extract `.debug_*` sections into `sections_`, or (B) implement a **buffer-based `DwarfEmitter`** that writes directly into `merged_debug_*` vectors (no MC layer). Option (A) reuses all of `DwarfStreamer`’s emission logic and avoids reimplementing line tables, ranges, abbrevs, etc.
- **AddressesMap:** One per input. Return the **relocation delta** for that object: `(text_addr_ + new_text_offset) - orig_text_addr`. No manual address patching.

---

## 2. Option A (Recommended): DwarfStreamer + Buffer Output

### 2.1 Flow

1. **Buffer:** Create a `raw_pwrite_stream` that writes to a buffer (e.g. `llvm::SmallVector<char>` or a custom stream that appends to `std::vector<uint8_t>`).
2. **Streamer:** Call `DwarfStreamer::createStreamer(Triple("amdgcn-amd-amdhsa"), OutputFileType::Object, BufferStream, WarningHandler)`. Ensure AMDGPU target is initialized (`InitializeAMDGPUTarget*` — already done in device_linker).
3. **Linker:** `DWARFLinker` with `setOutputDWARFEmitter(Streamer.get())`.
4. **Inputs:** For dispatcher and each kernel:
   - Open the **actual ELF file** (path from `disp_path_` or `kernel->source_file`).
   - `ObjectFile::createObjectFile(MemoryBuffer)`.
   - `DWARFContext::create(*Obj, ProcessDebugRelocations::Process, ...)` (critical: **Process**, not Ignore).
   - Build **DeviceLinkerAddressesMap** for this object: store `orig_text_addr`, `new_text_offset` (and `text_addr_`), implement `getSubprogramRelocAdjustment` / `getExprOpAddressRelocAdjustment` to return `(text_addr_ + new_text_offset) - orig_text_addr`; `hasValidRelocs()` = true.
   - `DWARFFile File(name, std::move(ctx), std::move(addr_map))`.
   - `linker.addObjectFile(File, nullptr, OnCUDieLoaded)` (track max DWARF version in callback).
5. **Link:** `setTargetDWARFVersion(max_version)`, then `linker.link()`.
6. **Finish:** `Streamer->finish()`.
7. **Extract sections:** Parse the buffer as an ELF object file (using LLVM Object APIs or your existing ELF parser). Read out `.debug_info`, `.debug_abbrev`, `.debug_str`, `.debug_str_offsets`, `.debug_addr`, `.debug_rnglists`, `.debug_ranges`, `.debug_line`, `.debug_line_str`. Add each as a `SectionInfo` into `sections_` (same as current “add merged debug sections” step).

### 2.2 Advantages

- No custom DIE/abbrev/line/ranges emission; `DwarfStreamer` handles all of it and produces valid DWARF.
- Matches how llvm-dwarfutil and dsymutil use DWARFLinker.
- Single place to fix if LLVM’s emission has a bug.

### 2.3 Dependencies

- LLVM build must include `DwarfStreamer` and the AMDGPU MC backend (already required for device linker).
- Need a `raw_pwrite_stream` implementation that writes to a resizable buffer and supports the object writer (e.g. `MCObjectStreamer` will write section bytes and fixups). LLVM’s `raw_svector_ostream` is `raw_ostream`; for `raw_pwrite_stream` you may need a small adapter or use an LLVM helper that backs an object writer with a buffer.

---

## 3. Option B: Custom Buffer-Based DwarfEmitter

If Option A is blocked (e.g. object writer not easy to drive into a buffer), implement a **custom `DwarfEmitter`** that writes only to `std::vector<uint8_t>` section buffers.

### 3.1 Required Methods (see `DWARFLinker.h`)

- **emitSectionContents(SecData, SecKind)** — append `SecData` to the appropriate `merged_debug_*` for `SecKind`.
- **emitAbbrevs(Abbrevs, DwarfVersion)** — serialize the abbreviation table into `merged_debug_abbrev_` (see `DwarfStreamer::emitAbbrevs`).
- **emitStrings(Pool)** — serialize string pool into `merged_debug_str_`.
- **emitStringOffsets(StringOffsets, TargetDWARFVersion)** — write `.debug_str_offsets` into `merged_debug_str_offsets_`.
- **emitLineStrings(Pool)** — into `merged_debug_line_str_`.
- **emitCompileUnitHeader(Unit, DwarfVersion)** — write CU header (length, version, unit_type, addr_size, abbrev_offset) into `merged_debug_info_`; track offset for this unit.
- **emitDIE(Die)** — serialize the DIE tree into `merged_debug_info_` (see `AsmPrinter::emitDwarfDIE` / DwarfStreamer’s use of it; you’d need to replicate DIE encoding or use a lower-level LLVM helper if available).
- **emitDwarfDebugRangeListHeader / Fragment / Footer** — for `.debug_rnglists` / `.debug_ranges`.
- **emitDwarfDebugAddrsHeader / emitDwarfDebugAddrs / Footer** — for `.debug_addr`.
- **emitLineTableForUnit(LineTable, Unit, ...)** — re-emit line table into `merged_debug_line_` with addresses and line_strp offsets adjusted (this is the heaviest; DwarfStreamer has helpers for prologue and rows).
- **emitDebugNames / emitApple*** — can be no-ops or minimal if you don’t need accelerator tables.
- **emitCIE / emitFDE** — no-op unless you merge `.debug_frame`.
- **emitMacroTables** — no-op if you don’t need macros.
- **get*SectionSize()** — return current size of the corresponding buffer.
- **finish()** — no-op or finalize.

### 3.2 MCSymbol Handling

The interface uses `MCSymbol*` for range/list headers (e.g. `emitDwarfDebugRangeListHeader` returns `MCSymbol* EndLabel`; footer emits the label). Options:

- **Fake symbols:** Create a minimal `MCContext` and create temp symbols; map each symbol to the current offset in the appropriate buffer when “emitting” the label; when the linker later asks for a “label difference,” compute offset difference from your map. This is brittle if the linker assumes a real MC layer.
- **Offsets instead of symbols:** If the linker only needs to backpatch a length, you can record “pending length at offset X, fill when we have EndOffset” and patch the 4-byte length in the buffer in the Footer call. That avoids MC entirely but may require local changes or a fork of the linker callbacks.

Option B is more work and duplicates logic already in `DwarfStreamer`; prefer Option A unless blocked.

---

## 4. AddressesMap (Both Options)

Implement **DeviceLinkerAddressesMap** per input object:

- **Constructor:** Take `orig_text_addr`, `new_text_addr` (i.e. `text_addr_ + new_text_offset`), and optionally size of the text segment for this object.
- **hasValidRelocs():** return `true` (we always have “live” code in the merged image).
- **getSubprogramRelocAdjustment(DIE, Verbose):** If the DIE has an address (e.g. `DW_AT_low_pc`), return `new_text_addr - orig_text_addr`. Otherwise `std::nullopt`.
- **getExprOpAddressRelocAdjustment(U, Op, Start, End, Verbose):** For `DW_OP_addr` / `DW_OP_addrx` etc., return the same delta `new_text_addr - orig_text_addr`.
- **getLibraryInstallName():** `std::nullopt`.
- **applyValidRelocs:** return `false` (we don’t apply relocs to our output buffer; the linker has already used the map when cloning).
- **needToSaveValidRelocs:** return `false`.
- **updateAndSaveValidRelocs / updateRelocationsWithUnitOffset:** no-op.
- **clear():** no-op.

This is the same idea as llvm-dwarfutil’s `ObjFileAddressMap`, but returning a non-zero delta.

---

## 5. What to Remove (One Edit Is Fine)

Remove or stop using in one pass:

### 5.1 Flags and Branches

- **Remove** `kUseDWARFLinker` and `kUsePhase3Emission`. There is only one path: DWARFLinker.
- In `link()`, **remove** the `if (kUseDWARFLinker) ... else if (kUsePhase3Emission) ... else mergeDebugInfo()`. Always call the new “merge via DWARFLinker” function (e.g. `mergeDebugInfoWithDWARFLinker()` renamed to `mergeDebugInfo()` or inlined).

### 5.2 collectSections() — Debug Merging

- **Remove** all code that merges `.debug_abbrev`, `.debug_str`, `.debug_str_offsets`, `.debug_addr`, `.debug_rnglists`, `.debug_ranges`, `.debug_info` from dispatcher and kernels into `merged_debug_*` and builds `debug_info_chunks_` (the block that starts around “First add dispatcher’s debug sections” and “Then add each specialized kernel’s debug sections”).
- **Remove** building of `debug_line_chunks_` and appending to `debug_line.data` and `merged_debug_line_str_` for the purpose of debug info (DWARFLinker will emit `.debug_line` and `.debug_line_str`). If `.debug_line` is still needed for something else before link, keep only what’s necessary; otherwise remove.
- **Keep** in `collectSections()` everything that builds non-debug sections (e.g. `.text`, `.rodata`, symtabs, etc.) and layout. **Keep** `kernel_text_offsets_` and any other state needed for `AddressesMap` (e.g. `text_addr_` is set in `computeLayout()`).

### 5.3 Entire Functions to Remove

- **mergeDebugInfo()** — full body (manual CU header patching, `createTempElfForMergedDebugSections`, `patchAddressesUsingLLVM`, manual `.debug_str_offsets` / `.debug_ranges` patching, attribute patching, line_strp patching, adding sections).
- **mergeDebugSectionsViaLLVMEmission()** — if present.
- **patchAddressesUsingLLVM()**.
- **createTempElfForMergedDebugSections()**.
- **findDwarfAttrPositionsUsingLLVM()** and **findDwarfAttrPositionsFromElf()**.
- **patchStrOffsetsBaseToZero()**.
- **createMinimalElfForDwarf()** and **createMinimalElfForLineTable()** (minimal ELF builders for LLVM parsing).
- **patchDwarf5StringOffsets()** (line table patching).
- **findLineStrpInChunkManual()**.
- **parseAbbrevTable()**, **parseAbbrevTableWithAttrs()**.
- **getFormFixedSize()**, **skipVariableForm()** (if only used by the above).
- **getDwarfVersionFromDebugInfo()** can stay if still needed for a quick check; otherwise remove.

### 5.4 Data Structures to Remove or Simplify

- **DebugInfoChunk** and **debug_info_chunks_** — not needed if we never do manual merge; remove. (If you keep them only for “which kernel is at which offset” for logging, you can keep a minimal list; otherwise remove.)
- **DwarfAttrPositions** and any `dwarf_attr_positions` on kernels — remove.
- **DebugLineChunk** and **debug_line_chunks_** — remove if line table is only from DWARFLinker.
- **KernelInfo** — drop `.debug_*` and `dwarf_attr_positions`; keep only what’s needed for loading the object and building `AddressesMap` (e.g. path, `orig_text_addr`; `new_text_offset` comes from `kernel_text_offsets_`).

### 5.5 Extraction in extractKernelInfo()

- **Remove** extraction of `.debug_abbrev`, `.debug_info`, `.debug_str`, `.debug_str_offsets`, `.debug_addr`, `.debug_rnglists`, `.debug_ranges` into `KernelInfo` if we no longer merge them manually. We only need to open the kernel file again when building `DWARFFile` for the linker (and we have the path). So you can remove that extraction and the relocations application for debug sections there.

### 5.6 Current mergeDebugInfoWithDWARFLinker()

- **Replace** its body with the new flow: create streamer (Option A) or buffer emitter (Option B), create one `DWARFFile` per input with `ProcessDebugRelocations::Process` and `DeviceLinkerAddressesMap`, add all, set target version, link, finish, then either parse streamer buffer (A) or use buffers already filled (B) and add sections to `sections_`.
- **Remove** the old “clear merged_*, create DeviceLinkerDwarfEmitter, create minimal ELF per kernel” approach that led to the “invalid abbreviation” error. No minimal ELF for inputs — only full object files.

---

## 6. Order of Objects and Target Version

- Add **dispatcher first**, then **all kernel objects** in the same order as `kernel_text_offsets_` (so that layout and `new_text_offset` match).
- In **OnCUDieLoaded**, track `max_dwarf_version = std::max(max_dwarf_version, Unit.getVersion())`.
- After all **addObjectFile** calls, call **setTargetDWARFVersion(max_dwarf_version)** (or a minimum like 5 if you require DWARF5).
- Then **link()**.

---

## 7. Validation and Testing

- **llvm-dwarfdump --verify** on the merged ELF (e.g. `merged_device.elf` or the final output that contains debug sections) must report **no errors**.
- **rocgdb** and **llvm-symbolizer** on addresses in the merged `.text` should show correct source lines and symbols.
- **Existing tests:** Re-run any device-linker or smoke tests (e.g. `smoke_test`, build_with_device_linker) and ensure they still pass.
- **Reproducibility:** Keep a fixed set of inputs (one build’s dispatcher + kernel objects) for before/after comparison.

---

## 8. Suggested Implementation Order

1. **Implement DeviceLinkerAddressesMap** (per-object delta; `hasValidRelocs` true; reloc adjustment = `new_text_addr - orig_text_addr`). Unit test with a single object if possible.
2. **Switch to ProcessDebugRelocations::Process** when creating `DWARFContext` for each input in the current `mergeDebugInfoWithDWARFLinker()` and feed **real ELF files** (dispatcher path + kernel path from `kernel->source_file`). Keep the current custom emitter temporarily; see if “invalid abbreviation” goes away with Process + real files.
3. If the current custom emitter still fails, **implement Option A:** buffer stream + `DwarfStreamer::createStreamer` + parse output ELF to get `.debug_*` and add to `sections_`. Remove the old custom emitter.
4. **Remove** all manual merge/patch code and the `mergeDebugInfo()` path in one pass as in §5.
5. **Run** `llvm-dwarfdump --verify` and tests; fix any regressions.

---

## 9. Files to Touch

- **device_linker.cpp:** Main changes (single merge path, remove manual DWARF, add Option A or B, AddressesMap, input feeding).
- **CMakeLists.txt (project):** No change if already linking full LLVM (with DWARFLinker and MC). If you switch to a custom buffer emitter that uses only DWARFLinker (no MC), you might avoid pulling in AMDGPU MC; otherwise keep as is.
- **DEVICE_LINKER.md / CONTEXT.md:** Update to state that debug merging is done solely by DWARFLinker; no manual patching.

---

## 10. Summary

| Current | After full conversion |
|--------|------------------------|
| Two paths (DWARFLinker + manual) | Single path: DWARFLinker only |
| Manual concat + patch of .debug_* | Linker produces all .debug_* via DwarfStreamer or buffer emitter |
| Minimal ELF per kernel for DWARFLinker | Real object file per input; ProcessDebugRelocations::Process |
| Custom emitter (never reached due to input error) | DwarfStreamer → buffer → extract sections (Option A) or buffer-based emitter (Option B) |
| AddressesMap returns delta but inputs wrong | Same AddressesMap; inputs fixed (real ELF + Process) |
| 800+ chunks, abbrev/CU/line/addr patching | No chunks; no manual patching |

Doing the conversion in **one edit** is acceptable: remove all code listed in §5 and replace the merge path with the Option A (or B) flow above.
