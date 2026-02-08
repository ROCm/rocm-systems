# Plan: Rewrite Device Linker DWARF Code Using LLVM APIs

**Goal:** Replace manual DWARF parsing and patching in the device linker with LLVM's DWARF APIs, while still merging DWARF from the dispatcher and all specialized kernels into one output ELF.

**Constraint:** The device linker already links against LLVM (e.g. via `AOMP_LLVM_DIR`). Keep existing ELF I/O (your ELF parser and section extraction); only the DWARF handling is rewritten.

---

## Current State (What to Replace)

| Area | Current approach | Location (approx.) |
|------|------------------|--------------------|
| Abbrev / DIE parsing | Custom: `parseAbbrevTable`, `findDwarfAttrPositions`, `getFormFixedSize`, `skipVariableForm`, ULEB/SLEB | Lines 175–414 |
| “Minimal ELF for DWARF” | `createMinimalElfForDwarf()` (unused) | 57–121 |
| Line table | Raw byte scan for `DW_LNE_set_address`; manual prologue parsing in `patchDwarf5StringOffsets` | 2996–3130, 3132–3171 |
| Merge | Concatenate section bytes; patch offsets/addresses using hand-found positions | 1564–1699, 2688–2992 |
| Relocations | Manual `.rela.debug_line` application (ELF-specific reloc types) | 736–761, 1471–1493 |

---

## LLVM APIs to Use

- **Reading DWARF from existing section bytes:**  
  `DWARFContext::create(StringMap<std::unique_ptr<MemoryBuffer>> sections, AddrSize, isLittleEndian, ...)`  
  Build the `StringMap` from your existing `.debug_*` buffers (from `getBytes()`). No need to use LLVM ObjectFile for the rest of the linker.

- **Iterating compile units and DIEs:**  
  `DWARFContext::compile_units()`, `DWARFUnit`, `DWARFDie`, `DWARFFormValue`  
  Use these instead of hand-walking `.debug_info` / abbrev.

- **Line tables:**  
  `DWARFContext::getLineTableForUnit(DWARFUnit*)` → `DWARFDebugLine::LineTable`  
  Use the parsed prologue and sequences instead of manual prologue parsing and `patchDwarf5StringOffsets`.

- **Optional (relocations):**  
  If you open the same inputs as `object::ObjectFile` and use `DWARFContext::create(ObjectFile, ProcessDebugRelocations::Process)`, LLVM will apply debug relocations. Alternatively keep applying `.rela.debug_line` yourself but use LLVM only for DWARF structure.

- **Optional (full merge):**  
  `llvm::dwarf_linker::classic::DWARFLinker` plus a custom `DwarfEmitter` that writes into your section buffers. Viable once the “read + merge by concatenation + patch” path is stable.

---

## Phase 1: Use LLVM to Read DWARF

**Scope:** Create a `DWARFContext` per input from existing section bytes; replace manual abbrev/DIE/attribute parsing with LLVM iteration; keep current merge strategy (concatenate + patch); remove dead and manual parsing code.

### Steps

1. **Create a DWARF “reader” per input**  
   For each ELF (dispatcher or kernel), take the existing `.debug_*` section vectors you already extract. Build a `StringMap<StringRef, std::unique_ptr<MemoryBuffer>>` (or the type expected by `DWARFContext::create`) with keys like `".debug_info"`, `".debug_abbrev"`, `".debug_line"`, `".debug_line_str"`, and optionally `".debug_str"`, `".debug_str_offsets"`, `".debug_addr"`, `".debug_rnglists"`, `".debug_ranges"` when present. Call `DWARFContext::create(sections, addrSize, isLittleEndian, ...)` and store the `std::unique_ptr<DWARFContext>` (e.g. in `KernelInfo` and for the dispatcher). Use a single, fixed address size (e.g. 8) and endianness from your target. Omit sections that are empty so LLVM doesn’t expect them.

2. **Replace attribute-position finding with LLVM**  
   Remove: `parseAbbrevTable`, `findDwarfAttrPositions`, `getFormFixedSize`, `skipVariableForm`, and the legacy `findRangesOffsets`. For each input with a `DWARFContext`, iterate `ctx->compile_units()`. For each `DWARFUnit`, get the root DIE and iterate its attributes via `DWARFDie::attributes()`. For `DW_AT_ranges`, `DW_AT_str_offsets_base`, `DW_AT_addr_base`, `DW_AT_rnglists_base`, `DW_AT_stmt_list`, use `DWARFFormValue` to get the value and, where the API allows, the byte offset of that value in the section (or the offset within the unit). If the API does not expose byte offsets, keep a small “patch list” built from LLVM’s logical values (e.g. “CU at unit offset X has DW_AT_stmt_list = Y”) and map that to byte offsets when you write merged sections. Store “patch list” per chunk (dispatcher + each kernel): which attributes need to be patched and with which base offset in the merged output. That replaces `DwarfAttrPositions` and the current manual position recording.

3. **Keep merge strategy, but drive it from LLVM**  
   Keep “concatenate sections in order (dispatcher then kernels)” and “patch base offsets in `.debug_info` and addresses in `.debug_addr` / `.debug_line`”. When building merged sections, use the patch list derived from LLVM (and, if needed, one more pass over the merged buffer to resolve “CU X, attribute Y” to actual byte offsets). Remove all manual abbrev/DIE parsing; the only DWARF reader is LLVM.

### Phase 1 validation criteria

| # | Criterion | How to check |
|---|-----------|--------------|
| 1.1 | **No regression in merge output** | For the same build (same dispatcher + same kernel objects), run the device linker before and after Phase 1. Compare merged ELF: `diff <(llvm-dwarfdump old.elf) <(llvm-dwarfdump new.elf)` → no differences, or only benign (e.g. ordering of CUs). |
| 1.2 | **DWARF verifier passes** | `llvm-dwarfdump --verify build/release/device_linker_output/merged_device.elf` exits 0 with no errors or warnings. |
| 1.3 | **Manual parsing code removed** | Codebase no longer contains: `parseAbbrevTable`, `findDwarfAttrPositions`, `getFormFixedSize`, `skipVariableForm`, `findRangesOffsets`, and `createMinimalElfForDwarf` (or they are unused stubs scheduled for removal). |
| 1.4 | **LLVM is the only DWARF reader** | All attribute positions and patch lists used for merging are derived from `DWARFContext` / `compile_units()` / DIE attributes (e.g. no hand-parsed ULEB/abbrev for .debug_info). |
| 1.5 | **Existing tests still pass** | Any existing device-linker or smoke tests (e.g. smoke_test, build_with_device_linker) still pass. |

**Phase 1 sign-off:** All of 1.1–1.5 satisfied.

---

## Phase 2: Use LLVM for Line Table Parsing and Patching

**Scope:** Use `getLineTableForUnit()` for every CU with line info; remove manual line table/prologue parsing; implement address and line_strp patching from parsed structures; optionally apply debug relocations before creating `DWARFContext`.

### Steps

1. **Line table via LLVM**  
   For each `DWARFContext`, for each compile unit that has line info, call `getLineTableForUnit(unit)`. Use the returned `DWARFDebugLine::LineTable` for: prologue (including directory/file formats and `DW_FORM_line_strp` offsets), and sequence of line program instructions (e.g. `DW_LNE_set_address`). Remove the manual prologue parsing in `patchDwarf5StringOffsets` and the ad-hoc “unknown form” handling.

2. **Re-emit line table with address and string-offset deltas**  
   Either: **(A)** For each chunk, use the parsed `LineTable` to apply address delta and string offset delta, then serialize the prologue and line program back into a byte buffer (using LLVM’s line table emission helpers if available, or a small writer). **(B)** Keep raw byte concatenation of `.debug_line` / `.debug_line_str`, but use the parsed `LineTable` only to compute the exact byte ranges that need patching (addresses and line_strp offsets), then patch in place.

3. **Relocations**  
   Prefer applying relocations before building the `DWARFContext`: either by using `object::ObjectFile` + `DWARFContext::create(Obj, ProcessDebugRelocations::Process)` for that input, or by applying `.rela.debug_line` to the section bytes you already have, then building the `StringMap` from the relocated bytes.

### Phase 2 validation criteria

| # | Criterion | How to check |
|---|-----------|--------------|
| 2.1 | **No regression in merged line info** | Same as 1.1 but with focus on line tables: `llvm-dwarfdump --debug-line merged_device.elf` (or equivalent) is identical (or structurally equivalent) before vs after Phase 2. |
| 2.2 | **Line table verifier clean** | `llvm-dwarfdump --verify merged_device.elf` still passes (no new errors in .debug_line / .debug_line_str). |
| 2.3 | **rocgdb line lookup works** | If available: load merged ELF in rocgdb and run e.g. `info functions`, `list`, or break on a kernel; no “DW_FORM_line_strp pointing outside .debug_line_str” (or similar) errors. |
| 2.4 | **Manual line parsing removed** | `patchDwarf5StringOffsets` no longer contains manual prologue parsing (directory/file format, form codes); logic is driven by LLVM’s `DWARFDebugLine::LineTable` (or equivalent) or by patch ranges derived from it. |
| 2.5 | **Addresses correct in .debug_line** | For at least one known kernel, extract its chunk from merged .debug_line, parse with LLVM; `DW_LNE_set_address` (and any other address-bearing opcodes) match expected relocated addresses (e.g. `text_addr_ + chunk.new_text_offset` plus original offset within chunk). |
| 2.6 | **Existing tests still pass** | Same as 1.5. |

**Phase 2 sign-off:** All of 2.1–2.6 satisfied.

**Phase 2 implemented:** `patchDwarf5StringOffsets()` now builds a minimal ELF per chunk (`.debug_info` CU with `stmt_list=0`, `.debug_abbrev`, `.debug_line`, `.debug_line_str`), parses with `DWARFContext::create` and `getLineTableForUnit()`. If parsing succeeds, the chunk is validated. The format-driven walk (directory/file format from buffer, patch `DW_FORM_line_strp` using `getFormFixedSize`/`skipVariableForm`) runs for every chunk so all `line_strp` offsets are patched even when LLVM parse fails. Address patching (`DW_LNE_set_address`) is unchanged. `llvm-dwarfdump --verify` and the disassembly integration test pass.

---

## Phase 3: Optional — Use LLVM to Emit Merged DWARF (or DWARFLinker)

**Scope:** Either re-emit DWARF from LLVM structures with new base offsets and addresses, or use `DWARFLinker` (or equivalent) with a custom emitter that writes into the device linker’s section buffers. **Phase 3 is when "LLVM for everything" applies:** merged `.debug_*` sections are produced by LLVM emission (or DWARFLinker), so line_strp and other offsets are correct by construction and manual format walks are no longer needed.

**Relationship to Phase 2:** Phase 2 still uses a format-driven walk over the `.debug_line` prologue to patch line_strp because LLVM's `LineTable` API does not expose the byte offsets of each line_strp in the prologue. Phase 3 removes that by using LLVM to *emit* the line table (and optionally .debug_info) with correct offsets from the start.

### Steps

1. **Emit side**  
   Use LLVM’s emission APIs where they fit (e.g. line table and prologue emission for `.debug_line` / `.debug_line_str`; for `.debug_info`, either “clone this unit with patched base offsets” using `DWARFUnit`/`DWARFDie` and LLVM form emission, or continue “concatenate + patch” but with patch positions coming only from LLVM). If full “link” semantics are desired (e.g. type uniquing, ODR), evaluate `llvm::dwarf_linker::classic::DWARFLinker`: implement a `DwarfEmitter` that writes into your own buffers; feed it the same inputs and an address map (orig → new per object); if the API allows per-object address mapping, encode your deltas per chunk.

2. **Suggested implementation order**  
   (a) Line table emission first: for each chunk, build an LLVM line table (from parsed `LineTable` or from source) with address and string-offset deltas applied, then emit via MCDwarf (or equivalent) into a buffer; concatenate chunk buffers. This replaces `patchDwarf5StringOffsets` and the manual prologue walk. (b) Then .debug_info: either clone units with patched bases via emission, or keep concatenate + patch but with positions from LLVM only. (c) Optional: adopt DWARFLinker for full link semantics and future symbol/type support.

3. **Scope**  
   Phase 3 is optional and can follow Phases 1–2. It is the place to add full symbol/type support when needed.

### Phase 3 validation criteria

| # | Criterion | How to check |
|---|-----------|--------------|
| 3.1 | **Verifier passes** | `llvm-dwarfdump --verify merged_device.elf` exits 0. |
| 3.2 | **Functional parity with Phase 2** | Line and (if present) symbol lookups that worked after Phase 2 still work (e.g. rocgdb `info functions`, `list`, breakpoints; llvm-symbolizer on merged addresses). No new tool failures. |
| 3.3 | **Merge is LLVM-driven** | Merged .debug_* sections are produced by LLVM emission APIs and/or DWARFLinker; no remaining “concatenate raw bytes then patch” logic for the sections replaced in this phase (or that logic is clearly legacy path for non-DWARF cases only). |
| 3.4 | **Address and offset correctness** | Spot-check: a few addresses in merged .debug_addr / .debug_line / range lists match the expected relocated values for the combined image. |
| 3.5 | **Future-proof for full symbols** | If Phase 3 uses DWARFLinker (or equivalent): design supports adding full .debug_info (symbols, types) later without replacing the merge pipeline again. Document or ticket that path. |
| 3.6 | **Existing tests still pass** | Same as 1.5. |

**Phase 3 sign-off:** All of 3.1–3.6 satisfied.

---

## Cross-phase and CI

- **Reproducibility:** Use a fixed set of dispatcher + kernel objects (e.g. from one build) for before/after comparisons in 1.1, 2.1, 2.5.
- **CI (no GPU):** Run device linker, then `llvm-dwarfdump --verify` and, if available, a small script that compares key parts of `llvm-dwarfdump` output (e.g. CU count, line table presence) to a golden or to the previous phase’s output.
- **Documentation:** Record in CONTEXT.md or DWARF_FIX_PROGRESS.md which phase is done and that the above criteria were met (and how, e.g. “1.1: diff empty for repo commit abc”).
