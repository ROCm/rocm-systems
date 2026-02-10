# DWARFLinker Full Conversion – Progress Log

**Started:** 2026-02-09  
**Plan:** `DWARFLINKER_FULL_CONVERSION_PLAN.md`  
**Rule:** Stop if any step diverges significantly from plan. No sudo.

---

## Step 1: DeviceLinkerAddressesMap

- [x] Verify returns delta = (text_addr_ + new_text_offset) - orig_text_addr — AddressesMap already correct (new_text_start_ - orig_text_start_).
- [x] Verify hasValidRelocs() = true — has_relocs_ = true.
- [x] Add setNumThreads(1) before link() — added before linker.link().

---

## Step 2: Process + real ELF

- [x] Use ProcessDebugRelocations::Process when creating DWARFContext — changed for both dispatcher and kernel DWARFContext::create (was Ignore).
- [x] Feed real ELF files — already using real ELF (MappedFile + ObjectFile) for kernels and dispatcher.

---

## Step 3: DwarfStreamer only (custom emitter removed)

- [x] Custom emitter path removed per user request. Only DwarfStreamer is used; if createStreamer fails, merge fails with fatal error.
- [x] createStreamer made to succeed: LLVMInitializeAMDGPUTargetInfo, LLVMInitializeAMDGPUTarget, LLVMInitializeAMDGPUTargetMC, LLVMInitializeAMDGPUAsmPrinter called before createStreamer.
- [x] Post-link: parse streamer buffer as object file, extract .debug_* sections, add to sections_.
- [ ] **Current blocker:** Abort (SIGABRT, exit 134) during `linker.link()` after adding 860 objects. Crash immediately after "Linking DWARF info (added 860 objects...)". Need to debug LLVM DWARFLinker/DwarfStreamer with AMDGPU + many CUs (e.g. run under gdb to get backtrace).

---

## Step 4: Remove manual DWARF code

- [ ] Remove mergeDebugInfo(), patchAddressesUsingLLVM, createTempElf*, findDwarfAttrPositions*, patchStrOffsetsBaseToZero, createMinimalElf*, patchDwarf5StringOffsets, parseAbbrev*, getFormFixedSize, skipVariableForm, etc.
- [ ] Remove debug merge from collectSections(); single path = DWARFLinker only

---

## Step 5: Verify

- [ ] llvm-dwarfdump --verify on merged_device.elf
- [ ] Build succeeds; note DWARF phase time and RSS if possible

---

## Notes

(Progress and decisions recorded below.)
