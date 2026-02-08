# Plan: Adopt DWARFLinker for DWARF Merging

## Goal
Replace manual DWARF patching with `llvm::dwarf_linker::classic::DWARFLinker` to fix remaining DWARF issues (overlapping ranges, containment errors) and eliminate error-prone manual patching.

## Current Problems
1. **Range list containment errors**: 36,480 "DIE address ranges are not contained by parent ranges" errors
2. **Overlapping ranges**: 11,856 "DIEs have overlapping address ranges" errors  
3. **Manual patching complexity**: Current code manually patches `.debug_addr` and `.debug_rnglists` by:
   - Finding addresses via LLVM parsing
   - Manually patching raw bytes with address deltas
   - Handling different range list entry types (DW_RLE_startx_endx, DW_RLE_offset_pair, etc.)
4. **CU low_pc issues**: CU's `DW_AT_low_pc` patching doesn't always match what range lists expect

## Why DWARFLinker?
- **Purpose-built for linking**: Designed to merge DWARF from multiple objects with address relocation
- **Handles all DWARF5 features**: Properly handles `.debug_rnglists`, `.debug_addr`, range containment
- **Address mapping**: Built-in support for per-object address relocation (exactly what we need)
- **Valid output**: Produces valid DWARF by construction, eliminating patching errors

## Implementation Plan

### Phase 1: Research and Setup (1-2 days)

#### 1.1 Understand DWARFLinker API
- [ ] Study `llvm::dwarf_linker::classic::DWARFLinker` API documentation
- [ ] Review LLVM source: `llvm/lib/DWARFLinker/DWARFLinker.cpp`
- [ ] Understand `DwarfEmitter` interface requirements
- [ ] Check if parallel or classic version is better for our use case

#### 1.2 Identify Required Components
- [ ] `DWARFLinker` - main linker class
- [ ] `DwarfEmitter` - custom emitter that writes to our buffers
- [ ] Address map - maps original addresses to new addresses per object
- [ ] Object file inputs - need to provide DWARFContext for each input

#### 1.3 Check LLVM Version Compatibility
- [ ] Verify ROCm LLVM version has DWARFLinker (check `/opt/rocm/llvm/include/llvm/DWARFLinker/`)
- [ ] If not available, check what version introduced it
- [ ] Document any version requirements

### Phase 2: Custom DwarfEmitter Implementation (2-3 days)

#### 2.1 Create Custom Emitter Class
```cpp
#include "llvm/DWARFLinker/Classic/DWARFLinker.h"
#include "llvm/DWARFLinker/DWARFLinkerBase.h"

class DeviceLinkerDwarfEmitter : public llvm::dwarf_linker::classic::DwarfEmitter {
    // Write to device linker's section buffers
    std::vector<uint8_t>& debug_info_out_;
    std::vector<uint8_t>& debug_abbrev_out_;
    std::vector<uint8_t>& debug_str_out_;
    std::vector<uint8_t>& debug_str_offsets_out_;
    std::vector<uint8_t>& debug_addr_out_;
    std::vector<uint8_t>& debug_rnglists_out_;
    std::vector<uint8_t>& debug_line_out_;
    std::vector<uint8_t>& debug_line_str_out_;
    
public:
    DeviceLinkerDwarfEmitter(/* buffers */);
    
    // Required DwarfEmitter methods:
    void emitSectionContents(llvm::StringRef SecData, 
                             llvm::dwarf_linker::DebugSectionKind SecKind) override;
    void emitAbbrevs(const std::vector<std::unique_ptr<DIEAbbrev>>& Abbrevs,
                     unsigned DwarfVersion) override;
    void emitStrings(const NonRelocatableStringpool& Pool) override;
    void emitStringOffsets(const llvm::SmallVector<uint64_t>& StringOffsets,
                          uint16_t TargetDWARFVersion) override;
    void emitLineStrings(const NonRelocatableStringpool& Pool) override;
    void emitCompileUnitHeader(CompileUnit& Unit, unsigned DwarfVersion) override;
    void emitDIE(DIE& Die) override;
    // ... other required methods
};
```

#### 2.2 Implement Required Emitter Methods
- [ ] `emitAbbrevs()` - emit abbreviation tables
- [ ] `emitCompileUnitHeader()` - emit CU headers
- [ ] `emitDIE()` - emit debug information entries
- [ ] `emitDebugLine()` - emit line tables
- [ ] `emitDebugAddr()` - emit address tables
- [ ] `emitDebugRngLists()` - emit range lists
- [ ] `emitDebugStr()` - emit string tables
- [ ] `emitDebugStrOffsets()` - emit string offset tables

#### 2.3 Handle Address Mapping
- [ ] Create address map per input object: `orig_addr -> new_addr`
- [ ] For each kernel: `new_addr = text_addr_ + chunk.new_text_offset + (orig_addr - chunk.orig_text_addr)`
- [ ] For dispatcher: similar mapping
- [ ] Pass address map to DWARFLinker

### Phase 3: Integration with Device Linker (3-4 days)

#### 3.1 Replace mergeDebugInfo() Logic
- [ ] Create new `mergeDebugInfoWithDWARFLinker()` function
- [ ] Create `DeviceLinkerDwarfEmitter` instance with output buffers
- [ ] Create `DWARFLinker` instance:
  ```cpp
  auto ErrorHandler = [](const Twine& Err, StringRef Context, const DWARFDie* DIE) {
      fprintf(stderr, "DWARFLinker error: %s\n", Err.str().c_str());
  };
  auto WarningHandler = [](const Twine& Warn, StringRef Context, const DWARFDie* DIE) {
      fprintf(stderr, "DWARFLinker warning: %s\n", Warn.str().c_str());
  };
  auto StringsTranslator = [](StringRef Str) { return Str; };  // No translation needed
  
  llvm::dwarf_linker::classic::DWARFLinker linker(ErrorHandler, WarningHandler, StringsTranslator);
  linker.setEmitter(&emitter);
  ```
- [ ] For each input (dispatcher + kernels):
  - [ ] Create `DWARFContext` from input ELF sections (already have this)
  - [ ] Create `DeviceLinkerAddressesMap` with address mapping for that input
  - [ ] Create `DWARFFile`:
    ```cpp
    auto addresses = std::make_unique<DeviceLinkerAddressesMap>(
        chunk.orig_text_addr, text_addr_ + chunk.new_text_offset, chunk.text_size);
    auto dwarf_file = std::make_unique<DWARFFile>(
        chunk.source_file, std::move(dwarf_context), std::move(addresses));
    linker.addObjectFile(*dwarf_file);
    ```
- [ ] Call `linker.link()` - produces output via emitter
- [ ] Extract merged sections from emitter buffers (already in vectors)

#### 3.2 Handle Section Concatenation
- [ ] DWARFLinker will merge all inputs into single sections
- [ ] No need for manual concatenation - linker handles it
- [ ] Verify section sizes match expectations

#### 3.3 Update Section Writing
- [ ] Replace `merged_debug_*_` vectors with emitter output
- [ ] Update `writeOutput()` to use new merged sections
- [ ] Remove old patching code paths

### Phase 4: Address Mapping Details (1-2 days)

#### 4.1 Implement Custom AddressesMap
```cpp
#include "llvm/DWARFLinker/AddressesMap.h"

class DeviceLinkerAddressesMap : public llvm::dwarf_linker::AddressesMap {
    uint64_t orig_text_start_;  // Original .text address in input
    uint64_t new_text_start_;    // New .text address in merged output  
    uint64_t text_size_;         // Size of .text section
    bool has_relocs_;            // Whether relocations are valid
    
public:
    DeviceLinkerAddressesMap(uint64_t orig_start, uint64_t new_start, uint64_t size)
        : orig_text_start_(orig_start), new_text_start_(new_start), 
          text_size_(size), has_relocs_(true) {}
    
    bool hasValidRelocs() override { return has_relocs_; }
    
    std::optional<int64_t> getSubprogramRelocAdjustment(
        const llvm::DWARFDie& DIE, bool Verbose) override {
        // Get address from DIE, compute delta
        auto low_pc = DIE.find(llvm::dwarf::DW_AT_low_pc);
        if (!low_pc) return std::nullopt;
        
        uint64_t addr;
        if (low_pc->getAsAddress(&addr)) {
            if (addr >= orig_text_start_ && addr < orig_text_start_ + text_size_) {
                return (int64_t)new_text_start_ - (int64_t)orig_text_start_;
            }
        }
        return std::nullopt;
    }
    
    std::optional<int64_t> getExprOpAddressRelocAdjustment(
        llvm::DWARFUnit& U, const llvm::DWARFExpression::Operation& Op,
        uint64_t StartOffset, uint64_t EndOffset, bool Verbose) override {
        // Similar logic for expression operands
        // ...
    }
    
    bool applyValidRelocs(llvm::MutableArrayRef<char> Data, uint64_t BaseOffset,
                         bool IsLittleEndian) override {
        // Apply relocations if needed (may not be necessary if AddressesMap handles it)
        return false;
    }
    
    // ... implement other required methods
};
```

#### 4.2 Handle Special Addresses
- [ ] Zero addresses (CU low_pc placeholder) - let DWARFLinker handle
- [ ] UINT64_MAX (invalid/sentinel) - preserve as-is
- [ ] Addresses outside .text - no relocation needed

### Phase 5: Testing and Validation (2-3 days)

#### 5.1 Basic Functionality
- [ ] Build with new code
- [ ] Verify no crashes
- [ ] Check merged ELF has all expected sections
- [ ] Verify section sizes are reasonable

#### 5.2 DWARF Validation
- [ ] Run `llvm-dwarfdump --verify merged_device.elf`
- [ ] **Target**: Zero errors (currently 48,339 errors)
- [ ] Verify no "containment" errors
- [ ] Verify no "overlapping ranges" errors

#### 5.3 Functional Testing
- [ ] Test with rocgdb: `info functions`, `list`, breakpoints
- [ ] Test with llvm-symbolizer on merged addresses
- [ ] Compare before/after: same functionality, fewer errors

#### 5.4 Regression Testing
- [ ] Run existing device linker tests
- [ ] Verify no performance regressions
- [ ] Check build time impact

### Phase 6: Cleanup (1 day)

#### 6.1 Remove Old Code
- [ ] Delete `patchAddressesUsingLLVM()` - no longer needed
- [ ] Delete manual `.debug_rnglists` patching code
- [ ] Delete manual `.debug_addr` patching code
- [ ] Delete `createTempElfForMergedDebugSections()` if unused
- [ ] Keep `mergeDebugInfo()` as fallback or delete if DWARFLinker works

#### 6.2 Update Documentation
- [ ] Update `CONTEXT.md` with DWARFLinker approach
- [ ] Update `DWARF_FIX_PROGRESS.md` with completion status
- [ ] Document any limitations or gotchas

## Implementation Details

### Custom Emitter Interface
The `DwarfEmitter` interface likely requires implementing methods like:
- `emitAbbrev()` / `emitAbbrevs()`
- `emitCompileUnitHeader()`
- `emitDIE()` / `emitDIEAttributes()`
- `emitDebugLine()` / `emitLineTablePrologue()`
- `emitDebugAddr()`
- `emitDebugRngLists()`
- `emitDebugStr()`
- `emitDebugStrOffsets()`

Each method receives data from DWARFLinker and writes it to our buffers.

### Address Map Format
DWARFLinker likely expects an address map interface. Check if it's:
- Per-object map: `(object_id, orig_addr) -> new_addr`
- Or global map: `orig_addr -> new_addr` (if objects don't overlap)

### Input Format
DWARFLinker likely needs:
- `DWARFContext` for each input object
- Address map per object
- Options (e.g., error handling, verbosity)

## Success Criteria

1. **Zero verification errors**: `llvm-dwarfdump --verify` reports no errors
2. **Functional parity**: All existing functionality works (rocgdb, llvm-symbolizer)
3. **Code simplification**: Removed ~500+ lines of manual patching code
4. **Maintainability**: Future DWARF changes handled by LLVM, not manual code

## Risks and Mitigations

### Risk 1: DWARFLinker API changes or incompatibilities
- **Mitigation**: Start with research phase, verify API availability
- **Fallback**: Keep old code path behind flag initially

### Risk 2: Performance impact
- **Mitigation**: Benchmark before/after, optimize if needed
- **Note**: DWARFLinker is optimized C++ code, likely faster than manual patching

### Risk 3: Missing features in DWARFLinker
- **Mitigation**: Check LLVM source/docs for feature completeness
- **Fallback**: Hybrid approach - use DWARFLinker for most, manual for edge cases

### Risk 4: ROCm LLVM version doesn't have DWARFLinker
- **Mitigation**: Check in Phase 1.1
- **Fallback**: Consider upgrading LLVM or using different approach

## Timeline Estimate
- **Phase 1**: 1-2 days (research)
- **Phase 2**: 2-3 days (emitter implementation)
- **Phase 3**: 3-4 days (integration)
- **Phase 4**: 1-2 days (address mapping)
- **Phase 5**: 2-3 days (testing)
- **Phase 6**: 1 day (cleanup)

**Total**: ~10-15 days

## Status: DWARFLinker Available! ✅

**Finding**: DWARFLinker IS available in `/COD/LATEST/aomp/llvm/include/llvm/DWARFLinker/`. The API structure is:

- **DWARFLinker**: `llvm::dwarf_linker::classic::DWARFLinker` in `DWARFLinker/Classic/DWARFLinker.h`
- **DwarfEmitter**: Abstract base class with methods like `emitSectionContents()`, `emitAbbrevs()`, `emitStrings()`, `emitCompileUnitHeader()`, `emitDIE()`, etc.
- **AddressesMap**: Abstract interface for address mapping (`getSubprogramRelocAdjustment()`, `getExprOpAddressRelocAdjustment()`, etc.)
- **DWARFFile**: Container for `DWARFContext` + `AddressesMap` per input object

**API Usage Pattern**:
```cpp
DWARFLinker linker(ErrorHandler, WarningHandler, StringsTranslator);
linker.addObjectFile(dwarfFile1, Loader, OnCUDieLoaded);
linker.addObjectFile(dwarfFile2, Loader, OnCUDieLoaded);
// ... add all inputs
linker.link();  // Produces output via DwarfEmitter
```

## Alternatives

### Option A: Use LLVM's Lower-Level Emission APIs
Instead of DWARFLinker, use LLVM's emission APIs directly:
- `MCDwarf` for line table emission
- Manual DIE cloning with `DWARFDie` and attribute patching
- Build range lists using LLVM's range list builder APIs
- **Pros**: Available in current LLVM, more control
- **Cons**: More manual work, need to handle merging ourselves

### Option B: Build DWARFLinker from LLVM Source
- Checkout LLVM source that has DWARFLinker
- Build just the DWARFLinker library
- Link against it
- **Pros**: Full DWARFLinker functionality
- **Cons**: Additional build complexity, version management

### Option C: Fix Current Approach with Better LLVM Usage
- Use LLVM's `DWARFAddressRangesVector` to get correct ranges
- Use LLVM to compute correct CU low_pc/high_pc from ranges
- Use LLVM's range list parsing to rebuild them correctly
- **Pros**: No new dependencies, incremental improvement
- **Cons**: Still some manual work, but less error-prone

### Option D: Wait for ROCm LLVM Upgrade
- ROCm may upgrade to LLVM version with DWARFLinker
- Track ROCm release notes
- **Pros**: Official support
- **Cons**: Timeline uncertain

## Recommended Approach: Option C (Hybrid)

Given that:
1. LLVM reading APIs already solved many problems
2. DWARFLinker isn't available
3. We need a solution now

**Recommended**: Enhance current LLVM-based approach:
1. Use LLVM's `DWARFAddressRangesVector` to get all ranges for each DIE
2. Use LLVM to compute correct parent-child containment
3. Rebuild range lists using LLVM's understanding of the structure
4. Let LLVM validate ranges as we build them

This gives us most of DWARFLinker's benefits without requiring it.

## Next Steps - Ready to Implement!

1. **Phase 1.1**: Verify DWARFLinker headers are accessible in build system ✅
   - **Found**: Headers at `/COD/LATEST/aomp/llvm/include/llvm/DWARFLinker/`
   - **Found**: Libraries: `libLLVMDWARFLinker.a` and `libLLVMDWARFLinkerClassic.a`
   - **Status**: ✅ Already configured! 
     - `CMakeLists.txt` line 1305: `set(AOMP_LLVM_DIR "/COD/LATEST/aomp/llvm")`
     - Line 1309: `target_include_directories(device_linker_tool PRIVATE ${AOMP_LLVM_DIR}/include)`
     - Line 1318: `target_link_libraries(device_linker_tool PRIVATE LLVM)`
   - **Action**: May need to explicitly link DWARFLinker libraries if `LLVM` target doesn't include them:
     ```cmake
     # In CMakeLists.txt around line 1318, add explicit libraries if needed:
     target_link_libraries(device_linker_tool PRIVATE 
         LLVM
         ${AOMP_LLVM_DIR}/lib/libLLVMDWARFLinkerClassic.a
         # LLVM target might already include these, test first
     )
     ```

2. **Phase 2**: Implement `DeviceLinkerDwarfEmitter` class
   - Start with `emitSectionContents()` - simplest method
   - Then implement `emitAbbrevs()`, `emitStrings()`, etc.
   - Test with minimal case first

3. **Phase 4**: Implement `DeviceLinkerAddressesMap` class
   - Implement `getSubprogramRelocAdjustment()` first
   - Test address mapping logic

4. **Phase 3**: Integrate with device linker
   - Replace `mergeDebugInfo()` call with `mergeDebugInfoWithDWARFLinker()`
   - Keep old code as fallback behind flag initially

## Implementation Priority
Start with Phase 2.1 (emitter) + Phase 4.1 (address map) as these are the core interfaces needed. Then Phase 3.1 (integration) can wire them together.

## Quick Start Checklist

### Immediate Next Steps:
1. ✅ **Headers available**: `/COD/LATEST/aomp/llvm/include/llvm/DWARFLinker/`
2. ✅ **Libraries available**: `libLLVMDWARFLinkerClassic.a` exists
3. ✅ **Build system ready**: Already using AOMP_LLVM_DIR
4. ⏭️ **Test compilation**: Try including a DWARFLinker header to verify it compiles
5. ⏭️ **Implement emitter**: Start with `DeviceLinkerDwarfEmitter` class
6. ⏭️ **Implement address map**: Start with `DeviceLinkerAddressesMap` class
7. ⏭️ **Wire together**: Create `mergeDebugInfoWithDWARFLinker()` function

### First Test:
```cpp
// In device_linker.cpp, add test include:
#include "llvm/DWARFLinker/Classic/DWARFLinker.h"

// If this compiles, we're ready to implement!
```
