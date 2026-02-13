# Device Linker: Proper Design

## AI Assistant Rules (MUST FOLLOW)

1. **Document every mistake** - When a mistake is made, immediately add it to the "Common Build Mistakes" section below.

2. **Re-read this document first** - When encountering ANY problem, re-read this design document BEFORE attempting a fix. The solution may already be documented.

3. **Review code changes before compiling** - After making a code change, STOP and review the change with the user before running the build. Do not chain multiple changes together without review.

4. **Two-strike rule** - If a fix-compile cycle fails twice, STOP and check with the user. Do not continue attempting fixes independently.

5. **Do not assume** - If something worked before but isn't working now, investigate what changed rather than guessing at solutions.

---

## Status

**Single-GPU: WORKING** - Single GPU tests pass.

**Sequential Multi-GPU: WORKING** - Each GPU works when initialized separately.

**Parallel Multi-GPU: KERNEL HANG** - Multi-GPU AllReduce hangs indefinitely (kernel never completes). The old COMGR errors are resolved; the issue is now a kernel execution hang during `hipStreamSynchronize()`.

**Specialized Kernel Symbols: WORKING** - 2493 `ncclDevFunc_*` symbols now appear in `.symtab` with correct addresses (pointing to actual function entry, not code block start).

**Debug Line Tables: WORKING** - Full DWARF5 support including `.debug_line_str` string offset patching. Debuggers now show correct file names and line numbers.

### Summary

- `ONLY_FUNCS` works with device linker (24 specialized kernels generated)
- ELF structure matches IFC exactly (9 segments, section ordering, relocations)
- `.note` metadata patched to set `uses_dynamic_stack: false`
- ABS symbols patched to set `has_dyn_sized_stack` and `has_recursion` to 0
- `.AMDGPU.gpr_maximums` section added to match IFC
- **Symbol section indices now correct**: `ncclDevFuncTable_*` symbols point to `.data.rel.ro`
- **Symbol addresses remapped**: Function table symbols moved from `.bss` to `.data.rel.ro`
- **Specialized kernel symbols preserved**: All `ncclDevFunc_*` symbols added to `.symtab` for debugging
- **GPU auto-detection with features**: Detects full target (e.g., `gfx950:sramecc+:xnack-`) from rocminfo

### Critical Design Constraint

**Symbol section indices must match the section containing the symbol's address.**

When merging ELFs, symbols must be remapped:
1. Update `st_value` to the new address in the merged ELF
2. Update `st_shndx` to the section index containing that address
3. Special handling for function tables: they move from `.bss` (dispatcher) to `.data.rel.ro` (merged)

The device linker now correctly maps:
- `ncclDevFuncTable_1` → `.data.rel.ro` @ 0x2fa000
- `ncclDevFuncTable_2` → `.data.rel.ro` @ 0x2fbae0  
- `ncclDevFuncTable_4` → `.data.rel.ro` @ 0x2fd5c0
- `__hip_cuid_*` → `.bss`
- Kernel descriptors → `.rodata`
- Kernel code → `.text`

### Test Results

| Test | Result |
|------|--------|
| Single GPU (device 0) | PASS |
| Single GPU (device 1) | PASS |
| Sequential (dev 0 then dev 1, separate comms) | PASS |
| Parallel (ncclCommInitAll or threads) | FAIL |
| IFC build parallel | PASS |

### Root Cause Analysis (Current Issue: Kernel Hang)

The COMGR concurrent loading issue is **RESOLVED**. The current problem is a **kernel execution hang** during multi-GPU AllReduce.

**Symptoms:**
- Single-GPU AllReduce works correctly
- Multi-GPU AllReduce hangs indefinitely (kernel never completes)
- Initialization succeeds, channels are connected, kernel is launched
- Hang occurs during `hipStreamSynchronize()`

**Likely Root Cause:**
The kernel hangs because connection pointers accessed from `ncclShmem.groups[].recvConns[]` 
and `sendConns[]` are NULL or invalid. These pointers are populated in the Primitives 
constructor from `channel->peers[peer]`, which comes from `ncclShmem.channel.peers`.

**Key code path (prims_simple.h):**
```cpp
// In Primitives constructor:
if (flags & (RoleWaitRecv|RolePostRecv)) loadRecvConn(channel->peers[peer], ...);
if (flags & (RoleWaitSend|RolePostSend)) loadSendConn(channel->peers[peer], ...);
```

The `peers` array is copied from host memory during kernel initialization, but may not 
contain valid pointers in the device linker build.

### Systematic ELF Verification (Completed)

| Check | Status |
|-------|--------|
| ELF header (e_type, e_machine, e_flags) | MATCH |
| Program headers (9 segments) | MATCH |
| Section ordering | MATCH |
| Symbol section indices (st_shndx) | FIXED - now correct |
| Function table symbols → .data.rel.ro | FIXED |
| .hash nchain = .dynsym count | MATCH |
| .dynamic entries | MATCH |
| .rela.dyn relocations | CORRECT (1662 R_AMDGPU_RELATIVE64) |
| Kernel metadata structure | VALID (different from IFC but correct) |
| Note section msgpack | VALID |

### Compilation Flags (Critical for Structure Layout)

All compilation units MUST have these flags for consistent structure layouts:

| Flag | Purpose | Affects Layout |
|------|---------|----------------|
| `DEVICE_LINKER` | Function table dispatch | No |
| `ENABLE_FAULT_INJECTION` | Adds faults field to ncclShmemData | Yes |
| `ENABLE_WARP_SPEED` | Adds warpComm/warpChannel fields | Yes |
| `ENABLE_LL128` | Protocol selection | Possibly |

**Current Flag Status by Unit:**

| Unit | DEVICE_LINKER | ENABLE_FAULT_INJECTION | ENABLE_WARP_SPEED | ENABLE_LL128 |
|------|---------------|------------------------|-------------------|--------------|
| Host code (librccl.so) | ✓ | ✓ | ✓ | ✓ |
| Specialized kernels | ✓ | ✓ | ✓ | ✓ |
| Dispatcher (CMake) | ✓ | ✓ | ✓ | ✓ |
| Dispatcher (script) | ✓ | ✓ | ✓ | ✓ |

### Known Differences from IFC (Acceptable)

1. **Function table symbols**: GLOBAL PROTECTED in .dynsym (IFC: LOCAL HIDDEN in .symtab only)
2. **Function table size**: 6872 bytes (IFC: 104 bytes) - expected due to more functions
3. **Kernel args**: 4096-byte struct by_value (IFC: global_buffer pointers)
4. **.rodata flags**: 0x2 (IFC: 0x32 with MERGE|STRINGS) - not critical
5. **.comment section**: Missing (IFC has it) - not critical

### Next Steps

1. Compare ncclShmemGroup initialization between system RCCL and device linker build
2. Verify connection pointer setup in host code
3. Check if there's a code path difference for multi-GPU that device linker misses
4. Use `rocgdb` to inspect connection pointers at kernel hang point

## Goal

Merge dispatcher device code + specialized kernel device code into a single ELF that the HIP runtime can load and execute.

---

## Architecture Overview

### The 6 Generic Kernels

Production RCCL has 6 dispatcher kernels:

**Regular (3):**
1. `ncclDevKernel_Generic_1<ncclDevKernelArgsStorage<4096>>`
2. `ncclDevKernel_Generic_2<ncclDevKernelArgsStorage<4096>>`
3. `ncclDevKernel_Generic_4<ncclDevKernelArgsStorage<4096>>`

**Debug (3):**
4. `ncclDevKernelDebug_Generic_1<ncclDevKernelArgsStorage<4096>>`
5. `ncclDevKernelDebug_Generic_2<ncclDevKernelArgsStorage<4096>>`
6. `ncclDevKernelDebug_Generic_4<ncclDevKernelArgsStorage<4096>>`

These are defined in `src/device/common.cu`. They call `ncclKernelMain` which:
1. Calls `loadWorkBatchToShmem()` to load work from kernel args into shared memory
2. Sets `ncclShmem.funcId` from the work batch
3. Dispatches to the appropriate device function

### Dispatch Mechanism

With `USE_INDIRECT_FUNCTION_CALL` defined, dispatch goes through function pointer tables:
```cpp
if (COLL_UNROLL == 1)
    ncclDevFuncTable_1[ncclShmem.funcId]();
else if (COLL_UNROLL == 2)
    ncclDevFuncTable_2[ncclShmem.funcId]();
else
    ncclDevFuncTable_4[ncclShmem.funcId]();
```

Without it, dispatch uses `NCCL_CALL_FUNCTIONS_*` macros (switch/binary search).

### Specialized Device Functions

There are ~2500 specialized device functions (`ncclDevFunc_*`) generated by `generate_specialized.py`.
Each function implements a specific collective operation (e.g., `ncclDevFunc_AllReduce_RING_LL_Sum_f32_0_0_2`).

---

## Key Technical Constraints

### No `-fgpu-rdc`

The device linker approach requires `-fno-gpu-rdc` (no relocatable device code).
Each compilation unit's device code is self-contained.
This means `extern __device__` symbols cannot be resolved across compilation units at compile time.

### Shared Memory (LDS) Handling

**The Problem:**
- Generic kernels declare `__shared__ ncclShmemData ncclShmem`
- Specialized device functions reference `ncclShmem`
- In non-RDC mode, each compilation unit has its own symbols

**The Solution:**
- LDS is allocated by the kernel at launch time
- All LDS usage starts at offset 0
- Specialized functions have no other LDS declarations - they just access ncclShmem at offset 0
- Generic kernels declare ncclShmem the same way
- When device linker merges the ELFs, all functions access the same LDS layout

### Function Pointer Tables

- Function tables are in `.data.rel.ro` (read-only after relocation)
- Tables are declared/defined in the dispatcher compilation unit
- Device linker populates them with `R_AMDGPU_RELATIVE64` relocations
- Runtime fills in actual addresses when loading the code object

---

## HIP Compilation Pipeline

Running `hipcc -###` reveals 4 steps:

### Step 1: Device Compilation
```bash
clang-22 -cc1 -triple amdgcn-amd-amdhsa \
    -fcuda-is-device -target-cpu gfx942 \
    -emit-obj -o /tmp/common-gfx942-XXXXXX.o \
    -x hip common.cu.cpp
```
Compiles HIP source to device object (relocatable).

### Step 2: Device Linking  
```bash
lld -flavor gnu -m elf64_amdgpu \
    --no-undefined -shared \
    -plugin-opt=-amdgpu-internalize-symbols \
    -plugin-opt=mcpu=gfx942 \
    --whole-archive -o /tmp/common-gfx942-YYYYYY.out \
    /tmp/common-gfx942-XXXXXX.o --no-whole-archive
```
Links device object into shared ELF (`ET_DYN`).

### Step 3: Bundle Creation
```bash
clang-offload-bundler -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx942 \
    -input=/dev/null \
    -input=/tmp/common-gfx942-YYYYYY.out \
    -output=/tmp/common-ZZZZZZ.hipfb
```
Bundles null host + device ELF into fatbin.

### Step 4: Host Compilation
```bash
clang-22 -cc1 -triple x86_64-unknown-linux-gnu \
    -fcuda-include-gpubinary /tmp/common-ZZZZZZ.hipfb \
    -emit-obj -o skeleton_dispatcher.o \
    -x hip common.cu.cpp
```
Compiles HIP source for host, embeds fatbin into `.hip_fatbin` section.

## Input

### 1. Dispatcher ELF
Compiled from `common.cu`, contains:
- 3 generic kernels: `ncclDevKernel_Generic_{1,2,4}`
- Function table declarations: `ncclDevFuncTable_{1,2,4}` (in .bss, uninitialized)
- Scratch variables: `ncclDeviceScratchTrigger`, `ncclDeviceScratchSink` (in .bss)
- Helper functions: `ncclDeviceScratchRecurse`, `ncclDeviceScratchReserve`

### 2. Specialized kernel ELFs
~2500 files, each contains:
- One `ncclDevFunc_*` function
- Kernel metadata in `.note`

### 3. Host table
`host_table.cpp` mapping function names → funcId

## Output

Single ET_DYN ELF with:
- All code merged into `.text`
- Function tables populated in `.data`
- Kernel descriptors in `.rodata`
- Metadata in `.note`
- Proper relocations in `.rela.dyn`

---

## Phase 1: Parse All Inputs

```
For dispatcher:
  - Record section addresses and sizes (.text, .rodata, .bss, etc.)
  - Extract symbol table (name → address, section, size)
  - Identify PC-relative references in .text (s_getpc + s_add patterns)
  
For each specialized kernel:
  - Extract device ELF from .hip_fatbin
  - Parse .text to get code bytes
  - Parse .note to get resource requirements (VGPR, SGPR, LDS, stack)
  - Get function name from symbol table
  - Map name → funcId using host_table
```

## Phase 2: Plan Layout

Define the merged ELF layout to **match IFC exactly**:

```
Address   Section        Contents
────────────────────────────────────────────────────────
0x000     [ELF header]   64 bytes
0x040     [Program hdrs] 9 × 56 = 504 bytes  
0x238     .note          AMDGPU metadata (variable size, 8-byte padded)
0xXXX     .dynsym        Dynamic symbols
0xXXX     .gnu.hash      Hash table
0xXXX     .hash          Hash table  
0xXXX     .dynstr        Dynamic strings
0xXXX     .rela.dyn      Relocations for function pointers
0xXXX     .rodata        Kernel descriptors (3 × 64 = 192 bytes)
0xXXX     .text          Dispatcher code + specialized kernels (page-aligned)
0xXXX     .data.rel.ro   Function tables (zeroed, filled by relocations)
0xXXX     .dynamic       Dynamic section
0xXXX     .relro_padding NOBITS section for page alignment
0xXXX     .data          Scratch variables only
0xXXX     .bss           Uninitialized data (NOBITS)
[non-alloc sections: .symtab, .strtab, .shstrtab - file offset only, no vaddr]
[section headers]
```

**Key rules:**
1. Calculate ALL sizes first, then assign addresses. No overlaps.
2. `.note` section must be padded to 8-byte boundaries for HSA loader compatibility.
3. `.text` section must be page-aligned (0x1000).
4. `.data.rel.ro` starts at page boundary for RELRO segment.

### Program Headers (Segments) - Match IFC

Create 9 program headers to match IFC's working structure:

```
#   Type       Flags   Covers
──────────────────────────────────────────────────────────────
1   PHDR       R       Program header table itself (9 × 56 bytes)
2   LOAD       R       Read-only: .note, .dynsym, .gnu.hash, .hash, .dynstr, .rela.dyn, .rodata
3   LOAD       R E     Executable: .text
4   LOAD       RW      Relocatable read-only: .data.rel.ro, .dynamic, .relro_padding
5   LOAD       RW      True writable: .data, .bss
6   DYNAMIC    RW      Points to .dynamic section
7   GNU_RELRO  R       Marks .data.rel.ro + .dynamic as read-only after relocations
8   GNU_STACK  RW      Stack permissions (size 0)
9   NOTE       R       Points to .note section
```

**Critical layout rules:**
- Segment 4 (LOAD RW for RELRO) must contain `.data.rel.ro`, `.dynamic`, and `.relro_padding`
- Segment 5 (LOAD RW for data) must contain `.data` and `.bss` only
- GNU_RELRO segment covers same range as segment 4
- Non-allocated sections (.symtab, .strtab, .shstrtab) must NOT be in any LOAD segment

## Phase 3: Build Section Contents

### 3.1 .text section

```
Offset 0:           Dispatcher .text (verbatim copy)
Offset disp_size:   Specialized kernel 1
Offset next:        Specialized kernel 2
...
```

Record: `specialized_func_addr[funcId] = text_base + offset`

### 3.2 .data.rel.ro section (function tables - read-only after relocation)

```
Offset 0:                    ncclDevFuncTable_1[FUNC_COUNT]  (FUNC_COUNT × 8 bytes, zeroed)
Offset FUNC_COUNT*8:         ncclDevFuncTable_2[FUNC_COUNT]  (zeroed)
Offset FUNC_COUNT*8*2:       ncclDevFuncTable_4[FUNC_COUNT]  (zeroed)
```

Data is zeroed; R_AMDGPU_RELATIVE64 relocations fill values at load time.
After relocation processing, GNU_RELRO marks this section read-only.

### 3.2b .data section (true writable scratch variables)

```
Offset 0:     ncclDeviceScratchTrigger (4 bytes, initialized to 0)
Offset 4:     ncclDeviceScratchSink (4 bytes)
```

### 3.2c .bss section (uninitialized data)

NOBITS section for any additional uninitialized variables from dispatcher.

### 3.3 .rodata section

```
Copy dispatcher's .rodata (3 kernel descriptors, 64 bytes each)
Update kernel_code_entry_byte_offset for each KD
```

### 3.4 .note section

```
Copy dispatcher's .note
Update private_segment_fixed_size to max(all kernels)
Update other resource fields as needed
```

### 3.5 Symbol tables (.dynsym, .symtab)

```
For each symbol from dispatcher:
  - Update st_value: old_addr + (new_section_base - old_section_base)
  - Update st_shndx: map old section index → new section index
```

### 3.6 .rela.dyn section

```
For each non-zero function table entry:
  Create R_AMDGPU_RELATIVE64 relocation:
    - r_offset = address of table entry
    - r_info = 0x0D (R_AMDGPU_RELATIVE64)
    - r_addend = function address (ELF virtual address)
```

## Phase 4: Patch References

### 4.1 PC-relative references in .text

The dispatcher uses patterns like:
```asm
s_getpc_b64 s[0:1]           ; s[0:1] = PC of next instruction
s_add_u32 s0, s0, <imm32>    ; s0 += immediate
```

For each such pattern:
1. Calculate: `old_target = old_pc + old_immediate`
2. **Look up the symbol at old_target** (don't guess based on address ranges!)
3. Calculate: `new_target = new address of that symbol`
4. Calculate: `new_immediate = new_target - new_pc`
5. Patch the immediate

**Key insight:** Build a symbol lookup table. For any address, find which symbol contains it.

### 4.2 Kernel descriptor entry offsets

For each KD at `.rodata` offset `kd_off`:
```
old_entry_offset = read KD[0x10:0x18]
old_entry_abs = old_rodata_addr + kd_off + old_entry_offset
text_offset = old_entry_abs - old_text_addr
new_entry_abs = new_text_addr + text_offset  
new_entry_offset = new_entry_abs - (new_rodata_addr + kd_off)
write new_entry_offset to KD[0x10:0x18]
```

## Phase 5: Write Output

Write sections in layout order:
1. ELF header
2. Program headers
3. Allocated sections in address order
4. Non-allocated sections
5. Section headers

Ensure:
- All section sh_addr matches actual content location
- All section sh_offset matches file position
- Program headers cover all LOAD segments correctly
- .dynamic section has correct DT_* entries

---

## Data Structures

### Symbol Lookup Table

```cpp
struct SymbolInfo {
    std::string name;
    uint64_t addr;
    uint64_t size;
    uint16_t section_idx;
    enum Section { TEXT, RODATA, DATA, BSS } section_type;
};

// Map: address range → symbol
std::map<uint64_t, SymbolInfo> symbol_by_addr;

// Lookup: given an address, find the symbol that contains it
SymbolInfo* findSymbol(uint64_t addr) {
    auto it = symbol_by_addr.upper_bound(addr);
    if (it == symbol_by_addr.begin()) return nullptr;
    --it;
    if (addr < it->second.addr + it->second.size)
        return &it->second;
    return nullptr;
}
```

### Section Mapping

```cpp
struct SectionMapping {
    uint16_t old_index;
    uint16_t new_index;
    uint64_t old_addr;
    uint64_t new_addr;
    uint64_t size;
};

std::vector<SectionMapping> section_map;
```

---

## Verification Checklist

Before testing, verify with llvm-readelf:

1. **No overlapping sections** - each address range used by exactly one section
2. **Symbols point to correct sections** - st_shndx matches where st_value falls
3. **Relocations are valid** - r_offset within .data, r_addend within .text
4. **KD entry offsets work** - entry_offset + kd_addr lands in .text
5. **PC-relative patches work** - PC + immediate lands on intended symbol
6. **Compare with production** - same section types, flags, similar structure

### Verification Commands

```bash
# Check for section overlaps
llvm-readelf -S merged.elf | sort -k4

# Check symbol section indices
llvm-readelf --symbols merged.elf | grep -E "Table|Scratch"

# Check relocations
llvm-readelf -r merged.elf

# Disassemble and check PC-relative refs
llvm-objdump -d merged.elf | grep -A2 s_getpc

# Compare with production
llvm-readelf -S production.elf
llvm-readelf -S merged.elf
```

---

## Implementation Order

1. **Write `parseDispatcher()`** - extract all section info, symbols, PC-relative refs
2. **Write `parseSpecializedKernels()`** - extract code, metadata, funcId mapping
3. **Write `planLayout()`** - calculate all addresses/sizes upfront, no overlaps
4. **Write `buildSections()`** - create section content with correct data
5. **Write `patchReferences()`** - fix up all cross-references using symbol lookup
6. **Write `writeElf()`** - output the file with correct headers
7. **Test with single-GPU first**
8. **Test with multi-GPU**

---

## Lessons Learned

1. **Don't guess** - look up symbols to determine what an address refers to
2. **No piecemeal patching** - calculate everything upfront, then write
3. **Section indices matter** - symbols must have correct st_shndx
4. **Compare with production** - byte-level verification catches bugs early
5. **Understand ELF structure** - headers, program headers, section headers all interrelated
6. **Non-alloc section offsets must be recalculated** - if you modify `.symtab`/`.strtab` after layout, recalculate file offsets before writing
7. **ELF local/global symbol ordering** - local symbols must precede globals; `sh_info` marks the boundary
8. **Compilation flag consistency is critical** - structure layout changes (ENABLE_WARP_SPEED, ENABLE_FAULT_INJECTION) must be identical across host, dispatcher, and specialized kernels
9. **Shared memory declarations** - `__shared__` vs `extern __shared__` affects LDS usage; dynamic allocation requires `extern __shared__` with incomplete array size
10. **DWARF5 string tables** - when merging `.debug_line_str` sections, must patch `DW_FORM_line_strp` offsets in `.debug_line` prologues
11. **Specialized kernel symbol addresses** - must account for helper functions before the actual `ncclDevFunc_` entry; use `func_offset` to compute correct `st_value`

---

## Debugging Support

### Specialized Kernel Symbols

The device linker preserves all specialized kernel symbols (`ncclDevFunc_*`) in `.symtab` for debugging:

- **Type**: `STT_FUNC` (function)
- **Binding**: `STB_GLOBAL` (must be global because ELF requires local symbols before globals)
- **Visibility**: `STV_HIDDEN` (won't pollute dynamic symbol table)
- **Section**: `.text`

This allows debuggers (e.g., `rocgdb`) to:
- Show function names in stack traces
- Set breakpoints on specialized kernels
- Provide source-level debugging of device code

Verify symbols with:
```bash
llvm-readelf --symbols merged_device.elf | grep ncclDevFunc_
```

### GPU Auto-Detection

The device linker auto-detects the GPU target including feature flags:
- Extracts full target from `rocminfo` (e.g., `gfx950:sramecc+:xnack-`)
- Falls back to base arch (e.g., `gfx950`) if features not found
- Matches what `clang-offload-bundler` expects for unbundling

Override with `--target <arch>` if needed.

### Debug Line Tables Support

The device linker supports merging DWARF5 debug info for line-level debugging:

**To enable:**
1. Add `-gline-tables-only` to specialized kernel compilation flags in CMake
2. Rebuild specialized kernels
3. Run device linker - it will automatically:
   - Extract `.debug_line` and `.debug_line_str` from dispatcher and specialized kernels
   - Concatenate sections
   - Patch `DW_LNE_set_address` opcodes with correct code addresses
   - **Patch DWARF5 string offsets** (`DW_FORM_line_strp`) when merging `.debug_line_str` sections

**DWARF5 String Offset Fix:**
DWARF5 stores directory/file names as offsets into `.debug_line_str`. When concatenating multiple 
`.debug_line_str` sections, the device linker adjusts these offsets via `patchDwarf5StringOffsets()`:
- Parses DWARF5 line table prologues to find directory and file name entries
- Identifies entries using `DW_FORM_line_strp` form
- Adjusts each string offset by `str_offset_base` (cumulative offset from earlier kernels)
- Handles all DWARF forms including `DW_FORM_data16` (MD5 checksums)

**Symbol Address Fix:**
Specialized kernel symbols (`ncclDevFunc_*`) now point to the actual function entry, not the 
start of the extracted code block (which includes helper functions):
```cpp
sym.st_value = text_addr_ + text_off + kern->func_offset;  // Actual function entry
sym.st_size = kern->code.size() - kern->func_offset;       // Function size only
```

**What works:**
- Line-based breakpoints in `rocgdb`
- Line information in stack traces
- Source stepping through specialized kernel code
- Correct file names in debugger output

**What doesn't work (limitation of -gline-tables-only):**
- Variable inspection (no type info)
- Full debug info (use full `-g` for that, but increases binary size)

---

## Debugging Tools Reference

### ROCm LLVM Object Tools

#### KD (Kernel Descriptor) Disassembly
```bash
# Decode KDs with human-readable field names
llvm-objdump -Dr --section=.rodata <device.elf>
```

#### Raw Section Contents with Relocations
```bash
# Show hex dump with relocation annotations
llvm-objdump -s -r --section=.rodata <device.elf>

# Useful for function tables - shows where R_AMDGPU_RELATIVE64 will be applied
llvm-objdump -s -r --section=.data.rel.ro <device.elf>
```

#### Offloading Bundle Extraction
```bash
# Automatically extract embedded device code from host .so
llvm-objdump --offloading <librccl.so>

# Then analyze the extracted device ELF:
llvm-objdump -Dr --section=.rodata librccl.so.0.hipv4-amdgcn-amd-amdhsa--gfx942*
```

### Key KD Fields (64-byte descriptor at each .kd symbol)

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0x00 | 4 | group_segment_fixed_size | LDS allocation in bytes |
| 0x04 | 4 | private_segment_fixed_size | Scratch size (should match max callee) |
| 0x08 | 8 | kernarg_size | Kernel argument buffer size |
| 0x10 | 8 | kernel_code_entry_byte_offset | Signed offset from KD to entry point |
| 0x2c | 4 | reserved (accum_offset encoding) | Affects AGPR allocation |
| 0x30 | 4 | compute_pgm_rsrc1 | VGPR/SGPR allocation |
| 0x34 | 4 | compute_pgm_rsrc2 | SCRATCH_EN (bit 0), USER_SGPR (bits 1-5) |
| 0x38 | 2 | kernel_code_properties | SGPR enables (dispatch_ptr, queue_ptr, etc.) |

### Comparing Builds
```bash
# Extract IFC device ELF
llvm-objcopy --dump-section=.hip_fatbin=/tmp/ifc_fatbin.bin /path/to/ifc/librccl.so
clang-offload-bundler --unbundle -type=o -targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
    -input=/tmp/ifc_fatbin.bin -output=/tmp/ifc_device.elf

# Compare KDs side-by-side
llvm-objdump -Dr --section=.rodata /tmp/ifc_device.elf | grep -A40 "Generic_2.*\.kd"
llvm-objdump -Dr --section=.rodata merged_device.elf | grep -A40 "Generic_2.*\.kd"
```

---

## Common Build Mistakes (DO NOT REPEAT)

### CMake Configuration

| Mistake | Correct |
|---------|---------|
| `-DDEVICE_LINKER_ENABLED=ON` | `-DDEVICE_LINKER=ON` (use DEVICE_LINKER, not DEVICE_LINKER_ENABLED) |
| Using `SPECIALIZED_KERNELS_ONLY` | OBSOLETE - use `DEVICE_LINKER` instead |
| Missing `USE_INDIRECT_FUNCTION_CALL` | Must be defined for function pointer dispatch |
| `ONLY_FUNCS` cached from previous build | Delete `CMakeCache.txt` when changing config |
| Using `ONLY_FUNCS` for "faster" testing | Don't limit kernels - full build should be fast with DEVICE_LINKER |

### Device Linker Pipeline

| Mistake | Correct |
|---------|---------|
| Forgetting to rebuild `device_linker` after code changes | **Always rebuild**: `g++ -O2 -std=c++17 -o device_linker device_linker.cpp -lpthread` |
| `--offload-compress` in dispatcher compile | Remove it - causes CCOB extraction failures |
| Wrong `--target-arch` (e.g., gfx942 vs gfx950) | Auto-detected from rocminfo now; override with `--target` if needed |
| Target missing feature flags (e.g., `gfx950` vs `gfx950:sramecc+:xnack-`) | Auto-detection now includes features; bundler requires exact match |

### Link Errors| Mistake | Correct |
|---------|---------|
| `sym_kernels_stub.cpp` duplicate symbol | Remove `ncclSymkGetKernelPtr` from stub - it's in `sym_kernels.cc` |

### Device Linker Runtime

| Mistake | Symptom | Correct |
|---------|---------|---------|
| Wrong target arch | "Processing 1 input files", "Mapped 0 kernel functions" | Ensure `--target-arch` matches the .o files |
| Specialized kernels not found | 0 relocations generated | Check `--input-dir` path and that .o files exist for target arch |

### Resolved Issues

**COMGR Concurrent Loading (FIXED):** The "Cannot Find Global Var Sizes" error during concurrent 
multi-GPU loading has been resolved. The device linker ELF now loads correctly on multiple GPUs.

**Dispatcher Inclusion (FIXED):** The build system now correctly compiles the dispatcher separately 
and the device linker merges it with specialized kernels.

**Shared Memory Limit (FIXED):** `ncclShmemPerWarp` now uses `extern __shared__` (dynamic allocation) 
for CUDA arch >= 700, matching the standard build behavior. This resolved the error:
`cudaArch 940 ncclMaxSharedMem 32832 exceeds device/fn maxSharedMem 27760`

### Current Unresolved Issue

**Problem**: Multi-GPU AllReduce hangs indefinitely during kernel execution.

**Status**: Single-GPU works; multi-GPU kernel launches but never completes.

**Investigation**: Connection pointers in `ncclShmem.groups[].recvConns[]` and `sendConns[]` 
may be NULL or invalid. Use `rocgdb` to inspect LDS contents at hang point.

### Build Command Reference

```bash
# Correct full build command:
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
      -DDEVICE_LINKER_ENABLED=ON \
      -DGPU_TARGETS=gfx950 \
      -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -GNinja ..

# After reconfiguring, ALWAYS delete CMakeCache.txt first:
rm -f CMakeCache.txt# If sym_kernels_stub.cpp error, edit build/sym_kernels_stub.cpp:
# Remove the ncclSymkGetKernelPtr function definition
```
