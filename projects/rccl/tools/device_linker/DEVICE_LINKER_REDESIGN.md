# Device Linker: Proper Design

## Goal

Merge dispatcher device code + specialized kernel device code into a single ELF that the HIP runtime can load and execute.

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

Define the merged ELF layout with **NO OVERLAP**:

```
Address   Section        Contents
────────────────────────────────────────────────────────
0x000     [ELF header]   64 bytes
0x040     [Program hdrs] 3 × 56 = 168 bytes  
0x0E8     .note          AMDGPU metadata (variable size)
0xXXX     .dynsym        Dynamic symbols
0xXXX     .gnu.hash      Hash table
0xXXX     .hash          Hash table  
0xXXX     .dynstr        Dynamic strings
0xXXX     .rodata        Kernel descriptors (3 × 64 = 192 bytes)
0xXXX     .text          Dispatcher code + specialized kernels
0xXXX     .data          Function tables + scratch variables
0xXXX     .rela.dyn      Relocations for function pointers
0xXXX     .dynamic       Dynamic section
[non-alloc sections: .symtab, .strtab, .shstrtab]
[section headers]
```

**Key rule: Calculate ALL sizes first, then assign addresses. No overlaps.**

## Phase 3: Build Section Contents

### 3.1 .text section

```
Offset 0:           Dispatcher .text (verbatim copy)
Offset disp_size:   Specialized kernel 1
Offset next:        Specialized kernel 2
...
```

Record: `specialized_func_addr[funcId] = text_base + offset`

### 3.2 .data section

```
Offset 0:                    ncclDevFuncTable_1[FUNC_COUNT]  (FUNC_COUNT × 8 bytes)
Offset FUNC_COUNT*8:         ncclDevFuncTable_2[FUNC_COUNT]  
Offset FUNC_COUNT*8*2:       ncclDevFuncTable_4[FUNC_COUNT]
Offset FUNC_COUNT*8*3:       ncclDeviceScratchTrigger (4 bytes, initialized to 0)
Offset FUNC_COUNT*8*3+4:     ncclDeviceScratchSink (4 bytes)
```

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
