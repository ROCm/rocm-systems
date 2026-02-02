# IFC vs Device Linker: Comprehensive ELF Comparison

**Date:** February 2026  
**Purpose:** Document all differences between IFC (Indirect Function Call / production) build and Device Linker build to identify root cause of multi-GPU failure.

## Executive Summary

| Aspect | IFC | Device Linker | Issue? |
|--------|-----|---------------|--------|
| Single GPU | ✅ PASS | ✅ PASS | No |
| Multi-GPU Concurrent | ✅ PASS | ❌ FAIL | **YES** |
| Sequential Multi-GPU | ✅ PASS | ✅ PASS | No |

**Root cause hypothesis:** The failure is specific to concurrent code object loading by COMGR/HSA runtime. Sequential loading works fine, suggesting a race condition or concurrent-specific code path that doesn't handle our ELF format correctly.

---

## 1. File Size Comparison

| File | IFC | Device Linker | Ratio |
|------|-----|---------------|-------|
| Device ELF | 116,328,024 bytes (111 MB) | 18,420,136 bytes (17.6 MB) | 6.3x smaller |
| .text section | 114,587,860 bytes | 1,731,328 bytes | 66x smaller |
| .note section | 26,780 bytes | 4,592 bytes | 5.8x smaller |

**Why smaller?** Device linker only includes unroll=2 specialized kernels (859) vs IFC's all unroll factors (2,577 kernels × 3 tables).

---

## 2. ELF Header Comparison

| Field | IFC | Device Linker | Match? |
|-------|-----|---------------|--------|
| Class | ELF64 | ELF64 | ✅ |
| OS/ABI | AMDGPU - HSA | AMDGPU - HSA | ✅ |
| ABI Version | 4 | 4 | ✅ |
| Type | DYN (Shared object) | DYN (Shared object) | ✅ |
| Machine | EM_AMDGPU | EM_AMDGPU | ✅ |
| Flags | 0x54c (gfx942, xnack, sramecc) | 0x54c (gfx942, xnack, sramecc) | ✅ |
| Program headers | 9 | 9 | ✅ |
| Section headers | 19 | 23 | ❌ (DL has debug sections) |

---

## 3. Section Comparison

### 3.1 Section Count and Types

| Section | IFC | Device Linker | Notes |
|---------|-----|---------------|-------|
| .note | ✅ | ✅ | Different sizes |
| .dynsym | ✅ | ✅ | Different sizes |
| .gnu.hash | ✅ | ✅ | |
| .hash | ✅ | ✅ | |
| .dynstr | ✅ | ✅ | |
| .rela.dyn | ✅ | ✅ | Different counts |
| .rodata | ✅ | ✅ | **Different flags** |
| .text | ✅ | ✅ | |
| .data.rel.ro | ✅ | ✅ | |
| .dynamic | ✅ | ✅ | |
| .relro_padding | ✅ | ✅ | |
| .data | ✅ | ✅ | |
| .bss | ✅ | ✅ | |
| .AMDGPU.gpr_maximums | ✅ (empty) | ✅ (empty) | |
| .comment | ✅ | ❌ Missing | Minor |
| .symtab | ✅ | ✅ | |
| .strtab | ✅ | ✅ | |
| .shstrtab | ✅ | ✅ | |
| .debug_line | ❌ | ✅ | Extra in DL |
| .debug_line_str | ❌ | ✅ | Extra in DL |
| .debug_abbrev | ❌ | ✅ | Extra in DL |
| .debug_info | ❌ | ✅ | Extra in DL |
| .debug_str | ❌ | ✅ | Extra in DL |

### 3.2 Section Size Comparison

| Section | IFC Size | DL Size | IFC Flags | DL Flags | Notes |
|---------|----------|---------|-----------|----------|-------|
| .note | 26,780 | 4,592 | A | A | IFC 5.8x larger |
| .dynsym | 6,384 | 264 | A | A | IFC 24x larger |
| .rela.dyn | 62,136 | 20,616 | A | A | IFC 3x larger |
| .rodata | 676,224 | 20,824 | **AMS** | **A** | **FLAGS DIFFER** |
| .text | 114,587,860 | 1,731,328 | AX | AX | |
| .data.rel.ro | 20,640 | 20,640 | WA | WA | ✅ Same |
| .dynamic | 176 | 176 | WA | WA | ✅ Same |
| .data | 96 | 96 | WA | WA | ✅ Same |
| .bss | 228 | 107 | WA | WA | |

**Key difference:** `.rodata` flags are `AMS` (Alloc+Merge+Strings) in IFC vs just `A` (Alloc) in Device Linker. This likely doesn't affect runtime behavior.

---

## 4. Program Headers (Segments)

Both have identical segment structure:

| # | Type | Flags | Contents |
|---|------|-------|----------|
| 0 | PHDR | R | Program header table |
| 1 | LOAD | R | .note, .dynsym, .gnu.hash, .hash, .dynstr, .rela.dyn, .rodata |
| 2 | LOAD | R E | .text |
| 3 | LOAD | RW | .data.rel.ro, .dynamic, .relro_padding |
| 4 | LOAD | RW | .data, .bss |
| 5 | DYNAMIC | RW | .dynamic |
| 6 | GNU_RELRO | R | .data.rel.ro, .dynamic, .relro_padding |
| 7 | GNU_STACK | RW | (empty) |
| 8 | NOTE | R | .note |

**Result:** ✅ Segment structure matches exactly.

---

## 5. Dynamic Section Comparison

| Entry | IFC | Device Linker | Match? |
|-------|-----|---------------|--------|
| RELA | 0xb850 | 0x17c0 | (addresses differ, OK) |
| RELASZ | 62,136 | 20,616 | (counts differ, OK) |
| RELAENT | 24 | 24 | ✅ |
| RELACOUNT | 2,589 | 859 | (counts differ, OK) |
| SYMTAB | (addr) | (addr) | ✅ |
| SYMENT | 24 | 24 | ✅ |
| STRTAB | (addr) | (addr) | ✅ |
| STRSZ | 9,454 | 473 | (sizes differ, OK) |
| GNU_HASH | (addr) | (addr) | ✅ |
| HASH | (addr) | (addr) | ✅ |
| NULL | 0 | 0 | ✅ |

**Result:** ✅ Same 11 entries with same structure. Value differences are expected due to different content sizes.

---

## 6. Symbol Table Differences

### 6.1 Dynamic Symbols (.dynsym)

| Metric | IFC | Device Linker |
|--------|-----|---------------|
| Total entries | 266 | 11 |
| `__hip_cuid_*` symbols | ~200 | 1 |
| Kernel functions | 6 (Generic + Debug) | 3 (Generic only) |
| Kernel descriptors (.kd) | 18 | 3 |
| Function table symbols | 0 (not in .dynsym) | 3 (LOCAL HIDDEN) |
| oneRankReduce functions | 12 | 0 |

**Key difference:** IFC has many more compilation units (each creates a `__hip_cuid_*`), and has oneRankReduce helper kernels.

### 6.2 Function Table Symbol Differences

| Aspect | IFC | Device Linker |
|--------|-----|---------------|
| Symbol name | `_ZL18ncclDevFuncTable_*` | `ncclDevFuncTable_*` |
| Linkage | Internal (`_ZL` prefix) | External |
| In .dynsym? | NO | YES (LOCAL HIDDEN) |
| In .symtab? | YES (LOCAL HIDDEN) | YES (duplicated!) |
| Size | 6,880 bytes (860 × 8) | 6,872 bytes (859 × 8) |

**Potential issue:** Device linker has function tables in `.dynsym` (even as LOCAL HIDDEN), while IFC keeps them only in `.symtab`. Also, DL has duplicate entries (LOCAL HIDDEN + GLOBAL PROTECTED).

### 6.3 Specialized Function Symbols

| Aspect | IFC | Device Linker |
|--------|-----|---------------|
| Count | 2,577 | 859 |
| Binding | LOCAL | GLOBAL |
| Visibility | HIDDEN | HIDDEN |
| In .dynsym? | NO | NO |
| In .symtab? | YES | YES |

**Note:** IFC has 3x more because it includes all unroll factors (1, 2, 4). Device linker build only has unroll=2.

### 6.4 Missing Symbol: `__clang_gpu_used_external`

| Aspect | IFC | Device Linker |
|--------|-----|---------------|
| Present? | YES | **NO** |
| Size | 96 bytes | - |
| Type | OBJECT | - |
| Section | .data | - |
| Relocations | 12 (pointing to kernel entries) | - |

**This symbol tracks which device functions are "used" by the compiler. Its absence could affect runtime behavior.**

---

## 7. Relocation Differences

### 7.1 Relocation Counts

| Category | IFC | Device Linker |
|----------|-----|---------------|
| Function table 1 | 859 | 0 |
| Function table 2 | 859 | 859 |
| Function table 4 | 859 | 0 |
| `__clang_gpu_used_external` | 12 | 0 |
| **Total** | **2,589** | **859** |

**Note:** Device linker only populates table_2 because the build only includes unroll=2 kernels.

### 7.2 Relocation Structure

Both use identical relocation format:
- Type: `R_AMDGPU_RELATIVE64` (0x0d)
- r_offset: Address in .data.rel.ro (function table entry)
- r_addend: Target address in .text (function entry point)

### 7.3 Missing `__clang_gpu_used_external` Relocations

IFC has 12 relocations in `.data` section that point to oneRankReduce kernel entries:

| Offset | Target Function |
|--------|-----------------|
| 0x6df7330 | oneRankReduce<int> |
| 0x6df7338 | oneRankReduce<unsigned char> |
| 0x6df7340 | oneRankReduce<int> |
| 0x6df7348 | oneRankReduce<unsigned int> |
| 0x6df7350 | oneRankReduce<long> |
| 0x6df7358 | oneRankReduce<unsigned long> |
| 0x6df7360 | oneRankReduce<fp8_e4m3> |
| 0x6df7368 | oneRankReduce<fp8_e5m2> |
| 0x6df7370 | oneRankReduce<half> |
| 0x6df7378 | oneRankReduce<bfloat16> |
| 0x6df7380 | oneRankReduce<float> |
| 0x6df7388 | oneRankReduce<double> |

Device linker is missing these entirely.

---

## 8. Kernel Descriptor Comparison

### 8.1 Generic_2 Kernel Descriptor

| Field | IFC | Device Linker | Match? |
|-------|-----|---------------|--------|
| group_segment_fixed_size (LDS) | 19,808 | 19,776 | ❌ (-32 bytes) |
| private_segment_fixed_size | 0 | 1,096 | ❌ |
| kernarg_size | 4,352 | 4,352 | ✅ |
| accum_offset | 128 | 128 | ✅ |
| next_free_vgpr | 136 | 304 | ❌ (2.2x more) |
| next_free_sgpr | 112 | 112 | ✅ |
| enable_private_segment | 1 | 1 | ✅ |
| system_sgpr_workgroup_id_x | 1 | 1 | ✅ |
| system_sgpr_workgroup_id_y | 1 | 1 | ✅ |
| system_sgpr_workgroup_id_z | 1 | 1 | ✅ |
| system_vgpr_workitem_id | 2 | 2 | ✅ |
| user_sgpr_dispatch_ptr | 1 | 1 | ✅ |
| user_sgpr_queue_ptr | 1 | 1 | ✅ |
| user_sgpr_kernarg_segment_ptr | 1 | 1 | ✅ |
| user_sgpr_dispatch_id | 1 | 1 | ✅ |
| uses_dynamic_stack | 1 | 1 | ✅ |

**Key differences:**
1. LDS size differs by 32 bytes (minor)
2. Device linker sets scratch size (1096) because specialized functions need it; IFC relies on dynamic allocation
3. Device linker uses 2.2x more VGPRs (likely due to different code generation)

### 8.2 ABS Resource Symbols

Both have ABS symbols tracking resource usage. Sample comparison:

| Symbol Suffix | IFC (Generic_2) | Device Linker (Generic_2) |
|---------------|-----------------|---------------------------|
| .private_seg_size | 0 | 0 |
| .num_vgpr | (varies by kernel) | 0x97 (151) |
| .num_agpr | 0 | 0 |
| .numbered_sgpr | (varies) | 0x6a (106) |
| .uses_vcc | 1 | 1 |
| .uses_flat_scratch | 0 | 1 |
| .has_dyn_sized_stack | 0 | 0 |
| .has_recursion | 0 | 0 |

---

## 9. .note Section (Kernel Metadata)

### 9.1 Kernel Count

| Build | Kernel Count | Kernels |
|-------|--------------|---------|
| IFC | 18 | 12 oneRankReduce + 3 Generic + 3 Debug |
| Device Linker | 3 | 3 Generic only |

### 9.2 Kernel Argument Structure

| Aspect | IFC (oneRankReduce) | IFC (Generic) | Device Linker (Generic) |
|--------|---------------------|---------------|-------------------------|
| First args | `global_buffer` (pointers) | `by_value` (4096 bytes) | `by_value` (4096 bytes) |
| kernarg_size | 296 bytes | 4,352 bytes | 4,352 bytes |

### 9.3 Resource Requirements in Metadata

| Field | IFC (oneRankReduce) | IFC (Generic) | DL (Generic) |
|-------|---------------------|---------------|--------------|
| private_segment_fixed_size | 0 | 0 | 1,096 |
| sgpr_count | 33 | 112 | 106 |
| vgpr_count | 44 | 136 | 151 |
| uses_dynamic_stack | false | true | true |

---

## 10. Key Differences Summary

### 10.1 Structural Differences (Likely NOT Causing Failure)

| Difference | Reason | Risk |
|------------|--------|------|
| File size (6x smaller) | Only unroll=2 kernels | None |
| .rodata flags (AMS vs A) | Build flag difference | Low |
| Debug sections present | Device linker adds them | None |
| Missing .comment | Not added by device linker | None |
| Fewer dynsym entries | Single compilation unit | Low |

### 10.2 Potentially Significant Differences

| Difference | IFC | Device Linker | Risk Level |
|------------|-----|---------------|------------|
| `__clang_gpu_used_external` | Present (96 bytes, 12 relocs) | **Missing** | **HIGH** |
| Function table linkage | Internal (`_ZL`) | External | **MEDIUM** |
| Function tables in .dynsym | No | Yes (LOCAL HIDDEN) | **MEDIUM** |
| Duplicate function table symbols | No | Yes | **LOW** |
| oneRankReduce kernels | 12 present | Missing | LOW (stub used) |

### 10.3 KD Differences (Likely NOT Causing Failure)

| Difference | Notes |
|------------|-------|
| LDS size (-32 bytes) | Minor, shouldn't affect functionality |
| private_segment_fixed_size (0 vs 1096) | DL pre-allocates for callees, both approaches valid |
| VGPR count (136 vs 304) | Different code paths, shouldn't cause failure |

---

## 11. Theories for Multi-GPU Failure

### Theory 1: Missing `__clang_gpu_used_external` (HIGH PRIORITY)

The `__clang_gpu_used_external` symbol is compiler-generated and tracks which device functions are "used". Its absence might:
- Cause COMGR to skip certain initialization
- Trigger a different code path during concurrent loading
- Result in missing metadata during multi-GPU setup

**Test:** Add this symbol with dummy relocations and verify.

### Theory 2: Function Table Symbol Visibility (MEDIUM PRIORITY)

IFC uses `_ZL` (internal linkage) for function tables, keeping them out of `.dynsym`. Device linker exports them as LOCAL HIDDEN in `.dynsym`.

This might:
- Trigger different symbol resolution during concurrent loading
- Cause race conditions in the dynamic linker

**Test:** Change function tables to internal linkage.

### Theory 3: COMGR Race Condition (MEDIUM PRIORITY)

The failure only happens during concurrent loading. COMGR may have:
- Thread-unsafe code when processing relocations
- Shared state that gets corrupted with our ELF format
- Assumptions about symbol order/structure that we violate

**Test:** Add serialization (mutex) around code object loading.

### Theory 4: Kernel Count in .note (LOW PRIORITY)

IFC has 18 kernels in metadata, we have 3. The runtime might:
- Expect certain helper kernels to exist
- Use metadata to initialize per-GPU state

**Test:** Add dummy kernel metadata for oneRankReduce kernels.

---

## 12. Recommended Investigation Order

1. **Add `__clang_gpu_used_external`** - Create the symbol with 12 relocations pointing to specialized kernel entries (or dispatcher entries)

2. **Change function table linkage** - Use `_ZL` prefix for internal linkage, remove from `.dynsym`

3. **Serialize code object loading** - As a workaround, add mutex around HIP kernel registration

4. **Debug COMGR** - Build with symbols, trace the "Cannot Find Global Var Sizes" error

5. **Match .note exactly** - Add oneRankReduce metadata even if kernels aren't present

---

## 13. Appendix: Verification Commands

```bash
# Extract device ELF from host library
llvm-objdump --offloading librccl.so

# Or manually:
llvm-objcopy --dump-section=.hip_fatbin=fatbin.bin librccl.so
clang-offload-bundler --unbundle -type=o \
    -targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
    -input=fatbin.bin -output=device.elf

# Compare sections
llvm-readelf -S ifc_device.elf
llvm-readelf -S dl_device.elf

# Compare symbols
llvm-readelf --symbols ifc_device.elf | grep -E "FuncTable|clang_gpu"
llvm-readelf --symbols dl_device.elf | grep -E "FuncTable|clang_gpu"

# Compare KDs
llvm-objdump -Dr --section=.rodata ifc_device.elf | grep -A40 "Generic_2.*\.kd"
llvm-objdump -Dr --section=.rodata dl_device.elf | grep -A40 "Generic_2.*\.kd"

# Compare relocations
llvm-readelf -r ifc_device.elf | head -50
llvm-readelf -r dl_device.elf | head -50

# Check dynamic section
llvm-readelf -d ifc_device.elf
llvm-readelf -d dl_device.elf
```

---

## 14. Files Referenced

| File | Description |
|------|-------------|
| `/tmp/elf_comparison/ifc_device.elf` | Extracted IFC device ELF |
| `/tmp/elf_comparison/dl_device.elf` | Device linker merged ELF |
| `/work/lmeadows/rocm-systems/projects/rccl/build-ifc-only/librccl.so.1.0` | IFC build |
| `/work/lmeadows/rocm-systems/projects/rccl/build/release/device_linker_output/merged_device.elf` | Device linker output |

---

*Generated from analysis on February 1, 2026*
