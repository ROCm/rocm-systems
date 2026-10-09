# hsaco format for AIE (NPU) kernels

This document gives the format of an HSA code object (hsaco) that holds AIE kernels for AMD XDNA/XDNA2 NPUs. It also gives the rules that the ROCR loader and the [mlir-aie](https://github.com/Xilinx/mlir-aie) packer apply to that format.

## 1. Scope and sources

The data in this document comes from the source code given below. If this document and the source code do not agree, the source code is correct.

| Component | Repository | Files |
| --- | --- | --- |
| ROCR loader | This repository | `runtime/hsa-runtime/core/inc/amd_aie_section.h`, `core/runtime/amd_aie_code.cpp`, `core/runtime/amd_aie_elf.cpp`, `loader/executable.cpp` |
| ROCR dispatch | This repository | `core/driver/xdna/amd_xdna_driver.cpp`, `inc/hsa_ext_amd_aie.h` |
| Packer and dumper | [mlir-aie](https://github.com/Xilinx/mlir-aie) | `python/compiler/hsaco/format.py`, `pack.py`, `elf.py`, `dump.py` |

The C++ header `amd_aie_section.h` is the authority for the section layout. The file `format.py` is a copy of that layout.

## 2. Terms

| Term | Meaning |
| --- | --- |
| hsaco | The ELF64 container file that the application gives to `hsa_code_object_reader_create_from_memory` or `..._from_file`. |
| AIE section | One section in the hsaco that holds the AIE kernels for one architecture. |
| arch | The AIE architecture name: `aie2` (Phoenix, npu1) or `aie2p` (Strix, Strix Halo, Krackan: npu4, npu5, npu6). |
| PDI | Programmable Device Image. The binary that configures the AIE array. |
| Instruction sequence | The `insts.bin` file from `aiecc`. A stream of 32-bit words that the NPU firmware runs. |
| Full ELF | An ELF32 file from `aiecc --get-full-elf`. It holds the PDI, the control code, and relocations in one file. |
| Control code | The instruction stream inside a full ELF, in a section with a name that starts with `.ctrltext`. |
| Patch site | A byte offset in the control code where the runtime writes an address. |
| Kernel descriptor | The host structure `AieKernelDescriptor`. The HSA kernel object handle points to it. |
| BO | Buffer object of the XDNA kernel driver. |

All multi-byte fields are little-endian. All sizes and offsets are in bytes, unless the text gives a different unit.

## 3. Overview

Two different ELF files have a part in this format. Do not confuse them:

1. The **hsaco**. This is an ELF64 file. It holds one AIE section for each arch.
2. The **full ELF**. This is an ELF32 file. A kernel of kind `FullElf` holds a full ELF as a blob inside the AIE section. The full ELF is thus an ELF32 file inside an ELF64 file.

The build pipeline is:

1. `aiecc` compiles an IRON or MLIR-AIE design. It makes one of these outputs:
   - `insts.bin` and `main.pdi` (and, as an option, `final.xclbin`).
   - A full ELF (`aiecc --get-full-elf`).
2. `aie-hsaco` puts the outputs into an AIE section in the hsaco.
3. `aie-hsaco-dump` reads the hsaco and validates each AIE section. This step is optional.
4. The application loads the hsaco on an AIE agent with `hsa_executable_load_agent_code_object`.
5. The application gets a kernel object handle with `hsa_executable_get_symbol_by_name` and `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT`.
6. The application puts the handle into an `hsa_amd_aie_kernel_dispatch_packet_t` and submits the packet to an AIE queue.

## 4. hsaco container

### 4.1 ELF header

When the output file does not exist, `aie-hsaco` makes a minimum container with these values:

| Field | Value |
| --- | --- |
| `EI_CLASS` | `ELFCLASS64` (2) |
| `EI_DATA` | `ELFDATA2LSB` (1) |
| `e_type` | `ET_REL` (1) |
| `e_machine` | `EM_AMDGPU` (224) |
| `e_shnum` | 2 |
| Sections | `[0]` `SHT_NULL`, `[1]` `.shstrtab` |

The `.shstrtab` section is necessary. `llvm-objcopy` writes the name of a new section into `.shstrtab`. If the ELF has no `.shstrtab`, `--add-section` does nothing and does not give an error.

The ROCR loader does not examine `e_type` or `e_machine`. It examines only the ELF magic. It then opens the file as an ELF64 image. Thus a container from a different tool is also satisfactory if it is a valid ELF64 file.

### 4.2 AIE section in the container

`aie-hsaco` adds one section for each arch with this command:

```bash
llvm-objcopy --remove-section=<arch> --add-section=<arch>=<file> \
             --set-section-flags=<arch>=noload,readonly <in> <out>
```

The section has these properties:

| Property | Value |
| --- | --- |
| Name | The arch name: `aie2` or `aie2p`. |
| `sh_type` | `SHT_PROGBITS` |
| `sh_flags` | `SHF_ALLOC` is clear. The loader does not map the section into a program image. |
| `sh_addralign` | 1. The section can start at an unaligned file offset. |

If a section with the same name exists, `aie-hsaco` replaces it. It does not add a second section.

One hsaco can hold an `aie2` section and an `aie2p` section. Run `aie-hsaco` one time for each arch. Do not run two `aie-hsaco` commands on one file at the same time. Each command reads the file, writes a temporary copy, and renames the copy over the original. The second rename removes the section that the first command added. No error occurs.

### 4.3 How ROCR finds the AIE section

An AIE section is a section that agrees with all of these conditions:

- `sh_size` is equal to or more than 48 (the header size).
- The section is fully in the file.
- The first 4 bytes are the magic `0x4B454941` (`"AIEK"`).

ROCR uses the AIE load path if the file has one or more AIE sections. It then examines the sections in section-header order, and selects the first AIE section whose name is equal to the arch name of the agent. ROCR does not parse the AIE sections for other arches.

If the file has no AIE section for the arch of the agent, the load fails with `HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS`. If the selected section is not correct, the load fails with `HSA_STATUS_ERROR_INVALID_CODE_OBJECT`.

Thus one hsaco can hold an `aie2` section and an `aie2p` section, in any sequence, and each agent loads its own section.

The XDNA driver backend sets the arch name of the agent (`HsaNodeProperties::AMDName`):

| Device | Arch name | Columns that the agent reports (`num_cols()`) |
| --- | --- | --- |
| Phoenix (npu1) | `aie2` | Device columns minus 1. NPU1 has only N−1 shim DMAs. |
| Strix, Strix Halo, Krackan (npu4/5/6) | `aie2p` | Device columns. |

### 4.4 GPU code in the same container

The packer can add an AIE section to an hsaco that already holds GPU code. On an AIE agent,
ROCR uses the AIE load path if it finds an AIE section. This document does not verify how the
GPU load path operates on a file that also holds an AIE section.

## 5. AIE section layout

The section has four regions in this sequence:

```text
+-------------------+  offset 0
| Section header    |  header_size bytes (48 in version 1.0)
+-------------------+  header_size
| Kernel table      |  kernel_count * kernel_entry_size bytes (44 per entry in version 1.0)
+-------------------+  string_table_offset
| String table      |  string_table_size bytes
+-------------------+  blob_pool_offset
| Blob pool         |  to the end of the section
+-------------------+  section size
```

All offsets in the header and in the kernel table are relative to the start of the section. There is no padding between regions, and there is no alignment of blobs. ROCR copies each blob into a new 64-byte-aligned device buffer at load time, so the alignment in the file is not
important.

### 5.1 Section header (`aie_section_header`, 48 bytes)

| Offset | Size | Field | Value and rules |
| --- | --- | --- | --- |
| 0 | 4 | `magic` | `0x4B454941` (`'A','I','E','K'`). ROCR defines this value. It does not come from an external format. |
| 4 | 2 | `version_major` | 1. ROCR and the dumper refuse any other value. |
| 6 | 2 | `version_minor` | 0. Readers do not examine it. |
| 8 | 4 | `header_size` | Offset of the kernel table. 48 in version 1.0. |
| 12 | 4 | `kernel_count` | Number of kernel table entries. Must be 1 or more for ROCR. |
| 16 | 4 | `kernel_entry_size` | Stride between kernel table entries. Must be 44 or more. |
| 20 | 4 | `string_table_offset` | Section offset of the string table. |
| 24 | 4 | `string_table_size` | Size of the string table. |
| 28 | 4 | `blob_pool_offset` | Section offset of the blob pool. The pool continues to the end of the section. |
| 32 | 16 | `reserved[4]` | Must be 0. No reader examines these words. |

A reader must use `header_size` and `kernel_entry_size` from the file. It must not use its own structure sizes. This rule lets an old reader read a section that has new fields at the end of the header or of the entry.

### 5.2 Kernel table entry (`aie_kernel_entry`, 44 bytes)

| Offset | Size | Field | Value and rules |
| --- | --- | --- | --- |
| 0 | 4 | `name_offset` | Offset of the kernel name, relative to `string_table_offset`. |
| 4 | 4 | `insts_offset` | Section offset of the instruction blob (`PdiInsts`) or the full-ELF blob (`FullElf`). |
| 8 | 4 | `insts_size` | Size of that blob. Must be more than 0. |
| 12 | 4 | `pdi_offset` | Section offset of the PDI blob. 0 if there is no PDI. |
| 16 | 4 | `pdi_size` | Size of the PDI blob. 0 if there is no PDI. |
| 20 | 4 | `kernarg_size` | Size of the kernel argument buffer. See section 9.1. |
| 24 | 4 | `num_cols` | Number of NPU columns of the partition that the kernel was compiled for. See section 8. |
| 28 | 4 | `kind` | Payload kind (`AieKernelKind`). See section 5.4. |
| 32 | 12 | `reserved[3]` | Must be 0. No reader examines these words. |

### 5.3 String table and blob pool

- The string table holds the kernel names. Each name is a UTF-8 string with a NUL terminator. The packer writes the names in kernel table order, with no padding.
- The HSA symbol name of a kernel is the string in the string table. ROCR does not add a suffix such as `.kd`.
- Kernel names must be unique in one section. ROCR and the packer both refuse a duplicate name.
- The blob pool holds the instruction blobs, the PDI blobs, and the full-ELF blobs. The packer stores identical blobs one time only (it compares the raw bytes). Thus, two entries can have the same `insts_offset` or `pdi_offset`. All entries from one full ELF point to one copy of that ELF.

### 5.4 Kernel kinds (`AieKernelKind`)

| Value | Name | `insts_*` locate | `pdi_*` |
| --- | --- | --- | --- |
| 0 | `PdiInsts` | The instruction sequence. | The PDI. ROCR requires a PDI. |
| 1 | `FullElf` | A complete full ELF (ELF32). | Must be 0. The PDI is inside the full ELF. |
| 2 and more | — | Not permitted on disk. | — |

The value 2 is also `AieKernelKind::Count` and `AieKernelKind::Undecided`. `Undecided` is a queue state in the runtime only. It must never be in a file.

The on-disk values are fixed. Do not use a value again for a different meaning.

### 5.5 Version rules

- Increase `version_minor` for an additive change only. Examples: a new field at the end of the header or of the entry, or a new meaning for a reserved word that an old reader can ignore safely.
- Increase `version_major` for all other changes. A new `AieKernelKind` value is a major change, because each reader refuses a `kind` that it does not know.
- When you change the layout, change `amd_aie_section.h` and `format.py` together.

## 6. Kind `PdiInsts`

A `PdiInsts` entry holds two blobs:

| Blob | Source file | Rules |
| --- | --- | --- |
| Instruction sequence | `insts.bin` from `aiecc` | `insts_size` must be a multiple of 4. The driver gives `insts_size / 4` to the firmware as the word count. |
| PDI | `main.pdi` from `aiecc`, or the PDI in the `AIE_PARTITION` section of an xclbin | `pdi_size` must be more than 0 for ROCR. |

The section format permits a `PdiInsts` entry with no PDI. The packer permits it too. ROCR refuses it at load time with `HSA_STATUS_ERROR_INVALID_CODE_OBJECT`.

For an xclbin, the packer uses `xclbinutil --dump-section AIE_PARTITION:JSON:<file>`. The xclbin must hold exactly one PDI.

## 7. Kind `FullElf`: the nested full ELF

The blob at `insts_offset` is a complete full ELF. The packer makes one kernel table entry for each kernel in the full ELF. All these entries point to the same blob.

### 7.1 ELF header requirements

ROCR (`aie_elf::Parse`) requires these values:

| Field | Required value |
| --- | --- |
| `EI_CLASS` | `ELFCLASS32` (1). The packer refuses an ELF64 full ELF. |
| `EI_DATA` | `ELFDATA2LSB` (1). |
| `EI_OSABI` | The value for the arch of the AIE section that holds the full ELF. Only `aie2p` has a value: 69 (`0x45`). |
| `EI_ABIVERSION` | 1, or a different value. This value selects the relocation encoding (section 7.4). |
| `e_shentsize` | 40 (`sizeof(Elf32_Shdr)`). |
| `e_shnum` | More than 0. |
| `e_shstrndx` | Less than `e_shnum`. |

Full ELF operates on aie2p only. `aie2` has no full-ELF format. The ROCR loader refuses a `FullElf` entry in an `aie2` section. The driver also refuses a full-ELF batch on an `aie2` device.

### 7.2 Sections that ROCR uses

ROCR finds these sections by name, except for the string tables. It finds each string table through the `sh_link` of its symbol table, and does not examine the name of the string table:

| Section | Necessary | Use |
| --- | --- | --- |
| `.symtab` | Yes. `sh_entsize` must be 16. | Group signature symbols and kernel symbols. |
| `.strtab` | Yes | Names for `.symtab`. The section that the `sh_link` of `.symtab` gives. |
| `.dynsym` | For relocations | Symbols that the relocations refer to. `sh_entsize` must be 16. |
| `.dynstr` | For relocations | Names for `.dynsym`. The section that the `sh_link` of `.dynsym` gives. |
| `.rela.dyn` | For relocations | `Elf32_Rela` entries. `sh_entsize` must be 12. |
| `SHT_GROUP` sections | Yes, 1 or more | One group for each kernel instance. |
| `.ctrltext*` | One in each group | The control code. Must be `SHT_PROGBITS` and not empty. |
| `.pdi*` | One for each PDI that a kernel loads | The PDI. A PDI relocation symbol names it. More than one section can have the same name only if their contents are the same. |
| `.note.xrt.configuration` | No (packer only) | The partition column count. See section 8. |

ROCR reads relocations only if `.rela.dyn`, `.dynsym` and `.dynstr` are all present. If one of them is missing, no kernel has a PDI patch site. The loader then refuses each kernel.

### 7.3 Kernel names: the COMDAT group encoding

Each `SHT_GROUP` section identifies one kernel instance. The encoding is not standard ELF:

1. The `sh_info` of the group is a `.symtab` index. That symbol is the **instance symbol**.  Its name is the instance name, for example `sequence`.
2. The `st_shndx` of the instance symbol is **also a `.symtab` index**, not a section index. That symbol is the **kernel symbol**, for example `_Z4mainPcPcPc`.
3. The kernel name is the demangled kernel symbol. The demangler reads only the form  `_Z<length><name>`. It keeps `<name>` and discards the remainder. A symbol in a different form stays unchanged.
4. The full kernel name is `<kernel>:<instance>`, for example `main:sequence`.

`readelf -s` shows `st_shndx` as a section index (`Ndx`). For instance symbols, this display is not correct.

The group data is an array of 32-bit words:

- Word 0 is the group flags. ROCR and the packer both require `GRP_COMDAT` (`0x1`).
- Words 1 and more are the section indices of the group members.

ROCR uses the member whose name starts with `.ctrltext` as the control code. If a group has more than one such member, ROCR uses the last one. ROCR and the packer both ignore a group with no `.ctrltext` member. Both refuse the ELF if no group has a `.ctrltext` member. ROCR also refuses the ELF if two groups with a `.ctrltext` member have the same full kernel name.

The hsaco kernel name of a `FullElf` entry must be equal to the `<kernel>:<instance>` name in the full ELF. Two hsaco entries must not point to the same kernel in one full ELF. Kernels in the full ELF that no hsaco entry names are not loaded.

### 7.4 Relocations

Each `Elf32_Rela` entry in `.rela.dyn` gives one patch site. ROCR decodes it as follows:

| Item | Source |
| --- | --- |
| Symbol | `.dynsym[ELF32_R_SYM(r_info)]`. Its name is in `.dynstr`. |
| Patched section | The `st_shndx` of the symbol. ROCR uses the relocation only if this is the `.ctrltext` section of a group. Other relocations are ignored. |
| Byte offset in the control code | `r_offset`. |
| Patch scheme and addend, `EI_ABIVERSION == 1` | Scheme = `r_addend & 0xF`. Addend = `(uint32_t)r_addend >> 4`. |
| Patch scheme and addend, other versions | Scheme = `ELF32_R_TYPE(r_info)`. Addend = `r_addend`. |

The symbol name gives the meaning of the patch site:

| Symbol name | Meaning | Required scheme |
| --- | --- | --- |
| Starts with `.pdi` | The device address of the PDI in the section with this name. If more than one section has this name, their contents must be the same, or ROCR refuses the ELF. | 8 (`Address64`) |
| Decimal digits only, 0 to 4095 | The address of kernel argument N. | 5 (`ShimDma48`) |
| Any other name | Not supported (for example scratch pads or control packets). ROCR refuses the ELF. | — |

ROCR applies these limits:

- A kernel can have more than one PDI patch site, and the sites can name different PDI sections. For example, the control code from MLIR-AIR loads an empty PDI before and after the design runs. ROCR copies each different PDI into device memory one time for each kernel. Sites that name PDIs with the same contents use one copy.
- Each PDI patch site offset must not be 0, must be a multiple of 4, and `offset + 8` must not be more than the control-code size.
- Each argument patch site offset must be a multiple of 4, and `offset + 12` must not be more than the control-code size.
- An argument can have more than one patch site. An argument index can have no patch site.
- The number of kernel arguments is the highest argument index plus 1.
- The loader refuses a kernel that has no PDI patch site.

Preemption save and restore sections, control packets and scalar arguments are not supported.

### 7.5 Patch schemes

The loader applies the PDI patches once, at load time, to its copy of the control code. The driver applies the argument patches at dispatch time, on a new copy of that control code. It never patches the loader's copy, because `ShimDma48` adds to the existing value.

**`Address64` (8), for a PDI.** The loader writes the 64-bit device address of the PDI at each patch site of that PDI. The low word is first. This is a store, not an addition.

**`ShimDma48` (5), for arguments.** The patch site is three 32-bit words, `w[0]`, `w[1]` and `w[2]`. The driver does these steps, with `addr = argument address + addend`:

```cpp
base  = ((w[2] & 0xFFFF) << 32) | w[1]
base += addr + 0x80000000
w[1]  = base & 0xFFFFFFFC
w[2]  = (w[2] & 0xFFFF0000) | (base >> 32)
```

The driver does not change `w[0]`.

## 8. Column count (`num_cols`)

`num_cols` is the column count of the partition that the kernel was compiled for. It is not the number of columns that the design uses. For example, an npu2 design in column 0 only is still an 8-column kernel.

The packer gets the value as follows:

| Input form | Source of `num_cols` |
| --- | --- |
| PDI and instruction sequence | The user must give it. The two files do not record it. |
| xclbin and instruction sequence | `aie_partition.partition.column_width` in the `AIE_PARTITION` JSON. |
| Full ELF | The `.note.xrt.configuration` section. |

The `.note.xrt.configuration` section must hold exactly one note: owner `"XRT\0"`, type 6, `descsz` 4. The descriptor is the column count as a `uint32`. XRT reads the same note with `xrt::elf::get_partition_size()`.

For the xclbin and full-ELF forms, a value from the user is a cross-check. If it is not equal to the value in the file, the packer stops with an error. The packer uses the value from the user only if the file records no value.

The packer refuses `num_cols == 0`. The ROCR loader refuses a kernel if `num_cols` is 0 or more than the column count of the agent (section 4.3).

At dispatch, the driver makes the hardware context with `num_cols × rows` tiles. The value is the highest `num_cols` of the kernels in the batch. A queue context gets larger when a batch needs more columns. It does not get smaller in the same mode.

## 9. HSA interface

### 9.1 Kernel argument buffer

The packet field `kernarg_address` points to `2 × num_kernargs` consecutive `uint64_t` values:

| Index | Value |
| --- | --- |
| `0` to `num_kernargs − 1` | The device address of each argument buffer. |
| `num_kernargs` to `2 × num_kernargs − 1` | The size of each argument buffer. |

Thus the correct `kernarg_size` is `16 × number of arguments`. For example, use 32 for two
buffer arguments.

- `PdiInsts`: the loader does not examine `kernarg_size`. It gives the value to the   application through the symbol.
- `FullElf`: the full ELF is the authority. The loader calculates `16 × (highest argument index + 1)`. If the entry has 0, the loader uses the calculated
  value. If the entry has a different value that is not 0, the load fails.
- `FullElf`: at dispatch, `num_kernargs` in the packet must be equal to the number of arguments in the full ELF.

### 9.2 Symbol information

| `hsa_executable_symbol_info_t` | Value for an AIE kernel |
| --- | --- |
| `KIND` | `HSA_SYMBOL_KIND_KERNEL` |
| `LINKAGE` | `HSA_SYMBOL_LINKAGE_PROGRAM` |
| `KERNEL_OBJECT` | The address of the `AieKernelDescriptor`. 0 until the executable is frozen. |
| `KERNEL_KERNARG_SEGMENT_SIZE` | `kernarg_size` (for `FullElf`, the value from section 9.1). |
| `KERNEL_KERNARG_SEGMENT_ALIGNMENT` | 64 |
| `KERNEL_GROUP_SEGMENT_SIZE` | 0 |
| `KERNEL_PRIVATE_SEGMENT_SIZE` | 0 |

The kernel descriptor is a host structure. It is not part of the file format. Its first word is a version (`kAieKernelDescriptorVersion`, now 1). The driver examines this word to find a handle that is not valid.

## 10. Load procedure in ROCR

`hsa_executable_load_agent_code_object` does these steps for an AIE agent:

1. If the agent is an AIE agent and the file has an AIE section (section 4.3), use the AIE path. On a platform that is not Linux, the result is `HSA_STATUS_ERROR_INVALID_CODE_OBJECT`.
2. Select the AIE section for the arch of the agent (section 4.3).
3. Parse the header and the kernel table of that section (section 11).
4. Refuse the load if a kernel name is already in the executable for this agent (`HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED`).
5. For each kernel:
   1. Refuse a `kind` of 2 or more.
   2. Refuse `num_cols` of 0, or more than the agent columns.
   3. `PdiInsts`: refuse an entry with no PDI. Copy the instruction blob and the PDI blob into device memory. Get the BO handles.
   4. `FullElf`: parse the full ELF (one time for each distinct blob). Find the kernel by name. Examine `kernarg_size`. Copy each different PDI into device memory and get its BO handle. Keep the control code and the patch sites in host memory.
6. Publish the symbols. Before this step, a failure releases all device memory from steps 5.3 and 5.4.

Device memory for blobs comes from the device SVM region (the device heap). Each buffer has 64-byte alignment. The loader copies each distinct blob one time. It identifies a blob by its address and size in the hsaco buffer. This does not apply to the PDIs of a full-ELF kernel: the loader copies them out of the nested ELF and places them once per kernel, so two kernels that load the same PDI each get their own copy. After the copy, the loader flushes the CPU cache for the buffer one time.

The loaded code object is not visible to `hsa_ven_amd_loader_executable_iterate_loaded_code_objects`. It has no `r_debug` link-map entry. Its load base, load size and delta are 0.

## 11. Validation summary

The table shows which tool examines each rule. "—" means that the tool does not examine the rule.

| Rule | ROCR loader | `aie-hsaco` (write) | `aie-hsaco-dump` |
| --- | --- | --- | --- |
| `magic == 0x4B454941` | Yes (finds section) | Writes it | Yes |
| `version_major == 1` | Yes | Writes it | Yes |
| `header_size >= 48` | Yes | Writes 48 | Yes |
| `kernel_entry_size >= 44` | Yes | Writes 44 | Yes |
| Kernel table in section | Yes | — | Yes |
| String table after kernel table, blob pool after string table | Yes | Writes in this sequence | Yes |
| `kernel_count >= 1` | Yes | Yes | — |
| `kind < 2` | Yes | Yes | Yes |
| `insts_size > 0` | Yes | Yes | Yes |
| `PdiInsts`: `insts_size % 4 == 0` | Yes | — | — |
| Blobs in section | Yes | — | Yes |
| Blobs in blob pool (not before `blob_pool_offset`) | Yes | Writes them there | Yes |
| `pdi_size == 0` → `pdi_offset == 0` | Yes | Yes | — |
| `PdiInsts` has a PDI | Yes | — | — |
| `FullElf` has no separate PDI | Yes | Yes | Yes |
| Name offset in string table | Yes | Yes | Yes |
| Name has NUL terminator in string table | Yes | Yes | Yes |
| Unique kernel names | Yes | Yes | — |
| `num_cols >= 1` | Yes | Yes | — |
| `num_cols <=` agent columns | Yes | — | — |
| `reserved` words are 0 | — | Writes 0 | — |
| Full ELF is ELF32 | Yes | Yes | — |
| Full ELF `EI_OSABI` agrees with the section arch | Yes | — | — |
| Full ELF groups are COMDAT | Yes | Yes | — |
| Full ELF groups without `.ctrltext` are not kernels | Yes | Yes | — |
| Full ELF patch sites in range | Yes | — | — |
| Full ELF PDI section names are not ambiguous | Yes | — | — |
| Full ELF kernel names are unique | Yes | — | — |

## 12. The packer command line

```bash
aie-hsaco --hsaco OUT.hsaco --arch {aie2,aie2p} --kernel SPEC [--kernel SPEC ...]
```

| `--kernel` form | Result |
| --- | --- |
| `NAME:INSTS[:PDI]:KERNARG_SIZE:NUM_COLS` | One `PdiInsts` entry. |
| `xclbin:NAME:XCLBIN:INSTS:KERNARG_SIZE[:NUM_COLS]` | One `PdiInsts` entry. The PDI comes from the xclbin. |
| `elf:PATH[:KERNARG_SIZE[:NUM_COLS]]` | One `FullElf` entry for each COMDAT group that has a `.ctrltext` member, in sorted name sequence. `KERNARG_SIZE` is 0 if not given. |

The colon grammar has no escape. It cannot hold a path that contains a colon. A kernel cannot have the name `elf` or `xclbin`. For these cases, use the long options: `--kernel-name` or `--kernel-elf` starts a kernel, and `--kernel-insts`, `--kernel-pdi`, `--kernel-xclbin`, `--kernel-kernarg` and `--kernel-cols` give its fields.

The kernel table has the kernels in the sequence of the command line.

Examples from the rocrtst AIE suite:

```bash
aie-hsaco --hsaco vsadd.hsaco     --arch aie2p --kernel vsadd:insts.bin:design.pdi:32:1
aie-hsaco --hsaco vsadd_elf.hsaco --arch aie2p --kernel elf:aie.elf:0:1
```

The second command makes the kernel `main:sequence`.

## 13. Example section

This is the `aie2p` section from the command below. The input files are 8-byte test files, not real kernels. The command ran with the packer at `e2994ea5223`.

```bash
aie-hsaco --hsaco t.hsaco --arch aie2p \
  --kernel 'k1:insts.bin:main.pdi:32:4' --kernel 'k2:insts.bin:main.pdi:16:4'
```

```text
0000: 41 49 45 4b 01 00 00 00 30 00 00 00 02 00 00 00
0010: 2c 00 00 00 88 00 00 00 06 00 00 00 8e 00 00 00
0020: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0030: 00 00 00 00 8e 00 00 00 08 00 00 00 96 00 00 00
0040: 08 00 00 00 20 00 00 00 04 00 00 00 00 00 00 00
0050: 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 00
0060: 8e 00 00 00 08 00 00 00 96 00 00 00 08 00 00 00
0070: 10 00 00 00 04 00 00 00 00 00 00 00 00 00 00 00
0080: 00 00 00 00 00 00 00 00 6b 31 00 6b 32 00 01 02
0090: 03 04 05 06 07 08 50 44 49 44 41 54 41 21
```

| Offset | Bytes | Meaning |
| --- | --- | --- |
| `0x00` | `41 49 45 4b` | `magic` = `"AIEK"` |
| `0x04` | `01 00` / `00 00` | Version 1.0 |
| `0x08` | `0x30` | `header_size` = 48 |
| `0x0c` | `2` | `kernel_count` |
| `0x10` | `0x2c` | `kernel_entry_size` = 44 |
| `0x14` | `0x88`, `6` | String table at `0x88`, size 6 |
| `0x1c` | `0x8e` | Blob pool at `0x8e` |
| `0x30` | entry 0 | `k1`: name 0, insts `0x8e`/8, PDI `0x96`/8, kernarg 32, cols 4, kind 0 |
| `0x5c` | entry 1 | `k2`: name 3, insts `0x8e`/8, PDI `0x96`/8, kernarg 16, cols 4, kind 0 |
| `0x88` | `k1\0k2\0` | String table |
| `0x8e` | 8 bytes | Instruction blob. The two entries use one copy. |
| `0x96` | `PDIDATA!` | PDI blob. The two entries use one copy. |

In the container, the section starts at file offset `0x51`. This offset is not aligned.

## 14. Known differences between the tools

1. **`PdiInsts` with no PDI.** The format and the packer permit it. ROCR refuses it.
2. **`FullElf` in an `aie2` section.** The packer permits it. ROCR refuses it at load time (section 7.1).
