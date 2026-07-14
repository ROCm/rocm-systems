// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translate_test.cpp
/// @brief CPU-only unit tests for the DBT translation pipeline.
///
/// Tests encoding correctness, legalization table integrity, and structural
/// properties of translated code objects — without requiring a GPU. Covers:
///   - Coherency bit remapping (GFX940→GFX12, GFX9→GFX12)
///   - Encoding field preservation across SOP1/SOP2/SOPP/SMEM/VOP3 formats
///   - Decode-encode round-trip for CDNA4→RDNA4
///   - Legalization table lookup and zero-ILLEGAL invariant across all ISA pairs
///   - Waitcnt decode/encode (GFX9 monolithic → GFX12 split counters)
///
/// These tests complement the hardware tests in hsa_translate_test.cpp which
/// verify correctness on real DBT host GPUs.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/encoding_fields.h"
#include "rocjitsu/code/dbt/generated/encoding_gfx1250_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna1.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_gfx1250_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna2_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna2_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_5_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna4_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up_for_test(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

std::array<uint32_t, 2> pack_rdna4_smem_load_for_test(uint16_t sdata, uint16_t sbase, uint8_t op,
                                                      uint32_t byte_offset) {
  constexpr uint8_t smem_encoding = 0x3d;
  constexpr uint8_t no_soffset = 0x7c;
  return {
      (static_cast<uint32_t>(sbase & 0x3Fu)) | (static_cast<uint32_t>(sdata & 0x7Fu) << 6) |
          (static_cast<uint32_t>(op & 0x3Fu) << 13) | (static_cast<uint32_t>(smem_encoding) << 26),
      (byte_offset & 0x00FF'FFFFu) | (static_cast<uint32_t>(no_soffset) << 25),
  };
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_text_and_rodata() {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t rodata_size = 4;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t shstrtab_offset = rodata_offset + rodata_size;
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 4;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 3;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  const std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  const uint32_t rodata_word = 0xA5A55A5Au;
  std::memcpy(image.data() + rodata_offset, &rodata_word, sizeof(rodata_word));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = sizeof(uint32_t);

  shdrs[3].sh_name = shstrtab_name;
  shdrs[3].sh_type = SHT_STRTAB;
  shdrs[3].sh_offset = shstrtab_offset;
  shdrs[3].sh_size = shstrtab.size();
  shdrs[3].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_load_segments() {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t rodata_size = 4;
  constexpr uint64_t load_align = 0x1000;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t rodata_symbol_name = add_elf_name(strtab, "rodata_object");
  const uint32_t text_symbol_name = add_elf_name(strtab, "text_start");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 3;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = rodata_offset;
  phdrs[1].p_vaddr = rodata_vaddr;
  phdrs[1].p_paddr = rodata_vaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  const std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  const uint32_t rodata_word = 0xA5A55A5Au;
  std::memcpy(image.data() + rodata_offset, &rodata_word, sizeof(rodata_word));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = rodata_symbol_name;
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = rodata_size;
  syms[2].st_name = text_symbol_name;
  syms[2].st_shndx = 1;
  syms[2].st_value = text_vaddr;
  syms[2].st_size = text_size;
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = rodata_vaddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = sizeof(uint32_t);

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_offset;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 4;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = strtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = strtab_offset;
  shdrs[4].sh_size = strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t>
make_minimal_amdgpu_elf_with_descriptor_after_text(const std::vector<uint32_t> &text_words) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = sizeof(KD);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = add_elf_name(strtab, "kernel.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 2;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = rodata_offset;
  phdrs[1].p_vaddr = rodata_vaddr;
  phdrs[1].p_paddr = rodata_vaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  KD kd{};
  kd.kernel_code_entry_byte_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  std::memcpy(image.data() + rodata_offset, &kd, sizeof(kd));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = kd_symbol_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = sizeof(KD);
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = rodata_vaddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = 64;

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_offset;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 4;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = strtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = strtab_offset;
  shdrs[4].sh_size = strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_descriptor_after_text() {
  return make_minimal_amdgpu_elf_with_descriptor_after_text({0xBF800000u, 0xBF800000u});
}

std::vector<uint8_t>
make_minimal_amdgpu_elf_with_descriptor_and_text(std::span<const uint32_t> text_words,
                                                 uint32_t elf_mach) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = sizeof(KD);
  const uint64_t text_size = text_words.size_bytes();

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = add_elf_name(strtab, "kernel.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 2;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = elf_mach;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = rodata_offset;
  phdrs[1].p_vaddr = rodata_vaddr;
  phdrs[1].p_paddr = rodata_vaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  KD kd{};
  kd.kernel_code_entry_byte_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  std::memcpy(image.data() + rodata_offset, &kd, sizeof(kd));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = kd_symbol_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = sizeof(KD);
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = rodata_vaddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = 64;

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_offset;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 4;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = strtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = strtab_offset;
  shdrs[4].sh_size = strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_two_kernel_descriptors_and_text(
    std::span<const uint32_t> text_words, uint32_t elf_mach, uint64_t entry0_word,
    uint64_t entry1_word) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = 2 * sizeof(KD);
  const uint64_t text_size = text_words.size_bytes();

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kernel0_name = add_elf_name(strtab, "kernel0.kd");
  const uint32_t kernel1_name = add_elf_name(strtab, "kernel1.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 3;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  ehdr.e_flags = elf_mach;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = rodata_offset;
  phdrs[1].p_vaddr = rodata_vaddr;
  phdrs[1].p_paddr = rodata_vaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  std::array<KD, 2> descriptors{};
  descriptors[0].kernel_code_entry_byte_offset =
      static_cast<int64_t>(text_vaddr + entry0_word * sizeof(uint32_t)) -
      static_cast<int64_t>(rodata_vaddr);
  descriptors[1].kernel_code_entry_byte_offset =
      static_cast<int64_t>(text_vaddr + entry1_word * sizeof(uint32_t)) -
      static_cast<int64_t>(rodata_vaddr + sizeof(KD));
  std::memcpy(image.data() + rodata_offset, descriptors.data(), rodata_size);
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = kernel0_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = sizeof(KD);
  syms[2].st_name = kernel1_name;
  syms[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[2].st_shndx = 2;
  syms[2].st_value = rodata_vaddr + sizeof(KD);
  syms[2].st_size = sizeof(KD);
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = rodata_vaddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = 64;

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_offset;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 4;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = strtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = strtab_offset;
  shdrs[4].sh_size = strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_two_kernel_descriptors() {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const std::array<uint32_t, 2> text_words = {kCdna4SEndpgm, kCdna4SEndpgm};
  return make_minimal_amdgpu_elf_with_two_kernel_descriptors_and_text(
      text_words, EF_AMDGPU_MACH_AMDGCN_GFX950, 0, 1);
}

std::vector<uint8_t>
make_minimal_amdgpu_elf_with_two_kernel_descriptors(const std::vector<uint32_t> &text_words) {
  return make_minimal_amdgpu_elf_with_two_kernel_descriptors_and_text(
      text_words, EF_AMDGPU_MACH_AMDGCN_GFX950, 0, 1);
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_relocation_after_text() {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t data_size = 8;
  constexpr uint64_t load_align = 0x1000;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t data_name = add_elf_name(shstrtab, ".data");
  const uint32_t rela_name = add_elf_name(shstrtab, ".rela.dyn");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  const uint64_t data_offset = text_offset + text_size;
  const uint64_t data_vaddr = text_vaddr + text_size + load_align;
  const uint64_t rela_offset = data_offset + data_size;
  constexpr size_t rela_count = 1;
  const uint64_t shstrtab_offset = rela_offset + rela_count * sizeof(Elf64_Rela);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 5;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 4;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x6; // PF_R | PF_W
  phdrs[1].p_offset = data_offset;
  phdrs[1].p_vaddr = data_vaddr;
  phdrs[1].p_paddr = data_vaddr;
  phdrs[1].p_filesz = data_size;
  phdrs[1].p_memsz = data_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  const std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  const uint64_t data_word = 0x1234567890ABCDEFull;
  std::memcpy(image.data() + data_offset, &data_word, sizeof(data_word));

  Elf64_Rela rela{};
  rela.r_offset = data_vaddr;
  std::memcpy(image.data() + rela_offset, &rela, sizeof(rela));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = data_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC | SHF_WRITE;
  shdrs[2].sh_addr = data_vaddr;
  shdrs[2].sh_offset = data_offset;
  shdrs[2].sh_size = data_size;
  shdrs[2].sh_addralign = sizeof(uint64_t);

  shdrs[3].sh_name = rela_name;
  shdrs[3].sh_type = SHT_RELA;
  shdrs[3].sh_offset = rela_offset;
  shdrs[3].sh_size = rela_count * sizeof(Elf64_Rela);
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Rela);

  shdrs[4].sh_name = shstrtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = shstrtab_offset;
  shdrs[4].sh_size = shstrtab.size();
  shdrs[4].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t> make_large_amdgpu_elf_with_waitcnt_entry() {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t rodata_offset = 0x100;
  constexpr uint64_t rodata_vaddr = 0x100;
  constexpr uint64_t text_offset = 0x1000;
  constexpr uint64_t text_vaddr = 0x1000;
  constexpr uint64_t text_size = 0x21000;
  constexpr uint64_t rodata_size = sizeof(KD);
  constexpr uint64_t load_align = 0x1000;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = add_elf_name(strtab, "kernel.kd");

  const uint64_t strtab_offset = text_offset + text_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 2;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x4; // PF_R
  phdrs[0].p_offset = rodata_offset;
  phdrs[0].p_vaddr = rodata_vaddr;
  phdrs[0].p_paddr = rodata_vaddr;
  phdrs[0].p_filesz = rodata_size;
  phdrs[0].p_memsz = rodata_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x5; // PF_R | PF_X
  phdrs[1].p_offset = text_offset;
  phdrs[1].p_vaddr = text_vaddr;
  phdrs[1].p_paddr = text_vaddr;
  phdrs[1].p_filesz = text_size;
  phdrs[1].p_memsz = text_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  KD kd{};
  kd.kernel_code_entry_byte_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  std::memcpy(image.data() + rodata_offset, &kd, sizeof(kd));

  std::vector<uint32_t> text_words(text_size / sizeof(uint32_t),
                                   build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = cdna4::build_sopp(cdna4::kSWaitcntSopp)[0]; // Expands on RDNA4.
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = kd_symbol_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 1;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = sizeof(KD);
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = rodata_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC;
  shdrs[1].sh_addr = rodata_vaddr;
  shdrs[1].sh_offset = rodata_offset;
  shdrs[1].sh_size = rodata_size;
  shdrs[1].sh_addralign = 64;

  shdrs[2].sh_name = text_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[2].sh_addr = text_vaddr;
  shdrs[2].sh_offset = text_offset;
  shdrs[2].sh_size = text_size;
  shdrs[2].sh_addralign = 256;

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_offset;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 4;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = strtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = strtab_offset;
  shdrs[4].sh_size = strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

const Section *find_section(const CodeObject &co, std::string_view name) {
  for (const auto &section : co.all_sections()) {
    if (section->name() == name)
      return section.get();
  }
  return nullptr;
}

rocr::llvm::amdhsa::kernel_descriptor_t read_kernel_descriptor_for_test(const void *descriptor) {
  rocr::llvm::amdhsa::kernel_descriptor_t kd{};
  std::memcpy(&kd, descriptor, sizeof(kd));
  return kd;
}

void write_kernel_descriptor_for_test(void *descriptor,
                                      const rocr::llvm::amdhsa::kernel_descriptor_t &kd) {
  std::memcpy(descriptor, &kd, sizeof(kd));
}

bool has_error_containing(const TranslatedCodeObject &result, DiagnosticKind kind,
                          std::string_view message) {
  return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                     [&](const TranslationDiagnostic &diagnostic) {
                       return diagnostic.severity == DiagnosticSeverity::Error &&
                              diagnostic.kind == kind &&
                              diagnostic.message.find(message) != std::string::npos;
                     });
}

void enable_workgroup_id_x_sgpr(std::vector<uint8_t> &image) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

  AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(KD));

  KD kd{};
  std::memcpy(&kd, image.data() + rodata->sectionOffset(), sizeof(kd));
  kd.compute_pgm_rsrc2 |= rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X;
  std::memcpy(image.data() + rodata->sectionOffset(), &kd, sizeof(kd));
}

bool section_contains_vop3_opcode(const Section *section, uint16_t opcode) {
  if (section == nullptr)
    return false;
  const auto *words = reinterpret_cast<const uint32_t *>(section->data());
  const size_t word_count = section->size() / sizeof(uint32_t);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  for (size_t i = 0; i < word_count;) {
    const uint32_t word = words[i];
    if ((word >> 26) == 0x35u && ((word >> 16) & 0x3FFu) == opcode)
      return true;
    size_t inst_words = 1;
    if (decoder) {
      try {
        std::unique_ptr<Instruction> inst(decoder->decode(words + i));
        inst_words = std::max<size_t>(1, static_cast<size_t>(inst->size()) / sizeof(uint32_t));
      } catch (...) {
      }
    }
    i += std::min(inst_words, word_count - i);
  }
  return false;
}

bool section_contains_vop3_inst(const Section *section, uint16_t opcode, uint8_t vdst,
                                uint16_t src0, uint16_t src1) {
  if (section == nullptr)
    return false;
  const auto *words = reinterpret_cast<const uint32_t *>(section->data());
  const size_t word_count = section->size() / sizeof(uint32_t);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  for (size_t i = 0; i + 1 < word_count;) {
    const uint32_t w0 = words[i];
    if ((w0 >> 26) == 0x35u && ((w0 >> 16) & 0x3FFu) == opcode &&
        static_cast<uint8_t>(w0 & 0xFFu) == vdst) {
      const uint32_t w1 = words[i + 1u];
      if (static_cast<uint16_t>(w1 & 0x1FFu) == src0 &&
          static_cast<uint16_t>((w1 >> 9) & 0x1FFu) == src1)
        return true;
    }
    size_t inst_words = 1;
    if (decoder) {
      try {
        std::unique_ptr<Instruction> inst(decoder->decode(words + i));
        inst_words = std::max<size_t>(1, static_cast<size_t>(inst->size()) / sizeof(uint32_t));
      } catch (...) {
      }
    }
    i += std::min(inst_words, word_count - i);
  }
  return false;
}

bool section_contains_vop3_inst_literal(const Section *section, uint16_t opcode,
                                        std::optional<uint8_t> vdst, uint16_t src0, uint16_t src1,
                                        uint32_t literal) {
  if (section == nullptr)
    return false;
  const auto *words = reinterpret_cast<const uint32_t *>(section->data());
  const size_t word_count = section->size() / sizeof(uint32_t);
  for (size_t i = 0; i + 2 < word_count; ++i) {
    const uint32_t w0 = words[i];
    const uint32_t w1 = words[i + 1u];
    if ((w0 >> 26) == 0x35u && ((w0 >> 16) & 0x3FFu) == opcode &&
        (!vdst || static_cast<uint8_t>(w0 & 0xFFu) == *vdst) &&
        static_cast<uint16_t>(w1 & 0x1FFu) == src0 &&
        static_cast<uint16_t>((w1 >> 9) & 0x1FFu) == src1 && words[i + 2u] == literal)
      return true;
  }
  return false;
}

std::optional<uint16_t> section_vop1_src_for_dst(const Section *section, uint8_t opcode,
                                                 uint8_t vdst) {
  if (section == nullptr)
    return std::nullopt;
  const auto *words = reinterpret_cast<const uint32_t *>(section->data());
  const size_t word_count = section->size() / sizeof(uint32_t);
  for (size_t i = 0; i < word_count; ++i) {
    const uint32_t word = words[i];
    if ((word >> 25) == 0x3Fu && ((word >> 9) & 0x7Fu) == opcode &&
        static_cast<uint8_t>((word >> 17) & 0xFFu) == vdst)
      return static_cast<uint16_t>(word & 0x1FFu);
  }
  return std::nullopt;
}

bool src_reads_vgpr_for_test(uint16_t src, uint8_t vgpr) {
  return src >= 256u && src < 512u && static_cast<uint8_t>(src - 256u) == vgpr;
}

[[maybe_unused]] bool
section_contains_low_first_overlapping_v_mul_u64_pair(const Section *section) {
  if (section == nullptr)
    return false;
  const auto *words = reinterpret_cast<const uint32_t *>(section->data());
  const size_t word_count = section->size() / sizeof(uint32_t);
  std::vector<size_t> starts;
  starts.reserve(word_count);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  for (size_t i = 0; i < word_count;) {
    starts.push_back(i);
    size_t inst_words = 1;
    if (decoder) {
      try {
        std::unique_ptr<Instruction> inst(decoder->decode(words + i));
        inst_words = std::max<size_t>(1, static_cast<size_t>(inst->size()) / sizeof(uint32_t));
      } catch (...) {
      }
    }
    i += std::min(inst_words, word_count - i);
  }

  for (size_t inst_index = 0; inst_index + 1 < starts.size(); ++inst_index) {
    const size_t lo_index = starts[inst_index];
    const size_t hi_index = starts[inst_index + 1];
    if (lo_index + 1 >= word_count || hi_index + 1 >= word_count)
      continue;
    const uint32_t lo_w0 = words[lo_index];
    const uint32_t hi_w0 = words[hi_index];
    if ((lo_w0 >> 26) != 0x35u || (hi_w0 >> 26) != 0x35u)
      continue;
    if (((lo_w0 >> 16) & 0x3FFu) != 812u || ((hi_w0 >> 16) & 0x3FFu) != 813u)
      continue;

    const uint8_t lo_vdst = static_cast<uint8_t>(lo_w0 & 0xFFu);
    const uint8_t hi_vdst = static_cast<uint8_t>(hi_w0 & 0xFFu);
    if (hi_vdst != static_cast<uint8_t>(lo_vdst + 1u))
      continue;

    const uint32_t lo_w1 = words[lo_index + 1];
    const uint32_t hi_w1 = words[hi_index + 1];
    const uint16_t lo_src0 = static_cast<uint16_t>(lo_w1 & 0x1FFu);
    const uint16_t lo_src1 = static_cast<uint16_t>((lo_w1 >> 9) & 0x1FFu);
    const uint16_t hi_src0 = static_cast<uint16_t>(hi_w1 & 0x1FFu);
    const uint16_t hi_src1 = static_cast<uint16_t>((hi_w1 >> 9) & 0x1FFu);
    if (lo_src0 != hi_src0 || lo_src1 != hi_src1)
      continue;
    if (src_reads_vgpr_for_test(lo_src0, lo_vdst) || src_reads_vgpr_for_test(lo_src1, lo_vdst))
      return true;
  }
  return false;
}

constexpr uint16_t hwreg_mode_vgpr_msb_for_test() {
  constexpr uint8_t kHwregMode = 1;
  constexpr uint8_t kVgprMsbModeOffset = 12;
  constexpr uint8_t kVgprMsbModeSize = 8;
  return static_cast<uint16_t>((kHwregMode & 0x3Fu) | ((kVgprMsbModeOffset & 0x1Fu) << 6) |
                               (((kVgprMsbModeSize - 1u) & 0x1Fu) << 11));
}

constexpr uint16_t hwreg_mode_replay_mode_for_test() {
  constexpr uint8_t kHwregMode = 1;
  constexpr uint8_t kReplayModeOffset = 25;
  constexpr uint8_t kReplayModeSize = 1;
  return static_cast<uint16_t>((kHwregMode & 0x3Fu) | ((kReplayModeOffset & 0x1Fu) << 6) |
                               (((kReplayModeSize - 1u) & 0x1Fu) << 11));
}

constexpr uint32_t build_sopk_for_test(uint8_t op, uint16_t simm16, uint8_t sdst = 0) {
  return 0xB0000000u | (simm16 & 0xFFFFu) | ((sdst & 0x7Fu) << 16) | ((op & 0x1Fu) << 23);
}

constexpr uint32_t build_vop1_for_test(uint8_t op, uint8_t vdst, uint16_t src0) {
  return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
}

constexpr uint32_t rdna4_vgpr_msb_reset_setreg_for_test() {
  constexpr uint8_t kOpSSetregImm32B32 = 19;
  return build_sopk_for_test(kOpSSetregImm32B32, hwreg_mode_vgpr_msb_for_test());
}

std::vector<uint32_t> section_words_for_test(const Section &section) {
  std::vector<uint32_t> words(section.size() / sizeof(uint32_t));
  std::memcpy(words.data(), section.data(), words.size() * sizeof(uint32_t));
  return words;
}

class SyntheticSectionForTest final : public Section {
public:
  SyntheticSectionForTest(std::string name, std::unique_ptr<char[]> data, size_t size,
                          uint64_t offset)
      : Section(std::move(name), std::move(data)), size_(size), offset_(offset) {}

  std::size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return offset_; }

private:
  size_t size_ = 0;
  uint64_t offset_ = 0;
};

std::vector<uint32_t> section_tail_words_for_test(const Section &section, size_t byte_offset) {
  EXPECT_EQ(byte_offset % sizeof(uint32_t), 0u);
  EXPECT_EQ(section.size() % sizeof(uint32_t), 0u);
  EXPECT_GE(section.size(), byte_offset);
  if (byte_offset > section.size() || byte_offset % sizeof(uint32_t) != 0 ||
      section.size() % sizeof(uint32_t) != 0)
    return {};

  const size_t byte_count = section.size() - byte_offset;
  std::vector<uint32_t> words(byte_count / sizeof(uint32_t));
  std::memcpy(words.data(), section.data() + byte_offset, byte_count);
  return words;
}

std::vector<uint32_t> local_expanded_text_words_for_test(const CodeObject &translated,
                                                         size_t original_text_size) {
  EXPECT_EQ(find_section(translated, ".rj_translations"), nullptr);
  EXPECT_FALSE(translated.text_sections().empty());
  if (translated.text_sections().empty())
    return {};

  const Section *text = translated.text_sections()[0];
  EXPECT_GT(text->size(), original_text_size);
  return section_tail_words_for_test(*text, original_text_size);
}

const Section *local_expanded_text_section_for_test(const CodeObject &translated,
                                                    size_t original_text_size) {
  EXPECT_EQ(find_section(translated, ".rj_translations"), nullptr);
  EXPECT_FALSE(translated.text_sections().empty());
  if (translated.text_sections().empty())
    return nullptr;

  const Section *text = translated.text_sections()[0];
  EXPECT_GT(text->size(), original_text_size);
  const auto words = section_tail_words_for_test(*text, original_text_size);
  if (words.empty())
    return nullptr;

  const size_t byte_count = words.size() * sizeof(uint32_t);
  auto data = std::make_unique<char[]>(byte_count);
  std::memcpy(data.get(), words.data(), byte_count);

  static thread_local std::unique_ptr<SyntheticSectionForTest> section;
  section =
      std::make_unique<SyntheticSectionForTest>(".text.expanded_tail", std::move(data), byte_count,
                                                text->sectionOffset() + original_text_size);
  return section.get();
}

std::optional<size_t> find_word_sequence_for_test(std::span<const uint32_t> words,
                                                  std::span<const uint32_t> needle) {
  if (needle.empty() || needle.size() > words.size())
    return std::nullopt;
  for (size_t i = 0; i + needle.size() <= words.size(); ++i) {
    if (std::equal(needle.begin(), needle.end(), words.begin() + static_cast<std::ptrdiff_t>(i)))
      return i;
  }
  return std::nullopt;
}

std::optional<size_t> find_branch_terminated_prefix_for_test(std::span<const uint32_t> words,
                                                             std::span<const uint32_t> prefix) {
  if (prefix.empty())
    return std::nullopt;
  for (size_t i = 0; i + prefix.size() < words.size(); ++i) {
    if (!std::equal(prefix.begin(), prefix.end(), words.begin() + static_cast<std::ptrdiff_t>(i)))
      continue;
    if (((words[i + prefix.size()] >> 16) & 0x7Fu) == sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4))
      return i;
  }
  return std::nullopt;
}

[[nodiscard]] constexpr uint32_t pack_sopc_for_test(uint32_t op, uint32_t ssrc0, uint32_t ssrc1) {
  return 0xBF000000u | ((op & 0x7Fu) << 16) | ((ssrc1 & 0xFFu) << 8) | (ssrc0 & 0xFFu);
}

std::optional<uint32_t> exec_lo_literal_at_for_test(std::span<const uint32_t> words, size_t index) {
  if (index + 1 >= words.size())
    return std::nullopt;
  auto sop1 = std::bit_cast<gfx1250::Sop1MachineInst>(words[index]);
  if (sop1.encoding == 0x17Du && sop1.op == 0u && sop1.sdst == 126u && sop1.ssrc0 == 0xFFu)
    return words[index + 1];
  return std::nullopt;
}

std::optional<size_t>
find_gfx1250_rdna4_entry_stub_body_word_offset_for_test(std::span<const uint32_t> words) {
  for (size_t i = 0; i + 2 < words.size(); ++i) {
    if (words[i] != rdna4_vgpr_msb_reset_setreg_for_test() || words[i + 1] != 0)
      continue;
    if (((words[i + 2] >> 16) & 0x7Fu) == sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4))
      return i + 3;
  }
  return std::nullopt;
}

void expect_gfx1250_rdna4_entry_stub_for_test(const CodeObject &translated) {
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto words = section_words_for_test(*translations);
  EXPECT_TRUE(find_gfx1250_rdna4_entry_stub_body_word_offset_for_test(words).has_value());
}

std::vector<uint32_t>
entry_body_words_after_gfx1250_rdna4_stub_for_test(std::span<const uint32_t> words) {
  const auto body_offset = find_gfx1250_rdna4_entry_stub_body_word_offset_for_test(words);
  EXPECT_TRUE(body_offset.has_value());
  if (!body_offset)
    return {words.begin(), words.end()};
  return {words.begin() + static_cast<std::ptrdiff_t>(*body_offset), words.end()};
}

std::vector<uint32_t> entry_body_words_after_gfx1250_rdna4_stub_for_test(const Section &section) {
  return entry_body_words_after_gfx1250_rdna4_stub_for_test(section_words_for_test(section));
}

constexpr uint16_t hwreg_gfx1250_grid_mode_for_test() {
  constexpr uint8_t kHwregGfx1250GridMode = 28;
  constexpr uint8_t kGridModeOffset = 6;
  constexpr uint8_t kGridModeSize = 4;
  return static_cast<uint16_t>((kHwregGfx1250GridMode & 0x3Fu) | ((kGridModeOffset & 0x1Fu) << 6) |
                               (((kGridModeSize - 1u) & 0x1Fu) << 11));
}

constexpr uint8_t kNullSgprForTest = 124;
constexpr uint16_t kRdna4M0ForTest = 125;
constexpr uint16_t kInlineMinusOneForTest = 193;
constexpr uint32_t kGfx1250PrivateBorrowedVgprCountForTest = 38u;
constexpr uint32_t kGfx1250HighBankShadowLowSaveVgprCountForTest = 33u;

std::optional<size_t> find_vop3_sdst_for_test(std::span<const uint32_t> words, uint16_t op,
                                              uint8_t vdst, uint16_t src0, uint16_t src1) {
  for (size_t i = 0; i + 1 < words.size(); ++i) {
    rdna4::Vop3SdstEncMachineInst decoded{};
    std::memcpy(&decoded, words.data() + i, sizeof(decoded));
    if (decoded.encoding == 0x35u && decoded.op == op && decoded.vdst == vdst &&
        decoded.src0 == src0 && decoded.src1 == src1)
      return i;
  }
  return std::nullopt;
}

std::optional<size_t> find_vop3_for_test(std::span<const uint32_t> words, uint16_t op, uint8_t vdst,
                                         uint16_t src0, uint16_t src1,
                                         std::optional<uint16_t> src2 = std::nullopt) {
  for (size_t i = 0; i + 1 < words.size(); ++i) {
    rdna4::Vop3MachineInst decoded{};
    std::memcpy(&decoded, words.data() + i, sizeof(decoded));
    if (decoded.encoding == 0x35u && decoded.op == op && decoded.vdst == vdst &&
        decoded.src0 == src0 && decoded.src1 == src1 && (!src2 || decoded.src2 == *src2))
      return i;
  }
  return std::nullopt;
}

std::optional<size_t> find_vop2_for_test(std::span<const uint32_t> words, uint8_t op,
                                         std::optional<uint8_t> vdst, uint16_t src0,
                                         uint8_t vsrc1) {
  for (size_t i = 0; i < words.size(); ++i) {
    const auto decoded = std::bit_cast<rdna4::Vop2MachineInst>(words[i]);
    if (decoded.op == op && (!vdst || decoded.vdst == *vdst) && decoded.src0 == src0 &&
        decoded.vsrc1 == vsrc1)
      return i;
  }
  return std::nullopt;
}

std::optional<size_t> find_vop1_for_test(std::span<const uint32_t> words, uint8_t op,
                                         std::optional<uint8_t> vdst, uint16_t src0) {
  for (size_t i = 0; i < words.size(); ++i) {
    const auto decoded = std::bit_cast<rdna4::Vop1MachineInst>(words[i]);
    if (decoded.op == op && (!vdst || decoded.vdst == *vdst) && decoded.src0 == src0)
      return i;
  }
  return std::nullopt;
}

void expect_sub_u64_carry_chain(const CodeObject &translated, uint8_t vdst, uint16_t src0,
                                uint16_t src1) {
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  constexpr uint16_t kOpSubCoU32 = 769u;
  const auto base = find_vop3_sdst_for_test(cave_words, kOpSubCoU32, vdst, src0, src1);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 7u, cave_words.size());

  rdna4::Vop3SdstEncMachineInst sub_lo{};
  std::memcpy(&sub_lo, cave_words.data() + *base, sizeof(sub_lo));
  EXPECT_EQ(sub_lo.encoding, 0x35u);
  EXPECT_EQ(sub_lo.op, kOpSubCoU32);
  EXPECT_EQ(sub_lo.vdst, vdst);
  EXPECT_LE(sub_lo.sdst, 105u);
  EXPECT_EQ(sub_lo.sdst & 1u, 0u);
  EXPECT_EQ(sub_lo.src0, src0);
  EXPECT_EQ(sub_lo.src1, src1);

  EXPECT_EQ(cave_words[*base + 2u],
            build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3SdstEncMachineInst sub_hi{};
  std::memcpy(&sub_hi, cave_words.data() + *base + 3u, sizeof(sub_hi));
  EXPECT_EQ(sub_hi.encoding, 0x35u);
  EXPECT_EQ(sub_hi.op, 289u);
  EXPECT_EQ(sub_hi.vdst, vdst + 1u);
  EXPECT_EQ(sub_hi.sdst, kNullSgprForTest);
  EXPECT_EQ(sub_hi.src0, src0 + 1u);
  EXPECT_EQ(sub_hi.src1, src1 + 1u);
  EXPECT_EQ(sub_hi.src2, sub_lo.sdst);
  EXPECT_EQ(cave_words[*base + 5u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ((cave_words[*base + 6u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

void expect_add_u64_carry_chain(const CodeObject &translated, uint8_t vdst, uint16_t src0,
                                uint16_t src1, std::optional<uint32_t> literal_word = std::nullopt,
                                std::optional<uint16_t> src0_hi = std::nullopt) {
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const size_t expected_words = literal_word ? 8u : 7u;
  const auto cave_words = section_words_for_test(*translations);
  constexpr uint16_t kOpAddCoU32 = 768u;
  const auto base = find_vop3_sdst_for_test(cave_words, kOpAddCoU32, vdst, src0, src1);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + expected_words, cave_words.size());

  rdna4::Vop3SdstEncMachineInst add_lo{};
  std::memcpy(&add_lo, cave_words.data() + *base, sizeof(add_lo));
  EXPECT_EQ(add_lo.encoding, 0x35u);
  EXPECT_EQ(add_lo.op, kOpAddCoU32);
  EXPECT_EQ(add_lo.vdst, vdst);
  EXPECT_LE(add_lo.sdst, 105u);
  EXPECT_EQ(add_lo.sdst & 1u, 0u);
  EXPECT_EQ(add_lo.src0, src0);
  EXPECT_EQ(add_lo.src1, src1);

  const size_t wait_index = literal_word ? 3u : 2u;
  if (literal_word)
    EXPECT_EQ(cave_words[*base + 2u], *literal_word);
  EXPECT_EQ(cave_words[*base + wait_index],
            build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3SdstEncMachineInst add_hi{};
  std::memcpy(&add_hi, cave_words.data() + *base + wait_index + 1u, sizeof(add_hi));
  EXPECT_EQ(add_hi.encoding, 0x35u);
  EXPECT_EQ(add_hi.op, 288u);
  EXPECT_EQ(add_hi.vdst, vdst + 1u);
  EXPECT_EQ(add_hi.sdst, kNullSgprForTest);
  EXPECT_EQ(add_hi.src0, src0_hi.value_or(literal_word ? scalar_positive_inline_u32(0)
                                                       : static_cast<uint16_t>(src0 + 1u)));
  EXPECT_EQ(add_hi.src1, src1 + 1u);
  EXPECT_EQ(add_hi.src2, add_lo.sdst);
  EXPECT_EQ(cave_words[*base + wait_index + 3u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ((cave_words[*base + wait_index + 4u] >> 16) & 0x7Fu,
            sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

[[nodiscard]] constexpr uint16_t scalar_negative_inline_i32(int16_t value) {
  return static_cast<uint16_t>(192 - value);
}

TEST(CoherencyRemap, Gfx940ToGfx12AgentScope) {
  auto coh = remap_gfx940_to_gfx12({1, 0, 0});
  EXPECT_EQ(coh.scope, 1);
  EXPECT_EQ(coh.th, 0);
}

TEST(CoherencyRemap, Gfx940ToGfx12SystemScope) {
  auto coh = remap_gfx940_to_gfx12({1, 1, 0});
  EXPECT_EQ(coh.scope, 3);
  EXPECT_EQ(coh.th, 0);
}

TEST(CoherencyRemap, Gfx940ToGfx12NonTemporal) {
  auto coh = remap_gfx940_to_gfx12({0, 0, 1});
  EXPECT_EQ(coh.scope, 0);
  EXPECT_EQ(coh.th, 3);
}

TEST(CoherencyRemap, Gfx9GlcToGfx12) {
  auto coh_glc1 = remap_gfx9_to_gfx12({1});
  EXPECT_EQ(coh_glc1.scope, 2);
  EXPECT_EQ(coh_glc1.th, 0);

  auto coh_glc0 = remap_gfx9_to_gfx12({0});
  EXPECT_EQ(coh_glc0.scope, 0);
  EXPECT_EQ(coh_glc0.th, 0);
}

TEST(EncodingTranslator, Sop1PreservesRegisters) {
  cdna4::Sop1MachineInst src{};
  src.ssrc0 = 42;
  src.sdst = 17;
  src.op = 3;
  src.encoding = 0x17D;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP1, w0, 0, 0, 5);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 42);
  EXPECT_EQ(dst.sdst, 17);
  EXPECT_EQ(dst.op, 5);
  EXPECT_EQ(dst.encoding, 0x17D);
}

TEST(EncodingTranslator, Sop2PreservesRegisters) {
  cdna4::Sop2MachineInst src{};
  src.ssrc0 = 10;
  src.ssrc1 = 20;
  src.sdst = 30;
  src.op = 7;
  src.encoding = 0x2;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP2, w0, 0, 0, 7);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop2MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 10);
  EXPECT_EQ(dst.ssrc1, 20);
  EXPECT_EQ(dst.sdst, 30);
  EXPECT_EQ(dst.op, 7);
}

TEST(InstructionBuilder, Sop2SetsEncodingPrefix) {
  const uint32_t word = build_s_lshl_b32(1, 2, 3, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ((word >> 30) & 0x3u, 0x2u);
}

TEST(EncodingTranslator, SoppPreservesSimm16) {
  cdna4::SoppMachineInst src{};
  src.simm16 = 0xABCD;
  src.op = 12;
  src.encoding = 0x17F;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOPP, w0, 0, 0, 12);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::SoppMachineInst>(result.words[0]);
  EXPECT_EQ(dst.simm16, 0xABCD);
  EXPECT_EQ(dst.op, 12);
}

TEST(EncodingTranslator, SmemRemapsCoherency) {
  cdna4::SmemMachineInst src{};
  src.sbase = 5;
  src.sdata = 3;
  src.glc = 1;
  src.nv = 0;
  src.op = 0;
  src.offset = 0x100;
  src.soffset = 0x7F;
  src.encoding = 0x3D;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SMEM, words[0], words[1], 0, 0);

  ASSERT_EQ(result.word_count, 2);
  rdna4::SmemMachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.sbase, 5);
  EXPECT_EQ(dst.sdata, 3);
  EXPECT_EQ(dst.scope, 2);
  EXPECT_EQ(dst.th, 0);
  EXPECT_EQ(dst.nv, 0);
  EXPECT_EQ(dst.soffset, 0x7C); // CDNA4 null (0x7F) → RDNA4 null (0x7C)
}

TEST(EncodingTranslator, Vop3PreservesModifiers) {
  cdna4::Vop3MachineInst src{};
  src.vdst = 10;
  src.src0 = 100;
  src.src1 = 200;
  src.src2 = 50;
  src.clamp = 1;
  src.omod = 2;
  src.neg = 5;
  src.abs = 3;
  src.op = 100;
  src.encoding = 0x35;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_VOP3, words[0], words[1], 0, 100);

  ASSERT_EQ(result.word_count, 2);
  rdna4::Vop3MachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.vdst, 10);
  EXPECT_EQ(dst.src0, 100);
  EXPECT_EQ(dst.src1, 200);
  EXPECT_EQ(dst.src2, 50);
  EXPECT_EQ(dst.clamp, 1);
  EXPECT_EQ(dst.omod, 2);
  EXPECT_EQ(dst.neg, 5);
  EXPECT_EQ(dst.abs, 3);
}

TEST(EncodingTranslator, Cdna4ToCdna3Vop2VectorAddPreservesOperands) {
  cdna4::Vop2MachineInst src{};
  src.src0 = 3;
  src.vsrc1 = 4;
  src.vdst = 5;
  src.op = 1;       // V_ADD_F32 on CDNA3 and CDNA4.
  src.encoding = 0; // GFX9-family VOP2 prefix.
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3(kEnc_VOP2, w0, 0, 0, 1);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<cdna3::Vop2MachineInst>(result.words[0]);
  EXPECT_EQ(dst.src0, 3);
  EXPECT_EQ(dst.vsrc1, 4);
  EXPECT_EQ(dst.vdst, 5);
  EXPECT_EQ(dst.op, 1);
  EXPECT_EQ(dst.encoding, 0);
}

TEST(EncodingTranslator, UnknownEncodingReturnsEmpty) {
  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(0xFFFF, 0, 0, 0, 0);
  EXPECT_EQ(result.word_count, 0);
}

TEST(EncodingTranslator, DecodeEncodeRoundTrip) {
  cdna4::Sop1MachineInst src{};
  src.ssrc0 = 55;
  src.sdst = 33;
  src.op = 4;
  src.encoding = 0x17D;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto fields = cdna4_to_rdna4::decode_sop1_cdna4(w0);
  EXPECT_EQ(fields.ssrc0, 55u);
  EXPECT_EQ(fields.sdst, 33u);
  EXPECT_EQ(fields.op, 4u);

  auto result = cdna4_to_rdna4::encode_sop1_rdna4(fields, 4);
  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 55);
  EXPECT_EQ(dst.sdst, 33);
  EXPECT_EQ(dst.op, 4);
}

TEST(EncodingTranslator, Gfx1250ToRdna4PreservesSoppEncoding) {
  gfx1250::SoppMachineInst inst{};
  inst.simm16 = 0;
  inst.op = 1; // s_endpgm
  const uint32_t w0 = std::bit_cast<uint32_t>(inst);

  auto result = gfx1250_to_rdna4::translate_encoding_gfx1250_to_rdna4(kEnc_SOPP, w0, 0, 0, 1);
  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::SoppMachineInst>(result.words[0]);
  EXPECT_EQ(dst.simm16, 0);
  EXPECT_EQ(dst.op, 1);
}

TEST(EncodingTranslator, Gfx1250ToRdna4ClearsUnusedVop3Src2Bits) {
  struct Vop3Case {
    uint32_t w0;
    uint32_t w1;
    uint16_t op;
    uint32_t expected_w1;
  };

  constexpr std::array<Vop3Case, 3> cases = {{
      {0xD5160004u, 0x02000000u, 278u, 0x00000000u}, // v_max_num_f32 v4, s0, s0
      {0xD71C0007u, 0x02020F0Bu, 796u, 0x00020F0Bu}, // v_ldexp_f32 v7, v11, v7
      {0xD7600000u, 0x02013F04u, 864u, 0x00013F04u}, // v_readlane_b32 s0, v4, 31
  }};

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.op);
    const uint32_t exact_encoding_id = test_case.w0 >> 23;
    auto result = gfx1250_to_rdna4::translate_encoding_gfx1250_to_rdna4(
        exact_encoding_id, test_case.w0, test_case.w1, 0, test_case.op);
    ASSERT_EQ(result.word_count, 2);
    EXPECT_EQ(result.words[0], test_case.w0);
    EXPECT_EQ(result.words[1], test_case.expected_w1);

    rdna4::Vop3MachineInst dst{};
    std::memcpy(&dst, result.words, sizeof(dst));
    EXPECT_EQ(dst.src2, 0u);
    EXPECT_EQ(dst.neg & 0x4u, 0u);
    EXPECT_EQ(dst.abs & 0x4u, 0u);
    EXPECT_EQ(dst.opsel & 0x4u, 0u);
  }

  gfx1250::Vop3MachineInst cmp{};
  cmp.encoding = 0x35;
  cmp.op = 81; // v_cmp_lt_i64
  cmp.vdst = 2;
  cmp.src0 = 256 + 4;
  cmp.src1 = 256 + 8;
  cmp.src2 = scalar_positive_inline_u32(0);
  uint32_t cmp_words[2];
  std::memcpy(cmp_words, &cmp, sizeof(cmp));

  auto cmp_result = gfx1250_to_rdna4::translate_encoding_gfx1250_to_rdna4(
      cmp_words[0] >> 23, cmp_words[0], cmp_words[1], 0, cmp.op);
  ASSERT_EQ(cmp_result.word_count, 2);
  rdna4::Vop3MachineInst cmp_dst{};
  std::memcpy(&cmp_dst, cmp_result.words, sizeof(cmp_dst));
  EXPECT_EQ(cmp_dst.src2, 0u);
}

TEST(EncodingTranslator, Gfx1250ToRdna4PreservesFp8StochasticDstByteSelect) {
  for (const uint16_t op : {875u, 876u}) {
    SCOPED_TRACE(op);
    gfx1250::Vop3MachineInst inst{};
    inst.encoding = 0x35;
    inst.op = op;
    inst.vdst = 7;
    inst.src0 = 256 + 1;
    inst.src1 = 256 + 2;
    inst.src2 = scalar_positive_inline_u32(0);
    inst.opsel = 0x4;
    uint32_t words[2];
    std::memcpy(words, &inst, sizeof(inst));

    auto result = gfx1250_to_rdna4::translate_encoding_gfx1250_to_rdna4(words[0] >> 23, words[0],
                                                                        words[1], 0, inst.op);
    ASSERT_EQ(result.word_count, 2);
    rdna4::Vop3MachineInst dst{};
    std::memcpy(&dst, result.words, sizeof(dst));
    EXPECT_EQ(dst.src2, 0u);
    EXPECT_EQ(dst.opsel & 0x4u, 0x4u);
  }
}

TEST(EncodingTranslator, Gfx1250ToRdna4PreservesUsedVop3Src2Bits) {
  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 521;
  inst.vdst = 7;
  inst.src0 = 256 + 1;
  inst.src1 = 256 + 2;
  inst.src2 = 256 + 3;
  inst.abs = 0x7;
  inst.opsel = 0xF;
  inst.neg = 0x7;
  uint32_t words[2];
  std::memcpy(words, &inst, sizeof(inst));

  auto result = gfx1250_to_rdna4::translate_encoding_gfx1250_to_rdna4(words[0] >> 23, words[0],
                                                                      words[1], 0, inst.op);
  ASSERT_EQ(result.word_count, 2);
  rdna4::Vop3MachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.src2, 256u + 3u);
  EXPECT_EQ(dst.abs, 0x7u);
  EXPECT_EQ(dst.opsel, 0xFu);
  EXPECT_EQ(dst.neg, 0x7u);
}

TEST(EncodingTranslator, Gfx1250ToRdna4ClearsUnusedVop3SdstSrc2Bits) {
  gfx1250::Vop3SdstEncMachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 768;
  inst.vdst = 5;
  inst.sdst = 12;
  inst.src0 = 256 + 4;
  inst.src1 = 256 + 8;
  inst.src2 = scalar_positive_inline_u32(0);
  inst.neg = 0x7;
  uint32_t words[2];
  std::memcpy(words, &inst, sizeof(inst));

  auto result = gfx1250_to_rdna4::translate_encoding_gfx1250_to_rdna4(words[0] >> 23, words[0],
                                                                      words[1], 0, inst.op);
  ASSERT_EQ(result.word_count, 2);
  rdna4::Vop3SdstEncMachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.src2, 0u);
  EXPECT_EQ(dst.neg, 0x3u);

  inst.op = 766;
  inst.src2 = 256 + 9;
  inst.neg = 0x7;
  std::memcpy(words, &inst, sizeof(inst));
  result = gfx1250_to_rdna4::translate_encoding_gfx1250_to_rdna4(words[0] >> 23, words[0], words[1],
                                                                 0, inst.op);
  ASSERT_EQ(result.word_count, 2);
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.src2, 256u + 9u);
  EXPECT_EQ(dst.neg, 0x7u);
}

TEST(LegalizationLookup, FindsKnownInstruction) {
  const auto *entry = lookup(kLegalization_cdna4_to_rdna4, 0, 0);
  EXPECT_NE(entry, nullptr);
  if (entry) {
    EXPECT_NE(entry->action, Action::Illegal);
  }
}

TEST(LegalizationLookup, ReturnsNullForUnknown) {
  const auto *entry = lookup(kLegalization_cdna4_to_rdna4, 0xFFFF, 0xFFFF);
  EXPECT_EQ(entry, nullptr);
}

TEST(LegalizationTable, NoIllegalEntries_Cdna4ToRdna4) {
  for (const auto &e : kLegalization_cdna4_to_rdna4) {
    EXPECT_NE(e.action, Action::Illegal)
        << "ILLEGAL at encoding_id=" << e.src_encoding_id << " opcode=" << e.src_opcode;
  }
}

#define CHECK_NO_ILLEGAL(pair)                                                                     \
  TEST(LegalizationTable, NoIllegalEntries_##pair) {                                               \
    for (const auto &e : kLegalization_##pair) {                                                   \
      EXPECT_NE(e.action, Action::Illegal)                                                         \
          << "ILLEGAL at encoding_id=" << e.src_encoding_id << " opcode=" << e.src_opcode;         \
    }                                                                                              \
    EXPECT_GT(std::size(kLegalization_##pair), 0u) << "table is empty";                            \
  }

CHECK_NO_ILLEGAL(cdna1_to_cdna2)
CHECK_NO_ILLEGAL(cdna1_to_cdna3)
CHECK_NO_ILLEGAL(cdna1_to_cdna4)
CHECK_NO_ILLEGAL(cdna1_to_rdna1)
CHECK_NO_ILLEGAL(cdna1_to_rdna2)
CHECK_NO_ILLEGAL(cdna1_to_rdna3)
CHECK_NO_ILLEGAL(cdna1_to_rdna4)
CHECK_NO_ILLEGAL(cdna2_to_cdna3)
CHECK_NO_ILLEGAL(cdna2_to_cdna4)
CHECK_NO_ILLEGAL(cdna2_to_rdna3)
CHECK_NO_ILLEGAL(cdna2_to_rdna4)
CHECK_NO_ILLEGAL(cdna3_to_cdna4)
CHECK_NO_ILLEGAL(cdna3_to_rdna3)
CHECK_NO_ILLEGAL(cdna3_to_rdna4)
CHECK_NO_ILLEGAL(cdna4_to_cdna3)
CHECK_NO_ILLEGAL(cdna4_to_rdna3)
CHECK_NO_ILLEGAL(gfx1250_to_rdna4)
CHECK_NO_ILLEGAL(rdna1_to_cdna3)
CHECK_NO_ILLEGAL(rdna1_to_cdna4)
CHECK_NO_ILLEGAL(rdna1_to_rdna2)
CHECK_NO_ILLEGAL(rdna1_to_rdna3)
CHECK_NO_ILLEGAL(rdna1_to_rdna4)
CHECK_NO_ILLEGAL(rdna2_to_rdna3)
CHECK_NO_ILLEGAL(rdna2_to_rdna4)
CHECK_NO_ILLEGAL(rdna3_5_to_rdna4)
CHECK_NO_ILLEGAL(rdna3_to_cdna4)
CHECK_NO_ILLEGAL(rdna3_to_rdna4)
CHECK_NO_ILLEGAL(rdna4_to_cdna4)

#undef CHECK_NO_ILLEGAL

TEST(SemanticRules, HandwrittenRulesAreSortedForBinarySearch) {
  EXPECT_TRUE(std::ranges::is_sorted(semantic_expand_rules_cdna4_to_rdna4()));
  EXPECT_TRUE(std::ranges::is_sorted(semantic_expand_rules_cdna4_to_rdna3()));
  EXPECT_TRUE(std::ranges::is_sorted(semantic_expand_rules_gfx1250_to_rdna4()));
}

TEST(CodeObjectPatcher, ElfFeatureFlagsAreNormalizedForTargetArchitecture) {
  constexpr uint32_t kAnyXnackAndSramEcc = 0x500;
  EXPECT_EQ(elf_flags_for_target(EF_AMDGPU_MACH_AMDGCN_GFX1250 | kAnyXnackAndSramEcc,
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250),
            EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  auto *source_header = reinterpret_cast<Elf64_Ehdr *>(image.data());
  source_header->e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250 | kAnyXnackAndSramEcc;

  AmdGpuCodeObject rdna_source(image.data(), image.size());
  ASSERT_TRUE(rdna_source.is_valid());
  CodeObjectPatcher rdna_patcher(rdna_source);
  rdna_patcher.update_elf_flags(EF_AMDGPU_MACH_AMDGCN_GFX1201);
  const std::vector<uint8_t> rdna_bytes = rdna_patcher.emit();
  const auto *rdna_header = reinterpret_cast<const Elf64_Ehdr *>(rdna_bytes.data());
  EXPECT_EQ(rdna_header->e_flags, EF_AMDGPU_MACH_AMDGCN_GFX1201);

  source_header->e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950 | kAnyXnackAndSramEcc;
  AmdGpuCodeObject cdna_source(image.data(), image.size());
  ASSERT_TRUE(cdna_source.is_valid());
  CodeObjectPatcher cdna_patcher(cdna_source);
  cdna_patcher.update_elf_flags(EF_AMDGPU_MACH_AMDGCN_GFX942);
  const std::vector<uint8_t> cdna_bytes = cdna_patcher.emit();
  const auto *cdna_header = reinterpret_cast<const Elf64_Ehdr *>(cdna_bytes.data());
  EXPECT_EQ(cdna_header->e_flags, EF_AMDGPU_MACH_AMDGCN_GFX942 | kAnyXnackAndSramEcc);
}

TEST(CodeObjectPatcher, CaveBodyMaterializesInRjTranslationsAfterText) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::array<uint32_t, 2> cave_words = {0xDEADBEEFu, 0xCAFEBABEu};
  patcher.append_cave_body(cave_words);
  ASSERT_TRUE(patcher.append_cave_section());

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());

  const Section *text = patched.text_sections()[0];
  ASSERT_NE(text, nullptr);
  EXPECT_EQ(text->size(), 8u) << ".text must keep its original size";

  const Section *translations = find_section(patched, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  EXPECT_EQ(translations->sectionOffset(), text->sectionOffset() + text->size())
      << ".rj_translations must be physically placed immediately after .text";
  ASSERT_EQ(translations->size(), cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(translations->data(), cave_words.data(), translations->size()), 0);

  const Section *rodata = find_section(patched, ".rodata");
  ASSERT_NE(rodata, nullptr);
  EXPECT_EQ(rodata->sectionOffset(), translations->sectionOffset() + translations->size())
      << "sections following .text must be shifted after the cave section";
  uint32_t rodata_word = 0;
  std::memcpy(&rodata_word, rodata->data(), sizeof(rodata_word));
  EXPECT_EQ(rodata_word, 0xA5A55A5Au);
}

TEST(CodeObjectPatcher, CaveInsertionPreservesLoadSegmentAlignment) {
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t padded_file_delta = 2 * load_align;

  auto image = make_minimal_amdgpu_elf_with_load_segments();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::vector<uint32_t> cave_words(load_align / sizeof(uint32_t) + 1, 0xDEADBEEFu);
  patcher.append_cave_body(cave_words);
  ASSERT_TRUE(patcher.append_cave_section());

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());

  const Section *text = patched.text_sections()[0];
  const Section *translations = find_section(patched, ".rj_translations");
  const Section *rodata = find_section(patched, ".rodata");
  ASSERT_NE(translations, nullptr);
  ASSERT_NE(rodata, nullptr);
  EXPECT_EQ(translations->sectionOffset(), text->sectionOffset() + text->size());
  EXPECT_EQ(translations->size(), cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(translations->vaddr(), text->vaddr() + text->size());

  EXPECT_EQ(rodata->sectionOffset(), text->sectionOffset() + text->size() + padded_file_delta)
      << "file padding should preserve later PT_LOAD p_offset/p_vaddr congruence";
  EXPECT_EQ(rodata->vaddr(), text->vaddr() + text->size() + load_align + padded_file_delta)
      << "later allocated sections must move after the expanded RX LOAD segment";

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(patched_bytes.data());
  ASSERT_EQ(ehdr->e_phnum, 2u);
  const auto *phdrs = reinterpret_cast<const Elf64_Phdr *>(patched_bytes.data() + ehdr->e_phoff);
  const auto *shdrs = reinterpret_cast<const Elf64_Shdr *>(patched_bytes.data() + ehdr->e_shoff);
  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    ASSERT_NE(phdrs[i].p_align, 0u);
    EXPECT_EQ(phdrs[i].p_offset % phdrs[i].p_align, phdrs[i].p_vaddr % phdrs[i].p_align)
        << "PT_LOAD " << i << " must remain loader-congruent";
  }
  EXPECT_EQ(phdrs[0].p_filesz, text->size() + padded_file_delta);
  EXPECT_EQ(phdrs[0].p_memsz, text->size() + padded_file_delta);
  EXPECT_EQ(phdrs[1].p_offset, rodata->sectionOffset());
  EXPECT_EQ(phdrs[1].p_vaddr, rodata->vaddr());
  EXPECT_EQ(phdrs[1].p_paddr, rodata->vaddr());
  EXPECT_LE(phdrs[0].p_vaddr + phdrs[0].p_memsz, phdrs[1].p_vaddr)
      << "expanded RX LOAD must not overlap the following LOAD in virtual memory";

  const auto *symtab = std::find_if(shdrs, shdrs + ehdr->e_shnum, [](const Elf64_Shdr &shdr) {
    return shdr.sh_type == SHT_SYMTAB;
  });
  ASSERT_NE(symtab, shdrs + ehdr->e_shnum);
  ASSERT_EQ(symtab->sh_entsize, sizeof(Elf64_Sym));
  ASSERT_GE(symtab->sh_size / symtab->sh_entsize, 3u);
  const auto *symbols =
      reinterpret_cast<const Elf64_Sym *>(patched_bytes.data() + symtab->sh_offset);
  EXPECT_EQ(symbols[1].st_value, rodata->vaddr())
      << "defined symbols in moved sections must track the section virtual address";
  EXPECT_EQ(symbols[2].st_value, text->vaddr())
      << "symbols in unmoved .text must keep their original virtual address";
}

TEST(CodeObjectPatcher, CaveInsertionPreservesMovedKernelDescriptorEntryAddress) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t padded_file_delta = 2 * load_align;

  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());
  const auto *original_rodata = find_section(co, ".rodata");
  ASSERT_NE(original_rodata, nullptr);
  KD original_kd{};
  std::memcpy(&original_kd, original_rodata->data(), sizeof(original_kd));

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::vector<uint32_t> cave_words(load_align / sizeof(uint32_t) + 1, 0xDEADBEEFu);
  patcher.append_cave_body(cave_words);
  ASSERT_TRUE(patcher.append_cave_section());

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *patched_text = patched.text_sections()[0];
  const auto *patched_rodata = find_section(patched, ".rodata");
  ASSERT_NE(patched_rodata, nullptr);

  KD patched_kd{};
  std::memcpy(&patched_kd, patched_rodata->data(), sizeof(patched_kd));
  EXPECT_EQ(patched_rodata->vaddr(), original_rodata->vaddr() + padded_file_delta);
  EXPECT_EQ(static_cast<uint64_t>(static_cast<int64_t>(patched_rodata->vaddr()) +
                                  patched_kd.kernel_code_entry_byte_offset),
            patched_text->vaddr())
      << "KERNEL_CODE_ENTRY_BYTE_OFFSET is relative to the descriptor address";
  EXPECT_EQ(patched_kd.kernel_code_entry_byte_offset,
            original_kd.kernel_code_entry_byte_offset - static_cast<int64_t>(padded_file_delta));
}

TEST(CodeObjectPatcher, CaveInsertionUpdatesRelocationOffsetsIntoMovedSections) {
  constexpr uint64_t load_align = 0x1000;

  auto image = make_minimal_amdgpu_elf_with_relocation_after_text();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::vector<uint32_t> cave_words(load_align / sizeof(uint32_t) + 1, 0xDEADBEEFu);
  patcher.append_cave_body(cave_words);
  ASSERT_TRUE(patcher.append_cave_section());

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());

  const auto *data = find_section(patched, ".data");
  const auto *rela_dyn = find_section(patched, ".rela.dyn");
  ASSERT_NE(data, nullptr);
  ASSERT_NE(rela_dyn, nullptr);
  ASSERT_EQ(rela_dyn->size(), sizeof(Elf64_Rela));

  Elf64_Rela rela{};
  std::memcpy(&rela, rela_dyn->data(), sizeof(rela));
  EXPECT_EQ(rela.r_offset, data->vaddr())
      << "ET_DYN relocation r_offset is the relocated storage address";
}

TEST(KernelDescriptorTranslator, Gfx1250IsWave32EvenWhenDescriptorBitClear) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *desc = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);
  AMDHSA_BITS_SET(desc->kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                  0);

  KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250);
  const auto translations = translator.translate_image(image, shdrs[1].sh_offset, shdrs[1].sh_size,
                                                       KernelDescriptorTranslationOptions{});

  ASSERT_EQ(translations.size(), 1u);
  EXPECT_EQ(translations[0].guest_wavefront_size, 32);
  EXPECT_EQ(translations[0].host_wavefront_size, 32);
  EXPECT_EQ(translations[0].target_wave_size, 32);
  EXPECT_TRUE(translations[0].supported);
}

TEST(KernelDescriptorTranslator, Gfx1250Supports1024AddressableVgprs) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *desc = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);

  KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250);
  const auto translations = translator.translate_image(image, shdrs[1].sh_offset, shdrs[1].sh_size,
                                                       KernelDescriptorTranslationOptions{});

  ASSERT_EQ(translations.size(), 1u);
  EXPECT_EQ(translations[0].guest_vgpr_count, 1024u);
  EXPECT_EQ(translations[0].host_vgpr_count, 1024u);
  EXPECT_EQ(translations[0].target_vgpr_count, 1024u);
  EXPECT_EQ(translations[0].target_vgpr_granulated, 63u);
  EXPECT_TRUE(translations[0].supported);
}

TEST(KernelDescriptorTranslator, RecomputedDescriptorPreservesDiscoveredSymbol) {
  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);

  KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4);
  const auto discovered = translator.translate_image(image, shdrs[1].sh_offset, shdrs[1].sh_size,
                                                     KernelDescriptorTranslationOptions{});
  ASSERT_EQ(discovered.size(), 1u);
  ASSERT_FALSE(discovered[0].symbol_name.empty());

  KernelDescriptorTranslationOptions updated_options;
  updated_options.minimum_vgprs = discovered[0].target_vgpr_count + 4;
  const auto recomputed = translator.translate_descriptor(
      image, discovered[0].descriptor_file_offset, discovered[0].entry_text_offset, updated_options,
      discovered[0].symbol_name);

  ASSERT_TRUE(recomputed.has_value());
  EXPECT_EQ(recomputed->symbol_name, discovered[0].symbol_name);
}

TEST(CodeObjectPatcher, Gfx1250PatchSetsWave32DescriptorBit) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250);
  const auto translations = translator.translate_image(
      image, co.text_sections()[0]->sectionOffset(), co.text_sections()[0]->size(),
      KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translations.size(), 1u);

  CodeObjectPatcher patcher(co);
  ASSERT_TRUE(
      patcher.apply_kernel_descriptor_translation(translations[0], ROCJITSU_CODE_ARCH_GFX1250));

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());

  const Section *rodata = find_section(patched, ".rodata");
  ASSERT_NE(rodata, nullptr);
  KD patched_desc{};
  std::memcpy(&patched_desc, rodata->data(), sizeof(patched_desc));
  EXPECT_EQ(AMDHSA_BITS_GET(patched_desc.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
            1u);
}

TEST(CodeObjectPatcher, PrologueLongBranchAvoidsRdna4GridCaptureSgprs) {
  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());
  const Section *rodata = find_section(co, ".rodata");
  ASSERT_NE(rodata, nullptr);

  KdTranslation translation;
  translation.descriptor_file_offset = rodata->sectionOffset();
  translation.entry_text_offset = 0;
  translation.target_source_sgpr_count = 71;
  translation.rdna4_grid_x_sgpr = 71;
  translation.rdna4_grid_yz_sgpr = 72;
  translation.target_sgpr_count = 82;
  translation.prologue_words = {build_s_mov_b32(71, 108 + 9, ROCJITSU_CODE_ARCH_RDNA4),
                                build_s_mov_b32(72, 108 + 7, ROCJITSU_CODE_ARCH_RDNA4)};

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(200000);
  ASSERT_TRUE(patcher.apply_kernel_descriptor_translation(translation, ROCJITSU_CODE_ARCH_RDNA4));

  const auto cave = patcher.cave_body();
  ASSERT_EQ(cave.size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(cave.size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), cave.data(), cave.size());

  EXPECT_NE(std::ranges::find(cave_words, pack_sop1(71, 74, 0)), cave_words.end())
      << "the prologue return branch must not clobber captured grid payload SGPRs";
  EXPECT_EQ(std::ranges::find(cave_words, pack_sop1(71, 72, 0)), cave_words.end())
      << "s72 carries the captured ttmp7 payload until original entry code consumes it";
}

TEST(CodeObjectPatcher, MetadataPrivateSegmentPatchMatchesKernelSymbol) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();

  static constexpr std::array<uint8_t, 28> kPrivateSizeKey = {
      0xBB, '.', 'p', 'r', 'i', 'v', 'a', 't', 'e', '_', 's', 'e', 'g', 'm',
      'e',  'n', 't', '_', 'f', 'i', 'x', 'e', 'd', '_', 's', 'i', 'z', 'e'};
  static constexpr std::array<uint8_t, 8> kSymbolKey = {0xA7, '.', 's', 'y', 'm', 'b', 'o', 'l'};

  const auto append_msgpack_string = [](std::vector<uint8_t> &bytes, std::string_view value) {
    ASSERT_LE(value.size(), 31u);
    bytes.push_back(static_cast<uint8_t>(0xA0u | value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  };

  std::vector<uint8_t> metadata;
  std::vector<size_t> private_value_offsets;
  const auto append_kernel_metadata = [&](std::string_view symbol) {
    metadata.insert(metadata.end(), kPrivateSizeKey.begin(), kPrivateSizeKey.end());
    private_value_offsets.push_back(image.size() + metadata.size());
    metadata.push_back(0);
    metadata.insert(metadata.end(), kSymbolKey.begin(), kSymbolKey.end());
    append_msgpack_string(metadata, symbol);
  };
  append_kernel_metadata("first.kd");
  append_kernel_metadata("target.kd");
  append_kernel_metadata("third.kd");
  image.insert(image.end(), metadata.begin(), metadata.end());

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  KdTranslation translation;
  translation.symbol_name = "target.kd";
  translation.private_spill_zone_base = 0;
  translation.private_spill_zone_bytes = 20;
  translation.target_private_size = 20;

  CodeObjectPatcher patcher(co);
  ASSERT_TRUE(patcher.patch_metadata_private_segment_fixed_sizes({&translation, 1}));

  const auto patched = patcher.emit();
  ASSERT_EQ(private_value_offsets.size(), 3u);
  EXPECT_EQ(patched[private_value_offsets[0]], 0u);
  EXPECT_EQ(patched[private_value_offsets[1]], 20u);
  EXPECT_EQ(patched[private_value_offsets[2]], 0u);
}

TEST(CodeObjectPatcher, MetadataPrivateSegmentResizeRefreshesTextOffset) {
  static constexpr std::array<uint8_t, 28> kPrivateSizeKey = {
      0xBB, '.', 'p', 'r', 'i', 'v', 'a', 't', 'e', '_', 's', 'e', 'g', 'm',
      'e',  'n', 't', '_', 'f', 'i', 'x', 'e', 'd', '_', 's', 'i', 'z', 'e'};
  static constexpr std::array<uint8_t, 8> kSymbolKey = {0xA7, '.', 's', 'y', 'm', 'b', 'o', 'l'};

  const auto append_msgpack_string = [](std::vector<uint8_t> &bytes, std::string_view value) {
    ASSERT_LE(value.size(), 31u);
    bytes.push_back(static_cast<uint8_t>(0xA0u | value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  };

  std::vector<uint8_t> metadata;
  metadata.insert(metadata.end(), kPrivateSizeKey.begin(), kPrivateSizeKey.end());
  metadata.push_back(0);
  metadata.insert(metadata.end(), kSymbolKey.begin(), kSymbolKey.end());
  append_msgpack_string(metadata, "target.kd");

  std::vector<uint8_t> note;
  Elf64_Nhdr nhdr{};
  nhdr.n_namesz = 4;
  nhdr.n_descsz = static_cast<uint32_t>(metadata.size());
  nhdr.n_type = 32;
  note.resize(sizeof(nhdr));
  std::memcpy(note.data(), &nhdr, sizeof(nhdr));
  note.insert(note.end(), {'A', 'M', 'D', '\0'});
  note.insert(note.end(), metadata.begin(), metadata.end());
  note.resize(align_up_for_test(note.size(), 4), 0);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t note_name = add_elf_name(shstrtab, ".note");
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  constexpr uint64_t note_offset = 0x100;
  constexpr uint64_t text_alignment = 256;
  const uint64_t text_offset = align_up_for_test(note_offset + note.size(), text_alignment);
  constexpr std::array<uint32_t, 2> text_words = {0x11111111u, 0x22222222u};
  constexpr uint64_t text_size = text_words.size() * sizeof(uint32_t);
  const uint64_t shstrtab_offset = text_offset + text_size;
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 4;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = 2;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 3;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, 2> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_filesz = image.size();
  phdrs[0].p_memsz = image.size();
  phdrs[0].p_align = 0x1000;
  phdrs[1].p_type = PT_NOTE;
  phdrs[1].p_offset = note_offset;
  phdrs[1].p_vaddr = note_offset;
  phdrs[1].p_paddr = note_offset;
  phdrs[1].p_filesz = note.size();
  phdrs[1].p_memsz = note.size();
  phdrs[1].p_align = 4;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), sizeof(phdrs));

  std::memcpy(image.data() + note_offset, note.data(), note.size());
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = note_name;
  shdrs[1].sh_type = SHT_NOTE;
  shdrs[1].sh_flags = SHF_ALLOC;
  shdrs[1].sh_addr = note_offset;
  shdrs[1].sh_offset = note_offset;
  shdrs[1].sh_size = note.size();
  shdrs[1].sh_addralign = 4;

  shdrs[2].sh_name = text_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[2].sh_addr = text_offset;
  shdrs[2].sh_offset = text_offset;
  shdrs[2].sh_size = text_size;
  shdrs[2].sh_addralign = text_alignment;

  shdrs[3].sh_name = shstrtab_name;
  shdrs[3].sh_type = SHT_STRTAB;
  shdrs[3].sh_offset = shstrtab_offset;
  shdrs[3].sh_size = shstrtab.size();
  shdrs[3].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  CodeObjectPatcher patcher(co);
  ASSERT_EQ(patcher.text_offset(), text_offset);

  KdTranslation translation;
  translation.symbol_name = "target.kd";
  translation.private_spill_zone_base = 0;
  translation.private_spill_zone_bytes = 600;
  translation.target_private_size = 600;

  ASSERT_TRUE(patcher.patch_metadata_private_segment_fixed_sizes({&translation, 1}));
  EXPECT_EQ(patcher.text_offset(), text_offset + text_alignment)
      << "resizable note growth must preserve later allocated-section alignment and refresh the "
         "cached text offset";

  constexpr std::array<uint32_t, 2> new_text_words = {0xAAAAAAAAu, 0xBBBBBBBBu};
  patcher.overwrite_text({reinterpret_cast<const uint8_t *>(new_text_words.data()), text_size});

  const auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());
  const Section *text = patched.text_sections()[0];
  ASSERT_NE(text, nullptr);
  ASSERT_EQ(text->size(), text_size);
  EXPECT_EQ(text->sectionOffset(), text_offset + text_alignment);
  EXPECT_EQ(text->sectionOffset() % text_alignment, 0u);
  EXPECT_EQ(text->vaddr() % text_alignment, 0u);
  EXPECT_EQ(std::memcmp(text->data(), new_text_words.data(), text_size), 0)
      << "overwrite_text must target the shifted .text section, not the old file offset";

  const auto *patched_ehdr = reinterpret_cast<const Elf64_Ehdr *>(patched_bytes.data());
  const auto *patched_phdrs =
      reinterpret_cast<const Elf64_Phdr *>(patched_bytes.data() + patched_ehdr->e_phoff);
  EXPECT_EQ(patched_phdrs[1].p_type, PT_NOTE);
  EXPECT_EQ(patched_phdrs[1].p_filesz, note.size() + text_alignment);
  EXPECT_EQ(patched_phdrs[1].p_memsz, note.size() + text_alignment);
}

TEST(CodeObjectPatcher, MetadataTargetIsaPatchRewritesSameLengthTarget) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  const std::string target = "amdgcn-amd-amdhsa--gfx1250";
  image.insert(image.end(), target.begin(), target.end());

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  CodeObjectPatcher patcher(co);
  ASSERT_FALSE(patcher.patch_metadata_target_isa("gfx1250", "gfx12010"));
  ASSERT_TRUE(patcher.patch_metadata_target_isa("amdgcn-amd-amdhsa--gfx1250",
                                                "amdgcn-amd-amdhsa--gfx1201"));

  const auto patched = patcher.emit();
  const std::string patched_text(patched.begin(), patched.end());
  EXPECT_NE(patched_text.find("amdgcn-amd-amdhsa--gfx1201"), std::string::npos);
  EXPECT_EQ(patched_text.find("amdgcn-amd-amdhsa--gfx1250"), std::string::npos);
}

TEST(CodeObjectPatcher, MetadataVgprPatchDoesNotLowerExistingCounts) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  static constexpr std::array<uint8_t, 12> kVgprCountKey = {0xAB, '.', 'v', 'g', 'p', 'r',
                                                            '_',  'c', 'o', 'u', 'n', 't'};

  image.insert(image.end(), kVgprCountKey.begin(), kVgprCountKey.end());
  const size_t high_value_offset = image.size();
  image.push_back(0xCC);
  image.push_back(128);

  image.insert(image.end(), kVgprCountKey.begin(), kVgprCountKey.end());
  const size_t low_value_offset = image.size();
  image.push_back(24);

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  CodeObjectPatcher patcher(co);
  ASSERT_TRUE(patcher.patch_metadata_vgpr_count(127));

  const auto patched = patcher.emit();
  ASSERT_LT(high_value_offset + 1, patched.size());
  ASSERT_LT(low_value_offset, patched.size());
  EXPECT_EQ(patched[high_value_offset], 0xCCu);
  EXPECT_EQ(patched[high_value_offset + 1], 128u);
  EXPECT_EQ(patched[low_value_offset], 127u);
}

TEST(CodeObjectPatcher, MetadataSgprPatchRaisesExistingCounts) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  static constexpr std::array<uint8_t, 12> kSgprCountKey = {0xAB, '.', 's', 'g', 'p', 'r',
                                                            '_',  'c', 'o', 'u', 'n', 't'};

  image.insert(image.end(), kSgprCountKey.begin(), kSgprCountKey.end());
  const size_t high_value_offset = image.size();
  image.push_back(96);

  image.insert(image.end(), kSgprCountKey.begin(), kSgprCountKey.end());
  const size_t low_value_offset = image.size();
  image.push_back(78);

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  CodeObjectPatcher patcher(co);
  ASSERT_TRUE(patcher.patch_metadata_sgpr_count(88));

  const auto patched = patcher.emit();
  ASSERT_LT(high_value_offset, patched.size());
  ASSERT_LT(low_value_offset, patched.size());
  EXPECT_EQ(patched[high_value_offset], 96u);
  EXPECT_EQ(patched[low_value_offset], 88u);
}

TEST(BinaryTranslator, CaveBranchOverflowLeavesCodeObjectUnchanged) {
  auto image = make_large_amdgpu_elf_with_waitcnt_entry();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(co);

  EXPECT_EQ(result.elf_bytes, image);
  const bool diagnosed = std::any_of(
      result.diagnostics.begin(), result.diagnostics.end(),
      [](const TranslationDiagnostic &diagnostic) {
        return diagnostic.severity == DiagnosticSeverity::Error &&
               diagnostic.kind == DiagnosticKind::ResourceLimit &&
               diagnostic.message.find("branch range") != std::string::npos &&
               diagnostic.message.find("leaving code object unchanged") != std::string::npos;
      });
  EXPECT_TRUE(diagnosed);
}

TEST(BinaryTranslator, Gfx1250ScaledVglobalUsesCaveAndDropsClause) {
  constexpr uint32_t kGfx1250SClause1 = 0xBF850001u;
  constexpr uint32_t kGfx1250GlobalLoadB32ScaleW0 = 0xEE050004u;
  constexpr uint32_t kGfx1250GlobalLoadB32ScaleW1 = 0x00010001u;
  constexpr uint32_t kGfx1250GlobalLoadB32ScaleW2 = 0x00000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 5> text_words = {kGfx1250SClause1, kGfx1250GlobalLoadB32ScaleW0,
                                              kGfx1250GlobalLoadB32ScaleW1,
                                              kGfx1250GlobalLoadB32ScaleW2, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4))
      << "s_clause must not cover DBT branch stubs after VMEM is moved to a cave";
  EXPECT_EQ((patched_text[1] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base =
      find_vop2_for_test(cave_words, 24u, std::nullopt, scalar_positive_inline_u32(2), 0u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LT(*base + 5u, cave_words.size());

  const auto lshl = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base]);
  EXPECT_EQ(lshl.src0, scalar_positive_inline_u32(2));
  EXPECT_EQ(lshl.vsrc1, 0u);
  EXPECT_EQ(lshl.op, 24u);
  EXPECT_EQ(cave_words[*base + 1u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::VglobalMachineInst mem{};
  std::memcpy(&mem, cave_words.data() + *base + 2u, sizeof(mem));
  EXPECT_EQ(mem.op, 20u);
  EXPECT_EQ(mem.saddr, 4u);
  EXPECT_EQ(mem.vdst, 1u);
  EXPECT_EQ(mem.vaddr, lshl.vdst);
  EXPECT_EQ(cave_words[*base + 3u], 0x00000001u)
      << "gfx1250 scale_offset bit must be cleared for RDNA4";
  EXPECT_EQ((cave_words[*base + 5u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250RelativeVgprAccessKeepsSemanticScratchAboveSourceFile) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250GlobalLoadB32ScaleW0 = 0xEE050004u;
  constexpr uint32_t kGfx1250GlobalLoadB32ScaleW1 = 0x00010001u;
  constexpr uint32_t kGfx1250GlobalLoadB32ScaleW2 = 0x00000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop1MachineInst movrels{};
  movrels.encoding = 0x3F;
  movrels.op = 67;
  movrels.vdst = 20;
  movrels.src0 = 256 + 4;
  const std::array<uint32_t, 5> text_words = {
      kGfx1250GlobalLoadB32ScaleW0,
      kGfx1250GlobalLoadB32ScaleW1,
      kGfx1250GlobalLoadB32ScaleW2,
      std::bit_cast<uint32_t>(movrels),
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *desc = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 1);

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto shift =
      find_vop2_for_test(cave_words, 24u, std::nullopt, scalar_positive_inline_u32(2), 0u);
  ASSERT_TRUE(shift.has_value());
  const auto address = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*shift]);
  EXPECT_GE(address.vdst, 32u)
      << "M0-relative source arrays make every descriptor-backed source VGPR unsafe for scratch";

  const Section *rodata = find_section(translated, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(KD));
  const auto *target_desc = reinterpret_cast<const KD *>(rodata->data());
  const uint32_t target_granulated = AMDHSA_BITS_GET(
      target_desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_GE((target_granulated + 1u) * 8u, 33u);
}

TEST(BinaryTranslator, Gfx1250GlobalAtomicAddF64LowersToRdna4CasLoop) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 4> text_words = {
      0xEE154004u, // global_atomic_add_f64 v2, v[0:1], s[4:5] scope:SCOPE_DEV
      0x00080000u,
      0x00000002u,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  bool found_cas = false;
  bool found_add = false;
  bool found_exec_narrow = false;
  for (size_t i = 0; i < cave_words.size(); ++i) {
    if (i + 2u < cave_words.size()) {
      rdna4::VglobalMachineInst candidate{};
      std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
      found_cas |= candidate.encoding == 0xEEu && candidate.op == 0x42u && candidate.th == 1u;
    }
    const auto vop2 = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[i]);
    found_add |= vop2.op == 2u;
    const auto sop2 = std::bit_cast<rdna4::Sop2MachineInst>(cave_words[i]);
    found_exec_narrow |= sop2.op == 35u && sop2.sdst == 126u && sop2.ssrc0 == 126u;
  }
  EXPECT_TRUE(found_cas);
  EXPECT_TRUE(found_add);
  EXPECT_TRUE(found_exec_narrow);
}

TEST(BinaryTranslator, Gfx1250ScaledGlobalLoadU16UsesTwoByteAddresses) {
  // global_load_u16 v8, v3, s[6:7] scale_offset
  constexpr uint32_t kGfx1250GlobalLoadU16ScaleW0 = 0xEE048006u;
  constexpr uint32_t kGfx1250GlobalLoadU16ScaleW1 = 0x00010008u;
  constexpr uint32_t kGfx1250GlobalLoadU16ScaleW2 = 0x00000003u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {kGfx1250GlobalLoadU16ScaleW0,
                                              kGfx1250GlobalLoadU16ScaleW1,
                                              kGfx1250GlobalLoadU16ScaleW2, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base =
      find_vop2_for_test(cave_words, 24u, std::nullopt, scalar_positive_inline_u32(1), 3u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LT(*base + 4u, cave_words.size());

  const auto lshl = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base]);
  EXPECT_EQ(cave_words[*base + 1u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::VglobalMachineInst mem{};
  std::memcpy(&mem, cave_words.data() + *base + 2u, sizeof(mem));
  EXPECT_EQ(mem.op, 18u);
  EXPECT_EQ(mem.saddr, 6u);
  EXPECT_EQ(mem.vdst, 8u);
  EXPECT_EQ(mem.vaddr, lshl.vdst);
  EXPECT_EQ(cave_words[*base + 3u], 0x00000008u)
      << "gfx1250 scale_offset bit must be cleared for RDNA4";
}

TEST(BinaryTranslator, Gfx1250ScaledScratchLoadB32MaterializesByteAddress) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::VscratchMachineInst load{};
  load.saddr = 4;
  load.op = 20; // scratch_load_b32
  load.encoding = 0xED;
  load.vdst = 8;
  load.scale_offset = 1;
  load.sve = 1;
  load.vaddr = 3;

  std::array<uint32_t, 4> text_words{};
  std::memcpy(text_words.data(), &load, sizeof(load));
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base =
      find_vop2_for_test(cave_words, 24u, std::nullopt, scalar_positive_inline_u32(2), 3u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LT(*base + 4u, cave_words.size());

  const auto lshl = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base]);
  EXPECT_EQ(cave_words[*base + 1u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::VscratchMachineInst mem{};
  std::memcpy(&mem, cave_words.data() + *base + 2u, sizeof(mem));
  EXPECT_EQ(mem.op, 20u);
  EXPECT_EQ(mem.saddr, 4u);
  EXPECT_EQ(mem.vdst, 8u);
  EXPECT_EQ(mem.vaddr, lshl.vdst);
  EXPECT_EQ(cave_words[*base + 3u], 0x00020008u)
      << "gfx1250 scale_offset must be cleared while preserving the scratch vaddr-enable bit";
}

TEST(BinaryTranslator, Gfx1250ScaledSmemMaterializesByteOffset) {
  // s_load_b32 s14, s[4:5], s21 offset:0 scale_offset nv
  constexpr uint32_t kGfx1250SLoadB32W0 = 0xF4100382u;
  constexpr uint32_t kGfx1250SLoadB32W1 = 0x2B000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {kGfx1250SLoadB32W0, kGfx1250SLoadB32W1,
                                              kGfx1250SEndpgm};

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> decoded(decoder->decode(text_words.data()));
  ASSERT_NE(decoded, nullptr);
  EXPECT_EQ(decoded->encoding_id(), 0x1E8u);

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  constexpr uint8_t kOpSMulI32 = 44;
  const auto mul = std::ranges::find_if(cave_words, [](uint32_t word) {
    return (word >> 30u) == 2u && ((word >> 23u) & 0x7Fu) == kOpSMulI32;
  });
  ASSERT_NE(mul, cave_words.end());
  const size_t mul_index = static_cast<size_t>(std::distance(cave_words.begin(), mul));
  ASSERT_LT(mul_index + 2u, cave_words.size());
  const uint8_t scaled_soffset = static_cast<uint8_t>((*mul >> 16u) & 0x7Fu);
  EXPECT_EQ(*mul & 0xFFu, 21u);
  EXPECT_EQ((*mul >> 8u) & 0xFFu, scalar_positive_inline_u32(4));

  rdna4::SmemMachineInst mem{};
  std::memcpy(&mem, cave_words.data() + mul_index + 1u, sizeof(mem));
  EXPECT_EQ(mem.op, 0u);
  EXPECT_EQ(mem.sbase, 2u);
  EXPECT_EQ(mem.sdata, 14u);
  EXPECT_EQ(mem.soffset, scaled_soffset);
  EXPECT_EQ(mem.ioffset, 0u);
  EXPECT_EQ(mem.nv, 0u);
}

TEST(BinaryTranslator, Gfx1250DirectVglobalWaitsOnValuVgprBeforeMemory) {
  constexpr uint32_t kGfx1250GlobalLoadB32W0 = 0xEE050004u;
  constexpr uint32_t kGfx1250GlobalLoadB32W1 = 0x00000001u;
  constexpr uint32_t kGfx1250GlobalLoadB32W2 = 0x00000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {kGfx1250GlobalLoadB32W0, kGfx1250GlobalLoadB32W1,
                                              kGfx1250GlobalLoadB32W2, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto wait_it = std::ranges::find(
      cave_words, build_s_wait_alu(kWaitAluDepctrVaVdstVmVsrc0, ROCJITSU_CODE_ARCH_RDNA4));
  ASSERT_NE(wait_it, cave_words.end());
  const auto base = static_cast<size_t>(std::distance(cave_words.begin(), wait_it));
  ASSERT_LE(base + 4u, cave_words.size());

  rdna4::VglobalMachineInst mem{};
  std::memcpy(&mem, cave_words.data() + base + 1u, sizeof(mem));
  EXPECT_EQ(mem.op, 20u);
  EXPECT_EQ(mem.saddr, 4u);
  EXPECT_EQ(mem.vdst, 1u);
  EXPECT_EQ((cave_words[base + 4u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250DirectAdjacentValuConsumerGetsRdna4Delay) {
  constexpr uint32_t kGfx1250VCvtF32F16V68V16 = 0x7E881710u;
  constexpr uint32_t kGfx1250VAddF32V66V68V66 = 0x06848544u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {kGfx1250VCvtF32F16V68V16, kGfx1250VAddF32V66V68V66,
                                              kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const std::array<uint32_t, 2> expected = {
      build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4),
      kGfx1250VAddF32V66V68V66,
  };
  EXPECT_TRUE(find_word_sequence_for_test(cave_words, expected).has_value());
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyVdsWaitSkipsScalarBarrierGap) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint32_t kGfx1250VLshlRevB32V9By2 = 0x30121282u;
  constexpr uint32_t kGfx1250SBarrierSignalMinus1 = 0xBE804EC1u;
  constexpr uint32_t kGfx1250SBarrierWait = 0xBF94FFFFu;
  constexpr uint32_t kGfx1250DsStoreB32V9V66W0 = 0xD8340000u;
  constexpr uint32_t kGfx1250DsStoreB32V9V66W1 = 0x00004209u;
  constexpr uint64_t kTextSize = 0x21000;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250VLshlRevB32V9By2;
  text_words[1] = kGfx1250SBarrierSignalMinus1;
  text_words[2] = kGfx1250SBarrierWait;
  text_words[3] = kGfx1250DsStoreB32V9V66W0;
  text_words[4] = kGfx1250DsStoreB32V9V66W1;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());

  const auto cave_words = local_expanded_text_words_for_test(translated, kTextSize);
  ASSERT_FALSE(cave_words.empty());
  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(cave_words);
  std::optional<size_t> store_index;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::VdsMachineInst ds{};
    std::memcpy(&ds, copied_words.data() + i, sizeof(ds));
    if (ds.encoding == 0x36u && ds.op == 0x0Du && ds.addr == 9u && ds.data0 == 66u) {
      store_index = i;
      break;
    }
  }

  ASSERT_TRUE(store_index.has_value());
  ASSERT_GT(*store_index, 0u);
  EXPECT_EQ(copied_words[*store_index - 1u],
            build_s_wait_alu(kWaitAluDepctrVaVdstVmVsrc0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyVbufferStoreWaitSkipsVectorCompareGap) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint32_t kGfx1250VAddF32V12V13V12 = 0x0618190Du;
  constexpr uint32_t kGfx1250VLshlRevB32V14By2 = 0x301C1C82u;
  constexpr uint32_t kGfx1250SMovB32S8Imm16 = 0xBE880090u;
  constexpr std::array<uint32_t, 2> kGfx1250VCmpLtU32VccV67S8 = {0xD449006Au, 0x02001143u};
  constexpr uint32_t kGfx1250SMovB32ExecVcc = 0xBEFE006Au;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::VbufferMachineInst store{};
  store.soffset = 0x7C;
  store.op = 26; // buffer_store_b32
  store.encoding = 0x31;
  store.vdata = 12;
  store.rsrc = 40;
  store.offen = 1;
  store.vaddr = 14;
  std::array<uint32_t, 3> store_words{};
  std::memcpy(store_words.data(), &store, sizeof(store));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250VAddF32V12V13V12;
  text_words[1] = kGfx1250VLshlRevB32V14By2;
  text_words[2] = kGfx1250SMovB32S8Imm16;
  text_words[3] = kGfx1250VCmpLtU32VccV67S8[0];
  text_words[4] = kGfx1250VCmpLtU32VccV67S8[1];
  text_words[5] = kGfx1250SMovB32ExecVcc;
  text_words[6] = store_words[0];
  text_words[7] = store_words[1];
  text_words[8] = store_words[2];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());

  const auto copied_words = local_expanded_text_words_for_test(translated, kTextSize);
  ASSERT_FALSE(copied_words.empty());
  std::optional<size_t> store_index;
  for (size_t i = 0; i + 2u < copied_words.size(); ++i) {
    rdna4::VbufferMachineInst store{};
    std::memcpy(&store, copied_words.data() + i, sizeof(store));
    if (store.encoding == 0x31u && store.op == 26u && store.vdata == 12u && store.vaddr == 14u &&
        store.rsrc == 40u && store.offen == 1u) {
      store_index = i;
      break;
    }
  }

  ASSERT_TRUE(store_index.has_value());
  ASSERT_GT(*store_index, 0u);
  EXPECT_EQ(copied_words[*store_index - 1u],
            build_s_wait_alu(kWaitAluDepctrVaVdstVmVsrc0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250GlobalLoadTr4B64ExpandsThroughOrdinaryLoadAndBpermute) {
  constexpr uint32_t kGfx1250GlobalLoadTr4B64ScaleW0 = 0xEE1CC004u;
  constexpr uint32_t kGfx1250GlobalLoadTr4B64ScaleW1 = 0x00010016u;
  constexpr uint32_t kGfx1250GlobalLoadTr4B64ScaleW2 = 0x00000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {kGfx1250GlobalLoadTr4B64ScaleW0,
                                              kGfx1250GlobalLoadTr4B64ScaleW1,
                                              kGfx1250GlobalLoadTr4B64ScaleW2, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_GT(translations->size(), 32u * sizeof(uint32_t));

  const auto words =
      std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(translations->data()),
                                translations->size() / sizeof(uint32_t));
  size_t rdna4_global_load_b64 = 0;
  size_t gfx1250_global_load_tr4 = 0;
  size_t ds_bpermute = 0;
  for (size_t i = 0; i < words.size(); ++i) {
    const uint32_t word = words[i];
    if (((word >> 24) & 0xFFu) == 0xEEu) {
      const uint32_t op = (word >> 14) & 0xFFu;
      rdna4_global_load_b64 += op == 21u;
      gfx1250_global_load_tr4 += op == 115u;
    }
    if (((word >> 26) & 0x3Fu) == 0x36u && ((word >> 18) & 0xFFu) == 0xB3u)
      ++ds_bpermute;
  }

  EXPECT_EQ(rdna4_global_load_b64, 1u);
  EXPECT_EQ(gfx1250_global_load_tr4, 0u);
  EXPECT_EQ(ds_bpermute, 32u);
}

TEST(BinaryTranslator, Gfx1250GlobalLoadTr6B96ExpandsThroughOrdinaryLoadAndBpermute) {
  constexpr uint32_t kGfx1250GlobalLoadTr6B96W0 = 0xEE1D0006u;
  constexpr uint32_t kGfx1250GlobalLoadTr6B96W1 = 0x0000000Eu;
  constexpr uint32_t kGfx1250GlobalLoadTr6B96W2 = 0x00000001u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {kGfx1250GlobalLoadTr6B96W0,
                                              kGfx1250GlobalLoadTr6B96W1,
                                              kGfx1250GlobalLoadTr6B96W2, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_GT(translations->size(), 48u * sizeof(uint32_t));

  const auto words =
      std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(translations->data()),
                                translations->size() / sizeof(uint32_t));
  size_t rdna4_global_load_b96 = 0;
  size_t gfx1250_global_load_tr6 = 0;
  size_t ds_bpermute = 0;
  for (uint32_t word : words) {
    if (((word >> 24) & 0xFFu) == 0xEEu) {
      const uint32_t op = (word >> 14) & 0xFFu;
      rdna4_global_load_b96 += op == 22u;
      gfx1250_global_load_tr6 += op == 116u;
    }
    if (((word >> 26) & 0x3Fu) == 0x36u && ((word >> 18) & 0xFFu) == 0xB3u)
      ++ds_bpermute;
  }

  EXPECT_EQ(rdna4_global_load_b96, 1u);
  EXPECT_EQ(gfx1250_global_load_tr6, 0u);
  EXPECT_EQ(ds_bpermute, 48u);
}

TEST(BinaryTranslator, Gfx1250TensorDmaLowersThroughGlobalAndDsOps) {
  constexpr uint32_t kGfx1250TensorLoadW0 = 0xD0710001u;
  constexpr uint32_t kGfx1250TensorLoadW1 = 0x7C000000u;
  constexpr uint32_t kGfx1250TensorLoadW2 = 0x7C7C141Cu;
  constexpr uint32_t kGfx1250SWaitTensorcnt0 = 0xBFCB0000u;
  constexpr uint32_t kGfx1250TensorStoreW0 = 0xD0714001u;
  constexpr uint32_t kGfx1250TensorStoreW1 = 0x7C000000u;
  constexpr uint32_t kGfx1250TensorStoreW2 = 0x7C7C1408u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 9> text_words = {
      kGfx1250TensorLoadW0,    kGfx1250TensorLoadW1,    kGfx1250TensorLoadW2,
      kGfx1250SWaitTensorcnt0, kGfx1250TensorStoreW0,   kGfx1250TensorStoreW1,
      kGfx1250TensorStoreW2,   kGfx1250SWaitTensorcnt0, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ((patched_text[4] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[5], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[6], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[7], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[8], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  size_t global_load_b32 = 0;
  size_t global_store_b32 = 0;
  size_t ds_load_b32 = 0;
  size_t ds_store_b32 = 0;
  size_t tensor_dma = 0;
  std::optional<uint32_t> current_exec_mask;
  std::vector<uint32_t> global_store_exec_masks;
  for (size_t i = 0; i < cave_words.size(); ++i) {
    const uint32_t word = cave_words[i];
    if (auto mask = exec_lo_literal_at_for_test(cave_words, i))
      current_exec_mask = *mask;
    if (((word >> 24) & 0xFFu) == 0xEEu) {
      const uint32_t op = (word >> 14) & 0xFFu;
      global_load_b32 += op == 20u;
      if (op == 26u) {
        ++global_store_b32;
        if (current_exec_mask)
          global_store_exec_masks.push_back(*current_exec_mask);
      }
    }
    if (i + 1 < cave_words.size()) {
      rdna4::VdsMachineInst ds{};
      const uint32_t pair[] = {cave_words[i], cave_words[i + 1]};
      std::memcpy(&ds, pair, sizeof(ds));
      if (ds.encoding == 0x36u) {
        ds_store_b32 += ds.op == 0x0Du;
        ds_load_b32 += ds.op == 0x36u;
      }
    }
    if ((word >> 23) == 0x1A0u) {
      const uint32_t op = (word >> 14) & 0xFFu;
      tensor_dma += op == 196u || op == 197u;
    }
  }

  EXPECT_GE(global_load_b32, 3u);
  EXPECT_GE(global_store_b32, 3u);
  EXPECT_GE(ds_load_b32, 3u);
  EXPECT_GE(ds_store_b32, 4u);
  EXPECT_EQ(tensor_dma, 0u);
  ASSERT_EQ(global_store_exec_masks.size(), global_store_b32);
  EXPECT_EQ(static_cast<size_t>(std::count(global_store_exec_masks.begin(),
                                           global_store_exec_masks.end(), 0x0000FFFFu)),
            global_store_b32);
  EXPECT_EQ(static_cast<size_t>(std::count(global_store_exec_masks.begin(),
                                           global_store_exec_masks.end(), 0x0000000Fu)),
            0u);
}

TEST(BinaryTranslator, Gfx1250GlobalLoadTr8AndTr16UseNativeRdna4TransposeOps) {
  constexpr uint32_t kGfx1250GlobalLoadTr8B64ScaleW0 = 0xEE160008u;
  constexpr uint32_t kGfx1250GlobalLoadTr8B64ScaleW1 = 0x00010018u;
  constexpr uint32_t kGfx1250GlobalLoadTr8B64ScaleW2 = 0x00000000u;
  constexpr uint32_t kGfx1250GlobalLoadTr16B128ScaleW0 = 0xEE15C00Au;
  constexpr uint32_t kGfx1250GlobalLoadTr16B128ScaleW1 = 0x00010006u;
  constexpr uint32_t kGfx1250GlobalLoadTr16B128ScaleW2 = 0x00000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 7> text_words = {kGfx1250GlobalLoadTr8B64ScaleW0,
                                              kGfx1250GlobalLoadTr8B64ScaleW1,
                                              kGfx1250GlobalLoadTr8B64ScaleW2,
                                              kGfx1250GlobalLoadTr16B128ScaleW0,
                                              kGfx1250GlobalLoadTr16B128ScaleW1,
                                              kGfx1250GlobalLoadTr16B128ScaleW2,
                                              kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_FALSE(translated.text_sections().empty());
  const Section *code = translations ? translations : translated.text_sections()[0];

  const auto words = std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(code->data()),
                                               code->size() / sizeof(uint32_t));
  size_t rdna4_tr8 = 0;
  size_t rdna4_tr16 = 0;
  for (uint32_t word : words) {
    if (((word >> 24) & 0xFFu) != 0xEEu)
      continue;
    const uint32_t op = (word >> 14) & 0xFFu;
    rdna4_tr16 += op == 87u;
    rdna4_tr8 += op == 88u;
  }

  EXPECT_EQ(rdna4_tr8, 1u);
  EXPECT_EQ(rdna4_tr16, 1u);
}

TEST(BinaryTranslator, Gfx1250DsTransposeLoadsExpandThroughDirectAndOrdinaryLoads) {
  auto ds_load_tr = [](uint8_t op, uint8_t vdst, uint8_t vaddr) {
    return std::array<uint32_t, 2>{
        static_cast<uint32_t>((0x36u << 26) | (static_cast<uint32_t>(op) << 18)),
        static_cast<uint32_t>(vaddr | (static_cast<uint32_t>(vdst) << 24))};
  };

  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto tr4 = ds_load_tr(0xFA, 0, 20);
  const auto tr8 = ds_load_tr(0xFD, 2, 21);
  const auto tr6 = ds_load_tr(0xFB, 4, 22);
  const auto tr16 = ds_load_tr(0xFC, 8, 23);
  const std::array<uint32_t, 9> text_words = {tr4[0], tr4[1],  tr8[0],  tr8[1],         tr6[0],
                                              tr6[1], tr16[0], tr16[1], kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_GT(translations->size(), 128u * sizeof(uint32_t));

  const auto words =
      std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(translations->data()),
                                translations->size() / sizeof(uint32_t));
  size_t ds_load_b32 = 0;
  size_t ds_load_u8 = 0;
  size_t ds_load_b96 = 0;
  size_t ds_transpose = 0;
  size_t ds_bpermute = 0;
  for (uint32_t word : words) {
    if (((word >> 26) & 0x3Fu) != 0x36u)
      continue;
    const uint32_t op = (word >> 18) & 0xFFu;
    ds_load_b32 += op == 0x36u;
    ds_load_u8 += op == 0x3Au;
    ds_load_b96 += op == 0xFEu;
    ds_transpose += op >= 0xFAu && op <= 0xFDu;
    ds_bpermute += op == 0xB3u;
  }

  EXPECT_EQ(ds_load_b32, 4u);
  EXPECT_EQ(ds_load_u8, 48u);
  EXPECT_EQ(ds_load_b96, 1u);
  EXPECT_EQ(ds_transpose, 0u);
  EXPECT_EQ(ds_bpermute, 128u);
}

TEST(BinaryTranslator, Gfx1250DsStore2AddrB64ExpandsToByteOffsetStores) {
  constexpr uint32_t kGfx1250DsStore2AddrB64W0 =
      (0x36u << 26) | (0x4Eu << 18) | (0x90u << 8) | 0x70u;
  constexpr uint32_t kGfx1250DsStore2AddrB64W1 = 13u | (18u << 8) | (20u << 16);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {kGfx1250DsStore2AddrB64W0, kGfx1250DsStore2AddrB64W1,
                                              kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_FALSE(translated.text_sections().empty());
  const Section *code = translations ? translations : translated.text_sections()[0];

  const auto words = std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(code->data()),
                                               code->size() / sizeof(uint32_t));
  std::vector<uint32_t> store_b64_offsets;
  size_t store_2addr_b64 = 0;
  for (uint32_t word : words) {
    if (((word >> 26) & 0x3Fu) != 0x36u)
      continue;
    const uint32_t op = (word >> 18) & 0xFFu;
    store_2addr_b64 += op == 0x4Eu;
    if (op == 0x4Du)
      store_b64_offsets.push_back((((word >> 8) & 0xFFu) << 8) | (word & 0xFFu));
  }

  EXPECT_EQ(store_2addr_b64, 0u);
  ASSERT_EQ(store_b64_offsets.size(), 2u);
  EXPECT_EQ(store_b64_offsets[0], 112u * 8u);
  EXPECT_EQ(store_b64_offsets[1], 144u * 8u);
}

TEST(BinaryTranslator, Gfx1250SSetVgprMsbZeroLowersToModeSetregWithoutWarning) {
  constexpr uint32_t kGfx1250SSetVgprMsb0 = 0xBF860000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 2> text_words = {kGfx1250SSetVgprMsb0, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_NE(patched_text[0], kGfx1250SSetVgprMsb0);
  EXPECT_EQ(patched_text[1], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto body_offset = find_gfx1250_rdna4_entry_stub_body_word_offset_for_test(cave_words);
  ASSERT_TRUE(body_offset.has_value());
  ASSERT_GE(*body_offset, 3u);

  const auto setreg = std::bit_cast<rdna4::SopkMachineInst>(cave_words[*body_offset - 3u]);
  EXPECT_EQ(setreg.op, 19u);
  EXPECT_EQ(setreg.sdst, 0u);
  EXPECT_EQ(setreg.simm16, hwreg_mode_vgpr_msb_for_test());
  EXPECT_EQ(cave_words[*body_offset - 2u], 0u);
  EXPECT_EQ((cave_words[*body_offset - 1u] >> 16) & 0x7Fu,
            sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250UnreachableSSetVgprMsbZeroIsNopped) {
  constexpr uint32_t kGfx1250SSetVgprMsb0 = 0xBF860000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  const std::array<uint32_t, 4> text_words = {
      kGfx1250SEndpgm,
      kGfx1250SNop0,
      kGfx1250SSetVgprMsb0,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], kGfx1250SEndpgm);
  EXPECT_EQ(patched_text[1], kGfx1250SNop0);
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250Rdna4EntryStubMaterializesWorkgroupIdSgprs) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 1> text_words = {kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_FALSE(source.text_sections().empty());

  KernelDescriptorTranslator descriptor_translator(ROCJITSU_CODE_ARCH_GFX1250,
                                                   ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations = descriptor_translator.translate_image(
      {image.data(), image.size()}, source.text_sections()[0]->sectionOffset(),
      source.text_sections()[0]->size(), KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(translations.empty());

  auto *kd = reinterpret_cast<kernel_descriptor_t *>(image.data() +
                                                     translations[0].descriptor_file_offset);
  kd->compute_pgm_rsrc2 = 0;
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 12);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y, 1);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, 1);

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations_section = find_section(translated, ".rj_translations");
  ASSERT_NE(translations_section, nullptr);
  const auto cave_words = section_words_for_test(*translations_section);

  constexpr uint16_t kTtmpBase = 108;
  const uint16_t shift16 = scalar_positive_inline_u32(16);
  const std::vector<uint32_t> expected_prefix = {
      build_s_mov_b32(12, kTtmpBase + 9, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_mov_b32(15, kTtmpBase + 9, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_mov_b32(16, kTtmpBase + 7, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_mov_b32(13, 16, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_lshl_b32(13, 13, shift16, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_lshr_b32(13, 13, shift16, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_lshr_b32(14, 16, shift16, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_mov_b32(kRdna4M0ForTest, kInlineMinusOneForTest, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      rdna4_vgpr_msb_reset_setreg_for_test(),
      0,
  };

  ASSERT_GE(cave_words.size(), expected_prefix.size() + 1u);
  const auto prefix_it = std::search(cave_words.begin(), cave_words.end(), expected_prefix.begin(),
                                     expected_prefix.end());
  ASSERT_NE(prefix_it, cave_words.end());
  const auto branch_index =
      static_cast<size_t>(std::distance(cave_words.begin(), prefix_it)) + expected_prefix.size();
  ASSERT_LT(branch_index, cave_words.size());
  EXPECT_EQ((cave_words[branch_index] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyRemapsTtmpGridReadsToCapturedSgprs) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;
  constexpr uint8_t kSCselectB32 = 48u;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = pack_sop2(kSCselectB32, 4, 117, 3);
  text_words[1] = pack_sop2(kSCselectB32, 5, 115, 3);
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());

  const auto cave_words = local_expanded_text_words_for_test(translated, kTextSize);
  ASSERT_FALSE(cave_words.empty());
  constexpr uint16_t kTtmpBase = 108;
  EXPECT_NE(
      std::ranges::find(cave_words, build_s_mov_b32(8, kTtmpBase + 9, ROCJITSU_CODE_ARCH_RDNA4)),
      cave_words.end());
  EXPECT_NE(
      std::ranges::find(cave_words, build_s_mov_b32(9, kTtmpBase + 7, ROCJITSU_CODE_ARCH_RDNA4)),
      cave_words.end());

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(cave_words);
  ASSERT_GE(copied_words.size(), 2u);
  EXPECT_NE(std::ranges::find(copied_words, pack_sop2(kSCselectB32, 4, 8, 3)), copied_words.end());
  EXPECT_NE(std::ranges::find(copied_words, pack_sop2(kSCselectB32, 5, 9, 3)), copied_words.end());
}

TEST(BinaryTranslator, Gfx1250Mode25SSetregImm32PreservesReplayMode) {
  constexpr uint32_t kGfx1250SSetregImm32Mode25 = 0xB9800641u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {
      kGfx1250SSetregImm32Mode25,
      1,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], kGfx1250SSetregImm32Mode25);
  EXPECT_EQ(patched_text[1], 1u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250IreeRawBufferDescriptorPrologueLowersToRdna4) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto build_gfx1250_sopk = [](uint8_t op, uint8_t sdst, uint16_t simm16) {
    gfx1250::SopkMachineInst inst{};
    inst.encoding = 0xBu;
    inst.op = op;
    inst.sdst = sdst;
    inst.simm16 = simm16;
    return std::bit_cast<uint32_t>(inst);
  };

  const std::array<uint32_t, 8> text_words = {
      build_gfx1250_sopk(0, 2, 1024),
      build_gfx1250_sopk(0, 6, 512),
      build_gfx1250_sopk(0, 6, 1024),
      build_gfx1250_sopk(0, 2, 8192),
      build_gfx1250_sopk(0, 2, 512),
      build_gfx1250_sopk(0, 6, 8192),
      build_gfx1250_sopk(17, 4, hwreg_gfx1250_grid_mode_for_test()),
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text.back(), kGfx1250SEndpgm);
  for (size_t i = 0; i + 2 < patched_text.size(); ++i)
    EXPECT_EQ((patched_text[i] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[6], pack_sop1(0, 4, scalar_positive_inline_u32(0)))
      << "gfx1250 grid-mode hwreg read should select the RDNA4 TTMP grid-id path";

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  const auto contains_literal_mov = [&](uint8_t sdst, uint32_t literal) {
    const uint32_t mov = pack_sop1(0, sdst, 255);
    for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
      if (cave_words[i] == mov && cave_words[i + 1] == literal)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(contains_literal_mov(3, 0x31016000u))
      << "gfx1250 i8 s[0:3] raw-buffer config word must be made explicit for RDNA4";
  EXPECT_TRUE(contains_literal_mov(6, 0x20000u))
      << "gfx1250 descriptor word2 128-byte units must become RDNA4 byte bounds";
  EXPECT_TRUE(contains_literal_mov(6, 0x10000u))
      << "gfx1250 fp8 descriptor word2 128-byte units must become RDNA4 byte bounds";
  EXPECT_TRUE(contains_literal_mov(2, 0x20000u))
      << "gfx1250 descriptor word2 128-byte units must become RDNA4 byte bounds";
  EXPECT_TRUE(contains_literal_mov(2, 0x100000u))
      << "gfx1250 descriptor word2 128-byte units must become RDNA4 byte bounds";
  EXPECT_TRUE(contains_literal_mov(2, 0x10000u))
      << "gfx1250 i8 descriptor word2 128-byte units must become RDNA4 byte bounds";
  EXPECT_TRUE(contains_literal_mov(6, 0x100000u))
      << "gfx1250 i8 descriptor word2 128-byte units must become RDNA4 byte bounds";
}

TEST(BinaryTranslator, Gfx1250InlineZeroS11IsNotResourceDescriptorConfig) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 2> text_words = {pack_sop1(0, 11, scalar_positive_inline_u32(0)),
                                              kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_NE(patched_text[0], pack_sop1(0, 11, 255))
      << "generic scalar zero-high-half address math must not be rewritten as a descriptor word";
  EXPECT_EQ(patched_text[1], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250DescriptorConfigRewriteStopsAtScalarUse) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SAddNcU64UsesS14S15 = 0xA9A80E28u;
  constexpr std::array<uint32_t, 3> kBufferLoadB32S12 = {0xC405007Cu, 0x00801802u, 0};
  const std::array<uint32_t, 6> text_words = {
      pack_sop1(0, 15, scalar_positive_inline_u32(0)),
      kGfx1250SAddNcU64UsesS14S15,
      kBufferLoadB32S12[0],
      kBufferLoadB32S12[1],
      kBufferLoadB32S12[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_FALSE(has_word_pair(pack_sop1(0, 15, 255), 0x31016000u))
      << "s15 is ordinary scalar data once consumed before the future buffer access";
  EXPECT_NE(std::ranges::find(all_words, pack_sop1(0, 15, scalar_positive_inline_u32(0))),
            all_words.end());
}

TEST(BinaryTranslator, Gfx1250GlobalMemoryDoesNotTriggerRawBufferDescriptorRewrite) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> kGlobalLoadB32 = {0xEE05007Cu, 0x0000001Au, 0x0000000Cu};
  const std::array<uint32_t, 6> text_words = {
      pack_sop1(0, 3, scalar_positive_inline_u32(0)),
      pack_sop1(0, 15, 3),
      kGlobalLoadB32[0],
      kGlobalLoadB32[1],
      kGlobalLoadB32[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_FALSE(has_word_pair(pack_sop1(0, 15, 255), 0x31016000u))
      << "global memory encodings share high bits with VBUFFER but do not consume raw descriptors";
}

TEST(BinaryTranslator, Gfx1250ZeroRawBufferDescriptorPrologueLowersWhenConsumedByVbuffer) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SDelayAlu = 0xBF870009u;
  constexpr std::array<uint32_t, 3> kBufferLoadU16S0 = {0xC404807Cu, 0x00800000u, 0};
  constexpr std::array<uint32_t, 3> kBufferLoadU16S8 = {0xC404807Cu, 0x00801001u, 0};
  constexpr std::array<uint32_t, 3> kBufferLoadB32S12 = {0xC405007Cu, 0x00801802u, 0};

  const std::array<uint32_t, 17> text_words = {
      pack_sop1(0, 2, scalar_positive_inline_u32(0)),
      kGfx1250SDelayAlu,
      pack_sop1(0, 3, 2),
      pack_sop1(0, 10, 2),
      pack_sop1(0, 11, 2),
      pack_sop1(0, 14, 2),
      pack_sop1(0, 15, 2),
      kBufferLoadU16S0[0],
      kBufferLoadU16S0[1],
      kBufferLoadU16S0[2],
      kBufferLoadU16S8[0],
      kBufferLoadU16S8[1],
      kBufferLoadU16S8[2],
      kBufferLoadB32S12[0],
      kBufferLoadB32S12[1],
      kBufferLoadB32S12[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);

  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> all_words(cave_words);
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto contains_literal_mov = [&](uint8_t sdst, uint32_t literal) {
    const uint32_t mov = pack_sop1(0, sdst, 255);
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == mov && all_words[i + 1] == literal)
        return true;
    }
    return false;
  };
  EXPECT_TRUE(contains_literal_mov(2, 0xFFFFFFFFu));
  EXPECT_TRUE(contains_literal_mov(3, 0x31016000u));
  EXPECT_TRUE(contains_literal_mov(11, 0x31016000u));
  EXPECT_TRUE(contains_literal_mov(15, 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250B64RawBufferDescriptorPairLowersWhenConsumedByVbuffer) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> kBufferLoadU16S8 = {0xC404807Cu, 0x00801001u, 0};

  const std::array<uint32_t, 6> text_words = {
      pack_sop1(1, 10, scalar_positive_inline_u32(8)),
      pack_sopp(37, 3),
      kBufferLoadU16S8[0],
      kBufferLoadU16S8[1],
      kBufferLoadU16S8[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);

  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> all_words(cave_words);
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 10, 255), 0x800u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 11, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250PaddedRawBufferBaseCopyUsesPointerDwordsForRdna4) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SDelayAlu = 0xBF870009u;
  constexpr uint8_t kSAndB32 = 22;
  constexpr uint8_t kSOrB32 = 24;
  constexpr std::array<uint32_t, 3> kBufferLoadB128S28 = {0xC405C07Cu, 0x4080382Cu, 0x0000001Eu};

  const std::array<uint32_t, 16> text_words = {
      pack_sop1(1, 28, 56),
      pack_sop1(0, 31, 255),
      0x31016000u,
      pack_sop2(kSAndB32, 8, 30, 255),
      0x7Fu,
      kGfx1250SDelayAlu,
      build_s_lshl_b32(8, 8, scalar_positive_inline_u32(25), ROCJITSU_CODE_ARCH_GFX1250),
      pack_sop2(kSAndB32, 29, 29, 255),
      0x01FF'FFFFu,
      kGfx1250SDelayAlu,
      pack_sop2(kSOrB32, 29, 29, 8),
      build_s_lshr_b32(30, 30, scalar_positive_inline_u32(7), ROCJITSU_CODE_ARCH_GFX1250),
      kBufferLoadB128S28[0],
      kBufferLoadB128S28[1],
      kBufferLoadB128S28[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  EXPECT_NE(std::ranges::find(all_words, pack_sop1(0, 28, 57)), all_words.end())
      << "gfx1250 raw-buffer bases are zero-padded before the pointer low dword";
  EXPECT_NE(std::ranges::find(all_words, pack_sop1(0, 29, 58)), all_words.end())
      << "RDNA4 raw-buffer bases must receive the real pointer high dword";
  EXPECT_EQ(std::ranges::find(all_words, pack_sop1(1, 28, 56)), all_words.end())
      << "the padded gfx1250 base copy must not survive in translated descriptor setup";
}

TEST(BinaryTranslator, Gfx1250RawBufferBaseCopyPreservesUnpaddedTupleWhenStrideFollows) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SDelayAlu = 0xBF870009u;
  constexpr uint8_t kSAndB32 = 22;
  constexpr uint8_t kSOrB32 = 24;
  constexpr std::array<uint32_t, 3> kBufferLoadB128S28 = {0xC405C07Cu, 0x4080382Cu, 0x0000001Eu};

  const std::array<uint32_t, 17> text_words = {
      pack_sop1(1, 28, 56),
      pack_sop1(0, 31, 255),
      0x31016000u,
      pack_sop2(kSAndB32, 8, 30, 255),
      0x7Fu,
      kGfx1250SDelayAlu,
      build_s_lshl_b32(8, 8, scalar_positive_inline_u32(25), ROCJITSU_CODE_ARCH_GFX1250),
      pack_sop2(kSAndB32, 29, 29, 255),
      0x01FF'FFFFu,
      kGfx1250SDelayAlu,
      pack_sop2(kSOrB32, 29, 29, 8),
      build_s_lshr_b32(30, 30, scalar_positive_inline_u32(7), ROCJITSU_CODE_ARCH_GFX1250),
      pack_sop1(0, 40, 58),
      kBufferLoadB128S28[0],
      kBufferLoadB128S28[1],
      kBufferLoadB128S28[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  EXPECT_NE(std::ranges::find(all_words, pack_sop1(1, 28, 56)), all_words.end())
      << "ordinary pointer+stride tuples must keep s56:s57 as the descriptor base";
  EXPECT_EQ(std::ranges::find(all_words, pack_sop1(0, 28, 57)), all_words.end())
      << "source tail s58 is a stride operand here, not the pointer high dword";
  EXPECT_EQ(std::ranges::find(all_words, pack_sop1(0, 29, 58)), all_words.end())
      << "source tail s58 must not be consumed as the RDNA4 pointer high dword";
  EXPECT_NE(std::ranges::find(all_words, pack_sop1(0, 40, 58)), all_words.end())
      << "the scalar stride use must remain available after descriptor lowering";
}

TEST(BinaryTranslator, Gfx1250SWaitKmcntAlsoDrainsRdna4Loadcnt) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SWaitKmcnt0 = 0xBFC70000u;
  constexpr uint32_t kRdna4SWaitLoadcnt0 = 0xBFC00000u;

  const std::array<uint32_t, 2> text_words = {kGfx1250SWaitKmcnt0, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  EXPECT_NE(std::ranges::find(all_words, kRdna4SWaitLoadcnt0), all_words.end())
      << "gfx1250 kmcnt waits can guard scalar memory values consumed on RDNA4 loadcnt";
  EXPECT_NE(std::ranges::find(all_words, kGfx1250SWaitKmcnt0), all_words.end())
      << "the original kmcnt ordering must still be preserved";
  EXPECT_EQ(std::ranges::find(all_words, pack_sop1(0, 56, 57)), all_words.end())
      << "wait lowering must not canonicalize a hardcoded SGPR tuple";
}

TEST(BinaryTranslator, Gfx1250SendmsgRtnWaitsForRdna4Kmcnt) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kRdna4SWaitKmcnt0 = 0xBFC70000u;
  const uint32_t kSendmsgRtnB32 = pack_sop1(0x4C, 6, 0x83);
  const uint32_t kSendmsgRtnB64 = pack_sop1(0x4D, 2, 0x83);
  const std::array<uint32_t, 3> text_words = {
      kSendmsgRtnB32,
      kSendmsgRtnB64,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  EXPECT_NE(std::ranges::find(all_words, kSendmsgRtnB32), all_words.end());
  EXPECT_NE(std::ranges::find(all_words, kSendmsgRtnB64), all_words.end());
  EXPECT_EQ(std::ranges::count(all_words, kRdna4SWaitKmcnt0), 2u)
      << "each translated send-message return must wait for its GFX12 response";
}

TEST(BinaryTranslator, Gfx1250B64RawBufferDescriptorPairLowersBeforeByteStore) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> kBufferStoreB8S4 = {0xC406007Cu, 0x40800803u, 1};

  const std::array<uint32_t, 5> text_words = {
      pack_sop1(1, 6, scalar_positive_inline_u32(8)),
      kBufferStoreB8S4[0],
      kBufferStoreB8S4[1],
      kBufferStoreB8S4[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 6, 255), 0x800u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 7, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250HighRawBufferDescriptorPairLowersBeforeByteStore) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> kBufferStoreB8S20 = {0xC406007Cu, 0x40802802u, 1};

  const std::array<uint32_t, 5> text_words = {
      pack_sop1(1, 22, scalar_positive_inline_u32(8)),
      kBufferStoreB8S20[0],
      kBufferStoreB8S20[1],
      kBufferStoreB8S20[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 22, 255), 0x800u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 23, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250B64RawBufferDescriptorPairLowersThroughNonzeroZeroCompare) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint8_t kSCmpEqU32 = 6;
  constexpr std::array<uint32_t, 3> kBufferStoreB32S0 = {0xC406807Cu, 0x00800000u, 0};
  const auto pack_sopc = [](uint32_t op, uint32_t ssrc0, uint32_t ssrc1) {
    return 0xBF000000u | ((op & 0x7Fu) << 16) | ((ssrc1 & 0xFFu) << 8) | (ssrc0 & 0xFFu);
  };

  const std::array<uint32_t, 6> text_words = {
      pack_sop1(1, 2, scalar_positive_inline_u32(8)),
      pack_sopc(kSCmpEqU32, 2, scalar_positive_inline_u32(0)),
      kBufferStoreB32S0[0],
      kBufferStoreB32S0[1],
      kBufferStoreB32S0[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);

  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> all_words(cave_words);
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 2, 255), 0x800u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 3, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250B64RawBufferDescriptorPairScansThroughBranchSuccessor) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kSCselectB32S8Ttmp9S1 = 0x98080175u;
  constexpr uint32_t kVCmpxGtU32 = 0x7D980282u;
  constexpr uint32_t kSCbranchExeczSkipStore = 0xBFA50003u;
  constexpr std::array<uint32_t, 3> kBufferStoreB32S0 = {0xC406807Cu, 0x00800000u, 0};

  const std::array<uint32_t, 9> text_words = {
      pack_sop1(1, 2, scalar_positive_inline_u32(8)),
      kSCselectB32S8Ttmp9S1,
      kVCmpxGtU32,
      kSCbranchExeczSkipStore,
      kBufferStoreB32S0[0],
      kBufferStoreB32S0[1],
      kBufferStoreB32S0[2],
      kGfx1250SEndpgm,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint64_t, 1> entry_offsets = {0};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250, entry_offsets);
  ASSERT_GE(blocks.size(), 3u);
  ASSERT_EQ(blocks[0]->start_offset(), 0u);
  ASSERT_EQ(blocks[0]->successors().size(), 2u);
  EXPECT_EQ(blocks[0]->successors()[0]->start_offset(), 28u);
  EXPECT_EQ(blocks[0]->successors()[1]->start_offset(), 16u);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 2, 255), 0x800u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 3, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250ZeroDescriptorWord2ScansThroughBranchAndConfigCopy) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kVCmpxGtU32 = 0x7D980282u;
  constexpr uint32_t kSCbranchExeczSkipLoad = 0xBFA50008u;
  constexpr std::array<uint32_t, 3> kBufferLoadB32S4 = {0xC405007Cu, 0x00800801u, 0};

  const std::array<uint32_t, 13> text_words = {
      pack_sop1(0, 6, scalar_positive_inline_u32(0)),
      kVCmpxGtU32,
      kSCbranchExeczSkipLoad,
      pack_sop2(25, 4, 4, 254),
      0x00000000u,
      0x20000000u,
      pack_sop1(0, 7, 6),
      kBufferLoadB32S4[0],
      kBufferLoadB32S4[1],
      kBufferLoadB32S4[2],
      kGfx1250SEndpgm,
      pack_sop1(0, 6, scalar_positive_inline_u32(1)),
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 6, 255), 0xFFFFFFFFu));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 7, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250InlineZeroConfigLowersBeforeCopiedDescriptorUse) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SWaitKmcnt0 = 0xBFC70000u;
  constexpr uint32_t kGfx1250SClause3 = 0xBF850003u;
  constexpr std::array<uint32_t, 2> kSAddNcU64S4S8Lit = {0xA984FF08u, 0x00000400u};
  constexpr std::array<uint32_t, 3> kBufferLoadB128S0 = {0xC405C07Cu, 0x00800002u, 0};
  constexpr std::array<uint32_t, 3> kBufferLoadB128S4 = {0xC405C07Cu, 0x0080080Au, 0};

  const std::array<uint32_t, 17> text_words = {
      pack_sop1(0, 2, scalar_positive_inline_u32(8)),
      pack_sop1(0, 3, scalar_positive_inline_u32(0)),
      pack_sop1(0, 6, 2),
      pack_sop1(0, 7, 3),
      kGfx1250SWaitKmcnt0,
      pack_sop1(0, 0, 8),
      pack_sop1(0, 1, 9),
      kSAddNcU64S4S8Lit[0],
      kSAddNcU64S4S8Lit[1],
      kGfx1250SClause3,
      kBufferLoadB128S0[0],
      kBufferLoadB128S0[1],
      kBufferLoadB128S0[2],
      kBufferLoadB128S4[0],
      kBufferLoadB128S4[1],
      kBufferLoadB128S4[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 2, 255), 0x800u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 3, 255), 0x31016000u));
  EXPECT_NE(std::ranges::find(all_words, pack_sop1(0, 6, 2)), all_words.end());
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 7, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250DescriptorConfigRewritePreservesLaterScalarZeroVopdMove) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> kBufferLoadU16S8 = {0xC404807Cu, 0x00801001u, 0};
  constexpr std::array<uint32_t, 3> kVopdLshlMovS11 = {0xCF448082u, 0x0009000Bu, 0x09000011u};
  const auto build_vop1 = [](uint8_t op, uint8_t vdst, uint16_t src0) {
    return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
  };

  const std::array<uint32_t, 8> text_words = {
      pack_sop1(0, 11, scalar_positive_inline_u32(0)),
      kBufferLoadU16S8[0],
      kBufferLoadU16S8[1],
      kBufferLoadU16S8[2],
      kVopdLshlMovS11[0],
      kVopdLshlMovS11[1],
      kVopdLshlMovS11[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 11, 255), 0x31016000u))
      << "the buffer load still needs the RDNA4 raw-buffer config word";
  EXPECT_NE(std::ranges::find(all_words, build_vop1(1, 9, scalar_positive_inline_u32(0))),
            all_words.end())
      << "later ordinary scalar-zero reuse must not observe the descriptor config literal";
  EXPECT_EQ(std::ranges::find(all_words, build_vop1(1, 9, 11)), all_words.end());
}

TEST(BinaryTranslator, Gfx1250DescriptorWord2RewritePreservesLaterExecMaskZero) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint8_t kSAndNot1B32 = 34;
  constexpr uint8_t kVccLo = 106;
  constexpr uint8_t kExecLo = 126;
  constexpr std::array<uint32_t, 3> kBufferLoadB128S0 = {0xC405C07Cu, 0x00800000u, 0};

  const std::array<uint32_t, 6> text_words = {
      pack_sop1(0, 2, scalar_positive_inline_u32(0)),
      kBufferLoadB128S0[0],
      kBufferLoadB128S0[1],
      kBufferLoadB128S0[2],
      pack_sop2(kSAndNot1B32, kVccLo, kExecLo, 2),
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 2, 255), 0xFFFFFFFFu))
      << "the buffer load still needs an unbounded RDNA4 descriptor range";
  EXPECT_NE(std::ranges::find(
                all_words, pack_sop2(kSAndNot1B32, kVccLo, kExecLo, scalar_positive_inline_u32(0))),
            all_words.end())
      << "later ordinary exec-mask use must not observe the descriptor range literal";
  EXPECT_EQ(std::ranges::find(all_words, pack_sop2(kSAndNot1B32, kVccLo, kExecLo, 2)),
            all_words.end());
}

TEST(BinaryTranslator, Gfx1250ScalarZeroMaskWithoutDescriptorUseStaysSgpr) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint8_t kSAndNot1B32 = 34;
  constexpr uint8_t kSCbranchVccz = 35;
  constexpr uint8_t kVccLo = 106;
  constexpr uint8_t kExecLo = 126;

  const std::array<uint32_t, 4> text_words = {
      pack_sop1(0, 2, scalar_positive_inline_u32(0)),
      pack_sop2(kSAndNot1B32, kVccLo, kExecLo, 2),
      pack_sopp(kSCbranchVccz, 1),
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  EXPECT_NE(std::ranges::find(all_words, pack_sop2(kSAndNot1B32, kVccLo, kExecLo, 2)),
            all_words.end())
      << "plain scalar-zero mask uses must not be mistaken for descriptor promotion";
  EXPECT_EQ(std::ranges::find(
                all_words, pack_sop2(kSAndNot1B32, kVccLo, kExecLo, scalar_positive_inline_u32(0))),
            all_words.end());
}

TEST(BinaryTranslator, Gfx1250VbufferB32OpcodesSurviveRdna4Encoding) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> kBufferLoadB32S0 = {0xC405007Cu, 0x00800000u, 0};
  constexpr std::array<uint32_t, 3> kBufferStoreB32S0 = {0xC406807Cu, 0x00800000u, 0};

  const std::array<uint32_t, 7> text_words = {
      kBufferLoadB32S0[0],  kBufferLoadB32S0[1],  kBufferLoadB32S0[2], kBufferStoreB32S0[0],
      kBufferStoreB32S0[1], kBufferStoreB32S0[2], kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  EXPECT_NE(std::ranges::find(all_words, kBufferLoadB32S0[0]), all_words.end());
  EXPECT_NE(std::ranges::find(all_words, kBufferStoreB32S0[0]), all_words.end());
  EXPECT_EQ(std::ranges::find(all_words, 0xC404807Cu), all_words.end())
      << "buffer_load_b32 must not translate to buffer_load_u16";
  EXPECT_EQ(std::ranges::find(all_words, 0xC406407Cu), all_words.end())
      << "buffer_store_b32 must not translate to buffer_store_b16";
}

TEST(BinaryTranslator, Gfx1250LiteralAndMovkRawBufferDescriptorBoundsLowerWhenConsumedByVbuffer) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> kBufferStoreB32S4 = {0xC406807Cu, 0x00800800u, 0};
  constexpr std::array<uint32_t, 3> kBufferStoreB32S8 = {0xC406807Cu, 0x00801000u, 0};
  const auto pack_sopk = [](uint32_t op, uint32_t sdst, uint32_t simm16) {
    return 0xB0000000u | ((op & 0x1Fu) << 23) | ((sdst & 0x7Fu) << 16) | (simm16 & 0xFFFFu);
  };

  const std::array<uint32_t, 11> text_words = {
      pack_sop1(0, 7, scalar_positive_inline_u32(0)),
      pack_sopk(0, 6, 0x67),
      kBufferStoreB32S4[0],
      kBufferStoreB32S4[1],
      kBufferStoreB32S4[2],
      pack_sop1(1, 10, 255),
      0x67,
      kBufferStoreB32S8[0],
      kBufferStoreB32S8[1],
      kBufferStoreB32S8[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);

  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> all_words(cave_words);
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop1(0, 6, 255), 0x3400u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 7, 255), 0x31016000u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 10, 255), 0x3400u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 11, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250StaticRawBufferDescriptorHighBitsLowerToRdna4Bounds) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> kBufferLoadU16S0 = {0xC404807Cu, 0x00800000u, 0};
  constexpr std::array<uint32_t, 3> kBufferStoreB32S4 = {0xC406807Cu, 0x00800800u, 0};

  const std::array<uint32_t, 16> text_words = {
      pack_sop1(0, 2, scalar_positive_inline_u32(0)),
      pack_sop2(24, 1, 5, 255),
      0x04000000u,
      pack_sop1(0, 3, 2),
      kBufferLoadU16S0[0],
      kBufferLoadU16S0[1],
      kBufferLoadU16S0[2],
      pack_sop2(25, 4, 12, 254),
      0,
      0x08000000u,
      pack_sop1(0, 6, 2),
      pack_sop1(0, 7, 2),
      kBufferStoreB32S4[0],
      kBufferStoreB32S4[1],
      kBufferStoreB32S4[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);

  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> all_words(cave_words);
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_word_pair(pack_sop2(22, 1, 5, 255), 0x01FF'FFFFu));
  EXPECT_TRUE(std::ranges::find(all_words, pack_sop1(0, 4, 12)) != all_words.end());
  EXPECT_TRUE(has_word_pair(pack_sop2(22, 5, 13, 255), 0x01FF'FFFFu));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 2, 255), 0xFFFFFFFFu));
  EXPECT_TRUE(std::ranges::find(all_words, pack_sop1(0, 6, 2)) != all_words.end());
}

TEST(BinaryTranslator, Gfx1250RawBufferConfigMaterializesAfterScalarZeroUse) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint8_t kSAndB32 = 22;
  constexpr uint8_t kSOrB32 = 24;
  constexpr std::array<uint32_t, 3> kBufferStoreB128S0 = {0xC407407Cu, 0x40800000u, 4u};

  const std::array<uint32_t, 12> text_words = {
      pack_sop1(0, 3, scalar_positive_inline_u32(0)),
      pack_sop2(kSAndB32, 4, 3, 3),
      pack_sop2(kSAndB32, 1, 1, 255),
      0x01FF'FFFFu,
      pack_sop1(0, 2, 255),
      0x01FF'FFFFu,
      pack_sop2(kSOrB32, 1, 1, 255),
      0xFC00'0000u,
      kBufferStoreB128S0[0],
      kBufferStoreB128S0[1],
      kBufferStoreB128S0[2],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };
  const auto word_pair_count = [&](uint32_t w0, uint32_t w1) {
    size_t count = 0;
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        ++count;
    }
    return count;
  };

  EXPECT_FALSE(has_word_pair(pack_sop2(kSAndB32, 1, 1, 255), 0x01FF'FFFFu))
      << "RDNA4 raw-buffer descriptors keep the unmasked host pointer high word";
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 2, 255), 0xFFFF'FFFFu));
  EXPECT_EQ(word_pair_count(pack_sop1(0, 3, 255), 0x31016000u), 1u)
      << "the scalar zero high half must survive until descriptor finalization, then become the "
         "RDNA4 raw-buffer config word before the store";
}

TEST(BinaryTranslator, Gfx1250MixedRawBufferDescriptorHighBitsLowerAcrossLongBlock) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr std::array<uint32_t, 3> kBufferStoreB32S8 = {0xC406807Cu, 0x00801000u, 0};

  std::vector<uint32_t> text_words = {
      pack_sop2(25, 8, 10, 254),
      0,
      0x88000000u,
      pack_sop1(1, 10, scalar_positive_inline_u32(2)),
  };
  text_words.insert(text_words.end(), 80, kGfx1250SNop0);
  text_words.insert(text_words.end(), kBufferStoreB32S8.begin(), kBufferStoreB32S8.end());
  text_words.push_back(kGfx1250SEndpgm);

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);

  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> all_words(cave_words);
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());
  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_TRUE(std::ranges::find(all_words, pack_sop1(0, 8, 10)) != all_words.end());
  EXPECT_TRUE(has_word_pair(pack_sop2(22, 9, 11, 255), 0x01FF'FFFFu));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 10, 255), 0x200u));
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 11, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250TensileRawBufferDescriptorHighWordLowersForRdna4) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> kBufferStoreB128S16 = {0xC407407Cu, 0x40802018u, 4u};

  const std::array<uint32_t, 16> text_words = {
      pack_sop1(1, 16, 36),
      pack_sop1(0, 18, 255),
      0xFFFFF000u,
      pack_sop1(0, 19, scalar_positive_inline_u32(0)),
      pack_sop2(22, 5, 18, 255),
      0x7Fu,
      build_s_lshl_b32(5, 5, scalar_positive_inline_u32(25), ROCJITSU_CODE_ARCH_GFX1250),
      pack_sop2(22, 17, 17, 255),
      0x01FF'FFFFu,
      pack_sop2(24, 17, 17, 5),
      build_s_lshr_b32(18, 18, scalar_positive_inline_u32(7), ROCJITSU_CODE_ARCH_GFX1250),
      kBufferStoreB128S16[0],
      kBufferStoreB128S16[1],
      kBufferStoreB128S16[2],
      kGfx1250SEndpgm,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_FALSE(has_word_pair(pack_sop2(22, 17, 17, 255), 0x01FF'FFFFu))
      << "RDNA4 raw-buffer descriptors need the full host pointer high word";
  EXPECT_FALSE(has_word_pair(pack_sop2(22, 17, 17, 255), 0xFC00'0000u));
  EXPECT_EQ(std::ranges::find(all_words, pack_sop2(24, 17, 17, 5)), all_words.end())
      << "gfx1250 range bits must not be ORed into the RDNA4 raw-buffer base high word";
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 18, 255), 0xFFFF'F000u))
      << "RDNA4 needs the full pre-shift raw-buffer range, not gfx1250's split word2 field";
  EXPECT_EQ(std::ranges::find(all_words, build_s_lshr_b32(18, 18, scalar_positive_inline_u32(7),
                                                          ROCJITSU_CODE_ARCH_GFX1250)),
            all_words.end());
  EXPECT_TRUE(has_word_pair(pack_sop1(0, 19, 255), 0x31016000u));
}

TEST(BinaryTranslator, Gfx1250SplitRawBufferDescriptorPackLowersWithoutVisibleVbufferUse) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SDelayAlu = 0xBF870009u;
  constexpr uint8_t kSAndB32 = 22;
  constexpr uint8_t kSOrB32 = 24;

  const std::array<uint32_t, 11> text_words = {
      pack_sop2(kSAndB32, 8, 30, 255),
      0x7Fu,
      kGfx1250SDelayAlu,
      build_s_lshl_b32(8, 8, scalar_positive_inline_u32(25), ROCJITSU_CODE_ARCH_GFX1250),
      pack_sop2(kSAndB32, 29, 29, 255),
      0x01FF'FFFFu,
      kGfx1250SDelayAlu,
      pack_sop2(kSOrB32, 29, 29, 8),
      build_s_lshr_b32(30, 30, scalar_positive_inline_u32(7), ROCJITSU_CODE_ARCH_GFX1250),
      kGfx1250SEndpgm,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> all_words;
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    all_words.resize(translations->size() / sizeof(uint32_t));
    std::memcpy(all_words.data(), translations->data(), translations->size());
  }
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  const auto has_word_pair = [&](uint32_t w0, uint32_t w1) {
    for (size_t i = 0; i + 1 < all_words.size(); ++i) {
      if (all_words[i] == w0 && all_words[i + 1] == w1)
        return true;
    }
    return false;
  };

  EXPECT_FALSE(has_word_pair(pack_sop2(kSAndB32, 29, 29, 255), 0x01FF'FFFFu))
      << "RDNA4 raw-buffer descriptors keep the full base high word";
  EXPECT_EQ(std::ranges::find(all_words, pack_sop2(kSOrB32, 29, 29, 8)), all_words.end())
      << "gfx1250 split range bits must not be packed into the RDNA4 base high word";
  EXPECT_EQ(std::ranges::find(all_words, build_s_lshr_b32(30, 30, scalar_positive_inline_u32(7),
                                                          ROCJITSU_CODE_ARCH_GFX1250)),
            all_words.end())
      << "RDNA4 raw-buffer word2 keeps the full pre-shift range value";
}

TEST(BinaryTranslator, Gfx1250SmemNvClearsForRdna4) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {
      0xF4104100u,
      0xF8000000u,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], 0xF4004100u);
  EXPECT_EQ(patched_text[1], 0xF8000000u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250S3CopyFromS2WithoutVbufferUseStaysScalar) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {
      pack_sop1(0, 2, scalar_positive_inline_u32(0)),
      pack_sop1(0, 3, 2),
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_NE(patched_text[0], pack_sop1(0, 2, 255));
  EXPECT_NE(patched_text[1], pack_sop1(0, 3, 255));
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250FmaMixF32LowersF16SourcesToF32Fma) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 2> kFmaMixInlineZero = {0xCC200000u, 0x1A020300u};
  const std::array<uint32_t, 3> text_words = {
      kFmaMixInlineZero[0],
      kFmaMixInlineZero[1],
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  size_t original_mix_count = 0;
  std::optional<size_t> base;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) == 0xCCu && ((w0 >> 16) & 0x7Fu) == 32u)
      ++original_mix_count;

    rdna4::Vop3MachineInst decoded{};
    std::memcpy(&decoded, cave_words.data() + i, sizeof(decoded));
    if (decoded.encoding == 0x35u && decoded.op == 531u && decoded.vdst == 0u &&
        decoded.src2 == scalar_positive_inline_u32(0)) {
      base = i;
      break;
    }
  }

  EXPECT_EQ(original_mix_count, 0u);
  ASSERT_TRUE(base.has_value());

  rdna4::Vop3MachineInst fma{};
  std::memcpy(&fma, cave_words.data() + *base, sizeof(fma));
  ASSERT_GE(fma.src0, 256u);
  ASSERT_GE(fma.src1, 256u);
  const auto src0_tmp = static_cast<uint8_t>(fma.src0 - 256u);
  const auto src1_tmp = static_cast<uint8_t>(fma.src1 - 256u);
  EXPECT_NE(src0_tmp, 0u);
  EXPECT_NE(src1_tmp, 1u);
  EXPECT_NE(src0_tmp, src1_tmp);
  EXPECT_TRUE(
      find_vop3_for_test(cave_words, 0x18Bu, src0_tmp, static_cast<uint16_t>(256u + src0_tmp), 0)
          .has_value());
  EXPECT_TRUE(
      find_vop3_for_test(cave_words, 0x18Bu, src1_tmp, static_cast<uint16_t>(256u + src1_tmp), 0)
          .has_value());
  ASSERT_LT(*base + 2u, cave_words.size());
  EXPECT_EQ((cave_words[*base + 2u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250FmaMixF32LowersHgemmF16OperandToF32Fma) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 21;
  inst.op = 32;
  inst.encoding = 0xCC;
  inst.src0 = 57;
  inst.src1 = 256 + 29;
  inst.src2 = 256 + 21;
  inst.opsel = 0;
  inst.opsel_hi = 0x2;
  inst.pad_14 = 0;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  size_t original_mix_count = 0;
  std::optional<size_t> base;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) == 0xCCu && ((w0 >> 16) & 0x7Fu) == 32u)
      ++original_mix_count;

    rdna4::Vop3MachineInst decoded{};
    std::memcpy(&decoded, cave_words.data() + i, sizeof(decoded));
    if (decoded.encoding == 0x35u && decoded.op == 531u && decoded.vdst == 21u &&
        decoded.src0 == 57u && decoded.src2 == 256u + 21u) {
      base = i;
      break;
    }
  }

  EXPECT_EQ(original_mix_count, 0u);
  ASSERT_TRUE(base.has_value());

  rdna4::Vop3MachineInst fma{};
  std::memcpy(&fma, cave_words.data() + *base, sizeof(fma));
  ASSERT_GE(fma.src1, 256u);
  const auto src1_tmp = static_cast<uint8_t>(fma.src1 - 256u);
  EXPECT_NE(src1_tmp, 21u);
  EXPECT_NE(src1_tmp, 29u);
  EXPECT_TRUE(find_vop2_for_test(cave_words, 28u, src1_tmp, scalar_positive_inline_u32(0), 29u)
                  .has_value());
  EXPECT_TRUE(
      find_vop3_for_test(cave_words, 0x18Bu, src1_tmp, static_cast<uint16_t>(256u + src1_tmp), 0)
          .has_value());
}

TEST(BinaryTranslator, Gfx1250Rdna4SemanticFloorKeepsV224Addressable) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x60;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 24;
  inst.opsel = 0x4;
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *rodata = find_section(translated, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(KD));

  const auto *desc = reinterpret_cast<const KD *>(rodata->data());
  const uint32_t granulated = AMDHSA_BITS_GET(desc->compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t addressable_vgprs = (granulated + 1u) * 8u;
  EXPECT_EQ(addressable_vgprs, 232u);
  EXPECT_GT(addressable_vgprs, 224u)
      << "the f16 K32 lowering may use v224 as its lane-xor address temporary";
}

TEST(BinaryTranslator, Gfx1250ReplayModeSetregIsPreservedForRdna4) {
  constexpr uint32_t kGfx1250SSetReplayMode = 0xB9800641u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {kGfx1250SSetReplayMode, 1u, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  std::vector<uint32_t> translated_words;
  for (const Section *text : translated.text_sections()) {
    const auto words = section_words_for_test(*text);
    translated_words.insert(translated_words.end(), words.begin(), words.end());
  }
  if (const Section *translations = find_section(translated, ".rj_translations")) {
    const auto words = section_words_for_test(*translations);
    translated_words.insert(translated_words.end(), words.begin(), words.end());
  }

  bool found_replay_setreg = false;
  for (size_t i = 0; i + 1 < translated_words.size(); ++i) {
    const auto setreg = std::bit_cast<rdna4::SopkMachineInst>(translated_words[i]);
    if (setreg.op == 19u && setreg.sdst == 0u &&
        setreg.simm16 == hwreg_mode_replay_mode_for_test() && translated_words[i + 1] == 1u) {
      found_replay_setreg = true;
      break;
    }
  }
  EXPECT_TRUE(found_replay_setreg);
}

TEST(BinaryTranslator, Gfx1250PcSop1OpsPreserveRdna4Encodings) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Sop1MachineInst get_pc{};
  get_pc.encoding = 0x17D;
  get_pc.op = 71;
  get_pc.sdst = 12;

  gfx1250::Sop1MachineInst set_pc{};
  set_pc.encoding = 0x17D;
  set_pc.op = 72;
  set_pc.ssrc0 = 12;

  gfx1250::Sop1MachineInst swap_pc{};
  swap_pc.encoding = 0x17D;
  swap_pc.op = 73;
  swap_pc.sdst = 26;
  swap_pc.ssrc0 = 12;

  const std::array<uint32_t, 4> text_words = {
      std::bit_cast<uint32_t>(get_pc),
      std::bit_cast<uint32_t>(set_pc),
      std::bit_cast<uint32_t>(swap_pc),
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], text_words[0]);
  EXPECT_EQ(patched_text[1], text_words[1]);
  EXPECT_EQ(patched_text[2], text_words[2]);
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250StaticSwappcTargetPreservesCallerLiveVgprs) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint16_t kPcSgpr = 12;
  constexpr uint16_t kTmpSgpr = 8;
  constexpr uint16_t kReturnSgpr = 26;
  constexpr uint32_t kHelperWord = 16;
  constexpr uint32_t kHelperOffset = kHelperWord * sizeof(uint32_t);
  constexpr uint32_t kLiteral = kHelperOffset - 8;
  constexpr uint16_t kRdna4VCvtF32F16Vop3 = 0x18Bu;

  gfx1250::Sop1MachineInst swap_pc{};
  swap_pc.encoding = 0x17D;
  swap_pc.op = 73;
  swap_pc.sdst = kReturnSgpr;
  swap_pc.ssrc0 = kPcSgpr;

  gfx1250::Vop1MachineInst use_v0_after_call{};
  use_v0_after_call.encoding = 0x3F;
  use_v0_after_call.op = 1;
  use_v0_after_call.vdst = 2;
  use_v0_after_call.src0 = 256;

  gfx1250::Vop1MachineInst high_cvt{};
  high_cvt.encoding = 0x3F;
  high_cvt.op = 11;
  high_cvt.vdst = 41;
  high_cvt.src0 = 256 + 172;

  gfx1250::Sop1MachineInst set_pc{};
  set_pc.encoding = 0x17D;
  set_pc.op = 72;
  set_pc.ssrc0 = kReturnSgpr;

  std::array<uint32_t, 18> text_words{};
  text_words.fill(kGfx1250SNop0);
  text_words[0] = pack_sop1(71, kPcSgpr, 0);
  text_words[1] = pack_sop2(2, kTmpSgpr, 255, scalar_positive_inline_u32(4));
  text_words[2] = kLiteral;
  text_words[3] = pack_sop2(0, kPcSgpr, kPcSgpr, kTmpSgpr);
  text_words[4] = pack_sop2(4, kPcSgpr + 1u, kPcSgpr + 1u, scalar_positive_inline_u32(0));
  text_words[5] = std::bit_cast<uint32_t>(swap_pc);
  text_words[6] = std::bit_cast<uint32_t>(use_v0_after_call);
  text_words[7] = kGfx1250SEndpgm;
  text_words[kHelperWord] = std::bit_cast<uint32_t>(high_cvt);
  text_words[kHelperWord + 1] = std::bit_cast<uint32_t>(set_pc);

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<rdna4::Vop3MachineInst> convert;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == kRdna4VCvtF32F16Vop3 &&
        candidate.vdst == 41u) {
      convert = candidate;
      break;
    }
  }
  ASSERT_TRUE(convert.has_value());
  ASSERT_GE(convert->src0, 256u);
  const auto tmp_vgpr = static_cast<uint8_t>(convert->src0 - 256u);
  EXPECT_NE(tmp_vgpr, 0u);
  EXPECT_TRUE(
      find_vop2_for_test(cave_words, 25, tmp_vgpr, scalar_positive_inline_u32(16), 44).has_value());
}

TEST(BinaryTranslator, Gfx1250SharedStaticSwappcTargetUsesMergedCallerLiveness) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint16_t kPcSgpr = 12;
  constexpr uint16_t kTmpSgpr = 8;
  constexpr uint16_t kReturnSgpr = 26;
  constexpr uint32_t kEntry1Word = 8;
  constexpr uint32_t kHelperWord = 24;
  constexpr uint32_t kHelperOffset = kHelperWord * sizeof(uint32_t);
  constexpr uint16_t kRdna4VCvtF32F16Vop3 = 0x18Bu;

  gfx1250::Sop1MachineInst swap_pc{};
  swap_pc.encoding = 0x17D;
  swap_pc.op = 73;
  swap_pc.sdst = kReturnSgpr;
  swap_pc.ssrc0 = kPcSgpr;

  gfx1250::Vop1MachineInst use_v0_after_call{};
  use_v0_after_call.encoding = 0x3F;
  use_v0_after_call.op = 1;
  use_v0_after_call.vdst = 2;
  use_v0_after_call.src0 = 256;

  gfx1250::Vop1MachineInst use_v1_after_call{};
  use_v1_after_call.encoding = 0x3F;
  use_v1_after_call.op = 1;
  use_v1_after_call.vdst = 3;
  use_v1_after_call.src0 = 257;

  gfx1250::Vop1MachineInst high_cvt{};
  high_cvt.encoding = 0x3F;
  high_cvt.op = 11;
  high_cvt.vdst = 41;
  high_cvt.src0 = 256 + 172;

  gfx1250::Sop1MachineInst set_pc{};
  set_pc.encoding = 0x17D;
  set_pc.op = 72;
  set_pc.ssrc0 = kReturnSgpr;

  auto append_call = [&](std::array<uint32_t, 26> &words, uint32_t base_word, uint32_t caller_use) {
    const uint32_t base_offset = base_word * sizeof(uint32_t);
    const uint32_t literal = kHelperOffset - (base_offset + 2u * sizeof(uint32_t));
    words[base_word + 0] = pack_sop1(71, kPcSgpr, 0);
    words[base_word + 1] = pack_sop2(2, kTmpSgpr, 255, scalar_positive_inline_u32(4));
    words[base_word + 2] = literal;
    words[base_word + 3] = pack_sop2(0, kPcSgpr, kPcSgpr, kTmpSgpr);
    words[base_word + 4] = pack_sop2(4, kPcSgpr + 1u, kPcSgpr + 1u, scalar_positive_inline_u32(0));
    words[base_word + 5] = std::bit_cast<uint32_t>(swap_pc);
    words[base_word + 6] = caller_use;
    words[base_word + 7] = kGfx1250SEndpgm;
  };

  std::array<uint32_t, 26> text_words{};
  text_words.fill(kGfx1250SNop0);
  append_call(text_words, 0, std::bit_cast<uint32_t>(use_v0_after_call));
  append_call(text_words, kEntry1Word, std::bit_cast<uint32_t>(use_v1_after_call));
  text_words[kHelperWord] = std::bit_cast<uint32_t>(high_cvt);
  text_words[kHelperWord + 1] = std::bit_cast<uint32_t>(set_pc);

  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors_and_text(
      text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250, 0, kEntry1Word);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.ok()) << testing::PrintToString(result.diagnostics);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<rdna4::Vop3MachineInst> convert;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == kRdna4VCvtF32F16Vop3 &&
        candidate.vdst == 41u) {
      convert = candidate;
      break;
    }
  }
  ASSERT_TRUE(convert.has_value());
  ASSERT_GE(convert->src0, 256u);
  const auto tmp_vgpr = static_cast<uint8_t>(convert->src0 - 256u);
  EXPECT_NE(tmp_vgpr, 0u);
  EXPECT_NE(tmp_vgpr, 1u);
  EXPECT_TRUE(
      find_vop2_for_test(cave_words, 25, tmp_vgpr, scalar_positive_inline_u32(16), 44).has_value());
}

TEST(BinaryTranslator, Gfx1250SSetVgprMsbResetWithPreviousModeLowersToModeSetreg) {
  constexpr uint32_t kGfx1250SSetVgprMsbResetFromDstBank1 = 0xBF864000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 2> text_words = {kGfx1250SSetVgprMsbResetFromDstBank1,
                                              kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_NE(patched_text[0], kGfx1250SSetVgprMsbResetFromDstBank1);
  EXPECT_EQ(patched_text[1], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250SSetVgprMsbNonzeroRolesLowerToModeSetreg) {
  struct Case {
    uint16_t simm16;
    uint32_t expected_mode_literal;
  };
  constexpr std::array<Case, 4> cases = {{
      {0x0001u, 0x04u}, // src0 bank 1 -> MODE bits 14:15
      {0x0008u, 0x20u}, // src1 bank 2 -> MODE bits 16:17
      {0x0030u, 0xC0u}, // src2 bank 3 -> MODE bits 18:19
      {0x0040u, 0x01u}, // dst bank 1 -> MODE bits 12:13
  }};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const auto &c : cases) {
    SCOPED_TRACE(c.simm16);
    const std::array<uint32_t, 2> text_words = {0xBF860000u | c.simm16, kGfx1250SEndpgm};

    auto image =
        make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
    AmdGpuCodeObject co(image.data(), image.size());
    ASSERT_TRUE(co.is_valid());

    BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                EF_AMDGPU_MACH_AMDGCN_GFX1201);
    auto result = translator.translate(co);
    ASSERT_FALSE(result.elf_bytes.empty());
    EXPECT_TRUE(result.warnings.empty());

    AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *text = translated.text_sections()[0];
    ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

    std::array<uint32_t, text_words.size()> patched_text{};
    std::memcpy(patched_text.data(), text->data(), text->size());
    EXPECT_NE(patched_text[0], text_words[0]);
    EXPECT_EQ(patched_text[1], kGfx1250SEndpgm);
  }
}

TEST(BinaryTranslator, Gfx1250VMovB64LowersToTwoB32Moves) {
  constexpr uint32_t kGfx1250VMovB64V10S2 = 0x7E143A02u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 2> text_words = {kGfx1250VMovB64V10S2, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const std::array<uint32_t, 2> prefix = {build_vop1_for_test(1, 10, 2),
                                          build_vop1_for_test(1, 11, 3)};
  const auto base = find_branch_terminated_prefix_for_test(cave_words, prefix);
  ASSERT_TRUE(base.has_value());

  const auto mov_lo = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[*base]);
  EXPECT_EQ(mov_lo.op, 1u);
  EXPECT_EQ(mov_lo.vdst, 10u);
  EXPECT_EQ(mov_lo.src0, 2u);

  const auto mov_hi = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[*base + 1u]);
  EXPECT_EQ(mov_hi.op, 1u);
  EXPECT_EQ(mov_hi.vdst, 11u);
  EXPECT_EQ(mov_hi.src0, 3u);

  EXPECT_EQ((cave_words[*base + 2u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VMovB64SignExtendsInlineConstant) {
  constexpr uint16_t kInlineOne = 129;
  constexpr uint16_t kInlineZero = 128;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 29;
  inst.vdst = 8;
  inst.src0 = kInlineOne;

  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};
  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const std::array<uint32_t, 2> prefix = {build_vop1_for_test(1, 8, kInlineOne),
                                          build_vop1_for_test(1, 9, kInlineZero)};
  EXPECT_TRUE(find_branch_terminated_prefix_for_test(cave_words, prefix).has_value());
}

TEST(BinaryTranslator, Gfx1250VMovB64LowersLiteral64) {
  constexpr uint16_t kLiteral64Src = 254;
  constexpr uint16_t kLiteral32Src = 255;
  constexpr uint32_t kLiteralLo = 0;
  constexpr uint32_t kLiteralHi = 0x80000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 29;
  inst.vdst = 0;
  inst.src0 = kLiteral64Src;

  const std::array<uint32_t, 4> text_words = {std::bit_cast<uint32_t>(inst), kLiteralLo, kLiteralHi,
                                              kGfx1250SEndpgm};
  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const std::array<uint32_t, 4> prefix = {build_vop1_for_test(1, 0, kLiteral32Src), kLiteralLo,
                                          build_vop1_for_test(1, 1, kLiteral32Src), kLiteralHi};
  EXPECT_TRUE(find_branch_terminated_prefix_for_test(cave_words, prefix).has_value());
}

TEST(BinaryTranslator, Gfx1250VMinNumF64E32MaterializesLiteral64) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint8_t kOpVMinNumF64 = 13;
  constexpr uint8_t kDestination = 4;
  constexpr uint8_t kRegisterSource = 6;
  constexpr uint32_t kLiteralLo = 0xFFFFFFFFu;
  constexpr uint32_t kLiteralHi = 0x7FEFFFFFu;
  gfx1250::Vop2MachineInst inst{};
  inst.src0 = 254;
  inst.vsrc1 = kRegisterSource;
  inst.vdst = kDestination;
  inst.op = kOpVMinNumF64;

  std::array<uint32_t, 4> text_words{};
  text_words[0] = std::bit_cast<uint32_t>(inst);
  text_words[1] = kLiteralLo;
  text_words[2] = kLiteralHi;
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<size_t> lowering;
  for (uint16_t literal_pair = 0; literal_pair <= 254 && !lowering; ++literal_pair) {
    if (literal_pair == kDestination || literal_pair + 1u == kDestination ||
        literal_pair == kRegisterSource || literal_pair + 1u == kRegisterSource)
      continue;
    rdna4::Vop2MachineInst lowered{};
    lowered.src0 = static_cast<uint16_t>(256u + literal_pair);
    lowered.vsrc1 = kRegisterSource;
    lowered.vdst = kDestination;
    lowered.op = kOpVMinNumF64;
    const std::array<uint32_t, 6> expected = {
        build_vop1_for_test(1, static_cast<uint8_t>(literal_pair), 255),
        kLiteralLo,
        build_vop1_for_test(1, static_cast<uint8_t>(literal_pair + 1u), 255),
        kLiteralHi,
        build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4),
        std::bit_cast<uint32_t>(lowered),
    };
    lowering = find_word_sequence_for_test(cave_words, expected);
  }
  EXPECT_TRUE(lowering.has_value());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250VMovB64SignExtendsLiteral32) {
  constexpr uint16_t kLiteral32Src = 255;
  constexpr uint16_t kInlineNegativeOne = 193;
  constexpr uint32_t kLiteral = 0x80000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 29;
  inst.vdst = 2;
  inst.src0 = kLiteral32Src;

  const std::array<uint32_t, 3> text_words = {std::bit_cast<uint32_t>(inst), kLiteral,
                                              kGfx1250SEndpgm};
  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const std::array<uint32_t, 3> prefix = {build_vop1_for_test(1, 2, kLiteral32Src), kLiteral,
                                          build_vop1_for_test(1, 3, kInlineNegativeOne)};
  EXPECT_TRUE(find_branch_terminated_prefix_for_test(cave_words, prefix).has_value());
}

TEST(BinaryTranslator, Gfx1250VMovB16PacksHighHalfIntoLowPhysicalVgpr) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 28;
  inst.vdst = 128 + 7;
  inst.src0 = 256 + 40;

  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  const auto build_vop2_test = [](uint8_t op, uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
    return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
  };

  constexpr uint8_t kTmp = 0;
  constexpr uint8_t kDstPhys = 7;
  EXPECT_NE(
      std::find(cave_words.begin(), cave_words.end(), build_vop2_test(27, kDstPhys, 255, kDstPhys)),
      cave_words.end())
      << "high-half v_mov_b16 must preserve the low half of the physical destination";
  EXPECT_NE(std::find(cave_words.begin(), cave_words.end(),
                      build_vop2_test(28, kDstPhys, 256 + kTmp, kDstPhys)),
            cave_words.end())
      << "high-half v_mov_b16 must merge into vdst & 0x7f, not physical v128+";
  EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), 0x0000FFFFu), cave_words.end());
  EXPECT_EQ((cave_words.back() >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VMovB64HighVdstLowersToTwoB32Moves) {
  constexpr uint32_t kGfx1250VMovB64V154V132 = 0x7F343B84u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 2> text_words = {kGfx1250VMovB64V154V132, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const std::array<uint32_t, 2> prefix = {build_vop1_for_test(1, 154, 388),
                                          build_vop1_for_test(1, 155, 389)};
  const auto base = find_branch_terminated_prefix_for_test(cave_words, prefix);
  ASSERT_TRUE(base.has_value());

  const auto mov_lo = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[*base]);
  EXPECT_EQ(mov_lo.op, 1u);
  EXPECT_EQ(mov_lo.vdst, 154u);
  EXPECT_EQ(mov_lo.src0, 388u);

  const auto mov_hi = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[*base + 1u]);
  EXPECT_EQ(mov_hi.op, 1u);
  EXPECT_EQ(mov_hi.vdst, 155u);
  EXPECT_EQ(mov_hi.src0, 389u);
}

TEST(BinaryTranslator, Gfx1250Vop3VMovB64LowersInPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 10;
  inst.op = 413;
  inst.encoding = 0x35;
  inst.src0 = 2;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());

  const auto mov_lo = std::bit_cast<rdna4::Vop1MachineInst>(patched_text[0]);
  EXPECT_EQ(mov_lo.op, 1u);
  EXPECT_EQ(mov_lo.vdst, 10u);
  EXPECT_EQ(mov_lo.src0, 2u);

  const auto mov_hi = std::bit_cast<rdna4::Vop1MachineInst>(patched_text[1]);
  EXPECT_EQ(mov_hi.op, 1u);
  EXPECT_EQ(mov_hi.vdst, 11u);
  EXPECT_EQ(mov_hi.src0, 3u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250Vop3VMovB64SignExtendsInlineConstant) {
  constexpr uint16_t kInlineNegativeOne = 193;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 10;
  inst.op = 413;
  inst.encoding = 0x35;
  inst.src0 = kInlineNegativeOne;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto mov_lo = std::bit_cast<rdna4::Vop1MachineInst>(patched_text[0]);
  EXPECT_EQ(mov_lo.op, 1u);
  EXPECT_EQ(mov_lo.vdst, 10u);
  EXPECT_EQ(mov_lo.src0, kInlineNegativeOne);
  const auto mov_hi = std::bit_cast<rdna4::Vop1MachineInst>(patched_text[1]);
  EXPECT_EQ(mov_hi.op, 1u);
  EXPECT_EQ(mov_hi.vdst, 11u);
  EXPECT_EQ(mov_hi.src0, kInlineNegativeOne);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250Vop3SingleSrcClearsReservedSourcesInPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250Op644W0 = 0xD6840004u;
  constexpr uint32_t kGfx1250Op644W1 = 0x02010004u;
  const std::array<uint32_t, 3> text_words = {
      kGfx1250Op644W0,
      kGfx1250Op644W1,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], kGfx1250Op644W0);
  EXPECT_EQ(patched_text[1], 0x00000004u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250VSubNcU64E32LowersToCarryChain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop2MachineInst inst{};
  inst.src0 = 256 + 2;
  inst.vsrc1 = 4;
  inst.vdst = 10;
  inst.op = 41;
  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  expect_sub_u64_carry_chain(translated, 10, 256 + 2, 256 + 4);
}

TEST(BinaryTranslator, Gfx1250VSubNcU64E32LowersLiteral64) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint64_t kLiteral = 0xFFFF'FFD4'9E9A'C6DBull;
  gfx1250::Vop2MachineInst inst{};
  inst.src0 = 254;
  inst.vsrc1 = 6;
  inst.vdst = 4;
  inst.op = 41;
  const std::array<uint32_t, 4> text_words = {
      std::bit_cast<uint32_t>(inst),
      static_cast<uint32_t>(kLiteral),
      static_cast<uint32_t>(kLiteral >> 32u),
      kGfx1250SEndpgm,
  };
  ASSERT_EQ(text_words[0], 0x52080CFEu);

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  constexpr uint16_t kOpSubCoU32 = 769u;
  const auto base = find_vop3_sdst_for_test(cave_words, kOpSubCoU32, 4, 255, 256 + 6);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 9u, cave_words.size());

  rdna4::Vop3SdstEncMachineInst sub_lo{};
  std::memcpy(&sub_lo, cave_words.data() + *base, sizeof(sub_lo));
  EXPECT_EQ(sub_lo.src0, 255u);
  EXPECT_EQ(sub_lo.src1, 256u + 6u);
  EXPECT_EQ(cave_words[*base + 2u], static_cast<uint32_t>(kLiteral));
  EXPECT_EQ(cave_words[*base + 3u],
            build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3SdstEncMachineInst sub_hi{};
  std::memcpy(&sub_hi, cave_words.data() + *base + 4u, sizeof(sub_hi));
  EXPECT_EQ(sub_hi.op, 289u);
  EXPECT_EQ(sub_hi.vdst, 5u);
  EXPECT_EQ(sub_hi.sdst, kNullSgprForTest);
  EXPECT_EQ(sub_hi.src0, 255u);
  EXPECT_EQ(sub_hi.src1, 256u + 7u);
  EXPECT_EQ(sub_hi.src2, sub_lo.sdst);
  EXPECT_EQ(cave_words[*base + 6u], static_cast<uint32_t>(kLiteral >> 32u));
  EXPECT_EQ(cave_words[*base + 7u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ((cave_words[*base + 8u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VAddNcU64E32LowersToCarryChain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop2MachineInst inst{};
  inst.src0 = 256 + 2;
  inst.vsrc1 = 4;
  inst.vdst = 10;
  inst.op = 40;
  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  expect_add_u64_carry_chain(translated, 10, 256 + 2, 256 + 4);
}

TEST(BinaryTranslator, Gfx1250VAddNcU64E32LowersFlatScratchBasePair) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint16_t kFlatScratchBaseLo = 230;
  constexpr uint16_t kRdna4SrcPrivateBase = 237;
  gfx1250::Vop2MachineInst inst{};
  inst.src0 = kFlatScratchBaseLo;
  inst.vsrc1 = 0;
  inst.vdst = 2;
  inst.op = 40;
  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};
  ASSERT_EQ(text_words[0], 0x500400E6u);

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<size_t> aperture_move;
  uint8_t aperture_sgpr = 0;
  for (size_t i = 0; i < cave_words.size(); ++i) {
    const auto mov = std::bit_cast<rdna4::Sop1MachineInst>(cave_words[i]);
    if (mov.encoding == 0x17Du && mov.op == 1u && mov.ssrc0 == kRdna4SrcPrivateBase) {
      aperture_move = i;
      aperture_sgpr = static_cast<uint8_t>(mov.sdst);
      break;
    }
  }
  ASSERT_TRUE(aperture_move.has_value());
  ASSERT_LE(*aperture_move + 4u, cave_words.size());
  EXPECT_EQ(aperture_sgpr & 1u, 0u);
  EXPECT_LE(aperture_sgpr, 105u);
  EXPECT_EQ(cave_words[*aperture_move + 1u],
            build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4));

  const auto mov_offset = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[*aperture_move + 2u]);
  EXPECT_EQ(mov_offset.op, 1u);
  EXPECT_EQ(mov_offset.vdst, 2u);
  EXPECT_EQ(mov_offset.src0, 256u);

  const auto mov_aperture = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[*aperture_move + 3u]);
  EXPECT_EQ(mov_aperture.op, 1u);
  EXPECT_EQ(mov_aperture.vdst, 3u);
  EXPECT_EQ(mov_aperture.src0, static_cast<uint16_t>(aperture_sgpr + 1u));
  EXPECT_EQ((cave_words[*aperture_move + 4u] >> 16) & 0x7Fu,
            sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VAddF16E32PreservesHalfSelectorsInRdna4Vop3) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint16_t kRdna4VAddF16Vop3 = 306u;
  gfx1250::Vop2MachineInst inst{};
  inst.src0 = 256 + 129;
  inst.vsrc1 = 129;
  inst.vdst = 129;
  inst.op = 50;
  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base = find_vop3_for_test(cave_words, kRdna4VAddF16Vop3, 1, 256 + 1, 256 + 1);
  ASSERT_TRUE(base.has_value());

  rdna4::Vop3MachineInst lowered{};
  std::memcpy(&lowered, cave_words.data() + *base, sizeof(lowered));
  EXPECT_EQ(lowered.opsel, 0xBu)
      << "gfx1250 src0.h, src1.h, and dst.h must use RDNA4 VOP3 op_sel bits";
}

TEST(BinaryTranslator, Gfx1250VCvtF32F16E32HighSrcUsesRdna4Vop3) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint16_t kRdna4VCvtF32F16Vop3 = 0x18Bu;

  gfx1250::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 11;
  inst.vdst = 41;
  inst.src0 = 256 + 172;
  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<rdna4::Vop3MachineInst> convert;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == kRdna4VCvtF32F16Vop3 &&
        candidate.vdst == 41u) {
      convert = candidate;
      break;
    }
  }
  ASSERT_TRUE(convert.has_value());
  EXPECT_NE(convert->src0, static_cast<uint16_t>(256 + 172))
      << "gfx1250 v44.h must not be interpreted as RDNA4 v172";

  ASSERT_GE(convert->src0, 256u);
  ASSERT_LT(convert->src0, 512u);
  const auto tmp_vgpr = static_cast<uint8_t>(convert->src0 - 256u);
  EXPECT_TRUE(
      find_vop2_for_test(cave_words, 25, tmp_vgpr, scalar_positive_inline_u32(16), 44).has_value())
      << "the high f16 half must be materialized from v44 before conversion";
}

TEST(BinaryTranslator, Gfx1250VAddNcU64E32PositiveInlineLowersToCarryChain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop2MachineInst inst{};
  inst.src0 = scalar_positive_inline_u32(64);
  inst.vsrc1 = 2;
  inst.vdst = 12;
  inst.op = 40;
  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  expect_add_u64_carry_chain(translated, 12, scalar_positive_inline_u32(64), 256 + 2, std::nullopt,
                             scalar_positive_inline_u32(0));
}

TEST(BinaryTranslator, Gfx1250VAddNcU64E32NegativeInlineLowersToCarryChain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop2MachineInst inst{};
  inst.src0 = scalar_negative_inline_i32(-4);
  inst.vsrc1 = 2;
  inst.vdst = 12;
  inst.op = 40;
  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  expect_add_u64_carry_chain(translated, 12, scalar_negative_inline_i32(-4), 256 + 2, std::nullopt,
                             scalar_negative_inline_i32(-1));
}

TEST(BinaryTranslator, Gfx1250VAddNcU64E32LiteralLowersToCarryChain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop2InstLiteralMachineInst inst{};
  inst.src0 = 255;
  inst.vsrc1 = 4;
  inst.vdst = 10;
  inst.op = 40;
  inst.simm32 = 0x4000;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  expect_add_u64_carry_chain(translated, 10, 255, 256 + 4, 0x4000);
}

TEST(BinaryTranslator, Gfx1250VAddNcU64E32NegativeLiteralLowersToSignExtendedCarryChain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint16_t kScalarInlineNegativeOne = 193u;
  gfx1250::Vop2InstLiteralMachineInst inst{};
  inst.src0 = 255;
  inst.vsrc1 = 4;
  inst.vdst = 10;
  inst.op = 40;
  inst.simm32 = 0xFFFFFFFFu;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  expect_add_u64_carry_chain(translated, 10, 255, 256 + 4, 0xFFFFFFFFu, kScalarInlineNegativeOne);
}

TEST(BinaryTranslator, Gfx1250VopdXyAddFmacLiteralLowersInPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {
      0xC900150Eu, // v_dual_add_f32 v10, v14, v10
      0x0A0800FFu, // v_dual_fmac_f32 v9, literal, v0
      0x32A5705Fu,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto add = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[0]);
  EXPECT_EQ(add.op, 3u);
  EXPECT_EQ(add.src0, 256u + 14u);
  EXPECT_EQ(add.vsrc1, 10u);
  EXPECT_EQ(add.vdst, 10u);

  const auto fmac = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[1]);
  EXPECT_EQ(fmac.op, 43u);
  EXPECT_EQ(fmac.src0, 255u);
  EXPECT_EQ(fmac.vsrc1, 0u);
  EXPECT_EQ(fmac.vdst, 9u);
  EXPECT_EQ(patched_text[2], 0x32A5705Fu);
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250VopdXyDualMaxLowersInPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {
      0xCA940301u, // v_dual_max_num_f32 v1, v1, v1
      0x01020502u, // v_dual_max_num_f32 v2, v2, v2
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto max_x = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[0]);
  EXPECT_EQ(max_x.op, 22u);
  EXPECT_EQ(max_x.src0, 256u + 1u);
  EXPECT_EQ(max_x.vsrc1, 1u);
  EXPECT_EQ(max_x.vdst, 1u);

  const auto max_y = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[1]);
  EXPECT_EQ(max_y.op, 22u);
  EXPECT_EQ(max_y.src0, 256u + 2u);
  EXPECT_EQ(max_y.vsrc1, 2u);
  EXPECT_EQ(max_y.vdst, 2u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250VopdXyCndmaskLiteralLowersInPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {
      0xCA922914u, // v_dual_max_num_f32 v20, v20, v20
      0x14040AFFu, // v_dual_cndmask_b32 v5, literal, v5, vcc
      0xFF800000u,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto max_x = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[0]);
  EXPECT_EQ(max_x.op, 22u);
  EXPECT_EQ(max_x.src0, 256u + 20u);
  EXPECT_EQ(max_x.vsrc1, 20u);
  EXPECT_EQ(max_x.vdst, 20u);

  const auto cndmask = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[1]);
  EXPECT_EQ(cndmask.op, 1u);
  EXPECT_EQ(cndmask.src0, 255u);
  EXPECT_EQ(cndmask.vsrc1, 5u);
  EXPECT_EQ(cndmask.vdst, 5u);
  EXPECT_EQ(patched_text[2], 0xFF800000u);
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250VopdXyAddCndmaskLowersViaWildcardRule) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {
      0xC9120080u, // v_dual_add_f32 v0, 0, v0
      0x00000280u, // v_dual_cndmask_b32 v1, 0, v1, vcc
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto add = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[0]);
  EXPECT_EQ(add.op, 3u);
  EXPECT_EQ(add.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(add.vsrc1, 0u);
  EXPECT_EQ(add.vdst, 0u);

  const auto cndmask = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[1]);
  EXPECT_EQ(cndmask.op, 1u);
  EXPECT_EQ(cndmask.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(cndmask.vsrc1, 1u);
  EXPECT_EQ(cndmask.vdst, 1u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250VopdXyAddAddNcLiteralLowersViaWildcardRule) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {
      0xC9200501u, // v_dual_add_f32 v1, v1, v2
      0x010026FFu, // v_dual_add_nc_u32 v0, literal, v19
      0x00000090u,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto add = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[0]);
  EXPECT_EQ(add.op, 3u);
  EXPECT_EQ(add.src0, 256u + 1u);
  EXPECT_EQ(add.vsrc1, 2u);
  EXPECT_EQ(add.vdst, 1u);

  const auto add_nc = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[1]);
  EXPECT_EQ(add_nc.op, 37u);
  EXPECT_EQ(add_nc.src0, 255u);
  EXPECT_EQ(add_nc.vsrc1, 19u);
  EXPECT_EQ(add_nc.vdst, 0u);
  EXPECT_EQ(patched_text[2], 0x00000090u);
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250SMovB64Literal64SignExtendedLowersToLiteral32) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {
      pack_sop1(1, 20, 254),
      0xFFFFF000u,
      0xFFFFFFFFu,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], pack_sop1(1, 20, 255));
  EXPECT_EQ(patched_text[1], 0xFFFFF000u);
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250SMulU64Literal64MaterializesHostSgprPair) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint8_t kOpSMulU64 = 85;
  constexpr uint8_t kDestination = 36;
  constexpr uint32_t kLiteralLo = 0x55555556u;
  constexpr uint32_t kLiteralHi = 0x55555555u;
  const std::array<uint32_t, 4> text_words = {
      pack_sop2(kOpSMulU64, kDestination, kDestination, 254),
      kLiteralLo,
      kLiteralHi,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<size_t> lowering;
  for (uint8_t literal_pair = 0; literal_pair <= 124 && !lowering; literal_pair += 2) {
    if (literal_pair == kDestination || literal_pair + 1u == kDestination)
      continue;
    const std::array<uint32_t, 7> expected = {
        pack_sop1(0, literal_pair, 255),
        kLiteralLo,
        pack_sop1(0, static_cast<uint8_t>(literal_pair + 1u), 255),
        kLiteralHi,
        build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4),
        pack_sop2(kOpSMulU64, kDestination, kDestination, literal_pair),
        build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4),
    };
    lowering = find_word_sequence_for_test(cave_words, expected);
  }
  EXPECT_TRUE(lowering.has_value());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250SCmpLgU64Literal64MaterializesHostSgprPair) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint8_t kOpSCmpLgU64 = 17;
  constexpr uint8_t kSource = 36;
  constexpr uint32_t kLiteralLo = 0x9E9AC6DBu;
  constexpr uint32_t kLiteralHi = 0xFFFFFFD4u;
  const std::array<uint32_t, 4> text_words = {
      pack_sopc_for_test(kOpSCmpLgU64, kSource, 254),
      kLiteralLo,
      kLiteralHi,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<size_t> lowering;
  for (uint8_t literal_pair = 0; literal_pair <= 124 && !lowering; literal_pair += 2) {
    if (literal_pair == kSource || literal_pair + 1u == kSource)
      continue;
    const std::array<uint32_t, 6> expected = {
        pack_sop1(0, literal_pair, 255),
        kLiteralLo,
        pack_sop1(0, static_cast<uint8_t>(literal_pair + 1u), 255),
        kLiteralHi,
        build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4),
        pack_sopc_for_test(kOpSCmpLgU64, kSource, literal_pair),
    };
    lowering = find_word_sequence_for_test(cave_words, expected);
  }
  EXPECT_TRUE(lowering.has_value());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250SOrB64Literal64LowersInPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {
      pack_sop2(25, 4, 8, 254),
      0x00000000u,
      0x60000000u,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], pack_sop2(25, 4, 8, scalar_positive_inline_u32(0)));
  EXPECT_EQ(patched_text[1], pack_sop2(24, 5, 5, 255));
  EXPECT_EQ(patched_text[2], 0x60000000u);
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250SAddNcU64Literal32PreservesSccAroundNative) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SAddNcU64 = 83u;
  const std::array<uint32_t, 3> text_words = {
      pack_sop2(kGfx1250SAddNcU64, 2, 2, 255),
      0x00002000u,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], text_words[2]);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  std::optional<size_t> base;
  for (uint32_t tmp = 0; tmp <= 105 && !base; ++tmp) {
    if (tmp == 2 || tmp == 3)
      continue;
    const std::array<uint32_t, 5> expected = {
        pack_sop2(48, tmp, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0)),
        text_words[0],
        text_words[1],
        build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4),
        pack_sopc_for_test(7, tmp, scalar_positive_inline_u32(0)),
    };
    base = find_word_sequence_for_test(cave_words, expected);
  }
  ASSERT_TRUE(base.has_value());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250SSubNcU64InlinePreservesSccAroundNative) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SSubNcU64 = 84u;
  const std::array<uint32_t, 2> text_words = {
      pack_sop2(kGfx1250SSubNcU64, 4, 4, scalar_positive_inline_u32(8)),
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], text_words[1]);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  std::optional<size_t> base;
  for (uint32_t tmp = 0; tmp <= 105 && !base; ++tmp) {
    if (tmp == 4 || tmp == 5)
      continue;
    const std::array<uint32_t, 4> expected = {
        pack_sop2(48, tmp, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0)),
        text_words[0],
        build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4),
        pack_sopc_for_test(7, tmp, scalar_positive_inline_u32(0)),
    };
    base = find_word_sequence_for_test(cave_words, expected);
  }
  ASSERT_TRUE(base.has_value());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250SAddNcU64Lit64LowersToCarryOutSequence) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SAddNcU64 = 83u;
  const std::array<uint32_t, 4> text_words = {
      pack_sop2(kGfx1250SAddNcU64, 2, 2, 254),
      0xAF123456u,
      0x00000000u,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  std::optional<size_t> base;
  for (uint32_t tmp = 0; tmp < 128 && !base; ++tmp) {
    const std::array<uint32_t, 9> expected = {
        pack_sop2(48, tmp, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0)),
        pack_sop2(0, 2, 2, 255),
        0xAF123456u,
        build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
        build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4),
        pack_sop2(4, 3, 3, scalar_positive_inline_u32(0)),
        build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
        build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4),
        pack_sopc_for_test(7, tmp, scalar_positive_inline_u32(0)),
    };
    base = find_word_sequence_for_test(cave_words, expected);
  }
  ASSERT_TRUE(base.has_value());
  ASSERT_LT(*base + 9, cave_words.size());
  EXPECT_EQ((cave_words[*base + 9] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250SAndB64Literal64PadsShortInPlaceLowering) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {
      pack_sop2(23, 24, 22, 254),
      0x00000000u,
      0xFFFFFFFFu,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], pack_sop2(22, 24, 22, scalar_positive_inline_u32(0)));
  EXPECT_EQ(patched_text[1], pack_sop2(22, 25, 23, scalar_negative_inline_i32(-1)));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250SAndB64AddressMaskClearsRdna4HighOffset) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<std::array<uint32_t, 2>, 4> kAddressMasks = {{
      {0xFFFFF000u, 0x7FFFFFFFu},
      {0xFFFFE000u, 0x7FFFFFFFu},
      {0xFFFFC000u, 0x7FFFFFFFu},
      {0xFFFFC000u, 0x1FFFFFFFu},
  }};
  for (const auto [mask_lo, mask_hi] : kAddressMasks) {
    const std::array<uint32_t, 4> text_words = {
        pack_sop2(23, 2, 2, 254),
        mask_lo,
        mask_hi,
        kGfx1250SEndpgm,
    };

    auto image =
        make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
    AmdGpuCodeObject co(image.data(), image.size());
    ASSERT_TRUE(co.is_valid());

    BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                EF_AMDGPU_MACH_AMDGCN_GFX1201);
    auto result = translator.translate(co);
    ASSERT_FALSE(result.elf_bytes.empty());
    EXPECT_TRUE(result.warnings.empty());

    AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *text = translated.text_sections()[0];
    ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

    std::array<uint32_t, text_words.size()> patched_text{};
    std::memcpy(patched_text.data(), text->data(), text->size());
    EXPECT_EQ(patched_text[0], pack_sop2(22, 2, 2, 255));
    EXPECT_EQ(patched_text[1], mask_lo);
    EXPECT_EQ(patched_text[2], pack_sop1(0, 3, scalar_positive_inline_u32(0)));
    EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
    expect_gfx1250_rdna4_entry_stub_for_test(translated);
  }
}

TEST(BinaryTranslator, Gfx1250SAndB64AddressMaskNonInPlaceClearsRdna4HighOffset) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text_words = {
      pack_sop2(23, 4, 2, 254),
      0xFFFFE000u,
      0x7FFFFFFFu,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  std::vector<uint32_t> all_words = section_words_for_test(*translations);
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size() % sizeof(uint32_t), 0u);
  const size_t old_size = all_words.size();
  all_words.resize(old_size + text->size() / sizeof(uint32_t));
  std::memcpy(all_words.data() + old_size, text->data(), text->size());

  EXPECT_NE(std::ranges::find(all_words, 0xFFFFE000u), all_words.end());
  const std::array<uint32_t, 1> expected = {
      pack_sop1(0, 5, scalar_positive_inline_u32(0)),
  };
  ASSERT_TRUE(find_word_sequence_for_test(all_words, expected).has_value());
  const std::array<uint32_t, 2> stale_high_mask = {
      pack_sop2(22, 5, 3, 255),
      0x7FFFFFFFu,
  };
  EXPECT_FALSE(find_word_sequence_for_test(all_words, stale_high_mask).has_value());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250SAndB32AddressMaskPairClearsRdna4HighOffset) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<std::array<uint32_t, 2>, 3> kAddressMasks = {{
      {0xFFFFE000u, 0x7FFFFFFFu},
      {0xFFFFC000u, 0x7FFFFFFFu},
      {0xFFFFC000u, 0x1FFFFFFFu},
  }};
  for (const auto [mask_lo, mask_hi] : kAddressMasks) {
    const std::array<uint32_t, 5> text_words = {
        pack_sop2(22, 0, 8, 255), mask_lo, pack_sop2(22, 1, 9, 255), mask_hi, kGfx1250SEndpgm,
    };

    auto image =
        make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
    AmdGpuCodeObject co(image.data(), image.size());
    ASSERT_TRUE(co.is_valid());

    BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                EF_AMDGPU_MACH_AMDGCN_GFX1201);
    auto result = translator.translate(co);
    ASSERT_FALSE(result.elf_bytes.empty());
    EXPECT_TRUE(result.warnings.empty());

    AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *text = translated.text_sections()[0];
    ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

    std::vector<uint32_t> all_words;
    if (const Section *translations = find_section(translated, ".rj_translations"))
      all_words = section_words_for_test(*translations);
    const size_t old_size = all_words.size();
    all_words.resize(old_size + text->size() / sizeof(uint32_t));
    std::memcpy(all_words.data() + old_size, text->data(), text->size());

    EXPECT_NE(std::ranges::find(all_words, mask_lo), all_words.end());
    const std::array<uint32_t, 2> expected_high = {
        pack_sop1(0, 1, scalar_positive_inline_u32(0)),
        build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
    };
    EXPECT_TRUE(find_word_sequence_for_test(all_words, expected_high).has_value());
    const std::array<uint32_t, 2> stale_high_mask = {
        pack_sop2(22, 1, 9, 255),
        mask_hi,
    };
    EXPECT_FALSE(find_word_sequence_for_test(all_words, stale_high_mask).has_value());
    expect_gfx1250_rdna4_entry_stub_for_test(translated);
  }
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyLowersSAndB64Literal64) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;
  constexpr uint32_t kLiteralLo = 0xBF118008u;
  constexpr uint32_t kLiteralHi = 0xA9800610u;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = pack_sop2(23, 62, 22, 254);
  text_words[1] = kLiteralLo;
  text_words[2] = kLiteralHi;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 5 * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 5u);
  EXPECT_EQ(copied_words[0], pack_sop2(22, 62, 22, 255));
  EXPECT_EQ(copied_words[1], kLiteralLo);
  EXPECT_EQ(copied_words[2], pack_sop2(22, 63, 23, 255));
  EXPECT_EQ(copied_words[3], kLiteralHi);
  EXPECT_EQ(copied_words[4], build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyPreservesSccAroundNativeSAddNcU64) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint32_t kGfx1250SAddNcU64 = 83u;
  constexpr uint64_t kTextSize = 0x21000;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = pack_sop2(kGfx1250SAddNcU64, 2, 2, 255);
  text_words[1] = 0x00002000u;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 5u);
  std::optional<size_t> base;
  for (uint32_t tmp = 0; tmp <= 105 && !base; ++tmp) {
    if (tmp == 2 || tmp == 3)
      continue;
    const std::array<uint32_t, 5> expected = {
        pack_sop2(48, tmp, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0)),
        text_words[0],
        text_words[1],
        build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4),
        pack_sopc_for_test(7, tmp, scalar_positive_inline_u32(0)),
    };
    base = find_word_sequence_for_test(copied_words, expected);
  }
  ASSERT_TRUE(base.has_value());
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyInsertsRdna4ScalarDependencyWaits) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = 0xD45B0000u; // v_cmp_le_u64_e64 s0, s[8:9], s[18:19]
  text_words[1] = 0x02002408u;
  text_words[2] = pack_sop2(22, 4, 4, 0); // s_and_b32 s4, s4, s0
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 8 * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 8u);
  const uint32_t valu_sdst_wait = build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4);
  const uint32_t salu_scc_delay = build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4);
  const uint32_t scalar_consumer = pack_sop2(22, 4, 4, 0);
  EXPECT_TRUE(std::ranges::any_of(copied_words, [](uint32_t word) {
    return (word >> 26) == 0x35u && ((word >> 16) & 0x3FFu) == 91u;
  }));
  const auto scalar_it = std::ranges::find(copied_words, scalar_consumer);
  ASSERT_NE(scalar_it, copied_words.end());
  const auto scalar_index = static_cast<size_t>(std::distance(copied_words.begin(), scalar_it));
  ASSERT_GT(scalar_index, 0u);
  ASSERT_LT(scalar_index + 3u, copied_words.size());
  EXPECT_EQ(copied_words[scalar_index - 1u], valu_sdst_wait);
  EXPECT_EQ(copied_words[scalar_index + 1u], salu_scc_delay);
  EXPECT_EQ(copied_words[scalar_index + 2u],
            build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(copied_words[scalar_index + 3u], kGfx1250SNop0);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyDelaysAfterSccDefiningScaledAdd) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  const uint32_t scaled_add = pack_sop2(15, 0, 55, 0); // s_lshl2_add_u32 s0, s55, s0
  const uint32_t carry_in =
      pack_sop2(4, 1, 1, scalar_positive_inline_u32(0)); // s_add_co_ci_u32 s1, s1, 0
  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = scaled_add;
  text_words[1] = carry_in;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  const uint32_t salu_scc_delay = build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4);
  const uint32_t salu_sgpr_wait = build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4);
  const std::array<uint32_t, 6> expected = {
      scaled_add, salu_scc_delay, salu_sgpr_wait, carry_in, salu_scc_delay, salu_sgpr_wait,
  };
  EXPECT_TRUE(find_word_sequence_for_test(copied_words, expected).has_value());
}

TEST(BinaryTranslator, Gfx1250WaitXcntLowersToRdna4MemoryCounterDrain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint32_t kGfx1250SWaitXcnt2 = 0xBFC50002u;
  constexpr uint64_t kTextSize = 0x21000;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SWaitXcnt2;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 5 * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 5u);
  EXPECT_EQ(copied_words[0], pack_sopp(64, 0));
  EXPECT_EQ(copied_words[1], pack_sopp(65, 0));
  EXPECT_EQ(copied_words[2], pack_sopp(70, 0));
  EXPECT_EQ(copied_words[3], pack_sopp(71, 0));
  EXPECT_EQ(copied_words[4], kGfx1250SNop0);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyUsesVccWaitAfterE32Compare) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;
  constexpr uint32_t kCmpNeU64E32 = 0x7CBA091Au; // v_cmp_ne_u64_e32 vcc_lo, v[26:27], v[4:5]
  constexpr uint32_t kSAndVcc = 0x8B006A00u;     // s_and_b32 s0, s0, vcc_lo

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kCmpNeU64E32;
  text_words[1] = kSAndVcc;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 6u * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 6u);
  const auto scalar_it = std::ranges::find(copied_words, kSAndVcc);
  ASSERT_NE(scalar_it, copied_words.end());
  const auto scalar_index = static_cast<size_t>(std::distance(copied_words.begin(), scalar_it));
  ASSERT_GT(scalar_index, 0u);
  ASSERT_LT(scalar_index + 2u, copied_words.size());
  EXPECT_EQ(copied_words[scalar_index - 1u],
            build_s_wait_alu(kWaitAluDepctrVaVcc0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(copied_words[scalar_index + 1u],
            build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(copied_words[scalar_index + 2u],
            build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyUsesSdstWaitAfterSgprCompare) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;
  constexpr uint32_t kSAndVccFromCmp =
      pack_sop2(22, 106, 126, 12); // s_and_b32 vcc_lo, exec_lo, s12

  gfx1250::Vop3MachineInst cmp{};
  cmp.encoding = 0x35;
  cmp.op = 86; // v_cmp_ge_i64 s12, s[6:7], s[12:13]
  cmp.vdst = 12;
  cmp.src0 = 6;
  cmp.src1 = 12;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  std::memcpy(text_words.data(), &cmp, sizeof(cmp));
  text_words[2] = kSAndVccFromCmp;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 7u * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  const auto scalar_it = std::ranges::find(copied_words, kSAndVccFromCmp);
  ASSERT_NE(scalar_it, copied_words.end());
  const auto scalar_index = static_cast<size_t>(std::distance(copied_words.begin(), scalar_it));
  ASSERT_GT(scalar_index, 0u);
  ASSERT_LT(scalar_index + 1u, copied_words.size());
  EXPECT_EQ(copied_words[scalar_index - 1u],
            build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(copied_words[scalar_index + 1u],
            build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyUsesSdstWaitAfterCmpx) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;
  constexpr uint32_t kCmpxNeU32E32 = 0x7D9A1A80u; // v_cmpx_ne_u32_e32 0, v13

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kCmpxNeU32E32;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 5u * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 3u);
  const auto cmpx_it = std::ranges::find(copied_words, kCmpxNeU32E32);
  ASSERT_NE(cmpx_it, copied_words.end());
  const auto cmpx_index = static_cast<size_t>(std::distance(copied_words.begin(), cmpx_it));
  ASSERT_LT(cmpx_index + 1u, copied_words.size());
  EXPECT_EQ(copied_words[cmpx_index + 1u],
            build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyWaitsAfterReadfirstlaneScalarResult) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;
  constexpr uint32_t kReadfirstlaneS2V2 = 0x7E040502u;     // v_readfirstlane_b32 s2, v2
  const uint32_t scalar_consumer = pack_sop2(22, 4, 4, 2); // s_and_b32 s4, s4, s2

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kReadfirstlaneS2V2;
  text_words[1] = scalar_consumer;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 8 * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 8u);
  const auto scalar_it = std::ranges::find(copied_words, scalar_consumer);
  ASSERT_NE(scalar_it, copied_words.end());
  const auto scalar_index = static_cast<size_t>(std::distance(copied_words.begin(), scalar_it));
  ASSERT_GT(scalar_index, 0u);
  EXPECT_EQ(copied_words[scalar_index - 1u],
            build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VMulU64E32LowersFullLow64Product) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto build_vop2 = [](uint8_t op, uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
    return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
  };

  const std::array<uint32_t, 2> text_words = {
      build_vop2(42, 8, 10, 6), // v_mul_u64_e32 v[8:9], s[10:11], v[6:7]
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);

  EXPECT_TRUE(section_contains_vop3_inst(translations, 813, 9, 10, 256 + 6))
      << "high product must start with mul_hi(src0.lo, src1.lo)";
  EXPECT_TRUE(section_contains_vop3_inst(translations, 812, 8, 11, 256 + 6))
      << "high product must include src0.hi * src1.lo";
  EXPECT_TRUE(section_contains_vop3_inst(translations, 812, 8, 10, 256 + 7))
      << "high product must include src0.lo * src1.hi";
  EXPECT_TRUE(section_contains_vop3_inst(translations, 812, 8, 10, 256 + 6))
      << "low word must still be src0.lo * src1.lo";
}

TEST(BinaryTranslator, Gfx1250VMulU64E32LowersLiteral64WithDistinctHighWord) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kLiteralLo = 0x6A09D977u;
  constexpr uint32_t kLiteralHi = 0x00000001u;
  const std::array<uint32_t, 4> text_words = {
      0x540000FEu, // v_mul_u64_e32 v[0:1], 0x16a09d977, v[0:1]
      kLiteralLo,
      kLiteralHi,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);

  EXPECT_TRUE(
      section_contains_vop3_inst_literal(translations, 813, std::nullopt, 255, 256, kLiteralLo))
      << "mul_hi must use the literal low word";
  EXPECT_TRUE(
      section_contains_vop3_inst_literal(translations, 812, std::nullopt, 255, 256, kLiteralHi))
      << "the first cross product must use the literal high word";
  EXPECT_TRUE(
      section_contains_vop3_inst_literal(translations, 812, std::nullopt, 255, 257, kLiteralLo))
      << "the second cross product must use the literal low word";
  EXPECT_TRUE(
      section_contains_vop3_inst_literal(translations, 812, std::nullopt, 255, 256, kLiteralLo))
      << "the low product must use the literal low word";
}

TEST(BinaryTranslator, Gfx1250VMulU64E32UsesScratchForDestinationSourceAlias) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto build_vop2 = [](uint8_t op, uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
    return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
  };

  const std::array<uint32_t, 2> text_words = {
      build_vop2(42, 6, 26, 6), // v_mul_u64_e32 v[6:7], s[26:27], v[6:7]
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  EXPECT_FALSE(section_contains_vop3_opcode(translations, 0))
      << "gfx1250 v_mul_u64 must be fully expanded";

  EXPECT_TRUE(section_contains_vop3_inst(translations, 812, 0, 27, 256 + 6))
      << "scratch high product must include src0.hi * original src1.lo";
  EXPECT_TRUE(section_contains_vop3_inst(translations, 812, 0, 26, 256 + 7))
      << "scratch high product must include src0.lo * original src1.hi";

  const auto low_copy_src = section_vop1_src_for_dst(translations, 1, 6);
  const auto high_copy_src = section_vop1_src_for_dst(translations, 1, 7);
  ASSERT_TRUE(low_copy_src);
  ASSERT_TRUE(high_copy_src);
  ASSERT_GE(*low_copy_src, 256u);
  ASSERT_GE(*high_copy_src, 256u);
  EXPECT_EQ(*high_copy_src, static_cast<uint16_t>(*low_copy_src + 1u));
  EXPECT_NE(*low_copy_src, 256u + 6u);
  EXPECT_NE(*low_copy_src, 256u + 7u);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyRaisesDescriptorForLoweredHighVgprDestinations) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  const auto build_vop2 = [](uint8_t op, uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
    return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
  };

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = build_vop2(42, 128, 6, 128); // v_mul_u64_e32 v[128:129], s6, v128
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];

  KernelDescriptorTranslator descriptor_parser(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto descriptor_infos = descriptor_parser.translate_image(
      result.elf_bytes, text->sectionOffset(), text->size(), KernelDescriptorTranslationOptions{});
  ASSERT_EQ(descriptor_infos.size(), 1u);
  EXPECT_GE(descriptor_infos[0].guest_vgpr_count, 130u)
      << "lowered v_mul_u64 writes vdst+1, so the translated descriptor must "
         "cover both high destination lanes";
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyRaisesDescriptorForRelativeAccessScratch) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250GlobalLoadB32ScaleW0 = 0xEE050004u;
  constexpr uint32_t kGfx1250GlobalLoadB32ScaleW1 = 0x00010001u;
  constexpr uint32_t kGfx1250GlobalLoadB32ScaleW2 = 0x00000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop1MachineInst movrels{};
  movrels.encoding = 0x3F;
  movrels.op = 67;
  movrels.vdst = 20;
  movrels.src0 = 256 + 4;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250GlobalLoadB32ScaleW0;
  text_words[1] = kGfx1250GlobalLoadB32ScaleW1;
  text_words[2] = kGfx1250GlobalLoadB32ScaleW2;
  text_words[3] = std::bit_cast<uint32_t>(movrels);
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *desc = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 7);

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_GT(text->size(), text_words.size() * sizeof(uint32_t));

  KernelDescriptorTranslator descriptor_parser(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto descriptor_infos = descriptor_parser.translate_image(
      result.elf_bytes, text->sectionOffset(), text->size(), KernelDescriptorTranslationOptions{});
  ASSERT_EQ(descriptor_infos.size(), 1u);
  EXPECT_GE(descriptor_infos[0].guest_vgpr_count, 129u)
      << "the expanded scale_offset lowering uses v128 after M0-relative access reserves the "
         "128-register source file";
}

TEST(BinaryTranslator, Gfx1250Vop3CndmaskDoesNotHitRawVMulU64E32Fallback) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {
      0xD5010006u, // v_cndmask_b32_e64 v6, 0, 1, s2
      0x00090280u,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  rdna4::Vop3MachineInst cndmask{};
  std::memcpy(&cndmask, patched_text.data(), sizeof(cndmask));
  EXPECT_EQ(cndmask.encoding, 0x35u);
  EXPECT_EQ(cndmask.op, 257u);
  EXPECT_EQ(cndmask.vdst, 6u);
  EXPECT_EQ(cndmask.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(cndmask.src1, scalar_positive_inline_u32(1));
  EXPECT_EQ(cndmask.src2, 2u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250Vop3VMadNcU64U32LowersToRdna4MadCoNullCarry) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 10;
  inst.op = 762;
  inst.encoding = 0x35;
  inst.src0 = 256 + 14;
  inst.src1 = 24;
  inst.src2 = 256 + 10;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  rdna4::Vop3SdstEncMachineInst mad{};
  std::memcpy(&mad, patched_text.data(), sizeof(mad));
  EXPECT_EQ(mad.encoding, 0x35u);
  EXPECT_EQ(mad.op, 766u);
  EXPECT_EQ(mad.vdst, 10u);
  EXPECT_EQ(mad.sdst, 124u);
  EXPECT_EQ(mad.src0, 256u + 14u);
  EXPECT_EQ(mad.src1, 24u);
  EXPECT_EQ(mad.src2, 256u + 10u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250Vop3VMadNcI64I32LowersToRdna4MadCoNullCarry) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text_words = {
      0xD6FB0010u,
      0x00297102u, // v_mad_nc_i64_i32 v[16:17], v2, 56, s[10:11]
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  rdna4::Vop3SdstEncMachineInst mad{};
  std::memcpy(&mad, patched_text.data(), sizeof(mad));
  EXPECT_EQ(mad.encoding, 0x35u);
  EXPECT_EQ(mad.op, 767u);
  EXPECT_EQ(mad.vdst, 16u);
  EXPECT_EQ(mad.sdst, 124u);
  EXPECT_EQ(mad.src0, 256u + 2u);
  EXPECT_EQ(mad.src1, scalar_positive_inline_u32(56));
  EXPECT_EQ(mad.src2, 10u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250Vop3VMadU32PreservesAddendWithScratch) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 11;
  inst.op = 565;
  inst.encoding = 0x35;
  inst.src0 = 256 + 15;
  inst.src1 = 24;
  inst.src2 = 256 + 11;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<size_t> base;
  for (size_t i = 0; i + 5u < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == 812u && candidate.src0 == 256u + 15u &&
        candidate.src1 == 24u) {
      base = i;
      break;
    }
  }
  ASSERT_TRUE(base.has_value());

  rdna4::Vop3MachineInst mul{};
  std::memcpy(&mul, cave_words.data() + *base, sizeof(mul));
  EXPECT_EQ(mul.encoding, 0x35u);
  EXPECT_EQ(mul.op, 812u);
  EXPECT_NE(mul.vdst, 11u);
  EXPECT_NE(mul.vdst, 15u);
  EXPECT_EQ(mul.src0, 256u + 15u);
  EXPECT_EQ(mul.src1, 24u);

  EXPECT_EQ(cave_words[*base + 2u], build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3MachineInst add{};
  std::memcpy(&add, cave_words.data() + *base + 3u, sizeof(add));
  EXPECT_EQ(add.encoding, 0x35u);
  EXPECT_EQ(add.op, 293u);
  EXPECT_EQ(add.vdst, 11u);
  EXPECT_EQ(add.src0, 256u + 11u);
  EXPECT_EQ(add.src1, 256u + mul.vdst);
  EXPECT_EQ((cave_words[*base + 5u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250Vop3VMadU32LiteralPreservesAddendWithScratch) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 1;
  inst.op = 565;
  inst.encoding = 0x35;
  inst.src0 = 255;
  inst.src1 = 256 + 2;
  inst.src2 = 256 + 1;

  std::array<uint32_t, 4> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = 0x1800;
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  EXPECT_FALSE(section_contains_vop3_opcode(translations, 0))
      << "gfx1250 v_mad_u32 must not remain as raw target VOP3 opcode 0";

  std::optional<size_t> base;
  for (size_t i = 0; i + 6u < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == 812u && candidate.src0 == 255u &&
        candidate.src1 == 256u + 2u) {
      base = i;
      break;
    }
  }
  ASSERT_TRUE(base.has_value());

  rdna4::Vop3MachineInst mul{};
  std::memcpy(&mul, cave_words.data() + *base, sizeof(mul));
  EXPECT_EQ(mul.encoding, 0x35u);
  EXPECT_EQ(mul.op, 812u);
  EXPECT_NE(mul.vdst, 1u);
  EXPECT_NE(mul.vdst, 2u);
  EXPECT_EQ(mul.src0, 255u);
  EXPECT_EQ(mul.src1, 256u + 2u);
  EXPECT_EQ(cave_words[*base + 2u], 0x1800u);
  EXPECT_EQ(cave_words[*base + 3u], build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3MachineInst add{};
  std::memcpy(&add, cave_words.data() + *base + 4u, sizeof(add));
  EXPECT_EQ(add.encoding, 0x35u);
  EXPECT_EQ(add.op, 293u);
  EXPECT_EQ(add.vdst, 1u);
  EXPECT_EQ(add.src0, 256u + 1u);
  EXPECT_EQ(add.src1, 256u + mul.vdst);
  EXPECT_EQ((cave_words[*base + 6u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250Vop3VMadU32LiteralAddendIsEncodedOnRdna4Add) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kLiteralAddend = 0x9E37DC32u;
  constexpr std::array<uint32_t, 4> text_words = {0xD6350001u, 0x03FC090Cu, kLiteralAddend,
                                                  kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  EXPECT_FALSE(section_contains_vop3_opcode(translations, 565))
      << "gfx1250 v_mad_u32 with a literal addend must be expanded";

  std::optional<size_t> add_index;
  for (size_t i = 0; i + 2u < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == 293u && candidate.vdst == 1u &&
        candidate.src0 == 255u) {
      add_index = i;
      break;
    }
  }
  ASSERT_TRUE(add_index.has_value());
  EXPECT_EQ(cave_words[*add_index + 2u], kLiteralAddend);
  EXPECT_EQ((cave_words[*add_index + 3u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250Vop3VSubNcU64LowersToCarryChain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 20;
  inst.op = 297;
  inst.encoding = 0x35;
  inst.src0 = 256 + 6;
  inst.src1 = 256 + 8;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  expect_sub_u64_carry_chain(translated, 20, 256 + 6, 256 + 8);
}

TEST(BinaryTranslator, Gfx1250Vop3VAddNcU64LowersToCarryChain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 20;
  inst.op = 296;
  inst.encoding = 0x35;
  inst.src0 = 256 + 6;
  inst.src1 = 256 + 8;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  expect_add_u64_carry_chain(translated, 20, 256 + 6, 256 + 8);
}

TEST(BinaryTranslator, Gfx1250VLshlAddU64ConstShiftLowersToCarryChain) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 10;
  inst.op = 594;
  inst.encoding = 0x35;
  inst.src0 = 256 + 2;
  inst.src1 = scalar_positive_inline_u32(4);
  inst.src2 = 6;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 11 <= cave_words.size(); ++i) {
    rdna4::Vop3MachineInst align_hi{};
    std::memcpy(&align_hi, cave_words.data() + i, sizeof(align_hi));
    rdna4::Vop3MachineInst shl_lo{};
    std::memcpy(&shl_lo, cave_words.data() + i + 2, sizeof(shl_lo));
    rdna4::Vop3SdstEncMachineInst add_lo{};
    std::memcpy(&add_lo, cave_words.data() + i + 5, sizeof(add_lo));
    rdna4::Vop3SdstEncMachineInst add_hi{};
    std::memcpy(&add_hi, cave_words.data() + i + 8, sizeof(add_hi));
    if (align_hi.op == 534u && align_hi.vdst == 11u && align_hi.src0 == 256u + 3u &&
        align_hi.src1 == 256u + 2u && align_hi.src2 == scalar_positive_inline_u32(28) &&
        shl_lo.op == 280u && shl_lo.vdst == 10u && shl_lo.src0 == scalar_positive_inline_u32(4) &&
        shl_lo.src1 == 256u + 2u &&
        cave_words[i + 4] == build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4) && add_lo.op == 768u &&
        add_lo.vdst == 10u && add_lo.sdst <= 105u && (add_lo.sdst & 1u) == 0u &&
        add_lo.src0 == 256u + 10u && add_lo.src1 == 6u &&
        cave_words[i + 7] == build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4) &&
        add_hi.op == 288u && add_hi.vdst == 11u && add_hi.sdst == kNullSgprForTest &&
        add_hi.src0 == 256u + 11u && add_hi.src1 == 7u && add_hi.src2 == add_lo.sdst &&
        ((cave_words[i + 10] >> 16) & 0x7Fu) == sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250VLshlAddU32DstAddendUsesScratchAndWait) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 5;
  inst.op = 582;
  inst.encoding = 0x35;
  inst.src0 = 256 + 3;
  inst.src1 = scalar_positive_inline_u32(11);
  inst.src2 = 256 + 5;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 6 <= cave_words.size(); ++i) {
    rdna4::Vop3MachineInst shl{};
    std::memcpy(&shl, cave_words.data() + i, sizeof(shl));
    rdna4::Vop3MachineInst add{};
    std::memcpy(&add, cave_words.data() + i + 3, sizeof(add));
    if (shl.op == 280u && shl.vdst != 5u && shl.vdst != 3u &&
        shl.src0 == scalar_positive_inline_u32(11) && shl.src1 == 256u + 3u &&
        cave_words[i + 2] == build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4) && add.op == 293u &&
        add.vdst == 5u && add.src0 == 256u + 5u && add.src1 == 256u + shl.vdst &&
        ((cave_words[i + 5] >> 16) & 0x7Fu) == sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250VLshlOrB32DstOrSourceUsesScratchAndDelay) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 5;
  inst.op = 598;
  inst.encoding = 0x35;
  inst.src0 = 256 + 3;
  inst.src1 = scalar_positive_inline_u32(3);
  inst.src2 = 256 + 5;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 5 <= cave_words.size(); ++i) {
    rdna4::Vop3MachineInst shl{};
    std::memcpy(&shl, cave_words.data() + i, sizeof(shl));
    const auto or_inst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[i + 3]);
    if (shl.op == 280u && shl.vdst != 5u && shl.vdst != 3u &&
        shl.src0 == scalar_positive_inline_u32(3) && shl.src1 == 256u + 3u &&
        cave_words[i + 2] == build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4) && or_inst.op == 28u &&
        or_inst.vdst == 5u && or_inst.src0 == 256u + 5u && or_inst.vsrc1 == shl.vdst &&
        ((cave_words[i + 4] >> 16) & 0x7Fu) == sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250VLshlrevB64E32ConstShiftLowersToB32Sequence) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop2MachineInst inst{};
  inst.vdst = 10;
  inst.op = 31;
  inst.src0 = scalar_positive_inline_u32(12);
  inst.vsrc1 = 10;
  const std::array<uint32_t, 2> text_words = {std::bit_cast<uint32_t>(inst), kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 5 <= cave_words.size(); ++i) {
    rdna4::Vop3MachineInst hi{};
    std::memcpy(&hi, cave_words.data() + i, sizeof(hi));
    rdna4::Vop3MachineInst lo{};
    std::memcpy(&lo, cave_words.data() + i + 2, sizeof(lo));
    if (hi.op == 534u && hi.vdst == 11u && hi.src0 == 256u + 11u && hi.src1 == 256u + 10u &&
        hi.src2 == scalar_positive_inline_u32(20) && lo.op == 280u && lo.vdst == 10u &&
        lo.src0 == scalar_positive_inline_u32(12) && lo.src1 == 256u + 10u &&
        ((cave_words[i + 4] >> 16) & 0x7Fu) == sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250VMadNcU64U32LiteralMaterializesSgpr) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 10;
  inst.op = 762;
  inst.encoding = 0x35;
  inst.src0 = 255;
  inst.src1 = 256 + 2;
  inst.src2 = 256 + 4;

  std::array<uint32_t, 4> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = 0x800;
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 6 <= cave_words.size(); ++i) {
    rdna4::Vop3SdstEncMachineInst mad{};
    std::memcpy(&mad, cave_words.data() + i + 3, sizeof(mad));
    if (cave_words[i + 2] == build_s_wait_alu(kWaitAluDepctrSaSdst0, ROCJITSU_CODE_ARCH_RDNA4) &&
        mad.op == 766u && mad.vdst == 10u && mad.sdst == kNullSgprForTest && mad.src0 <= 105u &&
        mad.src1 == 256u + 2u && mad.src2 == 256u + 4u &&
        ((cave_words[i + 5] >> 16) & 0x7Fu) == sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4)) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250VMaxU64LowersToCompareSelect) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 12;
  inst.op = 793;
  inst.encoding = 0x35;
  inst.src0 = 256 + 4;
  inst.src1 = scalar_positive_inline_u32(1);

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  auto cave_words = section_words_for_test(*translations);
  auto cmp_it = std::ranges::find_if(cave_words, [](uint32_t word) {
    return (word >> 26) == 0x35u && ((word >> 16) & 0x3FFu) == 92u;
  });
  ASSERT_NE(cmp_it, cave_words.end());
  const auto cmp_index = static_cast<size_t>(std::distance(cave_words.begin(), cmp_it));
  ASSERT_LT(cmp_index + 8u, cave_words.size());

  rdna4::Vop3MachineInst cmp{};
  std::memcpy(&cmp, cave_words.data() + cmp_index, sizeof(cmp));
  EXPECT_EQ(cmp.encoding, 0x35u);
  EXPECT_EQ(cmp.op, 92u);
  EXPECT_LE(cmp.vdst, 105u);
  EXPECT_EQ(cmp.vdst & 1u, 0u);
  EXPECT_EQ(cmp.src0, 256u + 4u);
  EXPECT_EQ(cmp.src1, scalar_positive_inline_u32(1));
  EXPECT_EQ(cave_words[cmp_index + 2u],
            build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3MachineInst select_lo{};
  std::memcpy(&select_lo, cave_words.data() + cmp_index + 3u, sizeof(select_lo));
  EXPECT_EQ(select_lo.op, 257u);
  EXPECT_EQ(select_lo.vdst, 12u);
  EXPECT_EQ(select_lo.src0, scalar_positive_inline_u32(1));
  EXPECT_EQ(select_lo.src1, 256u + 4u);
  EXPECT_EQ(select_lo.src2, cmp.vdst);

  rdna4::Vop3MachineInst select_hi{};
  std::memcpy(&select_hi, cave_words.data() + cmp_index + 5u, sizeof(select_hi));
  EXPECT_EQ(select_hi.op, 257u);
  EXPECT_EQ(select_hi.vdst, 13u);
  EXPECT_EQ(select_hi.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(select_hi.src1, 256u + 5u);
  EXPECT_EQ(select_hi.src2, cmp.vdst);
  EXPECT_EQ(cave_words[cmp_index + 7u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250RemainingMinMax64VariantsLowerToCompareSelect) {
  struct Case {
    uint16_t source_opcode;
    uint16_t compare_opcode;
  };
  constexpr Case cases[] = {
      {792, 89}, // v_min_u64 -> v_cmp_lt_u64
      {795, 84}, // v_max_i64 -> v_cmp_gt_i64
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const auto &test_case : cases) {
    SCOPED_TRACE(testing::Message() << "source opcode " << test_case.source_opcode);
    gfx1250::Vop3MachineInst inst{};
    inst.vdst = 12;
    inst.op = test_case.source_opcode;
    inst.encoding = 0x35;
    inst.src0 = 256 + 4;
    inst.src1 = 256 + 6;

    std::array<uint32_t, 3> text_words{};
    std::memcpy(text_words.data(), &inst, sizeof(inst));
    text_words[2] = kGfx1250SEndpgm;
    auto image =
        make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
    AmdGpuCodeObject co(image.data(), image.size());
    ASSERT_TRUE(co.is_valid());

    BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                EF_AMDGPU_MACH_AMDGCN_GFX1201);
    auto result = translator.translate(co);
    ASSERT_FALSE(result.elf_bytes.empty());
    EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

    AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    const Section *translations = find_section(translated, ".rj_translations");
    ASSERT_NE(translations, nullptr);
    const auto cave_words = section_words_for_test(*translations);
    const auto cmp = std::ranges::find_if(cave_words, [&](uint32_t word) {
      return (word >> 26u) == 0x35u && ((word >> 16u) & 0x3FFu) == test_case.compare_opcode;
    });
    ASSERT_NE(cmp, cave_words.end());
  }
}

TEST(BinaryTranslator, Gfx1250VMinI64LowersToCompareSelect) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 20;
  inst.op = 794;
  inst.encoding = 0x35;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 10;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  auto cave_words = section_words_for_test(*translations);
  auto cmp_it = std::ranges::find_if(cave_words, [](uint32_t word) {
    return (word >> 26) == 0x35u && ((word >> 16) & 0x3FFu) == 81u;
  });
  ASSERT_NE(cmp_it, cave_words.end());
  const auto cmp_index = static_cast<size_t>(std::distance(cave_words.begin(), cmp_it));
  ASSERT_LT(cmp_index + 8u, cave_words.size());

  rdna4::Vop3MachineInst cmp{};
  std::memcpy(&cmp, cave_words.data() + cmp_index, sizeof(cmp));
  EXPECT_EQ(cmp.encoding, 0x35u);
  EXPECT_EQ(cmp.op, 81u);
  EXPECT_LE(cmp.vdst, 105u);
  EXPECT_EQ(cmp.vdst & 1u, 0u);
  EXPECT_EQ(cmp.src0, 256u + 8u);
  EXPECT_EQ(cmp.src1, 256u + 10u);
  EXPECT_EQ(cave_words[cmp_index + 2u],
            build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3MachineInst select_lo{};
  std::memcpy(&select_lo, cave_words.data() + cmp_index + 3u, sizeof(select_lo));
  EXPECT_EQ(select_lo.op, 257u);
  EXPECT_EQ(select_lo.vdst, 20u);
  EXPECT_EQ(select_lo.src0, 256u + 10u);
  EXPECT_EQ(select_lo.src1, 256u + 8u);
  EXPECT_EQ(select_lo.src2, cmp.vdst);

  rdna4::Vop3MachineInst select_hi{};
  std::memcpy(&select_hi, cave_words.data() + cmp_index + 5u, sizeof(select_hi));
  EXPECT_EQ(select_hi.op, 257u);
  EXPECT_EQ(select_hi.vdst, 21u);
  EXPECT_EQ(select_hi.src0, 256u + 11u);
  EXPECT_EQ(select_hi.src1, 256u + 9u);
  EXPECT_EQ(select_hi.src2, cmp.vdst);
  EXPECT_EQ(cave_words[cmp_index + 7u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VMinI64ScalarSourcesStageThroughVgprs) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 4;
  inst.op = 794;
  inst.encoding = 0x35;
  inst.src0 = 18;
  inst.src1 = 8;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  auto cave_words = section_words_for_test(*translations);
  auto cmp_it = std::ranges::find_if(cave_words, [](uint32_t word) {
    return (word >> 26) == 0x35u && ((word >> 16) & 0x3FFu) == 81u;
  });
  ASSERT_NE(cmp_it, cave_words.end());
  const auto cmp_index = static_cast<size_t>(std::distance(cave_words.begin(), cmp_it));
  ASSERT_GE(cmp_index, 5u);
  ASSERT_LT(cmp_index + 8u, cave_words.size());

  rdna4::Vop1MachineInst mov_src0_lo{};
  std::memcpy(&mov_src0_lo, cave_words.data() + cmp_index - 5u, sizeof(mov_src0_lo));
  EXPECT_EQ(mov_src0_lo.op, 1u);
  EXPECT_EQ(mov_src0_lo.src0, 18u);

  rdna4::Vop1MachineInst mov_src0_hi{};
  std::memcpy(&mov_src0_hi, cave_words.data() + cmp_index - 4u, sizeof(mov_src0_hi));
  EXPECT_EQ(mov_src0_hi.op, 1u);
  EXPECT_EQ(mov_src0_hi.vdst, mov_src0_lo.vdst + 1u);
  EXPECT_EQ(mov_src0_hi.src0, 19u);

  rdna4::Vop1MachineInst mov_src1_lo{};
  std::memcpy(&mov_src1_lo, cave_words.data() + cmp_index - 3u, sizeof(mov_src1_lo));
  EXPECT_EQ(mov_src1_lo.op, 1u);
  EXPECT_EQ(mov_src1_lo.src0, 8u);

  rdna4::Vop1MachineInst mov_src1_hi{};
  std::memcpy(&mov_src1_hi, cave_words.data() + cmp_index - 2u, sizeof(mov_src1_hi));
  EXPECT_EQ(mov_src1_hi.op, 1u);
  EXPECT_EQ(mov_src1_hi.vdst, mov_src1_lo.vdst + 1u);
  EXPECT_EQ(mov_src1_hi.src0, 9u);

  EXPECT_EQ(cave_words[cmp_index - 1u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3MachineInst cmp{};
  std::memcpy(&cmp, cave_words.data() + cmp_index, sizeof(cmp));
  EXPECT_EQ(cmp.src0, 256u + mov_src0_lo.vdst);
  EXPECT_EQ(cmp.src1, 256u + mov_src1_lo.vdst);

  rdna4::Vop3MachineInst select_lo{};
  std::memcpy(&select_lo, cave_words.data() + cmp_index + 3u, sizeof(select_lo));
  EXPECT_EQ(select_lo.op, 257u);
  EXPECT_EQ(select_lo.src0, cmp.src1);
  EXPECT_EQ(select_lo.src1, cmp.src0);
  EXPECT_EQ(select_lo.src2, cmp.vdst);

  rdna4::Vop3MachineInst select_hi{};
  std::memcpy(&select_hi, cave_words.data() + cmp_index + 5u, sizeof(select_hi));
  EXPECT_EQ(select_hi.op, 257u);
  EXPECT_EQ(select_hi.src0, cmp.src1 + 1u);
  EXPECT_EQ(select_hi.src1, cmp.src0 + 1u);
  EXPECT_EQ(select_hi.src2, cmp.vdst);
}

TEST(BinaryTranslator, Gfx1250VMinI64DoesNotClobberAliasedVectorSource) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 2;
  inst.op = 794;
  inst.encoding = 0x35;
  inst.src0 = 2;
  inst.src1 = 256 + 2;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;
  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto cmp_it = std::ranges::find_if(cave_words, [](uint32_t word) {
    return (word >> 26u) == 0x35u && ((word >> 16u) & 0x3FFu) == 81u;
  });
  ASSERT_NE(cmp_it, cave_words.end());
  const size_t cmp_index = static_cast<size_t>(std::distance(cave_words.begin(), cmp_it));
  ASSERT_GE(cmp_index, 5u);

  rdna4::Vop3MachineInst cmp{};
  std::memcpy(&cmp, cave_words.data() + cmp_index, sizeof(cmp));
  EXPECT_NE(cmp.src0, cmp.src1);
  EXPECT_EQ(cmp.src1, 256u + 2u);
  EXPECT_NE(cmp.src0, 256u + 2u);
}

TEST(BinaryTranslator, Gfx1250VMinI64LiteralLowersToCompareSelect) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 2;
  inst.op = 794;
  inst.encoding = 0x35;
  inst.src0 = 255;
  inst.src1 = 256 + 24;

  std::array<uint32_t, 4> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = 0x1000;
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  auto cave_words = section_words_for_test(*translations);
  auto cmp_it = std::ranges::find_if(cave_words, [](uint32_t word) {
    return (word >> 26) == 0x35u && ((word >> 16) & 0x3FFu) == 81u;
  });
  ASSERT_NE(cmp_it, cave_words.end());
  const auto cmp_index = static_cast<size_t>(std::distance(cave_words.begin(), cmp_it));
  ASSERT_LT(cmp_index + 10u, cave_words.size());

  rdna4::Vop3MachineInst cmp{};
  std::memcpy(&cmp, cave_words.data() + cmp_index, sizeof(cmp));
  EXPECT_EQ(cmp.encoding, 0x35u);
  EXPECT_EQ(cmp.op, 81u);
  EXPECT_LE(cmp.vdst, 105u);
  EXPECT_EQ(cmp.vdst & 1u, 0u);
  EXPECT_EQ(cmp.src0, 255u);
  EXPECT_EQ(cmp.src1, 256u + 24u);
  EXPECT_EQ(cave_words[cmp_index + 2u], 0x1000u);
  EXPECT_EQ(cave_words[cmp_index + 3u],
            build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3MachineInst select_lo{};
  std::memcpy(&select_lo, cave_words.data() + cmp_index + 4u, sizeof(select_lo));
  EXPECT_EQ(select_lo.op, 257u);
  EXPECT_EQ(select_lo.vdst, 2u);
  EXPECT_EQ(select_lo.src0, 256u + 24u);
  EXPECT_EQ(select_lo.src1, 255u);
  EXPECT_EQ(select_lo.src2, cmp.vdst);
  EXPECT_EQ(cave_words[cmp_index + 6u], 0x1000u);

  rdna4::Vop3MachineInst select_hi{};
  std::memcpy(&select_hi, cave_words.data() + cmp_index + 7u, sizeof(select_hi));
  EXPECT_EQ(select_hi.op, 257u);
  EXPECT_EQ(select_hi.vdst, 3u);
  EXPECT_EQ(select_hi.src0, 256u + 25u);
  EXPECT_EQ(select_hi.src1, scalar_positive_inline_u32(0));
  EXPECT_EQ(select_hi.src2, cmp.vdst);
  EXPECT_EQ(cave_words[cmp_index + 9u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VPkAddF32LowersToTwoVAddF32s) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 41;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 30;
  inst.src1 = 256 + 34;
  inst.src2 = scalar_positive_inline_u32(0);
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base = find_vop3_for_test(cave_words, 259u, 42u, 256u + 30u, 256u + 34u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 5u, cave_words.size());

  rdna4::Vop3MachineInst add_lo{};
  std::memcpy(&add_lo, cave_words.data() + *base, sizeof(add_lo));
  EXPECT_EQ(add_lo.encoding, 0x35u);
  EXPECT_EQ(add_lo.op, 259u);
  EXPECT_EQ(add_lo.vdst, 42u);
  EXPECT_EQ(add_lo.src0, 256u + 30u);
  EXPECT_EQ(add_lo.src1, 256u + 34u);
  EXPECT_EQ(add_lo.clamp, 0u);
  EXPECT_EQ(add_lo.neg, 0u);

  rdna4::Vop3MachineInst add_hi{};
  std::memcpy(&add_hi, cave_words.data() + *base + 2, sizeof(add_hi));
  EXPECT_EQ(add_hi.encoding, 0x35u);
  EXPECT_EQ(add_hi.op, 259u);
  EXPECT_EQ(add_hi.vdst, 43u);
  EXPECT_EQ(add_hi.src0, 256u + 31u);
  EXPECT_EQ(add_hi.src1, 256u + 35u);
  EXPECT_EQ(add_hi.clamp, 0u);
  EXPECT_EQ(add_hi.neg, 0u);
  EXPECT_EQ((cave_words[*base + 4u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VPkAddBf16LowersThroughF32AndRepacks) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 35;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 30;
  inst.src1 = 256 + 34;
  inst.src2 = scalar_positive_inline_u32(0);

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  size_t add_count = 0;
  bool found_pack = false;
  for (size_t i = 0; i < cave_words.size(); ++i) {
    if (i + 1u < cave_words.size()) {
      rdna4::Vop3MachineInst candidate{};
      std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
      if (candidate.encoding == 0x35u && candidate.op == 259u)
        ++add_count;
    }
    const auto candidate = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[i]);
    found_pack |= candidate.op == 28u && candidate.vdst == 42u;
  }
  EXPECT_EQ(add_count, 2u);
  EXPECT_TRUE(found_pack);
}

TEST(BinaryTranslator, Gfx1250VPkMulF32LowersToTwoVMulF32s) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 44;
  inst.op = 40;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 12;
  inst.src1 = 256 + 20;
  inst.src2 = scalar_positive_inline_u32(0);
  inst.opsel = 0x2;
  inst.opsel_hi = 0x1;
  inst.neg = 0x2;
  inst.neg_hi = 0x1;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base = find_vop3_for_test(cave_words, 264u, 44u, 256u + 12u, 256u + 21u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 5u, cave_words.size());

  rdna4::Vop3MachineInst mul_lo{};
  std::memcpy(&mul_lo, cave_words.data() + *base, sizeof(mul_lo));
  EXPECT_EQ(mul_lo.encoding, 0x35u);
  EXPECT_EQ(mul_lo.op, 264u);
  EXPECT_EQ(mul_lo.vdst, 44u);
  EXPECT_EQ(mul_lo.src0, 256u + 12u);
  EXPECT_EQ(mul_lo.src1, 256u + 21u);
  EXPECT_EQ(mul_lo.neg, 0x2u);

  rdna4::Vop3MachineInst mul_hi{};
  std::memcpy(&mul_hi, cave_words.data() + *base + 2, sizeof(mul_hi));
  EXPECT_EQ(mul_hi.encoding, 0x35u);
  EXPECT_EQ(mul_hi.op, 264u);
  EXPECT_EQ(mul_hi.vdst, 45u);
  EXPECT_EQ(mul_hi.src0, 256u + 13u);
  EXPECT_EQ(mul_hi.src1, 256u + 20u);
  EXPECT_EQ(mul_hi.neg, 0x1u);
  EXPECT_EQ((cave_words[*base + 4u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VPkFmaF32LowersToTwoVFmaF32s) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 46;
  inst.op = 31;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 10;
  inst.src1 = 256 + 20;
  inst.src2 = 256 + 30;
  inst.opsel = 0x4;
  inst.opsel_hi = 0x3;
  inst.pad_14 = 1;
  inst.clamp = 1;
  inst.neg = 0x5;
  inst.neg_hi = 0x6;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base = find_vop3_for_test(cave_words, 531u, 46u, 256u + 10u, 256u + 20u, 256u + 31u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 5u, cave_words.size());

  rdna4::Vop3MachineInst fma_lo{};
  std::memcpy(&fma_lo, cave_words.data() + *base, sizeof(fma_lo));
  EXPECT_EQ(fma_lo.encoding, 0x35u);
  EXPECT_EQ(fma_lo.op, 531u);
  EXPECT_EQ(fma_lo.vdst, 46u);
  EXPECT_EQ(fma_lo.src0, 256u + 10u);
  EXPECT_EQ(fma_lo.src1, 256u + 20u);
  EXPECT_EQ(fma_lo.src2, 256u + 31u);
  EXPECT_EQ(fma_lo.clamp, 1u);
  EXPECT_EQ(fma_lo.neg, 0x5u);

  rdna4::Vop3MachineInst fma_hi{};
  std::memcpy(&fma_hi, cave_words.data() + *base + 2, sizeof(fma_hi));
  EXPECT_EQ(fma_hi.encoding, 0x35u);
  EXPECT_EQ(fma_hi.op, 531u);
  EXPECT_EQ(fma_hi.vdst, 47u);
  EXPECT_EQ(fma_hi.src0, 256u + 11u);
  EXPECT_EQ(fma_hi.src1, 256u + 21u);
  EXPECT_EQ(fma_hi.src2, 256u + 31u);
  EXPECT_EQ(fma_hi.clamp, 1u);
  EXPECT_EQ(fma_hi.neg, 0x6u);
  EXPECT_EQ((cave_words[*base + 4u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VPkFmaF32StagesHighLaneReadOfOldDstLow) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 2;
  inst.op = 31;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 16;
  inst.src1 = 256;
  inst.src2 = 256 + 2;
  inst.opsel_hi = 0x1;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base = find_vop3_for_test(cave_words, 531u, 2u, 256u + 16u, 256u, 256u + 2u);
  ASSERT_TRUE(base.has_value());
  ASSERT_GE(*base, 2u);
  ASSERT_LE(*base + 5u, cave_words.size());

  const auto stage = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[*base - 2u]);
  ASSERT_EQ(stage.op, 1u);
  EXPECT_NE(stage.vdst, 2u);
  EXPECT_NE(stage.vdst, 3u);
  EXPECT_EQ(stage.src0, 256u + 2u);
  EXPECT_EQ(cave_words[*base - 1u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3MachineInst fma_hi{};
  std::memcpy(&fma_hi, cave_words.data() + *base + 2, sizeof(fma_hi));
  EXPECT_EQ(fma_hi.encoding, 0x35u);
  EXPECT_EQ(fma_hi.op, 531u);
  EXPECT_EQ(fma_hi.vdst, 3u);
  EXPECT_EQ(fma_hi.src0, 256u + 17u);
  EXPECT_EQ(fma_hi.src1, 256u);
  EXPECT_EQ(fma_hi.src2, 256u + stage.vdst);
  EXPECT_EQ((cave_words[*base + 4u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VFmaMixF32Bf16LowersToBf16SourcesAndF32Fma) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 24;
  inst.op = 0x3D;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 10;
  inst.src1 = 256 + 11;
  inst.src2 = 256 + 12;
  inst.opsel = 0x1;
  inst.opsel_hi = 0x3;
  inst.clamp = 1;
  inst.neg = 0x5;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  size_t original_mix_count = 0;
  std::optional<size_t> base;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) == 0xCCu && ((w0 >> 16) & 0x7Fu) == 0x3Du)
      ++original_mix_count;

    rdna4::Vop3MachineInst decoded{};
    std::memcpy(&decoded, cave_words.data() + i, sizeof(decoded));
    if (decoded.encoding == 0x35u && decoded.op == 531u && decoded.vdst == 24u &&
        decoded.src2 == 256u + 12u) {
      base = i;
      break;
    }
  }

  EXPECT_EQ(original_mix_count, 0u);
  ASSERT_TRUE(base.has_value());
  ASSERT_GE(*base, 5u);
  ASSERT_LE(*base + 2u, cave_words.size());

  rdna4::Vop3MachineInst fma{};
  std::memcpy(&fma, cave_words.data() + *base, sizeof(fma));
  ASSERT_GE(fma.src0, 256u);
  ASSERT_GE(fma.src1, 256u);
  const auto src0_tmp = static_cast<uint8_t>(fma.src0 - 256u);
  const auto src1_tmp = static_cast<uint8_t>(fma.src1 - 256u);
  EXPECT_NE(src0_tmp, 10u);
  EXPECT_NE(src1_tmp, 11u);
  EXPECT_NE(src0_tmp, src1_tmp);
  EXPECT_EQ(fma.clamp, 1u);
  EXPECT_EQ(fma.neg, 0x5u);

  const auto src0_extract = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base - 5u]);
  EXPECT_EQ(src0_extract.op, 25u);
  EXPECT_EQ(src0_extract.vdst, src0_tmp);
  EXPECT_EQ(src0_extract.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(src0_extract.vsrc1, 10u);

  const auto src0_shift = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base - 4u]);
  EXPECT_EQ(src0_shift.op, 24u);
  EXPECT_EQ(src0_shift.vdst, src0_tmp);
  EXPECT_EQ(src0_shift.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(src0_shift.vsrc1, src0_tmp);

  const auto src1_extract = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base - 3u]);
  EXPECT_EQ(src1_extract.op, 28u);
  EXPECT_EQ(src1_extract.vdst, src1_tmp);
  EXPECT_EQ(src1_extract.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(src1_extract.vsrc1, 11u);

  const auto src1_shift = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base - 2u]);
  EXPECT_EQ(src1_shift.op, 24u);
  EXPECT_EQ(src1_shift.vdst, src1_tmp);
  EXPECT_EQ(src1_shift.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(src1_shift.vsrc1, src1_tmp);
  EXPECT_EQ(cave_words[*base - 1u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ((cave_words[*base + 2u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VCvtPkBf16F32LowersToIntegerRnePack) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 152;
  inst.op = 877;
  inst.encoding = 0x35;
  inst.src0 = 256 + 152;
  inst.src1 = 256 + 153;
  inst.src2 = scalar_positive_inline_u32(0);

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  EXPECT_FALSE(find_vop3_for_test(cave_words, 877u, 152u, 256u + 152u, 256u + 153u).has_value());

  const auto base =
      find_vop2_for_test(cave_words, 25u, std::nullopt, scalar_positive_inline_u32(16), 152u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LT(*base + 14u, cave_words.size());

  const auto lo_round_bit = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base]);
  ASSERT_EQ(lo_round_bit.op, 25u);
  const uint8_t tmp = lo_round_bit.vdst;
  EXPECT_EQ(lo_round_bit.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(lo_round_bit.vsrc1, 152u);

  const auto lo_mask = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 1u]);
  EXPECT_EQ(lo_mask.op, 27u);
  EXPECT_EQ(lo_mask.vdst, tmp);
  EXPECT_EQ(lo_mask.src0, scalar_positive_inline_u32(1));
  EXPECT_EQ(lo_mask.vsrc1, tmp);

  const auto lo_add_bias = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 2u]);
  ASSERT_EQ(lo_add_bias.op, 37u);
  const uint8_t lo = lo_add_bias.vdst;
  EXPECT_EQ(lo_add_bias.src0, 255u);
  EXPECT_EQ(lo_add_bias.vsrc1, 152u);
  EXPECT_EQ(cave_words[*base + 3u], 0x00007FFFu);

  const auto lo_add_lsb = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 4u]);
  EXPECT_EQ(lo_add_lsb.op, 37u);
  EXPECT_EQ(lo_add_lsb.vdst, lo);
  EXPECT_EQ(lo_add_lsb.src0, 256u + tmp);
  EXPECT_EQ(lo_add_lsb.vsrc1, lo);

  const auto lo_shift = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 5u]);
  EXPECT_EQ(lo_shift.op, 25u);
  EXPECT_EQ(lo_shift.vdst, lo);
  EXPECT_EQ(lo_shift.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(lo_shift.vsrc1, lo);

  const auto hi_round_bit = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 6u]);
  EXPECT_EQ(hi_round_bit.op, 25u);
  EXPECT_EQ(hi_round_bit.vdst, tmp);
  EXPECT_EQ(hi_round_bit.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(hi_round_bit.vsrc1, 153u);

  const auto hi_add_bias = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 8u]);
  ASSERT_EQ(hi_add_bias.op, 37u);
  const uint8_t hi = hi_add_bias.vdst;
  EXPECT_EQ(hi_add_bias.src0, 255u);
  EXPECT_EQ(hi_add_bias.vsrc1, 153u);
  EXPECT_EQ(cave_words[*base + 9u], 0x00007FFFu);

  const auto hi_shift = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 11u]);
  EXPECT_EQ(hi_shift.op, 25u);
  EXPECT_EQ(hi_shift.vdst, hi);
  EXPECT_EQ(hi_shift.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(hi_shift.vsrc1, hi);

  const auto pack_hi = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 12u]);
  EXPECT_EQ(pack_hi.op, 24u);
  EXPECT_EQ(pack_hi.vdst, hi);
  EXPECT_EQ(pack_hi.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(pack_hi.vsrc1, hi);

  const auto pack = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 13u]);
  EXPECT_EQ(pack.op, 28u);
  EXPECT_EQ(pack.vdst, 152u);
  EXPECT_EQ(pack.src0, 256u + lo);
  EXPECT_EQ(pack.vsrc1, hi);
  EXPECT_EQ((cave_words[*base + 14u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VCvtPkBf16F32MaterializesScalarSources) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 1;
  inst.op = 877;
  inst.encoding = 0x35;
  inst.src0 = 7;
  inst.src1 = 0;
  inst.src2 = scalar_positive_inline_u32(0);

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::vector<uint8_t> scalar_copies;
  for (uint32_t word : cave_words) {
    const auto mov = std::bit_cast<rdna4::Vop1MachineInst>(word);
    if (mov.op == 1u && (mov.src0 == 7u || mov.src0 == 0u))
      scalar_copies.push_back(mov.vdst);
  }
  ASSERT_EQ(scalar_copies.size(), 2u);
  EXPECT_EQ(scalar_copies[0], scalar_copies[1]);
  EXPECT_TRUE(find_vop2_for_test(cave_words, 25u, std::nullopt, scalar_positive_inline_u32(16),
                                 scalar_copies[0])
                  .has_value());
}

TEST(BinaryTranslator, Gfx1250VCvtF32Bf16LowersToHalfwordShift) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst lo{};
  lo.vdst = 12;
  lo.op = 498;
  lo.encoding = 0x35;
  lo.src0 = 256 + 20;

  gfx1250::Vop3MachineInst hi = lo;
  hi.vdst = 13;
  hi.src0 = 256 + 21;
  hi.opsel = 1;

  std::array<uint32_t, 5> text_words{};
  std::memcpy(text_words.data(), &lo, sizeof(lo));
  std::memcpy(text_words.data() + 2, &hi, sizeof(hi));
  text_words[4] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  std::vector<uint32_t> all_words;
  for (const Section *text : translated.text_sections()) {
    const auto text_words = section_words_for_test(*text);
    all_words.insert(all_words.end(), text_words.begin(), text_words.end());
  }
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  all_words.insert(all_words.end(), cave_words.begin(), cave_words.end());

  size_t original_convert_count = 0;
  for (size_t i = 0; i + 1 < all_words.size(); ++i) {
    rdna4::Vop3MachineInst decoded{};
    std::memcpy(&decoded, all_words.data() + i, sizeof(decoded));
    if (decoded.encoding == 0x35u && decoded.op == 498u)
      ++original_convert_count;
  }
  EXPECT_EQ(original_convert_count, 0u);

  const auto find_lshl_to = [&](uint8_t vdst) -> std::optional<std::pair<size_t, uint8_t>> {
    for (size_t i = 0; i < all_words.size(); ++i) {
      const auto decoded = std::bit_cast<rdna4::Vop2MachineInst>(all_words[i]);
      if (decoded.op == 24u && decoded.vdst == vdst &&
          decoded.src0 == scalar_positive_inline_u32(16))
        return std::make_pair(i, decoded.vsrc1);
    }
    return std::nullopt;
  };

  const auto lo_shift = find_lshl_to(12);
  ASSERT_TRUE(lo_shift.has_value());
  ASSERT_GT(lo_shift->first, 0u);
  const auto lo_materialize =
      std::bit_cast<rdna4::Vop2MachineInst>(all_words[lo_shift->first - 1u]);
  EXPECT_EQ(lo_materialize.op, 28u);
  EXPECT_EQ(lo_materialize.vdst, lo_shift->second);
  EXPECT_EQ(lo_materialize.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(lo_materialize.vsrc1, 20u);

  const auto hi_shift = find_lshl_to(13);
  ASSERT_TRUE(hi_shift.has_value());
  ASSERT_GT(hi_shift->first, 0u);
  const auto hi_materialize =
      std::bit_cast<rdna4::Vop2MachineInst>(all_words[hi_shift->first - 1u]);
  EXPECT_EQ(hi_materialize.op, 25u);
  EXPECT_EQ(hi_materialize.vdst, hi_shift->second);
  EXPECT_EQ(hi_materialize.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(hi_materialize.vsrc1, 21u);
}

TEST(BinaryTranslator, Gfx1250VCvtF32Fp8E5M3LowersClampSelectorToIntegerSequence) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kQuietNaN = 0x7FC00000u;
  constexpr std::array<uint32_t, 3> text_words = {0xD5EC8001u, 0x02010101u, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  EXPECT_FALSE(section_contains_vop3_opcode(translations, 492u))
      << "RDNA4 must not execute gfx1250's clamp-selected E5M3 conversion";

  std::optional<size_t> final_select;
  for (size_t i = 0; i + 3u < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == 257u && candidate.vdst == 1u &&
        candidate.src1 == 255u && cave_words[i + 2u] == kQuietNaN) {
      final_select = i;
      break;
    }
  }
  ASSERT_TRUE(final_select.has_value());
  EXPECT_EQ((cave_words[*final_select + 3u] >> 16) & 0x7Fu,
            sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VCvtF16Fp8Vop1LowersThroughF32Scratch) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 2> text_words = {0x7E02EF01u, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<rdna4::Vop3MachineInst> decode;
  for (size_t i = 0; i + 1u < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == 492u && candidate.src0 == 257u) {
      decode = candidate;
      break;
    }
  }
  ASSERT_TRUE(decode.has_value());
  EXPECT_EQ(decode->opsel, 0u);
  EXPECT_TRUE(find_vop1_for_test(cave_words, 10u, 1u, static_cast<uint16_t>(256u + decode->vdst))
                  .has_value());
}

TEST(BinaryTranslator, Gfx1250VCvtF16Bf8Vop3PreservesByteSelectorThroughF32Scratch) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 2;
  inst.op = 504;
  inst.encoding = 0x35;
  inst.src0 = 256 + 3;
  inst.opsel = 2;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<rdna4::Vop3MachineInst> decode;
  for (size_t i = 0; i + 1u < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == 493u && candidate.src0 == 259u) {
      decode = candidate;
      break;
    }
  }
  ASSERT_TRUE(decode.has_value());
  EXPECT_EQ(decode->opsel, 2u);
  EXPECT_TRUE(find_vop1_for_test(cave_words, 10u, 2u, static_cast<uint16_t>(256u + decode->vdst))
                  .has_value());
}

TEST(BinaryTranslator, Gfx1250VCvtPkFp8F16LowersThroughTwoF32Sources) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> text_words = {0xD7720001u, 0x02010101u, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::vector<rdna4::Vop3MachineInst> half_converts;
  std::optional<rdna4::Vop3MachineInst> pack;
  for (size_t i = 0; i + 1u < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding != 0x35u)
      continue;
    if (candidate.op == 395u && candidate.src0 == 257u)
      half_converts.push_back(candidate);
    if (candidate.op == 873u && candidate.vdst == 1u)
      pack = candidate;
  }
  ASSERT_EQ(half_converts.size(), 2u);
  EXPECT_EQ(half_converts[0].opsel, 0u);
  EXPECT_EQ(half_converts[1].opsel, 1u);
  ASSERT_TRUE(pack.has_value());
  EXPECT_EQ(pack->src0, static_cast<uint16_t>(256u + half_converts[0].vdst));
  EXPECT_EQ(pack->src1, static_cast<uint16_t>(256u + half_converts[1].vdst));
}

TEST(BinaryTranslator, Gfx1250VCvtPkF16Fp8Vop1DecodesBothBytesBeforePackingHalves) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 2> text_words = {0x7E02EB01u, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::vector<rdna4::Vop3MachineInst> decodes;
  for (size_t i = 0; i + 1u < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == 492u && candidate.src0 == 257u)
      decodes.push_back(candidate);
  }
  ASSERT_EQ(decodes.size(), 2u);
  EXPECT_EQ(decodes[0].opsel, 0u);
  EXPECT_EQ(decodes[1].opsel, 2u);
  EXPECT_TRUE(find_vop1_for_test(cave_words, 10u, static_cast<uint8_t>(decodes[0].vdst),
                                 static_cast<uint16_t>(256u + decodes[0].vdst))
                  .has_value());
  EXPECT_TRUE(find_vop1_for_test(cave_words, 10u, static_cast<uint8_t>(decodes[1].vdst),
                                 static_cast<uint16_t>(256u + decodes[1].vdst))
                  .has_value());
  EXPECT_TRUE(find_vop2_for_test(cave_words, 28u, 1u, static_cast<uint16_t>(256u + decodes[0].vdst),
                                 decodes[1].vdst)
                  .has_value());
}

TEST(BinaryTranslator, Gfx1250VCvtSrFp8F16LowersThroughF32StochasticConvert) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 3> text_words = {0xD7740003u, 0x02020501u, kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  std::optional<rdna4::Vop3MachineInst> widen;
  std::optional<rdna4::Vop3MachineInst> stochastic;
  for (size_t i = 0; i + 1u < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding != 0x35u)
      continue;
    if (candidate.op == 395u && candidate.src0 == 257u)
      widen = candidate;
    if (candidate.op == 875u && candidate.vdst == 3u && candidate.src1 == 258u)
      stochastic = candidate;
  }
  ASSERT_TRUE(widen.has_value());
  ASSERT_TRUE(stochastic.has_value());
  EXPECT_EQ(stochastic->src0, static_cast<uint16_t>(256u + widen->vdst));
  EXPECT_EQ(stochastic->opsel, 0u);
}

TEST(BinaryTranslator, Gfx1250ScaledFiniteLowpStochasticRoundingConsumesSeedBits) {
  struct TestCase {
    uint16_t op;
    uint16_t seed_shift;
  };
  constexpr std::array<TestCase, 3> test_cases = {{{0x297u, 10u}, {0x2D3u, 12u}, {0x2D4u, 11u}}};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const auto test_case : test_cases) {
    SCOPED_TRACE(test_case.op);
    gfx1250::Vop3MachineInst inst{};
    inst.vdst = 64;
    inst.op = test_case.op;
    inst.encoding = 0x35;
    inst.src0 = 256 + 32;
    inst.src1 = 256 + 48;
    inst.src2 = 256 + 49;

    std::array<uint32_t, 3> text_words{};
    std::memcpy(text_words.data(), &inst, sizeof(inst));
    text_words[2] = kGfx1250SEndpgm;

    auto image =
        make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
    AmdGpuCodeObject co(image.data(), image.size());
    ASSERT_TRUE(co.is_valid());

    BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                EF_AMDGPU_MACH_AMDGCN_GFX1201);
    auto result = translator.translate(co);
    ASSERT_FALSE(result.elf_bytes.empty());
    EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

    AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    const Section *translations = find_section(translated, ".rj_translations");
    ASSERT_NE(translations, nullptr);
    const auto cave_words = section_words_for_test(*translations);

    const auto seed_load = find_vop1_for_test(cave_words, 1u, std::nullopt, 256u + 48u);
    ASSERT_TRUE(seed_load.has_value());
    const auto seed_tmp = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[*seed_load]).vdst;
    const auto rounding_shift = find_vop2_for_test(
        cave_words, 25u, std::nullopt, scalar_positive_inline_u32(test_case.seed_shift), seed_tmp);
    const auto lfsr_shift = find_vop2_for_test(cave_words, 25u, std::nullopt,
                                               scalar_positive_inline_u32(31u), seed_tmp);
    ASSERT_TRUE(rounding_shift.has_value());
    ASSERT_TRUE(lfsr_shift.has_value());
    EXPECT_LT(*rounding_shift, *lfsr_shift)
        << "the seed must affect finite low-precision rounding before it is advanced";
  }
}

TEST(BinaryTranslator, Gfx1250VCvtPkF16F32LowersToScalarConvertsAndPack) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 32;
  inst.op = 879;
  inst.encoding = 0x35;
  inst.src0 = 256 + 32;
  inst.src1 = 256 + 33;
  inst.src2 = scalar_positive_inline_u32(0);

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  EXPECT_FALSE(find_vop3_for_test(cave_words, 879u, 32u, 256u + 32u, 256u + 33u).has_value());

  std::optional<size_t> base;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    rdna4::Vop3MachineInst decoded{};
    std::memcpy(&decoded, cave_words.data() + i, sizeof(decoded));
    if (decoded.encoding == 0x35u && decoded.op == 0x18Au && decoded.src0 == 256u + 32u) {
      base = i;
      break;
    }
  }
  ASSERT_TRUE(base.has_value());
  ASSERT_LT(*base + 10u, cave_words.size());

  rdna4::Vop3MachineInst lo_convert{};
  std::memcpy(&lo_convert, cave_words.data() + *base, sizeof(lo_convert));
  ASSERT_EQ(lo_convert.op, 0x18Au);
  ASSERT_EQ(lo_convert.src0, 256u + 32u);
  const uint8_t lo = lo_convert.vdst;
  EXPECT_NE(lo, 32u);
  EXPECT_NE(lo, 33u);

  rdna4::Vop3MachineInst hi_convert{};
  std::memcpy(&hi_convert, cave_words.data() + *base + 2u, sizeof(hi_convert));
  ASSERT_EQ(hi_convert.op, 0x18Au);
  ASSERT_EQ(hi_convert.src0, 256u + 33u);
  const uint8_t hi = hi_convert.vdst;
  EXPECT_NE(hi, 32u);
  EXPECT_NE(hi, 33u);
  EXPECT_NE(hi, lo);

  EXPECT_EQ(cave_words[*base + 4u], build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));

  const auto mask_lo = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 5u]);
  EXPECT_EQ(mask_lo.op, 27u);
  EXPECT_EQ(mask_lo.vdst, lo);
  EXPECT_EQ(mask_lo.src0, 255u);
  EXPECT_EQ(mask_lo.vsrc1, lo);
  EXPECT_EQ(cave_words[*base + 6u], 0x0000FFFFu);

  const auto pack_hi = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 7u]);
  EXPECT_EQ(pack_hi.op, 24u);
  EXPECT_EQ(pack_hi.vdst, hi);
  EXPECT_EQ(pack_hi.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(pack_hi.vsrc1, hi);

  EXPECT_EQ(cave_words[*base + 8u], build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));

  const auto pack = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 9u]);
  EXPECT_EQ(pack.op, 28u);
  EXPECT_EQ(pack.vdst, 32u);
  EXPECT_EQ(pack.src0, 256u + lo);
  EXPECT_EQ(pack.vsrc1, hi);
  EXPECT_EQ((cave_words[*base + 10u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250WmmaF32F16K32SplitsThroughContiguousK16Cave) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x60;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 24;
  inst.opsel_hi = 3;
  inst.neg = 0x4;
  inst.neg_hi = 0x4;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  struct WmmaFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
    uint8_t neg;
    uint8_t neg_hi;
  };
  std::vector<WmmaFields> wmma;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu || ((w0 >> 16) & 0x7Fu) != 64u)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    wmma.push_back(
        {static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
         static_cast<uint16_t>((w1 >> 9) & 0x1FFu), static_cast<uint16_t>((w1 >> 18) & 0x1FFu),
         static_cast<uint8_t>((w1 >> 29) & 0x7u), static_cast<uint8_t>((w0 >> 8) & 0x7u)});
  }
  ASSERT_EQ(wmma.size(), 2u);

  constexpr uint32_t kRdna4SWaitKmcnt0 = 0xBFC70000u;
  constexpr uint32_t kRdna4SWaitIdle0 = 0xBF8A0000u;
  const uint32_t kRdna4SWaitValuVgpr =
      build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4);
  const uint32_t kRdna4SWaitValuScalar =
      build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitKmcnt0), 2)
      << "each RDNA4 WMMA must wait on the matrix pipe before reusing the accumulator";
  EXPECT_EQ(std::find(cave_words.begin(), cave_words.end(), kRdna4SWaitIdle0), cave_words.end())
      << "WMMA lowering should use the documented kmcnt wait, not the broader idle wait";
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitValuVgpr), 2)
      << "accumulator copies must be waited on before WMMA/final-copy consumers";
  EXPECT_EQ(std::find(cave_words.begin(), cave_words.end(), kRdna4SWaitValuScalar),
            cave_words.end())
      << "WMMA split waits must target VGPR destinations, not scalar destinations";

  EXPECT_NE(wmma[0].vdst, 42u) << "unaligned original destination should use an aligned scratch "
                                  "accumulator";
  EXPECT_GE(wmma[0].src0, 256u);
  EXPECT_GE(wmma[0].src1, 256u);
  EXPECT_EQ(wmma[0].src0, 256u + 8u);
  EXPECT_EQ(wmma[0].src1, 256u + 16u);
  EXPECT_EQ(wmma[0].vdst, 208u);
  EXPECT_EQ((wmma[0].src0 - 256u) % 4u, 0u);
  EXPECT_EQ((wmma[0].src1 - 256u) % 4u, 0u);
  EXPECT_EQ(wmma[0].vdst % 8u, 0u);
  EXPECT_EQ(wmma[0].neg, 0x4u);
  EXPECT_EQ(wmma[0].neg_hi, 0x4u);

  EXPECT_EQ(wmma[0].src2, static_cast<uint16_t>(256u + wmma[0].vdst));
  EXPECT_EQ(wmma[1].vdst, wmma[0].vdst);
  EXPECT_EQ(wmma[1].src0, 256u + 12u);
  EXPECT_EQ(wmma[1].src1, 256u + 20u);
  EXPECT_EQ(wmma[1].src2, static_cast<uint16_t>(256u + wmma[0].vdst));
  EXPECT_EQ(wmma[1].vdst % 8u, 0u);
  EXPECT_EQ(wmma[1].neg, 0u) << "the C modifier must only be applied before the first K16 chunk";
  EXPECT_EQ(wmma[1].neg_hi, 0u) << "the C modifier must only be applied before the first K16 chunk";

  const auto build_vop1_test = [](uint8_t op, uint8_t vdst, uint16_t src0) {
    return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
  };
  for (uint8_t word = 0; word < 8; ++word) {
    const uint32_t expected_mov = build_vop1_test(
        1, static_cast<uint8_t>(42u + word), static_cast<uint16_t>(256u + wmma[0].vdst + word));
    auto mov_it = std::find(cave_words.begin(), cave_words.end(), expected_mov);
    EXPECT_NE(mov_it, cave_words.end())
        << "final scratch accumulator word should be copied to the original destination";
    if (word == 0 && mov_it != cave_words.end()) {
      ASSERT_NE(mov_it, cave_words.begin());
      EXPECT_EQ(*(mov_it - 1), kRdna4SWaitValuVgpr)
          << "scratch accumulator copy must wait for the second RDNA4 WMMA result";
    }
  }
  EXPECT_EQ((cave_words.back() >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250WmmaF32Bf16K32SplitsThroughContiguousK16Cave) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x62;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 24;
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  struct WmmaFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
  };
  std::vector<WmmaFields> wmma;
  size_t original_k32_count = 0;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu)
      continue;
    const uint32_t op = (w0 >> 16) & 0x7Fu;
    if (op == 0x62u) {
      ++original_k32_count;
      continue;
    }
    if (op != 65u)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    wmma.push_back({static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
                    static_cast<uint16_t>((w1 >> 9) & 0x1FFu),
                    static_cast<uint16_t>((w1 >> 18) & 0x1FFu)});
  }
  EXPECT_EQ(original_k32_count, 0u);
  ASSERT_EQ(wmma.size(), 2u);

  constexpr uint32_t kRdna4SWaitKmcnt0 = 0xBFC70000u;
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitKmcnt0), 2);

  EXPECT_NE(wmma[0].vdst, 42u);
  EXPECT_GE(wmma[0].src0, 256u);
  EXPECT_GE(wmma[0].src1, 256u);
  EXPECT_EQ(wmma[0].src0, 256u + 8u);
  EXPECT_EQ(wmma[0].src1, 256u + 16u);
  EXPECT_EQ(wmma[0].vdst, 208u);
  EXPECT_EQ((wmma[0].src0 - 256u) % 4u, 0u);
  EXPECT_EQ((wmma[0].src1 - 256u) % 4u, 0u);
  EXPECT_EQ(wmma[0].vdst % 8u, 0u);
  EXPECT_EQ(wmma[0].src2, static_cast<uint16_t>(256u + wmma[0].vdst));
  EXPECT_EQ(wmma[1].vdst, wmma[0].vdst);
  EXPECT_EQ(wmma[1].src0, 256u + 12u);
  EXPECT_EQ(wmma[1].src1, 256u + 20u);
  EXPECT_EQ(wmma[1].src2, static_cast<uint16_t>(256u + wmma[0].vdst));

  for (uint8_t word = 0; word < 8; ++word) {
    const uint32_t expected_mov = build_vop1_for_test(
        1, static_cast<uint8_t>(42u + word), static_cast<uint16_t>(256u + wmma[0].vdst + word));
    EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), expected_mov), cave_words.end());
  }
  EXPECT_EQ((cave_words.back() >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

void expect_gfx1250_packed16_k32_wmma_split(uint8_t source_op, uint8_t target_op) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = source_op;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 24;
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  struct WmmaFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
  };
  std::vector<WmmaFields> wmma;
  size_t original_count = 0;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu)
      continue;
    const uint8_t op = static_cast<uint8_t>((w0 >> 16) & 0x7Fu);
    original_count += op == source_op;
    if (op != target_op)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    wmma.push_back({static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
                    static_cast<uint16_t>((w1 >> 9) & 0x1FFu),
                    static_cast<uint16_t>((w1 >> 18) & 0x1FFu)});
  }
  EXPECT_EQ(original_count, 0u);
  ASSERT_EQ(wmma.size(), 2u);
  EXPECT_EQ(wmma[0].vdst, 208u);
  EXPECT_EQ(wmma[0].src0, 256u + 8u);
  EXPECT_EQ(wmma[0].src1, 256u + 16u);
  EXPECT_EQ(wmma[0].src2, 256u + 208u);
  EXPECT_EQ(wmma[1].vdst, 208u);
  EXPECT_EQ(wmma[1].src0, 256u + 12u);
  EXPECT_EQ(wmma[1].src1, 256u + 20u);
  EXPECT_EQ(wmma[1].src2, 256u + 208u);
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), 0xBFC70000u), 2);

  for (uint8_t word = 0; word < 4; ++word) {
    const uint32_t expected_mov = build_vop1_for_test(1, static_cast<uint8_t>(42u + word),
                                                      static_cast<uint16_t>(256u + 208u + word));
    EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), expected_mov), cave_words.end());
  }
}

TEST(BinaryTranslator, Gfx1250WmmaF16F16K32SplitsThroughPackedK16Cave) {
  expect_gfx1250_packed16_k32_wmma_split(0x61, 66);
}

TEST(BinaryTranslator, Gfx1250WmmaBf16Bf16K32SplitsThroughPackedK16Cave) {
  expect_gfx1250_packed16_k32_wmma_split(0x63, 67);
}

TEST(BinaryTranslator, Gfx1250WmmaBf16F32K32AccumulatesInF32BeforePacking) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x64;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 24;
  inst.opsel_hi = 3;
  inst.neg = 0x4;
  inst.neg_hi = 0x4;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  struct WmmaFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
    uint8_t neg;
    uint8_t neg_hi;
  };
  std::vector<WmmaFields> wmma;
  size_t original_count = 0;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu)
      continue;
    const uint8_t op = static_cast<uint8_t>((w0 >> 16) & 0x7Fu);
    original_count += op == 0x64u;
    if (op != 65u)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    wmma.push_back(
        {static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
         static_cast<uint16_t>((w1 >> 9) & 0x1FFu), static_cast<uint16_t>((w1 >> 18) & 0x1FFu),
         static_cast<uint8_t>((w1 >> 29) & 0x7u), static_cast<uint8_t>((w0 >> 8) & 0x7u)});
  }
  EXPECT_EQ(original_count, 0u);
  ASSERT_EQ(wmma.size(), 2u);
  EXPECT_EQ(wmma[0].vdst, 208u);
  EXPECT_EQ(wmma[0].src0, 256u + 8u);
  EXPECT_EQ(wmma[0].src1, 256u + 16u);
  EXPECT_EQ(wmma[0].src2, 256u + 208u);
  EXPECT_EQ(wmma[0].neg, 0x4u);
  EXPECT_EQ(wmma[0].neg_hi, 0x4u);
  EXPECT_EQ(wmma[1].vdst, 208u);
  EXPECT_EQ(wmma[1].src0, 256u + 12u);
  EXPECT_EQ(wmma[1].src1, 256u + 20u);
  EXPECT_EQ(wmma[1].src2, 256u + 208u);
  EXPECT_EQ(wmma[1].neg, 0u);
  EXPECT_EQ(wmma[1].neg_hi, 0u);
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), 0xBFC70000u), 2);

  for (uint8_t word = 0; word < 4; ++word) {
    const auto packed = std::ranges::find_if(cave_words, [&](uint32_t encoded) {
      const auto vop2 = std::bit_cast<rdna4::Vop2MachineInst>(encoded);
      return vop2.op == 28u && vop2.vdst == 42u + word;
    });
    EXPECT_NE(packed, cave_words.end());
  }
}

TEST(BinaryTranslator, Gfx1250WmmaI32Iu8K64SplitsThroughRelayoutCave) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x72;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 24;
  inst.opsel_hi = 3;
  inst.clamp = 1;
  inst.neg = 1;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  struct WmmaFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
    uint8_t clamp;
    uint8_t neg;
  };
  std::vector<WmmaFields> wmma;
  std::vector<size_t> wmma_indices;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu || ((w0 >> 16) & 0x7Fu) != 68u)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    wmma_indices.push_back(i);
    wmma.push_back(
        {static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
         static_cast<uint16_t>((w1 >> 9) & 0x1FFu), static_cast<uint16_t>((w1 >> 18) & 0x1FFu),
         static_cast<uint8_t>((w0 >> 15) & 0x1u), static_cast<uint8_t>((w1 >> 29) & 0x7u)});
  }
  ASSERT_EQ(wmma.size(), 4u);
  ASSERT_EQ(wmma_indices.size(), wmma.size());

  constexpr uint8_t kVccLo = 106;
  std::optional<size_t> save_vcc_index;
  std::optional<uint8_t> vcc_save_sgpr;
  for (size_t i = 0; i < cave_words.size(); ++i) {
    const auto sop1 = std::bit_cast<gfx1250::Sop1MachineInst>(cave_words[i]);
    if (sop1.encoding == 0x17Du && sop1.op == 1u && sop1.ssrc0 == kVccLo && sop1.sdst != kVccLo) {
      save_vcc_index = i;
      vcc_save_sgpr = static_cast<uint8_t>(sop1.sdst);
      break;
    }
  }
  ASSERT_TRUE(save_vcc_index) << "i8 WMMA lowering must preserve VCC for later predicates";
  ASSERT_TRUE(vcc_save_sgpr);

  std::optional<size_t> restore_vcc_index;
  for (size_t i = *save_vcc_index + 1; i < cave_words.size(); ++i) {
    const auto sop1 = std::bit_cast<gfx1250::Sop1MachineInst>(cave_words[i]);
    if (sop1.encoding == 0x17Du && sop1.op == 1u && sop1.sdst == kVccLo &&
        sop1.ssrc0 == *vcc_save_sgpr) {
      restore_vcc_index = i;
      break;
    }
  }
  ASSERT_TRUE(restore_vcc_index) << "i8 WMMA lowering must restore VCC after split WMMA ops";
  EXPECT_LT(*save_vcc_index, wmma_indices.front());
  EXPECT_GT(*restore_vcc_index, wmma_indices.back());

  constexpr uint32_t kRdna4SWaitKmcnt0 = 0xBFC70000u;
  constexpr uint32_t kRdna4SWaitIdle0 = 0xBF8A0000u;
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitKmcnt0), 4)
      << "each split RDNA4 i8 WMMA must wait before reusing the accumulator";
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitIdle0), 4)
      << "split i8 WMMA accumulation must drain before the next K16 chunk reuses "
         "the accumulator";

  EXPECT_NE(wmma[0].vdst, 42u) << "unaligned original destination should use an aligned scratch "
                                  "accumulator";
  EXPECT_GE(wmma[0].src0, 256u);
  EXPECT_GE(wmma[0].src1, 256u);
  EXPECT_EQ((wmma[0].src0 - 256u) % 2u, 0u);
  EXPECT_EQ((wmma[0].src1 - 256u) % 2u, 0u);
  EXPECT_EQ(wmma[0].vdst % 8u, 0u);

  for (const WmmaFields &fields : wmma) {
    EXPECT_EQ(fields.vdst, wmma[0].vdst);
    EXPECT_EQ(fields.src0, wmma[0].src0);
    EXPECT_EQ(fields.src1, wmma[0].src1);
    EXPECT_EQ(fields.src2, static_cast<uint16_t>(256u + wmma[0].vdst));
    EXPECT_EQ(fields.clamp, 1u) << "gfx1250 clamp bit must be preserved";
    EXPECT_EQ(fields.neg, 1u) << "gfx1250 neg_lo signedness bits must be preserved";
  }

  const auto build_vop1_test = [](uint8_t op, uint8_t vdst, uint16_t src0) {
    return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
  };
  std::optional<size_t> last_final_copy_index;
  for (uint8_t word = 0; word < 8; ++word) {
    const uint32_t expected_mov = build_vop1_test(
        1, static_cast<uint8_t>(42u + word), static_cast<uint16_t>(256u + wmma[0].vdst + word));
    auto copy_it = std::find(cave_words.begin(), cave_words.end(), expected_mov);
    EXPECT_NE(copy_it, cave_words.end())
        << "final scratch accumulator word should be copied to the original destination";
    if (copy_it != cave_words.end())
      last_final_copy_index =
          std::max(last_final_copy_index.value_or(0u),
                   static_cast<size_t>(std::distance(cave_words.begin(), copy_it)));
  }
  ASSERT_TRUE(last_final_copy_index);
  ASSERT_LT(*last_final_copy_index + 1u, cave_words.size());
  EXPECT_EQ(cave_words[*last_final_copy_index + 1u],
            build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4))
      << "scratch accumulator copy-back must be visible before later source-stream "
         "consumers read vdst";
  EXPECT_EQ((cave_words.back() >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250I8WmmaSemanticTempsGrowDescriptor) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 16;
  inst.op = 0x72;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 0;
  inst.src1 = 256 + 8;
  inst.src2 = 256 + 16;
  inst.opsel_hi = 3;

  std::vector<uint32_t> text_words;
  text_words.resize(2);
  std::memcpy(text_words.data(), &inst, sizeof(inst));

  auto append_live_self_move = [&](uint8_t reg) {
    gfx1250::Vop1MachineInst mov{};
    mov.encoding = 0x3F;
    mov.op = 1;
    mov.vdst = reg;
    mov.src0 = static_cast<uint16_t>(256u + reg);
    text_words.push_back(std::bit_cast<uint32_t>(mov));
  };
  for (uint16_t reg = 24; reg < 206; ++reg)
    append_live_self_move(static_cast<uint8_t>(reg));
  text_words.push_back(kGfx1250SEndpgm);

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *rodata = find_section(translated, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(KD));

  const auto *desc = reinterpret_cast<const KD *>(rodata->data());
  const uint32_t granulated = AMDHSA_BITS_GET(desc->compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t addressable_vgprs = (granulated + 1u) * 8u;
  EXPECT_GT(addressable_vgprs, 210u)
      << "i8 WMMA relayout lowering can place lane-xor and A/B temporaries at v206..v210";
}

TEST(BinaryTranslator, Gfx1250SwmmacF32F16K64SplitsThroughRdna4SparseOps) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x65;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 32;
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  struct SwmmacFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
  };
  std::vector<SwmmacFields> swmmac;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu || ((w0 >> 16) & 0x7Fu) != 0x50u)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    swmmac.push_back({static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
                      static_cast<uint16_t>((w1 >> 9) & 0x1FFu),
                      static_cast<uint16_t>((w1 >> 18) & 0x1FFu)});
  }
  ASSERT_EQ(swmmac.size(), 2u);

  constexpr uint32_t kRdna4SWaitKmcnt0 = 0xBFC70000u;
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitKmcnt0), 2)
      << "each split RDNA4 sparse WMMA must wait before reusing the accumulator";
  EXPECT_NE(swmmac[0].vdst, 42u)
      << "unaligned original destination should use an aligned scratch accumulator";
  EXPECT_EQ(swmmac[0].vdst % 8u, 0u);
  EXPECT_EQ((swmmac[0].src0 - 256u) % 4u, 0u);
  EXPECT_EQ((swmmac[0].src1 - 256u) % 4u, 0u);
  for (const SwmmacFields &fields : swmmac) {
    EXPECT_EQ(fields.vdst, swmmac[0].vdst);
    EXPECT_EQ(fields.src0, swmmac[0].src0);
    EXPECT_EQ(fields.src1, swmmac[0].src1);
    EXPECT_EQ(fields.src2, swmmac[0].src2);
    EXPECT_GE(fields.src2, 256u);
    EXPECT_NE(fields.src2, 256u + 32u)
        << "gfx1250 metadata must be relaid out into the RDNA4 K32 selector format";
  }

  const auto build_vop1_test = [](uint8_t op, uint8_t vdst, uint16_t src0) {
    return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
  };
  for (uint8_t word = 0; word < 8; ++word) {
    const uint32_t expected_mov = build_vop1_test(
        1, static_cast<uint8_t>(42u + word), static_cast<uint16_t>(256u + swmmac[0].vdst + word));
    EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), expected_mov), cave_words.end())
        << "final scratch accumulator word should be copied to the original destination";
  }
  EXPECT_EQ((cave_words.back() >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250SwmmacF32Fp8K128SplitsAndRelayoutsSelectors) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x73;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 32;
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  struct SwmmacFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
  };
  std::vector<SwmmacFields> swmmac;
  for (size_t i = 0; i + 1u < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu || ((w0 >> 16) & 0x7Fu) != 0x57u)
      continue;
    const uint32_t w1 = cave_words[i + 1u];
    swmmac.push_back({static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
                      static_cast<uint16_t>((w1 >> 9) & 0x1FFu),
                      static_cast<uint16_t>((w1 >> 18) & 0x1FFu)});
  }
  ASSERT_EQ(swmmac.size(), 4u);

  constexpr uint32_t kRdna4SWaitKmcnt0 = 0xBFC70000u;
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitKmcnt0), 4);
  EXPECT_NE(swmmac[0].vdst, 42u);
  EXPECT_EQ(swmmac[0].vdst % 8u, 0u);
  EXPECT_NE(swmmac[0].src2, 256u + 32u)
      << "gfx1250's interleaved two-dword selectors require an RDNA4 temporary";
  for (const SwmmacFields &fields : swmmac) {
    EXPECT_EQ(fields.vdst, swmmac[0].vdst);
    EXPECT_EQ(fields.src0, swmmac[0].src0);
    EXPECT_EQ(fields.src1, swmmac[0].src1);
    EXPECT_EQ(fields.src2, swmmac[0].src2);
  }

  const auto build_vop1_test = [](uint8_t op, uint8_t vdst, uint16_t src0) {
    return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
  };
  for (uint8_t word = 0; word < 8; ++word) {
    const uint32_t expected_mov = build_vop1_test(
        1, static_cast<uint8_t>(42u + word), static_cast<uint16_t>(256u + swmmac[0].vdst + word));
    EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), expected_mov), cave_words.end());
  }
}

TEST(BinaryTranslator, Gfx1250SwmmacF32F16K64RedirectsBankCrossingSrcRun) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc1Bank1 = 0xBF860004u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 64;
  inst.op = 0x65;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 246;
  inst.src2 = 256 + 32;
  inst.opsel_hi = 3;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc1Bank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto cave_words = section_words_for_test(*translations);
  struct SwmmacFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
  };
  std::vector<SwmmacFields> swmmac;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu || ((w0 >> 16) & 0x7Fu) != 0x50u)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    swmmac.push_back({static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
                      static_cast<uint16_t>((w1 >> 9) & 0x1FFu),
                      static_cast<uint16_t>((w1 >> 18) & 0x1FFu)});
  }
  ASSERT_EQ(swmmac.size(), 2u);

  for (const SwmmacFields &fields : swmmac) {
    ASSERT_GE(fields.src1, 256u);
    const uint16_t src1_base = static_cast<uint16_t>(fields.src1 - 256u);
    EXPECT_LE(static_cast<uint16_t>(src1_base + 16u), 256u)
        << "RDNA4 sparse f16 source1 must be a contiguous wave32 low-bank run";
    EXPECT_NE(fields.src1, 256u + 246u)
        << "cross-bank gfx1250 source runs must be redirected before semantic lowering";
  }
}

TEST(BinaryTranslator, Gfx1250SwmmacI32Iu8K128SplitsThroughRdna4SparseOps) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x7B;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 32;
  inst.opsel_hi = 3;
  inst.clamp = 1;
  inst.neg = 2;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  struct SwmmacFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
    uint8_t clamp;
    uint8_t neg;
  };
  std::vector<SwmmacFields> swmmac;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu || ((w0 >> 16) & 0x7Fu) != 0x54u)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    swmmac.push_back(
        {static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
         static_cast<uint16_t>((w1 >> 9) & 0x1FFu), static_cast<uint16_t>((w1 >> 18) & 0x1FFu),
         static_cast<uint8_t>((w0 >> 15) & 0x1u), static_cast<uint8_t>((w1 >> 29) & 0x7u)});
  }
  ASSERT_EQ(swmmac.size(), 4u);

  constexpr uint32_t kRdna4SWaitKmcnt0 = 0xBFC70000u;
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitKmcnt0), 4)
      << "each split RDNA4 sparse i8 WMMA must wait before reusing the accumulator";
  EXPECT_NE(swmmac[0].vdst, 42u)
      << "unaligned original destination should use an aligned scratch accumulator";
  EXPECT_EQ(swmmac[0].vdst % 8u, 0u);
  EXPECT_EQ((swmmac[0].src0 - 256u) % 2u, 0u);
  EXPECT_EQ((swmmac[0].src1 - 256u) % 2u, 0u);
  for (size_t i = 0; i < swmmac.size(); ++i) {
    EXPECT_EQ(swmmac[i].vdst, swmmac[0].vdst);
    EXPECT_EQ(swmmac[i].src0, swmmac[0].src0);
    EXPECT_EQ(swmmac[i].src1, swmmac[0].src1);
    EXPECT_EQ(swmmac[i].src2, static_cast<uint16_t>(256u + 32u + (i / 2u)))
        << "gfx1250 K128 i8 sparse chunks split the two index words by K64 half";
    EXPECT_EQ(swmmac[i].clamp, 1u);
    EXPECT_EQ(swmmac[i].neg, 2u);
  }

  const auto build_vop1_test = [](uint8_t op, uint8_t vdst, uint16_t src0) {
    return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
  };
  for (uint8_t word = 0; word < 8; ++word) {
    const uint32_t expected_mov = build_vop1_test(
        1, static_cast<uint8_t>(42u + word), static_cast<uint16_t>(256u + swmmac[0].vdst + word));
    EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), expected_mov), cave_words.end())
        << "final scratch accumulator word should be copied to the original destination";
  }
  EXPECT_EQ((cave_words.back() >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

void expect_gfx1250_fp8_wmma_splits_through_relayout_cave(uint8_t gfx1250_op,
                                                          size_t expected_chunks,
                                                          uint8_t opsel = 0) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = gfx1250_op;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 32;
  inst.src2 = 256 + 64;
  inst.opsel = opsel;
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);

  KernelDescriptorTranslator descriptor_parser(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto descriptor_infos = descriptor_parser.translate_image(
      result.elf_bytes, text->sectionOffset(), text->size(), KernelDescriptorTranslationOptions{});
  ASSERT_EQ(descriptor_infos.size(), 1u);
  EXPECT_EQ(descriptor_infos[0].target_private_size, 0u)
      << "K64 fp8 lowering should not reserve the K128 private spill zone";
  EXPECT_LT(descriptor_infos[0].guest_vgpr_count, 261u)
      << "K64 fp8 lowering should not pay the high-bank K128 resource floor";

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  struct WmmaFields {
    uint8_t vdst;
    uint16_t src0;
    uint16_t src1;
    uint16_t src2;
    uint8_t neg;
  };
  std::vector<WmmaFields> wmma;
  const uint8_t target_op = static_cast<uint8_t>(0x46u + (gfx1250_op - 0x6Au));
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu || ((w0 >> 16) & 0x7Fu) != target_op)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    wmma.push_back({static_cast<uint8_t>(w0 & 0xFFu), static_cast<uint16_t>(w1 & 0x1FFu),
                    static_cast<uint16_t>((w1 >> 9) & 0x1FFu),
                    static_cast<uint16_t>((w1 >> 18) & 0x1FFu),
                    static_cast<uint8_t>((w1 >> 29) & 0x7u)});
  }
  ASSERT_EQ(wmma.size(), expected_chunks);

  constexpr uint32_t kRdna4SWaitKmcnt0 = 0xBFC70000u;
  constexpr uint32_t kRdna4SWaitIdle0 = 0xBF8A0000u;
  EXPECT_EQ(
      static_cast<size_t>(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitKmcnt0)),
      expected_chunks)
      << "each split RDNA4 fp8 WMMA must wait before reusing the accumulator";
  EXPECT_EQ(std::find(cave_words.begin(), cave_words.end(), kRdna4SWaitIdle0), cave_words.end())
      << "WMMA lowering should use the matrix counter wait, not the broader idle wait";

  EXPECT_NE(wmma[0].vdst, 42u) << "unaligned original destination should use an aligned scratch "
                                  "accumulator";
  EXPECT_GE(wmma[0].src0, 256u);
  EXPECT_GE(wmma[0].src1, 256u);
  EXPECT_EQ((wmma[0].src0 - 256u) % 2u, 0u);
  EXPECT_EQ((wmma[0].src1 - 256u) % 2u, 0u);
  EXPECT_EQ(wmma[0].vdst % 8u, 0u);

  for (const WmmaFields &fields : wmma) {
    EXPECT_EQ(fields.vdst, wmma[0].vdst);
    EXPECT_EQ(fields.src0, wmma[0].src0);
    EXPECT_EQ(fields.src1, wmma[0].src1);
    EXPECT_EQ(fields.src2, static_cast<uint16_t>(256u + wmma[0].vdst));
    EXPECT_EQ(fields.neg, 0u);
  }

  const auto build_vop1_test = [](uint8_t op, uint8_t vdst, uint16_t src0) {
    return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
  };
  for (uint8_t word = 0; word < 8; ++word) {
    const uint32_t expected_mov = build_vop1_test(
        1, static_cast<uint8_t>(42u + word), static_cast<uint16_t>(256u + wmma[0].vdst + word));
    EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), expected_mov), cave_words.end())
        << "final scratch accumulator word should be copied to the original destination";
  }
  EXPECT_EQ((cave_words.back() >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250WmmaF32Fp8K64SplitsThroughRelayoutCave) {
  expect_gfx1250_fp8_wmma_splits_through_relayout_cave(0x6A, 4);
}

TEST(BinaryTranslator, Gfx1250WmmaF32Fp8K64AcceptsMatrixAReuseHint) {
  constexpr uint8_t kMatrixAReuse = 0x4;
  expect_gfx1250_fp8_wmma_splits_through_relayout_cave(0x6A, 4, kMatrixAReuse);
}

TEST(BinaryTranslator, Gfx1250WmmaF32MixedF8K64PreservesInputFormats) {
  expect_gfx1250_fp8_wmma_splits_through_relayout_cave(0x6B, 4);
  expect_gfx1250_fp8_wmma_splits_through_relayout_cave(0x6C, 4);
  expect_gfx1250_fp8_wmma_splits_through_relayout_cave(0x6D, 4);
}

void expect_gfx1250_f16_f8_wmma_k64_split(uint8_t gfx1250_op) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = gfx1250_op;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 32;
  inst.src2 = 256 + 64;
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;
  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  const uint8_t target_op = static_cast<uint8_t>(0x46u + (gfx1250_op - 0x6Eu));
  size_t wmma = 0;
  size_t unpack_f16 = 0;
  size_t pack_f16 = 0;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) == 0xCCu && ((w0 >> 16) & 0x7Fu) == target_op)
      ++wmma;
    if ((w0 >> 26) == 0x35u && ((w0 >> 16) & 0x3FFu) == 0x18Bu)
      ++unpack_f16;
    if ((w0 >> 26) == 0x35u && ((w0 >> 16) & 0x3FFu) == 0x18Au)
      ++pack_f16;
  }
  EXPECT_EQ(wmma, 4u);
  EXPECT_EQ(unpack_f16, 8u) << "four packed f16 accumulator words must widen to eight f32 lanes";
  EXPECT_EQ(pack_f16, 8u) << "the eight f32 results must narrow once after the final K16 chunk";
}

TEST(BinaryTranslator, Gfx1250WmmaF16F8K64UsesF32AccumulatorAndPreservesFormats) {
  for (uint8_t op = 0x6E; op <= 0x71; ++op)
    expect_gfx1250_f16_f8_wmma_k64_split(op);
}

TEST(BinaryTranslator, Gfx1250WmmaF32Fp8K128FallsBackThroughDot4Cave) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 80;
  inst.op = 0x80;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 32;
  inst.src2 = 256 + 64;
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];

  KernelDescriptorTranslator descriptor_parser(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto descriptor_infos = descriptor_parser.translate_image(
      result.elf_bytes, text->sectionOffset(), text->size(), KernelDescriptorTranslationOptions{});
  ASSERT_EQ(descriptor_infos.size(), 1u);
  EXPECT_EQ(descriptor_infos[0].target_lds_size, 0u);
  EXPECT_GE(descriptor_infos[0].target_private_size, 5u * sizeof(uint32_t))
      << "K128 fp8 scopes keep the private spill zone available for dense fallback cases";
  EXPECT_LT(descriptor_infos[0].guest_vgpr_count, 261u)
      << "K128 fp8 dot4 fallback should not raise the launch VGPR count for high-bank scratch";

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  size_t dot4 = 0;
  size_t wmma = 0;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu)
      continue;
    const uint8_t op = static_cast<uint8_t>((w0 >> 16) & 0x7Fu);
    if (op == 38u)
      ++dot4;
    if (op == 70u)
      ++wmma;
  }
  EXPECT_EQ(dot4, 2u * 8u * 32u)
      << "K128 fp8 updates each half wave once per output register and packed K group";
  EXPECT_EQ(wmma, 0u) << "K128 fp8 avoids the compact WMMA split because it is not bit-exact";
  constexpr uint32_t kRdna4SWaitDscnt0 = 0xBFC60000u;
  EXPECT_EQ(
      static_cast<size_t>(std::count(cave_words.begin(), cave_words.end(), kRdna4SWaitDscnt0)), 32u)
      << "K128 fp8 batches all eight output words behind one DS wait per packed K group";
  EXPECT_EQ((cave_words.back() >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

void expect_gfx1250_f8_wmma_k128_dot4(uint8_t gfx1250_op, bool f16_output) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 80;
  inst.op = gfx1250_op;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 32;
  inst.src2 = 256 + 64;
  inst.opsel_hi = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;
  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());
  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  constexpr std::array<uint8_t, 4> kDotOps{0x26, 0x24, 0x25, 0x27};
  const uint8_t format = static_cast<uint8_t>((gfx1250_op - (f16_output ? 0x84u : 0x80u)) & 3u);
  size_t dot4 = 0;
  size_t pack_f16 = 0;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) == 0xCCu && ((w0 >> 16) & 0x7Fu) == kDotOps[format])
      ++dot4;
    if ((w0 >> 26) == 0x35u && ((w0 >> 16) & 0x3FFu) == 0x18Au)
      ++pack_f16;
  }
  EXPECT_EQ(dot4, 2u * 8u * 32u);
  EXPECT_EQ(pack_f16, f16_output ? 8u : 0u);
}

TEST(BinaryTranslator, Gfx1250WmmaMixedF8K128PreservesFormatsThroughDot4Fallback) {
  for (uint8_t op = 0x81; op <= 0x83; ++op)
    expect_gfx1250_f8_wmma_k128_dot4(op, false);
  for (uint8_t op = 0x84; op <= 0x87; ++op)
    expect_gfx1250_f8_wmma_k128_dot4(op, true);
}

TEST(BinaryTranslator, Gfx1250WmmaF8f6f4ByteFormatsUseExactDot4Fallback) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint8_t, 4> kDotOps{0x26, 0x24, 0x25, 0x27};

  for (uint8_t matrix_a_fmt = 0; matrix_a_fmt <= 1; ++matrix_a_fmt) {
    for (uint8_t matrix_b_fmt = 0; matrix_b_fmt <= 1; ++matrix_b_fmt) {
      gfx1250::Vop3pMachineInst inst{};
      inst.vdst = 80;
      inst.op = 0x33;
      inst.encoding = 0xCC;
      inst.src0 = 256 + 8;
      inst.src1 = 256 + 32;
      inst.src2 = 256 + 64;
      inst.opsel = matrix_a_fmt;
      inst.opsel_hi = matrix_b_fmt;

      std::array<uint32_t, 3> text_words{};
      std::memcpy(text_words.data(), &inst, sizeof(inst));
      text_words[2] = kGfx1250SEndpgm;
      auto image = make_minimal_amdgpu_elf_with_descriptor_and_text(text_words,
                                                                    EF_AMDGPU_MACH_AMDGCN_GFX1250);
      AmdGpuCodeObject co(image.data(), image.size());
      ASSERT_TRUE(co.is_valid());

      BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                  EF_AMDGPU_MACH_AMDGCN_GFX1201);
      auto result = translator.translate(co);
      ASSERT_FALSE(result.elf_bytes.empty());
      EXPECT_TRUE(result.warnings.empty());
      AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(translated.is_valid());
      const Section *translations = find_section(translated, ".rj_translations");
      ASSERT_NE(translations, nullptr);
      const auto cave_words = section_words_for_test(*translations);

      const uint8_t expected_dot_op = kDotOps[(matrix_a_fmt << 1u) | matrix_b_fmt];
      size_t dot4 = 0;
      for (const uint32_t word : cave_words) {
        if ((word >> 24) == 0xCCu && ((word >> 16) & 0x7Fu) == expected_dot_op)
          ++dot4;
      }
      EXPECT_EQ(dot4, 2u * 8u * 32u);
    }
  }
}

TEST(BinaryTranslator, Gfx1250WmmaF8f6f4SubByteFormatsUseScalarFallback) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 80;
  inst.op = 0x33;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 32;
  inst.src2 = 256 + 64;
  inst.opsel = 2;    // FP6
  inst.opsel_hi = 2; // FP6

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;
  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());
  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  size_t fmac = 0;
  for (const uint32_t word : cave_words) {
    if ((word >> 26) == 0x35u && ((word >> 16) & 0x3FFu) == 299u)
      ++fmac;
  }
  EXPECT_EQ(fmac, 8u * 128u) << "the translation cave lowers every K update for all eight outputs";
}

TEST(BinaryTranslator, Gfx1250LocalCaveTerminatesActiveAppendedCaveChain) {
  constexpr uint32_t kGfx1250SClause1 = 0xBF850001u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21020u;
  constexpr size_t kWmmaCount = 48u;
  constexpr size_t kBranchWord = kWmmaCount * 2u;
  constexpr size_t kPaddingIslandWord = 0x1F000u / sizeof(uint32_t);
  constexpr size_t kFarBlockWord = kPaddingIslandWord + 1u;
  constexpr size_t kLocalDetourWord = 0x21000u / sizeof(uint32_t);
  constexpr size_t kRemoteRewriteWord = kLocalDetourWord + 1u;
  constexpr size_t kIndirectExitWord = kRemoteRewriteWord + 1u;

  // Keep the first large rewrite outside direct SOPP range of the appended
  // section.  The one-word padding hole can host a branch island, but not the
  // rewrite body.  An unresolved indirect exit suppresses whole-scope copying
  // and leaves a small padding run after the reachable CFG.
  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SClause1);
  gfx1250::Vop3pMachineInst wmma{};
  wmma.vdst = 80;
  wmma.op = 0x80;
  wmma.encoding = 0xCC;
  wmma.src0 = 256 + 8;
  wmma.src1 = 256 + 32;
  wmma.src2 = 256 + 64;
  wmma.opsel_hi = 3;
  for (size_t index = 0; index < kWmmaCount; ++index)
    std::memcpy(text_words.data() + index * 2u, &wmma, sizeof(wmma));

  text_words[kBranchWord] = build_s_branch(static_cast<int16_t>(kFarBlockWord - kBranchWord - 1u),
                                           ROCJITSU_CODE_ARCH_GFX1250);
  text_words[kPaddingIslandWord] = kGfx1250SNop0;

  // This scalar rewrite fits in the small local padding cave.  The following
  // u64 rewrite cannot fit there and must use the remote appended section.
  // Extending the old appended chain across the local branch stub would copy
  // that stub into the chain and then NOP its return path.
  text_words[kLocalDetourWord] = 0xBE840080u;          // s_mov_b32 s4, 0
  text_words[kRemoteRewriteWord] = 0x5014150Cu;        // v_add_nc_u64 v[10:11], v[12:13], v[10:11]
  text_words[kIndirectExitWord] = pack_sop1(72, 0, 2); // s_setpc_b64 s[2:3]
  for (size_t word = kIndirectExitWord + 1u; word < text_words.size(); ++word)
    text_words[word] = kGfx1250SNop0;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto patched_words = section_words_for_test(*translated.text_sections()[0]);
  ASSERT_GT(patched_words.size(), kRemoteRewriteWord);
  EXPECT_EQ((patched_words[kLocalDetourWord] >> 16) & 0x7Fu,
            sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4))
      << "a later appended rewrite must not absorb and NOP the local detour stub";
  EXPECT_EQ((patched_words[kRemoteRewriteWord] >> 16) & 0x7Fu,
            sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250WmmaScaleF32Fp8K128LowersThroughScalarFallback) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 4> kScaledWmmaPair{
      0xCC350000u,
      0x02020900u,
      0xCC330006u,
      0x02026912u,
  };
  std::array<uint32_t, 5> text_words{
      kScaledWmmaPair[0], kScaledWmmaPair[1], kScaledWmmaPair[2],
      kScaledWmmaPair[3], kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[4], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  size_t mul_f32 = 0;
  size_t add_f32 = 0;
  bool raw_pair_survived = false;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    if (i + kScaledWmmaPair.size() <= cave_words.size() &&
        std::equal(kScaledWmmaPair.begin(), kScaledWmmaPair.end(), cave_words.begin() + i)) {
      raw_pair_survived = true;
    }

    const uint32_t w0 = cave_words[i];
    if (((w0 >> 26) & 0x3Fu) != 0x35u)
      continue;
    const uint16_t op = static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
    if (op == 264u)
      ++mul_f32;
    if (op == 259u)
      ++add_f32;
  }

  EXPECT_FALSE(raw_pair_survived) << "scaled gfx1250 WMMA pair must not be copied into RDNA4 code";
  EXPECT_EQ(mul_f32, 3u * 8u * 128u)
      << "each scalar FP8 product must apply both A and B scale factors";
  EXPECT_EQ(add_f32, 8u * 128u) << "each scalar FP8 product contributes one f32 accumulation";
  EXPECT_NE(std::find_if(cave_words.begin(), cave_words.end(),
                         [](uint32_t word) {
                           return ((word >> 16) & 0x7Fu) ==
                                  sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4);
                         }),
            cave_words.end());
}

TEST(BinaryTranslator, Gfx1250WmmaScaleF32Fp8K128AcceptsInlineZeroScales) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 4> kScaledWmmaPair{
      0xCC350000u,
      0x02010080u,
      0xCC330000u,
      0x02024966u,
  };
  std::array<uint32_t, 5> text_words{
      kScaledWmmaPair[0], kScaledWmmaPair[1], kScaledWmmaPair[2],
      kScaledWmmaPair[3], kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[4], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  size_t mul_f32 = 0;
  size_t add_f32 = 0;
  size_t f32_one_literals = 0;
  bool raw_pair_survived = false;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    if (i + kScaledWmmaPair.size() <= cave_words.size() &&
        std::equal(kScaledWmmaPair.begin(), kScaledWmmaPair.end(), cave_words.begin() + i)) {
      raw_pair_survived = true;
    }

    const uint32_t w0 = cave_words[i];
    if (w0 == 0x3F800000u)
      ++f32_one_literals;
    if (((w0 >> 26) & 0x3Fu) != 0x35u)
      continue;
    const uint16_t op = static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
    if (op == 264u)
      ++mul_f32;
    if (op == 259u)
      ++add_f32;
  }

  EXPECT_FALSE(raw_pair_survived)
      << "scaled gfx1250 WMMA with inline scale operands must not be copied into RDNA4 code";
  EXPECT_EQ(mul_f32, 3u * 8u * 128u);
  EXPECT_EQ(add_f32, 8u * 128u);
  EXPECT_EQ(f32_one_literals, 2u * 8u * 128u)
      << "inline zero scale operands select default unity scaling for each A/B scale";
}

TEST(BinaryTranslator, Gfx1250WmmaScaleF32Fp4K128LowersThroughScalarFallback) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst scale{};
  scale.op = 0x35;
  scale.encoding = 0xCC;
  scale.src0 = 256 + 0;
  scale.src1 = 256 + 4;

  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x33;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 24;
  inst.opsel = 4;
  inst.pad_14 = 1;

  std::array<uint32_t, 5> text_words{};
  std::memcpy(text_words.data(), &scale, sizeof(scale));
  std::memcpy(text_words.data() + 2, &inst, sizeof(inst));
  text_words[4] = kGfx1250SEndpgm;
  const std::array<uint32_t, 4> raw_pair{
      text_words[0],
      text_words[1],
      text_words[2],
      text_words[3],
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[4], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  size_t mul_f32 = 0;
  size_t add_f32 = 0;
  bool raw_pair_survived = false;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    if (i + raw_pair.size() <= cave_words.size() &&
        std::equal(raw_pair.begin(), raw_pair.end(), cave_words.begin() + i)) {
      raw_pair_survived = true;
    }

    const uint32_t w0 = cave_words[i];
    if (((w0 >> 26) & 0x3Fu) != 0x35u)
      continue;
    const uint16_t op = static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
    if (op == 264u)
      ++mul_f32;
    if (op == 259u)
      ++add_f32;
  }

  EXPECT_FALSE(raw_pair_survived)
      << "scaled FP4 gfx1250 WMMA pair must not be copied into RDNA4 code";
  EXPECT_EQ(mul_f32, 3u * 8u * 128u)
      << "each scalar FP4 product must apply both A and B scale factors";
  EXPECT_EQ(add_f32, 8u * 128u) << "each scalar FP4 product contributes one f32 accumulation";
}

TEST(BinaryTranslator, Gfx1250WmmaScale16F32Fp4K128LowersThroughScalarFallback) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst scale{};
  scale.op = 0x3A;
  scale.encoding = 0xCC;
  scale.src0 = 256 + 0;
  scale.src1 = 256 + 2;

  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x33;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 24;
  inst.opsel = 4;
  inst.pad_14 = 1;

  std::array<uint32_t, 5> text_words{};
  std::memcpy(text_words.data(), &scale, sizeof(scale));
  std::memcpy(text_words.data() + 2, &inst, sizeof(inst));
  text_words[4] = kGfx1250SEndpgm;
  const std::array<uint32_t, 4> raw_pair{
      text_words[0],
      text_words[1],
      text_words[2],
      text_words[3],
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[4], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  size_t mul_f32 = 0;
  size_t add_f32 = 0;
  bool raw_pair_survived = false;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    if (i + raw_pair.size() <= cave_words.size() &&
        std::equal(raw_pair.begin(), raw_pair.end(), cave_words.begin() + i)) {
      raw_pair_survived = true;
    }

    const uint32_t w0 = cave_words[i];
    if (((w0 >> 26) & 0x3Fu) != 0x35u)
      continue;
    const uint16_t op = static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
    if (op == 264u)
      ++mul_f32;
    if (op == 259u)
      ++add_f32;
  }

  EXPECT_FALSE(raw_pair_survived)
      << "scale16 FP4 gfx1250 WMMA pair must not be copied into RDNA4 code";
  EXPECT_EQ(mul_f32, 3u * 8u * 128u)
      << "each scalar FP4 product must apply both A and B scale factors";
  EXPECT_EQ(add_f32, 8u * 128u) << "each scalar FP4 product contributes one f32 accumulation";
}

TEST(BinaryTranslator, Gfx1250WmmaScaleF32Fp4K128SemanticTempsGrowDescriptor) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst scale{};
  scale.op = 0x35;
  scale.encoding = 0xCC;
  scale.src0 = 256 + 24;
  scale.src1 = 256 + 25;

  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 16;
  inst.op = 0x33;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 0;
  inst.src1 = 256 + 8;
  inst.src2 = scalar_positive_inline_u32(0);
  inst.opsel = 4;
  inst.pad_14 = 1;

  std::vector<uint32_t> text_words;
  text_words.resize(4);
  std::memcpy(text_words.data(), &scale, sizeof(scale));
  std::memcpy(text_words.data() + 2, &inst, sizeof(inst));

  auto append_live_self_move = [&](uint8_t reg) {
    gfx1250::Vop1MachineInst mov{};
    mov.encoding = 0x3F;
    mov.op = 1;
    mov.vdst = reg;
    mov.src0 = static_cast<uint16_t>(256u + reg);
    text_words.push_back(std::bit_cast<uint32_t>(mov));
  };
  for (uint16_t reg = 26; reg < 206; ++reg)
    append_live_self_move(static_cast<uint8_t>(reg));
  text_words.push_back(kGfx1250SEndpgm);

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *rodata = find_section(translated, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(KD));

  const auto *desc = reinterpret_cast<const KD *>(rodata->data());
  const uint32_t granulated = AMDHSA_BITS_GET(desc->compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t addressable_vgprs = (granulated + 1u) * 8u;
  EXPECT_GT(addressable_vgprs, 215u)
      << "scaled FP4 K128 fallback uses a ten-VGPR temporary run after the live source footprint";
}

TEST(BinaryTranslator, Gfx1250WmmaF32Fp8K128BorrowsLowScratchThroughPrivateSpill) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 64;
  inst.op = 0x80;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 0;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 64;
  inst.opsel_hi = 3;

  std::vector<uint32_t> text_words;
  text_words.resize(2);
  std::memcpy(text_words.data(), &inst, sizeof(inst));

  auto append_live_self_move = [&](uint8_t reg) {
    gfx1250::Vop1MachineInst mov{};
    mov.encoding = 0x3F;
    mov.op = 1;
    mov.vdst = reg;
    mov.src0 = static_cast<uint16_t>(256u + reg);
    text_words.push_back(std::bit_cast<uint32_t>(mov));
  };
  for (uint16_t reg = 32; reg < 64; ++reg)
    append_live_self_move(static_cast<uint8_t>(reg));
  for (uint16_t reg = 72; reg < 256; ++reg)
    append_live_self_move(static_cast<uint8_t>(reg));
  text_words.push_back(kGfx1250SEndpgm);

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];

  KernelDescriptorTranslator descriptor_parser(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto descriptor_infos = descriptor_parser.translate_image(
      result.elf_bytes, text->sectionOffset(), text->size(), KernelDescriptorTranslationOptions{});
  ASSERT_EQ(descriptor_infos.size(), 1u);
  EXPECT_EQ(descriptor_infos[0].target_lds_size, 0u);
  EXPECT_EQ(descriptor_infos[0].target_private_size,
            kGfx1250PrivateBorrowedVgprCountForTest * sizeof(uint32_t));
  EXPECT_LT(descriptor_infos[0].guest_vgpr_count, 261u);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  size_t entry_mode_resets = 0;
  size_t non_entry_mode_writes = 0;
  const uint32_t mode_setreg = rdna4_vgpr_msb_reset_setreg_for_test();
  for (size_t i = 0; i < cave_words.size(); ++i) {
    if (cave_words[i] != mode_setreg)
      continue;
    if (i + 1 < cave_words.size() && cave_words[i + 1] == 0)
      ++entry_mode_resets;
    else
      ++non_entry_mode_writes;
  }
  EXPECT_LE(entry_mode_resets, 1u) << "at most the kernel-entry VGPR_MSB reset is expected";
  EXPECT_EQ(non_entry_mode_writes, 0u)
      << "borrowed low scratch must not depend on RDNA4 VGPR_MSB mode writes";

  size_t stores = 0;
  size_t loads = 0;
  size_t wmma = 0;
  size_t first_store = cave_words.size();
  size_t first_load = cave_words.size();
  constexpr uint32_t kScratchStoreB32OffOff = 0xED06807Cu;
  constexpr uint32_t kScratchLoadB32OffOff = 0xED05007Cu;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    if (cave_words[i] == kScratchStoreB32OffOff) {
      if (first_store == cave_words.size())
        first_store = i;
      ++stores;
    }
    if (cave_words[i] == kScratchLoadB32OffOff) {
      if (first_load == cave_words.size())
        first_load = i;
      ++loads;
    }

    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) != 0xCCu || ((w0 >> 16) & 0x7Fu) != 70u)
      continue;
    const uint32_t w1 = cave_words[i + 1];
    ++wmma;
    EXPECT_EQ(static_cast<uint8_t>(w0 & 0xFFu), 64u);
    EXPECT_EQ(static_cast<uint16_t>(w1 & 0x1FFu), static_cast<uint16_t>(256u + 32u));
    EXPECT_EQ(static_cast<uint16_t>((w1 >> 9) & 0x1FFu), static_cast<uint16_t>(256u + 34u));
  }
  EXPECT_EQ(stores, 5u);
  EXPECT_EQ(loads, 5u);
  EXPECT_EQ(wmma, 8u);
  ASSERT_NE(first_store, cave_words.size());
  ASSERT_NE(first_load, cave_words.size());
  EXPECT_NE(std::find(cave_words.begin(), cave_words.begin() + first_store, pack_sopp(64, 0)),
            cave_words.begin() + first_store)
      << "borrow-save must not race pending vector-memory loads into borrowed VGPRs";
  EXPECT_NE(std::find(cave_words.begin(), cave_words.begin() + first_store, pack_sopp(70, 0)),
            cave_words.begin() + first_store)
      << "borrow-save must not race pending LDS loads into borrowed VGPRs";
  ASSERT_LT(first_store + stores * 3u, cave_words.size());
  EXPECT_EQ(cave_words[first_store + stores * 3u], pack_sopp(65, 0))
      << "scratch stores must complete before borrowed VGPRs are reused";
  ASSERT_LT(first_load + loads * 3u, cave_words.size());
  EXPECT_EQ(cave_words[first_load + loads * 3u], pack_sopp(64, 0))
      << "scratch restores must complete before EXEC is restored";
}

TEST(BinaryTranslator, Gfx1250WmmaF32F8f6f4Fp4Fp4LowersThroughDot8Fallback) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 42;
  inst.op = 0x33;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 8;
  inst.src1 = 256 + 16;
  inst.src2 = 256 + 24;
  inst.opsel = 4;
  inst.pad_14 = 1;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  size_t dot8 = 0;
  size_t fmac = 0;
  for (size_t i = 0; i + 1 < cave_words.size(); ++i) {
    const uint32_t w0 = cave_words[i];
    if ((w0 >> 24) == 0xCCu && ((w0 >> 16) & 0x7Fu) == 0x19u)
      ++dot8;
    if (((w0 >> 26) & 0x3Fu) == 0x35u && ((w0 >> 16) & 0x3FFu) == 299u)
      ++fmac;
  }

  EXPECT_EQ(dot8, 8u * 16u * 4u)
      << "each output and FP4 K-group should lower through four unsigned u4 dot ops";
  EXPECT_EQ(fmac, 8u * 16u) << "each FP4 K-group contributes one scaled f32 accumulation";
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), 0x3E800000u), 8u * 16u)
      << "scaled FP4 integer dots must be multiplied by 0.25 before accumulation";
  EXPECT_NE(std::find_if(cave_words.begin(), cave_words.end(),
                         [](uint32_t word) {
                           return ((word >> 16) & 0x7Fu) ==
                                  sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4);
                         }),
            cave_words.end());
}

TEST(BinaryTranslator, Gfx1250WmmaF32K4LowersToFmacCave) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3pMachineInst inst{};
  inst.vdst = 2;
  inst.op = 0x5D;
  inst.encoding = 0xCC;
  inst.src0 = 256 + 10;
  inst.src1 = 256 + 40;
  inst.src2 = scalar_positive_inline_u32(0);
  inst.opsel = 0x4;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  constexpr uint16_t kRdna4FmacF32Vop3 = 299;
  size_t fmac_count = 0;
  for (uint32_t word : cave_words) {
    if ((word >> 26) == 0x35u && ((word >> 16) & 0x3FFu) == kRdna4FmacF32Vop3)
      ++fmac_count;
  }
  EXPECT_EQ(fmac_count, 32u) << "f32 K4 WMMA should lower to 8 outputs x 4 fmac terms";
  EXPECT_EQ((cave_words.back() >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VBitop3B32C8LowersInPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 3;
  inst.op = 564;
  inst.encoding = 0x35;
  inst.src0 = 10;
  inst.src1 = scalar_positive_inline_u32(31);
  inst.src2 = 256 + 1;
  inst.abs = 1;
  inst.omod = 3;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto or_inst = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[0]);
  EXPECT_EQ(or_inst.op, 28u);
  EXPECT_EQ(or_inst.vdst, 3u);
  EXPECT_EQ(or_inst.src0, 10u);
  EXPECT_EQ(or_inst.vsrc1, 1u);

  const auto and_inst = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[1]);
  EXPECT_EQ(and_inst.op, 27u);
  EXPECT_EQ(and_inst.vdst, 3u);
  EXPECT_EQ(and_inst.src0, scalar_positive_inline_u32(31));
  EXPECT_EQ(and_inst.vsrc1, 3u);
  EXPECT_EQ(patched_text[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250PermlaneFamilyLowersThroughRdna4BpermuteFi) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kRdna4DsBpermuteFiW0 = (0xCDu << 18) | (0x36u << 26);
  constexpr std::array<uint16_t, 4> kOps = {0x270u, 0x271u, 0x272u, 0x273u};

  for (uint16_t op : kOps) {
    SCOPED_TRACE(op);
    gfx1250::Vop3MachineInst inst{};
    inst.vdst = 5;
    inst.op = op;
    inst.encoding = 0x35;
    inst.src0 = 256 + 1;
    inst.src1 = 2;
    inst.src2 = 3;

    std::array<uint32_t, 3> text_words{};
    std::memcpy(text_words.data(), &inst, sizeof(inst));
    text_words[2] = kGfx1250SEndpgm;

    auto image =
        make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
    AmdGpuCodeObject co(image.data(), image.size());
    ASSERT_TRUE(co.is_valid());

    BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                EF_AMDGPU_MACH_AMDGCN_GFX1201);
    auto result = translator.translate(co);
    ASSERT_FALSE(result.elf_bytes.empty());
    EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

    AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    ASSERT_FALSE(translated.text_sections().empty());
    auto lowered_words = section_words_for_test(*translated.text_sections()[0]);
    if (const Section *translations = find_section(translated, ".rj_translations")) {
      const auto cave_words = section_words_for_test(*translations);
      lowered_words.insert(lowered_words.end(), cave_words.begin(), cave_words.end());
    }

    bool found_bpermute_fi = false;
    for (size_t i = 0; i + 1u < lowered_words.size(); ++i) {
      if (lowered_words[i] != kRdna4DsBpermuteFiW0)
        continue;
      const uint32_t operands = lowered_words[i + 1u];
      EXPECT_EQ((operands >> 8) & 0xFFu, 1u);
      EXPECT_EQ((operands >> 24) & 0xFFu, 5u);
      found_bpermute_fi = true;
      break;
    }
    EXPECT_TRUE(found_bpermute_fi);
  }
}

TEST(BinaryTranslator, Gfx1250PermlaneIdxGenLowersToRdna4Xor) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 5;
  inst.op = 0x314;
  inst.encoding = 0x35;
  inst.src0 = 256 + 1;
  inst.src1 = 2;

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto lowered_words = section_words_for_test(*translated.text_sections()[0]);
  ASSERT_FALSE(lowered_words.empty());
  const auto xor_inst = std::bit_cast<rdna4::Vop2MachineInst>(lowered_words[0]);
  EXPECT_EQ(xor_inst.op, 29u);
  EXPECT_EQ(xor_inst.vdst, 5u);
  EXPECT_EQ(xor_inst.src0, 2u);
  EXPECT_EQ(xor_inst.vsrc1, 1u);
}

TEST(BinaryTranslator, Gfx1250VBitop3B32C8LiteralLowersInPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kMaskLiteral = 0x0FFFFFF0u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 2;
  inst.op = 564;
  inst.encoding = 0x35;
  inst.src0 = 256 + 7;
  inst.src1 = 255;
  inst.src2 = 34;
  inst.abs = 1;
  inst.omod = 3;

  std::array<uint32_t, 4> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kMaskLiteral;
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto or_inst = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[0]);
  EXPECT_EQ(or_inst.op, 28u);
  EXPECT_EQ(or_inst.vdst, 2u);
  EXPECT_EQ(or_inst.src0, 34u);
  EXPECT_EQ(or_inst.vsrc1, 7u);

  const auto and_inst = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[1]);
  EXPECT_EQ(and_inst.op, 27u);
  EXPECT_EQ(and_inst.vdst, 2u);
  EXPECT_EQ(and_inst.src0, 255u);
  EXPECT_EQ(and_inst.vsrc1, 2u);
  EXPECT_EQ(patched_text[2], kMaskLiteral);
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250VBitop3B326cPreservesAliasedS1ThroughScratch) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kMaskLiteral = 0x000000FFu;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 2;
  inst.op = 564;
  inst.encoding = 0x35;
  inst.src0 = 256 + 4;
  inst.src1 = 256 + 2;
  inst.src2 = 255;
  inst.neg = 4;
  inst.abs = 5;
  inst.omod = 1;

  std::array<uint32_t, 4> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kMaskLiteral;
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  auto lowered_words = section_words_for_test(*translated.text_sections()[0]);
  const auto cave_words = section_words_for_test(*translations);
  lowered_words.insert(lowered_words.end(), cave_words.begin(), cave_words.end());

  const auto and_base = find_vop2_for_test(lowered_words, 27u, std::nullopt, 255u, 4u);
  ASSERT_TRUE(and_base.has_value());
  ASSERT_LT(*and_base + 1u, lowered_words.size());
  const auto and_inst = std::bit_cast<rdna4::Vop2MachineInst>(lowered_words[*and_base]);
  EXPECT_NE(and_inst.vdst, 2u);
  EXPECT_EQ(lowered_words[*and_base + 1u], kMaskLiteral);

  const auto xor_base = find_vop2_for_test(lowered_words, 29u, 2u, 256u + 2u, and_inst.vdst);
  EXPECT_TRUE(xor_base.has_value()) << "S1 must be read after the product is computed in scratch";
}

TEST(BinaryTranslator, Gfx1250VBitop3B3278PreservesAliasedS1ThroughScratch) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kMaskLiteral = 0x7FFFFFFFu;
  constexpr std::array<uint32_t, 4> text_words = {0xD6340702u, 0x0BFE0503u, kMaskLiteral,
                                                  kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  auto lowered_words = section_words_for_test(*translated.text_sections()[0]);
  const auto cave_words = section_words_for_test(*translations);
  lowered_words.insert(lowered_words.end(), cave_words.begin(), cave_words.end());

  const auto and_base = find_vop2_for_test(lowered_words, 27u, std::nullopt, 255u, 2u);
  ASSERT_TRUE(and_base.has_value());
  ASSERT_LT(*and_base + 1u, lowered_words.size());
  EXPECT_EQ(lowered_words[*and_base + 1u], kMaskLiteral);
  const auto and_inst = std::bit_cast<rdna4::Vop2MachineInst>(lowered_words[*and_base]);
  EXPECT_NE(and_inst.vdst, 2u);

  const auto xor_base = find_vop2_for_test(lowered_words, 29u, 2u, 256u + 3u, and_inst.vdst);
  ASSERT_TRUE(xor_base.has_value()) << "S1 must be consumed before the aliased VDST write";
}

TEST(BinaryTranslator, Gfx1250VBitop3B16EcHighDstLowersThroughB32MaskMerge) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kMaskLiteral = 0x000000FFu;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 5;
  inst.op = 563;
  inst.encoding = 0x35;
  inst.src0 = 256 + 5;
  inst.src1 = 256 + 8;
  inst.src2 = 255;
  inst.opsel = 0x9; // src0.h, src1.l, src2.l, vdst.h.
  inst.abs = 5;
  inst.omod = 3;
  inst.neg = 4;

  std::array<uint32_t, 4> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kMaskLiteral;
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto translation_words = section_words_for_test(*translations);
  const auto base =
      find_vop2_for_test(translation_words, 25u, std::nullopt, scalar_positive_inline_u32(16), 5u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 13u, translation_words.size());
  std::array<uint32_t, 13> cave_words{};
  std::copy_n(translation_words.begin() + static_cast<std::ptrdiff_t>(*base), cave_words.size(),
              cave_words.begin());

  const auto src0_hi = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[0]);
  ASSERT_EQ(src0_hi.op, 25u);
  const uint8_t tmp0 = static_cast<uint8_t>(src0_hi.vdst);
  const uint8_t tmp1 = static_cast<uint8_t>(tmp0 + 1u);
  const uint8_t tmp2 = static_cast<uint8_t>(tmp0 + 2u);
  const uint8_t result_tmp = static_cast<uint8_t>(tmp0 + 3u);
  EXPECT_EQ(src0_hi.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(src0_hi.vsrc1, 5u);

  const auto src1_lo = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[1]);
  EXPECT_EQ(src1_lo.op, 28u);
  EXPECT_EQ(src1_lo.vdst, tmp1);
  EXPECT_EQ(src1_lo.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(src1_lo.vsrc1, 8u);

  const auto src2_lit = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[2]);
  EXPECT_EQ(src2_lit.op, 1u);
  EXPECT_EQ(src2_lit.vdst, tmp2);
  EXPECT_EQ(src2_lit.src0, 255u);
  EXPECT_EQ(cave_words[3], kMaskLiteral);

  const auto and_inst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[4]);
  EXPECT_EQ(and_inst.op, 27u);
  EXPECT_EQ(and_inst.vdst, result_tmp);
  EXPECT_EQ(and_inst.src0, 256u + tmp0);
  EXPECT_EQ(and_inst.vsrc1, tmp2);

  const auto or_inst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[5]);
  EXPECT_EQ(or_inst.op, 28u);
  EXPECT_EQ(or_inst.vdst, result_tmp);
  EXPECT_EQ(or_inst.src0, 256u + tmp1);
  EXPECT_EQ(or_inst.vsrc1, result_tmp);

  const auto mask_result = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[6]);
  EXPECT_EQ(mask_result.op, 27u);
  EXPECT_EQ(mask_result.vdst, result_tmp);
  EXPECT_EQ(mask_result.src0, 255u);
  EXPECT_EQ(mask_result.vsrc1, result_tmp);
  EXPECT_EQ(cave_words[7], 0x0000FFFFu);

  const auto shift_result = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[8]);
  EXPECT_EQ(shift_result.op, 24u);
  EXPECT_EQ(shift_result.vdst, result_tmp);
  EXPECT_EQ(shift_result.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(shift_result.vsrc1, result_tmp);

  const auto preserve_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[9]);
  EXPECT_EQ(preserve_dst.op, 27u);
  EXPECT_EQ(preserve_dst.vdst, 5u);
  EXPECT_EQ(preserve_dst.src0, 255u);
  EXPECT_EQ(preserve_dst.vsrc1, 5u);
  EXPECT_EQ(cave_words[10], 0x0000FFFFu);

  const auto merge_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[11]);
  EXPECT_EQ(merge_dst.op, 28u);
  EXPECT_EQ(merge_dst.vdst, 5u);
  EXPECT_EQ(merge_dst.src0, 256u + result_tmp);
  EXPECT_EQ(merge_dst.vsrc1, 5u);
  EXPECT_EQ((cave_words[12] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VBitop3B16F8LowDstLowersThroughB32MaskMerge) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kMaskLiteral = 0x0000FF00u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 6;
  inst.op = 563;
  inst.encoding = 0x35;
  inst.src0 = 256 + 4;
  inst.src1 = 256 + 6;
  inst.src2 = 255;
  inst.opsel = 0x2; // src0.l, src1.h, src2.l, vdst.l.
  inst.abs = 7;
  inst.omod = 3;

  std::array<uint32_t, 4> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kMaskLiteral;
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto translation_words = section_words_for_test(*translations);
  const auto base =
      find_vop2_for_test(translation_words, 28u, std::nullopt, scalar_positive_inline_u32(0), 4u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 12u, translation_words.size());
  std::array<uint32_t, 12> cave_words{};
  std::copy_n(translation_words.begin() + static_cast<std::ptrdiff_t>(*base), cave_words.size(),
              cave_words.begin());

  const auto src0_lo = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[0]);
  ASSERT_EQ(src0_lo.op, 28u);
  const uint8_t tmp0 = static_cast<uint8_t>(src0_lo.vdst);
  const uint8_t tmp1 = static_cast<uint8_t>(tmp0 + 1u);
  const uint8_t tmp2 = static_cast<uint8_t>(tmp0 + 2u);
  const uint8_t result_tmp = static_cast<uint8_t>(tmp0 + 3u);
  EXPECT_EQ(src0_lo.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(src0_lo.vsrc1, 4u);

  const auto src1_hi = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[1]);
  EXPECT_EQ(src1_hi.op, 25u);
  EXPECT_EQ(src1_hi.vdst, tmp1);
  EXPECT_EQ(src1_hi.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(src1_hi.vsrc1, 6u);

  const auto src2_lit = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[2]);
  EXPECT_EQ(src2_lit.op, 1u);
  EXPECT_EQ(src2_lit.vdst, tmp2);
  EXPECT_EQ(src2_lit.src0, 255u);
  EXPECT_EQ(cave_words[3], kMaskLiteral);

  const auto and_inst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[4]);
  EXPECT_EQ(and_inst.op, 27u);
  EXPECT_EQ(and_inst.vdst, result_tmp);
  EXPECT_EQ(and_inst.src0, 256u + tmp1);
  EXPECT_EQ(and_inst.vsrc1, tmp2);

  const auto or_inst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[5]);
  EXPECT_EQ(or_inst.op, 28u);
  EXPECT_EQ(or_inst.vdst, result_tmp);
  EXPECT_EQ(or_inst.src0, 256u + tmp0);
  EXPECT_EQ(or_inst.vsrc1, result_tmp);

  const auto mask_result = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[6]);
  EXPECT_EQ(mask_result.op, 27u);
  EXPECT_EQ(mask_result.vdst, result_tmp);
  EXPECT_EQ(mask_result.src0, 255u);
  EXPECT_EQ(mask_result.vsrc1, result_tmp);
  EXPECT_EQ(cave_words[7], 0x0000FFFFu);

  const auto preserve_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[8]);
  EXPECT_EQ(preserve_dst.op, 27u);
  EXPECT_EQ(preserve_dst.vdst, 6u);
  EXPECT_EQ(preserve_dst.src0, 255u);
  EXPECT_EQ(preserve_dst.vsrc1, 6u);
  EXPECT_EQ(cave_words[9], 0xFFFF0000u);

  const auto merge_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[10]);
  EXPECT_EQ(merge_dst.op, 28u);
  EXPECT_EQ(merge_dst.vdst, 6u);
  EXPECT_EQ(merge_dst.src0, 256u + result_tmp);
  EXPECT_EQ(merge_dst.vsrc1, 6u);
  EXPECT_EQ((cave_words[11] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VBitop3B16FeHighDstLowersThroughB32OrMerge) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kLiteral = 0x00000F0Eu;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 3;
  inst.op = 563;
  inst.encoding = 0x35;
  inst.src0 = 256;
  inst.src1 = 255;
  inst.src2 = 256 + 4;
  inst.opsel = 0x8; // src0.l, src1.l, src2.l, vdst.h.
  inst.abs = 7;
  inst.omod = 3;
  inst.neg = 6;

  std::array<uint32_t, 4> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kLiteral;
  text_words[3] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto translation_words = section_words_for_test(*translations);
  const auto base =
      find_vop2_for_test(translation_words, 28u, std::nullopt, scalar_positive_inline_u32(0), 0u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 13u, translation_words.size());
  std::array<uint32_t, 13> cave_words{};
  std::copy_n(translation_words.begin() + static_cast<std::ptrdiff_t>(*base), cave_words.size(),
              cave_words.begin());

  const auto src0_lo = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[0]);
  ASSERT_EQ(src0_lo.op, 28u);
  const uint8_t tmp0 = static_cast<uint8_t>(src0_lo.vdst);
  const uint8_t tmp1 = static_cast<uint8_t>(tmp0 + 1u);
  const uint8_t tmp2 = static_cast<uint8_t>(tmp0 + 2u);
  const uint8_t result_tmp = static_cast<uint8_t>(tmp0 + 3u);
  EXPECT_EQ(src0_lo.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(src0_lo.vsrc1, 0u);

  const auto src1_lit = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[1]);
  EXPECT_EQ(src1_lit.op, 1u);
  EXPECT_EQ(src1_lit.vdst, tmp1);
  EXPECT_EQ(src1_lit.src0, 255u);
  EXPECT_EQ(cave_words[2], kLiteral);

  const auto src2_lo = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[3]);
  EXPECT_EQ(src2_lo.op, 28u);
  EXPECT_EQ(src2_lo.vdst, tmp2);
  EXPECT_EQ(src2_lo.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(src2_lo.vsrc1, 4u);

  const auto or_src01 = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[4]);
  EXPECT_EQ(or_src01.op, 28u);
  EXPECT_EQ(or_src01.vdst, result_tmp);
  EXPECT_EQ(or_src01.src0, 256u + tmp0);
  EXPECT_EQ(or_src01.vsrc1, tmp1);

  const auto or_src2 = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[5]);
  EXPECT_EQ(or_src2.op, 28u);
  EXPECT_EQ(or_src2.vdst, result_tmp);
  EXPECT_EQ(or_src2.src0, 256u + tmp2);
  EXPECT_EQ(or_src2.vsrc1, result_tmp);

  const auto mask_result = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[6]);
  EXPECT_EQ(mask_result.op, 27u);
  EXPECT_EQ(mask_result.vdst, result_tmp);
  EXPECT_EQ(mask_result.src0, 255u);
  EXPECT_EQ(mask_result.vsrc1, result_tmp);
  EXPECT_EQ(cave_words[7], 0x0000FFFFu);

  const auto shift_result = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[8]);
  EXPECT_EQ(shift_result.op, 24u);
  EXPECT_EQ(shift_result.vdst, result_tmp);
  EXPECT_EQ(shift_result.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(shift_result.vsrc1, result_tmp);

  const auto preserve_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[9]);
  EXPECT_EQ(preserve_dst.op, 27u);
  EXPECT_EQ(preserve_dst.vdst, 3u);
  EXPECT_EQ(preserve_dst.src0, 255u);
  EXPECT_EQ(preserve_dst.vsrc1, 3u);
  EXPECT_EQ(cave_words[10], 0x0000FFFFu);

  const auto merge_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[11]);
  EXPECT_EQ(merge_dst.op, 28u);
  EXPECT_EQ(merge_dst.vdst, 3u);
  EXPECT_EQ(merge_dst.src0, 256u + result_tmp);
  EXPECT_EQ(merge_dst.vsrc1, 3u);
  EXPECT_EQ((cave_words[12] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VLshlrevB16HighDstLowersThroughB32MaskMerge) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 9;
  inst.op = 824;
  inst.encoding = 0x35;
  inst.src0 = scalar_positive_inline_u32(8);
  inst.src1 = 256 + 4;
  inst.opsel = 0x8; // src0.l, src1.l, vdst.h.

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base = find_vop1_for_test(cave_words, 1u, std::nullopt, scalar_positive_inline_u32(8));
  ASSERT_TRUE(base.has_value());
  ASSERT_LT(*base + 11u, cave_words.size());

  const auto shift_mov = std::bit_cast<rdna4::Vop1MachineInst>(cave_words[*base]);
  ASSERT_EQ(shift_mov.op, 1u);
  const uint8_t tmp_shift = static_cast<uint8_t>(shift_mov.vdst);
  const uint8_t tmp_value = static_cast<uint8_t>(tmp_shift + 1u);
  const uint8_t result_tmp = static_cast<uint8_t>(tmp_shift + 2u);
  EXPECT_EQ(shift_mov.src0, scalar_positive_inline_u32(8));

  const auto value_lo = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 1u]);
  EXPECT_EQ(value_lo.op, 28u);
  EXPECT_EQ(value_lo.vdst, tmp_value);
  EXPECT_EQ(value_lo.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(value_lo.vsrc1, 4u);

  const auto mask_shift = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 2u]);
  EXPECT_EQ(mask_shift.op, 27u);
  EXPECT_EQ(mask_shift.vdst, tmp_shift);
  EXPECT_EQ(mask_shift.src0, 255u);
  EXPECT_EQ(mask_shift.vsrc1, tmp_shift);
  EXPECT_EQ(cave_words[*base + 3u], 0x0000000Fu);

  const auto lshl = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 4u]);
  EXPECT_EQ(lshl.op, 24u);
  EXPECT_EQ(lshl.vdst, result_tmp);
  EXPECT_EQ(lshl.src0, 256u + tmp_shift);
  EXPECT_EQ(lshl.vsrc1, tmp_value);

  const auto mask_result = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 5u]);
  EXPECT_EQ(mask_result.op, 27u);
  EXPECT_EQ(mask_result.vdst, result_tmp);
  EXPECT_EQ(mask_result.src0, 255u);
  EXPECT_EQ(mask_result.vsrc1, result_tmp);
  EXPECT_EQ(cave_words[*base + 6u], 0x0000FFFFu);

  const auto shift_result = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 7u]);
  EXPECT_EQ(shift_result.op, 24u);
  EXPECT_EQ(shift_result.vdst, result_tmp);
  EXPECT_EQ(shift_result.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(shift_result.vsrc1, result_tmp);

  const auto preserve_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 8u]);
  EXPECT_EQ(preserve_dst.op, 27u);
  EXPECT_EQ(preserve_dst.vdst, 9u);
  EXPECT_EQ(preserve_dst.src0, 255u);
  EXPECT_EQ(preserve_dst.vsrc1, 9u);
  EXPECT_EQ(cave_words[*base + 9u], 0x0000FFFFu);

  const auto merge_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 10u]);
  EXPECT_EQ(merge_dst.op, 28u);
  EXPECT_EQ(merge_dst.vdst, 9u);
  EXPECT_EQ(merge_dst.src0, 256u + result_tmp);
  EXPECT_EQ(merge_dst.vsrc1, 9u);
  EXPECT_EQ((cave_words[*base + 11u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250VOrB16HighDstLowersThroughB32MaskMerge) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  gfx1250::Vop3MachineInst inst{};
  inst.vdst = 8;
  inst.op = 867;
  inst.encoding = 0x35;
  inst.src0 = 256 + 10;
  inst.src1 = 256 + 8;
  inst.opsel = 0xA; // src0.l, src1.h, vdst.h.

  std::array<uint32_t, 3> text_words{};
  std::memcpy(text_words.data(), &inst, sizeof(inst));
  text_words[2] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base =
      find_vop2_for_test(cave_words, 28u, std::nullopt, scalar_positive_inline_u32(0), 10u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LT(*base + 9u, cave_words.size());

  const auto src0_lo = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base]);
  ASSERT_EQ(src0_lo.op, 28u);
  const uint8_t tmp0 = static_cast<uint8_t>(src0_lo.vdst);
  const uint8_t tmp1 = static_cast<uint8_t>(tmp0 + 1u);
  const uint8_t result_tmp = static_cast<uint8_t>(tmp0 + 2u);
  EXPECT_EQ(src0_lo.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(src0_lo.vsrc1, 10u);

  const auto src1_hi = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 1u]);
  EXPECT_EQ(src1_hi.op, 25u);
  EXPECT_EQ(src1_hi.vdst, tmp1);
  EXPECT_EQ(src1_hi.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(src1_hi.vsrc1, 8u);

  const auto or_inst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 2u]);
  EXPECT_EQ(or_inst.op, 28u);
  EXPECT_EQ(or_inst.vdst, result_tmp);
  EXPECT_EQ(or_inst.src0, 256u + tmp0);
  EXPECT_EQ(or_inst.vsrc1, tmp1);

  const auto mask_result = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 3u]);
  EXPECT_EQ(mask_result.op, 27u);
  EXPECT_EQ(mask_result.vdst, result_tmp);
  EXPECT_EQ(mask_result.src0, 255u);
  EXPECT_EQ(mask_result.vsrc1, result_tmp);
  EXPECT_EQ(cave_words[*base + 4u], 0x0000FFFFu);

  const auto shift_result = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 5u]);
  EXPECT_EQ(shift_result.op, 24u);
  EXPECT_EQ(shift_result.vdst, result_tmp);
  EXPECT_EQ(shift_result.src0, scalar_positive_inline_u32(16));
  EXPECT_EQ(shift_result.vsrc1, result_tmp);

  const auto preserve_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 6u]);
  EXPECT_EQ(preserve_dst.op, 27u);
  EXPECT_EQ(preserve_dst.vdst, 8u);
  EXPECT_EQ(preserve_dst.src0, 255u);
  EXPECT_EQ(preserve_dst.vsrc1, 8u);
  EXPECT_EQ(cave_words[*base + 7u], 0x0000FFFFu);

  const auto merge_dst = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 8u]);
  EXPECT_EQ(merge_dst.op, 28u);
  EXPECT_EQ(merge_dst.vdst, 8u);
  EXPECT_EQ(merge_dst.src0, 256u + result_tmp);
  EXPECT_EQ(merge_dst.vsrc1, 8u);
  EXPECT_EQ((cave_words[*base + 9u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

[[nodiscard]] constexpr std::array<uint32_t, 3>
make_gfx1250_vopd3(uint8_t opx, uint16_t srcx0, uint8_t vsrcx1, uint8_t vsrcx2, uint8_t vdstx,
                   uint8_t opy, uint16_t srcy0, uint8_t vsrcy1, uint8_t vsrcy2, uint8_t vdsty,
                   uint8_t negx = 0, uint8_t negy = 0) {
  return {
      0xCF000000u | (static_cast<uint32_t>(opx) << 18) | (static_cast<uint32_t>(opy) << 12) |
          (srcx0 & 0x1FFu),
      (srcy0 & 0x1FFu) | ((negx & 0x7u) << 9) | ((negy & 0x7u) << 12) |
          (static_cast<uint32_t>(vsrcx1) << 16) | (static_cast<uint32_t>(vsrcx2) << 24),
      static_cast<uint32_t>(vdstx) | (static_cast<uint32_t>(vsrcy1) << 8) |
          (static_cast<uint32_t>(vsrcy2) << 16) | (static_cast<uint32_t>(vdsty) << 24),
  };
}

TEST(BinaryTranslator, Gfx1250Vopd3MovBitopLowersInPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto vopd = make_gfx1250_vopd3(8, 1, 0, 0, 7, 18, 0, 4, 0x54, 6);
  const std::array<uint32_t, 4> text_words = {vopd[0], vopd[1], vopd[2], kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto mov = std::bit_cast<rdna4::Vop1MachineInst>(patched_text[0]);
  EXPECT_EQ(mov.op, 1u);
  EXPECT_EQ(mov.vdst, 7u);
  EXPECT_EQ(mov.src0, 1u);

  const auto bitop = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[1]);
  EXPECT_EQ(bitop.op, 28u);
  EXPECT_EQ(bitop.vdst, 6u);
  EXPECT_EQ(bitop.src0, 0u);
  EXPECT_EQ(bitop.vsrc1, 4u);
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250Vopd3HazardReadsBeforeClobber) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto vopd = make_gfx1250_vopd3(8, 1, 0, 0, 4, 18, 0, 4, 0x54, 6);
  const std::array<uint32_t, 4> text_words = {vopd[0], vopd[1], vopd[2], kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  const auto bitop = std::bit_cast<rdna4::Vop2MachineInst>(patched_text[0]);
  EXPECT_EQ(bitop.op, 28u);
  EXPECT_EQ(bitop.vdst, 6u);
  EXPECT_EQ(bitop.src0, 0u);
  EXPECT_EQ(bitop.vsrc1, 4u);

  const auto mov = std::bit_cast<rdna4::Vop1MachineInst>(patched_text[1]);
  EXPECT_EQ(mov.op, 1u);
  EXPECT_EQ(mov.vdst, 4u);
  EXPECT_EQ(mov.src0, 1u);
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250Vopd3DualCndmaskLowersThroughCave) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto vopd = make_gfx1250_vopd3(9, scalar_positive_inline_u32(0), 8, 2, 0, 9,
                                       scalar_positive_inline_u32(0), 9, 3, 1);
  const std::array<uint32_t, 4> text_words = {vopd[0], vopd[1], vopd[2], kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base =
      find_vop3_for_test(cave_words, 257u, 0u, scalar_positive_inline_u32(0), 256u + 8u, 2u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 5u, cave_words.size());

  rdna4::Vop3MachineInst cndmask_x{};
  std::memcpy(&cndmask_x, cave_words.data() + *base, sizeof(cndmask_x));
  EXPECT_EQ(cndmask_x.encoding, 0x35u);
  EXPECT_EQ(cndmask_x.op, 257u);
  EXPECT_EQ(cndmask_x.vdst, 0u);
  EXPECT_EQ(cndmask_x.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(cndmask_x.src1, 256u + 8u);
  EXPECT_EQ(cndmask_x.src2, 2u);

  rdna4::Vop3MachineInst cndmask_y{};
  std::memcpy(&cndmask_y, cave_words.data() + *base + 2, sizeof(cndmask_y));
  EXPECT_EQ(cndmask_y.encoding, 0x35u);
  EXPECT_EQ(cndmask_y.op, 257u);
  EXPECT_EQ(cndmask_y.vdst, 1u);
  EXPECT_EQ(cndmask_y.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(cndmask_y.src1, 256u + 9u);
  EXPECT_EQ(cndmask_y.src2, 3u);
  EXPECT_EQ((cave_words[*base + 4u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250Vopd3DualFmaLowersThroughCave) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto vopd = make_gfx1250_vopd3(19, 256 + 1, 2, 3, 4, 19, 256 + 5, 6, 7, 8, 0x5, 0x6);
  const std::array<uint32_t, 4> text_words = {vopd[0], vopd[1], vopd[2], kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base = find_vop3_for_test(cave_words, 531u, 4u, 256u + 1u, 256u + 2u, 256u + 3u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 5u, cave_words.size());

  rdna4::Vop3MachineInst fma_x{};
  std::memcpy(&fma_x, cave_words.data() + *base, sizeof(fma_x));
  EXPECT_EQ(fma_x.encoding, 0x35u);
  EXPECT_EQ(fma_x.op, 531u);
  EXPECT_EQ(fma_x.vdst, 4u);
  EXPECT_EQ(fma_x.src0, 256u + 1u);
  EXPECT_EQ(fma_x.src1, 256u + 2u);
  EXPECT_EQ(fma_x.src2, 256u + 3u);
  EXPECT_EQ(fma_x.neg, 0x5u);

  rdna4::Vop3MachineInst fma_y{};
  std::memcpy(&fma_y, cave_words.data() + *base + 2, sizeof(fma_y));
  EXPECT_EQ(fma_y.encoding, 0x35u);
  EXPECT_EQ(fma_y.op, 531u);
  EXPECT_EQ(fma_y.vdst, 8u);
  EXPECT_EQ(fma_y.src0, 256u + 5u);
  EXPECT_EQ(fma_y.src1, 256u + 6u);
  EXPECT_EQ(fma_y.src2, 256u + 7u);
  EXPECT_EQ(fma_y.neg, 0x6u);
  EXPECT_EQ((cave_words[*base + 4u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250Vopd3DualFmaWithoutNegLowersToMulAddThroughCave) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto vopd = make_gfx1250_vopd3(19, 256 + 1, 2, 3, 4, 19, 256 + 5, 6, 7, 8);
  const std::array<uint32_t, 4> text_words = {vopd[0], vopd[1], vopd[2], kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ((patched_text[0] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);

  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);
  const auto base = find_vop2_for_test(cave_words, 8u, 4u, 256u + 1u, 2u);
  ASSERT_TRUE(base.has_value());
  ASSERT_LE(*base + 5u, cave_words.size());

  const auto mul_x = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base]);
  EXPECT_EQ(mul_x.op, 8u);
  EXPECT_EQ(mul_x.vdst, 4u);
  EXPECT_EQ(mul_x.src0, 256u + 1u);
  EXPECT_EQ(mul_x.vsrc1, 2u);

  const auto add_x = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 1u]);
  EXPECT_EQ(add_x.op, 3u);
  EXPECT_EQ(add_x.vdst, 4u);
  EXPECT_EQ(add_x.src0, 256u + 3u);
  EXPECT_EQ(add_x.vsrc1, 4u);

  const auto mul_y = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 2u]);
  EXPECT_EQ(mul_y.op, 8u);
  EXPECT_EQ(mul_y.vdst, 8u);
  EXPECT_EQ(mul_y.src0, 256u + 5u);
  EXPECT_EQ(mul_y.vsrc1, 6u);

  const auto add_y = std::bit_cast<rdna4::Vop2MachineInst>(cave_words[*base + 3u]);
  EXPECT_EQ(add_y.op, 3u);
  EXPECT_EQ(add_y.vdst, 8u);
  EXPECT_EQ(add_y.src0, 256u + 7u);
  EXPECT_EQ(add_y.vsrc1, 8u);
  EXPECT_EQ((cave_words[*base + 4u] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250UnhandledVopd3DoesNotUseEncodingFallback) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto vopd = make_gfx1250_vopd3(33, 0, 0, 0, 1, 0, 0, 0, 0, 2);
  const std::array<uint32_t, 4> text_words = {vopd[0], vopd[1], vopd[2], kGfx1250SEndpgm};

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.warnings.size(), 1u);
  EXPECT_NE(result.warnings[0].find("EXPAND not yet implemented for v_dual_add_f64"),
            std::string::npos);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[3], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250VMovB64UsesExpandedCopyWhenEntryPrologueCaveIsTooFar) {
  constexpr uint32_t kGfx1250VMovB64V10S2 = 0x7E143A02u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  std::vector<uint32_t> text_words(0x21000 / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250VMovB64V10S2;
  text_words[1] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_GT(text->size(), text_words.size() * sizeof(uint32_t));

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(result.elf_bytes.data());
  EXPECT_EQ(ehdr->e_flags & EF_AMDGPU_MACH, EF_AMDGPU_MACH_AMDGCN_GFX1201);

  uint32_t original_first_word = 0;
  std::memcpy(&original_first_word, text->data(), sizeof(original_first_word));
  EXPECT_EQ(original_first_word, kGfx1250VMovB64V10S2);

  const auto cave_words =
      local_expanded_text_words_for_test(translated, text_words.size() * sizeof(uint32_t));
  ASSERT_FALSE(cave_words.empty());
  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(cave_words);
  ASSERT_GE(copied_words.size(), 2u);

  const auto mov_lo = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[0]);
  EXPECT_EQ(mov_lo.op, 1u);
  EXPECT_EQ(mov_lo.vdst, 10u);
  EXPECT_EQ(mov_lo.src0, 2u);

  const auto mov_hi = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[1]);
  EXPECT_EQ(mov_hi.op, 1u);
  EXPECT_EQ(mov_hi.vdst, 11u);
  EXPECT_EQ(mov_hi.src0, 3u);
}

TEST(BinaryTranslator, Gfx1250VMovB64UsesExpandedCopyWhenDeadTextCaveCannotHostPrologue) {
  constexpr uint32_t kGfx1250VMovB64V10S2 = 0x7E143A02u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint64_t kTextSize = 0x21000;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250VMovB64V10S2);
  text_words[0] = kGfx1250VMovB64V10S2;
  text_words[1] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_GT(text->size(), text_words.size() * sizeof(uint32_t));

  uint32_t original_first_word = 0;
  std::memcpy(&original_first_word, text->data(), sizeof(original_first_word));
  EXPECT_EQ(original_first_word, kGfx1250VMovB64V10S2);

  const auto cave_words =
      local_expanded_text_words_for_test(translated, text_words.size() * sizeof(uint32_t));
  ASSERT_FALSE(cave_words.empty());
  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(cave_words);
  ASSERT_GE(copied_words.size(), 2u);

  const auto mov_lo = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[0]);
  EXPECT_EQ(mov_lo.op, 1u);
  EXPECT_EQ(mov_lo.vdst, 10u);
  EXPECT_EQ(mov_lo.src0, 2u);

  const auto mov_hi = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[1]);
  EXPECT_EQ(mov_hi.op, 1u);
  EXPECT_EQ(mov_hi.vdst, 11u);
  EXPECT_EQ(mov_hi.src0, 3u);
}

TEST(BinaryTranslator, Gfx1250VMovB64UsesExpandedCopyWhenBackwardCaveCannotHostPrologue) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint32_t kGfx1250VMovB64V10S2 = 0x7E143A02u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kSourceOffset = 0x22000;
  constexpr uint64_t kTextSize = 0x44000;
  static_assert(kSourceOffset % sizeof(uint32_t) == 0);

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SEndpgm);
  std::fill(text_words.begin(), text_words.begin() + kSourceOffset / sizeof(uint32_t),
            kGfx1250SNop0);
  text_words[kSourceOffset / sizeof(uint32_t)] = kGfx1250VMovB64V10S2;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *kd = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);
  kd->kernel_code_entry_byte_offset = static_cast<int64_t>(shdrs[1].sh_addr + kSourceOffset) -
                                      static_cast<int64_t>(shdrs[2].sh_addr);

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_GT(text->size(), text_words.size() * sizeof(uint32_t));

  uint32_t original_entry_word = 0;
  std::memcpy(&original_entry_word, text->data() + kSourceOffset, sizeof(original_entry_word));
  EXPECT_EQ(original_entry_word, kGfx1250VMovB64V10S2);

  const auto cave_words =
      local_expanded_text_words_for_test(translated, text_words.size() * sizeof(uint32_t));
  ASSERT_FALSE(cave_words.empty());
  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(cave_words);
  ASSERT_GE(copied_words.size(), 2u);

  const auto mov_lo = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[0]);
  EXPECT_EQ(mov_lo.op, 1u);
  EXPECT_EQ(mov_lo.vdst, 10u);
  EXPECT_EQ(mov_lo.src0, 2u);

  const auto mov_hi = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[1]);
  EXPECT_EQ(mov_hi.op, 1u);
  EXPECT_EQ(mov_hi.vdst, 11u);
  EXPECT_EQ(mov_hi.src0, 3u);
}

TEST(BinaryTranslator, Gfx1250LargeReachableTextUsesExpandedCopyWhenCaveTooFar) {
  constexpr uint32_t kGfx1250VMovB64V10S2 = 0x7E143A02u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250VMovB64V10S2;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_GT(text->size(), text_words.size() * sizeof(uint32_t));

  uint32_t original_first_word = 0;
  std::memcpy(&original_first_word, text->data(), sizeof(original_first_word));
  EXPECT_EQ(original_first_word, kGfx1250VMovB64V10S2);

  const auto cave_words =
      local_expanded_text_words_for_test(translated, text_words.size() * sizeof(uint32_t));
  ASSERT_GE(cave_words.size(), 2u);
  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(cave_words);
  ASSERT_GE(copied_words.size(), 2u);
  const auto mov_lo = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[0]);
  EXPECT_EQ(mov_lo.op, 1u);
  EXPECT_EQ(mov_lo.vdst, 10u);
  EXPECT_EQ(mov_lo.src0, 2u);

  const auto mov_hi = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[1]);
  EXPECT_EQ(mov_hi.op, 1u);
  EXPECT_EQ(mov_hi.vdst, 11u);
  EXPECT_EQ(mov_hi.src0, 3u);

  KernelDescriptorTranslator translated_parser(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto translated_infos = translated_parser.translate_image(
      result.elf_bytes, text->sectionOffset(), text->size(), KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translated_infos.size(), 1u);
  EXPECT_EQ(translated_infos[0].entry_text_offset, text_words.size() * sizeof(uint32_t));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyRelocatesMaterializedPcRelativeHelperAddress) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint32_t kGfx1250SSetVgprMsb0 = 0xBF860000u;
  constexpr uint64_t kTextSize = 0x21000;
  constexpr uint16_t kPcSgpr = 12;
  constexpr uint16_t kTmpSgpr = 8;
  constexpr uint32_t kGetpcWord = 0;
  constexpr uint32_t kBranchWord = 5;
  constexpr uint32_t kHelperWord = 6;
  constexpr uint32_t kSkipWord = 8;
  constexpr int32_t kOriginalLiteral =
      static_cast<int32_t>(kHelperWord * sizeof(uint32_t) - 2 * sizeof(uint32_t));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[kGetpcWord] = pack_sop1(71, kPcSgpr, 0);
  text_words[1] = pack_sop2(2, kTmpSgpr, 255, scalar_positive_inline_u32(4));
  text_words[2] = static_cast<uint32_t>(kOriginalLiteral);
  text_words[3] = pack_sop2(0, kPcSgpr, kPcSgpr, kTmpSgpr);
  text_words[4] = pack_sop2(4, kPcSgpr + 1u, kPcSgpr + 1u, scalar_positive_inline_u32(0));
  text_words[kBranchWord] = build_s_branch(static_cast<int16_t>(kSkipWord - kBranchWord - 1u),
                                           ROCJITSU_CODE_ARCH_GFX1250);
  text_words[kHelperWord] = kGfx1250SSetVgprMsb0;
  text_words[kHelperWord + 1u] = kGfx1250SEndpgm;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto cave_words = local_expanded_text_words_for_test(translated, kTextSize);
  ASSERT_FALSE(cave_words.empty());

  std::optional<size_t> getpc_word;
  std::optional<size_t> add_tmp_word;
  for (size_t i = 0; i + 2 < cave_words.size(); ++i) {
    if (cave_words[i] != pack_sop2(2, kTmpSgpr, 255, scalar_positive_inline_u32(4)))
      continue;

    for (size_t scan = i; scan > 0 && i - scan < 4; --scan) {
      if (cave_words[scan - 1] == pack_sop1(71, kPcSgpr, 0)) {
        getpc_word = scan - 1;
        add_tmp_word = i;
        break;
      }
    }
    if (getpc_word)
      break;
  }
  ASSERT_TRUE(getpc_word.has_value());
  ASSERT_TRUE(add_tmp_word.has_value());

  const int64_t helper_pc = static_cast<int64_t>(kHelperWord * sizeof(uint32_t));
  const int64_t getpc_pc = static_cast<int64_t>(kTextSize + *getpc_word * sizeof(uint32_t));
  const int64_t expected_literal = helper_pc - getpc_pc - 2 * sizeof(uint32_t);
  ASSERT_GE(expected_literal, std::numeric_limits<int32_t>::min());
  ASSERT_LE(expected_literal, std::numeric_limits<int32_t>::max());
  EXPECT_EQ(static_cast<int32_t>(cave_words[*add_tmp_word + 1]),
            static_cast<int32_t>(expected_literal));
  EXPECT_NE(cave_words[*add_tmp_word + 1], static_cast<uint32_t>(kOriginalLiteral));

  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  std::vector<uint32_t> patched_text(text->size() / sizeof(uint32_t));
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[kHelperWord], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyRelocatesDirectPcRelativeLiteral64Address) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint32_t kGfx1250SAddNcU64 = 83u;
  constexpr uint64_t kTextSize = 0x21000;
  constexpr uint16_t kPcSgpr = 12;
  constexpr uint32_t kGetpcWord = 0;
  constexpr uint32_t kBranchWord = 4;
  constexpr uint32_t kAddressWord = 1024;
  constexpr uint32_t kEndWord = 8;
  constexpr int64_t kOriginalLiteral =
      static_cast<int64_t>(kAddressWord * sizeof(uint32_t)) - sizeof(uint32_t);

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[kGetpcWord] = pack_sop1(71, kPcSgpr, 0);
  text_words[1] = pack_sop2(kGfx1250SAddNcU64, kPcSgpr, kPcSgpr, 254);
  text_words[2] = static_cast<uint32_t>(kOriginalLiteral);
  text_words[3] = 0;
  text_words[kBranchWord] =
      build_s_branch(static_cast<int16_t>(kEndWord - kBranchWord - 1u), ROCJITSU_CODE_ARCH_GFX1250);
  text_words[kAddressWord] = kGfx1250SNop0;
  text_words[kEndWord] = kGfx1250SEndpgm;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto cave_words = local_expanded_text_words_for_test(translated, kTextSize);
  ASSERT_FALSE(cave_words.empty());

  std::optional<size_t> getpc_word;
  std::optional<size_t> literal_word;
  for (size_t i = 0; i + 4 < cave_words.size(); ++i) {
    if (cave_words[i] != pack_sop1(71, kPcSgpr, 0))
      continue;
    for (size_t scan = i + 1; scan < std::min(cave_words.size(), i + 6); ++scan) {
      if (cave_words[scan] == pack_sop2(0, kPcSgpr, kPcSgpr, 255)) {
        getpc_word = i;
        literal_word = scan + 1;
        break;
      }
    }
    if (literal_word)
      break;
  }
  ASSERT_TRUE(getpc_word.has_value());
  ASSERT_TRUE(literal_word.has_value());

  const int64_t address_pc = static_cast<int64_t>(kAddressWord * sizeof(uint32_t));
  const int64_t getpc_pc = static_cast<int64_t>(kTextSize + *getpc_word * sizeof(uint32_t));
  const int64_t expected_literal = address_pc - getpc_pc - sizeof(uint32_t);
  ASSERT_GE(expected_literal, std::numeric_limits<int32_t>::min());
  ASSERT_LE(expected_literal, std::numeric_limits<int32_t>::max());
  EXPECT_EQ(static_cast<int32_t>(cave_words[*literal_word]),
            static_cast<int32_t>(expected_literal));
  EXPECT_NE(cave_words[*literal_word], static_cast<uint32_t>(kOriginalLiteral));
}

TEST(BinaryTranslator, Gfx1250LongCaveReturnPreservesSccBeforeConditionalBranch) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250SCmpEqU32S6_3 =
      pack_sopc_for_test(6, 6, scalar_positive_inline_u32(3));
  constexpr uint32_t kGfx1250SCbranchScc1 = 0xBFA20000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr size_t kCompareCount = 0x5000;

  std::vector<uint32_t> text_words(kCompareCount + 2, kGfx1250SCmpEqU32S6_3);
  text_words[kCompareCount] = kGfx1250SCbranchScc1;
  text_words[kCompareCount + 1] = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  const auto cave_words = section_words_for_test(*translations);

  const uint32_t scc_one = scalar_positive_inline_u32(1);
  const uint32_t scc_zero = scalar_positive_inline_u32(0);
  std::optional<size_t> trailer_index;
  std::optional<uint32_t> trailer_scc_sgpr;
  std::optional<uint32_t> trailer_sgpr_pair;
  for (size_t i = 2; i + 7 < cave_words.size() && !trailer_index; ++i) {
    for (uint32_t tmp = 0; tmp <= 105; ++tmp) {
      if (cave_words[i] != pack_sop2(48, tmp, scc_one, scc_zero))
        continue;

      const uint32_t getpc = cave_words[i + 1];
      const uint32_t pair = (getpc >> 16) & 0x7Fu;
      if (pair >= 127 || getpc != pack_sop1(71, pair, 0))
        continue;
      if (cave_words[i + 2] != pack_sop2(0, pair, pair, 255))
        continue;
      if (cave_words[i + 4] != pack_sop2(4, pair + 1u, pair + 1u, 255))
        continue;
      if (cave_words[i + 6] != pack_sopc_for_test(7, tmp, scc_zero))
        continue;
      if (cave_words[i + 7] != pack_sop1(72, 0, pair))
        continue;

      trailer_index = i;
      trailer_scc_sgpr = tmp;
      trailer_sgpr_pair = pair;
      break;
    }
  }

  ASSERT_TRUE(trailer_index.has_value())
      << "far cave returns to original s_cbranch_scc* must preserve SCC across s_setpc";
  ASSERT_TRUE(trailer_scc_sgpr.has_value());
  ASSERT_TRUE(trailer_sgpr_pair.has_value());
  EXPECT_EQ(cave_words[*trailer_index - 2u], kGfx1250SCmpEqU32S6_3);
  EXPECT_EQ(cave_words[*trailer_index - 1u],
            build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));

  const Section *rodata = find_section(translated, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(KD));
  const auto *desc = reinterpret_cast<const KD *>(rodata->data());
  const uint32_t granulated = AMDHSA_BITS_GET(
      desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  const uint32_t addressable_sgprs = (granulated + 1u) * 8u;
  const uint32_t required_sgprs = std::max(*trailer_scc_sgpr + 1u, *trailer_sgpr_pair + 2u);
  EXPECT_GE(addressable_sgprs, required_sgprs)
      << "long cave returns must grow the kernel descriptor for their scratch SGPRs";
}

TEST(BinaryTranslator, Gfx1250InPlaceNopsVgprMsbModeWhenNoShadowPlan) {
  constexpr uint32_t kGfx1250SSetVgprMsbDstBank1 = 0xBF860040u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 2> text_words = {
      kGfx1250SSetVgprMsbDstBank1,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[1], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250StaticSetpcTargetIsTranslatedInPlace) {
  constexpr uint32_t kGfx1250SSetVgprMsb0 = 0xBF860000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint16_t kPcSgpr = 24;
  constexpr uint16_t kTmpSgpr = 26;
  constexpr uint32_t kTargetWord = 8;
  constexpr uint32_t kTargetOffset = kTargetWord * sizeof(uint32_t);
  constexpr uint32_t kGetpcOffset = 0;
  constexpr uint32_t kLiteral = kTargetOffset - kGetpcOffset - 8;

  const std::array<uint32_t, 10> text_words = {
      pack_sop1(71, kPcSgpr, 0),
      pack_sop2(2, kTmpSgpr, 255, scalar_positive_inline_u32(4)),
      kLiteral,
      pack_sop2(0, kPcSgpr, kPcSgpr, kTmpSgpr),
      pack_sop2(4, kPcSgpr + 1u, kPcSgpr + 1u, scalar_positive_inline_u32(0)),
      pack_sop1(72, 0, kPcSgpr),
      kGfx1250SEndpgm,
      kGfx1250SNop0,
      kGfx1250SSetVgprMsb0,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  std::array<uint32_t, text_words.size()> patched_text{};
  std::memcpy(patched_text.data(), text->data(), text->size());
  EXPECT_EQ(patched_text[kTargetWord], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_text[kTargetWord + 1u], kGfx1250SEndpgm);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsHighBankDstVgpr) {
  constexpr uint32_t kGfx1250SSetVgprMsbDstBank1 = 0xBF860040u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop1MachineInst mov{};
  mov.encoding = 0x3F;
  mov.op = 1;
  mov.vdst = 0;
  mov.src0 = scalar_positive_inline_u32(1);

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbDstBank1;
  text_words[1] = std::bit_cast<uint32_t>(mov);
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_GT(text->size(), text_words.size() * sizeof(uint32_t));

  uint32_t original_mov_word = 0;
  std::memcpy(&original_mov_word, text->data() + sizeof(uint32_t), sizeof(original_mov_word));
  EXPECT_EQ(original_mov_word, std::bit_cast<uint32_t>(mov));

  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 2 * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 2u);
  EXPECT_EQ(copied_words[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  const auto shadow_mov = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[1]);
  EXPECT_EQ(shadow_mov.op, 1u);
  EXPECT_EQ(shadow_mov.vdst, 64u);
  EXPECT_EQ(shadow_mov.src0, scalar_positive_inline_u32(1));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyIgnoresHighBankVNop) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc1Bank1 = 0xBF860004u;
  constexpr uint32_t kGfx1250VNopE32 = 0x7E000000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc1Bank1;
  text_words[1] = kGfx1250VNopE32;
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyCarriesHighBankModeAcrossBasicBlocks) {
  constexpr uint32_t kGfx1250SSetVgprMsbDstBank1 = 0xBF860040u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop1MachineInst mov{};
  mov.encoding = 0x3F;
  mov.op = 1;
  mov.vdst = 0;
  mov.src0 = scalar_positive_inline_u32(1);

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbDstBank1;
  text_words[1] = build_s_branch(0, ROCJITSU_CODE_ARCH_GFX1250);
  text_words[2] = std::bit_cast<uint32_t>(mov);
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 3 * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 3u);
  EXPECT_EQ(copied_words[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ((copied_words[1] >> 16) & 0x7Fu, sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4));

  const auto shadow_mov = std::bit_cast<rdna4::Vop1MachineInst>(copied_words[2]);
  EXPECT_EQ(shadow_mov.op, 1u);
  EXPECT_EQ(shadow_mov.vdst, 64u);
  EXPECT_EQ(shadow_mov.src0, scalar_positive_inline_u32(1));
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyAllowsJoinModeConflictKilledBeforeUse) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank2 = 0xBF860002u;
  constexpr uint32_t kGfx1250SSetVgprMsb0 = 0xBF860000u;
  constexpr uint32_t kGfx1250SCbranchScc1 = 0xBFA20000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::VdsMachineInst load{};
  load.encoding = 0x36;
  load.op = 0xFF;
  load.addr = 12;
  load.vdst = 32;

  std::array<uint32_t, 2> load_words{};
  std::memcpy(load_words.data(), &load, sizeof(load));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Bank1;
  text_words[1] = kGfx1250SCbranchScc1 | 1u; // branch to word 3
  text_words[2] = kGfx1250SSetVgprMsb0;
  text_words[3] = kGfx1250SSetVgprMsbSrc0Bank2;
  text_words[4] = load_words[0];
  text_words[5] = load_words[1];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  constexpr uint32_t kScratchLoadB32OffOff = 0xED05007Cu;
  constexpr uint32_t kSelector2Logical12ScratchOffset =
      (kGfx1250PrivateBorrowedVgprCountForTest + 256u + 12u) * sizeof(uint32_t) << 8;

  bool found_private_reload = false;
  bool found_shadow_load = false;
  for (size_t i = 0; i + 2u < copied_words.size(); ++i) {
    if (copied_words[i] == kScratchLoadB32OffOff && (copied_words[i + 1] & 0xFFu) == 76u &&
        copied_words[i + 2] == kSelector2Logical12ScratchOffset) {
      found_private_reload = true;
    }

    rdna4::VdsMachineInst shadow_load{};
    std::memcpy(&shadow_load, copied_words.data() + i, sizeof(shadow_load));
    if (shadow_load.encoding == 0x36u && shadow_load.op == 0xFFu && shadow_load.addr == 76u &&
        shadow_load.vdst == 32u) {
      found_shadow_load = true;
    }
  }

  EXPECT_TRUE(found_private_reload)
      << "join-local s_set_vgpr_msb should make conflicting incoming modes irrelevant";
  EXPECT_TRUE(found_shadow_load);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyCarriesConflictingModesThroughScalarControl) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank2 = 0xBF860002u;
  constexpr uint32_t kGfx1250SSetVgprMsb0 = 0xBF860000u;
  constexpr uint32_t kGfx1250SCbranchScc1 = 0xBFA20000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::VdsMachineInst load{};
  load.encoding = 0x36;
  load.op = 0xFF;
  load.addr = 12;
  load.vdst = 32;

  std::array<uint32_t, 2> load_words{};
  std::memcpy(load_words.data(), &load, sizeof(load));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Bank1;
  text_words[1] = kGfx1250SCbranchScc1 | 1u; // branch to word 3
  text_words[2] = kGfx1250SSetVgprMsb0;
  text_words[3] = kGfx1250SCbranchScc1 | 1u; // scalar-only join, branch to word 5
  text_words[4] = kGfx1250SNop0;
  text_words[5] = kGfx1250SSetVgprMsbSrc0Bank2;
  text_words[6] = load_words[0];
  text_words[7] = load_words[1];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found_shadow_load = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::VdsMachineInst shadow_load{};
    std::memcpy(&shadow_load, copied_words.data() + i, sizeof(shadow_load));
    if (shadow_load.encoding == 0x36u && shadow_load.op == 0xFFu && shadow_load.addr == 76u &&
        shadow_load.vdst == 32u) {
      found_shadow_load = true;
      break;
    }
  }

  EXPECT_TRUE(found_shadow_load)
      << "scalar-only control blocks may carry multiple modes until a later s_set_vgpr_msb";
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsHighBankDsLoadAddress) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::VdsMachineInst load{};
  load.encoding = 0x36;
  load.op = 0xFF;
  load.addr = 12;
  load.vdst = 32;

  std::array<uint32_t, 2> load_words{};
  std::memcpy(load_words.data(), &load, sizeof(load));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Bank1;
  text_words[1] = load_words[0];
  text_words[2] = load_words[1];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 3 * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 3u);
  EXPECT_EQ(copied_words[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  std::optional<rdna4::VdsMachineInst> shadow_load;
  for (size_t i = 1; i + 1u < copied_words.size(); ++i) {
    rdna4::VdsMachineInst candidate{};
    std::memcpy(&candidate, copied_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x36u && candidate.op == 0xFFu) {
      shadow_load = candidate;
      break;
    }
  }
  ASSERT_TRUE(shadow_load.has_value());
  EXPECT_EQ(shadow_load->addr, 76u);
  EXPECT_EQ(shadow_load->vdst, 32u);
}

TEST(BinaryTranslator, Gfx1250TensorDmaTensileU8DataFastPathBranchesOnTileDim0) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint32_t, 4> text_words = {
      0xD0710001u, // tensor_load_to_lds s[16:19], s[20:27]
      0x7C000000u,
      0x7C7C1410u,
      kGfx1250SEndpgm,
  };

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = find_section(translated, ".rj_translations");
  ASSERT_NE(translations, nullptr);

  const auto cave_words = section_words_for_test(*translations);
  const auto contains_word = [&cave_words](uint32_t word) {
    return std::ranges::find(cave_words, word) != cave_words.end();
  };
  EXPECT_TRUE(contains_word(0x07500000u)) << "padded u8 Tensor DMA descriptor flags";
  EXPECT_TRUE(contains_word(0x00800000u)) << "128-byte tile dim0 data path";
  EXPECT_TRUE(contains_word(0x01000000u)) << "256-byte tile dim0 data path";
  EXPECT_FALSE(
      find_word_sequence_for_test(cave_words, std::span<const uint32_t, 3>(text_words.data(), 3))
          .has_value())
      << "gfx1250 Tensor DMA instruction must be semantically expanded";
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsHighBankBufferLoadAddress) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::VbufferMachineInst load{};
  load.encoding = 0x31;
  load.op = 23; // buffer_load_b128
  load.vdata = 20;
  load.rsrc = 0;
  load.soffset = 0x7F;
  load.offen = 1;
  load.vaddr = 12;

  std::array<uint32_t, 3> load_words{};
  std::memcpy(load_words.data(), &load, sizeof(load));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Bank1;
  text_words[1] = load_words[0];
  text_words[2] = load_words[1];
  text_words[3] = load_words[2];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  std::optional<rdna4::VbufferMachineInst> shadow_load;
  for (size_t i = 0; i + 2u < copied_words.size(); ++i) {
    rdna4::VbufferMachineInst candidate{};
    std::memcpy(&candidate, copied_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x31u && candidate.op == 23u && candidate.offen == 1u &&
        candidate.idxen == 0u) {
      shadow_load = candidate;
      break;
    }
  }

  ASSERT_TRUE(shadow_load.has_value());
  EXPECT_EQ(shadow_load->vaddr, 76u);
  EXPECT_EQ(shadow_load->vdata, 20u);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsHighBankBufferD16LoadData) {
  constexpr uint32_t kGfx1250SSetVgprMsbDstBank1 = 0xBF860040u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::VbufferMachineInst load{};
  load.encoding = 0x31;
  load.op = 35; // buffer_load_d16_hi_b16
  load.vdata = 20;
  load.rsrc = 0;
  load.soffset = 0x7F;
  load.offen = 1;
  load.vaddr = 12;

  std::array<uint32_t, 3> load_words{};
  std::memcpy(load_words.data(), &load, sizeof(load));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbDstBank1;
  text_words[1] = load_words[0];
  text_words[2] = load_words[1];
  text_words[3] = load_words[2];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  std::optional<rdna4::VbufferMachineInst> shadow_load;
  for (size_t i = 0; i + 2u < copied_words.size(); ++i) {
    rdna4::VbufferMachineInst candidate{};
    std::memcpy(&candidate, copied_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x31u && candidate.op == 35u && candidate.offen == 1u &&
        candidate.idxen == 0u) {
      shadow_load = candidate;
      break;
    }
  }

  ASSERT_TRUE(shadow_load.has_value());
  EXPECT_EQ(shadow_load->vaddr, 12u);
  EXPECT_EQ(shadow_load->vdata, 84u);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsHighBankBufferStoreData) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc1Bank1 = 0xBF860004u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::VbufferMachineInst store{};
  store.encoding = 0x31;
  store.op = 29; // buffer_store_b128
  store.vdata = 20;
  store.rsrc = 0;
  store.soffset = 0x7F;
  store.offen = 1;
  store.vaddr = 12;

  std::array<uint32_t, 3> store_words{};
  std::memcpy(store_words.data(), &store, sizeof(store));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc1Bank1;
  text_words[1] = store_words[0];
  text_words[2] = store_words[1];
  text_words[3] = store_words[2];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  std::optional<rdna4::VbufferMachineInst> shadow_store;
  for (size_t i = 0; i + 2u < copied_words.size(); ++i) {
    rdna4::VbufferMachineInst candidate{};
    std::memcpy(&candidate, copied_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x31u && candidate.op == 29u && candidate.offen == 1u &&
        candidate.idxen == 0u) {
      shadow_store = candidate;
      break;
    }
  }

  ASSERT_TRUE(shadow_store.has_value());
  EXPECT_EQ(shadow_store->vaddr, 12u);
  EXPECT_EQ(shadow_store->vdata, 84u);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyPreservesLiveLowVgprAcrossHighBankDsAddress) {
  constexpr uint32_t kGfx1250SSetVgprMsb0 = 0xBF860000u;
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  const auto make_mov_b32 = [](uint8_t vdst, uint16_t src0) {
    gfx1250::Vop1MachineInst mov{};
    mov.encoding = 0x3F;
    mov.op = 1;
    mov.vdst = vdst;
    mov.src0 = src0;
    return std::bit_cast<uint32_t>(mov);
  };

  gfx1250::VdsMachineInst load{};
  load.encoding = 0x36;
  load.op = 0xFF;
  load.addr = 13;
  load.vdst = 32;

  std::array<uint32_t, 2> load_words{};
  std::memcpy(load_words.data(), &load, sizeof(load));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  size_t index = 0;
  text_words[index++] = make_mov_b32(77, scalar_positive_inline_u32(1));
  text_words[index++] = kGfx1250SSetVgprMsbSrc0Bank1;
  text_words[index++] = load_words[0];
  text_words[index++] = load_words[1];
  text_words[index++] = kGfx1250SSetVgprMsb0;
  for (uint16_t reg = 0; reg < 256; ++reg)
    text_words[index++] =
        make_mov_b32(static_cast<uint8_t>(reg), static_cast<uint16_t>(256u + reg));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  constexpr uint32_t kScratchStoreB32OffOff = 0xED06807Cu;
  constexpr uint32_t kScratchLoadB32OffOff = 0xED05007Cu;
  constexpr uint32_t kHighPrivateBase = kGfx1250PrivateBorrowedVgprCountForTest * sizeof(uint32_t);
  constexpr uint32_t kHighLogical13Offset = (kHighPrivateBase + 13u * sizeof(uint32_t)) << 8;
  constexpr uint32_t kLowSaveSlot0Offset = (kHighPrivateBase + 14u * sizeof(uint32_t)) << 8;

  std::optional<size_t> low_save_store;
  std::optional<size_t> high_addr_reload;
  std::optional<size_t> shadow_ds_load;
  std::optional<size_t> low_restore_load;
  for (size_t i = 0; i + 2u < cave_words.size(); ++i) {
    if (cave_words[i] == kScratchStoreB32OffOff && (cave_words[i + 1] >> 23) == 77u &&
        cave_words[i + 2] == kLowSaveSlot0Offset)
      low_save_store = i;
    if (cave_words[i] == kScratchLoadB32OffOff && (cave_words[i + 1] & 0xFFu) == 77u &&
        cave_words[i + 2] == kHighLogical13Offset)
      high_addr_reload = i;
    if (cave_words[i] == kScratchLoadB32OffOff && (cave_words[i + 1] & 0xFFu) == 77u &&
        cave_words[i + 2] == kLowSaveSlot0Offset)
      low_restore_load = i;

    rdna4::VdsMachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x36u && candidate.op == 0xFFu && candidate.addr == 77u &&
        candidate.vdst == 32u)
      shadow_ds_load = i;
  }

  ASSERT_TRUE(low_save_store.has_value());
  ASSERT_TRUE(high_addr_reload.has_value());
  ASSERT_TRUE(shadow_ds_load.has_value());
  ASSERT_TRUE(low_restore_load.has_value());
  EXPECT_LT(*low_save_store, *high_addr_reload);
  EXPECT_LT(*high_addr_reload, *shadow_ds_load);
  EXPECT_LT(*shadow_ds_load, *low_restore_load);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyPreservesOffsetDsLoadB128SecondWord) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::VdsMachineInst load0{};
  load0.encoding = 0x36;
  load0.op = 0xFF;
  load0.addr = 159;
  load0.vdst = 208;

  gfx1250::VdsMachineInst load1{};
  load1.encoding = 0x36;
  load1.op = 0xFF;
  load1.offset0 = 16;
  load1.addr = 159;
  load1.vdst = 212;

  std::array<uint32_t, 2> load0_words{};
  std::array<uint32_t, 2> load1_words{};
  std::memcpy(load0_words.data(), &load0, sizeof(load0));
  std::memcpy(load1_words.data(), &load1, sizeof(load1));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> decoded_load1(decoder->decode(load1_words.data()));
  ASSERT_NE(decoded_load1, nullptr);
  EXPECT_EQ(decoded_load1->mnemonic(), "ds_load_b128");
  EXPECT_EQ(decoded_load1->size(), sizeof(gfx1250::VdsMachineInst));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = load0_words[0];
  text_words[1] = load0_words[1];
  text_words[2] = load1_words[0];
  text_words[3] = load1_words[1];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  const std::array<uint64_t, 1> entry_offsets = {0};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250, entry_offsets);
  ASSERT_FALSE(blocks.empty());
  auto inst_it = blocks[0]->instructions().begin();
  ASSERT_NE(inst_it, blocks[0]->instructions().end());
  EXPECT_EQ((*inst_it).mnemonic(), "ds_load_b128");
  EXPECT_EQ((*inst_it).size(), sizeof(gfx1250::VdsMachineInst));
  ++inst_it;
  ASSERT_NE(inst_it, blocks[0]->instructions().end());
  EXPECT_EQ((*inst_it).mnemonic(), "ds_load_b128");
  EXPECT_EQ((*inst_it).size(), sizeof(gfx1250::VdsMachineInst));

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 4 * sizeof(uint32_t));

  const auto translation_words = section_words_for_test(*translations);
  const std::array<uint32_t, 2> expected0 = {load0_words[0], load0_words[1]};
  const std::array<uint32_t, 2> expected1 = {load1_words[0], load1_words[1]};
  EXPECT_TRUE(find_word_sequence_for_test(translation_words, expected0).has_value());
  EXPECT_TRUE(find_word_sequence_for_test(translation_words, expected1).has_value());
}

TEST(BinaryTranslator, Gfx1250InPlaceShadowsHighBankDsLoadAddress) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x1000;

  gfx1250::VdsMachineInst load{};
  load.encoding = 0x36;
  load.op = 0xFF;
  load.addr = 12;
  load.vdst = 32;

  std::array<uint32_t, 2> load_words{};
  std::memcpy(load_words.data(), &load, sizeof(load));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Bank1;
  text_words[1] = load_words[0];
  text_words[2] = load_words[1];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];

  std::array<uint32_t, 3> patched_words{};
  std::memcpy(patched_words.data(), text->data(), patched_words.size() * sizeof(uint32_t));
  EXPECT_EQ(patched_words[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::VdsMachineInst shadow_load{};
  std::memcpy(&shadow_load, patched_words.data() + 1, sizeof(shadow_load));
  EXPECT_EQ(shadow_load.encoding, 0x36u);
  EXPECT_EQ(shadow_load.op, 0xFFu);
  EXPECT_EQ(shadow_load.addr, 76u);
  EXPECT_EQ(shadow_load.vdst, 32u);
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250InPlaceShadowsHighBankSelector2DsLoadAddress) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank2 = 0xBF860002u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x1000;

  gfx1250::VdsMachineInst load{};
  load.encoding = 0x36;
  load.op = 0xFF;
  load.addr = 12;
  load.vdst = 32;

  std::array<uint32_t, 2> load_words{};
  std::memcpy(load_words.data(), &load, sizeof(load));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Bank2;
  text_words[1] = load_words[0];
  text_words[2] = load_words[1];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);
  EXPECT_NE(result.elf_bytes, image);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  expect_gfx1250_rdna4_entry_stub_for_test(translated);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopySpillsBankedHighBankSelector2DsLoadAddress) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank2 = 0xBF860002u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::VdsMachineInst load{};
  load.encoding = 0x36;
  load.op = 0xFF;
  load.addr = 12;
  load.vdst = 32;

  std::array<uint32_t, 2> load_words{};
  std::memcpy(load_words.data(), &load, sizeof(load));

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Bank2;
  text_words[1] = load_words[0];
  text_words[2] = load_words[1];
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  constexpr uint32_t kScratchLoadB32OffOff = 0xED05007Cu;
  constexpr uint32_t kSelector2Logical12ScratchOffset =
      (kGfx1250PrivateBorrowedVgprCountForTest + 256u + 12u) * sizeof(uint32_t) << 8;

  bool found_private_reload = false;
  for (size_t i = 0; i + 2u < copied_words.size(); ++i) {
    if (copied_words[i] == kScratchLoadB32OffOff && (copied_words[i + 1] & 0xFFu) == 76u &&
        copied_words[i + 2] == kSelector2Logical12ScratchOffset) {
      found_private_reload = true;
      break;
    }
  }
  EXPECT_TRUE(found_private_reload)
      << "selector 2 logical v12 should reload from its own banked private slot";

  bool found_shadow_load = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::VdsMachineInst shadow_load{};
    std::memcpy(&shadow_load, copied_words.data() + i, sizeof(shadow_load));
    if (shadow_load.encoding == 0x36u && shadow_load.op == 0xFFu && shadow_load.addr == 76u &&
        shadow_load.vdst == 32u) {
      found_shadow_load = true;
      break;
    }
  }
  EXPECT_TRUE(found_shadow_load);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVLshlAddU64HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0DstBank1 = 0xBF860041u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 594;
  inst.vdst = 4;
  inst.src0 = 256 + 4;
  inst.src1 = scalar_positive_inline_u32(2);
  inst.src2 = 18;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 10u * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 10u);
  EXPECT_EQ(copied_words[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3MachineInst align_hi{};
  std::memcpy(&align_hi, copied_words.data() + 1, sizeof(align_hi));
  EXPECT_EQ(align_hi.op, 534u);
  EXPECT_EQ(align_hi.vdst, 69u);
  EXPECT_EQ(align_hi.src0, 256u + 69u);
  EXPECT_EQ(align_hi.src1, 256u + 68u);
  EXPECT_EQ(align_hi.src2, scalar_positive_inline_u32(30));

  rdna4::Vop3MachineInst shl_lo{};
  std::memcpy(&shl_lo, copied_words.data() + 3, sizeof(shl_lo));
  EXPECT_EQ(shl_lo.op, 280u);
  EXPECT_EQ(shl_lo.vdst, 68u);
  EXPECT_EQ(shl_lo.src0, scalar_positive_inline_u32(2));
  EXPECT_EQ(shl_lo.src1, 256u + 68u);

  rdna4::Vop3SdstEncMachineInst add_lo{};
  std::memcpy(&add_lo, copied_words.data() + 5, sizeof(add_lo));
  EXPECT_EQ(add_lo.op, 768u);
  EXPECT_EQ(add_lo.vdst, 68u);
  EXPECT_LE(add_lo.sdst, 105u);
  EXPECT_EQ(add_lo.sdst & 1u, 0u);
  EXPECT_EQ(add_lo.src0, 256u + 68u);
  EXPECT_EQ(add_lo.src1, 18u);

  EXPECT_EQ(copied_words[7], build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3SdstEncMachineInst add_hi{};
  std::memcpy(&add_hi, copied_words.data() + 8, sizeof(add_hi));
  EXPECT_EQ(add_hi.op, 288u);
  EXPECT_EQ(add_hi.vdst, 69u);
  EXPECT_EQ(add_hi.sdst, kNullSgprForTest);
  EXPECT_EQ(add_hi.src0, 256u + 69u);
  EXPECT_EQ(add_hi.src1, 19u);
  EXPECT_EQ(add_hi.src2, add_lo.sdst);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVLshlAddU32HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0DstBank1 = 0xBF860041u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 582;
  inst.vdst = 5;
  inst.src0 = 256 + 3;
  inst.src1 = scalar_positive_inline_u32(10);
  inst.src2 = 256 + 5;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 5u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst shl{};
    std::memcpy(&shl, copied_words.data() + i, sizeof(shl));
    rdna4::Vop3MachineInst add{};
    std::memcpy(&add, copied_words.data() + i + 3, sizeof(add));
    if (shl.op == 280u && shl.vdst == 69u && shl.src0 == scalar_positive_inline_u32(10) &&
        shl.src1 == 256u + 67u &&
        copied_words[i + 2] == build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4) && add.op == 293u &&
        add.vdst == 69u && add.src0 == 256u + 5u && add.src1 == 256u + 69u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVLshlAddU32OverlappingDstSrc2) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc2DstBank1 = 0xBF860050u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 582;
  inst.vdst = 5;
  inst.src0 = 256 + 3;
  inst.src1 = scalar_positive_inline_u32(5);
  inst.src2 = 256 + 5;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc2DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = 256;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201, options);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);

  bool found_add = false;
  for (size_t i = 0; i + 2u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst shl{};
    std::memcpy(&shl, copied_words.data() + i, sizeof(shl));
    rdna4::Vop3MachineInst add{};
    std::memcpy(&add, copied_words.data() + i + 3, sizeof(add));
    if (shl.op == 280u && shl.src0 == scalar_positive_inline_u32(5) && shl.src1 == 256u + 3u &&
        copied_words[i + 2] == build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4) && add.op == 293u &&
        add.vdst == 69u && add.src0 == 256u + 69u && add.src1 == 256u + shl.vdst) {
      EXPECT_NE(shl.vdst, 69u);
      found_add = true;
    }
  }

  EXPECT_TRUE(found_add);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVLshlOrB32HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 598;
  inst.vdst = 5;
  inst.src0 = 256 + 3;
  inst.src1 = scalar_positive_inline_u32(4);
  inst.src2 = 256 + 8;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Bank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 4u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst shl{};
    std::memcpy(&shl, copied_words.data() + i, sizeof(shl));
    const auto or_inst = std::bit_cast<rdna4::Vop2MachineInst>(copied_words[i + 3]);
    if (shl.op == 280u && shl.vdst == 5u && shl.src0 == scalar_positive_inline_u32(4) &&
        shl.src1 == 256u + 67u &&
        copied_words[i + 2] == build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4) &&
        or_inst.op == 28u && or_inst.vdst == 5u && or_inst.src0 == 256u + 8u &&
        or_inst.vsrc1 == 5u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVCvtF32U32HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0DstBank1 = 0xBF860041u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 6;
  inst.vdst = 5;
  inst.src0 = 256 + 3;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (uint32_t word : copied_words) {
    const auto cvt = std::bit_cast<rdna4::Vop1MachineInst>(word);
    if (cvt.encoding == 0x3Fu && cvt.op == 6u && cvt.vdst == 69u && cvt.src0 == 256u + 67u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVRcpIflagF32HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0DstBank1 = 0xBF860041u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 43;
  inst.vdst = 5;
  inst.src0 = 256 + 3;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (uint32_t word : copied_words) {
    const auto rcp = std::bit_cast<rdna4::Vop1MachineInst>(word);
    if (rcp.encoding == 0x3Fu && rcp.op == 43u && rcp.vdst == 69u && rcp.src0 == 256u + 67u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVAddCoU32HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Src1DstBank1 = 0xBF860045u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3SdstEncMachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 768;
  inst.vdst = 5;
  inst.sdst = 20;
  inst.src0 = 256 + 3;
  inst.src1 = 256 + 8;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Src1DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::Vop3SdstEncMachineInst add{};
    std::memcpy(&add, copied_words.data() + i, sizeof(add));
    if (add.encoding == 0x35u && add.op == 768u && add.vdst == 69u && add.sdst == 20u &&
        add.src0 == 256u + 67u && add.src1 == 256u + 72u && add.src2 == 0u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVAddNcU32Vop3HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Src1DstBank1 = 0xBF860045u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 293;
  inst.vdst = 5;
  inst.src0 = 256 + 3;
  inst.src1 = 256 + 8;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Src1DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst add{};
    std::memcpy(&add, copied_words.data() + i, sizeof(add));
    if (add.encoding == 0x35u && add.op == 293u && add.vdst == 69u && add.src0 == 256u + 67u &&
        add.src1 == 256u + 72u && add.src2 == 0u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVMulF32E32HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Src1DstBank1 = 0xBF860045u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop2MachineInst inst{};
  inst.op = 8;
  inst.vdst = 5;
  inst.src0 = 256 + 3;
  inst.vsrc1 = 8;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Src1DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (uint32_t word : copied_words) {
    const auto mul = std::bit_cast<rdna4::Vop2MachineInst>(word);
    if (mul.encoding == 0u && mul.op == 8u && mul.vdst == 69u && mul.src0 == 256u + 67u &&
        mul.vsrc1 == 72u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVCmpGeI32Vop3HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Src1Bank1 = 0xBF860005u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 70;
  inst.vdst = 20;
  inst.src0 = 256 + 3;
  inst.src1 = 256 + 8;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Src1Bank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst cmp{};
    std::memcpy(&cmp, copied_words.data() + i, sizeof(cmp));
    if (cmp.encoding == 0x35u && cmp.op == 70u && cmp.vdst == 20u && cmp.src0 == 256u + 67u &&
        cmp.src1 == 256u + 72u && cmp.src2 == 0u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVCndmaskB32Vop3HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Src1DstBank1 = 0xBF860045u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 0x101;
  inst.vdst = 5;
  inst.src0 = 256 + 3;
  inst.src1 = 256 + 8;
  inst.src2 = 0;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Src1DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst cndmask{};
    std::memcpy(&cndmask, copied_words.data() + i, sizeof(cndmask));
    if (cndmask.encoding == 0x35u && cndmask.op == 0x101u && cndmask.vdst == 69u &&
        cndmask.src0 == 256u + 67u && cndmask.src1 == 256u + 72u && cndmask.src2 == 0u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVAddLshlU32Vop3HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbDstBank1 = 0xBF860040u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 0x247;
  inst.vdst = 5;
  inst.src0 = 256 + 3;
  inst.src1 = 256 + 8;
  inst.src2 = 256 + 12;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbDstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst add_lshl{};
    std::memcpy(&add_lshl, copied_words.data() + i, sizeof(add_lshl));
    if (add_lshl.encoding == 0x35u && add_lshl.op == 0x247u && add_lshl.vdst == 69u &&
        add_lshl.src0 == 256u + 3u && add_lshl.src1 == 256u + 8u && add_lshl.src2 == 256u + 12u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVFmaMixF32Vop3pHighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc1Bank1 = 0xBF860004u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3pMachineInst inst{};
  inst.encoding = 0xCC;
  inst.op = 0x20;
  inst.vdst = 5;
  inst.src0 = 256 + 3;
  inst.src1 = 256 + 8;
  inst.src2 = 256 + 12;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc1Bank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst fma{};
    std::memcpy(&fma, copied_words.data() + i, sizeof(fma));
    if (fma.encoding == 0x35u && fma.op == 531u && fma.vdst == 5u && fma.src0 == 256u + 3u &&
        fma.src1 == 256u + 72u && fma.src2 == 256u + 12u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVPkMulF32HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3pMachineInst inst{};
  inst.encoding = 0xCC;
  inst.op = 40;
  inst.vdst = 10;
  inst.src0 = 256 + 3;
  inst.src1 = 256 + 12;
  inst.opsel_hi = 0x1;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Bank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found_lo = false;
  bool found_hi = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst mul{};
    std::memcpy(&mul, copied_words.data() + i, sizeof(mul));
    if (mul.op == 264u && mul.vdst == 10u && mul.src0 == 256u + 67u && mul.src1 == 256u + 12u) {
      found_lo = true;
    }
    if (mul.op == 264u && mul.vdst == 11u && mul.src0 == 256u + 68u && mul.src1 == 256u + 12u) {
      found_hi = true;
    }
  }
  EXPECT_TRUE(found_lo);
  EXPECT_TRUE(found_hi);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVCvtPkF16F32HighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Src1DstBank1 = 0xBF860045u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop3MachineInst inst{};
  inst.encoding = 0x35;
  inst.op = 879;
  inst.vdst = 5;
  inst.src0 = 256 + 3;
  inst.src1 = 256 + 4;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc0Src1DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  bool found_lo_convert = false;
  bool found_hi_convert = false;
  bool found_pack = false;
  for (size_t i = 0; i + 1u < copied_words.size(); ++i) {
    rdna4::Vop3MachineInst convert{};
    std::memcpy(&convert, copied_words.data() + i, sizeof(convert));
    if (convert.op == 0x18Au && convert.src0 == 256u + 67u)
      found_lo_convert = true;
    if (convert.op == 0x18Au && convert.src0 == 256u + 68u)
      found_hi_convert = true;

    const auto pack = std::bit_cast<rdna4::Vop2MachineInst>(copied_words[i]);
    if (pack.op == 28u && pack.vdst == 69u)
      found_pack = true;
  }
  EXPECT_TRUE(found_lo_convert);
  EXPECT_TRUE(found_hi_convert);
  EXPECT_TRUE(found_pack);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyShadowsVAddU64LiteralHighBankOperands) {
  constexpr uint32_t kGfx1250SSetVgprMsbSrc1DstBank1 = 0xBF860044u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  gfx1250::Vop2InstLiteralMachineInst inst{};
  inst.src0 = 255;
  inst.vsrc1 = 0;
  inst.vdst = 0;
  inst.op = 40;
  inst.simm32 = 0x4000;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  text_words[0] = kGfx1250SSetVgprMsbSrc1DstBank1;
  std::memcpy(text_words.data() + 1, &inst, sizeof(inst));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_GE(translations->size(), 7u * sizeof(uint32_t));

  const auto copied_words = entry_body_words_after_gfx1250_rdna4_stub_for_test(*translations);
  ASSERT_GE(copied_words.size(), 7u);
  EXPECT_EQ(copied_words[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3SdstEncMachineInst add_lo{};
  std::memcpy(&add_lo, copied_words.data() + 1, sizeof(add_lo));
  EXPECT_EQ(add_lo.op, 768u);
  EXPECT_EQ(add_lo.vdst, 64u);
  EXPECT_LE(add_lo.sdst, 105u);
  EXPECT_EQ(add_lo.sdst & 1u, 0u);
  EXPECT_EQ(add_lo.src0, 255u);
  EXPECT_EQ(add_lo.src1, 256u + 64u);
  EXPECT_EQ(copied_words[3], 0x4000u);

  EXPECT_EQ(copied_words[4], build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));

  rdna4::Vop3SdstEncMachineInst add_hi{};
  std::memcpy(&add_hi, copied_words.data() + 5, sizeof(add_hi));
  EXPECT_EQ(add_hi.op, 288u);
  EXPECT_EQ(add_hi.vdst, 65u);
  EXPECT_EQ(add_hi.sdst, kNullSgprForTest);
  EXPECT_EQ(add_hi.src0, scalar_positive_inline_u32(0));
  EXPECT_EQ(add_hi.src1, 256u + 65u);
  EXPECT_EQ(add_hi.src2, add_lo.sdst);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopyPreservesLowSourceAliasedWithHighBankShadow) {
  constexpr uint32_t kGfx1250SSetVgprMsb0 = 0xBF860000u;
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  const auto make_mov_b32 = [](uint8_t vdst, uint16_t src0) {
    gfx1250::Vop1MachineInst mov{};
    mov.encoding = 0x3F;
    mov.op = 1;
    mov.vdst = vdst;
    mov.src0 = src0;
    return std::bit_cast<uint32_t>(mov);
  };

  gfx1250::Vop2MachineInst add{};
  add.src0 = 256u + 2u;
  add.vsrc1 = 66;
  add.vdst = 66;
  add.op = 40;

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  size_t index = 0;
  text_words[index++] = kGfx1250SSetVgprMsbSrc0Bank1;
  text_words[index++] = std::bit_cast<uint32_t>(add);
  text_words[index++] = kGfx1250SSetVgprMsb0;
  for (uint16_t reg = 0; reg < 256; ++reg)
    text_words[index++] =
        make_mov_b32(static_cast<uint8_t>(reg), static_cast<uint16_t>(256u + reg));
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  constexpr uint32_t kScratchLoadB32OffOff = 0xED05007Cu;
  constexpr uint32_t kScratchStoreB32OffOff = 0xED06807Cu;
  constexpr uint32_t kHighLogical2ScratchOffset =
      (kGfx1250PrivateBorrowedVgprCountForTest + 2u) * sizeof(uint32_t) << 8;
  constexpr uint32_t kHighLogical3ScratchOffset =
      (kGfx1250PrivateBorrowedVgprCountForTest + 3u) * sizeof(uint32_t) << 8;

  std::vector<uint8_t> high_source_reload_vgprs;
  for (size_t i = 0; i + 2u < cave_words.size(); ++i) {
    if (cave_words[i] != kScratchLoadB32OffOff)
      continue;
    if (cave_words[i + 2] == kHighLogical2ScratchOffset ||
        cave_words[i + 2] == kHighLogical3ScratchOffset)
      high_source_reload_vgprs.push_back(static_cast<uint8_t>(cave_words[i + 1] & 0xFFu));
  }
  ASSERT_EQ(high_source_reload_vgprs.size(), 2u);
  EXPECT_NE(high_source_reload_vgprs[0], 66u);
  EXPECT_NE(high_source_reload_vgprs[0], 67u);
  EXPECT_NE(high_source_reload_vgprs[1], 66u);
  EXPECT_NE(high_source_reload_vgprs[1], 67u);

  std::vector<uint8_t> borrowed_store_vgprs;
  for (size_t i = 0; i + 2u < cave_words.size(); ++i) {
    if (cave_words[i] != kScratchStoreB32OffOff)
      continue;
    if (cave_words[i + 2] == 0u || cave_words[i + 2] == (sizeof(uint32_t) << 8))
      borrowed_store_vgprs.push_back(static_cast<uint8_t>(cave_words[i + 1] >> 23));
  }
  ASSERT_EQ(borrowed_store_vgprs.size(), 2u);
  EXPECT_EQ(borrowed_store_vgprs, high_source_reload_vgprs)
      << "no-free fallback should save and restore the borrowed low temporary pair";

  std::optional<rdna4::Vop3SdstEncMachineInst> add_lo;
  for (size_t i = 0; i + 1u < cave_words.size(); ++i) {
    rdna4::Vop3SdstEncMachineInst candidate{};
    std::memcpy(&candidate, cave_words.data() + i, sizeof(candidate));
    if (candidate.encoding == 0x35u && candidate.op == 768u && candidate.vdst == 66u &&
        candidate.src1 == 256u + 66u) {
      add_lo = candidate;
      break;
    }
  }
  ASSERT_TRUE(add_lo.has_value());
  EXPECT_NE(add_lo->src0, 256u + 66u)
      << "private reload of high v258 must not clobber the low v66 source";
  EXPECT_NE(add_lo->src0, 256u + 67u);
}

TEST(BinaryTranslator, Gfx1250ExpandedCopySpillsAliasedHighBankShadowWindow) {
  constexpr uint32_t kGfx1250SSetVgprMsb0 = 0xBF860000u;
  constexpr uint32_t kGfx1250SSetVgprMsbSrc0Bank1 = 0xBF860001u;
  constexpr uint32_t kGfx1250SSetVgprMsbDstBank1 = 0xBF860040u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop0 = 0xBF800000u;
  constexpr uint64_t kTextSize = 0x21000;

  const auto make_mov_b32 = [](uint8_t vdst, uint16_t src0) {
    gfx1250::Vop1MachineInst mov{};
    mov.encoding = 0x3F;
    mov.op = 1;
    mov.vdst = vdst;
    mov.src0 = src0;
    return std::bit_cast<uint32_t>(mov);
  };

  std::vector<uint32_t> text_words(kTextSize / sizeof(uint32_t), kGfx1250SNop0);
  size_t index = 0;
  text_words[index++] = kGfx1250SSetVgprMsbDstBank1;
  text_words[index++] = make_mov_b32(0, scalar_positive_inline_u32(1));
  text_words[index++] = kGfx1250SSetVgprMsb0;
  for (uint16_t reg = 0; reg < 256; ++reg)
    text_words[index++] =
        make_mov_b32(static_cast<uint8_t>(reg), static_cast<uint16_t>(256u + reg));
  text_words[index++] = kGfx1250SSetVgprMsbSrc0Bank1;
  text_words[index++] = make_mov_b32(10, 256u);
  text_words.back() = kGfx1250SEndpgm;

  auto image =
      make_minimal_amdgpu_elf_with_descriptor_and_text(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.warnings.empty());

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text = translated.text_sections()[0];

  KernelDescriptorTranslator descriptor_parser(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto descriptor_infos = descriptor_parser.translate_image(
      result.elf_bytes, text->sectionOffset(), text->size(), KernelDescriptorTranslationOptions{});
  ASSERT_EQ(descriptor_infos.size(), 1u);
  EXPECT_EQ(descriptor_infos[0].target_private_size,
            (kGfx1250PrivateBorrowedVgprCountForTest + 1u +
             kGfx1250HighBankShadowLowSaveVgprCountForTest) *
                sizeof(uint32_t));

  const Section *translations = local_expanded_text_section_for_test(translated, kTextSize);
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(translations->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> cave_words(translations->size() / sizeof(uint32_t));
  std::memcpy(cave_words.data(), translations->data(), translations->size());

  constexpr uint32_t kScratchStoreB32OffOff = 0xED06807Cu;
  constexpr uint32_t kScratchLoadB32OffOff = 0xED05007Cu;
  std::optional<size_t> high_store;
  std::optional<size_t> high_load;
  std::optional<size_t> low_save_store;
  std::optional<size_t> low_restore_load;
  constexpr uint32_t kHighLogical0Offset =
      kGfx1250PrivateBorrowedVgprCountForTest * sizeof(uint32_t) << 8;
  constexpr uint32_t kLowSaveSlot0Offset =
      (kGfx1250PrivateBorrowedVgprCountForTest + 1u) * sizeof(uint32_t) << 8;
  for (size_t i = 0; i + 2u < cave_words.size(); ++i) {
    if (cave_words[i] == kScratchStoreB32OffOff && cave_words[i + 1] == (64u << 23) &&
        cave_words[i + 2] == kHighLogical0Offset)
      high_store = i;
    if (cave_words[i] == kScratchLoadB32OffOff && cave_words[i + 1] == 64u &&
        cave_words[i + 2] == kHighLogical0Offset)
      high_load = i;
    if (cave_words[i] == kScratchStoreB32OffOff && cave_words[i + 1] == (64u << 23) &&
        cave_words[i + 2] == kLowSaveSlot0Offset)
      low_save_store = i;
    if (cave_words[i] == kScratchLoadB32OffOff && cave_words[i + 1] == 64u &&
        cave_words[i + 2] == kLowSaveSlot0Offset)
      low_restore_load = i;
  }

  ASSERT_TRUE(high_store.has_value());
  ASSERT_TRUE(high_load.has_value());
  ASSERT_TRUE(low_save_store.has_value());
  ASSERT_TRUE(low_restore_load.has_value());
  EXPECT_LT(*low_save_store, *high_store);
  EXPECT_LT(*high_store, *low_restore_load);
  EXPECT_LT(*low_restore_load, *high_load);
}

TEST(BinaryTranslator, TranslatesIreeGfx1250HsacosWithoutWarnings) {
#ifndef IREE_GFX1250_VALIDATION_DIR
  GTEST_SKIP() << "IREE gfx1250 validation artifact directory was not configured";
#else
  const std::filesystem::path root(IREE_GFX1250_VALIDATION_DIR);
  if (!std::filesystem::exists(root))
    GTEST_SKIP() << "IREE gfx1250 validation artifact directory is missing: " << root;

  std::vector<std::filesystem::path> hsacos;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file())
      continue;
    if (entry.path().extension() == ".hsaco")
      hsacos.push_back(entry.path());
  }
  std::ranges::sort(hsacos);

  ASSERT_EQ(hsacos.size(), 17u) << "workspace artifact set should match instruction_audit.json";
  const std::set<std::string> expected_deferred_warnings;
  std::set<std::string> unique_warnings;
  for (const auto &path : hsacos) {
    SCOPED_TRACE(path.string());
    AmdGpuCodeObject co(path.string());
    ASSERT_TRUE(co.is_valid());
    ASSERT_EQ(co.target_id(), ROCJITSU_CODE_TARGET_GFX1250);

    BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                EF_AMDGPU_MACH_AMDGCN_GFX1201);
    auto result = translator.translate(co);
    ASSERT_FALSE(result.elf_bytes.empty());
    unique_warnings.insert(result.warnings.begin(), result.warnings.end());

    AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    const Section *translations = find_section(translated, ".rj_translations");
    EXPECT_FALSE(section_contains_vop3_opcode(translations, 0))
        << "gfx1250 v_mul_u64 must not remain in expanded translated code";
    EXPECT_FALSE(section_contains_low_first_overlapping_v_mul_u64_pair(translations))
        << "v_mul_u64 low-half lowering must not clobber an operand before high-half lowering";
    const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(result.elf_bytes.data());
    EXPECT_EQ(ehdr->e_flags & EF_AMDGPU_MACH, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  }

  std::set<std::string> unexpected_warnings;
  std::ranges::set_difference(unique_warnings, expected_deferred_warnings,
                              std::inserter(unexpected_warnings, unexpected_warnings.end()));
  EXPECT_TRUE(unexpected_warnings.empty()) << "Unexpected translation warnings:\n"
                                           << testing::PrintToString(unexpected_warnings);

  std::set<std::string> missing_warnings;
  std::ranges::set_difference(expected_deferred_warnings, unique_warnings,
                              std::inserter(missing_warnings, missing_warnings.end()));
  EXPECT_TRUE(missing_warnings.empty()) << "Expected deferred WMMA warnings disappeared:\n"
                                        << testing::PrintToString(missing_warnings);
#endif
}

TEST(BinaryTranslator, TranslatesNoGapIreeGfx1250HsacosWithoutWarnings) {
#ifndef IREE_GFX1250_VALIDATION_DIR
  GTEST_SKIP() << "IREE gfx1250 validation artifact directory was not configured";
#else
  const std::filesystem::path root(IREE_GFX1250_VALIDATION_DIR);
  const std::array<std::filesystem::path, 2> hsacos = {
      root / "extra/regression_linalg_ops_dynamic/dumps/binaries/"
             "module_linalg_ops_dynamic_linked_rocm_hsaco_fb.hsaco",
      root / "extra/regression_reduction_broadcast_elementwise/dumps/binaries/"
             "module_reduction_broadcast_elementwise_linked_rocm_hsaco_fb.hsaco",
  };

  for (const auto &path : hsacos) {
    SCOPED_TRACE(path.string());
    ASSERT_TRUE(std::filesystem::exists(path));
    AmdGpuCodeObject co(path.string());
    ASSERT_TRUE(co.is_valid());
    ASSERT_EQ(co.target_id(), ROCJITSU_CODE_TARGET_GFX1250);

    BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                EF_AMDGPU_MACH_AMDGCN_GFX1201);
    auto result = translator.translate(co);
    ASSERT_FALSE(result.elf_bytes.empty());
    EXPECT_TRUE(result.warnings.empty()) << "Translation warnings:\n"
                                         << testing::PrintToString(result.warnings);

    AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(result.elf_bytes.data());
    EXPECT_EQ(ehdr->e_flags & EF_AMDGPU_MACH, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  }
#endif
}

} // namespace
} // namespace rocjitsu

// --- WaitcntTranslator tests ---
#include "rocjitsu/code/dbt/waitcnt_translator.h"

using rocjitsu::decode_waitcnt_gfx9;
using rocjitsu::encode_waitcnt_gfx12;
using rocjitsu::WaitcntValues;

TEST(WaitcntTranslator, DecodeVmcnt0) {
  auto v = decode_waitcnt_gfx9(0x0000);
  EXPECT_EQ(v.vmcnt, 0);
  EXPECT_EQ(v.lgkmcnt, 0);
  EXPECT_EQ(v.expcnt, 0);
}

TEST(WaitcntTranslator, DecodeAllRelaxed) {
  auto v = decode_waitcnt_gfx9(0xCF7F);
  EXPECT_EQ(v.vmcnt, 0x3F);
  EXPECT_EQ(v.lgkmcnt, 0x0F);
  EXPECT_EQ(v.expcnt, 0x07);
}

TEST(WaitcntTranslator, DecodeVmcnt15Lgkm0) {
  uint16_t simm16 = 0x000F;
  auto v = decode_waitcnt_gfx9(simm16);
  EXPECT_EQ(v.vmcnt, 15);
  EXPECT_EQ(v.lgkmcnt, 0);
  EXPECT_EQ(v.expcnt, 0);
}

TEST(WaitcntTranslator, EncodeAllZeroProducesMultipleWords) {
  WaitcntValues v{0, 0, 0};
  auto words = encode_waitcnt_gfx12(v);
  EXPECT_GE(words.size(), 3u);
}

TEST(WaitcntTranslator, EncodeAllRelaxedProducesNop) {
  WaitcntValues v{0x3F, 0x0F, 0x07};
  auto words = encode_waitcnt_gfx12(v);
  ASSERT_EQ(words.size(), 1u);
  uint8_t op = (words[0] >> 16) & 0x7F;
  EXPECT_EQ(op, 0);
}

TEST(WaitcntTranslator, EncodeVmcnt0EmitsLoadcntAndStorecnt) {
  WaitcntValues v{0, 0x0F, 0x07};
  auto words = encode_waitcnt_gfx12(v);
  EXPECT_GE(words.size(), 2u);

  bool has_loadcnt = false;
  bool has_storecnt_dscnt = false;
  for (auto w : words) {
    uint8_t op = (w >> 16) & 0x7F;
    if (op == 64)
      has_loadcnt = true;
    if (op == 73)
      has_storecnt_dscnt = true;
  }
  EXPECT_TRUE(has_loadcnt);
  EXPECT_TRUE(has_storecnt_dscnt);
}

namespace cdna3 = rocjitsu::cdna3;
namespace cdna4 = rocjitsu::cdna4;

constexpr uint16_t kAnyExpectedField = 0xffff;

enum class ExpectedCdna3Kind {
  Vop3,
  Vop3p,
  Vop2,
  Vop1,
  Sop1,
  Vop3pMfma,
  Ds,
  Mubuf,
  Sopp,
};

struct ExpectedCdna3Inst {
  ExpectedCdna3Kind kind = ExpectedCdna3Kind::Vop3;
  uint16_t op = 0;
  uint16_t vdst = kAnyExpectedField;
  uint16_t acc_cd = kAnyExpectedField;
  uint16_t src0 = kAnyExpectedField;
  uint16_t src1 = kAnyExpectedField;
  uint16_t src2 = kAnyExpectedField;
};

struct Cdna4ToCdna3SemanticRuleCase {
  const char *name = "";
  uint16_t encoding_id = 0;
  uint16_t opcode = 0;
  std::array<uint32_t, 2> words{};
  std::vector<ExpectedCdna3Inst> expected{};
};

ExpectedCdna3Inst expect_vop3(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_vop3p(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3p;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_vop2(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop2;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_vop1(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop1;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_sop1(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Sop1;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_mfma(uint16_t op, uint16_t vdst, uint16_t acc_cd, uint16_t src0,
                              uint16_t src1, uint16_t src2) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3pMfma;
  inst.op = op;
  inst.vdst = vdst;
  inst.acc_cd = acc_cd;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  return inst;
}

ExpectedCdna3Inst expect_ds(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Ds;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_mubuf(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Mubuf;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_sopp(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Sopp;
  inst.op = op;
  return inst;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_bitop3_sequence(bool b16) {
  // Truth table 0xde lowers to S2 ^ S1 ^ (S1 & S2) ^ S0 ^ (S0 & S1).
  std::vector<ExpectedCdna3Inst> expected = {
      expect_vop3(321), // v_mov_b32
      expect_vop3(277), // v_xor_b32
      expect_vop3(275), // v_and_b32
      expect_vop3(277), // v_xor_b32
      expect_vop3(277), // v_xor_b32
      expect_vop3(275), // v_and_b32
      expect_vop3(277), // v_xor_b32
  };
  if (b16) {
    expected.push_back(expect_vop3(274)); // v_lshlrev_b32
    expected.push_back(expect_vop3(272)); // v_lshrrev_b32
  }
  expected.push_back(expect_vop3(321)); // v_mov_b32 copy scratch accumulator to vdst.
  expected.push_back(expect_sopp(2));   // s_branch back to original fallthrough.
  return expected;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_mfma_sequence(uint16_t narrow_op,
                                                            uint16_t src2 = 128) {
  return {
      // Wide-K MFMA lowering is high-half first, then low-half. This ordering
      // is part of the layout contract validated against the Qwen kernels.
      expect_mfma(narrow_op, 0, 1, 258, 262, src2),
      expect_mfma(narrow_op, 0, 1, 256, 260, 256),
      expect_sopp(2),
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_buffer_load_lds_sequence(uint16_t mubuf_op,
                                                                       uint16_t ds_op) {
  return {
      expect_sop1(1),         // s_mov_b64 save EXEC.
      expect_sop1(0),         // s_mov_b32 exec_lo, -1.
      expect_sop1(0),         // s_mov_b32 exec_hi, -1.
      expect_vop3(652),       // v_mbcnt_lo_u32_b32
      expect_vop3(653),       // v_mbcnt_hi_u32_b32
      expect_vop3(274),       // v_lshlrev_b32 lane_id, 4
      expect_vop3(308),       // v_add_u32 m0, lane_offset
      expect_sop1(1),         // s_mov_b64 restore EXEC.
      expect_mubuf(mubuf_op), // buffer_load_dwordx{3,4} into scratch VGPRs.
      expect_sopp(12),        // s_waitcnt 0 before consuming VMEM data.
      expect_ds(ds_op),       // ds_write_b96/b128
      expect_sopp(12),        // s_waitcnt lgkmcnt(0) for the explicit DS write.
      expect_sopp(2),         // s_branch back to original fallthrough.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_permlane32_swap_sequence() {
  return {
      expect_sop1(1),   // s_mov_b64 save EXEC.
      expect_sop1(0),   // s_mov_b32 exec_lo, -1.
      expect_sop1(0),   // s_mov_b32 exec_hi, -1.
      expect_vop3(652), // v_mbcnt_lo_u32_b32
      expect_vop3(653), // v_mbcnt_hi_u32_b32
      expect_vop3(277), // v_xor_b32 lane, 32.
      expect_vop3(274), // v_lshlrev_b32 byte address.
      expect_ds(63),    // ds_bpermute_b32 from old vdst high half.
      expect_ds(63),    // ds_bpermute_b32 from old src low half.
      expect_sopp(12),  // s_waitcnt lgkmcnt(0).
      expect_sop1(0),   // s_mov_b32 exec_lo, low-half mask.
      expect_sop1(0),   // s_mov_b32 exec_hi, low-half mask.
      expect_vop3(321), // v_mov_b32 src <- old vdst high.
      expect_sop1(0),   // s_mov_b32 exec_lo, high-half mask.
      expect_sop1(0),   // s_mov_b32 exec_hi, high-half mask.
      expect_vop3(321), // v_mov_b32 vdst <- old src low.
      expect_sop1(1),   // s_mov_b64 restore EXEC.
      expect_sopp(2),   // s_branch back to original fallthrough.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_raw_b16_pack_sequence() {
  return {
      expect_vop3(321), // v_mov_b32 -1
      expect_vop3(272), // v_lshrrev_b32 16, mask
      expect_vop3(275), // v_and_b32 low half
      expect_vop3(275), // v_and_b32 high half
      expect_vop3(274), // v_lshlrev_b32 16, high half
      expect_vop3(276), // v_or_b32
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_pk_f16_f32_sequence() {
  return {
      expect_vop3(330), // v_cvt_f16_f32 low half into scratch.
      expect_vop3(330), // v_cvt_f16_f32 high half into scratch.
      expect_vop3(274), // v_lshlrev_b32 16, high half.
      expect_vop3(276), // v_or_b32 pack low/high halves into vdst.
      expect_sopp(2),   // s_branch back to original fallthrough.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_pk_bf16_f32_sequence() {
  return {
      expect_vop3(cdna3::kVMovB32Vop3),     // copy the low source.
      expect_vop3(cdna3::kVLshrrevB32Vop3), // extract the BF16 lsb.
      expect_vop3(cdna3::kVAndB32Vop3),     // isolate the BF16 lsb.
      expect_vop2(cdna3::kVAddU32Vop2),     // add the 0x7fff rounding bias.
      expect_vop3(cdna3::kVAddU32Vop3),     // apply low RNE bias.
      expect_vop3(cdna3::kVLshrrevB32Vop3), // keep the rounded low BF16 half.
      expect_vop3(cdna3::kVMovB32Vop3),     // copy the high source.
      expect_vop3(cdna3::kVLshrrevB32Vop3), // extract the BF16 lsb.
      expect_vop3(cdna3::kVAndB32Vop3),     // isolate the BF16 lsb.
      expect_vop2(cdna3::kVAddU32Vop2),     // add the 0x7fff rounding bias.
      expect_vop3(cdna3::kVAddU32Vop3),     // apply high RNE bias.
      expect_vop3(cdna3::kVLshrrevB32Vop3), // keep the rounded high BF16 half.
      expect_vop3(cdna3::kVLshlrevB32Vop3), // move the high half into position.
      expect_vop3(cdna3::kVOrB32Vop3),      // pack the low and high halves.
      expect_sopp(2),
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_f32_bf16_sequence() {
  return {
      expect_vop3(cdna3::kVLshlrevB32Vop3), // place BF16 bits in the FP32 high half.
      expect_sopp(2),
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_dot2_f32_bf16_sequence() {
  return {
      expect_vop3(cdna3::kVLshlrevB32Vop3), // widen A low BF16.
      expect_vop3(cdna3::kVLshlrevB32Vop3), // widen B low BF16.
      expect_vop3(cdna3::kVLshlrevB32Vop3), // widen A high BF16.
      expect_vop3(cdna3::kVLshlrevB32Vop3), // widen B high BF16.
      expect_vop3(cdna3::kVMulF32Vop3),
      expect_vop3(cdna3::kVMulF32Vop3),
      expect_vop3(cdna3::kVAddF32Vop3),
      expect_vop3(cdna3::kVAddF32Vop3),
      expect_sopp(2),
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_ds_read_b64_tr_b16_sequence(bool acc_dst = false) {
  std::vector<ExpectedCdna3Inst> expected = {
      expect_sop1(1),   // s_mov_b64 save EXEC.
      expect_sop1(0),   // s_mov_b32 exec_lo, all lanes.
      expect_sop1(0),   // s_mov_b32 exec_hi, all lanes.
      expect_ds(118),   // ds_read_b64
      expect_sopp(12),  // s_waitcnt lgkmcnt(0)
      expect_vop3(652), // v_mbcnt_lo_u32_b32
      expect_vop3(653), // v_mbcnt_hi_u32_b32
      expect_vop3(275), // v_and_b32
      expect_vop3(274), // v_lshlrev_b32
      expect_vop3(275), // v_and_b32
      expect_vop3(274), // v_lshlrev_b32
      expect_vop3(275), // v_and_b32
      expect_vop3(276), // v_or_b32
      expect_vop3(308), // v_add_u32
      expect_vop3(274), // v_lshlrev_b32
      expect_vop3(276), // v_or_b32
      expect_ds(63),    // ds_bpermute_b32
      expect_ds(63),    // ds_bpermute_b32
      expect_sopp(12),  // s_waitcnt lgkmcnt(0)
      expect_vop3(493), // v_perm_b32
      expect_vop3(308), // v_add_u32
      expect_ds(63),    expect_ds(63), expect_sopp(12), expect_vop3(493), expect_vop3(308),
      expect_ds(63),    expect_ds(63), expect_sopp(12), expect_vop3(493), expect_vop3(308),
      expect_ds(63),    expect_ds(63), expect_sopp(12), expect_vop3(493),
      expect_sop1(1), // s_mov_b64 restore EXEC.
  };
  auto first_pack = expected_cdna3_raw_b16_pack_sequence();
  expected.insert(expected.end(), first_pack.begin(), first_pack.end());
  auto second_pack = expected_cdna3_raw_b16_pack_sequence();
  expected.insert(expected.end(), second_pack.begin(), second_pack.end());
  if (acc_dst) {
    expected.push_back(expect_vop3p(cdna3::kVAccvgprWriteVop3p));
    expected.push_back(expect_vop3p(cdna3::kVAccvgprWriteVop3p));
  }
  expected.push_back(expect_sopp(2));
  return expected;
}

template <typename MachineInst>
std::array<uint32_t, 2> encode_two_word_inst(const MachineInst &inst) {
  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

std::array<uint32_t, 2> make_cdna4_bitop3_words(uint16_t opcode, uint8_t vdst) {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = opcode;
  inst.vdst = vdst;
  inst.src0 = static_cast<uint16_t>(256 + vdst + 1);
  inst.src1 = static_cast<uint16_t>(256 + vdst + 2);
  inst.src2 = static_cast<uint16_t>(256 + vdst + 3);

  // Use a non-trivial LUT so the generic expansion emits real AND/XOR work and
  // cannot accidentally pass by lowering to a simple move.
  constexpr uint8_t kTruthTable = 0xde;
  inst.omod = kTruthTable >> 6;
  inst.abs = (kTruthTable >> 3) & 0x7;
  inst.neg = kTruthTable & 0x7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_bitop3_b16_unsupported_op_sel_words() {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = 563;
  inst.vdst = 8;
  inst.src0 = 256 + 9;
  inst.src1 = 256 + 10;
  inst.src2 = 256 + 11;
  inst.op_sel = 1;
  inst.omod = 1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_pk_f16_f32_words() {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = 615;
  inst.vdst = 0;
  inst.src0 = 256 + 1;
  inst.src1 = 256 + 2;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_pk_bf16_f32_words() {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = cdna4::kVCvtPkBf16F32Vop3;
  inst.vdst = 0;
  inst.src0 = 256 + 1;
  inst.src1 = 256 + 2;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_f32_bf16_words(uint16_t encoding_id) {
  rocjitsu::cdna4::Vop1MachineInst inst{};
  inst.encoding = 0x3f;
  inst.op = cdna4::kVCvtF32Bf16Vop1;
  inst.vdst = static_cast<uint8_t>((encoding_id - 0xFCu) << 6);
  inst.src0 = 256 + 1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_dot2_f32_bf16_words() {
  rocjitsu::cdna4::Vop3pMachineInst inst{};
  inst.encoding = 0x1A7;
  inst.op = cdna4::kVDot2F32Bf16Vop3p;
  inst.vdst = 0;
  inst.src0 = 256 + 1;
  inst.src1 = 256 + 2;
  inst.src2 = 256;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_permlane_swap_b32_words(uint16_t encoding_id, uint16_t opcode) {
  rocjitsu::cdna4::Vop1MachineInst inst{};
  // The legalization table's VOP1 encoding ids (0xfc..0xff) are the generated
  // primary-decode ids, not the raw 7-bit VOP1 selector.  Primary decode looks
  // at bits 31:23, so VOP1 contributes its fixed selector in bits 31:25 and
  // VDST[7:6] in bits 24:23.  Keep the real VOP1 selector at 0x3f and vary
  // VDST's high bits to exercise each generated semantic rule.
  inst.encoding = 0x3f;
  inst.op = opcode;
  inst.vdst = static_cast<uint8_t>((encoding_id - 0xFCu) << 6);
  inst.src0 = 256 + 1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_permlane16_swap_b32_words(uint16_t encoding_id) {
  return make_cdna4_permlane_swap_b32_words(encoding_id, cdna4::kVPermlane16SwapB32Vop1);
}

std::array<uint32_t, 2> make_cdna4_permlane32_swap_b32_words(uint16_t encoding_id) {
  return make_cdna4_permlane_swap_b32_words(encoding_id, cdna4::kVPermlane32SwapB32Vop1);
}

std::array<uint32_t, 2> make_cdna4_mfma_words(uint8_t opcode, uint8_t vdst, uint16_t src0,
                                              uint16_t src1, uint16_t src2 = 128) {
  rocjitsu::cdna4::Vop3pMfmaMachineInst inst{};
  inst.encoding = 0x1A7;
  inst.op = opcode;
  inst.vdst = vdst;
  inst.acc_cd = 1;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_mfma_vgpr_dst_alias_words() {
  rocjitsu::cdna4::Vop3pMfmaMachineInst inst{};
  inst.encoding = 0x1A7;
  inst.op = 84;
  inst.vdst = 0;
  inst.acc_cd = 0;
  // Ordinary-VGPR destination v[0:3] overlaps the first wide source window.
  // The lowering must therefore place the first narrow MFMA's partial result in
  // scratch and report that scratch through TranslationContext::require_vgprs().
  inst.src0 = 256;
  inst.src1 = 260;
  inst.src2 = 128;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_dot2c_unimplemented_expand_words() {
  // v_dot2c_f32_bf16 is present in CDNA4 and not CDNA3. The generated
  // legalization table marks raw encoding-id 88/opcode 22 as EXPAND, but no
  // handwritten semantic rule exists yet.
  return {0x2C000000U, 0x00000000U};
}

std::array<uint32_t, 2> make_cdna4_ds_read_b64_tr_b16_words(bool acc = false) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = 227;
  inst.addr = 2;
  inst.vdst = 0;
  inst.acc = acc ? 1 : 0;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_buffer_load_lds_words(uint8_t op) {
  rocjitsu::cdna4::MubufMachineInst inst{};
  inst.encoding = 0x38;
  inst.op = op;
  inst.lds = 1;
  inst.offen = 1;
  inst.vaddr = 2;
  inst.vdata = 0;
  inst.srsrc = 4;
  inst.soffset = 0;
  return encode_two_word_inst(inst);
}

std::vector<Cdna4ToCdna3SemanticRuleCase> cdna4_to_cdna3_semantic_rule_cases() {
  return {
      {"VBitop3B16", 0x1A4, 563, make_cdna4_bitop3_words(563, 8),
       expected_cdna3_bitop3_sequence(true)},
      {"VBitop3B32", 0x1A4, 564, make_cdna4_bitop3_words(564, 16),
       expected_cdna3_bitop3_sequence(false)},
      {"VCvtPkF16F32", 0x1A4, 615, make_cdna4_cvt_pk_f16_f32_words(),
       expected_cdna3_cvt_pk_f16_f32_sequence()},
      {"VCvtPkBf16F32", 0x1A4, cdna4::kVCvtPkBf16F32Vop3, make_cdna4_cvt_pk_bf16_f32_words(),
       expected_cdna3_cvt_pk_bf16_f32_sequence()},
      {"VCvtF32Bf16E32", 0xFC, cdna4::kVCvtF32Bf16Vop1, make_cdna4_cvt_f32_bf16_words(0xFC),
       expected_cdna3_cvt_f32_bf16_sequence()},
      {"VCvtF32Bf16E32Hi1", 0xFD, cdna4::kVCvtF32Bf16Vop1, make_cdna4_cvt_f32_bf16_words(0xFD),
       expected_cdna3_cvt_f32_bf16_sequence()},
      {"VCvtF32Bf16E32Hi2", 0xFE, cdna4::kVCvtF32Bf16Vop1, make_cdna4_cvt_f32_bf16_words(0xFE),
       expected_cdna3_cvt_f32_bf16_sequence()},
      {"VCvtF32Bf16E32Hi3", 0xFF, cdna4::kVCvtF32Bf16Vop1, make_cdna4_cvt_f32_bf16_words(0xFF),
       expected_cdna3_cvt_f32_bf16_sequence()},
      {"VDot2F32Bf16", 0x1A7, cdna4::kVDot2F32Bf16Vop3p, make_cdna4_dot2_f32_bf16_words(),
       expected_cdna3_dot2_f32_bf16_sequence()},
      {"VPermlane16SwapB32E32", 0xFC, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(0xFC), expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane16SwapB32E32Hi1", 0xFD, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(0xFD), expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane16SwapB32E32Hi2", 0xFE, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(0xFE), expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane16SwapB32E32Hi3", 0xFF, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(0xFF), expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane32SwapB32E32", 0xFC, 90, make_cdna4_permlane32_swap_b32_words(0xFC),
       expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane32SwapB32E32Hi1", 0xFD, 90, make_cdna4_permlane32_swap_b32_words(0xFD),
       expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane32SwapB32E32Hi2", 0xFE, 90, make_cdna4_permlane32_swap_b32_words(0xFE),
       expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane32SwapB32E32Hi3", 0xFF, 90, make_cdna4_permlane32_swap_b32_words(0xFF),
       expected_cdna3_permlane32_swap_sequence()},
      {"MfmaF32_16x16x32Bf16", 0x1A7, cdna4::kVMfmaF3216x16x32Bf16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3216x16x32Bf16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3216x16x16Bf16Vop3pMfma)},
      {"MfmaF32_32x32x16Bf16", 0x1A7, cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3232x32x8Bf16Vop3pMfma)},
      {"MfmaF32_16x16x32F16", cdna4::encoding::kVop3p, cdna4::kVMfmaF3216x16x32F16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3216x16x32F16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3216x16x16F16Vop3pMfma)},
      {"MfmaF32_32x32x16F16", cdna4::encoding::kVop3p, cdna4::kVMfmaF3232x32x16F16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3232x32x16F16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3232x32x8F16Vop3pMfma)},
      {"MfmaF32_16x16x32F16AccumVgpr", cdna4::encoding::kVop3p,
       cdna4::kVMfmaF3216x16x32F16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3216x16x32F16Vop3pMfma, 0, 256, 260, 272),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3216x16x16F16Vop3pMfma, 272)},
      {"MfmaF32_32x32x16F16AccumVgpr", cdna4::encoding::kVop3p,
       cdna4::kVMfmaF3232x32x16F16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3232x32x16F16Vop3pMfma, 0, 256, 260, 272),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3232x32x8F16Vop3pMfma, 272)},
      {"DsReadB64TrB16", cdna4::encoding::kDsHi3, cdna4::kDsReadB64TrB16Ds,
       make_cdna4_ds_read_b64_tr_b16_words(), expected_cdna3_ds_read_b64_tr_b16_sequence()},
      {"DsReadB64TrB16Acc", cdna4::encoding::kDsHi7, cdna4::kDsReadB64TrB16Ds,
       make_cdna4_ds_read_b64_tr_b16_words(/*acc=*/true),
       expected_cdna3_ds_read_b64_tr_b16_sequence(/*acc_dst=*/true)},
      {"BufferLoadDwordLds", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordMubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordMubuf),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordMubuf, cdna3::kDsWriteB32Ds)},
      {"BufferLoadDwordx3Lds", 0x1C0, 22, make_cdna4_buffer_load_lds_words(22),
       expected_cdna3_buffer_load_lds_sequence(22, 222)},
      {"BufferLoadDwordx4Lds", 0x1C0, 23, make_cdna4_buffer_load_lds_words(23),
       expected_cdna3_buffer_load_lds_sequence(23, 223)},
  };
}

bool has_cdna4_to_cdna3_semantic_rule(uint16_t encoding_id, uint16_t opcode) {
  for (const auto &rule : rocjitsu::semantic_expand_rules_cdna4_to_cdna3()) {
    if (rule.src_encoding_id == encoding_id && rule.src_opcode == opcode)
      return true;
  }
  return false;
}

bool has_cdna4_to_cdna3_semantic_rule_case(uint16_t encoding_id, uint16_t opcode) {
  for (const auto &test_case : cdna4_to_cdna3_semantic_rule_cases()) {
    if (test_case.encoding_id == encoding_id && test_case.opcode == opcode)
      return true;
  }
  return false;
}

void expect_field_matches(uint16_t expected, uint16_t actual, std::string_view field_name) {
  if (expected != kAnyExpectedField) {
    EXPECT_EQ(actual, expected) << field_name;
  }
}

void expect_cdna3_instruction_matches(const rocjitsu::Instruction &inst,
                                      const ExpectedCdna3Inst &expected) {
  const uint32_t *raw = inst.raw_encoding();
  ASSERT_NE(raw, nullptr);

  switch (expected.kind) {
  case ExpectedCdna3Kind::Vop3: {
    rocjitsu::cdna3::Vop3MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x34u);
    EXPECT_EQ(actual.op, expected.op);
    expect_field_matches(expected.vdst, static_cast<uint16_t>(actual.vdst), "vdst");
    expect_field_matches(expected.src0, static_cast<uint16_t>(actual.src0), "src0");
    expect_field_matches(expected.src1, static_cast<uint16_t>(actual.src1), "src1");
    expect_field_matches(expected.src2, static_cast<uint16_t>(actual.src2), "src2");
    break;
  }
  case ExpectedCdna3Kind::Vop3p: {
    rocjitsu::cdna3::Vop3pMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x1A7u);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Vop2: {
    rocjitsu::cdna3::Vop2MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0u);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Vop1: {
    rocjitsu::cdna3::Vop1MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Sop1: {
    rocjitsu::cdna3::Sop1MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x17Du);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Vop3pMfma: {
    rocjitsu::cdna3::Vop3pMfmaMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x1A7u);
    EXPECT_EQ(actual.op, expected.op);
    expect_field_matches(expected.vdst, static_cast<uint16_t>(actual.vdst), "vdst");
    expect_field_matches(expected.acc_cd, static_cast<uint16_t>(actual.acc_cd), "acc_cd");
    expect_field_matches(expected.src0, static_cast<uint16_t>(actual.src0), "src0");
    expect_field_matches(expected.src1, static_cast<uint16_t>(actual.src1), "src1");
    expect_field_matches(expected.src2, static_cast<uint16_t>(actual.src2), "src2");
    break;
  }
  case ExpectedCdna3Kind::Ds: {
    rocjitsu::cdna3::DsMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x36u);
    EXPECT_EQ(actual.op, expected.op);
    expect_field_matches(expected.vdst, static_cast<uint16_t>(actual.vdst), "vdst");
    break;
  }
  case ExpectedCdna3Kind::Mubuf: {
    rocjitsu::cdna3::MubufMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x38u);
    EXPECT_EQ(actual.op, expected.op);
    EXPECT_EQ(actual.lds, 0u);
    break;
  }
  case ExpectedCdna3Kind::Sopp: {
    rocjitsu::cdna3::SoppMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x17Fu);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  }
}

void expect_cdna3_translated_text_matches(const rocjitsu::Section &text,
                                          const std::vector<ExpectedCdna3Inst> &expected) {
  ASSERT_EQ(text.size() % sizeof(uint32_t), 0u);
  ASSERT_GT(text.size(), 0u);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);

  const auto *words = reinterpret_cast<const uint32_t *>(text.data());
  const size_t word_count = text.size() / sizeof(uint32_t);
  std::vector<std::unique_ptr<rocjitsu::Instruction>> actual;
  for (size_t pc = 0; pc < word_count;) {
    SCOPED_TRACE(pc);
    auto inst = std::unique_ptr<rocjitsu::Instruction>(decoder->decode(&words[pc]));
    ASSERT_NE(inst, nullptr);
    ASSERT_GT(inst->size(), 0u);
    ASSERT_EQ(inst->size() % sizeof(uint32_t), 0u);
    ASSERT_LE(pc + inst->size() / sizeof(uint32_t), word_count);
    pc += inst->size() / sizeof(uint32_t);
    actual.push_back(std::move(inst));
  }

  ASSERT_FALSE(expected.empty());
  ASSERT_EQ(expected.back().kind, ExpectedCdna3Kind::Sopp);
  ASSERT_EQ(expected.back().op, 2u) << "legacy cave fixture must end in s_branch fallthrough";

  // Compact CDNA relocation emits the semantic body directly into .text, so
  // the old appended-cave fixture's final return branch is absent. Some e32
  // fixture encodings occupy a two-dword MachineInst container; the second
  // source dword remains independently reachable after the semantic prefix and
  // is intentionally outside this rule-level assertion.
  const size_t translated_instruction_count = expected.size() - 1;
  ASSERT_GE(actual.size(), translated_instruction_count);
  for (size_t i = 0; i < translated_instruction_count; ++i) {
    SCOPED_TRACE(i);
    expect_cdna3_instruction_matches(*actual[i], expected[i]);
  }
}

void expect_cdna3_translated_descriptor_vgprs_at_least(const std::vector<uint8_t> &image,
                                                       uint32_t expected_minimum) {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto infos = parser.translate_image(image, translated.text_sections()[0]->sectionOffset(),
                                            translated.text_sections()[0]->size(),
                                            rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(infos.size(), 1u);
  EXPECT_GE(infos[0].target_vgpr_count, expected_minimum);
}

uint32_t build_s_getpc_b64(uint16_t sdst, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rocjitsu::build_sop1_encoding(arch, 0x47, sdst, 0);
  default:
    return rocjitsu::build_sop1_encoding(arch, 0x1c, sdst, 0);
  }
}

uint32_t build_s_setpc_b64(uint16_t ssrc0, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rocjitsu::build_sop1_encoding(arch, 0x48, 0, ssrc0);
  default:
    return rocjitsu::build_sop1_encoding(arch, 0x1d, 0, ssrc0);
  }
}

uint32_t build_s_swappc_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rocjitsu::build_sop1_encoding(arch, 0x49, sdst, ssrc0);
  default:
    return rocjitsu::build_sop1_encoding(arch, 0x1e, sdst, ssrc0);
  }
}

uint32_t build_s_call_b64(uint16_t sdst, int16_t simm16) {
  return cdna4::build_sopk(cdna4::kSCallB64Sopk, {.simm16 = static_cast<uint16_t>(simm16),
                                                  .sdst = static_cast<uint8_t>(sdst)})[0];
}

uint32_t build_s_trap(uint16_t simm16) {
  // CDNA1-4 encode S_TRAP at SOPP opcode 0x12. This helper is intentionally
  // local to the CDNA4->CDNA3 tests below; RDNA3+ uses a different SOPP opcode.
  return cdna4::build_sopp(cdna4::kSTrapSopp, {.simm16 = simm16})[0];
}

uint32_t build_s_add_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return cdna4::build_sop2(cdna4::kSAddU32Sop2, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                                 .ssrc1 = static_cast<uint8_t>(ssrc1),
                                                 .sdst = static_cast<uint8_t>(sdst)})[0];
}

uint32_t build_s_addc_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return cdna4::build_sop2(cdna4::kSAddcU32Sop2, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                                  .sdst = static_cast<uint8_t>(sdst)})[0];
}

std::vector<std::unique_ptr<rocjitsu::Instruction>>
decode_text_instructions(const rocjitsu::Section &text, rj_code_arch_t arch) {
  std::vector<std::unique_ptr<rocjitsu::Instruction>> decoded;
  auto decoder = rocjitsu::Decoder::create(arch);
  if (!decoder)
    return decoded;

  const auto *words = reinterpret_cast<const rj_code_binary_inst_t *>(text.data());
  const size_t word_count = text.size() / sizeof(rj_code_binary_inst_t);
  size_t word_offset = 0;
  while (word_offset < word_count) {
    std::unique_ptr<rocjitsu::Instruction> inst(
        decoder->decode(words + word_offset, word_offset * sizeof(rj_code_binary_inst_t)));
    if (!inst)
      break;
    word_offset += static_cast<size_t>(inst->size()) / sizeof(rj_code_binary_inst_t);
    decoded.push_back(std::move(inst));
  }
  return decoded;
}

// --- Synthetic BinaryTranslator integration tests ---
TEST(BinaryTranslatorE2E, TranslatesMultiKernelCodeObject) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_two_kernel_descriptors();
  rocjitsu::AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());
  const auto *text = co.text_sections()[0];
  const auto *rodata = rocjitsu::find_section(co, ".rodata");
  ASSERT_NE(rodata, nullptr);

  rocjitsu::KernelDescriptorTranslator original_parser(ROCJITSU_CODE_ARCH_CDNA4,
                                                       ROCJITSU_CODE_ARCH_RDNA4);
  const auto original_infos = original_parser.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(original_infos.size(), 2u);

  std::vector<uint64_t> original_entries;
  std::vector<uint64_t> original_descriptor_offsets;
  for (const auto &info : original_infos) {
    original_entries.push_back(info.entry_text_offset);
    original_descriptor_offsets.push_back(info.descriptor_file_offset);
  }
  std::ranges::sort(original_entries);
  std::ranges::sort(original_descriptor_offsets);
  EXPECT_EQ(original_entries, (std::vector<uint64_t>{0, sizeof(uint32_t)}));
  EXPECT_EQ(original_descriptor_offsets,
            (std::vector<uint64_t>{rodata->sectionOffset(), rodata->sectionOffset() + sizeof(KD)}));

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.ok());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_EQ(translated.text_sections()[0]->size(), text->size());
  EXPECT_EQ(rocjitsu::find_section(translated, ".rj_translations"), nullptr)
      << "this fixture should exercise multi-kernel descriptor handling without code caves";

  const auto *translated_header =
      reinterpret_cast<const rocjitsu::Elf64_Ehdr *>(result.elf_bytes.data());
  EXPECT_EQ(translated_header->e_flags & rocjitsu::EF_AMDGPU_MACH,
            rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200);

  rocjitsu::KernelDescriptorTranslator translated_parser(ROCJITSU_CODE_ARCH_RDNA4,
                                                         ROCJITSU_CODE_ARCH_RDNA4);
  const auto translated_infos = translated_parser.translate_image(
      result.elf_bytes, translated.text_sections()[0]->sectionOffset(),
      translated.text_sections()[0]->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translated_infos.size(), 2u);

  std::vector<uint64_t> translated_entries;
  std::vector<uint64_t> translated_descriptor_offsets;
  for (const auto &info : translated_infos) {
    translated_entries.push_back(info.entry_text_offset);
    translated_descriptor_offsets.push_back(info.descriptor_file_offset);
  }
  std::ranges::sort(translated_entries);
  std::ranges::sort(translated_descriptor_offsets);
  EXPECT_EQ(translated_entries, (std::vector<uint64_t>{0, sizeof(uint32_t)}));
  EXPECT_EQ(translated_descriptor_offsets, original_descriptor_offsets);
}

TEST(BinaryTranslatorE2E, DuplicatesSharedReachableBlocksPerKernel) {
  constexpr uint32_t kCdna4SEndpgm = cdna4::build_sopp(cdna4::kSEndpgmSopp)[0];
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // kernel0: 0x00 -> helper 0x08.
      rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4), // kernel1: 0x04 -> helper 0x08.
      kCdna4SEndpgm,                                         // Shared source helper.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_two_kernel_descriptors(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *text = translated.text_sections()[0];

  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto infos = parser.translate_image(result.elf_bytes, text->sectionOffset(), text->size(),
                                            rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(infos.size(), 2u);
  std::vector<uint64_t> translated_entries;
  for (const auto &info : infos)
    translated_entries.push_back(info.entry_text_offset);
  std::ranges::sort(translated_entries);
  ASSERT_EQ(translated_entries[0], 0u);
  ASSERT_GT(translated_entries[1], 2 * sizeof(uint32_t));
  ASSERT_LE(translated_entries[1] + 2 * sizeof(uint32_t), text->size());

  const auto *target_words = reinterpret_cast<const uint32_t *>(text->data());
  const uint64_t second_entry_word = translated_entries[1] / sizeof(uint32_t);
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], kCdna4SEndpgm);
  EXPECT_EQ(target_words[second_entry_word], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[second_entry_word + 1], kCdna4SEndpgm);
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3SemanticExpandRulesHaveTranslationFixtures) {
  const auto test_cases = cdna4_to_cdna3_semantic_rule_cases();
  const auto rules = rocjitsu::semantic_expand_rules_cdna4_to_cdna3();

  for (const auto &rule : rules) {
    EXPECT_TRUE(has_cdna4_to_cdna3_semantic_rule_case(rule.src_encoding_id, rule.src_opcode))
        << "missing fixture for CDNA4->CDNA3 semantic rule encoding=0x" << std::hex
        << rule.src_encoding_id << " opcode=" << rule.src_opcode << std::dec;
  }
  for (const auto &test_case : test_cases) {
    EXPECT_TRUE(has_cdna4_to_cdna3_semantic_rule(test_case.encoding_id, test_case.opcode))
        << "test fixture has no CDNA4->CDNA3 semantic rule: " << test_case.name;
  }
}

class Cdna4ToCdna3SemanticRuleTranslationTest
    : public ::testing::TestWithParam<Cdna4ToCdna3SemanticRuleCase> {};

TEST_P(Cdna4ToCdna3SemanticRuleTranslationTest, TranslatesSingleInstruction) {
  const auto &test_case = GetParam();
  SCOPED_TRACE(test_case.name);

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());

  const auto *source_text = source_layout.text_sections()[0];
  ASSERT_EQ(source_text->size(), test_case.words.size() * sizeof(uint32_t));
  std::memcpy(image.data() + source_text->sectionOffset(), test_case.words.data(),
              test_case.words.size() * sizeof(uint32_t));

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_EQ(rocjitsu::find_section(translated, ".rj_translations"), nullptr)
      << "CDNA4-to-CDNA3 relocation uses develop's compact .text layout";
  expect_cdna3_translated_text_matches(*translated.text_sections()[0], test_case.expected);
}

INSTANTIATE_TEST_SUITE_P(ImplementedRules, Cdna4ToCdna3SemanticRuleTranslationTest,
                         ::testing::ValuesIn(cdna4_to_cdna3_semantic_rule_cases()),
                         [](const ::testing::TestParamInfo<Cdna4ToCdna3SemanticRuleCase> &info) {
                           return std::string(info.param.name);
                         });

TEST(BinaryTranslatorE2E, Cdna4ToCdna3Bitop3ScratchGrowsDescriptor) {
  constexpr uint16_t kScratchFloor = 120;
  const auto words = make_cdna4_bitop3_words(564, 16);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({words[0], words[1]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = kScratchFloor;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  ASSERT_FALSE(result.elf_bytes.empty());
  // This fixture's LUT needs a two-VGPR scratch run. The conservative liveness
  // floor forces that run above the descriptor's original allocation, so missing
  // require_vgprs() feedback would leave the patched descriptor too small.
  expect_cdna3_translated_descriptor_vgprs_at_least(result.elf_bytes, kScratchFloor + 2);
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3MfmaPartialScratchGrowsDescriptor) {
  constexpr uint16_t kScratchFloor = 120;
  const auto words = make_cdna4_mfma_vgpr_dst_alias_words();
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({words[0], words[1]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = kScratchFloor;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  ASSERT_FALSE(result.elf_bytes.empty());
  // The 16x16x32 F16 lowering uses a four-VGPR partial accumulator when an
  // ordinary destination overlaps the still-needed wide A/B source window.
  expect_cdna3_translated_descriptor_vgprs_at_least(result.elf_bytes, kScratchFloor + 4);
}

TEST(BinaryTranslatorE2E, RelocatedKernelCompactsReachableBodyAndPatchesBranches) {
  constexpr uint32_t kCdna4SEndpgm = cdna4::build_sopp(cdna4::kSEndpgmSopp)[0];
  constexpr uint32_t kCdna4SCbranchScc1ToSourceTarget =
      cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = 4})[0];
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> source 0x0c.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x04 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x08 unreachable.
      kCdna4SCbranchScc1ToSourceTarget,                      // 0x0c -> 0x20, else 0x10.
      rocjitsu::build_s_branch(4, ROCJITSU_CODE_ARCH_CDNA4), // 0x10 -> source 0x24.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x14 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x18 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x1c unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x20 conditional target.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x24 fallthrough-branch target.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  // This kernel does not use the kernarg-preload compatibility entry path, so
  // relocation emits only the reachable CFG body. Source gaps disappear and
  // branch immediates are patched against the compact target layout.
  const std::vector<uint32_t> expected = {
      rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3),
      cdna3::build_sopp(cdna3::kSCbranchScc1Sopp, {.simm16 = 1})[0],
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA3),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3),
      kCdna4SEndpgm,
  };
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(target_words[i], expected[i]);
  }
}

TEST(BinaryTranslatorE2E, RelocatedKernelCompactsReachableBlocksAfterEntry) {
  std::vector<uint32_t> words(74, rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  words[0] = rocjitsu::build_s_branch(63, ROCJITSU_CODE_ARCH_CDNA4); // 0x00 -> 0x100.
  words[64] = rocjitsu::build_s_branch(7, ROCJITSU_CODE_ARCH_CDNA4); // 0x100 -> 0x120.
  words[72] = rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);    // Reachable target.

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  const auto *source_rodata = rocjitsu::find_section(source_layout, ".rodata");
  ASSERT_NE(source_rodata, nullptr);
  ASSERT_GE(source_rodata->size(), sizeof(rocr::llvm::amdhsa::kernel_descriptor_t));
  auto source_kd =
      rocjitsu::read_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd.kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  rocjitsu::write_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset(),
                                             source_kd);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *text = translated.text_sections()[0];
  ASSERT_GE(text->size(), words.size() * sizeof(uint32_t));

  const auto *target_words = reinterpret_cast<const uint32_t *>(text->data());
  // The synthesized preload launch window occupies words 0 and 64. The compact
  // relocated body starts after that protected window, and the source 0x120
  // target lands immediately after the source 0x100 branch instead of remaining
  // at the original word 72.
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(64, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[64], rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[65], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[66], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[67], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[72], rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, RewritesRecoveredSetpcTargetAfterRelocation) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 20;
  constexpr uint32_t kRelocatedGetpcDelta = 16;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),      // 0x04.
      kOriginalGetpcDelta,                                     // 0x08.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),    // 0x10.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),      // 0x14 unreachable.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),      // 0x18 target.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  // The recovered source target is 0x18, but the unreachable 0x14 word is not
  // emitted in the compact body. The old PC builder must therefore be rewritten
  // from getpc+20 to getpc+16 while preserving the indirect setpc consumer.
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand));
  EXPECT_EQ(target_words[2], kRelocatedGetpcDelta);
  EXPECT_EQ(target_words[3], build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0));
  EXPECT_EQ(target_words[4], build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[5], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[6], rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, RewritesRecoveredSwappcTargetAfterRelocation) {
  constexpr uint16_t kPcSreg = 10;
  constexpr uint16_t kReturnSreg = 20;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 24;
  constexpr uint32_t kRelocatedGetpcDelta = 20;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),               // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),                 // 0x04.
      kOriginalGetpcDelta,                                                // 0x08.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),            // 0x0c.
      build_s_swappc_b64(kReturnSreg, kPcSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x10.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                 // 0x14 fallthrough.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                 // 0x18 unreachable.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                 // 0x1c target.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand));
  EXPECT_EQ(target_words[2], kRelocatedGetpcDelta);
  EXPECT_EQ(target_words[3], build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0));
  EXPECT_EQ(target_words[4], build_s_swappc_b64(kReturnSreg, kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[5], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[6], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[7], rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, TranslatesDirectSCallWithSetpcReturn) {
  constexpr uint16_t kReturnSreg = 30;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00 -> call block at 0x0c.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x04 unreachable gap.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x08 unreachable gap.
      build_s_call_b64(kReturnSreg, 2),                         // 0x0c -> callee at 0x18.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),       // 0x10 call continuation.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x14 unreachable gap.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x18 callee return.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_call_b64(kReturnSreg, 1))
      << "the direct call target must be recomputed after unreachable source gaps are compacted";
  EXPECT_EQ(target_words[2], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[3], build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA3));

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_GE(decoded.size(), 4u);
  ASSERT_EQ(decoded[1]->mnemonic(), "s_call_b64");
  ASSERT_TRUE(decoded[1]->branch_offset_bytes().has_value());
  EXPECT_EQ(*decoded[1]->branch_offset_bytes(), 4)
      << "translated call should branch from word 1 to the relocated return block at word 3";
}

TEST(BinaryTranslatorE2E, TranslatesDirectSCallWhenCalleeBranchesToSetpcReturn) {
  constexpr uint16_t kReturnSreg = 30;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00 -> call block at 0x0c.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x04 unreachable gap.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x08 unreachable gap.
      build_s_call_b64(kReturnSreg, 2),                         // 0x0c -> callee at 0x18.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),       // 0x10 call continuation.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x14 unreachable gap.
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),    // 0x18 callee -> return.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x1c unreachable gap.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x20 callee return.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_call_b64(kReturnSreg, 1));
  EXPECT_EQ(target_words[2], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[3], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[4], build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, TranslatesSwappcCallWhenCalleeSetpcBranchesToReturn) {
  constexpr uint16_t kCallTargetSreg = 10;
  constexpr uint16_t kReturnTargetSreg = 12;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalCallTargetDelta = 28;
  constexpr uint32_t kOriginalReturnTargetDelta = 20;
  constexpr uint32_t kRelocatedCallTargetDelta = 20;
  constexpr uint32_t kRelocatedReturnTargetDelta = 16;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kCallTargetSreg, ROCJITSU_CODE_ARCH_CDNA4),               // 0x00.
      build_s_add_u32(kCallTargetSreg, kCallTargetSreg, kLiteralOperand),         // 0x04.
      kOriginalCallTargetDelta,                                                   // 0x08.
      build_s_addc_u32(kCallTargetSreg + 1, kCallTargetSreg + 1, kInlineInt0),    // 0x0c.
      build_s_swappc_b64(kReturnSreg, kCallTargetSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x10.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                     // 0x14 continuation.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                     // 0x18 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                     // 0x1c unreachable.
      build_s_getpc_b64(kReturnTargetSreg, ROCJITSU_CODE_ARCH_CDNA4),         // 0x20 callee.
      build_s_add_u32(kReturnTargetSreg, kReturnTargetSreg, kLiteralOperand), // 0x24.
      kOriginalReturnTargetDelta,                                             // 0x28.
      build_s_addc_u32(kReturnTargetSreg + 1, kReturnTargetSreg + 1, kInlineInt0), // 0x2c.
      build_s_setpc_b64(kReturnTargetSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x30 -> return.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),             // 0x34 unreachable.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4),       // 0x38 return.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kCallTargetSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[2], kRelocatedCallTargetDelta)
      << "the swappc call target should relocate to the compact callee body";
  EXPECT_EQ(target_words[4],
            build_s_swappc_b64(kReturnSreg, kCallTargetSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[5], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[6], build_s_getpc_b64(kReturnTargetSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[8], kRelocatedReturnTargetDelta)
      << "the callee's recovered setpc edge is what reaches the return block";
  EXPECT_EQ(target_words[10], build_s_setpc_b64(kReturnTargetSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[11], build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, RewritesOneRecoveredBuilderUsedByTwoSetpcConsumers) {
  constexpr uint16_t kPcSreg = 12;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 28;
  constexpr uint32_t kRelocatedGetpcDelta = 24;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),            // 0x04.
      kOriginalGetpcDelta,                                           // 0x08.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),       // 0x0c.
      cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = 1})[0], // 0x10 -> consumer.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x14 first consumer.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x18 carried consumer.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x1c unreachable gap.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x20 shared target.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand));
  EXPECT_EQ(target_words[2], kRelocatedGetpcDelta);
  EXPECT_EQ(target_words[3], build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0));
  EXPECT_EQ(target_words[4], cdna3::build_sopp(cdna3::kSCbranchScc1Sopp, {.simm16 = 1})[0]);
  EXPECT_EQ(target_words[5], build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[6], build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[7], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, RelocatesDirectCallReturnAcrossShiftedOffsets) {
  constexpr uint16_t kReturnSreg = 28;
  std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(4, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00 -> call block at 0x14.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x04 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x08 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x0c unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x10 unreachable.
      build_s_call_b64(kReturnSreg, 6),                         // 0x14 -> callee at 0x30.
      rocjitsu::build_s_branch(6, ROCJITSU_CODE_ARCH_CDNA4),    // 0x18 continuation -> 0x34.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x1c unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x20 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x24 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x28 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x2c unreachable.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x30 callee return.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),       // 0x34 final continuation.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_call_b64(kReturnSreg, 1))
      << "source call target 0x30 should relocate to compact word 3";
  EXPECT_EQ(target_words[2], rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA3))
      << "the call continuation branch should relocate to compact word 4";
  EXPECT_EQ(target_words[3], build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[4], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_GE(decoded.size(), 5u);
  ASSERT_TRUE(decoded[1]->branch_offset_bytes().has_value());
  EXPECT_EQ(*decoded[1]->branch_offset_bytes(), 4);
}

TEST(BinaryTranslatorE2E, TrapTerminatesCfgBeforeFollowingFunction) {
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> trap block.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x04 unreachable gap.
      build_s_trap(2),                                       // 0x08 terminates.
      build_s_setpc_b64(30, ROCJITSU_CODE_ARCH_CDNA4),       // 0x0c next function body.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x10 unreachable.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_GE(decoded.size(), 2u);
  EXPECT_EQ(decoded[0]->mnemonic(), "s_branch");
  EXPECT_EQ(decoded[1]->mnemonic(), "s_trap");
  // The real regression this covers is a bogus CFG fallthrough from S_TRAP into
  // the following ELF function/padding bytes. If that edge is present, the
  // unrecovered S_SETPC_B64 below the trap becomes reachable and translation
  // fails with an indirect-branch diagnostic.
  EXPECT_TRUE(std::none_of(decoded.begin(), decoded.end(),
                           [](const auto &inst) { return inst->mnemonic() == "s_setpc_b64"; }));
}

TEST(BinaryTranslatorE2E, RejectsUnrecoveredIndirectBranchInstructions) {
  struct Case {
    const char *name;
    std::vector<uint32_t> words;
    const char *mnemonic;
  };

  const std::array<Case, 3> cases = {{
      {"SetpcS0", {0xBE801D00u, 0x00000000u}, "s_setpc_b64"},
      {"SetpcS30WithoutCall",
       {build_s_setpc_b64(30, ROCJITSU_CODE_ARCH_CDNA4),
        rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4)},
       "s_setpc_b64"},
      {"Swappc", {0xBE801E00u, 0x00000000u}, "s_swappc_b64"},
  }};

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(test_case.words);
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
    auto result = translator.translate(source);

    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::has_error_containing(
        result, rocjitsu::DiagnosticKind::Legalization,
        "indirect branch or call target recovery is not implemented"));
    const auto diagnostic =
        std::find_if(result.diagnostics.begin(), result.diagnostics.end(),
                     [&](const auto &d) { return d.mnemonic == test_case.mnemonic; });
    EXPECT_NE(diagnostic, result.diagnostics.end());
  }
}

TEST(BinaryTranslatorE2E, RejectsDirectBranchTargetBeforeText) {
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(-2, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> -0x04.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(
      rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::Legalization,
                                     "direct branch target is outside the source .text range"));
}

TEST(BinaryTranslatorE2E, RejectsDirectBranchTargetAbsentFromRelocatedBody) {
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> .text end.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "direct branch target is not present in the kernel-local relocated body"));
}

TEST(BinaryTranslatorE2E, RejectsDescriptorPrologueBranchRangeOverflow) {
  constexpr size_t kBodyWordsPastBranchRange = 32769;
  std::vector<uint32_t> words(kBodyWordsPastBranchRange, 0xBF800000u);
  words.push_back(0xBF810000u);

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_workgroup_id_x_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(
      result, rocjitsu::DiagnosticKind::ResourceLimit,
      "kernel descriptor prologue branch range exceeds s_branch simm16"));
}

TEST(BinaryTranslatorE2E, ExpandLegalizationWithoutSemanticRuleFails) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());

  const auto words = make_cdna4_dot2c_unimplemented_expand_words();
  const auto *source_text = source_layout.text_sections()[0];
  ASSERT_EQ(source_text->size(), words.size() * sizeof(uint32_t));
  std::memcpy(image.data() + source_text->sectionOffset(), words.data(),
              words.size() * sizeof(uint32_t));

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  ASSERT_FALSE(result.diagnostics.empty());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &d) {
    return d.kind == rocjitsu::DiagnosticKind::ExpandMissing;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->severity, rocjitsu::DiagnosticSeverity::Error);
  EXPECT_EQ(diagnostic->guest_offset, std::optional<uint64_t>(0));
  EXPECT_FALSE(diagnostic->required_work.empty());
}

TEST(BinaryTranslatorE2E, DebugContinueAfterFailureCollectsMultipleExpandDiagnostics) {
  const auto first = make_cdna4_dot2c_unimplemented_expand_words();
  const auto second = make_cdna4_dot2c_unimplemented_expand_words();
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {first[0], first[1], second[0], second[1]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_continue_after_failure = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image)
      << "continued-failure diagnostics must not emit partially translated code";

  std::vector<uint64_t> expand_offsets;
  for (const auto &diagnostic : result.diagnostics) {
    if (diagnostic.kind == rocjitsu::DiagnosticKind::ExpandMissing &&
        diagnostic.guest_offset.has_value())
      expand_offsets.push_back(*diagnostic.guest_offset);
  }
  EXPECT_EQ(expand_offsets, (std::vector<uint64_t>{0, 8}));
}

TEST(BinaryTranslatorE2E, MatchedSemanticExpandRuleFailureIsDiagnostic) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());

  const auto words = make_cdna4_bitop3_b16_unsupported_op_sel_words();
  const auto *source_text = source_layout.text_sections()[0];
  ASSERT_EQ(source_text->size(), words.size() * sizeof(uint32_t));
  std::memcpy(image.data() + source_text->sectionOffset(), words.data(),
              words.size() * sizeof(uint32_t));

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  ASSERT_FALSE(result.diagnostics.empty());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &d) {
    return d.kind == rocjitsu::DiagnosticKind::ExpandFailed;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->severity, rocjitsu::DiagnosticSeverity::Error);
  EXPECT_EQ(diagnostic->guest_offset, std::optional<uint64_t>(0));
  EXPECT_FALSE(diagnostic->message.empty());
}

TEST(KernelDescriptorTranslator, IgnoresNonAllocExecutableSectionsForEntryRange) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<rocjitsu::Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<rocjitsu::Elf64_Shdr *>(image.data() + ehdr->e_shoff);

  constexpr uint64_t fake_exec_vaddr = 0x9000;
  shdrs[5].sh_flags = rocjitsu::SHF_EXECINSTR;
  shdrs[5].sh_addr = fake_exec_vaddr;
  shdrs[5].sh_size = sizeof(uint32_t);

  auto *kd = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);
  kd->kernel_code_entry_byte_offset =
      static_cast<int64_t>(fake_exec_vaddr) - static_cast<int64_t>(shdrs[2].sh_addr);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations = translator.translate_image(
      image, shdrs[1].sh_offset, shdrs[1].sh_size, rocjitsu::KernelDescriptorTranslationOptions{});
  EXPECT_TRUE(translations.empty())
      << "non-loadable executable sections must not extend valid kernel entry range";
}

TEST(KernelDescriptorTranslator, Gfx1250DescriptorsAreWave32Only) {
  using namespace rocr::llvm::amdhsa;
  using KD = kernel_descriptor_t;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<rocjitsu::Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<rocjitsu::Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *kd = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);

  AMDHSA_BITS_SET(kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 0);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations = translator.translate_image(
      image, shdrs[1].sh_offset, shdrs[1].sh_size, rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_EQ(translations[0].guest_wavefront_size, 32u);
  EXPECT_EQ(translations[0].host_wavefront_size, 32u);
  EXPECT_EQ(translations[0].target_wave_size, 32u);
  EXPECT_FALSE(translations[0].force_wave64);
  EXPECT_TRUE(translations[0].supported);
}

TEST(KernelDescriptorTranslator, Gfx1250ToRdna4InitializesBlockMemoryM0) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<rocjitsu::Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<rocjitsu::Elf64_Shdr *>(image.data() + ehdr->e_shoff);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations = translator.translate_image(
      image, shdrs[1].sh_offset, shdrs[1].sh_size, rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translations.size(), 1u);

  const std::vector<uint32_t> expected = {
      rocjitsu::build_s_mov_b32(rocjitsu::kRdna4M0ForTest, rocjitsu::kInlineMinusOneForTest,
                                ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      0xB9803B01u,
      0,
  };
  ASSERT_GE(translations[0].prologue_words.size(), expected.size());
  const auto suffix = std::span(translations[0].prologue_words).last(expected.size());
  EXPECT_TRUE(std::ranges::equal(suffix, expected))
      << "gfx1250 block operations require M0=-1 at translated kernel entry";
  EXPECT_EQ(expected[0], 0xBEFD00C1u) << "RDNA4 s_mov_b32 m0, -1 encoding changed";
  EXPECT_TRUE(translations[0].supported);
}

TEST(CodeObjectPatcher, Gfx1250ToRdna4PatchSetsWave32DescriptorBit) {
  using namespace rocr::llvm::amdhsa;
  using KD = kernel_descriptor_t;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<rocjitsu::Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<rocjitsu::Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *kd = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);

  AMDHSA_BITS_SET(kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 0);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations = translator.translate_image(
      image, shdrs[1].sh_offset, shdrs[1].sh_size, rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translations.size(), 1u);
  ASSERT_EQ(translations[0].target_wave_size, 32u);

  rocjitsu::AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  rocjitsu::CodeObjectPatcher patcher(co);
  ASSERT_TRUE(
      patcher.apply_kernel_descriptor_translation(translations[0], ROCJITSU_CODE_ARCH_RDNA4));

  const auto patched = patcher.emit();
  const auto *patched_ehdr = reinterpret_cast<const rocjitsu::Elf64_Ehdr *>(patched.data());
  const auto *patched_shdrs =
      reinterpret_cast<const rocjitsu::Elf64_Shdr *>(patched.data() + patched_ehdr->e_shoff);
  ASSERT_GE(patched.size(), patched_shdrs[2].sh_offset + sizeof(KD));
  const auto *patched_kd =
      reinterpret_cast<const KD *>(patched.data() + patched_shdrs[2].sh_offset);
  EXPECT_EQ(AMDHSA_BITS_GET(patched_kd->kernel_code_properties,
                            KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
            1u);
}

TEST(KernelDescriptorTranslator, Gfx1250ToRdna4MaterializesKernargPreload) {
  using namespace rocr::llvm::amdhsa;
  using KD = kernel_descriptor_t;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<rocjitsu::Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<rocjitsu::Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *kd = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);

  kd->compute_pgm_rsrc2 = 0;
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 29);
  kd->kernel_code_properties = 0;
  AMDHSA_BITS_SET(kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR,
                  1);
  AMDHSA_BITS_SET(kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
  kd->kernarg_preload = 0;
  AMDHSA_BITS_SET(kd->kernarg_preload, KERNARG_PRELOAD_SPEC_LENGTH, 27);
  AMDHSA_BITS_SET(kd->kernarg_preload, KERNARG_PRELOAD_SPEC_OFFSET, 0);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations = translator.translate_image(
      image, shdrs[1].sh_offset, shdrs[1].sh_size, rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translations.size(), 1u);

  std::vector<uint32_t> expected;
  auto append_smem = [&expected](uint16_t sdata, uint8_t op, uint32_t byte_offset) {
    const auto words = rocjitsu::pack_rdna4_smem_load_for_test(sdata, 0, op, byte_offset);
    expected.insert(expected.end(), words.begin(), words.end());
  };
  append_smem(2, 1, 0);    // s_load_b64  s[2:3],   s[0:1], 0x00
  append_smem(4, 2, 8);    // s_load_b128 s[4:7],   s[0:1], 0x08
  append_smem(8, 4, 24);   // s_load_b512 s[8:23],  s[0:1], 0x18
  append_smem(24, 2, 88);  // s_load_b128 s[24:27], s[0:1], 0x58
  append_smem(28, 0, 104); // s_load_b32  s28,      s[0:1], 0x68
  expected.push_back(rocjitsu::pack_sopp(64, 0));
  expected.push_back(rocjitsu::pack_sopp(71, 0));
  expected.push_back(rocjitsu::build_s_mov_b32(29, 108 + 9, ROCJITSU_CODE_ARCH_RDNA4));
  expected.push_back(
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
  expected.push_back(rocjitsu::build_s_mov_b32(30, 108 + 7, ROCJITSU_CODE_ARCH_RDNA4));
  expected.push_back(
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
  expected.push_back(rocjitsu::build_s_mov_b32(
      rocjitsu::kRdna4M0ForTest, rocjitsu::kInlineMinusOneForTest, ROCJITSU_CODE_ARCH_RDNA4));
  expected.push_back(
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
  expected.push_back(0xB9803B01u);
  expected.push_back(0);

  EXPECT_EQ(translations[0].prologue_words, expected)
      << "gfx1250 kernarg preload must be rebuilt before the translated RDNA4 entry";
  EXPECT_TRUE(translations[0].clears_kernarg_preload)
      << "the redirected RDNA4 descriptor must enter the DBT prologue directly";
  EXPECT_TRUE(translations[0].supported);

  rocjitsu::AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  rocjitsu::CodeObjectPatcher patcher(co);
  ASSERT_TRUE(
      patcher.apply_kernel_descriptor_translation(translations[0], ROCJITSU_CODE_ARCH_RDNA4));
  const auto patched = patcher.emit();
  const auto *patched_ehdr = reinterpret_cast<const rocjitsu::Elf64_Ehdr *>(patched.data());
  const auto *patched_shdrs =
      reinterpret_cast<const rocjitsu::Elf64_Shdr *>(patched.data() + patched_ehdr->e_shoff);
  ASSERT_GE(patched.size(), patched_shdrs[2].sh_offset + sizeof(KD));
  const auto *patched_kd =
      reinterpret_cast<const KD *>(patched.data() + patched_shdrs[2].sh_offset);
  EXPECT_EQ(patched_kd->kernarg_preload, 0u);
}

TEST(KernelDescriptorTranslator, Rdna4Wave32VgprDescriptorFieldIsCapped) {
  using namespace rocr::llvm::amdhsa;
  using KD = kernel_descriptor_t;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<rocjitsu::Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<rocjitsu::Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *kd = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);

  AMDHSA_BITS_SET(kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.minimum_vgprs = 280;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations =
      translator.translate_image(image, shdrs[1].sh_offset, shdrs[1].sh_size, options);

  ASSERT_EQ(translations.size(), 1u);
  EXPECT_EQ(translations[0].target_wave_size, 32u);
  EXPECT_EQ(translations[0].target_vgpr_count, 280u);
  EXPECT_EQ(translations[0].target_vgpr_granulated, 31u)
      << "gfx1201 hangs when the wave32 launch descriptor uses values above 31";
  EXPECT_TRUE(translations[0].supported);
}

TEST(KernelDescriptorTranslator, Gfx1250HighBankOverrideLowersRdna4LaunchVgprs) {
  using namespace rocr::llvm::amdhsa;
  using KD = kernel_descriptor_t;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  auto *ehdr = reinterpret_cast<rocjitsu::Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<rocjitsu::Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *kd = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);

  AMDHSA_BITS_SET(kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 42);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.target_vgpr_count_override = 256;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations =
      translator.translate_image(image, shdrs[1].sh_offset, shdrs[1].sh_size, options);

  ASSERT_EQ(translations.size(), 1u);
  EXPECT_EQ(translations[0].guest_vgpr_count, 688u);
  EXPECT_EQ(translations[0].target_vgpr_count, 256u);
  EXPECT_EQ(translations[0].target_vgpr_allocation_count, 256u);
  EXPECT_EQ(translations[0].target_vgpr_granulated, 31u);
  EXPECT_TRUE(translations[0].supported);
}

TEST(BinaryTranslator, Gfx1250OversizedLdsVirtualizesEntireAllocation) {
  using namespace rocr::llvm::amdhsa;
  using KD = kernel_descriptor_t;

  rocjitsu::gfx1250::VdsMachineInst store{};
  store.encoding = 0x36;
  store.op = rocjitsu::gfx1250::kDsStoreB128Vds;
  store.addr = 4;
  store.data0 = 8;
  store.offset0 = 32;
  rocjitsu::gfx1250::VdsMachineInst load{};
  load.encoding = 0x36;
  load.op = rocjitsu::gfx1250::kDsLoadU8Vds;
  load.addr = 4;
  load.vdst = 12;
  load.offset0 = 64;
  std::array<uint32_t, 5> text_words{};
  std::memcpy(text_words.data(), &store, sizeof(store));
  std::memcpy(text_words.data() + 2, &load, sizeof(load));
  text_words[4] = 0xBFB00000u;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_and_text(
      text_words, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto *ehdr = reinterpret_cast<rocjitsu::Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<rocjitsu::Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *kd = reinterpret_cast<KD *>(image.data() + shdrs[2].sh_offset);
  kd->group_segment_fixed_size = 128u * 1024u;
  kd->kernarg_size = 8;
  AMDHSA_BITS_SET(kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR,
                  1);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA4,
                                        rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201);
  const auto result = translator.translate(source);
  EXPECT_FALSE(rocjitsu::has_error_diagnostic(result.diagnostics));
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *rodata = rocjitsu::find_section(translated, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *translated_kd = reinterpret_cast<const KD *>(rodata->data());
  EXPECT_EQ(translated_kd->group_segment_fixed_size, 0u);
  EXPECT_GT(translated_kd->kernarg_size, kd->kernarg_size);

  const auto *metadata_section = rocjitsu::find_section(translated, ".rocjitsu.lds");
  ASSERT_NE(metadata_section, nullptr);
  const auto metadata = rocjitsu::parse_virtual_lds_metadata(std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()));
  ASSERT_TRUE(metadata.has_value());
  ASSERT_EQ(metadata->size(), 1u);
  EXPECT_EQ((*metadata)[0].static_lds_bytes, 128u * 1024u);
  EXPECT_EQ((*metadata)[0].normal_descriptor_vaddr, (*metadata)[0].virtual_descriptor_vaddr);

  const auto *caves = rocjitsu::find_section(translated, ".rj_translations");
  ASSERT_NE(caves, nullptr);
  const auto *words = reinterpret_cast<const uint32_t *>(caves->data());
  size_t wrapper_stride_loads = 0;
  const std::array<uint32_t, 3> stride_offsets{
      (*metadata)[0].backing_pointer_kernarg_offset + 8u,
      (*metadata)[0].backing_pointer_kernarg_offset + 12u,
      (*metadata)[0].backing_pointer_kernarg_offset + 16u,
  };
  for (size_t i = 0; i + 1 < caves->size() / sizeof(uint32_t); ++i) {
    rocjitsu::rdna4::SmemMachineInst smem{};
    std::memcpy(&smem, words + i, sizeof(smem));
    if (smem.encoding != 0x3D || smem.sdata != (*metadata)[0].virtual_lds_base_sgpr + 2u ||
        std::ranges::find(stride_offsets, smem.ioffset) == stride_offsets.end()) {
      continue;
    }
    // The stride fields live in the kernarg wrapper at s[0:1], not in the
    // allocation whose pointer was loaded into the virtual-LDS base pair.
    EXPECT_EQ(smem.sbase, 0u);
    ++wrapper_stride_loads;
  }
  EXPECT_EQ(wrapper_stride_loads, stride_offsets.size());

  bool found_global_store = false;
  size_t global_store_index = 0;
  uint8_t global_address_pair = 0;
  for (size_t i = 0; i + 2 < caves->size() / sizeof(uint32_t); ++i) {
    rocjitsu::rdna4::VglobalMachineInst global{};
    std::memcpy(&global, words + i, sizeof(global));
    if (global.encoding == 0xEE && global.op == rocjitsu::rdna4::kGlobalStoreB128Vglobal &&
        global.vsrc == store.data0 && global.ioffset == store.offset0) {
      found_global_store = true;
      global_store_index = i;
      global_address_pair = global.vaddr;
      break;
    }
  }
  ASSERT_TRUE(found_global_store);
  ASSERT_GE(global_store_index, 3u);
  EXPECT_NE(global_address_pair, store.addr);
  const uint32_t copy_low = words[global_store_index - 3u];
  const uint32_t clear_high = words[global_store_index - 2u];
  EXPECT_EQ(copy_low & 0x1FFu, 256u + store.addr);
  EXPECT_EQ((copy_low >> 17u) & 0xFFu, global_address_pair);
  EXPECT_EQ(clear_high & 0x1FFu, 128u);
  EXPECT_EQ((clear_high >> 17u) & 0xFFu, global_address_pair + 1u);

  bool found_global_load = false;
  bool found_workgroup_invalidate = false;
  for (size_t i = 0; i + 2 < caves->size() / sizeof(uint32_t); ++i) {
    if (words[i] == 0xEE0AC000u && words[i + 1u] == 0x00040000u &&
        words[i + 2u] == rocjitsu::pack_sopp(rocjitsu::rdna4::kSWaitLoadcntSopp, 0)) {
      found_workgroup_invalidate = true;
    }
    rocjitsu::rdna4::VglobalMachineInst global{};
    std::memcpy(&global, words + i, sizeof(global));
    if (global.encoding == 0xEE && global.op == rocjitsu::rdna4::kGlobalLoadU8Vglobal &&
        global.vdst == load.vdst && global.ioffset == load.offset0 && global.scope == 1) {
      found_global_load = true;
    }
  }
  EXPECT_TRUE(found_workgroup_invalidate);
  EXPECT_TRUE(found_global_load);
}
