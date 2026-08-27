// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translate_test.cpp
/// @brief CPU-only unit tests for the DBT translation pipeline.
///
/// Tests structural properties of translated code objects without requiring a
/// GPU, including ELF rewriting, semantic lowering, and revision translation.
///
/// These tests complement the hardware tests in hsa_translate_test.cpp which
/// verify correctness on real DBT host GPUs.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/binary_translator_internal.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/legalization/gfx1250_b0_to_a0.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/dbt/waitcnt_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/code/rj_code_internal.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "support/elf_test_support.h"
#include "support/translate_test_support.h"
#include "util/data_types.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

using test_support::append_elf_section_for_test;
using test_support::decode_one;
using test_support::enable_workgroup_id_x_sgpr;
using test_support::find_section;
using test_support::has_error_containing;
using test_support::kKernelDescriptorSize;
using test_support::make_large_amdgpu_elf_with_waitcnt_entry;
using test_support::make_minimal_amdgpu_elf_with_descriptor_after_text;
using test_support::make_minimal_amdgpu_elf_with_two_kernel_descriptors;
using test_support::read_elf_array_for_test;
using test_support::read_elf_struct_for_test;
using test_support::read_kernel_descriptor_entry_offset;
using test_support::read_kernel_descriptor_for_test;
using test_support::TestKernelDescriptor;
using test_support::TestRuntimeTextReference;
using test_support::TestRuntimeTextRelocation;
using test_support::write_elf_struct_for_test;
using test_support::write_kernel_descriptor_entry_offset;
using test_support::write_kernel_descriptor_for_test;
using test_support::write_value_for_test;

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.resize(names.size() + name.size() + 1);
  if (!name.empty())
    std::memcpy(names.data() + offset, name.data(), name.size());
  return offset;
}

uint64_t align_up_for_test(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

void write_bytes_for_test(std::vector<uint8_t> &image, uint64_t offset, const void *src,
                          size_t size) {
  assert(offset <= image.size());
  assert(size <= image.size() - offset);
  std::memcpy(image.data() + offset, src, size);
}

constexpr uint64_t kKernargPreloadSkipBytes = 256;

std::vector<uint8_t>
make_minimal_amdgpu_elf_with_text_and_rodata(std::span<const uint8_t> text_bytes) {
  constexpr uint64_t text_offset = 0x100;
  const uint64_t text_size = text_bytes.size();
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

  if (!text_bytes.empty())
    std::memcpy(image.data() + text_offset, text_bytes.data(), text_size);

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

std::vector<uint8_t> make_minimal_amdgpu_elf_with_text_and_rodata() {
  constexpr std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
  return make_minimal_amdgpu_elf_with_text_and_rodata(std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(text_words.data()), sizeof(text_words)));
}

std::vector<uint8_t>
make_minimal_amdgpu_elf_with_text_words_and_rodata(std::span<const uint32_t> text_words) {
  return make_minimal_amdgpu_elf_with_text_and_rodata(std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(text_words.data()), text_words.size_bytes()));
}

std::vector<uint8_t> make_minimal_gfx1250_elf_with_empty_text_and_rodata() {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  assert(shdrs.size() >= 2);

  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  shdrs[1].sh_size = 0;
  write_elf_struct_for_test(image, 0, ehdr);
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
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
  syms[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
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

// reloc_type defaults to a representative non-inert record, which is what the
// r_offset tests want. Naming R_AMDGPU_RELATIVE64 with an addend inside the
// section that follows .text instead exercises the addend shift: the stored
// value is load_bias + r_addend, so a .text that grows past its load alignment
// moves the section and the addend has to move with it.
std::vector<uint8_t>
make_minimal_amdgpu_elf_with_relocation_after_text(bool place_reloc_in_text = false,
                                                   uint32_t reloc_type = R_AMDGPU_ABS64,
                                                   int64_t reloc_addend = 0) {
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
  // By default the relocation place is in .data (safe: DBT shifts it with the
  // moved section). place_reloc_in_text points it inside .text, which DBT cannot
  // remap after relocating instructions and must reject.
  rela.r_offset = place_reloc_in_text ? text_vaddr : data_vaddr;
  rela.r_info = (uint64_t{0} << 32) | reloc_type;
  rela.r_addend = reloc_addend;
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

// Build an ET_DYN object whose .data relocation resolves against a symbol of a
// chosen type, defined either in .text or .data. Supported zero-addend text
// symbols can follow their relocated st_value; section symbols, nonzero
// addends, and unrecognized dynamic relocation types remain unsupported.
// ROCr rejects R_AMDGPU_NONE in target-less dynamic sections, while explicit-target
// sections are skipped for supported code objects and therefore treat NONE as inert.
constexpr std::array kSupportedExplicitSymbolRelocations = {
    R_AMDGPU_ABS32_LO,
    R_AMDGPU_ABS32_HI,
    R_AMDGPU_ABS64,
    R_AMDGPU_ABS32,
};
constexpr uint32_t kUnrecognizedAmdGpuRelocation = 12; // Unassigned in the AMDGPU ELF ABI.
constexpr uint8_t kUnsupportedRuntimeSymbolType = 15;
constexpr std::array kUnsupportedExplicitSymbolRelocations = {
    kUnrecognizedAmdGpuRelocation,
};
std::vector<uint8_t>
make_amdgpu_elf_with_symbol_relocation(uint8_t sym_type, bool defined_in_text, int64_t addend = 0,
                                       uint32_t relocation_type = R_AMDGPU_ABS64) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t data_size = 8;
  constexpr uint64_t load_align = 0x1000;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t data_name = add_elf_name(shstrtab, ".data");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t rela_name = add_elf_name(shstrtab, ".rela.dyn");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t sym_name = add_elf_name(strtab, "target");

  constexpr uint16_t text_index = 1;
  constexpr uint16_t data_index = 2;

  const uint64_t data_offset = text_offset + text_size;
  const uint64_t data_vaddr = text_vaddr + text_size + load_align;
  const uint64_t symtab_offset = align_up_for_test(data_offset + data_size, 8);
  constexpr size_t sym_count = 2; // STN_UNDEF + target
  const uint64_t strtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t rela_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t rela_count = 1;
  const uint64_t shstrtab_offset = rela_offset + rela_count * sizeof(Elf64_Rela);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 7;
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
  ehdr.e_shstrndx = 6;
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

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = sym_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, sym_type);
  syms[1].st_shndx = defined_in_text ? text_index : data_index;
  syms[1].st_value = defined_in_text ? text_vaddr : data_vaddr;
  syms[1].st_size = defined_in_text ? text_size : data_size;
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  Elf64_Rela rela{};
  rela.r_offset = data_vaddr; // place in .data, safely shifted with the section
  rela.r_info = (static_cast<uint64_t>(1) << 32) | relocation_type;
  rela.r_addend = addend;
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

  shdrs[5].sh_name = rela_name;
  shdrs[5].sh_type = SHT_RELA;
  shdrs[5].sh_offset = rela_offset;
  shdrs[5].sh_size = rela_count * sizeof(Elf64_Rela);
  shdrs[5].sh_link = 3; // .symtab
  shdrs[5].sh_addralign = 8;
  shdrs[5].sh_entsize = sizeof(Elf64_Rela);

  shdrs[6].sh_name = shstrtab_name;
  shdrs[6].sh_type = SHT_STRTAB;
  shdrs[6].sh_offset = shstrtab_offset;
  shdrs[6].sh_size = shstrtab.size();
  shdrs[6].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

// Build an ET_DYN object with a symbol-less relocation of the given type in
// .rela.dyn (symbol index 0), placed in .data, with the supplied addend. Used to
// prove the text-symbol guard also catches an R_AMDGPU_RELATIVE64 whose addend
// lands in the source .text virtual-address interval (the loader forms the stored
// value from load_bias + r_addend, which DBT would leave pointing at stale PC).
// text_vaddr is 0x1100 with size 8, so an addend of 0x1100 is in-text and 0x2108
// (the .data vaddr) is out-of-text.
std::vector<uint8_t> make_amdgpu_elf_with_relative_relocation(uint32_t reloc_type, int64_t addend) {
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
  const uint64_t rela_offset = align_up_for_test(data_offset + data_size, 8);
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

  Elf64_Rela rela{};
  rela.r_offset = data_vaddr; // place in .data, safely shifted with the section
  rela.r_info = static_cast<uint64_t>(reloc_type); // symbol index 0, low 32 bits type
  rela.r_addend = addend;
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
  shdrs[3].sh_link = 0; // no symtab needed for symbol-zero relocations
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

TEST(BinaryTranslatorE2E, EmptyTextSameArchIsSuccessfulNoOp) {
  const auto image = make_minimal_gfx1250_elf_with_empty_text_and_rodata();
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.target_id(), ROCJITSU_CODE_TARGET_GFX1250);
  ASSERT_FALSE(source.text_sections().empty());
  ASSERT_EQ(source.text_sections()[0]->size(), 0u);

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.dispatchable());
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_TRUE(result.rewrite_discharge_verified);
  EXPECT_EQ(result.elf_bytes, image);
  const auto warning = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == DiagnosticKind::DataOnly;
  });
  ASSERT_NE(warning, result.diagnostics.end());
  EXPECT_EQ(warning->severity, DiagnosticSeverity::Warning);
  EXPECT_EQ(warning->message,
            "code object has no executable sections, segments, or callable symbols; leaving "
            "unchanged");
}

TEST(BinaryTranslatorE2E, RejectsUnsupportedCrossTargetCdna5TranslationInBothDirections) {
  constexpr std::array<std::pair<uint32_t, uint32_t>, 2> kDirections{{
      {EF_AMDGPU_MACH_AMDGCN_GFX1251, EF_AMDGPU_MACH_AMDGCN_GFX1250},
      {EF_AMDGPU_MACH_AMDGCN_GFX1250, EF_AMDGPU_MACH_AMDGCN_GFX1251},
  }};

  for (const auto &[source_mach, target_mach] : kDirections) {
    auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
    auto header = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    header.e_flags = source_mach;
    write_elf_struct_for_test(image, 0, header);
    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, target_mach);
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(std::ranges::any_of(result.diagnostics, [](const auto &diagnostic) {
      return diagnostic.message == "cross-target CDNA5 translation is unsupported";
    })) << (result.diagnostics.empty() ? "" : result.diagnostics.front().message);
  }
}

TEST(BinaryTranslatorE2E, PreservesGfx1251IdentityForSameTargetDataOnlyObject) {
  auto image = make_minimal_gfx1250_elf_with_empty_text_and_rodata();
  auto header = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1251;
  write_elf_struct_for_test(image, 0, header);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.target_id(), ROCJITSU_CODE_TARGET_GFX1251);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5,
                              EF_AMDGPU_MACH_AMDGCN_GFX1251);
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_EQ(result.elf_bytes, image);
  const auto output_header = read_elf_struct_for_test<Elf64_Ehdr>(result.elf_bytes, 0);
  EXPECT_EQ(output_header.e_flags & EF_AMDGPU_MACH, EF_AMDGPU_MACH_AMDGCN_GFX1251);
}

TEST(BinaryTranslatorE2E, UsesGfx1251DecoderForSameTargetExecutableObject) {
  const std::vector<uint32_t> kText = {
      0xCC4B4004u, // v_pk_add_f64 v[4:7], v[8:11], v[12:15]
      0x1A021908u,
      0xBFB00000u, // s_endpgm
  };
  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text(kText);
  auto header = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1251;
  write_elf_struct_for_test(image, 0, header);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.target_id(), ROCJITSU_CODE_TARGET_GFX1251);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5,
                              EF_AMDGPU_MACH_AMDGCN_GFX1251);
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  const auto output_header = read_elf_struct_for_test<Elf64_Ehdr>(result.elf_bytes, 0);
  EXPECT_EQ(output_header.e_flags & EF_AMDGPU_MACH, EF_AMDGPU_MACH_AMDGCN_GFX1251);
}

TEST(BinaryTranslatorE2E, TruncatedImageFailsBeforeReadingElfHeader) {
  const std::array<uint8_t, sizeof(Elf64_Ehdr) - 1> image{};
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_FALSE(source.is_valid());
  ASSERT_LT(source.image_size(), sizeof(Elf64_Ehdr));

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::ResourceLimit,
                                   "too small to contain an ELF header"));
}

TEST(BinaryTranslatorE2E, EmptyTextCrossArchStillFails) {
  const auto image = make_minimal_gfx1250_elf_with_empty_text_and_rodata();
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::ResourceLimit,
                                   "does not expose a non-empty .text section"));
}

TEST(BinaryTranslatorE2E, EmptyTextSameArchDifferentMachineStillFails) {
  auto image = make_minimal_gfx1250_elf_with_empty_text_and_rodata();
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1200);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.target_id(), ROCJITSU_CODE_TARGET_GFX1200);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_RDNA4,
                              EF_AMDGPU_MACH_AMDGCN_GFX1201);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::ResourceLimit,
                                   "does not expose a non-empty .text section"));
}

TEST(BinaryTranslatorE2E, EmptyTextGfx1250StillRequiresRevisions) {
  const auto image = make_minimal_gfx1250_elf_with_empty_text_and_rodata();
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                   "requires both input and output silicon revisions"));
}

TEST(BinaryTranslatorE2E, EmptyTextGfx1250StillRejectsA0ToB0) {
  const auto image = make_minimal_gfx1250_elf_with_empty_text_and_rodata();
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250A0;
  options.output_revision = ProcessorRevision::Gfx1250B0;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                   "A0-to-B0 translation is not supported"));
}

TEST(BinaryTranslatorE2E, Gfx1250InvalidInstructionIsDiagnosedAndLeavesObjectUnchanged) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto invalid = cdna5::build_vop1(cdna5::kVReadfirstlaneB32Vop1, {.src0 = 255});
  const std::vector<uint32_t> words = {invalid[0], kGfx1250SEndpgm};
  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                   "does not support 32-bit literals"));
}

TEST(BinaryTranslatorE2E, DescriptorlessExecutableTextIsSuccessfulNoOp) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  write_value_for_test<uint32_t>(image, 0x100, 0xbf850004u);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_FALSE(source.text_sections().empty());
  ASSERT_GT(source.text_sections()[0]->size(), 0u);

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.dispatchable());
  EXPECT_EQ(result.elf_bytes, image);
  const auto warning = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == DiagnosticKind::NothingToTranslate;
  });
  ASSERT_NE(warning, result.diagnostics.end());
  EXPECT_EQ(warning->severity, DiagnosticSeverity::Warning);
  EXPECT_NE(warning->message.find("no kernel descriptors"), std::string::npos);
}

TEST(BinaryTranslatorE2E, RewriteDischargeCannotVerifySkippedKernelStubs) {
  const auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  options.skip_failed_kernels = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.dispatchable());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_FALSE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                   "cannot be combined with skip-failed-kernels"));
  EXPECT_TRUE(std::ranges::none_of(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == DiagnosticKind::KernelSkipped;
  }));
}

TEST(BinaryTranslatorE2E, RewriteDischargeReportsUnsupportedCoreProfile) {
  const auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslatorOptions options;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA4, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::ResidualRewrite,
                                   "unavailable for this translation profile"));
}

// A code relocation names a branch destination inside the text being written, so it has to be a
// whole instruction there. An offset equal to the size names the byte one past the end, which
// would still compute a literal and leave it pointing outside `.text`.
TEST(CodeObjectPatcher, ReplaceTextRejectsCodeRelocationTargetPastEndOfText) {
  const std::array<uint32_t, 4> text_words = {0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u};
  const auto bytes = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text_words.data()),
                                              sizeof(text_words));
  const auto attempt = [&](uint64_t target_text_offset) {
    auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
    AmdGpuCodeObject co(image.data(), image.size());
    EXPECT_TRUE(co.is_valid());
    CodeObjectPatcher patcher(co);
    const std::array<PcRelativeTextRelocation, 1> code_relocations = {
        PcRelativeTextRelocation{.target_getpc_offset = 0,
                                 .target_literal_offset = 4,
                                 .target_text_offset = target_text_offset}};
    return patcher.replace_text(bytes, {}, {}, code_relocations);
  };

  EXPECT_FALSE(attempt(sizeof(text_words)))
      << "a target one past the end of .text is not an instruction";
  EXPECT_FALSE(attempt(sizeof(text_words) - 2))
      << "a target without room for a whole instruction word is not an instruction";
  EXPECT_FALSE(attempt(1)) << "an offset interior to the first instruction is not a destination";
  EXPECT_FALSE(attempt(sizeof(uint32_t) + 2))
      << "an offset interior to a later instruction is not a destination";
  EXPECT_TRUE(attempt(sizeof(text_words) - sizeof(uint32_t)))
      << "the last instruction in .text is a legitimate branch destination";
}

TEST(CodeObjectPatcher, ResolvesAllocatedDataPayloadsAndEndpoints) {
  std::array<Elf64_Shdr, 5> sections{};
  sections[0].sh_addr = 0x1000;
  sections[0].sh_size = 0x8;
  sections[1].sh_flags = SHF_ALLOC;
  sections[1].sh_addr = 0x2000;
  sections[1].sh_size = 0x10;
  sections[2].sh_flags = SHF_ALLOC;
  sections[2].sh_addr = 0x2010;
  sections[2].sh_size = 0;
  sections[3].sh_flags = SHF_ALLOC;
  sections[3].sh_addr = 0x2010;
  sections[3].sh_size = 0x8;
  sections[4].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[4].sh_addr = 0x3000;
  sections[4].sh_size = 0x8;

  struct Case {
    uint64_t target;
    size_t section_index;
    uint64_t section_offset;
  };
  for (const Case &test :
       {Case{0x2000, 1, 0}, Case{0x200f, 1, 0xf}, Case{0x2011, 3, 1}, Case{0x2018, 3, 8}}) {
    const auto resolved = resolve_allocated_data_section_address(sections, test.target);
    ASSERT_TRUE(resolved.has_value()) << "target 0x" << std::hex << test.target;
    EXPECT_EQ(resolved->section_index, test.section_index);
    EXPECT_EQ(resolved->section_offset, test.section_offset);
  }
  EXPECT_TRUE(resolve_allocated_data_section_address(sections, 0x2010))
      << "either nonempty section may own their shared boundary";

  std::array<Elf64_Shdr, 1> empty_section{};
  empty_section[0].sh_flags = SHF_ALLOC;
  empty_section[0].sh_addr = 0x4000;
  EXPECT_FALSE(resolve_allocated_data_section_address(empty_section, 0x4000));
  EXPECT_FALSE(resolve_allocated_data_section_address(sections, 0x1000));
  EXPECT_FALSE(resolve_allocated_data_section_address(sections, 0x2019));
  EXPECT_FALSE(resolve_allocated_data_section_address(sections, 0x2100));
  EXPECT_FALSE(resolve_allocated_data_section_address(sections, 0x3000));

  ASSERT_TRUE(resolve_allocated_data_section_address(sections, 0x2010));
  EXPECT_FALSE(resolve_pc_relative_data_section_address(sections, 0x2010, 0x2010, 0x8))
      << "source text takes precedence over a data section ending at the same address";
}

class RewriteDischargeBoundTestInstruction final : public Instruction {
public:
  explicit RewriteDischargeBoundTestInstruction(size_t word_count)
      : Instruction("bounded_decode_test", nullptr) {
    size_ = static_cast<int>(word_count * sizeof(uint32_t));
  }
};

class RewriteDischargeBoundTestDecoder final : public Decoder {
public:
  RewriteDischargeBoundTestDecoder(size_t max_instruction_words, size_t decoded_words,
                                   bool reject = false)
      : max_instruction_words_(max_instruction_words), decoded_words_(decoded_words),
        reject_(reject) {}

  std::size_t max_instruction_words() const override { return max_instruction_words_; }

  DecodeResult decode(const rj_code_binary_inst_t *words,
                      const DecodeErrorEmitter &emit_error) override {
    ++decode_calls;
    observed_words.assign(words, words + max_instruction_words_);
    if (reject_)
      return emit_error.emit() << "test decoder rejected encoding";
    return std::make_unique<RewriteDischargeBoundTestInstruction>(decoded_words_);
  }

  size_t decode_calls = 0;
  std::vector<uint32_t> observed_words;

private:
  size_t max_instruction_words_;
  size_t decoded_words_;
  bool reject_;
};

TEST(BinaryTranslatorInternal, RewriteDischargeDecodeRejectsWidthBeyondDecoderMaximum) {
  RewriteDischargeBoundTestDecoder decoder(/*max_instruction_words=*/2, /*decoded_words=*/3);
  internal::RewriteDischargeInstructionDecoder bounded_decoder(decoder);
  constexpr std::array<uint32_t, 3> words = {0x11111111u, 0x22222222u, 0x33333333u};
  const auto bytes =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words));
  std::unique_ptr<Instruction> instruction;

  EXPECT_EQ(bounded_decoder.decode(bytes, /*source_offset=*/20, instruction),
            internal::RewriteDischargeDecodeStatus::TruncatedInstruction);
  ASSERT_NE(instruction, nullptr);
  EXPECT_EQ(instruction->src_loc(), 20u);
  EXPECT_EQ(decoder.decode_calls, 1u);
  EXPECT_EQ(decoder.observed_words, (std::vector<uint32_t>{0x11111111u, 0x22222222u}));
}

TEST(BinaryTranslatorInternal, RewriteDischargeDecodeRejectsZeroBoundWithoutDecoding) {
  RewriteDischargeBoundTestDecoder decoder(/*max_instruction_words=*/0, /*decoded_words=*/1);
  internal::RewriteDischargeInstructionDecoder bounded_decoder(decoder);
  constexpr std::array<uint32_t, 1> words = {0x11111111u};
  const auto bytes =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words));
  std::unique_ptr<Instruction> instruction =
      std::make_unique<RewriteDischargeBoundTestInstruction>(1);

  EXPECT_FALSE(bounded_decoder.has_valid_lookahead_bound());
  EXPECT_EQ(bounded_decoder.decode(bytes, /*source_offset=*/0, instruction),
            internal::RewriteDischargeDecodeStatus::InvalidLookaheadBound);
  EXPECT_EQ(instruction, nullptr);
  EXPECT_EQ(decoder.decode_calls, 0u);
}

TEST(BinaryTranslatorInternal, RewriteDischargeDecodeReportsRejectedEncoding) {
  RewriteDischargeBoundTestDecoder decoder(/*max_instruction_words=*/1, /*decoded_words=*/1,
                                           /*reject=*/true);
  internal::RewriteDischargeInstructionDecoder bounded_decoder(decoder);
  constexpr std::array<uint32_t, 1> words = {0x11111111u};
  const auto bytes =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words));
  std::unique_ptr<Instruction> instruction;
  util::StringDiagnostic decode_error;

  EXPECT_EQ(bounded_decoder.decode(bytes, /*source_offset=*/8, instruction, decode_error.emitter()),
            internal::RewriteDischargeDecodeStatus::InvalidEncoding);
  EXPECT_EQ(instruction, nullptr);
  EXPECT_EQ(decode_error.message(), "test decoder rejected encoding");
}

TEST(BinaryTranslatorInternal, RewriteDischargeDecodeClearsUnusedWordsAcrossCalls) {
  RewriteDischargeBoundTestDecoder decoder(/*max_instruction_words=*/2, /*decoded_words=*/1);
  internal::RewriteDischargeInstructionDecoder bounded_decoder(decoder);
  constexpr std::array<uint32_t, 2> full_words = {0x11111111u, 0x22222222u};
  const auto full_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(full_words.data()), sizeof(full_words));
  std::unique_ptr<Instruction> instruction;

  ASSERT_EQ(bounded_decoder.decode(full_bytes, /*source_offset=*/0, instruction),
            internal::RewriteDischargeDecodeStatus::Success);
  ASSERT_EQ(decoder.observed_words, (std::vector<uint32_t>{0x11111111u, 0x22222222u}));

  constexpr std::array<uint32_t, 1> short_words = {0x33333333u};
  const auto short_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(short_words.data()), sizeof(short_words));
  EXPECT_EQ(bounded_decoder.decode(short_bytes, /*source_offset=*/4, instruction),
            internal::RewriteDischargeDecodeStatus::Success);
  EXPECT_EQ(decoder.observed_words, (std::vector<uint32_t>{0x33333333u, 0u}));
}

TEST(BinaryTranslatorE2E, RewriteDischargeRejectsIdentityOutputWithResidualTrigger) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  constexpr auto source_clause = cdna5::build_sopp(cdna5::kSClauseSopp, {.simm16 = 4});
  write_value_for_test<uint32_t>(image, 0x100, source_clause[0]);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  const auto residual = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == DiagnosticKind::ResidualRewrite;
  });
  ASSERT_NE(residual, result.diagnostics.end());
  EXPECT_EQ(residual->severity, DiagnosticSeverity::Error);
  EXPECT_EQ(residual->output_offset, std::optional<uint64_t>(0));
  EXPECT_EQ(residual->mnemonic, "s_clause");
}

TEST(BinaryTranslatorE2E, RewriteDischargeRejectsNonDwordTextTail) {
  constexpr std::array<uint8_t, 7> text = {0x00, 0x00, 0x80, 0xbf, 0xaa, 0xbb, 0xcc};
  for (size_t tail_size = 1; tail_size <= 3; ++tail_size) {
    SCOPED_TRACE(tail_size);
    auto image = make_minimal_amdgpu_elf_with_text_and_rodata(
        std::span<const uint8_t>(text).first(sizeof(uint32_t) + tail_size));
    write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                   EF_AMDGPU_MACH_AMDGCN_GFX1250);
    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.rewrite_discharge_checked);
    EXPECT_FALSE(result.rewrite_discharge_verified);
    EXPECT_TRUE(
        has_error_containing(result, DiagnosticKind::ResidualRewrite, "partial instruction word"));
  }
}

TEST(BinaryTranslatorE2E, RewriteDischargeRejectsTruncatedMultiwordInstruction) {
  constexpr std::array<uint32_t, 4> scaled_wmma = {0xCC350000u, 0x0202954Eu, 0xCC332042u,
                                                   0x050A01CAu};
  for (size_t word_count = 1; word_count < scaled_wmma.size(); ++word_count) {
    SCOPED_TRACE(word_count);
    auto image = make_minimal_amdgpu_elf_with_text_words_and_rodata(
        std::span<const uint32_t>(scaled_wmma).first(word_count));
    write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                   EF_AMDGPU_MACH_AMDGCN_GFX1250);
    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.rewrite_discharge_checked);
    EXPECT_FALSE(result.rewrite_discharge_verified);
    EXPECT_TRUE(std::ranges::any_of(result.diagnostics, [](const auto &diagnostic) {
      return diagnostic.kind == DiagnosticKind::ResidualRewrite;
    }));
  }
}

TEST(BinaryTranslatorE2E, RewriteDischargeRejectsResidualFlatScratchBaseSelector) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  constexpr auto source = cdna5::build_sop1(cdna5::kSMovB64Sop1, {.ssrc0 = 231, .sdst = 10});
  write_value_for_test<uint32_t>(image, 0x100, source[0]);
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(code_object);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  const auto residual = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == DiagnosticKind::ResidualRewrite;
  });
  ASSERT_NE(residual, result.diagnostics.end());
  EXPECT_EQ(residual->severity, DiagnosticSeverity::Error);
  EXPECT_EQ(residual->output_offset, std::optional<uint64_t>(0));
  EXPECT_EQ(residual->mnemonic, "s_mov_b64");
}

TEST(BinaryTranslatorE2E, Gfx1250TranslationIgnoresUnreferencedVisibleTextSymbols) {
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::vector<uint32_t> words = {clear_m0[0], cluster[0], cluster[1], cluster[2],
                                       kGfx1250SEndpgm};

  for (const uint8_t symbol_bind : {kElfSymbolBindGlobal, kElfSymbolBindWeak}) {
    SCOPED_TRACE(symbol_bind == kElfSymbolBindGlobal ? "STB_GLOBAL" : "STB_WEAK");
    auto image = make_minimal_amdgpu_elf_with_descriptor_after_text(words, words.size());
    const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                   EF_AMDGPU_MACH_AMDGCN_GFX1250);
    const auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
    const auto text_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
    const auto symtab_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
    ASSERT_NE(text_it, shdrs.end());
    ASSERT_NE(symtab_it, shdrs.end());

    auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                      symtab_it->sh_size / sizeof(Elf64_Sym));
    ASSERT_EQ(symbols.size(), 3u);
    symbols[2].st_info = elf_symbol_info(symbol_bind, kElfSymbolTypeFunc);
    symbols[2].st_shndx = static_cast<uint16_t>(text_it - shdrs.begin());
    symbols[2].st_value = text_it->sh_addr + sizeof(uint32_t);
    symbols[2].st_size = 4 * sizeof(uint32_t);
    write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                         symbols.size() * sizeof(Elf64_Sym));

    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    EXPECT_FALSE(result.rewrite_discharge_checked)
        << "ordinary translation must ignore symbol metadata without relying on the verifier";
    AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    ASSERT_EQ(translated.text_sections().size(), 1u);
    const Section *translated_text = translated.text_sections().front();
    ASSERT_EQ(translated_text->size() % sizeof(uint32_t), 0u);
    const auto translated_words =
        std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(translated_text->data()),
                                  translated_text->size() / sizeof(uint32_t));
    // Ordinary text symbols are CFG split points, but visibility alone does not
    // make them external entries or require their bodies to be preserved.
    EXPECT_EQ(std::ranges::count(translated_words, cluster[0]), 1);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250TranslationIgnoresUnreferencedSymbolInsideInstruction) {
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr auto source_clause = cdna5::build_sopp(cdna5::kSClauseSopp, {.simm16 = 4});
  constexpr auto target_nop = cdna5::build_sopp(cdna5::kSNopSopp, {.simm16 = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::vector<uint32_t> words = {clear_m0[0], cluster[0],       cluster[1],
                                       cluster[2],  source_clause[0], kGfx1250SEndpgm};
  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text(words, words.size());

  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto text_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
  const auto symtab_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
  ASSERT_NE(text_it, shdrs.end());
  ASSERT_NE(symtab_it, shdrs.end());

  auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                    symtab_it->sh_size / sizeof(Elf64_Sym));
  ASSERT_EQ(symbols.size(), 3u);
  symbols[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  symbols[2].st_shndx = static_cast<uint16_t>(text_it - shdrs.begin());
  symbols[2].st_value = text_it->sh_addr + 2 * sizeof(uint32_t);
  symbols[2].st_size = sizeof(uint32_t);
  write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                       symbols.size() * sizeof(Elf64_Sym));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_FALSE(result.rewrite_discharge_checked);
  EXPECT_NE(result.elf_bytes, image);
  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_EQ(translated.text_sections().size(), 1u);
  const Section *translated_text = translated.text_sections().front();
  const auto translated_words =
      std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(translated_text->data()),
                                translated_text->size() / sizeof(uint32_t));
  EXPECT_EQ(std::ranges::count(translated_words, source_clause[0]), 0);
  EXPECT_EQ(std::ranges::count(translated_words, target_nop[0]), 1);
}

TEST(BinaryTranslatorE2E, RewriteDischargeIgnoresUnreferencedLocalTextLabel) {
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::vector<uint32_t> words = {clear_m0[0], cluster[0], cluster[1], cluster[2],
                                       kGfx1250SEndpgm};
  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text(words, words.size());

  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto text_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
  const auto symtab_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
  ASSERT_NE(text_it, shdrs.end());
  ASSERT_NE(symtab_it, shdrs.end());

  auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                    symtab_it->sh_size / sizeof(Elf64_Sym));
  ASSERT_EQ(symbols.size(), 3u);
  std::swap(symbols[1], symbols[2]);
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindLocal, kElfSymbolTypeFunc);
  symbols[1].st_shndx = static_cast<uint16_t>(text_it - shdrs.begin());
  symbols[1].st_value = text_it->sh_addr + sizeof(uint32_t);
  symbols[1].st_size = 4 * sizeof(uint32_t);
  symtab_it->sh_info = 2;
  write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                       symbols.size() * sizeof(Elf64_Sym));
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_TRUE(result.rewrite_discharge_verified);
  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_EQ(translated.text_sections().size(), 1u);
  const Section *translated_text = translated.text_sections().front();
  const auto translated_words =
      std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(translated_text->data()),
                                translated_text->size() / sizeof(uint32_t));
  EXPECT_EQ(std::ranges::count(translated_words, cluster[0]), 1)
      << "an unreferenced local label must not be treated as another executable body";
}

TEST(BinaryTranslatorE2E, RewriteDischargeHonorsRelocationBackedLocalTextEntry) {
  for (const uint32_t relocation_type : kSupportedExplicitSymbolRelocations) {
    SCOPED_TRACE(relocation_type);
    auto image = make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeFunc, true, /*addend=*/0,
                                                        relocation_type);
    const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                   EF_AMDGPU_MACH_AMDGCN_GFX1250);
    auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
    const auto text_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
    const auto symtab_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
    ASSERT_NE(text_it, shdrs.end());
    ASSERT_NE(symtab_it, shdrs.end());

    constexpr auto compound = cdna5::build_vop3p(
        cdna5::kVWmmaF3216x16x128F8f6f4Vop3p, {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272});
    static_assert(compound.size() == 2);
    write_bytes_for_test(image, text_it->sh_offset, compound.data(), sizeof(compound));
    auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                      symtab_it->sh_size / sizeof(Elf64_Sym));
    ASSERT_EQ(symbols.size(), 2u);
    symbols[1].st_info = elf_symbol_info(kElfSymbolBindLocal, kElfSymbolTypeFunc);
    symbols[1].st_value = text_it->sh_addr + sizeof(uint32_t);
    symbols[1].st_size = sizeof(uint32_t);
    symtab_it->sh_info = 2;
    write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                         symbols.size() * sizeof(Elf64_Sym));
    write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(result.rewrite_discharge_checked);
    EXPECT_FALSE(result.rewrite_discharge_verified);
    const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &candidate) {
      return candidate.kind == DiagnosticKind::ResidualRewrite;
    });
    ASSERT_NE(diagnostic, result.diagnostics.end());
    EXPECT_EQ(diagnostic->message,
              "rewrite-discharge verification found an invalid final executable entry");
  }
}

TEST(BinaryTranslatorE2E, PreservesEtRelNotypeAbs64EntryPolicy) {
  auto image = make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeNone, /*defined_in_text=*/true,
                                                      /*addend=*/0, R_AMDGPU_ABS64);
  auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto text_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
  const auto data_it = std::ranges::find_if(shdrs, [](const Elf64_Shdr &section) {
    return (section.sh_flags & SHF_ALLOC) != 0 && (section.sh_flags & SHF_EXECINSTR) == 0;
  });
  const auto symtab_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(text_it, shdrs.end());
  ASSERT_NE(data_it, shdrs.end());
  ASSERT_NE(symtab_it, shdrs.end());
  ASSERT_NE(rela_it, shdrs.end());

  ehdr.e_type = ET_REL;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                    symtab_it->sh_size / sizeof(Elf64_Sym));
  ASSERT_EQ(symbols.size(), 2u);
  symbols[1].st_value = sizeof(uint32_t);
  auto rela = read_elf_struct_for_test<Elf64_Rela>(image, rela_it->sh_offset);
  rela.r_offset = 0;
  rela_it->sh_info = static_cast<uint32_t>(data_it - shdrs.begin());

  constexpr auto compound = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                                               {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272});
  write_bytes_for_test(image, text_it->sh_offset, compound.data(), sizeof(compound));
  write_elf_struct_for_test(image, 0, ehdr);
  write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                       symbols.size() * sizeof(Elf64_Sym));
  write_elf_struct_for_test(image, rela_it->sh_offset, rela);
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  CodeObjectPatcher patcher(source);
  EXPECT_FALSE(patcher.has_unsupported_relocation_to_text());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(has_error_containing(result, DiagnosticKind::Legalization,
                                    "unsupported relocation referencing .text"));
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::ResidualRewrite,
                                   "invalid final executable entry"));
}

TEST(BinaryTranslatorE2E, PreservesExplicitTargetStaticRelocationPolicy) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {kGfx1250SEndpgm, kGfx1250SEndpgm}, TestRuntimeTextReference{
                                              .relocation = TestRuntimeTextRelocation::Abs64,
                                              .relocation_type = kUnrecognizedAmdGpuRelocation,
                                              .target_text_offset = sizeof(uint32_t),
                                          });
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto rodata_it = std::ranges::find_if(shdrs, [](const Elf64_Shdr &section) {
    return (section.sh_flags & SHF_ALLOC) != 0 && (section.sh_flags & SHF_EXECINSTR) == 0;
  });
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(rodata_it, shdrs.end());
  ASSERT_NE(rela_it, shdrs.end());
  rela_it->sh_info = static_cast<uint32_t>(rodata_it - shdrs.begin());
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  CodeObjectPatcher patcher(source);
  EXPECT_FALSE(patcher.has_unsupported_relocation_to_text());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_TRUE(result.rewrite_discharge_verified);
}

TEST(BinaryTranslatorE2E, ExplicitTargetNonAbs64FunctionReferenceCreatesTextEntry) {
  auto image = make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeFunc, /*defined_in_text=*/true,
                                                      /*addend=*/0, kUnrecognizedAmdGpuRelocation);
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto text_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
  const auto data_it = std::ranges::find_if(shdrs, [](const Elf64_Shdr &section) {
    return (section.sh_flags & SHF_ALLOC) != 0 && (section.sh_flags & SHF_EXECINSTR) == 0;
  });
  const auto symtab_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(text_it, shdrs.end());
  ASSERT_NE(data_it, shdrs.end());
  ASSERT_NE(symtab_it, shdrs.end());
  ASSERT_NE(rela_it, shdrs.end());

  constexpr auto compound = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                                               {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272});
  static_assert(compound.size() == 2);
  write_bytes_for_test(image, text_it->sh_offset, compound.data(), sizeof(compound));
  auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                    symtab_it->sh_size / sizeof(Elf64_Sym));
  ASSERT_EQ(symbols.size(), 2u);
  symbols[1].st_value = text_it->sh_addr + sizeof(uint32_t);
  symbols[1].st_size = sizeof(uint32_t);
  rela_it->sh_info = static_cast<uint32_t>(data_it - shdrs.begin());
  write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                       symbols.size() * sizeof(Elf64_Sym));
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  CodeObjectPatcher patcher(source);
  EXPECT_FALSE(patcher.has_unsupported_relocation_to_text());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::ResidualRewrite,
                                   "invalid final executable entry"));
}

TEST(BinaryTranslatorE2E, RejectsDynamicNoneBeforeResolvingSymbolMetadata) {
  constexpr auto compound = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                                               {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272});
  static_assert(compound.size() == 2);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {compound[0], compound[1], kGfx1250SEndpgm},
      TestRuntimeTextReference{
          .relocation = TestRuntimeTextRelocation::Abs64,
          .relocation_type = R_AMDGPU_NONE,
          .target_text_offset = sizeof(uint32_t),
      });
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto text_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
  const auto rodata_it = std::ranges::find_if(shdrs, [](const Elf64_Shdr &section) {
    return (section.sh_flags & SHF_ALLOC) != 0 && (section.sh_flags & SHF_EXECINSTR) == 0;
  });
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(text_it, shdrs.end());
  ASSERT_NE(rodata_it, shdrs.end());
  ASSERT_NE(rela_it, shdrs.end());

  // Keep the second kernel entry at the instruction following the compound.
  // Offset four is then referenced only by the target-less dynamic NONE record.
  // Poison both ways entry discovery could reach its symbol: record-level NONE
  // rejection must win before sh_link or the symbol index is inspected.
  write_kernel_descriptor_entry_offset(
      image.data() + rodata_it->sh_offset + kKernelDescriptorSize,
      static_cast<int64_t>(text_it->sh_addr + sizeof(compound)) -
          static_cast<int64_t>(rodata_it->sh_addr + kKernelDescriptorSize));
  rela_it->sh_link = static_cast<uint32_t>(shdrs.size());
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  auto rela = read_elf_struct_for_test<Elf64_Rela>(image, rela_it->sh_offset);
  rela.r_offset = text_it->sh_addr;
  rela.r_info = (static_cast<uint64_t>(99) << 32) | R_AMDGPU_NONE;
  rela.r_addend = 17;
  write_elf_struct_for_test(image, rela_it->sh_offset, rela);

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  CodeObjectPatcher patcher(source);
  EXPECT_FALSE(patcher.has_relocations_within_text());
  EXPECT_TRUE(patcher.has_rocr_rejected_none_relocation());
  EXPECT_FALSE(patcher.has_unsupported_relocation_to_text());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                   "R_AMDGPU_NONE outside a valid explicit-target"));
  EXPECT_FALSE(has_error_containing(result, DiagnosticKind::ResourceLimit,
                                    "could not recover relocation-backed executable entries"));
}

TEST(BinaryTranslatorE2E, RejectsDynamicNoneBeforeInspectingInvalidPlace) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {kGfx1250SEndpgm, kGfx1250SEndpgm}, TestRuntimeTextReference{
                                              .relocation = TestRuntimeTextRelocation::Abs64,
                                              .relocation_type = R_AMDGPU_NONE,
                                              .target_text_offset = 0,
                                          });
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(rela_it, shdrs.end());
  auto relocation = read_elf_struct_for_test<Elf64_Rela>(image, rela_it->sh_offset);
  relocation.r_offset = std::numeric_limits<uint64_t>::max();
  relocation.r_info = R_AMDGPU_NONE;
  write_elf_struct_for_test(image, rela_it->sh_offset, relocation);

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                   "R_AMDGPU_NONE outside a valid explicit-target"));
}

TEST(BinaryTranslatorE2E, IgnoresTargetlessShtRelNoneLikeRocr) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {kGfx1250SEndpgm, kGfx1250SEndpgm}, TestRuntimeTextReference{
                                              .relocation = TestRuntimeTextRelocation::Abs64,
                                              .relocation_type = R_AMDGPU_NONE,
                                              .target_text_offset = 0,
                                          });
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(rela_it, shdrs.end());
  const size_t relocation_section_index = static_cast<size_t>(rela_it - shdrs.begin());
  const Elf64_Rel rel{.r_offset = std::numeric_limits<uint64_t>::max(), .r_info = R_AMDGPU_NONE};
  write_elf_struct_for_test(image, rela_it->sh_offset, rel);
  shdrs[relocation_section_index].sh_type = SHT_REL;
  shdrs[relocation_section_index].sh_size = sizeof(Elf64_Rel);
  shdrs[relocation_section_index].sh_entsize = sizeof(Elf64_Rel);
  shdrs[relocation_section_index].sh_link = static_cast<uint32_t>(shdrs.size());
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  CodeObjectPatcher patcher(source);
  EXPECT_FALSE(patcher.has_rocr_rejected_none_relocation());
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_TRUE(result.rewrite_discharge_verified);
}

TEST(BinaryTranslatorE2E, IgnoresExplicitTargetNoneRelocationToTextSymbol) {
  constexpr auto compound = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                                               {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272});
  static_assert(compound.size() == 2);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {compound[0], compound[1], kGfx1250SEndpgm},
      TestRuntimeTextReference{
          .relocation = TestRuntimeTextRelocation::Abs64,
          .relocation_type = R_AMDGPU_NONE,
          .target_text_offset = sizeof(uint32_t),
      });
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto text_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
  const auto rodata_it = std::ranges::find_if(shdrs, [](const Elf64_Shdr &section) {
    return (section.sh_flags & SHF_ALLOC) != 0 && (section.sh_flags & SHF_EXECINSTR) == 0;
  });
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(text_it, shdrs.end());
  ASSERT_NE(rodata_it, shdrs.end());
  ASSERT_NE(rela_it, shdrs.end());

  // ROCr skips explicit-target/static relocation sections for supported code objects.
  // Point this record at allocated data and make its symbol-table link unusable. NONE must be
  // ignored before entry recovery resolves that metadata, so it neither creates the otherwise
  // invalid entry at the compound instruction's second dword nor requires symbol remapping.
  rela_it->sh_info = static_cast<uint32_t>(rodata_it - shdrs.begin());
  rela_it->sh_link = static_cast<uint32_t>(shdrs.size());
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  write_kernel_descriptor_entry_offset(
      image.data() + rodata_it->sh_offset + kKernelDescriptorSize,
      static_cast<int64_t>(text_it->sh_addr + sizeof(compound)) -
          static_cast<int64_t>(rodata_it->sh_addr + kKernelDescriptorSize));
  auto rela = read_elf_struct_for_test<Elf64_Rela>(image, rela_it->sh_offset);
  rela.r_offset = text_it->sh_addr;
  rela.r_addend = 17;
  write_elf_struct_for_test(image, rela_it->sh_offset, rela);

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  CodeObjectPatcher patcher(source);
  EXPECT_FALSE(patcher.has_relocations_within_text());
  EXPECT_FALSE(patcher.has_rocr_rejected_none_relocation());
  EXPECT_FALSE(patcher.has_unsupported_relocation_to_text());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_TRUE(result.rewrite_discharge_verified);
}

TEST(BinaryTranslatorE2E, RejectsUnsupportedSymbolRelocationsToText) {
  for (const uint32_t relocation_type : kUnsupportedExplicitSymbolRelocations) {
    SCOPED_TRACE(relocation_type);
    auto image = make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeFunc, true, /*addend=*/0,
                                                        relocation_type);
    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                     "unsupported relocation referencing .text"));
  }
}

TEST(BinaryTranslatorE2E, RejectsUnsupportedRelocationBackedTextSymbolTypes) {
  for (const uint8_t symbol_type : {kElfSymbolTypeNone, kUnsupportedRuntimeSymbolType}) {
    SCOPED_TRACE(static_cast<uint32_t>(symbol_type));
    auto image = make_amdgpu_elf_with_symbol_relocation(symbol_type, /*defined_in_text=*/true);
    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_FALSE(result.rewrite_discharge_checked);
    EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                     "unsupported relocation referencing .text"));
  }
}

TEST(BinaryTranslatorE2E, RewriteDischargeDoesNotPromoteRelocationBackedNonEntrySymbol) {
  for (const uint8_t symbol_type : {kElfSymbolTypeObject, kElfSymbolTypeAmdGpuHsaKernel}) {
    SCOPED_TRACE(static_cast<uint32_t>(symbol_type));
    auto image = make_amdgpu_elf_with_symbol_relocation(
        symbol_type, true, /*addend=*/0, kSupportedExplicitSymbolRelocations.front());
    const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                   EF_AMDGPU_MACH_AMDGCN_GFX1250);
    auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
    const auto text_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
    const auto symtab_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
    ASSERT_NE(text_it, shdrs.end());
    ASSERT_NE(symtab_it, shdrs.end());

    constexpr auto compound =
        cdna5::build_vop3(cdna5::kVAddF32Vop3, {.vdst = 8, .src0 = 256, .src1 = 257, .src2 = 0});
    static_assert(compound.size() == 2);
    write_bytes_for_test(image, text_it->sh_offset, compound.data(), sizeof(compound));
    auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                      symtab_it->sh_size / sizeof(Elf64_Sym));
    ASSERT_EQ(symbols.size(), 2u);
    symbols[1].st_value = text_it->sh_addr + sizeof(uint32_t);
    symbols[1].st_size = sizeof(uint32_t);
    write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                         symbols.size() * sizeof(Elf64_Sym));

    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(result.rewrite_discharge_checked);
    EXPECT_TRUE(result.rewrite_discharge_verified);
  }
}

TEST(BinaryTranslatorE2E, RewriteDischargeIgnoresExplicitNonAllocatedRelocation) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {kGfx1250SEndpgm, kGfx1250SEndpgm}, TestRuntimeTextReference{
                                              .relocation = TestRuntimeTextRelocation::Abs64,
                                              .target_text_offset = 2,
                                          });
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto strtab_it = std::ranges::find_if(shdrs, [&](const Elf64_Shdr &section) {
    return section.sh_type == SHT_STRTAB &&
           static_cast<size_t>(&section - shdrs.data()) != ehdr.e_shstrndx;
  });
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(strtab_it, shdrs.end());
  ASSERT_NE(rela_it, shdrs.end());

  rela_it->sh_info = static_cast<uint32_t>(strtab_it - shdrs.begin());
  rela_it->sh_entsize = 0;
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_TRUE(result.rewrite_discharge_verified);
}

TEST(BinaryTranslatorE2E, RewriteDischargeIgnoresMalformedUnreferencedSymbolTable) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<uint8_t, sizeof(Elf64_Sym) + 8> unused_symbols{};

  for (const uint64_t entry_size : {uint64_t{0}, uint64_t{sizeof(unused_symbols)}}) {
    SCOPED_TRACE(entry_size);
    auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
        {kGfx1250SEndpgm, kGfx1250SEndpgm}, TestRuntimeTextReference{
                                                .relocation = TestRuntimeTextRelocation::Relative64,
                                                .target_text_offset = 0,
                                            });
    const auto header = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    const auto sections =
        read_elf_array_for_test<Elf64_Shdr>(image, header.e_shoff, header.e_shnum);
    const auto valid_symtab = std::ranges::find_if(
        sections, [](const Elf64_Shdr &candidate) { return candidate.sh_type == SHT_SYMTAB; });
    ASSERT_NE(valid_symtab, sections.end());

    Elf64_Shdr irrelevant_symtab{};
    irrelevant_symtab.sh_name = valid_symtab->sh_name;
    irrelevant_symtab.sh_type = SHT_SYMTAB;
    irrelevant_symtab.sh_link = valid_symtab->sh_link;
    irrelevant_symtab.sh_addralign = alignof(Elf64_Sym);
    irrelevant_symtab.sh_entsize = entry_size;
    append_elf_section_for_test(image, irrelevant_symtab, unused_symbols);

    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    EXPECT_TRUE(result.rewrite_discharge_checked);
    EXPECT_TRUE(result.rewrite_discharge_verified);
  }
}

TEST(BinaryTranslatorE2E, RejectsAllocatedExecutableNobitsSections) {
  for (const bool replace_text : {false, true}) {
    SCOPED_TRACE(replace_text);
    auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
    const auto header = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                   EF_AMDGPU_MACH_AMDGCN_GFX1250);
    auto sections = read_elf_array_for_test<Elf64_Shdr>(image, header.e_shoff, header.e_shnum);
    Elf64_Shdr &nobits = sections[replace_text ? 1 : 2];
    nobits.sh_type = SHT_NOBITS;
    nobits.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
    nobits.sh_offset = image.size() + 0x1000;
    write_bytes_for_test(image, header.e_shoff, sections.data(),
                         sections.size() * sizeof(Elf64_Shdr));

    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    EXPECT_EQ(source.allocated_executable_sections().size(), replace_text ? 1u : 2u);
    EXPECT_EQ(source.text_sections().size(), replace_text ? 0u : 1u);

    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_FALSE(result.rewrite_discharge_checked);
    EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                     "exactly one allocated executable section named .text"));
  }
}

TEST(BinaryTranslatorE2E, UsesActualTextSectionIndexForRelocationBackedEntries) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {kGfx1250SEndpgm, kGfx1250SEndpgm}, TestRuntimeTextReference{
                                              .relocation = TestRuntimeTextRelocation::Abs64,
                                              .target_text_offset = 2,
                                          });
  const auto header = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  auto sections = read_elf_array_for_test<Elf64_Shdr>(image, header.e_shoff, header.e_shnum);
  const auto text = std::ranges::find_if(sections, [](const Elf64_Shdr &candidate) {
    return (candidate.sh_flags & SHF_EXECINSTR) != 0;
  });
  ASSERT_NE(text, sections.end());
  ASSERT_GT(text - sections.begin(), 0);

  // A preceding NOBITS header may legally reuse the text section's file offset.
  // Section identity, not the non-unique (offset, size) pair, must decide which
  // symbol-table section index denotes executable code.
  Elf64_Shdr &collision = sections[static_cast<size_t>(text - sections.begin() - 1)];
  collision.sh_type = SHT_NOBITS;
  collision.sh_offset = text->sh_offset;
  collision.sh_size = text->sh_size;
  write_bytes_for_test(image, header.e_shoff, sections.data(),
                       sections.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.text_sections().size(), 1u);
  ASSERT_EQ(source.text_sections().front()->sectionHeaderIndex(),
            std::optional<size_t>(static_cast<size_t>(text - sections.begin())));
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_FALSE(result.rewrite_discharge_checked);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                   "translation found an invalid executable entry"));
}

TEST(BinaryTranslatorE2E, TranslationFailsClosedOnMalformedAllocatedRelocation) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {kGfx1250SEndpgm, kGfx1250SEndpgm}, TestRuntimeTextReference{
                                              .relocation = TestRuntimeTextRelocation::Abs64,
                                              .target_text_offset = 0,
                                          });
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(rela_it, shdrs.end());
  rela_it->sh_link = static_cast<uint32_t>(shdrs.size());
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_FALSE(result.rewrite_discharge_checked);
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &candidate) {
    return candidate.kind == DiagnosticKind::Legalization;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->message,
            "translation could not recover relocation-backed executable entries");
}

TEST(BinaryTranslatorE2E, RewriteDischargeIgnoresUnreferencedVisibleSymbolMetadata) {
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr std::array<int64_t, 2> target_deltas = {
      static_cast<int64_t>(sizeof(uint32_t)),
      -static_cast<int64_t>(sizeof(uint32_t)),
  };

  for (const int64_t target_delta : target_deltas) {
    SCOPED_TRACE(target_delta);
    auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
        {clear_m0[0], cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
    const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                   EF_AMDGPU_MACH_AMDGCN_GFX1250);
    const auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
    const auto text_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
    const auto symtab_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
    ASSERT_NE(text_it, shdrs.end());
    ASSERT_NE(symtab_it, shdrs.end());

    auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                      symtab_it->sh_size / sizeof(Elf64_Sym));
    ASSERT_EQ(symbols.size(), 3u);
    symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
    symbols[1].st_shndx = static_cast<uint16_t>(text_it - shdrs.begin());
    symbols[1].st_value = text_it->sh_addr;
    symbols[1].st_size = sizeof(uint32_t);
    symbols[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
    symbols[2].st_shndx = static_cast<uint16_t>(text_it - shdrs.begin());
    symbols[2].st_value =
        static_cast<uint64_t>(static_cast<int64_t>(text_it->sh_addr) + target_delta);
    symbols[2].st_size = sizeof(uint32_t);
    write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                         symbols.size() * sizeof(Elf64_Sym));

    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    EXPECT_TRUE(result.rewrite_discharge_checked);
    EXPECT_TRUE(result.rewrite_discharge_verified);
  }
}

TEST(BinaryTranslatorE2E, TranslationFailsClosedOnMultipleExecutableSections) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  ASSERT_GE(shdrs.size(), 3u);
  shdrs[2].sh_flags |= SHF_EXECINSTR;
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.text_sections().size(), 1u);
  ASSERT_EQ(source.allocated_executable_sections().size(), 2u);
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_FALSE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &candidate) {
    return candidate.kind == DiagnosticKind::Legalization;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->severity, DiagnosticSeverity::Error);
  EXPECT_EQ(diagnostic->guest_offset, std::nullopt);
  EXPECT_EQ(diagnostic->output_offset, std::nullopt);
  EXPECT_TRUE(diagnostic->mnemonic.empty());
  EXPECT_EQ(diagnostic->message,
            "translation requires exactly one allocated executable section named .text");
}

TEST(BinaryTranslatorE2E, TranslationRejectsSoleExecutableSectionNotNamedText) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  ASSERT_GE(shdrs.size(), 3u);
  shdrs[1].sh_name = shdrs[2].sh_name;
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(source.text_sections().empty());
  ASSERT_EQ(source.allocated_executable_sections().size(), 1u);
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_FALSE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  EXPECT_TRUE(has_error_containing(
      result, DiagnosticKind::Legalization,
      "translation requires exactly one allocated executable section named .text"));
}

TEST(BinaryTranslatorE2E, TranslationRejectsMalformedNameOnAdditionalExecutableSection) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  ASSERT_GE(shdrs.size(), 3u);
  shdrs[2].sh_flags |= SHF_EXECINSTR;
  shdrs[2].sh_name = std::numeric_limits<uint32_t>::max();
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_FALSE(source.is_valid());
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_FALSE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::ResourceLimit,
                                   "translation could not parse input code object"));
}

TEST(BinaryTranslatorE2E, TranslationRejectsMalformedNameOnSoleExecutableSection) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  ASSERT_GE(shdrs.size(), 3u);
  shdrs[1].sh_name = std::numeric_limits<uint32_t>::max();
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_FALSE(source.is_valid());
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_FALSE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::ResourceLimit,
                                   "translation could not parse input code object"));
}

TEST(BinaryTranslatorE2E, RewriteDischargeIgnoresUnreferencedVisibleTextSymbolEntry) {
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const uint8_t symbol_type : {kElfSymbolTypeFunc, kElfSymbolTypeNone}) {
    SCOPED_TRACE(symbol_type == kElfSymbolTypeFunc ? "STT_FUNC" : "STT_NOTYPE");
    auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
        {clear_m0[0], cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});

    const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                   EF_AMDGPU_MACH_AMDGCN_GFX1250);
    const auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
    const auto symtab_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
    ASSERT_NE(symtab_it, shdrs.end());
    auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                      symtab_it->sh_size / sizeof(Elf64_Sym));
    ASSERT_EQ(symbols.size(), 3u);
    symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, symbol_type);
    symbols[1].st_shndx = 1;
    symbols[1].st_value = 0x1100;
    symbols[1].st_size = sizeof(uint32_t);
    symbols[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, symbol_type);
    symbols[2].st_shndx = 1;
    symbols[2].st_value = 0x1100 + sizeof(uint32_t);
    symbols[2].st_size = 4 * sizeof(uint32_t);
    write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                         symbols.size() * sizeof(Elf64_Sym));

    AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    BinaryTranslatorOptions options;
    options.input_revision = ProcessorRevision::Gfx1250B0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
    options.verify_rewrite_discharge = true;
    BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
    const auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    EXPECT_TRUE(result.rewrite_discharge_checked);
    EXPECT_TRUE(result.rewrite_discharge_verified);
  }
}

TEST(BinaryTranslatorE2E, RewriteDischargeDoesNotCreditPredecessorAcrossRelativeEntry) {
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {clear_m0[0], cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});

  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto text_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
  const auto data_it = std::ranges::find_if(shdrs, [](const Elf64_Shdr &section) {
    return section.sh_type == SHT_PROGBITS && (section.sh_flags & SHF_ALLOC) != 0 &&
           (section.sh_flags & SHF_EXECINSTR) == 0;
  });
  const auto symtab_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
  ASSERT_NE(text_it, shdrs.end());
  ASSERT_NE(data_it, shdrs.end());
  ASSERT_NE(symtab_it, shdrs.end());

  auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                    symtab_it->sh_size / sizeof(Elf64_Sym));
  ASSERT_EQ(symbols.size(), 3u);
  for (size_t symbol_index = 1; symbol_index < symbols.size(); ++symbol_index) {
    symbols[symbol_index].st_shndx = SHN_UNDEF;
    symbols[symbol_index].st_value = 0;
    symbols[symbol_index].st_size = 0;
  }
  write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                       symbols.size() * sizeof(Elf64_Sym));

  Elf64_Rela relocation{};
  relocation.r_offset = data_it->sh_addr;
  relocation.r_info = R_AMDGPU_RELATIVE64;
  relocation.r_addend =
      static_cast<int64_t>(text_it->sh_addr + static_cast<uint64_t>(sizeof(uint32_t)));
  write_bytes_for_test(image, data_it->sh_offset, &relocation, sizeof(relocation));

  data_it->sh_type = SHT_RELA;
  data_it->sh_size = sizeof(Elf64_Rela);
  data_it->sh_link = static_cast<uint32_t>(symtab_it - shdrs.begin());
  data_it->sh_info = 0;
  data_it->sh_addralign = alignof(Elf64_Rela);
  data_it->sh_entsize = sizeof(Elf64_Rela);
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_FALSE(result.rewrite_discharge_verified);
  const auto residual = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == DiagnosticKind::ResidualRewrite;
  });
  ASSERT_NE(residual, result.diagnostics.end());
  EXPECT_EQ(residual->severity, DiagnosticSeverity::Error);
  EXPECT_EQ(residual->output_offset, std::optional<uint64_t>(sizeof(uint32_t)));
  EXPECT_EQ(residual->mnemonic, "cluster_load_b32");
}

TEST(BinaryTranslatorE2E, RejectsRelocationBackedEntryOutsideKernelTranslationScopes) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  auto image = make_minimal_amdgpu_elf_with_two_kernel_descriptors(
      {kGfx1250SEndpgm, kGfx1250SEndpgm, kGfx1250SNop},
      TestRuntimeTextReference{
          .relocation = TestRuntimeTextRelocation::Relative64,
          .target_text_offset = 2 * sizeof(uint32_t),
      });

  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto text_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return (section.sh_flags & SHF_EXECINSTR) != 0; });
  const auto symtab_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
  ASSERT_NE(text_it, shdrs.end());
  ASSERT_NE(symtab_it, shdrs.end());

  auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab_it->sh_offset,
                                                    symtab_it->sh_size / sizeof(Elf64_Sym));
  ASSERT_EQ(symbols.size(), 3u);
  symbols[1].st_shndx = SHN_UNDEF;
  symbols[1].st_value = 0;
  symbols[1].st_size = 0;
  write_bytes_for_test(image, symtab_it->sh_offset, symbols.data(),
                       symbols.size() * sizeof(Elf64_Sym));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_FALSE(result.rewrite_discharge_checked)
      << "required runtime entries must be enforced during ordinary translation";
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &candidate) {
    return candidate.kind == DiagnosticKind::Legalization &&
           candidate.message ==
               "relocation-backed executable entry is outside every kernel translation scope";
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->severity, DiagnosticSeverity::Error);
  EXPECT_EQ(diagnostic->guest_offset, 2 * sizeof(uint32_t));
}

TEST(CodeObjectPatcher, ReplaceTextGrowsTextAndShiftsFollowingSections) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  const std::array<uint32_t, 4> text_words = {0xBF800000u, 0xBF800000u, 0xDEADBEEFu, 0xCAFEBABEu};
  const auto *text_bytes = reinterpret_cast<const uint8_t *>(text_words.data());
  ASSERT_TRUE(patcher.replace_text({text_bytes, text_words.size() * sizeof(uint32_t)}));

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());

  const Section *text = patched.text_sections()[0];
  ASSERT_NE(text, nullptr);
  EXPECT_EQ(text->size(), text_words.size() * sizeof(uint32_t));
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_EQ(patched.text_sections()[0]->name(), ".text");
  EXPECT_EQ(find_section(patched, ".rj_translations"), nullptr);
  EXPECT_EQ(std::memcmp(text->data(), text_words.data(), text->size()), 0);

  const Section *rodata = find_section(patched, ".rodata");
  ASSERT_NE(rodata, nullptr);
  EXPECT_EQ(rodata->sectionOffset(), text->sectionOffset() + text->size())
      << "sections following .text must be shifted after the grown text";
  uint32_t rodata_word = 0;
  std::memcpy(&rodata_word, rodata->data(), sizeof(rodata_word));
  EXPECT_EQ(rodata_word, 0xA5A55A5Au);
}

// COMPUTE_PGM_RSRC3 is carried over verbatim only when source and target agree on its layout.
// "GFX10 or later" is not a fine enough test: INST_PREF_SIZE is GFX11+, and IMAGE_OP is GFX11+,
// so handing a GFX11 or GFX12 word to a GFX10 target would set bits GFX10 requires to be zero.
// Equally, a GFX12 word must survive intact on a GFX12 target, including the GFX125-only
// NAMED_BAR_CNT and an INST_PREF_SIZE above the 63 that GFX11's narrower field can hold.
TEST(CodeObjectPatcher, Rsrc3IsCarriedOnlyBetweenMatchingLayouts) {
  using namespace rocr::llvm::amdhsa;

  const auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject probe(image.data(), image.size());
  ASSERT_TRUE(probe.is_valid());
  const Section *probe_rodata = find_section(probe, ".rodata");
  ASSERT_NE(probe_rodata, nullptr);
  ASSERT_GE(probe_rodata->size(), sizeof(kernel_descriptor_t));

  // A GFX12-shaped source word: INST_PREF_SIZE beyond GFX11's 6-bit range, plus a GFX125-only
  // named-barrier allocation.
  uint32_t source_rsrc3 = 0;
  AMDHSA_BITS_SET(source_rsrc3, COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE, 107u);
  AMDHSA_BITS_SET(source_rsrc3, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT, 3u);

  auto patched_rsrc3 = [&](rj_code_arch_t source_arch,
                           rj_code_arch_t target_arch) -> std::optional<uint32_t> {
    auto local_image = image;
    AmdGpuCodeObject local_co(local_image.data(), local_image.size());
    if (!local_co.is_valid())
      return std::nullopt;
    const Section *rodata = find_section(local_co, ".rodata");
    if (rodata == nullptr)
      return std::nullopt;

    auto descriptor = read_kernel_descriptor_for_test(rodata->data());
    descriptor.compute_pgm_rsrc3 = source_rsrc3;
    write_kernel_descriptor_for_test(local_image.data() + rodata->sectionOffset(), descriptor);

    AmdGpuCodeObject seeded(local_image.data(), local_image.size());
    if (!seeded.is_valid())
      return std::nullopt;

    KdTranslation translation{};
    translation.descriptor_file_offset = rodata->sectionOffset();
    translation.target_wave_size = 32;
    translation.source_arch = source_arch;

    CodeObjectPatcher patcher(seeded);
    if (!patcher.apply_kernel_descriptor_translation(translation, target_arch))
      return std::nullopt;
    const auto patched_image = patcher.emit();
    return read_kernel_descriptor_for_test(patched_image.data() +
                                           translation.descriptor_file_offset)
        .compute_pgm_rsrc3;
  };

  // Matching layout: carried verbatim, both the wide INST_PREF_SIZE and NAMED_BAR_CNT.
  const auto same_layout = patched_rsrc3(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(same_layout.has_value());
  EXPECT_EQ(*same_layout, source_rsrc3);
  EXPECT_EQ(AMDHSA_BITS_GET(*same_layout, COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE), 107u);
  EXPECT_EQ(AMDHSA_BITS_GET(*same_layout, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT), 3u);

  // GFX10 targets have no INST_PREF_SIZE and no IMAGE_OP; the GFX12 word must not be inherited.
  for (const rj_code_arch_t gfx10 : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const auto rebuilt = patched_rsrc3(ROCJITSU_CODE_ARCH_CDNA5, gfx10);
    ASSERT_TRUE(rebuilt.has_value());
    EXPECT_NE(*rebuilt, source_rsrc3);
    EXPECT_EQ(AMDHSA_BITS_GET(*rebuilt, COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE), 0u)
        << "GFX10 reserves the INST_PREF_SIZE bits and requires them to be zero";
  }

  // A GFX9/CDNA source encodes ACCUM_OFFSET here, so it is rebuilt for a GFX12 target.
  const auto from_cdna = patched_rsrc3(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(from_cdna.has_value());
  EXPECT_NE(*from_cdna, source_rsrc3);
  EXPECT_EQ(AMDHSA_BITS_GET(*from_cdna, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT), 0u);

  // The narrow case: GFX11 -> GFX10. Both are "GFX10+" and both use SHARED_VGPR_COUNT, so a
  // coarse family check treats them as interchangeable -- but GFX11's INST_PREF_SIZE at 9:4 and
  // IMAGE_OP at 31 are reserved on GFX10 and must not be carried over.
  uint32_t gfx11_rsrc3 = 0;
  AMDHSA_BITS_SET(gfx11_rsrc3, COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE, 21u);
  AMDHSA_BITS_SET(gfx11_rsrc3, COMPUTE_PGM_RSRC3_GFX10_PLUS_IMAGE_OP, 1u);
  source_rsrc3 = gfx11_rsrc3;
  for (const rj_code_arch_t gfx10 : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const auto rebuilt = patched_rsrc3(ROCJITSU_CODE_ARCH_RDNA3, gfx10);
    ASSERT_TRUE(rebuilt.has_value());
    EXPECT_EQ(AMDHSA_BITS_GET(*rebuilt, COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE), 0u)
        << "GFX10 reserves INST_PREF_SIZE; a GFX11 word must not be carried over";
    EXPECT_EQ(AMDHSA_BITS_GET(*rebuilt, COMPUTE_PGM_RSRC3_GFX10_PLUS_IMAGE_OP), 0u)
        << "IMAGE_OP is GFX11+; GFX10 reserves bit 31";
  }

  // GFX11 -> GFX11 still round-trips, so the split does not over-rebuild.
  const auto gfx11_same = patched_rsrc3(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(gfx11_same.has_value());
  EXPECT_EQ(*gfx11_same, gfx11_rsrc3);

  // GFX125 -> GFX120. Both are GFX12, and both use the same 8-bit INST_PREF_SIZE, but LLVM
  // reserves bits 21:14 on GFX120 while GFX125 fills them with NAMED_BAR_CNT,
  // ENABLE_DYNAMIC_VGPR, TCP_SPLIT and ENABLE_DIDT_THROTTLE. A shared "GFX12+" bucket would
  // carry those straight into target-reserved bits.
  uint32_t gfx125_rsrc3 = 0;
  AMDHSA_BITS_SET(gfx125_rsrc3, COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE, 107u);
  AMDHSA_BITS_SET(gfx125_rsrc3, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT, 3u);
  AMDHSA_BITS_SET(gfx125_rsrc3, COMPUTE_PGM_RSRC3_GFX125_TCP_SPLIT, 5u);
  AMDHSA_BITS_SET(gfx125_rsrc3, COMPUTE_PGM_RSRC3_GFX125_ENABLE_DYNAMIC_VGPR, 1u);
  source_rsrc3 = gfx125_rsrc3;
  const auto to_gfx120 = patched_rsrc3(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(to_gfx120.has_value());
  EXPECT_EQ(AMDHSA_BITS_GET(*to_gfx120, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT), 0u)
      << "bits 21:14 are reserved on GFX120 and must not inherit GFX125 state";
  EXPECT_EQ(AMDHSA_BITS_GET(*to_gfx120, COMPUTE_PGM_RSRC3_GFX125_TCP_SPLIT), 0u);
  EXPECT_EQ(AMDHSA_BITS_GET(*to_gfx120, COMPUTE_PGM_RSRC3_GFX125_ENABLE_DYNAMIC_VGPR), 0u);

  // GFX125 -> GFX125 keeps all of it.
  const auto gfx125_same = patched_rsrc3(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(gfx125_same.has_value());
  EXPECT_EQ(*gfx125_same, gfx125_rsrc3);
}

// SHARED_VGPR_COUNT is the one carried field this patcher invalidates itself: it rewrites both
// the wave size and the RSRC1 VGPR allocation the field is constrained against. LLVM requires
// the count to be zero for wave32, and for wave64 requires
// (compute_pgm_rsrc1.vgprs + 1) * 4 + shared_vgpr_count * 8 <= 256.
//
// The field counts registers the unchanged kernel body uses, so a descriptor that no longer
// leaves room for them is refused rather than clamped: clamping would keep the body's demand
// while shrinking what the descriptor reserves for it.
TEST(CodeObjectPatcher, SharedVgprCountIsReconciledWithTheTargetAllocation) {
  using namespace rocr::llvm::amdhsa;

  const auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();

  struct PatchOutcome {
    bool seeded = false;  ///< Fixture setup succeeded.
    bool patched = false; ///< The patcher accepted the request.
    uint32_t shared_count = 0;
  };

  auto patched = [&](uint32_t source_shared, uint32_t target_wave,
                     uint32_t vgpr_granulated) -> PatchOutcome {
    auto local_image = image;
    AmdGpuCodeObject probe(local_image.data(), local_image.size());
    if (!probe.is_valid())
      return {};
    const Section *rodata = find_section(probe, ".rodata");
    if (rodata == nullptr)
      return {};

    auto descriptor = read_kernel_descriptor_for_test(rodata->data());
    descriptor.compute_pgm_rsrc3 = 0;
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX10_PLUS_SHARED_VGPR_COUNT,
                    source_shared);
    write_kernel_descriptor_for_test(local_image.data() + rodata->sectionOffset(), descriptor);

    AmdGpuCodeObject seeded(local_image.data(), local_image.size());
    if (!seeded.is_valid())
      return {};

    KdTranslation translation{};
    translation.descriptor_file_offset = rodata->sectionOffset();
    translation.target_wave_size = target_wave;
    translation.target_vgpr_granulated = vgpr_granulated;
    translation.source_arch = ROCJITSU_CODE_ARCH_RDNA3;

    PatchOutcome outcome;
    outcome.seeded = true;
    CodeObjectPatcher patcher(seeded);
    outcome.patched =
        patcher.apply_kernel_descriptor_translation(translation, ROCJITSU_CODE_ARCH_RDNA3);
    if (!outcome.patched)
      return outcome;
    const auto out = patcher.emit();
    outcome.shared_count = AMDHSA_BITS_GET(
        read_kernel_descriptor_for_test(out.data() + translation.descriptor_file_offset)
            .compute_pgm_rsrc3,
        COMPUTE_PGM_RSRC3_GFX10_PLUS_SHARED_VGPR_COUNT);
    return outcome;
  };

  // Wave64 with a small VGPR allocation: the source request still fits, so it survives.
  const auto roomy = patched(/*source_shared=*/4, /*target_wave=*/64, /*vgpr_granulated=*/1);
  ASSERT_TRUE(roomy.seeded);
  ASSERT_TRUE(roomy.patched);
  EXPECT_EQ(roomy.shared_count, 4u);

  // Exactly at the limit: (33 + 1) * 4 = 136 arch VGPRs plus 15 * 8 = 120 shared reaches 256.
  const auto exact = patched(/*source_shared=*/15, /*target_wave=*/64, /*vgpr_granulated=*/33);
  ASSERT_TRUE(exact.seeded);
  ASSERT_TRUE(exact.patched);
  EXPECT_EQ(exact.shared_count, 15u);

  // One block past the limit is refused rather than trimmed to fit.
  const auto over = patched(/*source_shared=*/15, /*target_wave=*/64, /*vgpr_granulated=*/34);
  ASSERT_TRUE(over.seeded);
  EXPECT_FALSE(over.patched) << "an allocation that crowds out the shared blocks must be refused";

  // Wave64 with a large allocation: (63 + 1) * 4 = 256 VGPRs leaves no room at all. Clamping to
  // zero here used to report success while removing every block the body still uses.
  const auto tight = patched(/*source_shared=*/9, /*target_wave=*/64, /*vgpr_granulated=*/63);
  ASSERT_TRUE(tight.seeded);
  EXPECT_FALSE(tight.patched) << "a 9-to-0 reduction is a resource change, not a translation";

  // Wave32 has no shared VGPR blocks at all, so there is nowhere to put the source's request.
  const auto wave32 = patched(/*source_shared=*/9, /*target_wave=*/32, /*vgpr_granulated=*/1);
  ASSERT_TRUE(wave32.seeded);
  EXPECT_FALSE(wave32.patched) << "wave32 cannot carry shared VGPR blocks";

  // A source that asked for nothing is unaffected by any of the above.
  const auto none = patched(/*source_shared=*/0, /*target_wave=*/32, /*vgpr_granulated=*/63);
  ASSERT_TRUE(none.seeded);
  ASSERT_TRUE(none.patched);
  EXPECT_EQ(none.shared_count, 0u);
}

TEST(CodeObjectPatcher, AppliesArchSpecificWgpModeBit) {
  using namespace rocr::llvm::amdhsa;

  const auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  const Section *rodata = find_section(co, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(kernel_descriptor_t));

  auto patched_rsrc1 = [&](rj_code_arch_t arch) -> std::optional<uint32_t> {
    AmdGpuCodeObject local_co(image.data(), image.size());
    if (!local_co.is_valid())
      return std::nullopt;

    KdTranslation translation{};
    translation.descriptor_file_offset = rodata->sectionOffset();
    translation.target_wave_size = 32;

    CodeObjectPatcher patcher(local_co);
    if (!patcher.apply_kernel_descriptor_translation(translation, arch))
      return std::nullopt;

    const auto patched_image = patcher.emit();
    const auto kd =
        read_kernel_descriptor_for_test(patched_image.data() + translation.descriptor_file_offset);
    return kd.compute_pgm_rsrc1;
  };

  const auto cdna3_rsrc1 = patched_rsrc1(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_TRUE(cdna3_rsrc1.has_value());
  EXPECT_EQ(AMDHSA_BITS_GET(*cdna3_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE), 0u);
  EXPECT_EQ(AMDHSA_BITS_GET(*cdna3_rsrc1, COMPUTE_PGM_RSRC1_MEM_ORDERED), 0u);
  EXPECT_EQ(AMDHSA_BITS_GET(*cdna3_rsrc1, COMPUTE_PGM_RSRC1_FWD_PROGRESS), 0u);

  const auto rdna1_rsrc1 = patched_rsrc1(ROCJITSU_CODE_ARCH_RDNA1);
  ASSERT_TRUE(rdna1_rsrc1.has_value());
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna1_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna1_rsrc1, COMPUTE_PGM_RSRC1_MEM_ORDERED), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna1_rsrc1, COMPUTE_PGM_RSRC1_FWD_PROGRESS), 1u);

  const auto rdna2_rsrc1 = patched_rsrc1(ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(rdna2_rsrc1.has_value());
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna2_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna2_rsrc1, COMPUTE_PGM_RSRC1_MEM_ORDERED), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna2_rsrc1, COMPUTE_PGM_RSRC1_FWD_PROGRESS), 1u);

  const auto rdna3_rsrc1 = patched_rsrc1(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(rdna3_rsrc1.has_value());
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna3_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna3_rsrc1, COMPUTE_PGM_RSRC1_MEM_ORDERED), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna3_rsrc1, COMPUTE_PGM_RSRC1_FWD_PROGRESS), 1u);

  const auto rdna3_5_rsrc1 = patched_rsrc1(ROCJITSU_CODE_ARCH_RDNA3_5);
  ASSERT_TRUE(rdna3_5_rsrc1.has_value());
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna3_5_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna3_5_rsrc1, COMPUTE_PGM_RSRC1_MEM_ORDERED), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna3_5_rsrc1, COMPUTE_PGM_RSRC1_FWD_PROGRESS), 1u);

  const auto rdna4_rsrc1 = patched_rsrc1(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(rdna4_rsrc1.has_value());
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna4_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna4_rsrc1, COMPUTE_PGM_RSRC1_MEM_ORDERED), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*rdna4_rsrc1, COMPUTE_PGM_RSRC1_FWD_PROGRESS), 1u);

  const auto gfx1250_rsrc1 = patched_rsrc1(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(gfx1250_rsrc1.has_value());
  EXPECT_EQ(AMDHSA_BITS_GET(*gfx1250_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE), 0u);
  EXPECT_EQ(AMDHSA_BITS_GET(*gfx1250_rsrc1, COMPUTE_PGM_RSRC1_MEM_ORDERED), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(*gfx1250_rsrc1, COMPUTE_PGM_RSRC1_FWD_PROGRESS), 1u);
}

TEST(CodeObjectPatcher, PreservesPrivateEnableForZeroFixedDynamicStack) {
  using namespace rocr::llvm::amdhsa;

  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  const Section *source_rodata = find_section(source_layout, ".rodata");
  ASSERT_NE(source_rodata, nullptr);

  kernel_descriptor_t source_descriptor{};
  AMDHSA_BITS_SET(source_descriptor.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1);
  AMDHSA_BITS_SET(source_descriptor.kernel_code_properties, KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK,
                  1);
  write_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset(),
                                   source_descriptor);

  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  KdTranslation translation{};
  translation.descriptor_file_offset = source_rodata->sectionOffset();
  translation.target_private_size = 0;
  translation.target_wave_size = 64;

  CodeObjectPatcher patcher(code_object);
  ASSERT_TRUE(patcher.apply_kernel_descriptor_translation(translation, ROCJITSU_CODE_ARCH_CDNA3));
  const auto patched_image = patcher.emit();
  const auto patched_descriptor =
      read_kernel_descriptor_for_test(patched_image.data() + translation.descriptor_file_offset);

  EXPECT_EQ(patched_descriptor.private_segment_fixed_size, 0u);
  EXPECT_EQ(AMDHSA_BITS_GET(patched_descriptor.kernel_code_properties,
                            KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(patched_descriptor.compute_pgm_rsrc2,
                            COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
            1u);
}

TEST(CodeObjectPatcher, ReplaceTextPreservesLoadSegmentAlignment) {
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t padded_file_delta = 2 * load_align;

  auto image = make_minimal_amdgpu_elf_with_load_segments();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  const std::vector<uint32_t> text_words(load_align / sizeof(uint32_t) + 3, 0xDEADBEEFu);
  const auto *text_bytes = reinterpret_cast<const uint8_t *>(text_words.data());
  ASSERT_TRUE(patcher.replace_text({text_bytes, text_words.size() * sizeof(uint32_t)}));

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());

  const Section *text = patched.text_sections()[0];
  const Section *rodata = find_section(patched, ".rodata");
  ASSERT_NE(rodata, nullptr);
  EXPECT_EQ(find_section(patched, ".rj_translations"), nullptr);
  EXPECT_EQ(text->size(), text_words.size() * sizeof(uint32_t));

  EXPECT_EQ(rodata->sectionOffset(), text->sectionOffset() + 8u + padded_file_delta)
      << "file padding should preserve later PT_LOAD p_offset/p_vaddr congruence";
  EXPECT_EQ(rodata->vaddr(), text->vaddr() + 8 + load_align + padded_file_delta)
      << "later allocated sections must move after the expanded RX LOAD segment";

  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(patched_bytes, 0);
  ASSERT_EQ(ehdr.e_phnum, 2u);
  const auto phdrs = read_elf_array_for_test<Elf64_Phdr>(patched_bytes, ehdr.e_phoff, ehdr.e_phnum);
  const auto shdrs = read_elf_array_for_test<Elf64_Shdr>(patched_bytes, ehdr.e_shoff, ehdr.e_shnum);
  for (uint16_t i = 0; i < ehdr.e_phnum; ++i) {
    ASSERT_NE(phdrs[i].p_align, 0u);
    EXPECT_EQ(phdrs[i].p_offset % phdrs[i].p_align, phdrs[i].p_vaddr % phdrs[i].p_align)
        << "PT_LOAD " << i << " must remain loader-congruent";
  }
  EXPECT_EQ(phdrs[0].p_filesz, 8u + padded_file_delta);
  EXPECT_EQ(phdrs[0].p_memsz, 8u + padded_file_delta);
  EXPECT_EQ(phdrs[1].p_offset, rodata->sectionOffset());
  EXPECT_EQ(phdrs[1].p_vaddr, rodata->vaddr());
  EXPECT_EQ(phdrs[1].p_paddr, rodata->vaddr());
  EXPECT_LE(phdrs[0].p_vaddr + phdrs[0].p_memsz, phdrs[1].p_vaddr)
      << "expanded RX LOAD must not overlap the following LOAD in virtual memory";

  const auto symtab = std::find_if(shdrs.begin(), shdrs.end(), [](const Elf64_Shdr &shdr) {
    return shdr.sh_type == SHT_SYMTAB;
  });
  ASSERT_NE(symtab, shdrs.end());
  ASSERT_EQ(symtab->sh_entsize, sizeof(Elf64_Sym));
  ASSERT_GE(symtab->sh_size / symtab->sh_entsize, 3u);
  const auto symbols = read_elf_array_for_test<Elf64_Sym>(patched_bytes, symtab->sh_offset,
                                                          symtab->sh_size / symtab->sh_entsize);
  EXPECT_EQ(symbols[1].st_value, rodata->vaddr())
      << "defined symbols in moved sections must track the section virtual address";
  EXPECT_EQ(symbols[2].st_value, text->vaddr())
      << "symbols in unmoved .text must keep their original virtual address";
  EXPECT_EQ(symbols[2].st_size, text->size())
      << "function symbols spanning old .text must cover appended translated cave code";
}

TEST(CodeObjectPatcher, ReplaceTextRelocatesTextSymbolsWithExactOffsetMap) {
  auto image = make_minimal_amdgpu_elf_with_load_segments();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  CodeObjectPatcher patcher(co);
  const std::array<uint32_t, 4> text_words = {0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u};
  const auto *text_bytes = reinterpret_cast<const uint8_t *>(text_words.data());
  constexpr std::array<TextOffsetRelocation, 2> relocations = {
      TextOffsetRelocation{.source_offset = 0, .target_offset = 4},
      TextOffsetRelocation{.source_offset = 8, .target_offset = 16},
  };
  ASSERT_TRUE(patcher.replace_text({text_bytes, sizeof(text_words)}, relocations));

  const auto patched_bytes = patcher.emit();
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(patched_bytes, 0);
  const auto shdrs = read_elf_array_for_test<Elf64_Shdr>(patched_bytes, ehdr.e_shoff, ehdr.e_shnum);
  const auto symtab = std::find_if(shdrs.begin(), shdrs.end(), [](const Elf64_Shdr &shdr) {
    return shdr.sh_type == SHT_SYMTAB;
  });
  ASSERT_NE(symtab, shdrs.end());
  const auto symbols = read_elf_array_for_test<Elf64_Sym>(patched_bytes, symtab->sh_offset,
                                                          symtab->sh_size / symtab->sh_entsize);
  ASSERT_GE(symbols.size(), 3u);
  EXPECT_EQ(symbols[2].st_value, 0x1104u);
  EXPECT_EQ(symbols[2].st_size, 12u);
}

TEST(CodeObjectPatcher, AppendsNonAllocSectionWithoutMovingLoadableSegments) {
  auto image = make_minimal_amdgpu_elf_with_load_segments();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  const Section *original_text = find_section(co, ".text");
  const Section *original_rodata = find_section(co, ".rodata");
  ASSERT_NE(original_text, nullptr);
  ASSERT_NE(original_rodata, nullptr);
  const uint64_t original_text_offset = original_text->sectionOffset();
  const uint64_t original_text_vaddr = original_text->vaddr();
  const uint64_t original_rodata_offset = original_rodata->sectionOffset();
  const uint64_t original_rodata_vaddr = original_rodata->vaddr();

  CodeObjectPatcher patcher(co);
  const std::array<uint8_t, 8> payload = {'R', 'J', 'L', 'D', 'S', 1, 2, 3};
  ASSERT_TRUE(patcher.append_nonalloc_section(".rocjitsu.lds", payload, 8));

  const auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());

  const Section *metadata = find_section(patched, ".rocjitsu.lds");
  ASSERT_NE(metadata, nullptr);
  ASSERT_EQ(metadata->size(), payload.size());
  EXPECT_EQ(std::memcmp(metadata->data(), payload.data(), payload.size()), 0);
  EXPECT_EQ(metadata->flags() & SHF_ALLOC, 0u);

  const Section *patched_text = find_section(patched, ".text");
  const Section *patched_rodata = find_section(patched, ".rodata");
  ASSERT_NE(patched_text, nullptr);
  ASSERT_NE(patched_rodata, nullptr);
  EXPECT_EQ(patched_text->sectionOffset(), original_text_offset);
  EXPECT_EQ(patched_text->vaddr(), original_text_vaddr);
  EXPECT_EQ(patched_rodata->sectionOffset(), original_rodata_offset);
  EXPECT_EQ(patched_rodata->vaddr(), original_rodata_vaddr);
}

TEST(CodeObjectPatcher, ReplaceTextPreservesMovedKernelDescriptorEntryAddress) {
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t padded_file_delta = 2 * load_align;

  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());
  const auto *original_rodata = find_section(co, ".rodata");
  ASSERT_NE(original_rodata, nullptr);
  const int64_t original_entry_offset =
      read_kernel_descriptor_entry_offset(original_rodata->data());

  CodeObjectPatcher patcher(co);
  const std::vector<uint32_t> text_words(load_align / sizeof(uint32_t) + 3, 0xDEADBEEFu);
  const auto *text_bytes = reinterpret_cast<const uint8_t *>(text_words.data());
  ASSERT_TRUE(patcher.replace_text({text_bytes, text_words.size() * sizeof(uint32_t)}));

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *patched_text = patched.text_sections()[0];
  const auto *patched_rodata = find_section(patched, ".rodata");
  ASSERT_NE(patched_rodata, nullptr);

  const int64_t patched_entry_offset = read_kernel_descriptor_entry_offset(patched_rodata->data());
  EXPECT_EQ(patched_rodata->vaddr(), original_rodata->vaddr() + padded_file_delta);
  EXPECT_EQ(
      static_cast<uint64_t>(static_cast<int64_t>(patched_rodata->vaddr()) + patched_entry_offset),
      patched_text->vaddr())
      << "KERNEL_CODE_ENTRY_BYTE_OFFSET is relative to the descriptor address";
  EXPECT_EQ(patched_entry_offset, original_entry_offset - static_cast<int64_t>(padded_file_delta));
}

TEST(CodeObjectPatcher, ReplaceTextUpdatesRelocationOffsetsIntoMovedSections) {
  constexpr uint64_t load_align = 0x1000;

  auto image = make_minimal_amdgpu_elf_with_relocation_after_text();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  const std::vector<uint32_t> text_words(load_align / sizeof(uint32_t) + 3, 0xDEADBEEFu);
  const auto *text_bytes = reinterpret_cast<const uint8_t *>(text_words.data());
  ASSERT_TRUE(patcher.replace_text({text_bytes, text_words.size() * sizeof(uint32_t)}));

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

// The r_offset test above moves the relocation's storage. This one moves what the
// relocation names: an R_AMDGPU_RELATIVE64 addend pointing into .data resolves to
// load_bias + r_addend, so growing .text past its load alignment relocates .data
// and the addend has to follow it or the pointer lands short of the moved section.
TEST(CodeObjectPatcher, ReplaceTextShiftsRelativeAddendsIntoMovedSections) {
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t kTextVaddr = 0x1100;
  constexpr uint64_t kTextSize = 8;
  constexpr uint64_t kOriginalDataVaddr = kTextVaddr + kTextSize + load_align;

  auto image = make_minimal_amdgpu_elf_with_relocation_after_text(
      /*place_reloc_in_text=*/false, rocjitsu::R_AMDGPU_RELATIVE64,
      static_cast<int64_t>(kOriginalDataVaddr));
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  const std::vector<uint32_t> text_words(load_align / sizeof(uint32_t) + 3, 0xDEADBEEFu);
  const auto *text_bytes = reinterpret_cast<const uint8_t *>(text_words.data());
  ASSERT_TRUE(patcher.replace_text({text_bytes, text_words.size() * sizeof(uint32_t)}));

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());

  const auto *data = find_section(patched, ".data");
  const auto *rela_dyn = find_section(patched, ".rela.dyn");
  ASSERT_NE(data, nullptr);
  ASSERT_NE(rela_dyn, nullptr);
  ASSERT_EQ(rela_dyn->size(), sizeof(Elf64_Rela));
  ASSERT_GT(data->vaddr(), kOriginalDataVaddr) << "test only proves anything if .data moved";

  Elf64_Rela rela{};
  std::memcpy(&rela, rela_dyn->data(), sizeof(rela));
  EXPECT_EQ(static_cast<uint64_t>(rela.r_addend), data->vaddr())
      << "RELATIVE64 addend naming a section above .text must move with that section";
}

TEST(CodeObjectPatcher, DetectsRelocationsWithinText) {
  auto safe_image =
      make_minimal_amdgpu_elf_with_relocation_after_text(/*place_reloc_in_text=*/false);
  AmdGpuCodeObject safe_co(safe_image.data(), safe_image.size());
  ASSERT_TRUE(safe_co.is_valid());
  CodeObjectPatcher safe_patcher(safe_co);
  EXPECT_FALSE(safe_patcher.has_relocations_within_text());

  auto text_image =
      make_minimal_amdgpu_elf_with_relocation_after_text(/*place_reloc_in_text=*/true);
  AmdGpuCodeObject text_co(text_image.data(), text_image.size());
  ASSERT_TRUE(text_co.is_valid());
  CodeObjectPatcher text_patcher(text_co);
  EXPECT_TRUE(text_patcher.has_relocations_within_text());
}

TEST(AmdGpuElf, ClassifiesRocrRelocationSectionModesAndActions) {
  Elf64_Ehdr dynamic_header{};
  dynamic_header.e_type = ET_DYN;
  std::array<Elf64_Shdr, 3> sections{};
  sections[1].sh_type = SHT_PROGBITS;
  sections[1].sh_flags = SHF_ALLOC;
  sections[2].sh_type = SHT_NULL;
  Elf64_Shdr relocations{};
  relocations.sh_type = SHT_RELA;
  Elf64_Sym function{};
  function.st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  constexpr uint64_t abs64_info = (uint64_t{1} << 32) | R_AMDGPU_ABS64;

  const auto expect_actions = [&](RocrRelocationSectionMode expected_mode,
                                  RocrNoneRelocationAction expected_none,
                                  TextSymbolRelocationAction expected_abs64) {
    const auto mode = classify_rocr_relocation_section(dynamic_header, sections, relocations);
    EXPECT_EQ(mode, expected_mode);
    EXPECT_EQ(classify_rocr_none_relocation(mode, R_AMDGPU_NONE), expected_none);
    EXPECT_EQ(classify_text_symbol_relocation(mode, abs64_info,
                                              /*has_explicit_addend=*/true, /*addend=*/0, function),
              expected_abs64);
  };

  relocations.sh_info = SHN_UNDEF;
  expect_actions(RocrRelocationSectionMode::Dynamic, RocrNoneRelocationAction::Rejected,
                 TextSymbolRelocationAction::RequiresExecutableEntry);

  relocations.sh_info = 1;
  expect_actions(RocrRelocationSectionMode::ExplicitTarget, RocrNoneRelocationAction::Ignored,
                 TextSymbolRelocationAction::RequiresExecutableEntry);

  relocations.sh_info = 2;
  expect_actions(RocrRelocationSectionMode::Malformed, RocrNoneRelocationAction::Rejected,
                 TextSymbolRelocationAction::Unsupported);

  relocations.sh_info = static_cast<uint32_t>(sections.size());
  expect_actions(RocrRelocationSectionMode::Malformed, RocrNoneRelocationAction::Rejected,
                 TextSymbolRelocationAction::Unsupported);

  dynamic_header.e_type = ET_REL;
  relocations.sh_info = SHN_UNDEF;
  expect_actions(RocrRelocationSectionMode::NotApplicable, RocrNoneRelocationAction::Ignored,
                 TextSymbolRelocationAction::RequiresExecutableEntry);
}

TEST(CodeObjectPatcher, RejectsDynamicNoneBeforeInspectingPlaceOrSymbolMetadata) {
  const auto mutate_relocation = [](std::vector<uint8_t> image, auto mutate) {
    const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
    const auto rela_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
    EXPECT_NE(rela_it, shdrs.end());
    if (rela_it == shdrs.end())
      return image;
    const size_t rela_index = static_cast<size_t>(rela_it - shdrs.begin());
    auto relocation = read_elf_struct_for_test<Elf64_Rela>(image, rela_it->sh_offset);
    mutate(shdrs, rela_index, relocation);
    write_elf_struct_for_test(image, rela_it->sh_offset, relocation);
    write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
    return image;
  };
  const auto expect_rejected = [](std::string_view name, const std::vector<uint8_t> &image) {
    SCOPED_TRACE(name);
    AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    CodeObjectPatcher patcher(object);
    EXPECT_TRUE(patcher.has_rocr_rejected_none_relocation());
  };
  const auto expect_malformed = [](std::string_view name, const std::vector<uint8_t> &image) {
    SCOPED_TRACE(name);
    AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    CodeObjectPatcher patcher(object);
    EXPECT_TRUE(patcher.has_malformed_rocr_relocation_section());
    EXPECT_FALSE(patcher.has_rocr_rejected_none_relocation())
        << "malformed section metadata owns the loader rejection before record dispatch";
  };

  expect_rejected("text symbol", make_amdgpu_elf_with_symbol_relocation(
                                     kElfSymbolTypeFunc, /*defined_in_text=*/true,
                                     /*addend=*/0, R_AMDGPU_NONE));
  expect_rejected("data symbol", make_amdgpu_elf_with_symbol_relocation(
                                     kElfSymbolTypeObject, /*defined_in_text=*/false,
                                     /*addend=*/0, R_AMDGPU_NONE));

  const auto base = make_amdgpu_elf_with_symbol_relocation(
      kElfSymbolTypeFunc, /*defined_in_text=*/true, /*addend=*/0, R_AMDGPU_NONE);
  expect_rejected("symbol zero", mutate_relocation(base, [](auto &, size_t, auto &relocation) {
                    relocation.r_info = R_AMDGPU_NONE;
                  }));
  expect_rejected("out-of-range symbol",
                  mutate_relocation(base, [](auto &, size_t, auto &relocation) {
                    relocation.r_info = (static_cast<uint64_t>(99) << 32) | R_AMDGPU_NONE;
                  }));
  expect_rejected("invalid symbol table link",
                  mutate_relocation(base, [](auto &shdrs, size_t rela_index, auto &) {
                    shdrs[rela_index].sh_link = static_cast<uint32_t>(shdrs.size());
                  }));
  expect_rejected("place inside text",
                  mutate_relocation(base, [](auto &, size_t, auto &relocation) {
                    relocation.r_offset = 0x1100;
                  }));
  expect_rejected("place outside allocated storage",
                  mutate_relocation(base, [](auto &, size_t, auto &relocation) {
                    relocation.r_offset = std::numeric_limits<uint64_t>::max();
                    relocation.r_info = R_AMDGPU_NONE;
                    relocation.r_addend = 91;
                  }));
  expect_malformed("out-of-range explicit target",
                   mutate_relocation(base, [](auto &shdrs, size_t rela_index, auto &) {
                     shdrs[rela_index].sh_info = static_cast<uint32_t>(shdrs.size());
                   }));
  expect_malformed("explicit target naming SHT_NULL",
                   mutate_relocation(base, [](auto &shdrs, size_t rela_index, auto &) {
                     constexpr uint32_t target_index = 2;
                     shdrs[target_index].sh_type = SHT_NULL;
                     shdrs[rela_index].sh_info = target_index;
                   }));

  auto rel_image = mutate_relocation(base, [](auto &shdrs, size_t rela_index, auto &relocation) {
    const Elf64_Rel rel{.r_offset = relocation.r_offset, .r_info = relocation.r_info};
    relocation = {};
    std::memcpy(&relocation, &rel, sizeof(rel));
    shdrs[rela_index].sh_type = SHT_REL;
    shdrs[rela_index].sh_size = sizeof(Elf64_Rel);
    shdrs[rela_index].sh_entsize = sizeof(Elf64_Rel);
  });
  AmdGpuCodeObject rel_object(rel_image.data(), rel_image.size());
  ASSERT_TRUE(rel_object.is_valid());
  CodeObjectPatcher rel_patcher(rel_object);
  EXPECT_FALSE(rel_patcher.has_rocr_rejected_none_relocation())
      << "ROCr does not materialize SHT_REL as a relocation section";
  EXPECT_FALSE(rel_patcher.has_unsupported_relocation_to_text());

  AmdGpuCodeObject direct_object(base.data(), base.size());
  ASSERT_TRUE(direct_object.is_valid());
  CodeObjectPatcher direct_patcher(direct_object);
  constexpr std::array<uint32_t, 2> replacement = {0xBF800000u, 0xBF800000u};
  constexpr std::array<TextOffsetRelocation, 1> placements = {
      TextOffsetRelocation{.source_offset = 0, .target_offset = 0},
  };
  const auto replacement_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(replacement.data()), sizeof(replacement));
  EXPECT_FALSE(direct_patcher.replace_text(replacement_bytes, placements));
}

TEST(BinaryTranslatorE2E, RejectsRelocationTargetingText) {
  // A relocation whose place is inside .text cannot be remapped after DBT
  // relocates instructions, so translation must fail closed and leave the code
  // object unchanged rather than apply the relocation to the wrong bytes.
  auto image = make_minimal_amdgpu_elf_with_relocation_after_text(/*place_reloc_in_text=*/true);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::Legalization,
                                             "relocation place inside .text"));
}

TEST(CodeObjectPatcher, ClassifiesTextSymbolRelocationCompatibility) {
  auto dynamic_none = make_amdgpu_elf_with_symbol_relocation(
      kElfSymbolTypeFunc, /*defined_in_text=*/true, /*addend=*/0, R_AMDGPU_NONE);
  AmdGpuCodeObject dynamic_none_co(dynamic_none.data(), dynamic_none.size());
  ASSERT_TRUE(dynamic_none_co.is_valid());
  CodeObjectPatcher dynamic_none_patcher(dynamic_none_co);
  EXPECT_TRUE(dynamic_none_patcher.has_rocr_rejected_none_relocation())
      << "ROCr rejects NONE in a target-less dynamic relocation section";
  EXPECT_FALSE(dynamic_none_patcher.has_unsupported_relocation_to_text())
      << "dynamic NONE compatibility is a record-level, not a text-target, decision";

  auto explicit_none = dynamic_none;
  const auto explicit_none_ehdr = read_elf_struct_for_test<Elf64_Ehdr>(explicit_none, 0);
  auto explicit_none_shdrs = read_elf_array_for_test<Elf64_Shdr>(
      explicit_none, explicit_none_ehdr.e_shoff, explicit_none_ehdr.e_shnum);
  const auto explicit_none_data =
      std::ranges::find_if(explicit_none_shdrs, [](const auto &section) {
        return (section.sh_flags & SHF_ALLOC) != 0 && (section.sh_flags & SHF_EXECINSTR) == 0;
      });
  const auto explicit_none_rela = std::ranges::find_if(
      explicit_none_shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(explicit_none_data, explicit_none_shdrs.end());
  ASSERT_NE(explicit_none_rela, explicit_none_shdrs.end());
  explicit_none_rela->sh_info =
      static_cast<uint32_t>(explicit_none_data - explicit_none_shdrs.begin());
  write_bytes_for_test(explicit_none, explicit_none_ehdr.e_shoff, explicit_none_shdrs.data(),
                       explicit_none_shdrs.size() * sizeof(Elf64_Shdr));
  AmdGpuCodeObject explicit_none_co(explicit_none.data(), explicit_none.size());
  ASSERT_TRUE(explicit_none_co.is_valid());
  CodeObjectPatcher explicit_none_patcher(explicit_none_co);
  EXPECT_FALSE(explicit_none_patcher.has_rocr_rejected_none_relocation());
  EXPECT_FALSE(explicit_none_patcher.has_unsupported_relocation_to_text())
      << "ROCr skips explicit-target/static relocation sections for supported code objects";

  for (const uint32_t relocation_type : kSupportedExplicitSymbolRelocations) {
    for (uint8_t sym_type :
         {kElfSymbolTypeFunc, kElfSymbolTypeObject, kElfSymbolTypeAmdGpuHsaKernel}) {
      SCOPED_TRACE(relocation_type);
      auto text_image = make_amdgpu_elf_with_symbol_relocation(sym_type, /*defined_in_text=*/true,
                                                               /*addend=*/0, relocation_type);
      AmdGpuCodeObject text_co(text_image.data(), text_image.size());
      ASSERT_TRUE(text_co.is_valid()) << "sym_type=" << static_cast<int>(sym_type);
      CodeObjectPatcher text_patcher(text_co);
      EXPECT_FALSE(text_patcher.has_unsupported_relocation_to_text())
          << "zero-addend ordinary text sym_type=" << static_cast<int>(sym_type)
          << " can follow its relocated st_value";

      auto data_image = make_amdgpu_elf_with_symbol_relocation(sym_type, /*defined_in_text=*/false,
                                                               /*addend=*/0, relocation_type);
      AmdGpuCodeObject data_co(data_image.data(), data_image.size());
      ASSERT_TRUE(data_co.is_valid()) << "sym_type=" << static_cast<int>(sym_type);
      CodeObjectPatcher data_patcher(data_co);
      EXPECT_FALSE(data_patcher.has_unsupported_relocation_to_text())
          << "data-defined sym_type=" << static_cast<int>(sym_type) << " must be accepted";
    }
  }

  for (const uint8_t symbol_type :
       {kElfSymbolTypeNone, kElfSymbolTypeSection, kUnsupportedRuntimeSymbolType}) {
    SCOPED_TRACE(static_cast<uint32_t>(symbol_type));
    auto image = make_amdgpu_elf_with_symbol_relocation(symbol_type, /*defined_in_text=*/true);
    AmdGpuCodeObject co(image.data(), image.size());
    ASSERT_TRUE(co.is_valid());
    CodeObjectPatcher patcher(co);
    EXPECT_TRUE(patcher.has_unsupported_relocation_to_text())
        << "ROCr does not derive this text symbol's runtime address from st_value";
  }

  for (const uint32_t relocation_type : kUnsupportedExplicitSymbolRelocations) {
    SCOPED_TRACE(relocation_type);
    auto image = make_amdgpu_elf_with_symbol_relocation(
        kElfSymbolTypeFunc, /*defined_in_text=*/true, /*addend=*/0, relocation_type);
    AmdGpuCodeObject co(image.data(), image.size());
    ASSERT_TRUE(co.is_valid());
    CodeObjectPatcher patcher(co);
    EXPECT_TRUE(patcher.has_unsupported_relocation_to_text())
        << "a relocation that does not consume the symbol value cannot follow relocated .text";
  }

  auto null_target = make_amdgpu_elf_with_symbol_relocation(
      kElfSymbolTypeFunc, /*defined_in_text=*/true, /*addend=*/0, R_AMDGPU_ABS64);
  const auto null_target_ehdr = read_elf_struct_for_test<Elf64_Ehdr>(null_target, 0);
  auto null_target_shdrs = read_elf_array_for_test<Elf64_Shdr>(
      null_target, null_target_ehdr.e_shoff, null_target_ehdr.e_shnum);
  const auto null_target_data = std::ranges::find_if(null_target_shdrs, [](const auto &section) {
    return (section.sh_flags & SHF_ALLOC) != 0 && (section.sh_flags & SHF_EXECINSTR) == 0;
  });
  const auto null_target_rela = std::ranges::find_if(
      null_target_shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(null_target_data, null_target_shdrs.end());
  ASSERT_NE(null_target_rela, null_target_shdrs.end());
  null_target_data->sh_type = SHT_NULL;
  null_target_rela->sh_info = static_cast<uint32_t>(null_target_data - null_target_shdrs.begin());
  write_bytes_for_test(null_target, null_target_ehdr.e_shoff, null_target_shdrs.data(),
                       null_target_shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject null_target_co(null_target.data(), null_target.size());
  ASSERT_TRUE(null_target_co.is_valid());
  CodeObjectPatcher null_target_patcher(null_target_co);
  EXPECT_TRUE(null_target_patcher.has_malformed_rocr_relocation_section())
      << "ROCr cannot parse a nonzero SHT_NULL section before relocation dispatch";
  EXPECT_FALSE(null_target_patcher.has_unsupported_relocation_to_text())
      << "malformed section metadata is not an established .text reference";

  auto relative_image = make_amdgpu_elf_with_symbol_relocation(
      kElfSymbolTypeFunc, /*defined_in_text=*/true, /*addend=*/0, R_AMDGPU_RELATIVE64);
  AmdGpuCodeObject relative_co(relative_image.data(), relative_image.size());
  ASSERT_TRUE(relative_co.is_valid());
  CodeObjectPatcher relative_patcher(relative_co);
  EXPECT_FALSE(relative_patcher.has_unsupported_relocation_to_text())
      << "RELATIVE64 derives its target from the addend, not the redundant symbol value";

  auto addend_image = make_amdgpu_elf_with_symbol_relocation(
      kElfSymbolTypeFunc, /*defined_in_text=*/true, /*addend=*/4);
  AmdGpuCodeObject addend_co(addend_image.data(), addend_image.size());
  ASSERT_TRUE(addend_co.is_valid());
  CodeObjectPatcher addend_patcher(addend_co);
  EXPECT_TRUE(addend_patcher.has_unsupported_relocation_to_text())
      << "a named symbol plus addend can target an independently moved instruction";

  // Convert the fixture's explicit-addend RELA record into an implicit-addend
  // REL record. Without relocation-type-specific decoding, DBT cannot prove
  // what source text offset is stored at the relocation place.
  auto rel_image =
      make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeFunc, /*defined_in_text=*/true);
  const auto rel_ehdr = read_elf_struct_for_test<Elf64_Ehdr>(rel_image, 0);
  auto rel_shdrs =
      read_elf_array_for_test<Elf64_Shdr>(rel_image, rel_ehdr.e_shoff, rel_ehdr.e_shnum);
  const auto rela = std::ranges::find_if(
      rel_shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(rela, rel_shdrs.end());
  const size_t rela_index = static_cast<size_t>(rela - rel_shdrs.begin());
  const auto rela_record = read_elf_struct_for_test<Elf64_Rela>(rel_image, rela->sh_offset);
  const Elf64_Rel rel_record{.r_offset = rela_record.r_offset, .r_info = rela_record.r_info};
  std::memcpy(rel_image.data() + rela->sh_offset, &rel_record, sizeof(rel_record));
  rel_shdrs[rela_index].sh_type = SHT_REL;
  rel_shdrs[rela_index].sh_size = sizeof(Elf64_Rel);
  rel_shdrs[rela_index].sh_entsize = sizeof(Elf64_Rel);
  std::memcpy(rel_image.data() + rel_ehdr.e_shoff, rel_shdrs.data(),
              rel_shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject rel_co(rel_image.data(), rel_image.size());
  ASSERT_TRUE(rel_co.is_valid());
  CodeObjectPatcher rel_patcher(rel_co);
  EXPECT_TRUE(rel_patcher.has_unsupported_relocation_to_text())
      << "REL keeps an implicit addend that generic text-symbol remapping cannot inspect";
}

TEST(BinaryTranslatorE2E, RejectsMalformedRelocationWithNonzeroNullTarget) {
  auto image = make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeFunc, /*defined_in_text=*/true,
                                                      /*addend=*/0, R_AMDGPU_ABS64);
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
  const auto data_it = std::ranges::find_if(shdrs, [](const auto &section) {
    return (section.sh_flags & SHF_ALLOC) != 0 && (section.sh_flags & SHF_EXECINSTR) == 0;
  });
  const auto rela_it = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
  ASSERT_NE(data_it, shdrs.end());
  ASSERT_NE(rela_it, shdrs.end());
  data_it->sh_type = SHT_NULL;
  rela_it->sh_info = static_cast<uint32_t>(data_it - shdrs.begin());
  write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::Legalization,
                                   "malformed ROCr relocation-section metadata"));
}

TEST(CodeObjectPatcher, RejectsLoaderInvalidRocrRelocationsWithoutTextMap) {
  const auto expect_rejected = [](std::string_view name, auto mutate_sections) {
    SCOPED_TRACE(name);
    auto image = make_amdgpu_elf_with_symbol_relocation(
        kElfSymbolTypeFunc, /*defined_in_text=*/true, /*addend=*/0, R_AMDGPU_ABS64);
    const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    auto shdrs = read_elf_array_for_test<Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);
    const auto rela_it = std::ranges::find_if(
        shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_RELA; });
    ASSERT_NE(rela_it, shdrs.end());
    mutate_sections(shdrs, *rela_it);
    auto relocation = read_elf_struct_for_test<Elf64_Rela>(image, rela_it->sh_offset);
    relocation.r_offset = std::numeric_limits<uint64_t>::max();
    write_elf_struct_for_test(image, rela_it->sh_offset, relocation);
    rela_it->sh_link = static_cast<uint32_t>(shdrs.size());
    write_bytes_for_test(image, ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

    AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    CodeObjectPatcher patcher(object);
    EXPECT_TRUE(patcher.has_malformed_rocr_relocation_section());
    EXPECT_FALSE(patcher.has_rocr_rejected_none_relocation())
        << "malformed section metadata owns the loader rejection before record dispatch";
    EXPECT_FALSE(patcher.has_unsupported_relocation_to_text());

    std::vector<uint8_t> replacement(patcher.text_bytes().begin(), patcher.text_bytes().end());
    replacement.resize(replacement.size() + sizeof(uint32_t));
    EXPECT_FALSE(patcher.replace_text(replacement));
    EXPECT_TRUE(std::ranges::equal(patcher.image_bytes(), image));
  };

  expect_rejected("out-of-range sh_info", [](auto &sections, Elf64_Shdr &relocations) {
    relocations.sh_info = static_cast<uint32_t>(sections.size());
  });
  expect_rejected("nonzero SHT_NULL sh_info", [](auto &sections, Elf64_Shdr &relocations) {
    constexpr uint32_t target_index = 2;
    ASSERT_LT(target_index, sections.size());
    sections[target_index].sh_type = SHT_NULL;
    relocations.sh_info = target_index;
  });
  auto dynamic_none = make_amdgpu_elf_with_symbol_relocation(
      kElfSymbolTypeFunc, /*defined_in_text=*/true, /*addend=*/0, R_AMDGPU_NONE);
  AmdGpuCodeObject dynamic_none_object(dynamic_none.data(), dynamic_none.size());
  ASSERT_TRUE(dynamic_none_object.is_valid());
  CodeObjectPatcher dynamic_none_patcher(dynamic_none_object);
  EXPECT_FALSE(dynamic_none_patcher.has_malformed_rocr_relocation_section());
  EXPECT_TRUE(dynamic_none_patcher.has_rocr_rejected_none_relocation());
  std::vector<uint8_t> dynamic_none_replacement(dynamic_none_patcher.text_bytes().begin(),
                                                dynamic_none_patcher.text_bytes().end());
  dynamic_none_replacement.resize(dynamic_none_replacement.size() + sizeof(uint32_t));
  EXPECT_FALSE(dynamic_none_patcher.replace_text(dynamic_none_replacement));
  EXPECT_TRUE(std::ranges::equal(dynamic_none_patcher.image_bytes(), dynamic_none));
}

TEST(CodeObjectPatcher, AllowsAppendOnlyGrowthWithoutTextMapForMappingIncompatibleReference) {
  auto image = make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeFunc, /*defined_in_text=*/true,
                                                      /*addend=*/4, R_AMDGPU_ABS64);
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  CodeObjectPatcher patcher(object);
  ASSERT_FALSE(patcher.has_malformed_rocr_relocation_section());
  ASSERT_FALSE(patcher.has_rocr_rejected_none_relocation());
  ASSERT_TRUE(patcher.has_unsupported_relocation_to_text());

  std::vector<uint8_t> replacement(patcher.text_bytes().begin(), patcher.text_bytes().end());
  replacement.resize(replacement.size() + sizeof(uint32_t));
  EXPECT_TRUE(patcher.replace_text(replacement));
}

TEST(CodeObjectPatcher, RelocatesReferencedTextSymbolWithExactOffsetMap) {
  auto image = make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeFunc, /*defined_in_text=*/true);
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  CodeObjectPatcher patcher(object);

  const std::array<uint32_t, 4> expanded_text = {0xBF800000u, 0xBF800000u, 0xBF800000u,
                                                 0xBF800000u};
  constexpr std::array<TextOffsetRelocation, 2> mappings = {
      TextOffsetRelocation{.source_offset = 0, .target_offset = 4},
      TextOffsetRelocation{.source_offset = 8, .target_offset = 16},
  };
  const auto bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(expanded_text.data()), sizeof(expanded_text));
  ASSERT_TRUE(patcher.replace_text(bytes, mappings));

  const auto patched = patcher.emit();
  const auto ehdr = read_elf_struct_for_test<Elf64_Ehdr>(patched, 0);
  const auto shdrs = read_elf_array_for_test<Elf64_Shdr>(patched, ehdr.e_shoff, ehdr.e_shnum);
  const auto symtab = std::ranges::find_if(
      shdrs, [](const Elf64_Shdr &section) { return section.sh_type == SHT_SYMTAB; });
  ASSERT_NE(symtab, shdrs.end());
  const auto symbols = read_elf_array_for_test<Elf64_Sym>(patched, symtab->sh_offset,
                                                          symtab->sh_size / sizeof(Elf64_Sym));
  ASSERT_GE(symbols.size(), 2u);
  EXPECT_EQ(symbols[1].st_value, 0x1104u);
  EXPECT_EQ(symbols[1].st_size, 12u);
}

TEST(CodeObjectPatcher, RejectsReferencedTextSymbolWithoutExactOffsetMap) {
  auto image = make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeFunc, /*defined_in_text=*/true);
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  CodeObjectPatcher patcher(object);

  const std::array<uint32_t, 3> expanded_text = {0xBF800000u, 0xBF800000u, 0xBF800000u};
  constexpr std::array<TextOffsetRelocation, 1> incomplete_mapping = {
      TextOffsetRelocation{.source_offset = 4, .target_offset = 8},
  };
  const auto bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(expanded_text.data()), sizeof(expanded_text));
  EXPECT_FALSE(patcher.replace_text(bytes, incomplete_mapping));
}

TEST(CodeObjectPatcher, RejectsConflictingReferencedTextSymbolPlacements) {
  auto image = make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeFunc,
                                                      /*defined_in_text=*/true);
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  CodeObjectPatcher patcher(object);

  const std::array<uint32_t, 4> expanded_text = {0xBF800000u, 0xBF800000u, 0xBF800000u,
                                                 0xBF800000u};
  constexpr std::array<TextOffsetRelocation, 2> conflicting = {
      TextOffsetRelocation{.source_offset = 0, .target_offset = 4},
      TextOffsetRelocation{.source_offset = 0, .target_offset = 8},
  };
  const auto bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(expanded_text.data()), sizeof(expanded_text));
  EXPECT_FALSE(patcher.replace_text(bytes, conflicting));
}

TEST(BinaryTranslatorE2E, RejectsRelocationToTextSectionSymbol) {
  auto image =
      make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeSection, /*defined_in_text=*/true);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::Legalization,
                                             "unsupported relocation referencing .text"));
}

// Independently pin the AMDGPU ELF ABI values so the tests cannot mask a
// regression by constructing the same wrong relocation types the patcher checks
// for. The fixtures use the production constants, while these assertions anchor
// those constants to the authoritative ABI numbers.
static_assert(rocjitsu::R_AMDGPU_NONE == 0, "R_AMDGPU_NONE must be the AMDGPU ELF ABI value 0");
static_assert(rocjitsu::R_AMDGPU_ABS32_LO == 1,
              "R_AMDGPU_ABS32_LO must be the AMDGPU ELF ABI value 1");
static_assert(rocjitsu::R_AMDGPU_ABS32_HI == 2,
              "R_AMDGPU_ABS32_HI must be the AMDGPU ELF ABI value 2");
static_assert(rocjitsu::R_AMDGPU_ABS64 == 3, "R_AMDGPU_ABS64 must be the AMDGPU ELF ABI value 3");
static_assert(rocjitsu::R_AMDGPU_ABS32 == 6, "R_AMDGPU_ABS32 must be the AMDGPU ELF ABI value 6");
static_assert(rocjitsu::R_AMDGPU_RELATIVE64 == 13,
              "R_AMDGPU_RELATIVE64 must be the AMDGPU ELF ABI value 13");
static_assert(rocjitsu::kElfSymbolTypeAmdGpuHsaKernel == 10,
              "STT_AMDGPU_HSA_KERNEL must be the AMDGPU ELF ABI value 10");

// R_AMDGPU_RELATIVE64 carries symbol index 0, so the target is entirely in the
// explicit addend. DBT can remap an in-text addend through its exact offset map.
TEST(CodeObjectPatcher, SupportsRelative64AddendIntoText) {
  // Reference the production ABI constant so the test can never drift from it.
  constexpr uint32_t kRelative64 = rocjitsu::R_AMDGPU_RELATIVE64;
  constexpr int64_t kInTextAddend = 0x1100;    // == text_vaddr
  constexpr int64_t kOutOfTextAddend = 0x2108; // == data_vaddr

  auto in_text = make_amdgpu_elf_with_relative_relocation(kRelative64, kInTextAddend);
  AmdGpuCodeObject in_text_co(in_text.data(), in_text.size());
  ASSERT_TRUE(in_text_co.is_valid());
  CodeObjectPatcher in_text_patcher(in_text_co);
  EXPECT_FALSE(in_text_patcher.has_unsupported_relocation_to_text())
      << "RELATIVE64 addend inside .text has explicit remapping semantics";
  // The addend place is in .data, so the in-text-place guard must NOT be what
  // catches it -- the symbol-zero addend check is the code path under test.
  EXPECT_FALSE(in_text_patcher.has_relocations_within_text());

  auto out_of_text = make_amdgpu_elf_with_relative_relocation(kRelative64, kOutOfTextAddend);
  AmdGpuCodeObject out_co(out_of_text.data(), out_of_text.size());
  ASSERT_TRUE(out_co.is_valid());
  CodeObjectPatcher out_patcher(out_co);
  EXPECT_FALSE(out_patcher.has_unsupported_relocation_to_text())
      << "RELATIVE64 addend outside .text must be accepted";
}

TEST(BinaryTranslator, InlineExpansionAvoidsCaveBranchOverflow) {
  auto image = make_large_amdgpu_elf_with_waitcnt_entry();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(co);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  const bool diagnosed = std::any_of(
      result.diagnostics.begin(), result.diagnostics.end(),
      [](const TranslationDiagnostic &diagnostic) {
        return diagnostic.severity == DiagnosticSeverity::Error &&
               diagnostic.kind == DiagnosticKind::ResourceLimit &&
               diagnostic.message.find("branch range") != std::string::npos &&
               diagnostic.message.find("leaving code object unchanged") != std::string::npos;
      });
  EXPECT_FALSE(diagnosed);
}

TEST(BinaryTranslator, InlineExpansionIgnoresUnreachableTextTail) {
  auto image = make_large_amdgpu_elf_with_waitcnt_entry();
  AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());

  const auto *source_text = source_layout.text_sections()[0];
  auto *source_words = reinterpret_cast<uint32_t *>(image.data() + source_text->sectionOffset());
  source_words[1] = 0xBF810000u; // CDNA4 s_endpgm; the remaining large tail is unreachable.

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(co);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_EQ(find_section(translated, ".rj_translations"), nullptr);

  const auto expected_waitcnt = encode_waitcnt_gfx12(decode_waitcnt_gfx9(0));
  ASSERT_FALSE(expected_waitcnt.empty());
  const auto *text = translated.text_sections()[0];
  ASSERT_GE(text->size(), (expected_waitcnt.size() + 1) * sizeof(uint32_t));

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  for (size_t i = 0; i < expected_waitcnt.size(); ++i)
    EXPECT_EQ(target_words[i], expected_waitcnt[i]);
  EXPECT_EQ(target_words[expected_waitcnt.size()], build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(BinaryTranslator, SynthesizesKernargPreloadEntrySkipWindow) {
  auto image = make_large_amdgpu_elf_with_waitcnt_entry();
  AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());
  const auto *source_rodata = find_section(source_layout, ".rodata");
  ASSERT_NE(source_rodata, nullptr);
  ASSERT_GE(source_rodata->size(), sizeof(rocr::llvm::amdhsa::kernel_descriptor_t));

  auto source_kd = read_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd.kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  write_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset(), source_kd);

  const auto *source_text = source_layout.text_sections()[0];
  auto *source_words = reinterpret_cast<uint32_t *>(image.data() + source_text->sectionOffset());
  source_words[0] = build_s_branch(63, ROCJITSU_CODE_ARCH_CDNA4);
  source_words[64] = 0xBF810000u; // CDNA4 s_endpgm at the post-preload body entry.

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(co);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *text = translated.text_sections()[0];
  ASSERT_GT(text->size(), kKernargPreloadSkipBytes);
  const auto *target_words = reinterpret_cast<const uint32_t *>(text->data());
  EXPECT_EQ(target_words[0], build_s_branch(64, ROCJITSU_CODE_ARCH_CDNA3))
      << "old firmware enters the synthesized compatibility stub, which branches to the "
         "translated compatibility source entry";
  for (size_t i = 1; i < 64; ++i)
    EXPECT_EQ(target_words[i], build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3))
        << "the synthesized launch window must keep the compatible firmware entry exactly 256 "
           "bytes after the descriptor entry";
  EXPECT_EQ(target_words[64], build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA3))
      << "compatible firmware enters at descriptor entry + 256 and branches to the translated "
         "preloaded-kernarg source entry";
  EXPECT_EQ(target_words[65], build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3))
      << "the original compatibility source block is translated in the compact body";
  EXPECT_EQ(target_words[66], 0xBF810000u)
      << "the original compatible-firmware source entry is translated in the compact body";

  const auto *target_rodata = find_section(translated, ".rodata");
  ASSERT_NE(target_rodata, nullptr);
  ASSERT_GE(target_rodata->size(), sizeof(rocr::llvm::amdhsa::kernel_descriptor_t));
  const auto target_kd =
      read_kernel_descriptor_for_test(translated.image_data() + target_rodata->sectionOffset());
  EXPECT_EQ(target_kd.kernel_code_entry_byte_offset, source_kd.kernel_code_entry_byte_offset)
      << "the descriptor is redirected to the synthesized compatibility entry; compatible "
         "firmware still reaches the synthesized +256 entry by adding the ABI skip";
}

TEST(BinaryTranslator, SynthesizesKernargPreloadEntrySkipWindowWithDescriptorPrologue) {
  constexpr uint64_t kSourceEntryBytes = 512;
  constexpr size_t kSourceEntryWord = kSourceEntryBytes / sizeof(uint32_t);
  constexpr size_t kSourcePreloadEntryWord =
      (kSourceEntryBytes + kKernargPreloadSkipBytes) / sizeof(uint32_t);
  constexpr uint16_t kScalarOperandTtmpBase = 108;
  constexpr uint16_t kTtmpRdna4GridX = 9;

  std::vector<uint32_t> words(kSourcePreloadEntryWord + 1,
                              build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  words[0] = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  words[kSourceEntryWord] = build_s_branch(63, ROCJITSU_CODE_ARCH_CDNA4);
  words[kSourcePreloadEntryWord] = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  enable_workgroup_id_x_sgpr(image);

  AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());
  const auto *source_text = source_layout.text_sections()[0];
  const auto *source_rodata = find_section(source_layout, ".rodata");
  ASSERT_NE(source_rodata, nullptr);
  ASSERT_GE(source_rodata->size(), sizeof(rocr::llvm::amdhsa::kernel_descriptor_t));

  auto source_kd = read_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd.kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  source_kd.kernel_code_entry_byte_offset =
      static_cast<int64_t>(source_text->vaddr() + kSourceEntryBytes) -
      static_cast<int64_t>(source_rodata->vaddr());
  write_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset(), source_kd);

  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(co);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *text = translated.text_sections()[0];
  ASSERT_GT(text->size(), kKernargPreloadSkipBytes + 3 * sizeof(uint32_t));
  const auto *target_words = reinterpret_cast<const uint32_t *>(text->data());

  const uint32_t workgroup_id_x_prologue =
      build_s_mov_b32(0, kScalarOperandTtmpBase + kTtmpRdna4GridX, ROCJITSU_CODE_ARCH_RDNA4);
  const uint32_t prologue_delay = build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4);
  const auto expect_launch_stub = [&](size_t word_index, int16_t branch_offset) {
    EXPECT_EQ(target_words[word_index], workgroup_id_x_prologue)
        << "the synthesized kernarg-preload launch stub must materialize descriptor ABI SGPRs "
           "before branching into the relocated body";
    EXPECT_EQ(target_words[word_index + 1], prologue_delay)
        << "the synthesized launch stub must preserve scalar producer/consumer hazards";
    EXPECT_EQ(target_words[word_index + 2], build_s_branch(branch_offset, ROCJITSU_CODE_ARCH_RDNA4))
        << "the synthesized launch stub branches only after the descriptor ABI prologue";
  };
  expect_launch_stub(0, 64);
  expect_launch_stub(kKernargPreloadSkipBytes / sizeof(uint32_t), 1);

  EXPECT_EQ(target_words[67], build_s_branch(0, ROCJITSU_CODE_ARCH_RDNA4))
      << "the original compatibility source entry is translated in the compact body";
  EXPECT_EQ(target_words[68], build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4))
      << "the original preloaded-kernarg source entry is translated in the compact body";

  const auto *target_rodata = find_section(translated, ".rodata");
  ASSERT_NE(target_rodata, nullptr);
  ASSERT_GE(target_rodata->size(), sizeof(rocr::llvm::amdhsa::kernel_descriptor_t));
  const auto target_kd =
      read_kernel_descriptor_for_test(translated.image_data() + target_rodata->sectionOffset());
  const int64_t target_entry_text_offset = static_cast<int64_t>(target_rodata->vaddr()) +
                                           target_kd.kernel_code_entry_byte_offset -
                                           static_cast<int64_t>(text->vaddr());
  EXPECT_EQ(target_entry_text_offset, 0)
      << "the descriptor must be redirected from the moved source entry to the synthesized "
         "compatibility launch stub";
  EXPECT_NE(target_kd.kernel_code_entry_byte_offset, source_kd.kernel_code_entry_byte_offset)
      << "the source descriptor entry is deliberately nonzero, so this assertion proves the "
         "descriptor was repointed rather than passing because both entries were zero";
}

TEST(InstructionBuilder, PatchPcrelBranchOffsetInRange) {
  const uint32_t source_word = build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4);
  auto inst = decode_one(source_word, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(inst, nullptr);

  std::array<uint32_t, 1> words = {build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3)};
  EXPECT_TRUE(patch_pcrel_branch_offset(*inst, words, -8, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(words[0], build_s_branch(-2, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(InstructionBuilder, PatchPcrelBranchOffsetRejectsOutOfRange) {
  const uint32_t source_word = build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4);
  auto inst = decode_one(source_word, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(inst, nullptr);

  std::array<uint32_t, 1> words = {build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3)};
  EXPECT_FALSE(patch_pcrel_branch_offset(*inst, words, 32768LL * 4, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(words[0], build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));

  EXPECT_FALSE(patch_pcrel_branch_offset(*inst, words, -32769LL * 4, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(words[0], build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(InstructionBuilder, PatchPcrelBranchOffsetRejectsNonBranch) {
  const uint32_t source_word = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  auto inst = decode_one(source_word, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(inst, nullptr);

  std::array<uint32_t, 1> words = {build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3)};
  EXPECT_FALSE(patch_pcrel_branch_offset(*inst, words, 4, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(words[0], build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(InstructionBuilder, PatchPcrelBranchOffsetRejectsMisalignedDelta) {
  const uint32_t source_word = build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4);
  auto inst = decode_one(source_word, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(inst, nullptr);

  std::array<uint32_t, 1> words = {build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3)};
  EXPECT_FALSE(patch_pcrel_branch_offset(*inst, words, 2, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(words[0], build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
}

/// @brief One incomplete-consumer body with a second, independent PC builder.
///
/// @details The consumer's own fact is incomplete: one path builds a concrete PC
/// in s[8:9], the other reaches the setpc with the pair unconstrained. Whether
/// the translation may keep that dynamic transfer depends entirely on the second
/// builder in s[10:11]: @p bypass_builder_delta decides whether its value lands
/// on a relocatable block start or on an address DBT cannot move.
std::vector<uint32_t> make_incomplete_indirect_consumer_body(uint32_t bypass_builder_delta) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kBypassSreg = 10;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  return {
      pack_sopp(5, 5),                                     // 0x00: cbranch scc0 -> bypass at 0x18.
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x04: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x08: s_add_u32.
      40,                                                  // 0x0c: target delta -> 0x30.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x10: s_addc_u32.
      build_s_branch(5, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14 -> consumer at 0x2c.
      pack_sop1(0x1c, kBypassSreg, 0),                     // 0x18: bypass-path s_getpc_b64.
      pack_sop2(0, kBypassSreg, kBypassSreg, kLiteralOperand),     // 0x1c: s_add_u32.
      bypass_builder_delta,                                        // 0x20.
      pack_sop2(4, kBypassSreg + 1, kBypassSreg + 1, kInlineInt0), // 0x24: s_addc_u32.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                    // 0x28: closes the chain.
      pack_sop1(0x1d, 0, kPcSreg),                                 // 0x2c: joined consumer setpc.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                    // 0x30: builder target.
  };
}

TEST(BinaryTranslatorE2E, IncompleteIndirectConsumerFailsClosed) {
  // The bypass builder computes 0x1c + 0x100000, far outside .text. That value
  // is a PC-derived address DBT cannot rewrite to a relocated one, so the scope
  // still contains a potential stale PC. The unconstrained path into the setpc
  // may hold exactly such a value, so the whole translation must fail closed
  // rather than relocate only the known builder and leave the dynamic transfer
  // pointing at stale bytes.
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      make_incomplete_indirect_consumer_body(0x100000));
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  ASSERT_FALSE(result.diagnostics.empty());
  EXPECT_NE(result.diagnostics.front().message.find("unconstrained predecessor path"),
            std::string::npos)
      << result.diagnostics.front().message;
  EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
}

TEST(BinaryTranslatorE2E, IncompleteIndirectConsumerFailsClosedOnOpenPcBuilderChain) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kOpenSreg = 10;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // The s[10:11] getpc at 0x14 is the last instruction of its block, so nothing
  // in this block proves what its value becomes: this is exactly the shape whose
  // delta add lives in a successor, where an unmodeled write would leave the
  // original delta in place. The scope therefore cannot be proven free of stale
  // PC values and the incomplete consumer must keep failing closed.
  std::vector<uint32_t> words = {
      pack_sopp(5, 5),                                     // 0x00: cbranch scc0 -> bypass at 0x18.
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x04: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x08: s_add_u32.
      36,                                                  // 0x0c: target delta -> 0x2c.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x10: s_addc_u32 (chain closed).
      pack_sop1(0x1c, kOpenSreg, 0),                       // 0x14: bare s_getpc_b64 at block end.
      pack_sop2(0, kOpenSreg, kOpenSreg, kLiteralOperand), // 0x18: its s_add_u32, next block.
      12,                                                  // 0x1c.
      pack_sop2(4, kOpenSreg + 1, kOpenSreg + 1, kInlineInt0), // 0x20: s_addc_u32.
      pack_sop1(0x1d, 0, kPcSreg),                             // 0x24: joined consumer setpc.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                // 0x28: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                // 0x2c: builder target.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  ASSERT_FALSE(result.diagnostics.empty());
  EXPECT_NE(result.diagnostics.front().message.find("unconstrained predecessor path"),
            std::string::npos)
      << result.diagnostics.front().message;
  EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
}

TEST(BinaryTranslatorE2E, IncompleteIndirectConsumerTranslatesWhenScopeHasNoStalePcValues) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kBypassSreg = 10;

  // Same body, except the bypass builder now targets 0x1c + 0x14 = 0x30, the
  // same relocatable block start as the consumed builder. Every PC-derived value
  // in the scope can therefore be rewritten to its relocated address, so no path
  // into the setpc can deliver a stale one and the dynamic transfer is safe to
  // keep even though the consumer's own fact stays incomplete.
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      make_incomplete_indirect_consumer_body(0x14));
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
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  // Assert the relocated deltas structurally instead of by absolute index: for a
  // rewritten builder, getpc_next + delta must land on the relocated s_endpgm.
  const auto expect_builder_targets_endpgm = [&](uint16_t pc_sreg) {
    const uint32_t getpc = pack_sop1(0x1c, pc_sreg, 0);
    size_t found = 0;
    for (size_t i = 0; i + 2 < word_count; ++i) {
      if (target_words[i] != getpc)
        continue;
      ++found;
      const uint64_t target = (i + 1) * sizeof(uint32_t) + target_words[i + 2];
      ASSERT_EQ(target % sizeof(uint32_t), 0u);
      ASSERT_LT(target / sizeof(uint32_t), word_count);
      EXPECT_EQ(target_words[target / sizeof(uint32_t)], build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3))
          << "builder for s[" << pc_sreg << ":" << pc_sreg + 1
          << "] was not rewritten to its relocated target";
    }
    EXPECT_EQ(found, 1u);
  };
  expect_builder_targets_endpgm(kPcSreg);
  // The unconsumed builder must be relocated too: it is the producer whose
  // stale value the fail-closed path exists to prevent.
  expect_builder_targets_endpgm(kBypassSreg);

  EXPECT_NE(std::find(target_words, target_words + word_count, pack_sop1(0x1d, 0, kPcSreg)),
            target_words + word_count)
      << "an incomplete consumer must keep its dynamic transfer, not become a direct window";
}

TEST(BinaryTranslatorE2E, IncompleteIndirectConsumerFailsClosedOnNonContiguousBuilderRange) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kBypassSreg = 10;
  constexpr uint16_t kUnrelatedSreg = 20;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // The bypass builder in s[10:11] targets the relocatable block start at 0x34,
  // so on its own it would satisfy the whole-scope proof. But an unrelated
  // s_mov_b32 sits between its s_add_u32 and s_addc_u32: the pair survives that
  // move (it writes s20, not the pair), so the recovered builder range spans it.
  // patch_recovered_builder_fixups NOPs the whole range, which would erase the
  // move. The proof must fail closed on the non-contiguous range, keeping the
  // incomplete consumer's original refusal and leaving the object unchanged.
  std::vector<uint32_t> words = {
      pack_sopp(5, 6),                                     // 0x00: cbranch scc0 -> bypass at 0x1c.
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x04: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x08: s_add_u32.
      44,                                                  // 0x0c: target delta -> 0x34.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x10: s_addc_u32.
      build_s_branch(6, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14 -> consumer at 0x30.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x18: padding before bypass.
      pack_sop1(0x1c, kBypassSreg, 0),                     // 0x1c: bypass s_getpc_b64.
      pack_sop2(0, kBypassSreg, kBypassSreg, kLiteralOperand), // 0x20: s_add_u32.
      20,                                                      // 0x24: -> 0x20 + 20 = 0x34.
      build_s_mov_b32(kUnrelatedSreg, 0,
                      ROCJITSU_CODE_ARCH_CDNA4), // 0x28: unrelated write, in range.
      pack_sop2(4, kBypassSreg + 1, kBypassSreg + 1, kInlineInt0), // 0x2c: s_addc_u32.
      pack_sop1(0x1d, 0, kPcSreg),                                 // 0x30: joined consumer setpc.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                    // 0x34: builder target.
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  ASSERT_FALSE(result.diagnostics.empty());
  EXPECT_NE(result.diagnostics.front().message.find("unconstrained predecessor path"),
            std::string::npos)
      << result.diagnostics.front().message;
  EXPECT_EQ(result.elf_bytes, image)
      << "a non-contiguous builder range must fail closed, preserving the unrelated instruction";
}

TEST(BinaryTranslatorE2E, IncompleteConsumerFailsClosedAtKernargPreloadFirmwareEntry) {
  // The same incomplete-consumer body that translates when rooted at the ordinary
  // kernel entry (IncompleteIndirectConsumerTranslatesWhenScopeHasNoStalePcValues)
  // must instead fail closed when its scope root is the kernarg-preload firmware
  // entry (descriptor entry + 256). Before that entry runs, the command processor
  // copies caller-controlled kernarg words into user SGPRs, so the pair the
  // consumer reads on its unconstrained path can hold an original .text pointer
  // that no in-scope builder or relocation rewrites. The entry-state gate must not
  // treat the +256 entry as a safe root.
  constexpr size_t kPreloadEntryWord = kKernargPreloadSkipBytes / sizeof(uint32_t);

  // Descriptor entry (word 0) is a trivial ordinary path. The incomplete-consumer
  // body is placed at the +256 preload firmware entry, so it is reachable only
  // from that root.
  std::vector<uint32_t> words(kPreloadEntryWord, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  words[0] = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  const auto body = make_incomplete_indirect_consumer_body(0x14);
  words.insert(words.end(), body.begin(), body.end());

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  enable_workgroup_id_x_sgpr(image);

  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  const auto *source_rodata = find_section(source_layout, ".rodata");
  ASSERT_NE(source_rodata, nullptr);
  auto source_kd = read_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd.kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  write_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset(), source_kd);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  ASSERT_FALSE(result.diagnostics.empty());
  EXPECT_NE(result.diagnostics.front().message.find("unconstrained predecessor path"),
            std::string::npos)
      << result.diagnostics.front().message;
  EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
}

// The external-entry soundness gate (rejecting relocation-table-dispatched
// callees as unconstrained roots) is unit-tested directly in
// analysis/liveness_test.cpp (BinaryTranslatorInternal.ScopeRootsReject
// RelocationTableCallee): reaching its rejecting branch end-to-end would require
// a descriptor-bearing relocation-table dispatch fixture, whereas the pure gate
// exercises both the accept and reject paths precisely. The entry-state ACCEPT
// path is already covered end-to-end by
// IncompleteIndirectConsumerTranslatesWhenScopeHasNoStalePcValues above.

// A device function reached by more than one kernel scope is cloned once per scope, so each
// scope's branches resolve through its own placement map. A stored function pointer holds exactly
// one value and cannot choose between clones, so the translator nominates one clone canonical and
// rewrites every R_AMDGPU_RELATIVE64 addend to name it.
//
// That nomination is only sound while the clones are interchangeable, and a virtual-LDS sidecar
// scope is not interchangeable with its hardware-LDS twin: the sidecar clone is lowered against a
// different LDS model, so a pointer that reached it under a dispatch configured for the hardware
// model -- or the reverse -- would execute the wrong lowering. When two scopes emitting the same
// address-taken body disagree on that variant, the offset is recorded as variant-conflicted and
// withheld from the canonical map, which leaves the addend path fail-closed on it instead of
// silently answering with whichever clone happened to be nominated first.
//
// Sidecar variants exist only for the CDNA4 -> CDNA3 pair, so both tests below use it. They share
// one fixture and differ in a single input field -- kernel 0's group_segment_fixed_size -- which is
// what makes the refusal attributable to the variant disagreement rather than to the cloning.
namespace {

constexpr uint16_t kSharedHelperReturnSreg = 30;
constexpr size_t kSharedHelperEntryWord = 4;
/// @brief Static LDS beyond the CDNA3 host's 64 KiB per-workgroup limit, which is what makes the
/// hosting scope acquire a virtual-LDS sidecar variant alongside its hardware-LDS one.
constexpr uint32_t kOversizedGroupSegmentBytes = 105600u;

/// @brief Two kernels that both call one address-taken helper, so the helper is cloned per scope.
///
/// @details The helper is named by an R_AMDGPU_RELATIVE64 slot, which is what makes it
/// address-taken, and is reached by a direct call from each kernel, which is what makes every
/// scope emit its own clone rather than leaving it to be adopted by one.
///
/// @param oversized_kernel0_lds Give kernel 0 more static LDS than the host can dispatch, so its
/// scope gains a sidecar variant and the two variants disagree about the helper they both emit.
[[nodiscard]] std::vector<uint8_t>
make_shared_address_taken_helper_image(bool oversized_kernel0_lds) {
  const std::vector<uint32_t> words = {
      // word 0: kernel 0 entry, calls the helper at word 4.
      build_s_call_b64(kSharedHelperReturnSreg, 3, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // word 1
      // word 2: kernel 1 entry, calls the same helper.
      build_s_call_b64(kSharedHelperReturnSreg, 1, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                             // word 3
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                             // word 4: helper entry
      build_s_setpc_b64(kSharedHelperReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // word 5: helper return
  };

  auto image = test_support::make_minimal_amdgpu_elf_with_two_kernels_and_function_pointers(
      words, /*kernel1_entry_word=*/2, {{.offset_word = kSharedHelperEntryWord, .words = 2}});
  if (!oversized_kernel0_lds)
    return image;

  AmdGpuCodeObject layout(image.data(), image.size());
  const auto *rodata = find_section(layout, ".rodata");
  if (rodata == nullptr)
    return image;
  // Kernel 0's descriptor is first in .rodata, so this leaves kernel 1 on the hardware-LDS model.
  write_value_for_test<uint32_t>(
      image, rodata->sectionOffset() + offsetof(TestKernelDescriptor, group_segment_fixed_size),
      kOversizedGroupSegmentBytes);
  // The sidecar descriptor owns a wrapper ABI that needs an initialized workgroup id and room in
  // the User SGPRs, so the source descriptor has to declare both for the variant to be computable.
  uint32_t rsrc2 = 0;
  AMDHSA_BITS_SET(rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  AMDHSA_BITS_SET(rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  write_value_for_test<uint32_t>(
      image, rodata->sectionOffset() + offsetof(TestKernelDescriptor, compute_pgm_rsrc2), rsrc2);
  uint16_t properties = 0;
  AMDHSA_BITS_SET(properties,
                  rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  write_value_for_test<uint16_t>(
      image, rodata->sectionOffset() + offsetof(TestKernelDescriptor, kernel_code_properties),
      properties);
  return image;
}

/// @brief `.text`-relative offsets of every clone of the helper body in a translated image.
///
/// @details Each clone ends in the helper's `s_setpc_b64` return, so counting those locates the
/// clones without depending on where the layout placed them.
[[nodiscard]] std::vector<uint64_t>
translated_helper_entry_offsets(const AmdGpuCodeObject &object) {
  std::vector<uint64_t> entries;
  if (object.text_sections().empty())
    return entries;
  const Section &text = *object.text_sections().front();
  const auto *words = reinterpret_cast<const uint32_t *>(text.data());
  const size_t word_count = text.size() / sizeof(uint32_t);
  const uint32_t ret = build_s_setpc_b64(kSharedHelperReturnSreg, ROCJITSU_CODE_ARCH_CDNA3);
  for (size_t i = 1; i < word_count; ++i) {
    if (words[i] == ret)
      entries.push_back((i - 1) * sizeof(uint32_t));
  }
  return entries;
}

/// @brief The single `R_AMDGPU_RELATIVE64` addend the fixture's pointer slot carries.
[[nodiscard]] std::optional<uint64_t> only_relative64_addend(const AmdGpuCodeObject &object) {
  const auto *rela = find_section(object, ".rela.dyn");
  if (rela == nullptr || rela->size() != sizeof(Elf64_Rela))
    return std::nullopt;
  Elf64_Rela entry{};
  std::memcpy(&entry, rela->data(), sizeof(entry));
  if (elf_reloc_type(entry.r_info) != R_AMDGPU_RELATIVE64 || entry.r_addend < 0)
    return std::nullopt;
  return static_cast<uint64_t>(entry.r_addend);
}

} // namespace

// The control. Both kernels stay on the hardware-LDS model, so the two scopes that clone the
// helper agree on variant, nothing is recorded as conflicted, and the stored pointer is answered
// from the canonical placement -- the clone the lowest-offset scope emitted. Without this the
// refusal below could be passing because the body is cloned at all rather than because the clones
// are not equivalent.
TEST(BinaryTranslatorE2E, AgreeingScopesResolveAnAddressTakenCloneThroughCanonicalPlacement) {
  const auto image = make_shared_address_taken_helper_image(/*oversized_kernel0_lds=*/false);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto clones = translated_helper_entry_offsets(translated);
  ASSERT_EQ(clones.size(), 2u) << "each kernel scope emits its own clone of the shared helper";

  const auto addend = only_relative64_addend(translated);
  ASSERT_TRUE(addend.has_value());
  const uint64_t text_vaddr = translated.text_sections().front()->vaddr();
  ASSERT_GE(*addend, text_vaddr);
  EXPECT_EQ(*addend - text_vaddr, clones.front())
      << "the stored pointer names the canonical clone, not the caller-local one";
}

// The same fixture with kernel 0 given more static LDS than the host can dispatch. Its scope now
// has a hardware-LDS variant and a virtual-LDS sidecar variant, both emitting the shared helper
// and disagreeing about how LDS in it is lowered, so no single clone can answer for the pointer.
// The object must be refused whole rather than translated with the addend pointing at whichever
// clone was nominated first: that artifact would look correct and would run the hardware-LDS
// lowering under a sidecar dispatch.
TEST(BinaryTranslatorE2E, VariantConflictedAddressTakenCloneRefusesInsteadOfNamingOneClone) {
  const auto image = make_shared_address_taken_helper_image(/*oversized_kernel0_lds=*/true);
  AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  const auto result = translator.translate(source);
  EXPECT_FALSE(result.ok())
      << "a code address whose clones disagree on kernel-scope variant has no single answer";
  EXPECT_TRUE(has_error_containing(result, DiagnosticKind::ResourceLimit,
                                   "relocated .text could not be materialized safely"))
      << (result.diagnostics.empty() ? "" : result.diagnostics.front().message);
  EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
}

} // namespace
} // namespace rocjitsu
