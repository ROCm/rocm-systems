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
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/encoding_fields.h"
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
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/dbt/waitcnt_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/code/rj_code_internal.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/builders.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/register_set.h"
#include "support/elf_test_support.h"

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
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

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

template <typename T>
T read_elf_struct_for_test(const std::vector<uint8_t> &image, uint64_t offset) {
  T value{};
  assert(offset <= image.size());
  assert(sizeof(T) <= image.size() - offset);
  std::memcpy(&value, image.data() + offset, sizeof(value));
  return value;
}

template <typename T>
std::vector<T> read_elf_array_for_test(const std::vector<uint8_t> &image, uint64_t offset,
                                       size_t count) {
  std::vector<T> values(count);
  assert(offset <= image.size());
  assert(count <= (image.size() - offset) / sizeof(T));
  std::memcpy(values.data(), image.data() + offset, count * sizeof(T));
  return values;
}

std::optional<Elf64_Sym> find_elf_symbol_for_test(const std::vector<uint8_t> &image,
                                                  std::string_view name) {
  const auto range_is_in_image = [&](uint64_t offset, uint64_t size) {
    const uint64_t image_size = image.size();
    return offset <= image_size && size <= image_size - offset;
  };
  if (!range_is_in_image(0, sizeof(Elf64_Ehdr)))
    return std::nullopt;

  const auto header = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  const uint64_t section_table_size = static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr);
  if (!range_is_in_image(header.e_shoff, section_table_size))
    return std::nullopt;

  const auto sections = read_elf_array_for_test<Elf64_Shdr>(image, header.e_shoff, header.e_shnum);
  for (const Elf64_Shdr &symtab : sections) {
    if ((symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM) ||
        symtab.sh_entsize != sizeof(Elf64_Sym) || symtab.sh_size % sizeof(Elf64_Sym) != 0 ||
        symtab.sh_link >= sections.size() || !range_is_in_image(symtab.sh_offset, symtab.sh_size)) {
      continue;
    }
    const Elf64_Shdr &strtab = sections[symtab.sh_link];
    if (strtab.sh_type != SHT_STRTAB || !range_is_in_image(strtab.sh_offset, strtab.sh_size))
      continue;
    const auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab.sh_offset,
                                                            symtab.sh_size / sizeof(Elf64_Sym));
    for (const Elf64_Sym &symbol : symbols) {
      if (symbol.st_name >= strtab.sh_size)
        continue;
      const char *begin =
          reinterpret_cast<const char *>(image.data() + strtab.sh_offset + symbol.st_name);
      const size_t remaining = strtab.sh_size - symbol.st_name;
      const auto *end = static_cast<const char *>(std::memchr(begin, '\0', remaining));
      if (end != nullptr && std::string_view(begin, static_cast<size_t>(end - begin)) == name)
        return symbol;
    }
  }
  return std::nullopt;
}

template <typename T>
void write_elf_struct_for_test(std::vector<uint8_t> &image, uint64_t offset, const T &value) {
  assert(offset <= image.size());
  assert(sizeof(T) <= image.size() - offset);
  std::memcpy(image.data() + offset, &value, sizeof(value));
}

void write_bytes_for_test(std::vector<uint8_t> &image, uint64_t offset, const void *src,
                          size_t size) {
  assert(offset <= image.size());
  assert(size <= image.size() - offset);
  std::memcpy(image.data() + offset, src, size);
}

template <typename T>
void write_value_for_test(std::vector<uint8_t> &image, uint64_t offset, T value) {
  write_bytes_for_test(image, offset, &value, sizeof(value));
}

using TestKernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;
constexpr size_t kKernelDescriptorSize = sizeof(TestKernelDescriptor);
constexpr size_t kKernelDescriptorEntryOffset =
    offsetof(TestKernelDescriptor, kernel_code_entry_byte_offset);
constexpr uint64_t kKernargPreloadSkipBytes = 256;

void write_kernel_descriptor_entry_offset(void *descriptor, int64_t entry_offset) {
  auto *bytes = static_cast<uint8_t *>(descriptor);
  std::memcpy(bytes + kKernelDescriptorEntryOffset, &entry_offset, sizeof(entry_offset));
}

int64_t read_kernel_descriptor_entry_offset(const void *descriptor) {
  const auto *bytes = static_cast<const uint8_t *>(descriptor);
  int64_t entry_offset = 0;
  std::memcpy(&entry_offset, bytes + kKernelDescriptorEntryOffset, sizeof(entry_offset));
  return entry_offset;
}

TestKernelDescriptor read_kernel_descriptor_for_test(const void *descriptor) {
  TestKernelDescriptor kd{};
  std::memcpy(&kd, descriptor, sizeof(kd));
  return kd;
}

void write_kernel_descriptor_for_test(void *descriptor, const TestKernelDescriptor &kd) {
  std::memcpy(descriptor, &kd, sizeof(kd));
}

std::vector<uint8_t> make_kernel_descriptor_bytes(int64_t entry_offset) {
  std::vector<uint8_t> descriptor(kKernelDescriptorSize, 0);
  write_kernel_descriptor_entry_offset(descriptor.data(), entry_offset);
  return descriptor;
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

std::vector<uint8_t> make_minimal_amdgpu_elf_with_descriptor_after_text(
    const std::vector<uint32_t> &text_words,
    std::optional<size_t> text_function_words = std::nullopt) {
  if (text_function_words && *text_function_words > text_words.size())
    throw std::invalid_argument("text function extent exceeds .text fixture");
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = kKernelDescriptorSize;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = add_elf_name(strtab, "kernel.kd");
  const uint32_t text_symbol_name = text_function_words ? add_elf_name(strtab, "kernel") : 0;

  // The kernel descriptor requires 8-byte alignment (tests reinterpret_cast the
  // .rodata bytes to TestKernelDescriptor). An odd text_words count leaves
  // text_offset + text_size only 4-aligned, so pad up to 8. text_offset and
  // text_vaddr are both 8-aligned and differ by a multiple of load_align, so
  // padding both keeps the PT_LOAD p_offset == p_vaddr (mod p_align) congruence.
  const uint64_t rodata_offset = align_up_for_test(text_offset + text_size, 8);
  const uint64_t rodata_vaddr = align_up_for_test(text_vaddr + text_size, 8) + load_align;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  const size_t sym_count = text_function_words ? 3 : 2;
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

  const auto descriptor = make_kernel_descriptor_bytes(static_cast<int64_t>(text_vaddr) -
                                                       static_cast<int64_t>(rodata_vaddr));
  std::memcpy(image.data() + rodata_offset, descriptor.data(), descriptor.size());
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::vector<Elf64_Sym> syms(sym_count);
  syms[1].st_name = kd_symbol_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = kKernelDescriptorSize;
  if (text_function_words) {
    syms[2].st_name = text_symbol_name;
    syms[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
    syms[2].st_shndx = 1;
    syms[2].st_value = text_vaddr;
    syms[2].st_size = *text_function_words * sizeof(uint32_t);
  }
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

template <size_t N>
std::vector<uint8_t>
make_gfx1250_image_with_live_sgprs(const std::array<uint32_t, N> &instruction_words,
                                   uint16_t live_sgprs) {
  if (live_sgprs > REGISTER_SET_ALLOCATABLE_SGPRS) {
    ADD_FAILURE() << "liveness wall exceeds the allocatable SGPR range";
    return {};
  }
  constexpr uint16_t kGfx1250M0Operand = 125;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  std::vector<uint32_t> words(instruction_words.begin(), instruction_words.end());
  for (uint16_t sgpr = 0; sgpr < live_sgprs; ++sgpr) {
    words.push_back(gfx1250::build_sop1(gfx1250::kSMovB32Sop1, {.ssrc0 = static_cast<uint8_t>(sgpr),
                                                                .sdst = kGfx1250M0Operand})[0]);
  }
  words.push_back(kGfx1250SEndpgm);
  return make_minimal_amdgpu_elf_with_descriptor_after_text(words);
}

std::unique_ptr<Instruction> decode_one(uint32_t word, rj_code_arch_t arch) {
  auto decoder = Decoder::create(arch);
  if (!decoder)
    return nullptr;
  return std::unique_ptr<Instruction>(decoder->decode(&word));
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

std::vector<uint8_t> make_minimal_amdgpu_elf_with_two_kernel_descriptors(
    const std::vector<uint32_t> &text_words = {0xBF810000u, 0xBF810000u}) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = 2 * kKernelDescriptorSize;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kernel0_name = add_elf_name(strtab, "kernel0.kd");
  const uint32_t kernel1_name = add_elf_name(strtab, "kernel1.kd");

  // The kernel descriptors require 8-byte alignment (tests reinterpret_cast the
  // .rodata bytes to TestKernelDescriptor). An odd text_words count leaves
  // text_offset + text_size only 4-aligned, so pad up to 8. text_offset and
  // text_vaddr are both 8-aligned and differ by a multiple of load_align, so
  // padding both keeps the PT_LOAD p_offset == p_vaddr (mod p_align) congruence.
  const uint64_t rodata_offset = align_up_for_test(text_offset + text_size, 8);
  const uint64_t rodata_vaddr = align_up_for_test(text_vaddr + text_size, 8) + load_align;
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

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  std::vector<uint8_t> descriptors(rodata_size, 0);
  write_kernel_descriptor_entry_offset(descriptors.data(), static_cast<int64_t>(text_vaddr) -
                                                               static_cast<int64_t>(rodata_vaddr));
  write_kernel_descriptor_entry_offset(
      descriptors.data() + kKernelDescriptorSize,
      static_cast<int64_t>(text_vaddr + sizeof(uint32_t)) -
          static_cast<int64_t>(rodata_vaddr + kKernelDescriptorSize));
  std::memcpy(image.data() + rodata_offset, descriptors.data(), descriptors.size());
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = kernel0_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = kKernelDescriptorSize;
  syms[2].st_name = kernel1_name;
  syms[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[2].st_shndx = 2;
  syms[2].st_value = rodata_vaddr + kKernelDescriptorSize;
  syms[2].st_size = kKernelDescriptorSize;
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

std::vector<uint8_t>
make_minimal_amdgpu_elf_with_relocation_after_text(bool place_reloc_in_text = false) {
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
// chosen type, defined either in .text or .data. Ordinary zero-addend text
// symbols can follow their relocated st_value; section symbols and nonzero
// addends need relocation-specific reconstruction and remain unsupported.
std::vector<uint8_t> make_amdgpu_elf_with_symbol_relocation(uint8_t sym_type, bool defined_in_text,
                                                            int64_t addend = 0) {
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
  rela.r_info = (static_cast<uint64_t>(1) << 32) | R_AMDGPU_ABS64;
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

std::vector<uint8_t> make_large_amdgpu_elf_with_waitcnt_entry() {
  constexpr uint64_t rodata_offset = 0x100;
  constexpr uint64_t rodata_vaddr = 0x100;
  constexpr uint64_t text_offset = 0x1000;
  constexpr uint64_t text_vaddr = 0x1000;
  constexpr uint64_t text_size = 0x21000;
  constexpr uint64_t rodata_size = kKernelDescriptorSize;
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

  write_bytes_for_test(image, offsetof(Elf64_Ehdr, e_ident), EI_MAGIC, EI_MAGIC_SIZE);
  image[offsetof(Elf64_Ehdr, e_ident) + EI_CLASS] = ELFCLASS64;
  image[offsetof(Elf64_Ehdr, e_ident) + EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_type), ET_DYN);
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_machine), EM_AMDGPU);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_version), 1);
  write_value_for_test<uint64_t>(image, offsetof(Elf64_Ehdr, e_phoff), sizeof(Elf64_Ehdr));
  write_value_for_test<uint64_t>(image, offsetof(Elf64_Ehdr, e_shoff), shoff);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX950);
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_ehsize), sizeof(Elf64_Ehdr));
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_phentsize), sizeof(Elf64_Phdr));
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_phnum), phdr_count);
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_shentsize), sizeof(Elf64_Shdr));
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_shnum), section_count);
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_shstrndx), 5);

  const uint64_t phdr0 = sizeof(Elf64_Ehdr);
  write_value_for_test<uint32_t>(image, phdr0 + offsetof(Elf64_Phdr, p_type), PT_LOAD);
  write_value_for_test<uint32_t>(image, phdr0 + offsetof(Elf64_Phdr, p_flags), 0x4); // PF_R
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_offset), rodata_offset);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_vaddr), rodata_vaddr);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_paddr), rodata_vaddr);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_filesz), rodata_size);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_memsz), rodata_size);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_align), load_align);

  const uint64_t phdr1 = phdr0 + sizeof(Elf64_Phdr);
  write_value_for_test<uint32_t>(image, phdr1 + offsetof(Elf64_Phdr, p_type), PT_LOAD);
  write_value_for_test<uint32_t>(image, phdr1 + offsetof(Elf64_Phdr, p_flags), 0x5); // PF_R | PF_X
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_offset), text_offset);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_vaddr), text_vaddr);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_paddr), text_vaddr);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_filesz), text_size);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_memsz), text_size);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_align), load_align);

  const auto descriptor = make_kernel_descriptor_bytes(static_cast<int64_t>(text_vaddr) -
                                                       static_cast<int64_t>(rodata_vaddr));
  std::memcpy(image.data() + rodata_offset, descriptor.data(), descriptor.size());

  std::vector<uint32_t> text_words(text_size / sizeof(uint32_t),
                                   build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = cdna4::build_sopp(cdna4::kSWaitcntSopp)[0]; // Expands on RDNA4.
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  const uint64_t sym1 = symtab_offset + sizeof(Elf64_Sym);
  write_value_for_test<uint32_t>(image, sym1 + offsetof(Elf64_Sym, st_name), kd_symbol_name);
  write_value_for_test<unsigned char>(image, sym1 + offsetof(Elf64_Sym, st_info),
                                      elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject));
  write_value_for_test<uint16_t>(image, sym1 + offsetof(Elf64_Sym, st_shndx), 1);
  write_value_for_test<uint64_t>(image, sym1 + offsetof(Elf64_Sym, st_value), rodata_vaddr);
  write_value_for_test<uint64_t>(image, sym1 + offsetof(Elf64_Sym, st_size), kKernelDescriptorSize);

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  const auto write_shdr = [&](uint64_t index, uint32_t name, uint32_t type, uint64_t flags,
                              uint64_t addr, uint64_t offset, uint64_t size, uint32_t link,
                              uint32_t info, uint64_t addralign, uint64_t entsize) {
    const uint64_t base = shoff + index * sizeof(Elf64_Shdr);
    write_value_for_test<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_name), name);
    write_value_for_test<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_type), type);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_flags), flags);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_addr), addr);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_offset), offset);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_size), size);
    write_value_for_test<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_link), link);
    write_value_for_test<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_info), info);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_addralign), addralign);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_entsize), entsize);
  };
  write_shdr(1, rodata_name, SHT_PROGBITS, SHF_ALLOC, rodata_vaddr, rodata_offset, rodata_size, 0,
             0, 64, 0);
  write_shdr(2, text_name, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_vaddr, text_offset,
             text_size, 0, 0, 256, 0);
  write_shdr(3, symtab_name, SHT_SYMTAB, 0, 0, symtab_offset, sym_count * sizeof(Elf64_Sym), 4, 1,
             8, sizeof(Elf64_Sym));
  write_shdr(4, strtab_name, SHT_STRTAB, 0, 0, strtab_offset, strtab.size(), 0, 0, 1, 0);
  write_shdr(5, shstrtab_name, SHT_STRTAB, 0, 0, shstrtab_offset, shstrtab.size(), 0, 0, 1, 0);
  return image;
}

const Section *find_section(const CodeObject &co, std::string_view name) {
  for (const auto &section : co.all_sections()) {
    if (section->name() == name)
      return section.get();
  }
  return nullptr;
}

std::optional<uint64_t> loaded_vaddr_to_file_offset(const std::vector<uint8_t> &image,
                                                    uint64_t vaddr) {
  return test_support::loaded_vaddr_to_file_offset(image, vaddr);
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

void enable_kernarg_segment_ptr_sgpr(std::vector<uint8_t> &image, uint32_t kernarg_size = 16) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

  AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(KD));

  KD kd{};
  std::memcpy(&kd, image.data() + rodata->sectionOffset(), sizeof(kd));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(kd.kernel_code_properties,
                  rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  kd.kernarg_size = kernarg_size;
  std::memcpy(image.data() + rodata->sectionOffset(), &kd, sizeof(kd));
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
  src.op = cdna4::kSCmovB64Sop1;
  src.encoding = cdna4::encoding::kSop1;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP1, w0, 0, 0, rdna4::kSBrevB64Sop1);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 42);
  EXPECT_EQ(dst.sdst, 17);
  EXPECT_EQ(dst.op, rdna4::kSBrevB64Sop1);
  EXPECT_EQ(dst.encoding, rdna4::encoding::kSop1);
}

TEST(EncodingTranslator, Sop2PreservesRegisters) {
  cdna4::Sop2MachineInst src{};
  src.ssrc0 = 10;
  src.ssrc1 = 20;
  src.sdst = 30;
  src.op = cdna4::kSMinU32Sop2;
  src.encoding = 0x2;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP2, w0, 0, 0, rdna4::kSMinU32Sop2);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop2MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 10);
  EXPECT_EQ(dst.ssrc1, 20);
  EXPECT_EQ(dst.sdst, 30);
  EXPECT_EQ(dst.op, rdna4::kSMinU32Sop2);
}

TEST(InstructionBuilder, Sop2SetsEncodingPrefix) {
  const uint32_t word = build_s_lshl_b32(1, 2, 3, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ((word >> 30) & 0x3u, 0x2u);
}

TEST(EncodingTranslator, SoppPreservesSimm16) {
  cdna4::SoppMachineInst src{};
  src.simm16 = 0xABCD;
  src.op = cdna4::kSWaitcntSopp;
  src.encoding = cdna4::encoding::kSopp;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOPP, w0, 0, 0, rdna4::kSWaitcntSopp);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::SoppMachineInst>(result.words[0]);
  EXPECT_EQ(dst.simm16, 0xABCD);
  EXPECT_EQ(dst.op, rdna4::kSWaitcntSopp);
}

TEST(EncodingTranslator, SmemRemapsCoherency) {
  cdna4::SmemMachineInst src{};
  src.sbase = 5;
  src.sdata = 3;
  src.glc = 1;
  src.nv = 0;
  src.op = cdna4::kSLoadDwordSmem;
  src.offset = 0x100;
  src.soffset = 0x7F;
  src.encoding = 0x3D;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SMEM, words[0], words[1], 0,
                                                                  rdna4::kSLoadB32Smem);

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
  src.op = cdna4::kVCmpGtF64Vop3;
  src.encoding = 0x35;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_VOP3, words[0], words[1], 0,
                                                                  rdna4::kVCmpGtF64Vop3);

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
  src.op = cdna4::kVAddF32Vop2;
  src.encoding = 0; // GFX9-family VOP2 prefix.
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result =
      cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3(kEnc_VOP2, w0, 0, 0, cdna3::kVAddF32Vop2);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<cdna3::Vop2MachineInst>(result.words[0]);
  EXPECT_EQ(dst.src0, 3);
  EXPECT_EQ(dst.vsrc1, 4);
  EXPECT_EQ(dst.vdst, 5);
  EXPECT_EQ(dst.op, cdna3::kVAddF32Vop2);
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
  src.op = cdna4::kSNotB32Sop1;
  src.encoding = cdna4::encoding::kSop1;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto fields = cdna4_to_rdna4::decode_sop1_cdna4(w0);
  EXPECT_EQ(fields.ssrc0, 55u);
  EXPECT_EQ(fields.sdst, 33u);
  EXPECT_EQ(fields.op, cdna4::kSNotB32Sop1);

  auto result = cdna4_to_rdna4::encode_sop1_rdna4(fields, rdna4::kSBrevB32Sop1);
  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 55);
  EXPECT_EQ(dst.sdst, 33);
  EXPECT_EQ(dst.op, rdna4::kSBrevB32Sop1);
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

// Cross-ISA legalization of the literal FMA-K / MAD-K family. The fieldless
// OPR_SIMM32 literal now participates in the operand signature (with its size
// canonicalized to the 32-bit encoding word), so identical literal FMAs match
// faithfully across ISAs. These pin the three outcomes that drives:
//   - same opcode on both ISAs           -> Identity
//   - same instruction, different opcode -> Substitute (with remapped target)
//   - the V_FMAAK_F16 case whose per-ISA simm32 size (16 on rdna1/2, 32 on
//     rdna3/4) would fragment the match without the canonicalization
// VOP2 encoding_id is 0 on CDNA guests and 4 on RDNA guests.
TEST(LegalizationFmaK, Cdna3ToCdna4MadmkF16IsIdentity) {
  const auto *e = lookup(kLegalization_cdna3_to_cdna4, /*enc=*/0, /*op=*/36); // v_madmk_f16
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->action, Action::Identity);
  EXPECT_EQ(e->target_opcode, 36);
}

TEST(LegalizationFmaK, Cdna1ToRdna1MadmkF32IsSubstitute) {
  // cdna1 v_madmk_f32 (op 23) -> rdna1 v_madmk_f32 (op 32): same instruction,
  // different opcode number, so the literal-preserving remap is a Substitute.
  const auto *e = lookup(kLegalization_cdna1_to_rdna1, /*enc=*/0, /*op=*/23);
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->action, Action::Substitute);
  EXPECT_EQ(e->target_opcode, 32);
}

TEST(LegalizationFmaK, Rdna1ToRdna3FmaakF16IsIdentity) {
  // Regression guard for the simm32-size canonicalization: without it, this
  // degrades to Lower because the literal's declared size differs (16 vs 32)
  // for the same instruction across rdna generations.
  const auto *e = lookup(kLegalization_rdna1_to_rdna3, /*enc=*/4, /*op=*/56); // v_fmaak_f16
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->action, Action::Identity);
  EXPECT_EQ(e->target_opcode, 56);
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

  const auto gfx1250_rsrc1 = patched_rsrc1(ROCJITSU_CODE_ARCH_GFX1250);
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
  for (uint8_t sym_type : {kElfSymbolTypeFunc, kElfSymbolTypeNone, kElfSymbolTypeObject}) {
    auto text_image = make_amdgpu_elf_with_symbol_relocation(sym_type, /*defined_in_text=*/true);
    AmdGpuCodeObject text_co(text_image.data(), text_image.size());
    ASSERT_TRUE(text_co.is_valid()) << "sym_type=" << static_cast<int>(sym_type);
    CodeObjectPatcher text_patcher(text_co);
    EXPECT_FALSE(text_patcher.has_unsupported_relocation_to_text())
        << "zero-addend ordinary text sym_type=" << static_cast<int>(sym_type)
        << " can follow its relocated st_value";

    auto data_image = make_amdgpu_elf_with_symbol_relocation(sym_type, /*defined_in_text=*/false);
    AmdGpuCodeObject data_co(data_image.data(), data_image.size());
    ASSERT_TRUE(data_co.is_valid()) << "sym_type=" << static_cast<int>(sym_type);
    CodeObjectPatcher data_patcher(data_co);
    EXPECT_FALSE(data_patcher.has_unsupported_relocation_to_text())
        << "data-defined sym_type=" << static_cast<int>(sym_type) << " must be accepted";
  }

  auto section_image =
      make_amdgpu_elf_with_symbol_relocation(kElfSymbolTypeSection, /*defined_in_text=*/true);
  AmdGpuCodeObject section_co(section_image.data(), section_image.size());
  ASSERT_TRUE(section_co.is_valid());
  CodeObjectPatcher section_patcher(section_co);
  EXPECT_TRUE(section_patcher.has_unsupported_relocation_to_text())
      << "STT_SECTION keeps its in-text offset in the relocation addend";

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

// Independently pin the AMDGPU ELF ABI value so the tests below cannot mask a
// regression by constructing the same wrong relocation type the patcher checks
// for. The fixtures use the production constant (so they never drift from it),
// but this assertion anchors that constant to the authoritative ABI number.
static_assert(rocjitsu::R_AMDGPU_RELATIVE64 == 13,
              "R_AMDGPU_RELATIVE64 must be the AMDGPU ELF ABI value 13");

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

} // namespace
} // namespace rocjitsu

namespace cdna3 = rocjitsu::cdna3;
namespace cdna4 = rocjitsu::cdna4;
namespace gfx1250 = rocjitsu::gfx1250;
namespace rdna4 = rocjitsu::rdna4;

/// @brief Build explicit gfx1250 revision options without partial aggregate
/// initializers, which are rejected by the test suite's warning policy.
rocjitsu::BinaryTranslatorOptions
gfx1250_revision_options(rocjitsu::ProcessorRevision input_revision,
                         rocjitsu::ProcessorRevision output_revision) {
  rocjitsu::BinaryTranslatorOptions options;
  options.input_revision = input_revision;
  options.output_revision = output_revision;
  return options;
}

constexpr uint16_t kAnyExpectedField = 0xffff;

enum class ExpectedCdna3Kind {
  Vop3,
  Vop3p,
  Vop2,
  Vop1,
  Sop2,
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
  uint16_t acc = kAnyExpectedField;
  uint16_t acc_cd = kAnyExpectedField;
  uint16_t src0 = kAnyExpectedField;
  uint16_t src1 = kAnyExpectedField;
  uint16_t src2 = kAnyExpectedField;
  uint16_t data0 = kAnyExpectedField;
  uint16_t vdata = kAnyExpectedField;
};

struct Cdna4ToCdna3SemanticRuleCase {
  const char *name = "";
  uint16_t encoding_id = 0;
  uint16_t opcode = 0;
  std::array<uint32_t, 2> words{};
  std::vector<ExpectedCdna3Inst> expected{};
  size_t word_count = 2;
};

ExpectedCdna3Inst expect_vop3(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_vop3(uint16_t op, uint16_t vdst, uint16_t src0, uint16_t src1,
                              uint16_t src2 = kAnyExpectedField) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3;
  inst.op = op;
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
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

ExpectedCdna3Inst expect_sop2(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Sop2;
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
                              uint16_t src1, uint16_t src2, uint16_t acc = 0) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3pMfma;
  inst.op = op;
  inst.vdst = vdst;
  inst.acc = acc;
  inst.acc_cd = acc_cd;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  return inst;
}

ExpectedCdna3Inst expect_ds(uint16_t op, uint16_t data0 = kAnyExpectedField) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Ds;
  inst.op = op;
  inst.data0 = data0;
  return inst;
}

ExpectedCdna3Inst expect_mubuf(uint16_t op, uint16_t vdata = kAnyExpectedField) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Mubuf;
  inst.op = op;
  inst.vdata = vdata;
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
      expect_vop3(cdna3::kVMovB32Vop3), // v_mov_b32
      expect_vop3(cdna3::kVXorB32Vop3), // v_xor_b32
      expect_vop3(cdna3::kVAndB32Vop3), // v_and_b32
      expect_vop3(cdna3::kVXorB32Vop3), // v_xor_b32
      expect_vop3(cdna3::kVXorB32Vop3), // v_xor_b32
      expect_vop3(cdna3::kVAndB32Vop3), // v_and_b32
      expect_vop3(cdna3::kVXorB32Vop3), // v_xor_b32
  };
  if (b16) {
    expected.push_back(expect_vop3(cdna3::kVLshlrevB32Vop3)); // v_lshlrev_b32
    expected.push_back(expect_vop3(cdna3::kVLshrrevB32Vop3)); // v_lshrrev_b32
  }
  expected.push_back(
      expect_vop3(cdna3::kVMovB32Vop3)); // v_mov_b32 copy scratch accumulator to vdst.
  return expected;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_bitop3_fast_sequence(uint16_t op, bool b16) {
  std::vector<ExpectedCdna3Inst> expected = {expect_vop3(op)};
  if (b16) {
    // Compact V_BITOP3_B16 fast paths still need the same high-half cleanup as
    // the generic ANF expansion; otherwise the CDNA3 B32 op leaves stale bits
    // above the architectural 16-bit result.
    expected.push_back(expect_vop3(cdna3::kVLshlrevB32Vop3)); // v_lshlrev_b32
    expected.push_back(expect_vop3(cdna3::kVLshrrevB32Vop3)); // v_lshrrev_b32
  }
  return expected;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_mfma_sequence(uint16_t narrow_op, uint16_t src2 = 128,
                                                            uint16_t source_acc = 0) {
  // Wide-K MFMA lowering is intentionally high-half first, then low-half.  The
  // order is part of the layout contract verified against MI300X Qwen q_proj.
  return {
      expect_mfma(narrow_op, 0, 1, 258, 262, src2, source_acc),
      expect_mfma(narrow_op, 0, 1, 256, 260, 256, source_acc),
  };
}

std::vector<ExpectedCdna3Inst>
expected_cdna3_buffer_load_lds_sequence(uint16_t mubuf_op, uint16_t ds_op,
                                        uint16_t scratch_data = kAnyExpectedField) {
  return {
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 save EXEC.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, -1.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, -1.
      expect_vop3(cdna3::kVMbcntLoU32B32Vop3), // v_mbcnt_lo_u32_b32
      expect_vop3(cdna3::kVMbcntHiU32B32Vop3), // v_mbcnt_hi_u32_b32
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32 lane_id, 4
      expect_vop3(cdna3::kVAddU32Vop3),        // v_add_u32 m0, lane_offset
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 restore EXEC.
      expect_mubuf(mubuf_op, scratch_data),    // buffer_load_dword{x3,x4} into scratch VGPRs.
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt 0 before consuming VMEM data.
      expect_ds(ds_op, scratch_data),          // ds_write_b32/b96/b128
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt lgkmcnt(0) for the explicit DS write.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_permlane32_swap_sequence() {
  return {
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 save EXEC.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, -1.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, -1.
      expect_vop3(cdna3::kVMbcntLoU32B32Vop3), // v_mbcnt_lo_u32_b32
      expect_vop3(cdna3::kVMbcntHiU32B32Vop3), // v_mbcnt_hi_u32_b32
      expect_vop3(cdna3::kVXorB32Vop3),        // v_xor_b32 lane, 32.
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32 byte address.
      expect_ds(cdna3::kDsBpermuteB32Ds),      // ds_bpermute_b32 from old vdst high half.
      expect_ds(cdna3::kDsBpermuteB32Ds),      // ds_bpermute_b32 from old src low half.
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt lgkmcnt(0).
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, low-half mask.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, low-half mask.
      expect_vop3(cdna3::kVMovB32Vop3),        // v_mov_b32 src <- old vdst high.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, high-half mask.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, high-half mask.
      expect_vop3(cdna3::kVMovB32Vop3),        // v_mov_b32 vdst <- old src low.
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 restore EXEC.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_raw_b16_pack_sequence() {
  return {
      expect_vop3(cdna3::kVMovB32Vop3),     // v_mov_b32 -1
      expect_vop3(cdna3::kVLshrrevB32Vop3), // v_lshrrev_b32 16, mask
      expect_vop3(cdna3::kVAndB32Vop3),     // v_and_b32 low half
      expect_vop3(cdna3::kVAndB32Vop3),     // v_and_b32 high half
      expect_vop3(cdna3::kVLshlrevB32Vop3), // v_lshlrev_b32 16, high half
      expect_vop3(cdna3::kVOrB32Vop3),      // v_or_b32
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_pk_f16_f32_sequence() {
  return {
      expect_vop3(cdna3::kVCvtF16F32Vop3),  // v_cvt_f16_f32 low half into scratch.
      expect_vop3(cdna3::kVCvtF16F32Vop3),  // v_cvt_f16_f32 high half into scratch.
      expect_vop3(cdna3::kVLshlrevB32Vop3), // v_lshlrev_b32 16, high half.
      expect_vop3(cdna3::kVOrB32Vop3),      // v_or_b32 pack low/high halves into vdst.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_pk_bf16_f32_sequence() {
  // Per-half NaN-safe RNE lowering (emit_cdna3_f32_to_bf16_rne): compute the
  // rounded value and the NaN-preserving value in parallel, then blend with an
  // exponent-all-ones mask via v_bfi_b32. See semantic/cdna4_to_cdna3.cpp.
  auto half = []() {
    return std::vector<ExpectedCdna3Inst>{
        expect_vop3(cdna3::kVMovB32Vop3),     // f = source bits into scratch.
        expect_vop3(cdna3::kVLshrrevB32Vop3), // (f >> 16)
        expect_vop3(cdna3::kVAndB32Vop3),     // & 1  -> rounding lsb
        expect_vop2(cdna3::kVAddU32Vop2),     // + 0x7fff  (rounding bias)
        expect_vop3(cdna3::kVAddU32Vop3),     // f + bias  -> rounded (t0)
        expect_vop3(cdna3::kVBfeU32Vop3),     // f & 0xffff -> low bits
        expect_vop3(cdna3::kVMinU32Vop3),     // min(1, low) -> nonzero flag
        expect_vop3(cdna3::kVLshlrevB32Vop3), // flag << 16
        expect_vop3(cdna3::kVOrB32Vop3),      // f | (flag<<16) -> NaN-preserving (t1)
        expect_vop2(cdna3::kVAndB32Vop2),     // f & 0x7f800000  (exponent field)
        expect_vop2(cdna3::kVXorB32Vop2),     // ^ 0x7f800000    -> 0 iff exp all-ones
        expect_vop3(cdna3::kVSubU32Vop3),     // - 1
        expect_vop3(cdna3::kVLshrrevB32Vop3), // >> 31
        expect_vop3(cdna3::kVSubU32Vop3),     // 0 - x  -> all-ones mask (t2)
        expect_vop3(cdna3::kVBfiB32Vop3),     // bfi(mask, t1, t0)  -> select
        expect_vop3(cdna3::kVLshrrevB32Vop3), // >> 16  -> BF16 half
    };
  };
  std::vector<ExpectedCdna3Inst> expected = half();
  const std::vector<ExpectedCdna3Inst> high = half();
  expected.insert(expected.end(), high.begin(), high.end());
  expected.push_back(expect_vop3(cdna3::kVLshlrevB32Vop3)); // high half into position.
  expected.push_back(expect_vop3(cdna3::kVOrB32Vop3));      // pack low/high into vdst.
  return expected;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_f32_bf16_sequence() {
  return {
      expect_vop3(
          cdna3::kVLshlrevB32Vop3), // v_lshlrev_b32 16, src; BF16 bits become FP32 high half.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_f32_bf16_sdwa_word0_sequence() {
  return {
      expect_vop3(cdna3::kVLshlrevB32Vop3, 4, rocjitsu::scalar_positive_inline_u32(16), 256 + 4),
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_f32_bf16_sdwa_word1_sequence() {
  return {
      expect_vop3(cdna3::kVLshrrevB32Vop3, 4, rocjitsu::scalar_positive_inline_u32(16), 256 + 76),
      expect_vop3(cdna3::kVLshlrevB32Vop3, 4, rocjitsu::scalar_positive_inline_u32(16), 256 + 4),
  };
}

std::vector<ExpectedCdna3Inst>
expected_cdna3_dot2_f32_bf16_sequence(uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2,
                                      uint8_t scratch_base, uint8_t op_sel = 0,
                                      uint8_t op_sel_hi = 0, uint8_t neg = 0, uint8_t neg_hi = 0,
                                      uint8_t op_sel_hi_2 = 0, bool clamp = false) {
  const uint8_t a_lo = scratch_base;
  const uint8_t b_lo = static_cast<uint8_t>(scratch_base + 1);
  const uint8_t a_hi = static_cast<uint8_t>(scratch_base + 2);
  const uint8_t b_hi = static_cast<uint8_t>(scratch_base + 3);
  std::vector<ExpectedCdna3Inst> expected;
  auto expect_widen = [&](uint8_t dst, uint16_t src, bool high_half, bool negate) {
    if (high_half) {
      expected.push_back(
          expect_vop3(cdna3::kVLshrrevB32Vop3, dst, rocjitsu::scalar_positive_inline_u32(16), src));
      expected.push_back(expect_vop3(cdna3::kVLshlrevB32Vop3, dst,
                                     rocjitsu::scalar_positive_inline_u32(16), 256 + dst));
    } else {
      expected.push_back(
          expect_vop3(cdna3::kVLshlrevB32Vop3, dst, rocjitsu::scalar_positive_inline_u32(16), src));
    }
    if (negate)
      expected.push_back(expect_vop2(cdna3::kVXorB32Vop2)); // v_xor_b32 literal flips FP32 sign.
  };
  expect_widen(a_lo, src0, (op_sel & 0x1u) != 0, (neg & 0x1u) != 0);
  expect_widen(b_lo, src1, (op_sel & 0x2u) != 0, (neg & 0x2u) != 0);
  expect_widen(a_hi, src0, op_sel_hi_2 != 0, (neg_hi & 0x1u) != 0);
  expect_widen(b_hi, src1, (op_sel_hi & 0x2u) != 0, (neg_hi & 0x2u) != 0);
  expected.push_back(expect_vop3(cdna3::kVMulF32Vop3, a_lo, 256 + a_lo, 256 + b_lo));
  expected.push_back(expect_vop3(cdna3::kVMulF32Vop3, a_hi, 256 + a_hi, 256 + b_hi));
  expected.push_back(expect_vop3(cdna3::kVAddF32Vop3, a_lo, 256 + a_lo, 256 + a_hi));
  expected.push_back(expect_vop3(cdna3::kVAddF32Vop3, vdst, 256 + a_lo, src2));
  if (clamp)
    expected.push_back(
        expect_vop3(cdna3::kVMed3F32Vop3, vdst, 256 + vdst, 128, 242)); // v_med3_f32 clamps [0,1].
  return expected;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_ds_read_b64_tr_b16_sequence(bool acc_dst = false) {
  std::vector<ExpectedCdna3Inst> expected = {
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 save EXEC.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, all lanes.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, all lanes.
      expect_ds(cdna3::kDsReadB64Ds),          // ds_read_b64
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt lgkmcnt(0)
      expect_vop3(cdna3::kVMbcntLoU32B32Vop3), // v_mbcnt_lo_u32_b32
      expect_vop3(cdna3::kVMbcntHiU32B32Vop3), // v_mbcnt_hi_u32_b32
      expect_vop3(cdna3::kVAndB32Vop3),        // v_and_b32
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32
      expect_vop3(cdna3::kVAndB32Vop3),        // v_and_b32
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32
      expect_vop3(cdna3::kVAndB32Vop3),        // v_and_b32
      expect_vop3(cdna3::kVOrB32Vop3),         // v_or_b32
      expect_vop3(cdna3::kVAddU32Vop3),        // v_add_u32
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32
      expect_vop3(cdna3::kVOrB32Vop3),         // v_or_b32
      expect_ds(cdna3::kDsBpermuteB32Ds),      // ds_bpermute_b32
      expect_ds(cdna3::kDsBpermuteB32Ds),      // ds_bpermute_b32
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt lgkmcnt(0)
      expect_vop3(cdna3::kVPermB32Vop3),       // v_perm_b32
      expect_vop3(cdna3::kVAddU32Vop3),        // v_add_u32
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_sopp(cdna3::kSWaitcntSopp),
      expect_vop3(cdna3::kVPermB32Vop3),
      expect_vop3(cdna3::kVAddU32Vop3),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_sopp(cdna3::kSWaitcntSopp),
      expect_vop3(cdna3::kVPermB32Vop3),
      expect_vop3(cdna3::kVAddU32Vop3),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_sopp(cdna3::kSWaitcntSopp),
      expect_vop3(cdna3::kVPermB32Vop3),
      expect_sop1(cdna3::kSMovB64), // s_mov_b64 restore EXEC.
  };
  auto first_pack = expected_cdna3_raw_b16_pack_sequence();
  expected.insert(expected.end(), first_pack.begin(), first_pack.end());
  auto second_pack = expected_cdna3_raw_b16_pack_sequence();
  expected.insert(expected.end(), second_pack.begin(), second_pack.end());
  if (acc_dst) {
    expected.push_back(expect_vop3p(cdna3::kVAccvgprWriteVop3p)); // v_accvgpr_write_b32 low dword.
    expected.push_back(expect_vop3p(cdna3::kVAccvgprWriteVop3p)); // v_accvgpr_write_b32 high dword.
  }
  return expected;
}

template <typename MachineInst>
std::array<uint32_t, 2> encode_two_word_inst(const MachineInst &inst) {
  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

std::array<uint32_t, 2> make_cdna4_bitop3_words(uint16_t opcode, uint8_t vdst,
                                                uint8_t truth_table = 0xde) {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = opcode;
  inst.vdst = vdst;
  inst.src0 = static_cast<uint16_t>(256 + vdst + 1);
  inst.src1 = static_cast<uint16_t>(256 + vdst + 2);
  inst.src2 = static_cast<uint16_t>(256 + vdst + 3);

  inst.omod = truth_table >> 6;
  inst.abs = (truth_table >> 3) & 0x7;
  inst.neg = truth_table & 0x7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_v_lshl_add_u64_words(uint8_t vdst, uint16_t src0, uint16_t src1,
                                                        uint16_t src2) {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = cdna4::kVLshlAddU64Vop3;
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_bitop3_b16_unsupported_op_sel_words() {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = cdna4::kVBitop3B16Vop3;
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
  inst.op = cdna4::kVCvtPkF16F32Vop3;
  inst.vdst = 0;
  inst.src0 = 256 + 1;
  inst.src1 = 256 + 2;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_pk_bf16_f32_words(uint8_t vdst = 0, uint16_t src0 = 256 + 1,
                                                         uint16_t src1 = 256 + 2) {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = cdna4::kVCvtPkBf16F32Vop3;
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.src1 = src1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_f32_bf16_words(uint16_t encoding_id) {
  rocjitsu::cdna4::Vop1MachineInst inst{};
  // As with the VOP1 permlane cases below, semantic-rule lookup sees generated
  // primary-decode ids 0xfc..0xff while the raw instruction keeps the hardware
  // VOP1 selector in bits 31:25. Vary VDST high bits to exercise each id.
  inst.encoding = 0x3f;
  inst.op = cdna4::kVCvtF32Bf16Vop1;
  inst.vdst = static_cast<uint8_t>((encoding_id - 0xFCu) << 6);
  inst.src0 = 256 + 1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_dot2_f32_bf16_words(uint8_t vdst, uint16_t src0, uint16_t src1,
                                                       uint16_t src2, uint8_t op_sel = 0,
                                                       uint8_t op_sel_hi = 0, uint8_t neg = 0,
                                                       uint8_t neg_hi = 0, uint8_t op_sel_hi_2 = 0,
                                                       bool clamp = false) {
  rocjitsu::cdna4::Vop3pMachineInst inst{};
  inst.encoding = cdna4::encoding::kVop3p;
  inst.op = cdna4::kVDot2F32Bf16Vop3p;
  inst.vdst = vdst;
  inst.op_sel = op_sel;
  inst.op_sel_hi = op_sel_hi;
  inst.op_sel_hi_2 = op_sel_hi_2;
  inst.clamp = clamp ? 1 : 0;
  inst.neg = neg;
  inst.neg_hi = neg_hi;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_f32_bf16_sdwa_word0_words() {
  rocjitsu::cdna4::Vop1VopSdwaMachineInst inst{};
  inst.encoding = 0x3f;
  inst.op = cdna4::kVCvtF32Bf16Vop1;
  inst.vdst = 4;
  inst.src0 = 249;
  inst.vsrc0 = 4;
  inst.dst_sel = 6;
  inst.dst_unused = 2;
  inst.src0_sel = 4;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_f32_bf16_sdwa_word1_words() {
  rocjitsu::cdna4::Vop1VopSdwaMachineInst inst{};
  inst.encoding = 0x3f;
  inst.op = cdna4::kVCvtF32Bf16Vop1;
  inst.vdst = 4;
  inst.src0 = 249;
  inst.vsrc0 = 76;
  inst.dst_sel = 6;
  inst.dst_unused = 2;
  inst.src0_sel = 5;
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
  inst.op = opcode & 0xFF;
  inst.vdst = static_cast<uint8_t>((encoding_id - cdna4::encoding::kVop1) << 6);
  inst.src0 = 256 + 1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_permlane16_swap_b32_words(uint16_t encoding_id) {
  return make_cdna4_permlane_swap_b32_words(encoding_id, cdna4::kVPermlane16SwapB32Vop1);
}

std::array<uint32_t, 2> make_cdna4_permlane32_swap_b32_words(uint16_t encoding_id) {
  return make_cdna4_permlane_swap_b32_words(encoding_id, cdna4::kVPermlane32SwapB32Vop1);
}

std::array<uint32_t, 2> make_cdna4_mfma_words(uint16_t opcode, uint8_t vdst, uint16_t src0,
                                              uint16_t src1, uint16_t src2 = 128, uint8_t acc = 0) {
  rocjitsu::cdna4::Vop3pMfmaMachineInst inst{};
  inst.encoding = cdna4::encoding::kVop3p;
  inst.op = opcode & 0x7F;
  inst.vdst = vdst;
  inst.acc_cd = 1;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  inst.acc = acc;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_mfma_vgpr_dst_alias_words() {
  rocjitsu::cdna4::Vop3pMfmaMachineInst inst{};
  inst.encoding = cdna4::encoding::kVop3pMfma;
  inst.op = cdna4::kVMfmaF3216x16x32F16Vop3pMfma;
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

std::array<uint32_t, 2> make_cdna4_ds_read_b64_tr_b16_words(uint16_t byte_offset = 0,
                                                            uint8_t addr = 2, bool acc = false) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadB64TrB16Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.vdst = 0;
  inst.acc = acc ? 1 : 0;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_b32_words(uint16_t byte_offset = 0x0134,
                                                     uint8_t addr = 4, uint8_t vdst = 7) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadB32Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read2_b32_words(uint8_t offset0 = 1, uint8_t offset1 = 2,
                                                      uint8_t addr = 12, uint8_t vdst = 20) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsRead2B32Ds;
  inst.offset0 = offset0;
  inst.offset1 = offset1;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read2st64_b32_words(uint8_t offset0 = 2,
                                                          uint8_t offset1 = 3) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsRead2st64B32Ds;
  inst.offset0 = offset0;
  inst.offset1 = offset1;
  inst.addr = 12;
  inst.vdst = 20;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read2_b64_words(uint8_t offset0 = 3, uint8_t offset1 = 68,
                                                      uint8_t addr = 58, uint8_t vdst = 66) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsRead2B64Ds;
  inst.offset0 = offset0;
  inst.offset1 = offset1;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read2st64_b64_words(uint8_t offset0 = 2, uint8_t offset1 = 3,
                                                          uint8_t addr = 98, uint8_t vdst = 112) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsRead2st64B64Ds;
  inst.offset0 = offset0;
  inst.offset1 = offset1;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_b64_words(uint8_t addr = 4, uint8_t vdst = 8,
                                                     uint16_t byte_offset = 0) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadB64Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_b128_words(uint8_t addr = 4, uint8_t vdst = 8,
                                                      uint16_t byte_offset = 0, bool acc = false) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadB128Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.vdst = vdst;
  inst.acc = acc ? 1 : 0;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_u16_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadU16Ds;
  inst.offset0 = 0x20;
  inst.addr = 4;
  inst.vdst = 7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_u8_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadU8Ds;
  inst.offset0 = 0x20;
  inst.addr = 8;
  inst.vdst = 12;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_i8_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadI8Ds;
  inst.offset0 = 0x20;
  inst.addr = 8;
  inst.vdst = 12;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b8_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB8Ds;
  inst.offset0 = 0x10;
  inst.addr = 4;
  inst.data0 = 7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b8_d16_hi_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB8D16HiDs;
  inst.offset0 = 0x10;
  inst.addr = 4;
  inst.data0 = 7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b32_words(uint8_t addr = 4, uint8_t data = 7,
                                                      uint16_t byte_offset = 0) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB32Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.data0 = data;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b96_words(uint8_t addr = 4, uint8_t data = 8,
                                                      uint16_t byte_offset = 0) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB96Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.data0 = data;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b64_words(uint8_t addr = 4, uint8_t data = 8,
                                                      uint16_t byte_offset = 0) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB64Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.data0 = data;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b128_words(uint8_t addr = 4, uint8_t data = 8,
                                                       uint16_t byte_offset = 0, bool acc = false) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB128Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.data0 = data;
  inst.acc = acc ? 1 : 0;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write2_b32_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWrite2B32Ds;
  inst.offset0 = 1;
  inst.offset1 = 2;
  inst.addr = 4;
  inst.data0 = 7;
  inst.data1 = 9;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write2_b64_words(uint8_t addr = 4, uint8_t data0 = 8,
                                                       uint8_t data1 = 12) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWrite2B64Ds;
  inst.offset0 = 1;
  inst.offset1 = 2;
  inst.addr = addr;
  inst.data0 = data0;
  inst.data1 = data1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write2st64_b64_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWrite2st64B64Ds;
  inst.offset0 = 1;
  inst.offset1 = 2;
  inst.addr = 4;
  inst.data0 = 8;
  inst.data1 = 12;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_add_u32_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsAddU32Ds;
  inst.addr = 3;
  inst.data0 = 7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_buffer_load_lds_words(uint16_t op, uint8_t vaddr = 2) {
  rocjitsu::cdna4::MubufMachineInst inst{};
  inst.encoding = 0x38;
  inst.op = op & 0x7F;
  inst.lds = 1;
  inst.offen = 1;
  inst.vaddr = vaddr;
  inst.vdata = 0;
  inst.srsrc = 4;
  inst.soffset = 0;
  return encode_two_word_inst(inst);
}

std::vector<Cdna4ToCdna3SemanticRuleCase> cdna4_to_cdna3_semantic_rule_cases() {
  return {
      {"VBitop3B16", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B16Vop3, 8), expected_cdna3_bitop3_sequence(true)},
      {"VBitop3B16FastEc", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B16Vop3, 8, 0xec),
       expected_cdna3_bitop3_fast_sequence(cdna3::kVAndOrB32Vop3, true)},
      {"VBitop3B16FastF8", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B16Vop3, 8, 0xf8),
       expected_cdna3_bitop3_fast_sequence(cdna3::kVAndOrB32Vop3, true)},
      {"VBitop3B16FastFe", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B16Vop3, 8, 0xfe),
       expected_cdna3_bitop3_fast_sequence(cdna3::kVOr3B32Vop3, true)},
      {"VBitop3B32", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B32Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B32Vop3, 16), expected_cdna3_bitop3_sequence(false)},
      {"VCvtPkF16F32", cdna4::encoding::kVop3OpHi4, cdna4::kVCvtPkF16F32Vop3,
       make_cdna4_cvt_pk_f16_f32_words(), expected_cdna3_cvt_pk_f16_f32_sequence()},
      {"VCvtPkBf16F32", cdna4::encoding::kVop3OpHi4, cdna4::kVCvtPkBf16F32Vop3,
       make_cdna4_cvt_pk_bf16_f32_words(), expected_cdna3_cvt_pk_bf16_f32_sequence()},
      {"VCvtF32Bf16E32", cdna4::encoding::kVop1, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_words(cdna4::encoding::kVop1),
       expected_cdna3_cvt_f32_bf16_sequence(), 1},
      {"VCvtF32Bf16E32Hi1", cdna4::encoding::kVop1Hi1, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_words(cdna4::encoding::kVop1Hi1),
       expected_cdna3_cvt_f32_bf16_sequence(), 1},
      {"VCvtF32Bf16E32Hi2", cdna4::encoding::kVop1Hi2, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_words(cdna4::encoding::kVop1Hi2),
       expected_cdna3_cvt_f32_bf16_sequence(), 1},
      {"VCvtF32Bf16E32Hi3", cdna4::encoding::kVop1Hi3, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_words(cdna4::encoding::kVop1Hi3),
       expected_cdna3_cvt_f32_bf16_sequence(), 1},
      {"VCvtF32Bf16SdwaWord0", cdna4::encoding::kVop1, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_sdwa_word0_words(),
       expected_cdna3_cvt_f32_bf16_sdwa_word0_sequence(), 2},
      {"VCvtF32Bf16SdwaWord1", cdna4::encoding::kVop1, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_sdwa_word1_words(),
       expected_cdna3_cvt_f32_bf16_sdwa_word1_sequence(), 2},
      {"VDot2F32Bf16AccumAlias", cdna4::encoding::kVop3p, cdna4::kVDot2F32Bf16Vop3p,
       make_cdna4_dot2_f32_bf16_words(/*vdst=*/0, /*src0=*/256 + 1, /*src1=*/256 + 2,
                                      /*src2=*/256),
       expected_cdna3_dot2_f32_bf16_sequence(/*vdst=*/0, /*src0=*/256 + 1,
                                             /*src1=*/256 + 2, /*src2=*/256,
                                             /*scratch_base=*/3)},
      {"VDot2F32Bf16OpSelNeg", cdna4::encoding::kVop3p, cdna4::kVDot2F32Bf16Vop3p,
       make_cdna4_dot2_f32_bf16_words(/*vdst=*/0, /*src0=*/256 + 1, /*src1=*/256 + 2,
                                      /*src2=*/256, /*op_sel=*/1, /*op_sel_hi=*/2,
                                      /*neg=*/1, /*neg_hi=*/2),
       expected_cdna3_dot2_f32_bf16_sequence(/*vdst=*/0, /*src0=*/256 + 1,
                                             /*src1=*/256 + 2, /*src2=*/256,
                                             /*scratch_base=*/3, /*op_sel=*/1,
                                             /*op_sel_hi=*/2, /*neg=*/1, /*neg_hi=*/2)},
      {"VDot2F32Bf16OpSelHi2Clamp", cdna4::encoding::kVop3p, cdna4::kVDot2F32Bf16Vop3p,
       make_cdna4_dot2_f32_bf16_words(/*vdst=*/0, /*src0=*/256 + 1, /*src1=*/256 + 2,
                                      /*src2=*/256, /*op_sel=*/0, /*op_sel_hi=*/0,
                                      /*neg=*/0, /*neg_hi=*/0, /*op_sel_hi_2=*/1,
                                      /*clamp=*/true),
       expected_cdna3_dot2_f32_bf16_sequence(/*vdst=*/0, /*src0=*/256 + 1,
                                             /*src1=*/256 + 2, /*src2=*/256,
                                             /*scratch_base=*/3, /*op_sel=*/0,
                                             /*op_sel_hi=*/0, /*neg=*/0, /*neg_hi=*/0,
                                             /*op_sel_hi_2=*/1, /*clamp=*/true)},
      {"VPermlane16SwapB32E32", cdna4::encoding::kVop1, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(cdna4::encoding::kVop1),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane16SwapB32E32Hi1", cdna4::encoding::kVop1Hi1, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(cdna4::encoding::kVop1Hi1),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane16SwapB32E32Hi2", cdna4::encoding::kVop1Hi2, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(cdna4::encoding::kVop1Hi2),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane16SwapB32E32Hi3", cdna4::encoding::kVop1Hi3, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(cdna4::encoding::kVop1Hi3),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane32SwapB32E32", cdna4::encoding::kVop1, cdna4::kVPermlane32SwapB32Vop1,
       make_cdna4_permlane32_swap_b32_words(cdna4::encoding::kVop1),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane32SwapB32E32Hi1", cdna4::encoding::kVop1Hi1, cdna4::kVPermlane32SwapB32Vop1,
       make_cdna4_permlane32_swap_b32_words(cdna4::encoding::kVop1Hi1),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane32SwapB32E32Hi2", cdna4::encoding::kVop1Hi2, cdna4::kVPermlane32SwapB32Vop1,
       make_cdna4_permlane32_swap_b32_words(cdna4::encoding::kVop1Hi2),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane32SwapB32E32Hi3", cdna4::encoding::kVop1Hi3, cdna4::kVPermlane32SwapB32Vop1,
       make_cdna4_permlane32_swap_b32_words(cdna4::encoding::kVop1Hi3),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"MfmaF32_16x16x32Bf16", cdna4::encoding::kVop3p, cdna4::kVMfmaF3216x16x32Bf16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3216x16x32Bf16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3216x16x16Bf16Vop3pMfma)},
      {"MfmaF32_32x32x16Bf16", cdna4::encoding::kVop3p, cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3232x32x8Bf16Vop3pMfma)},
      {"MfmaF32_32x32x16Bf16Source1Acc", cdna4::encoding::kVop3p,
       cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma, 0, 256, 260, 272, 2),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3232x32x8Bf16Vop3pMfma, 272, 2)},
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
       make_cdna4_ds_read_b64_tr_b16_words(/*byte_offset=*/0, /*addr=*/2, /*acc=*/true),
       expected_cdna3_ds_read_b64_tr_b16_sequence(/*acc_dst=*/true)},
      {"BufferLoadDwordLds", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordMubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordMubuf),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordMubuf, cdna3::kDsWriteB32Ds)},
      {"BufferLoadDwordx3Lds", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx3Mubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordx3Mubuf),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordx3Mubuf,
                                               cdna3::kDsWriteB96Ds)},
      {"BufferLoadDwordx3LdsEvenScratch", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx3Mubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordx3Mubuf, /*vaddr=*/0),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordx3Mubuf, cdna3::kDsWriteB96Ds,
                                               /*scratch_data=*/2)},
      {"BufferLoadDwordx4Lds", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx4Mubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordx4Mubuf),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordx4Mubuf,
                                               cdna3::kDsWriteB128Ds)},
      {"BufferLoadDwordx4LdsEvenScratch", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx4Mubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordx4Mubuf, /*vaddr=*/0),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordx4Mubuf,
                                               cdna3::kDsWriteB128Ds, /*scratch_data=*/2)},
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
    EXPECT_EQ(actual.encoding, cdna3::encoding::kVop3p);
    EXPECT_EQ(actual.op, expected.op);
    expect_field_matches(expected.vdst, static_cast<uint16_t>(actual.vdst), "vdst");
    expect_field_matches(expected.src0, static_cast<uint16_t>(actual.src0), "src0");
    expect_field_matches(expected.src1, static_cast<uint16_t>(actual.src1), "src1");
    expect_field_matches(expected.src2, static_cast<uint16_t>(actual.src2), "src2");
    break;
  }
  case ExpectedCdna3Kind::Vop2: {
    rocjitsu::cdna3::Vop2MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Vop1: {
    rocjitsu::cdna3::Vop1MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Sop2: {
    rocjitsu::cdna3::Sop2MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x2u);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Sop1: {
    rocjitsu::cdna3::Sop1MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, cdna3::encoding::kSop1);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Vop3pMfma: {
    rocjitsu::cdna3::Vop3pMfmaMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, cdna3::encoding::kVop3pMfma);
    EXPECT_EQ(actual.op, expected.op);
    expect_field_matches(expected.vdst, static_cast<uint16_t>(actual.vdst), "vdst");
    expect_field_matches(expected.acc, static_cast<uint16_t>(actual.acc), "acc");
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
    expect_field_matches(expected.data0, static_cast<uint16_t>(actual.data0), "data0");
    break;
  }
  case ExpectedCdna3Kind::Mubuf: {
    rocjitsu::cdna3::MubufMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x38u);
    EXPECT_EQ(actual.op, expected.op);
    EXPECT_EQ(actual.lds, 0u);
    expect_field_matches(expected.vdata, static_cast<uint16_t>(actual.vdata), "vdata");
    break;
  }
  case ExpectedCdna3Kind::Sopp: {
    rocjitsu::cdna3::SoppMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, cdna3::encoding::kSopp);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  }
}

void expect_cdna3_text_matches(const rocjitsu::Section &text,
                               const std::vector<ExpectedCdna3Inst> &expected,
                               bool allow_non_nop_tail = false) {
  ASSERT_EQ(text.size() % sizeof(uint32_t), 0u);

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

  ASSERT_GE(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(i);
    expect_cdna3_instruction_matches(*actual[i], expected[i]);
  }
  if (allow_non_nop_tail)
    return;
  for (size_t i = expected.size(); i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    const uint32_t *raw = actual[i]->raw_encoding();
    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(*raw, rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
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

std::optional<uint8_t> find_vop2_literal_add(const uint32_t *words, size_t word_count,
                                             uint8_t vsrc1, uint32_t literal) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::Vop2InstLiteralMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0u && actual.op == 52u && actual.src0 == 0xFFu &&
        actual.vsrc1 == vsrc1 && actual.simm32 == literal)
      return static_cast<uint8_t>(actual.vdst);
  }
  return std::nullopt;
}

size_t count_cdna3_s_mov_b32_literal(const uint32_t *words, size_t word_count, uint8_t sdst,
                                     uint32_t literal) {
  size_t count = 0;
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::Sop1MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x17Du && actual.op == 0u && actual.sdst == sdst &&
        actual.ssrc0 == 0xFFu && words[i + 1] == literal) {
      ++count;
    }
  }
  return count;
}

bool contains_vop3_mov_b32(const uint32_t *words, size_t word_count, uint8_t vdst, uint16_t src0) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::Vop3MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x34u && actual.op == 321u && actual.vdst == vdst && actual.src0 == src0)
      return true;
  }
  return false;
}

bool contains_flat_global_load(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vdst,
                               uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.vdst == vdst &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

bool contains_flat_global_load_acc(const uint32_t *words, size_t word_count, uint8_t op,
                                   uint8_t vdst, uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.vdst == vdst &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.acc == 1u && actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

std::optional<uint8_t> find_flat_global_load_saddr(const uint32_t *words, size_t word_count,
                                                   uint8_t op, uint8_t vdst, uint8_t addr,
                                                   uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.vdst == vdst &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.saddr);
  }
  return std::nullopt;
}

bool contains_flat_global_load_addr(const uint32_t *words, size_t word_count, uint8_t op,
                                    uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.addr == addr &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

std::optional<uint8_t> find_flat_global_load_vdst(const uint32_t *words, size_t word_count,
                                                  uint8_t op, uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.addr == addr &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.vdst);
  }
  return std::nullopt;
}

std::optional<uint8_t> find_flat_global_load_addr_for_vdst(const uint32_t *words, size_t word_count,
                                                           uint8_t op, uint8_t vdst,
                                                           uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.vdst == vdst &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.addr);
  }
  return std::nullopt;
}

bool contains_flat_global_store(const uint32_t *words, size_t word_count, uint8_t op, uint8_t data,
                                uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.data == data &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

bool contains_flat_global_store_acc(const uint32_t *words, size_t word_count, uint8_t op,
                                    uint8_t data, uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.data == data &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.acc == 1u && actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

std::optional<uint8_t> find_flat_global_store_data(const uint32_t *words, size_t word_count,
                                                   uint8_t op, uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.addr == addr &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.data);
  }
  return std::nullopt;
}

std::optional<uint8_t> find_flat_global_store_addr_for_data(const uint32_t *words,
                                                            size_t word_count, uint8_t op,
                                                            uint8_t data, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.data == data &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.addr);
  }
  return std::nullopt;
}

bool contains_flat_global_store_op(const uint32_t *words, size_t word_count, uint8_t op) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.lds == 0u &&
        actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

bool contains_flat_global_op_with_null_saddr(const uint32_t *words, size_t word_count, uint8_t op) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.lds == 0u &&
        actual.saddr == 0x7Fu)
      return true;
  }
  return false;
}

bool contains_mubuf_lds_op(const uint32_t *words, size_t word_count, uint8_t op) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::MubufMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x38u && actual.op == op && actual.lds != 0u)
      return true;
  }
  return false;
}

bool contains_flat_scratch_dword(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vgpr,
                                 uint16_t offset, bool is_load) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatScratchMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    const uint8_t encoded_vgpr =
        is_load ? static_cast<uint8_t>(actual.vdst) : static_cast<uint8_t>(actual.data);
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 1u && actual.saddr == 0x7Fu &&
        encoded_vgpr == vgpr && actual.offset == offset)
      return true;
  }
  return false;
}

bool contains_flat_scratch_dword_offset(const uint32_t *words, size_t word_count, uint8_t op,
                                        uint16_t offset, bool is_load) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatScratchMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 1u && actual.saddr == 0x7Fu &&
        actual.offset == offset) {
      const bool decoded_load = actual.op == 20u;
      if (decoded_load == is_load)
        return true;
    }
  }
  return false;
}

bool contains_cdna3_vop1(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vdst,
                         uint16_t src0) {
  for (size_t i = 0; i < word_count; ++i) {
    rocjitsu::cdna3::Vop1MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x3Fu && actual.op == op && actual.vdst == vdst && actual.src0 == src0)
      return true;
  }
  return false;
}

size_t count_cdna3_vop1_writes(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vdst) {
  size_t count = 0;
  for (size_t i = 0; i < word_count; ++i) {
    rocjitsu::cdna3::Vop1MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x3Fu && actual.op == op && actual.vdst == vdst)
      ++count;
  }
  return count;
}

bool contains_cdna3_vop3p(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vdst,
                          uint16_t src0) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::Vop3pMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x1A7u && actual.op == op && actual.vdst == vdst && actual.src0 == src0)
      return true;
  }
  return false;
}

std::optional<uint8_t> find_cdna3_vop1_vdst(const uint32_t *words, size_t word_count, uint8_t op,
                                            uint16_t src0) {
  for (size_t i = 0; i < word_count; ++i) {
    rocjitsu::cdna3::Vop1MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x3Fu && actual.op == op && actual.src0 == src0)
      return static_cast<uint8_t>(actual.vdst);
  }
  return std::nullopt;
}

bool contains_sopp(const uint32_t *words, size_t word_count, uint8_t op) {
  for (size_t i = 0; i < word_count; ++i) {
    rocjitsu::cdna3::SoppMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x17Fu && actual.op == op)
      return true;
  }
  return false;
}

bool contains_smem_load_dwordx2_with_wait(const uint32_t *words, size_t word_count, uint8_t sdata,
                                          uint8_t sbase_sgpr, uint32_t offset) {
  for (size_t i = 0; i + 2 < word_count; ++i) {
    rocjitsu::cdna3::SmemMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x30u && actual.op == 1u && actual.sdata == sdata &&
        actual.sbase == (sbase_sgpr / 2u) && actual.imm == 1u && actual.offset == offset &&
        words[i + 2] == rocjitsu::pack_sopp(cdna3::kSWaitcntSopp, 0))
      return true;
  }
  return false;
}

void expect_cdna3_translated_descriptor_sgprs_eq(const std::vector<uint8_t> &image,
                                                 uint32_t expected) {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto infos = parser.translate_image(image, translated.text_sections()[0]->sectionOffset(),
                                            translated.text_sections()[0]->size(),
                                            rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(infos.size(), 1u);
  EXPECT_EQ(infos[0].target_sgpr_count, expected);
}

struct InstructionWordsView {
  const uint32_t *words = nullptr;
  size_t word_count = 0;
};

uint32_t cdna3_descriptor_vgpr_allocation_count(const rocjitsu::TestKernelDescriptor &descriptor) {
  const uint32_t granulated =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                      rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  return (granulated + 1u) * 8u;
}

uint32_t cdna3_descriptor_sgpr_count(const rocjitsu::TestKernelDescriptor &descriptor) {
  const uint32_t granulated =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                      rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  return (granulated + 1u) * 8u;
}

std::optional<rocjitsu::VirtualLdsKernelMetadata>
find_virtual_lds_metadata_record_for_test(const std::vector<uint8_t> &image,
                                          std::string_view kernel_name = "kernel") {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  if (!translated.is_valid()) {
    ADD_FAILURE() << "translated code object is invalid";
    return std::nullopt;
  }

  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  if (metadata_section == nullptr) {
    ADD_FAILURE() << "missing " << rocjitsu::kVirtualLdsMetadataSectionName << " section";
    return std::nullopt;
  }

  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  if (!parsed.has_value()) {
    ADD_FAILURE() << "could not parse virtual-LDS metadata";
    return std::nullopt;
  }

  const auto it =
      std::ranges::find_if(*parsed, [&](const rocjitsu::VirtualLdsKernelMetadata &record) {
        return record.kernel_name == kernel_name;
      });
  if (it == parsed->end()) {
    ADD_FAILURE() << "missing virtual-LDS metadata for " << kernel_name;
    return std::nullopt;
  }
  return *it;
}

std::optional<rocjitsu::SidecarVariantMetadata>
find_sidecar_metadata_record_for_test(const std::vector<uint8_t> &image,
                                      std::string_view kernel_name = "kernel") {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  const auto *section = rocjitsu::find_section(translated, rocjitsu::kSidecarMetadataSectionName);
  if (section == nullptr)
    return std::nullopt;
  const auto parsed = rocjitsu::parse_sidecar_metadata(
      {reinterpret_cast<const uint8_t *>(section->data()), section->size()});
  if (!parsed)
    return std::nullopt;
  const auto record = std::ranges::find_if(*parsed, [&](const auto &candidate) {
    return candidate.kernel_name == kernel_name &&
           candidate.variant_name == rocjitsu::kVirtualLdsSidecarVariantName;
  });
  return record == parsed->end() ? std::nullopt : std::optional{*record};
}

std::optional<rocjitsu::KernargExtensionMetadata>
find_kernarg_extension_record_for_test(const std::vector<uint8_t> &image,
                                       std::string_view kernel_name = "kernel") {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  const auto *section =
      rocjitsu::find_section(translated, rocjitsu::kKernargExtensionMetadataSectionName);
  if (section == nullptr)
    return std::nullopt;
  const auto parsed = rocjitsu::parse_kernarg_extension_metadata(
      {reinterpret_cast<const uint8_t *>(section->data()), section->size()});
  if (!parsed)
    return std::nullopt;
  const auto record = std::ranges::find_if(*parsed, [&](const auto &candidate) {
    return candidate.kernel_name == kernel_name &&
           candidate.variant_name == rocjitsu::kVirtualLdsSidecarVariantName;
  });
  return record == parsed->end() ? std::nullopt : std::optional{*record};
}

std::optional<rocjitsu::KernargExtensionLayout>
make_kernarg_extension_layout_for_test(const rocjitsu::KernargExtensionMetadata &metadata) {
  std::vector<rocjitsu::KernargExtensionPayloadLayout> payloads;
  for (const auto &payload : metadata.payloads)
    payloads.push_back({.size = payload.size, .alignment = payload.alignment});
  return rocjitsu::make_kernarg_extension_layout(metadata.original_kernarg_size, payloads);
}

std::optional<rocjitsu::TestKernelDescriptor>
read_descriptor_at_loaded_vaddr_for_test(const std::vector<uint8_t> &image, uint64_t vaddr) {
  const auto descriptor_file_offset = rocjitsu::loaded_vaddr_to_file_offset(image, vaddr);
  if (!descriptor_file_offset.has_value()) {
    ADD_FAILURE() << "could not map descriptor vaddr 0x" << std::hex << vaddr << std::dec;
    return std::nullopt;
  }
  if (*descriptor_file_offset > image.size() ||
      sizeof(rocjitsu::TestKernelDescriptor) > image.size() - *descriptor_file_offset) {
    ADD_FAILURE() << "descriptor vaddr 0x" << std::hex << vaddr << std::dec
                  << " maps outside the ELF image";
    return std::nullopt;
  }
  return rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      image, *descriptor_file_offset);
}

std::optional<rocjitsu::TestKernelDescriptor>
read_virtual_lds_sidecar_descriptor_for_test(const std::vector<uint8_t> &image,
                                             std::string_view kernel_name = "kernel") {
  const auto record = find_sidecar_metadata_record_for_test(image, kernel_name);
  if (!record.has_value())
    return std::nullopt;

  return read_descriptor_at_loaded_vaddr_for_test(image, record->variant_descriptor_vaddr);
}

std::optional<InstructionWordsView>
virtual_lds_sidecar_entry_words_for_test(const std::vector<uint8_t> &image,
                                         std::string_view kernel_name = "kernel") {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  if (!translated.is_valid()) {
    ADD_FAILURE() << "translated code object is invalid";
    return std::nullopt;
  }

  const auto record = find_sidecar_metadata_record_for_test(image, kernel_name);
  const auto descriptor = read_virtual_lds_sidecar_descriptor_for_test(image, kernel_name);
  if (!record.has_value() || !descriptor.has_value())
    return std::nullopt;

  const int64_t entry_delta = rocjitsu::read_kernel_descriptor_entry_offset(&*descriptor);
  const int64_t entry_vaddr_signed =
      static_cast<int64_t>(record->variant_descriptor_vaddr) + entry_delta;
  if (entry_vaddr_signed < 0) {
    ADD_FAILURE() << "sidecar entry resolves before address zero";
    return std::nullopt;
  }

  const uint64_t entry_vaddr = static_cast<uint64_t>(entry_vaddr_signed);
  const auto entry_file_offset = rocjitsu::loaded_vaddr_to_file_offset(image, entry_vaddr);
  if (!entry_file_offset.has_value()) {
    ADD_FAILURE() << "could not map sidecar entry vaddr 0x" << std::hex << entry_vaddr << std::dec;
    return std::nullopt;
  }

  for (const auto *text : translated.text_sections()) {
    const uint64_t text_begin = text->sectionOffset();
    const uint64_t text_end = text_begin + text->size();
    if (*entry_file_offset < text_begin || *entry_file_offset >= text_end)
      continue;

    const uint64_t byte_count = text_end - *entry_file_offset;
    return InstructionWordsView{
        reinterpret_cast<const uint32_t *>(image.data() + *entry_file_offset),
        static_cast<size_t>(byte_count / sizeof(uint32_t)),
    };
  }

  ADD_FAILURE() << "sidecar entry does not point inside a text section";
  return std::nullopt;
}

void expect_cdna3_sidecar_descriptor_sgprs_eq(const std::vector<uint8_t> &image,
                                              uint32_t expected) {
  const auto sidecar_kd = read_virtual_lds_sidecar_descriptor_for_test(image);
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_EQ(cdna3_descriptor_sgpr_count(*sidecar_kd), expected);
}

void expect_cdna3_sidecar_descriptor_vgprs_at_least(const std::vector<uint8_t> &image,
                                                    uint32_t expected_minimum) {
  const auto sidecar_kd = read_virtual_lds_sidecar_descriptor_for_test(image);
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_GE(cdna3_descriptor_vgpr_allocation_count(*sidecar_kd), expected_minimum);
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

std::array<uint32_t, 2> build_cdna4_smem_load(uint8_t op, uint8_t sdata, uint8_t sbase,
                                              uint32_t byte_offset) {
  rocjitsu::cdna4::SmemMachineInst inst{};
  inst.encoding = 0x30;
  inst.op = op;
  inst.sbase = (sbase / 2) & 0x3f;
  inst.sdata = sdata & 0x7f;
  inst.imm = 1;
  inst.offset = byte_offset & 0x1fffff;
  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
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

void expect_nop_words(const uint32_t *words, size_t begin, size_t end, rj_code_arch_t arch) {
  for (size_t i = begin; i < end; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(words[i], rocjitsu::build_s_nop(0, arch));
  }
}

// --- Synthetic BinaryTranslator integration tests ---
TEST(BinaryTranslatorE2E, TranslatesMultiKernelCodeObject) {
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
            (std::vector<uint64_t>{rodata->sectionOffset(),
                                   rodata->sectionOffset() + rocjitsu::kKernelDescriptorSize}));

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

TEST(BinaryTranslatorE2E, SkipFailedKernelKeepsIndependentKernelTranslating) {
  constexpr uint16_t kReturnSreg = 0;
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4),
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_two_kernel_descriptors(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  options.debug_continue_after_failure = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  // A skipped kernel is only a warning, so ok() stays true, but the artifact is
  // NOT dispatchable: its s_endpgm stub would silently complete if dispatched. Code
  // object output paths (CLI, API consumers) must gate on dispatchable().
  EXPECT_FALSE(result.dispatchable());
  ASSERT_FALSE(result.elf_bytes.empty());
  const auto skipped = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == rocjitsu::DiagnosticKind::KernelSkipped;
  });
  ASSERT_NE(skipped, result.diagnostics.end());
  EXPECT_EQ(skipped->severity, rocjitsu::DiagnosticSeverity::Warning);
  EXPECT_EQ(skipped->guest_offset, std::optional<uint64_t>(0));
  EXPECT_NE(skipped->message.find("kernel0"), std::string::npos);
  EXPECT_NE(skipped->message.find("indirect branch"), std::string::npos);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *translated_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(translated_words[0], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3))
      << "the failed kernel body exits safely if it is dispatched";
  EXPECT_NE(skipped->message.find("S_ENDPGM"), std::string::npos);
  EXPECT_NE(skipped->message.find("INVALID OUTPUTS"), std::string::npos);

  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto infos = parser.translate_image(
      result.elf_bytes, translated.text_sections()[0]->sectionOffset(),
      translated.text_sections()[0]->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(infos.size(), 2u);
  std::vector<uint64_t> entries;
  for (const auto &info : infos)
    entries.push_back(info.entry_text_offset);
  std::ranges::sort(entries);
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0], 0u);
  EXPECT_EQ(entries[1], sizeof(uint32_t))
      << "the independent kernel follows the one-instruction skipped body";
  ASSERT_EQ(entries[1] % sizeof(uint32_t), 0u);
  ASSERT_LT(entries[1], translated.text_sections()[0]->size());
  EXPECT_EQ(translated_words[entries[1] / sizeof(uint32_t)],
            rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3))
      << "the independent kernel still translates instead of using the skipped-kernel trap";
}

TEST(BinaryTranslatorE2E, OversizedTargetLdsDescriptorEmitsVirtualVariantInsteadOfSkipping) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_two_kernel_descriptors();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), 2 * rocjitsu::kKernelDescriptorSize);

  // gfx950/CDNA4 can advertise 160 KiB LDS kernels, but gfx942/CDNA3 can only
  // dispatch 64 KiB per workgroup in the checked-in topology. Keep the normal
  // descriptor for launches that fit, and emit a virtual-LDS sidecar descriptor
  // for runtime packet rewriting when static plus dynamic LDS exceeds the host.
  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      105600u);
  rocjitsu::write_value_for_test<uint32_t>(
      image, rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, kernarg_size), 16u);
  uint32_t source_rsrc2 = 0;
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                  1);
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                  1);
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  // COMPUTE_PGM_RSRC2.LDS_SIZE is programmed by CP from the dispatch packet, not
  // by the code-object descriptor. Seed a stale guest value here so the test
  // proves translated normal and virtual descriptors do not preserve it.
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE, 22);
  rocjitsu::write_value_for_test<uint32_t>(
      image, rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, compute_pgm_rsrc2),
      source_rsrc2);
  uint16_t source_properties = 0;
  AMDHSA_BITS_SET(source_properties,
                  rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  rocjitsu::write_value_for_test<uint16_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, kernel_code_properties),
      source_properties);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *translated_rodata = rocjitsu::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.group_segment_fixed_size, 105600u);
  EXPECT_EQ(normal_kd.kernarg_size, 16u);
  EXPECT_EQ(AMDHSA_BITS_GET(normal_kd.compute_pgm_rsrc2,
                            rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE),
            0u);

  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes, "kernel0");
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes, "kernel0");
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(record.kernel_name, "kernel0");
  EXPECT_EQ(sidecar->normal_descriptor_vaddr, translated.kernel_descriptor_offset("kernel0"));
  EXPECT_NE(sidecar->variant_descriptor_vaddr, sidecar->normal_descriptor_vaddr);
  EXPECT_EQ(record.static_lds_bytes, 105600u);
  EXPECT_EQ(extension->original_kernarg_size, 16u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 24u);
  EXPECT_NE(record.virtual_lds_base_sgpr, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagWorkgroupIdX, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagWorkgroupIdY, 0u);
  EXPECT_EQ(record.flags & rocjitsu::kVirtualLdsFlagWorkgroupIdZ, 0u);

  const auto virtual_descriptor_offset =
      rocjitsu::loaded_vaddr_to_file_offset(result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(virtual_kd.kernarg_size, 48u);
  EXPECT_EQ(AMDHSA_BITS_GET(virtual_kd.compute_pgm_rsrc2,
                            rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE),
            0u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarPreservesReservedBytesAfterTextGrowth) {
  // Regression for the sidecar being copied from a stale pre-.text-growth file
  // offset. The DS-read fixture lowers to a flat-global load, which GROWS .text
  // and thus shifts the descriptor section that follows it. The sidecar template
  // must come from a snapshot of the source descriptor, not from re-reading the
  // now-shifted offset (which would read relocated instruction bytes). Seed the
  // reserved regions with recognizable patterns and assert the emitted sidecar
  // carries them unchanged.
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  // Seed reserved1[20] with a distinctive byte pattern. reserved1 is a field DBT
  // never rewrites, so it is only correct in the sidecar if the template was
  // copied from the (pre-growth) source descriptor snapshot.
  const uint64_t reserved1_off =
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, reserved1);
  std::array<uint8_t, 20> reserved1_pattern{};
  for (size_t i = 0; i < reserved1_pattern.size(); ++i)
    reserved1_pattern[i] = static_cast<uint8_t>(0xA0 + i);
  for (size_t i = 0; i < reserved1_pattern.size(); ++i)
    image[reserved1_off + i] = reserved1_pattern[i];

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        rocjitsu::BinaryTranslatorOptions{});
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  // Confirm .text actually grew (otherwise the regression could not occur).
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_GT(translated.text_sections()[0]->size(), 3u * sizeof(uint32_t));

  const auto sidecar_kd = read_virtual_lds_sidecar_descriptor_for_test(result.elf_bytes, "kernel");
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_TRUE(std::equal(std::begin(sidecar_kd->reserved1), std::end(sidecar_kd->reserved1),
                         reserved1_pattern.begin()))
      << "sidecar reserved1 was not copied from the source descriptor snapshot";
}

TEST(BinaryTranslatorE2E, VirtualLdsWrapperIgnoresInlineKernargLoadsBeyondDescriptorSize) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  constexpr uint16_t kKernargSgpr = 0;
  constexpr uint16_t kInlineInt0 = 128;
  constexpr uint16_t kInlineInt16 = 144;
  constexpr uint32_t kDescriptorKernargSize = 48;
  constexpr uint32_t kExpectedStateOffset = 56;
  constexpr uint32_t kExpectedWrapperBytes = 80;
  const auto load_header = build_cdna4_smem_load(/*op=*/0, /*sdata=*/16, kKernargSgpr,
                                                 /*byte_offset=*/0);
  const auto load_userargs = build_cdna4_smem_load(/*op=*/4, /*sdata=*/20, kKernargSgpr,
                                                   /*byte_offset=*/0x40);

  std::vector<uint32_t> words = {
      load_header[0],
      load_header[1],
      build_s_add_u32(kKernargSgpr, kKernargSgpr, kInlineInt16),
      build_s_addc_u32(kKernargSgpr + 1, kKernargSgpr + 1, kInlineInt0),
      load_userargs[0],
      load_userargs[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, kDescriptorKernargSize);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));
  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      6144u);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *translated_rodata = rocjitsu::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.kernarg_size, kDescriptorKernargSize);

  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(extension->original_kernarg_size, kDescriptorKernargSize);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), kExpectedStateOffset);

  const auto virtual_descriptor_offset =
      rocjitsu::loaded_vaddr_to_file_offset(result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.kernarg_size, kExpectedWrapperBytes);
}

TEST(BinaryTranslatorE2E, VirtualLdsWrapperIgnoresDirectKernargLoadsInSharedHelperBlocks) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  constexpr uint16_t kKernargSgpr = 0;
  constexpr uint32_t kDescriptorKernargSize = 0x90;
  constexpr uint32_t kExpectedStateOffset = 0x98;
  constexpr uint32_t kExpectedWrapperBytes = 0xB0;
  const auto load_helper_tail =
      build_cdna4_smem_load(/*op=*/0, /*sdata=*/16, kKernargSgpr, /*byte_offset=*/0xA0);

  // Model Tensile-style helper code that is present in the code object but not
  // reached by the local descriptor CFG. The virtual-LDS dispatcher still must
  // not size its wrapper prefix from a decoded direct kernarg load range from
  // the same ABI kernarg SGPR pair.
  std::vector<uint32_t> words = {
      rocjitsu::pack_sopp(/*op=*/5, /*simm16=*/2),
      load_helper_tail[0],
      load_helper_tail[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, kDescriptorKernargSize);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));
  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      6144u);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *translated_rodata = rocjitsu::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.kernarg_size, kDescriptorKernargSize);

  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(extension->original_kernarg_size, kDescriptorKernargSize);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), kExpectedStateOffset);

  const auto virtual_descriptor_offset =
      rocjitsu::loaded_vaddr_to_file_offset(result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.kernarg_size, kExpectedWrapperBytes);
}

TEST(BinaryTranslatorE2E, DirectBranchRelocationUsesLongBranchWindowWhenExpandedOutOfRange) {
  constexpr size_t kTargetWord = 20000;
  constexpr uint16_t kLongBranchScratchSgpr = 8;
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;

  std::vector<uint32_t> words;
  words.reserve(kTargetWord + 1);
  words.push_back(
      rocjitsu::pack_sopp(cdna3::kSCbranchExecnzSopp, static_cast<uint16_t>(kTargetWord - 1)));
  for (size_t i = 1; i < kTargetWord; ++i)
    words.push_back(rocjitsu::pack_sopp(cdna3::kSCbranchScc0Sopp, 0));
  words.push_back(kCdna4SEndpgm);

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());

  // The source branch is in range, but reserving branch windows for the many
  // intervening branches pushes the relocated target outside the SOPP simm16
  // range. DBT inverts the condition to skip over the long transfer when EXEC
  // is zero, then rebuilds the target PC in a descriptor-backed SGPR pair. This
  // specifically guards the highest adjacent conditional pair: opcode 38 must
  // invert to 37, not invalid opcode 39.
  EXPECT_EQ(target_words[0], rocjitsu::pack_sopp(cdna3::kSCbranchExeczSopp,
                                                 rocjitsu::kMaxDirectBranchTransferWords - 1));
  EXPECT_EQ(target_words[1], build_s_getpc_b64(kLongBranchScratchSgpr, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[5], build_s_setpc_b64(kLongBranchScratchSgpr, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[6], rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, Gfx1250LongBranchInvertsExecNzToExecZ) {
  constexpr size_t kTargetWord = 20000;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  std::vector<uint32_t> words;
  words.reserve(kTargetWord + 1);
  words.push_back(
      rocjitsu::pack_sopp(gfx1250::kSCbranchExecnzSopp, static_cast<uint16_t>(kTargetWord - 1)));
  for (size_t i = 1; i < kTargetWord; ++i)
    words.push_back(rocjitsu::pack_sopp(gfx1250::kSCbranchScc0Sopp, 0));
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::pack_sopp(gfx1250::kSCbranchExeczSopp,
                                                 rocjitsu::kMaxDirectBranchTransferWords - 1));
}

TEST(BinaryTranslatorE2E, DirectBranchWindowKeepsFourKibibyteBranchesCompact) {
  const uint32_t branch_word = rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_GFX1250);
  const auto branch = rocjitsu::decode_one(branch_word, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(branch, nullptr);

  constexpr uint64_t kCompactDistance = 4 * 1024;
  EXPECT_EQ(rocjitsu::direct_branch_patch_window_bytes(*branch, 0, kCompactDistance, true),
            sizeof(uint32_t));
  EXPECT_EQ(rocjitsu::direct_branch_patch_window_bytes(*branch, 0,
                                                       kCompactDistance + sizeof(uint32_t), true),
            rocjitsu::kMaxDirectBranchTransferWords * sizeof(uint32_t));
}

TEST(BinaryTranslatorE2E, DirectBranchWindowReservesLongFormForExpandedConditional) {
  const uint32_t branch_word =
      rocjitsu::pack_sopp(gfx1250::kSCbranchScc0Sopp, static_cast<uint16_t>(1391));
  const auto branch = rocjitsu::decode_one(branch_word, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(branch, nullptr);

  constexpr uint64_t kNvfp4BranchDistance = 5568;
  EXPECT_EQ(rocjitsu::direct_branch_patch_window_bytes(*branch, 0, kNvfp4BranchDistance, true),
            rocjitsu::kMaxDirectBranchTransferWords * sizeof(uint32_t));
  EXPECT_EQ(rocjitsu::direct_branch_patch_window_bytes(*branch, 0, kNvfp4BranchDistance, false),
            2 * sizeof(uint32_t));
}

TEST(BinaryTranslatorE2E, DirectBranchRelocationUsesIslandsWhenSgprsAreFull) {
  using namespace rocr::llvm::amdhsa;

  constexpr size_t kTargetWord = 20000;
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;

  std::vector<uint32_t> words;
  words.reserve(kTargetWord + 1);
  words.push_back(
      rocjitsu::pack_sopp(cdna3::kSCbranchScc1Sopp, static_cast<uint16_t>(kTargetWord - 1)));
  for (size_t i = 1; i < kTargetWord; ++i)
    words.push_back(rocjitsu::pack_sopp(cdna3::kSCbranchScc0Sopp, 0));
  words.push_back(kCdna4SEndpgm);

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  // 112 SGPRs is the largest CDNA descriptor allocation this translator accepts.
  // There is no legal descriptor-backed pair left for the getpc/add/setpc long
  // direct-branch sequence. Full-SGPR kernels therefore need SOPP-only branch
  // islands instead of forcing an unpatchable s112:s113 scratch requirement.
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());
  expect_cdna3_translated_descriptor_sgprs_eq(result.elf_bytes, 112);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  // The original s_cbranch_scc1 is out of range after every conditional branch
  // reserves a two-word islandable window. DBT inverts it to skip over a local
  // s_branch into a private island slot; no scratch SGPR is required.
  EXPECT_EQ(target_words[0], rocjitsu::pack_sopp(cdna3::kSCbranchScc0Sopp, 1));
  rocjitsu::cdna3::SoppMachineInst island_branch{};
  std::memcpy(&island_branch, target_words + 1, sizeof(island_branch));
  EXPECT_EQ(island_branch.encoding, 0x17Fu);
  EXPECT_EQ(island_branch.op, 2u);
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

  ASSERT_LE(test_case.word_count, test_case.words.size());
  const std::vector<uint32_t> source_words(test_case.words.begin(),
                                           test_case.words.begin() + test_case.word_count);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(source_words);
  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());

  const auto *source_text = source_layout.text_sections()[0];
  ASSERT_EQ(source_text->size(), source_words.size() * sizeof(uint32_t));

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_EQ(rocjitsu::find_section(translated, ".rj_translations"), nullptr);
  const bool has_sidecar =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName) != nullptr;
  expect_cdna3_text_matches(*translated.text_sections()[0], test_case.expected, has_sidecar);
}

INSTANTIATE_TEST_SUITE_P(ImplementedRules, Cdna4ToCdna3SemanticRuleTranslationTest,
                         ::testing::ValuesIn(cdna4_to_cdna3_semantic_rule_cases()),
                         [](const ::testing::TestParamInfo<Cdna4ToCdna3SemanticRuleCase> &info) {
                           return std::string(info.param.name);
                         });

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaEmitsShiftAndCarryAdd) {
  // v[4:5] = (v[0:1] << 2) + v[2:3]. vdst does not alias the addend, so the shift
  // goes straight into vdst. Verify a v_lshlrev_b64 (the shift the old lowering
  // dropped) precedes the carry-add pair.
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/4, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/256 + 2);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_RDNA4);
  const bool has_shift = std::any_of(decoded.begin(), decoded.end(), [](const auto &inst) {
    return inst->mnemonic() == "v_lshlrev_b64";
  });
  EXPECT_TRUE(has_shift) << "lowering must materialize the shift, not drop it";
  const bool has_add = std::any_of(decoded.begin(), decoded.end(), [](const auto &inst) {
    return inst->mnemonic() == "v_add_co_u32";
  });
  EXPECT_TRUE(has_add) << "lowering must emit the 64-bit carry add";
}

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaHandlesDestinationAliasingAddend) {
  // v[2:3] = (v[0:1] << 2) + v[2:3]: the destination aliases the VGPR addend, the
  // common address-computation pattern emitted by real kernels (e.g. vector_add).
  // The lowering must shift into a dead scratch pair rather than clobbering the
  // addend, and must still succeed (the earlier fail-closed guard wrongly rejected
  // this).
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/2, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/256 + 2);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_RDNA4);

  const auto shift = std::find_if(decoded.begin(), decoded.end(), [](const auto &inst) {
    return inst->mnemonic() == "v_lshlrev_b64";
  });
  ASSERT_NE(shift, decoded.end()) << "lowering must materialize the shift";
  // The shift destination must not be v2/v3 (the aliased addend); it must land in
  // a separate scratch pair so the following add can still read the addend.
  ASSERT_GE((*shift)->num_dst_operands(), 1);
  const auto shift_dst = (*shift)->dst_operand(0)->to_register_ref();
  ASSERT_TRUE(shift_dst.has_value());
  EXPECT_NE(shift_dst->index, 2u)
      << "shift result must not overwrite the aliased addend v[2:3] before the add reads it";
}

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaRejectsInlineConstantAddend) {
  // v[4:5] = (v[0:1] << 2) + <inline constant 0>. The 64-bit addend is encoded as
  // a single inline-constant operand (128), so the high half cannot be derived as
  // src2 + 1 (129 == constant 1). The lowering must fail closed rather than
  // silently add 1 to the high word.
  constexpr uint16_t kInlineConstZero = 128;
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/4, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/kInlineConstZero);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::ExpandFailed,
                                             "non-register 64-bit addend"));
}

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaRejectsV255Addend) {
  // v[4:5] = (v[0:1] << 2) + v[255:256]. src2=v255 (selector 511) has no valid
  // high half: the derived src2+1 selector is 512, which does not encode a VGPR.
  // The lowering must fail closed rather than emit an add reading an invalid
  // operand.
  constexpr uint16_t kV255 = 256 + 255;
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/4, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/kV255);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::ExpandFailed,
                                             "non-register 64-bit addend"));
}

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaCarryUsesScalarSgprNotVcc) {
  // v_lshl_add_u64 defines no carry output and VCC is not liveness-tracked, so
  // the carry chain must target a dead ordinary SGPR pair, never VCC (which could
  // be live across the instruction). Verify the emitted v_add_co_u32 /
  // v_add_co_ci_u32 write a scalar SDST that is not VCC.
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/4, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/256 + 2);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_RDNA4);

  // VCC_LO is scalar selector 106. The carry SDST must be an ordinary SGPR.
  constexpr uint32_t kVccLo = 106;
  bool saw_add = false;
  for (const auto &inst : decoded) {
    const auto mnemonic = inst->mnemonic();
    if (mnemonic != "v_add_co_u32" && mnemonic != "v_add_co_ci_u32")
      continue;
    saw_add = true;
    ASSERT_GE(inst->num_dst_operands(), 2)
        << mnemonic << " must expose an explicit scalar carry destination";
    const auto sdst = inst->dst_operand(1)->to_register_ref();
    ASSERT_TRUE(sdst.has_value()) << mnemonic << " carry SDST must be a register";
    EXPECT_NE(sdst->index, kVccLo) << mnemonic << " must not clobber VCC as its carry destination";
  }
  EXPECT_TRUE(saw_add) << "lowering must emit the carry-add pair";
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadB32ToFlatGlobalLoad) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto saddr =
      find_flat_global_load_saddr(target_words, target_word_count, /*op=*/20, /*vdst=*/7,
                                  /*addr=*/4, /*offset=*/0x134);
  ASSERT_TRUE(saddr.has_value());
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(target_words, target_word_count, *saddr,
                                                   /*sbase_sgpr=*/0, /*offset=*/24));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersMubufDwordLdsToFlatGlobalStore) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto mubuf = make_cdna4_buffer_load_lds_words(/*op=*/20);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {mubuf[0], mubuf[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  // A virtual-LDS sidecar cannot leave the MUBUF LDS bit set: the side effect
  // would target zero-sized hardware LDS instead of rocjitsu's backing buffer.
  EXPECT_FALSE(contains_mubuf_lds_op(target_words, target_word_count, /*op=*/20));
  EXPECT_TRUE(contains_flat_global_store_op(target_words, target_word_count, /*op=*/28));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarReservesFreshSgprsBelowCdnaSpecialTail) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_mov_b32(/*sdst=*/47, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      ds[0],
      ds[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  // A HipKittens-style descriptor can allocate more SGPRs than the ordinary
  // guest body names because VCC/flat-scratch/XNACK and granularity padding are
  // included in COMPUTE_PGM_RSRC1. Virtual LDS must not treat that tail as
  // ordinary scratch space.
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  6);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  EXPECT_EQ(parsed->front().virtual_lds_base_sgpr, 48u);
  expect_cdna3_sidecar_descriptor_sgprs_eq(result.elf_bytes, 64);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarKeepsDescriptorSpecialTailFree) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  expect_cdna3_translated_descriptor_sgprs_eq(result.elf_bytes, 112);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  EXPECT_EQ(parsed->front().virtual_lds_base_sgpr, 2u);

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  // A large SGPR allocation does not imply all of those numbers are ordinary
  // scratchable registers. VCC/flat-scratch/XNACK live in the descriptor tail,
  // so a guest body that only names low SGPRs should place virtual-LDS scratch
  // below that tail instead of borrowing a high pair with spill-per-use.
  EXPECT_FALSE(contains_sopp(target_words, target_word_count, /*op=*/8));
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(target_words, target_word_count, /*sdata=*/2,
                                                   /*sbase_sgpr=*/0, /*offset=*/24));
  rocjitsu::cdna3::FlatMachineInst actual{};
  bool found_load = false;
  for (size_t i = 0; i + 1 < target_word_count; ++i) {
    std::memcpy(&actual, target_words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == 20u && actual.seg == 2u && actual.addr == 4u &&
        actual.vdst == 7u && actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr == 2u) {
      found_load = true;
      break;
    }
  }
  EXPECT_TRUE(found_load);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillsTouchedHighSgprPairWhenDescriptorSgprsAreFull) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  std::vector<uint32_t> words;
  words.reserve(102 + 3);
  for (uint16_t sgpr = 0; sgpr < 102; ++sgpr)
    words.push_back(rocjitsu::build_s_mov_b32(sgpr, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(ds[0]);
  words.push_back(ds[1]);
  words.push_back(kCdna4SEndpgm);

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  expect_cdna3_translated_descriptor_sgprs_eq(result.elf_bytes, 112);

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  const auto *target_words = entry->words;
  const size_t target_word_count = entry->word_count;
  EXPECT_TRUE(contains_sopp(target_words, target_word_count, /*op=*/8));
  const auto saved_lo =
      find_cdna3_vop1_vdst(target_words, target_word_count, /*op=*/1, /*src0=*/100);
  const auto saved_hi =
      find_cdna3_vop1_vdst(target_words, target_word_count, /*op=*/1, /*src0=*/101);
  ASSERT_TRUE(saved_lo.has_value());
  ASSERT_TRUE(saved_hi.has_value());
  EXPECT_NE(*saved_lo, *saved_hi);
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(target_words, target_word_count, /*sdata=*/100,
                                                   /*sbase_sgpr=*/0, /*offset=*/24));
  // Spill-per-use lowering first installs the runtime backing pointer in the
  // borrowed SGPR pair, then restores the guest scalar values after the lowered
  // memory access. The two operations can use different VGPR temps because the
  // entry prologue saves the backing pointer in persistent scratch.
  EXPECT_GE(count_cdna3_vop1_writes(target_words, target_word_count, /*op=*/2, /*vdst=*/100), 2u);
  EXPECT_GE(count_cdna3_vop1_writes(target_words, target_word_count, /*op=*/2, /*vdst=*/101), 2u);
}

TEST(BinaryTranslatorE2E, VirtualLdsMubufSemanticRuleUsesSpillPerUseAccessEmitter) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto mubuf = make_cdna4_buffer_load_lds_words(/*op=*/20);
  std::vector<uint32_t> words;
  words.reserve(102 + 3);
  for (uint16_t sgpr = 0; sgpr < 102; ++sgpr)
    words.push_back(rocjitsu::build_s_mov_b32(sgpr, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(mubuf[0]);
  words.push_back(mubuf[1]);
  words.push_back(kCdna4SEndpgm);

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  EXPECT_TRUE(contains_flat_global_op_with_null_saddr(entry->words, entry->word_count, /*op=*/28));
  EXPECT_TRUE(contains_sopp(entry->words, entry->word_count, /*op=*/8));
  EXPECT_GE(count_cdna3_vop1_writes(entry->words, entry->word_count, /*op=*/2, /*vdst=*/100), 2u);
  EXPECT_GE(count_cdna3_vop1_writes(entry->words, entry->word_count, /*op=*/2, /*vdst=*/101), 2u);
}

TEST(BinaryTranslatorE2E, VirtualLdsTransposeSemanticRuleUsesSpillPerUseAccessEmitter) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b64_tr_b16_words();
  std::vector<uint32_t> words;
  words.reserve(102 + 3);
  for (uint16_t sgpr = 0; sgpr < 102; ++sgpr)
    words.push_back(rocjitsu::build_s_mov_b32(sgpr, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(ds[0]);
  words.push_back(ds[1]);
  words.push_back(kCdna4SEndpgm);

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  EXPECT_TRUE(contains_flat_global_op_with_null_saddr(entry->words, entry->word_count, /*op=*/21));
  EXPECT_TRUE(contains_sopp(entry->words, entry->word_count, /*op=*/8));
  EXPECT_GE(count_cdna3_vop1_writes(entry->words, entry->word_count, /*op=*/2, /*vdst=*/100), 2u);
  EXPECT_GE(count_cdna3_vop1_writes(entry->words, entry->word_count, /*op=*/2, /*vdst=*/101), 2u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillPerUseGrowsDescriptorForBorrowedHighSgprs) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      ds[0],
      ds[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  2);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  expect_cdna3_sidecar_descriptor_sgprs_eq(result.elf_bytes, 104);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  EXPECT_EQ(parsed->front().virtual_lds_base_sgpr, 98u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillPerUseCanGrowTinyNoAgprKernelsPastAccumOffset) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b32_words(/*addr=*/0, /*data=*/1);
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      ds[0],
      ds[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  2);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  bool found_full_vaddr_store = false;
  for (size_t i = 0; i + 1 < target_word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, target_words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == 28u && actual.seg == 2u && actual.addr == 0u &&
        actual.saddr == 0x7Fu) {
      found_full_vaddr_store = true;
      break;
    }
  }
  EXPECT_TRUE(found_full_vaddr_store);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarRaisesDescriptorForExplicitHighDsVgprs) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds_write = make_cdna4_ds_write_b32_words(/*addr=*/62, /*data=*/0, /*byte_offset=*/256);
  const auto ds_read = make_cdna4_ds_read_b32_words(/*byte_offset=*/256, /*addr=*/62, /*vdst=*/116);
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      ds_write[0],
      ds_write[1],
      ds_read[0],
      ds_read[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  2);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  // The source descriptor intentionally under-declares the inline-assembly
  // VGPRs. Virtual-LDS lowering must grow the target descriptor for both the
  // synthetic GLOBAL address high half v63 and the explicit read destination
  // v116 before the runtime dispatches the translated body.
  expect_cdna3_sidecar_descriptor_vgprs_at_least(result.elf_bytes, 117);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillPerUseCapturesWrapperBeforeGuestClobber) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  const uint32_t touch_s100 =
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t touch_s101 =
      rocjitsu::build_s_mov_b32(/*sdst=*/101, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t clobber_dispatch_ptr =
      rocjitsu::build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {touch_s100, touch_s101, clobber_dispatch_ptr, ds[0], ds[1], kCdna4SEndpgm});

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(source_kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                  1);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  const auto *entry_words = entry->words;
  const size_t entry_word_count = std::min(entry->word_count, size_t{48});
  // The descriptor entry must consume the wrapper kernarg pointer before the
  // guest body executes `s_mov_b32 s0, 0`. Spill-per-use virtual LDS then
  // reloads the backing pointer from private scratch, not from a guest-owned
  // ABI pointer SGPR.
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(entry_words, entry_word_count,
                                                   /*sdata=*/100, /*sbase_sgpr=*/2,
                                                   /*offset=*/8));

  EXPECT_TRUE(contains_flat_scratch_dword_offset(entry_words, entry_word_count, /*op=*/28,
                                                 /*offset=*/0, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(entry_words, entry_word_count, /*op=*/28,
                                                 /*offset=*/4, /*is_load=*/false));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillPerUseKeepsEntryPrologueStableAfterVgprGrowth) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  const auto grow_vgprs =
      make_cdna4_permlane32_swap_b32_words(/*encoding_id=*/cdna4::encoding::kVop1Hi3);
  const uint32_t touch_s100 =
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t touch_s101 =
      rocjitsu::build_s_mov_b32(/*sdst=*/101, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {touch_s100, touch_s101, ds[0], ds[1], grow_vgprs[0], grow_vgprs[1], kCdna4SEndpgm});

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(source_kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                  1);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = 193;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  expect_cdna3_sidecar_descriptor_vgprs_at_least(result.elf_bytes, 196);

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  const auto *entry_words = entry->words;
  const size_t entry_word_count = std::min(entry->word_count, size_t{48});
  // The test-only liveness floor makes the permlane lowering grow the final
  // target VGPR count to at least v195 after the entry prologue has already
  // been emitted. Spill-per-use virtual LDS must still reuse the original low
  // entry temps during descriptor recomputation instead of regenerating the
  // prologue with v194/v195.
  const auto saved_lo = find_cdna3_vop1_vdst(entry_words, entry_word_count, /*op=*/1, /*src0=*/100);
  const auto saved_hi = find_cdna3_vop1_vdst(entry_words, entry_word_count, /*op=*/1, /*src0=*/101);
  ASSERT_TRUE(saved_lo.has_value());
  ASSERT_TRUE(saved_hi.has_value());
  EXPECT_LT(*saved_lo, 194u);
  EXPECT_LT(*saved_hi, 194u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarEmitsRuntimeMetadataSection) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);

  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(record.kernel_name, "kernel");
  EXPECT_EQ(sidecar->normal_descriptor_vaddr, translated.kernel_descriptor_offset("kernel"));
  EXPECT_NE(sidecar->variant_descriptor_vaddr, sidecar->normal_descriptor_vaddr);
  EXPECT_EQ(record.static_lds_bytes, 6144u);
  EXPECT_EQ(extension->original_kernarg_size, 16u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 24u);
  EXPECT_NE(record.virtual_lds_base_sgpr, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);

  const auto normal_kd =
      read_descriptor_at_loaded_vaddr_for_test(result.elf_bytes, sidecar->normal_descriptor_vaddr);
  ASSERT_TRUE(normal_kd.has_value());
  EXPECT_EQ(normal_kd->kernarg_size, 16u);

  const auto sidecar_kd =
      read_descriptor_at_loaded_vaddr_for_test(result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_EQ(sidecar_kd->kernarg_size, 48u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarZeroKernargUsesWrapperKernarg) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(source_kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                  1);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  const auto saddr =
      find_flat_global_load_saddr(target_words, target_word_count, /*op=*/20, /*vdst=*/7,
                                  /*addr=*/4, /*offset=*/0x134);
  ASSERT_TRUE(saddr.has_value());
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(target_words, target_word_count, *saddr,
                                                   /*sbase_sgpr=*/2, /*offset=*/8));

  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(extension->original_kernarg_size, 0u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 8u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);

  const auto normal_kd =
      read_descriptor_at_loaded_vaddr_for_test(result.elf_bytes, sidecar->normal_descriptor_vaddr);
  const auto sidecar_kd =
      read_descriptor_at_loaded_vaddr_for_test(result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(normal_kd.has_value());
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_EQ(record.normal_private_segment_size, normal_kd->private_segment_fixed_size);
  EXPECT_EQ(record.virtual_private_segment_size, sidecar_kd->private_segment_fixed_size);
  EXPECT_EQ(sidecar_kd->kernarg_size, 32u);
  EXPECT_EQ(AMDHSA_BITS_GET(sidecar_kd->kernel_code_properties,
                            KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR),
            1u)
      << "kernel_code_properties=0x" << std::hex << sidecar_kd->kernel_code_properties << std::dec;
  EXPECT_EQ(AMDHSA_BITS_GET(sidecar_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT), 4u);
}

TEST(BinaryTranslatorE2E, LdsKernelEmitsNormalAndVirtualDescriptorVariants) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);

  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(record.kernel_name, "kernel");
  EXPECT_EQ(sidecar->normal_descriptor_vaddr, translated.kernel_descriptor_offset("kernel"));
  EXPECT_NE(sidecar->variant_descriptor_vaddr, sidecar->normal_descriptor_vaddr);
  EXPECT_EQ(record.static_lds_bytes, 6144u);
  EXPECT_EQ(extension->original_kernarg_size, 16u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 24u);
  EXPECT_NE(record.virtual_lds_base_sgpr, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);

  const auto *translated_rodata = rocjitsu::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.group_segment_fixed_size, 6144u);
  EXPECT_EQ(normal_kd.kernarg_size, 16u);

  const auto virtual_descriptor_offset =
      rocjitsu::loaded_vaddr_to_file_offset(result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(virtual_kd.kernarg_size, 48u);
}

TEST(BinaryTranslatorE2E, MubufLdsKernelEmitsNormalAndVirtualDescriptorVariants) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto mubuf = make_cdna4_buffer_load_lds_words(/*op=*/23);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {mubuf[0], mubuf[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  // MUBUF instructions encode the LDS side effect in a modifier bit, not in the
  // mnemonic. The sidecar detector must inspect the decoded machine fields, or
  // MUBUF-only LDS kernels would keep only the normal hardware-LDS descriptor.
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);

  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(record.kernel_name, "kernel");
  EXPECT_EQ(sidecar->normal_descriptor_vaddr, translated.kernel_descriptor_offset("kernel"));
  EXPECT_NE(sidecar->variant_descriptor_vaddr, sidecar->normal_descriptor_vaddr);
  EXPECT_EQ(record.static_lds_bytes, 6144u);
  EXPECT_EQ(extension->original_kernarg_size, 16u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 24u);
  EXPECT_NE(record.virtual_lds_base_sgpr, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);

  const auto *translated_rodata = rocjitsu::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.group_segment_fixed_size, 6144u);
  EXPECT_EQ(normal_kd.kernarg_size, 16u);

  const auto virtual_descriptor_offset =
      rocjitsu::loaded_vaddr_to_file_offset(result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(virtual_kd.kernarg_size, 48u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarMaterializesLargeDsReadB32Offset) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  constexpr uint16_t kLargeDsOffset = 0x1234;
  const auto ds = make_cdna4_ds_read_b32_words(kLargeDsOffset);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/7,
                                        /*addr=*/4, /*offset=*/0));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadU16ToFlatGlobalLoadUshort) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_u16_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/18, /*vdst=*/7,
                                        /*addr=*/4, /*offset=*/0x20));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadU8ToFlatGlobalLoadUbyte) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_u8_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/16, /*vdst=*/12,
                                        /*addr=*/8, /*offset=*/0x20));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadI8ToFlatGlobalLoadSbyte) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_i8_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/17, /*vdst=*/12,
                                        /*addr=*/8, /*offset=*/0x20));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWriteB8ToFlatGlobalStoreByte) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b8_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/24, /*data=*/7,
                                         /*addr=*/4, /*offset=*/0x10));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWriteB8D16HiToFlatGlobalStoreByteD16Hi) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b8_d16_hi_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/25, /*data=*/7,
                                         /*addr=*/4, /*offset=*/0x10));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesDsWriteB96DataOverlappingAddressPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b96_words(/*addr=*/4, /*data=*/4);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_data = find_flat_global_store_data(target_words, target_word_count, /*op=*/30,
                                                       /*addr=*/4, /*offset=*/0);
  ASSERT_TRUE(staged_data.has_value());
  EXPECT_NE(*staged_data, 4u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, *staged_data,
                                  /*src0=*/256 + 4));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1,
                                  static_cast<uint8_t>(*staged_data + 1), /*src0=*/256 + 5));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1,
                                  static_cast<uint8_t>(*staged_data + 2), /*src0=*/256 + 6));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddDsAddressIntoEvenGlobalAddressPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b128_words(/*addr=*/7, /*data=*/58);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_addr =
      find_flat_global_store_addr_for_data(target_words, target_word_count, /*op=*/31,
                                           /*data=*/58, /*offset=*/0);
  ASSERT_TRUE(staged_addr.has_value());
  EXPECT_EQ(*staged_addr % 2, 0u);
  EXPECT_NE(*staged_addr, 7u);
  EXPECT_EQ(find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/7, /*literal=*/0),
            staged_addr);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersAccDsWriteB128ToAccFlatGlobalStore) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b128_words(/*addr=*/28, /*data=*/30, /*byte_offset=*/0x40,
                                                 /*acc=*/true);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store_acc(target_words, target_word_count, /*op=*/31,
                                             /*data=*/30, /*addr=*/28, /*offset=*/0x40));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddWideStoreDataIntoEvenTuple) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b64_words(/*addr=*/4, /*data=*/7);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_data = find_flat_global_store_data(target_words, target_word_count, /*op=*/29,
                                                       /*addr=*/4, /*offset=*/0);
  ASSERT_TRUE(staged_data.has_value());
  EXPECT_EQ(*staged_data % 2, 0u);
  EXPECT_NE(*staged_data, 7u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, *staged_data,
                                  /*src0=*/256 + 7));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1,
                                  static_cast<uint8_t>(*staged_data + 1), /*src0=*/256 + 8));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddWideLoadResultIntoEvenTuple) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b64_words(/*addr=*/4, /*vdst=*/7);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_vdst = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/21,
                                                      /*addr=*/4, /*offset=*/0);
  ASSERT_TRUE(staged_vdst.has_value());
  EXPECT_EQ(*staged_vdst % 2, 0u);
  EXPECT_NE(*staged_vdst, 7u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/7,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/8,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 1)));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadB128ToFlatGlobalLoad) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b128_words(/*addr=*/28, /*vdst=*/30, /*byte_offset=*/0x40);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/23, /*vdst=*/30,
                                        /*addr=*/28, /*offset=*/0x40));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersAccDsReadB128ToAccFlatGlobalLoad) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b128_words(/*addr=*/28, /*vdst=*/30, /*byte_offset=*/0x40,
                                                /*acc=*/true);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load_acc(target_words, target_word_count, /*op=*/23,
                                            /*vdst=*/30, /*addr=*/28, /*offset=*/0x40));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddDsReadB128AddressIntoEvenGlobalPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b128_words(/*addr=*/27, /*vdst=*/30);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_addr =
      find_flat_global_load_addr_for_vdst(target_words, target_word_count, /*op=*/23,
                                          /*vdst=*/30, /*offset=*/0);
  ASSERT_TRUE(staged_addr.has_value());
  EXPECT_EQ(*staged_addr % 2, 0u);
  EXPECT_NE(*staged_addr, 27u);
  EXPECT_EQ(find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/27, /*literal=*/0),
            staged_addr);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddDsReadB128ResultIntoEvenTuple) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b128_words(/*addr=*/28, /*vdst=*/31);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_vdst = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/23,
                                                      /*addr=*/28, /*offset=*/0);
  ASSERT_TRUE(staged_vdst.has_value());
  EXPECT_EQ(*staged_vdst % 2, 0u);
  EXPECT_NE(*staged_vdst, 31u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/31,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/32,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 1)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/33,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 2)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/34,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 3)));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWrite2B32ToFlatGlobalStores) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write2_b32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/28, /*data=*/7,
                                         /*addr=*/4, /*offset=*/4));
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/28, /*data=*/9,
                                         /*addr=*/4, /*offset=*/8));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWrite2B64ToFlatGlobalStores) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write2_b64_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/8,
                                         /*addr=*/4, /*offset=*/8));
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/12,
                                         /*addr=*/4, /*offset=*/16));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesDsWrite2B64DataOverlappingAddressPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write2_b64_words(/*addr=*/8, /*data0=*/8, /*data1=*/12);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_data = find_flat_global_store_data(target_words, target_word_count, /*op=*/29,
                                                       /*addr=*/8, /*offset=*/8);
  ASSERT_TRUE(staged_data.has_value());
  EXPECT_NE(*staged_data, 8u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, *staged_data,
                                  /*src0=*/256 + 8));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1,
                                  /*vdst=*/static_cast<uint8_t>(*staged_data + 1),
                                  /*src0=*/256 + 9));
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/12,
                                         /*addr=*/8, /*offset=*/16));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWrite2st64B64ToFlatGlobalStores) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write2st64_b64_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/8,
                                         /*addr=*/4, /*offset=*/512));
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/12,
                                         /*addr=*/4, /*offset=*/1024));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsRead2B32ToFlatGlobalLoads) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2_b32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/20,
                                        /*addr=*/12, /*offset=*/4));
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/21,
                                        /*addr=*/12, /*offset=*/8));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesDsRead2B32ResultsOverlappingAddressPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2_b32_words(/*offset0=*/1, /*offset1=*/2, /*addr=*/20,
                                                /*vdst=*/20);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto temp0 = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/20,
                                                /*addr=*/20, /*offset=*/4);
  const auto temp1 = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/20,
                                                /*addr=*/20, /*offset=*/8);
  ASSERT_TRUE(temp0.has_value());
  ASSERT_TRUE(temp1.has_value());
  EXPECT_NE(*temp0, 20u);
  EXPECT_NE(*temp1, 21u);
  EXPECT_EQ(*temp1, static_cast<uint8_t>(*temp0 + 1));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/20,
                                  /*src0=*/static_cast<uint16_t>(256u + *temp0)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/21,
                                  /*src0=*/static_cast<uint16_t>(256u + *temp1)));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsRead2st64B32ToFlatGlobalLoads) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2st64_b32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/20,
                                        /*addr=*/12, /*offset=*/512));
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/21,
                                        /*addr=*/12, /*offset=*/768));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsRead2B64ToFlatGlobalLoads) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2_b64_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/21, /*vdst=*/66,
                                        /*addr=*/58, /*offset=*/24));
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/21, /*vdst=*/68,
                                        /*addr=*/58, /*offset=*/544));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddWideRead2ResultsIntoEvenTuple) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2_b64_words(/*offset0=*/3, /*offset1=*/68, /*addr=*/58,
                                                /*vdst=*/67);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_vdst = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/21,
                                                      /*addr=*/58, /*offset=*/24);
  ASSERT_TRUE(staged_vdst.has_value());
  EXPECT_EQ(*staged_vdst % 2, 0u);
  EXPECT_NE(*staged_vdst, 67u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/67,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/68,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 1)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/69,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 2)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/70,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 3)));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsRead2st64B64ToFlatGlobalLoads) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2st64_b64_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/21, /*vdst=*/112,
                                        /*addr=*/98, /*offset=*/1024));
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/21, /*vdst=*/114,
                                        /*addr=*/98, /*offset=*/1536));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarUsesDistinctSpillSlotsForTwoAddressTemps) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  // vdst aliases addr, so the replacement must preserve the original address
  // VGPR. Offsets 9 and 10 exceed the flat/global immediate range after B64 ST64
  // scaling. The descriptor is already at the ordinary VGPR ceiling and the body
  // touches s[100:101], so virtual LDS must use the spill-per-use path: the
  // preserved address temp and two backing-pointer VGPR temps are live together
  // and must receive distinct private scratch slots.
  const auto ds = make_cdna4_ds_read2st64_b64_words(/*offset0=*/9, /*offset1=*/10,
                                                    /*addr=*/4, /*vdst=*/4);
  const uint32_t touch_s100 =
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t touch_s101 =
      rocjitsu::build_s_mov_b32(/*sdst=*/101, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {touch_s100, touch_s101, ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 63);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/28,
                                                 /*offset=*/16, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/28,
                                                 /*offset=*/20, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/28,
                                                 /*offset=*/24, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/20,
                                                 /*offset=*/16, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/20,
                                                 /*offset=*/20, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/20,
                                                 /*offset=*/24, /*is_load=*/true));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadB64TrB16LoadToFlatGlobal) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b64_tr_b16_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_addr =
      find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/2, /*literal=*/0);
  ASSERT_TRUE(staged_addr.has_value());
  EXPECT_EQ(*staged_addr % 2, 0u);
  EXPECT_TRUE(contains_vop3_mov_b32(target_words, target_word_count,
                                    static_cast<uint8_t>(*staged_addr + 1),
                                    rocjitsu::scalar_positive_inline_u32(0)));
  EXPECT_TRUE(contains_flat_global_load_addr(target_words, target_word_count, /*op=*/21,
                                             *staged_addr, /*offset=*/0));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddDsReadB64TrB16AddressIntoEvenGlobalPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b64_tr_b16_words(/*byte_offset=*/0, /*addr=*/7);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_addr =
      find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/7, /*literal=*/0);
  ASSERT_TRUE(staged_addr.has_value());
  EXPECT_EQ(*staged_addr % 2, 0u);
  EXPECT_NE(*staged_addr, 7u);
  EXPECT_TRUE(contains_vop3_mov_b32(target_words, target_word_count,
                                    static_cast<uint8_t>(*staged_addr + 1),
                                    rocjitsu::scalar_positive_inline_u32(0)));
  EXPECT_TRUE(contains_flat_global_load_addr(target_words, target_word_count, /*op=*/21,
                                             *staged_addr, /*offset=*/0));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarMaterializesLargeDsReadB64TrB16Offset) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  constexpr uint16_t kLargeDsOffset = 0x1234;
  const auto ds = make_cdna4_ds_read_b64_tr_b16_words(kLargeDsOffset);
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  const auto temp =
      find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/2, kLargeDsOffset);
  ASSERT_TRUE(temp.has_value());
  EXPECT_EQ(*temp % 2, 0u);
  EXPECT_TRUE(contains_vop3_mov_b32(target_words, target_word_count,
                                    static_cast<uint8_t>(*temp + 1),
                                    rocjitsu::scalar_positive_inline_u32(0)));
  EXPECT_TRUE(contains_flat_global_load_addr(target_words, target_word_count, /*op=*/21, *temp,
                                             /*offset=*/0));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarRejectsUnsupportedDsMemoryOpcode) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_add_u32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));
  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      105600u);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &d) {
    return d.message.find("virtual LDS lowering does not support this DS opcode") !=
           std::string::npos;
  });
  EXPECT_NE(diagnostic, result.diagnostics.end());
}

TEST(BinaryTranslatorE2E, SkipFailedVirtualLdsSidecarKeepsNormalDescriptor) {
  // Static LDS that fits the host limit: the sidecar is only needed for
  // *dynamic* LDS overflow, so the normal hardware-LDS descriptor is a valid
  // launch target on its own. A sidecar lowering failure must therefore leave
  // the normal descriptor intact.
  constexpr uint32_t kBelowHostLdsBytes = 1024u;
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_add_u32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));
  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      kBelowHostLdsBytes);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());
  const auto skipped = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == rocjitsu::DiagnosticKind::KernelSkipped &&
           diagnostic.message.find("virtual LDS lowering does not support this DS opcode") !=
               std::string::npos;
  });
  ASSERT_NE(skipped, result.diagnostics.end());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *translated_text = translated.text_sections()[0];
  const auto *translated_rodata = rocjitsu::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  ASSERT_GE(translated_rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));
  const auto normal_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, translated_rodata->sectionOffset());
  const int64_t normal_entry_vaddr = static_cast<int64_t>(translated_rodata->vaddr()) +
                                     rocjitsu::read_kernel_descriptor_entry_offset(&normal_kd);
  ASSERT_GE(normal_entry_vaddr, 0);
  const auto normal_entry_file_offset = rocjitsu::loaded_vaddr_to_file_offset(
      result.elf_bytes, static_cast<uint64_t>(normal_entry_vaddr));
  ASSERT_TRUE(normal_entry_file_offset.has_value());

  // The optional virtual-LDS sidecar failed, but below-threshold launches still
  // use the normal descriptor. Its entry must therefore continue to point at the
  // translated hardware-LDS body, not at the skipped-kernel stub appended for
  // the sidecar variant.
  EXPECT_EQ(*normal_entry_file_offset, translated_text->sectionOffset());
  EXPECT_EQ(rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName), nullptr);
}

TEST(BinaryTranslatorE2E, SkipFailedVirtualLdsSidecarStubsOversizedNormalDescriptor) {
  // Static LDS that exceeds the host limit: the normal descriptor advertises
  // more hardware LDS than the host has, so it is only launchable through its
  // virtual sidecar. When the sidecar lowering fails and is skipped, the normal
  // descriptor must be stubbed too — leaving it dispatchable would fault the
  // host at launch. This is the regression guard for the sidecar-dependent case.
  constexpr uint32_t kOverHostLdsBytes = 105600u;
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_add_u32_words();
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));
  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      kOverHostLdsBytes);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());
  const auto skipped = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == rocjitsu::DiagnosticKind::KernelSkipped &&
           diagnostic.message.find("virtual LDS lowering does not support this DS opcode") !=
               std::string::npos;
  });
  ASSERT_NE(skipped, result.diagnostics.end());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *translated_rodata = rocjitsu::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  ASSERT_GE(translated_rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));
  const auto normal_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      result.elf_bytes, translated_rodata->sectionOffset());
  const int64_t normal_entry_vaddr = static_cast<int64_t>(translated_rodata->vaddr()) +
                                     rocjitsu::read_kernel_descriptor_entry_offset(&normal_kd);
  ASSERT_GE(normal_entry_vaddr, 0);
  const auto normal_entry_file_offset = rocjitsu::loaded_vaddr_to_file_offset(
      result.elf_bytes, static_cast<uint64_t>(normal_entry_vaddr));
  ASSERT_TRUE(normal_entry_file_offset.has_value());

  // The normal descriptor's advertised static LDS (105600) exceeds the CDNA3
  // host limit, so it could only ever launch via the failed sidecar. Its entry
  // must now point at the skipped-kernel endpgm stub, and the descriptor must
  // advertise no fixed LDS so a launch attempt cannot fault the host.
  const auto *entry_words =
      reinterpret_cast<const uint32_t *>(result.elf_bytes.data() + *normal_entry_file_offset);
  EXPECT_EQ(entry_words[0], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(normal_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(rocjitsu::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName), nullptr);
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3Bitop3ScratchGrowsDescriptor) {
  constexpr uint16_t kScratchFloor = 120;
  const auto words = make_cdna4_bitop3_words(cdna4::kVBitop3B32Vop3, 16);
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

TEST(BinaryTranslatorE2E, Cdna4ToCdna3Bitop3UsesSpillBackedScratchWhenVgprsAreFull) {
  using namespace rocr::llvm::amdhsa;

  const auto words = make_cdna4_bitop3_words(cdna4::kVBitop3B32Vop3, 16);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({words[0], words[1]});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 63);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  // With find_free_run() forced above the physical VGPR namespace, the bitop3
  // lowering must borrow descriptor-backed v0/v1, preserve them in the reusable
  // per-lane semantic spill window, and restore them after writing VDST. The
  // forbidden set excludes v16..v19 so the borrowed window cannot clobber the
  // destination or any source while the LUT expression is being synthesized.
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/0, /*offset=*/0, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/1, /*offset=*/4, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/0, /*offset=*/0, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/1, /*offset=*/4, /*is_load=*/true));
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3CvtPkBf16SpillScratchAvoidsVgprSources) {
  using namespace rocr::llvm::amdhsa;

  const auto words = make_cdna4_cvt_pk_bf16_f32_words(/*vdst=*/0, /*src0=*/256 + 4,
                                                      /*src1=*/256 + 1);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({words[0], words[1]});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  1);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  // Force liveness to report no dead physical VGPR window. The BF16 lowering
  // must then spill a borrowed three-VGPR window, but that window cannot include
  // VDST or the still-needed high source. The old allocator borrowed v[1:3]
  // here and clobbered SRC1 before converting the high packed half.
  EXPECT_FALSE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                           /*vgpr=*/1, /*offset=*/0, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/2, /*offset=*/0, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/3, /*offset=*/4, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/4, /*offset=*/8, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/2, /*offset=*/0, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/3, /*offset=*/4, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/4, /*offset=*/8, /*is_load=*/true));
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3MfmaPartialScratchGrowsDescriptor) {
  constexpr uint16_t kScratchFloor = 121;
  constexpr uint16_t kAlignedScratch = 122;
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
  expect_cdna3_translated_descriptor_vgprs_at_least(result.elf_bytes, kAlignedScratch + 4);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  std::optional<rocjitsu::cdna3::Vop3pMfmaMachineInst> partial_mfma;
  for (size_t i = 0; i + 1 < target_word_count; ++i) {
    rocjitsu::cdna3::Vop3pMfmaMachineInst actual{};
    std::memcpy(&actual, target_words + i, sizeof(actual));
    if (actual.encoding == 0x1A7u && actual.op == 77u && actual.acc_cd == 0u) {
      partial_mfma = actual;
      break;
    }
  }
  ASSERT_TRUE(partial_mfma.has_value());
  EXPECT_EQ(partial_mfma->vdst, kAlignedScratch);
  EXPECT_EQ(partial_mfma->vdst % 2u, 0u);
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3Dot2ScratchGrowsDescriptor) {
  constexpr uint16_t kScratchFloor = 121;
  const auto words = make_cdna4_dot2_f32_bf16_words(/*vdst=*/0, /*src0=*/256 + 1,
                                                    /*src1=*/256 + 2, /*src2=*/256);
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
  // The BF16 dot2 fallback widens four packed halves into ordinary VGPR
  // temporaries before issuing scalar FP32 arithmetic. If liveness selects a
  // scratch run above the guest allocation, the descriptor must grow to cover
  // all four generated VGPRs.
  expect_cdna3_translated_descriptor_vgprs_at_least(result.elf_bytes, kScratchFloor + 4);
}

TEST(BinaryTranslatorE2E, Rdna4ScratchAllocationDoesNotWrapPastV255) {
  const auto words = make_cdna4_mfma_words(cdna4::kVMfmaF3216x16x16F16Vop3pMfma, 0, 256, 260);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({words[0], words[1]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4, 0,
                                        options);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "MFMA lowering could not find a free VGPR for ds_bpermute addresses"));
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3PermlaneRejectsDescriptorFullExecSave) {
  using namespace rocr::llvm::amdhsa;

  const auto permlane =
      make_cdna4_permlane32_swap_b32_words(/*encoding_id=*/cdna4::encoding::kVop1);
  std::vector<uint32_t> words = {permlane[0], permlane[1]};
  for (uint16_t sgpr = 0; sgpr < 102; ++sgpr) {
    // Keep every ordinary SGPR live after the permlane replacement point. With
    // the descriptor already full, the EXEC-save helper must then fail instead
    // of borrowing and clobbering one of these guest scalar values.
    words.push_back(rocjitsu::build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_CDNA4));
  }
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  12);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &d) {
    return d.kind == rocjitsu::DiagnosticKind::ExpandFailed &&
           d.message.find("No free descriptor-backed SGPR pair for EXEC save/restore") !=
               std::string::npos;
  });
  EXPECT_NE(diagnostic, result.diagnostics.end());
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3PermlaneExecSaveReservesSpecialSgprTail) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto permlane =
      make_cdna4_permlane32_swap_b32_words(/*encoding_id=*/cdna4::encoding::kVop1);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {permlane[0], permlane[1], kCdna4SEndpgm});

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  // This mirrors the Qwen SDPA `attn_fwd` shape: the source descriptor allocates
  // 64 SGPRs, while the semantic permlane lowering introduces an EXEC-save pair
  // starting at s64. Growing only through s65 leaves no room for the
  // architecture-owned VCC/flat-scratch/XNACK tail, so DBT must reserve that
  // tail explicitly when materializing the generated ordinary SGPR pair.
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  7);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());
  expect_cdna3_translated_descriptor_sgprs_eq(result.elf_bytes, 80);
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3Permlane16SwapWritesBothRowPairs) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto permlane =
      make_cdna4_permlane16_swap_b32_words(/*encoding_id=*/cdna4::encoding::kVop1);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {permlane[0], permlane[1], kCdna4SEndpgm});

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  // V_PERMLANE16_SWAP_B32 swaps both 16-lane row pairs: lanes 0..15 with
  // 16..31, and lanes 32..47 with 48..63. The lowering therefore emits the
  // same low/high row masks into EXEC_LO and EXEC_HI.
  constexpr uint8_t kExecLoSgpr = 126;
  constexpr uint8_t kExecHiSgpr = 127;
  EXPECT_EQ(
      count_cdna3_s_mov_b32_literal(target_words, target_word_count, kExecLoSgpr, 0x0000ffffu), 1u);
  EXPECT_EQ(
      count_cdna3_s_mov_b32_literal(target_words, target_word_count, kExecLoSgpr, 0xffff0000u), 1u);
  EXPECT_EQ(
      count_cdna3_s_mov_b32_literal(target_words, target_word_count, kExecHiSgpr, 0x0000ffffu), 1u);
  EXPECT_EQ(
      count_cdna3_s_mov_b32_literal(target_words, target_word_count, kExecHiSgpr, 0xffff0000u), 1u);
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
  // branch immediates are patched against the compact target layout. Conditional
  // direct branches still reserve their patch window, so the compact body keeps
  // one target-ISA NOP slot beside each conditional transfer.
  const std::vector<uint32_t> expected = {
      rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3),
      cdna3::build_sopp(cdna3::kSCbranchScc1Sopp, {.simm16 = 2})[0],
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3),
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

TEST(BinaryTranslatorE2E, PatchesRecoveredSetpcTargetAfterRelocation) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 20;
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
  ASSERT_GE(translated.text_sections()[0]->size(), 11 * sizeof(uint32_t));
  // Recovered indirect jumps now patch the consumer site rather than rewriting
  // the source-side getpc/add builder. The fixed six-word transfer window keeps
  // block placement deterministic before the final target address is known.
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand));
  EXPECT_EQ(target_words[2], kOriginalGetpcDelta);
  EXPECT_EQ(target_words[3], build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0));
  EXPECT_EQ(target_words[4], rocjitsu::build_s_branch(5, ROCJITSU_CODE_ARCH_CDNA3));
  expect_nop_words(target_words, 5, 10, ROCJITSU_CODE_ARCH_CDNA3);
  EXPECT_EQ(target_words[10], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, PatchesRecoveredSwappcTargetAfterRelocation) {
  constexpr uint16_t kPcSreg = 10;
  constexpr uint16_t kReturnSreg = 20;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 24;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),               // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),                 // 0x04.
      kOriginalGetpcDelta,                                                // 0x08.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),            // 0x0c.
      build_s_swappc_b64(kReturnSreg, kPcSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x10.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x14 unreachable continuation.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4), // 0x18 unreachable.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x1c target.
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
  ASSERT_GE(translated.text_sections()[0]->size(), 11 * sizeof(uint32_t));
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand));
  EXPECT_EQ(target_words[2], kOriginalGetpcDelta);
  EXPECT_EQ(target_words[3], build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0));
  // The target is non-returning, so relocation drops the dead continuation.
  // The pre-existing recovered-transfer window remains in place on this branch.
  EXPECT_EQ(target_words[4], build_s_call_b64(kReturnSreg, 5));
  expect_nop_words(target_words, 5, 10, ROCJITSU_CODE_ARCH_CDNA3);
  EXPECT_EQ(target_words[10], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
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

TEST(BinaryTranslatorE2E, TranslatesNestedCallReturningThroughOuterPair) {
  constexpr uint16_t kOuterReturnSreg = 30;
  constexpr uint16_t kInnerReturnSreg = 28;
  const std::vector<uint32_t> words = {
      build_s_call_b64(kOuterReturnSreg, 1),              // 0x00 -> outer callee at 0x08.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 outer continuation.
      build_s_call_b64(kInnerReturnSreg, 1),              // 0x08 -> inner callee at 0x10.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x0c inner continuation.
      rocjitsu::pack_sopp(5, 1),                          // 0x10 -> outer return at 0x18.
      build_s_setpc_b64(kInnerReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x14 inner return.
      build_s_setpc_b64(kOuterReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x18 outer return.
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
  ASSERT_GE(decoded.size(), words.size());
  const auto count_mnemonic = [&](std::string_view mnemonic) {
    return std::ranges::count_if(
        decoded, [&](const auto &inst) { return inst != nullptr && inst->mnemonic() == mnemonic; });
  };
  EXPECT_EQ(count_mnemonic("s_call_b64"), 2);
  EXPECT_EQ(count_mnemonic("s_setpc_b64"), 2);
}

TEST(BinaryTranslatorE2E, TranslatesSwappcCallWhenCalleeSetpcBranchesToReturn) {
  constexpr uint16_t kCallTargetSreg = 10;
  constexpr uint16_t kReturnTargetSreg = 12;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalCallTargetDelta = 28;
  constexpr uint32_t kOriginalReturnTargetDelta = 20;
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
  ASSERT_GE(translated.text_sections()[0]->size(), 22 * sizeof(uint32_t));
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kCallTargetSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[2], kOriginalCallTargetDelta);
  EXPECT_EQ(target_words[4], build_s_call_b64(kReturnSreg, 6))
      << "the swappc call window should patch directly to the compact callee body";
  expect_nop_words(target_words, 5, 10, ROCJITSU_CODE_ARCH_CDNA3);
  EXPECT_EQ(target_words[10], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[11], build_s_getpc_b64(kReturnTargetSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[13], kOriginalReturnTargetDelta);
  EXPECT_EQ(target_words[15], rocjitsu::build_s_branch(5, ROCJITSU_CODE_ARCH_CDNA3))
      << "the callee's recovered setpc window is what reaches the return block";
  expect_nop_words(target_words, 16, 21, ROCJITSU_CODE_ARCH_CDNA3);
  EXPECT_EQ(target_words[21], build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, PatchesOneRecoveredBuilderUsedByTwoSetpcConsumers) {
  constexpr uint16_t kPcSreg = 12;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 28;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),            // 0x04.
      kOriginalGetpcDelta,                                           // 0x08.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),       // 0x0c.
      cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = 1})[0], // 0x10 -> second consumer.
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
  ASSERT_GE(translated.text_sections()[0]->size(), 18 * sizeof(uint32_t));
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand));
  EXPECT_EQ(target_words[2], kOriginalGetpcDelta);
  EXPECT_EQ(target_words[3], build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0));
  EXPECT_EQ(target_words[4], cdna3::build_sopp(cdna3::kSCbranchScc1Sopp, {.simm16 = 7})[0]);
  EXPECT_EQ(target_words[5], rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[6], rocjitsu::build_s_branch(11, ROCJITSU_CODE_ARCH_CDNA3));
  expect_nop_words(target_words, 7, 12, ROCJITSU_CODE_ARCH_CDNA3);
  EXPECT_EQ(target_words[12], rocjitsu::build_s_branch(5, ROCJITSU_CODE_ARCH_CDNA3));
  expect_nop_words(target_words, 13, 18, ROCJITSU_CODE_ARCH_CDNA3);
  EXPECT_EQ(target_words[18], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, RewritesDistinctBuildersForOneMultiTargetSetpcConsumer) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalTargetADelta = 44;
  constexpr uint32_t kOriginalTargetBDelta = 28;
  constexpr uint32_t kRelocatedTargetADelta = 40;
  constexpr uint32_t kRelocatedTargetBDelta = 24;
  const std::vector<uint32_t> words = {
      cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = 5})[0], // 0x00 -> builder B.
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x04 builder A.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),            // 0x08.
      kOriginalTargetADelta,                                         // 0x0c -> target A.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),       // 0x10.
      rocjitsu::build_s_branch(5, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14 -> consumer.
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x18 builder B.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),            // 0x1c.
      kOriginalTargetBDelta,                                         // 0x20 -> target B.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),       // 0x24.
      rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4),         // 0x28 -> consumer.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x2c multi-target consumer.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x30 unreachable.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x34 target A.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x38 target B.
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
  ASSERT_GE(translated.text_sections()[0]->size(), 15 * sizeof(uint32_t));
  EXPECT_EQ(target_words[4], kRelocatedTargetADelta)
      << "builder A should be rewritten once for its relocated target";
  EXPECT_EQ(target_words[9], kRelocatedTargetBDelta)
      << "builder B should be rewritten once for its relocated target";
  EXPECT_EQ(target_words[12], build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3))
      << "one consumer with multiple possible targets must stay indirect";
  EXPECT_EQ(target_words[13], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[14], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
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

TEST(BinaryTranslatorE2E, EndpgmAfterTrapTerminatesCfgBeforeFollowingFunction) {
  // A resumable S_TRAP is not a CFG terminator: on hardware it transfers to the
  // trap handler and may return to the next instruction. A real terminator
  // (S_ENDPGM) must follow it to end the block. The S_ENDPGM terminates the CFG
  // so the following ELF function bytes stay unreachable and the unrecovered
  // S_SETPC_B64 below is never decoded into the CFG.
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> trap block.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x04 unreachable gap.
      build_s_trap(3),                                       // 0x08 falls through to endpgm.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x0c terminates the block.
      build_s_setpc_b64(30, ROCJITSU_CODE_ARCH_CDNA4),       // 0x10 next function body.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x14 unreachable.
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
  ASSERT_GE(decoded.size(), 3u);
  EXPECT_EQ(decoded[0]->mnemonic(), "s_branch");
  EXPECT_EQ(decoded[1]->mnemonic(), "s_trap");
  EXPECT_EQ(decoded[2]->mnemonic(), "s_endpgm");
  // The trailing S_ENDPGM (a real terminator) prevents a bogus fallthrough into
  // the following ELF function bytes, so the unrecovered S_SETPC_B64 is never
  // reached and translation does not emit it.
  EXPECT_TRUE(std::none_of(decoded.begin(), decoded.end(),
                           [](const auto &inst) { return inst->mnemonic() == "s_setpc_b64"; }));
}

TEST(BinaryTranslatorE2E, ResumableTrapFallsThroughAndDoesNotHideFollowingCode) {
  // Trap IDs other than ROCr's assertion/abort trap remain resumable, so code
  // after them is reachable. Here the fallthrough reaches an unrecovered
  // S_SETPC_B64, which the translator must surface as an error.
  const std::vector<uint32_t> words = {
      build_s_trap(3),                                 // 0x00 falls through.
      build_s_setpc_b64(30, ROCJITSU_CODE_ARCH_CDNA4), // 0x04 reachable, unrecovered.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(
      rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::Legalization,
                                     "indirect branch or call target recovery is not implemented"));
}

TEST(BinaryTranslatorE2E, RocrAbortTrapTerminatesCfgBeforeFollowingCode) {
  // ROCr reserves trap ID 2 for assertion/abort handling. That path does not
  // return, so bytes after S_TRAP 2 must not be pulled into the current CFG.
  const std::vector<uint32_t> words = {
      build_s_trap(2),                                 // 0x00 terminates the program path.
      build_s_setpc_b64(30, ROCJITSU_CODE_ARCH_CDNA4), // 0x04 unrelated, unreachable code.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
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
  ASSERT_FALSE(decoded.empty());
  EXPECT_EQ(decoded[0]->mnemonic(), "s_trap");
  EXPECT_TRUE(std::none_of(decoded.begin(), decoded.end(),
                           [](const auto &inst) { return inst->mnemonic() == "s_setpc_b64"; }));
}

TEST(BinaryTranslatorE2E, RocrAbortDeadEdgeDoesNotPoisonRecoveredCall) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Exercise BinaryTranslator's ExplicitOnly entry policy rather than calling
  // BasicBlock::build with the policy directly:
  //
  //   builder --conditional--------------------------> call
  //                 |
  //                 +--> s_trap 2 -X-> dead branch --^
  //
  // Treating the dead post-trap block as an inferred entry would add an
  // unconstrained path to s[8:9] and make translation fail closed.
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),      // 0x04.
      40,                                                      // 0x08: 0x04 + 40 = helper 0x2c.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c.
      cdna4::build_sopp(cdna4::kSCbranchScc0Sopp, {.simm16 = 3})[0],      // 0x10 -> 0x20.
      build_s_trap(2),                                                    // 0x14.
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),              // 0x18 -> 0x20.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                 // 0x1c.
      build_s_swappc_b64(kReturnSreg, kPcSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x20.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                 // 0x24.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                 // 0x28.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4),           // 0x2c.
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
  EXPECT_TRUE(std::ranges::any_of(
      decoded, [](const auto &inst) { return inst->mnemonic() == "s_call_b64"; }));
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

TEST(BinaryTranslatorE2E, PlacesDescriptorPrologueBeforeLargeRelocatedBody) {
  constexpr size_t kBodyWordsPastBranchRange = 32769;
  std::vector<uint32_t> words(kBodyWordsPastBranchRange, 0xBF800000u);
  words.push_back(0xBF810000u);

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_workgroup_id_x_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);

  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_FALSE(rocjitsu::has_error_containing(
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

TEST(BinaryTranslatorE2E, Gfx1250CopiesUnaffectedInstructionsForB0ToA0) {
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({kGfx1250SNop, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_EQ(result.host_arch, ROCJITSU_CODE_ARCH_GFX1250);
}

TEST(BinaryTranslatorE2E, Gfx1250ReplacesSClauseWithNopForB0ToA0) {
  constexpr auto source_clause = gfx1250::build_sopp(gfx1250::kSClauseSopp, {.simm16 = 4});
  constexpr auto source_end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp, {.simm16 = 0});
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_clause[0], source_end[0]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], gfx1250::build_sopp(gfx1250::kSNopSopp, {.simm16 = 0})[0]);
  EXPECT_EQ(target_words[1], source_end[0]);
}

TEST(BinaryTranslatorE2E, Gfx1250PreservesSClauseOutsideB0ToA0) {
  constexpr auto source_clause = gfx1250::build_sopp(gfx1250::kSClauseSopp, {.simm16 = 4});
  constexpr auto source_end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp, {.simm16 = 0});

  for (const rocjitsu::ProcessorRevision revision : {
           rocjitsu::ProcessorRevision::Gfx1250A0,
           rocjitsu::ProcessorRevision::Gfx1250B0,
       }) {
    auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_clause[0], source_end[0]});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
                                          gfx1250_revision_options(revision, revision));
    auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(target_words[0], source_clause[0]);
    EXPECT_EQ(target_words[1], source_end[0]);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250RelocatesDirectCallForB0ToA0) {
  constexpr uint16_t kReturnSreg = 30;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(2, ROCJITSU_CODE_ARCH_GFX1250),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      rocjitsu::build_s_call_b64(kReturnSreg, 2, ROCJITSU_CODE_ARCH_GFX1250),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_GFX1250),
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_EQ(target_words[1],
            rocjitsu::build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_EQ(target_words[2], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_EQ(target_words[3], rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_GFX1250));

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_GE(decoded.size(), 4u);
  EXPECT_EQ(decoded[1]->mnemonic(), "s_call_i64");
  ASSERT_TRUE(decoded[1]->branch_offset_bytes().has_value());
  EXPECT_EQ(*decoded[1]->branch_offset_bytes(), 4);
}

TEST(BinaryTranslatorE2E, Gfx1250DirectTailTransferCompactionIsIdempotent) {
  constexpr uint16_t kReturnSreg = 30;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_GFX1250),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0],
            rocjitsu::build_s_call_b64(kReturnSreg, 0, ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_EQ(target_words[1], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250KernargPreloadUsesSingleDescriptorEntry) {
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({kGfx1250SNop, kGfx1250SEndpgm});

  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  const auto *source_rodata = rocjitsu::find_section(source_layout, ".rodata");
  ASSERT_NE(source_rodata, nullptr);
  auto source_kd =
      rocjitsu::read_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd.kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  AMDHSA_BITS_SET(source_kd.kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_OFFSET, 2);
  rocjitsu::write_kernel_descriptor_for_test(image.data() + source_rodata->sectionOffset(),
                                             source_kd);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *target_rodata = rocjitsu::find_section(translated, ".rodata");
  ASSERT_NE(target_rodata, nullptr);
  const auto target_kd = rocjitsu::read_kernel_descriptor_for_test(translated.image_data() +
                                                                   target_rodata->sectionOffset());
  EXPECT_EQ(target_kd.kernarg_preload, source_kd.kernarg_preload)
      << "gfx1250 preload fields remain active while DBT uses the descriptor's single entry";
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesUnaffectedLiteralOperandsForB0ToA0) {
  constexpr uint32_t kVMovB32Literal = 0x7E0202FFu;
  constexpr uint32_t kLiteral = 0x11223344u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kVMovB32Literal, kLiteral, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], kVMovB32Literal);
  EXPECT_EQ(target_words[1], kLiteral);
  EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesCvtPkFp8F32WhenClampIsClear) {
  constexpr auto source_cvt = gfx1250::build_vop3(
      gfx1250::kVCvtPkFp8F32Vop3, {.vdst = 30, .clamp = 0, .src0 = 256 + 22, .src1 = 256 + 2});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_cvt[0], source_cvt[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], source_cvt[0]);
  EXPECT_EQ(target_words[1], source_cvt[1]);
  EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250CvtPkFp8F32ClampSetStillFailsClosed) {
  constexpr auto source_cvt = gfx1250::build_vop3(
      gfx1250::kVCvtPkFp8F32Vop3, {.vdst = 30, .clamp = 1, .src0 = 256 + 22, .src1 = 256 + 2});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_cvt[0], source_cvt[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &item) {
    return item.severity == rocjitsu::DiagnosticSeverity::Error &&
           item.kind == rocjitsu::DiagnosticKind::ExpandMissing;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->mnemonic, "v_cvt_pk_fp8_f32");
}

TEST(BinaryTranslatorE2E, Gfx1250LiteralCvtPkFp8F32ClampSetFailsClosed) {
  constexpr auto source_cvt = gfx1250::build_vop3(
      gfx1250::kVCvtPkFp8F32Vop3, {.vdst = 30, .clamp = 1, .src0 = 255, .src1 = 256 + 2});
  constexpr uint32_t kLiteral = 0x3F800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_cvt[0], source_cvt[1], kLiteral, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &item) {
    return item.severity == rocjitsu::DiagnosticSeverity::Error &&
           item.kind == rocjitsu::DiagnosticKind::ExpandMissing;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->mnemonic, "v_cvt_pk_fp8_f32");
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesLiteralCvtF32Fp8E32WithoutClampEmulation) {
  constexpr auto conversion =
      gfx1250::build_vop1(gfx1250::kVCvtF32Fp8Vop1, {.src0 = 255, .vdst = 30});
  constexpr uint32_t kLiteral = 0x3F800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {conversion[0], kLiteral, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], conversion[0]);
  EXPECT_EQ(target_words[1], kLiteral);
  EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250EmulatesCvtF32Fp8E5m3ForA0) {
  constexpr auto conversion = gfx1250::build_vop3(
      gfx1250::kVCvtF32Fp8Vop3, {.vdst = 30, .opsel = 2, .clamp = 1, .src0 = 256 + 22});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {conversion[0], conversion[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "v_cvt_f32_fp8"; }),
            0);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "v_cndmask_b32"; }),
            2);

  // Assert the value-defining literal constants of the emitted sequence, not just
  // mnemonic counts: an off-by-one in the NaN threshold, exponent clamp, or the
  // F16/F32 reconstruction constants would otherwise pass. These constants are
  // distinctive, so scan the raw translated text words for each. (Reading them from
  // the raw text is robust; a decoded VOP3's raw_encoding() only spans its 2-dword
  // base and does not expose the trailing literal.)
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto contains_word = [&](uint32_t value) {
    return std::any_of(target_words, target_words + target_word_count,
                       [&](uint32_t w) { return w == value; });
  };
  EXPECT_TRUE(contains_word(0xffu)) << "missing E5M3 NaN threshold literal 0xff";
  EXPECT_TRUE(contains_word(0xf7u)) << "missing E5M3 exponent-clamp literal 0xf7";
  EXPECT_TRUE(contains_word(0x47800000u)) << "missing F16 reconstruction constant";
  EXPECT_TRUE(contains_word(0x7fa3d000u)) << "missing F32 max-finite constant";

  // Pin the byte-select. The source opsel=2 must map through the VOP3
  // byte-select bit swap to byte 1, so the single v_bfe_u32 that isolates the
  // packed byte reads bit offset 1*8=8 (inline-encoded as 128+8=136). Without
  // the swap it would read byte 2 (offset 16, encoded 144) and silently unpack
  // the wrong lane's fp8 datum. Inspect the decoded operand so a byte-select
  // regression cannot pass.
  int bfe_count = 0;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() != "v_bfe_u32")
      continue;
    ++bfe_count;
    const rocjitsu::Operand *offset = inst->src_operand(1);
    ASSERT_NE(offset, nullptr);
    EXPECT_EQ(offset->encoding_value(), 136) << "E5M3 byte-select must be byte 1 (offset 8)";
  }
  EXPECT_EQ(bfe_count, 1) << "E5M3 unpack must isolate the byte with exactly one v_bfe_u32";
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3CompareMasksTargetDeadSgprs) {
  constexpr auto conversion = gfx1250::build_vop3(
      gfx1250::kVCvtF32Fp8Vop3, {.vdst = 30, .opsel = 2, .clamp = 1, .src0 = 256 + 22});
  constexpr uint16_t kLiveSgprs = 8;
  auto image = rocjitsu::make_gfx1250_image_with_live_sgprs(conversion, kLiveSgprs);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  std::optional<uint16_t> nan_mask;
  std::optional<uint16_t> exp31_mask;
  std::vector<uint16_t> cndmask_predicates;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "v_cmp_eq_u32" || inst->mnemonic() == "v_cmp_lt_u32") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      gfx1250::Vop3MachineInst encoded{};
      std::memcpy(&encoded, inst->raw_encoding(), sizeof(encoded));
      if (inst->mnemonic() == "v_cmp_eq_u32")
        nan_mask = static_cast<uint16_t>(encoded.vdst);
      else
        exp31_mask = static_cast<uint16_t>(encoded.vdst);
    } else if (inst->mnemonic() == "v_cndmask_b32") {
      const rocjitsu::Operand *predicate = inst->src_operand(2);
      ASSERT_NE(predicate, nullptr);
      cndmask_predicates.push_back(predicate->encoding_value());
    }
  }
  ASSERT_TRUE(nan_mask.has_value());
  ASSERT_TRUE(exp31_mask.has_value());
  ASSERT_EQ(cndmask_predicates.size(), 2u);
  EXPECT_NE(*nan_mask, *exp31_mask);
  EXPECT_GE(*nan_mask, kLiveSgprs);
  EXPECT_GE(*exp31_mask, kLiveSgprs);
  EXPECT_EQ(cndmask_predicates[0], *exp31_mask);
  EXPECT_EQ(cndmask_predicates[1], *nan_mask);
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3UnpackFailsClosedWithoutPredicateSgprs) {
  constexpr auto conversion = gfx1250::build_vop3(
      gfx1250::kVCvtF32Fp8Vop3, {.vdst = 30, .opsel = 2, .clamp = 1, .src0 = 256 + 22});
  for (const uint16_t live_sgprs :
       {static_cast<uint16_t>(rocjitsu::REGISTER_SET_ALLOCATABLE_SGPRS - 1u),
        static_cast<uint16_t>(rocjitsu::REGISTER_SET_ALLOCATABLE_SGPRS)}) {
    SCOPED_TRACE(live_sgprs);
    auto image = rocjitsu::make_gfx1250_image_with_live_sgprs(conversion, live_sgprs);
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::has_error_containing(
        result, rocjitsu::DiagnosticKind::ExpandFailed,
        "gfx1250 E5M3 unpack could not allocate two dead SGPR masks"));
  }
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesWmmaOutsideB0ToA0Profile) {
  constexpr auto source_wmma = gfx1250::build_vop3p(
      gfx1250::kVWmmaF3216x16x32Bf16Vop3p, {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], source_wmma[0]);
  EXPECT_EQ(target_words[1], source_wmma[1]);
  EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250EqualRevisionsUseIdentityProfile) {
  constexpr auto source_ds2 = gfx1250::build_vds(
      gfx1250::kDsLoad2addrB32Vds, {.offset0 = 1, .offset1 = 3, .addr = 7, .vdst = 9});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  for (const rocjitsu::ProcessorRevision revision :
       {rocjitsu::ProcessorRevision::Gfx1250A0, rocjitsu::ProcessorRevision::Gfx1250B0}) {
    SCOPED_TRACE(revision == rocjitsu::ProcessorRevision::Gfx1250A0 ? "A0-to-A0" : "B0-to-B0");
    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
                                          gfx1250_revision_options(revision, revision));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(target_words[0], source_ds2[0]);
    EXPECT_EQ(target_words[1], source_ds2[1]);
    EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250RejectsA0ToB0Translation) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250A0,
                               rocjitsu::ProcessorRevision::Gfx1250B0));
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::Legalization,
                                             "A0-to-B0 translation is not supported"));
}

TEST(BinaryTranslatorE2E, Gfx1250ExpandsEveryDs2VariantForA0) {
  struct Ds2Case {
    const char *name;
    uint16_t source_opcode;
    uint16_t target_opcode;
    uint8_t element_dwords;
    bool stride64;
  };
  constexpr std::array cases = {
      Ds2Case{"store_b32", gfx1250::kDsStore2addrB32Vds, gfx1250::kDsStoreB32Vds, 1, false},
      Ds2Case{"store_stride64_b32", gfx1250::kDsStore2addrStride64B32Vds, gfx1250::kDsStoreB32Vds,
              1, true},
      Ds2Case{"storexchg_b32", gfx1250::kDsStorexchg2addrRtnB32Vds, gfx1250::kDsStorexchgRtnB32Vds,
              1, false},
      Ds2Case{"storexchg_stride64_b32", gfx1250::kDsStorexchg2addrStride64RtnB32Vds,
              gfx1250::kDsStorexchgRtnB32Vds, 1, true},
      Ds2Case{"load_b32", gfx1250::kDsLoad2addrB32Vds, gfx1250::kDsLoadB32Vds, 1, false},
      Ds2Case{"load_stride64_b32", gfx1250::kDsLoad2addrStride64B32Vds, gfx1250::kDsLoadB32Vds, 1,
              true},
      Ds2Case{"store_b64", gfx1250::kDsStore2addrB64Vds, gfx1250::kDsStoreB64Vds, 2, false},
      Ds2Case{"store_stride64_b64", gfx1250::kDsStore2addrStride64B64Vds, gfx1250::kDsStoreB64Vds,
              2, true},
      Ds2Case{"storexchg_b64", gfx1250::kDsStorexchg2addrRtnB64Vds, gfx1250::kDsStorexchgRtnB64Vds,
              2, false},
      Ds2Case{"storexchg_stride64_b64", gfx1250::kDsStorexchg2addrStride64RtnB64Vds,
              gfx1250::kDsStorexchgRtnB64Vds, 2, true},
      Ds2Case{"load_b64", gfx1250::kDsLoad2addrB64Vds, gfx1250::kDsLoadB64Vds, 2, false},
      Ds2Case{"load_stride64_b64", gfx1250::kDsLoad2addrStride64B64Vds, gfx1250::kDsLoadB64Vds, 2,
              true},
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr gfx1250::VdsBuilderFields source_fields{
      .offset0 = 3, .offset1 = 5, .addr = 20, .data0 = 30, .data1 = 40, .vdst = 50};

  for (const Ds2Case &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto source_ds2 = gfx1250::build_vds(test_case.source_opcode, source_fields);
    auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    ASSERT_GE(translated.text_sections()[0]->size(), 6 * sizeof(uint32_t));

    const uint16_t scale = static_cast<uint16_t>(test_case.element_dwords * sizeof(uint32_t) *
                                                 (test_case.stride64 ? 64 : 1));
    const bool plain_store = test_case.target_opcode == gfx1250::kDsStoreB32Vds ||
                             test_case.target_opcode == gfx1250::kDsStoreB64Vds;
    const auto first =
        gfx1250::build_vds(test_case.target_opcode,
                           {.offset0 = static_cast<uint8_t>((source_fields.offset0 * scale) & 0xff),
                            .offset1 = static_cast<uint8_t>((source_fields.offset0 * scale) >> 8),
                            .addr = source_fields.addr,
                            .data0 = source_fields.data0,
                            .vdst = static_cast<uint8_t>(plain_store ? 0 : source_fields.vdst)});
    const auto second =
        gfx1250::build_vds(test_case.target_opcode,
                           {.offset0 = static_cast<uint8_t>((source_fields.offset1 * scale) & 0xff),
                            .offset1 = static_cast<uint8_t>((source_fields.offset1 * scale) >> 8),
                            .addr = source_fields.addr,
                            .data0 = source_fields.data1,
                            .vdst = static_cast<uint8_t>(
                                plain_store ? 0 : source_fields.vdst + test_case.element_dwords)});
    EXPECT_EQ(target_words[0], first[0]);
    EXPECT_EQ(target_words[1], first[1]);
    EXPECT_EQ(target_words[2], second[0]);
    EXPECT_EQ(target_words[3], second[1]);
    EXPECT_EQ(target_words[4], gfx1250::build_sopp(gfx1250::kSWaitDscntSopp, {.simm16 = 0})[0]);
    EXPECT_EQ(target_words[5], kGfx1250SEndpgm);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2LoadPreservesAliasedAddressUntilBothLoadsIssue) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  // ADDR=v7 aliases the first destination half v7. The second load must issue
  // first, exactly matching the executable far-DS2 alias oracle.
  constexpr gfx1250::VdsBuilderFields fields{.offset0 = 1, .offset1 = 3, .addr = 7, .vdst = 7};
  constexpr auto source_ds2 = gfx1250::build_vds(gfx1250::kDsLoad2addrB32Vds, fields);
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());

  constexpr auto second =
      gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.offset0 = 12, .addr = fields.addr, .vdst = 8});
  constexpr auto first =
      gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.offset0 = 4, .addr = fields.addr, .vdst = 7});
  EXPECT_EQ(target_words[0], second[0]);
  EXPECT_EQ(target_words[1], second[1]);
  EXPECT_EQ(target_words[2], first[0]);
  EXPECT_EQ(target_words[3], first[1]);
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2ExchangeCyclicOverlapFailsClosed) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  // First destination v10 aliases ADDR; second destination v11 aliases DATA0.
  // Neither issue order preserves every compound source operand.
  constexpr auto source_ds2 = gfx1250::build_vds(
      gfx1250::kDsStorexchg2addrRtnB32Vds, {.addr = 10, .data0 = 11, .data1 = 20, .vdst = 10});
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &item) {
    return item.kind == rocjitsu::DiagnosticKind::ExpandFailed;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_NE(diagnostic->message.find("cyclic"), std::string::npos);
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2OversizedStride64OffsetFailsClosed) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr auto source_ds2 = gfx1250::build_vds(gfx1250::kDsLoad2addrStride64B64Vds,
                                                 {.offset1 = 255, .addr = 4, .vdst = 8});
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &item) {
    return item.kind == rocjitsu::DiagnosticKind::ExpandFailed;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_NE(diagnostic->message.find("16-bit"), std::string::npos);
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2MovesData1VgprBankToSecondData0Role) {
  constexpr auto set_vgpr_msb = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x10});
  constexpr auto source_ds2 =
      gfx1250::build_vds(gfx1250::kDsStore2addrB32Vds,
                         {.offset0 = 1, .offset1 = 2, .addr = 4, .data0 = 8, .data1 = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  // The generated bank transitions carry these immediates, and each generated
  // s_set_vgpr_msb must be preceded by an s_wait_xcnt drain. Collect the generated
  // transitions dynamically so the assertion does not depend on absolute offsets.
  std::vector<uint16_t> generated_modes;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() != "s_set_vgpr_msb")
      continue;
    ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
    const uint16_t imm = static_cast<uint16_t>(decoded[index]->raw_encoding()[0] & 0xffffu);
    // The 0x0010 transition is the copied source s_set_vgpr_msb; the two generated
    // transitions (0x1014, 0x1410) must each be guarded by s_wait_xcnt.
    if (imm != 0x0010) {
      ASSERT_GT(index, 0u);
      EXPECT_EQ(decoded[index - 1]->mnemonic(), "s_wait_xcnt");
    }
    generated_modes.push_back(imm);
  }
  EXPECT_EQ(generated_modes, (std::vector<uint16_t>{0x0010, 0x1014, 0x1410}));
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2AdjustsDestinationBankForSecondLoad) {
  constexpr auto source_ds2 = gfx1250::build_vds(
      gfx1250::kDsLoad2addrB64Vds, {.offset0 = 1, .offset1 = 2, .addr = 4, .vdst = 254});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  // Two generated bank transitions (0x0040, 0x4000), each guarded by s_wait_xcnt.
  std::vector<uint16_t> generated_modes;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() != "s_set_vgpr_msb")
      continue;
    ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
    ASSERT_GT(index, 0u);
    EXPECT_EQ(decoded[index - 1]->mnemonic(), "s_wait_xcnt");
    generated_modes.push_back(static_cast<uint16_t>(decoded[index]->raw_encoding()[0] & 0xffffu));
  }
  EXPECT_EQ(generated_modes, (std::vector<uint16_t>{0x0040, 0x4000}));
}

TEST(SemanticTranslator, Gfx1250ClassifiesLivenessFreeExpandRules) {
  std::vector<uint32_t> liveness_free_rules;
  for (const rocjitsu::TranslationRule &rule : rocjitsu::semantic_expand_rules_gfx1250_b0_to_a0()) {
    if (!rule.requires_liveness) {
      liveness_free_rules.push_back((static_cast<uint32_t>(rule.src_encoding_id) << 16) |
                                    rule.src_opcode);
    }
  }

  const std::vector<uint32_t> expected = {
      (static_cast<uint32_t>(gfx1250::encoding::kSopp) << 16) | gfx1250::kSClauseSopp,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3p) << 16) |
          gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3p) << 16) | gfx1250::kVWmmaI3216x16x64Iu8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3p) << 16) |
          gfx1250::kVSwmmacI3216x16x128Iu8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3pOpHi1) << 16) |
          gfx1250::kVWmmaF3216x16x128Fp8Fp8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3pOpHi1) << 16) |
          gfx1250::kVWmmaF3216x16x128Fp8Bf8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3pOpHi1) << 16) |
          gfx1250::kVWmmaF3216x16x128Bf8Fp8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3pOpHi1) << 16) |
          gfx1250::kVWmmaF3216x16x128Bf8Bf8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3pOpHi1) << 16) |
          gfx1250::kVWmmaF1616x16x128Fp8Fp8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3pOpHi1) << 16) |
          gfx1250::kVWmmaF1616x16x128Fp8Bf8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3pOpHi1) << 16) |
          gfx1250::kVWmmaF1616x16x128Bf8Fp8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3pOpHi1) << 16) |
          gfx1250::kVWmmaF1616x16x128Bf8Bf8Vop3p,
      (static_cast<uint32_t>(gfx1250::encoding::kVop3pOpHi1) << 16) |
          gfx1250::kVWmmaF3232x16x128F4Vop3p,
  };
  EXPECT_EQ(liveness_free_rules, expected);
}

TEST(BinaryTranslatorE2E, Gfx1250LowersF32K128Fp8Bf8WmmaToNeutralRegularScale) {
  struct WmmaCase {
    const char *name;
    uint16_t source_opcode;
    uint8_t matrix_a_fmt;
    uint8_t matrix_b_fmt;
  };
  constexpr std::array cases = {
      WmmaCase{"fp8_fp8", gfx1250::kVWmmaF3216x16x128Fp8Fp8Vop3p, 0, 0},
      WmmaCase{"fp8_bf8", gfx1250::kVWmmaF3216x16x128Fp8Bf8Vop3p, 0, 1},
      WmmaCase{"bf8_fp8", gfx1250::kVWmmaF3216x16x128Bf8Fp8Vop3p, 1, 0},
      WmmaCase{"bf8_bf8", gfx1250::kVWmmaF3216x16x128Bf8Bf8Vop3p, 1, 1},
  };
  constexpr gfx1250::Vop3pBuilderFields fields{
      .vdst = 54,
      .neg_hi = 5,
      .clamp = 1,
      .src0 = 256 + 16,
      .src1 = 256 + 32,
      .src2 = 256 + 48,
      .neg = 3,
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  EXPECT_EQ(rocjitsu::semantic_expand_rules_gfx1250_b0_to_a0().size(), 38u);
  for (const WmmaCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto source_wmma = gfx1250::build_vop3p(test_case.source_opcode, fields);
    auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    const auto decoded =
        decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
    EXPECT_EQ(std::ranges::count_if(decoded,
                                    [](const auto &candidate) {
                                      return candidate->mnemonic() ==
                                             "v_wmma_scale_f32_16x16x128_f8f6f4";
                                    }),
              1);
    EXPECT_EQ(std::ranges::count_if(decoded,
                                    [](const auto &candidate) {
                                      return candidate->mnemonic().find("16x16x64") !=
                                             std::string_view::npos;
                                    }),
              0);

    const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
    const auto prefix = std::find_if(words, words + word_count, [](uint32_t word) {
      return ((word >> 24) & 0xffu) == 0xccu && ((word >> 16) & 0xffu) == 0x35u;
    });
    ASSERT_NE(prefix, words + word_count);
    ASSERT_GE(words + word_count - prefix, 4);
    gfx1250::Vop3pMachineInst scale_prefix{};
    std::memcpy(&scale_prefix, prefix, sizeof(scale_prefix));
    EXPECT_EQ(scale_prefix.src0, 128u);
    EXPECT_EQ(scale_prefix.src1, 128u);
    EXPECT_EQ(scale_prefix.src2, 256u);
    EXPECT_EQ(scale_prefix.neg, 0u);
    EXPECT_EQ(scale_prefix.neg_hi, 0u);
    EXPECT_EQ(scale_prefix.opsel, 0u);
    EXPECT_EQ(scale_prefix.opsel_hi, 0u);

    gfx1250::Vop3pMachineInst replacement{};
    std::memcpy(&replacement, prefix + 2, sizeof(replacement));
    EXPECT_EQ(replacement.op, gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p);
    EXPECT_EQ(replacement.vdst, fields.vdst);
    EXPECT_EQ(replacement.neg_hi, fields.neg_hi);
    EXPECT_EQ(replacement.clamp, fields.clamp);
    EXPECT_EQ(replacement.src0, fields.src0);
    EXPECT_EQ(replacement.src1, fields.src1);
    EXPECT_EQ(replacement.src2, fields.src2);
    EXPECT_EQ(replacement.opsel, test_case.matrix_a_fmt);
    EXPECT_EQ(replacement.opsel_hi, test_case.matrix_b_fmt);
    EXPECT_EQ(replacement.neg, fields.neg);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250F32K128WmmaRejectsNonVgprMatrixInputs) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  for (const gfx1250::Vop3pBuilderFields fields : {
           gfx1250::Vop3pBuilderFields{.vdst = 54, .src0 = 128, .src1 = 256 + 32, .src2 = 256 + 48},
           gfx1250::Vop3pBuilderFields{.vdst = 54, .src0 = 256 + 16, .src1 = 128, .src2 = 256 + 48},
       }) {
    const auto source_wmma = gfx1250::build_vop3p(gfx1250::kVWmmaF3216x16x128Fp8Fp8Vop3p, fields);
    auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(
        rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::ExpandFailed,
                                       "K=128 WMMA matrix operands are not ordinary VGPR ranges"));
  }
}

TEST(BinaryTranslatorE2E, Gfx1250F16K128Fp8Bf8WmmaStillFailsClosedForA0) {
  // A two-K=64 lowering would materialize the first result as packed f16 before
  // feeding it back as C, adding an intermediate rounding that the K=128 form
  // does not perform. Keep every f16 form fail-closed until an exact lowering
  // preserves the single final f16 round.
  constexpr std::array source_opcodes = {
      gfx1250::kVWmmaF1616x16x128Fp8Fp8Vop3p,
      gfx1250::kVWmmaF1616x16x128Fp8Bf8Vop3p,
      gfx1250::kVWmmaF1616x16x128Bf8Fp8Vop3p,
      gfx1250::kVWmmaF1616x16x128Bf8Bf8Vop3p,
  };
  constexpr gfx1250::Vop3pBuilderFields fields{
      .vdst = 54, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const uint16_t source_opcode : source_opcodes) {
    const auto source_wmma = gfx1250::build_vop3p(source_opcode, fields);
    auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
    EXPECT_TRUE(
        rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::ExpandFailed,
                                       "f16 K=128 WMMA A0 lowering is not yet implemented"));
  }
}

TEST(BinaryTranslatorE2E, Gfx1250MasksM0AroundOffFormClusterLoadForA0) {
  // An off-form (NULL-saddr) cluster load is NOT demoted to a global load. Like
  // every cluster-load form, it stays a cluster load and is wrapped so it runs
  // with M0 = 0; the opcode is unchanged.
  constexpr auto cluster = gfx1250::build_vglobal(gfx1250::kClusterLoadB32Vglobal,
                                                  {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  // The load stays a cluster load (no demotion to global_load).
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "cluster_load_b32"; }),
            1);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "global_load_b32"; }),
            0);
  // And it is framed by an M0=0 save/clear/restore. M0 encodes as 125 on gfx1250,
  // inline 0 as 128; SOP1 s_mov is SSRC0[7:0], SDST[22:16].
  constexpr uint16_t kGfx1250M0Operand = 125;
  constexpr uint16_t kInlineZeroOperand = 128;
  bool cleared_m0 = false;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() != "s_mov_b32" || inst->raw_encoding() == nullptr)
      continue;
    const uint32_t word = inst->raw_encoding()[0];
    if (((word >> 16) & 0x7fu) == kGfx1250M0Operand && (word & 0xffu) == kInlineZeroOperand)
      cleared_m0 = true;
  }
  EXPECT_TRUE(cleared_m0) << "off-form cluster load must still set M0 (125) = inline 0 (128)";
}

TEST(BinaryTranslatorE2E, Gfx1250MasksM0AroundSaddrClusterLoadForA0) {
  constexpr auto cluster =
      gfx1250::build_vglobal(gfx1250::kClusterLoadB64Vglobal, {.saddr = 4, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "cluster_load_b64"; }),
            1);
  // The load is wrapped to run with M0 = 0, saving/restoring the original M0
  // through a dead SGPR. Assert operands, not just mnemonics: M0 encodes as 125 on
  // gfx1250 and NULL as 124, so a write to 124 would be a discarded NULL write.
  // SOP1 (s_mov): SSRC0[7:0], SDST[22:16]; inline 0 encodes as 128. The sequence
  // is: s_mov scratch, M0 (save) / s_mov M0, 0 (clear) / cluster_load /
  // s_mov M0, scratch (restore).
  constexpr uint16_t kGfx1250M0Operand = 125;
  constexpr uint16_t kInlineZeroOperand = 128;
  bool saved_m0 = false;    // some s_mov reads M0 as source
  bool cleared_m0 = false;  // some s_mov writes M0 from inline 0
  bool restored_m0 = false; // some s_mov writes M0 from a non-inline (scratch) source
  for (const auto &inst : decoded) {
    if (inst->mnemonic() != "s_mov_b32" || inst->raw_encoding() == nullptr)
      continue;
    const uint32_t word = inst->raw_encoding()[0];
    const uint16_t ssrc0 = word & 0xffu;
    const uint16_t sdst = (word >> 16) & 0x7fu;
    if (ssrc0 == kGfx1250M0Operand)
      saved_m0 = true;
    if (sdst == kGfx1250M0Operand && ssrc0 == kInlineZeroOperand)
      cleared_m0 = true;
    if (sdst == kGfx1250M0Operand && ssrc0 != kInlineZeroOperand)
      restored_m0 = true;
  }
  EXPECT_TRUE(saved_m0) << "no s_mov_b32 reads M0 (125) to save it";
  EXPECT_TRUE(cleared_m0) << "no s_mov_b32 sets M0 (125) = inline 0 (128)";
  EXPECT_TRUE(restored_m0) << "no s_mov_b32 restores M0 (125) from the saved scratch";
}

TEST(BinaryTranslatorE2E, Gfx1250MaterializesDsAddtidAddressForA0) {
  constexpr auto addtid = gfx1250::build_vds(gfx1250::kDsLoadAddtidB32Vds,
                                             {.offset0 = 0x34, .offset1 = 0x12, .vdst = 8});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {addtid[0], addtid[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "ds_load_addtid_b32"; }),
            0);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "ds_load_b32"; }),
            1);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &inst) {
                                    return inst->mnemonic() == "v_mbcnt_lo_u32_b32" ||
                                           inst->mnemonic() == "v_mbcnt_hi_u32_b32";
                                  }),
            2);
  // Pin the A0 ADDTID address formula (M0 + tid*4) & 0xfffff, not just mnemonics:
  // this is the stepping-specific contract, so a miscompiled shift/mask/base would
  // silently target the wrong LDS offset. The emitted sequence is
  //   tid   = v_mbcnt_lo/hi           (checked by count above)
  //   tid*4 = v_lshlrev_b32 by 2
  //   +M0   = v_add_nc_u32 SRC0 = M0  (gfx1250 M0 = 125; NULL is 124 — the inverse
  //                                    of CDNA. Reading NULL would use base 0.)
  //   &mask = v_bfe_u32 offset 0 width 20  (0xfffff = 20-bit LDS byte address)
  // Inline integer N encodes as 128+N in a VOP3 source; VOP3 SRC0 is the low 9
  // bits of word[1], SRC1 the next 9 (bits [17:9]), SRC2 the next 9 (word[1]
  // bits... read from the raw words the builders emit).
  constexpr uint16_t kGfx1250M0Operand = 125;
  constexpr uint16_t kInline = 128; // inline integer 0 encodes as 128.

  auto v_lshl = std::ranges::find_if(
      decoded, [](const auto &inst) { return inst->mnemonic() == "v_lshlrev_b32"; });
  ASSERT_NE(v_lshl, decoded.end());
  ASSERT_NE((*v_lshl)->raw_encoding(), nullptr);
  // SRC0 is the shift amount: inline 2 -> encoding 130 (shift left by 2 == *4).
  EXPECT_EQ((*v_lshl)->raw_encoding()[1] & 0x1ffu, static_cast<uint16_t>(kInline + 2));

  auto v_add = std::ranges::find_if(
      decoded, [](const auto &inst) { return inst->mnemonic() == "v_add_nc_u32"; });
  ASSERT_NE(v_add, decoded.end());
  ASSERT_NE((*v_add)->raw_encoding(), nullptr);
  EXPECT_EQ((*v_add)->raw_encoding()[1] & 0x1ffu, kGfx1250M0Operand);

  auto v_bfe = std::ranges::find_if(
      decoded, [](const auto &inst) { return inst->mnemonic() == "v_bfe_u32"; });
  ASSERT_NE(v_bfe, decoded.end());
  ASSERT_NE((*v_bfe)->raw_encoding(), nullptr);
  // SRC1 (bits [17:9]) is the field offset (inline 0 -> 128); SRC2 (bits [26:18])
  // is the field width (inline 20 -> 148). Together: (value >> 0) & ((1<<20)-1).
  EXPECT_EQ(((*v_bfe)->raw_encoding()[1] >> 9) & 0x1ffu, kInline);
  EXPECT_EQ(((*v_bfe)->raw_encoding()[1] >> 18) & 0x1ffu, static_cast<uint16_t>(kInline + 20));
}

TEST(BinaryTranslatorE2E, Gfx1250FailsClosedOnUnsupportedBarrierSignalIsfirst) {
  // s_barrier_signal_isfirst is classified as needing an expansion but has no rule
  // yet, so the translation must fail closed (ExpandMissing) rather than pass the
  // instruction through. This pins the NOT-YET-SUPPORTED contract for the
  // classifier-only mnemonics.
  constexpr auto barrier = gfx1250::build_sop1(gfx1250::kSBarrierSignalIsfirstSop1, {.ssrc0 = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({barrier[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image) << "a fail-closed translation must leave the object unchanged";
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &d) {
    return d.kind == rocjitsu::DiagnosticKind::ExpandMissing;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->severity, rocjitsu::DiagnosticSeverity::Error);
}

TEST(BinaryTranslatorE2E, Gfx1250WrapsBareF8f6f4WmmaInNeutralScaleForA0) {
  constexpr gfx1250::Vop3pBuilderFields fields{
      .vdst = 32,
      .neg_hi = 4,
      .opsel = 4,
      .src0 = 256 + 64,
      .src1 = 256 + 128,
      .src2 = 256 + 192,
      .opsel_hi = 1,
      .neg = 3,
  };
  constexpr auto wmma = gfx1250::build_vop3p(gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p, fields);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr size_t kPrefixAndMatrixWords = 4;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {wmma[0], wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_GE(decoded.size(), 2u);
  EXPECT_EQ(decoded[0]->mnemonic(), "v_wmma_scale_f32_16x16x128_f8f6f4");
  EXPECT_EQ(decoded[0]->size(), 4 * static_cast<int>(sizeof(uint32_t)));
  EXPECT_EQ(decoded[1]->mnemonic(), "s_endpgm");

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  ASSERT_GE(translated.text_sections()[0]->size() / sizeof(uint32_t), kPrefixAndMatrixWords + 1u);
  gfx1250::Vop3pMachineInst prefix{};
  std::memcpy(&prefix, target_words, sizeof(prefix));
  EXPECT_EQ(prefix.op, 0x35u);
  EXPECT_EQ(prefix.src0, 128u);
  EXPECT_EQ(prefix.src1, 128u);
  EXPECT_EQ(prefix.src2, 256u);
  EXPECT_EQ(prefix.neg, 0u);
  EXPECT_EQ(prefix.neg_hi, 0u);
  EXPECT_EQ(prefix.opsel, 0u);
  EXPECT_EQ(prefix.opsel_hi, 0u);
  EXPECT_EQ(prefix.pad_14, 0u);
  EXPECT_EQ(target_words[2], wmma[0]);
  EXPECT_EQ(target_words[3], wmma[1]);
  EXPECT_EQ(target_words[kPrefixAndMatrixWords], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesSupportedK64LowPrecisionWmmaThroughForA0) {
  struct Case {
    const char *name;
    uint16_t opcode;
  };
  const std::array<Case, 8> cases = {{
      {"v_wmma_f32_16x16x64_fp8_fp8", gfx1250::kVWmmaF3216x16x64Fp8Fp8Vop3p},
      {"v_wmma_f32_16x16x64_fp8_bf8", gfx1250::kVWmmaF3216x16x64Fp8Bf8Vop3p},
      {"v_wmma_f32_16x16x64_bf8_fp8", gfx1250::kVWmmaF3216x16x64Bf8Fp8Vop3p},
      {"v_wmma_f32_16x16x64_bf8_bf8", gfx1250::kVWmmaF3216x16x64Bf8Bf8Vop3p},
      {"v_wmma_f16_16x16x64_fp8_fp8", gfx1250::kVWmmaF1616x16x64Fp8Fp8Vop3p},
      {"v_wmma_f16_16x16x64_fp8_bf8", gfx1250::kVWmmaF1616x16x64Fp8Bf8Vop3p},
      {"v_wmma_f16_16x16x64_bf8_fp8", gfx1250::kVWmmaF1616x16x64Bf8Fp8Vop3p},
      {"v_wmma_f16_16x16x64_bf8_bf8", gfx1250::kVWmmaF1616x16x64Bf8Bf8Vop3p},
  }};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  for (const auto &c : cases) {
    const auto wmma = gfx1250::build_vop3p(
        c.opcode, {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
    auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {wmma[0], wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid()) << c.name;
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << c.name << ": "
                             << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(target_words[0], wmma[0]) << c.name;
    EXPECT_EQ(target_words[1], wmma[1]) << c.name;
    EXPECT_EQ(target_words[2], kGfx1250SEndpgm) << c.name;
  }
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesA0CompatibleLowPrecisionSwmmac) {
  // FP8/BF8 sparse K=128 instructions exist on both steppings. They are not in
  // the gfx1250 B0-additions table and their opcodes fit the A0 VOP3P opcode
  // field, so B0-to-A0 translation must preserve their authoritative bytes.
  constexpr auto swmmac =
      gfx1250::build_vop3p(gfx1250::kVSwmmacF3216x16x128Fp8Fp8Vop3p,
                           {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {swmmac[0], swmmac[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto text = translated.text_sections();
  ASSERT_EQ(text.size(), 1u);
  ASSERT_GE(text[0]->size(), 2 * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(text[0]->data(), swmmac.data(), 2 * sizeof(uint32_t)), 0);
}

TEST(BinaryTranslatorE2E, Gfx1250FailsClosedOnFlatScratchBaseHi64BitSource) {
  // A 64-bit scalar source using the special FLAT_SCRATCH_BASE_HI value (encoding
  // 231) is not supported on this target and must fail closed rather than copy the
  // instruction through unchanged. s_mov_b64 reads a 64-bit source.
  constexpr uint16_t kFlatScratchBaseHi = 231;
  constexpr auto mov64 =
      gfx1250::build_sop1(gfx1250::kSMovB64Sop1, {.ssrc0 = kFlatScratchBaseHi, .sdst = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({mov64[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image) << "a fail-closed translation must leave the object unchanged";
}

TEST(BinaryTranslatorE2E, Gfx1250Allows32BitFlatScratchBaseHiSource) {
  // The same special value in a 32-bit source position is unaffected; s_mov_b32
  // must translate cleanly (identity, same-arch stepping copy).
  constexpr uint16_t kFlatScratchBaseHi = 231;
  constexpr auto mov32 =
      gfx1250::build_sop1(gfx1250::kSMovB32Sop1, {.ssrc0 = kFlatScratchBaseHi, .sdst = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({mov32[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
}

namespace {
// Populate a C-API code object handle owning a copy of @p image, for
// rj_code_translate. rj_code_object_t is non-copyable (RefCounted), so fill a
// caller-provided handle in place.
void init_c_code_object(rj_code_object_t &handle, const std::vector<uint8_t> &image) {
  handle.owned_co = std::make_unique<rocjitsu::AmdGpuCodeObject>(image.data(), image.size());
  handle.co = handle.owned_co.get();
}
} // namespace

TEST(RjCodeTranslateCApi, RejectsSameArchGfx1250) {
  // The C ABI carries no silicon revision, and gfx1250 A0 and B0 share an ELF
  // machine ID, so a same-architecture translation is direction-ambiguous.
  // Revision-aware B0-to-A0 translation uses the C++ BinaryTranslator instead.
  constexpr auto mov = gfx1250::build_sop1(gfx1250::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({mov[0], kGfx1250SEndpgm});
  rj_code_object_t source{};
  init_c_code_object(source, image);
  ASSERT_TRUE(source.co->is_valid());

  rj_code_dbt_options_t options{};
  options.guest_arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.host_arch = ROCJITSU_CODE_ARCH_GFX1250;
  rj_code_object_t *translated = nullptr;
  EXPECT_NE(rj_code_translate(&source, &options, &translated), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(translated, nullptr);
}

TEST(BinaryTranslatorEnforcesGfx1250Revisions, SameArchUnspecifiedRevisionFailsClosed) {
  // Direct BinaryTranslator use with same-arch gfx1250 and no revisions fails
  // closed because no translation profile was selected.
  constexpr auto cluster =
      gfx1250::build_vglobal(gfx1250::kClusterLoadB64Vglobal, {.saddr = 4, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250);
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
}

TEST(BinaryTranslatorEnforcesGfx1250Revisions, TranslatesGfx1250B0ToA0WithRevisions) {
  // With both revisions provided, B0-to-A0 translation selects its profile. The
  // cluster load remains a cluster load framed by an M0=0 sequence.
  constexpr auto cluster =
      gfx1250::build_vglobal(gfx1250::kClusterLoadB64Vglobal, {.saddr = 4, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "cluster_load_b64"; }),
            1);
}

TEST(BinaryTranslatorE2E, Gfx1250GeneratedVgprMsbTransitionsCarryPreviousState) {
  constexpr uint16_t kAllVgprMsbFieldsHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto original_mode =
      gfx1250::build_sopk(gfx1250::kSSetregImm32B32Sopk, {.simm16 = kAllVgprMsbFieldsHwreg});
  constexpr auto addtid =
      gfx1250::build_vds(gfx1250::kDsStoreAddtidB32Vds, {.offset0 = 4, .data0 = 8});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {original_mode[0], 1u << 14, addtid[0], addtid[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_GE(decoded.size(), 5u);
  EXPECT_EQ(decoded[0]->mnemonic(), "s_setreg_imm32_b32");
  // The first generated transition is preceded by an s_nop (MODE-write separation)
  // and an s_wait_xcnt (memory drain) before the s_set_vgpr_msb.
  EXPECT_EQ(decoded[1]->mnemonic(), "s_nop");
  EXPECT_EQ(decoded[2]->mnemonic(), "s_wait_xcnt");
  EXPECT_EQ(decoded[3]->mnemonic(), "s_set_vgpr_msb");
  ASSERT_NE(decoded[3]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[3]->raw_encoding()[0] & 0xffffu, 0x0100u);

  // Every generated s_set_vgpr_msb must be immediately preceded by an s_wait_xcnt.
  std::vector<uint16_t> generated_modes;
  for (size_t index = 1; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
      ASSERT_GT(index, 0u);
      EXPECT_EQ(decoded[index - 1]->mnemonic(), "s_wait_xcnt")
          << "s_set_vgpr_msb at index " << index << " is not guarded by s_wait_xcnt";
      generated_modes.push_back(static_cast<uint16_t>(decoded[index]->raw_encoding()[0] & 0xffffu));
    }
  }
  // Input mode has SRC0 bank 1, SRC1 bank 0. The ds_store_addtid_b32 store-data
  // operand is a SRC1-role VGPR, so the store mode carries src1_bank (0) in the
  // SRC1 field, not src0_bank. With src1_bank 0 the store transition is a no-op
  // and is elided, leaving only the compute transition and the final restore.
  EXPECT_EQ(generated_modes, (std::vector<uint16_t>{0x0100, 0x0001}));
}

TEST(BinaryTranslatorE2E, Gfx1250AddtidStoreModeUsesSrc1BankNotSrc0) {
  // Regression for the ADDTID store-data bank: the emitted ds_store_b32 keeps
  // the original store-data VGPR in data0, a SRC1-role operand, so its high bank
  // must come from src1_bank. Choose differing nonzero banks (SRC0 bank 1, SRC1
  // bank 2) so a src0_bank/src1_bank swap is observable: the store mode must be
  // 0x08 (bank 2 in the SRC1 field), not 0x04 (bank 1).
  constexpr uint16_t kAllVgprMsbFieldsHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto original_mode =
      gfx1250::build_sopk(gfx1250::kSSetregImm32B32Sopk, {.simm16 = kAllVgprMsbFieldsHwreg});
  constexpr uint32_t kModeLiteral = (1u << 14) | (2u << 16); // SRC0 bank 1, SRC1 bank 2.
  constexpr auto addtid =
      gfx1250::build_vds(gfx1250::kDsStoreAddtidB32Vds, {.offset0 = 4, .data0 = 8});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {original_mode[0], kModeLiteral, addtid[0], addtid[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);

  // Assert the full generated s_set_vgpr_msb sequence, not just the store mode's
  // low byte, so this also catches a regression that drops the previous-mode
  // carry (SIMM16[15:8]) or restores the wrong original mode. The set_vgpr_msb
  // byte is {DST[7:6], SRC2[5:4], SRC1[3:2], SRC0[1:0]}; the high byte carries
  // the immediately-preceding mode. The live mode is SRC0 bank 1, SRC1 bank 2 =
  // 0x09. Transitions: compute (new 0x00, prev 0x09) = 0x0900; store (new
  // src1_bank 2 in the SRC1 field 0x08, prev 0x00) = 0x0008; restore (new 0x09,
  // prev 0x08) = 0x0809. A src0_bank/src1_bank swap would make the store mode
  // 0x04 instead of 0x08.
  std::vector<uint16_t> generated_modes;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
      generated_modes.push_back(static_cast<uint16_t>(decoded[index]->raw_encoding()[0] & 0xffffu));
    }
  }
  EXPECT_EQ(generated_modes, (std::vector<uint16_t>{0x0900, 0x0008, 0x0809}))
      << "ADDTID store mode must carry src1_bank (2) in the SRC1 field, and every transition "
         "must record the previous mode in SIMM16[15:8]";
}

TEST(BinaryTranslatorE2E, Gfx1250ConservativelyPadsIu8WmmaForA0) {
  struct Case {
    uint16_t opcode;
    size_t required_slots;
  };
  constexpr std::array cases = {
      Case{gfx1250::kVWmmaI3216x16x64Iu8Vop3p, 9},
      Case{gfx1250::kVSwmmacI3216x16x128Iu8Vop3p, 5},
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  for (const Case &test_case : cases) {
    const auto source_wmma = gfx1250::build_vop3p(
        test_case.opcode, {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
    auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(target_words[0], source_wmma[0]);
    EXPECT_EQ(target_words[1], source_wmma[1]);
    const uint32_t v_nop = gfx1250::build_vop1(gfx1250::kVNopVop1)[0];
    for (size_t slot = 0; slot < test_case.required_slots; ++slot)
      EXPECT_EQ(target_words[2 + slot], v_nop);
    EXPECT_EQ(target_words[2 + test_case.required_slots], kGfx1250SEndpgm);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250TensorLoadAlwaysClearsDynamicMulticastMask) {
  // Multicast is selected dynamically by D# group-1 bits [15:0], not by an
  // instruction bit. A0 must therefore clear the runtime mask for every tensor
  // load and restore the descriptor word after the instruction.
  constexpr auto source_tensor = gfx1250::build_vimage(
      gfx1250::kTensorLoadToLdsVimage,
      {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_tensor[0], source_tensor[1], source_tensor[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_GE(translated.text_sections()[0]->size(), 7 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  // D0 occupies s[8:11] and D1 occupies s[0:7], so the first dead ordinary
  // scalar register is s12.
  constexpr auto save_d1 = gfx1250::build_sop1(gfx1250::kSMovB32Sop1, {.ssrc0 = 0, .sdst = 12});
  constexpr auto clear_mask =
      gfx1250::build_sop2(gfx1250::kSPackHhB32B16Sop2, {.ssrc0 = 128, .ssrc1 = 0, .sdst = 0});
  constexpr auto restore_d1 = gfx1250::build_sop1(gfx1250::kSMovB32Sop1, {.ssrc0 = 12, .sdst = 0});
  EXPECT_EQ(target_words[0], save_d1[0]);
  EXPECT_EQ(target_words[1], clear_mask[0]);
  EXPECT_EQ(target_words[2], source_tensor[0]);
  EXPECT_EQ(target_words[3], source_tensor[1]);
  EXPECT_EQ(target_words[4], source_tensor[2]);
  EXPECT_EQ(target_words[5], restore_d1[0]);
  EXPECT_EQ(target_words[6], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250RegularWmmaScaleEncodesUnusedSrc2AsVgpr) {
  // Real VOP3PX2 encoding from the mxscale offline oracle. Bits [58:50] are
  // zero in the B0 object even though SQ decodes that unused field as a scalar
  // dependency. The A0 output must encode VGPR0 without changing other bits.
  constexpr std::array<uint32_t, 4> source_scale = {0xCC350000u, 0x0202954Eu, 0xCC332042u,
                                                    0x050A01CAu};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_scale[0], source_scale[1], source_scale[2], source_scale[3], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  constexpr uint32_t kScaleSrc2Mask = 0x1ffu << 18;
  EXPECT_EQ(target_words[0], source_scale[0]);
  EXPECT_EQ(target_words[1], (source_scale[1] & ~kScaleSrc2Mask) | (0x100u << 18));
  EXPECT_EQ(target_words[2], source_scale[2]);
  EXPECT_EQ(target_words[3], source_scale[3]);
  EXPECT_EQ(target_words[4], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250PreservesVop3DppExtensionWord) {
  // Real rocSPARSE encoding: v_fma_f32_e64_dpp is a 64-bit VOP3 followed by
  // one DPP control DWORD. The decoder must consume and preserve all 12 bytes,
  // rather than treating the control word as a new instruction.
  constexpr std::array<uint32_t, 3> source_vop3_dpp = {0xD6130000u, 0x020200FAu, 0xFF08E403u};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_vop3_dpp[0], source_vop3_dpp[1], source_vop3_dpp[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], source_vop3_dpp[0]);
  EXPECT_EQ(target_words[1], source_vop3_dpp[1]);
  EXPECT_EQ(target_words[2], source_vop3_dpp[2]);
  EXPECT_EQ(target_words[3], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250SplitsRegularScaleFp4AcrossMForA0) {
  constexpr auto scale =
      gfx1250::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p, {.vdst = 96,
                                                                .neg_hi = 5,
                                                                .clamp = 1,
                                                                .src0 = 256 + 16,
                                                                .src1 = 256 + 32,
                                                                .src2 = 256 + 48,
                                                                .neg = 3});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);
  for (uint16_t half = 0; half < 2; ++half) {
    const size_t offset = pass_offsets[half];
    ASSERT_LE(offset + 4, word_count);
    EXPECT_EQ((words[offset] >> 11) & 0x1u, half);
    EXPECT_EQ(words[offset] & ((1u << 13) | (1u << 14)), 0u);
    EXPECT_EQ((words[offset + 1] >> 18) & 0x1ffu, 0x100u);

    gfx1250::Vop3pMachineInst replacement{};
    std::memcpy(&replacement, words + offset + 2, sizeof(replacement));
    EXPECT_EQ(replacement.op, gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p);
    EXPECT_EQ(replacement.vdst, 96u + half * 8u);
    EXPECT_EQ(replacement.src0, 256u + 16u + half * 8u);
    EXPECT_EQ(replacement.src1, 256u + 32u);
    EXPECT_EQ(replacement.src2, 256u + 48u + half * 8u);
    EXPECT_EQ(replacement.opsel, 4u);
    EXPECT_EQ(replacement.pad_14, 1u);
    EXPECT_EQ(replacement.opsel_hi, 0u);
    EXPECT_EQ(replacement.neg_hi, 5u);
    EXPECT_EQ(replacement.clamp, 1u);
    EXPECT_EQ(replacement.neg, 3u);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4RejectsSharedBOverlap) {
  constexpr auto scale =
      gfx1250::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 96, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "regular-Scale 32x16 lower destination overlaps an input needed by the upper half"));
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4RejectsScaleSourceOverlap) {
  struct ScaleCase {
    const char *name;
    uint16_t scale_src0;
    uint16_t scale_src1;
  };
  constexpr std::array cases = {
      ScaleCase{"scale_src0", 256 + 96, 256 + 66},
      ScaleCase{"scale_src1", 256 + 64, 256 + 96},
  };
  // Scale operands ignore VGPR-MSB and always name bank-zero registers. Keep
  // matrix A/B in nonzero banks while D remains in bank zero so a lowering that
  // incorrectly applies the matrix SRC0/SRC1 bank to the scale operands misses
  // this destructive overlap.
  constexpr auto set_vgpr_msb = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x09});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const ScaleCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto scale = gfx1250::build_vop3p(
        0x35, {.src0 = test_case.scale_src0, .src1 = test_case.scale_src1, .src2 = 0});
    auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {set_vgpr_msb[0], scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::has_error_containing(
        result, rocjitsu::DiagnosticKind::ExpandFailed,
        "regular-Scale 32x16 lower destination overlaps an input needed by the upper half"));
  }
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4AdjustsUpperCAndDestinationBanks) {
  constexpr auto set_vgpr_msb = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x90});
  constexpr auto scale =
      gfx1250::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 252, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 252});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);
  gfx1250::Vop3pMachineInst upper{};
  std::memcpy(&upper, words + pass_offsets[1] + 2, sizeof(upper));
  EXPECT_EQ(upper.vdst, 4u);
  EXPECT_EQ(upper.src0, 256u + 24u);
  EXPECT_EQ(upper.src2, 256u + 4u);

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint8_t> modes;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      modes.push_back(static_cast<uint8_t>(inst->raw_encoding()[0] & 0xffu));
    }
  }
  EXPECT_NE(std::ranges::find(modes, 0xe0u), modes.end());
  EXPECT_EQ(modes.back(), 0x90u);
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4AdjustsUpperABank) {
  constexpr auto scale =
      gfx1250::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 96, .src0 = 256 + 252, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);
  gfx1250::Vop3pMachineInst upper{};
  std::memcpy(&upper, words + pass_offsets[1] + 2, sizeof(upper));
  EXPECT_EQ(upper.src0, 256u + 4u);

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint8_t> modes;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      modes.push_back(static_cast<uint8_t>(inst->raw_encoding()[0] & 0xffu));
    }
  }
  EXPECT_NE(std::ranges::find(modes, 0x01u), modes.end());
  EXPECT_EQ(modes.back(), 0u);
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4RejectsBank3UpperHalfOverflow) {
  constexpr auto set_vgpr_msb = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0xf0});
  constexpr auto scale =
      gfx1250::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 252, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 252});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "regular-Scale 32x16 FP4 split exceeds the VGPR address space"));
}

TEST(BinaryTranslatorE2E, Gfx1250SplitsScale16Fp4AcrossMAndKForA0) {
  constexpr auto scale16 =
      gfx1250::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p, {.vdst = 96,
                                                                .neg_hi = 5,
                                                                .clamp = 1,
                                                                .src0 = 256 + 16,
                                                                .src1 = 256 + 32,
                                                                .src2 = 256 + 48,
                                                                .neg = 7});
  constexpr auto overwrite_masked_a =
      gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 128, .vdst = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale16[0], scale16[1], matrix[0], matrix[1], overwrite_masked_a[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 4u);

  std::array<uint16_t, 4> scale_a_sources{};
  for (uint16_t pass_index = 0; pass_index < 4; ++pass_index) {
    const uint16_t m_half = pass_index / 2u;
    const bool accumulate = (pass_index & 1u) != 0;
    const size_t offset = pass_offsets[pass_index];
    ASSERT_LE(offset + 4, word_count);
    EXPECT_EQ((words[offset] >> 11) & 0x1u, m_half);
    EXPECT_EQ(words[offset] & ((1u << 13) | (1u << 14)), 0u);
    scale_a_sources[pass_index] = static_cast<uint16_t>(words[offset + 1] & 0x1ffu);

    gfx1250::Vop3pMachineInst replacement{};
    std::memcpy(&replacement, words + offset + 2, sizeof(replacement));
    const uint16_t destination = static_cast<uint16_t>(96u + m_half * 8u);
    EXPECT_EQ(replacement.op, gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p);
    EXPECT_EQ(replacement.vdst, destination);
    EXPECT_EQ(replacement.src0, 256u)
        << "the dead v0 range should be selected as unspilled masked-A scratch";
    EXPECT_EQ(replacement.src1, 256u + 32u);
    EXPECT_EQ(replacement.src2, accumulate ? 256u + destination : 256u + 48u + m_half * 8u);
    EXPECT_EQ(replacement.opsel, 4u);
    EXPECT_EQ(replacement.pad_14, 1u);
    EXPECT_EQ(replacement.opsel_hi, 0u);
    EXPECT_EQ(replacement.clamp, 1u);
    EXPECT_EQ(replacement.neg_hi & 0x3u, 1u);
    EXPECT_EQ(replacement.neg & 0x3u, 3u);
    EXPECT_EQ((replacement.neg_hi >> 2) & 0x1u, accumulate ? 0u : 1u);
    EXPECT_EQ((replacement.neg >> 2) & 0x1u, accumulate ? 0u : 1u);
  }
  EXPECT_NE(scale_a_sources[0], scale_a_sources[1]);
  EXPECT_EQ(scale_a_sources[0], scale_a_sources[2]);
  EXPECT_EQ(scale_a_sources[1], scale_a_sources[3]);

  // The following source VALU defines the selected unspilled scratch range.
  // Keep it outside the final FP4 WMMA's one-slot input-read window.
  const uint32_t v_nop = gfx1250::build_vop1(gfx1250::kVNopVop1)[0];
  ASSERT_LT(pass_offsets.back() + 5, word_count);
  EXPECT_EQ(words[pass_offsets.back() + 4], v_nop);
  EXPECT_EQ(words[pass_offsets.back() + 5], overwrite_masked_a[0]);
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16Fp4AdjustsUpperCAndDestinationBanks) {
  constexpr auto set_vgpr_msb = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x90});
  constexpr auto scale16 =
      gfx1250::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 252, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 252});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 4u);
  for (size_t pass_index = 2; pass_index < 4; ++pass_index) {
    gfx1250::Vop3pMachineInst replacement{};
    std::memcpy(&replacement, words + pass_offsets[pass_index] + 2, sizeof(replacement));
    EXPECT_EQ(replacement.vdst, 4u);
    EXPECT_EQ(replacement.src2, 256u + 4u);
  }

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint8_t> modes;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      modes.push_back(static_cast<uint8_t>(inst->raw_encoding()[0] & 0xffu));
    }
  }
  EXPECT_NE(std::ranges::find(modes, 0xe0u), modes.end());
  EXPECT_NE(std::ranges::find(modes, 0xf0u), modes.end());
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16Fp4RejectsBank3UpperHalfOverflow) {
  constexpr auto set_vgpr_msb = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0xf0});
  constexpr auto scale16 =
      gfx1250::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 252, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 252});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(
      rocjitsu::has_error_containing(result, rocjitsu::DiagnosticKind::ExpandFailed,
                                     "Scale16 M=32 C/D split exceeds the VGPR address space"));
}

TEST(BinaryTranslatorE2E, Gfx1250ConvertsScale16F8f6f4ToRegularScaleForA0) {
  constexpr auto scale16 = gfx1250::build_vop3p(
      0x3a, {.opsel = 1, .src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256, .opsel_hi = 1});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p,
                           {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 128});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &inst) {
                                    return inst->mnemonic() == "v_wmma_scale_f32_16x16x128_f8f6f4";
                                  }),
            2);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &inst) {
                                    return inst->mnemonic().starts_with("v_wmma_scale16_f32_");
                                  }),
            0);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &inst) { return inst->mnemonic() == "v_max_u32"; }),
            0);

  // Assert the value-defining operands of the two emitted scale passes, not just
  // that they are scale-form WMMAs. Each pass is a 4-dword VOP3PX2 whose prefix
  // half carries the regular-scale prefix opcode in word[0] bits [23:16] and its
  // scale-A/scale-B sources in word[1] bits [8:0]/[17:9]. The two passes read the
  // low and high scale halves; the second pass additionally accumulates the first
  // pass's result (its C source becomes the intermediate D and the reuse bit is
  // cleared). Read straight from the emitted words so operand miscompiles are
  // caught.
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  constexpr uint32_t kScalePrefixOp = 0x35;
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < target_word_count; ++i) {
    if (((target_words[i] >> 16) & 0xffu) == kScalePrefixOp)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u) << "expected two regular-scale prefix passes";
  for (const size_t pass_offset : pass_offsets) {
    EXPECT_EQ((target_words[pass_offset] >> 11) & 1u, 1u)
        << "M=16 must preserve the source A-scale lane selector";
    EXPECT_EQ((target_words[pass_offset + 1] >> 27) & 1u, 1u)
        << "Matrix B selection comes from SCL_OPSEL_HI[0]";
  }
  // The source scale operands were src0=v64, src1=v66 (VGPR encodings 256+64,
  // 256+66). The two passes read the low and high scale halves derived from those.
  const uint16_t first_scale_a = static_cast<uint16_t>(target_words[pass_offsets[0] + 1] & 0x1ffu);
  const uint16_t second_scale_a = static_cast<uint16_t>(target_words[pass_offsets[1] + 1] & 0x1ffu);
  EXPECT_GE(first_scale_a, 256u) << "first pass scale-A must be a VGPR";
  EXPECT_GE(second_scale_a, 256u) << "second pass scale-A must be a VGPR";
  EXPECT_NE(first_scale_a, second_scale_a) << "the two passes must read different scale halves";

  // FP8/FP8 is not the all-FP4 fast case. The gfx1250 hazard table requires
  // three VALU co-execution slots before generated VALU may overwrite matrix A.
  const uint32_t v_nop = gfx1250::build_vop1(gfx1250::kVNopVop1)[0];
  ASSERT_LT(pass_offsets[0] + 7, target_word_count);
  EXPECT_EQ(target_words[pass_offsets[0] + 4], v_nop);
  EXPECT_EQ(target_words[pass_offsets[0] + 5], v_nop);
  EXPECT_EQ(target_words[pass_offsets[0] + 6], v_nop);
  EXPECT_NE(target_words[pass_offsets[0] + 7], v_nop);
  ASSERT_LT(pass_offsets[1] + 7, target_word_count);
  EXPECT_EQ(target_words[pass_offsets[1] + 4], v_nop);
  EXPECT_EQ(target_words[pass_offsets[1] + 5], v_nop);
  EXPECT_EQ(target_words[pass_offsets[1] + 6], v_nop);
  EXPECT_NE(target_words[pass_offsets[1] + 7], v_nop);

  // The FP8 matrix-A format takes the lane-mask K-selection path: each pass
  // isolates its own K=16 half by cndmask'ing matrix-A lanes against a lane
  // mask built from the low/high 16-bit halves. Pin those two mask literals so a
  // regression that drops the split (or masks the wrong half) is caught, not
  // just the WMMA opcode count.
  const auto contains_word = [&](uint32_t value) {
    return std::any_of(target_words, target_words + target_word_count,
                       [&](uint32_t w) { return w == value; });
  };
  EXPECT_TRUE(contains_word(0x0000ffffu)) << "missing low-half Scale16 lane mask";
  EXPECT_TRUE(contains_word(0xffff0000u)) << "missing high-half Scale16 lane mask";
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16UsesSpillBackedScratchWhenVgprsAreFull) {
  using namespace rocr::llvm::amdhsa;

  constexpr auto scale16 =
      gfx1250::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto descriptor = rocjitsu::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  rocjitsu::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(), descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
                                        options);
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);

  struct ScratchAccess {
    size_t word_offset;
    uint8_t vgpr;
    uint32_t byte_offset;
  };
  std::vector<ScratchAccess> stores;
  std::vector<ScratchAccess> loads;
  size_t wait_store_count = 0;
  size_t wait_load_count = 0;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "s_wait_storecnt") {
      ++wait_store_count;
      continue;
    }
    if (inst->mnemonic() == "s_wait_loadcnt") {
      ++wait_load_count;
      continue;
    }
    if (inst->mnemonic() != "scratch_store_b32" && inst->mnemonic() != "scratch_load_b32")
      continue;
    ASSERT_NE(inst->raw_encoding(), nullptr);
    gfx1250::VscratchMachineInst scratch{};
    std::memcpy(&scratch, inst->raw_encoding(), sizeof(scratch));
    const bool is_load = inst->mnemonic() == "scratch_load_b32";
    (is_load ? loads : stores)
        .push_back(
            ScratchAccess{.word_offset = inst->src_loc() / sizeof(uint32_t),
                          .vgpr = static_cast<uint8_t>(is_load ? scratch.vdst : scratch.vsrc),
                          .byte_offset = scratch.ioffset});
  }

  // FP4 Scale16 needs eight masked-A lanes and five scalar temporaries. With
  // every free register forced above the low-bank policy, the lowering borrows
  // v[0:12], saves every lane, and restores the same range and offsets.
  ASSERT_EQ(stores.size(), 13u);
  ASSERT_EQ(loads.size(), 13u);
  for (uint8_t i = 0; i < 13; ++i) {
    EXPECT_EQ(stores[i].vgpr, i);
    EXPECT_EQ(stores[i].byte_offset, static_cast<uint32_t>(i) * sizeof(uint32_t));
    EXPECT_EQ(loads[i].vgpr, i);
    EXPECT_EQ(loads[i].byte_offset, static_cast<uint32_t>(i) * sizeof(uint32_t));
  }
  EXPECT_EQ(wait_store_count, 1u);
  EXPECT_EQ(wait_load_count, 1u);

  const uint32_t wait_store = gfx1250::build_sopp(gfx1250::kSWaitStorecntSopp, {.simm16 = 0})[0];
  const uint32_t wait_load = gfx1250::build_sopp(gfx1250::kSWaitLoadcntSopp, {.simm16 = 0})[0];
  const uint32_t v_nop = gfx1250::build_vop1(gfx1250::kVNopVop1)[0];
  ASSERT_LT(stores.back().word_offset + 3, target_word_count);
  EXPECT_EQ(target_words[stores.back().word_offset + 3], wait_store);
  ASSERT_GT(loads.front().word_offset, 0u);
  EXPECT_EQ(target_words[loads.front().word_offset - 1], v_nop)
      << "the final WMMA must clear its co-execution hazard before restoration";
  ASSERT_LT(loads.back().word_offset + 3, target_word_count);
  EXPECT_EQ(target_words[loads.back().word_offset + 3], wait_load);
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16SpillOffsetFailureLeavesElfUnchanged) {
  using namespace rocr::llvm::amdhsa;

  constexpr auto scale16 =
      gfx1250::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto descriptor = rocjitsu::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  descriptor.private_segment_fixed_size = 0x7ffffffdu;
  rocjitsu::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(), descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
                                        options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "Scale16 WMMA could not allocate its contiguous scratch range"));
}

TEST(BinaryTranslatorE2E, Gfx1250FailsClosedOnStandalone32x16Fp4ForA0) {
  // The standalone 32x16 FP4 A0 split is deferred. Each M=16 half needs its own
  // neutral regular-scale prefix; emitting bare matrix bodies from a semantic
  // expansion would bypass recursive legalization. Until that direct lowering
  // exists, the source instruction must fail closed.
  constexpr auto matrix =
      gfx1250::build_vop3p(gfx1250::kVWmmaF3232x16x128F4Vop3p,
                           {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 40, .src2 = 256 + 64});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
}

TEST(BinaryTranslatorE2E, Gfx1250RecoversAddNcU64PcBuilder) {
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint32_t kTargetOffset = 20;
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  const std::vector<uint32_t> words = {
      rocjitsu::build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_GFX1250),
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_GFX1250, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand),
      kTargetOffset - sizeof(uint32_t),
      rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_GFX1250),
      kGfx1250SNop,
      kGfx1250SEndpgm,
  };
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
}

TEST(BinaryTranslatorE2E, Gfx1250IgnoresUndecodablePaddingAfterTerminator) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kGfx1250SEndpgm, 0, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
}

TEST(BinaryTranslatorE2E, Gfx1250RejectsReachableZeroInKernelBody) {
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kGfx1250SNop, 0, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "reachable kernel code falls through into undecodable .text bytes"));
}

TEST(BinaryTranslatorE2E, Gfx1250AcceptsClangUnreachableKernelStub) {
  // clang emits this one-instruction body for rocPRIM target-specialized
  // trampolines whose source ends in __builtin_unreachable(). The following
  // zero is alignment padding, not a second instruction.
  constexpr uint32_t kSetReplayMode = 0xb9800641u;
  constexpr uint32_t kLiteralOne = 1u;
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kSetReplayMode, kLiteralOne, 0}, 2);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(decoded.size(), 2u)
      << "the synthesized endpgm must be the only instruction after the stub body";
  ASSERT_NE(decoded[0]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[0]->mnemonic(), "s_setreg_imm32_b32");
  EXPECT_EQ(decoded[0]->raw_encoding()[0], kSetReplayMode);
  EXPECT_EQ(decoded[0]->raw_encoding()[1], kLiteralOne);
  EXPECT_EQ(decoded[1]->mnemonic(), "s_endpgm");

  const auto kernel_symbol = rocjitsu::find_elf_symbol_for_test(result.elf_bytes, "kernel");
  ASSERT_TRUE(kernel_symbol.has_value());
  EXPECT_EQ(kernel_symbol->st_size, 3 * sizeof(uint32_t))
      << "the translated function extent must include its synthesized terminator";
}

TEST(BinaryTranslatorE2E, Gfx1250AcceptsClangUnreachableKernelStubWithPrefetchPrologue) {
  // Newer clang adds the exact gfx1250 prefetch prologue in front of the same
  // unreachable rocPRIM target-specialization body.
  constexpr uint32_t kGlobalPrefetchB8[] = {0xee174000u, 0x00040000u, 0u};
  constexpr uint32_t kVNop = 0x7e000000u;
  constexpr uint32_t kSetReplayMode[] = {0xb9800641u, 1u};
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kGlobalPrefetchB8[0], kGlobalPrefetchB8[1], kGlobalPrefetchB8[2], kVNop, kSetReplayMode[0],
       kSetReplayMode[1], 0});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(decoded.size(), 4u)
      << "the synthesized endpgm must be the only instruction after the stub body";
  EXPECT_EQ(decoded[0]->mnemonic(), "global_prefetch_b8");
  EXPECT_EQ(decoded[1]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[2]->mnemonic(), "s_setreg_imm32_b32");
  ASSERT_NE(decoded[2]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[2]->raw_encoding()[0], kSetReplayMode[0]);
  EXPECT_EQ(decoded[2]->raw_encoding()[1], kSetReplayMode[1]);
  EXPECT_EQ(decoded[3]->mnemonic(), "s_endpgm");
}

TEST(BinaryTranslatorE2E, Gfx1250MaterializesSectionFinalClangUnreachableKernelStub) {
  constexpr uint32_t kSetReplayMode = 0xb9800641u;
  constexpr uint32_t kLiteralOne = 1u;
  auto image =
      rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text({kSetReplayMode, kLiteralOne});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto first = translator.translate(source);
  ASSERT_TRUE(first.ok()) << (first.diagnostics.empty() ? "" : first.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject first_output(first.elf_bytes.data(), first.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*first_output.text_sections()[0], ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(decoded.size(), 2u);
  EXPECT_EQ(decoded[0]->mnemonic(), "s_setreg_imm32_b32");
  EXPECT_EQ(decoded[1]->mnemonic(), "s_endpgm");

  auto second = translator.translate(first_output);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, first.elf_bytes);
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

TEST(KernelDescriptorTranslator, Gfx1250UsesWave32SixteenVgprGranularity) {
  using namespace rocr::llvm::amdhsa;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  const auto *text = rocjitsu::find_section(layout, ".text");
  ASSERT_NE(rodata, nullptr);
  ASSERT_NE(text, nullptr);

  auto descriptor = rocjitsu::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  // Leave ENABLE_WAVEFRONT_SIZE32 clear to verify that the Wave32-only gfx1250
  // architecture does not misinterpret a legacy producer's missing bit.
  rocjitsu::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(), descriptor);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                                  ROCJITSU_CODE_ARCH_GFX1250);
  const auto translations = translator.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});

  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].guest_wavefront_size, 32);
  EXPECT_EQ(translations[0].target_wave_size, 32);
  EXPECT_EQ(translations[0].guest_vgpr_count, 1024u);
  EXPECT_EQ(translations[0].target_vgpr_count, 1024u);
  EXPECT_EQ(translations[0].target_vgpr_granulated, 63u);
}

TEST(KernelDescriptorTranslator, Gfx1250AcceptsLdsAboveTcpPartitionSize) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  const auto *text = rocjitsu::find_section(layout, ".text");
  ASSERT_NE(rodata, nullptr);
  ASSERT_NE(text, nullptr);

  // gfx1250's descriptor limit is 320 KiB per workgroup. A TCP may be
  // configured with a smaller LDS/vector-cache partition, but that partition
  // size is not the architectural descriptor validation limit.
  constexpr uint32_t kStaticLdsBytes = 176128u;
  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      kStaticLdsBytes);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                                  ROCJITSU_CODE_ARCH_GFX1250);
  const auto translations = translator.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});

  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].target_lds_size, kStaticLdsBytes);
}

TEST(KernelDescriptorTranslator, CdnaToCdnaRejectsOversizedLdsWithoutVirtualization) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);

  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      105600u);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations = translator.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_FALSE(translations[0].supported);
  EXPECT_EQ(translations[0].target_lds_size, 105600u);
  EXPECT_FALSE(translations[0].needs_lds_overflow_buf);
  EXPECT_EQ(translations[0].lds_overflow_size, 0u);

  const bool reported_lds_limit =
      std::ranges::any_of(translations[0].diagnostics, [](const auto &diagnostic) {
        return diagnostic.message.find("target LDS size exceeds host per-workgroup limit") !=
               std::string::npos;
      });
  EXPECT_TRUE(reported_lds_limit);
}

TEST(KernelDescriptorTranslator, CdnaToCdnaVirtualizesOversizedStaticLdsDescriptor) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);

  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      105600u);
  auto source_descriptor = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      image, rodata->sectionOffset());
  uint32_t source_rsrc2 = source_descriptor.compute_pgm_rsrc2;
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE, 22);
  rocjitsu::write_value_for_test<uint32_t>(
      image, rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, compute_pgm_rsrc2),
      source_rsrc2);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].target_lds_size, 0u);
  EXPECT_TRUE(translations[0].needs_lds_overflow_buf);
  EXPECT_EQ(translations[0].lds_overflow_size, 105600u);
  EXPECT_EQ(translations[0].kernarg_size, 16u);
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 16u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 24u);
  EXPECT_EQ(translations[0].target_kernarg_size, 48u);

  rocjitsu::AmdGpuCodeObject mutated(image.data(), image.size());
  ASSERT_TRUE(mutated.is_valid());
  rocjitsu::CodeObjectPatcher patcher(mutated);
  auto patch_plan = translations[0];
  patch_plan.target_entry_text_offset = patch_plan.entry_text_offset;
  patch_plan.target_body_entry_text_offset = patch_plan.entry_text_offset;
  ASSERT_TRUE(patcher.apply_kernel_descriptor_translation(patch_plan, ROCJITSU_CODE_ARCH_CDNA3));

  const auto patched_image = patcher.emit();
  const auto patched_kd = rocjitsu::read_elf_struct_for_test<rocjitsu::TestKernelDescriptor>(
      patched_image, rodata->sectionOffset());
  EXPECT_EQ(patched_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(patched_kd.kernarg_size, 48u);
  EXPECT_EQ(AMDHSA_BITS_GET(patched_kd.compute_pgm_rsrc2,
                            rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE),
            0u);
}

TEST(KernelDescriptorTranslator, VirtualLdsPreservesKernargPreloadRangeWhenSizeIsZero) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, /*kernarg_size=*/0);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_OFFSET, 2);
  AMDHSA_BITS_SET(source_kd->kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 6);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].kernarg_size, 32u);
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 32u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 40u);
  EXPECT_EQ(translations[0].target_kernarg_size, 64u);
}

TEST(KernelDescriptorTranslator, VirtualLdsKeepsOddKernargPreloadCopyExtentExact) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, /*kernarg_size=*/0);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_OFFSET, 0);
  AMDHSA_BITS_SET(source_kd->kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 11);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].kernarg_size, 44u);
  // Wrapper fields remain aligned independently of the exact source-copy size.
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 48u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 56u);
  EXPECT_EQ(translations[0].target_kernarg_size, 80u);
}

TEST(KernelDescriptorTranslator, VirtualLdsAcceptsZeroKernargSizeWithWrapper) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, /*kernarg_size=*/0);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);

  rocjitsu::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() + offsetof(rocjitsu::TestKernelDescriptor, group_segment_fixed_size),
      105600u);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_TRUE(translations[0].needs_lds_overflow_buf);
  EXPECT_EQ(translations[0].kernarg_size, 0u);
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 0u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 8u);
  EXPECT_EQ(translations[0].target_kernarg_size, 32u);
}

TEST(KernelDescriptorTranslator, VirtualLdsAddsKernargSegmentPointerWhenMissing) {
  using namespace rocr::llvm::amdhsa;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 105600u;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1);
  AMDHSA_BITS_SET(source_kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                  1);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_TRUE(translations[0].needs_lds_overflow_buf);
  EXPECT_EQ(translations[0].kernarg_size, 0u);
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 0u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 8u);
  EXPECT_EQ(translations[0].target_kernarg_size, 32u);
  EXPECT_FALSE(translations[0].source_has_kernarg_segment_ptr);
  EXPECT_TRUE(translations[0].has_kernarg_segment_ptr);
  EXPECT_EQ(translations[0].kernarg_segment_ptr_sgpr, 2u);
  EXPECT_EQ(translations[0].source_user_sgpr_count, 2u);
  EXPECT_EQ(translations[0].target_user_sgpr_count, 4u);
  EXPECT_EQ(translations[0].workgroup_id_sgpr_x, 2);
  EXPECT_EQ(translations[0].lds_overflow_workgroup_id_sgpr_x, 4);
  EXPECT_EQ(translations[0].user_sgpr_repair_start, 2u);
  // gfx950 initializes architected FLAT_SCRATCH rather than an additional
  // ordinary SGPR, so only the enabled workgroup-id SGPR needs repair.
  EXPECT_EQ(translations[0].user_sgpr_repair_count, 1u);
  EXPECT_TRUE(translations[0].has_dispatch_ptr);
  EXPECT_EQ(translations[0].dispatch_ptr_sgpr, 0u);
}

TEST(KernelDescriptorTranslator, VirtualLdsRejectsMissingKernargSegmentPointerWithFullUserSgprs) {
  using namespace rocr::llvm::amdhsa;

  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::TestKernelDescriptor));

  auto *source_kd =
      reinterpret_cast<rocjitsu::TestKernelDescriptor *>(image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 105600u;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_FALSE(translations[0].supported);
  EXPECT_TRUE(translations[0].needs_lds_overflow_buf);
  const bool reported_user_sgpr_limit =
      std::ranges::any_of(translations[0].diagnostics, [](const auto &diagnostic) {
        return diagnostic.message.find("USER_SGPR_COUNT") != std::string::npos &&
               diagnostic.message.find("16 SGPR") != std::string::npos;
      });
  EXPECT_TRUE(reported_user_sgpr_limit);
}

TEST(KernelDescriptorTranslator, IgnoresNonAllocExecutableSectionsForEntryRange) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_descriptor_after_text();
  const auto ehdr = rocjitsu::read_elf_struct_for_test<rocjitsu::Elf64_Ehdr>(image, 0);
  auto shdrs =
      rocjitsu::read_elf_array_for_test<rocjitsu::Elf64_Shdr>(image, ehdr.e_shoff, ehdr.e_shnum);

  constexpr uint64_t fake_exec_vaddr = 0x9000;
  shdrs[5].sh_flags = rocjitsu::SHF_EXECINSTR;
  shdrs[5].sh_addr = fake_exec_vaddr;
  shdrs[5].sh_size = sizeof(uint32_t);
  for (size_t i = 0; i < shdrs.size(); ++i)
    rocjitsu::write_elf_struct_for_test(image, ehdr.e_shoff + i * sizeof(rocjitsu::Elf64_Shdr),
                                        shdrs[i]);

  rocjitsu::write_kernel_descriptor_entry_offset(image.data() + shdrs[2].sh_offset,
                                                 static_cast<int64_t>(fake_exec_vaddr) -
                                                     static_cast<int64_t>(shdrs[2].sh_addr));

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations = translator.translate_image(
      image, shdrs[1].sh_offset, shdrs[1].sh_size, rocjitsu::KernelDescriptorTranslationOptions{});
  EXPECT_TRUE(translations.empty())
      << "non-loadable executable sections must not extend valid kernel entry range";
}
