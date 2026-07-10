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
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
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

std::vector<uint8_t>
make_minimal_amdgpu_elf_with_descriptor_after_text(const std::vector<uint32_t> &text_words) {
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

  const auto descriptor = make_kernel_descriptor_bytes(static_cast<int64_t>(text_vaddr) -
                                                       static_cast<int64_t>(rodata_vaddr));
  std::memcpy(image.data() + rodata_offset, descriptor.data(), descriptor.size());
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, sym_count> syms{};
  syms[1].st_name = kd_symbol_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = kKernelDescriptorSize;
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
  text_words[0] = pack_sopp(12, 0); // CDNA4 s_waitcnt 0 expands on RDNA4.
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

TEST(BinaryTranslator, LocalCaveIgnoresUnreachableTextTail) {
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

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], build_s_branch(1, ROCJITSU_CODE_ARCH_RDNA4))
      << "The expansion cave should be placed immediately after the reachable entry block, not "
         "after the large unreachable .text tail";
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

constexpr uint16_t kAnyExpectedField = 0xffff;

enum class ExpectedCdna3Kind {
  Vop3,
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
      expect_mfma(narrow_op, 0, 1, 256, 260, src2),
      expect_mfma(narrow_op, 0, 1, 258, 262, 256),
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

std::vector<ExpectedCdna3Inst> expected_cdna3_ds_read_b64_tr_b16_sequence() {
  std::vector<ExpectedCdna3Inst> expected = {
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
      expect_ds(63),    expect_ds(63), expect_sopp(12), expect_vop3(493),
  };

  auto first_pack = expected_cdna3_raw_b16_pack_sequence();
  expected.insert(expected.end(), first_pack.begin(), first_pack.end());
  expected.insert(expected.end(), {
                                      expect_vop3(308),
                                      expect_ds(63),
                                      expect_ds(63),
                                      expect_sopp(12),
                                      expect_vop3(493),
                                      expect_vop3(308),
                                      expect_ds(63),
                                      expect_ds(63),
                                      expect_sopp(12),
                                      expect_vop3(493),
                                  });
  auto second_pack = expected_cdna3_raw_b16_pack_sequence();
  expected.insert(expected.end(), second_pack.begin(), second_pack.end());
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

std::array<uint32_t, 2> make_cdna4_permlane32_swap_b32_words(uint16_t encoding_id) {
  rocjitsu::cdna4::Vop1MachineInst inst{};
  // The legalization table's VOP1 encoding ids (0xfc..0xff) are the generated
  // primary-decode ids, not the raw 7-bit VOP1 selector.  Primary decode looks
  // at bits 31:23, so VOP1 contributes its fixed selector in bits 31:25 and
  // VDST[7:6] in bits 24:23.  Keep the real VOP1 selector at 0x3f and vary
  // VDST's high bits to exercise each generated semantic rule.
  inst.encoding = 0x3f;
  inst.op = 90;
  inst.vdst = static_cast<uint8_t>((encoding_id - 0xFCu) << 6);
  inst.src0 = 256 + 1;
  return encode_two_word_inst(inst);
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

std::array<uint32_t, 2> make_cdna4_ds_read_b64_tr_b16_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = 227;
  inst.addr = 2;
  inst.vdst = 0;
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
      {"VPermlane32SwapB32E32", 0xFC, 90, make_cdna4_permlane32_swap_b32_words(0xFC),
       expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane32SwapB32E32Hi1", 0xFD, 90, make_cdna4_permlane32_swap_b32_words(0xFD),
       expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane32SwapB32E32Hi2", 0xFE, 90, make_cdna4_permlane32_swap_b32_words(0xFE),
       expected_cdna3_permlane32_swap_sequence()},
      {"VPermlane32SwapB32E32Hi3", 0xFF, 90, make_cdna4_permlane32_swap_b32_words(0xFF),
       expected_cdna3_permlane32_swap_sequence()},
      {"MfmaF32_16x16x32F16", 0x1A7, 84, make_cdna4_mfma_words(84, 0, 256, 260),
       expected_cdna3_mfma_sequence(77)},
      {"MfmaF32_32x32x16F16", 0x1A7, 85, make_cdna4_mfma_words(85, 0, 256, 260),
       expected_cdna3_mfma_sequence(76)},
      {"MfmaF32_16x16x32F16AccumVgpr", 0x1A7, 84, make_cdna4_mfma_words(84, 0, 256, 260, 272),
       expected_cdna3_mfma_sequence(77, 272)},
      {"MfmaF32_32x32x16F16AccumVgpr", 0x1A7, 85, make_cdna4_mfma_words(85, 0, 256, 260, 272),
       expected_cdna3_mfma_sequence(76, 272)},
      {"DsReadB64TrB16", 0x1B3, 227, make_cdna4_ds_read_b64_tr_b16_words(),
       expected_cdna3_ds_read_b64_tr_b16_sequence()},
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

void expect_cdna3_local_cave_matches(const rocjitsu::Section &text, uint64_t cave_offset,
                                     const std::vector<ExpectedCdna3Inst> &expected) {
  ASSERT_EQ(text.size() % sizeof(uint32_t), 0u);
  ASSERT_GT(text.size(), cave_offset);
  ASSERT_EQ(cave_offset % sizeof(uint32_t), 0u);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);

  const auto *words = reinterpret_cast<const uint32_t *>(text.data() + cave_offset);
  const size_t word_count = (text.size() - cave_offset) / sizeof(uint32_t);
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

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
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
    return rocjitsu::pack_sop1(0x47, sdst, 0);
  default:
    return rocjitsu::pack_sop1(0x1c, sdst, 0);
  }
}

uint32_t build_s_setpc_b64(uint16_t ssrc0, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rocjitsu::pack_sop1(0x48, 0, ssrc0);
  default:
    return rocjitsu::pack_sop1(0x1d, 0, ssrc0);
  }
}

uint32_t build_s_swappc_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rocjitsu::pack_sop1(0x49, sdst, ssrc0);
  default:
    return rocjitsu::pack_sop1(0x1e, sdst, ssrc0);
  }
}

uint32_t build_s_call_b64(uint16_t sdst, int16_t simm16) {
  constexpr uint32_t kSopkEncodingPrefix = 0xb;
  constexpr uint32_t kSCallB64Opcode = 0x15;
  return (kSopkEncodingPrefix << 28) | (kSCallB64Opcode << 23) | ((sdst & 0x7fu) << 16) |
         static_cast<uint16_t>(simm16);
}

uint32_t build_s_trap(uint16_t simm16) {
  // CDNA1-4 encode S_TRAP at SOPP opcode 0x12. This helper is intentionally
  // local to the CDNA4->CDNA3 tests below; RDNA3+ uses a different SOPP opcode.
  constexpr uint32_t kCdnaSoppTrapOpcode = 0x12;
  return rocjitsu::pack_sopp(kCdnaSoppTrapOpcode, simm16);
}

uint32_t build_s_add_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return rocjitsu::pack_sop2(0, sdst, ssrc0, ssrc1);
}

uint32_t build_s_addc_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return rocjitsu::pack_sop2(4, sdst, ssrc0, ssrc1);
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

TEST(BinaryTranslatorE2E, DuplicatesSharedReachableBlocksPerKernel) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
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
  EXPECT_EQ(rocjitsu::find_section(translated, ".rj_translations"), nullptr);
  expect_cdna3_local_cave_matches(*translated.text_sections()[0], source_text->size(),
                                  test_case.expected);
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
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  constexpr uint32_t kCdna4SCbranchScc1ToSourceTarget = rocjitsu::pack_sopp(5, 4);
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
      rocjitsu::pack_sopp(5, 1),
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
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),      // 0x04.
      kOriginalGetpcDelta,                                     // 0x08.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c.
      rocjitsu::pack_sopp(5, 1),                               // 0x10 -> second consumer.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),    // 0x14 first consumer.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),    // 0x18 carried consumer.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),      // 0x1c unreachable gap.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),      // 0x20 shared target.
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
  EXPECT_EQ(target_words[4], rocjitsu::pack_sopp(5, 1));
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
