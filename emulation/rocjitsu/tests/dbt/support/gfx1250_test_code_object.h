// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::test_support {
namespace detail {

inline uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

inline uint32_t add_name(std::vector<uint8_t> &table, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(table.size());
  table.insert(table.end(), name.begin(), name.end());
  table.push_back('\0');
  return offset;
}

} // namespace detail

/// Build a small, valid gfx1250 code object with one kernel descriptor.
inline std::vector<uint8_t> make_gfx1250_code_object(std::span<const uint32_t> text_words) {
  constexpr uint64_t kTextOffset = 0x100;
  constexpr uint64_t kTextVaddr = 0x1100;
  constexpr uint64_t kLoadAlignment = 0x1000;
  constexpr uint64_t kKernelDescriptorSize = 64;
  constexpr uint64_t kKernelEntryOffsetField = 16;
  const uint64_t text_size = text_words.size_bytes();

  std::vector<uint8_t> section_names{'\0'};
  const uint32_t text_name = detail::add_name(section_names, ".text");
  const uint32_t rodata_name = detail::add_name(section_names, ".rodata");
  const uint32_t symtab_name = detail::add_name(section_names, ".symtab");
  const uint32_t strtab_name = detail::add_name(section_names, ".strtab");
  const uint32_t shstrtab_name = detail::add_name(section_names, ".shstrtab");

  std::vector<uint8_t> symbol_names{'\0'};
  const uint32_t descriptor_symbol_name = detail::add_name(symbol_names, "kernel.kd");

  const uint64_t rodata_offset = detail::align_up(kTextOffset + text_size, 8);
  const uint64_t rodata_vaddr = detail::align_up(kTextVaddr + text_size, 8) + kLoadAlignment;
  const uint64_t strtab_offset = rodata_offset + kKernelDescriptorSize;
  const uint64_t symtab_offset = detail::align_up(strtab_offset + symbol_names.size(), 8);
  constexpr size_t kSymbolCount = 2;
  const uint64_t shstrtab_offset = symtab_offset + kSymbolCount * sizeof(Elf64_Sym);
  const uint64_t section_headers_offset =
      detail::align_up(shstrtab_offset + section_names.size(), 8);
  constexpr uint16_t kSectionCount = 6;

  std::vector<uint8_t> image(section_headers_offset + kSectionCount * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_DYN;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_phoff = sizeof(Elf64_Ehdr);
  header.e_shoff = section_headers_offset;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_phentsize = sizeof(Elf64_Phdr);
  header.e_phnum = 2;
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = kSectionCount;
  header.e_shstrndx = 5;
  std::memcpy(image.data(), &header, sizeof(header));

  std::array<Elf64_Phdr, 2> program_headers{};
  program_headers[0].p_type = PT_LOAD;
  program_headers[0].p_flags = 0x5;
  program_headers[0].p_offset = kTextOffset;
  program_headers[0].p_vaddr = kTextVaddr;
  program_headers[0].p_paddr = kTextVaddr;
  program_headers[0].p_filesz = text_size;
  program_headers[0].p_memsz = text_size;
  program_headers[0].p_align = kLoadAlignment;
  program_headers[1].p_type = PT_LOAD;
  program_headers[1].p_flags = 0x4;
  program_headers[1].p_offset = rodata_offset;
  program_headers[1].p_vaddr = rodata_vaddr;
  program_headers[1].p_paddr = rodata_vaddr;
  program_headers[1].p_filesz = kKernelDescriptorSize;
  program_headers[1].p_memsz = kKernelDescriptorSize;
  program_headers[1].p_align = kLoadAlignment;
  std::memcpy(image.data() + header.e_phoff, program_headers.data(), sizeof(program_headers));

  std::memcpy(image.data() + kTextOffset, text_words.data(), text_size);
  const int64_t entry_offset =
      static_cast<int64_t>(kTextVaddr) - static_cast<int64_t>(rodata_vaddr);
  std::memcpy(image.data() + rodata_offset + kKernelEntryOffsetField, &entry_offset,
              sizeof(entry_offset));
  std::memcpy(image.data() + strtab_offset, symbol_names.data(), symbol_names.size());

  std::array<Elf64_Sym, kSymbolCount> symbols{};
  symbols[1].st_name = descriptor_symbol_name;
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[1].st_shndx = 2;
  symbols[1].st_value = rodata_vaddr;
  symbols[1].st_size = kKernelDescriptorSize;
  std::memcpy(image.data() + symtab_offset, symbols.data(), sizeof(symbols));
  std::memcpy(image.data() + shstrtab_offset, section_names.data(), section_names.size());

  std::array<Elf64_Shdr, kSectionCount> section_headers{};
  section_headers[1].sh_name = text_name;
  section_headers[1].sh_type = SHT_PROGBITS;
  section_headers[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  section_headers[1].sh_addr = kTextVaddr;
  section_headers[1].sh_offset = kTextOffset;
  section_headers[1].sh_size = text_size;
  section_headers[1].sh_addralign = sizeof(uint32_t);
  section_headers[2].sh_name = rodata_name;
  section_headers[2].sh_type = SHT_PROGBITS;
  section_headers[2].sh_flags = SHF_ALLOC;
  section_headers[2].sh_addr = rodata_vaddr;
  section_headers[2].sh_offset = rodata_offset;
  section_headers[2].sh_size = kKernelDescriptorSize;
  section_headers[2].sh_addralign = 64;
  section_headers[3].sh_name = symtab_name;
  section_headers[3].sh_type = SHT_SYMTAB;
  section_headers[3].sh_offset = symtab_offset;
  section_headers[3].sh_size = sizeof(symbols);
  section_headers[3].sh_link = 4;
  section_headers[3].sh_info = 1;
  section_headers[3].sh_addralign = 8;
  section_headers[3].sh_entsize = sizeof(Elf64_Sym);
  section_headers[4].sh_name = strtab_name;
  section_headers[4].sh_type = SHT_STRTAB;
  section_headers[4].sh_offset = strtab_offset;
  section_headers[4].sh_size = symbol_names.size();
  section_headers[4].sh_addralign = 1;
  section_headers[5].sh_name = shstrtab_name;
  section_headers[5].sh_type = SHT_STRTAB;
  section_headers[5].sh_offset = shstrtab_offset;
  section_headers[5].sh_size = section_names.size();
  section_headers[5].sh_addralign = 1;
  std::memcpy(image.data() + section_headers_offset, section_headers.data(),
              sizeof(section_headers));
  return image;
}

} // namespace rocjitsu::test_support
