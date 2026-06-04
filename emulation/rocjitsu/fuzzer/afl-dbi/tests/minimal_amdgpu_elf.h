// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"

#include "hsa/AMDHSAKernelDescriptor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rocjitsu::fuzzer::afl::test {

inline constexpr uint64_t kMinimalAmdGpuElfTextOffset = 0x100;
inline constexpr uint64_t kMinimalAmdGpuElfTextVaddr = 0x1100;
inline constexpr uint64_t kMinimalAmdGpuElfRodataOffset = 0x110;
inline constexpr uint64_t kMinimalAmdGpuElfRodataVaddr = 0x2000;
inline constexpr uint64_t kMinimalAmdGpuElfKernelDescriptorOffset = kMinimalAmdGpuElfRodataOffset;
inline constexpr uint64_t kMinimalAmdGpuElfKernelEntryOffset = kMinimalAmdGpuElfTextOffset;

namespace detail {

inline uint32_t add_elf_name(std::vector<uint8_t> &table, const char *name) {
  const uint32_t offset = static_cast<uint32_t>(table.size());
  const auto *bytes = reinterpret_cast<const uint8_t *>(name);
  table.insert(table.end(), bytes, bytes + std::strlen(name) + 1);
  return offset;
}

inline uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace detail

inline std::vector<uint8_t> make_minimal_amdgpu_elf() {
  using KernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;

  constexpr std::array<uint32_t, 4> text_words = {
      0xbf800000u, // s_nop 0
      0xbf810000u, // s_endpgm
      0xbf800000u, // s_nop 0
      0xbf800000u, // s_nop 0
  };
  constexpr uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t rodata_size = sizeof(KernelDescriptor);
  constexpr uint64_t load_align = 0x1000;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = detail::add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = detail::add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = detail::add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = detail::add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = detail::add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = detail::add_elf_name(strtab, "kernel.kd");

  const uint64_t strtab_offset = kMinimalAmdGpuElfRodataOffset + rodata_size;
  const uint64_t symtab_offset = detail::align_up(strtab_offset + strtab.size(), 8);
  constexpr size_t symbol_count = 2;
  const uint64_t shstrtab_offset = symtab_offset + symbol_count * sizeof(Elf64_Sym);
  const uint64_t shoff = detail::align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t program_header_count = 2;

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
  ehdr.e_phnum = program_header_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, program_header_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = kMinimalAmdGpuElfTextOffset;
  phdrs[0].p_vaddr = kMinimalAmdGpuElfTextVaddr;
  phdrs[0].p_paddr = kMinimalAmdGpuElfTextVaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = kMinimalAmdGpuElfRodataOffset;
  phdrs[1].p_vaddr = kMinimalAmdGpuElfRodataVaddr;
  phdrs[1].p_paddr = kMinimalAmdGpuElfRodataVaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  std::memcpy(image.data() + kMinimalAmdGpuElfTextOffset, text_words.data(), text_size);

  KernelDescriptor kd{};
  kd.kernel_code_entry_byte_offset = static_cast<int64_t>(kMinimalAmdGpuElfTextVaddr) -
                                     static_cast<int64_t>(kMinimalAmdGpuElfRodataVaddr);
  std::memcpy(image.data() + kMinimalAmdGpuElfRodataOffset, &kd, sizeof(kd));

  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<Elf64_Sym, symbol_count> syms{};
  syms[1].st_name = kd_symbol_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = kMinimalAmdGpuElfRodataVaddr;
  syms[1].st_size = sizeof(KernelDescriptor);
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = kMinimalAmdGpuElfTextVaddr;
  shdrs[1].sh_offset = kMinimalAmdGpuElfTextOffset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = kMinimalAmdGpuElfRodataVaddr;
  shdrs[2].sh_offset = kMinimalAmdGpuElfRodataOffset;
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

} // namespace rocjitsu::fuzzer::afl::test
