// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dbt_test_utils.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <array>
#include <cstring>
#include <vector>

namespace rocjitsu::dbt_test {

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

} // namespace

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

std::vector<uint8_t> make_minimal_amdgpu_elf_with_descriptor_after_text() {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t text_size = 8;
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

  const std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
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

std::vector<uint8_t> make_minimal_amdgpu_elf_with_two_kernel_descriptors() {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = 2 * sizeof(KD);

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

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const std::array<uint32_t, 2> text_words = {kCdna4SEndpgm, kCdna4SEndpgm};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  std::array<KD, 2> descriptors{};
  descriptors[0].kernel_code_entry_byte_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  descriptors[1].kernel_code_entry_byte_offset =
      static_cast<int64_t>(text_vaddr + sizeof(uint32_t)) -
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
  text_words[0] = pack_sopp(12, 0); // CDNA4 s_waitcnt 0 expands on RDNA4.
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

} // namespace rocjitsu::dbt_test
