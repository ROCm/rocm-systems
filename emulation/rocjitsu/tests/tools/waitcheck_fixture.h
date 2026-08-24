// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcheck_fixture.h
/// @brief Synthetic gfx12 code objects for waitcheck tool/preload smoke tests.

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::waitcheck_test {

inline uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

inline uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

template <typename T> void append_inst(std::vector<uint32_t> &words, const T &inst) {
  static_assert(sizeof(T) % sizeof(uint32_t) == 0);
  std::array<uint32_t, sizeof(T) / sizeof(uint32_t)> encoded =
      std::bit_cast<std::array<uint32_t, sizeof(T) / sizeof(uint32_t)>>(inst);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

[[nodiscard]] inline rdna4::SoppMachineInst s_wait_loadcnt(uint32_t count) {
  rdna4::SoppMachineInst inst{};
  inst.encoding = 0x17F;
  inst.op = 64;
  inst.simm16 = count;
  return inst;
}

[[nodiscard]] inline rdna4::Vop1MachineInst v_mov_b32(uint32_t vdst, uint32_t src_vgpr) {
  rdna4::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 1;
  inst.vdst = vdst;
  inst.src0 = 256 + src_vgpr;
  return inst;
}

[[nodiscard]] inline uint32_t v_add_f32_e32_word(uint32_t vdst, uint32_t src0, uint32_t vsrc1) {
  auto inst = std::bit_cast<rdna4::Vop2MachineInst>(0x06000000U);
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.vsrc1 = vsrc1;
  return std::bit_cast<uint32_t>(inst);
}

[[nodiscard]] inline uint32_t s_mov_b32_word(uint32_t sdst, uint32_t ssrc0) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE800000U);
  inst.sdst = sdst;
  inst.ssrc0 = ssrc0;
  return std::bit_cast<uint32_t>(inst);
}

[[nodiscard]] inline uint32_t s_setpc_b64_word(uint32_t ssrc0) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE804800U);
  inst.ssrc0 = ssrc0;
  return std::bit_cast<uint32_t>(inst);
}

[[nodiscard]] inline rdna4::VglobalMachineInst global_load_b32(uint32_t vdst) {
  rdna4::VglobalMachineInst inst{};
  inst.encoding = 0xEE;
  inst.op = 20;
  inst.vdst = vdst;
  inst.vaddr = 8;
  inst.saddr = 0;
  return inst;
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx_multi_kernel_code_object(
    const std::vector<std::pair<std::string, std::vector<uint32_t>>> &kernels, uint32_t mach,
    bool wave32 = false) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t kernel_descriptor_size = 64;
  std::vector<uint32_t> text_words;
  std::vector<uint64_t> entry_offsets;
  for (const auto &[name, words] : kernels) {
    (void)name;
    entry_offsets.push_back(text_words.size() * sizeof(uint32_t));
    text_words.insert(text_words.end(), words.begin(), words.end());
  }
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  std::vector<uint32_t> kd_symbol_names;
  std::vector<uint32_t> function_symbol_names;
  for (const auto &[name, words] : kernels) {
    (void)words;
    kd_symbol_names.push_back(add_elf_name(strtab, name + ".kd"));
    function_symbol_names.push_back(add_elf_name(strtab, name));
  }

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t rodata_size = kernels.size() * kernel_descriptor_size;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  const size_t sym_count = 2 * kernels.size() + 1;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = 1;
  ehdr.e_ident[EI_VERSION] = 1;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_flags = mach;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shoff = shoff;
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
  constexpr size_t kernel_code_entry_byte_offset_offset = 16;
  constexpr size_t kernel_code_properties_offset = 56;
  constexpr uint16_t wavefront_size32_bit = 1u << 10u;
  for (size_t i = 0; i < kernels.size(); ++i) {
    std::array<uint8_t, kernel_descriptor_size> kernel_descriptor{};
    const uint64_t descriptor_vaddr = rodata_vaddr + i * kernel_descriptor_size;
    const int64_t entry_offset = static_cast<int64_t>(text_vaddr + entry_offsets[i]) -
                                 static_cast<int64_t>(descriptor_vaddr);
    std::memcpy(kernel_descriptor.data() + kernel_code_entry_byte_offset_offset, &entry_offset,
                sizeof(entry_offset));
    const uint16_t kernel_code_properties = wave32 ? wavefront_size32_bit : 0;
    std::memcpy(kernel_descriptor.data() + kernel_code_properties_offset, &kernel_code_properties,
                sizeof(kernel_code_properties));
    std::memcpy(image.data() + rodata_offset + i * kernel_descriptor_size, kernel_descriptor.data(),
                kernel_descriptor.size());
  }
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::vector<Elf64_Sym> syms(sym_count);
  for (size_t i = 0; i < kernels.size(); ++i) {
    Elf64_Sym &descriptor_symbol = syms[2 * i + 1];
    descriptor_symbol.st_name = kd_symbol_names[i];
    descriptor_symbol.st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
    descriptor_symbol.st_shndx = 2;
    descriptor_symbol.st_value = rodata_vaddr + i * kernel_descriptor_size;
    descriptor_symbol.st_size = kernel_descriptor_size;

    Elf64_Sym &function_symbol = syms[2 * i + 2];
    function_symbol.st_name = function_symbol_names[i];
    function_symbol.st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
    function_symbol.st_shndx = 1;
    function_symbol.st_value = text_vaddr + entry_offsets[i];
    function_symbol.st_size = kernels[i].second.size() * sizeof(uint32_t);
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
  shdrs[2].sh_addralign = kernel_descriptor_size;

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

[[nodiscard]] inline std::vector<uint8_t> make_gfx_function_only_code_object(
    const std::vector<std::pair<std::string, std::vector<uint32_t>>> &functions, uint32_t mach,
    bool wave32 = false) {
  std::vector<uint8_t> image = make_gfx_multi_kernel_code_object(functions, mach, wave32);
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *symbols = reinterpret_cast<Elf64_Sym *>(image.data() + shdrs[3].sh_offset);
  for (size_t i = 0; i < functions.size(); ++i)
    symbols[2 * i + 1] = {};
  return image;
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx_code_object(const std::vector<uint32_t> &text_words, uint32_t mach, bool wave32 = false) {
  return make_gfx_multi_kernel_code_object({{"waitcheck", text_words}}, mach, wave32);
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx_symbol_less_code_object(const std::vector<uint32_t> &text_words, uint32_t mach,
                                 bool wave32 = false) {
  std::vector<uint8_t> image = make_gfx_code_object(text_words, mach, wave32);
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  for (size_t section_index = 0; section_index < ehdr->e_shnum; ++section_index) {
    const Elf64_Shdr &section = shdrs[section_index];
    if (section.sh_type != SHT_SYMTAB)
      continue;
    std::memset(image.data() + section.sh_offset, 0, section.sh_size);
  }
  return image;
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx1200_code_object(const std::vector<uint32_t> &text_words) {
  return make_gfx_code_object(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1200);
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx1201_code_object(const std::vector<uint32_t> &text_words) {
  return make_gfx_code_object(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1201);
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1201_multi_kernel_code_object(
    const std::vector<std::pair<std::string, std::vector<uint32_t>>> &kernels) {
  return make_gfx_multi_kernel_code_object(kernels, EF_AMDGPU_MACH_AMDGCN_GFX1201);
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1201_function_only_code_object(
    const std::vector<std::pair<std::string, std::vector<uint32_t>>> &functions) {
  return make_gfx_function_only_code_object(functions, EF_AMDGPU_MACH_AMDGCN_GFX1201);
}

[[nodiscard]] inline std::vector<uint8_t> make_relocatable_code_object(std::vector<uint8_t> image) {
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  ehdr.e_type = ET_REL;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));
  return image;
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx942_code_object(const std::vector<uint32_t> &text_words) {
  return make_gfx_code_object(text_words, EF_AMDGPU_MACH_AMDGCN_GFX942);
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx1100_code_object(const std::vector<uint32_t> &text_words) {
  return make_gfx_code_object(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1100);
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx1150_code_object(const std::vector<uint32_t> &text_words) {
  return make_gfx_code_object(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1150);
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx1151_code_object(const std::vector<uint32_t> &text_words) {
  return make_gfx_code_object(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1151);
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx1250_code_object(const std::vector<uint32_t> &text_words) {
  return make_gfx_code_object(text_words, EF_AMDGPU_MACH_AMDGCN_GFX1250);
}

[[nodiscard]] inline std::vector<uint8_t>
make_gfx950_incomplete_direct_to_lds_code_object(bool include_definite_wait_hazard = false) {
  std::vector<uint32_t> text_words;
  if (include_definite_wait_hazard) {
    text_words.insert(text_words.end(), {
                                            0xE0501000u,
                                            0x80000008u, // buffer_load_dword v0, v8, s[0:3] offen
                                            0x7E020300u, // v_mov_b32 v1, v0 without a wait
                                        });
  }
  text_words.insert(text_words.end(), {
                                          0xE05D1000u,
                                          0x80100008u, // buffer_load_dwordx4 ... lds; M0 unknown
                                          0x7E180280u, // v_mov_b32 v12, 0
                                          0xD9FE0040u,
                                          0x1200000Cu, // ds_read_b128 v[18:21], v12 offset:64
                                          0xBF810000u, // s_endpgm
                                      });
  return make_gfx_code_object(text_words, EF_AMDGPU_MACH_AMDGCN_GFX950);
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx942_missing_wait_code_object() {
  return make_gfx942_code_object({0xE0501000u, 0x80000008u, 0x7E020300u});
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1100_missing_wait_code_object() {
  return make_gfx1100_code_object({0xDC520000u, 0x00000008u, 0x7E020300u});
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1150_missing_wait_code_object() {
  return make_gfx1150_code_object({0xDC520000u, 0x00000008u, 0x7E020300u});
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1151_missing_wait_code_object() {
  return make_gfx1151_code_object({0xDC520000u, 0x00000008u, 0x7E020300u});
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1200_missing_wait_code_object() {
  std::vector<uint32_t> text_words;
  append_inst(text_words, global_load_b32(0));
  append_inst(text_words, v_mov_b32(1, 0));
  return make_gfx1200_code_object(text_words);
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1200_correct_wait_code_object() {
  std::vector<uint32_t> text_words;
  append_inst(text_words, global_load_b32(0));
  append_inst(text_words, s_wait_loadcnt(0));
  append_inst(text_words, v_mov_b32(1, 0));
  return make_gfx1200_code_object(text_words);
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1201_correct_wait_code_object() {
  std::vector<uint32_t> text_words;
  append_inst(text_words, global_load_b32(0));
  append_inst(text_words, s_wait_loadcnt(0));
  append_inst(text_words, v_mov_b32(1, 0));
  return make_gfx1201_code_object(text_words);
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1201_invalid_instruction_code_object() {
  return make_gfx1201_code_object({0x00800000U});
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1200_invalid_instruction_code_object() {
  return make_gfx1200_code_object({0x00800000U});
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1200_padded_code_object() {
  std::vector<uint32_t> text_words;
  append_inst(text_words, s_wait_loadcnt(0));
  text_words.push_back(0xBFB00000U); // s_endpgm
  text_words.push_back(0x00000000U);
  text_words.push_back(0x00000000U);
  return make_gfx1200_code_object(text_words);
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1200_function_only_padded_code_object() {
  std::vector<uint32_t> first;
  append_inst(first, s_wait_loadcnt(0));
  first.push_back(0xBE80481EU); // s_setpc_b64 s[30:31]
  first.push_back(0x00000000U);
  first.push_back(0x00000000U);

  std::vector<uint32_t> second;
  append_inst(second, global_load_b32(0));
  append_inst(second, v_mov_b32(1, 0));
  second.push_back(0xBE80481EU); // s_setpc_b64 s[30:31]

  auto image = make_gfx_function_only_code_object({{"first", first}, {"second", second}},
                                                  EF_AMDGPU_MACH_AMDGCN_GFX1200);
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *syms = reinterpret_cast<Elf64_Sym *>(image.data() + shdrs[3].sh_offset);
  syms[2].st_size = 2 * sizeof(uint32_t);
  syms[4].st_size = second.size() * sizeof(uint32_t);
  return image;
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1250_unterminated_stub_code_object() {
  auto image = make_gfx1250_code_object({0xB9800641U, 0x00000001U, 0x00000000U});
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *syms = reinterpret_cast<Elf64_Sym *>(image.data() + shdrs[3].sh_offset);
  syms[2].st_size = 2 * sizeof(uint32_t);
  return image;
}

[[nodiscard]] inline std::vector<uint8_t> make_gfx1250_vmax_u64_code_object() {
  return make_gfx1250_code_object({0xD7190002U, 0x0201020CU});
}

} // namespace rocjitsu::waitcheck_test
