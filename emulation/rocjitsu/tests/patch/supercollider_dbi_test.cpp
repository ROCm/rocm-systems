// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/supercollider_dbi.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

inline constexpr uint8_t kElfSymbolBindLocal = 0;
inline constexpr uint16_t kRdna4VccLo = 106;
inline constexpr uint32_t kRdna4Wave64AllVgprsGranulated = 63;

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

std::vector<uint8_t>
make_rdna4_lds_code_object(std::span<const uint32_t> text_words,
                           std::string_view kernel_name = "lds_probe",
                           uint32_t vgpr_granulated = kRdna4Wave64AllVgprsGranulated) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t rodata_vaddr = 0x2100;
  constexpr uint64_t kernel_descriptor_size = 64;

  const uint64_t text_size = text_words.size() * sizeof(uint32_t);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kernel_symbol_name = add_elf_name(strtab, kernel_name);
  const std::string kd_name = std::string(kernel_name) + ".kd";
  const uint32_t kd_symbol_name = add_elf_name(strtab, kd_name);

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t strtab_offset = rodata_offset + kernel_descriptor_size;
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 3;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;

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
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1201;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  static_assert(sizeof(KD) == kernel_descriptor_size);
  KD kernel_descriptor{};
  const int64_t entry_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  kernel_descriptor.kernel_code_entry_byte_offset = entry_offset;
  AMDHSA_BITS_SET(kernel_descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, vgpr_granulated);
  std::memcpy(image.data() + rodata_offset, &kernel_descriptor, sizeof(kernel_descriptor));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Sym, sym_count> symbols{};
  symbols[1].st_name = kernel_symbol_name;
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  symbols[1].st_shndx = 1;
  symbols[1].st_value = text_vaddr;
  symbols[1].st_size = text_size;
  symbols[2].st_name = kd_symbol_name;
  symbols[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[2].st_shndx = 2;
  symbols[2].st_value = rodata_vaddr;
  symbols[2].st_size = kernel_descriptor_size;
  std::memcpy(image.data() + symtab_offset, symbols.data(), symbols.size() * sizeof(Elf64_Sym));

  std::array<Elf64_Shdr, section_count> sections{};
  sections[1].sh_name = text_name;
  sections[1].sh_type = SHT_PROGBITS;
  sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[1].sh_addr = text_vaddr;
  sections[1].sh_offset = text_offset;
  sections[1].sh_size = text_size;
  sections[1].sh_addralign = sizeof(uint32_t);
  sections[2].sh_name = rodata_name;
  sections[2].sh_type = SHT_PROGBITS;
  sections[2].sh_flags = SHF_ALLOC;
  sections[2].sh_addr = rodata_vaddr;
  sections[2].sh_offset = rodata_offset;
  sections[2].sh_size = kernel_descriptor_size;
  sections[2].sh_addralign = 64;
  sections[3].sh_name = symtab_name;
  sections[3].sh_type = SHT_SYMTAB;
  sections[3].sh_offset = symtab_offset;
  sections[3].sh_size = sym_count * sizeof(Elf64_Sym);
  sections[3].sh_link = 4;
  sections[3].sh_info = 1;
  sections[3].sh_addralign = 8;
  sections[3].sh_entsize = sizeof(Elf64_Sym);
  sections[4].sh_name = strtab_name;
  sections[4].sh_type = SHT_STRTAB;
  sections[4].sh_offset = strtab_offset;
  sections[4].sh_size = strtab.size();
  sections[4].sh_addralign = 1;
  sections[5].sh_name = shstrtab_name;
  sections[5].sh_type = SHT_STRTAB;
  sections[5].sh_offset = shstrtab_offset;
  sections[5].sh_size = shstrtab.size();
  sections[5].sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));

  return image;
}

std::vector<uint8_t> make_rdna4_code_object_with_local_function(
    std::span<const uint32_t> kernel_words, std::span<const uint32_t> function_words,
    std::span<const uint32_t> tail_words = {},
    uint32_t vgpr_granulated = kRdna4Wave64AllVgprsGranulated) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t rodata_vaddr = 0x2100;
  constexpr uint64_t kernel_descriptor_size = 64;

  const uint64_t kernel_size = kernel_words.size() * sizeof(uint32_t);
  const uint64_t function_size = function_words.size() * sizeof(uint32_t);
  const uint64_t tail_size = tail_words.size() * sizeof(uint32_t);
  const uint64_t text_size = kernel_size + function_size + tail_size;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kernel_symbol_name = add_elf_name(strtab, "lds_probe");
  const uint32_t function_symbol_name = add_elf_name(strtab, "lds_helper");
  const uint32_t kd_symbol_name = add_elf_name(strtab, "lds_probe.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t strtab_offset = rodata_offset + kernel_descriptor_size;
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 4;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;

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
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1201;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, kernel_words.data(), kernel_size);
  std::memcpy(image.data() + text_offset + kernel_size, function_words.data(), function_size);
  if (tail_size != 0)
    std::memcpy(image.data() + text_offset + kernel_size + function_size, tail_words.data(),
                tail_size);

  static_assert(sizeof(KD) == kernel_descriptor_size);
  KD kernel_descriptor{};
  const int64_t entry_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  kernel_descriptor.kernel_code_entry_byte_offset = entry_offset;
  AMDHSA_BITS_SET(kernel_descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, vgpr_granulated);
  std::memcpy(image.data() + rodata_offset, &kernel_descriptor, sizeof(kernel_descriptor));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Sym, sym_count> symbols{};
  symbols[1].st_name = kernel_symbol_name;
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
  symbols[1].st_shndx = 1;
  symbols[1].st_value = text_vaddr;
  symbols[1].st_size = kernel_size;
  symbols[2].st_name = function_symbol_name;
  symbols[2].st_info = elf_symbol_info(kElfSymbolBindLocal, kElfSymbolTypeFunc);
  symbols[2].st_shndx = 1;
  symbols[2].st_value = text_vaddr + kernel_size;
  symbols[2].st_size = function_size;
  symbols[3].st_name = kd_symbol_name;
  symbols[3].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[3].st_shndx = 2;
  symbols[3].st_value = rodata_vaddr;
  symbols[3].st_size = kernel_descriptor_size;
  std::memcpy(image.data() + symtab_offset, symbols.data(), symbols.size() * sizeof(Elf64_Sym));

  std::array<Elf64_Shdr, section_count> sections{};
  sections[1].sh_name = text_name;
  sections[1].sh_type = SHT_PROGBITS;
  sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[1].sh_addr = text_vaddr;
  sections[1].sh_offset = text_offset;
  sections[1].sh_size = text_size;
  sections[1].sh_addralign = sizeof(uint32_t);
  sections[2].sh_name = rodata_name;
  sections[2].sh_type = SHT_PROGBITS;
  sections[2].sh_flags = SHF_ALLOC;
  sections[2].sh_addr = rodata_vaddr;
  sections[2].sh_offset = rodata_offset;
  sections[2].sh_size = kernel_descriptor_size;
  sections[2].sh_addralign = 64;
  sections[3].sh_name = symtab_name;
  sections[3].sh_type = SHT_SYMTAB;
  sections[3].sh_offset = symtab_offset;
  sections[3].sh_size = sym_count * sizeof(Elf64_Sym);
  sections[3].sh_link = 4;
  sections[3].sh_info = 2;
  sections[3].sh_addralign = 8;
  sections[3].sh_entsize = sizeof(Elf64_Sym);
  sections[4].sh_name = strtab_name;
  sections[4].sh_type = SHT_STRTAB;
  sections[4].sh_offset = strtab_offset;
  sections[4].sh_size = strtab.size();
  sections[4].sh_addralign = 1;
  sections[5].sh_name = shstrtab_name;
  sections[5].sh_type = SHT_STRTAB;
  sections[5].sh_offset = shstrtab_offset;
  sections[5].sh_size = shstrtab.size();
  sections[5].sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));

  return image;
}

std::vector<uint8_t> make_rdna4_supported_lds_code_object() {
  const std::array<uint32_t, 6> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFC60000u,              // s_wait_dscnt
      0xBFB00000u,              // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_unsupported_lds_code_object() {
  const std::array<uint32_t, 11> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xD8000000u, 0x00000000u, // ds_add_u32
      0xBF940000u,              // s_barrier_wait
      0xBFC60000u,              // s_wait_dscnt
      0xF4042000u, 0x00000000u, // s_dcache_inv
      0xBFB00000u,              // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

std::vector<uint8_t> make_rdna4_flat_memory_code_object() {
  const std::array<uint32_t, 13> text_words = {
      0xEC050000u, 0x00000000u, 0x00000000u, // flat_load_b32
      0xEC068000u, 0x00000000u, 0x00000000u, // flat_store_b32
      0xEE050000u, 0x00000000u, 0x00000000u, // global_load_b32
      0xED050000u, 0x00000000u, 0x00000000u, // scratch_load_b32
      0xBFB00000u,                           // s_endpgm
  };
  return make_rdna4_lds_code_object(text_words);
}

TEST(SuperColliderDbi, DisabledModeDoesNotParseCodeObject) {
  const std::vector<uint8_t> bytes = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  SuperColliderDbiOptions options;
  options.enabled = false;
  options.delay_nops = 32;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.input_size, bytes.size());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.warnings.empty());
  EXPECT_TRUE(result.target_name.empty());
  EXPECT_TRUE(result.kernels.empty());
}

TEST(SuperColliderDbi, EnabledModeRejectsInvalidCodeObject) {
  const std::vector<uint8_t> bytes = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  SuperColliderDbiOptions options;
  options.enabled = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.input_size, bytes.size());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
  EXPECT_TRUE(result.target_name.empty());
  EXPECT_TRUE(result.kernels.empty());
}

TEST(SuperColliderDbi, StubRejectsEmptyCodeObject) {
  const std::vector<uint8_t> bytes;
  SuperColliderDbiOptions options;
  options.enabled = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.input_size, 0u);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
}

TEST(SuperColliderDbi, CountsFlatGlobalAndScratchMemoryInstructions) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_memory_code_object();
  SuperColliderDbiOptions options;
  options.enabled = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);

  const SuperColliderDbiKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 52u);
  EXPECT_EQ(kernel.stats.instruction_count, 5u);
  EXPECT_EQ(kernel.stats.lds_read_count, 0u);
  EXPECT_EQ(kernel.stats.lds_write_count, 0u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 0u);
  EXPECT_EQ(kernel.stats.ds_other_count, 0u);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_write_count, 1u);
  EXPECT_EQ(kernel.stats.flat_atomic_count, 0u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_private_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_private_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_global_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 2u);
  EXPECT_EQ(kernel.stats.global_memory_count, 1u);
  EXPECT_EQ(kernel.stats.scratch_memory_count, 1u);
  EXPECT_EQ(kernel.preflight_action, SuperColliderDbiPreflightAction::Skip);
  ASSERT_GE(kernel.preflight_reasons.size(), 4u);
  EXPECT_EQ(kernel.preflight_reasons[0], "no supported non-atomic LDS reads or writes");
  EXPECT_EQ(kernel.preflight_reasons[1], "flat/generic memory instructions observed: 2");
  EXPECT_EQ(kernel.preflight_reasons[2], "global memory instructions observed: 1");
  EXPECT_EQ(kernel.preflight_reasons[3], "scratch memory instructions observed: 1");
  EXPECT_TRUE(kernel.lds_sites.empty());
  ASSERT_EQ(kernel.flat_sites.size(), 2u);
  EXPECT_EQ(kernel.flat_sites[0].kind, SuperColliderDbiLdsAccessKind::Read);
  EXPECT_EQ(kernel.flat_sites[0].mnemonic, "flat_load_b32");
  EXPECT_EQ(kernel.flat_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.flat_sites[0].file_offset, 0x100u);
  EXPECT_EQ(kernel.flat_sites[0].size, 12u);
  EXPECT_EQ(kernel.flat_sites[0].width_bits, 32u);
  EXPECT_EQ(kernel.flat_sites[0].address_space_hint, SuperColliderDbiFlatAddressSpaceHint::Unknown);
  ASSERT_TRUE(kernel.flat_sites[0].dst_vgpr);
  ASSERT_TRUE(kernel.flat_sites[0].addr_vgpr);
  EXPECT_FALSE(kernel.flat_sites[0].data_vgpr);
  EXPECT_EQ(*kernel.flat_sites[0].dst_vgpr, 0u);
  EXPECT_LT(*kernel.flat_sites[0].addr_vgpr, 256u);
  ASSERT_TRUE(kernel.flat_sites[0].raw_saddr);
  ASSERT_TRUE(kernel.flat_sites[0].raw_vaddr);
  ASSERT_TRUE(kernel.flat_sites[0].raw_vdst);
  ASSERT_TRUE(kernel.flat_sites[0].raw_ioffset);
  EXPECT_EQ(*kernel.flat_sites[0].raw_saddr, 0u);
  EXPECT_EQ(*kernel.flat_sites[0].raw_vdst, 0u);
  EXPECT_EQ(*kernel.flat_sites[0].raw_ioffset, 0);
  EXPECT_EQ(kernel.flat_sites[1].kind, SuperColliderDbiLdsAccessKind::Write);
  EXPECT_EQ(kernel.flat_sites[1].mnemonic, "flat_store_b32");
  EXPECT_EQ(kernel.flat_sites[1].text_offset, 12u);
  EXPECT_EQ(kernel.flat_sites[1].file_offset, 0x10cu);
  EXPECT_EQ(kernel.flat_sites[1].size, 12u);
  EXPECT_EQ(kernel.flat_sites[1].width_bits, 32u);
  EXPECT_EQ(kernel.flat_sites[1].address_space_hint, SuperColliderDbiFlatAddressSpaceHint::Unknown);
  EXPECT_FALSE(kernel.flat_sites[1].dst_vgpr);
  ASSERT_TRUE(kernel.flat_sites[1].addr_vgpr);
  ASSERT_TRUE(kernel.flat_sites[1].data_vgpr);
  EXPECT_LT(*kernel.flat_sites[1].addr_vgpr, 256u);
  EXPECT_EQ(*kernel.flat_sites[1].data_vgpr, 0u);
  ASSERT_TRUE(kernel.flat_sites[1].raw_saddr);
  ASSERT_TRUE(kernel.flat_sites[1].raw_vaddr);
  ASSERT_TRUE(kernel.flat_sites[1].raw_vsrc);
  ASSERT_TRUE(kernel.flat_sites[1].raw_ioffset);
  EXPECT_EQ(*kernel.flat_sites[1].raw_saddr, 0u);
  EXPECT_EQ(*kernel.flat_sites[1].raw_vsrc, 0u);
  EXPECT_EQ(*kernel.flat_sites[1].raw_ioffset, 0);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(SuperColliderDbi, ClassifiesObviousSharedBaseFlatLoad) {
  const std::array<uint32_t, 9> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);

  const SuperColliderDbiKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 36u);
  EXPECT_EQ(kernel.stats.instruction_count, 5u);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_write_count, 0u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_private_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().kind, SuperColliderDbiLdsAccessKind::Read);
  EXPECT_EQ(kernel.flat_sites.front().mnemonic, "flat_load_b32");
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint,
            SuperColliderDbiFlatAddressSpaceHint::Group);
  ASSERT_TRUE(kernel.flat_sites.front().addr_vgpr);
  EXPECT_EQ(*kernel.flat_sites.front().addr_vgpr, 0u);
  ASSERT_TRUE(kernel.flat_sites.front().dst_vgpr);
  EXPECT_EQ(*kernel.flat_sites.front().dst_vgpr, 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(SuperColliderDbi, ClassifiesHighHalfSharedBaseFlatLoadAsMaybeGroup) {
  const std::array<uint32_t, 9> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000080u,              // v_mov_b32_e64 v0, 0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);

  const SuperColliderDbiKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint,
            SuperColliderDbiFlatAddressSpaceHint::MaybeGroup);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(SuperColliderDbi, PropagatesSharedBaseThroughVectorAddCarryAddressConstruction) {
  const std::array<uint32_t, 11> text_words = {
      0xBE8E01EBu,                           // s_mov_b64 s[14:15], src_shared_base
      0xBE81000Fu,                           // s_mov_b32 s1, s15
      0xD5810000u, 0x00000080u,              // v_mov_b32_e64 v0, 0
      0xD5200100u, 0x00220001u,              // v_add_co_ci_u32_e64 v0, s1, s1, v0, s8
      0x7E020300u,                           // v_mov_b32_e32 v1, v0
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);

  const SuperColliderDbiKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint,
            SuperColliderDbiFlatAddressSpaceHint::MaybeGroup);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(SuperColliderDbi, InventoriesLocalFunctionFlatSharedAccesses) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  SuperColliderDbiOptions options;
  options.enabled = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.kernels.front().name, "lds_probe");
  EXPECT_EQ(result.kernels.front().code_size, 4u);
  EXPECT_EQ(result.kernels.front().stats.flat_group_hint_count, 0u);
  ASSERT_EQ(result.functions.size(), 1u);

  const SuperColliderDbiFunctionInfo &function = result.functions.front();
  EXPECT_EQ(function.name, "lds_helper");
  EXPECT_TRUE(function.decoded);
  EXPECT_EQ(function.entry_text_offset, 4u);
  EXPECT_EQ(function.text_file_offset, 0x100u);
  EXPECT_EQ(function.code_size, 36u);
  EXPECT_EQ(function.stats.instruction_count, 5u);
  EXPECT_EQ(function.stats.flat_read_count, 1u);
  EXPECT_EQ(function.stats.flat_group_hint_count, 1u);
  EXPECT_EQ(function.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(function.flat_sites.size(), 1u);
  EXPECT_EQ(function.flat_sites.front().address_space_hint,
            SuperColliderDbiFlatAddressSpaceHint::Group);
  EXPECT_EQ(function.flat_sites.front().text_offset, 24u);
  EXPECT_EQ(function.flat_sites.front().file_offset, 0x118u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(SuperColliderDbi, FlatTrapProofRewritesLikelyGroupLocalFunctionSite) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_flat_trap = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineFlatTrapRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  EXPECT_EQ(result.patches.front().original_size, 12u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  std::array<uint32_t, 3> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words[0], 0xBF900000u); // s_trap 0
  EXPECT_EQ(rewritten_words[1], 0xBF800000u); // s_nop 0
  EXPECT_EQ(rewritten_words[2], 0xBF800000u); // s_nop 0
}

TEST(SuperColliderDbi, FlatCheckTrapProofReportsUnpaddedCandidateCounts) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.patches.empty());

  bool saw_padding_counts = false;
  for (const std::string &warning : result.warnings) {
    saw_padding_counts |= warning.find("supported_candidates=1") != std::string::npos &&
                          warning.find("scratchable_candidates=1") != std::string::npos &&
                          warning.find("max_observed_padding_words=0") != std::string::npos &&
                          warning.find("append_cave_reachable_candidates=1") != std::string::npos;
  }
  EXPECT_TRUE(saw_padding_counts);
}

TEST(SuperColliderDbi, FlatCheckTrapProofUsesReachableUncoveredNopCave) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::array<uint32_t, 12> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::LocalCaveFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 40u);
  EXPECT_EQ(result.patches.front().original_size, 12u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 3> expected_anchor = {
      build_s_branch(3, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x118,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 12> expected_cave = {
      0xEC05007Cu,
      0x00000002u,
      0x00000000u, // original flat_load_b32 v2, v[0:1]
      0xBF800000u, // delay
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // duplicate flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_branch(-13, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_cave.size()> cave_words{};
  std::memcpy(cave_words.data(), result.elf_bytes.data() + 0x128,
              cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(SuperColliderDbi, FlatLoadCheckTrapProofRewritesPaddedLocalFunctionSite) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 11> expected_words = {
      0xEC05007Cu, 0x00000002u, 0x00000000u, // original flat_load_b32 v2, v[0:1]
      0xBF800000u,                           // delay
      0xEC05007Cu, 0x00000005u, 0x00000000u, // duplicate flat_load_b32 v5, v[0:1]
      0xBFC60000u,                           // s_wait_dscnt 0
      0x7C9A0B02u,                           // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u,                           // s_cbranch_vccz +1
      0xBF900000u,                           // s_trap 0
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, FlatStoreCheckTrapProofRewritesPaddedLocalFunctionSite) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC06807Cu, 0x01000000u, 0x00000000u, // flat_store_b32 v[0:1], v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 11> expected_words = {
      0xEC06807Cu, 0x01000000u, 0x00000000u, // original flat_store_b32 v[0:1], v2
      0xBF800000u,                           // delay
      0xEC05007Cu, 0x00000005u, 0x00000000u, // readback flat_load_b32 v5, v[0:1]
      0xBFC60000u,                           // s_wait_dscnt 0
      0x7C9A0B02u,                           // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u,                           // s_cbranch_vccz +1
      0xBF900000u,                           // s_trap 0
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, FlatStoreCheckTrapProofCanUseSleepDelay) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC06807Cu, 0x01000000u, 0x00000000u, // flat_store_b32 v[0:1], v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_flat_check_trap = true;
  options.delay_mode = SuperColliderDbiDelayMode::Sleep;
  options.delay_nops = 9;
  options.scratch_vgpr = 5;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 11> expected_words = {
      0xEC06807Cu,
      0x01000000u,
      0x00000000u, // original flat_store_b32 v[0:1], v2
      build_s_sleep(9, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // readback flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, FlatStoreCheckTrapProofCanUseSleepVarDelay) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 17> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC06807Cu, 0x01000000u, 0x00000000u, // flat_store_b32 v[0:1], v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_flat_check_trap = true;
  options.delay_mode = SuperColliderDbiDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = kRdna4VccLo;
  options.scratch_vgpr = 5;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 11> expected_words = {
      0xEC06807Cu,
      0x01000000u,
      0x00000000u, // original flat_store_b32 v[0:1], v2
      build_s_sleep_var(kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // readback flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x118,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, CountsRdna4LdsAndSynchronizationInstructions) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  SuperColliderDbiOptions options;
  options.enabled = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.target_name, "gfx1201");
  EXPECT_EQ(result.arch_name, "rdna4");

  const SuperColliderDbiKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.name, "lds_probe");
  EXPECT_TRUE(kernel.has_text_range);
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 44u);
  EXPECT_EQ(kernel.stats.instruction_count, 7u);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 1u);
  EXPECT_EQ(kernel.stats.ds_other_count, 0u);
  EXPECT_EQ(kernel.stats.barrier_count, 1u);
  EXPECT_EQ(kernel.stats.wait_count, 1u);
  EXPECT_EQ(kernel.stats.fence_like_count, 1u);
  EXPECT_EQ(kernel.stats.decode_error_count, 0u);
  ASSERT_EQ(kernel.lds_sites.size(), 3u);
  EXPECT_EQ(kernel.lds_sites[0].kind, SuperColliderDbiLdsAccessKind::Write);
  EXPECT_TRUE(kernel.lds_sites[0].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[0].mnemonic, "ds_store_b32");
  EXPECT_EQ(kernel.lds_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.lds_sites[0].file_offset, 0x100u);
  EXPECT_EQ(kernel.lds_sites[0].size, 8u);
  EXPECT_EQ(kernel.lds_sites[0].width_bits, 32u);
  ASSERT_TRUE(kernel.lds_sites[0].addr_vgpr);
  ASSERT_TRUE(kernel.lds_sites[0].data_vgpr);
  EXPECT_FALSE(kernel.lds_sites[0].dst_vgpr);
  EXPECT_EQ(*kernel.lds_sites[0].addr_vgpr, 0u);
  EXPECT_EQ(*kernel.lds_sites[0].data_vgpr, 0u);
  EXPECT_EQ(kernel.lds_sites[1].kind, SuperColliderDbiLdsAccessKind::Read);
  EXPECT_TRUE(kernel.lds_sites[1].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[1].mnemonic, "ds_load_b32");
  EXPECT_EQ(kernel.lds_sites[1].text_offset, 8u);
  EXPECT_EQ(kernel.lds_sites[1].file_offset, 0x108u);
  EXPECT_EQ(kernel.lds_sites[1].size, 8u);
  EXPECT_EQ(kernel.lds_sites[1].width_bits, 32u);
  ASSERT_TRUE(kernel.lds_sites[1].dst_vgpr);
  ASSERT_TRUE(kernel.lds_sites[1].addr_vgpr);
  EXPECT_FALSE(kernel.lds_sites[1].data_vgpr);
  EXPECT_EQ(*kernel.lds_sites[1].dst_vgpr, 0u);
  EXPECT_EQ(*kernel.lds_sites[1].addr_vgpr, 0u);
  EXPECT_EQ(kernel.lds_sites[2].kind, SuperColliderDbiLdsAccessKind::Atomic);
  EXPECT_FALSE(kernel.lds_sites[2].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[2].mnemonic, "ds_add_u32");
  EXPECT_EQ(kernel.lds_sites[2].text_offset, 16u);
  EXPECT_EQ(kernel.lds_sites[2].file_offset, 0x110u);
  EXPECT_EQ(kernel.lds_sites[2].size, 8u);
  EXPECT_EQ(kernel.lds_sites[2].width_bits, 32u);
  EXPECT_EQ(kernel.preflight_action, SuperColliderDbiPreflightAction::Skip);
  ASSERT_GE(kernel.preflight_reasons.size(), 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(SuperColliderDbi, FaultDropBarrierModeRewritesSelectedBarrier) {
  const std::array<uint32_t, 5> text_words = {
      0xBF940000u,              // s_barrier_wait
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBF940000u,              // s_barrier_wait
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.fault_drop_barrier = true;
  options.fault_barrier_index = 1;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineBarrierNopRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 12u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 12u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  uint32_t first_barrier = 0;
  uint32_t rewritten_barrier = 0;
  std::memcpy(&first_barrier, result.elf_bytes.data() + 0x100, sizeof(first_barrier));
  std::memcpy(&rewritten_barrier, result.elf_bytes.data() + 0x100 + 12, sizeof(rewritten_barrier));
  EXPECT_EQ(first_barrier, 0xBF940000u);
  EXPECT_EQ(rewritten_barrier, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(SuperColliderDbi, FaultDropBarrierModeSkipsRocclrRuntimeHelpers) {
  const std::array<uint32_t, 3> text_words = {
      0xBF940000u, // s_barrier_wait
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "__amd_rocclr_fillBufferAligned");
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.fault_drop_barrier = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings)
    saw_skip_warning |= warning.find("skipped ROCclr runtime helper") != std::string::npos;
  EXPECT_TRUE(saw_skip_warning);
}

TEST(SuperColliderDbi, FaultDropBarrierModeComposesWithLdsCheckTrapPatch) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF940000u, // s_barrier_wait
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.fault_drop_barrier = true;
  options.scratch_vgpr = 5;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[1].kind, SuperColliderDbiPatchKind::InlineBarrierNopRewrite);
  EXPECT_EQ(result.patches[1].anchor_offset, 40u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  uint32_t rewritten_barrier = 0;
  std::memcpy(&rewritten_barrier, result.elf_bytes.data() + 0x100 + 40, sizeof(rewritten_barrier));
  EXPECT_EQ(rewritten_barrier, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  uint32_t original_access = 0;
  uint32_t duplicated_access = 0;
  std::memcpy(&original_access, result.elf_bytes.data() + 0x100, sizeof(original_access));
  std::memcpy(&duplicated_access, result.elf_bytes.data() + 0x100 + 8, sizeof(duplicated_access));
  EXPECT_EQ(original_access, 0xD8D80000u);
  EXPECT_EQ(duplicated_access, 0xD8D80000u);
}

TEST(SuperColliderDbi, MarksSupportedLdsKernelAsPreflightCandidate) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  SuperColliderDbiOptions options;
  options.enabled = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_EQ(result.kernels.size(), 1u);

  const SuperColliderDbiKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 24u);
  EXPECT_EQ(kernel.stats.instruction_count, 4u);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 0u);
  EXPECT_EQ(kernel.stats.fence_like_count, 0u);
  ASSERT_EQ(kernel.lds_sites.size(), 2u);
  EXPECT_EQ(kernel.lds_sites[0].kind, SuperColliderDbiLdsAccessKind::Write);
  EXPECT_TRUE(kernel.lds_sites[0].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.lds_sites[0].width_bits, 32u);
  EXPECT_EQ(kernel.lds_sites[1].kind, SuperColliderDbiLdsAccessKind::Read);
  EXPECT_TRUE(kernel.lds_sites[1].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[1].text_offset, 8u);
  EXPECT_EQ(kernel.lds_sites[1].width_bits, 32u);
  EXPECT_EQ(kernel.preflight_action, SuperColliderDbiPreflightAction::Candidate);
  EXPECT_GE(kernel.preflight_reasons.size(), 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(SuperColliderDbi, ProbeNopModeEmitsPatchedElfForCandidate) {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFC60000u,              // s_wait_dscnt
      0x06040F06u,              // v_add_f32_e32 v2, v6, v7
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_nop = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.text_sections.size(), 1u);
  EXPECT_EQ(result.kernels.front().preflight_action, SuperColliderDbiPreflightAction::Candidate);
  EXPECT_TRUE(result.modified);
  EXPECT_FALSE(result.elf_bytes.empty());
  EXPECT_NE(result.elf_bytes, bytes);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 20u);
  EXPECT_EQ(result.patches.front().original_size, 4u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_GT(patched.text_sections().front()->size(), result.text_sections.front().size);
}

TEST(SuperColliderDbi, ProbeNopModeRewritesExistingNopInPlace) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBF800000u, // s_nop 0
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_nop = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineNopRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBF800001u);
}

TEST(SuperColliderDbi, ProbeTrampolineNopModeSkipsExistingNopRewrite) {
  const std::array<uint32_t, 5> text_words = {
      0xBF800000u, // s_nop 0
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 12u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());

  uint32_t original_nop = 0;
  std::memcpy(&original_nop, result.elf_bytes.data() + 0x100, sizeof(original_nop));
  EXPECT_EQ(original_nop, 0xBF800000u);
}

TEST(SuperColliderDbi, ProbeTrampolineNopModeSkipsSClauseRun) {
  const std::array<uint32_t, 10> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBF850001u,              // s_clause 1
      0xF4002200u, 0xF8000020u, // s_load_b64 s[8:9], s[0:1], 0x20
      0xF4006000u, 0xF8000000u, // s_load_b256 s[0:7], s[0:1], 0x0
      0xBFC70000u,              // s_wait_kmcnt 0
      0x06040F06u,              // v_add_f32_e32 v2, v6, v7
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 32u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());

  std::array<uint32_t, 8> prefix_words{};
  std::memcpy(prefix_words.data(), result.elf_bytes.data() + 0x100,
              prefix_words.size() * sizeof(uint32_t));
  EXPECT_EQ(prefix_words[0], 0xD8340000u);
  EXPECT_EQ(prefix_words[1], 0x00000000u);
  EXPECT_EQ(prefix_words[2], 0xBF850001u);
  EXPECT_EQ(prefix_words[3], 0xF4002200u);
  EXPECT_EQ(prefix_words[4], 0xF8000020u);
  EXPECT_EQ(prefix_words[5], 0xF4006000u);
  EXPECT_EQ(prefix_words[6], 0xF8000000u);
  EXPECT_EQ(prefix_words[7], 0xBFC70000u);
}

TEST(SuperColliderDbi, ProbeTrampolineNopModeSkipsRocclrRuntimeHelpers) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "__amd_rocclr_fillBufferAligned");
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings)
    saw_skip_warning |= warning.find("skipped ROCclr runtime helper") != std::string::npos;
  EXPECT_TRUE(saw_skip_warning);
}

TEST(SuperColliderDbi, ProbeTrampolineNopModeSkipsCodeObjectWithoutCandidateSites) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings)
    saw_skip_warning |= warning.find("without supported DBI candidate") != std::string::npos;
  EXPECT_TRUE(saw_skip_warning);
}

TEST(SuperColliderDbi, ProbeNopModePrefersVectorAluAnchorOverMemoryAnchor) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_nop = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
}

TEST(SuperColliderDbi, ProbeEndpgmModeRewritesVectorAluInPlace) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBF800000u, // s_nop 0
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_nop = true;
  options.probe_endpgm = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineEndpgmRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBFB00000u);
}

TEST(SuperColliderDbi, ProbeLdsEndpgmModeRewritesFirstSupportedReadInPlace) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_endpgm = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsEndpgmRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBFB00000u);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeRewritesPaddedLoadInPlace) {
  const std::array<uint32_t, 13> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 48u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 13> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u,
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeCanPatchMultiplePaddedLoads) {
  const std::array<uint32_t, 23> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xD8D80000u,
      0x04000005u, // ds_load_b32 v4, v5
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;
  options.max_patches = 2;
  options.scratch_vgpr = 3;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].trampoline_offset, 8u);
  EXPECT_EQ(result.patches[0].original_size, 44u);
  EXPECT_EQ(result.patches[1].kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 44u);
  EXPECT_EQ(result.patches[1].trampoline_offset, 52u);
  EXPECT_EQ(result.patches[1].original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 22> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x04000005u, // original ds_load_b32 v4, v5
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000005u, // duplicate ds_load_b32 v3, v5
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0704u, // v_cmp_ne_u32_e32 vcc_lo, v4, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeRewritesPaddedU16D16LoadInPlace) {
  const std::array<uint32_t, 14> text_words = {
      0xDA980000u,
      0x01000002u, // ds_load_u16_d16 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 3;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 14> expected_words = {
      0xDA980000u,
      0x01000002u, // original ds_load_u16_d16 v1, v2
      0xBFC60000u, // s_wait_dscnt 0 for original d16 load
      build_v_mov_b32_e32(3, vector_source_vgpr(1), ROCJITSU_CODE_ARCH_RDNA4),
      0xBF800000u, // delay
      0xDA980000u,
      0x03000002u, // duplicate ds_load_u16_d16 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeRewritesPaddedU16D16HiLoadInPlace) {
  const std::array<uint32_t, 14> text_words = {
      0xDA9C0000u,
      0x01000002u, // ds_load_u16_d16_hi v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 3;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 14> expected_words = {
      0xDA9C0000u,
      0x01000002u, // original ds_load_u16_d16_hi v1, v2
      0xBFC60000u, // s_wait_dscnt 0 for original d16 load
      build_v_mov_b32_e32(3, vector_source_vgpr(1), ROCJITSU_CODE_ARCH_RDNA4),
      0xBF800000u, // delay
      0xDA9C0000u,
      0x03000002u, // duplicate ds_load_u16_d16_hi v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeCanUseSleepDelay) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_mode = SuperColliderDbiDelayMode::Sleep;
  options.delay_nops = 7;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 12> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      build_s_sleep(7, ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeCanUseSleepVarDelay) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_mode = SuperColliderDbiDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = kRdna4VccLo;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 12> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      build_s_sleep_var(kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeRejectsOversizedSleepDelay) {
  const std::array<uint32_t, 4> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.delay_mode = SuperColliderDbiDelayMode::Sleep;
  options.delay_nops = 65536;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("16-bit s_sleep"), std::string::npos);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeRewritesPaddedStoreInPlace) {
  const std::array<uint32_t, 13> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 48u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 13> expected_words = {
      0xD8340000u,
      0x00000102u, // original ds_store_b32 v2, v1
      0xBF800000u,
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // readback ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeRewritesPaddedB64LoadInPlace) {
  const std::array<uint32_t, 16> text_words = {
      0xD9D80000u,
      0x01000009u, // ds_load_b64 v[1:2], v9
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 2;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 60u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 16> expected_words = {
      0xD9D80000u,
      0x01000009u, // original ds_load_b64 v[1:2], v9
      0xBF800000u,
      0xBF800000u, // delay
      0xD9D80000u,
      0x05000009u, // duplicate ds_load_b64 v[5:6], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeRewritesPaddedB128StoreInPlace) {
  const std::array<uint32_t, 21> text_words = {
      0xDB7C0000u,
      0x00000109u, // ds_store_b128 v9, v[1:4]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 80u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 21> expected_words = {
      0xDB7C0000u,
      0x00000109u, // original ds_store_b128 v9, v[1:4]
      0xBF800000u, // delay
      0xDBFC0000u,
      0x05000009u, // readback ds_load_b128 v[5:8], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A0F03u, // v_cmp_ne_u32_e32 vcc_lo, v3, v7
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A1104u, // v_cmp_ne_u32_e32 vcc_lo, v4, v8
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeAutoScratchUsesLiveness) {
  const std::array<uint32_t, 14> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0x06080603u, // v_add_f32_e32 v4, v3, v3
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.delay_nops = 2;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 14> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u,
      0xBF800000u, // delay
      0xD8D80000u,
      0x05000002u, // duplicate ds_load_b32 v5, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(4, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 4, ROCJITSU_CODE_ARCH_RDNA4),
      0x06080603u, // original v_add_f32_e32 after padding
      0xBFB00000u, // original s_endpgm
  };
  std::array<uint32_t, expected_words.size()> rewritten_words{};
  std::memcpy(rewritten_words.data(), result.elf_bytes.data() + 0x100,
              rewritten_words.size() * sizeof(uint32_t));
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeAutoScratchStaysWithinDescriptorVgprs) {
  const std::array<uint32_t, 14> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xD8340000u,
      0x00000302u, // ds_store_b32 v2, v3
      0xBFB00000u, // s_endpgm
  };
  const uint32_t four_vgprs_granulated = 0;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", four_vgprs_granulated);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_FALSE(result.warnings.empty());

  bool saw_descriptor_bound = false;
  for (const std::string &warning : result.warnings)
    saw_descriptor_bound |= warning.find("supported_candidates=2") != std::string::npos &&
                            warning.find("scratchable_candidates=0") != std::string::npos;
  EXPECT_TRUE(saw_descriptor_bound);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeSkipsDescriptorEdgeB64LoadScratch) {
  const std::array<uint32_t, 5> text_words = {
      0xD9D80000u,
      0x01000009u, // ds_load_b64 v[1:2], v9
      0xD8D80000u,
      0x0D000002u, // ds_load_b32 v13, v2
      0xBFB00000u, // s_endpgm
  };
  const uint32_t sixteen_vgprs_granulated = 3;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", sixteen_vgprs_granulated);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 14u);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeUsesReachableUncoveredNopCave) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 12> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 1;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 16u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(3, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 12> expected_cave = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-14, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_cave.size()> cave_words{};
  std::memcpy(cave_words.data(), result.elf_bytes.data() + 0x110,
              cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeSelectsOneLocalCavePerKernel) {
  const std::array<uint32_t, 5> kernel_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xD8D80000u,
      0x04000005u, // ds_load_b32 v4, v5
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 25> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.max_patches = 2;
  options.scratch_vgpr = 6;
  options.delay_nops = 1;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 24u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeUsesReachableUncoveredNopCaveFor2addrB64Load) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD9DC0000u,
      0x01000009u, // ds_load_2addr_b64 v[1:4], v9
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 21> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 16u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(3, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 21> expected_cave = {
      0xD9DC0000u,
      0x01000009u, // original ds_load_2addr_b64 v[1:4], v9
      0xBF800000u, // delay
      0xD9DC0000u,
      0x05000009u, // duplicate ds_load_2addr_b64 v[5:8], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0F03u, // v_cmp_ne_u32_e32 vcc_lo, v3, v7
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A1104u, // v_cmp_ne_u32_e32 vcc_lo, v4, v8
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-23, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_cave.size()> cave_words{};
  std::memcpy(cave_words.data(), result.elf_bytes.data() + 0x110,
              cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeUsesReachableUncoveredNopCaveForB128Store) {
  const std::array<uint32_t, 3> kernel_words = {
      0xDB7C0000u,
      0x00000109u, // ds_store_b128 v9, v[1:4]
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 21> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::LocalCaveLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 16u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(3, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 21> expected_cave = {
      0xDB7C0000u,
      0x00000109u, // original ds_store_b128 v9, v[1:4]
      0xBF800000u, // delay
      0xDBFC0000u,
      0x05000009u, // readback ds_load_b128 v[5:8], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0F03u, // v_cmp_ne_u32_e32 vcc_lo, v3, v7
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A1104u, // v_cmp_ne_u32_e32 vcc_lo, v4, v8
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-23, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_cave.size()> cave_words{};
  std::memcpy(cave_words.data(), result.elf_bytes.data() + 0x110,
              cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeUsesAppendedTextCaveWhenNoLocalCaveFits) {
  const std::array<uint32_t, 3> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 12u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  EXPECT_GT(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(2, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);
}

TEST(SuperColliderDbi, ProbeLdsCheckTrapModeReportsExcessiveDelay) {
  const std::array<uint32_t, 3> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 300;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_FALSE(result.warnings.empty());
  EXPECT_NE(result.warnings.back().find("requested delay needs too much padding"),
            std::string::npos);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(SuperColliderDbi, ProbeLdsEndpgmModeCanRewritePreflightSkippedKernel) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.probe_lds_endpgm = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.kernels.front().preflight_action, SuperColliderDbiPreflightAction::Skip);
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, SuperColliderDbiPatchKind::InlineLdsEndpgmRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBFB00000u);
}

TEST(SuperColliderDbi, FailClosedRejectsUnsupportedLdsKernel) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  SuperColliderDbiOptions options;
  options.enabled = true;
  options.fail_closed = true;

  const auto result = try_patch_supercollider_dbi(bytes, options);

  ASSERT_FALSE(result.errors.empty());
  ASSERT_EQ(result.kernels.size(), 1u);
  const SuperColliderDbiKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.preflight_action, SuperColliderDbiPreflightAction::Reject);
  ASSERT_GE(kernel.preflight_reasons.size(), 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

} // namespace
} // namespace rocjitsu
