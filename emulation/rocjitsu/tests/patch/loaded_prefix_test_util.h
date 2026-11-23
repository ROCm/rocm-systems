// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "../dbi_test_util.h"
#include "../elf_test_support.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace rocjitsu::test {

inline constexpr rocjitsu::IsaFamily kLoadedPrefixArch = rocjitsu::IsaFamily::Cdna5;
inline constexpr uint64_t kLoadedPrefixSize = sizeof(uint64_t);

struct LoadedPrefixRelocation {
  std::vector<uint8_t> image;
  uint32_t barrier_word = 0;
  uint32_t nop_word = 0;
  uint64_t prefix_file_offset = 0;
};

// Build and qualify the common ELF input; operation admission and emitted-byte
// assertions stay with their observation or replacement test. Call through
// ASSERT_NO_FATAL_FAILURE so a malformed fixture cannot bypass qualification.
inline void make_loaded_prefix_relocation(uint64_t bytes_before_text, bool overlaps_text,
                                          LoadedPrefixRelocation &fixture) {
  fixture = {};
  constexpr rocjitsu::IsaFamily arch = kLoadedPrefixArch;
  const util::FailureOr<uint32_t> barrier =
      build_sop1_encoding(arch, cdna5::kSBarrierSignalSop1, 0, 193);
  const util::FailureOr<uint32_t> nop = build_s_nop(0, arch);
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(barrier.succeeded());
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(endpgm.succeeded());
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(
      {barrier.value(), nop.value(), endpgm.value()}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const AmdGpuCodeObject original(image.data(), image.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.text_sections().size(), 1u);
  const Section &original_text = *original.text_sections().front();
  const CodeObjectPatcher original_patcher(original);
  std::vector<Elf64_Shdr> sections = original_patcher.section_headers();
  ASSERT_EQ(sections.size(), 6u);
  ASSERT_EQ(sections[1].sh_flags, SHF_ALLOC | SHF_EXECINSTR);
  ASSERT_EQ(sections[2].sh_flags, SHF_ALLOC);
  ASSERT_EQ(sections[3].sh_type, SHT_SYMTAB);
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_VERSION] = 1;
  header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  const std::vector<Elf64_Phdr>::iterator text_load =
      std::ranges::find_if(*segments, [](const Elf64_Phdr &segment) {
        return segment.p_type == PT_LOAD && (segment.p_flags & PF_X) != 0;
      });
  const std::vector<Elf64_Phdr>::iterator data_load =
      std::ranges::find_if(*segments, [](const Elf64_Phdr &segment) {
        return segment.p_type == PT_LOAD && (segment.p_flags & PF_X) == 0;
      });
  const std::vector<Elf64_Shdr>::iterator symbols =
      std::ranges::find(sections, SHT_SYMTAB, &Elf64_Shdr::sh_type);
  ASSERT_NE(text_load, segments->end());
  ASSERT_NE(data_load, segments->end());
  ASSERT_NE(symbols, sections.end());
  ASSERT_EQ(text_load->p_offset, original_text.sectionOffset());
  ASSERT_EQ(text_load->p_vaddr, original_text.vaddr());
  ASSERT_GE(text_load->p_offset, kLoadedPrefixSize);
  ASSERT_LE(bytes_before_text, kLoadedPrefixSize);
  text_load->p_offset -= kLoadedPrefixSize;
  text_load->p_vaddr -= kLoadedPrefixSize;
  text_load->p_paddr -= kLoadedPrefixSize;
  text_load->p_filesz += kLoadedPrefixSize;
  text_load->p_memsz += kLoadedPrefixSize;
  text_load->p_flags |= 0x2u; // PF_W
  std::fill_n(image.data() + text_load->p_offset, kLoadedPrefixSize, uint8_t{0x5a});

  // This is actual loaded, file-backed padding, not an address before the LOAD.
  // ABS64 writes eight bytes even when only its tail reaches the first instruction.
  const uint64_t relocation_offset = header.e_shoff;
  const uint64_t relocation_vaddr = data_load->p_vaddr + relocation_offset - data_load->p_offset;
  const Elf64_Rela relocation = {.r_offset = original_text.vaddr() - bytes_before_text,
                                 .r_info = (uint64_t{1} << 32) | R_AMDGPU_ABS64,
                                 .r_addend = 0};
  const uint32_t symbol_table_index = static_cast<uint32_t>(symbols - sections.begin());
  symbols->sh_type = SHT_DYNSYM;
  header.e_shoff += sizeof(relocation);
  data_load->p_filesz = header.e_shoff - data_load->p_offset;
  data_load->p_memsz = data_load->p_filesz;
  sections.push_back({.sh_name = 0,
                      .sh_type = SHT_RELA,
                      .sh_flags = SHF_ALLOC,
                      .sh_addr = relocation_vaddr,
                      .sh_offset = relocation_offset,
                      .sh_size = sizeof(relocation),
                      .sh_link = symbol_table_index,
                      .sh_info = SHN_UNDEF,
                      .sh_addralign = alignof(Elf64_Rela),
                      .sh_entsize = sizeof(relocation)});
  header.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(header.e_shoff + sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data() + relocation_offset, &relocation, sizeof(relocation));
  std::memcpy(image.data(), &header, sizeof(header));
  std::memcpy(image.data() + header.e_phoff, segments->data(),
              segments->size() * sizeof(Elf64_Phdr));
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));

  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.target_id(), ROCJITSU_CODE_TARGET_GFX1250);
  ASSERT_EQ(source.text_sections().size(), 1u);
  const Section &text = *source.text_sections().front();
  ASSERT_EQ(text.vaddr(), original_text.vaddr());
  ASSERT_EQ(classify_rocr_relocation_section(header, sections, sections.back()),
            RocrRelocationSectionMode::Dynamic);
  ASSERT_EQ(sections[symbol_table_index].sh_type, SHT_DYNSYM);
  const Elf64_Sym symbol = test_support::read_elf_struct_for_test<Elf64_Sym>(
      image, sections[symbol_table_index].sh_offset + sizeof(Elf64_Sym));
  const std::optional<Elf64_Sym> named_symbol =
      test_support::find_elf_symbol_for_test(image, "test_kernel.kd");
  ASSERT_TRUE(named_symbol);
  ASSERT_NE(symbol.st_name, 0u);
  ASSERT_EQ(symbol.st_name, named_symbol->st_name);
  ASSERT_NE(symbol.st_shndx, *text.sectionHeaderIndex());
  ASSERT_NE(symbol.st_shndx, SHN_UNDEF);
  ASSERT_EQ(elf_reloc_type(relocation.r_info), R_AMDGPU_ABS64);
  ASSERT_EQ(elf_reloc_sym(relocation.r_info), 1u);
  ASSERT_LT(relocation.r_offset, text.vaddr());
  ASSERT_GE(relocation.r_offset, text_load->p_vaddr);
  ASSERT_LE(relocation.r_offset + sizeof(uint64_t), text_load->p_vaddr + text_load->p_filesz);
  ASSERT_EQ(relocation.r_offset + sizeof(uint64_t) > text.vaddr(), overlaps_text);
  ASSERT_TRUE(std::ranges::none_of(sections, [&](const Elf64_Shdr &section) {
    return (section.sh_flags & SHF_ALLOC) != 0 && relocation.r_offset >= section.sh_addr &&
           relocation.r_offset - section.sh_addr < section.sh_size;
  }));
  const uint64_t write_file_offset = text_load->p_offset + relocation.r_offset - text_load->p_vaddr;
  ASSERT_LT(write_file_offset, text.sectionOffset());
  ASSERT_LE(write_file_offset + sizeof(uint64_t), image.size());
  ASSERT_EQ(image[write_file_offset], 0x5a);
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), source.target_id());
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 1> barrier_words = {barrier.value()};
  const DecodeResult decoded = decoder->decode_window(barrier_words);
  ASSERT_TRUE(decoded.succeeded());
  ASSERT_EQ(decoded.value()->mnemonic(), "s_barrier_signal");
  ASSERT_EQ(decoded.value()->size(), sizeof(uint32_t));
  const CodeObjectPatcher patcher(source);
  EXPECT_FALSE(patcher.has_malformed_rocr_relocation_section());
  EXPECT_FALSE(patcher.has_rocr_rejected_none_relocation());
  EXPECT_FALSE(patcher.has_unsupported_relocation_to_text());
  EXPECT_EQ(patcher.has_relocations_within_text(), overlaps_text);

  fixture = {.image = std::move(image),
             .barrier_word = barrier.value(),
             .nop_word = nop.value(),
             .prefix_file_offset = text_load->p_offset};
}

} // namespace rocjitsu::test
