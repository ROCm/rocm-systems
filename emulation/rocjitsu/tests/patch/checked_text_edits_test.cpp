// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "../dbi_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/code_object_patcher.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {
constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
// gfx1200 / RDNA4, with the kernel entry at .text offset 0. All offsets below
// are text-relative; the ELF builder's file layout is a separate coordinate system.
//
// Offset  Bytes  Instruction / encoded words
// 0x00      4   s_nop 1
// 0x04      4   s_nop 2                         (unchanged control)
// 0x08      8   s_mov_b32 s0, 0x12345678          (0xbe8000ff, literal 0x12345678)
// 0x10      4   s_nop 3                         (unchanged control)
// 0x14     12   global_store_b32 v0, v1, s[2:3] (0xee068002, 0x00800000, 0)
// 0x20      4   s_endpgm                        (unchanged control)
// Total: 0x24 / 36 bytes. The repository decoder and the pinned SDK's gfx1200
// disassembler agree on these extents and mnemonics. The final zero word of the
// global store belongs to that instruction; it is not independently skippable padding.
std::vector<uint8_t> make_mixed_width_kernel_elf() {
  constexpr util::FailureOr<uint32_t> kNop1 = build_s_nop(1, kArch);
  constexpr util::FailureOr<uint32_t> kNop2 = build_s_nop(2, kArch);
  constexpr util::FailureOr<uint32_t> kNop3 = build_s_nop(3, kArch);
  constexpr util::FailureOr<uint32_t> kEndpgm = build_s_endpgm(kArch);
  static_assert(kNop1.succeeded() && kNop2.succeeded() && kNop3.succeeded() && kEndpgm.succeeded());
  return test::make_gfx1200_kernel_elf({kNop1.value(), kNop2.value(), 0xbe8000ffu, 0x12345678u,
                                        kNop3.value(), 0xee068002u, 0x00800000u, 0,
                                        kEndpgm.value()},
                                       0);
}

// Keep the kernel descriptor intact and put a symbol-less pointer in separate
// loaded data. No text symbol or direct branch names its target. nullopt names
// the data slot itself instead; nonallocated metadata retains the same r_offset.
std::vector<uint8_t> with_relative64_reference(std::vector<uint8_t> image,
                                               std::optional<uint64_t> text_offset,
                                               bool nonallocated_metadata = false,
                                               bool trailing_partial_record = false) {
  const AmdGpuCodeObject original(image.data(), image.size());
  const Section &text = *original.text_sections().front();
  std::vector<Elf64_Shdr> sections = CodeObjectPatcher(original).section_headers();
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  EXPECT_TRUE(segments);
  if (!segments)
    return {};
  const std::vector<Elf64_Phdr>::iterator data_load =
      std::ranges::find_if(*segments, [](const Elf64_Phdr &segment) {
        return segment.p_type == PT_LOAD && (segment.p_flags & PF_X) == 0;
      });
  const std::vector<Elf64_Shdr>::iterator symbols =
      std::ranges::find(sections, SHT_SYMTAB, &Elf64_Shdr::sh_type);
  EXPECT_NE(data_load, segments->end());
  EXPECT_NE(symbols, sections.end());
  if (data_load == segments->end() || symbols == sections.end())
    return {};
  const uint32_t symbol_index = static_cast<uint32_t>(symbols - sections.begin());
  const uint64_t data_offset = header.e_shoff;
  const uint64_t data_vaddr = data_load->p_vaddr + data_offset - data_load->p_offset;
  const uint64_t rela_offset = data_offset + sizeof(uint64_t);
  const uint64_t rela_vaddr = data_vaddr + sizeof(uint64_t);
  header.e_shoff = rela_offset + sizeof(Elf64_Rela);
  // Keep the section table aligned while admitting one incomplete tail byte.
  if (trailing_partial_record)
    header.e_shoff += alignof(Elf64_Shdr);
  image.resize(header.e_shoff);
  const uint64_t slot = 0;
  std::memcpy(image.data() + data_offset, &slot, sizeof(slot));
  const Elf64_Rela relocation = {
      .r_offset = data_vaddr,
      .r_info = R_AMDGPU_RELATIVE64,
      .r_addend = static_cast<int64_t>(text_offset ? text.vaddr() + *text_offset : data_vaddr)};
  std::memcpy(image.data() + rela_offset, &relocation, sizeof(relocation));
  sections.push_back({.sh_name = 0,
                      .sh_type = SHT_PROGBITS,
                      .sh_flags = SHF_ALLOC | SHF_WRITE,
                      .sh_addr = data_vaddr,
                      .sh_offset = data_offset,
                      .sh_size = sizeof(uint64_t),
                      .sh_link = 0,
                      .sh_info = 0,
                      .sh_addralign = alignof(uint64_t),
                      .sh_entsize = 0});
  sections.push_back({.sh_name = 0,
                      .sh_type = SHT_RELA,
                      .sh_flags = SHF_ALLOC,
                      .sh_addr = rela_vaddr,
                      .sh_offset = rela_offset,
                      .sh_size = sizeof(Elf64_Rela) + (trailing_partial_record ? 1u : 0u),
                      .sh_link = symbol_index,
                      .sh_info = nonallocated_metadata ? header.e_shstrndx : SHN_UNDEF,
                      .sh_addralign = alignof(Elf64_Rela),
                      .sh_entsize = sizeof(Elf64_Rela)});
  data_load->p_flags |= 0x2u; // PF_W
  data_load->p_filesz = header.e_shoff - data_load->p_offset;
  data_load->p_memsz = data_load->p_filesz;
  header.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(header.e_shoff + sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data(), &header, sizeof(header));
  std::memcpy(image.data() + header.e_phoff, segments->data(),
              segments->size() * sizeof(Elf64_Phdr));
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  return image;
}

// Reuse the relocation layout, with a text symbol at offset zero. Its endpoint
// stays outside the edited instruction; only the relocation addend names its interior.
std::vector<uint8_t> with_absolute64_text_reference(int64_t addend, bool nonallocated_metadata,
                                                    bool trailing_partial_record) {
  std::vector<uint8_t> image = with_relative64_reference(
      make_mixed_width_kernel_elf(), 0, nonallocated_metadata, trailing_partial_record);
  const AmdGpuCodeObject object(image.data(), image.size());
  const std::vector<Elf64_Shdr> sections = CodeObjectPatcher(object).section_headers();
  const Elf64_Shdr &relocs = sections.back();
  const Elf64_Shdr &symbols = sections[relocs.sh_link];
  const Section &text = *object.text_sections().front();
  constexpr uint32_t kTextSymbolIndex = 1;
  const Elf64_Sym symbol = {.st_name = 0,
                            .st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc),
                            .st_other = 0,
                            .st_shndx = static_cast<uint16_t>(*text.sectionHeaderIndex()),
                            .st_value = text.vaddr(),
                            .st_size = 0};
  std::memcpy(image.data() + symbols.sh_offset + kTextSymbolIndex * sizeof(symbol), &symbol,
              sizeof(symbol));
  Elf64_Rela relocation = {};
  std::memcpy(&relocation, image.data() + relocs.sh_offset, sizeof(relocation));
  relocation.r_info = (uint64_t{kTextSymbolIndex} << 32) | R_AMDGPU_ABS64;
  relocation.r_addend = addend;
  std::memcpy(image.data() + relocs.sh_offset, &relocation, sizeof(relocation));
  return image;
}

TEST(CheckedTextEdits, RejectsSymbolReferenceBeforePartialRelocationTail) {
  for (const bool trailing_partial_record : {false, true}) {
    SCOPED_TRACE(trailing_partial_record);
    // Offset 0 plus addend 12 points inside the eight-byte instruction at 8.
    const std::vector<uint8_t> image =
        with_absolute64_text_reference(12, false, trailing_partial_record);
    const AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    const CodeObjectPatcher patcher(object);
    ASSERT_FALSE(patcher.has_relocations_within_text());
    ASSERT_FALSE(patcher.has_malformed_rocr_relocation_section());
    ASSERT_FALSE(patcher.has_rocr_rejected_none_relocation());
    EXPECT_TRUE(patcher.has_unsupported_relocation_to_text());
  }
}

TEST(CheckedTextEdits, AllowsSupportedSymbolReferencesBeforePartialRelocationTail) {
  for (const bool trailing_partial_record : {false, true}) {
    SCOPED_TRACE(trailing_partial_record);
    for (const bool nonallocated_metadata : {false, true}) {
      SCOPED_TRACE(nonallocated_metadata);
      // A zero-addend reference follows the symbol at offset zero. A reference
      // from nonallocated metadata is ignored even when its addend names the interior.
      const std::vector<uint8_t> image = with_absolute64_text_reference(
          nonallocated_metadata ? 12 : 0, nonallocated_metadata, trailing_partial_record);
      const AmdGpuCodeObject object(image.data(), image.size());
      ASSERT_TRUE(object.is_valid());
      EXPECT_FALSE(CodeObjectPatcher(object).has_unsupported_relocation_to_text());
    }
  }
}

TEST(CheckedTextEdits, InstrumentorRejectsRelativeReferenceInsideAnchor) {
  const std::vector<uint8_t> image = with_relative64_reference(make_mixed_width_kernel_elf(), 12);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  Instrumentor instrumentor(object, kArch);
  instrumentor.add_point_by_offset(8);
  const Instrumentor::ValidationResult validated = instrumentor.validate_points();
  ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
  ASSERT_EQ(validated.sites.size(), 1u);
  ASSERT_EQ(validated.sites.front().original_size, 8u);
  const std::vector<uint8_t> original_image = image;
  const InstrumentedCodeObject result = instrumentor.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("anchor edit transaction:"), std::string::npos);
  EXPECT_NE(result.errors.front().find("relocation"), std::string::npos);
  EXPECT_EQ(image, original_image);
  EXPECT_EQ(std::memcmp(object.image_data(), original_image.data(), original_image.size()), 0);
}

} // namespace
} // namespace rocjitsu
