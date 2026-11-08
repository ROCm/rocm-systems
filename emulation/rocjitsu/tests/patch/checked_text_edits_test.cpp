// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "../dbi_test_util.h"
#include "../elf_test_support.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/operand_types.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

// Keep both executable sections loaded while only the exact .text name is
// indexed by Instrumentor. The branch at 0x304 may legally leave that section.
std::vector<uint8_t> make_branch_to_earlier_section_elf(uint32_t branch) {
  constexpr uint64_t kEarlyAddress = 0x200;
  constexpr uint64_t kTextAddress = 0x300;
  constexpr util::FailureOr<uint32_t> kNop = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA2);
  constexpr util::FailureOr<uint32_t> kEndpgm = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA2);
  static_assert(kNop.succeeded() && kEndpgm.succeeded());
  const std::array<uint32_t, 2> words = {kNop.value(), branch};
  std::vector<uint8_t> names = {'\0'};
  const uint32_t early_name = test::add_elf_name(names, ".text.early");
  const uint32_t text_name = test::add_elf_name(names, ".text");
  const uint32_t names_name = test::add_elf_name(names, ".shstrtab");
  const uint64_t names_offset = kTextAddress + sizeof(words);
  const uint64_t section_offset = test::align_up_for_test(names_offset + names.size(), 8);
  std::array<Elf64_Shdr, 4> sections = {};
  sections[1] = {.sh_name = early_name,
                 .sh_type = SHT_PROGBITS,
                 .sh_flags = SHF_ALLOC | SHF_EXECINSTR,
                 .sh_addr = kEarlyAddress,
                 .sh_offset = kEarlyAddress,
                 .sh_size = sizeof(uint32_t),
                 .sh_link = 0,
                 .sh_info = 0,
                 .sh_addralign = sizeof(uint32_t),
                 .sh_entsize = 0};
  sections[2] = {.sh_name = text_name,
                 .sh_type = SHT_PROGBITS,
                 .sh_flags = SHF_ALLOC | SHF_EXECINSTR,
                 .sh_addr = kTextAddress,
                 .sh_offset = kTextAddress,
                 .sh_size = sizeof(words),
                 .sh_link = 0,
                 .sh_info = 0,
                 .sh_addralign = sizeof(uint32_t),
                 .sh_entsize = 0};
  sections[3] = {.sh_name = names_name,
                 .sh_type = SHT_STRTAB,
                 .sh_flags = 0,
                 .sh_addr = 0,
                 .sh_offset = names_offset,
                 .sh_size = names.size(),
                 .sh_link = 0,
                 .sh_info = 0,
                 .sh_addralign = 1,
                 .sh_entsize = 0};
  Elf64_Ehdr header = {};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_DYN;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_phoff = sizeof(header);
  header.e_shoff = section_offset;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1030;
  header.e_ehsize = sizeof(header);
  header.e_phentsize = sizeof(Elf64_Phdr);
  header.e_phnum = 1;
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = sections.size();
  header.e_shstrndx = 3;
  const Elf64_Phdr segment = {.p_type = PT_LOAD,
                              .p_flags = 0x5, // PF_R | PF_X
                              .p_offset = kEarlyAddress,
                              .p_vaddr = kEarlyAddress,
                              .p_paddr = kEarlyAddress,
                              .p_filesz = names_offset - kEarlyAddress,
                              .p_memsz = names_offset - kEarlyAddress,
                              .p_align = sizeof(uint32_t)};
  std::vector<uint8_t> image(section_offset + sizeof(sections), 0);
  test_support::write_elf_struct_for_test(image, 0, header);
  test_support::write_elf_struct_for_test(image, header.e_phoff, segment);
  test_support::write_elf_struct_for_test(image, kEarlyAddress, kEndpgm.value());
  std::memcpy(image.data() + kTextAddress, words.data(), sizeof(words));
  std::memcpy(image.data() + names_offset, names.data(), names.size());
  std::memcpy(image.data() + section_offset, sections.data(), sizeof(sections));
  return image;
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

// Replace the descriptor symbol with a text label. These probe-free edits do not
// need a descriptor; keeping its old name would make this label look like one.
std::vector<uint8_t> with_text_symbol(uint64_t start, uint64_t size, bool relocatable) {
  std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject original(image.data(), image.size());
  const Section &text = *original.text_sections().front();
  std::vector<Elf64_Shdr> sections = CodeObjectPatcher(original).section_headers();
  const std::vector<Elf64_Shdr>::iterator symbols =
      std::ranges::find(sections, SHT_SYMTAB, &Elf64_Shdr::sh_type);
  EXPECT_NE(symbols, sections.end());
  if (symbols == sections.end())
    return {};
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  if (relocatable) {
    header.e_type = ET_REL;
    header.e_phoff = 0;
    header.e_phnum = 0;
    // Drop linked addresses along with the load segments. Otherwise growing
    // .text shifts .rodata without a segment alignment to preserve its layout.
    for (Elf64_Shdr &section : sections) {
      if ((section.sh_flags & SHF_ALLOC) != 0)
        section.sh_addr = 0;
    }
    // A nonzero section address makes erroneous subtraction observable: the
    // interior label at section offset 12 must not be normalized to offset 4.
    sections[*text.sectionHeaderIndex()].sh_addr = 8;
  }
  const Elf64_Sym symbol = {.st_name = 0,
                            .st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeNone),
                            .st_other = 0,
                            .st_shndx = static_cast<uint16_t>(*text.sectionHeaderIndex()),
                            .st_value = relocatable ? start : text.vaddr() + start,
                            .st_size = size};
  std::memcpy(image.data() + symbols->sh_offset + sizeof(Elf64_Sym), &symbol, sizeof(symbol));
  std::memcpy(image.data(), &header, sizeof(header));
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  return image;
}

// Exercise the public observation path, including the distinction between
// per-site eligibility and the reference checks performed on the final batch.
void expect_reference_result(const std::vector<uint8_t> &image, uint64_t anchor,
                             std::optional<uint64_t> rejected_reference,
                             rj_code_arch_t arch = kArch) {
  const std::vector<uint8_t> original_image = image;
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  Instrumentor instrumentor(object, arch);
  instrumentor.add_point_by_offset(anchor);
  const Instrumentor::ValidationResult validated = instrumentor.validate_points();
  ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
  ASSERT_EQ(validated.sites.size(), 1u);
  const InstrumentedCodeObject result = instrumentor.patch();
  if (rejected_reference) {
    EXPECT_TRUE(result.elf_bytes.empty());
    EXPECT_TRUE(result.sites.empty());
    ASSERT_FALSE(result.errors.empty());
    ASSERT_TRUE(result.edit_error);
    EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
    EXPECT_EQ(result.edit_error->offset, *rejected_reference);
  } else {
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    EXPECT_FALSE(result.edit_error);
    ASSERT_FALSE(result.elf_bytes.empty());
    ASSERT_EQ(result.sites.size(), 1u);
    EXPECT_EQ(result.sites.front().original_offset, anchor);
    const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    EXPECT_TRUE(patched.is_valid());
  }
  EXPECT_EQ(image, original_image);
  EXPECT_EQ(std::memcmp(object.image_data(), original_image.data(), original_image.size()), 0);
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
  EXPECT_TRUE(result.sites.empty());
  ASSERT_FALSE(result.errors.empty());
  ASSERT_TRUE(result.edit_error);
  EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
  EXPECT_EQ(result.edit_error->offset, 12u);
  EXPECT_NE(result.errors.front().find("anchor edit transaction:"), std::string::npos);
  EXPECT_NE(result.errors.front().find("relocation"), std::string::npos);
  EXPECT_EQ(image, original_image);
  EXPECT_EQ(std::memcmp(object.image_data(), original_image.data(), original_image.size()), 0);
}

TEST(CheckedTextEdits, InstrumentorChecksDirectReferencesIncludingUnreachableBranches) {
  for (const bool unreachable : {false, true}) {
    SCOPED_TRACE(unreachable);
    for (const uint64_t destination : {8u, 12u, 16u}) {
      SCOPED_TRACE(destination);
      std::vector<uint8_t> image = make_mixed_width_kernel_elf();
      const AmdGpuCodeObject object(image.data(), image.size());
      const uint64_t text_file_offset = object.text_sections().front()->sectionOffset();
      const uint64_t branch_offset = unreachable ? 4 : 0;
      if (unreachable) {
        // Jump over the second branch. The complete text scan must retain that
        // unreachable branch's reference, not just references in entry's CFG.
        const util::FailureOr<uint32_t> maybe_skip = build_s_branch(1, kArch);
        ASSERT_TRUE(maybe_skip.succeeded());
        const uint32_t skip = maybe_skip.value();
        std::memcpy(image.data() + text_file_offset, &skip, sizeof(skip));
      }
      const int16_t displacement = static_cast<int16_t>((destination - branch_offset - 4) / 4);
      const util::FailureOr<uint32_t> maybe_branch = build_s_branch(displacement, kArch);
      ASSERT_TRUE(maybe_branch.succeeded());
      const uint32_t branch = maybe_branch.value();
      std::memcpy(image.data() + text_file_offset + branch_offset, &branch, sizeof(branch));
      expect_reference_result(image, 8,
                              destination == 12 ? std::optional<uint64_t>(12) : std::nullopt);
    }
  }
}

TEST(CheckedTextEdits, PreservesBranchToEarlierExecutableSection) {
  constexpr rj_code_arch_t kRdna2 = ROCJITSU_CODE_ARCH_RDNA2;
  // The branch at 0x304 uses next PC 0x308: (0x200 - 0x308) / 4 = -66.
  const util::FailureOr<uint32_t> branch = build_s_branch(-66, kRdna2);
  ASSERT_TRUE(branch.succeeded());
  const std::vector<uint8_t> image = make_branch_to_earlier_section_elf(branch.value());
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  ASSERT_EQ(object.text_sections().size(), 1u);
  ASSERT_EQ(object.allocated_executable_sections().size(), 2u);
  const Section &text = *object.text_sections().front();
  ASSERT_EQ(text.vaddr(), 0x300u);
  const Section *early = test_support::find_section(object, ".text.early");
  ASSERT_NE(early, nullptr);
  ASSERT_EQ(early->vaddr(), 0x200u);
  ASSERT_EQ(early->flags() & (SHF_ALLOC | SHF_EXECINSTR), SHF_ALLOC | SHF_EXECINSTR);
  Instrumentor instrumentor(object, kRdna2);
  instrumentor.add_point_by_offset(0);
  const Instrumentor::ValidationResult validated = instrumentor.validate_points();
  ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
  ASSERT_EQ(validated.sites.size(), 1u);
  ASSERT_EQ(validated.sites.front().anchor_offset, 0u);
  const InstrumentedCodeObject result = instrumentor.patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.sites.size(), 1u);
  const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  ASSERT_EQ(patched.allocated_executable_sections().size(), 2u);
  const Section &patched_text = *patched.text_sections().front();
  EXPECT_EQ(patched_text.vaddr(), text.vaddr());
  ASSERT_GE(patched_text.size(), text.size());
  EXPECT_EQ(std::memcmp(patched_text.data() + 4, text.data() + 4, sizeof(uint32_t)), 0);
  const Section *patched_early = test_support::find_section(patched, ".text.early");
  ASSERT_NE(patched_early, nullptr);
  EXPECT_EQ(patched_early->vaddr(), early->vaddr());
  ASSERT_EQ(patched_early->size(), early->size());
  EXPECT_EQ(std::memcmp(patched_early->data(), early->data(), early->size()), 0);
  const std::unique_ptr<Decoder> decoder = Decoder::create(kRdna2);
  ASSERT_NE(decoder, nullptr);
  uint32_t patched_branch = 0;
  std::memcpy(&patched_branch, patched_text.data() + 4, sizeof(patched_branch));
  DecodeResult decoded = decoder->decode_window(std::span(&patched_branch, 1), 0x304);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->mnemonic(), "s_branch");
  ASSERT_TRUE(decoded.value()->branch_offset_bytes());
  const std::optional<uint64_t> destination =
      add_signed_offset(decoded.value()->src_loc() + decoded.value()->size(),
                        *decoded.value()->branch_offset_bytes());
  ASSERT_TRUE(destination);
  EXPECT_EQ(*destination, 0x200u);
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(object.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(CheckedTextEdits, RejectsBranchWhoseVirtualAddressUnderflows) {
  // Unlike an address before .text, (0x308 + -195 * 4) = -4 cannot be a VA.
  const util::FailureOr<uint32_t> branch = build_s_branch(-195, ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_TRUE(branch.succeeded());
  const std::vector<uint8_t> image = make_branch_to_earlier_section_elf(branch.value());
  expect_reference_result(image, 0, 0, ROCJITSU_CODE_ARCH_RDNA2);
}

TEST(CheckedTextEdits, BranchReferencesCheckEachVirtualAddressAddition) {
  struct AddressCase {
    const char *name;
    uint64_t address_base;
    uint64_t source_offset;
    int16_t displacement_dwords;
    std::optional<uint64_t> destination;
  };
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t kHighBase = uint64_t{1} << 63;
  const std::array<AddressCase, 8> cases = {{
      {.name = "before section",
       .address_base = 0x300,
       .source_offset = 4,
       .displacement_dwords = -66,
       .destination = 0x200},
      {.name = "zero address",
       .address_base = 0x300,
       .source_offset = 4,
       .displacement_dwords = -194,
       .destination = 0},
      {.name = "negative destination",
       .address_base = 0x300,
       .source_offset = 4,
       .displacement_dwords = -195,
       .destination = std::nullopt},
      {.name = "base above signed range",
       .address_base = kHighBase,
       .source_offset = 4,
       .displacement_dwords = -3,
       .destination = kHighBase - 4},
      {.name = "upper aligned address",
       .address_base = kMax - 7,
       .source_offset = 0,
       .displacement_dwords = 0,
       .destination = kMax - 3},
      {.name = "destination overflow",
       .address_base = kMax - 7,
       .source_offset = 0,
       .displacement_dwords = 1,
       .destination = std::nullopt},
      {.name = "next PC overflow",
       .address_base = kMax - 3,
       .source_offset = 0,
       .displacement_dwords = 0,
       .destination = std::nullopt},
      {.name = "instruction address overflow",
       .address_base = kMax - 3,
       .source_offset = 4,
       .displacement_dwords = 0,
       .destination = std::nullopt},
  }};
  const std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_NE(decoder, nullptr);
  for (const AddressCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const util::FailureOr<uint32_t> branch =
        build_s_branch(test_case.displacement_dwords, ROCJITSU_CODE_ARCH_RDNA2);
    ASSERT_TRUE(branch.succeeded());
    const uint32_t word = branch.value();
    DecodeResult decoded = decoder->decode_window(std::span(&word, 1), test_case.source_offset);
    ASSERT_TRUE(decoded.succeeded());
    ASSERT_TRUE(decoded.value()->branch_offset_bytes());
    InstructionBoundaries boundaries;
    boundaries.record(*decoded.value(), test_case.address_base);
    ASSERT_EQ(boundaries.entries().size(), 1u);
    EXPECT_EQ(boundaries.entries().front().offset, test_case.source_offset);
    EXPECT_EQ(boundaries.entries().front().size, sizeof(uint32_t));
    EXPECT_EQ(boundaries.reference_overflow(), !test_case.destination.has_value());
    if (test_case.destination) {
      ASSERT_EQ(boundaries.reference_addresses().size(), 1u);
      EXPECT_EQ(boundaries.reference_addresses().front(), *test_case.destination);
    } else {
      EXPECT_TRUE(boundaries.reference_addresses().empty());
    }
  }
}

TEST(CheckedTextEdits, InstrumentorRejectsWholeBatchOnLaterInteriorReference) {
  for (const uint64_t reference : {12u, 16u}) {
    SCOPED_TRACE(reference);
    const std::vector<uint8_t> image =
        with_relative64_reference(make_mixed_width_kernel_elf(), reference);
    const std::vector<uint8_t> original_image = image;
    const AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    Instrumentor instrumentor(object, kArch);
    instrumentor.add_point_by_offset(0);
    instrumentor.add_point_by_offset(8);
    const Instrumentor::ValidationResult validated = instrumentor.validate_points();
    ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
    ASSERT_EQ(validated.sites.size(), 2u);
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    if (reference == 12) {
      // The first anchor is valid, but the reference enters the second anchor's
      // literal. A failed transaction must not publish either applied site.
      ASSERT_FALSE(result.errors.empty());
      ASSERT_TRUE(result.edit_error);
      EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
      EXPECT_EQ(result.edit_error->offset, reference);
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.sites.empty());
      EXPECT_TRUE(result.patches.empty());
    } else {
      // A reference at the second anchor's end permits both edits.
      ASSERT_TRUE(result.errors.empty()) << result.errors.front();
      EXPECT_FALSE(result.edit_error);
      ASSERT_FALSE(result.elf_bytes.empty());
      ASSERT_EQ(result.sites.size(), 2u);
      EXPECT_EQ(result.patches.size(), 2u);
      const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
      EXPECT_TRUE(patched.is_valid());
    }
    EXPECT_EQ(image, original_image);
    EXPECT_EQ(std::memcmp(object.image_data(), original_image.data(), original_image.size()), 0);
  }
}

TEST(CheckedTextEdits, InstrumentorChecksSymbolStartAndEndReferences) {
  for (const bool end_reference : {false, true}) {
    SCOPED_TRACE(end_reference);
    for (const uint64_t endpoint : {8u, 12u, 16u}) {
      SCOPED_TRACE(endpoint);
      const std::vector<uint8_t> image =
          with_text_symbol(end_reference ? 0 : endpoint, end_reference ? endpoint : 0, false);
      expect_reference_result(image, 8,
                              endpoint == 12 ? std::optional<uint64_t>(12) : std::nullopt);
    }
  }
}

TEST(CheckedTextEdits, RelocatableSymbolsKeepSectionOffsetsWithNonzeroSectionAddress) {
  for (const bool end_reference : {false, true}) {
    SCOPED_TRACE(end_reference);
    for (const uint64_t endpoint : {8u, 12u, 16u}) {
      SCOPED_TRACE(endpoint);
      const std::vector<uint8_t> image =
          with_text_symbol(end_reference ? 0 : endpoint, end_reference ? endpoint : 0, true);
      expect_reference_result(image, 8,
                              endpoint == 12 ? std::optional<uint64_t>(12) : std::nullopt);
    }
  }
}

TEST(CheckedTextEdits, InstrumentorChecksDescriptorEntryInsideAnchor) {
  constexpr uint64_t kEntry = 256;
  const util::FailureOr<uint32_t> nop = build_s_nop(0, kArch);
  ASSERT_TRUE(nop.succeeded());
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(kArch);
  ASSERT_TRUE(endpgm.succeeded());
  for (const uint64_t anchor : {248u, 252u, 256u}) {
    SCOPED_TRACE(anchor);
    std::vector<uint32_t> words(67, nop.value());
    words[anchor / sizeof(uint32_t)] = 0xbe8000ffu;
    words[anchor / sizeof(uint32_t) + 1] = 0x12345678u;
    words.back() = endpgm.value();
    const std::vector<uint8_t> image =
        test::make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1200, 0, 0, false, false,
                                     false, false, false, false, 0, kEntry);
    const AmdGpuCodeObject object(image.data(), image.size());
    const Section &text = *object.text_sections().front();
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(image, text.sectionOffset(), text.size());
    ASSERT_EQ(kernels.size(), 1u);
    ASSERT_EQ(kernels.front().entry_text_offset, kEntry);
    expect_reference_result(image, anchor,
                            anchor == 252 ? std::optional<uint64_t>(kEntry) : std::nullopt);
  }
}

TEST(CheckedTextEdits, UnknownMachineUsesTheSameCallerArchitectureForValidationAndPatching) {
  const util::FailureOr<uint32_t> nop = build_s_nop(0, kArch);
  ASSERT_TRUE(nop.succeeded());
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(kArch);
  ASSERT_TRUE(endpgm.succeeded());
  const std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf({nop.value(), endpgm.value()}, 0, 3, 0);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  ASSERT_EQ(object.target_id(), ROCJITSU_CODE_TARGET_INVALID);
  expect_reference_result(image, 0, std::nullopt);
}

TEST(CheckedTextEdits, ConcreteVariantDecoderIsUsedForSourceAndFinalAnchor) {
  constexpr rj_code_arch_t kCdna5 = ROCJITSU_CODE_ARCH_CDNA5;
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(kCdna5);
  ASSERT_TRUE(endpgm.succeeded());
  // Same public LLVM gfx1251 encoding as the probe mismatch regression:
  // v_pk_add_nc_u64 v[4:7], v[8:11], v[12:15]. It is unavailable on gfx1250.
  const std::vector<uint32_t> words = {0xcc4c4004u, 0x1a021908u, endpgm.value()};
  for (const uint32_t machine : {EF_AMDGPU_MACH_AMDGCN_GFX1250, EF_AMDGPU_MACH_AMDGCN_GFX1251}) {
    SCOPED_TRACE(machine);
    const std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, machine);
    if (machine == EF_AMDGPU_MACH_AMDGCN_GFX1251) {
      expect_reference_result(image, 0, std::nullopt, kCdna5);
    } else {
      const std::vector<uint8_t> original_image = image;
      const AmdGpuCodeObject object(image.data(), image.size());
      ASSERT_TRUE(object.is_valid());
      Instrumentor instrumentor(object, kCdna5);
      instrumentor.add_point_by_offset(0);
      const Instrumentor::ValidationResult validated = instrumentor.validate_points();
      EXPECT_TRUE(validated.sites.empty());
      EXPECT_FALSE(validated.errors.empty());
      const InstrumentedCodeObject result = instrumentor.patch();
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.sites.empty());
      EXPECT_FALSE(result.errors.empty());
      EXPECT_EQ(image, original_image);
      EXPECT_EQ(std::memcmp(object.image_data(), original_image.data(), original_image.size()), 0);
    }
  }
}

TEST(CheckedTextEdits, RejectsDebugBranchesWhoseDestinationsAreUnavailable) {
  constexpr rj_code_arch_t kCdna3 = ROCJITSU_CODE_ARCH_CDNA3;
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(kCdna3);
  ASSERT_TRUE(endpgm.succeeded());
  // Gfx942 SOPP encodings for the four conditional-debug branch forms. Their
  // displacement is one dword, targeting the literal at offset 8 of [4,12).
  constexpr std::array<uint32_t, 4> kDebugBranches = {0xbf970001u, 0xbf980001u, 0xbf990001u,
                                                      0xbf9a0001u};
  for (const uint32_t branch : kDebugBranches) {
    SCOPED_TRACE(branch);
    const std::vector<uint8_t> image =
        test::make_gfx942_kernel_elf({branch, 0xbe8000ffu, 0x12345678u, endpgm.value()}, 0);
    expect_reference_result(image, 4, 0, kCdna3);
  }
  // A normal branch to the anchor's boundary remains supported.
  const util::FailureOr<uint32_t> branch = build_s_branch(0, kCdna3);
  ASSERT_TRUE(branch.succeeded());
  const std::vector<uint8_t> control =
      test::make_gfx942_kernel_elf({branch.value(), 0xbe8000ffu, 0x12345678u, endpgm.value()}, 0);
  expect_reference_result(control, 4, std::nullopt, kCdna3);
}

TEST(CheckedTextEdits, RejectsSubvectorLoopsWhoseDestinationsAreUnavailable) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    SCOPED_TRACE(arch);
    const uint32_t machine = arch == ROCJITSU_CODE_ARCH_RDNA1 ? EF_AMDGPU_MACH_AMDGCN_GFX1010
                                                              : EF_AMDGPU_MACH_AMDGCN_GFX1030;
    const std::array<uint16_t, 2> opcodes =
        arch == ROCJITSU_CODE_ARCH_RDNA1
            ? std::array{rdna1::kSSubvectorLoopBeginSopk, rdna1::kSSubvectorLoopEndSopk}
            : std::array{rdna2::kSSubvectorLoopBeginSopk, rdna2::kSSubvectorLoopEndSopk};
    const uint16_t literal_operand = arch == ROCJITSU_CODE_ARCH_RDNA1
                                         ? static_cast<uint16_t>(rdna1::OPR_SRC_SRC_LITERAL)
                                         : static_cast<uint16_t>(rdna2::OPR_SRC_SRC_LITERAL);
    const util::FailureOr<uint32_t> move = build_s_mov_b32(0, literal_operand, arch);
    ASSERT_TRUE(move.succeeded());
    const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
    ASSERT_TRUE(endpgm.succeeded());
    for (const uint16_t opcode : opcodes) {
      SCOPED_TRACE(opcode);
      // Both loop forms have label operands but no displacement metadata. At
      // offset zero, simm16=1 refers to the literal at offset 8 of [4,12).
      const util::FailureOr<uint32_t> loop = build_sopk_encoding(arch, opcode, 0, 1);
      ASSERT_TRUE(loop.succeeded());
      const std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
          {loop.value(), move.value(), 0x12345678u, endpgm.value()}, 0, 3, machine);
      expect_reference_result(image, 4, 0, arch);
    }
    // Ordinary branches with known destinations at either boundary still pass.
    for (const int16_t displacement : {0, 2}) {
      SCOPED_TRACE(displacement);
      const util::FailureOr<uint32_t> branch = build_s_branch(displacement, arch);
      ASSERT_TRUE(branch.succeeded());
      const std::vector<uint8_t> control = test::make_amdgpu_kernel_elf(
          {branch.value(), move.value(), 0x12345678u, endpgm.value()}, 0, 3, machine);
      expect_reference_result(control, 4, std::nullopt, arch);
    }
  }
}

} // namespace
} // namespace rocjitsu
