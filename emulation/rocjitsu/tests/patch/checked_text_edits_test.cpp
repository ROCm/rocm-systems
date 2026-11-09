// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "../dbi_test_util.h"
#include "../elf_test_support.h"
#include "replacement_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
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
#include "util/result.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {
// Test requests keep fixture bytes readable; all admission and writing go through
// InstrumentationKind::ReplaceWithInst. This helper never validates or mutates a candidate.
using ReplacementEdit = test::ExpectedReplacement;

rj_code_arch_t source_architecture(const AmdGpuCodeObject &source) {
  Elf64_Ehdr header = {};
  if (source.image_size() < sizeof(header))
    return ROCJITSU_CODE_ARCH_INVALID;
  std::memcpy(&header, source.image_data(), sizeof(header));
  return arch_for_elf_mach(header.e_flags);
}

FailureOr<InstrumentedCodeObject> apply_replacements(Instrumentor &instrumentor,
                                                     std::span<const ReplacementEdit> edits,
                                                     TextEditError &error) {
  error = {};
  for (const ReplacementEdit &edit : edits) {
    InstrumentationPoint point = {
        .anchor_offset = edit.offset,
        .kind = InstrumentationKind::ReplaceWithInst,
        .probe_symbol = {},
        .probe_args = {},
        .replacement = InstructionReplacement{.expected_bytes = edit.expected,
                                              .replacement_bytes = edit.replacement}};
    instrumentor.add_point(std::move(point));
  }
  InstrumentedCodeObject result = instrumentor.patch();
  if (!result.errors.empty()) {
    EXPECT_TRUE(result.elf_bytes.empty());
    EXPECT_TRUE(result.sites.empty());
    if (result.edit_error)
      error = *result.edit_error;
    else
      error.message = result.errors.front();
    return Result::failure();
  }
  EXPECT_FALSE(result.edit_error.has_value());
  return result;
}

FailureOr<InstrumentedCodeObject> apply_replacements(const AmdGpuCodeObject &source,
                                                     std::span<const ReplacementEdit> edits,
                                                     TextEditError &error) {
  Instrumentor instrumentor(source, source_architecture(source));
  return apply_replacements(instrumentor, edits, error);
}

constexpr rj_code_target_id_t kTarget = ROCJITSU_CODE_TARGET_GFX1200;
constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
constexpr util::FailureOr<uint32_t> kNop0Encoding = build_s_nop(0, kArch);
constexpr util::FailureOr<uint32_t> kNop1Encoding = build_s_nop(1, kArch);
constexpr util::FailureOr<uint32_t> kEndpgmEncoding = build_s_endpgm(kArch);
static_assert(kNop0Encoding.succeeded() && kNop1Encoding.succeeded() &&
              kEndpgmEncoding.succeeded());
constexpr uint32_t kNop0Word = kNop0Encoding.value();
constexpr uint32_t kNop1Word = kNop1Encoding.value();
constexpr uint32_t kEndpgmWord = kEndpgmEncoding.value();
std::vector<uint8_t> make_instruction_bytes(std::initializer_list<uint32_t> words) {
  std::vector<uint8_t> result(words.size() * 4);
  if (!result.empty())
    std::memcpy(result.data(), words.begin(), result.size());
  return result;
}
// gfx1200 / RDNA4, with the kernel entry at .text offset 0. All offsets below
// are text-relative; the ELF builder's file layout is a separate coordinate system.
//
// Offset  Bytes  Instruction / encoded words
// 0x00      4   s_nop 1                         (selected)
// 0x04      4   s_nop 2                         (unchanged control)
// 0x08      8   s_mov_b32 s0, 0x12345678          (0xbe8000ff, literal 0x12345678)
// 0x10      4   s_nop 3                         (unchanged control)
// 0x14     12   global_store_b32 v0, v1, s[2:3] (0xee068002, 0x00800000, 0)
// 0x20      4   s_endpgm                        (unchanged control)
// Total: 0x24 / 36 bytes. The repository decoder and the pinned SDK's gfx1200
// disassembler agree on these extents and mnemonics. The final zero word of the
// global store belongs to that instruction; it is not independently skippable padding.
//
// make_mixed_width_nop_edits selects the nonadjacent extents at 0, 8 and 20,
// replacing one whole instruction with respectively 1, 2 and 3 four-byte NOPs.
// Controls at 0x04, 0x10 and 0x20 expose writes between or beyond those extents.
// This exercises encoding, complete extents and preservation of unselected text.
// It never executes the store or establishes GPU register/memory state for it.
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
std::vector<ReplacementEdit> make_mixed_width_nop_edits() {
  const uint32_t nop = kNop0Word;
  // Columns: text offset, expected instruction bytes, replacement NOP bytes.
  return {{.offset = 0,
           .expected = make_instruction_bytes({kNop1Word}),
           .replacement = make_instruction_bytes({nop})},
          {.offset = 8,
           .expected = make_instruction_bytes({0xbe8000ffu, 0x12345678u}),
           .replacement = make_instruction_bytes({nop, nop})},
          {.offset = 20,
           .expected = make_instruction_bytes({0xee068002u, 0x00800000u, 0}),
           .replacement = make_instruction_bytes({nop, nop, nop})}};
}

// Keep a second executable section loaded outside the indexed .text. The later
// placement has its own LOAD so text growth moves its target by 4096 bytes.
std::vector<uint8_t> make_branch_to_other_section_elf(uint32_t branch,
                                                      bool helper_after_text = false) {
  const uint64_t helper_address = helper_after_text ? 0x1400 : 0x200;
  const uint64_t helper_file_offset = helper_after_text ? 0x400 : 0x200;
  constexpr uint64_t kTextAddress = 0x300;
  constexpr util::FailureOr<uint32_t> kNop = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA2);
  constexpr util::FailureOr<uint32_t> kEndpgm = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA2);
  static_assert(kNop.succeeded() && kEndpgm.succeeded());
  const std::array<uint32_t, 3> words = {kNop.value(), branch, kEndpgm.value()};
  std::vector<uint8_t> names = {'\0'};
  const uint32_t early_name =
      test::add_elf_name(names, helper_after_text ? ".text.late" : ".text.early");
  const uint32_t text_name = test::add_elf_name(names, ".text");
  const uint32_t names_name = test::add_elf_name(names, ".shstrtab");
  const uint64_t names_offset =
      std::max(kTextAddress + sizeof(words), helper_file_offset + sizeof(uint32_t));
  const uint64_t section_offset = test::align_up_for_test(names_offset + names.size(), 8);
  std::array<Elf64_Shdr, 4> sections = {};
  sections[1] = {.sh_name = early_name,
                 .sh_type = SHT_PROGBITS,
                 .sh_flags = SHF_ALLOC | SHF_EXECINSTR,
                 .sh_addr = helper_address,
                 .sh_offset = helper_file_offset,
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
  header.e_phnum = helper_after_text ? 2 : 1;
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = sections.size();
  header.e_shstrndx = 3;
  const Elf64_Phdr segment = {
      .p_type = PT_LOAD,
      .p_flags = 0x5, // PF_R | PF_X
      .p_offset = helper_after_text ? kTextAddress : helper_file_offset,
      .p_vaddr = helper_after_text ? kTextAddress : helper_address,
      .p_paddr = helper_after_text ? kTextAddress : helper_address,
      .p_filesz = helper_after_text ? sizeof(words) : names_offset - helper_file_offset,
      .p_memsz = helper_after_text ? sizeof(words) : names_offset - helper_file_offset,
      .p_align = sizeof(uint32_t)};
  std::vector<uint8_t> image(section_offset + sizeof(sections), 0);
  test_support::write_elf_struct_for_test(image, 0, header);
  test_support::write_elf_struct_for_test(image, header.e_phoff, segment);
  if (helper_after_text) {
    const Elf64_Phdr helper = {.p_type = PT_LOAD,
                               .p_flags = 0x5,
                               .p_offset = helper_file_offset,
                               .p_vaddr = helper_address,
                               .p_paddr = helper_address,
                               .p_filesz = sizeof(uint32_t),
                               .p_memsz = sizeof(uint32_t),
                               .p_align = 4096};
    test_support::write_elf_struct_for_test(image, header.e_phoff + sizeof(segment), helper);
  }
  test_support::write_elf_struct_for_test(image, helper_file_offset, kEndpgm.value());
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

std::optional<std::vector<uint8_t>>
expected_replacement_image(const AmdGpuCodeObject &source, std::span<const ReplacementEdit> edits) {
  return test::expected_replacement_image(source, edits, source_architecture(source));
}

// Test-side exact-image oracle, independent of the production transaction.
testing::AssertionResult matches_checked_output(const AmdGpuCodeObject &source,
                                                std::span<const ReplacementEdit> edits,
                                                std::span<const uint8_t> candidate) {
  const std::optional<std::vector<uint8_t>> expected = expected_replacement_image(source, edits);
  if (!expected)
    return testing::AssertionFailure() << "test oracle: invalid expected replacement layout";
  const AmdGpuCodeObject parsed(candidate.data(), candidate.size());
  if (!parsed.is_valid() || parsed.target_id() != source.target_id() ||
      !std::ranges::equal(*expected, candidate))
    return testing::AssertionFailure() << "test oracle: unexpected output bytes or target";
  return testing::AssertionSuccess();
}

TEST(CheckedTextEdits, ReusableFactsKeepRequestChecksAndSourceIdentity) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject source(image.data(), image.size());
  TextEditError error;
  Instrumentor facts(source, source_architecture(source));
  const InstructionBoundaries *context = facts.instruction_boundaries(error);
  ASSERT_NE(context, nullptr) << error.message;
  EXPECT_EQ(facts.instruction_boundaries(error), context);
  const std::vector<ReplacementEdit> original_edits = make_mixed_width_nop_edits();
  for (int attempt = 0; attempt < 2; ++attempt) {
    FailureOr<std::unique_ptr<Instrumentor>> batch = facts.fork(error);
    ASSERT_TRUE(batch.succeeded()) << error.message;
    EXPECT_EQ(batch.value()->instruction_boundaries(error), context);
    const FailureOr<InstrumentedCodeObject> result =
        apply_replacements(*batch.value(), original_edits, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(matches_checked_output(source, original_edits, result.value().elf_bytes));
    std::vector<ReplacementEdit> stale = original_edits;
    stale.back().expected[0] ^= 1;
    FailureOr<std::unique_ptr<Instrumentor>> stale_batch = facts.fork(error);
    ASSERT_TRUE(stale_batch.succeeded()) << error.message;
    EXPECT_EQ(stale_batch.value()->instruction_boundaries(error), context);
    EXPECT_TRUE(apply_replacements(*stale_batch.value(), stale, error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::StaleBytes);
    std::vector<ReplacementEdit> invalid = original_edits;
    invalid.back().replacement.assign(12, 0xff);
    FailureOr<std::unique_ptr<Instrumentor>> invalid_batch = facts.fork(error);
    ASSERT_TRUE(invalid_batch.succeeded()) << error.message;
    EXPECT_EQ(invalid_batch.value()->instruction_boundaries(error), context);
    EXPECT_TRUE(apply_replacements(*invalid_batch.value(), invalid, error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::InvalidInstruction);
    std::vector<ReplacementEdit> overlapping = original_edits;
    overlapping.push_back(original_edits.front());
    FailureOr<std::unique_ptr<Instrumentor>> overlap_batch = facts.fork(error);
    ASSERT_TRUE(overlap_batch.succeeded()) << error.message;
    EXPECT_EQ(overlap_batch.value()->instruction_boundaries(error), context);
    EXPECT_TRUE(apply_replacements(*overlap_batch.value(), overlapping, error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::Overlap);
  }
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
}

TEST(CheckedTextEdits, CachedFullTextFactsIncludeUnreachableReferences) {
  const util::FailureOr<uint32_t> branch = build_s_branch(-3, kArch);
  ASSERT_TRUE(branch.succeeded());
  const std::vector<uint8_t> image =
      test::make_gfx1200_kernel_elf({0xbe8000ffu, 0x12345678u, kEndpgmWord, branch.value()}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  TextEditError error;
  Instrumentor facts(source, source_architecture(source));
  const InstructionBoundaries *context = facts.instruction_boundaries(error);
  ASSERT_NE(context, nullptr) << error.message;
  const uint32_t nop = kNop0Word;
  const ReplacementEdit edit = {.offset = 0,
                                .expected = make_instruction_bytes({0xbe8000ffu, 0x12345678u}),
                                .replacement = make_instruction_bytes({nop, nop})};
  for (int attempt = 0; attempt < 2; ++attempt) {
    FailureOr<std::unique_ptr<Instrumentor>> batch = facts.fork(error);
    ASSERT_TRUE(batch.succeeded()) << error.message;
    EXPECT_EQ(batch.value()->instruction_boundaries(error), context);
    EXPECT_TRUE(apply_replacements(*batch.value(), std::span(&edit, 1), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
  }
}

TEST(CheckedTextEdits, ThreeNonadjacentCompleteExtentsPreserveOtherTextAndDescribeTrampolines) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  const std::vector<ReplacementEdit> requests = make_mixed_width_nop_edits();
  TextEditError error;
  FailureOr<InstrumentedCodeObject> result = apply_replacements(object, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  const Section &text = *object.text_sections().front();
  const std::optional<std::vector<uint8_t>> expected = expected_replacement_image(object, requests);
  ASSERT_TRUE(expected);
  EXPECT_EQ(result.value().elf_bytes, *expected);
  const AmdGpuCodeObject candidate(result.value().elf_bytes.data(),
                                   result.value().elf_bytes.size());
  EXPECT_EQ(candidate.target_id(), kTarget);
  ASSERT_EQ(result.value().sites.size(), requests.size());
  uint64_t cave_offset = text.size();
  for (size_t index = 0; index < requests.size(); ++index) {
    const InstrumentedSite &site = result.value().sites[index];
    EXPECT_EQ(site.kind, InstrumentationKind::ReplaceWithInst);
    EXPECT_EQ(site.original_offset, requests[index].offset);
    EXPECT_EQ(site.original_size, requests[index].expected.size());
    EXPECT_EQ(site.instrumentation_offset, cave_offset);
    EXPECT_EQ(site.instrumentation_size, 0u);
    EXPECT_EQ(site.guest_offset, cave_offset);
    EXPECT_EQ(site.guest_size, requests[index].replacement.size());
    // Only the requested NOP sequence occupies this guest range. The original
    // literal instruction and store are never replayed in the trampoline.
    EXPECT_EQ(std::memcmp(candidate.text_sections().front()->data() + site.guest_offset,
                          requests[index].replacement.data(), site.guest_size),
              0);
    cave_offset += requests[index].replacement.size() + sizeof(uint32_t);
  }
  EXPECT_EQ(candidate.text_sections().front()->size(), cave_offset);
  for (const uint64_t control_offset : {4u, 16u, 32u})
    EXPECT_EQ(std::memcmp(candidate.text_sections().front()->data() + control_offset,
                          text.data() + control_offset, sizeof(uint32_t)),
              0);
  EXPECT_EQ(std::memcmp(object.image_data(), image.data(), image.size()), 0);
  EXPECT_TRUE(matches_checked_output(object, requests, result.value().elf_bytes));
}

TEST(CheckedTextEdits, TrampolinePayloadLengthIsIndependentOfCompleteSourceExtent) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject source(image.data(), image.size());
  const uint32_t nop = kNop0Word;
  const std::vector<std::vector<uint8_t>> bodies = {
      {}, make_instruction_bytes({nop}), make_instruction_bytes({nop, nop, nop, nop})};
  for (const ReplacementEdit &original : make_mixed_width_nop_edits()) {
    SCOPED_TRACE(original.offset);
    for (const std::vector<uint8_t> &body : bodies) {
      SCOPED_TRACE(body.size());
      const ReplacementEdit edit = {
          .offset = original.offset, .expected = original.expected, .replacement = body};
      TextEditError error;
      const FailureOr<InstrumentedCodeObject> result =
          apply_replacements(source, std::span(&edit, 1), error);
      ASSERT_TRUE(result.succeeded()) << error.message;
      ASSERT_EQ(result.value().sites.size(), 1u);
      const InstrumentedSite &site = result.value().sites.front();
      EXPECT_EQ(site.original_size, original.expected.size());
      EXPECT_EQ(site.guest_offset, source.text_sections().front()->size());
      EXPECT_EQ(site.guest_size, body.size());
      EXPECT_EQ(site.instrumentation_offset, site.guest_offset);
      EXPECT_EQ(site.instrumentation_size, 0u);
      EXPECT_TRUE(matches_checked_output(source, std::span(&edit, 1), result.value().elf_bytes));
      EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
    }
  }
}

TEST(CheckedTextEdits, TerminalReplacementHasNoReturnBranchOrOriginalReplay) {
  const uint32_t nop = kNop1Word;
  const uint32_t end = kEndpgmWord;
  const std::vector<uint8_t> image = test::make_gfx1200_kernel_elf({nop, end}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  for (const uint64_t offset : {0u, 4u}) {
    SCOPED_TRACE(offset);
    const ReplacementEdit edit = {.offset = offset,
                                  .expected = make_instruction_bytes({offset == 0 ? nop : end}),
                                  .replacement = make_instruction_bytes({kNop0Word, end})};
    Instrumentor instrumentor(source, kArch);
    instrumentor.add_point(
        {.anchor_offset = edit.offset,
         .kind = InstrumentationKind::ReplaceWithInst,
         .probe_symbol = {},
         .probe_args = {},
         .replacement = InstructionReplacement{.expected_bytes = edit.expected,
                                               .replacement_bytes = edit.replacement}});
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 1u);
    ASSERT_EQ(result.patches.size(), 1u);
    EXPECT_EQ(result.patches.front().original_bytes, edit.expected);
    EXPECT_FALSE(result.patches.front().return_target);
    EXPECT_TRUE(matches_checked_output(source, std::span(&edit, 1), result.elf_bytes));
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    EXPECT_EQ(candidate.text_sections().front()->size(),
              source.text_sections().front()->size() + edit.replacement.size());
    EXPECT_EQ(result.sites.front().guest_size, edit.replacement.size());
  }
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
}

TEST(CheckedTextEdits, NonterminalDeletionNeedsARealSourceContinuation) {
  const uint32_t nop = kNop0Word;
  const uint32_t end = kEndpgmWord;
  const std::vector<uint8_t> image = test::make_gfx1200_kernel_elf({nop, end}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  for (const std::vector<uint8_t> &body :
       std::vector<std::vector<uint8_t>>{{}, make_instruction_bytes({nop})}) {
    const ReplacementEdit edit = {
        .offset = 4, .expected = make_instruction_bytes({end}), .replacement = body};
    TextEditError error;
    EXPECT_TRUE(apply_replacements(source, std::span(&edit, 1), error).failed());
    EXPECT_FALSE(error.message.empty());
    EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
  }
}

TEST(CheckedTextEdits, ContinuationMustNameAnInstructionRatherThanZeroPadding) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  const util::FailureOr<uint32_t> maybe_end = build_s_endpgm(arch);
  ASSERT_TRUE(maybe_end.succeeded());
  const uint32_t end = maybe_end.value();
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  const std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf({end, 0, 0}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  for (const bool deletion : {false, true}) {
    SCOPED_TRACE(deletion);
    const ReplacementEdit edit = {.offset = 0,
                                  .expected = make_instruction_bytes({end}),
                                  .replacement = deletion ? std::vector<uint8_t>{}
                                                          : make_instruction_bytes({nop})};
    TextEditError error;
    EXPECT_TRUE(apply_replacements(source, std::span(&edit, 1), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::InvalidInstruction);
    EXPECT_EQ(error.offset, 0u);
    EXPECT_EQ(image, pristine);
  }
  const ReplacementEdit terminal = {.offset = 0,
                                    .expected = make_instruction_bytes({end}),
                                    .replacement = make_instruction_bytes({end})};
  TextEditError error;
  const FailureOr<InstrumentedCodeObject> terminated =
      apply_replacements(source, std::span(&terminal, 1), error);
  ASSERT_TRUE(terminated.succeeded()) << error.message;
  EXPECT_TRUE(
      matches_checked_output(source, std::span(&terminal, 1), terminated.value().elf_bytes));
  const std::vector<uint8_t> continued_image =
      test::make_amdgpu_kernel_elf({nop, end, 0, 0}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const AmdGpuCodeObject continued(continued_image.data(), continued_image.size());
  const ReplacementEdit normal = {
      .offset = 0, .expected = make_instruction_bytes({nop}), .replacement = {}};
  const FailureOr<InstrumentedCodeObject> result =
      apply_replacements(continued, std::span(&normal, 1), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(matches_checked_output(continued, std::span(&normal, 1), result.value().elf_bytes));
  EXPECT_EQ(image, pristine);
}

TEST(CheckedTextEdits, PcDependentReplacementPayloadsAreNotMovedWithoutRelocation) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4;
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(1, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, endpgm.value()}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  const std::array<util::FailureOr<uint32_t>, 5> payloads = {
      build_s_branch(0, arch), build_s_getpc_b64(0, arch), build_s_setpc_b64(30, arch),
      build_s_swappc_b64(30, 0, arch), build_s_call_b64(30, 0, arch)};
  for (const util::FailureOr<uint32_t> &payload : payloads) {
    ASSERT_TRUE(payload.succeeded());
    SCOPED_TRACE(payload.value());
    const ReplacementEdit edit = {.offset = 0,
                                  .expected = make_instruction_bytes({nop}),
                                  .replacement = make_instruction_bytes({payload.value()})};
    TextEditError error;
    EXPECT_TRUE(apply_replacements(source, std::span(&edit, 1), error).failed());
    EXPECT_FALSE(error.message.empty());
    EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
  }
}

TEST(CheckedTextEdits, SectionOnlyTransformationDoesNotRequireLoadMapping) {
  std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_phnum = 0;
  std::memcpy(image.data(), &header, sizeof(header));
  const AmdGpuCodeObject object(image.data(), image.size());
  TextEditError error;
  const std::vector<ReplacementEdit> requests = make_mixed_width_nop_edits();
  const FailureOr<InstrumentedCodeObject> result = apply_replacements(object, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(matches_checked_output(object, requests, result.value().elf_bytes));
}

struct PcDataLayout {
  bool before_text;
  bool section_backed;
  bool zero_fill;
};

std::vector<uint8_t> make_retained_pc_data_elf(const PcDataLayout &layout) {
  // The getpc/add/addc words are the gfx950 sequence emitted by Clang for
  // an indexed constant load. Use the existing descriptor payload as data.
  std::vector<uint32_t> words(16, 0xbf800000u);
  words[1] = 0xbe821c00u;
  words[2] = 0x8002ff02u;
  words[4] = 0x8203ff03u;
  words.back() = 0xbf810000u;
  std::vector<uint8_t> image = test::make_gfx950_kernel_elf(words, 0);
  const Elf64_Ehdr header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  std::vector<Elf64_Shdr> sections =
      test_support::read_elf_array_for_test<Elf64_Shdr>(image, header.e_shoff, header.e_shnum);
  if (layout.before_text) {
    sections[2].sh_addr -= 0x2000;
    Elf64_Phdr segment = test_support::read_elf_struct_for_test<Elf64_Phdr>(
        image, header.e_phoff + sizeof(Elf64_Phdr));
    segment.p_vaddr -= 0x2000;
    segment.p_paddr -= 0x2000;
    test_support::write_elf_struct_for_test(image, header.e_phoff + sizeof(segment), segment);
    amdhsa::kernel_descriptor_t descriptor =
        test_support::read_elf_struct_for_test<amdhsa::kernel_descriptor_t>(image,
                                                                            sections[2].sh_offset);
    descriptor.kernel_code_entry_byte_offset += 0x2000;
    test_support::write_elf_struct_for_test(image, sections[2].sh_offset, descriptor);
    const uint64_t symbol_offset = sections[3].sh_offset + sizeof(Elf64_Sym);
    Elf64_Sym symbol = test_support::read_elf_struct_for_test<Elf64_Sym>(image, symbol_offset);
    symbol.st_value = sections[2].sh_addr;
    test_support::write_elf_struct_for_test(image, symbol_offset, symbol);
  }
  if (!layout.section_backed)
    sections[2].sh_flags = 0;
  test_support::write_elf_struct_for_test(image, header.e_shoff + 2 * sizeof(Elf64_Shdr),
                                          sections[2]);
  uint64_t target = sections[2].sh_addr;
  if (layout.zero_fill) {
    Elf64_Phdr segment = test_support::read_elf_struct_for_test<Elf64_Phdr>(image, header.e_phoff);
    segment.p_memsz += 64;
    test_support::write_elf_struct_for_test(image, header.e_phoff, segment);
    target = sections[1].sh_addr + sections[1].sh_size + 16;
  }
  const uint64_t delta = target - (sections[1].sh_addr + 8);
  test_support::write_value_for_test(image, sections[1].sh_offset + 12,
                                     static_cast<uint32_t>(delta));
  test_support::write_value_for_test(image, sections[1].sh_offset + 20,
                                     static_cast<uint32_t>(delta >> 32));
  return image;
}

TEST(CheckedTextEdits, RetainedPcDataReferenceRequiresStableDataAddresses) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4;
  const std::array<PcDataLayout, 4> layouts = {{
      {.before_text = false, .section_backed = true, .zero_fill = false},
      {.before_text = true, .section_backed = true, .zero_fill = false},
      {.before_text = false, .section_backed = false, .zero_fill = false},
      {.before_text = true, .section_backed = true, .zero_fill = true},
  }};
  for (const PcDataLayout &layout : layouts) {
    SCOPED_TRACE(layout.before_text);
    SCOPED_TRACE(layout.section_backed);
    SCOPED_TRACE(layout.zero_fill);
    const bool stable = layout.before_text && !layout.zero_fill;
    const std::vector<uint8_t> image = make_retained_pc_data_elf(layout);
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    const Section &text = *source.text_sections().front();
    const Section *data = test_support::find_section(source, ".rodata");
    ASSERT_NE(data, nullptr);
    const uint64_t original_target =
        text.vaddr() + 8 +
        test_support::read_elf_struct_for_test<uint32_t>(image, text.sectionOffset() + 12) +
        (uint64_t{
             test_support::read_elf_struct_for_test<uint32_t>(image, text.sectionOffset() + 20)}
         << 32);
    ASSERT_EQ(original_target, layout.zero_fill ? text.vaddr() + text.size() + 16 : data->vaddr());
    for (int mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(mode);
      Instrumentor seed(source, arch);
      Instrumentor *batch = &seed;
      std::unique_ptr<Instrumentor> child;
      TextEditError error;
      if (mode != 0) {
        if (mode == 2) {
          seed.add_point_by_offset(0);
          const Instrumentor::ValidationResult observed = seed.validate_points();
          ASSERT_TRUE(observed.errors.empty()) << observed.errors.front();
        }
        ASSERT_NE(seed.instruction_boundaries(error), nullptr) << error.message;
        FailureOr<std::unique_ptr<Instrumentor>> fork = seed.fork(error);
        ASSERT_TRUE(fork.succeeded()) << error.message;
        child = std::move(fork).value();
        batch = child.get();
      }
      batch->add_point(
          {.anchor_offset = 0,
           .kind = InstrumentationKind::ReplaceWithInst,
           .probe_symbol = {},
           .probe_args = {},
           .replacement = InstructionReplacement{
               .expected_bytes = make_instruction_bytes({0xbf800000u}), .replacement_bytes = {}}});
      const Instrumentor::ValidationResult validation = batch->validate_points();
      const InstrumentedCodeObjectDebug result = batch->patch_with_debug_summaries();
      if (!stable) {
        EXPECT_FALSE(validation.errors.empty());
        EXPECT_TRUE(validation.sites.empty());
        EXPECT_FALSE(result.errors.empty());
        EXPECT_TRUE(result.elf_bytes.empty());
        EXPECT_TRUE(result.sites.empty());
        EXPECT_TRUE(result.patches.empty());
        if (!result.errors.empty()) {
          EXPECT_NE(result.errors.front().find("PC-relative"), std::string::npos);
        }
      }
      if (result.errors.empty()) {
        const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
        ASSERT_TRUE(candidate.is_valid());
        const Section *moved = test_support::find_section(candidate, ".rodata");
        ASSERT_NE(moved, nullptr);
        EXPECT_EQ(original_target, moved->vaddr()) << "retained getpc computes a stale data VA";
      } else {
        EXPECT_FALSE(stable) << result.errors.front();
      }
      EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
    }
  }
}

TEST(CheckedTextEdits, ReplacementGrowthChecksFinalFileBackedDynamicWrites) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  const util::FailureOr<uint32_t> nop = build_s_nop(0, arch);
  const util::FailureOr<uint32_t> end = build_s_endpgm(arch);
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(end.succeeded());
  std::vector<uint32_t> words(16, nop.value());
  words.back() = end.value();
  for (const bool collides : {false, true}) {
    SCOPED_TRACE(collides);
    std::vector<uint8_t> image = with_relative64_reference(
        test::make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250), std::nullopt);
    Elf64_Ehdr header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
    header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    test_support::write_elf_struct_for_test(image, 0, header);
    const std::vector<Elf64_Shdr> sections =
        test_support::read_elf_array_for_test<Elf64_Shdr>(image, header.e_shoff, header.e_shnum);
    Elf64_Rela relocation =
        test_support::read_elf_struct_for_test<Elf64_Rela>(image, sections.back().sh_offset);
    relocation.r_info = (uint64_t{1} << 32) | R_AMDGPU_ABS64;
    relocation.r_addend = 0;
    if (collides) {
      relocation.r_offset = sections[1].sh_addr + sections[1].sh_size;
      Elf64_Phdr text_load =
          test_support::read_elf_struct_for_test<Elf64_Phdr>(image, header.e_phoff);
      text_load.p_filesz += sizeof(uint64_t);
      text_load.p_memsz += sizeof(uint64_t);
      text_load.p_flags |= 0x2u; // PF_W
      test_support::write_elf_struct_for_test(image, header.e_phoff, text_load);
    }
    test_support::write_elf_struct_for_test(image, sections.back().sh_offset, relocation);
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    ASSERT_FALSE(CodeObjectPatcher(source).has_relocations_within_text());
    Instrumentor batch(source, arch);
    batch.add_point(
        {.anchor_offset = 0,
         .kind = InstrumentationKind::ReplaceWithInst,
         .probe_symbol = {},
         .probe_args = {},
         .replacement = InstructionReplacement{
             .expected_bytes = make_instruction_bytes({nop.value()}), .replacement_bytes = {}}});
    const Instrumentor::ValidationResult validation = batch.validate_points();
    ASSERT_TRUE(validation.errors.empty()) << validation.errors.front();
    const InstrumentedCodeObjectDebug result = batch.patch_with_debug_summaries();
    if (collides) {
      EXPECT_FALSE(result.errors.empty());
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.sites.empty());
      EXPECT_TRUE(result.patches.empty());
    } else {
      ASSERT_TRUE(result.errors.empty()) << result.errors.front();
      const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(candidate.is_valid());
      EXPECT_FALSE(CodeObjectPatcher(candidate).has_relocations_within_text());
    }
    EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
  }
}

TEST(CheckedTextEdits, Gfx1100EditsRetainSourceTarget) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA3;
  const util::FailureOr<uint32_t> maybe_original = build_s_nop(1, arch);
  ASSERT_TRUE(maybe_original.succeeded());
  const uint32_t original = maybe_original.value();
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  const std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf({original, endpgm.value()}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1100);
  const AmdGpuCodeObject source(image.data(), image.size());
  const std::vector<ReplacementEdit> requests = {{.offset = 0,
                                                  .expected = make_instruction_bytes({original}),
                                                  .replacement = make_instruction_bytes({nop})}};
  TextEditError error;
  const FailureOr<InstrumentedCodeObject> result = apply_replacements(source, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_NE(source.target_id(), ROCJITSU_CODE_TARGET_INVALID);
  const AmdGpuCodeObject candidate(result.value().elf_bytes.data(),
                                   result.value().elf_bytes.size());
  EXPECT_EQ(candidate.target_id(), source.target_id());
  EXPECT_TRUE(matches_checked_output(source, requests, result.value().elf_bytes));
}

TEST(CheckedTextEdits, LegacyElfIdentityIsSourceDerivedAndUnchanged) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA1;
  const util::FailureOr<uint32_t> original = build_s_nop(1, arch);
  ASSERT_TRUE(original.succeeded());
  const util::FailureOr<uint32_t> replacement = build_s_nop(0, arch);
  ASSERT_TRUE(replacement.succeeded());
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf({original.value(), endpgm.value()}, 0,
                                                            3, EF_AMDGPU_MACH_AMDGCN_GFX908);
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_EQ(source.target_id(), ROCJITSU_CODE_TARGET_INVALID);
  const std::vector<ReplacementEdit> requests = {
      {.offset = 0,
       .expected = make_instruction_bytes({original.value()}),
       .replacement = make_instruction_bytes({replacement.value()})}};
  TextEditError error;
  const FailureOr<InstrumentedCodeObject> result = apply_replacements(source, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  const AmdGpuCodeObject candidate(result.value().elf_bytes.data(),
                                   result.value().elf_bytes.size());
  EXPECT_EQ(candidate.target_id(), ROCJITSU_CODE_TARGET_INVALID);
  const std::vector<uint8_t> expected = result.value().elf_bytes;
  EXPECT_TRUE(matches_checked_output(source, requests, expected));
  for (const uint32_t machine :
       {EF_AMDGPU_MACH_NONE, uint32_t{0xfe}, EF_AMDGPU_MACH_AMDGCN_GFX1010}) {
    std::vector<uint8_t> corrupt = expected;
    Elf64_Ehdr header = {};
    std::memcpy(&header, corrupt.data(), sizeof(header));
    header.e_flags = machine;
    std::memcpy(corrupt.data(), &header, sizeof(header));
    EXPECT_FALSE(matches_checked_output(source, requests, corrupt));
    if (machine != EF_AMDGPU_MACH_AMDGCN_GFX1010) {
      const AmdGpuCodeObject unknown(corrupt.data(), corrupt.size());
      EXPECT_TRUE(apply_replacements(unknown, requests, error).failed());
      EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedTarget);
    }
  }
}

TEST(CheckedTextEdits, LateFailuresReturnNoCandidateAndLeaveSourceUnchanged) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  for (int mode = 0; mode < 8; ++mode) {
    SCOPED_TRACE(mode);
    std::vector<ReplacementEdit> requests = make_mixed_width_nop_edits();
    switch (mode) {
    case 0:
      requests.back().expected[0] ^= 1;
      break;
    case 1:
      requests.back().offset = std::numeric_limits<uint64_t>::max() - 3;
      break;
    case 2:
      requests.push_back(requests.front());
      break;
    case 3:
      // Empty replacement is now deletion; missing expected source bytes is
      // still invalid and must reject the entire batch.
      requests.back().expected.clear();
      break;
    case 4:
      // A shorter complete sequence is valid. A partial dword is not.
      requests.back().replacement.resize(11);
      break;
    case 5:
      requests.back().offset = 24;
      break;
    case 6:
      requests.back().replacement = make_instruction_bytes({0xffffffffu, 0xffffffffu, 0xffffffffu});
      break;
    case 7:
      requests.back().offset = 21;
      break;
    }
    TextEditError error;
    EXPECT_TRUE(apply_replacements(object, requests, error).failed());
    EXPECT_NE(error.code, TextEditErrorCode::None);
    EXPECT_EQ(std::memcmp(object.image_data(), image.data(), image.size()), 0);
  }
}

TEST(CheckedTextEdits, RejectsInteriorLiteralAndPartialInstruction) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  // Columns: text offset, expected bytes, replacement bytes; each row is invalid.
  for (const ReplacementEdit &edit : std::vector<ReplacementEdit>{
           {.offset = 12,
            .expected = make_instruction_bytes({0x12345678u}),
            .replacement = make_instruction_bytes({kNop0Word})},
           {.offset = 8,
            .expected = make_instruction_bytes({0xbe8000ffu}),
            .replacement = make_instruction_bytes({kNop0Word})},
           {.offset = 20,
            .expected = make_instruction_bytes({0xee068002u, 0x00800000u}),
            .replacement = make_instruction_bytes({kNop0Word, kNop0Word})}}) {
    TextEditError error;
    EXPECT_TRUE(apply_replacements(object, std::span(&edit, 1), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::InvalidInstruction) << error.message;
  }
}

TEST(CheckedTextEdits, RejectsTruncatedOriginalEightAndTwelveByteInstructions) {
  for (const std::vector<uint32_t> &words :
       std::vector<std::vector<uint32_t>>{{0xbe8000ffu}, {0xee068002u, 0x00800000u}}) {
    const std::vector<uint8_t> image = test::make_gfx1200_kernel_elf(words, 0);
    const AmdGpuCodeObject object(image.data(), image.size());
    ReplacementEdit edit = {.offset = 0,
                            .expected = std::vector<uint8_t>(words.size() * 4),
                            .replacement = std::vector<uint8_t>(words.size() * 4)};
    std::memcpy(edit.expected.data(), words.data(), edit.expected.size());
    for (size_t offset = 0; offset < edit.replacement.size(); offset += 4) {
      const uint32_t nop = kNop0Word;
      std::memcpy(edit.replacement.data() + offset, &nop, 4);
    }
    TextEditError error;
    EXPECT_TRUE(apply_replacements(object, std::span(&edit, 1), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::InvalidInstruction) << error.message;
  }
}

TEST(CheckedTextEdits, EmitterOracleRejectsTextMetadataAndTargetCorruption) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  const std::vector<ReplacementEdit> requests = make_mixed_width_nop_edits();
  TextEditError error;
  FailureOr<InstrumentedCodeObject> result = apply_replacements(object, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  const AmdGpuCodeObject output(result.value().elf_bytes.data(), result.value().elf_bytes.size());
  ASSERT_TRUE(output.is_valid());
  const Section &text = *output.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
      result.value().elf_bytes, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_EQ(kernels.size(), 1u);
  ASSERT_FALSE(result.value().sites.empty());
  for (const uint64_t offset :
       {text.sectionOffset() + 4, text.sectionOffset(), uint64_t{offsetof(Elf64_Ehdr, e_flags)},
        kernels.front().descriptor_file_offset,
        text.sectionOffset() + result.value().sites.front().guest_offset}) {
    std::vector<uint8_t> candidate = result.value().elf_bytes;
    candidate[offset] ^= 1;
    EXPECT_FALSE(matches_checked_output(object, requests, candidate)) << offset;
  }
  std::vector<uint8_t> unknown = image;
  Elf64_Ehdr header = {};
  std::memcpy(&header, unknown.data(), sizeof(header));
  header.e_flags = EF_AMDGPU_MACH_NONE;
  std::memcpy(unknown.data(), &header, sizeof(header));
  const AmdGpuCodeObject unsupported(unknown.data(), unknown.size());
  EXPECT_TRUE(apply_replacements(unsupported, requests, error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedTarget);
  EXPECT_TRUE(apply_replacements(object, {}, error).failed());
}

TEST(CheckedTextEdits, ClauseAndTwelveByteTrampolinePoliciesStaySeparate) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  Instrumentor instrumentor(object, kArch);
  instrumentor.add_point_by_offset(20, InstrumentationKind::BeforeInst);
  EXPECT_TRUE(instrumentor.patch().elf_bytes.empty());

  // s_clause 0 followed by a complete global store. The transaction cannot
  // silently change the clause's instruction count or memory operation kinds.
  const std::vector<uint8_t> clause_image =
      test::make_gfx1200_kernel_elf({0xbf850000u, 0xee068002u, 0x00800000u, 0, kEndpgmWord}, 0);
  const AmdGpuCodeObject clause_object(clause_image.data(), clause_image.size());
  ReplacementEdit edit = make_mixed_width_nop_edits().back();
  edit.offset = 4;
  TextEditError error;
  EXPECT_TRUE(apply_replacements(clause_object, std::span(&edit, 1), error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::Clause) << error.message;
}

TEST(CheckedTextEdits, RejectsRelocationWritesAndInteriorSymbolEndpoints) {
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    std::vector<uint8_t> image = make_mixed_width_kernel_elf();
    const AmdGpuCodeObject original(image.data(), image.size());
    const CodeObjectPatcher patcher(original);
    const std::vector<Elf64_Shdr> sections = patcher.section_headers();
    const Section &text = *original.text_sections().front();
    Elf64_Ehdr header = {};
    std::memcpy(&header, image.data(), sizeof(header));
    for (size_t index = 0; index < sections.size(); ++index) {
      const Elf64_Shdr &section = sections[index];
      if (section.sh_type != SHT_SYMTAB)
        continue;
      if (mode == 0) {
        // Reuse the allocated descriptor storage for a relocation table.
        const std::vector<Elf64_Shdr>::const_iterator data =
            std::ranges::find_if(sections, [](const Elf64_Shdr &candidate) {
              return candidate.sh_type == SHT_PROGBITS && (candidate.sh_flags & SHF_ALLOC) != 0 &&
                     (candidate.sh_flags & SHF_EXECINSTR) == 0;
            });
        ASSERT_NE(data, sections.end());
        const size_t data_index = static_cast<size_t>(data - sections.begin());
        Elf64_Shdr relocation = *data;
        relocation.sh_type = SHT_RELA;
        relocation.sh_info = *text.sectionHeaderIndex();
        relocation.sh_link = index;
        relocation.sh_entsize = sizeof(Elf64_Rela);
        relocation.sh_size = sizeof(Elf64_Rela);
        Elf64_Rela record = {};
        record.r_offset = text.vaddr() + 20;
        record.r_info = R_AMDGPU_ABS64;
        std::memcpy(image.data() + relocation.sh_offset, &record, sizeof(record));
        std::memcpy(image.data() + header.e_shoff + data_index * header.e_shentsize, &relocation,
                    sizeof(relocation));
      } else {
        Elf64_Sym symbol = {};
        std::memcpy(&symbol, image.data() + section.sh_offset + sizeof(symbol), sizeof(symbol));
        symbol.st_shndx = *text.sectionHeaderIndex();
        symbol.st_info = elf_symbol_info(kElfSymbolBindLocal, kElfSymbolTypeFunc);
        symbol.st_value = text.vaddr() + (mode == 1 ? 12 : 8);
        symbol.st_size = mode == 1 ? 0 : (mode == 2 ? 4 : 8);
        std::memcpy(image.data() + section.sh_offset + sizeof(symbol), &symbol, sizeof(symbol));
      }
    }
    const AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    TextEditError error;
    FailureOr<InstrumentedCodeObject> result =
        apply_replacements(object, make_mixed_width_nop_edits(), error);
    if (mode == 3) {
      EXPECT_TRUE(result.succeeded()) << error.message;
    } else {
      EXPECT_TRUE(result.failed());
      EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
    }
  }
}

TEST(CheckedTextEdits, RejectsRelativeReferencesInsideInstructionsAtomically) {
  for (const uint64_t target_offset : {12u, 13u, 24u, 28u}) {
    SCOPED_TRACE(target_offset);
    const std::vector<uint8_t> image =
        with_relative64_reference(make_mixed_width_kernel_elf(), target_offset);
    const AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    const CodeObjectPatcher patcher(object);
    ASSERT_FALSE(patcher.has_relocations_within_text());
    ASSERT_FALSE(patcher.has_malformed_rocr_relocation_section());
    ASSERT_FALSE(patcher.has_rocr_rejected_none_relocation());
    ASSERT_FALSE(patcher.has_unsupported_relocation_to_text());
    TextEditError error;
    Instrumentor facts(object, source_architecture(object));
    const InstructionBoundaries *context = facts.instruction_boundaries(error);
    ASSERT_NE(context, nullptr) << error.message;
    const std::vector<uint8_t> original_image = image;
    const std::vector<ReplacementEdit> edits = make_mixed_width_nop_edits();
    EXPECT_TRUE(apply_replacements(object, edits, error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
    EXPECT_EQ(error.offset, target_offset);
    EXPECT_EQ(image, original_image);
    EXPECT_EQ(std::memcmp(object.image_data(), original_image.data(), original_image.size()), 0);

    // The unrelated first edit remains usable after the rejected transaction.
    const std::span<const ReplacementEdit> first(edits.data(), 1);
    const FailureOr<InstrumentedCodeObject> result = apply_replacements(object, first, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(matches_checked_output(object, first, result.value().elf_bytes));
  }
}

TEST(CheckedTextEdits, RejectsRelativeReferenceBeforePartialRelocationTail) {
  // Shared admission accepts the complete record before this incomplete tail.
  const std::vector<uint8_t> image =
      with_relative64_reference(make_mixed_width_kernel_elf(), 12, false, true);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  const CodeObjectPatcher patcher(object);
  ASSERT_FALSE(patcher.has_relocations_within_text());
  ASSERT_FALSE(patcher.has_malformed_rocr_relocation_section());
  ASSERT_FALSE(patcher.has_rocr_rejected_none_relocation());
  ASSERT_FALSE(patcher.has_unsupported_relocation_to_text());
  TextEditError error;
  Instrumentor facts(object, source_architecture(object));
  const InstructionBoundaries *context = facts.instruction_boundaries(error);
  ASSERT_NE(context, nullptr) << error.message;
  const std::vector<ReplacementEdit> edits = make_mixed_width_nop_edits();
  EXPECT_TRUE(apply_replacements(object, edits, error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
  EXPECT_EQ(error.offset, 12u);
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
    TextEditError error;
    Instrumentor facts(object, source_architecture(object));
    const InstructionBoundaries *context = facts.instruction_boundaries(error);
    EXPECT_EQ(context, nullptr);
    EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
    const ReplacementEdit edit = make_mixed_width_nop_edits()[1];
    EXPECT_TRUE(apply_replacements(object, std::span(&edit, 1), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
    EXPECT_EQ(std::memcmp(object.image_data(), image.data(), image.size()), 0);
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
      TextEditError error;
      const ReplacementEdit edit = make_mixed_width_nop_edits()[1];
      const FailureOr<InstrumentedCodeObject> result =
          apply_replacements(object, std::span(&edit, 1), error);
      ASSERT_TRUE(result.succeeded()) << error.message;
      EXPECT_TRUE(matches_checked_output(object, std::span(&edit, 1), result.value().elf_bytes));
    }
  }
}

TEST(CheckedTextEdits, NonDynamicRelativeReferencesFollowSharedAllocationPolicy) {
  constexpr Elf_Half kEtExec = 2;
  for (const Elf_Half elf_type : {ET_REL, kEtExec}) {
    SCOPED_TRACE(elf_type);
    for (const bool nonallocated_metadata : {false, true}) {
      SCOPED_TRACE(nonallocated_metadata);
      std::vector<uint8_t> image =
          with_relative64_reference(make_mixed_width_kernel_elf(), 12, nonallocated_metadata);
      const AmdGpuCodeObject original(image.data(), image.size());
      ASSERT_TRUE(original.is_valid());
      std::vector<Elf64_Shdr> sections = CodeObjectPatcher(original).section_headers();
      Elf64_Shdr &relocs = sections.back();
      Elf64_Rela relocation = {};
      std::memcpy(&relocation, image.data() + relocs.sh_offset, sizeof(relocation));
      if (!nonallocated_metadata) {
        const std::vector<Elf64_Shdr>::const_iterator data =
            std::ranges::find_if(sections, [&](const Elf64_Shdr &section) {
              return (section.sh_flags & SHF_ALLOC) != 0 &&
                     relocation.r_offset >= section.sh_addr &&
                     relocation.r_offset - section.sh_addr < section.sh_size;
            });
        ASSERT_NE(data, sections.end());
        relocs.sh_info = static_cast<uint32_t>(data - sections.begin());
      }
      // ET_REL places are relative to sh_info; ET_EXEC keeps the virtual place.
      if (elf_type == ET_REL)
        relocation.r_offset = 0;
      Elf64_Ehdr header = {};
      std::memcpy(&header, image.data(), sizeof(header));
      header.e_type = elf_type;
      std::memcpy(image.data(), &header, sizeof(header));
      std::memcpy(image.data() + header.e_shoff, sections.data(),
                  sections.size() * sizeof(Elf64_Shdr));
      std::memcpy(image.data() + relocs.sh_offset, &relocation, sizeof(relocation));

      const AmdGpuCodeObject object(image.data(), image.size());
      ASSERT_TRUE(object.is_valid());
      const CodeObjectPatcher patcher(object);
      ASSERT_FALSE(patcher.has_relocations_within_text());
      ASSERT_FALSE(patcher.has_unsupported_relocation_to_text());
      TextEditError error;
      Instrumentor facts(object, source_architecture(object));
      const InstructionBoundaries *context = facts.instruction_boundaries(error);
      ASSERT_NE(context, nullptr) << error.message;
      const std::vector<uint8_t> original_image = image;
      const std::vector<ReplacementEdit> edits = make_mixed_width_nop_edits();
      const FailureOr<InstrumentedCodeObject> result = apply_replacements(object, edits, error);
      if (nonallocated_metadata) {
        ASSERT_TRUE(result.succeeded()) << error.message;
        EXPECT_TRUE(matches_checked_output(object, edits, result.value().elf_bytes));
      } else {
        EXPECT_TRUE(result.failed());
        EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
        EXPECT_EQ(error.offset, 12u);
      }
      EXPECT_EQ(image, original_image);
      EXPECT_EQ(std::memcmp(object.image_data(), original_image.data(), original_image.size()), 0);
      // An interior reference does not prohibit unrelated edits to the same source.
      const std::span<const ReplacementEdit> first(edits.data(), 1);
      const FailureOr<InstrumentedCodeObject> unrelated = apply_replacements(object, first, error);
      ASSERT_TRUE(unrelated.succeeded()) << error.message;
      EXPECT_TRUE(matches_checked_output(object, first, unrelated.value().elf_bytes));
    }
  }
}

TEST(CheckedTextEdits, PreservesRelativeReferencesAtRetainedBoundaries) {
  for (const uint64_t target_offset : {0u, 4u, 8u, 16u, 20u, 32u, 36u}) {
    SCOPED_TRACE(target_offset);
    const std::vector<uint8_t> image =
        with_relative64_reference(make_mixed_width_kernel_elf(), target_offset);
    const AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    TextEditError error;
    const std::vector<ReplacementEdit> edits = make_mixed_width_nop_edits();
    const FailureOr<InstrumentedCodeObject> result = apply_replacements(object, edits, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(matches_checked_output(object, edits, result.value().elf_bytes));
  }
}

TEST(CheckedTextEdits, PreservesUnrelatedDataAndNonallocatedRelativeReferences) {
  for (const bool nonallocated_metadata : {false, true}) {
    SCOPED_TRACE(nonallocated_metadata);
    const std::vector<uint8_t> image = with_relative64_reference(
        make_mixed_width_kernel_elf(),
        nonallocated_metadata ? std::optional<uint64_t>(12) : std::nullopt, nonallocated_metadata);
    const AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    TextEditError error;
    const std::vector<ReplacementEdit> edits = make_mixed_width_nop_edits();
    const FailureOr<InstrumentedCodeObject> result = apply_replacements(object, edits, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(matches_checked_output(object, edits, result.value().elf_bytes));
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

TEST(CheckedTextEdits, PreservesAdditionalExecutableSection) {
  std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject original(image.data(), image.size());
  ASSERT_TRUE(original.is_valid());
  const std::vector<Elf64_Shdr> sections = CodeObjectPatcher(original).section_headers();
  const std::vector<Elf64_Shdr>::const_iterator data =
      std::ranges::find_if(sections, [](const Elf64_Shdr &candidate) {
        return candidate.sh_type == SHT_PROGBITS && (candidate.sh_flags & SHF_ALLOC) != 0 &&
               (candidate.sh_flags & SHF_EXECINSTR) == 0;
      });
  ASSERT_NE(data, sections.end());
  const size_t data_index = static_cast<size_t>(data - sections.begin());
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  Elf64_Shdr section = *data;
  const size_t offset = header.e_shoff + data_index * header.e_shentsize;
  section.sh_flags |= SHF_EXECINSTR;
  std::memcpy(image.data() + offset, &section, sizeof(section));
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_EQ(object.text_sections().size(), 1u);
  ASSERT_EQ(object.allocated_executable_sections().size(), 2u);
  const Section &source_text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> source_kernels = scan_kernel_descriptors(
      image, source_text.sectionOffset(), source_text.size(), source_text.sectionHeaderIndex());
  ASSERT_EQ(source_kernels.size(), 1u);
  ASSERT_GE(source_kernels.front().descriptor_file_offset, section.sh_offset);
  const uint64_t descriptor_within_section =
      source_kernels.front().descriptor_file_offset - section.sh_offset;
  ASSERT_LE(descriptor_within_section + sizeof(amdhsa::kernel_descriptor_t), section.sh_size);
  TextEditError error;
  const std::vector<ReplacementEdit> requests = make_mixed_width_nop_edits();
  const FailureOr<InstrumentedCodeObject> result = apply_replacements(object, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  std::vector<uint8_t> expected = result.value().elf_bytes;
  EXPECT_TRUE(matches_checked_output(object, requests, expected));
  // Section growth can move the executable sibling in the ELF file. Corrupt
  // its actual output bytes rather than the old file position now holding cave.
  const AmdGpuCodeObject output(expected.data(), expected.size());
  const std::vector<Elf64_Shdr> output_sections = CodeObjectPatcher(output).section_headers();
  ASSERT_LT(data_index, output_sections.size());
  const Elf64_Shdr &output_sibling = output_sections[data_index];
  ASSERT_EQ(output_sibling.sh_size, section.sh_size);
  ASSERT_LE(output_sibling.sh_offset + output_sibling.sh_size, expected.size());
  EXPECT_EQ(output_sibling.sh_flags, section.sh_flags);
  // The sibling contains the kernel descriptor. Growing .text moves that
  // descriptor, so only its entry displacement changes within the section.
  std::vector<uint8_t> expected_sibling(image.begin() + section.sh_offset,
                                        image.begin() + section.sh_offset + section.sh_size);
  const Section &output_text = *output.text_sections().front();
  const uint64_t output_descriptor_address = output_sibling.sh_addr + descriptor_within_section;
  ASSERT_LE(output_text.vaddr(), static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
                                     source_kernels.front().entry_text_offset);
  ASSERT_LE(output_descriptor_address, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  const int64_t entry_displacement =
      static_cast<int64_t>(output_text.vaddr() + source_kernels.front().entry_text_offset) -
      static_cast<int64_t>(output_descriptor_address);
  std::memcpy(expected_sibling.data() + descriptor_within_section +
                  offsetof(amdhsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
              &entry_displacement, sizeof(entry_displacement));
  EXPECT_EQ(std::memcmp(expected.data() + output_sibling.sh_offset, expected_sibling.data(),
                        expected_sibling.size()),
            0);
  const std::vector<KernelDescriptorInfo> output_kernels = scan_kernel_descriptors(
      expected, output_text.sectionOffset(), output_text.size(), output_text.sectionHeaderIndex());
  ASSERT_EQ(output_kernels.size(), source_kernels.size());
  EXPECT_EQ(output_kernels.front().entry_text_offset, source_kernels.front().entry_text_offset);
  EXPECT_EQ(output_kernels.front().descriptor_file_offset,
            output_sibling.sh_offset + descriptor_within_section);
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(object.image_data(), pristine.data(), pristine.size()), 0);
  expected[output_sibling.sh_offset] ^= 1;
  EXPECT_FALSE(matches_checked_output(object, requests, expected));
}

TEST(CheckedTextEdits, RejectsAliasedSections) {
  for (const bool file_alias : {false, true}) {
    std::vector<uint8_t> image = make_mixed_width_kernel_elf();
    const AmdGpuCodeObject original(image.data(), image.size());
    const CodeObjectPatcher patcher(original);
    const std::vector<Elf64_Shdr> sections = patcher.section_headers();
    const std::vector<Elf64_Shdr>::const_iterator data =
        std::ranges::find_if(sections, [](const Elf64_Shdr &candidate) {
          return candidate.sh_type == SHT_PROGBITS && (candidate.sh_flags & SHF_ALLOC) != 0 &&
                 (candidate.sh_flags & SHF_EXECINSTR) == 0;
        });
    ASSERT_NE(data, sections.end());
    const size_t data_index = static_cast<size_t>(data - sections.begin());
    Elf64_Shdr section = *data;
    section.sh_flags |= SHF_EXECINSTR;
    if (file_alias) {
      section.sh_offset = original.text_sections().front()->sectionOffset();
      section.sh_size = original.text_sections().front()->size();
    } else {
      section.sh_addr = original.text_sections().front()->vaddr();
    }
    Elf64_Ehdr header = {};
    std::memcpy(&header, image.data(), sizeof(header));
    std::memcpy(image.data() + header.e_shoff + data_index * header.e_shentsize, &section,
                sizeof(section));
    const AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    TextEditError error;
    EXPECT_TRUE(apply_replacements(object, make_mixed_width_nop_edits(), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::InvalidImage) << error.message;
    EXPECT_NE(error.message.find(file_alias ? "aliases another section" : "virtual addresses"),
              std::string::npos);
  }
}

struct LoadAliasFixture {
  std::vector<uint8_t> image;
  Elf64_Phdr alias;
};

std::optional<LoadAliasFixture> make_load_alias_fixture(std::optional<uint64_t> text_offset,
                                                        uint64_t file_size, uint64_t memory_size,
                                                        bool same_mapping = false) {
  std::vector<uint8_t> image =
      test::make_gfx1200_kernel_elf({kNop0Word, kNop1Word, kNop0Word, kEndpgmWord}, 0);
  const AmdGpuCodeObject original(image.data(), image.size());
  EXPECT_TRUE(original.is_valid());
  if (!original.is_valid() || original.text_sections().size() != 1)
    return std::nullopt;
  const Section &text = *original.text_sections().front();
  EXPECT_TRUE(!same_mapping || text_offset.has_value());
  if (same_mapping && !text_offset)
    return std::nullopt;
  const uint64_t file_offset = text_offset ? text.sectionOffset() + *text_offset : 0;
  EXPECT_LE(file_offset, image.size());
  EXPECT_LE(file_size, memory_size);
  if (file_offset > image.size() || file_size > memory_size)
    return std::nullopt;
  EXPECT_LE(file_size, image.size() - file_offset);
  if (file_size > image.size() - file_offset)
    return std::nullopt;
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_VERSION] = 1;
  header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  EXPECT_TRUE(segments);
  if (!segments)
    return std::nullopt;
  EXPECT_EQ(segments->size(), 2u);
  EXPECT_EQ(header.e_phentsize, sizeof(Elf64_Phdr));
  const uint64_t new_table_end = header.e_phoff + (segments->size() + 1) * sizeof(Elf64_Phdr);
  EXPECT_LE(new_table_end, text.sectionOffset());
  if (segments->size() != 2 || header.e_phentsize != sizeof(Elf64_Phdr) ||
      new_table_end > text.sectionOffset())
    return std::nullopt;
  const Elf64_Phdr &owner = segments->front();
  EXPECT_EQ(owner.p_type, PT_LOAD);
  EXPECT_NE(owner.p_flags & PF_X, 0u);
  EXPECT_EQ(owner.p_offset, text.sectionOffset());
  EXPECT_EQ(owner.p_vaddr, text.vaddr());
  EXPECT_GE(owner.p_filesz, text.size());
  constexpr uint64_t kAlignment = 0x1000;
  const uint64_t address =
      same_mapping ? text.vaddr() + *text_offset : 0x9000 + file_offset % kAlignment;
  const Elf64_Phdr alias = {.p_type = PT_LOAD,
                            .p_flags = 0x4, // PF_R
                            .p_offset = file_offset,
                            .p_vaddr = address,
                            .p_paddr = address,
                            .p_filesz = file_size,
                            .p_memsz = memory_size,
                            .p_align = kAlignment};
  EXPECT_EQ(alias.p_offset % alias.p_align, alias.p_vaddr % alias.p_align);
  if (!same_mapping) {
    EXPECT_GE(alias.p_vaddr, text.vaddr() + text.size());
  }
  if (text_offset && file_size != 0) {
    EXPECT_LE(*text_offset, text.size());
    if (*text_offset > text.size())
      return std::nullopt;
    EXPECT_LE(file_size, text.size() - *text_offset);
    if (file_size > text.size() - *text_offset)
      return std::nullopt;
    EXPECT_EQ(std::memcmp(image.data() + alias.p_offset, text.data() + *text_offset,
                          static_cast<size_t>(file_size)),
              0);
    const uint64_t original_address = text.vaddr() + *text_offset;
    EXPECT_EQ(alias.p_vaddr == original_address, same_mapping);
  }
  segments->push_back(alias);
  std::ranges::sort(*segments, {}, &Elf64_Phdr::p_vaddr);
  header.e_phnum = static_cast<uint16_t>(segments->size());
  std::memcpy(image.data(), &header, sizeof(header));
  std::memcpy(image.data() + header.e_phoff, segments->data(),
              segments->size() * sizeof(Elf64_Phdr));
  const AmdGpuCodeObject mapped(image.data(), image.size());
  EXPECT_TRUE(mapped.is_valid());
  if (!mapped.is_valid())
    return std::nullopt;
  EXPECT_EQ(mapped.text_sections().size(), 1u);
  if (mapped.text_sections().size() != 1)
    return std::nullopt;
  EXPECT_EQ(mapped.text_sections().front()->sectionOffset(), text.sectionOffset());
  EXPECT_EQ(mapped.text_sections().front()->vaddr(), text.vaddr());
  EXPECT_EQ(mapped.text_sections().front()->size(), text.size());
  const std::optional<std::vector<Elf64_Phdr>> readback =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  EXPECT_TRUE(readback);
  if (!readback)
    return std::nullopt;
  EXPECT_EQ(readback->size(), 3u);
  EXPECT_EQ(std::ranges::count(*readback, alias.p_vaddr, &Elf64_Phdr::p_vaddr), 1);
  const std::vector<Elf64_Phdr>::const_iterator actual_alias =
      std::ranges::find(*readback, alias.p_vaddr, &Elf64_Phdr::p_vaddr);
  EXPECT_NE(actual_alias, readback->end());
  if (actual_alias == readback->end())
    return std::nullopt;
  EXPECT_EQ(actual_alias->p_type, PT_LOAD);
  EXPECT_EQ(actual_alias->p_flags, 0x4u);
  EXPECT_EQ(actual_alias->p_offset, file_offset);
  EXPECT_EQ(actual_alias->p_filesz, file_size);
  EXPECT_EQ(actual_alias->p_memsz, memory_size);
  EXPECT_EQ(actual_alias->p_offset % actual_alias->p_align,
            actual_alias->p_vaddr % actual_alias->p_align);
  return LoadAliasFixture{.image = std::move(image), .alias = alias};
}

void expect_load_alias_mapping_preserved(const LoadAliasFixture &fixture,
                                         const InstrumentedCodeObject &result,
                                         bool unchanged_bytes = true) {
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(candidate.is_valid());
  ASSERT_EQ(candidate.text_sections().size(), 1u);
  EXPECT_GT(candidate.text_sections().front()->size(), 4 * sizeof(uint32_t));
  Elf64_Ehdr header = {};
  std::memcpy(&header, result.elf_bytes.data(), sizeof(header));
  const std::optional<std::vector<Elf64_Phdr>> segments = read_elf_table<Elf64_Phdr>(
      result.elf_bytes, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  ASSERT_EQ(segments->size(), 3u);
  const std::vector<Elf64_Phdr>::const_iterator alias =
      std::ranges::find(*segments, fixture.alias.p_vaddr, &Elf64_Phdr::p_vaddr);
  ASSERT_NE(alias, segments->end());
  EXPECT_EQ(alias->p_type, PT_LOAD);
  EXPECT_EQ(alias->p_flags, fixture.alias.p_flags);
  EXPECT_EQ(alias->p_offset, fixture.alias.p_offset);
  EXPECT_EQ(alias->p_vaddr, fixture.alias.p_vaddr);
  EXPECT_EQ(alias->p_paddr, fixture.alias.p_paddr);
  EXPECT_EQ(alias->p_filesz, fixture.alias.p_filesz);
  EXPECT_EQ(alias->p_memsz, fixture.alias.p_memsz);
  EXPECT_EQ(alias->p_align, fixture.alias.p_align);
  ASSERT_LE(alias->p_offset, result.elf_bytes.size());
  ASSERT_LE(alias->p_filesz, result.elf_bytes.size() - alias->p_offset);
  const int compared = std::memcmp(result.elf_bytes.data() + alias->p_offset,
                                   fixture.image.data() + fixture.alias.p_offset,
                                   static_cast<size_t>(fixture.alias.p_filesz));
  if (unchanged_bytes)
    EXPECT_EQ(compared, 0);
  else
    EXPECT_NE(compared, 0);
}

void expect_load_alias_edit_rejected(Instrumentor &instrumentor) {
  // A valid earlier edit must not publish a partial batch when the alias fails.
  instrumentor.add_point({.anchor_offset = 0,
                          .kind = InstrumentationKind::ReplaceWithInst,
                          .probe_symbol = {},
                          .probe_args = {},
                          .replacement = InstructionReplacement{
                              .expected_bytes = make_instruction_bytes({kNop0Word}),
                              .replacement_bytes = make_instruction_bytes({kNop1Word})}});
  instrumentor.add_point({.anchor_offset = 4,
                          .kind = InstrumentationKind::ReplaceWithInst,
                          .probe_symbol = {},
                          .probe_args = {},
                          .replacement = InstructionReplacement{
                              .expected_bytes = make_instruction_bytes({kNop1Word}),
                              .replacement_bytes = make_instruction_bytes({kNop0Word})}});
  const Instrumentor::ValidationResult resolved = instrumentor.validate_points();
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  EXPECT_FALSE(resolved.errors.empty());
  EXPECT_TRUE(resolved.sites.empty());
  EXPECT_TRUE(resolved.edit_error);
  if (resolved.edit_error) {
    EXPECT_EQ(resolved.edit_error->code, TextEditErrorCode::InvalidImage);
    EXPECT_EQ(resolved.edit_error->offset, 4u);
    EXPECT_NE(resolved.edit_error->message.find("PT_LOAD"), std::string::npos);
  }
  EXPECT_FALSE(result.errors.empty());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.sites.empty());
  EXPECT_TRUE(result.patches.empty());
  EXPECT_TRUE(result.edit_error);
  if (result.edit_error) {
    EXPECT_EQ(result.edit_error->code, TextEditErrorCode::InvalidImage);
    EXPECT_EQ(result.edit_error->offset, 4u);
    EXPECT_NE(result.edit_error->message.find("PT_LOAD"), std::string::npos);
  }
}

TEST(CheckedTextEdits, RejectsDivergentLoadFileAliasesAtEditedInstructions) {
  struct AliasRange {
    uint64_t offset;
    uint64_t size;
  };
  for (const AliasRange &range : std::array<AliasRange, 3>{AliasRange{.offset = 4, .size = 4},
                                                           AliasRange{.offset = 4, .size = 2},
                                                           AliasRange{.offset = 6, .size = 2}}) {
    SCOPED_TRACE(range.offset);
    SCOPED_TRACE(range.size);
    const std::optional<LoadAliasFixture> fixture =
        make_load_alias_fixture(range.offset, range.size, range.size);
    ASSERT_TRUE(fixture);
    const std::vector<uint8_t> pristine = fixture->image;
    const AmdGpuCodeObject source(fixture->image.data(), fixture->image.size());
    ASSERT_TRUE(source.is_valid());
    Instrumentor direct(source, kArch);
    expect_load_alias_edit_rejected(direct);
    for (const bool copied_cfg : {false, true}) {
      SCOPED_TRACE(copied_cfg);
      Instrumentor seed(source, kArch);
      if (copied_cfg) {
        seed.add_point_by_offset(0);
        const Instrumentor::ValidationResult observed = seed.validate_points();
        ASSERT_TRUE(observed.errors.empty()) << observed.errors.front();
        ASSERT_EQ(observed.sites.size(), 1u);
      }
      TextEditError error;
      const InstructionBoundaries *context = seed.instruction_boundaries(error);
      ASSERT_NE(context, nullptr) << error.message;
      ASSERT_NE(context->find(4), nullptr);
      EXPECT_EQ(context->find(4)->size, 4u);
      EXPECT_EQ(seed.instruction_boundaries(error), context);
      for (int attempt = 0; attempt < 2; ++attempt) {
        FailureOr<std::unique_ptr<Instrumentor>> fork = seed.fork(error);
        ASSERT_TRUE(fork.succeeded()) << error.message;
        EXPECT_EQ(fork.value()->instruction_boundaries(error), context);
        expect_load_alias_edit_rejected(*fork.value());
      }
      // The same shared facts must allow a different, unaliased source edit.
      FailureOr<std::unique_ptr<Instrumentor>> safe = seed.fork(error);
      ASSERT_TRUE(safe.succeeded()) << error.message;
      EXPECT_EQ(safe.value()->instruction_boundaries(error), context);
      const ReplacementEdit harmless = {.offset = 0,
                                        .expected = make_instruction_bytes({kNop0Word}),
                                        .replacement = make_instruction_bytes({kNop1Word})};
      const FailureOr<InstrumentedCodeObject> accepted =
          apply_replacements(*safe.value(), std::span(&harmless, 1), error);
      ASSERT_TRUE(accepted.succeeded()) << error.message;
      EXPECT_TRUE(
          matches_checked_output(source, std::span(&harmless, 1), accepted.value().elf_bytes));
      ASSERT_NO_FATAL_FAILURE(expect_load_alias_mapping_preserved(*fixture, accepted.value()));
    }
    EXPECT_EQ(fixture->image, pristine);
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  }
}

TEST(CheckedTextEdits, LoadFileAliasChecksKeepTouchingEmptyAndEquivalentMappingsEligible) {
  struct Control {
    std::optional<uint64_t> text_offset;
    uint64_t file_size;
    uint64_t memory_size;
    bool same_mapping;
  };
  const std::array<Control, 5> controls = {
      Control{.text_offset = 0, .file_size = 4, .memory_size = 4, .same_mapping = false},
      Control{.text_offset = 8, .file_size = 4, .memory_size = 4, .same_mapping = false},
      Control{.text_offset = 4, .file_size = 0, .memory_size = 8, .same_mapping = false},
      Control{.text_offset = std::nullopt, .file_size = 4, .memory_size = 4, .same_mapping = false},
      Control{.text_offset = 4, .file_size = 4, .memory_size = 4, .same_mapping = true}};
  for (size_t index = 0; index < controls.size(); ++index) {
    SCOPED_TRACE(index);
    const Control &control = controls[index];
    const std::optional<LoadAliasFixture> fixture = make_load_alias_fixture(
        control.text_offset, control.file_size, control.memory_size, control.same_mapping);
    ASSERT_TRUE(fixture);
    const std::vector<uint8_t> pristine = fixture->image;
    const AmdGpuCodeObject source(fixture->image.data(), fixture->image.size());
    ASSERT_TRUE(source.is_valid());
    Instrumentor seed(source, kArch);
    seed.add_point_by_offset(0);
    const Instrumentor::ValidationResult observed = seed.validate_points();
    ASSERT_TRUE(observed.errors.empty()) << observed.errors.front();
    ASSERT_EQ(observed.sites.size(), 1u);
    TextEditError error;
    const InstructionBoundaries *context = seed.instruction_boundaries(error);
    ASSERT_NE(context, nullptr) << error.message;
    FailureOr<std::unique_ptr<Instrumentor>> fork = seed.fork(error);
    ASSERT_TRUE(fork.succeeded()) << error.message;
    EXPECT_EQ(fork.value()->instruction_boundaries(error), context);
    Instrumentor direct(source, kArch);
    const std::array<Instrumentor *, 2> batches = {&direct, fork.value().get()};
    for (Instrumentor *batch : batches) {
      const ReplacementEdit edit = {.offset = 4,
                                    .expected = make_instruction_bytes({kNop1Word}),
                                    .replacement = make_instruction_bytes({kNop0Word})};
      const FailureOr<InstrumentedCodeObject> result =
          apply_replacements(*batch, std::span(&edit, 1), error);
      ASSERT_TRUE(result.succeeded()) << error.message;
      EXPECT_TRUE(matches_checked_output(source, std::span(&edit, 1), result.value().elf_bytes));
      ASSERT_NO_FATAL_FAILURE(
          expect_load_alias_mapping_preserved(*fixture, result.value(), !control.same_mapping));
    }
    EXPECT_EQ(fixture->image, pristine);
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  }
}

TEST(CheckedTextEdits, StreamingScanDistinguishesGfx1250ZeroLiteralFromPadding) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  const std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      {0xbe8000ffu, 0, 0, endpgm.value()}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const AmdGpuCodeObject object(image.data(), image.size());
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  TextEditError error;
  Instrumentor seed(object, arch);
  const InstructionBoundaries *facts = seed.instruction_boundaries(error);
  ASSERT_NE(facts, nullptr) << error.message;
  const InstructionBoundaries::Entry *literal = facts->find(0);
  ASSERT_NE(literal, nullptr);
  EXPECT_EQ(literal->size, 8u);
  EXPECT_EQ(facts->find(4), nullptr);
  EXPECT_EQ(facts->find(8), nullptr);
  const InstructionBoundaries::Entry *terminator = facts->find(12);
  ASSERT_NE(terminator, nullptr);
  EXPECT_EQ(terminator->size, 4u);
  // Offset8 is padding, so a fallthrough body has no valid continuation.
  // Terminate the positive replacement while retaining the literal/padding case.
  ReplacementEdit edit = {.offset = 0,
                          .expected = make_instruction_bytes({0xbe8000ffu, 0}),
                          .replacement = make_instruction_bytes({nop, endpgm.value()})};
  const FailureOr<InstrumentedCodeObject> result =
      apply_replacements(object, std::span(&edit, 1), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(matches_checked_output(object, std::span(&edit, 1), result.value().elf_bytes));
  edit = {.offset = 8,
          .expected = make_instruction_bytes({0}),
          .replacement = make_instruction_bytes({nop})};
  EXPECT_TRUE(apply_replacements(object, std::span(&edit, 1), error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::InvalidInstruction);
  EXPECT_EQ(error.offset, 8u);
}

TEST(CheckedTextEdits, RejectsBranchReferenceIntoReplacedLiteral) {
  const util::FailureOr<uint32_t> branch = build_s_branch(1, kArch);
  ASSERT_TRUE(branch.succeeded());
  const std::vector<uint8_t> image =
      test::make_gfx1200_kernel_elf({branch.value(), 0xbe8000ffu, 0x12345678u, kEndpgmWord}, 0);
  const AmdGpuCodeObject object(image.data(), image.size());
  const uint32_t nop = kNop0Word;
  const ReplacementEdit edit = {.offset = 4,
                                .expected = make_instruction_bytes({0xbe8000ffu, 0x12345678u}),
                                .replacement = make_instruction_bytes({nop, nop})};
  TextEditError error;
  EXPECT_TRUE(apply_replacements(object, std::span(&edit, 1), error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
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

TEST(CheckedTextEdits, ReplacementGrowthRejectsRetainedBranchesToMovedTargets) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA2;
  const util::FailureOr<uint32_t> nop = build_s_nop(0, arch);
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(nop.succeeded() && endpgm.succeeded());
  // Cases: later, same-text, removed, earlier, sectionless shifted LOAD,
  // containing LOAD file tail, containing LOAD zero-fill tail.
  for (int scenario = 0; scenario < 7; ++scenario) {
    SCOPED_TRACE(scenario);
    const bool later = scenario != 3;
    const uint64_t target = scenario == 1 ? 0x308 : scenario >= 5 ? 0x400 : later ? 0x1400 : 0x200;
    const int64_t displacement = static_cast<int64_t>(target) - 0x308;
    const util::FailureOr<uint32_t> branch = build_s_branch(displacement / 4, arch);
    ASSERT_TRUE(branch.succeeded());
    std::vector<uint8_t> image = make_branch_to_other_section_elf(branch.value(), later);
    if (scenario >= 4) {
      const Elf64_Ehdr header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
      const Elf64_Shdr unsectioned = {};
      test_support::write_elf_struct_for_test(image, header.e_shoff + sizeof(Elf64_Shdr),
                                              unsectioned);
      if (scenario >= 5) {
        Elf64_Phdr segment =
            test_support::read_elf_struct_for_test<Elf64_Phdr>(image, header.e_phoff);
        segment.p_filesz = scenario == 6 ? 12 : 0x104;
        segment.p_memsz = 0x104;
        const Elf64_Phdr unused = {};
        test_support::write_elf_struct_for_test(image, header.e_phoff, segment);
        test_support::write_elf_struct_for_test(image, header.e_phoff + sizeof(segment), unused);
      }
    }
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    const Section *helper =
        test_support::find_section(source, later ? ".text.late" : ".text.early");
    if (scenario < 4) {
      ASSERT_NE(helper, nullptr);
    }
    // The three sectionless negatives exercise the mapping predicates directly.
    for (int route = 0; route < (scenario < 4 ? 4 : 1); ++route) {
      SCOPED_TRACE(route);
      Instrumentor direct(source, arch);
      Instrumentor seed(source, arch);
      std::unique_ptr<Instrumentor> forked;
      Instrumentor *batch = &direct;
      TextEditError error;
      if (route == 1) {
        ASSERT_NE(direct.instruction_boundaries(error), nullptr) << error.message;
      } else if (route >= 2) {
        if (route == 3) {
          seed.add_point_by_offset(0);
          const Instrumentor::ValidationResult warm = seed.validate_points();
          ASSERT_TRUE(warm.errors.empty()) << warm.errors.front();
          ASSERT_EQ(warm.sites.size(), 1u);
        }
        FailureOr<std::unique_ptr<Instrumentor>> child = seed.fork(error);
        ASSERT_TRUE(child.succeeded()) << error.message;
        forked = std::move(child).value();
        batch = forked.get();
      }
      InstrumentationPoint point = {
          .anchor_offset = scenario == 2 ? 4u : 0u,
          .kind = InstrumentationKind::ReplaceWithInst,
          .probe_symbol = {},
          .probe_args = {},
          .replacement = InstructionReplacement{
              .expected_bytes =
                  make_instruction_bytes({scenario == 2 ? branch.value() : nop.value()}),
              .replacement_bytes = scenario == 2 ? make_instruction_bytes({endpgm.value()})
                                                 : std::vector<uint8_t>{}}};
      batch->add_point(std::move(point));
      const Instrumentor::ValidationResult validated = batch->validate_points();
      const InstrumentedCodeObject result = batch->patch();
      if (scenario == 0 || scenario >= 4) {
        EXPECT_FALSE(validated.errors.empty());
        EXPECT_TRUE(validated.sites.empty());
        EXPECT_FALSE(result.errors.empty());
        if (!validated.errors.empty()) {
          EXPECT_NE(validated.errors.front().find("retained direct branch"), std::string::npos);
        }
        if (!result.errors.empty()) {
          EXPECT_NE(result.errors.front().find("retained direct branch"), std::string::npos);
        }
        EXPECT_TRUE(result.elf_bytes.empty());
        EXPECT_TRUE(result.sites.empty());
        EXPECT_TRUE(result.edit_error.has_value());
        if (result.edit_error) {
          EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
          EXPECT_EQ(result.edit_error->offset, 4u);
        }
      } else {
        ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
        ASSERT_EQ(validated.sites.size(), 1u);
        ASSERT_TRUE(result.errors.empty()) << result.errors.front();
        ASSERT_EQ(result.sites.size(), 1u);
        const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
        ASSERT_TRUE(candidate.is_valid());
        const Section *text = test_support::find_section(candidate, ".text");
        const Section *moved_helper =
            test_support::find_section(candidate, later ? ".text.late" : ".text.early");
        ASSERT_NE(text, nullptr);
        ASSERT_NE(moved_helper, nullptr);
        EXPECT_EQ(moved_helper->vaddr(), helper->vaddr() + (later ? 4096 : 0));
        const std::unique_ptr<Decoder> decoder = Decoder::create(arch);
        ASSERT_NE(decoder, nullptr);
        uint32_t word = 0;
        std::memcpy(&word, text->data() + 4, sizeof(word));
        const DecodeResult decoded = decoder->decode_window(std::span(&word, 1), 4);
        ASSERT_TRUE(decoded.succeeded());
        ASSERT_TRUE(decoded.value()->branch_offset_bytes());
        const std::optional<uint64_t> destination = add_signed_offset(
            text->vaddr() + 4 + decoded.value()->size(), *decoded.value()->branch_offset_bytes());
        const uint64_t expected =
            scenario == 2 ? text->vaddr() + result.sites.front().guest_offset : target;
        EXPECT_EQ(destination, expected);
      }
      EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
    }
  }
}

TEST(CheckedTextEdits, PreservesBranchToEarlierExecutableSection) {
  constexpr rj_code_arch_t kRdna2 = ROCJITSU_CODE_ARCH_RDNA2;
  // The branch at 0x304 uses next PC 0x308: (0x200 - 0x308) / 4 = -66.
  const util::FailureOr<uint32_t> branch = build_s_branch(-66, kRdna2);
  ASSERT_TRUE(branch.succeeded());
  const std::vector<uint8_t> image = make_branch_to_other_section_elf(branch.value());
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
  const std::vector<uint8_t> image = make_branch_to_other_section_elf(branch.value());
  expect_reference_result(image, 0, 0, ROCJITSU_CODE_ARCH_RDNA2);
}

TEST(CheckedTextEdits, ReplacementFactsRetainEarlierSectionAddressesAcrossForks) {
  constexpr rj_code_arch_t kRdna2 = ROCJITSU_CODE_ARCH_RDNA2;
  const util::FailureOr<uint32_t> branch = build_s_branch(-66, kRdna2);
  const util::FailureOr<uint32_t> nop = build_s_nop(0, kRdna2);
  const util::FailureOr<uint32_t> replacement = build_s_nop(1, kRdna2);
  ASSERT_TRUE(branch.succeeded() && nop.succeeded() && replacement.succeeded());
  const std::vector<uint8_t> image = make_branch_to_other_section_elf(branch.value());
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor seed(source, kRdna2);
  TextEditError error;
  const InstructionBoundaries *facts = seed.instruction_boundaries(error);
  ASSERT_NE(facts, nullptr) << error.message;
  ASSERT_EQ(facts->reference_addresses().size(), 1u);
  EXPECT_EQ(facts->reference_addresses().front(), 0x200u);
  FailureOr<std::unique_ptr<Instrumentor>> fork = seed.fork(error);
  ASSERT_TRUE(fork.succeeded()) << error.message;
  EXPECT_EQ(fork.value()->instruction_boundaries(error), facts);
  Instrumentor direct(source, kRdna2);
  const ReplacementEdit edit = {.offset = 0,
                                .expected = make_instruction_bytes({nop.value()}),
                                .replacement = make_instruction_bytes({replacement.value()})};
  for (Instrumentor *batch : std::array<Instrumentor *, 2>{&direct, fork.value().get()}) {
    const FailureOr<InstrumentedCodeObject> result =
        apply_replacements(*batch, std::span(&edit, 1), error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(matches_checked_output(source, std::span(&edit, 1), result.value().elf_bytes));
  }
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
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
