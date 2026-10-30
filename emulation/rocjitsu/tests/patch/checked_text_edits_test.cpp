// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/checked_text_edits.h"

#include "../dbi_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instrumentor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>

namespace rocjitsu {
namespace {
constexpr rj_code_target_id_t kTarget = ROCJITSU_CODE_TARGET_GFX1200;
constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
std::vector<uint8_t> make_instruction_bytes(std::initializer_list<uint32_t> words) {
  std::vector<uint8_t> result(words.size() * 4);
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
// This exercises encoding, complete extents, equal-length edits and preservation.
// It never executes the store or establishes GPU register/memory state for it.
std::vector<uint8_t> make_mixed_width_kernel_elf() {
  return test::make_gfx1200_kernel_elf({build_s_nop(1, kArch), build_s_nop(2, kArch), 0xbe8000ffu,
                                        0x12345678u, build_s_nop(3, kArch), 0xee068002u,
                                        0x00800000u, 0, build_s_endpgm(kArch)},
                                       0);
}
std::vector<CheckedTextEdit> make_mixed_width_nop_edits() {
  const uint32_t nop = build_s_nop(0, kArch);
  // Columns: text offset, expected instruction bytes, replacement NOP bytes.
  return {
      {0, make_instruction_bytes({build_s_nop(1, kArch)}), make_instruction_bytes({nop})},
      {8, make_instruction_bytes({0xbe8000ffu, 0x12345678u}), make_instruction_bytes({nop, nop})},
      {20, make_instruction_bytes({0xee068002u, 0x00800000u, 0}),
       make_instruction_bytes({nop, nop, nop})}};
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
  auto segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  EXPECT_TRUE(segments);
  if (!segments)
    return {};
  const auto data_load = std::ranges::find_if(*segments, [](const Elf64_Phdr &segment) {
    return segment.p_type == PT_LOAD && (segment.p_flags & PF_X) == 0;
  });
  const auto symbols = std::ranges::find(sections, SHT_SYMTAB, &Elf64_Shdr::sh_type);
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

// Test-side exact-image oracle, independent of the production transaction.
testing::AssertionResult matches_checked_output(const AmdGpuCodeObject &source,
                                                std::span<const CheckedTextEdit> edits,
                                                std::span<const uint8_t> candidate) {
  const auto *data = reinterpret_cast<const uint8_t *>(source.image_data());
  std::vector<uint8_t> expected(data, data + source.image_size());
  const Section &text = *source.text_sections().front();
  for (const CheckedTextEdit &edit : edits) {
    if (edit.offset > text.size() || edit.replacement.size() > text.size() - edit.offset)
      return testing::AssertionFailure() << "test oracle: edit outside .text at " << edit.offset;
    std::memcpy(expected.data() + text.sectionOffset() + edit.offset, edit.replacement.data(),
                edit.replacement.size());
  }
  const AmdGpuCodeObject parsed(candidate.data(), candidate.size());
  if (!parsed.is_valid() || parsed.target_id() != source.target_id() ||
      !std::ranges::equal(expected, candidate))
    return testing::AssertionFailure() << "test oracle: unexpected output bytes or target";
  return testing::AssertionSuccess();
}

TEST(CheckedTextEdits, ReusableFactsKeepRequestChecksAndSourceIdentity) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject source(image.data(), image.size());
  TextEditError error;
  FailureOr<std::unique_ptr<CheckedTextEditContext>> context =
      CheckedTextEditContext::create(source, error);
  ASSERT_TRUE(context.succeeded()) << error.message;
  const std::vector<CheckedTextEdit> original_edits = make_mixed_width_nop_edits();
  for (int attempt = 0; attempt < 2; ++attempt) {
    const FailureOr<CheckedTextEditResult> result =
        apply_checked_text_edits(*context.value(), original_edits, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(matches_checked_output(source, original_edits, result.value().image));
    std::vector<CheckedTextEdit> stale = original_edits;
    stale.back().expected[0] ^= 1;
    EXPECT_TRUE(apply_checked_text_edits(*context.value(), stale, error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::StaleBytes);
    std::vector<CheckedTextEdit> invalid = original_edits;
    invalid.back().replacement.assign(12, 0xff);
    EXPECT_TRUE(apply_checked_text_edits(*context.value(), invalid, error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::InvalidInstruction);
  }
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
}

TEST(CheckedTextEdits, CachedFullTextFactsIncludeUnreachableReferences) {
  const std::vector<uint8_t> image = test::make_gfx1200_kernel_elf(
      {0xbe8000ffu, 0x12345678u, build_s_endpgm(kArch), build_s_branch(-3, kArch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  TextEditError error;
  FailureOr<std::unique_ptr<CheckedTextEditContext>> context =
      CheckedTextEditContext::create(source, error);
  ASSERT_TRUE(context.succeeded()) << error.message;
  const uint32_t nop = build_s_nop(0, kArch);
  const CheckedTextEdit edit = {.offset = 0,
                                .expected = make_instruction_bytes({0xbe8000ffu, 0x12345678u}),
                                .replacement = make_instruction_bytes({nop, nop})};
  for (int attempt = 0; attempt < 2; ++attempt) {
    EXPECT_TRUE(apply_checked_text_edits(*context.value(), std::span(&edit, 1), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
  }
}

TEST(CheckedTextEdits, ThreeNonadjacentCompleteExtentsPreserveAllOtherImageBytes) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  const std::vector<CheckedTextEdit> requests = make_mixed_width_nop_edits();
  TextEditError error;
  FailureOr<CheckedTextEditResult> result = apply_checked_text_edits(object, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  std::vector<uint8_t> expected = image;
  const Section &text = *object.text_sections().front();
  for (const CheckedTextEdit &edit : requests)
    std::memcpy(expected.data() + text.sectionOffset() + edit.offset, edit.replacement.data(),
                edit.replacement.size());
  EXPECT_EQ(result.value().image, expected);
  EXPECT_EQ(result.value().edits, requests);
  EXPECT_EQ(result.value().target, kTarget);
  ASSERT_EQ(result.value().mapping.size(), 1u);
  EXPECT_EQ(result.value().mapping.front().source_offset, 0u);
  EXPECT_EQ(result.value().mapping.front().candidate_offset, 0u);
  EXPECT_EQ(result.value().mapping.front().size, 36u);
  EXPECT_EQ(std::memcmp(object.image_data(), image.data(), image.size()), 0);
  EXPECT_TRUE(matches_checked_output(object, requests, result.value().image));
}

TEST(CheckedTextEdits, SectionOnlyTransformationDoesNotRequireLoadMapping) {
  std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_phnum = 0;
  std::memcpy(image.data(), &header, sizeof(header));
  const AmdGpuCodeObject object(image.data(), image.size());
  TextEditError error;
  const std::vector<CheckedTextEdit> requests = make_mixed_width_nop_edits();
  const FailureOr<CheckedTextEditResult> result = apply_checked_text_edits(object, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(matches_checked_output(object, requests, result.value().image));
}

TEST(CheckedTextEdits, Gfx1100EditsRetainSourceTarget) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA3;
  const uint32_t original = build_s_nop(1, arch);
  const uint32_t nop = build_s_nop(0, arch);
  const std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      {original, build_s_endpgm(arch)}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1100);
  const AmdGpuCodeObject source(image.data(), image.size());
  const std::vector<CheckedTextEdit> requests = {{.offset = 0,
                                                  .expected = make_instruction_bytes({original}),
                                                  .replacement = make_instruction_bytes({nop})}};
  TextEditError error;
  const FailureOr<CheckedTextEditResult> result = apply_checked_text_edits(source, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_NE(source.target_id(), ROCJITSU_CODE_TARGET_INVALID);
  EXPECT_EQ(result.value().target, source.target_id());
  std::vector<uint8_t> expected = image;
  std::memcpy(expected.data() + source.text_sections().front()->sectionOffset(), &nop, sizeof(nop));
  EXPECT_EQ(result.value().image, expected);
  EXPECT_TRUE(matches_checked_output(source, requests, result.value().image));
}

TEST(CheckedTextEdits, LegacyElfIdentityIsSourceDerivedAndUnchanged) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA1;
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      {build_s_nop(1, arch), build_s_endpgm(arch)}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX908);
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_EQ(source.target_id(), ROCJITSU_CODE_TARGET_INVALID);
  const std::vector<CheckedTextEdit> requests = {
      {.offset = 0,
       .expected = make_instruction_bytes({build_s_nop(1, arch)}),
       .replacement = make_instruction_bytes({build_s_nop(0, arch)})}};
  TextEditError error;
  const FailureOr<CheckedTextEditResult> result = apply_checked_text_edits(source, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().target, ROCJITSU_CODE_TARGET_INVALID);
  std::vector<uint8_t> expected = image;
  std::memcpy(expected.data() + result.value().section.file_offset,
              requests.front().replacement.data(), 4);
  EXPECT_EQ(result.value().image, expected);
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
      EXPECT_TRUE(apply_checked_text_edits(unknown, requests, error).failed());
      EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedTarget);
    }
  }
}

TEST(CheckedTextEdits, LateFailuresReturnNoCandidateAndLeaveSourceUnchanged) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  for (int mode = 0; mode < 8; ++mode) {
    SCOPED_TRACE(mode);
    std::vector<CheckedTextEdit> requests = make_mixed_width_nop_edits();
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
      requests.back().replacement.clear();
      break;
    case 4:
      requests.back().replacement.resize(8);
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
    EXPECT_TRUE(apply_checked_text_edits(object, requests, error).failed());
    EXPECT_NE(error.code, TextEditErrorCode::None);
    EXPECT_EQ(std::memcmp(object.image_data(), image.data(), image.size()), 0);
  }
}

TEST(CheckedTextEdits, RejectsInteriorLiteralAndPartialInstruction) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  // Columns: text offset, expected bytes, replacement bytes; each row is invalid.
  for (const CheckedTextEdit &edit : std::vector<CheckedTextEdit>{
           {12, make_instruction_bytes({0x12345678u}),
            make_instruction_bytes({build_s_nop(0, kArch)})},
           {8, make_instruction_bytes({0xbe8000ffu}),
            make_instruction_bytes({build_s_nop(0, kArch)})},
           {20, make_instruction_bytes({0xee068002u, 0x00800000u}),
            make_instruction_bytes({build_s_nop(0, kArch), build_s_nop(0, kArch)})}}) {
    TextEditError error;
    EXPECT_TRUE(apply_checked_text_edits(object, std::span(&edit, 1), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::InvalidInstruction) << error.message;
  }
}

TEST(CheckedTextEdits, RejectsTruncatedOriginalEightAndTwelveByteInstructions) {
  for (const std::vector<uint32_t> &words :
       std::vector<std::vector<uint32_t>>{{0xbe8000ffu}, {0xee068002u, 0x00800000u}}) {
    const std::vector<uint8_t> image = test::make_gfx1200_kernel_elf(words, 0);
    const AmdGpuCodeObject object(image.data(), image.size());
    CheckedTextEdit edit = {.offset = 0,
                            .expected = std::vector<uint8_t>(words.size() * 4),
                            .replacement = std::vector<uint8_t>(words.size() * 4)};
    std::memcpy(edit.expected.data(), words.data(), edit.expected.size());
    for (size_t offset = 0; offset < edit.replacement.size(); offset += 4) {
      const uint32_t nop = build_s_nop(0, kArch);
      std::memcpy(edit.replacement.data() + offset, &nop, 4);
    }
    TextEditError error;
    EXPECT_TRUE(apply_checked_text_edits(object, std::span(&edit, 1), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::InvalidInstruction) << error.message;
  }
}

TEST(CheckedTextEdits, EmitterOracleRejectsTextMetadataAndTargetCorruption) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  const std::vector<CheckedTextEdit> requests = make_mixed_width_nop_edits();
  TextEditError error;
  FailureOr<CheckedTextEditResult> result = apply_checked_text_edits(object, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_EQ(kernels.size(), 1u);
  for (const uint64_t offset :
       {text.sectionOffset() + 4, text.sectionOffset(), uint64_t{offsetof(Elf64_Ehdr, e_flags)},
        kernels.front().descriptor_file_offset}) {
    std::vector<uint8_t> candidate = result.value().image;
    candidate[offset] ^= 1;
    EXPECT_FALSE(matches_checked_output(object, requests, candidate)) << offset;
  }
  std::vector<uint8_t> unknown = image;
  Elf64_Ehdr header = {};
  std::memcpy(&header, unknown.data(), sizeof(header));
  header.e_flags = EF_AMDGPU_MACH_NONE;
  std::memcpy(unknown.data(), &header, sizeof(header));
  const AmdGpuCodeObject unsupported(unknown.data(), unknown.size());
  EXPECT_TRUE(apply_checked_text_edits(unsupported, requests, error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedTarget);
  EXPECT_TRUE(apply_checked_text_edits(object, {}, error).failed());
}

TEST(CheckedTextEdits, ClauseAndTwelveByteTrampolinePoliciesStaySeparate) {
  const std::vector<uint8_t> image = make_mixed_width_kernel_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  Instrumentor instrumentor(object, kArch);
  instrumentor.add_point_by_offset(20, InstrumentationKind::BeforeInst);
  EXPECT_TRUE(instrumentor.patch().elf_bytes.empty());

  // s_clause 0 followed by a complete global store. The transaction cannot
  // silently change the clause's instruction count or memory operation kinds.
  const std::vector<uint8_t> clause_image = test::make_gfx1200_kernel_elf(
      {0xbf850000u, 0xee068002u, 0x00800000u, 0, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject clause_object(clause_image.data(), clause_image.size());
  CheckedTextEdit edit = make_mixed_width_nop_edits().back();
  edit.offset = 4;
  TextEditError error;
  EXPECT_TRUE(apply_checked_text_edits(clause_object, std::span(&edit, 1), error).failed());
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
    FailureOr<CheckedTextEditResult> result =
        apply_checked_text_edits(object, make_mixed_width_nop_edits(), error);
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
    const FailureOr<std::unique_ptr<CheckedTextEditContext>> context =
        CheckedTextEditContext::create(object, error);
    ASSERT_TRUE(context.succeeded()) << error.message;
    const std::vector<uint8_t> original_image = image;
    const std::vector<CheckedTextEdit> edits = make_mixed_width_nop_edits();
    EXPECT_TRUE(apply_checked_text_edits(*context.value(), edits, error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
    EXPECT_EQ(error.offset, target_offset);
    EXPECT_EQ(image, original_image);
    EXPECT_EQ(std::memcmp(object.image_data(), original_image.data(), original_image.size()), 0);

    // The unrelated first edit remains usable after the rejected transaction.
    const std::span<const CheckedTextEdit> first(edits.data(), 1);
    const FailureOr<CheckedTextEditResult> result =
        apply_checked_text_edits(*context.value(), first, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(matches_checked_output(object, first, result.value().image));
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
  const FailureOr<std::unique_ptr<CheckedTextEditContext>> context =
      CheckedTextEditContext::create(object, error);
  ASSERT_TRUE(context.succeeded()) << error.message;
  const std::vector<CheckedTextEdit> edits = make_mixed_width_nop_edits();
  EXPECT_TRUE(apply_checked_text_edits(*context.value(), edits, error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
  EXPECT_EQ(error.offset, 12u);
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
      const FailureOr<std::unique_ptr<CheckedTextEditContext>> context =
          CheckedTextEditContext::create(object, error);
      ASSERT_TRUE(context.succeeded()) << error.message;
      const std::vector<uint8_t> original_image = image;
      const std::vector<CheckedTextEdit> edits = make_mixed_width_nop_edits();
      const FailureOr<CheckedTextEditResult> result =
          apply_checked_text_edits(*context.value(), edits, error);
      if (nonallocated_metadata) {
        ASSERT_TRUE(result.succeeded()) << error.message;
        EXPECT_TRUE(matches_checked_output(object, edits, result.value().image));
      } else {
        EXPECT_TRUE(result.failed());
        EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
        EXPECT_EQ(error.offset, 12u);
      }
      EXPECT_EQ(image, original_image);
      EXPECT_EQ(std::memcmp(object.image_data(), original_image.data(), original_image.size()), 0);
      // An interior reference does not prohibit unrelated edits to the same source.
      const std::span<const CheckedTextEdit> first(edits.data(), 1);
      const FailureOr<CheckedTextEditResult> unrelated =
          apply_checked_text_edits(*context.value(), first, error);
      ASSERT_TRUE(unrelated.succeeded()) << error.message;
      EXPECT_TRUE(matches_checked_output(object, first, unrelated.value().image));
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
    const std::vector<CheckedTextEdit> edits = make_mixed_width_nop_edits();
    const FailureOr<CheckedTextEditResult> result = apply_checked_text_edits(object, edits, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(matches_checked_output(object, edits, result.value().image));
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
    const std::vector<CheckedTextEdit> edits = make_mixed_width_nop_edits();
    const FailureOr<CheckedTextEditResult> result = apply_checked_text_edits(object, edits, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(matches_checked_output(object, edits, result.value().image));
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
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_EQ(object.text_sections().size(), 1u);
  ASSERT_EQ(object.allocated_executable_sections().size(), 2u);
  TextEditError error;
  const std::vector<CheckedTextEdit> requests = make_mixed_width_nop_edits();
  const FailureOr<CheckedTextEditResult> result = apply_checked_text_edits(object, requests, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  std::vector<uint8_t> expected = image;
  for (const CheckedTextEdit &edit : requests)
    std::memcpy(expected.data() + result.value().section.file_offset + edit.offset,
                edit.replacement.data(), edit.replacement.size());
  EXPECT_EQ(result.value().image, expected);
  EXPECT_TRUE(matches_checked_output(object, requests, expected));
  expected[section.sh_offset] ^= 1;
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
    EXPECT_TRUE(apply_checked_text_edits(object, make_mixed_width_nop_edits(), error).failed());
    EXPECT_EQ(error.code, TextEditErrorCode::InvalidImage) << error.message;
    EXPECT_NE(error.message.find(file_alias ? "aliases another section" : "virtual addresses"),
              std::string::npos);
  }
}

TEST(CheckedTextEdits, StreamingScanDistinguishesGfx1250ZeroLiteralFromPadding) {
  const std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf({0xbe8000ffu, 0, 0, build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)}, 0,
                                   3, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const AmdGpuCodeObject object(image.data(), image.size());
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5);
  CheckedTextEdit edit = {.offset = 0,
                          .expected = make_instruction_bytes({0xbe8000ffu, 0}),
                          .replacement = make_instruction_bytes({nop, nop})};
  TextEditError error;
  EXPECT_TRUE(apply_checked_text_edits(object, std::span(&edit, 1), error).succeeded())
      << error.message;
  edit = {.offset = 8,
          .expected = make_instruction_bytes({0}),
          .replacement = make_instruction_bytes({nop})};
  EXPECT_TRUE(apply_checked_text_edits(object, std::span(&edit, 1), error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::InvalidInstruction);
}

TEST(CheckedTextEdits, RejectsBranchReferenceIntoReplacedLiteral) {
  const std::vector<uint8_t> image = test::make_gfx1200_kernel_elf(
      {build_s_branch(1, kArch), 0xbe8000ffu, 0x12345678u, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject object(image.data(), image.size());
  const uint32_t nop = build_s_nop(0, kArch);
  const CheckedTextEdit edit = {.offset = 4,
                                .expected = make_instruction_bytes({0xbe8000ffu, 0x12345678u}),
                                .replacement = make_instruction_bytes({nop, nop})};
  TextEditError error;
  EXPECT_TRUE(apply_checked_text_edits(object, std::span(&edit, 1), error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences) << error.message;
}

} // namespace
} // namespace rocjitsu
