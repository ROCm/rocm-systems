// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../dbi_test_util.h"
#include "../elf_test_support.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {
constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
constexpr util::FailureOr<uint32_t> kNop0Encoding = build_s_nop(0, kArch);
constexpr util::FailureOr<uint32_t> kNop1Encoding = build_s_nop(1, kArch);
constexpr util::FailureOr<uint32_t> kEndpgmEncoding = build_s_endpgm(kArch);
static_assert(kNop0Encoding.succeeded() && kNop1Encoding.succeeded() &&
              kEndpgmEncoding.succeeded());
constexpr uint32_t kNop0Word = kNop0Encoding.value();
constexpr uint32_t kNop1Word = kNop1Encoding.value();
constexpr uint32_t kEndpgmWord = kEndpgmEncoding.value();

Elf64_Ehdr read_hsa_v5_header(std::span<const uint8_t> image) {
  Elf64_Ehdr header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_VERSION] = 1;
  header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return header;
}

void write_fixture_headers(std::span<uint8_t> image, const Elf64_Ehdr &header,
                           std::span<const Elf64_Phdr> segments,
                           std::span<const Elf64_Shdr> sections = {}) {
  test_support::write_elf_struct_for_test(image, 0, header);
  for (size_t index = 0; index < segments.size(); ++index)
    test_support::write_elf_struct_for_test(image, header.e_phoff + index * sizeof(Elf64_Phdr),
                                            segments[index]);
  for (size_t index = 0; index < sections.size(); ++index)
    test_support::write_elf_struct_for_test(image, header.e_shoff + index * sizeof(Elf64_Shdr),
                                            sections[index]);
}

void patch_eligible_points(const AmdGpuCodeObject &source, InstrumentedCodeObjectDebug &result,
                           std::initializer_list<uint64_t> offsets = {0},
                           rj_code_arch_t arch = kArch) {
  Instrumentor instrumentor(source, arch);
  for (uint64_t offset : offsets)
    instrumentor.add_point_by_offset(offset);
  const Instrumentor::ValidationResult eligible = instrumentor.validate_points();
  ASSERT_TRUE(eligible.errors.empty()) << eligible.errors.front();
  ASSERT_EQ(eligible.sites.size(), offsets.size());
  result = instrumentor.patch_with_debug_summaries();
}

void expect_rejected_without_output(const InstrumentedCodeObjectDebug &result) {
  EXPECT_FALSE(result.errors.empty());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.sites.empty());
  EXPECT_TRUE(result.patches.empty());
}

void expect_loaded_prefix_relocation_policy(uint64_t bytes_before_text, bool overlaps_text) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint64_t loaded_prefix_size = sizeof(uint64_t);
  const util::FailureOr<uint32_t> barrier =
      build_sop1_encoding(arch, cdna5::kSBarrierSignalSop1, 0, 193);
  const util::FailureOr<uint32_t> nop = build_s_nop(0, arch);
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(barrier.succeeded());
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(endpgm.succeeded());
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      {barrier.value(), nop.value(), endpgm.value()}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const AmdGpuCodeObject original(image.data(), image.size());
  ASSERT_TRUE(original.is_valid());
  const Section &original_text = *original.text_sections().front();
  const CodeObjectPatcher original_patcher(original);
  std::vector<Elf64_Shdr> sections = original_patcher.section_headers();
  Elf64_Ehdr header = read_hsa_v5_header(image);
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
  ASSERT_GE(text_load->p_offset, loaded_prefix_size);
  ASSERT_LE(bytes_before_text, loaded_prefix_size);
  text_load->p_offset -= loaded_prefix_size;
  text_load->p_vaddr -= loaded_prefix_size;
  text_load->p_paddr -= loaded_prefix_size;
  text_load->p_filesz += loaded_prefix_size;
  text_load->p_memsz += loaded_prefix_size;
  text_load->p_flags |= 0x2u; // PF_W
  std::fill_n(image.data() + text_load->p_offset, loaded_prefix_size, uint8_t{0x5a});

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
  test_support::write_elf_struct_for_test(image, relocation_offset, relocation);
  write_fixture_headers(image, header, *segments, sections);

  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.target_id(), ROCJITSU_CODE_TARGET_GFX1250);
  ASSERT_EQ(source.text_sections().size(), 1u);
  const Section &text = *source.text_sections().front();
  ASSERT_EQ(text.vaddr(), original_text.vaddr());
  ASSERT_EQ(classify_rocr_relocation_section(header, sections, sections.back()),
            RocrRelocationSectionMode::Dynamic);
  ASSERT_EQ(sections[symbol_table_index].sh_type, SHT_DYNSYM);
  Elf64_Sym symbol = {};
  std::memcpy(&symbol, image.data() + sections[symbol_table_index].sh_offset + sizeof(symbol),
              sizeof(symbol));
  ASSERT_NE(symbol.st_shndx, *text.sectionHeaderIndex());
  ASSERT_NE(symbol.st_shndx, SHN_UNDEF);
  ASSERT_EQ(elf_reloc_type(relocation.r_info), R_AMDGPU_ABS64);
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

  InstrumentedCodeObjectDebug result = {};
  ASSERT_NO_FATAL_FAILURE(patch_eligible_points(source, result, {4, 0}, arch));
  if (overlaps_text) {
    expect_rejected_without_output(result);
    EXPECT_TRUE(result.edit_error);
    if (result.edit_error) {
      EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
    }
  } else {
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 2u);
    ASSERT_EQ(result.patches.size(), 2u);
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    EXPECT_EQ(candidate.target_id(), source.target_id());
    const Section &emitted_text = *candidate.text_sections().front();
    for (const InstrumentedSite &site : result.sites) {
      uint32_t relocated = 0;
      ASSERT_LE(site.guest_offset, emitted_text.size());
      ASSERT_LE(sizeof(relocated), emitted_text.size() - site.guest_offset);
      std::memcpy(&relocated, emitted_text.data() + site.guest_offset, sizeof(relocated));
      EXPECT_EQ(relocated, site.original_offset == 0 ? barrier.value() : nop.value());
    }
    EXPECT_EQ(std::memcmp(result.elf_bytes.data() + text_load->p_offset,
                          pristine.data() + text_load->p_offset, loaded_prefix_size),
              0);
  }
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(BiEditSafety, RejectsRelocationWriteStartingInLoadedPrefix) {
  expect_loaded_prefix_relocation_policy(4, true);
}

TEST(BiEditSafety, AllowsRelocationWriteEndingAtTextBoundary) {
  expect_loaded_prefix_relocation_policy(8, false);
}

void expect_tail_relocation_growth_policy(bool moved_section) {
  std::vector<uint8_t> image =
      test::make_gfx1200_kernel_elf({kNop0Word, kNop1Word, kNop0Word, kEndpgmWord}, 0);
  const AmdGpuCodeObject original(image.data(), image.size());
  ASSERT_TRUE(original.is_valid());
  const Section &text = *original.text_sections().front();
  const uint64_t text_end = text.vaddr() + text.size();
  const uint64_t text_file_end = text.sectionOffset() + text.size();
  const CodeObjectPatcher original_patcher(original);
  std::vector<Elf64_Shdr> sections = original_patcher.section_headers();
  Elf64_Ehdr header = read_hsa_v5_header(image);
  std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  ASSERT_EQ(segments->size(), 2u);
  ASSERT_EQ(segments->front().p_vaddr, text.vaddr());
  ASSERT_EQ(segments->front().p_memsz, text.size());
  if (moved_section) {
    // This target moves with its own LOAD. The equal starting address must not
    // cause a pre-growth range check to reject an otherwise safe relocation.
    sections.push_back({.sh_name = 0,
                        .sh_type = SHT_NOBITS,
                        .sh_flags = SHF_ALLOC | SHF_WRITE,
                        .sh_addr = text_end,
                        .sh_offset = text_file_end,
                        .sh_size = sizeof(uint64_t),
                        .sh_link = 0,
                        .sh_info = 0,
                        .sh_addralign = alignof(uint64_t),
                        .sh_entsize = 0});
    segments->push_back({.p_type = PT_LOAD,
                         .p_flags = 0x6, // PF_R | PF_W
                         .p_offset = text_file_end,
                         .p_vaddr = text_end,
                         .p_paddr = text_end,
                         .p_filesz = 0,
                         .p_memsz = sizeof(uint64_t),
                         .p_align = 1});
  } else {
    // No allocated section owns this zero-fill tail, so relocation remapping
    // leaves its address at the start of the appended trampoline.
    segments->front().p_memsz += sizeof(uint64_t);
    segments->front().p_flags |= 0x2u; // PF_W
  }
  const std::vector<Elf64_Shdr>::iterator symbols =
      std::ranges::find(sections, SHT_SYMTAB, &Elf64_Shdr::sh_type);
  ASSERT_NE(symbols, sections.end());
  const uint32_t symbol_table_index = static_cast<uint32_t>(symbols - sections.begin());
  symbols->sh_type = SHT_DYNSYM;
  const uint64_t relocation_offset = header.e_shoff;
  const Elf64_Rela relocation = {
      .r_offset = text_end, .r_info = (uint64_t{1} << 32) | R_AMDGPU_ABS64, .r_addend = 0};
  const std::vector<Elf64_Phdr>::iterator data_load =
      std::ranges::find_if(*segments, [](const Elf64_Phdr &segment) {
        return segment.p_type == PT_LOAD && segment.p_filesz != 0 && (segment.p_flags & PF_X) == 0;
      });
  ASSERT_NE(data_load, segments->end());
  const uint64_t relocation_vaddr = data_load->p_vaddr + relocation_offset - data_load->p_offset;
  data_load->p_filesz = relocation_offset + sizeof(relocation) - data_load->p_offset;
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
  std::ranges::sort(*segments, {}, &Elf64_Phdr::p_vaddr);
  header.e_phnum = static_cast<uint16_t>(segments->size());
  ASSERT_LE(header.e_phoff + segments->size() * sizeof(Elf64_Phdr), text.sectionOffset());
  header.e_shoff += sizeof(relocation);
  header.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(header.e_shoff + sections.size() * sizeof(Elf64_Shdr));
  test_support::write_elf_struct_for_test(image, relocation_offset, relocation);
  write_fixture_headers(image, header, *segments, sections);
  ASSERT_EQ(classify_rocr_relocation_section(header, sections, sections.back()),
            RocrRelocationSectionMode::Dynamic);
  ASSERT_EQ(std::ranges::any_of(sections,
                                [&](const Elf64_Shdr &section) {
                                  return (section.sh_flags & SHF_ALLOC) != 0 &&
                                         text_end >= section.sh_addr &&
                                         text_end - section.sh_addr < section.sh_size;
                                }),
            moved_section);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  const CodeObjectPatcher patcher(source);
  ASSERT_FALSE(patcher.has_relocations_within_text());
  ASSERT_FALSE(patcher.has_malformed_rocr_relocation_section());
  ASSERT_FALSE(patcher.has_unsupported_relocation_to_text());
  InstrumentedCodeObjectDebug result = {};
  ASSERT_NO_FATAL_FAILURE(patch_eligible_points(source, result));
  if (moved_section) {
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 1u);
    ASSERT_EQ(result.patches.size(), 1u);
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    const Section &emitted_text = *candidate.text_sections().front();
    ASSERT_GT(emitted_text.size(), text.size());
    const CodeObjectPatcher emitted_patcher(candidate);
    EXPECT_FALSE(emitted_patcher.has_relocations_within_text());
    const std::vector<Elf64_Shdr> emitted_sections = emitted_patcher.section_headers();
    ASSERT_EQ(emitted_sections.size(), sections.size());
    Elf64_Rela emitted_relocation = {};
    std::memcpy(&emitted_relocation, result.elf_bytes.data() + emitted_sections.back().sh_offset,
                sizeof(emitted_relocation));
    const Elf64_Shdr &target = emitted_sections[emitted_sections.size() - 2];
    EXPECT_EQ(target.sh_type, SHT_NOBITS);
    EXPECT_GT(target.sh_addr, text_end);
    EXPECT_GE(target.sh_addr, emitted_text.vaddr() + emitted_text.size());
    EXPECT_EQ(emitted_relocation.r_offset, target.sh_addr);
    EXPECT_EQ(emitted_relocation.r_info, relocation.r_info);
    EXPECT_EQ(emitted_relocation.r_addend, relocation.r_addend);
    uint32_t relocated_word = 0;
    std::memcpy(&relocated_word, emitted_text.data() + result.sites.front().guest_offset,
                sizeof(relocated_word));
    EXPECT_EQ(relocated_word, kNop0Word);
  } else {
    expect_rejected_without_output(result);
  }
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(BiEditSafety, RejectsRelocationWriteIntoAppendedTrampoline) {
  expect_tail_relocation_growth_policy(false);
}

TEST(BiEditSafety, AllowsRelocationTargetMovedClearOfAppendedTrampoline) {
  expect_tail_relocation_growth_policy(true);
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
  Elf64_Ehdr header = read_hsa_v5_header(image);
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
  write_fixture_headers(image, header, *segments);
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
  Elf64_Ehdr header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(result.elf_bytes, 0);
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

TEST(BiEditSafety, RejectsDivergentLoadFileAliasesAtEditedInstructions) {
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
    const AmdGpuCodeObject source(pristine.data(), pristine.size());
    InstrumentedCodeObjectDebug result = {};
    ASSERT_NO_FATAL_FAILURE(patch_eligible_points(source, result, {0, 4}));
    expect_rejected_without_output(result);
    EXPECT_TRUE(result.edit_error);
    if (result.edit_error) {
      EXPECT_EQ(result.edit_error->code, TextEditErrorCode::InvalidImage);
      EXPECT_EQ(result.edit_error->offset, 4u);
    }
    // The same image remains eligible for an edit outside the aliased bytes.
    Instrumentor safe(source, kArch);
    safe.add_point_by_offset(0);
    const InstrumentedCodeObject accepted = safe.patch();
    ASSERT_TRUE(accepted.errors.empty()) << accepted.errors.front();
    ASSERT_EQ(accepted.sites.size(), 1u);
    ASSERT_NO_FATAL_FAILURE(expect_load_alias_mapping_preserved(*fixture, accepted));
    EXPECT_EQ(fixture->image, pristine);
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  }
}

TEST(BiEditSafety, RejectsTextGrowthIntoDivergentLoadZeroFill) {
  const std::optional<LoadAliasFixture> fixture = make_load_alias_fixture(12, 4, 64);
  ASSERT_TRUE(fixture);
  const std::vector<uint8_t> pristine = fixture->image;
  const AmdGpuCodeObject source(pristine.data(), pristine.size());
  ASSERT_TRUE(source.is_valid());
  const Section &text = *source.text_sections().front();
  ASSERT_EQ(text.size(), 16u);
  ASSERT_EQ(fixture->alias.p_offset + fixture->alias.p_filesz, text.sectionOffset() + text.size());
  ASSERT_GT(fixture->alias.p_memsz, fixture->alias.p_filesz);
  InstrumentedCodeObjectDebug result = {};
  ASSERT_NO_FATAL_FAILURE(patch_eligible_points(source, result));
  expect_rejected_without_output(result);
  EXPECT_EQ(fixture->image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(BiEditSafety, AllowsTextGrowthThroughEquivalentLoadMapping) {
  const std::optional<LoadAliasFixture> fixture = make_load_alias_fixture(12, 4, 4, true);
  ASSERT_TRUE(fixture);
  const AmdGpuCodeObject source(fixture->image.data(), fixture->image.size());
  ASSERT_TRUE(source.is_valid());
  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point_by_offset(0);
  const InstrumentedCodeObject result = instrumentor.patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 1u);
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(candidate.is_valid());
  const Section &text = *candidate.text_sections().front();
  ASSERT_GT(text.size(), 16u);
  Elf64_Ehdr header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(result.elf_bytes, 0);
  const std::optional<std::vector<Elf64_Phdr>> segments = read_elf_table<Elf64_Phdr>(
      result.elf_bytes, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  const std::vector<Elf64_Phdr>::const_iterator alias =
      std::ranges::find(*segments, fixture->alias.p_vaddr, &Elf64_Phdr::p_vaddr);
  ASSERT_NE(alias, segments->end());
  EXPECT_EQ(alias->p_offset, fixture->alias.p_offset);
  EXPECT_EQ(alias->p_vaddr, text.vaddr() + 12);
  EXPECT_GE(alias->p_filesz, text.size() - 12);
  EXPECT_GE(alias->p_memsz, alias->p_filesz);
  ASSERT_LE(alias->p_offset + text.size() - 12, result.elf_bytes.size());
  EXPECT_EQ(
      std::memcmp(result.elf_bytes.data() + alias->p_offset, text.data() + 12, text.size() - 12),
      0);
}

TEST(BiEditSafety, LoadFileAliasChecksKeepTouchingEmptyAndEquivalentMappingsEligible) {
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
    const AmdGpuCodeObject source(pristine.data(), pristine.size());
    Instrumentor instrumentor(source, kArch);
    instrumentor.add_point_by_offset(4);
    const InstrumentedCodeObject result = instrumentor.patch();
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 1u);
    ASSERT_NO_FATAL_FAILURE(
        expect_load_alias_mapping_preserved(*fixture, result, !control.same_mapping));
    EXPECT_EQ(fixture->image, pristine);
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  }
}

void replay_load_segments(std::span<const uint8_t> image, std::span<uint8_t> loaded) {
  ASSERT_GE(image.size(), sizeof(Elf64_Ehdr));
  const Elf64_Ehdr header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  const std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  // The modern loader shares one zero-initialized allocation between segments.
  std::ranges::fill(loaded, 0);
  for (const Elf64_Phdr &segment : *segments) {
    if (segment.p_type != PT_LOAD)
      continue;
    ASSERT_LE(segment.p_filesz, segment.p_memsz);
    ASSERT_LE(segment.p_vaddr, loaded.size());
    ASSERT_LE(segment.p_memsz, loaded.size() - segment.p_vaddr);
    if (segment.p_filesz == 0)
      continue;
    ASSERT_LE(segment.p_offset, image.size());
    ASSERT_LE(segment.p_filesz, image.size() - segment.p_offset);
    std::memcpy(loaded.data() + segment.p_vaddr, image.data() + segment.p_offset, segment.p_filesz);
  }
}

void expect_anchor_branch_to_cave(const Section &text, uint64_t cave_offset) {
  const std::optional<int16_t> displacement = compute_sopp_branch_simm16(0, cave_offset);
  ASSERT_TRUE(displacement);
  const util::FailureOr<uint32_t> branch = build_s_branch(*displacement, kArch);
  ASSERT_TRUE(branch.succeeded());
  ASSERT_GE(text.size(), sizeof(uint32_t));
  uint32_t anchor_word = 0;
  std::memcpy(&anchor_word, text.data(), sizeof(anchor_word));
  EXPECT_EQ(anchor_word, branch.value());
}

struct UnmovedLoadRange {
  uint64_t vaddr;
  uint64_t file_size;
  uint64_t memory_size;
  bool overlaps_cave;
};

void expect_unmoved_load_growth_policy(const UnmovedLoadRange &range) {
  constexpr uint64_t kTextFileOffset = 0x1ff0;
  constexpr uint64_t kTextVaddr = 0x1ff0;
  constexpr uint64_t kDescriptorVaddr = 0x4000;
  constexpr uint32_t kMarker = 0x12345678;
  const uint64_t marker_file_offset = 0x1000 + range.vaddr % 0x1000;
  ASSERT_LE(range.file_size, sizeof(kMarker));
  ASSERT_LE(range.file_size, range.memory_size);
  std::vector<uint8_t> image =
      test::make_gfx1200_kernel_elf({kNop0Word, kNop1Word, kNop0Word, kEndpgmWord}, 0);
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
  Elf64_Ehdr header = read_hsa_v5_header(image);
  std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  ASSERT_EQ(segments->size(), 2u);
  ASSERT_LE(original_text.sectionOffset(), kTextFileOffset);
  const uint64_t delta = kTextFileOffset - original_text.sectionOffset();
  image.insert(image.begin() + static_cast<std::ptrdiff_t>(original_text.sectionOffset()),
               static_cast<size_t>(delta), 0);
  header.e_shoff += delta;
  for (Elf64_Shdr &section : sections) {
    if (section.sh_type != SHT_NULL && section.sh_offset >= original_text.sectionOffset())
      section.sh_offset += delta;
  }
  for (Elf64_Phdr &segment : *segments)
    segment.p_offset += delta;
  sections[1].sh_addr = kTextVaddr;
  sections[2].sh_addr = kDescriptorVaddr;
  (*segments)[0].p_vaddr = kTextVaddr;
  (*segments)[0].p_paddr = kTextVaddr;
  (*segments)[1].p_vaddr = kDescriptorVaddr;
  (*segments)[1].p_paddr = kDescriptorVaddr;
  // Rebind the existing descriptor and its symbol after moving the fixture.
  amdhsa::kernel_descriptor_t descriptor = {};
  std::memcpy(&descriptor, image.data() + sections[2].sh_offset, sizeof(descriptor));
  descriptor.kernel_code_entry_byte_offset =
      static_cast<int64_t>(kTextVaddr) - static_cast<int64_t>(kDescriptorVaddr);
  std::memcpy(image.data() + sections[2].sh_offset, &descriptor, sizeof(descriptor));
  Elf64_Sym descriptor_symbol = {};
  std::memcpy(&descriptor_symbol, image.data() + sections[3].sh_offset + sizeof(Elf64_Sym),
              sizeof(descriptor_symbol));
  ASSERT_EQ(descriptor_symbol.st_shndx, 2u);
  descriptor_symbol.st_value = kDescriptorVaddr;
  std::memcpy(image.data() + sections[3].sh_offset + sizeof(Elf64_Sym), &descriptor_symbol,
              sizeof(descriptor_symbol));
  segments->push_back({.p_type = PT_LOAD,
                       .p_flags = 0x6, // PF_R | PF_W
                       .p_offset = marker_file_offset,
                       .p_vaddr = range.vaddr,
                       .p_paddr = range.vaddr,
                       .p_filesz = range.file_size,
                       .p_memsz = range.memory_size,
                       .p_align = 0x1000});
  std::ranges::sort(*segments, {}, &Elf64_Phdr::p_vaddr);
  for (const Elf64_Phdr &segment : *segments) {
    ASSERT_EQ(segment.p_offset % segment.p_align, segment.p_vaddr % segment.p_align);
  }
  header.e_phnum = static_cast<uint16_t>(segments->size());
  ASSERT_LT(marker_file_offset + sizeof(kMarker), sections[1].sh_offset);
  ASSERT_EQ(sections[1].sh_addr + sections[1].sh_size, 0x2000u);
  ASSERT_GE(range.vaddr, sections[1].sh_addr + sections[1].sh_size);
  ASSERT_LE(header.e_phoff + segments->size() * sizeof(Elf64_Phdr), marker_file_offset);
  test_support::write_value_for_test(image, marker_file_offset, kMarker);
  write_fixture_headers(image, header, *segments, sections);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  const Section &source_text = *source.text_sections().front();
  std::vector<uint8_t> original_loaded(0x10000, 0);
  ASSERT_NO_FATAL_FAILURE(replay_load_segments(image, original_loaded));
  ASSERT_EQ(std::memcmp(original_loaded.data() + source_text.vaddr(), source_text.data(),
                        source_text.size()),
            0);
  if (range.file_size == 0 && range.memory_size != 0) {
    uint32_t original_zero_fill = kMarker;
    std::memcpy(&original_zero_fill, original_loaded.data() + range.vaddr,
                sizeof(original_zero_fill));
    ASSERT_EQ(original_zero_fill, 0u);
  }
  InstrumentedCodeObjectDebug result = {};
  ASSERT_NO_FATAL_FAILURE(patch_eligible_points(source, result));
  if (!result.elf_bytes.empty()) {
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    ASSERT_EQ(candidate.text_sections().size(), 1u);
    ASSERT_EQ(result.sites.size(), 1u);
    const Section &emitted_text = *candidate.text_sections().front();
    ASSERT_EQ(emitted_text.size(), 28u);
    std::vector<uint8_t> loaded(0x10000, 0);
    ASSERT_NO_FATAL_FAILURE(replay_load_segments(result.elf_bytes, loaded));
    const uint64_t cave_offset = result.sites.front().instrumentation_offset;
    ASSERT_EQ(cave_offset, original_text.size());
    ASSERT_NO_FATAL_FAILURE(expect_anchor_branch_to_cave(emitted_text, cave_offset));
    ASSERT_LE(cave_offset, emitted_text.size());
    ASSERT_LE(sizeof(uint32_t), emitted_text.size() - cave_offset);
    uint32_t file_cave_word = 0;
    std::memcpy(&file_cave_word, emitted_text.data() + cave_offset, sizeof(file_cave_word));
    ASSERT_EQ(file_cave_word, kNop0Word);
    uint32_t loaded_cave_word = 0;
    std::memcpy(&loaded_cave_word, loaded.data() + emitted_text.vaddr() + cave_offset,
                sizeof(loaded_cave_word));
    EXPECT_EQ(loaded_cave_word, file_cave_word);
    if (range.overlaps_cave && range.file_size != 0) {
      EXPECT_EQ(loaded_cave_word, kMarker);
    }
    if (range.overlaps_cave && range.file_size == 0) {
      uint32_t emitted_zero_fill = 0;
      std::memcpy(&emitted_zero_fill, loaded.data() + range.vaddr, sizeof(emitted_zero_fill));
      EXPECT_EQ(emitted_zero_fill, 0u);
    }
    if (range.vaddr == 0x200c) {
      EXPECT_EQ(range.vaddr, emitted_text.vaddr() + emitted_text.size());
    }
  }
  if (range.overlaps_cave) {
    expect_rejected_without_output(result);
  } else {
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 1u);
    EXPECT_FALSE(result.elf_bytes.empty());
  }
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(BiEditSafety, RejectsUnmovedLoadOverwritingAppendedTrampoline) {
  expect_unmoved_load_growth_policy(
      {.vaddr = 0x2000, .file_size = 4, .memory_size = 4, .overlaps_cave = true});
}

TEST(BiEditSafety, AllowsUnmovedLoadClearOfAppendedTrampoline) {
  expect_unmoved_load_growth_policy(
      {.vaddr = 0x3000, .file_size = 4, .memory_size = 4, .overlaps_cave = false});
}

TEST(BiEditSafety, AllowsUnmovedLoadTouchingFinalTextEnd) {
  expect_unmoved_load_growth_policy(
      {.vaddr = 0x200c, .file_size = 4, .memory_size = 4, .overlaps_cave = false});
}

TEST(BiEditSafety, RejectsTextGrowthIntoUnmovedZeroFill) {
  expect_unmoved_load_growth_policy(
      {.vaddr = 0x2000, .file_size = 0, .memory_size = 4, .overlaps_cave = true});
}

TEST(BiEditSafety, AllowsEmptyUnmovedLoadAtAppendedText) {
  expect_unmoved_load_growth_policy(
      {.vaddr = 0x2000, .file_size = 0, .memory_size = 0, .overlaps_cave = false});
}

void expect_unedited_load_alias_growth_policy(uint64_t alias_text_offset, bool overlaps_cave) {
  std::optional<LoadAliasFixture> fixture = make_load_alias_fixture(4, 4, 4);
  ASSERT_TRUE(fixture);
  const AmdGpuCodeObject original(fixture->image.data(), fixture->image.size());
  ASSERT_TRUE(original.is_valid());
  const Section &original_text = *original.text_sections().front();
  Elf64_Ehdr header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(fixture->image, 0);
  std::optional<std::vector<Elf64_Phdr>> segments = read_elf_table<Elf64_Phdr>(
      fixture->image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  const std::vector<Elf64_Phdr>::iterator alias =
      std::ranges::find(*segments, fixture->alias.p_vaddr, &Elf64_Phdr::p_vaddr);
  ASSERT_NE(alias, segments->end());
  alias->p_vaddr = original_text.vaddr() + alias_text_offset;
  alias->p_paddr = alias->p_vaddr;
  alias->p_align = 1;
  std::ranges::sort(*segments, {}, &Elf64_Phdr::p_vaddr);
  ASSERT_GE(alias_text_offset, original_text.size());
  std::memcpy(fixture->image.data() + header.e_phoff, segments->data(),
              segments->size() * sizeof(Elf64_Phdr));
  const std::vector<uint8_t> pristine = fixture->image;
  const AmdGpuCodeObject source(pristine.data(), pristine.size());
  ASSERT_TRUE(source.is_valid());
  const Section &source_text = *source.text_sections().front();
  std::vector<uint8_t> original_loaded(0x10000, 0);
  ASSERT_NO_FATAL_FAILURE(replay_load_segments(pristine, original_loaded));
  ASSERT_EQ(std::memcmp(original_loaded.data() + source_text.vaddr(), source_text.data(),
                        source_text.size()),
            0);
  InstrumentedCodeObjectDebug result = {};
  ASSERT_NO_FATAL_FAILURE(patch_eligible_points(source, result));
  if (!result.elf_bytes.empty()) {
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    ASSERT_EQ(candidate.text_sections().size(), 1u);
    ASSERT_EQ(result.sites.size(), 1u);
    const Section &text = *candidate.text_sections().front();
    ASSERT_EQ(text.size(), 28u);
    const uint64_t cave_offset = result.sites.front().instrumentation_offset;
    ASSERT_EQ(cave_offset, original_text.size());
    ASSERT_NO_FATAL_FAILURE(expect_anchor_branch_to_cave(text, cave_offset));
    const uint64_t return_offset = text.size() - sizeof(uint32_t);
    ASSERT_EQ(return_offset, 24u);
    const std::optional<int16_t> return_displacement =
        compute_sopp_branch_simm16(return_offset, sizeof(uint32_t));
    ASSERT_TRUE(return_displacement);
    const util::FailureOr<uint32_t> return_branch = build_s_branch(*return_displacement, kArch);
    ASSERT_TRUE(return_branch.succeeded());
    uint32_t file_return_word = 0;
    std::memcpy(&file_return_word, text.data() + return_offset, sizeof(file_return_word));
    ASSERT_EQ(file_return_word, return_branch.value());
    std::vector<uint8_t> loaded(0x10000, 0);
    ASSERT_NO_FATAL_FAILURE(replay_load_segments(result.elf_bytes, loaded));
    uint32_t loaded_return_word = 0;
    std::memcpy(&loaded_return_word, loaded.data() + text.vaddr() + return_offset,
                sizeof(loaded_return_word));
    EXPECT_EQ(loaded_return_word, file_return_word);
    if (overlaps_cave) {
      EXPECT_EQ(loaded_return_word, kNop1Word);
    }
    EXPECT_EQ(std::memcmp(loaded.data() + text.vaddr(), text.data(), text.size()), 0);
  }
  if (overlaps_cave) {
    expect_rejected_without_output(result);
  } else {
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 1u);
    EXPECT_FALSE(result.elf_bytes.empty());
  }
  EXPECT_EQ(fixture->image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(BiEditSafety, RejectsUneditedLoadAliasOverwritingAppendedReturnBranch) {
  expect_unedited_load_alias_growth_policy(24, true);
}

TEST(BiEditSafety, AllowsUneditedLoadAliasClearOfAppendedText) {
  expect_unedited_load_alias_growth_policy(256, false);
}

enum class AnchorLoadBacking { SeparateCopy, SameFile, UneditedWord, ZeroFill };

void expect_anchor_load_backing_policy(AnchorLoadBacking backing) {
  std::vector<uint8_t> image =
      test::make_gfx1200_kernel_elf({kNop0Word, kNop1Word, kNop0Word, kEndpgmWord}, 0);
  const AmdGpuCodeObject original(image.data(), image.size());
  ASSERT_TRUE(original.is_valid());
  const Section &text = *original.text_sections().front();
  const uint64_t word_offset = backing == AnchorLoadBacking::UneditedWord ? 4 : 0;
  const uint64_t file_offset =
      backing == AnchorLoadBacking::SameFile ? text.sectionOffset() : 0x2100 + word_offset;
  const uint64_t file_size = backing == AnchorLoadBacking::ZeroFill ? 0 : 4;
  image.resize(0x2108, 0);
  if (backing != AnchorLoadBacking::SameFile)
    std::memcpy(image.data() + file_offset, text.data() + word_offset, sizeof(uint32_t));
  Elf64_Ehdr header = read_hsa_v5_header(image);
  std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  segments->push_back({.p_type = PT_LOAD,
                       .p_flags = 0x4,
                       .p_offset = file_offset,
                       .p_vaddr = text.vaddr() + word_offset,
                       .p_paddr = text.vaddr() + word_offset,
                       .p_filesz = file_size,
                       .p_memsz = sizeof(uint32_t),
                       .p_align = 0x1000});
  std::stable_sort(
      segments->begin(), segments->end(),
      [](const Elf64_Phdr &left, const Elf64_Phdr &right) { return left.p_vaddr < right.p_vaddr; });
  header.e_phnum = static_cast<uint16_t>(segments->size());
  ASSERT_LE(header.e_phoff + segments->size() * sizeof(Elf64_Phdr), text.sectionOffset());
  for (const Elf64_Phdr &segment : *segments)
    ASSERT_EQ(segment.p_offset % segment.p_align, segment.p_vaddr % segment.p_align);
  write_fixture_headers(image, header, *segments);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  std::vector<uint8_t> loaded(0x10000, 0);
  ASSERT_NO_FATAL_FAILURE(replay_load_segments(image, loaded));
  ASSERT_EQ(std::memcmp(loaded.data() + text.vaddr(), text.data(), text.size()), 0);
  InstrumentedCodeObjectDebug result = {};
  ASSERT_NO_FATAL_FAILURE(patch_eligible_points(source, result));
  if (!result.elf_bytes.empty()) {
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    ASSERT_EQ(result.sites.size(), 1u);
    const Section &emitted_text = *candidate.text_sections().front();
    ASSERT_NO_FATAL_FAILURE(
        expect_anchor_branch_to_cave(emitted_text, result.sites.front().instrumentation_offset));
    ASSERT_NO_FATAL_FAILURE(replay_load_segments(result.elf_bytes, loaded));
    uint32_t file_anchor = 0;
    uint32_t loaded_anchor = 0;
    std::memcpy(&file_anchor, emitted_text.data(), sizeof(file_anchor));
    std::memcpy(&loaded_anchor, loaded.data() + text.vaddr(), sizeof(loaded_anchor));
    EXPECT_EQ(loaded_anchor, file_anchor);
    EXPECT_EQ(std::memcmp(loaded.data() + text.vaddr(), emitted_text.data(), emitted_text.size()),
              0);
    if (backing == AnchorLoadBacking::SeparateCopy) {
      EXPECT_EQ(loaded_anchor, kNop0Word);
    }
  }
  if (backing == AnchorLoadBacking::SeparateCopy) {
    expect_rejected_without_output(result);
    ASSERT_TRUE(result.edit_error);
    EXPECT_EQ(result.edit_error->code, TextEditErrorCode::InvalidImage);
    EXPECT_EQ(result.edit_error->offset, 0u);
  } else {
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    EXPECT_EQ(result.sites.size(), 1u);
  }
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(BiEditSafety, RejectsSeparateFileCopyOverwritingEditedAnchor) {
  expect_anchor_load_backing_policy(AnchorLoadBacking::SeparateCopy);
}

TEST(BiEditSafety, KeepsEquivalentAnchorMappingsEligible) {
  expect_anchor_load_backing_policy(AnchorLoadBacking::SameFile);
}

TEST(BiEditSafety, KeepsSeparateFileCopyOfUneditedWordEligible) {
  expect_anchor_load_backing_policy(AnchorLoadBacking::UneditedWord);
}

TEST(BiEditSafety, KeepsZeroFileAnchorMappingEligible) {
  expect_anchor_load_backing_policy(AnchorLoadBacking::ZeroFill);
}

enum class BssLoadLayout { None, Fixed, MovedWithEmptyMapping };

struct GrowthLoadContract {
  uint64_t descriptor_file_offset;
  uint64_t descriptor_vaddr;
  uint64_t tail_file_offset;
  uint64_t tail_vaddr;
  bool accepted;
  BssLoadLayout bss = BssLoadLayout::None;
  uint64_t descriptor_load_file_offset = 0;
};

void expect_growth_load_contract(const GrowthLoadContract &layout) {
  constexpr uint64_t kTextOffset = 0x1ff0;
  constexpr uint64_t kTextVaddr = 0x1ff0;
  constexpr uint32_t kDataWord = 0x12345678;
  const std::vector<uint8_t> original_image =
      test::make_gfx1200_kernel_elf({kNop0Word, kNop1Word, kNop0Word, kEndpgmWord}, 0);
  const AmdGpuCodeObject original(original_image.data(), original_image.size());
  ASSERT_TRUE(original.is_valid());
  const CodeObjectPatcher original_patcher(original);
  std::vector<Elf64_Shdr> sections = original_patcher.section_headers();
  ASSERT_EQ(sections.size(), 6u);
  Elf64_Ehdr header = read_hsa_v5_header(original_image);
  std::vector<uint8_t> image(0x8000, 0);
  uint64_t metadata_offset = 0x5000;
  for (size_t index = 1; index < sections.size(); ++index) {
    Elf64_Shdr &section = sections[index];
    const uint64_t old_offset = section.sh_offset;
    section.sh_offset =
        index == 1 ? kTextOffset : (index == 2 ? layout.descriptor_file_offset : metadata_offset);
    ASSERT_LE(old_offset, original_image.size());
    ASSERT_LE(section.sh_size, original_image.size() - old_offset);
    ASSERT_LE(section.sh_offset, image.size());
    ASSERT_LE(section.sh_size, image.size() - section.sh_offset);
    std::memcpy(image.data() + section.sh_offset, original_image.data() + old_offset,
                section.sh_size);
    if (index >= 3)
      metadata_offset = test::align_up_for_test(section.sh_offset + section.sh_size, 8);
  }
  sections[1].sh_addr = kTextVaddr;
  sections[2].sh_addr = layout.descriptor_vaddr;
  amdhsa::kernel_descriptor_t descriptor = {};
  std::memcpy(&descriptor, image.data() + sections[2].sh_offset, sizeof(descriptor));
  descriptor.kernel_code_entry_byte_offset =
      static_cast<int64_t>(kTextVaddr) - static_cast<int64_t>(layout.descriptor_vaddr);
  std::memcpy(image.data() + sections[2].sh_offset, &descriptor, sizeof(descriptor));
  const uint64_t descriptor_load_offset = layout.descriptor_load_file_offset == 0
                                              ? layout.descriptor_file_offset
                                              : layout.descriptor_load_file_offset;
  std::memcpy(image.data() + descriptor_load_offset, &descriptor, sizeof(descriptor));
  Elf64_Sym symbol = {};
  std::memcpy(&symbol, image.data() + sections[3].sh_offset + sizeof(Elf64_Sym), sizeof(symbol));
  ASSERT_EQ(symbol.st_shndx, 2u);
  symbol.st_value = layout.descriptor_vaddr;
  std::memcpy(image.data() + sections[3].sh_offset + sizeof(Elf64_Sym), &symbol, sizeof(symbol));
  std::vector<Elf64_Phdr> segments = {{.p_type = PT_LOAD,
                                       .p_flags = 0x4,
                                       .p_offset = 0,
                                       .p_vaddr = 0,
                                       .p_paddr = 0,
                                       .p_filesz = 0x200,
                                       .p_memsz = 0x200,
                                       .p_align = 0x1000},
                                      {.p_type = PT_LOAD,
                                       .p_flags = 0x5,
                                       .p_offset = kTextOffset,
                                       .p_vaddr = kTextVaddr,
                                       .p_paddr = kTextVaddr,
                                       .p_filesz = 16,
                                       .p_memsz = 16,
                                       .p_align = 0x1000},
                                      {.p_type = PT_LOAD,
                                       .p_flags = 0x4,
                                       .p_offset = descriptor_load_offset,
                                       .p_vaddr = layout.descriptor_vaddr,
                                       .p_paddr = layout.descriptor_vaddr,
                                       .p_filesz = sizeof(descriptor),
                                       .p_memsz = sizeof(descriptor),
                                       .p_align = 0x1000},
                                      {.p_type = PT_LOAD,
                                       .p_flags = 0x6,
                                       .p_offset = layout.tail_file_offset,
                                       .p_vaddr = layout.tail_vaddr,
                                       .p_paddr = layout.tail_vaddr,
                                       .p_filesz = sizeof(kDataWord),
                                       .p_memsz = sizeof(kDataWord),
                                       .p_align = 0x1000}};
  if (layout.bss != BssLoadLayout::None) {
    const uint64_t file_offset = layout.bss == BssLoadLayout::Fixed ? 0x1000 : 0x2000;
    sections.push_back({.sh_name = 0,
                        .sh_type = SHT_NOBITS,
                        .sh_flags = SHF_ALLOC | SHF_WRITE,
                        .sh_addr = 0x6000,
                        .sh_offset = file_offset,
                        .sh_size = sizeof(uint32_t),
                        .sh_link = 0,
                        .sh_info = 0,
                        .sh_addralign = alignof(uint32_t),
                        .sh_entsize = 0});
    segments.push_back({.p_type = PT_LOAD,
                        .p_flags = 0x6,
                        .p_offset = file_offset,
                        .p_vaddr = 0x6000,
                        .p_paddr = 0x6000,
                        .p_filesz = 0,
                        .p_memsz = sizeof(uint32_t),
                        .p_align = 0x1000});
    if (layout.bss == BssLoadLayout::MovedWithEmptyMapping) {
      segments.push_back({.p_type = PT_LOAD,
                          .p_flags = 0x4,
                          .p_offset = file_offset + 2,
                          .p_vaddr = 0x6002,
                          .p_paddr = 0x6002,
                          .p_filesz = 0,
                          .p_memsz = 0,
                          .p_align = 1});
    }
  }
  std::ranges::sort(segments, {}, &Elf64_Phdr::p_vaddr);
  for (const Elf64_Phdr &segment : segments)
    ASSERT_EQ(segment.p_offset % segment.p_align, segment.p_vaddr % segment.p_align);
  header.e_phnum = static_cast<uint16_t>(segments.size());
  header.e_shnum = static_cast<uint16_t>(sections.size());
  header.e_shoff = metadata_offset;
  ASSERT_LE(header.e_phoff + segments.size() * sizeof(Elf64_Phdr), 0x200u);
  test_support::write_value_for_test(image, layout.tail_file_offset, kDataWord);
  write_fixture_headers(image, header, segments, sections);
  image.resize(header.e_shoff + sections.size() * sizeof(Elf64_Shdr));
  const uint64_t original_allocation = segments.back().p_vaddr + segments.back().p_memsz;
  for (const Elf64_Phdr &segment : segments)
    ASSERT_LE(segment.p_vaddr + segment.p_memsz, original_allocation);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  std::vector<uint8_t> loaded(0x10000, 0);
  ASSERT_NO_FATAL_FAILURE(replay_load_segments(image, loaded));
  ASSERT_EQ(std::memcmp(loaded.data() + layout.descriptor_vaddr, &descriptor, sizeof(descriptor)),
            0);
  InstrumentedCodeObjectDebug result = {};
  ASSERT_NO_FATAL_FAILURE(patch_eligible_points(source, result));
  if (!result.elf_bytes.empty()) {
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    ASSERT_EQ(result.sites.size(), 1u);
    const Section &text = *candidate.text_sections().front();
    ASSERT_EQ(text.vaddr() + text.size(), 0x200cu);
    ASSERT_NO_FATAL_FAILURE(
        expect_anchor_branch_to_cave(text, result.sites.front().instrumentation_offset));
    const CodeObjectPatcher emitted_patcher(candidate);
    const std::vector<Elf64_Shdr> emitted_sections = emitted_patcher.section_headers();
    header = test_support::read_elf_struct_for_test<Elf64_Ehdr>(result.elf_bytes, 0);
    const std::optional<std::vector<Elf64_Phdr>> emitted_segments = read_elf_table<Elf64_Phdr>(
        result.elf_bytes, header.e_phoff, header.e_phnum, header.e_phentsize);
    ASSERT_TRUE(emitted_segments);
    ASSERT_EQ(emitted_segments->front().p_vaddr, 0u);
    const uint64_t allocation = emitted_segments->back().p_vaddr + emitted_segments->back().p_memsz;
    for (const Elf64_Phdr &segment : *emitted_segments)
      EXPECT_LE(segment.p_vaddr + segment.p_memsz, allocation);
    ASSERT_NO_FATAL_FAILURE(replay_load_segments(result.elf_bytes, loaded));
    EXPECT_EQ(std::memcmp(loaded.data() + text.vaddr(), text.data(), text.size()), 0);
    const Elf64_Shdr &emitted_descriptor = emitted_sections[2];
    EXPECT_EQ(std::memcmp(loaded.data() + emitted_descriptor.sh_addr,
                          result.elf_bytes.data() + emitted_descriptor.sh_offset,
                          sizeof(descriptor)),
              0);
    Elf64_Sym emitted_symbol = {};
    std::memcpy(&emitted_symbol,
                result.elf_bytes.data() + emitted_sections[3].sh_offset + sizeof(Elf64_Sym),
                sizeof(emitted_symbol));
    EXPECT_EQ(emitted_symbol.st_value, emitted_descriptor.sh_addr);
    amdhsa::kernel_descriptor_t loaded_descriptor = {};
    std::memcpy(&loaded_descriptor, loaded.data() + emitted_symbol.st_value,
                sizeof(loaded_descriptor));
    EXPECT_EQ(emitted_symbol.st_value + loaded_descriptor.kernel_code_entry_byte_offset,
              kTextVaddr);
    if (layout.bss != BssLoadLayout::None) {
      EXPECT_EQ(emitted_sections.back().sh_type, SHT_NOBITS);
      EXPECT_EQ(emitted_sections.back().sh_addr, 0x7000u);
      uint32_t zero_fill = 1;
      std::memcpy(&zero_fill, loaded.data() + emitted_sections.back().sh_addr, sizeof(zero_fill));
      EXPECT_EQ(zero_fill, 0u);
    }
    if (layout.descriptor_vaddr == 0x4000) {
      EXPECT_EQ(emitted_descriptor.sh_addr, 0x5000u);
    } else {
      EXPECT_EQ(allocation, layout.tail_vaddr + sizeof(kDataWord));
      const std::vector<Elf64_Phdr>::const_iterator text_load =
          std::ranges::find(*emitted_segments, kTextVaddr, &Elf64_Phdr::p_vaddr);
      ASSERT_NE(text_load, emitted_segments->end());
      EXPECT_EQ(text_load->p_vaddr + text_load->p_memsz, 0x3000u);
    }
  }
  if (layout.accepted) {
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    EXPECT_EQ(result.sites.size(), 1u);
  } else {
    expect_rejected_without_output(result);
    EXPECT_FALSE(result.edit_error);
  }
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(BiEditSafety, RejectsMovedDescriptorLosingItsLoadBacking) {
  expect_growth_load_contract({.descriptor_file_offset = 0x1000,
                               .descriptor_vaddr = 0x4000,
                               .tail_file_offset = 0x4000,
                               .tail_vaddr = 0x6000,
                               .accepted = false});
}

TEST(BiEditSafety, KeepsMovedDescriptorWithMatchingLoadBackingEligible) {
  expect_growth_load_contract({.descriptor_file_offset = 0x3000,
                               .descriptor_vaddr = 0x4000,
                               .tail_file_offset = 0x4000,
                               .tail_vaddr = 0x6000,
                               .accepted = true});
}

TEST(BiEditSafety, RejectsPaddedLoadBeyondRocrAllocation) {
  expect_growth_load_contract({.descriptor_file_offset = 0x3000,
                               .descriptor_vaddr = 0x1000,
                               .tail_file_offset = 0x1200,
                               .tail_vaddr = 0x2200,
                               .accepted = false});
}

TEST(BiEditSafety, KeepsPaddedLoadWithinRocrAllocationEligible) {
  expect_growth_load_contract({.descriptor_file_offset = 0x3000,
                               .descriptor_vaddr = 0x1000,
                               .tail_file_offset = 0x1200,
                               .tail_vaddr = 0x3200,
                               .accepted = true});
}

TEST(BiEditSafety, RejectsMovedZeroFillLosingItsLoadBacking) {
  expect_growth_load_contract({.descriptor_file_offset = 0x3000,
                               .descriptor_vaddr = 0x4000,
                               .tail_file_offset = 0x4000,
                               .tail_vaddr = 0x8000,
                               .accepted = false,
                               .bss = BssLoadLayout::Fixed});
}

TEST(BiEditSafety, KeepsMovedZeroFillWithEmptyLoadEligible) {
  expect_growth_load_contract({.descriptor_file_offset = 0x3000,
                               .descriptor_vaddr = 0x4000,
                               .tail_file_offset = 0x4000,
                               .tail_vaddr = 0x8000,
                               .accepted = true,
                               .bss = BssLoadLayout::MovedWithEmptyMapping});
}

TEST(BiEditSafety, RejectsMovedDescriptorWithUntrackedSourceBacking) {
  expect_growth_load_contract({.descriptor_file_offset = 0x3000,
                               .descriptor_vaddr = 0x4000,
                               .tail_file_offset = 0x4000,
                               .tail_vaddr = 0x6000,
                               .accepted = false,
                               .descriptor_load_file_offset = 0x1000});
}

TEST(BiEditSafety, RejectsDescriptorWithoutDeclaredBackingInOrderedLoads) {
  expect_growth_load_contract({.descriptor_file_offset = 0x1000,
                               .descriptor_vaddr = 0x4000,
                               .tail_file_offset = 0x4000,
                               .tail_vaddr = 0x6000,
                               .accepted = false,
                               .descriptor_load_file_offset = 0x3000});
}

} // namespace
} // namespace rocjitsu
