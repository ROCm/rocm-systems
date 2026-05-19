// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_patcher_test.cpp
/// @brief CPU-only tests for DBT ELF patching and code cave placement.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/encoding_fields.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna1.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna2_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna2_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_5_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna4_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "dbt_test_utils.h"

namespace rocjitsu {
namespace {
using namespace dbt_test;
TEST(CodeObjectPatcher, CaveBodyMaterializesInRjTranslationsAfterText) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::array<uint32_t, 2> cave_words = {0xDEADBEEFu, 0xCAFEBABEu};
  patcher.append_cave_body(cave_words);
  ASSERT_TRUE(patcher.append_cave_section());

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());

  const Section *text = patched.text_sections()[0];
  ASSERT_NE(text, nullptr);
  EXPECT_EQ(text->size(), 8u) << ".text must keep its original size";

  const Section *translations = find_section(patched, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(patched.code_sections().size(), 2u);
  EXPECT_EQ(patched.code_sections()[0]->name(), ".text");
  EXPECT_EQ(patched.code_sections()[1]->name(), ".rj_translations");
  EXPECT_EQ(translations->sectionOffset(), text->sectionOffset() + text->size())
      << ".rj_translations must be physically placed immediately after .text";
  ASSERT_EQ(translations->size(), cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(translations->data(), cave_words.data(), translations->size()), 0);

  const Section *rodata = find_section(patched, ".rodata");
  ASSERT_NE(rodata, nullptr);
  EXPECT_EQ(rodata->sectionOffset(), translations->sectionOffset() + translations->size())
      << "sections following .text must be shifted after the cave section";
  uint32_t rodata_word = 0;
  std::memcpy(&rodata_word, rodata->data(), sizeof(rodata_word));
  EXPECT_EQ(rodata_word, 0xA5A55A5Au);
}

TEST(CodeObjectPatcher, CaveInsertionPreservesLoadSegmentAlignment) {
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t padded_file_delta = 2 * load_align;

  auto image = make_minimal_amdgpu_elf_with_load_segments();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::vector<uint32_t> cave_words(load_align / sizeof(uint32_t) + 1, 0xDEADBEEFu);
  patcher.append_cave_body(cave_words);
  ASSERT_TRUE(patcher.append_cave_section());

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());

  const Section *text = patched.text_sections()[0];
  const Section *translations = find_section(patched, ".rj_translations");
  const Section *rodata = find_section(patched, ".rodata");
  ASSERT_NE(translations, nullptr);
  ASSERT_NE(rodata, nullptr);
  EXPECT_EQ(translations->sectionOffset(), text->sectionOffset() + text->size());
  EXPECT_EQ(translations->size(), cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(translations->vaddr(), text->vaddr() + text->size());

  EXPECT_EQ(rodata->sectionOffset(), text->sectionOffset() + text->size() + padded_file_delta)
      << "file padding should preserve later PT_LOAD p_offset/p_vaddr congruence";
  EXPECT_EQ(rodata->vaddr(), text->vaddr() + text->size() + load_align + padded_file_delta)
      << "later allocated sections must move after the expanded RX LOAD segment";

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(patched_bytes.data());
  ASSERT_EQ(ehdr->e_phnum, 2u);
  const auto *phdrs = reinterpret_cast<const Elf64_Phdr *>(patched_bytes.data() + ehdr->e_phoff);
  const auto *shdrs = reinterpret_cast<const Elf64_Shdr *>(patched_bytes.data() + ehdr->e_shoff);
  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    ASSERT_NE(phdrs[i].p_align, 0u);
    EXPECT_EQ(phdrs[i].p_offset % phdrs[i].p_align, phdrs[i].p_vaddr % phdrs[i].p_align)
        << "PT_LOAD " << i << " must remain loader-congruent";
  }
  EXPECT_EQ(phdrs[0].p_filesz, text->size() + padded_file_delta);
  EXPECT_EQ(phdrs[0].p_memsz, text->size() + padded_file_delta);
  EXPECT_EQ(phdrs[1].p_offset, rodata->sectionOffset());
  EXPECT_EQ(phdrs[1].p_vaddr, rodata->vaddr());
  EXPECT_EQ(phdrs[1].p_paddr, rodata->vaddr());
  EXPECT_LE(phdrs[0].p_vaddr + phdrs[0].p_memsz, phdrs[1].p_vaddr)
      << "expanded RX LOAD must not overlap the following LOAD in virtual memory";

  const auto *symtab = std::find_if(shdrs, shdrs + ehdr->e_shnum, [](const Elf64_Shdr &shdr) {
    return shdr.sh_type == SHT_SYMTAB;
  });
  ASSERT_NE(symtab, shdrs + ehdr->e_shnum);
  ASSERT_EQ(symtab->sh_entsize, sizeof(Elf64_Sym));
  ASSERT_GE(symtab->sh_size / symtab->sh_entsize, 3u);
  const auto *symbols =
      reinterpret_cast<const Elf64_Sym *>(patched_bytes.data() + symtab->sh_offset);
  EXPECT_EQ(symbols[1].st_value, rodata->vaddr())
      << "defined symbols in moved sections must track the section virtual address";
  EXPECT_EQ(symbols[2].st_value, text->vaddr())
      << "symbols in unmoved .text must keep their original virtual address";
}

TEST(CodeObjectPatcher, CaveInsertionPreservesMovedKernelDescriptorEntryAddress) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t padded_file_delta = 2 * load_align;

  auto image = make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());
  const auto *original_rodata = find_section(co, ".rodata");
  ASSERT_NE(original_rodata, nullptr);
  KD original_kd{};
  std::memcpy(&original_kd, original_rodata->data(), sizeof(original_kd));

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::vector<uint32_t> cave_words(load_align / sizeof(uint32_t) + 1, 0xDEADBEEFu);
  patcher.append_cave_body(cave_words);
  ASSERT_TRUE(patcher.append_cave_section());

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *patched_text = patched.text_sections()[0];
  const auto *patched_rodata = find_section(patched, ".rodata");
  ASSERT_NE(patched_rodata, nullptr);

  KD patched_kd{};
  std::memcpy(&patched_kd, patched_rodata->data(), sizeof(patched_kd));
  EXPECT_EQ(patched_rodata->vaddr(), original_rodata->vaddr() + padded_file_delta);
  EXPECT_EQ(static_cast<uint64_t>(static_cast<int64_t>(patched_rodata->vaddr()) +
                                  patched_kd.kernel_code_entry_byte_offset),
            patched_text->vaddr())
      << "KERNEL_CODE_ENTRY_BYTE_OFFSET is relative to the descriptor address";
  EXPECT_EQ(patched_kd.kernel_code_entry_byte_offset,
            original_kd.kernel_code_entry_byte_offset - static_cast<int64_t>(padded_file_delta));
}

TEST(CodeObjectPatcher, CaveInsertionUpdatesRelocationOffsetsIntoMovedSections) {
  constexpr uint64_t load_align = 0x1000;

  auto image = make_minimal_amdgpu_elf_with_relocation_after_text();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::vector<uint32_t> cave_words(load_align / sizeof(uint32_t) + 1, 0xDEADBEEFu);
  patcher.append_cave_body(cave_words);
  ASSERT_TRUE(patcher.append_cave_section());

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());

  const auto *data = find_section(patched, ".data");
  const auto *rela_dyn = find_section(patched, ".rela.dyn");
  ASSERT_NE(data, nullptr);
  ASSERT_NE(rela_dyn, nullptr);
  ASSERT_EQ(rela_dyn->size(), sizeof(Elf64_Rela));

  Elf64_Rela rela{};
  std::memcpy(&rela, rela_dyn->data(), sizeof(rela));
  EXPECT_EQ(rela.r_offset, data->vaddr())
      << "ET_DYN relocation r_offset is the relocated storage address";
}

TEST(BinaryTranslator, CaveBranchOverflowLeavesCodeObjectUnchanged) {
  auto image = make_large_amdgpu_elf_with_waitcnt_entry();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(co);

  EXPECT_EQ(result.elf_bytes, image);
  const bool warned =
      std::any_of(result.warnings.begin(), result.warnings.end(), [](const std::string &warning) {
        return warning.find("branch range") != std::string::npos &&
               warning.find("leaving code object unchanged") != std::string::npos;
      });
  EXPECT_TRUE(warned);
}

} // namespace
} // namespace rocjitsu
