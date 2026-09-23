// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file relocation_function_table_hip_test.cpp
/// @brief End-to-end DBT coverage for a HIP-emitted device-function table.

#ifndef HAS_GFX1250_DEVICE_KERNELS
#error "relocation_function_table_hip_test.cpp requires a gfx1250-capable device compiler"
#endif

#include "decode_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/processor_revision.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/isa/decoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

std::string relocation_table_kernel_path() {
  return std::string(KERNEL_DIR) + "/relocation_function_table_dispatch.o";
}

void append_nonallocated_relative64(std::vector<uint8_t> &image, uint64_t slot_vaddr,
                                    uint64_t target_vaddr) {
  rocjitsu::Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  std::vector<rocjitsu::Elf64_Shdr> sections(ehdr.e_shnum);
  std::memcpy(sections.data(), image.data() + ehdr.e_shoff,
              sections.size() * sizeof(rocjitsu::Elf64_Shdr));

  const uint64_t relocation_offset = ehdr.e_shoff;
  rocjitsu::Elf64_Rela relocation{};
  relocation.r_offset = slot_vaddr;
  relocation.r_info = rocjitsu::R_AMDGPU_RELATIVE64;
  relocation.r_addend = static_cast<int64_t>(target_vaddr);
  image.insert(image.begin() + static_cast<std::ptrdiff_t>(relocation_offset),
               reinterpret_cast<const uint8_t *>(&relocation),
               reinterpret_cast<const uint8_t *>(&relocation) + sizeof(relocation));

  rocjitsu::Elf64_Shdr metadata_relocations{};
  metadata_relocations.sh_type = rocjitsu::SHT_RELA;
  metadata_relocations.sh_offset = relocation_offset;
  metadata_relocations.sh_size = sizeof(relocation);
  metadata_relocations.sh_info = ehdr.e_shstrndx;
  metadata_relocations.sh_addralign = alignof(rocjitsu::Elf64_Rela);
  metadata_relocations.sh_entsize = sizeof(rocjitsu::Elf64_Rela);
  sections.push_back(metadata_relocations);

  ehdr.e_shoff += sizeof(relocation);
  ehdr.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(ehdr.e_shoff + sections.size() * sizeof(rocjitsu::Elf64_Shdr));
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));
  std::memcpy(image.data() + ehdr.e_shoff, sections.data(),
              sections.size() * sizeof(rocjitsu::Elf64_Shdr));
}

} // namespace

TEST(RelocationFunctionTableHip, TranslatesEightWayDeviceDispatch) {
  rocjitsu::Executable executable(relocation_table_kernel_path());
  ASSERT_TRUE(executable.is_valid()) << "failed to load relocation table HIP fixture";
  ASSERT_EQ(executable.num_code_objects(ROCJITSU_CODE_TARGET_GFX1250), 1u);

  const auto *source = executable.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  ASSERT_NE(source, nullptr);
  ASSERT_EQ(source->text_sections().size(), 1u);
  EXPECT_NE(source->kernel_descriptor_offset("relocation_function_table_dispatch"), 0u);

  const auto source_tables = rocjitsu::discover_relocation_function_tables(*source);
  ASSERT_EQ(source_tables.size(), 1u);
  EXPECT_EQ(source_tables[0].table_size, 8u * sizeof(uint64_t));
  EXPECT_TRUE(source_tables[0].got_slot_vaddrs.empty());
  ASSERT_EQ(source_tables[0].entries.size(), 8u);

  // Every slot must name a distinct, exactly decoded function entry. This
  // checks the real linked ELF rather than relying on source declarations to
  // prove that the compiler retained all eight address-taken functions.
  std::vector<uint64_t> source_targets;
  source_targets.reserve(source_tables[0].entries.size());
  for (const auto &entry : source_tables[0].entries)
    source_targets.push_back(entry.target_text_offset);
  std::ranges::sort(source_targets);
  EXPECT_EQ(std::ranges::adjacent_find(source_targets), source_targets.end());

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const auto blocks =
      rocjitsu::build_valid_blocks(*source, *decoder, ROCJITSU_CODE_ARCH_CDNA5, source_targets);
  const auto dispatches =
      rocjitsu::analyze_relocation_pairs(blocks, source_tables, source->text_sections()[0]->vaddr())
          .dispatches;
  ASSERT_EQ(dispatches.size(), 1u);
  EXPECT_EQ(dispatches[0].table_index, 0u);

  rocjitsu::BinaryTranslatorOptions options;
  options.input_revision = rocjitsu::ProcessorRevision::Gfx1250B0;
  options.output_revision = rocjitsu::ProcessorRevision::Gfx1250A0;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  auto result = translator.translate(*source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? "translation failed without diagnostics"
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto translated_tables = rocjitsu::discover_relocation_function_tables(translated);
  ASSERT_EQ(translated_tables.size(), 1u);
  EXPECT_EQ(translated_tables[0].entries.size(), 8u);
  EXPECT_EQ(translated_tables[0].table_size, 8u * sizeof(uint64_t));

  const uint64_t translated_text_size = translated.text_sections()[0]->size();
  for (const auto &entry : translated_tables[0].entries)
    EXPECT_LT(entry.target_text_offset, translated_text_size);
}

TEST(RelocationFunctionTableHip, TranslationIgnoresExplicitNonAllocatedRelocation) {
  rocjitsu::Executable executable(relocation_table_kernel_path());
  ASSERT_TRUE(executable.is_valid()) << "failed to load relocation table HIP fixture";
  const auto *source = executable.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  ASSERT_NE(source, nullptr);
  ASSERT_EQ(source->text_sections().size(), 1u);

  const auto source_tables = rocjitsu::discover_relocation_function_tables(*source);
  ASSERT_EQ(source_tables.size(), 1u);
  ASSERT_EQ(source_tables[0].entries.size(), 8u);
  const uint64_t invalid_mid_instruction_target = source->text_sections()[0]->vaddr() + 2;
  const auto *source_bytes = reinterpret_cast<const uint8_t *>(source->image_data());
  std::vector<uint8_t> image(source_bytes, source_bytes + source->image_size());
  append_nonallocated_relative64(image, source_tables[0].entries[0].slot_vaddr,
                                 invalid_mid_instruction_target);

  rocjitsu::AmdGpuCodeObject augmented(image.data(), image.size());
  ASSERT_TRUE(augmented.is_valid());
  const auto augmented_tables = rocjitsu::discover_relocation_function_tables(augmented);
  ASSERT_EQ(augmented_tables.size(), 1u);
  EXPECT_EQ(augmented_tables[0].entries.size(), 8u);

  rocjitsu::BinaryTranslatorOptions options;
  options.input_revision = rocjitsu::ProcessorRevision::Gfx1250B0;
  options.output_revision = rocjitsu::ProcessorRevision::Gfx1250A0;
  options.verify_rewrite_discharge = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  const auto result = translator.translate(augmented);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_TRUE(result.rewrite_discharge_verified);
}
