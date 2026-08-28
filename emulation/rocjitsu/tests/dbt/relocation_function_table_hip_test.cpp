// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file relocation_function_table_hip_test.cpp
/// @brief End-to-end DBT coverage for a HIP-emitted device-function table.

#ifndef HAS_GFX1250_DEVICE_KERNELS
#error "relocation_function_table_hip_test.cpp requires a gfx1250-capable device compiler"
#endif

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/processor_revision.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/isa/decoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

std::string relocation_table_kernel_path() {
  return std::string(KERNEL_DIR) + "/relocation_function_table_dispatch.o";
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

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  const auto blocks =
      rocjitsu::BasicBlock::build(*source, *decoder, ROCJITSU_CODE_ARCH_GFX1250, source_targets);
  const auto dispatches = rocjitsu::discover_relocation_table_dispatches(
      blocks, source_tables, source->text_sections()[0]->vaddr());
  ASSERT_EQ(dispatches.size(), 1u);
  EXPECT_EQ(dispatches[0].table_index, 0u);

  rocjitsu::BinaryTranslatorOptions options;
  options.input_revision = rocjitsu::ProcessorRevision::Gfx1250B0;
  options.output_revision = rocjitsu::ProcessorRevision::Gfx1250A0;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
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
