// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file isa_registry_composition_test.cpp
/// @brief Multi-component static ISA registry composition tests.

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>

extern "C" size_t rj_test_narrow_target_count();
extern "C" bool rj_test_narrow_has_target(const char *id);
extern "C" size_t rj_test_downstream_target_count();
extern "C" bool rj_test_downstream_has_target(const char *id);

namespace {

TEST(IsaRegistryCompositionTest, ComponentsKeepIndependentStaticSubsets) {
  using namespace rocjitsu;

  const IsaTargetRegistry &full = default_isa_target_registry();
  const IsaTargetDescriptor *full_gfx1250 = full.find("gfx1250");
  ASSERT_NE(full_gfx1250, nullptr);
  EXPECT_TRUE(full_gfx1250->supports_execution);

  constexpr uint32_t kSNop = 0xBF800000u;
  std::unique_ptr<Decoder> full_decoder = Decoder::create(full, "gfx1250");
  ASSERT_NE(full_decoder, nullptr);
  std::unique_ptr<Instruction> full_instruction(full_decoder->decode(&kSNop));
  ASSERT_NE(full_instruction, nullptr);
  EXPECT_NE(full_instruction->execute, nullptr);

  ASSERT_EQ(rj_test_narrow_target_count(), 1u);
  EXPECT_TRUE(rj_test_narrow_has_target("gfx1250"));
  EXPECT_FALSE(rj_test_narrow_has_target("rdna4"));
  EXPECT_FALSE(rj_test_narrow_has_target("vendor-downstream-test"));

  ASSERT_EQ(rj_test_downstream_target_count(), 1u);
  EXPECT_TRUE(rj_test_downstream_has_target("vendor-downstream-test"));
  EXPECT_FALSE(rj_test_downstream_has_target("gfx1250"));
  EXPECT_EQ(full.find("vendor-downstream-test"), nullptr);
}

} // namespace
