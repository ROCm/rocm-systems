// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0_library_test.cpp
/// @brief Tests the fixed-profile gfx1250 B0-to-A0 shared-library API.

#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

#ifdef GFX1250_B0_TO_A0_FIXTURE
uint64_t source_identity(const std::vector<uint8_t> &bytes) {
  constexpr uint64_t kOffsetBasis = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t identity = kOffsetBasis;
  for (uint8_t byte : bytes) {
    identity ^= byte;
    identity *= kPrime;
  }
  return identity;
}
#endif

TEST(Gfx1250B0ToA0Library, RejectsInvalidArgumentsAndClearsOutputs) {
  auto *output = reinterpret_cast<uint8_t *>(0x1);
  size_t output_size = 1;
  rj_gfx1250_b0_to_a0_translation_info_t info{1, 1};
  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate_with_info(nullptr, 0, &output, &output_size, &info),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  EXPECT_EQ(info.source_code_object_id, 0u);
  EXPECT_EQ(info.changed_instruction_count, 0u);

  constexpr std::array<uint8_t, 64> kNotElf = {'N', 'O', 'T', 'E', 'L', 'F'};
  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate_with_info(kNotElf.data(), kNotElf.size(), &output,
                                                    &output_size, &info),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  EXPECT_NE(info.source_code_object_id, 0u);
  EXPECT_EQ(info.changed_instruction_count, 0u);

  rj_gfx1250_b0_to_a0_free(nullptr);
}

#ifdef GFX1250_B0_TO_A0_FIXTURE
TEST(Gfx1250B0ToA0Library, TranslatesRealGfx1250CodeObject) {
  std::ifstream input(GFX1250_B0_TO_A0_FIXTURE, std::ios::binary);
  ASSERT_TRUE(input) << GFX1250_B0_TO_A0_FIXTURE;
  const std::vector<uint8_t> source((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
  ASSERT_GE(source.size(), 4u);

  uint8_t *output = nullptr;
  size_t output_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  ASSERT_EQ(rj_gfx1250_b0_to_a0_translate_with_info(source.data(), source.size(), &output,
                                                    &output_size, &info),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(info.source_code_object_id, source_identity(source));
  EXPECT_GT(info.changed_instruction_count, 0u);
  constexpr std::array<uint8_t, 4> kElfMagic = {0x7f, 'E', 'L', 'F'};
  ASSERT_GE(output_size, kElfMagic.size());
  EXPECT_TRUE(std::equal(kElfMagic.begin(), kElfMagic.end(), output));

  rj_gfx1250_b0_to_a0_free(output);
}
#endif

} // namespace
