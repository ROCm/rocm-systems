// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcnt_translator_test.cpp
/// @brief Focused GFX9-to-GFX12 wait-counter translation tests.

#include "rocjitsu/code/dbt/waitcnt_translator.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/opcodes.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace rocjitsu {
namespace {

TEST(WaitcntTranslator, DecodeVmcnt0) {
  const auto values = decode_waitcnt_gfx9(0x0000);
  EXPECT_EQ(values.vmcnt, 0);
  EXPECT_EQ(values.lgkmcnt, 0);
  EXPECT_EQ(values.expcnt, 0);
}

TEST(WaitcntTranslator, DecodeAllRelaxed) {
  const auto values = decode_waitcnt_gfx9(0xCF7F);
  EXPECT_EQ(values.vmcnt, 0x3F);
  EXPECT_EQ(values.lgkmcnt, 0x0F);
  EXPECT_EQ(values.expcnt, 0x07);
}

TEST(WaitcntTranslator, DecodeVmcnt15Lgkm0) {
  const auto values = decode_waitcnt_gfx9(0x000F);
  EXPECT_EQ(values.vmcnt, 15);
  EXPECT_EQ(values.lgkmcnt, 0);
  EXPECT_EQ(values.expcnt, 0);
}

TEST(WaitcntTranslator, EncodeAllZeroProducesMultipleWords) {
  EXPECT_GE(encode_waitcnt_gfx12(WaitcntValues{0, 0, 0}).size(), 3u);
}

TEST(WaitcntTranslator, EncodeAllRelaxedProducesNop) {
  const auto words = encode_waitcnt_gfx12(WaitcntValues{0x3F, 0x0F, 0x07});
  ASSERT_EQ(words.size(), 1u);
  EXPECT_EQ((words[0] >> 16) & 0x7Fu, 0u);
}

TEST(WaitcntTranslator, EncodeVmcnt0EmitsLoadcntAndStorecnt) {
  const auto words = encode_waitcnt_gfx12(WaitcntValues{0, 0x0F, 0x07});
  bool has_loadcnt = false;
  bool has_storecnt_dscnt = false;
  for (uint32_t word : words) {
    const uint8_t op = (word >> 16) & 0x7F;
    has_loadcnt |= op == rdna4::kSWaitLoadcntSopp;
    has_storecnt_dscnt |= op == rdna4::kSWaitStorecntDscntSopp;
  }
  EXPECT_TRUE(has_loadcnt);
  EXPECT_TRUE(has_storecnt_dscnt);
}

} // namespace
} // namespace rocjitsu
