// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcnt_translator_test.cpp
/// @brief Focused GFX9-to-GFX12 wait-counter translation tests.

#include "rocjitsu/code/dbt/waitcnt_translator.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"

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

TEST(WaitcntTranslator, DecodeSplitVmBitsAndIgnoreReservedBits) {
  struct Case {
    uint16_t simm16;
    WaitcntValues expected;
  };
  // Pin the GFX9 layout independently of the shared helper. Its result orders
  // EXP before LGKM, whereas WaitcntValues orders LGKM before EXP.
  const Case cases[] = {
      {0x4000, {16, 0, 0}},  {0x8000, {32, 0, 0}}, {0x41a5, {21, 1, 2}},
      {0xc978, {56, 9, 7}},  {0x3080, {0, 0, 0}}, // Bits 7, 12 and 13 are reserved on GFX9.
      {0xffff, {63, 15, 7}},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(c.simm16);
    const auto values = decode_waitcnt_gfx9(c.simm16);
    EXPECT_EQ(values.vmcnt, c.expected.vmcnt);
    EXPECT_EQ(values.lgkmcnt, c.expected.lgkmcnt);
    EXPECT_EQ(values.expcnt, c.expected.expcnt);
  }
}

TEST(WaitcntTranslator, EncodeAllZeroProducesMultipleWords) {
  // vmcnt=0 splits into loadcnt(0) + storecnt/dscnt(0,0); lgkmcnt=0 also feeds
  // dscnt and emits kmcnt(0); expcnt=0 emits expcnt(0). The encoding is
  // deterministic, so pin the exact SOPP words to catch a regression in either
  // the opcode selection or the packed simm16 fields.
  const auto words = encode_waitcnt_gfx12(WaitcntValues{0, 0, 0});
  const std::vector<uint32_t> expected = {0xBFC00000u, 0xBFC90000u, 0xBFC70000u, 0xBFC40000u};
  EXPECT_EQ(words, expected);
}

TEST(WaitcntTranslator, EncodeAllRelaxedProducesNop) {
  const auto words = encode_waitcnt_gfx12(WaitcntValues{0x3F, 0x0F, 0x07});
  ASSERT_EQ(words.size(), 1u);
  EXPECT_EQ((words[0] >> 16) & 0x7Fu, 0u);
}

TEST(WaitcntTranslator, EncodeVmcnt0EmitsLoadcntAndStorecnt) {
  // vmcnt=0 with relaxed lgkm/exp yields exactly loadcnt(0) and
  // storecnt(0)/dscnt(15): the dscnt field stays at its no-wait maximum because
  // lgkmcnt is relaxed. Pin the exact words so a regression in the packed
  // storecnt/dscnt simm16 (sc<<4 | dc) is caught.
  const auto words = encode_waitcnt_gfx12(WaitcntValues{0, 0x0F, 0x07});
  const std::vector<uint32_t> expected = {0xBFC00000u, 0xBFC9000Fu};
  EXPECT_EQ(words, expected);

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
