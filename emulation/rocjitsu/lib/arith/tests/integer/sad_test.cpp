// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sums of absolute differences, sliding windows, and packed accumulators.

#include "integer/sad_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerSadU8W32, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 32>(rj_arith_v_sad_u8_w32, rocjitsu::arith::test::Sad::U8);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerSadHiU8W32, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 32>(rj_arith_v_sad_hi_u8_w32,
                                          rocjitsu::arith::test::Sad::HiU8);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerSadU16W32, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 32>(rj_arith_v_sad_u16_w32, rocjitsu::arith::test::Sad::U16);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerSadU32W32, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 32>(rj_arith_v_sad_u32_w32, rocjitsu::arith::test::Sad::U32);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerMsadU8W32, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 32>(rj_arith_v_msad_u8_w32,
                                          rocjitsu::arith::test::Sad::MaskedU8);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerQsadPkU16U8W32, Arithmetic) {
  rocjitsu::arith::test::check_sad<2, 32>(rj_arith_v_qsad_pk_u16_u8_w32,
                                          rocjitsu::arith::test::Sad::QuadPacked);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerMqsadPkU16U8W32, Arithmetic) {
  rocjitsu::arith::test::check_sad<2, 32>(rj_arith_v_mqsad_pk_u16_u8_w32,
                                          rocjitsu::arith::test::Sad::MaskedQuadPacked);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerMqsadU32U8W32, Arithmetic) {
  rocjitsu::arith::test::check_sad<4, 32>(rj_arith_v_mqsad_u32_u8_w32,
                                          rocjitsu::arith::test::Sad::MaskedQuad);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerSadU8W64, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 64>(rj_arith_v_sad_u8_w64, rocjitsu::arith::test::Sad::U8);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerSadHiU8W64, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 64>(rj_arith_v_sad_hi_u8_w64,
                                          rocjitsu::arith::test::Sad::HiU8);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerSadU16W64, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 64>(rj_arith_v_sad_u16_w64, rocjitsu::arith::test::Sad::U16);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerSadU32W64, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 64>(rj_arith_v_sad_u32_w64, rocjitsu::arith::test::Sad::U32);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerMsadU8W64, Arithmetic) {
  rocjitsu::arith::test::check_sad<1, 64>(rj_arith_v_msad_u8_w64,
                                          rocjitsu::arith::test::Sad::MaskedU8);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerQsadPkU16U8W64, Arithmetic) {
  rocjitsu::arith::test::check_sad<2, 64>(rj_arith_v_qsad_pk_u16_u8_w64,
                                          rocjitsu::arith::test::Sad::QuadPacked);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerMqsadPkU16U8W64, Arithmetic) {
  rocjitsu::arith::test::check_sad<2, 64>(rj_arith_v_mqsad_pk_u16_u8_w64,
                                          rocjitsu::arith::test::Sad::MaskedQuadPacked);
}

// Check SAD windows, packed accumulators, and multi-register aliases.
TEST(IntegerMqsadU32U8W64, Arithmetic) {
  rocjitsu::arith::test::check_sad<4, 64>(rj_arith_v_mqsad_u32_u8_w64,
                                          rocjitsu::arith::test::Sad::MaskedQuad);
}
