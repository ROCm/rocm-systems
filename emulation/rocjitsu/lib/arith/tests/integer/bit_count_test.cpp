// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for population counts, bit reversal, and significant-bit searches.

#include "integer/bit_count_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check integer counts/searches with routing and aliases.
TEST(IntegerClzI32U32W32, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<32, 1>(rj_arith_v_clz_i32_u32_w32,
                                                rocjitsu::arith::test::BitCount::Clz);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerClzI32U32W64, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<64, 1>(rj_arith_v_clz_i32_u32_w64,
                                                rocjitsu::arith::test::BitCount::Clz);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerCtzI32B32W32, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<32, 1>(rj_arith_v_ctz_i32_b32_w32,
                                                rocjitsu::arith::test::BitCount::Ctz);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerCtzI32B32W64, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<64, 1>(rj_arith_v_ctz_i32_b32_w64,
                                                rocjitsu::arith::test::BitCount::Ctz);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerClsI32W32, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<32, 1>(rj_arith_v_cls_i32_w32,
                                                rocjitsu::arith::test::BitCount::Cls);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerClsI32W64, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<64, 1>(rj_arith_v_cls_i32_w64,
                                                rocjitsu::arith::test::BitCount::Cls);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerBcntU32B32W32, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<32, 2>(rj_arith_v_bcnt_u32_b32_w32,
                                                rocjitsu::arith::test::BitCount::Bcnt);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerBcntU32B32W64, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<64, 2>(rj_arith_v_bcnt_u32_b32_w64,
                                                rocjitsu::arith::test::BitCount::Bcnt);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerMbcntLoU32B32W32, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<32, 2>(rj_arith_v_mbcnt_lo_u32_b32_w32,
                                                rocjitsu::arith::test::BitCount::MbcntLo);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerMbcntLoU32B32W64, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<64, 2>(rj_arith_v_mbcnt_lo_u32_b32_w64,
                                                rocjitsu::arith::test::BitCount::MbcntLo);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerMbcntHiU32B32W32, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<32, 2>(rj_arith_v_mbcnt_hi_u32_b32_w32,
                                                rocjitsu::arith::test::BitCount::MbcntHi);
}

// Check integer counts/searches with routing and aliases.
TEST(IntegerMbcntHiU32B32W64, Arithmetic) {
  rocjitsu::arith::test::check_bit_count<64, 2>(rj_arith_v_mbcnt_hi_u32_b32_w64,
                                                rocjitsu::arith::test::BitCount::MbcntHi);
}
