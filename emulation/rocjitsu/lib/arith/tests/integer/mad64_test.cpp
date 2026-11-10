// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for wide integer multiply-add and carry results.

#include "integer/mad64_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check 64-bit multiply-add, carry masks, and overlapping destination words.
TEST(IntegerMadU64U32W32, Arithmetic) {
  rocjitsu::arith::test::check_mad64<32>(rj_arith_v_mad_u64_u32_w32, /*is_signed=*/false);
}

// Check 64-bit multiply-add, carry masks, and overlapping destination words.
TEST(IntegerMadI64I32W32, Arithmetic) {
  rocjitsu::arith::test::check_mad64<32>(rj_arith_v_mad_i64_i32_w32, /*is_signed=*/true);
}

// Check 64-bit multiply-add, carry masks, and overlapping destination words.
TEST(IntegerMadCoU64U32W32, Arithmetic) {
  rocjitsu::arith::test::check_mad64<32>(rj_arith_v_mad_co_u64_u32_w32, /*is_signed=*/false);
}

// Check 64-bit multiply-add, carry masks, and overlapping destination words.
TEST(IntegerMadCoI64I32W32, Arithmetic) {
  rocjitsu::arith::test::check_mad64<32>(rj_arith_v_mad_co_i64_i32_w32, /*is_signed=*/true);
}

// Check 64-bit multiply-add, carry masks, and overlapping destination words.
TEST(IntegerMadU64U32W64, Arithmetic) {
  rocjitsu::arith::test::check_mad64<64>(rj_arith_v_mad_u64_u32_w64, /*is_signed=*/false);
}

// Check 64-bit multiply-add, carry masks, and overlapping destination words.
TEST(IntegerMadI64I32W64, Arithmetic) {
  rocjitsu::arith::test::check_mad64<64>(rj_arith_v_mad_i64_i32_w64, /*is_signed=*/true);
}

// Check 64-bit multiply-add, carry masks, and overlapping destination words.
TEST(IntegerMadCoU64U32W64, Arithmetic) {
  rocjitsu::arith::test::check_mad64<64>(rj_arith_v_mad_co_u64_u32_w64, /*is_signed=*/false);
}

// Check 64-bit multiply-add, carry masks, and overlapping destination words.
TEST(IntegerMadCoI64I32W64, Arithmetic) {
  rocjitsu::arith::test::check_mad64<64>(rj_arith_v_mad_co_i64_i32_w64, /*is_signed=*/true);
}
