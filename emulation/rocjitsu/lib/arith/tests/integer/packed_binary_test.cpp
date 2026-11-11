// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed 16-bit integer arithmetic and selection.

#include "integer/packed_binary_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkAddI16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_add_i16_w32,
                                                    rocjitsu::arith::test::HalfBinary::AddI16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkAddI16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_add_i16_w64,
                                                    rocjitsu::arith::test::HalfBinary::AddI16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkSubI16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_sub_i16_w32,
                                                    rocjitsu::arith::test::HalfBinary::SubI16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkSubI16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_sub_i16_w64,
                                                    rocjitsu::arith::test::HalfBinary::SubI16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkAddU16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_add_u16_w32,
                                                    rocjitsu::arith::test::HalfBinary::AddU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkAddU16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_add_u16_w64,
                                                    rocjitsu::arith::test::HalfBinary::AddU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkSubU16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_sub_u16_w32,
                                                    rocjitsu::arith::test::HalfBinary::SubU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkSubU16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_sub_u16_w64,
                                                    rocjitsu::arith::test::HalfBinary::SubU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMinI16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_min_i16_w32,
                                                    rocjitsu::arith::test::HalfBinary::MinI16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMinI16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_min_i16_w64,
                                                    rocjitsu::arith::test::HalfBinary::MinI16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMaxI16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_max_i16_w32,
                                                    rocjitsu::arith::test::HalfBinary::MaxI16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMaxI16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_max_i16_w64,
                                                    rocjitsu::arith::test::HalfBinary::MaxI16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMinU16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_min_u16_w32,
                                                    rocjitsu::arith::test::HalfBinary::MinU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMinU16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_min_u16_w64,
                                                    rocjitsu::arith::test::HalfBinary::MinU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMaxU16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_max_u16_w32,
                                                    rocjitsu::arith::test::HalfBinary::MaxU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMaxU16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_max_u16_w64,
                                                    rocjitsu::arith::test::HalfBinary::MaxU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMulLoU16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_mul_lo_u16_w32,
                                                    rocjitsu::arith::test::HalfBinary::MulLoU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkMulLoU16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_mul_lo_u16_w64,
                                                    rocjitsu::arith::test::HalfBinary::MulLoU16);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkLshlrevB16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_lshlrev_b16_w32,
                                                    rocjitsu::arith::test::HalfBinary::Lshl);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkLshlrevB16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_lshlrev_b16_w64,
                                                    rocjitsu::arith::test::HalfBinary::Lshl);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkLshrrevB16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_lshrrev_b16_w32,
                                                    rocjitsu::arith::test::HalfBinary::Lshr);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkLshrrevB16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_lshrrev_b16_w64,
                                                    rocjitsu::arith::test::HalfBinary::Lshr);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkAshrrevI16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<32, 2>(rj_arith_v_pk_ashrrev_i16_w32,
                                                    rocjitsu::arith::test::HalfBinary::Ashr);
}

// Check packed integer arithmetic and independent halfword selectors.
TEST(IntegerPkAshrrevI16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_binary<64, 2>(rj_arith_v_pk_ashrrev_i16_w64,
                                                    rocjitsu::arith::test::HalfBinary::Ashr);
}
