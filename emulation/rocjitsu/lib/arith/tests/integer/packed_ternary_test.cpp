// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed 16-bit multiply-add and selections with saturation.

#include "integer/packed_ternary_test.h"
#include "rj_arith/detail/rj_arith_integer.h"

#include <gtest/gtest.h>

// Check packed integer multiply-add and saturation.
TEST(IntegerPkMadU16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_mad_u16_w32, false);
}

// Check packed integer multiply-add and saturation.
TEST(IntegerPkMadU16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_mad_u16_w64, false);
}

// Check packed integer multiply-add and saturation.
TEST(IntegerPkMadI16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_mad_i16_w32, true);
}

// Check packed integer multiply-add and saturation.
TEST(IntegerPkMadI16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_mad_i16_w64, true);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkAddMinI16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_add_min_i16_w32, true,
                                                     rocjitsu::arith::test::PackedTernary::AddMin);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkAddMinI16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_add_min_i16_w64, true,
                                                     rocjitsu::arith::test::PackedTernary::AddMin);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkAddMinU16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_add_min_u16_w32, false,
                                                     rocjitsu::arith::test::PackedTernary::AddMin);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkAddMinU16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_add_min_u16_w64, false,
                                                     rocjitsu::arith::test::PackedTernary::AddMin);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkAddMaxI16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_add_max_i16_w32, true,
                                                     rocjitsu::arith::test::PackedTernary::AddMax);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkAddMaxI16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_add_max_i16_w64, true,
                                                     rocjitsu::arith::test::PackedTernary::AddMax);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkAddMaxU16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_add_max_u16_w32, false,
                                                     rocjitsu::arith::test::PackedTernary::AddMax);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkAddMaxU16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_add_max_u16_w64, false,
                                                     rocjitsu::arith::test::PackedTernary::AddMax);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkMin3I16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_min3_i16_w32, true,
                                                     rocjitsu::arith::test::PackedTernary::Min);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkMin3I16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_min3_i16_w64, true,
                                                     rocjitsu::arith::test::PackedTernary::Min);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkMin3U16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_min3_u16_w32, false,
                                                     rocjitsu::arith::test::PackedTernary::Min);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkMin3U16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_min3_u16_w64, false,
                                                     rocjitsu::arith::test::PackedTernary::Min);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkMax3I16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_max3_i16_w32, true,
                                                     rocjitsu::arith::test::PackedTernary::Max);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkMax3I16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_max3_i16_w64, true,
                                                     rocjitsu::arith::test::PackedTernary::Max);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkMax3U16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<32, 3>(rj_arith_v_pk_max3_u16_w32, false,
                                                     rocjitsu::arith::test::PackedTernary::Max);
}

// Check packed selection with all CPU levels, semantics, selectors and aliases.
TEST(IntegerPkMax3U16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_ternary<64, 3>(rj_arith_v_pk_max3_u16_w64, false,
                                                     rocjitsu::arith::test::PackedTernary::Max);
}
