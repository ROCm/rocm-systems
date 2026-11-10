// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed 16-bit multiply-add and saturation.

#include "integer/packed_mad_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check packed integer multiply-add and saturation.
TEST(IntegerPkMadU16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_mad<32, 3>(rj_arith_v_pk_mad_u16_w32, false);
}

// Check packed integer multiply-add and saturation.
TEST(IntegerPkMadU16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_mad<64, 3>(rj_arith_v_pk_mad_u16_w64, false);
}

// Check packed integer multiply-add and saturation.
TEST(IntegerPkMadI16W32, Arithmetic) {
  rocjitsu::arith::test::check_packed_mad<32, 3>(rj_arith_v_pk_mad_i16_w32, true);
}

// Check packed integer multiply-add and saturation.
TEST(IntegerPkMadI16W64, Arithmetic) {
  rocjitsu::arith::test::check_packed_mad<64, 3>(rj_arith_v_pk_mad_i16_w64, true);
}
