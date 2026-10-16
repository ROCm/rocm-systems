// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for signed and unsigned integer ordering predicates.

#include "compare/integer_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmp_lt_i16_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmp_le_i16_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmp_gt_i16_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmp_ge_i16_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_lt_i16_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_le_i16_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_gt_i16_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_ge_i16_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmp_lt_u16_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmp_le_u16_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmp_gt_u16_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmp_ge_u16_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_lt_u16_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_le_u16_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_gt_u16_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_ge_u16_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmp_lt_i32_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmp_le_i32_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmp_gt_i32_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmp_ge_i32_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_lt_i32_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_le_i32_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_gt_i32_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_ge_i32_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmp_lt_u32_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmp_le_u32_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmp_gt_u32_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmp_ge_u32_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_lt_u32_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_le_u32_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_gt_u32_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_ge_u32_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmp_lt_i64_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmp_le_i64_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmp_gt_i64_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmp_ge_i64_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_lt_i64_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_le_i64_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_gt_i64_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_ge_i64_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmp_lt_u64_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmp_le_u64_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmp_gt_u64_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmp_ge_u64_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_lt_u64_w32, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_le_u64_w32, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_gt_u64_w32, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_ge_u64_w32, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmp_lt_i16_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmp_le_i16_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmp_gt_i16_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmp_ge_i16_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_lt_i16_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_le_i16_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_gt_i16_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_ge_i16_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmp_lt_u16_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmp_le_u16_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmp_gt_u16_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmp_ge_u16_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_lt_u16_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_le_u16_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_gt_u16_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_ge_u16_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmp_lt_i32_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmp_le_i32_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmp_gt_i32_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmp_ge_i32_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_lt_i32_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_le_i32_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_gt_i32_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_ge_i32_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmp_lt_u32_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmp_le_u32_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmp_gt_u32_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmp_ge_u32_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_lt_u32_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_le_u32_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_gt_u32_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_ge_u32_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmp_lt_i64_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmp_le_i64_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmp_gt_i64_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmp_ge_i64_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_lt_i64_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_le_i64_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_gt_i64_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_ge_i64_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLtU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmp_lt_u64_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpLeU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmp_le_u64_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGtU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmp_gt_u64_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpGeU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmp_ge_u64_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLtU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_lt_u64_w64, [](uint64_t a, uint64_t b) { return a < b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxLeU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_le_u64_w64, [](uint64_t a, uint64_t b) { return a <= b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGtU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_gt_u64_w64, [](uint64_t a, uint64_t b) { return a > b; });
}

// Check signed and unsigned integer ordering predicates.
TEST(CompareCmpxGeU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_ge_u64_w64, [](uint64_t a, uint64_t b) { return a >= b; });
}
