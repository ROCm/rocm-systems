// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for always-true and always-false comparison masks.

#include "compare/constant_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

// Check always-true and always-false comparison masks.
TEST(CompareCmpFF16W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<16, 32, /*Floating=*/true>(rj_arith_v_cmp_f_f16_w32,
                                                                           false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTF16W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<16, 32, /*Floating=*/true>(rj_arith_v_cmp_t_f16_w32,
                                                                           true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFF16W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<16, 32, /*Floating=*/true>(
      rj_arith_v_cmpx_f_f16_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTF16W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<16, 32, /*Floating=*/true>(
      rj_arith_v_cmpx_t_f16_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFF32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/true>(rj_arith_v_cmp_f_f32_w32,
                                                                           false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTF32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/true>(rj_arith_v_cmp_t_f32_w32,
                                                                           true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFF32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/true>(
      rj_arith_v_cmpx_f_f32_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTF32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/true>(
      rj_arith_v_cmpx_t_f32_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFF64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/true>(rj_arith_v_cmp_f_f64_w32,
                                                                           false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTF64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/true>(rj_arith_v_cmp_t_f64_w32,
                                                                           true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFF64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/true>(
      rj_arith_v_cmpx_f_f64_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTF64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/true>(
      rj_arith_v_cmpx_t_f64_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFI32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/false>(
      rj_arith_v_cmp_f_i32_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTI32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/false>(
      rj_arith_v_cmp_t_i32_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFI32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/false>(
      rj_arith_v_cmpx_f_i32_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTI32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/false>(
      rj_arith_v_cmpx_t_i32_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFU32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/false>(
      rj_arith_v_cmp_f_u32_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTU32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/false>(
      rj_arith_v_cmp_t_u32_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFU32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/false>(
      rj_arith_v_cmpx_f_u32_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTU32W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 32, /*Floating=*/false>(
      rj_arith_v_cmpx_t_u32_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFI64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/false>(
      rj_arith_v_cmp_f_i64_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTI64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/false>(
      rj_arith_v_cmp_t_i64_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFI64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/false>(
      rj_arith_v_cmpx_f_i64_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTI64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/false>(
      rj_arith_v_cmpx_t_i64_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFU64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/false>(
      rj_arith_v_cmp_f_u64_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTU64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/false>(
      rj_arith_v_cmp_t_u64_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFU64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/false>(
      rj_arith_v_cmpx_f_u64_w32, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTU64W32, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 32, /*Floating=*/false>(
      rj_arith_v_cmpx_t_u64_w32, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFI32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/false>(
      rj_arith_v_cmp_f_i32_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTI32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/false>(
      rj_arith_v_cmp_t_i32_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFI32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/false>(
      rj_arith_v_cmpx_f_i32_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTI32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/false>(
      rj_arith_v_cmpx_t_i32_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFU32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/false>(
      rj_arith_v_cmp_f_u32_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTU32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/false>(
      rj_arith_v_cmp_t_u32_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFU32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/false>(
      rj_arith_v_cmpx_f_u32_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTU32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/false>(
      rj_arith_v_cmpx_t_u32_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFI64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/false>(
      rj_arith_v_cmp_f_i64_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTI64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/false>(
      rj_arith_v_cmp_t_i64_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFI64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/false>(
      rj_arith_v_cmpx_f_i64_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTI64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/false>(
      rj_arith_v_cmpx_t_i64_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFU64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/false>(
      rj_arith_v_cmp_f_u64_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTU64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/false>(
      rj_arith_v_cmp_t_u64_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFU64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/false>(
      rj_arith_v_cmpx_f_u64_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTU64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/false>(
      rj_arith_v_cmpx_t_u64_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFF16W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<16, 64, /*Floating=*/true>(rj_arith_v_cmp_f_f16_w64,
                                                                           false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTF16W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<16, 64, /*Floating=*/true>(rj_arith_v_cmp_t_f16_w64,
                                                                           true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFF16W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<16, 64, /*Floating=*/true>(
      rj_arith_v_cmpx_f_f16_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTF16W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<16, 64, /*Floating=*/true>(
      rj_arith_v_cmpx_t_f16_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFF32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/true>(rj_arith_v_cmp_f_f32_w64,
                                                                           false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTF32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/true>(rj_arith_v_cmp_t_f32_w64,
                                                                           true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFF32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/true>(
      rj_arith_v_cmpx_f_f32_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTF32W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<32, 64, /*Floating=*/true>(
      rj_arith_v_cmpx_t_f32_w64, true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpFF64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/true>(rj_arith_v_cmp_f_f64_w64,
                                                                           false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpTF64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/true>(rj_arith_v_cmp_t_f64_w64,
                                                                           true);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxFF64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/true>(
      rj_arith_v_cmpx_f_f64_w64, false);
}

// Check always-true and always-false comparison masks.
TEST(CompareCmpxTF64W64, Arithmetic) {
  rocjitsu::arith::test::check_constant_compare<64, 64, /*Floating=*/true>(
      rj_arith_v_cmpx_t_f64_w64, true);
}
