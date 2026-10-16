// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP add/subtract and reversed-subtract entry points.

#include "fp/binary_arithmetic_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_add_f32_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_add_f32_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_add_f32_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_add_f32_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_add_f64_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_add_f64_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_add_f64_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_add_f64_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_sub_f32_w32_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_sub_f32_w32, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_sub_f32_w64_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_sub_f32_w64, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_subrev_f32_w32_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_subrev_f32_w32, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_subrev_f32_w64_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_subrev_f32_w64, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}
