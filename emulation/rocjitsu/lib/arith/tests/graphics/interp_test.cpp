// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP16/FP32 interpolation stages and quad-lane operand routing.

#include "graphics/helpers.h"

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p10_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p10_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      0, 32, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p2_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p2_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      1, 32, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p10_f16_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p10_f16_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      0, 16, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p2_f16_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p2_f16_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      1, 16, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p10_rtz_f16_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p10_rtz_f16_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      2, 16, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p2_rtz_f16_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p2_rtz_f16_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      3, 16, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p10_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p10_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      0, 32, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p2_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p2_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      1, 32, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p10_f16_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p10_f16_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      0, 16, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p2_f16_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p2_f16_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      1, 16, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p10_rtz_f16_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p10_rtz_f16_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      2, 16, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_interp_p2_rtz_f16_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_interp_p2_rtz_f16_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      3, 16, false, false);
}
