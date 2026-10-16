// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for lit-multiply special values and gfx11/gfx12 modifier behavior.

#include "graphics/helpers.h"

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_mullit_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_mullit_f32_w32(flags, exec_mask, mode, d, a, b, c, status);
      },
      0, 0, true, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_mullit_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_mullit_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, c, status);
      },
      0, 0, true, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_mullit_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_mullit_f32_w64(flags, exec_mask, mode, d, a, b, c, status);
      },
      0, 0, true, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_mullit_f32_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_mullit_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, c, status);
      },
      0, 0, true, true);
}
