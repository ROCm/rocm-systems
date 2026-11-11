// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for VOPD dot-product wrappers.

#include "dual/helpers.h"

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualDot2accF32F16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_dot2acc_f32_f16_w32,
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dot2_f32_f16_w32(f, e, m, d, a, b, d, s);
      },
      0x3c003c00U, 0x40004000U, 0x3f800000U, 0x40a00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualDot2accF32Bf16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_dot2acc_f32_bf16_w32,
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dot2_f32_bf16_w32(f, e, m, d, a, b, d, s);
      },
      0x3f803f80U, 0x40004000U, 0x3f800000U, 0x40a00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualDot2accF32F16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_dot2acc_f32_f16_w32_gfx12,
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dot2_f32_f16_w32_gfx12(f, e, m, d, a, b, d, s);
      },
      0x3c003c00U, 0x40004000U, 0x3f800000U, 0x40a00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualDot2accF32Bf16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_dot2acc_f32_bf16_w32_gfx12,
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dot2_f32_bf16_w32_gfx12(f, e, m, d, a, b, d, s);
      },
      0x3f803f80U, 0x40004000U, 0x3f800000U, 0x40a00000U, false);
}
