// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for VOPD fused arithmetic and literal-operand wrappers.

#include "dual/helpers.h"

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualFmacF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_fmac_f32_w32, rj_arith_v_fmac_f32_w32,
                                    0x40000000U, 0x40400000U, 0x40800000U, 0x41200000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualFmacF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_fmac_f32_w32_gfx12,
                                    rj_arith_v_fmac_f32_w32_gfx12, 0x40000000U, 0x40400000U,
                                    0x40800000U, 0x41200000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualFmaakF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dual_fmaak_f32_w32(f, e, m, d, a, b, 0x40800000U, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_fmaak_f32_w32(f, e, m, d, a, b, 0x40800000U, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x41200000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualFmaakF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dual_fmaak_f32_w32_gfx12(f, e, m, d, a, b, 0x40800000U, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_fmaak_f32_w32_gfx12(f, e, m, d, a, b, 0x40800000U, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x41200000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualFmamkF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dual_fmamk_f32_w32(f, e, m, d, a, 0x40800000U, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_fmamk_f32_w32(f, e, m, d, a, 0x40800000U, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x41300000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualFmamkF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_dual_fmamk_f32_w32_gfx12(f, e, m, d, a, 0x40800000U, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        return rj_arith_v_fmamk_f32_w32_gfx12(f, e, m, d, a, 0x40800000U, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x41300000U, false);
}
