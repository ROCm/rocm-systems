// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for VOPD arithmetic, selection, and move wrappers.

#include "dual/helpers.h"

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_mul_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_mul_f32_w32(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        // VOPD compatibility historically omitted MUL FP status.
        if ((f & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
          f = (f & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
        }
        return rj_arith_v_mul_f32_w32(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40c00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_mul_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_mul_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        // VOPD compatibility historically omitted MUL FP status.
        if ((f & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
          f = (f & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
        }
        return rj_arith_v_mul_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40c00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_add_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_add_f32_w32(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_add_f32_w32(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40a00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_add_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_add_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_add_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40a00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_sub_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_sub_f32_w32(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_sub_f32_w32(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0xbf800000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_sub_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_sub_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_sub_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0xbf800000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_subrev_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_subrev_f32_w32(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_subrev_f32_w32(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x3f800000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_subrev_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_subrev_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_subrev_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x3f800000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_mul_dx9_zero_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_mul_dx9_zero_f32_w32(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_mul_dx9_zero_f32_w32(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40c00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_mul_dx9_zero_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_mul_dx9_zero_f32_w32_gfx12(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40c00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_mov_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_mov_b32_w32(f, e, m, d, a);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_mov_b32_w32(f, e, m, d, a);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40000000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_cndmask_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_cndmask_b32_w32(f, e, m, d, a, b, 0xaaaaaaaaU);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_cndmask_b32_w32(f, e, m, d, a, b, 0xaaaaaaaaU);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40000000U, true);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_max_num_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_max_num_f32_w32(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_max_num_f32_w32(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40400000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_min_num_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_min_num_f32_w32(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_min_num_f32_w32(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40000000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_max_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_max_f32_w32(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_max_f32_w32(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40400000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_min_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_min_f32_w32(f, e, m, d, a, b, s);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_min_f32_w32(f, e, m, d, a, b, s);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40000000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_add_nc_u32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_add_nc_u32_w32(f, e, m, d, a, b);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_add_nc_u32_w32(f, e, m, d, a, b);
      },
      2U, 3U, 0x40800000U, 5U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_lshlrev_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_lshlrev_b32_w32(f, e, m, d, a, b);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_lshlrev_b32_w32(f, e, m, d, a, b);
      },
      2U, 3U, 0x40800000U, 12U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(v_dual_and_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_dual_and_b32_w32(f, e, m, d, a, b);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)b;
        (void)s;
        return rj_arith_v_and_b32_w32(f, e, m, d, a, b);
      },
      2U, 3U, 0x40800000U, 2U, false);
}
