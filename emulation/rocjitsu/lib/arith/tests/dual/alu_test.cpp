// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for VOPD arithmetic, selection, and move wrappers.

#include "dual/helpers.h"
#include "rj_arith/detail/rj_arith_dual.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_integer.h"
#include "rj_arith/detail/rj_arith_register.h"

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualMulF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_mul_f32_w32,
      rocjitsu::arith::test::numeric_compatibility_reference(rj_arith_v_mul_f32_w32), 0x40000000U,
      0x40400000U, 0x40800000U, 0x40c00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualMulF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rocjitsu::arith::test::numeric_compatibility_reference(rj_arith_v_dual_mul_f32_w32_gfx12),
      rocjitsu::arith::test::numeric_compatibility_reference(rj_arith_v_mul_f32_w32_gfx12),
      0x40000000U, 0x40400000U, 0x40800000U, 0x40c00000U, false);
  // Literal causes follow shared/fp_status.h::fma, the pre-migration VOPD model.
  // In particular it retains INPUT_DENORM for a denormal paired with a NaN.
  struct Case {
    uint32_t a, b;
    uint64_t flags;
    uint32_t causes;
  };
  const Case cases[] = {
      {0x3f800000, 0x3f800000, 0, 0},
      {0x7f800001, 0x3f800000, 0, RJ_ARITH_EXCEPTION_INVALID},
      {1, 0x7fc12345, 0, RJ_ARITH_EXCEPTION_INPUT_DENORM},
      {1, 0x7f800000, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, RJ_ARITH_EXCEPTION_INVALID},
      {0x7f7fffff, 0x40000000, 0, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0x00800000, 0x3f000000, 0, 0},
      {0x00800000, 0x3f000000, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
       RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0x3f800001, 0x3f800001, RJ_ARITH_FP_ROUND_UP, RJ_ARITH_EXCEPTION_INEXACT},
  };
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const auto &test : cases) {
      for (uint32_t exec_mask : {0U, 1U << 31}) {
        for (int aliasing_offset : {0, 1, 3}) {
          rocjitsu::arith::test::Registers<4, 32> input;
          input.v[0][31] = test.a;
          input.v[1][31] = test.b;
          uint32_t expected[32], *reference = expected;
          ASSERT_EQ(rj_arith_v_mul_f32_w32_gfx12(
                        cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | test.flags,
                        exec_mask, 0, &reference, input.v, input.v + 1, nullptr),
                    RJ_ARITH_SUCCESS);
          uint32_t status = 0x80000040U;
          ASSERT_EQ(rj_arith_v_dual_mul_f32_w32_gfx12(
                        cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | test.flags, exec_mask, 0,
                        input.v + aliasing_offset, input.v, input.v + 1, &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x80000040U | (exec_mask ? test.causes : 0));
          if (exec_mask) {
            EXPECT_EQ(input.v[aliasing_offset][31], expected[31]);
          }
        }
      }
    }
  }
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualAddF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_add_f32_w32, rj_arith_v_add_f32_w32,
                                    0x40000000U, 0x40400000U, 0x40800000U, 0x40a00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualAddF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_add_f32_w32_gfx12,
      rocjitsu::arith::test::numeric_compatibility_reference(rj_arith_v_add_f32_w32_gfx12),
      0x40000000U, 0x40400000U, 0x40800000U, 0x40a00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualSubF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_sub_f32_w32, rj_arith_v_sub_f32_w32,
                                    0x40000000U, 0x40400000U, 0x40800000U, 0xbf800000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualSubF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_sub_f32_w32_gfx12,
      rocjitsu::arith::test::numeric_compatibility_reference(rj_arith_v_sub_f32_w32_gfx12),
      0x40000000U, 0x40400000U, 0x40800000U, 0xbf800000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualSubrevF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_subrev_f32_w32, rj_arith_v_subrev_f32_w32,
                                    0x40000000U, 0x40400000U, 0x40800000U, 0x3f800000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualSubrevF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_subrev_f32_w32_gfx12,
      rocjitsu::arith::test::numeric_compatibility_reference(rj_arith_v_subrev_f32_w32_gfx12),
      0x40000000U, 0x40400000U, 0x40800000U, 0x3f800000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualMulDx9ZeroF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_mul_dx9_zero_f32_w32,
                                    rj_arith_v_mul_dx9_zero_f32_w32, 0x40000000U, 0x40400000U,
                                    0x40800000U, 0x40c00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualMulDx9ZeroF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx12,
                                    rj_arith_v_mul_dx9_zero_f32_w32_gfx12, 0x40000000U, 0x40400000U,
                                    0x40800000U, 0x40c00000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualMovB32W32, Arithmetic) {
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
TEST(DualCndmaskB32W32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)s;
        return rj_arith_v_dual_cndmask_b32_w32(f, e, m, d, a, b, 0xaaaaaaaaU);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)s;
        return rj_arith_v_cndmask_b32_w32(f, e, m, d, a, b, 0xaaaaaaaaU);
      },
      0x40000000U, 0x40400000U, 0x40800000U, 0x40000000U, true);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualMaxNumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_max_num_f32_w32,
      rocjitsu::arith::test::numeric_compatibility_reference(rj_arith_v_max_num_f32_w32),
      0x40000000U, 0x40400000U, 0x40800000U, 0x40400000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualMinNumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      rj_arith_v_dual_min_num_f32_w32,
      rocjitsu::arith::test::numeric_compatibility_reference(rj_arith_v_min_num_f32_w32),
      0x40000000U, 0x40400000U, 0x40800000U, 0x40000000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualMaxF32W32, Arithmetic) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_max_f32_w32, rj_arith_v_max_f32_w32,
                                    0x40000000U, 0x40400000U, 0x40800000U, 0x40400000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualMinF32W32, Arithmetic) {
  rocjitsu::arith::test::check_dual(rj_arith_v_dual_min_f32_w32, rj_arith_v_min_f32_w32,
                                    0x40000000U, 0x40400000U, 0x40800000U, 0x40000000U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualAddNcU32W32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)s;
        return rj_arith_v_dual_add_nc_u32_w32(f, e, m, d, a, b);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)s;
        return rj_arith_v_add_nc_u32_w32(f, e, m, d, a, b);
      },
      2U, 3U, 0x40800000U, 5U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualLshlrevB32W32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)s;
        return rj_arith_v_dual_lshlrev_b32_w32(f, e, m, d, a, b);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)s;
        return rj_arith_v_lshlrev_b32_w32(f, e, m, d, a, b);
      },
      2U, 3U, 0x40800000U, 12U, false);
}

// Check VOPD results and status against the corresponding ordinary instruction.
TEST(DualAndB32W32, Arithmetic) {
  rocjitsu::arith::test::check_dual(
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)s;
        return rj_arith_v_dual_and_b32_w32(f, e, m, d, a, b);
      },
      [](auto f, auto e, auto m, auto d, auto a, auto b, auto s) {
        (void)s;
        return rj_arith_v_and_b32_w32(f, e, m, d, a, b);
      },
      2U, 3U, 0x40800000U, 2U, false);
}
