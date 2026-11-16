// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for rounding to integral values and rounding-related FP status.

#include "math/unary_test.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_math.h"

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_trunc_f32_w32, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_ceil_f32_w32, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_rndne_f32_w32, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_floor_f32_w32, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_trunc_f32_w32_gfx12, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_ceil_f32_w32_gfx12, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_rndne_f32_w32_gfx12, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_floor_f32_w32_gfx12, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_trunc_f32_w64, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_ceil_f32_w64, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_rndne_f32_w64, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_floor_f32_w64, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_trunc_f32_w64_gfx12, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_ceil_f32_w64_gfx12, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_rndne_f32_w64_gfx12, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_floor_f32_w64_gfx12, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_trunc_f16_w32, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_ceil_f16_w32, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_rndne_f16_w32, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_floor_f16_w32, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_trunc_f16_w32_gfx12, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_ceil_f16_w32_gfx12, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_rndne_f16_w32_gfx12, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_floor_f16_w32_gfx12, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_trunc_f16_w64, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_ceil_f16_w64, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_rndne_f16_w64, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_floor_f16_w64, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_trunc_f16_w64_gfx12, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_ceil_f16_w64_gfx12, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_rndne_f16_w64_gfx12, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_floor_f16_w64_gfx12, 3);
}

// Captured TRANS overflow ignores ROUND; integral OMOD still honors it.
TEST(MathOutputControls, OverflowAndSignalingNan) {
  using namespace rocjitsu::arith::test;
  for (auto cpu : supported_cpu_levels()) {
    uint32_t a[32]{}, d[32]{};
    const uint32_t *source[] = {a};
    uint32_t *destination[] = {d};
    uint32_t status = 0;
    const auto exact = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
    for (auto semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (auto call : {rj_arith_v_rcp_f16_w32_gfx12, rj_arith_v_exp_f16_w32_gfx12}) {
        a[0] = call == rj_arith_v_rcp_f16_w32_gfx12 ? 0x0400 : 0x4b80; // EXP(15).
        ASSERT_EQ(call(cpu | semantics | RJ_ARITH_FP_ROUND_ZERO, 1, RJ_ARITH_ALU_OMOD_4,
                       destination, source, &status),
                  0);
        EXPECT_EQ(d[0], 0x7c00U);
      }
    }
    a[0] = 0x42fe0000; // EXP(127), then multiply by two.
    ASSERT_EQ(rj_arith_v_exp_f32_w32_gfx12(exact | RJ_ARITH_FP_ROUND_ZERO, 1, RJ_ARITH_ALU_OMOD_2,
                                           destination, source, &status),
              0);
    EXPECT_EQ(d[0], 0x7f800000U);
    for (auto semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      a[0] = 0x7f7fffff;
      ASSERT_EQ(rj_arith_v_trunc_f32_w32_gfx12(cpu | semantics | RJ_ARITH_FP_ROUND_ZERO, 1,
                                               RJ_ARITH_ALU_OMOD_2, destination, source, &status),
                0);
      EXPECT_EQ(d[0], 0x7f7fffffU);
    }
    for (auto call :
         {rj_arith_v_trunc_f32_w32, rj_arith_v_ceil_f32_w32, rj_arith_v_floor_f32_w32,
          rj_arith_v_rndne_f32_w32, rj_arith_v_frexp_mant_f32_w32, rj_arith_v_rcp_iflag_f32_w32}) {
      for (auto ieee : {uint64_t{0}, RJ_ARITH_FP_IEEE_MODE}) {
        a[0] = 0x7f800001;
        ASSERT_EQ(call(exact | ieee, 1, 0, destination, source, &status), 0);
        EXPECT_EQ(d[0], ieee ? 0x7fc00001U : 0x7f800001U);
      }
    }
    for (auto call : {rj_arith_v_trunc_f16_w32, rj_arith_v_ceil_f16_w32, rj_arith_v_floor_f16_w32,
                      rj_arith_v_rndne_f16_w32, rj_arith_v_frexp_mant_f16_w32}) {
      for (auto ieee : {uint64_t{0}, RJ_ARITH_FP_IEEE_MODE}) {
        a[0] = 0x7c01;
        d[0] = 0;
        ASSERT_EQ(call(exact | ieee, 1, 0, destination, source, &status), 0);
        EXPECT_EQ(d[0], ieee ? 0x7e01U : 0x7c01U);
      }
    }
    for (auto call : {rj_arith_v_fract_f32_w32, rj_arith_v_fract_f16_w32}) {
      const bool half = call == rj_arith_v_fract_f16_w32;
      a[0] = half ? 0x7c01 : 0x7f800001;
      d[0] = 0;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, 0, destination, source, &status),
                0);
      EXPECT_EQ(d[0], a[0]);
    }
    a[0] = 0x813c;
    d[0] = 0;
    ASSERT_EQ(rj_arith_v_sin_f16_w32_gfx12(cpu | RJ_ARITH_SEMANTICS_LOOSE, 1,
                                           RJ_ARITH_ALU_OMOD_HALF, destination, source, &status),
              0);
    EXPECT_EQ(d[0], 0x8000U);
  }
}

// Hardware RCP overflow ignores guest ROUND, unlike ordinary FP64 arithmetic.
TEST(MathOutputControls, DoubleReciprocalOverflow) {
  using namespace rocjitsu::arith::test;
  for (auto cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_rcp_f64_w32, rj_arith_v_rcp_f64_w32_gfx12}) {
      uint32_t lo[32]{}, hi[32]{}, dlo[32]{}, dhi[32]{};
      const uint32_t *a[] = {lo, hi};
      uint32_t *d[] = {dlo, dhi};
      uint32_t status = 0;
      for (uint32_t sign : {0U, 0x80000000U}) {
        hi[0] = 0x00100000 | sign;
        for (auto rounding : {uint64_t{0}, RJ_ARITH_FP_ROUND_ZERO}) {
          auto flags =
              cpu | RJ_ARITH_SEMANTICS_LOOSE | rounding | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
          ASSERT_EQ(call(flags, 1, RJ_ARITH_ALU_OMOD_4, d, a, &status), 0);
          EXPECT_EQ((uint64_t(dhi[0]) << 32) | dlo[0],
                    (uint64_t(sign) << 32) | 0x7ff0000000000000ULL);
        }
      }
    }
  }
}

// Captured sine gates/scaling differ from cosine when denormals are preserved.
TEST(MathOutputControls, TrigDenormalScaling) {
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      uint32_t input[32]{}, result[32]{};
      const uint32_t *a[] = {input};
      uint32_t *d[] = {result};
      input[0] = 0x3e800000; // 0.25 turns.
      ASSERT_EQ(rj_arith_v_sin_f32_w32(cpu | semantics, 1, RJ_ARITH_ALU_OMOD_2, d, a, nullptr), 0);
      EXPECT_EQ(result[0], 0x3f800000U);
      ASSERT_EQ(rj_arith_v_sin_f32_w32(cpu | semantics | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 1,
                                       RJ_ARITH_ALU_OMOD_2, d, a, nullptr),
                0);
      EXPECT_EQ(result[0], 0x40000000U);
      for (uint32_t bits : {0x00100000U, 0x80100000U}) {
        input[0] = bits;
        ASSERT_EQ(
            rj_arith_v_sin_f32_w32_gfx12(cpu | semantics, 1, RJ_ARITH_ALU_OMOD_2, d, a, nullptr),
            0);
        EXPECT_EQ(result[0], 0U);
        ASSERT_EQ(rj_arith_v_cos_f32_w32(cpu | semantics, 1, RJ_ARITH_ALU_OMOD_2, d, a, nullptr),
                  0);
        EXPECT_EQ(result[0], 0x40000000U);
      }
    }
  }
}
