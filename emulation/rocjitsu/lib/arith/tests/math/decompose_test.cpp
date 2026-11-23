// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for fractional-part and mantissa decomposition.

#include "fp/captured_gfx11_test.h"
#include "math/unary_test.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <gtest/gtest.h>
#include <stdint.h>

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_fract_f32_w32, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_frexp_mant_f32_w32, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_fract_f32_w32_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_frexp_mant_f32_w32_gfx12, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_fract_f32_w64, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_frexp_mant_f32_w64, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_fract_f32_w64_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_frexp_mant_f32_w64_gfx12, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_fract_f16_w32, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_frexp_mant_f16_w32, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_fract_f16_w32_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_frexp_mant_f16_w32_gfx12, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_fract_f16_w64, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_frexp_mant_f16_w64, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_fract_f16_w64_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_frexp_mant_f16_w64_gfx12, 10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF16W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<32, true>(rj_arith_v_frexp_mant_f16_w32_gfx942,
                                                             10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF16W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<64, true>(rj_arith_v_frexp_mant_f16_w64_gfx942,
                                                             10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF32W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<32, false>(rj_arith_v_frexp_mant_f32_w32_gfx942,
                                                              10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF32W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<64, false>(rj_arith_v_frexp_mant_f32_w64_gfx942,
                                                              10);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF16W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_fract_f16_w32_gfx942, 9, false,
                                                      false);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF16W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_fract_f16_w64_gfx942, 9, false,
                                                      false);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF32W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_fract_f32_w32_gfx942, 9, false,
                                                       false);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF32W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_fract_f32_w64_gfx942, 9, false,
                                                       false);
}

namespace {

using UnaryFn = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                        uint32_t *);

// gfx1201 captures with IEEE set: guest rounding applies to the subtraction, a tiny negative
// input caps below one unless input flushing zeroes it, and NaNs are quieted with their payload.
// FP16 forms write the low half and keep the high half.
struct FractCase {
  UnaryFn call;
  uint64_t flags;
  uint32_t input, expected;
};

const FractCase fract_cases[] = {
    {rj_arith_v_fract_f32_w32_gfx12, 0, 0xb5a168adU, 0x3f7fffecU},
    {rj_arith_v_fract_f32_w32_gfx12, RJ_ARITH_FP_ROUND_ZERO, 0xb5a168adU, 0x3f7fffebU},
    {rj_arith_v_fract_f32_w32_gfx12, 0, 0x80000001U, 0x3f7fffffU},
    {rj_arith_v_fract_f32_w32_gfx12, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0x80000001U, 0},
    {rj_arith_v_fract_f32_w32_gfx12, 0, 0x7f800001U, 0x7fc00001U},
    {rj_arith_v_fract_f16_w32_gfx12, 0, 0xa6e1U, 0x3bc9U},
    {rj_arith_v_fract_f16_w32_gfx12, RJ_ARITH_FP_ROUND_ZERO, 0xa6e1U, 0x3bc8U},
    {rj_arith_v_fract_f16_w32_gfx12, 0, 0x8001U, 0x3bffU},
    {rj_arith_v_fract_f16_w32_gfx12, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0x8001U, 0},
    {rj_arith_v_fract_f16_w32_gfx12, 0, 0x7c01U, 0x7e01U},
};

} // namespace

// Pin the captured FRACT results in Compatibility and strict exact mode, so that a faster loose
// path cannot change what default-mode RDNA4 execution returns.
TEST(MathFractGfx12, ExactMatchesHardware) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : hardware_semantics()) {
      for (const FractCase &test : fract_cases) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << '/' << semantics << '/' << (&test - fract_cases));
        const bool half = test.call == rj_arith_v_fract_f16_w32_gfx12;
        Registers<2, 32> r;
        r.v[0][0] = half ? 0xa5a50000U | test.input : test.input;
        r.v[1][0] = 0xa5a5a5a5U;
        ASSERT_EQ(test.call(cpu | semantics | RJ_ARITH_FP_IEEE_MODE | test.flags, 1, 0, r.v + 1,
                            r.v, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[1][0], half ? 0xa5a50000U | test.expected : test.expected);
      }
    }
  }
}

namespace {

using rocjitsu::arith::test::CapturedGfx11;

// gfx1100 (W7900, mkm2) FRACT lanes: hw1100-noieee (2026-09-25, MODE.IEEE clear) for 0x1f0 and
// 0x100, the Track C hw1100-wide corpus (2026-11-18) for the directed-rounding words 0x1f5 and
// 0x1fa, and hw2-gfx1100 (2026-10-06, MODE.IEEE set) for 0x300/0x3f0. RDNA3 applies OMOD only with
// output denormals flushed and IEEE clear: mul:2 doubles 0.5 at 0x100 and leaves it alone at
// 0x1f0 and 0x300, so a kernel that always scales fails two rows. A tiny negative input caps
// below one under every rounding; signaling NaNs pass through with IEEE clear and quiet with it.
const CapturedGfx11 fract_f32_gfx11_lanes[] = {
    {"Mul2FlushScales", RJ_ARITH_ALU_OMOD_2, 0x100, 0x3f000000, 0, 0, 0x3f800000},
    {"Mul2MinNormalFlushScales", RJ_ARITH_ALU_OMOD_2, 0x100, 0x00800000, 0, 0, 0x01000000},
    {"Mul2DenormalsKeptIgnored", RJ_ARITH_ALU_OMOD_2, 0x1f0, 0x3f000000, 0, 0, 0x3f000000},
    {"Mul2IeeeIgnored", RJ_ARITH_ALU_OMOD_2, 0x300, 0x00800000, 0, 0, 0x00800000},
    {"TinyNegativeNearest", 0, 0x1f0, 0x80800000, 0, 0, 0x3f7fffff},
    {"TinyNegativeRoundUp", 0, 0x1f5, 0x80800000, 0, 0, 0x3f7fffff},
    {"TinyNegativeRoundDown", 0, 0x1fa, 0x80800000, 0, 0, 0x3f7fffff},
    {"MinusHalf", 0, 0x1f0, 0xbf000000, 0, 0, 0x3f000000},
    {"SignalingKeptIeeeClear", 0, 0x1f0, 0x7f800001, 0, 0, 0x7f800001},
};
// FP16 forms write the selected destination half and keep the other, which the capture filled
// with 0xa5a5.
const CapturedGfx11 fract_f16_gfx11_lanes[] = {
    {"TinyNegativeCaps", 0, 0x1f0, 0x8ced8400, 0, 0, 0xa5a53bff},
    {"PositiveFraction", 0, 0x1f0, 0x4daab800, 0, 0, 0xa5a53800},
    {"Mul2FlushScalesMinNormal", RJ_ARITH_ALU_OMOD_2, 0x100, 0xb19a0400, 0, 0, 0xa5a50800},
    {"Mul2FlushScalesHalf", RJ_ARITH_ALU_OMOD_2, 0x100, 0x6e8e3800, 0, 0, 0xa5a53c00},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D, 0x1f0, 0xd49c0000, 0, 0, 0x3400a5a5},
    {"HighSourceZero", RJ_ARITH_ALU_HIGH_A, 0x1f0, 0x7b3b8000, 0, 0, 0xa5a50000},
    {"SignalingKeptIeeeClear", 0, 0x1f0, 0xc3957c01, 0, 0, 0xa5a57c01},
    {"SignalingQuietedIeee", 0, 0x3f0, 0xc3957c01, 0, 0, 0xa5a57e01},
};

} // namespace

// Compatibility and strict exact reproduce the captured RDNA3 FRACT rules.
TEST(MathFractGfx11, CapturedOmodRoundingAndNans) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 32>(rj_arith_v_fract_f32_w32, fract_f32_gfx11_lanes);
  check_captured_gfx11_exact<64, 32>(rj_arith_v_fract_f32_w64, fract_f32_gfx11_lanes);
  check_captured_gfx11_exact<32, 16>(rj_arith_v_fract_f16_w32, fract_f16_gfx11_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_fract_f16_w64, fract_f16_gfx11_lanes);
}
