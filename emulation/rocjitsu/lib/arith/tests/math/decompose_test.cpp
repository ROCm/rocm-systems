// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for fractional-part and mantissa decomposition.

#include "math/unary_test.h"

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
