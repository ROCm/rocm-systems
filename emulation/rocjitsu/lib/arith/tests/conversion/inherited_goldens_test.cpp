// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Selected conversion boundaries retained from the retired VM numerical models.
// Public entry points preserve rounding, tininess, payload and seeded-rounding contracts.

#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <cstdint>
#include <gtest/gtest.h>

namespace {

using Unary = decltype(&rj_arith_v_cvt_f16_f32_w32_gfx12);

// Adapt the two integer-source FP32 APIs that have no FP-status parameter.
template <auto Call>
int without_status(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
                   const uint32_t *const *a, uint32_t *) {
  return Call(flags, exec_mask, modifiers, d, a);
}

// Check inherited literal results through each supported scalar/SIMD dispatch path.
TEST(ConversionInheritedGoldens, FloatingDestinations) {
  struct Case {
    Unary call;
    uint64_t input, expected, flags = 0, modifiers = 0;
  };
  const Case cases[] = {
      {rj_arith_v_cvt_f16_i16_w32, 0x12348000U, 0xf800},
      {rj_arith_v_cvt_f16_i16_w32, 0xffffffffU, 0xbc00},
      {rj_arith_v_cvt_f16_u16_w32, 0xffff0001U, 0x3c00},
      {rj_arith_v_cvt_f16_u16_w32, 0x0000fff0U, 0x7c00},
      {without_status<rj_arith_v_cvt_f32_i32_w32>, 0x80000001U, 0xceffffffU,
       RJ_ARITH_FP_ROUND_ZERO},
      {without_status<rj_arith_v_cvt_f32_u32_w32>, 0xffffffffU, 0x4fffffffU,
       RJ_ARITH_FP_ROUND_ZERO | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, RJ_ARITH_ALU_OMOD_2},
      {rj_arith_v_cvt_f16_i16_w32, 0x7fff, 0x3c00, 0, RJ_ARITH_ALU_CLAMP},
      {rj_arith_v_cvt_f16_i16_w32, 0xffff, 0, 0, RJ_ARITH_ALU_CLAMP},
      {rj_arith_v_cvt_f64_f32_w32, 1, 0xb6a0000000000000ULL, 0, RJ_ARITH_ALU_NEG_A},
      {rj_arith_v_cvt_f64_f32_w32, 1, 0x8000000000000000ULL, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS,
       RJ_ARITH_ALU_NEG_A},
      {rj_arith_v_cvt_f32_f64_w32, 0xbff0000000000000ULL, 0x3f800000, 0, RJ_ARITH_ALU_ABS_A},
      // Tininess is judged after rounding, before denormal encoding.
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0x387ff000U, 0x0400, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0xb87ff000U, 0x8400, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0x387fe000U, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0xb87fe000U, 0x8000, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0x387fe000U, 0x0400},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0x387ff000U, 0,
       RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_ZERO},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0x387fe001U, 0x0400,
       RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_UP},
      {rj_arith_v_cvt_f32_f64_w32_gfx12, 0x380fffffe0000000ULL, 0,
       RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {rj_arith_v_cvt_f32_f64_w32_gfx12, 0x380ffffff0000000ULL, 0x00800000,
       RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0x47800000U, 0x7c00},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0x47800000U, 0x7bff, RJ_ARITH_FP_ROUND_ZERO},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0xc7800000U, 0xfbff, RJ_ARITH_FP_ROUND_UP},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0xc7800000U, 0xfc00, RJ_ARITH_FP_ROUND_DOWN},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0x47800000U, 0x7bff, RJ_ARITH_FP16_OVFL},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0xff800000U, 0xfc00, RJ_ARITH_FP16_OVFL},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0x7f800001U, 0x7e00},
      {rj_arith_v_cvt_f16_f32_w32_gfx12, 0xffa12345U, 0xff09},
      {rj_arith_v_cvt_f32_f64_w32_gfx12, 0x7ff0000000000001ULL, 0x7fc00000},
  };
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const auto &test : cases) {
      uint32_t source[2][32]{}, result[2][32]{};
      source[0][0] = uint32_t(test.input);
      source[1][0] = uint32_t(test.input >> 32);
      const uint32_t *a[] = {source[0], source[1]};
      uint32_t *d[] = {result[0], result[1]};
      uint32_t status = 0;
      ASSERT_EQ(test.call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | test.flags, 1, test.modifiers, d,
                          a, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(result[0][0] | (uint64_t(result[1][0]) << 32), test.expected)
          << "input=" << test.input << " cpu=" << cpu;
    }
  }
}

// Check overflow, canonical NaNs and seed addition without a retained conversion model.
TEST(ConversionInheritedGoldens, Fp8Narrowing) {
  using Binary = decltype(&rj_arith_v_cvt_sr_fp8_f32_w32);
  struct Case {
    Binary call;
    uint32_t input, seed, expected;
    bool saturate = false;
  };
  const Case cases[] = {
      {rj_arith_v_cvt_pk_fp8_f32_w32, 0x43e80000U, 0, 0x7e},
      {rj_arith_v_cvt_pk_fp8_f32_w32, 0x43ec0000U, 0, 0x7f},
      {rj_arith_v_cvt_pk_fp8_f32_w32, 0xc3ec0000U, 0, 0xfe, true},
      {rj_arith_v_cvt_pk_bf8_f32_w32, 0x7f7fffffU, 0, 0x7c},
      {rj_arith_v_cvt_pk_bf8_f32_w32, 0x7f7fffffU, 0, 0x7b, true},
      {rj_arith_v_cvt_pk_fp8_f32_w32, 0x7f800000U, 0, 0x7f, true},
      {rj_arith_v_cvt_pk_bf8_f32_w32, 0xff800000U, 0, 0xfc, true},
      {rj_arith_v_cvt_pk_fp8_f32_w32, 0x7fc00000U, 0, 0xff},
      {rj_arith_v_cvt_pk_bf8_f32_w32, 0xff800001U, 0, 0xfe},
      {rj_arith_v_cvt_sr_fp8_f32_w32, 0x3f880000U, 0, 0x38},
      {rj_arith_v_cvt_sr_fp8_f32_w32, 0x3f880000U, 0x80000000U, 0x39},
      {rj_arith_v_cvt_sr_fp8_f32_w32, 0x384b85fdU, 0xffffffffU, 1},
  };
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const auto &test : cases) {
      uint32_t source[32]{}, other[32]{}, result[32]{};
      source[0] = test.input;
      other[0] = test.seed;
      const uint32_t *a = source, *b = other;
      uint32_t *d = result;
      uint64_t flags =
          cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | (test.saturate ? RJ_ARITH_FP16_OVFL : 0);
      ASSERT_EQ(test.call(flags, 1, 0, &d, &a, &b), RJ_ARITH_SUCCESS);
      EXPECT_EQ(result[0], test.expected) << "input=" << test.input << " cpu=" << cpu;
    }
  }
}

// Retain selected exact FP8/BF8 widening literals, including a subnormal and infinity.
TEST(ConversionInheritedGoldens, Fp8Widening) {
  using Widen = decltype(&rj_arith_v_cvt_f32_fp8_w32);
  struct Case {
    Widen call;
    uint32_t input, expected;
  };
  const Case cases[] = {
      {rj_arith_v_cvt_f32_fp8_w32, 0x7e, 0x43e00000U},
      {rj_arith_v_cvt_f32_fp8_w32, 1, 0x3b000000U},
      {rj_arith_v_cvt_f32_bf8_w32, 0xfc, 0xff800000U},
      {rj_arith_v_cvt_f32_fp8_w32, 0x7f, 0xffc00000U},
  };
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const auto &test : cases) {
      uint32_t source[32]{}, result[32]{};
      source[0] = test.input;
      const uint32_t *a = source;
      uint32_t *d = result;
      ASSERT_EQ(test.call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, 0, &d, &a), RJ_ARITH_SUCCESS);
      EXPECT_EQ(result[0], test.expected) << "input=" << test.input << " cpu=" << cpu;
    }
  }
}

} // namespace
