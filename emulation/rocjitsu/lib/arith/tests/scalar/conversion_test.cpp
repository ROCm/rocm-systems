// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar integer/FP conversions and packed FP16 results.

#include "scalar/fp_helpers.h"
#include <algorithm>
#include <limits>

#include "conversion/conversion16_reference.h"
#include "conversion/conversion32_reference.h"
#include "conversion/packed_conversion_reference.h"
#include "rj_arith/detail/rj_arith_scalar.h"

// Check scalar FP results and the expected semantics/status policy.
TEST(s_cvt_f32_i32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_cvt_f32_i32(f, m, d, a, status);
      },
      [](uint32_t a, uint32_t b, auto, bool flush) {
        (void)b;

        (void)flush;

        return rocjitsu::arith::test::conversion32_reference(0, a, 0);
      },
      false, 0);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_cvt_f32_u32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_cvt_f32_u32(f, m, d, a, status);
      },
      [](uint32_t a, uint32_t b, auto, bool flush) {
        (void)b;

        (void)flush;

        return rocjitsu::arith::test::conversion32_reference(1, a, 0);
      },
      false, 0);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_cvt_i32_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_cvt_i32_f32(f, m, d, a, status);
      },
      [](uint32_t a, uint32_t b, auto, bool flush) {
        (void)b;

        if (flush && !(a & 0x7f800000U)) {
          a &= 0x80000000U;
        }

        return rocjitsu::arith::test::conversion32_reference(2, a, 0);
      },
      false, 1);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_cvt_u32_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_cvt_u32_f32(f, m, d, a, status);
      },
      [](uint32_t a, uint32_t b, auto, bool flush) {
        (void)b;

        if (flush && !(a & 0x7f800000U)) {
          a &= 0x80000000U;
        }

        return rocjitsu::arith::test::conversion32_reference(3, a, 0);
      },
      false, 1);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_cvt_f16_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_cvt_f16_f32(f, m, d, a, status);
      },
      [](uint32_t a, uint32_t b, auto, bool flush) {
        (void)b;

        if (flush && !(a & 0x7f800000U)) {
          a &= 0x80000000U;
        }

        return rocjitsu::arith::test::conversion16_reference(4, a, 0, 0, false);
      },
      false, 2);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_cvt_f32_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_cvt_f32_f16(f, m, d, a, status);
      },
      [](uint32_t a, uint32_t b, auto, bool flush) {
        (void)b;

        if (flush && !(a & 0x7c00U)) {
          a &= 0x8000U;
        }

        return rocjitsu::arith::test::conversion16_reference(5, a, 0, 0, false);
      },
      false, 0);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_cvt_hi_f32_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_cvt_hi_f32_f16(f, m, d, a, status);
      },
      [](uint32_t a, uint32_t b, auto, bool flush) {
        (void)b;
        a >>= 16;
        if (flush && !(a & 0x7c00U)) {
          a &= 0x8000U;
        }

        return rocjitsu::arith::test::conversion16_reference(5, a, 0, 0, false);
      },
      false, 0);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_cvt_pk_rtz_f16_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_cvt_pk_rtz_f16_f32(f, m, d, a, b, status);
      },
      [](uint32_t a, uint32_t b, auto, bool flush) {
        (void)b;

        if (flush && !(a & 0x7f800000U)) {
          a &= 0x80000000U;
        }
        if (flush && !(b & 0x7f800000U)) {
          b &= 0x80000000U;
        }
        return rocjitsu::arith::test::packed_conversion_reference(0, a, b, 0);
      },
      false, 1);
}

// Literal boundaries check legacy saturation, signaling payloads, and halfword placement.
TEST(ScalarConversionGfx1250, ArithmeticAndPolicy) {
  using Unary = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint32_t *);
  struct Case {
    Unary call;
    uint32_t input, expected;
  };
  const Case cases[] = {{rj_arith_s_cvt_f32_i32_gfx1250, 0x80000000, 0xcf000000},
                        {rj_arith_s_cvt_f32_u32_gfx1250, 0xffffffff, 0x4f800000},
                        {rj_arith_s_cvt_i32_f32_gfx1250, 0x7f800000, 0x7fffffff},
                        {rj_arith_s_cvt_i32_f32_gfx1250, 0xff800000, 0x80000000},
                        {rj_arith_s_cvt_i32_f32_gfx1250, 0x7f800001, 0},
                        {rj_arith_s_cvt_u32_f32_gfx1250, 0x4f800000, 0xffffffff},
                        {rj_arith_s_cvt_u32_f32_gfx1250, 0xbf800000, 0},
                        {rj_arith_s_cvt_f16_f32_gfx1250, 0x33800000, 1},
                        {rj_arith_s_cvt_f16_f32_gfx1250, 0x7f800001, 0x7c01},
                        {rj_arith_s_cvt_f16_f32_gfx1250, 0x47800000, 0x7bff},
                        {rj_arith_s_cvt_f32_f16_gfx1250, 0x7c01, 0x7f802000},
                        {rj_arith_s_cvt_hi_f32_f16_gfx1250, 0x7c010000, 0x7f802000},
                        {rj_arith_s_cvt_hi_f32_f16_gfx1250, 0x00010000, 0x33800000}};
  using namespace rocjitsu::arith::test;
  for (const auto &test : cases) {
    for (auto semantics : semantics_modes()) {
      uint64_t flags = semantics | RJ_ARITH_FP16_OVFL;
      if (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
        flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
      }
      uint32_t d = test.input, status = 0x80000040U;
      bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                    semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
      ASSERT_EQ(test.call(flags, 0, &d, d, &status),
                strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
      EXPECT_EQ(d, strict ? test.input : test.expected);
      EXPECT_EQ(status, 0x80000040U);
    }
    uint32_t d = 123, status = 456;
    expect_invalid_flags(
        [&](auto f, auto m) {
          int error = test.call(f, m, &d, 0, &status);
          EXPECT_EQ(d, 123U);
          EXPECT_EQ(status, 456U);
          return error;
        },
        0);
  }
  struct Packed {
    uint32_t a, b, expected;
  };
  const Packed packed[] = {{0x7f800001, 0xffc12345, 0xfe097c01},
                           {0x47800000, 0xc7800000, 0xfbff7bff},
                           {0x33800000, 0xb3800000, 0x80010001}};
  for (const auto &test : packed) {
    for (auto semantics : semantics_modes()) {
      uint64_t flags = semantics;
      if (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
        flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
      }
      uint32_t d = test.a, status = 0x80000040U;
      bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                    semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
      ASSERT_EQ(rj_arith_s_cvt_pk_rtz_f16_f32_gfx1250(flags, 0, &d, d, test.b, &status),
                strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
      EXPECT_EQ(d, strict ? test.a : test.expected);
      EXPECT_EQ(status, 0x80000040U);
    }
  }
  uint32_t d = 123, status = 456;
  expect_invalid_flags(
      [&](auto f, auto m) {
        int error = rj_arith_s_cvt_pk_rtz_f16_f32_gfx1250(f, m, &d, 0, 0, &status);
        EXPECT_EQ(d, 123U);
        EXPECT_EQ(status, 456U);
        return error;
      },
      0);
}
