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
      true, 0, false);
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
      true, 0, false);
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
      true, 1, false);
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
      true, 1, false);
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
      true, 2, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 1, false);
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

// gfx1201 captures: special, denormal, overflowing and inexact inputs in default MODE
// 0xf0 and in flushing, directed and FP16_OVFL MODEs; inputs whose result changes with
// the rounding direction in all four directions; and one input under MODE 0xf3/0xfc,
// whose F32 and F16 rounding fields differ. c is the FMAC accumulator or the
// FMAAK/FMAMK literal.
const rocjitsu::arith::test::ScalarCapture s_cvt_f32_i32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x7fffffff, 0x0, 0x0, 0x4f000000},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7fffffff, 0x0, 0x0, 0x4f000000},
    {0xf0, 0x7fffffff, 0x0, 0x0, 0x4f000000},
    {0xf5, 0x7fffffff, 0x0, 0x0, 0x4f000000},
    {0xfa, 0x7fffffff, 0x0, 0x0, 0x4effffff},
    {0xff, 0x7fffffff, 0x0, 0x0, 0x4effffff},
    {0xf0, 0x80000001, 0x0, 0x0, 0xcf000000},
    {0xf5, 0x80000001, 0x0, 0x0, 0xceffffff},
    {0xfa, 0x80000001, 0x0, 0x0, 0xcf000000},
    {0xff, 0x80000001, 0x0, 0x0, 0xceffffff},
    {0xf0, 0x12345678, 0x0, 0x0, 0x4d91a2b4},
    {0xf5, 0x12345678, 0x0, 0x0, 0x4d91a2b4},
    {0xfa, 0x12345678, 0x0, 0x0, 0x4d91a2b3},
    {0xff, 0x12345678, 0x0, 0x0, 0x4d91a2b3},
    {0xf3, 0x7fffffff, 0x0, 0x0, 0x4effffff},
    {0xfc, 0x7fffffff, 0x0, 0x0, 0x4f000000},
};

TEST(s_cvt_f32_i32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto, auto, auto status) {
        return rj_arith_s_cvt_f32_i32(f, 0, d, a, status);
      },
      s_cvt_f32_i32_captures, 0, 32);
}

const rocjitsu::arith::test::ScalarCapture s_cvt_f32_u32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0xffffffff, 0x0, 0x0, 0x4f800000},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0xffffffff, 0x0, 0x0, 0x4f800000},
    {0xf0, 0xffffffff, 0x0, 0x0, 0x4f800000},
    {0xf5, 0xffffffff, 0x0, 0x0, 0x4f800000},
    {0xfa, 0xffffffff, 0x0, 0x0, 0x4f7fffff},
    {0xff, 0xffffffff, 0x0, 0x0, 0x4f7fffff},
    {0xf0, 0x7fffffff, 0x0, 0x0, 0x4f000000},
    {0xf5, 0x7fffffff, 0x0, 0x0, 0x4f000000},
    {0xfa, 0x7fffffff, 0x0, 0x0, 0x4effffff},
    {0xff, 0x7fffffff, 0x0, 0x0, 0x4effffff},
    {0xf0, 0x80000001, 0x0, 0x0, 0x4f000000},
    {0xf5, 0x80000001, 0x0, 0x0, 0x4f000001},
    {0xfa, 0x80000001, 0x0, 0x0, 0x4f000000},
    {0xff, 0x80000001, 0x0, 0x0, 0x4f000000},
    {0xf3, 0xffffffff, 0x0, 0x0, 0x4f7fffff},
    {0xfc, 0xffffffff, 0x0, 0x0, 0x4f800000},
};

TEST(s_cvt_f32_u32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto, auto, auto status) {
        return rj_arith_s_cvt_f32_u32(f, 0, d, a, status);
      },
      s_cvt_f32_u32_captures, 0, 32);
}

const rocjitsu::arith::test::ScalarCapture s_cvt_f16_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x800000, 0x0, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x0, 0x0, 0x7c00},
    {0xf0, 0x7f800001, 0x0, 0x0, 0x7e00},
    {0xf0, 0x40490fdb, 0x0, 0x0, 0x4248},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x800000, 0x0, 0x0, 0x0},
    {0x0, 0x7f7fffff, 0x0, 0x0, 0x7c00},
    {0x0, 0x7f800001, 0x0, 0x0, 0x7e00},
    {0x0, 0x40490fdb, 0x0, 0x0, 0x4248},
    {0x10, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x0, 0x0, 0x7c00},
    {0xf5, 0x7f7fffff, 0x0, 0x0, 0x7c00},
    {0xfa, 0x7f7fffff, 0x0, 0x0, 0x7bff},
    {0xff, 0x7f7fffff, 0x0, 0x0, 0x7bff},
    {0xf0, 0xff7fffff, 0x0, 0x0, 0xfc00},
    {0xf5, 0xff7fffff, 0x0, 0x0, 0xfbff},
    {0xfa, 0xff7fffff, 0x0, 0x0, 0xfc00},
    {0xff, 0xff7fffff, 0x0, 0x0, 0xfbff},
    {0xf0, 0x40490fdb, 0x0, 0x0, 0x4248},
    {0xf5, 0x40490fdb, 0x0, 0x0, 0x4249},
    {0xfa, 0x40490fdb, 0x0, 0x0, 0x4248},
    {0xff, 0x40490fdb, 0x0, 0x0, 0x4248},
    {0xf3, 0x7f7fffff, 0x0, 0x0, 0x7c00},
    {0xfc, 0x7f7fffff, 0x0, 0x0, 0x7bff},
    // FP16_OVFL saturates finite overflow in every rounding mode; infinity stays infinite.
    {0xf0, 0x47800000, 0x0, 0x0, 0x7c00},
    {0x8000f0, 0x47800000, 0x0, 0x0, 0x7bff},
    {0x8000f5, 0x47800000, 0x0, 0x0, 0x7bff},
    {0xf5, 0xc7800000, 0x0, 0x0, 0xfbff},
    {0x8000f0, 0xc7800000, 0x0, 0x0, 0xfbff},
    {0x8000f5, 0xc7800000, 0x0, 0x0, 0xfbff},
    {0x8000f0, 0x7f800000, 0x0, 0x0, 0x7c00},
    {0x8000f5, 0x7f800000, 0x0, 0x0, 0x7c00},
};

TEST(s_cvt_f16_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto, auto, auto status) {
        return rj_arith_s_cvt_f16_f32(f, 0, d, a, status);
      },
      s_cvt_f16_f32_captures, 32, 16);
}

// gfx1201 captures by operand class (NaNs in either order, signed zeros, denormals,
// ordered and out-of-range values) in default MODE 0xf0 and in MODE 0x00, where
// inputs and outputs are flushed.
const rocjitsu::arith::test::ScalarCapture s_cvt_i32_f32_captures[] = {
    {0xf0, 0x80000000, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x800000, 0x0, 0x0, 0x0},
    {0xf0, 0xbf800000, 0x0, 0x0, 0xffffffff},
    {0xf0, 0x7f7fffff, 0x0, 0x0, 0x7fffffff},
    {0xf0, 0xff7fffff, 0x0, 0x0, 0x80000000},
    {0xf0, 0x7f800000, 0x0, 0x0, 0x7fffffff},
    {0xf0, 0x7fc00000, 0x0, 0x0, 0x0},
    {0xf0, 0x7f800001, 0x0, 0x0, 0x0},
    {0x0, 0x80000000, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x800000, 0x0, 0x0, 0x0},
    {0x0, 0xbf800000, 0x0, 0x0, 0xffffffff},
    {0x0, 0x7f7fffff, 0x0, 0x0, 0x7fffffff},
    {0x0, 0xff7fffff, 0x0, 0x0, 0x80000000},
    {0x0, 0x7f800000, 0x0, 0x0, 0x7fffffff},
    {0x0, 0x7fc00000, 0x0, 0x0, 0x0},
    {0x0, 0x7f800001, 0x0, 0x0, 0x0},
};

TEST(s_cvt_i32_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto, auto, auto status) {
        return rj_arith_s_cvt_i32_f32(f, 0, d, a, status);
      },
      s_cvt_i32_f32_captures, 32, 0, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_cvt_u32_f32_captures[] = {
    {0xf0, 0x80000000, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x800000, 0x0, 0x0, 0x0},
    {0xf0, 0xbf800000, 0x0, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x0, 0x0, 0xffffffff},
    {0xf0, 0xff7fffff, 0x0, 0x0, 0x0},
    {0xf0, 0x7f800000, 0x0, 0x0, 0xffffffff},
    {0xf0, 0x7fc00000, 0x0, 0x0, 0x0},
    {0xf0, 0x7f800001, 0x0, 0x0, 0x0},
    {0x0, 0x80000000, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x800000, 0x0, 0x0, 0x0},
    {0x0, 0xbf800000, 0x0, 0x0, 0x0},
    {0x0, 0x7f7fffff, 0x0, 0x0, 0xffffffff},
    {0x0, 0xff7fffff, 0x0, 0x0, 0x0},
    {0x0, 0x7f800000, 0x0, 0x0, 0xffffffff},
    {0x0, 0x7fc00000, 0x0, 0x0, 0x0},
    {0x0, 0x7f800001, 0x0, 0x0, 0x0},
};

TEST(s_cvt_u32_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto, auto, auto status) {
        return rj_arith_s_cvt_u32_f32(f, 0, d, a, status);
      },
      s_cvt_u32_f32_captures, 32, 0, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_cvt_f32_f16_captures[] = {
    {0xf0, 0x8000, 0x0, 0x0, 0x80000000},
    {0xf0, 0x1, 0x0, 0x0, 0x33800000},
    {0xf0, 0x400, 0x0, 0x0, 0x38800000},
    {0xf0, 0xbc00, 0x0, 0x0, 0xbf800000},
    {0xf0, 0x7c00, 0x0, 0x0, 0x7f800000},
    {0xf0, 0x7e00, 0x0, 0x0, 0x7fc00000},
    {0xf0, 0x7c01, 0x0, 0x0, 0x7fc02000},
    {0x0, 0x8000, 0x0, 0x0, 0x80000000},
    {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x400, 0x0, 0x0, 0x38800000},
    {0x0, 0xbc00, 0x0, 0x0, 0xbf800000},
    {0x0, 0x7c00, 0x0, 0x0, 0x7f800000},
    {0x0, 0x7e00, 0x0, 0x0, 0x7fc00000},
    {0x0, 0x7c01, 0x0, 0x0, 0x7fc02000},
};

TEST(s_cvt_f32_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto, auto, auto status) {
        return rj_arith_s_cvt_f32_f16(f, 0, d, a, status);
      },
      s_cvt_f32_f16_captures, 16, 32, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_cvt_hi_f32_f16_captures[] = {
    {0xf0, 0x80000000, 0x0, 0x0, 0x80000000}, {0xf0, 0x20083ff, 0x0, 0x0, 0x38000000},
    {0xf0, 0x42003c00, 0x0, 0x0, 0x40400000}, {0xf0, 0xbc008000, 0x0, 0x0, 0xbf800000},
    {0xf0, 0xfc004000, 0x0, 0x0, 0xff800000}, {0xf0, 0x7e000001, 0x0, 0x0, 0x7fc00000},
    {0xf0, 0xfc01fe00, 0x0, 0x0, 0xffc02000}, {0x0, 0x80000000, 0x0, 0x0, 0x80000000},
    {0x0, 0x20083ff, 0x0, 0x0, 0x0},          {0x0, 0x42003c00, 0x0, 0x0, 0x40400000},
    {0x0, 0xbc008000, 0x0, 0x0, 0xbf800000},  {0x0, 0xfc004000, 0x0, 0x0, 0xff800000},
    {0x0, 0x7e000001, 0x0, 0x0, 0x7fc00000},  {0x0, 0xfc01fe00, 0x0, 0x0, 0xffc02000},
};

TEST(s_cvt_hi_f32_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto, auto, auto status) {
        return rj_arith_s_cvt_hi_f32_f16(f, 0, d, a, status);
      },
      s_cvt_hi_f32_f16_captures, 16, 32, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_cvt_pk_rtz_f16_f32_captures[] = {
    {0xf0, 0x7f800001, 0x7fc00000, 0x0, 0x7e007e00},
    {0xf0, 0x7fc00000, 0x7f800001, 0x0, 0x7e007e00},
    {0xf0, 0x7fc00000, 0x800000, 0x0, 0x7e00},
    {0xf0, 0x800000, 0x7f800001, 0x0, 0x7e000000},
    {0xf0, 0x0, 0x80000000, 0x0, 0x80000000},
    {0xf0, 0x80000000, 0x0, 0x0, 0x8000},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x1, 0x0, 0x0},
    {0xf0, 0x1, 0x800000, 0x0, 0x0},
    {0xf0, 0x800000, 0x800000, 0x0, 0x0},
    {0xf0, 0x7f800000, 0x800000, 0x0, 0x7c00},
    {0xf0, 0x7f7fffff, 0x800000, 0x0, 0x7bff},
    {0x0, 0x7f800001, 0x7fc00000, 0x0, 0x7e007e00},
    {0x0, 0x7fc00000, 0x7f800001, 0x0, 0x7e007e00},
    {0x0, 0x7fc00000, 0x800000, 0x0, 0x7e00},
    {0x0, 0x800000, 0x7f800001, 0x0, 0x7e000000},
    {0x0, 0x0, 0x80000000, 0x0, 0x80000000},
    {0x0, 0x80000000, 0x0, 0x0, 0x8000},
    {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x1, 0x0, 0x0},
    {0x0, 0x1, 0x800000, 0x0, 0x0},
    {0x0, 0x800000, 0x800000, 0x0, 0x0},
    {0x0, 0x7f800000, 0x800000, 0x0, 0x7c00},
    {0x0, 0x7f7fffff, 0x800000, 0x0, 0x7bff},
};

TEST(s_cvt_pk_rtz_f16_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_cvt_pk_rtz_f16_f32(f, 0, d, a, b, status);
      },
      s_cvt_pk_rtz_f16_f32_captures, 32, 16, /*rounded=*/false);
}
