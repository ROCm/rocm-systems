// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compact gfx1150 scalar FP witnesses: inherited IEEE policy, status and qualification.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
#include "rj_arith/detail/rj_arith_scalar.h"
#include <algorithm>
#include <bit>
#include <gtest/gtest.h>
#include <random>

namespace {
using Binary = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint32_t, uint32_t *);
using Unary = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint32_t *);
using Ternary = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint32_t, uint32_t, uint32_t *);
constexpr uint32_t kInitialFpStatus = 0x100;

// All new inherited models reject unqualified strict semantics before touching outputs.
template <typename Call> void check_rejections(Call call) {
  for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
    uint32_t result = 0xdeadbeef, status = kInitialFpStatus;
    EXPECT_EQ(call(semantics, &result, &status), RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(result, 0xdeadbeefU);
    EXPECT_EQ(status, kInitialFpStatus);
  }
}

// Exact normal results and one signaling-NaN witness distinguish old and new selection.
TEST(Gfx1150ScalarFp, Binary) {
  struct Case {
    Binary call;
    bool half;
    int result;
    bool selection;
  };
  const Case cases[] = {
      {rj_arith_s_add_f16_gfx1150, true, 3, false},  {rj_arith_s_sub_f16_gfx1150, true, 1, false},
      {rj_arith_s_mul_f16_gfx1150, true, 2, false},  {rj_arith_s_min_f16_gfx1150, true, 1, true},
      {rj_arith_s_max_f16_gfx1150, true, 2, true},   {rj_arith_s_add_f32_gfx1150, false, 3, false},
      {rj_arith_s_sub_f32_gfx1150, false, 1, false}, {rj_arith_s_mul_f32_gfx1150, false, 2, false},
      {rj_arith_s_min_f32_gfx1150, false, 1, true},  {rj_arith_s_max_f32_gfx1150, false, 2, true},
  };
  for (const auto &test : cases) {
    const uint32_t one = test.half ? 0x3c00 : 0x3f800000;
    const uint32_t two = test.half ? 0x4000 : 0x40000000;
    const uint32_t snan = test.half ? 0x7c11 : 0x7f800123;
    const uint32_t quiet = test.half ? 0x200 : 0x400000;
    for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
      for (uint64_t semantics :
           {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
            RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
        uint32_t result = 0, status = kInitialFpStatus;
        ASSERT_EQ(test.call(cpu | semantics, 0, &result, two, one, &status), RJ_ARITH_SUCCESS);
        EXPECT_EQ(result, test.result == 1   ? one
                          : test.result == 2 ? two
                          : test.half        ? 0x4200
                                             : 0x40400000);
        EXPECT_EQ(status, kInitialFpStatus);
      }
      for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
        uint32_t result = 0, status = kInitialFpStatus;
        ASSERT_EQ(test.call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee, 0, &result, snan, one,
                            &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(result, test.selection && !ieee ? one : snan | (ieee ? quiet : 0));
        EXPECT_EQ(status, kInitialFpStatus | RJ_ARITH_EXCEPTION_INVALID);
      }
    }
    // Finite, nearest-even arithmetic has the inherited ordinary IEEE result.
    // Fully specified engine/high bits keep input generation independent of the STL.
    std::minstd_rand random;
    for (int sample = 0; sample < 32; ++sample) {
      const uint32_t a = test.half ? 0x3800U | (random() >> 21) : 0x3f000000U | (random() >> 8);
      const uint32_t b = test.half ? 0x3800U | (random() >> 21) : 0x3f000000U | (random() >> 8);
      const double x = test.half ? rocjitsu::arith::test::half_value(a) : std::bit_cast<float>(a);
      const double y = test.half ? rocjitsu::arith::test::half_value(b) : std::bit_cast<float>(b);
      const double exact = test.selection     ? (test.result == 1 ? std::min(x, y) : std::max(x, y))
                           : test.result == 3 ? x + y
                           : test.result == 1 ? x - y
                                              : x * y;
      const uint32_t expected = test.half ? rocjitsu::arith::test::half_bits(exact)
                                          : std::bit_cast<uint32_t>(float(exact));
      uint32_t result = 0, status = kInitialFpStatus;
      ASSERT_EQ(test.call(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, &result, a, b, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(result, expected);
      ASSERT_EQ(test.call(RJ_ARITH_SEMANTICS_LOOSE, 0, &result, a, b, &status), RJ_ARITH_SUCCESS);
      const float actual = test.half ? float(rocjitsu::arith::test::half_value(result))
                                     : std::bit_cast<float>(result);
      const float reference = test.half ? float(rocjitsu::arith::test::half_value(expected))
                                        : std::bit_cast<float>(expected);
      rocjitsu::arith::test::expect_f32(std::bit_cast<uint32_t>(actual), reference,
                                        RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/1e-7f,
                                        /*rtol=*/1e-6f);
    }
    check_rejections([&](auto flags, auto result, auto status) {
      return test.call(flags, 0, result, two, one, status);
    });
  }
}

// Integral rounding shares its finite algorithm but only IEEE mode quiets old NaNs.
TEST(Gfx1150ScalarFp, Round) {
  struct Case {
    Unary call;
    bool half;
    int result;
  };
  const Case cases[] = {
      {rj_arith_s_ceil_f16_gfx1150, true, 2},   {rj_arith_s_floor_f16_gfx1150, true, 1},
      {rj_arith_s_trunc_f16_gfx1150, true, 1},  {rj_arith_s_rndne_f16_gfx1150, true, 2},
      {rj_arith_s_ceil_f32_gfx1150, false, 2},  {rj_arith_s_floor_f32_gfx1150, false, 1},
      {rj_arith_s_trunc_f32_gfx1150, false, 1}, {rj_arith_s_rndne_f32_gfx1150, false, 2},
  };
  for (const auto &test : cases) {
    const uint32_t input = test.half ? 0x3e00 : 0x3fc00000;
    const uint32_t one = test.half ? 0x3c00 : 0x3f800000;
    const uint32_t two = test.half ? 0x4000 : 0x40000000;
    const uint32_t snan = test.half ? 0x7c11 : 0x7f800123;
    for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
      uint32_t result = 0, status = kInitialFpStatus;
      ASSERT_EQ(test.call(RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee, 0, &result, input, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(result, test.result == 1 ? one : two);
      EXPECT_EQ(status, kInitialFpStatus);
      ASSERT_EQ(test.call(RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee, 0, &result, snan, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(result, snan | (ieee ? (test.half ? 0x200U : 0x400000U) : 0));
      EXPECT_EQ(status, kInitialFpStatus | RJ_ARITH_EXCEPTION_INVALID);
    }
    check_rejections([&](auto flags, auto result, auto status) {
      return test.call(flags, 0, result, input, status);
    });
  }
}

// FMA/FMAC retain accumulator versus literal operand ordering and IEEE NaNs.
TEST(Gfx1150ScalarFp, Fma) {
  struct Case {
    Ternary call;
    bool half;
    bool accumulator;
  };
  const Case cases[] = {
      {[](auto f, auto m, auto d, auto a, auto b, auto, auto status) {
         return rj_arith_s_fmac_f16_gfx1150(f, m, d, a, b, status);
       },
       true, true},
      {[](auto f, auto m, auto d, auto a, auto b, auto, auto status) {
         return rj_arith_s_fmac_f32_gfx1150(f, m, d, a, b, status);
       },
       false, true},
      {rj_arith_s_fmaak_f32_gfx1150, false, false},
      {rj_arith_s_fmamk_f32_gfx1150, false, false},
  };
  for (const auto &test : cases) {
    const uint32_t one = test.half ? 0x3c00 : 0x3f800000;
    const uint32_t two = test.half ? 0x4000 : 0x40000000;
    const uint32_t snan = test.half ? 0x7c11 : 0x7f800123;
    for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
      uint32_t result = one, status = kInitialFpStatus;
      ASSERT_EQ(
          test.call(RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee, 0, &result, one, one, one, &status),
          RJ_ARITH_SUCCESS);
      EXPECT_EQ(result, two);
      result = snan;
      ASSERT_EQ(
          test.call(RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee, 0, &result, one, one, snan, &status),
          RJ_ARITH_SUCCESS);
      EXPECT_EQ(result, snan | (ieee ? (test.half ? 0x200U : 0x400000U) : 0));
      EXPECT_EQ(status, kInitialFpStatus | RJ_ARITH_EXCEPTION_INVALID);
    }
    check_rejections([&](auto flags, auto result, auto status) {
      return test.call(flags, 0, result, one, one, one, status);
    });
  }
}

// Conversion goldens cover both directions, the high-half carrier and packed RTZ causes.
TEST(Gfx1150ScalarFp, Conversion) {
  struct Case {
    Unary call;
    uint32_t input, result;
  };
  const Case cases[] = {
      {rj_arith_s_cvt_f32_i32_gfx1150, 0x3, 0x40400000},
      {rj_arith_s_cvt_f32_u32_gfx1150, 0x3, 0x40400000},
      {rj_arith_s_cvt_i32_f32_gfx1150, 0x40400000, 0x3},
      {rj_arith_s_cvt_u32_f32_gfx1150, 0x40400000, 0x3},
      {rj_arith_s_cvt_f16_f32_gfx1150, 0x40400000, 0x4200},
      {rj_arith_s_cvt_f32_f16_gfx1150, 0x4200, 0x40400000},
      {rj_arith_s_cvt_hi_f32_f16_gfx1150, 0x4200a55a, 0x40400000},
  };
  // gfx1150 reports fractional INEXACT; gfx12 deliberately suppresses it.
  for (auto call : {rj_arith_s_cvt_i32_f32, rj_arith_s_cvt_i32_f32_gfx1150}) {
    uint32_t result = 0, status = kInitialFpStatus;
    ASSERT_EQ(call(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, &result, 0x3fc00000, &status),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(result, 1U);
    EXPECT_EQ(status,
              kInitialFpStatus |
                  (call == rj_arith_s_cvt_i32_f32_gfx1150 ? RJ_ARITH_EXCEPTION_INEXACT : 0));
    status = kInitialFpStatus;
    ASSERT_EQ(call(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, &result, 0x7f800000, &status),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(status, kInitialFpStatus | RJ_ARITH_EXCEPTION_INVALID);
  }
  for (const auto &test : cases) {
    uint32_t result = 0, status = kInitialFpStatus;
    ASSERT_EQ(test.call(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, &result, test.input, &status),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(result, test.result);
    EXPECT_EQ(status, kInitialFpStatus);
    check_rejections([&](auto flags, auto output, auto causes) {
      return test.call(flags, 0, output, test.input, causes);
    });
  }
  uint32_t result = 0, status = kInitialFpStatus;
  ASSERT_EQ(rj_arith_s_cvt_pk_rtz_f16_f32_gfx1150(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, &result,
                                                  0x3f800001, 0x40000000, &status),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(result, 0x40003c00U);
  EXPECT_EQ(status, kInitialFpStatus | RJ_ARITH_EXCEPTION_INEXACT);
  check_rejections([&](auto flags, auto output, auto causes) {
    return rj_arith_s_cvt_pk_rtz_f16_f32_gfx1150(flags, 0, output, 0x3f800001, 0x40000000, causes);
  });
}
} // namespace
