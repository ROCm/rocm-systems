// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar floating-point arithmetic and selection.

#include "compare/floating_point.h"
#include "rj_arith/detail/rj_arith_scalar.h"
#include "scalar/fp_helpers.h"
#include <algorithm>
#include <limits>

// Check scalar FP results and the expected semantics/status policy.
TEST(s_add_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_add_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(x + y);
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_sub_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_sub_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(x - y);
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_mul_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_mul_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(x * y);
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_min_num_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_min_num_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(
            (std::isnan(x) || std::isnan(y)) ? (std::isnan(x) ? y : x)
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) || std::signbit(y)) ? -1.0 : 1.0)
                : std::min(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_max_num_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_max_num_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(
            (std::isnan(x) || std::isnan(y)) ? (std::isnan(x) ? y : x)
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) && std::signbit(y)) ? -1.0 : 1.0)
                : std::max(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_minimum_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_minimum_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(
            (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN()
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) || std::signbit(y)) ? -1.0 : 1.0)
                : std::min(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_maximum_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_maximum_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(
            (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN()
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) && std::signbit(y)) ? -1.0 : 1.0)
                : std::max(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_fmac_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_fmac_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::fma(float(x), float(y), float(z)));
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_add_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_add_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(x + y);
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_sub_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_sub_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(x - y);
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_mul_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_mul_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(x * y);
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_min_num_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_min_num_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(
            (std::isnan(x) || std::isnan(y)) ? (std::isnan(x) ? y : x)
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) || std::signbit(y)) ? -1.0 : 1.0)
                : std::min(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_max_num_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_max_num_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(
            (std::isnan(x) || std::isnan(y)) ? (std::isnan(x) ? y : x)
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) && std::signbit(y)) ? -1.0 : 1.0)
                : std::max(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_minimum_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_minimum_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(
            (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN()
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) || std::signbit(y)) ? -1.0 : 1.0)
                : std::min(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_maximum_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_maximum_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(
            (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN()
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) && std::signbit(y)) ? -1.0 : 1.0)
                : std::max(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_fmac_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_fmac_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::fma(x, y, z));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_fmaak_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        return rj_arith_s_fmaak_f32(f, m, d, a, b, c, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::fma(float(x), float(y), float(z)));
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_fmamk_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        return rj_arith_s_fmamk_f32(f, m, d, a, c, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::fma(float(x), float(z), float(y)));
      },
      true, 0, false);
}

// All scalar binary forms share the legacy host arithmetic/selection reference.
TEST(ScalarBinaryGfx1250, ArithmeticAndPolicy) {
  using Call = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint32_t, uint32_t *);
  const Call calls[2][7] = {
      {rj_arith_s_add_f32_gfx1250, rj_arith_s_sub_f32_gfx1250, rj_arith_s_mul_f32_gfx1250,
       rj_arith_s_min_num_f32_gfx1250, rj_arith_s_max_num_f32_gfx1250,
       rj_arith_s_minimum_f32_gfx1250, rj_arith_s_maximum_f32_gfx1250},
      {rj_arith_s_add_f16_gfx1250, rj_arith_s_sub_f16_gfx1250, rj_arith_s_mul_f16_gfx1250,
       rj_arith_s_min_num_f16_gfx1250, rj_arith_s_max_num_f16_gfx1250,
       rj_arith_s_minimum_f16_gfx1250, rj_arith_s_maximum_f16_gfx1250},
  };
  using namespace rocjitsu::arith::test;
  for (int half = 0; half < 2; ++half) {
    const uint32_t sign = half ? 0x8000U : 0x80000000U;
    const uint32_t inf = half ? 0x7c00U : 0x7f800000U;
    const uint32_t quiet = half ? 0x200U : 0x400000U;
    for (int op = 0; op < 7; ++op) {
      for (auto semantics : semantics_modes()) {
        std::minstd_rand random;
        for (int sample = 0; sample < 128; ++sample) {
          uint32_t a = half ? random_f16_bits(random) : random_f32_bits(random);
          uint32_t b = half ? random_f16_bits(random) : random_f32_bits(random);
          if (sample < 100) {
            a = half ? compare_class_value<16>(sample % 10) : compare_class_value<32>(sample % 10);
            b = half ? compare_class_value<16>(sample / 10) : compare_class_value<32>(sample / 10);
          }
          if (sample == 100) {
            a = b = inf - 1;
          } // Finite overflow despite FP16_OVFL in Compatibility.
          const bool compatibility = semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY;
          const bool flush = sample % 2 == 0;
          const bool effective_flush = flush && (op >= 3 || !compatibility);
          uint64_t flags = semantics | (flush ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
          if (compatibility) {
            flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP16_OVFL;
          }
          const uint32_t raw_a = a, raw_b = b;
          if (effective_flush) {
            if (!(a & inf)) {
              a &= sign;
            }
            if (!(b & inf)) {
              b &= sign;
            }
          }
          const float x = half ? float(scalar_fp_value<true>(a, false)) : std::bit_cast<float>(a);
          const float y = half ? float(scalar_fp_value<true>(b, false)) : std::bit_cast<float>(b);
          uint32_t expected;
          if (op < 3) {
            const float value = op == 0 ? x + y : op == 1 ? x - y : x * y;
            expected = half ? scalar_fp_bits<true>(value) : std::bit_cast<uint32_t>(value);
          } else {
            const bool an = (a & (sign - 1)) > inf, bn = (b & (sign - 1)) > inf;
            const bool maximum = op == 4 || op == 6;
            if ((op >= 5 && (an || bn)) || (an && bn)) {
              expected = (an ? a : b) | quiet;
            } else if (an || bn) {
              expected = an ? b : a;
            } else if (x == y) {
              expected = maximum ? a & b : a | b;
            } else {
              expected = (maximum ? x > y : x < y) ? a : b;
            }
          }
          uint32_t output = raw_a, status = 0x80000040U;
          const int error = calls[half][op](flags, 0, &output, output, raw_b, &status);
          const bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
          ASSERT_EQ(error, strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
          if (!strict && op < 3 && (expected & (sign - 1)) > inf) {
            EXPECT_GT(output & (sign - 1), inf);
          } else {
            EXPECT_EQ(output, strict ? raw_a : expected);
          }
          EXPECT_EQ(status, 0x80000040U);
        }
      }
      uint32_t output = 123, status = 456;
      expect_invalid_flags(
          [&](auto flags, auto mode) {
            int error = calls[half][op](flags, mode, &output, 0, 0, &status);
            EXPECT_EQ(output, 123U);
            EXPECT_EQ(status, 456U);
            return error;
          },
          0);
    }
  }
}

// FP32 retains the legacy directed FMA model; FP16 retains two rounding stages.
TEST(ScalarFmaGfx1250, ArithmeticAndPolicy) {
  using Call = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint32_t, uint32_t, uint32_t *);
  const Call calls[] = {rj_arith_s_fmaak_f32_gfx1250, rj_arith_s_fmamk_f32_gfx1250,
                        +[](uint64_t f, uint64_t m, uint32_t *d, uint32_t a, uint32_t b, uint32_t,
                            uint32_t *s) { return rj_arith_s_fmac_f32_gfx1250(f, m, d, a, b, s); }};
  struct Case {
    uint32_t a, b, c, result[4];
  };
  const Case cases[] = {
      {0x3f800000, 0x3f800000, 0x33800000, {0x3f800000, 0x3f800001, 0x3f800000, 0x3f800000}},
      {0xbf800000, 0x3f800000, 0xb3800000, {0xbf800000, 0xbf800000, 0xbf800001, 0xbf800000}},
      {0x3f800000, 0x3f800000, 0xbf800000, {0, 0, 0x80000000, 0}},
      {0, 0x7f800000, 0x7fc12345, {0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000}},
      {0x7fc12345, 0x7fc45678, 0x7fc98765, {0x7fc12345, 0x7fc12345, 0x7fc12345, 0x7fc12345}}};
  using namespace rocjitsu::arith::test;
  for (auto call : calls) {
    for (auto semantics : semantics_modes()) {
      for (int rounding = 0; rounding < 4; ++rounding) {
        for (const auto &test : cases) {
          uint32_t d = test.c, status = 0x80000040U;
          const bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
          ASSERT_EQ(
              call(semantics | (uint64_t(rounding) << 24), 0, &d, test.a, test.b, test.c, &status),
              strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
          if (!strict && semantics == RJ_ARITH_SEMANTICS_LOOSE &&
              (test.a & 0x7fffffff) >= 0x7f800000) {
            EXPECT_GT(d & 0x7fffffffU, 0x7f800000U);
          } else if (!strict && semantics == RJ_ARITH_SEMANTICS_LOOSE && test.a == 0 &&
                     test.b == 0x7f800000) {
            EXPECT_GT(d & 0x7fffffffU, 0x7f800000U);
          } else {
            EXPECT_EQ(d, strict
                             ? test.c
                             : test.result[semantics == RJ_ARITH_SEMANTICS_LOOSE ? 0 : rounding]);
          }
          EXPECT_EQ(status, 0x80000040U);
        }
      }
    }
    for (int denorm = 0; denorm < 4; ++denorm) {
      uint32_t d = 0, status = 0;
      uint64_t flags = RJ_ARITH_SEMANTICS_COMPATIBILITY |
                       (denorm & 1 ? 0 : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) |
                       (denorm & 2 ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
      ASSERT_EQ(call(flags, 0, &d, 1, 0x3f800000, 0, &status), RJ_ARITH_SUCCESS);
      EXPECT_EQ(d, denorm == 3 ? 1U : 0U);
      EXPECT_EQ(status, 0U);
    }
    uint32_t d = 123, status = 456;
    expect_invalid_flags(
        [&](auto f, auto m) {
          int error = call(f, m, &d, 0, 0, 0, &status);
          EXPECT_EQ(d, 123U);
          EXPECT_EQ(status, 456U);
          return error;
        },
        0);
  }
  std::minstd_rand random;
  for (auto semantics : semantics_modes()) {
    for (int sample = 0; sample < 256; ++sample) {
      uint32_t a = random_f16_bits(random), b = random_f16_bits(random),
               c = random_f16_bits(random);
      if (sample == 0) {
        a = 0x7bff;
        b = 0x4000;
        c = 0;
      }
      if (sample == 1) {
        a = 1;
        b = 0x3c00;
        c = 0;
      }
      // Legacy two-stage rounding gives 0xe5a2; single FP16 rounding gives 0xe5a1.
      if (sample == 2) {
        a = 0x4782;
        b = 0xda00;
        c = 0x03e1;
      }
      float value =
          std::fma(float(scalar_fp_value<true>(a, false)), float(scalar_fp_value<true>(b, false)),
                   float(scalar_fp_value<true>(c, false)));
      uint32_t expected = scalar_fp_bits<true>(value), d = c, status = 0x80000040U;
      if (sample == 2) {
        EXPECT_EQ(expected, 0xe5a2U);
      }
      uint64_t flags = semantics;
      if (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
        flags |= RJ_ARITH_FP16_OVFL | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                 RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
      }
      bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                    semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
      ASSERT_EQ(rj_arith_s_fmac_f16_gfx1250(flags, 0, &d, a, b, &status),
                strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
      if (!strict && (expected & 0x7fff) > 0x7c00) {
        EXPECT_GT(d & 0x7fffU, 0x7c00U);
      } else {
        EXPECT_EQ(d, strict ? c : expected);
      }
      EXPECT_EQ(status, 0x80000040U);
    }
  }
  uint32_t d = 123, status = 456;
  expect_invalid_flags(
      [&](auto f, auto m) {
        int error = rj_arith_s_fmac_f16_gfx1250(f, m, &d, 0, 0, &status);
        EXPECT_EQ(d, 123U);
        EXPECT_EQ(status, 456U);
        return error;
      },
      0);
}
