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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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
      true, 0, false);
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

// gfx1201 captures: special, denormal, overflowing and inexact inputs in default MODE
// 0xf0 and in flushing, directed and FP16_OVFL MODEs; inputs whose result changes with
// the rounding direction in all four directions; and one input under MODE 0xf3/0xfc,
// whose F32 and F16 rounding fields differ. c is the FMAC accumulator or the
// FMAAK/FMAMK literal.
const rocjitsu::arith::test::ScalarCapture s_add_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x7f800001, 0x0, 0x0, 0x7fc00001},
    {0xf0, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf0, 0x3f800000, 0x800000, 0x0, 0x3f800000},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7f800001, 0x0, 0x0, 0x7fc00001},
    {0x0, 0x3f800000, 0x800000, 0x0, 0x3f800000},
    {0x0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0x0, 0x80eba725, 0x11a17a4, 0x0, 0x0},
    {0x10, 0x1, 0x0, 0x0, 0x0},
    {0x10, 0x800000, 0x1, 0x0, 0x800001},
    {0x10, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0x15, 0x7f7fffff, 0x1, 0x0, 0x7f800000},
    {0xf0, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf5, 0x3f800000, 0x1, 0x0, 0x3f800001},
    {0xfa, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xff, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf0, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xf5, 0xbf800000, 0x1, 0x0, 0xbf7fffff},
    {0xfa, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xff, 0xbf800000, 0x1, 0x0, 0xbf7fffff},
    {0xf0, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xf5, 0x3f000000, 0x1, 0x0, 0x3f000001},
    {0xfa, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xff, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xf3, 0xbf800000, 0x1, 0x0, 0xbf7fffff},
    {0xfc, 0xbf800000, 0x1, 0x0, 0xbf800000},
};

TEST(s_add_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_add_f32(f, 0, d, a, b, status);
      },
      s_add_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_sub_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x7f800001, 0x0, 0x0, 0x7fc00001},
    {0xf0, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf0, 0x3f800000, 0x800000, 0x0, 0x3f800000},
    {0xf0, 0xff7fffff, 0x7f7fffff, 0x0, 0xff800000},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7f800001, 0x0, 0x0, 0x7fc00001},
    {0x0, 0x3f800000, 0x800000, 0x0, 0x3f800000},
    {0x0, 0xff7fffff, 0x7f7fffff, 0x0, 0xff800000},
    {0x0, 0x80d8feb1, 0x811cab1c, 0x0, 0x0},
    {0x10, 0x1, 0x0, 0x0, 0x0},
    {0x10, 0x1, 0x1, 0x0, 0x0},
    {0x10, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0x15, 0x7f7fffff, 0x807fffff, 0x0, 0x7f800000},
    {0xf0, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf5, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xfa, 0x3f800000, 0x1, 0x0, 0x3f7fffff},
    {0xff, 0x3f800000, 0x1, 0x0, 0x3f7fffff},
    {0xf0, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xf5, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xfa, 0xbf800000, 0x1, 0x0, 0xbf800001},
    {0xff, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xf0, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xf5, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xfa, 0x3f000000, 0x1, 0x0, 0x3effffff},
    {0xff, 0x3f000000, 0x1, 0x0, 0x3effffff},
    {0xf3, 0x3f800000, 0x1, 0x0, 0x3f7fffff},
    {0xfc, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf0, 0x0, 0x7fc00000, 0x0, 0xffc00000},
};

TEST(s_sub_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_sub_f32(f, 0, d, a, b, status);
      },
      s_sub_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_mul_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x7f800000, 0x0, 0x0, 0xffc00000},
    {0xf0, 0x1, 0x1, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf0, 0x800000, 0x800000, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0xf0, 0x3effffff, 0x7f7fffff, 0x0, 0x7efffffe},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7f800000, 0x0, 0x0, 0xffc00000},
    {0x0, 0x800000, 0x800000, 0x0, 0x0},
    {0x0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0x0, 0x3effffff, 0x7f7fffff, 0x0, 0x7efffffe},
    {0x10, 0x1, 0x0, 0x0, 0x0},
    {0x10, 0x1, 0x1, 0x0, 0x0},
    {0x10, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf0, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xf5, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xfa, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xff, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xf0, 0xbfc00000, 0x1, 0x0, 0x80000002},
    {0xf5, 0xbfc00000, 0x1, 0x0, 0x80000001},
    {0xfa, 0xbfc00000, 0x1, 0x0, 0x80000002},
    {0xff, 0xbfc00000, 0x1, 0x0, 0x80000001},
    {0xf0, 0x40490fdb, 0x1, 0x0, 0x3},
    {0xf5, 0x40490fdb, 0x1, 0x0, 0x4},
    {0xfa, 0x40490fdb, 0x1, 0x0, 0x3},
    {0xff, 0x40490fdb, 0x1, 0x0, 0x3},
    {0xf3, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xfc, 0x3fc00000, 0x1, 0x0, 0x2},
};

TEST(s_mul_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_mul_f32(f, 0, d, a, b, status);
      },
      s_mul_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_add_f16_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},         {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x7c01, 0x0, 0x0, 0x7e01},   {0xf0, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xf0, 0x3c00, 0x400, 0x0, 0x3c00}, {0xf0, 0x7bff, 0x7bff, 0x0, 0x7c00},
    {0x0, 0x0, 0x0, 0x0, 0x0},          {0x0, 0x7c01, 0x0, 0x0, 0x7e01},
    {0x0, 0x3c00, 0x400, 0x0, 0x3c00},  {0x0, 0x7bff, 0x7bff, 0x0, 0x7c00},
    {0x0, 0x47b, 0x8751, 0x0, 0x8000},  {0x40, 0x1, 0x0, 0x0, 0x0},
    {0x40, 0x400, 0x1, 0x0, 0x401},     {0x40, 0x3c00, 0x1, 0x0, 0x3c00},
    {0x45, 0x7bff, 0x1, 0x0, 0x7c00},   {0xf0, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xf5, 0x3c00, 0x1, 0x0, 0x3c01},   {0xfa, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xff, 0x3c00, 0x1, 0x0, 0x3c00},   {0xf0, 0xbc00, 0x1, 0x0, 0xbc00},
    {0xf5, 0xbc00, 0x1, 0x0, 0xbbff},   {0xfa, 0xbc00, 0x1, 0x0, 0xbc00},
    {0xff, 0xbc00, 0x1, 0x0, 0xbbff},   {0xf0, 0x3800, 0x1, 0x0, 0x3800},
    {0xf5, 0x3800, 0x1, 0x0, 0x3801},   {0xfa, 0x3800, 0x1, 0x0, 0x3800},
    {0xff, 0x3800, 0x1, 0x0, 0x3800},   {0xf3, 0xbc00, 0x1, 0x0, 0xbc00},
    {0xfc, 0xbc00, 0x1, 0x0, 0xbbff},
};

TEST(s_add_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_add_f16(f, 0, d, a, b, status);
      },
      s_add_f16_captures, 16, 16);
}

const rocjitsu::arith::test::ScalarCapture s_sub_f16_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},          {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x7c01, 0x0, 0x0, 0x7e01},    {0xf0, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xf0, 0x3c00, 0x400, 0x0, 0x3c00},  {0xf0, 0xfbff, 0x7bff, 0x0, 0xfc00},
    {0x0, 0x0, 0x0, 0x0, 0x0},           {0x0, 0x7c01, 0x0, 0x0, 0x7e01},
    {0x0, 0x3c00, 0x400, 0x0, 0x3c00},   {0x0, 0xfbff, 0x7bff, 0x0, 0xfc00},
    {0x0, 0x868b, 0x85d8, 0x0, 0x8000},  {0x40, 0x1, 0x0, 0x0, 0x0},
    {0x40, 0x1, 0x1, 0x0, 0x0},          {0x40, 0x3c00, 0x1, 0x0, 0x3c00},
    {0x45, 0x7bff, 0x83ff, 0x0, 0x7c00}, {0xf0, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xf5, 0x3c00, 0x1, 0x0, 0x3c00},    {0xfa, 0x3c00, 0x1, 0x0, 0x3bff},
    {0xff, 0x3c00, 0x1, 0x0, 0x3bff},    {0xf0, 0xbc00, 0x1, 0x0, 0xbc00},
    {0xf5, 0xbc00, 0x1, 0x0, 0xbc00},    {0xfa, 0xbc00, 0x1, 0x0, 0xbc01},
    {0xff, 0xbc00, 0x1, 0x0, 0xbc00},    {0xf0, 0x3800, 0x1, 0x0, 0x3800},
    {0xf5, 0x3800, 0x1, 0x0, 0x3800},    {0xfa, 0x3800, 0x1, 0x0, 0x37ff},
    {0xff, 0x3800, 0x1, 0x0, 0x37ff},    {0xf3, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xfc, 0x3c00, 0x1, 0x0, 0x3bff},    {0xf0, 0x0, 0x7e00, 0x0, 0xfe00},
};

TEST(s_sub_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_sub_f16(f, 0, d, a, b, status);
      },
      s_sub_f16_captures, 16, 16);
}

const rocjitsu::arith::test::ScalarCapture s_mul_f16_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},          {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x7c00, 0x0, 0x0, 0xfe00},    {0xf0, 0x1, 0x1, 0x0, 0x0},
    {0xf0, 0x7bff, 0x83ff, 0x0, 0xc3fd}, {0xf0, 0x400, 0x400, 0x0, 0x0},
    {0xf0, 0x7bff, 0x7bff, 0x0, 0x7c00}, {0xf0, 0x3bff, 0x7bff, 0x0, 0x7bfe},
    {0x0, 0x0, 0x0, 0x0, 0x0},           {0x0, 0x7c00, 0x0, 0x0, 0xfe00},
    {0x0, 0x400, 0x400, 0x0, 0x0},       {0x0, 0x7bff, 0x7bff, 0x0, 0x7c00},
    {0x0, 0x3bff, 0x7bff, 0x0, 0x7bfe},  {0x40, 0x1, 0x0, 0x0, 0x0},
    {0x40, 0x1, 0x1, 0x0, 0x0},          {0x40, 0x7bff, 0x83ff, 0x0, 0xc3fd},
    {0xf0, 0x3e00, 0x1, 0x0, 0x2},       {0xf5, 0x3e00, 0x1, 0x0, 0x2},
    {0xfa, 0x3e00, 0x1, 0x0, 0x1},       {0xff, 0x3e00, 0x1, 0x0, 0x1},
    {0xf0, 0xbe00, 0x1, 0x0, 0x8002},    {0xf5, 0xbe00, 0x1, 0x0, 0x8001},
    {0xfa, 0xbe00, 0x1, 0x0, 0x8002},    {0xff, 0xbe00, 0x1, 0x0, 0x8001},
    {0xf0, 0x4248, 0x1, 0x0, 0x3},       {0xf5, 0x4248, 0x1, 0x0, 0x4},
    {0xfa, 0x4248, 0x1, 0x0, 0x3},       {0xff, 0x4248, 0x1, 0x0, 0x3},
    {0xf3, 0x3e00, 0x1, 0x0, 0x2},       {0xfc, 0x3e00, 0x1, 0x0, 0x1},
};

TEST(s_mul_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_mul_f16(f, 0, d, a, b, status);
      },
      s_mul_f16_captures, 16, 16);
}

const rocjitsu::arith::test::ScalarCapture s_fmac_f16_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},          {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x7c00, 0x0, 0x0, 0xfe00},    {0xf0, 0x1, 0x1, 0x0, 0x0},
    {0xf0, 0x7bff, 0x83ff, 0x0, 0xc3fd}, {0xf0, 0x400, 0x400, 0x0, 0x0},
    {0xf0, 0x7bff, 0x7bff, 0x0, 0x7c00}, {0xf0, 0x7bff, 0x7bff, 0x1, 0x7c00},
    {0xf0, 0x400, 0x400, 0x400, 0x400},  {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7c00, 0x0, 0x0, 0xfe00},     {0x0, 0x400, 0x400, 0x0, 0x0},
    {0x0, 0x7bff, 0x7bff, 0x0, 0x7c00},  {0x0, 0x400, 0x400, 0x400, 0x400},
    {0x40, 0x1, 0x0, 0x0, 0x0},          {0x40, 0x1, 0x1, 0x0, 0x0},
    {0x40, 0x7bff, 0x83ff, 0x0, 0xc3fd}, {0x40, 0x7bff, 0x7bff, 0x1, 0x7c00},
    {0xf0, 0x3e00, 0x1, 0x0, 0x2},       {0xf5, 0x3e00, 0x1, 0x0, 0x2},
    {0xfa, 0x3e00, 0x1, 0x0, 0x1},       {0xff, 0x3e00, 0x1, 0x0, 0x1},
    {0xf0, 0x3800, 0x83ff, 0x0, 0x8200}, {0xf5, 0x3800, 0x83ff, 0x0, 0x81ff},
    {0xfa, 0x3800, 0x83ff, 0x0, 0x8200}, {0xff, 0x3800, 0x83ff, 0x0, 0x81ff},
    {0xf0, 0x7bff, 0x83ff, 0x0, 0xc3fd}, {0xf5, 0x7bff, 0x83ff, 0x0, 0xc3fd},
    {0xfa, 0x7bff, 0x83ff, 0x0, 0xc3fe}, {0xff, 0x7bff, 0x83ff, 0x0, 0xc3fd},
    {0xf3, 0x3e00, 0x1, 0x0, 0x2},       {0xfc, 0x3e00, 0x1, 0x0, 0x1},
    {0xf0, 0x7c00, 0x0, 0x7e00, 0xfe00},
};

TEST(s_fmac_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_fmac_f16(f, 0, d, a, b, status);
      },
      s_fmac_f16_captures, 16, 16);
}

const rocjitsu::arith::test::ScalarCapture s_fmac_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x7f800000, 0x0, 0x0, 0xffc00000},
    {0xf0, 0x1, 0x1, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf0, 0x800000, 0x800000, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x1, 0x7f800000},
    {0xf0, 0x800000, 0x800000, 0x800000, 0x800000},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7f800000, 0x0, 0x0, 0xffc00000},
    {0x0, 0x800000, 0x800000, 0x0, 0x0},
    {0x0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0x0, 0x800000, 0x800000, 0x800000, 0x800000},
    {0x10, 0x1, 0x0, 0x0, 0x0},
    {0x10, 0x1, 0x1, 0x0, 0x0},
    {0x10, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0x10, 0x7f7fffff, 0x7f7fffff, 0x1, 0x7f800000},
    {0xf0, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xf5, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xfa, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xff, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xf0, 0x3f000000, 0x807fffff, 0x0, 0x80400000},
    {0xf5, 0x3f000000, 0x807fffff, 0x0, 0x803fffff},
    {0xfa, 0x3f000000, 0x807fffff, 0x0, 0x80400000},
    {0xff, 0x3f000000, 0x807fffff, 0x0, 0x803fffff},
    {0xf0, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf5, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xfa, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffe},
    {0xff, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf3, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xfc, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xf0, 0x7f800000, 0x0, 0x7fc00000, 0xffc00000},
};

TEST(s_fmac_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_fmac_f32(f, 0, d, a, b, status);
      },
      s_fmac_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_fmaak_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x1, 0x0, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x7f800000, 0x0, 0x40490fdb, 0xffc00000},
    {0xf0, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x800000, 0x800000, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x40490fdb, 0x7f800000},
    {0x0, 0x0, 0x0, 0x40490fdb, 0x40490fdb},
    {0x0, 0x7f800000, 0x0, 0x40490fdb, 0xffc00000},
    {0x0, 0x800000, 0x800000, 0x40490fdb, 0x40490fdb},
    {0x0, 0x7f7fffff, 0x7f7fffff, 0x40490fdb, 0x7f800000},
    {0x10, 0x1, 0x0, 0x40490fdb, 0x40490fdb},
    {0x10, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf5, 0x1, 0x1, 0x40490fdb, 0x40490fdc},
    {0xfa, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xff, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x807fffff, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf5, 0x807fffff, 0x1, 0x40490fdb, 0x40490fdb},
    {0xfa, 0x807fffff, 0x1, 0x40490fdb, 0x40490fda},
    {0xff, 0x807fffff, 0x1, 0x40490fdb, 0x40490fda},
    {0xf0, 0x800000, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf5, 0x800000, 0x1, 0x40490fdb, 0x40490fdc},
    {0xfa, 0x800000, 0x1, 0x40490fdb, 0x40490fdb},
    {0xff, 0x800000, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf3, 0x807fffff, 0x1, 0x40490fdb, 0x40490fda},
    {0xfc, 0x807fffff, 0x1, 0x40490fdb, 0x40490fdb},
};

TEST(s_fmaak_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto c, auto status) {
        return rj_arith_s_fmaak_f32(f, 0, d, a, b, c, status);
      },
      s_fmaak_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_fmamk_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x40490fdb, 0x0},
    {0xf0, 0x1, 0x0, 0x40490fdb, 0x3},
    {0xf0, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0xf0, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
    {0xf0, 0x7f800001, 0x0, 0x40490fdb, 0x7fc00001},
    {0xf0, 0x3fc00000, 0x0, 0x40490fdb, 0x4096cbe4},
    {0xf0, 0x400000, 0x0, 0x40490fdb, 0xc90fdb},
    {0xf0, 0x7f7fffff, 0x1, 0x40490fdb, 0x7f800000},
    {0xf0, 0x9e81b6, 0x81508a70, 0x40490fdb, 0x50e1c4},
    {0x0, 0x0, 0x0, 0x40490fdb, 0x0},
    {0x0, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
    {0x0, 0x7f800001, 0x0, 0x40490fdb, 0x7fc00001},
    {0x0, 0x3fc00000, 0x0, 0x40490fdb, 0x4096cbe4},
    {0x0, 0x9e81b6, 0x81508a70, 0x40490fdb, 0x0},
    {0x10, 0x1, 0x0, 0x40490fdb, 0x0},
    {0x10, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0x10, 0x400000, 0x0, 0x40490fdb, 0xc90fdb},
    {0x10, 0x7f7fffff, 0x1, 0x40490fdb, 0x7f800000},
    {0xf0, 0x1, 0x0, 0x40490fdb, 0x3},
    {0xf5, 0x1, 0x0, 0x40490fdb, 0x4},
    {0xfa, 0x1, 0x0, 0x40490fdb, 0x3},
    {0xff, 0x1, 0x0, 0x40490fdb, 0x3},
    {0xf0, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0xf5, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0xfa, 0x807fffff, 0x0, 0x40490fdb, 0x81490fda},
    {0xff, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0xf0, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
    {0xf5, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
    {0xfa, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f7fffff},
    {0xff, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f7fffff},
    {0xf3, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f7fffff},
    {0xfc, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
};

TEST(s_fmamk_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto c, auto status) {
        return rj_arith_s_fmamk_f32(f, 0, d, a, c, b, status);
      },
      s_fmamk_f32_captures, 32, 32);
}

// gfx1201 captures by operand class (NaNs in either order, signed zeros, denormals,
// ordered and out-of-range values) in default MODE 0xf0 and in MODE 0x00, where
// inputs and outputs are flushed.
const rocjitsu::arith::test::ScalarCapture s_min_num_f32_captures[] = {
    {0xf0, 0x7f800001, 0x7fc00000, 0x0, 0x7fc00001},
    {0xf0, 0x7fc00000, 0x7f800001, 0x0, 0x7fc00000},
    {0xf0, 0x7fc00000, 0x800000, 0x0, 0x800000},
    {0xf0, 0x800000, 0x7f800001, 0x0, 0x800000},
    {0xf0, 0x0, 0x80000000, 0x0, 0x80000000},
    {0xf0, 0x80000000, 0x0, 0x0, 0x80000000},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x1, 0x0, 0x1},
    {0xf0, 0x1, 0x800000, 0x0, 0x1},
    {0xf0, 0x800000, 0x800000, 0x0, 0x800000},
    {0xf0, 0x7f800000, 0x800000, 0x0, 0x800000},
    {0xf0, 0x7f7fffff, 0x800000, 0x0, 0x800000},
    {0x0, 0x7f800001, 0x7fc00000, 0x0, 0x7fc00001},
    {0x0, 0x7fc00000, 0x7f800001, 0x0, 0x7fc00000},
    {0x0, 0x7fc00000, 0x800000, 0x0, 0x800000},
    {0x0, 0x800000, 0x7f800001, 0x0, 0x800000},
    {0x0, 0x0, 0x80000000, 0x0, 0x80000000},
    {0x0, 0x80000000, 0x0, 0x0, 0x80000000},
    {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x1, 0x0, 0x0},
    {0x0, 0x1, 0x800000, 0x0, 0x0},
    {0x0, 0x800000, 0x800000, 0x0, 0x800000},
    {0x0, 0x7f800000, 0x800000, 0x0, 0x800000},
    {0x0, 0x7f7fffff, 0x800000, 0x0, 0x800000},
};

TEST(s_min_num_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_min_num_f32(f, 0, d, a, b, status);
      },
      s_min_num_f32_captures, 32, 32, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_max_num_f32_captures[] = {
    {0xf0, 0x7f800001, 0x7fc00000, 0x0, 0x7fc00001},
    {0xf0, 0x7fc00000, 0x7f800001, 0x0, 0x7fc00000},
    {0xf0, 0x7fc00000, 0x800000, 0x0, 0x800000},
    {0xf0, 0x800000, 0x7f800001, 0x0, 0x800000},
    {0xf0, 0x0, 0x80000000, 0x0, 0x0},
    {0xf0, 0x80000000, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x1, 0x1, 0x0, 0x1},
    {0xf0, 0x1, 0x800000, 0x0, 0x800000},
    {0xf0, 0x800000, 0x800000, 0x0, 0x800000},
    {0xf0, 0x7f800000, 0x800000, 0x0, 0x7f800000},
    {0xf0, 0x7f7fffff, 0x800000, 0x0, 0x7f7fffff},
    {0x0, 0x7f800001, 0x7fc00000, 0x0, 0x7fc00001},
    {0x0, 0x7fc00000, 0x7f800001, 0x0, 0x7fc00000},
    {0x0, 0x7fc00000, 0x800000, 0x0, 0x800000},
    {0x0, 0x800000, 0x7f800001, 0x0, 0x800000},
    {0x0, 0x0, 0x80000000, 0x0, 0x0},
    {0x0, 0x80000000, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x1, 0x0, 0x0},
    {0x0, 0x1, 0x800000, 0x0, 0x800000},
    {0x0, 0x800000, 0x800000, 0x0, 0x800000},
    {0x0, 0x7f800000, 0x800000, 0x0, 0x7f800000},
    {0x0, 0x7f7fffff, 0x800000, 0x0, 0x7f7fffff},
};

TEST(s_max_num_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_max_num_f32(f, 0, d, a, b, status);
      },
      s_max_num_f32_captures, 32, 32, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_minimum_f32_captures[] = {
    {0xf0, 0x7f800001, 0x7fc00000, 0x0, 0x7fc00001},
    {0xf0, 0x7fc00000, 0x7f800001, 0x0, 0x7fc00000},
    {0xf0, 0x7fc00000, 0x800000, 0x0, 0x7fc00000},
    {0xf0, 0x800000, 0x7f800001, 0x0, 0x7fc00001},
    {0xf0, 0x0, 0x80000000, 0x0, 0x80000000},
    {0xf0, 0x80000000, 0x0, 0x0, 0x80000000},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x1, 0x0, 0x1},
    {0xf0, 0x1, 0x800000, 0x0, 0x1},
    {0xf0, 0x800000, 0x800000, 0x0, 0x800000},
    {0xf0, 0x7f800000, 0x800000, 0x0, 0x800000},
    {0xf0, 0x7f7fffff, 0x800000, 0x0, 0x800000},
    {0x0, 0x7f800001, 0x7fc00000, 0x0, 0x7fc00001},
    {0x0, 0x7fc00000, 0x7f800001, 0x0, 0x7fc00000},
    {0x0, 0x7fc00000, 0x800000, 0x0, 0x7fc00000},
    {0x0, 0x800000, 0x7f800001, 0x0, 0x7fc00001},
    {0x0, 0x0, 0x80000000, 0x0, 0x80000000},
    {0x0, 0x80000000, 0x0, 0x0, 0x80000000},
    {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x1, 0x0, 0x0},
    {0x0, 0x1, 0x800000, 0x0, 0x0},
    {0x0, 0x800000, 0x800000, 0x0, 0x800000},
    {0x0, 0x7f800000, 0x800000, 0x0, 0x800000},
    {0x0, 0x7f7fffff, 0x800000, 0x0, 0x800000},
};

TEST(s_minimum_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_minimum_f32(f, 0, d, a, b, status);
      },
      s_minimum_f32_captures, 32, 32, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_maximum_f32_captures[] = {
    {0xf0, 0x7f800001, 0x7fc00000, 0x0, 0x7fc00001},
    {0xf0, 0x7fc00000, 0x7f800001, 0x0, 0x7fc00000},
    {0xf0, 0x7fc00000, 0x800000, 0x0, 0x7fc00000},
    {0xf0, 0x800000, 0x7f800001, 0x0, 0x7fc00001},
    {0xf0, 0x0, 0x80000000, 0x0, 0x0},
    {0xf0, 0x80000000, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x1, 0x1, 0x0, 0x1},
    {0xf0, 0x1, 0x800000, 0x0, 0x800000},
    {0xf0, 0x800000, 0x800000, 0x0, 0x800000},
    {0xf0, 0x7f800000, 0x800000, 0x0, 0x7f800000},
    {0xf0, 0x7f7fffff, 0x800000, 0x0, 0x7f7fffff},
    {0x0, 0x7f800001, 0x7fc00000, 0x0, 0x7fc00001},
    {0x0, 0x7fc00000, 0x7f800001, 0x0, 0x7fc00000},
    {0x0, 0x7fc00000, 0x800000, 0x0, 0x7fc00000},
    {0x0, 0x800000, 0x7f800001, 0x0, 0x7fc00001},
    {0x0, 0x0, 0x80000000, 0x0, 0x0},
    {0x0, 0x80000000, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x1, 0x0, 0x0},
    {0x0, 0x1, 0x800000, 0x0, 0x800000},
    {0x0, 0x800000, 0x800000, 0x0, 0x800000},
    {0x0, 0x7f800000, 0x800000, 0x0, 0x7f800000},
    {0x0, 0x7f7fffff, 0x800000, 0x0, 0x7f7fffff},
};

TEST(s_maximum_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_maximum_f32(f, 0, d, a, b, status);
      },
      s_maximum_f32_captures, 32, 32, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_min_num_f16_captures[] = {
    {0xf0, 0x7c01, 0x7e00, 0x0, 0x7e01}, {0xf0, 0x7e00, 0x7c01, 0x0, 0x7e00},
    {0xf0, 0x7e00, 0x400, 0x0, 0x400},   {0xf0, 0x400, 0x7c01, 0x0, 0x400},
    {0xf0, 0x0, 0x8000, 0x0, 0x8000},    {0xf0, 0x8000, 0x0, 0x0, 0x8000},
    {0xf0, 0x1, 0x0, 0x0, 0x0},          {0xf0, 0x1, 0x1, 0x0, 0x1},
    {0xf0, 0x1, 0x400, 0x0, 0x1},        {0xf0, 0x400, 0x400, 0x0, 0x400},
    {0xf0, 0x7c00, 0x400, 0x0, 0x400},   {0x0, 0x7c01, 0x7e00, 0x0, 0x7e01},
    {0x0, 0x7e00, 0x7c01, 0x0, 0x7e00},  {0x0, 0x7e00, 0x400, 0x0, 0x400},
    {0x0, 0x400, 0x7c01, 0x0, 0x400},    {0x0, 0x0, 0x8000, 0x0, 0x8000},
    {0x0, 0x8000, 0x0, 0x0, 0x8000},     {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x1, 0x0, 0x0},           {0x0, 0x1, 0x400, 0x0, 0x0},
    {0x0, 0x400, 0x400, 0x0, 0x400},     {0x0, 0x7c00, 0x400, 0x0, 0x400},
};

TEST(s_min_num_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_min_num_f16(f, 0, d, a, b, status);
      },
      s_min_num_f16_captures, 16, 16, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_max_num_f16_captures[] = {
    {0xf0, 0x7c01, 0x7e00, 0x0, 0x7e01}, {0xf0, 0x7e00, 0x7c01, 0x0, 0x7e00},
    {0xf0, 0x7e00, 0x400, 0x0, 0x400},   {0xf0, 0x400, 0x7c01, 0x0, 0x400},
    {0xf0, 0x0, 0x8000, 0x0, 0x0},       {0xf0, 0x8000, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x1},          {0xf0, 0x1, 0x1, 0x0, 0x1},
    {0xf0, 0x1, 0x400, 0x0, 0x400},      {0xf0, 0x400, 0x400, 0x0, 0x400},
    {0xf0, 0x7c00, 0x400, 0x0, 0x7c00},  {0x0, 0x7c01, 0x7e00, 0x0, 0x7e01},
    {0x0, 0x7e00, 0x7c01, 0x0, 0x7e00},  {0x0, 0x7e00, 0x400, 0x0, 0x400},
    {0x0, 0x400, 0x7c01, 0x0, 0x400},    {0x0, 0x0, 0x8000, 0x0, 0x0},
    {0x0, 0x8000, 0x0, 0x0, 0x0},        {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x1, 0x0, 0x0},           {0x0, 0x1, 0x400, 0x0, 0x400},
    {0x0, 0x400, 0x400, 0x0, 0x400},     {0x0, 0x7c00, 0x400, 0x0, 0x7c00},
};

TEST(s_max_num_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_max_num_f16(f, 0, d, a, b, status);
      },
      s_max_num_f16_captures, 16, 16, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_minimum_f16_captures[] = {
    {0xf0, 0x7c01, 0x7e00, 0x0, 0x7e01}, {0xf0, 0x7e00, 0x7c01, 0x0, 0x7e00},
    {0xf0, 0x7e00, 0x400, 0x0, 0x7e00},  {0xf0, 0x400, 0x7c01, 0x0, 0x7e01},
    {0xf0, 0x0, 0x8000, 0x0, 0x8000},    {0xf0, 0x8000, 0x0, 0x0, 0x8000},
    {0xf0, 0x1, 0x0, 0x0, 0x0},          {0xf0, 0x1, 0x1, 0x0, 0x1},
    {0xf0, 0x1, 0x400, 0x0, 0x1},        {0xf0, 0x400, 0x400, 0x0, 0x400},
    {0xf0, 0x7c00, 0x400, 0x0, 0x400},   {0x0, 0x7c01, 0x7e00, 0x0, 0x7e01},
    {0x0, 0x7e00, 0x7c01, 0x0, 0x7e00},  {0x0, 0x7e00, 0x400, 0x0, 0x7e00},
    {0x0, 0x400, 0x7c01, 0x0, 0x7e01},   {0x0, 0x0, 0x8000, 0x0, 0x8000},
    {0x0, 0x8000, 0x0, 0x0, 0x8000},     {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x1, 0x0, 0x0},           {0x0, 0x1, 0x400, 0x0, 0x0},
    {0x0, 0x400, 0x400, 0x0, 0x400},     {0x0, 0x7c00, 0x400, 0x0, 0x400},
};

TEST(s_minimum_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_minimum_f16(f, 0, d, a, b, status);
      },
      s_minimum_f16_captures, 16, 16, /*rounded=*/false);
}

const rocjitsu::arith::test::ScalarCapture s_maximum_f16_captures[] = {
    {0xf0, 0x7c01, 0x7e00, 0x0, 0x7e01}, {0xf0, 0x7e00, 0x7c01, 0x0, 0x7e00},
    {0xf0, 0x7e00, 0x400, 0x0, 0x7e00},  {0xf0, 0x400, 0x7c01, 0x0, 0x7e01},
    {0xf0, 0x0, 0x8000, 0x0, 0x0},       {0xf0, 0x8000, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x1},          {0xf0, 0x1, 0x1, 0x0, 0x1},
    {0xf0, 0x1, 0x400, 0x0, 0x400},      {0xf0, 0x400, 0x400, 0x0, 0x400},
    {0xf0, 0x7c00, 0x400, 0x0, 0x7c00},  {0x0, 0x7c01, 0x7e00, 0x0, 0x7e01},
    {0x0, 0x7e00, 0x7c01, 0x0, 0x7e00},  {0x0, 0x7e00, 0x400, 0x0, 0x7e00},
    {0x0, 0x400, 0x7c01, 0x0, 0x7e01},   {0x0, 0x0, 0x8000, 0x0, 0x0},
    {0x0, 0x8000, 0x0, 0x0, 0x0},        {0x0, 0x1, 0x0, 0x0, 0x0},
    {0x0, 0x1, 0x1, 0x0, 0x0},           {0x0, 0x1, 0x400, 0x0, 0x400},
    {0x0, 0x400, 0x400, 0x0, 0x400},     {0x0, 0x7c00, 0x400, 0x0, 0x7c00},
};

TEST(s_maximum_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_maximum_f16(f, 0, d, a, b, status);
      },
      s_maximum_f16_captures, 16, 16, /*rounded=*/false);
}
