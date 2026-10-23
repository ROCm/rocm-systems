// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar floating-point arithmetic and selection.

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
