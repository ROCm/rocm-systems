// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar floating-point rounding.

#include "scalar/fp_helpers.h"
#include <algorithm>
#include <limits>

// Check scalar FP results and the expected semantics/status policy.
TEST(s_ceil_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_ceil_f32(f, m, d, a, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::ceil(x));
      },
      true);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_floor_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_floor_f32(f, m, d, a, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::floor(x));
      },
      true);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_trunc_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_trunc_f32(f, m, d, a, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::trunc(x));
      },
      true);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_rndne_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_rndne_f32(f, m, d, a, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::copysign(
            std::floor(std::abs(x)) + (std::abs(x) - std::floor(std::abs(x)) > 0.5 ||
                                       (std::abs(x) - std::floor(std::abs(x)) == 0.5 &&
                                        std::fmod(std::floor(std::abs(x)), 2) != 0)),
            x));
      },
      true);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_ceil_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_ceil_f16(f, m, d, a, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::ceil(x));
      },
      true);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_floor_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_floor_f16(f, m, d, a, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::floor(x));
      },
      true);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_trunc_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_trunc_f16(f, m, d, a, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::trunc(x));
      },
      true);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_rndne_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_rndne_f16(f, m, d, a, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::copysign(
            std::floor(std::abs(x)) + (std::abs(x) - std::floor(std::abs(x)) > 0.5 ||
                                       (std::abs(x) - std::floor(std::abs(x)) == 0.5 &&
                                        std::fmod(std::floor(std::abs(x)), 2) != 0)),
            x));
      },
      true);
}
