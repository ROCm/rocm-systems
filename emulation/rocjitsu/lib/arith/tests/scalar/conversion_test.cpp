// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar integer/FP conversions and packed FP16 results.

#include "scalar/fp_helpers.h"
#include <algorithm>
#include <limits>

#include "conversion/conversion16_reference.h"
#include "conversion/conversion32_reference.h"
#include "conversion/packed_conversion_reference.h"

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
