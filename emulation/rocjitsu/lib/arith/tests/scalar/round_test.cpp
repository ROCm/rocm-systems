// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar floating-point rounding.

#include "scalar/fp_helpers.h"
#include <algorithm>
#include <limits>

namespace {

// Check the shared arithmetic contract, then exact NaN quieting without a host
// floating-point conversion erasing the guest payload or sign.
template <bool Half, typename Call, typename Reference>
void check_scalar_round(Call call, Reference reference) {
  using namespace rocjitsu::arith::test;
  check_scalar_fp<Half>(call, reference, true);
  constexpr uint32_t inf = Half ? 0x7c00U : 0x7f800000U;
  constexpr uint32_t quiet = Half ? 0x200U : 0x400000U;
  constexpr uint32_t sign = Half ? 0x8000U : 0x80000000U;
  // Integral rounding preserves sign/payload and sets the quiet bit.
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      if (semantics == RJ_ARITH_SEMANTICS_LOOSE) {
        continue;
      }
      for (uint32_t input : {inf | 1U, sign | inf | 0x15U, inf | quiet | 3U}) {
        uint32_t result = 0, status = 0x80000040U;
        ASSERT_EQ(call(cpu | semantics, 0, &result, input, 0, 0, &status), RJ_ARITH_SUCCESS);
        EXPECT_EQ(result, input | quiet);
        EXPECT_EQ(status, 0x80000040U | (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL &&
                                                 !(input & quiet)
                                             ? RJ_ARITH_EXCEPTION_INVALID
                                             : 0));
      }
    }
  }
}

} // namespace

// Check scalar FP results and the expected semantics/status policy.
TEST(s_ceil_f32, Arithmetic) {
  check_scalar_round<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_ceil_f32(f, m, d, a, status);
      },
      [](auto a, auto, auto, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush);
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::ceil(x));
      });
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_floor_f32, Arithmetic) {
  check_scalar_round<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_floor_f32(f, m, d, a, status);
      },
      [](auto a, auto, auto, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush);
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::floor(x));
      });
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_trunc_f32, Arithmetic) {
  check_scalar_round<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_trunc_f32(f, m, d, a, status);
      },
      [](auto a, auto, auto, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush);
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::trunc(x));
      });
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_rndne_f32, Arithmetic) {
  check_scalar_round<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_rndne_f32(f, m, d, a, status);
      },
      [](auto a, auto, auto, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush);
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::copysign(
            std::floor(std::abs(x)) + (std::abs(x) - std::floor(std::abs(x)) > 0.5 ||
                                       (std::abs(x) - std::floor(std::abs(x)) == 0.5 &&
                                        std::fmod(std::floor(std::abs(x)), 2) != 0)),
            x));
      });
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_ceil_f16, Arithmetic) {
  check_scalar_round<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_ceil_f16(f, m, d, a, status);
      },
      [](auto a, auto, auto, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush);
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::ceil(x));
      });
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_floor_f16, Arithmetic) {
  check_scalar_round<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_floor_f16(f, m, d, a, status);
      },
      [](auto a, auto, auto, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush);
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::floor(x));
      });
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_trunc_f16, Arithmetic) {
  check_scalar_round<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_trunc_f16(f, m, d, a, status);
      },
      [](auto a, auto, auto, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush);
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::trunc(x));
      });
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_rndne_f16, Arithmetic) {
  check_scalar_round<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_rndne_f16(f, m, d, a, status);
      },
      [](auto a, auto, auto, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush);
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::copysign(
            std::floor(std::abs(x)) + (std::abs(x) - std::floor(std::abs(x)) > 0.5 ||
                                       (std::abs(x) - std::floor(std::abs(x)) == 0.5 &&
                                        std::fmod(std::floor(std::abs(x)), 2) != 0)),
            x));
      });
}
