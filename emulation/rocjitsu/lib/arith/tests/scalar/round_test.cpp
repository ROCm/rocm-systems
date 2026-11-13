// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar floating-point rounding.

#include "rj_arith/detail/rj_arith_scalar.h"
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

// Computed integral references match the inherited model; NaN bits are explicit.
TEST(ScalarRoundGfx1250, ArithmeticAndPolicy) {
  using Call = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint32_t *);
  const Call calls[2][4] = {{rj_arith_s_ceil_f32_gfx1250, rj_arith_s_floor_f32_gfx1250,
                             rj_arith_s_trunc_f32_gfx1250, rj_arith_s_rndne_f32_gfx1250},
                            {rj_arith_s_ceil_f16_gfx1250, rj_arith_s_floor_f16_gfx1250,
                             rj_arith_s_trunc_f16_gfx1250, rj_arith_s_rndne_f16_gfx1250}};
  using namespace rocjitsu::arith::test;
  for (int half = 0; half < 2; ++half) {
    const uint32_t sign = half ? 0x8000U : 0x80000000U;
    const uint32_t inf = half ? 0x7c00U : 0x7f800000U;
    const uint32_t quiet = half ? 0x200U : 0x400000U;
    const uint32_t edges[] = {0,
                              sign,
                              1,
                              sign | 1,
                              inf,
                              sign | inf,
                              inf | 1,
                              sign | inf | 0x15U,
                              half ? 0x3800U : 0x3f000000U,
                              half ? 0xbe00U : 0xbfc00000U};
    for (int op = 0; op < 4; ++op) {
      for (auto semantics : semantics_modes()) {
        std::minstd_rand random;
        for (int sample = 0; sample < 128; ++sample) {
          uint32_t input = half ? random_f16_bits(random) : random_f32_bits(random);
          if (sample < 10) {
            input = edges[sample];
          }
          const bool flush = sample % 2 == 0;
          const bool effective_flush = flush && semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY;
          const uint32_t magnitude = input & (sign - 1);
          uint32_t expected;
          if (magnitude > inf) {
            expected = input | quiet;
          } else {
            const double x = half ? scalar_fp_value<true>(input, effective_flush)
                                  : scalar_fp_value<false>(input, effective_flush);
            const double y = op == 0   ? std::ceil(x)
                             : op == 1 ? std::floor(x)
                             : op == 2 ? std::trunc(x)
                                       : std::nearbyint(x);
            expected = half ? scalar_fp_bits<true>(y) : scalar_fp_bits<false>(y);
          }
          uint32_t output = input, status = 0x80000040U;
          const int error =
              calls[half][op](semantics | (flush ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0), 0,
                              &output, output, &status);
          const bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
          ASSERT_EQ(error, strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
          EXPECT_EQ(output, strict ? input : expected);
          EXPECT_EQ(status, 0x80000040U);
        }
      }
      uint32_t output = 123, status = 456;
      expect_invalid_flags(
          [&](auto flags, auto mode) {
            int error = calls[half][op](flags, mode, &output, 0, &status);
            EXPECT_EQ(output, 123U);
            EXPECT_EQ(status, 456U);
            return error;
          },
          0);
    }
  }
}
