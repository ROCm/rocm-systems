// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar-result special functions without EXEC masking.

#include "math/unary_test.h"

namespace {
using namespace rocjitsu::arith::test;

// Check scalar-result special functions and semantics support for the supplied entry point.
template <bool Half, class Call> void check_pseudo(Call call, int op) {
  bool exact = !Half || (op >= 4 && op <= 6);
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      std::minstd_rand random;
      for (int sample = 0; sample < 64; ++sample) {
        for (int scenario = 0; scenario < 8; ++scenario) {
          uint32_t a =
              Half ? uint32_t(random_f16_bits(random)) | (uint32_t(random_f16_bits(random)) << 16)
                   : random_f32_bits(random);
          bool precise = exact && semantics != RJ_ARITH_SEMANTICS_LOOSE;
          if (!Half && precise) {
            a = unary_inputs[sample % 32];
          }
          if (sample == 0) {
            a = Half ? 0x7e01 : 0x7fc00001U;
          }
          if (sample == 1) {
            a = Half ? 0x7c00 : 0x7f800000U;
          }
          if (!Half && precise) {
            a = unary_inputs[sample % 32];
          }
          uint64_t mode = uint64_t(scenario & 3) << 6;
          if (scenario & 4) {
            mode |= RJ_ARITH_ALU_CLAMP;
          }
          uint32_t d = 123, status = 0x80000000U;
          int wanted = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                           ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                       : !exact && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                           ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                           : RJ_ARITH_SUCCESS;
          ASSERT_EQ(call(cpu | semantics, mode, &d, a, &status), wanted);
          EXPECT_EQ(status, 0x80000000U);
          // Rejected modes must leave both the numeric and FP-status outputs untouched.
          if (wanted) {
            EXPECT_EQ(d, 123U);
            continue;
          }
          float expected;
          if constexpr (Half) {
            EXPECT_EQ(d >> 16, 0U);
            expected = f16_value(half_unary_reference(op, a, uint32_t(mode), false));
            expect_f32(bit_cast<uint32_t>(f16_value(uint16_t(d))), expected,
                       precise ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                               : RJ_ARITH_SEMANTICS_LOOSE,
                       /*atol=*/0x1p-24f, /*rtol=*/1.1e-3f);
          } else {
            expected = precise ? omod_f32_reference(bit_cast<float>(unary_goldens[op][sample % 32]),
                                                    uint32_t(mode))
                               : unary_reference(op, bit_cast<float>(a), uint32_t(mode));
            if (mode & RJ_ARITH_ALU_CLAMP) {
              expected = !(expected > 0) ? 0 : std::min(expected, 1.0f);
            }
            expect_f32(d, expected,
                       precise ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                               : RJ_ARITH_SEMANTICS_LOOSE,
                       /*atol=*/0.0f, /*rtol=*/2.5e-7f);
          }
          uint32_t alias = a;
          ASSERT_EQ(call(cpu | semantics, mode, &alias, alias, &status), RJ_ARITH_SUCCESS);
          EXPECT_EQ(alias, d);
        }
      }
    }
  }
  uint32_t d = 123, status = 456;
  expect_invalid_flags(
      [&](auto f, auto m) {
        int error = call(f, m, &d, 0, &status);
        EXPECT_EQ(d, 123U);
        EXPECT_EQ(status, 456U);
        return error;
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP);
}
} // namespace

// Check scalar-result special functions and semantics support.
TEST(v_s_exp_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_exp_f32, 7); }

// Check scalar-result special functions and semantics support.
TEST(v_s_log_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_log_f32, 8); }

// Check scalar-result special functions and semantics support.
TEST(v_s_rcp_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_rcp_f32, 5); }

// Check scalar-result special functions and semantics support.
TEST(v_s_rsq_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_rsq_f32, 6); }

// Check scalar-result special functions and semantics support.
TEST(v_s_sqrt_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_sqrt_f32, 4); }

// Check scalar-result special functions and semantics support.
TEST(v_s_exp_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_exp_f16, 7); }

// Check scalar-result special functions and semantics support.
TEST(v_s_log_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_log_f16, 8); }

// Check scalar-result special functions and semantics support.
TEST(v_s_rcp_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_rcp_f16, 5); }

// Check scalar-result special functions and semantics support.
TEST(v_s_rsq_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_rsq_f16, 6); }

// Check scalar-result special functions and semantics support.
TEST(v_s_sqrt_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_sqrt_f16, 4); }
