// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar predicate tests with SCC aliases and optional FP status.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/integer.h"
#include "common/semantics.h"
#include "compare/floating_point.h"
#include "rj_arith/rj_arith.h"
#include "scalar/integer_helpers.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check scalar predicates, SCC aliases, and FP-status expectations for the supplied entry point.
template <int Width, bool Floating, class Call, class Predicate>
void check_scalar_compare(Call call, Predicate predicate) {
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      for (bool flush : {false, true}) {
        std::minstd_rand random;
        for (int sample = 0; sample < 128; ++sample) {
          uint64_t a = scalar_input(random, sample),
                   b = scalar_input(random, (sample * 3 + 1) % 128);
          if constexpr (Floating) {
            if (sample < 100) {
              a = compare_class_value<Width>(sample % 10);
              b = compare_class_value<Width>(sample / 10);
            }
          }
          if (sample >= 100 && sample % 4 == 0) {
            b = a;
          }
          uint32_t result = 0xdeadbeefU, status = 0x80000040U, expected_status = status;
          uint64_t flags = cpu | semantics | (flush ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
          bool expected;
          if constexpr (Floating) {
            constexpr uint64_t inf = compare_infinity<Width>(), quiet = compare_quiet_bit<Width>(),
                               magnitude = (1ULL << (Width - 1)) - 1;
            auto am = a & magnitude, bm = b & magnitude;
            bool snan = (am > inf && !(am & quiet)) || (bm > inf && !(bm & quiet));
            bool denorm = (am && !(am & inf)) || (bm && !(bm & inf));
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              if (snan) {
                expected_status |= RJ_ARITH_EXCEPTION_INVALID;
              }
              if (am < inf && bm < inf && denorm && !flush) {
                expected_status |= RJ_ARITH_EXCEPTION_INPUT_DENORM;
              }
            }
            expected = predicate(compare_float_value<Width>(a, 0, 0, flush),
                                 compare_float_value<Width>(b, 0, 0, flush));
          } else {
            expected = predicate(a, b);
          }
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << flush << '/' << sample);
          ASSERT_EQ(call(flags, 0, &result, a, b, &status), RJ_ARITH_SUCCESS);
          EXPECT_EQ(result, uint32_t(expected));
          EXPECT_EQ(status, expected_status);
          // Scalar inputs may be sourced from SCC's own storage.
          uint32_t alias = uint32_t(a), expected_alias = 0, alias_status = 0;
          ASSERT_EQ(call(flags, 0, &expected_alias, alias, b, &alias_status), RJ_ARITH_SUCCESS);
          alias_status = 0;
          ASSERT_EQ(call(flags, 0, &alias, alias, b, &alias_status), RJ_ARITH_SUCCESS);
          EXPECT_EQ(alias, expected_alias);
        }
      }
    }
  }
  uint32_t output = 123, status = 456;
  expect_invalid_flags(
      [&](auto flags, auto mode) {
        int error = call(flags, mode, &output, 0, 0, &status);
        EXPECT_EQ(output, 123U);
        EXPECT_EQ(status, 456U);
        return error;
      },
      0);
}

} // namespace rocjitsu::arith::test
