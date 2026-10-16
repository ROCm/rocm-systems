// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar FP fixtures, numeric comparisons, and semantics/status contracts.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
#include "common/integer.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Scalar arithmetic uses the original suite's wider mathematical references.
// Integral rounding copies exactly representable integers or signed zeros,
// independently validated by the old captures. FP32 FMA uses the established
// std::fma reference also used by the vector suite. Unsupported numerical and
// FP-status modes are checked separately.
template <bool Half, class Call, class Reference>
void check_scalar_fp(Call call, Reference reference, bool exact = false, int output_format = 0,
                     bool report = true) {
  constexpr uint32_t inf = Half ? 0x7c00U : 0x7f800000U, quiet = Half ? 0x200U : 0x400000U;
  constexpr uint32_t sign = Half ? 0x8000U : 0x80000000U;
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      std::minstd_rand random;
      for (int sample = 0; sample < 128; ++sample) {
        uint32_t a = Half ? random_f16_bits(random) : random_f32_bits(random);
        uint32_t b = Half ? random_f16_bits(random) : random_f32_bits(random);
        uint32_t c = Half ? random_f16_bits(random) : random_f32_bits(random);
        const uint32_t edges[] = {0, sign, inf, inf | sign, inf | quiet | 1, inf | 1, 1, sign | 1};
        if (sample < 8) {
          a = edges[sample];
        }
        if (sample >= 8 && sample < 16) {
          b = edges[sample - 8];
        }
        if (Half) {
          a |= random_word(random) & 0xffff0000U;
          b |= random_word(random) & 0xffff0000U;
        }
        bool flush = sample % 3 == 0;
        uint64_t flags = cpu | semantics | (flush ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
        int wanted = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL && !report
                         ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                     : exact ? RJ_ARITH_SUCCESS
                     : semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                         ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                     : semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                         ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                         : RJ_ARITH_SUCCESS;
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << sample);
        uint32_t words[] = {123, c, 456}, status = 0x80000040U, expected_status = status;
        ASSERT_EQ(call(flags, 0, words + 1, a, b, c, &status), wanted);
        if (wanted) {
          EXPECT_EQ(words[1], c);
        } else {
          uint32_t expected = reference(a, b, c, flush);
          if (output_format == 1) {
            EXPECT_EQ(words[1], expected);
          } else if (Half || output_format == 2) {
            EXPECT_EQ(words[1] >> 16, 0U);
            expect_f32(bit_cast<uint32_t>(f16_value(uint16_t(words[1]))),
                       f16_value(uint16_t(expected)), exact ? semantics : RJ_ARITH_SEMANTICS_LOOSE,
                       /*atol=*/0x1p-24f, /*rtol=*/1e-3f);
          } else {
            expect_f32(words[1], bit_cast<float>(expected),
                       exact ? semantics : RJ_ARITH_SEMANTICS_LOOSE,
                       /*atol=*/0x1p-149f, /*rtol=*/1.2e-7f);
          }
          if (exact && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            uint32_t magnitude = a & (sign - 1);
            if (magnitude > inf && !(magnitude & quiet)) {
              expected_status |= RJ_ARITH_EXCEPTION_INVALID;
            } else if (magnitude && !(magnitude & inf) && !flush) {
              expected_status |= RJ_ARITH_EXCEPTION_INPUT_DENORM;
            }
          }
          // A separate alias call uses an identical snapshotted accumulator.
          uint32_t alias = c, alias_status = 0x80000040U;
          ASSERT_EQ(call(flags, 0, &alias, a, b, alias, &alias_status), RJ_ARITH_SUCCESS);
          EXPECT_EQ(alias, words[1]);
          EXPECT_EQ(alias_status, status);
        }
        EXPECT_EQ(status, expected_status);
        EXPECT_EQ(words[0], 123U);
        EXPECT_EQ(words[2], 456U);
      }
    }
  }
  uint32_t d = 123, status = 456;
  expect_invalid_flags(
      [&](auto flags, auto mode) {
        int error = call(flags, mode, &d, 0, 0, 0, &status);
        EXPECT_EQ(d, 123U);
        EXPECT_EQ(status, 456U);
        return error;
      },
      0);
}

// Decode FP16/FP32 scalar bits, optionally flushing subnormal inputs.
template <bool Half> double scalar_fp_value(uint32_t bits, bool flush) {
  double value = Half ? half_value(uint16_t(bits)) : double(bit_cast<float>(bits));
  if (flush && std::abs(value) < (Half ? 0x1p-14 : 0x1p-126)) {
    value = std::copysign(0.0, value);
  }
  return value;
}

// Encode a scalar reference value as FP16 or FP32 bits.
template <bool Half> uint32_t scalar_fp_bits(double value) {
  return Half ? half_bits(value) : bit_cast<uint32_t>(float(value));
}

} // namespace rocjitsu::arith::test
