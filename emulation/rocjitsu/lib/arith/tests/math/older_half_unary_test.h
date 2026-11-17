// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older-profile unary witnesses, selected-half aliases, masks and qualification checks.
#pragma once

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <bit>
#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {
// Selected inherited-model values distinguish directed integral rounding and NaN payloads.
template <int Lanes, class Call> void check_older_half_unary(Call call, int op) {
  const uint16_t inputs[] = {0x3e00, 0xbe00, 0x7d23, 0xfc00, 0x8001};
  const uint16_t expected[][5] = {
      {0x3c00, 0xbc00, 0x7f23, 0xfc00, 0x8000}, {0x4000, 0xbc00, 0x7f23, 0xfc00, 0x8000},
      {0x4000, 0xc000, 0x7f23, 0xfc00, 0x8000}, {0x3c00, 0xc000, 0x7f23, 0xfc00, 0xbc00},
      {0x3a00, 0xba00, 0x7f23, 0xfc00, 0xb800}, {0x3800, 0x3800, 0x7f23, 0xfe00, 0x3bff}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      uint32_t data[Lanes], original[Lanes];
      for (int lane = 0; lane < Lanes; ++lane) {
        data[lane] = original[lane] = (uint32_t(inputs[lane % 5]) << 16) | 0xbeef;
      }
      uint32_t *d = data;
      const uint32_t *a = data;
      uint32_t status = 0x1234;
      const auto exec_mask = WaveMask<Lanes>(0x8000000380000005ULL);
      const int error = call(cpu | semantics, exec_mask, RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D,
                             &d, &a, &status);
      const bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                          semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
      EXPECT_EQ(error == RJ_ARITH_SUCCESS, !strict);
      for (int lane = 0; lane < Lanes; ++lane) {
        const uint32_t want = !strict && ((exec_mask >> lane) & 1)
                                  ? (uint32_t(expected[op][lane % 5]) << 16) | 0xbeef
                                  : original[lane];
        // This target's unqualified fallback has Loose's numerical contract too.
        const bool loose = semantics == RJ_ARITH_SEMANTICS_LOOSE ||
                           semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE;
        if (loose && ((exec_mask >> lane) & 1)) {
          EXPECT_EQ(data[lane] & 0xffffU, original[lane] & 0xffffU);
          expect_f32(std::bit_cast<uint32_t>(f16_value(uint16_t(data[lane] >> 16))),
                     f16_value(expected[op][lane % 5]), RJ_ARITH_SEMANTICS_LOOSE,
                     /*atol=*/0x1p-24, /*rtol=*/1.1e-3);
        } else {
          EXPECT_EQ(data[lane], want) << op << ',' << lane << ',' << cpu << ',' << semantics;
        }
      }
      EXPECT_EQ(status, 0x1234U);
    }
  }
}
// A midpoint-adjacent negative input distinguishes every guest rounding direction.
template <int Lanes, class Call> void check_half_fract_rounding(Call call) {
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t rounding :
         {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      uint32_t input[Lanes]{}, output[Lanes]{};
      input[0] = 0xb401;
      const uint32_t *a = input;
      uint32_t *d = output;
      uint32_t status = 0x1234;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | rounding, WaveMask<Lanes>(1), 0, &d,
                     &a, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(output[0], rounding <= RJ_ARITH_FP_ROUND_UP ? 0x3a00U : 0x39ffU);
      EXPECT_EQ(status, 0x1234U);
    }
  }
}
} // namespace rocjitsu::arith::test
