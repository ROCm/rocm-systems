// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Selected exact gfx12 BF16 WMMA boundaries shared by Wave32 and Wave64.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include "wmma/helpers.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// gfx1201 cancellation witness supplied in the review of PR #13939:
// 2^64 * 2^64 + (-2^127) * 1 = 2^127, despite the first product exceeding FP32.
template <int Lanes, typename Call> void expect_bf16_overflow_cancellation(Call call) {
  constexpr int factors = 128 / Lanes, accumulators = 256 / Lanes;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      if (semantics == RJ_ARITH_SEMANTICS_LOOSE || semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
        continue;
      }
      // The Wave64 gfx1201 witness puts products in K groups 4 and 8:
      // positive first overflows, negative first cancels to 2^127.
      for (int separate_groups = 0; separate_groups <= (Lanes == 64); ++separate_groups) {
        for (int reverse : {0, 1}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << reverse << '/' << separate_groups);
          const int first = separate_groups ? 4 : 0, second = separate_groups ? 8 : 1;
          const int positive_k = reverse ? second : first, negative_k = reverse ? first : second;
          Registers<24, Lanes> r;
          for (auto *reg : r.v) {
            std::fill_n(reg, Lanes, 0U);
          }
          set_wmma_gfx12_half<Lanes>(r.v, 0, positive_k, 0x5f80U);           // A[0][k] = 2^64.
          set_wmma_gfx12_half<Lanes>(r.v + factors, 0, positive_k, 0x5f80U); // B[k][0] = 2^64.
          set_wmma_gfx12_half<Lanes>(r.v, 0, negative_k, 0xff00U);           // A[0][k] = -2^127.
          set_wmma_gfx12_half<Lanes>(r.v + factors, 0, negative_k, 0x3f80U); // B[k][0] = 1.
          ASSERT_EQ(
              call(cpu | semantics, 0, r.v + 4 * factors, r.v, r.v + factors, r.v + 2 * factors),
              RJ_ARITH_SUCCESS);
          const uint32_t expected = separate_groups && !reverse ? 0x7f800000U : 0x7f000000U;
          for (int reg = 0; reg < accumulators; ++reg) {
            for (int lane = 0; lane < Lanes; ++lane) {
              EXPECT_EQ(r.v[4 * factors + reg][lane], reg == 0 && lane == 0 ? expected : 0U);
            }
          }
        }
      }
    }
  }
}

// Selected gfx1201 capture: kGfx12WmmaBF16Cases[5] in
// rocjitsu/tests/fixtures/float_dot/gfx1201_cases.h (Radeon AI Pro R9700).
// Its independently recorded expected32 and expected64 fields are both 0x78bd8ea7,
// despite the different K-group accumulation orders.
template <int Lanes, typename Call> void expect_bf16_exponent_gaps(Call call) {
  constexpr int factors = 128 / Lanes, accumulators = 256 / Lanes;
  const uint16_t a[16] = {0x9844, 0x3f7a, 0x606f, 0xc18c, 0xbe5b, 0x891f, 0xbbfb, 0x93b8,
                          0xc3c6, 0x32f2, 0xf44e, 0x15cf, 0x5eeb, 0x2b55, 0x8bef, 0xa075};
  const uint16_t b[16] = {0x8e88, 0xec18, 0xd3c2, 0xf6ae, 0x0cdb, 0x87c5, 0x4587, 0x5825,
                          0x6e7d, 0x8ab5, 0x8773, 0x4108, 0x91c2, 0x3987, 0x62ab, 0xfbad};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      Registers<24, Lanes> r;
      for (int row = 0; row < 16; ++row) {
        for (int k = 0; k < 16; ++k) {
          set_wmma_gfx12_half<Lanes>(r.v, row, k, a[k]);
          set_wmma_gfx12_half<Lanes>(r.v + factors, row, k, b[k]);
        }
      }
      for (int reg = 2 * factors; reg < 4 * factors; ++reg) {
        std::fill_n(r.v[reg], Lanes, 0x48e26837U);
      }
      ASSERT_EQ(call(cpu | semantics, 0, r.v + 4 * factors, r.v, r.v + factors, r.v + 2 * factors),
                RJ_ARITH_SUCCESS);
      for (int reg = 4 * factors; reg < 4 * factors + accumulators; ++reg) {
        for (int lane = 0; lane < Lanes; ++lane) {
          EXPECT_EQ(r.v[reg][lane], 0x78bd8ea7U);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
