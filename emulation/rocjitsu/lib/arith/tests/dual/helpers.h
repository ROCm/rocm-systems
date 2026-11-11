// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// VOPD wrapper agreement with ordinary instruction entry points.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// VOPD arithmetic is defined to match its ordinary instruction. Those APIs
// have independent numerical tests; this checks decoding restrictions, old-D
// accumulation, aliasing and status forwarding without duplicating their oracles.
// A separate literal example prevents matching two broken arithmetic routes.
template <typename Call, typename Ordinary>
void check_dual(Call call, Ordinary ordinary, uint32_t a, uint32_t b, uint32_t c, uint32_t expected,
                bool select = false) {
  Registers<4, 32> input;
  for (int lane = 0; lane < 32; ++lane) {
    input.v[0][lane] = a;
    input.v[1][lane] = b;
    input.v[3][lane] = c;
  }
  uint32_t status = 0x80000000U;
  ASSERT_EQ(call(0, UINT32_MAX, 0, input.v + 3, input.v, input.v + 1, &status), RJ_ARITH_SUCCESS);
  for (int lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(input.v[3][lane], select ? (lane & 1 ? b : a) : expected);
  }
  EXPECT_EQ(status, 0x80000000U);
  std::minstd_rand random;
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      input.v[reg][lane] = random_word(random);
    }
  }
  for (int reg = 0; reg < 3; ++reg) {
    input.v[reg][reg * 4] = 0x7f800001U;
    input.v[reg][reg * 4 + 1] = 0x7f800000U;
    input.v[reg][reg * 4 + 2] = 1;
  }
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 10; ++scenario) {
        for (int d : {0, 1, 3}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << scenario << '/' << d);
          auto actual = input, reference = input;
          auto exec_mask = exec_mask_for_case<32>(scenario);
          uint64_t flags = cpu | semantics |
                           (scenario >= 5 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS
                                          : 0);
          uint32_t actual_status = 0x80000000U, reference_status = actual_status;
          int wanted = ordinary(flags, exec_mask, 0, reference.v + d, reference.v, reference.v + 1,
                                &reference_status);
          ASSERT_EQ(call(flags, exec_mask, 0, actual.v + d, actual.v, actual.v + 1, &actual_status),
                    wanted);
          EXPECT_EQ(actual_status, reference_status);
          EXPECT_EQ(0, std::memcmp(actual.words, reference.words, sizeof actual.words));
          actual.expect_preserved(input, d, wanted ? 0 : 1);
          for (int lane = 0; lane < 32; ++lane) {
            if (wanted || !((exec_mask >> lane) & 1)) {
              EXPECT_EQ(actual.v[d][lane], input.v[d][lane]);
            }
          }
        }
      }
    }
  }
  auto unchanged = input;
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t mode) {
        uint32_t state = 0x80000000U;
        int result =
            call(flags, UINT32_MAX, mode, unchanged.v + 3, unchanged.v, unchanged.v + 1, &state);
        unchanged.expect_preserved(input, 0, 0);
        EXPECT_EQ(state, 0x80000000U);
        return result;
      },
      0);
}

} // namespace rocjitsu::arith::test
