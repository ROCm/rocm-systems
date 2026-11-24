// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Check the inherited rectangle coverage model, masks, aliases and semantics policy.
#include "common/contracts.h"
#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_register.h"
#include <gtest/gtest.h>
#include <stdint.h>

namespace {
// Independently enumerate the wrapped rectangle's shader-engine assignments.
uint32_t coverage(uint32_t selector) {
  int xmin = (selector >> 6) & 3, ymin = (selector >> 4) & 3;
  int dx = (((selector >> 2) & 3) - xmin) & 3, dy = ((selector & 3) - ymin) & 3;
  uint32_t result = 0;
  for (int x = xmin; x <= xmin + dx; ++x) {
    for (int y = ymin; y <= ymin + dy; ++y) {
      int engine = (((x ^ (y >> 1)) & 1) << 1) | ((y ^ (x >> 1)) & 1);
      result |= 1U << engine;
    }
  }
  return result;
}
} // namespace

TEST(RegisterScreenPartition, Arithmetic) {
  using namespace rocjitsu::arith::test;
  Registers<3, 64> input;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int batch = 0; batch < 4; ++batch) {
        for (int lane = 0; lane < 64; ++lane) {
          input.v[0][lane] = 0xabcd0000U | uint32_t(batch * 64 + lane);
          input.v[1][lane] = input.v[2][lane] = 0xdeadbeefU;
        }
        for (uint64_t exec_mask : {0ULL, 0x80000000800000a5ULL, ~0ULL}) {
          for (int aliasing_offset : {0, 2}) {
            auto actual = input, expected = input;
            bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                          semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
            ASSERT_EQ(rj_arith_v_screen_partition_4se_b32_w64(cpu | semantics, exec_mask, 0,
                                                              actual.v + aliasing_offset, actual.v),
                      strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
            for (int lane = 0; lane < 64; ++lane) {
              if (!strict && ((exec_mask >> lane) & 1)) {
                expected.v[aliasing_offset][lane] = coverage(input.v[0][lane]);
              }
            }
            for (int reg = 0; reg < 3; ++reg) {
              for (int lane = 0; lane < 64; ++lane) {
                ASSERT_EQ(actual.v[reg][lane], expected.v[reg][lane]);
              }
            }
          }
        }
      }
    }
  }
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t controls) {
        return rj_arith_v_screen_partition_4se_b32_w64(flags, 0, controls, nullptr, nullptr);
      },
      0);
}
