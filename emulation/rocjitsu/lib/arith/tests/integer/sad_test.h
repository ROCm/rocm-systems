// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sums of absolute differences, sliding windows, and packed accumulators.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "integer/sad_reference.h"
#include "integer/word_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check SAD windows, packed accumulators, and multi-register aliases for the supplied entry
// point.
template <int Words, int Lanes, class Call> void check_sad(Call call, Sad operation) {
  using Mask = WaveMask<Lanes>;
  auto input = make_integer_inputs<Lanes>();
  // Near-limit addends expose independent saturation/wrap in each result;
  // zero B bytes distinguish masked SAD from ordinary absolute differences.
  for (int lane = 0; lane < 16; ++lane) {
    for (int word = 0; word < Words; ++word) {
      input.v[4 + word][lane] = (Words == 2 ? 0xfff0fff0U : 0xfffffff0U) + uint32_t(lane);
    }
    input.v[2][lane] &= lane % 2 ? 0x00ff00ffU : UINT32_MAX;
  }
  expect_alu_invalid_flags<Words == 1>(
      [&](uint64_t flags, uint64_t mode) {
        auto r = input;
        int error = call(flags, ~Mask(0), mode, r.v + 8, r.v, r.v + 2, r.v + 4);
        expect_integer_registers(r, input);
        EXPECT_EQ(call(flags, ~Mask(0), mode, nullptr, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return error;
      },
      RJ_ARITH_ALU_CLAMP);
  const auto modes = integer_modes(RJ_ARITH_ALU_CLAMP, Words == 1);
  const auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 5 + int(modes.size()); ++scenario) {
        uint64_t mode = scenario < 5 ? 0 : modes[scenario - 5];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario < 5 ? scenario : scenario % 2 ? 0 : 3);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        for (int aliasing_offset : {8, 0, 2, 4}) {
          auto r = input, expected = input;
          for (int lane = 0; lane < Lanes; ++lane) {
            int source;
            if (!integer_source(mode, exec_mask, lane, source)) {
              continue;
            }
            uint32_t c[4];
            for (int word = 0; word < 4; ++word) {
              c[word] = input.v[4 + word][lane];
            }
            auto result =
                sad_reference(operation, source < 0 ? 0 : input.v[0][source], input.v[1][lane],
                              input.v[2][lane], c, mode & RJ_ARITH_ALU_CLAMP);
            for (int word = 0; word < Words; ++word) {
              expected.v[aliasing_offset + word][lane] = result[word];
            }
          }
          ASSERT_EQ(call(cpu | semantics, exec_mask, mode,
                         exec_mask ? r.v + aliasing_offset : nullptr, exec_mask ? r.v : nullptr,
                         exec_mask ? r.v + 2 : nullptr, exec_mask ? r.v + 4 : nullptr),
                    RJ_ARITH_SUCCESS);
          expect_integer_registers(r, expected);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
