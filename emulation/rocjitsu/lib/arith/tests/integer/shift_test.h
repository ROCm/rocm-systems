// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 32/64-bit shifts and count masking.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "integer/shift_reference.h"
#include "integer/word_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check wide/narrow shifts, count masking, and aliases for the supplied entry point.
template <int Width, int Lanes, typename Call> void check_shift(Call call, Shift operation) {
  using Mask = WaveMask<Lanes>;
  const auto input = make_integer_inputs<Lanes>();
  expect_alu_invalid_flags<Width == 32>(
      [&](uint64_t flags, uint64_t mode) {
        auto r = input;
        int error = call(flags, ~Mask(0), mode, r.v + 8, r.v, r.v + 2);
        expect_integer_registers(r, input);
        EXPECT_EQ(call(flags, ~Mask(0), mode, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return error;
      },
      RJ_ARITH_ALU_CLAMP);
  const auto modes = integer_modes(RJ_ARITH_ALU_CLAMP, Width == 32);
  const auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 5 + int(modes.size()); ++scenario) {
        uint64_t mode = scenario < 5 ? 0 : modes[scenario - 5];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario < 5 ? scenario : scenario % 2 ? 0 : 3);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        for (int aliasing_offset : {8, 0, 2}) {
          auto r = input, expected = input;
          for (int lane = 0; lane < Lanes; ++lane) {
            int source;
            if (!integer_source(mode, exec_mask, lane, source)) {
              continue;
            }
            uint32_t count = source < 0 ? 0 : input.v[0][source];
            uint64_t value = input.v[2][lane];
            if constexpr (Width == 64) {
              value |= uint64_t(input.v[3][lane]) << 32;
            }
            uint64_t result = shift_reference(Width, operation, count, value);
            expected.v[aliasing_offset][lane] = uint32_t(result);
            if constexpr (Width == 64) {
              expected.v[aliasing_offset + 1][lane] = uint32_t(result >> 32);
            }
          }
          ASSERT_EQ(call(cpu | semantics, exec_mask, mode,
                         exec_mask ? r.v + aliasing_offset : nullptr, exec_mask ? r.v : nullptr,
                         exec_mask ? r.v + 2 : nullptr),
                    RJ_ARITH_SUCCESS);
          expect_integer_registers(r, expected);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
