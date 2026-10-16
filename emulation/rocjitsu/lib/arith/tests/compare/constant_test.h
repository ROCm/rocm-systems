// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Always-true and always-false comparison masks.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "compare/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Constant predicates ignore operands but retain EXEC/DPP filtering. Floating
// forms cannot report RDNA3 FP status; compatibility resolves to no-status exact.
template <int Width, int Lanes, bool Floating, class Call>
void check_constant_compare(Call call, bool value) {
  using Mask = WaveMask<Lanes>;
  auto invoke = [&](uint64_t flags, Mask exec_mask, uint64_t mode, Mask *d, uint32_t *status) {
    if constexpr (Floating) {
      return call(flags, exec_mask, mode, d, nullptr, nullptr, status);
    } else {
      return call(flags, exec_mask, mode, d, nullptr, nullptr);
    }
  };
  uint64_t low_modifiers = Floating
                               ? RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A |
                                     RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP |
                                     (Width == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0)
                               : 0;
  expect_alu_invalid_flags<Width != 64>(
      [&](uint64_t flags, uint64_t mode) {
        Mask d = ~Mask(0);
        uint32_t status = 0x80000040U;
        int error = invoke(flags, ~Mask(0), mode, &d, &status);
        EXPECT_EQ(d, ~Mask(0));
        EXPECT_EQ(status, 0x80000040U);
        return error;
      },
      low_modifiers);
  std::vector<uint64_t> modes = {0, low_modifiers};
  if constexpr (Width != 64) {
    for (uint64_t descriptor : dpp_modes) {
      modes.push_back(descriptor | low_modifiers);
    }
  }
  const auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < int(modes.size()) + 5; ++scenario) {
        uint64_t mode = scenario < 5 ? 0 : modes[scenario - 5];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);

        struct {
          Mask before, result, after;
        } output{Mask(17), ~Mask(0), Mask(23)};

        uint32_t status = 0x80000040U;
        uint64_t fp_flags = Floating
                                ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                      RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE
                                : 0;
        bool unsupported = Floating && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
        ASSERT_EQ(invoke(cpu | semantics | fp_flags, exec_mask, mode, &output.result, &status),
                  unsupported ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS : RJ_ARITH_SUCCESS);
        Mask expected = comparison_mask<Lanes>(mode, exec_mask, [&](int, int) { return value; });
        EXPECT_EQ(output.result, unsupported ? ~Mask(0) : expected);
        EXPECT_EQ(output.before, Mask(17));
        EXPECT_EQ(output.after, Mask(23));
        EXPECT_EQ(status, 0x80000040U);
        if constexpr (Floating) {
          if (!unsupported) {
            ASSERT_EQ(invoke(cpu | semantics | fp_flags, exec_mask, mode, &output.result, nullptr),
                      RJ_ARITH_SUCCESS);
            EXPECT_EQ(output.result, expected);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
