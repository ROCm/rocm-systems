// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Predicate bindings and NaN-only Compatibility status for gfx908/gfx10.
// Numeric references reuse the already-validated underlying comparison APIs.
#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "compare/floating_point.h"
#include "compare/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check the wrapper's numeric binding against the inherited comparator, with
// literal status rules independent of the implementation's unordered-mask call.
template <int Bits, int Lanes, bool Constant = false, typename Call>
void check_signaling_clamp_compare(Call call, Call numeric) {
  using Mask = WaveMask<Lanes>;
  auto input = make_compare_float_inputs<Bits, Lanes>();
  for (int half = 0; half < (Bits == 16 ? 2 : 1); ++half) {
    write_compare_operand(input.v, Lanes - 1, Bits,
                          compare_infinity<Bits>() | compare_quiet_bit<Bits>() | 1, half);
    write_compare_operand(input.v + 2, Lanes - 1, Bits, 0, half);
    write_compare_operand(input.v, 5, Bits, 1, half);
    // A preserved subnormal/finite pair must not add INPUT_DENORM.
    write_compare_operand(input.v + 2, 5, Bits, compare_class_value<Bits>(8), half);
  }
  constexpr uint64_t modifiers = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A |
                                 RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP |
                                 (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0);
  expect_alu_invalid_flags<Bits != 64>(
      [&](uint64_t flags, uint64_t mode) {
        Mask result = ~Mask(0);
        uint32_t status = 0x100;
        int error = call(flags, 0, mode, &result, nullptr, nullptr, &status);
        EXPECT_EQ(result, ~Mask(0));
        EXPECT_EQ(status, 0x100U);
        return error;
      },
      modifiers);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 4; ++scenario) {
        const Mask exec_masks[] = {~Mask(0), Mask(1) << (Lanes - 1), Mask(1) << 5, 0};
        const Mask exec_mask = exec_masks[scenario];
        const uint64_t mode = scenario
                                  ? RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B |
                                        (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0)
                                  : 0;
        const uint64_t flags =
            cpu | semantics |
            (scenario & 1 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE : 0);
        const bool rejected = !Constant && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
        Mask expected = ~Mask(0);
        if (!rejected) {
          const uint64_t numeric_flags =
              Constant && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                  ? (flags & ~RJ_ARITH_SEMANTICS_MASK) |
                        RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                  : flags;
          ASSERT_EQ(
              numeric(numeric_flags, exec_mask, mode, &expected, input.v, input.v + 2, nullptr),
              RJ_ARITH_SUCCESS);
        }
        uint32_t expected_status = 0x100;
        if (!rejected &&
            (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY ||
             (Constant && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL)) &&
            scenario) {
          for (int lane = 0; lane < Lanes; ++lane) {
            if ((exec_mask >> lane) & 1) {
              for (int operand : {0, 2}) {
                const uint64_t value = compare_operand(input.v + operand, lane, Bits, Bits == 16);
                if ((value & ((1ULL << (Bits - 1)) - 1)) > compare_infinity<Bits>()) {
                  expected_status |= RJ_ARITH_EXCEPTION_INVALID;
                }
              }
            }
          }
        }
        for (int aliasing_register : {-1, 0, 2}) {
          expect_comparison(
              input, expected, rejected ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS : RJ_ARITH_SUCCESS,
              aliasing_register, [&](auto &r, Mask *result) {
                uint32_t status = 0x100;
                int error = call(flags, exec_mask, mode, result, exec_mask ? r.v : nullptr,
                                 exec_mask ? r.v + 2 : nullptr, &status);
                EXPECT_EQ(status, expected_status);
                return error;
              });
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
