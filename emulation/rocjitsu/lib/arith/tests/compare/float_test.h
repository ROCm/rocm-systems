// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Floating-point predicate tests across widths, wave sizes, and gfx policies.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "compare/floating_point.h"
#include "compare/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Literal gfx12 exception cases from the inherited comparison model.
// Each cause is checked alone, accumulated, and with the
// exceptional lane inactive. Literal expectations avoid duplicating the model.
// Class indices refer to compare_class_value's ISA order.
struct CompareStatusCase {
  int a_class, b_class;
  uint32_t preserved, flushed, signaling_preserved, signaling_flushed;
};

static const CompareStatusCase compare_status_cases[] = {
    {8, 8, 0, 0, 0, 0}, // normal/normal
    {0, 8, RJ_ARITH_EXCEPTION_INVALID, RJ_ARITH_EXCEPTION_INVALID, RJ_ARITH_EXCEPTION_INVALID,
     RJ_ARITH_EXCEPTION_INVALID}, // sNaN/normal
    {8, 0, RJ_ARITH_EXCEPTION_INVALID, RJ_ARITH_EXCEPTION_INVALID, RJ_ARITH_EXCEPTION_INVALID,
     RJ_ARITH_EXCEPTION_INVALID}, // normal/sNaN
    {1, 8, 0, 0, RJ_ARITH_EXCEPTION_INVALID,
     RJ_ARITH_EXCEPTION_INVALID}, // qNaN signals only with CLAMP
    {8, 1, 0, 0, RJ_ARITH_EXCEPTION_INVALID, RJ_ARITH_EXCEPTION_INVALID},
    {7, 8, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0, RJ_ARITH_EXCEPTION_INPUT_DENORM,
     0}, // preserved subnormal/finite
    {8, 4, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0},
    {7, 9, 0, 0, 0, 0}, // infinity suppresses INPUT_DENORM
    {7, 1, 0, 0, RJ_ARITH_EXCEPTION_INVALID,
     RJ_ARITH_EXCEPTION_INVALID}, // NaN suppresses INPUT_DENORM
    {4, 0, RJ_ARITH_EXCEPTION_INVALID, RJ_ARITH_EXCEPTION_INVALID, RJ_ARITH_EXCEPTION_INVALID,
     RJ_ARITH_EXCEPTION_INVALID},
};

// Check FP predicate masks, modifiers, aliases, and FP-status policy for the supplied entry
// point.
template <int Width, int Lanes, typename Call>
void check_float_compare(Call call, bool gfx12, bool (*predicate)(double, double)) {
  using Mask = WaveMask<Lanes>;
  const auto input = make_compare_float_inputs<Width, Lanes>();
  const uint64_t low_modifiers = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A |
                                 RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP |
                                 (Width == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0);
  expect_alu_invalid_flags<Width != 64>(
      [&](uint64_t flags, uint64_t mode) {
        Mask d = ~Mask(0);
        uint32_t status = 0x80000040U;
        int error = call(flags, ~Mask(0), mode, &d, nullptr, nullptr, &status);
        EXPECT_EQ(d, ~Mask(0));
        EXPECT_EQ(status, 0x80000040U);
        return error;
      },
      low_modifiers);

  std::vector<uint64_t> modes = {0, RJ_ARITH_ALU_ABS_A, RJ_ARITH_ALU_NEG_B,
                                 RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_B |
                                     RJ_ARITH_ALU_NEG_B,
                                 RJ_ARITH_ALU_CLAMP};
  if constexpr (Width == 16) {
    modes.insert(modes.end(), {RJ_ARITH_ALU_HIGH_A, RJ_ARITH_ALU_HIGH_B,
                               RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B});
  }
  if constexpr (Width != 64) {
    for (uint64_t descriptor : dpp_modes) {
      modes.push_back(descriptor | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_B);
    }
  }
  const auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < int(modes.size()) + 5; ++scenario) {
        uint64_t mode = scenario < 5 ? 0 : modes[scenario - 5];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario);
        bool flush = scenario % 2;
        uint64_t flags = cpu | semantics | RJ_ARITH_FP_IEEE_MODE |
                         RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                         (flush ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
        bool reporting = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
        int error = reporting && !gfx12 ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS : RJ_ARITH_SUCCESS;
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        Mask expected = comparison_mask<Lanes>(mode, exec_mask, [&](int source, int lane) {
          return predicate(compare_float_value<Width>(
                               compare_operand(input.v, source, Width, mode & RJ_ARITH_ALU_HIGH_A),
                               mode, 0, flush),
                           compare_float_value<Width>(compare_operand(input.v + 2, lane, Width,
                                                                      mode & RJ_ARITH_ALU_HIGH_B),
                                                      mode, 1, flush));
        });
        for (int aliasing_register : {-1, 0, 2}) {
          expect_comparison(input, expected, error, aliasing_register, [&](auto &r, Mask *d) {
            uint32_t status = 0x80000040U;
            int result = call(flags, exec_mask, mode, d, exec_mask ? r.v : nullptr,
                              exec_mask ? r.v + 2 : nullptr, &status);
            if (!reporting || error || !exec_mask) {
              EXPECT_EQ(status, 0x80000040U);
            } else {
              EXPECT_EQ(status & ~(RJ_ARITH_EXCEPTION_INVALID | RJ_ARITH_EXCEPTION_INPUT_DENORM),
                        0x80000040U);
            }
            return result;
          });
        }
        // Modes without reporting must also accept null status storage.
        if (!reporting) {
          Mask d = 0;
          ASSERT_EQ(call(flags, exec_mask, mode, &d, exec_mask ? input.v : nullptr,
                         exec_mask ? input.v + 2 : nullptr, nullptr),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(d, expected);
        }
      }
    }
  }

  // Keep status cases separate from random predicate checks: no cross-product
  // of modifiers, predicates' data, masks and aliasing is needed here.
  for (uint64_t cpu : cpu_levels) {
    for (const auto &test : compare_status_cases) {
      for (int setting = 0; setting < 4; ++setting) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << '/' << test.a_class << '/' << test.b_class << '/' << setting);
        auto r = input;
        bool flush = setting & 1, signaling = setting & 2;
        int lane = Lanes - 1;
        uint64_t mode = (signaling ? RJ_ARITH_ALU_CLAMP : 0) | RJ_ARITH_ALU_NEG_A |
                        RJ_ARITH_ALU_ABS_B |
                        (Width == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0);
        write_compare_operand(r.v, lane, Width, compare_class_value<Width>(test.a_class),
                              Width == 16);
        write_compare_operand(r.v + 2, lane, Width, compare_class_value<Width>(test.b_class),
                              Width == 16);
        uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL |
                         (flush ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
        const uint32_t causes[] = {test.preserved, test.flushed, test.signaling_preserved,
                                   test.signaling_flushed};
        uint32_t status = 0;
        Mask d = ~Mask(0);
        for (Mask exec_mask : {Mask(1) << lane, Mask(0)}) {
          for (uint32_t initial : {0U, 0x80000040U}) {
            status = initial;
            ASSERT_EQ(call(flags, exec_mask, mode, &d, r.v, r.v + 2, &status),
                      gfx12 ? RJ_ARITH_SUCCESS : RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
            EXPECT_EQ(status, initial | (gfx12 && exec_mask ? causes[setting] : 0));
          }
        }
      }
    }
  }

  // A single sNaN routed through DPP checks exception participation exactly:
  // an inactive/filtered source can become zero, so it must not raise INVALID.
  if constexpr (Width != 64) {
    if (gfx12) {
      Registers<4, Lanes> r;
      for (int reg = 0; reg < 4; ++reg) {
        for (int lane = 0; lane < Lanes; ++lane) {
          r.v[reg][lane] = 0;
        }
      }
      write_compare_operand(r.v, Lanes - 1, Width, compare_class_value<Width>(0));
      for (uint64_t cpu : cpu_levels) {
        for (uint64_t mode : dpp_modes) {
          for (int scenario = 0; scenario < 5; ++scenario) {
            Mask exec_mask = exec_mask_for_case<Lanes>(scenario);
            SCOPED_TRACE(::testing::Message() << cpu << '/' << mode << '/' << scenario);
            Mask exceptional = comparison_mask<Lanes>(
                mode, exec_mask, [&](int source, int) { return source == Lanes - 1; });
            uint32_t status = 0x80000040U;
            Mask d = 0;
            ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, &d, r.v,
                           r.v + 2, &status),
                      RJ_ARITH_SUCCESS);
            EXPECT_EQ(status, 0x80000040U | (exceptional ? RJ_ARITH_EXCEPTION_INVALID : 0));
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
