// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Graphics test scenarios, operand aliases, and destination preservation.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "graphics/cube_reference.h"
#include "graphics/interp16_reference.h"
#include "graphics/interp32_reference.h"
#include "graphics/mullit_reference.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// The interpolation oracles were validated against gfx1201 captures by the
// original suite. Cube selection follows the ISA; the bounded normal inputs
// avoid overflow rounding ambiguity, with ties, signed zeros and quiet NaNs
// inserted explicitly. Lighting has a loose-only mathematical reference.
// Check the supplied graphics entry point against the family reference, including modifiers,
// aliases, and unsupported semantics.
template <int Lanes, class Call>
void check_graphics(Call call, int op, int interpolation, bool lighting = false,
                    bool gfx12 = true) {
  Registers<4, Lanes> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] =
          interpolation == 16 && (reg == 0 || (reg == 2 && !(op & 1)))
              ? uint32_t(random_f16_bits(random)) | (uint32_t(random_f16_bits(random)) << 16)
              : random_f32_bits(random);
    }
  }
  // Add exceptional values and equal magnitudes explicitly; random finite inputs alone rarely
  // exercise these branches.
  for (int reg = 0; reg < 3; ++reg) {
    input.v[reg][reg * 4] =
        interpolation == 16 && (reg == 0 || (reg == 2 && !(op & 1))) ? 0x7e007e00U : 0x7fc00000U;
    input.v[reg][reg * 4 + 1] =
        interpolation == 16 && (reg == 0 || (reg == 2 && !(op & 1))) ? 0x7c007c00U : 0x7f800000U;
    input.v[reg][15] = 0;
    input.v[reg][16] = 0x80000000U;
    if (!interpolation) {
      input.v[reg][17] = 0x3f800000U;
    }
  }
  uint64_t modifiers = interpolation
                           ? RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                                 RJ_ARITH_ALU_CLAMP | RJ_ARITH_INTERP_WAIT_EXP_MASK
                           : 0x1ff;
  if (interpolation == 16) {
    modifiers |= RJ_ARITH_ALU_HIGH_A | (op & 1 ? RJ_ARITH_ALU_HIGH_D : RJ_ARITH_ALU_HIGH_C);
  }
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 40; ++scenario) {
        for (int d : {0, 1, 2, 3}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << scenario << '/' << d);
          auto actual = input;
          auto exec_mask = exec_mask_for_case<Lanes>(scenario);
          uint64_t mode = interpolation == 16   ? interp16_mode(op, scenario * 7 % 64, scenario % 8)
                          : interpolation == 32 ? interp32_mode(scenario % 16, scenario % 8)
                                                : uint64_t(scenario * 37) & 0x1ff;
          if (!interpolation && scenario >= 33) {
            mode |= dpp_modes[scenario - 33];
          }
          uint64_t guest =
              (!interpolation && scenario % 3 == 0 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
              (scenario % 3 == 1 ? RJ_ARITH_FP16_OVFL : 0);
          uint32_t status = 0x80000000U;
          int error = call(cpu | semantics | guest, exec_mask, mode, actual.v + d, actual.v,
                           actual.v + 1, actual.v + 2, &status);
          int expected_error = RJ_ARITH_SUCCESS;
          if ((lighting || interpolation) &&
              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
            expected_error = RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
          }
          if ((lighting || interpolation) && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            expected_error = lighting ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                                      : RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
          }
          ASSERT_EQ(error, expected_error);
          EXPECT_EQ(status, 0x80000000U);
          if (error) {
            actual.expect_preserved(input, 0, 0);
            continue;
          }
          for (int lane = 0; lane < Lanes; ++lane) {
            int source = lane;
            bool enabled = (mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16))
                               ? dpp_source(mode, exec_mask, lane, source)
                               : bool((exec_mask >> lane) & 1);
            if (!enabled) {
              EXPECT_EQ(actual.v[d][lane], input.v[d][lane]);
              continue;
            }
            uint32_t a = source < 0 ? 0 : input.v[0][source], expected;
            if (interpolation == 16) {
              expected =
                  interp16_reference(op, input.v[0], input.v[1], input.v[2], input.v[d][lane], lane,
                                     scenario * 7 % 64, guest & RJ_ARITH_FP16_OVFL);
            } else if (interpolation == 32) {
              expected = bit_cast<uint32_t>(
                  interp32_reference(op, input.v[0], input.v[1], input.v[2], lane, scenario % 16));
            } else if (lighting) {
              expected = mullit_reference(a, input.v[1][lane], input.v[2][lane], uint32_t(mode),
                                          !gfx12, guest);
            } else {
              expected = cube_reference(op, a, input.v[1][lane], input.v[2][lane], uint32_t(mode));
            }
            // P2 writes one FP16 halfword: compare its numeric value tolerantly but require the
            // other half to remain bit-identical.
            if (interpolation == 16 && (op & 1)) {
              int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
              EXPECT_EQ(actual.v[d][lane] & ~(0xffffU << shift), expected & ~(0xffffU << shift));
              expect_f32(bit_cast<uint32_t>(f16_value(uint16_t(actual.v[d][lane] >> shift))),
                         f16_value(uint16_t(expected >> shift)), RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/0x1p-24f, /*rtol=*/1e-3f);
            } else {
              expect_f32(actual.v[d][lane], bit_cast<float>(expected),
                         !interpolation && !lighting ? semantics : RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/1e-44f, /*rtol=*/2e-7f);
            }
          }
          actual.expect_preserved(input, d, 1);
        }
      }
    }
  }
  auto unchanged = input;
  auto invalid = [&](uint64_t flags, uint64_t mode) {
    uint32_t status = 0x80000000U;
    int result = call(flags, WaveMask<Lanes>(~WaveMask<Lanes>(0)), mode, unchanged.v + 3,
                      unchanged.v, unchanged.v + 1, unchanged.v + 2, &status);
    unchanged.expect_preserved(input, 0, 0);
    EXPECT_EQ(status, 0x80000000U);
    return result;
  };
  if (interpolation) {
    expect_alu_invalid_flags<false>(invalid, modifiers);
  } else {
    expect_alu_invalid_flags<true>(invalid, modifiers);
  }
}

} // namespace rocjitsu::arith::test
