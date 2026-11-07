// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer predicate tests across signedness, widths, and wave sizes.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "compare/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Signed ordering is unsigned ordering after flipping the sign bit. This
// specification-derived reference avoids out-of-range signed conversions.
// Each named test supplies its relation; CMP and CMPX both return their mask.
template <int Width, int Lanes, bool Signed, typename Call>
void check_integer_compare(Call call, bool (*predicate)(uint64_t, uint64_t)) {
  using Mask = WaveMask<Lanes>;
  Registers<4, Lanes> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = random_word(random);
    }
  }
  constexpr uint64_t sign = 1ULL << (Width - 1);
  const uint64_t edges[] = {0, 1, sign - 1, sign, sign + 1, sign | (sign - 1)};
  for (int lane = 0; lane < Lanes; ++lane) {
    for (int half = 0; half < (Width == 16 ? 2 : 1); ++half) {
      if (lane < 12) {
        write_compare_operand(input.v, lane, Width, edges[(lane + half) % 6], half);
        write_compare_operand(input.v + 2, lane, Width, edges[(lane / 2 + half) % 6], half);
      } else if (lane % 4 == 0) {
        write_compare_operand(input.v + 2, lane, Width, compare_operand(input.v, lane, Width, half),
                              half);
      }
    }
  }
  // Equal high words with different low words catch truncated 64-bit compares.
  if constexpr (Width == 64) {
    write_compare_operand(input.v, 15, Width, 0x1234567800000001ULL);
    write_compare_operand(input.v + 2, 15, Width, 0x1234567800000002ULL);
  }
  const uint64_t low_modifiers = Width == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0;
  expect_alu_invalid_flags<Width != 64>(
      [&](uint64_t flags, uint64_t mode) {
        Mask result = ~Mask(0);
        int error = call(flags, ~Mask(0), mode, &result, nullptr, nullptr);
        EXPECT_EQ(result, ~Mask(0));
        return error;
      },
      low_modifiers);

  std::vector<uint64_t> modes = {0};
  if constexpr (Width == 16) {
    modes.insert(modes.end(), {RJ_ARITH_ALU_HIGH_A, RJ_ARITH_ALU_HIGH_B, low_modifiers});
  }
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
        Mask expected = comparison_mask<Lanes>(mode, exec_mask, [&](int source, int lane) {
          uint64_t a = compare_operand(input.v, source, Width, mode & RJ_ARITH_ALU_HIGH_A);
          uint64_t b = compare_operand(input.v + 2, lane, Width, mode & RJ_ARITH_ALU_HIGH_B);
          return predicate(a ^ (Signed ? sign : 0), b ^ (Signed ? sign : 0));
        });
        for (int aliasing_register : {-1, 0, 2}) {
          expect_comparison(input, expected, RJ_ARITH_SUCCESS, aliasing_register,
                            [&](auto &r, Mask *d) {
                              return call(cpu | semantics, exec_mask, mode, d,
                                          exec_mask ? r.v : nullptr, exec_mask ? r.v + 2 : nullptr);
                            });
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
