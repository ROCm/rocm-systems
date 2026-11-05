// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared integer register fixtures, routing, aliases, and whole-bank comparisons.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Guarded source and destination slots with room for overlapping operands.
// Default-seeded high-bit draws provide arbitrary bits; the first lanes cover
// zero, signed/unsigned limits and alternating bits.
template <int Lanes> Registers<12, Lanes> make_integer_inputs() {
  Registers<12, Lanes> input;
  std::minstd_rand random;
  const uint32_t edges[] = {0,           1,           0xffffffffU, 0x80000000U,
                            0x7fffffffU, 0xffff0000U, 0x0000ffffU, 0x80008000U,
                            0x7fff7fffU, 0xaaaaaaaaU, 0x55555555U, 15,
                            16,          31,          32,          63};
  for (int reg = 0; reg < 12; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      uint32_t bits = random_word(random);
      input.v[reg][lane] = lane < 16 ? edges[(lane + 3 * reg) % 16] : bits;
    }
  }
  return input;
}

// Compare the whole bank, including guards and inactive/untouched halves.
// Print per-word diagnostics only on failure, not a trace for every lane.
template <int Lanes>
void expect_integer_registers(const Registers<12, Lanes> &actual,
                              const Registers<12, Lanes> &expected) {
  if (std::memcmp(actual.words, expected.words, sizeof actual.words) == 0) {
    return;
  }
  for (int reg = 0; reg < 12; ++reg) {
    for (int word = 0; word < Lanes + 2; ++word) {
      EXPECT_EQ(actual.words[reg][word], expected.words[reg][word]) << reg << '/' << word;
    }
  }
}

// Enumerate individual and combined integer modifiers.
inline std::vector<uint64_t> integer_modes(uint64_t low_modifiers) {
  std::vector<uint64_t> modes{0};
  for (int bit = 0; bit < 32; ++bit) {
    if ((low_modifiers >> bit) & 1) {
      modes.push_back(1ULL << bit);
    }
  }
  if (low_modifiers) {
    modes.push_back(low_modifiers);
  }
  return modes;
}

// Each reference computes one complete destination word, including the old D
// half when relevant. Sample modes independently from masks, not a full product.
template <int Lanes, typename Call, typename Reference>
void check_integer_word(Call call, Reference reference, uint64_t low_modifiers,
                        const Registers<12, Lanes> &input = make_integer_inputs<Lanes>()) {
  using Mask = WaveMask<Lanes>;
  expect_alu_invalid_flags<false>(
      [&](uint64_t flags, uint64_t mode) {
        auto r = input;
        int error = call(flags, ~Mask(0), mode, r.v + 8, r.v, r.v + 2, r.v + 4);
        expect_integer_registers(r, input);
        EXPECT_EQ(call(flags, ~Mask(0), mode, nullptr, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return error;
      },
      low_modifiers);
  const auto modes = integer_modes(low_modifiers);
  const auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 5 + int(modes.size()); ++scenario) {
        uint64_t mode = scenario < 5 ? 0 : modes[scenario - 5];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario < 5 ? scenario : scenario % 2 ? 0 : 3);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        // Compute expectations from the original bank, so an aliased write cannot change a later
        // reference operand.
        for (int aliasing_offset : {8, 0, 2, 4}) {
          auto r = input, expected = input;
          for (int lane = 0; lane < Lanes; ++lane) {
            if (!((exec_mask >> lane) & 1)) {
              continue;
            }
            uint32_t a = input.v[0][lane];
            uint32_t b = input.v[2][lane];
            expected.v[aliasing_offset][lane] = reference(
                a, b, input.v[4][lane], input.v[aliasing_offset][lane], uint32_t(mode), lane);
          }
          uint64_t flags = cpu | semantics | (scenario % 2 ? RJ_ARITH_FP16_OVFL : 0);
          ASSERT_EQ(call(flags, exec_mask, mode, exec_mask ? r.v + aliasing_offset : nullptr,
                         exec_mask ? r.v : nullptr, exec_mask ? r.v + 2 : nullptr,
                         exec_mask ? r.v + 4 : nullptr),
                    RJ_ARITH_SUCCESS);
          expect_integer_registers(r, expected);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
