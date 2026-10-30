// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Register-operation scenarios, routing references, and preservation checks.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Raw-bit operations have specification-derived references, including NaN payloads.
// Reference and invocation see separate register banks so in-place routing always
// reads the original wave. Scenarios pair masks with selectors rather than taking
// an exhaustive Cartesian product.
template <int Lanes, bool Dpp = false, typename Call, typename Reference>
void check_register(Call call, Reference reference, uint64_t modifiers = 0) {
  Registers<6, Lanes> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 6; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = random_word(random);
    }
  }
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 40; ++scenario) {
        for (int aliasing_offset : {0, 1, 4}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << scenario << '/' << aliasing_offset);
          auto actual = input, expected = input;
          auto exec_mask = exec_mask_for_case<Lanes>(scenario);
          uint64_t mode = uint64_t(scenario) & modifiers;
          // Also exercise higher modifier bits and each DPP routing class.
          if (scenario >= 8) {
            mode = (modifiers & (1ULL << (scenario - 8)));
          }
          if constexpr (Dpp) {
            if (scenario >= 32) {
              mode |= scenario == 32 ? 0 : dpp_modes[scenario - 33];
            }
          }
          reference(expected, input, exec_mask, mode, aliasing_offset, scenario);
          ASSERT_EQ(call(cpu | semantics, exec_mask, mode, actual, aliasing_offset, scenario),
                    RJ_ARITH_SUCCESS);
          for (int reg = 0; reg < 6; ++reg) {
            for (int lane = 0; lane < Lanes; ++lane) {
              ASSERT_EQ(actual.v[reg][lane], expected.v[reg][lane])
                  << "register=" << reg << ", lane=" << lane;
            }
          }
          // Comparing the complete padded storage checks both guards as well.
          EXPECT_EQ(0, std::memcmp(actual.words, expected.words, sizeof actual.words));
        }
      }
    }
  }
  auto unchanged = input;
  expect_alu_invalid_flags<Dpp>(
      [&](uint64_t flags, uint64_t mode) {
        int result = call(flags, WaveMask<Lanes>(~WaveMask<Lanes>(0)), mode, unchanged, 4, 0);
        unchanged.expect_preserved(input, 0, 0);
        return result;
      },
      modifiers);
}

// Return the moved word or selected halfword, preserving untouched destination bits.
inline uint32_t moved_word(uint32_t a, uint32_t old, uint64_t mode, bool half) {
  if (!half) {
    if (mode & RJ_ARITH_ALU_ABS_A) {
      a &= 0x7fffffffU;
    }
    return a ^ (mode & RJ_ARITH_ALU_NEG_A ? 0x80000000U : 0);
  }
  uint32_t value = (a >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) & 0xffffU;
  if (mode & RJ_ARITH_ALU_ABS_A) {
    value &= 0x7fffU;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    value ^= 0x8000U;
  }
  int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  return (old & ~(0xffffU << shift)) | (value << shift);
}

} // namespace rocjitsu::arith::test
