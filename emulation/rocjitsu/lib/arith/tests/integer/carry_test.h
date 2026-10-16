// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer arithmetic with carry/borrow masks.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "integer/carry_reference.h"
#include "integer/word_test.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases for the supplied entry point.
template <int Lanes, bool WithInputCarry, class Call> void check_carry(Call call, Carry operation) {
  using Mask = WaveMask<Lanes>;
  auto input = make_integer_inputs<Lanes>();
  // Include full-width carry/borrow and carry-in wrapping B from UINT32_MAX.
  input.v[0][0] = 0;
  input.v[2][0] = UINT32_MAX;
  input.v[0][1] = UINT32_MAX;
  input.v[2][1] = 1;
  const Mask input_carry = Mask(0xa5a5a5a5a5a5a5a5ULL);
  auto invoke = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, Mask *carry,
                    const uint32_t *const *a, const uint32_t *const *b) {
    if constexpr (WithInputCarry) {
      return call(flags, exec_mask, mode, d, carry, a, b, input_carry);
    } else {
      return call(flags, exec_mask, mode, d, carry, a, b);
    }
  };
  expect_alu_invalid_flags<true>(
      [&](uint64_t flags, uint64_t mode) {
        auto r = input;
        Mask carry = ~Mask(0);
        int error = invoke(flags, ~Mask(0), mode, r.v + 8, &carry, r.v, r.v + 2);
        expect_integer_registers(r, input);
        EXPECT_EQ(carry, ~Mask(0));
        EXPECT_EQ(invoke(flags, ~Mask(0), mode, nullptr, &carry, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        EXPECT_EQ(carry, ~Mask(0));
        return error;
      },
      RJ_ARITH_ALU_CLAMP);
  const bool reverse = operation == Carry::Subrev || operation == Carry::SubrevCi;
  const auto modes = integer_modes(RJ_ARITH_ALU_CLAMP, true);
  const auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 5 + int(modes.size()); ++scenario) {
        uint64_t mode = scenario < 5 ? 0 : modes[scenario - 5];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario < 5 ? scenario : scenario % 2 ? 0 : 3);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        // Disjoint, D=A, D=B, carry=A, and carry=D. Scalar stores occur last.
        for (int alias = 0; alias < 5; ++alias) {
          int offset = alias == 1 || alias == 4 ? 0 : alias == 2 ? 2 : 8;
          auto r = input, expected = input;
          Mask carry = ~Mask(0), want_carry = 0;
          for (int lane = 0; lane < Lanes; ++lane) {
            int source;
            if (!integer_source(mode, exec_mask, lane, source)) {
              continue;
            }
            uint32_t a = reverse ? input.v[0][lane] : source < 0 ? 0 : input.v[0][source];
            uint32_t b = !reverse ? input.v[2][lane] : source < 0 ? 0 : input.v[2][source];
            auto result = carry_reference(operation, a, b, (input_carry >> lane) & 1,
                                          mode & RJ_ARITH_ALU_CLAMP);
            expected.v[offset][lane] = result.value;
            if (result.carry) {
              want_carry |= Mask(1) << lane;
            }
          }
          void *storage =
              alias >= 3 ? static_cast<void *>(r.v[0] + 1) : static_cast<void *>(&carry);
          if (alias >= 3) {
            std::memcpy(expected.v[0] + 1, &want_carry, sizeof want_carry);
          }
          ASSERT_EQ(invoke(cpu | semantics, exec_mask, mode, exec_mask ? r.v + offset : nullptr,
                           static_cast<Mask *>(storage), exec_mask ? r.v : nullptr,
                           exec_mask ? r.v + 2 : nullptr),
                    RJ_ARITH_SUCCESS);
          std::memcpy(&carry, storage, sizeof carry);
          EXPECT_EQ(carry, want_carry);
          expect_integer_registers(r, expected);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
