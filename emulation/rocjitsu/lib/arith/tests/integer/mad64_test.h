// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wide integer multiply-add and carry results.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "integer/mad64_reference.h"
#include "integer/word_test.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check 64-bit multiply-add, carry masks, and overlapping destination words for the supplied
// entry point.
template <int Lanes, typename Call> void check_mad64(Call call, bool is_signed) {
  using Mask = WaveMask<Lanes>;
  auto input = make_integer_inputs<Lanes>();
  // Old suite's directed cases distinguish extended sign from signed overflow,
  // and positive/negative saturation. Remaining lanes retain pseudorandom data.
  const uint32_t a[] = {0, 0x80000000U, 0x80000000U, 1, UINT32_MAX};
  const uint32_t b[] = {0, 0x80000000U, 0x7fffffffU, 1, UINT32_MAX};
  const uint64_t c[] = {UINT64_MAX, UINT64_MAX >> 1, 1ULL << 63, 0, UINT64_MAX};
  for (int lane = 0; lane < 5; ++lane) {
    input.v[0][lane] = a[lane];
    input.v[2][lane] = b[lane];
    input.v[4][lane] = uint32_t(c[lane]);
    input.v[5][lane] = uint32_t(c[lane] >> 32);
  }
  expect_alu_invalid_flags<false>(
      [&](uint64_t flags, uint64_t mode) {
        auto r = input;
        Mask carry = ~Mask(0);
        int error = call(flags, ~Mask(0), mode, r.v + 8, &carry, r.v, r.v + 2, r.v + 4);
        expect_integer_registers(r, input);
        EXPECT_EQ(carry, ~Mask(0));
        EXPECT_EQ(call(flags, ~Mask(0), mode, nullptr, &carry, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        EXPECT_EQ(carry, ~Mask(0));
        return error;
      },
      RJ_ARITH_ALU_CLAMP);
  const auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 7; ++scenario) {
        uint64_t mode = scenario >= 5 ? RJ_ARITH_ALU_CLAMP : 0;
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario < 5 ? scenario : scenario % 2 ? 0 : 3);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        // Sources overlap D; carry overlaps A/D; finally D0 and D1 are the
        // same register, where the high-word store must win. Carry is last.
        for (int alias = 0; alias < 7; ++alias) {
          int offset = alias == 1 || alias == 5 || alias == 6 ? 0
                       : alias == 2                           ? 2
                       : alias == 3                           ? 4
                                                              : 8;
          bool repeated = alias == 6;
          auto r = input, expected = input;
          Mask carry = ~Mask(0), want_carry = 0;
          for (int lane = 0; lane < Lanes; ++lane) {
            if (!((exec_mask >> lane) & 1)) {
              continue;
            }
            auto result = mad64_reference(is_signed, input.v[0][lane], input.v[2][lane],
                                          input.v[4][lane] | (uint64_t(input.v[5][lane]) << 32),
                                          mode & RJ_ARITH_ALU_CLAMP);
            expected.v[offset][lane] = uint32_t(result.value);
            expected.v[offset + !repeated][lane] = uint32_t(result.value >> 32);
            if (result.carry) {
              want_carry |= Mask(1) << lane;
            }
          }
          bool carry_alias = alias == 4 || alias == 5;
          void *storage =
              carry_alias ? static_cast<void *>(r.v[0] + 1) : static_cast<void *>(&carry);
          if (carry_alias) {
            std::memcpy(expected.v[0] + 1, &want_carry, sizeof want_carry);
          }
          uint32_t *dest[] = {r.v[offset], r.v[offset + !repeated]};
          ASSERT_EQ(call(cpu | semantics, exec_mask, mode, exec_mask ? dest : nullptr,
                         static_cast<Mask *>(storage), exec_mask ? r.v : nullptr,
                         exec_mask ? r.v + 2 : nullptr, exec_mask ? r.v + 4 : nullptr),
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
