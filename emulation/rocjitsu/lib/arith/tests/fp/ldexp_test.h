// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Power-of-two scaling with guest rounding and denormal controls.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>

#include <vector>

namespace rocjitsu::arith::test {

// Continue the FP input stream to supply bounded signed integer exponents.
template <int Width, int Lanes> rocjitsu::arith::test::Registers<10, Lanes> make_ldexp_inputs() {
  std::minstd_rand random;
  auto input = rocjitsu::arith::test::make_fp_inputs<Width, Lanes>(random);
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[2][lane] = uint32_t(int(random() >> 25) % 9 - 4);
  }
  return input;
}

// Fill active destination elements from the independent arithmetic reference.
template <int Width, int Lanes>
void compute_ldexp_expected(const rocjitsu::arith::test::Registers<10, Lanes> &input,
                            uint64_t exec_mask, uint64_t instruction_flags, int aliasing_offset,
                            rocjitsu::arith::test::Registers<10, Lanes> &expected_registers) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    const int a_half = Width == 16 && bool(instruction_flags & RJ_ARITH_ALU_HIGH_A);
    const int b_half = Width == 16 && bool(instruction_flags & RJ_ARITH_ALU_HIGH_B);
    const int d_half = Width == 16 && bool(instruction_flags & RJ_ARITH_ALU_HIGH_D);
    // DPP8 reverses each group; FI controls reads from inactive sources.
    int source_lane = instruction_flags & RJ_ARITH_DPP8 ? (lane & ~7) + 7 - (lane & 7) : lane;
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, source_lane, Width, a_half);
    if ((instruction_flags & RJ_ARITH_DPP8) && !(instruction_flags & RJ_ARITH_DPP_FI) &&
        !((exec_mask >> source_lane) & 1)) {
      a = 0;
    }
    uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, Width, b_half);

    double x = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(a, Width),
                                                  instruction_flags, 0);
    double value = std::ldexp(x, Width == 16 ? int16_t(b) : int32_t(b));
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      value = !(value > 0) ? 0 : std::min(value, 1.0);
    }
    uint64_t expected = rocjitsu::arith::test::fp_bits(value, Width);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, Width, expected,
                                    d_half);
  }
}

// Exercise this instruction contract for either wave size.
template <int Width, int Lanes, typename Call> void check_ldexp(Call call) {
  static_assert(Width == 16 || Width == 32 || Width == 64);
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = make_ldexp_inputs<Width, Lanes>();

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags,
                                                        uint64_t instruction_flags) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;

    int result =
        call(flags, exec_mask, instruction_flags, r.v + aliasing_offset, r.v, r.v + 2, &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0);
    EXPECT_EQ(call(flags, exec_mask, instruction_flags, nullptr, nullptr, nullptr, &fp_status),
              result);
    return result;
  });

  std::vector<uint64_t> instruction_flag_cases = {0, RJ_ARITH_ALU_NEG_A, RJ_ARITH_ALU_CLAMP};
  if constexpr (Width == 16) {
    instruction_flag_cases.push_back(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D);
  }
  if constexpr (Width != 64) {
    instruction_flag_cases.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT));
    instruction_flag_cases.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) |
                                     RJ_ARITH_DPP_FI | RJ_ARITH_ALU_NEG_A);
  }
  std::vector<int> aliasing_offsets = {6, 0, 2};
  if constexpr (Width == 64) {
    aliasing_offsets.push_back(1);
  }

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t instruction_flags : instruction_flag_cases) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : aliasing_offsets) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << instruction_flags
                                              << '/' << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, instruction_flags, r.v + aliasing_offset, r.v,
                              r.v + 2, &fp_status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            EXPECT_EQ(fp_status, 0x80000000U);
            compute_ldexp_expected<Width>(input, exec_mask, instruction_flags, aliasing_offset,
                                          expected_registers);

            rocjitsu::arith::test::expect_fp_result<Width>(
                input, r, expected_registers, exec_mask, aliasing_offset, instruction_flags,
                semantics,
                /*atol=*/Width == 64 ? 1e-14 : 1e-5, /*rtol=*/Width == 16 ? 5e-4
                                                     : Width == 32        ? 1e-6
                                                                          : 1e-14);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
