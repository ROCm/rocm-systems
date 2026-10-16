// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Floating-point exponent extraction and integer result checks.
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
#include <limits>
#include <random>
#include <stdint.h>
#include <type_traits>

#include <vector>

namespace rocjitsu::arith::test {

// Fill active destination elements from the independent arithmetic reference.
template <int Width, int Lanes>
void compute_frexp_exp_expected(const rocjitsu::arith::test::Registers<10, Lanes> &input,
                                uint64_t exec_mask, uint64_t mode, int aliasing_offset,
                                rocjitsu::arith::test::Registers<10, Lanes> &expected_registers) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    const int a_half = Width == 16 && bool(mode & RJ_ARITH_ALU_HIGH_A);
    const int d_half = Width == 16 && bool(mode & RJ_ARITH_ALU_HIGH_D);
    // DPP8 reverses each group; FI controls reads from inactive sources.
    int source_lane = mode & RJ_ARITH_DPP8 ? (lane & ~7) + 7 - (lane & 7) : lane;
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, source_lane, Width, a_half);
    if ((mode & RJ_ARITH_DPP8) && !(mode & RJ_ARITH_DPP_FI) && !((exec_mask >> source_lane) & 1)) {
      a = 0;
    }

    double x =
        rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(a, Width), mode, 0);
    int exponent = 0;
    if (std::isfinite(x) && x != 0) {
      std::frexp(x, &exponent);
    }
    uint64_t expected = Width == 16 ? uint16_t(exponent) : uint32_t(exponent);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane,
                                    Width == 16 ? 16 : 32, expected, d_half);
  }
}

// Check numeric results, inactive lanes, untouched register halves and guards.
template <int Lanes>
void expect_exponent_result(const rocjitsu::arith::test::Registers<10, Lanes> &input,
                            const rocjitsu::arith::test::Registers<10, Lanes> &r,
                            const rocjitsu::arith::test::Registers<10, Lanes> &expected_registers,
                            uint64_t exec_mask, int aliasing_offset) {
  r.expect_preserved(input, aliasing_offset, 1);
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      EXPECT_EQ(r.v[aliasing_offset][lane], input.v[aliasing_offset][lane]);
      continue;
    }
    EXPECT_EQ(r.v[aliasing_offset][lane], expected_registers.v[aliasing_offset][lane]);
  }
}

// Exercise this instruction contract for either wave size.
template <int Width, int Lanes, class Call> void check_frexp_exp(Call call) {
  static_assert(Width == 16 || Width == 32 || Width == 64);
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<Width, Lanes>();

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U;

    int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, &fp_status), result);
    return result;
  });

  std::vector<uint64_t> modes = {0, RJ_ARITH_ALU_NEG_A, RJ_ARITH_ALU_CLAMP};
  if constexpr (Width == 16) {
    modes.push_back(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D);
  }
  if constexpr (Width != 64) {
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT));
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) | RJ_ARITH_DPP_FI |
                    RJ_ARITH_ALU_NEG_A);
  }
  std::vector<int> aliasing_offsets = {6, 0, 2};

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : aliasing_offsets) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, &fp_status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            if (Width == 64 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            EXPECT_EQ(fp_status, 0x80000000U);
            compute_frexp_exp_expected<Width>(input, exec_mask, mode, aliasing_offset,
                                              expected_registers);

            expect_exponent_result(input, r, expected_registers, exec_mask, aliasing_offset);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
