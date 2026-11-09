// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Division fixup, exceptional operands, and FP-status expectations.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/div_fixup_reference.h"
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
void compute_div_fixup_expected(const rocjitsu::arith::test::Registers<10, Lanes> &input,
                                uint64_t exec_mask, uint64_t mode, int aliasing_offset,
                                rocjitsu::arith::test::Registers<10, Lanes> &expected_registers) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    const int a_half = Width == 16 && bool(mode & RJ_ARITH_ALU_HIGH_A);
    const int b_half = Width == 16 && bool(mode & RJ_ARITH_ALU_HIGH_B);
    const int c_half = Width == 16 && bool(mode & RJ_ARITH_ALU_HIGH_C);
    const int d_half = Width == 16 && bool(mode & RJ_ARITH_ALU_HIGH_D);
    // DPP8 reverses each group; FI controls reads from inactive sources.
    int source_lane = mode & RJ_ARITH_DPP8 ? (lane & ~7) + 7 - (lane & 7) : lane;
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, source_lane, Width, a_half);
    if ((mode & RJ_ARITH_DPP8) && !(mode & RJ_ARITH_DPP_FI) && !((exec_mask >> source_lane) & 1)) {
      a = 0;
    }
    uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, Width, b_half);
    uint64_t c = rocjitsu::arith::test::read_fp(input.v + 4, lane, Width, c_half);
    // Existing independent classification/scaling reference.
    uint64_t expected =
        rocjitsu::arith::test::fixup_reference(Width, a, b, c, uint32_t(mode) & 0x1ffU, false);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, Width, expected,
                                    d_half);
  }
}

// Exercise this instruction contract for either wave size.
template <int Width, int Lanes, typename Call, typename Cases>
void check_div_fixup(Call call, bool gfx12, const Cases &status_cases) {
  static_assert(Width == 16 || Width == 32 || Width == 64);
  static_assert(Lanes == 32 || Lanes == 64);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<Width, Lanes>();

  // Check SIMD cause precedence and numeric results against the scalar model.
  // Independent literal status cases below separately constrain that model.
  if (gfx12) {
    std::minstd_rand engine;
    auto word = [&] { return uint32_t(engine() >> 7) ^ (uint32_t(engine() >> 7) << 16); };
    auto cpus = supported_cpu_levels();
    for (int scenario = 0; scenario < 128; ++scenario) {
      Registers values;
      for (int reg = 0; reg < 10; ++reg) {
        for (int lane = 0; lane < Lanes; ++lane) {
          values.v[reg][lane] = word();
        }
      }
      uint64_t mode = word() & (Width == 16 ? 0x1fffU : 0x1ffU);
      uint64_t flags = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL |
                       (scenario & 1 ? RJ_ARITH_FP16_OVFL : 0) |
                       (scenario & 2 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
      Mask exec_mask = Mask(word());
      if constexpr (Lanes == 64) {
        exec_mask |= uint64_t(word()) << 32;
      }
      int destination = scenario % 7;
      auto expected = values;
      uint32_t expected_status = 0x80000000U;
      ASSERT_EQ(call(flags, exec_mask, mode, expected.v + destination, expected.v, expected.v + 2,
                     expected.v + 4, &expected_status),
                RJ_ARITH_SUCCESS);
      for (auto cpu : cpus) {
        auto actual = values;
        uint32_t status = 0x80000000U;
        SCOPED_TRACE(::testing::Message() << scenario << '/' << cpu);
        ASSERT_EQ(call(flags | cpu, exec_mask, mode, actual.v + destination, actual.v, actual.v + 2,
                       actual.v + 4, &status),
                  RJ_ARITH_SUCCESS);
        ASSERT_EQ(status, expected_status);
        for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
          for (int lane = 0; lane < Lanes; ++lane) {
            ASSERT_EQ(actual.v[destination + reg][lane], expected.v[destination + reg][lane]);
          }
        }
        actual.expect_preserved(values, destination, Width == 64 ? 2 : 1);
      }
    }
  }

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;

    int result =
        call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4, &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, nullptr, nullptr, &fp_status), result);
    return result;
  });

  std::vector<uint64_t> modes = {0, RJ_ARITH_ALU_NEG_A, RJ_ARITH_ALU_CLAMP};
  if constexpr (Width == 16) {
    modes.push_back(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D);
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT));
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) | RJ_ARITH_DPP_FI |
                    RJ_ARITH_ALU_NEG_A);
  }
  std::vector<int> aliasing_offsets = {6, 0, 2, 4};
  if constexpr (Width == 64) {
    aliasing_offsets.push_back(1);
  }

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

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4,
                              &fp_status);
            if (!gfx12 && Width == 16 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            bool reporting =
                gfx12 && (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
                          (Width != 16 && semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY));
            if (!reporting || !exec_mask) {
              EXPECT_EQ(fp_status, 0x80000000U);
            } else {
              EXPECT_EQ(fp_status & 0x80000000U, 0x80000000U);
            }
            compute_div_fixup_expected<Width>(input, exec_mask, mode, aliasing_offset,
                                              expected_registers);

            rocjitsu::arith::test::expect_fp_result<Width>(
                input, r, expected_registers, exec_mask, aliasing_offset, mode, semantics,
                /*atol=*/Width == 64 ? 1e-14 : 1e-5, /*rtol=*/Width == 16 ? 5e-4
                                                     : Width == 32        ? 1e-6
                                                                          : 1e-14);
          }
        }
      }
    }
  }
  if (gfx12) {
    for (uint64_t cpu : cpu_levels) {
      for (const auto &test : status_cases) {
        for (uint64_t mode : {uint64_t(0), uint64_t(RJ_ARITH_ALU_CLAMP)}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << mode << '/' << test.a << '/' << test.b << '/' << test.c);
          Registers r = input;
          const int lane = Lanes - 1;
          for (int part = 0; part < 1; ++part) {
            rocjitsu::arith::test::write_fp(r.v, lane, Width, test.a, part);
            rocjitsu::arith::test::write_fp(r.v + 2, lane, Width, test.b, part);
            rocjitsu::arith::test::write_fp(r.v + 4, lane, Width, test.c, part);
          }
          Mask exec_mask = Mask(1) << lane;
          uint32_t fp_status = 0x80000000U;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, &fp_status),
                    RJ_ARITH_SUCCESS);
          uint32_t causes = mode ? 0 : test.causes;
          EXPECT_EQ(fp_status, 0x80000000U | causes);
          // Preserve previously accumulated causes as well as unrelated bits.
          fp_status = 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
          for (int part = 0; part < 1; ++part) {
            rocjitsu::arith::test::write_fp(r.v + 4, lane, Width, test.c, part);
          }
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, &fp_status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
          Registers before = r;
          exec_mask = 0;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, &fp_status),
                    RJ_ARITH_SUCCESS);
          r.expect_preserved(before, 0, 0);
          EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
