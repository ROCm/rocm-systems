// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 FMA modifiers, aliasing, and guest FP-status checks.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/fma_reference.h"
#include "fp/fma_simd_test.h"
#include "fp/half_fma_reference.h"
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

// Exercise this instruction contract for either wave size.
template <int Lanes, typename Call, typename Cases>
void check_half_fma(Call entry_point, bool gfx12, const Cases &status_cases) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  static_assert(Lanes == 32 || Lanes == 64);
  const uint64_t clamp = RJ_ARITH_ALU_CLAMP;
  auto call = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                  uint32_t *fp_status) {
    return entry_point(flags, exec_mask, mode, d, a, b, c, fp_status);
  };
  check_fma_simd<16, Lanes>(call, {0, RJ_ARITH_ALU_NEG_A, RJ_ARITH_ALU_CLAMP, RJ_ARITH_ALU_OMOD_2,
                                   RJ_ARITH_ALU_OMOD_4, RJ_ARITH_ALU_OMOD_HALF,
                                   RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D});
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<16, Lanes>();

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

  std::vector<uint64_t> modes = {0, RJ_ARITH_ALU_NEG_A, clamp};

  modes.push_back(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D);
  modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT));
  modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) | RJ_ARITH_DPP_FI |
                  RJ_ARITH_ALU_NEG_A);

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : {6, 0, 2, 4}) {

            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4,
                              &fp_status);
            if (!gfx12 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            bool reporting = gfx12 && (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
            if (!reporting || !exec_mask) {
              EXPECT_EQ(fp_status, 0x80000000U);
            } else {
              EXPECT_EQ(fp_status & 0x80000000U, 0x80000000U);
            }

            compute_fma_expected<16>(input, exec_mask, mode, aliasing_offset, expected_registers,
                                     gfx12);

            // The independent half oracle specifies NaN encodings as well as finite values.
            if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
              int half = bool(mode & RJ_ARITH_ALU_HIGH_D);
              for (int lane = 0; lane < Lanes; ++lane) {
                if ((exec_mask >> lane) & 1) {
                  EXPECT_EQ(read_fp(r.v + aliasing_offset, lane, 16, half),
                            read_fp(expected_registers.v + aliasing_offset, lane, 16, half));
                }
              }
            }
            expect_fp_result<16>(input, r, expected_registers, exec_mask, aliasing_offset, mode,
                                 semantics, /*atol=*/1e-5, /*rtol=*/5e-4);
          }
        }
      }
    }
  }
  if (gfx12) {
    for (uint64_t cpu : cpu_levels) {
      for (const auto &test : status_cases) {
        for (uint64_t mode : {uint64_t(0), clamp}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << mode << '/' << test.a << '/' << test.b << '/' << test.c);
          Registers r = input;
          const int lane = Lanes - 1;
          rocjitsu::arith::test::write_fp(r.v, lane, 16, test.a);
          rocjitsu::arith::test::write_fp(r.v + 2, lane, 16, test.b);
          rocjitsu::arith::test::write_fp(r.v + 4, lane, 16, test.c);

          Mask exec_mask = Mask(1) << lane;
          uint32_t fp_status = 0x80000000U;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, &fp_status),
                    RJ_ARITH_SUCCESS);
          uint32_t causes = mode ? 0 : test.causes;
          EXPECT_EQ(fp_status, 0x80000000U | causes);
          // Preserve previously accumulated causes as well as unrelated bits.
          fp_status = 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
          rocjitsu::arith::test::write_fp(r.v + 4, lane, 16, test.c);

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
