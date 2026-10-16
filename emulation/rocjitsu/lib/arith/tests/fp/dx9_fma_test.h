// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DX9 zero-product FMA/FMAC behavior.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/fma_simd_test.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

// Fill active destination elements from the independent arithmetic reference.
template <int Lanes, bool Accumulate>
void compute_dx9_fma_expected(const rocjitsu::arith::test::Registers<10, Lanes> &input,
                              uint64_t exec_mask, uint64_t mode, int aliasing_offset,
                              rocjitsu::arith::test::Registers<10, Lanes> &expected_registers) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    // DPP8 reverses each group; FI controls reads from inactive sources.
    int source_lane = mode & RJ_ARITH_DPP8 ? (lane & ~7) + 7 - (lane & 7) : lane;
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, source_lane, 32, 0);
    if ((mode & RJ_ARITH_DPP8) && !(mode & RJ_ARITH_DPP_FI) && !((exec_mask >> source_lane) & 1)) {
      a = 0;
    }
    uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, 32, 0);
    uint64_t c =
        rocjitsu::arith::test::read_fp(input.v + (Accumulate ? aliasing_offset : 4), lane, 32, 0);

    double x = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(a, 32), mode, 0);
    double y = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(b, 32), mode, 1);
    double z = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(c, 32), mode, 2);
    double value = double(std::fma(float(x), float(y), float(z)));
    if (x == 0 || y == 0) {
      value = z;
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = !(value > 0) ? 0 : std::min(value, 1.0);
    }
    uint64_t expected = rocjitsu::arith::test::fp_bits(value, 32);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, 32, expected, 0);
  }
}

// Exercise this instruction contract for either wave size.
template <int Lanes, bool Accumulate, typename Call> void check_dx9_fma(Call entry_point) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  static_assert(Lanes == 32 || Lanes == 64);
  auto call = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
    if constexpr (Accumulate) {
      return entry_point(flags, exec_mask, mode, d, a, b);
    } else {
      return entry_point(flags, exec_mask, mode, d, a, b, c);
    }
  };
  check_fma_simd<32, Lanes>(
      [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
          uint32_t *) { return call(flags, exec_mask, mode, d, a, b, c); },
      {0, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_ALU_OMOD_HALF, RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP});
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  // (1 + 2^-23)^2 - 1 is just above 2^-22; only rounding up advances a bit.
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t rounding :
         {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      Registers r;
      r.v[0][0] = r.v[2][0] = 0x3f800001U;
      r.v[4][0] = r.v[6][0] = 0xbf800000U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | rounding, Mask(1),
                     0, r.v + 6, r.v, r.v + 2, r.v + 4),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[6][0], rounding == RJ_ARITH_FP_ROUND_UP ? 0x34800001U : 0x34800000U);
    }
  }
  Registers input = rocjitsu::arith::test::make_fp_inputs<32, Lanes>();

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);

    int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, nullptr, nullptr), result);
    return result;
  });

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : {uint64_t(0), uint64_t(RJ_ARITH_ALU_NEG_A), uint64_t(RJ_ARITH_ALU_CLAMP),
                            uint64_t(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT)),
                            uint64_t(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) |
                                     RJ_ARITH_DPP_FI | RJ_ARITH_ALU_NEG_A)}) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : {6, 0, 2, 4}) {
            if (Accumulate && aliasing_offset == 4) {
              continue;
            }
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4);
            // Unlike FMA, the DX9 FMAC form does not accept DPP encodings.
            if (Accumulate && (mode & RJ_ARITH_DPP8)) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_INVALID_FLAGS);
              r.expect_preserved(input, 0, 0);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            compute_dx9_fma_expected<Lanes, Accumulate>(input, exec_mask, mode, aliasing_offset,
                                                        expected_registers);

            rocjitsu::arith::test::expect_fp_result<32>(input, r, expected_registers, exec_mask,
                                                        aliasing_offset, 0, semantics,
                                                        /*atol=*/1e-5, /*rtol=*/1e-6);
          }
        }
      }
    }
  }
  // DX9 turns a zero product into the addend, even for Inf, and flushes
  // subnormal factors before that decision. Ordinary IEEE FMA would yield NaN.
  struct ZeroProduct {
    uint32_t a, b, c;
  };
  const ZeroProduct cases[] = {{0, 0x7f800000U, 0x3f800000U},
                               {0x7f800000U, 0x80000000U, 0xc0000000U},
                               {1, 0x7f800000U, 0x40000000U}};
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (const auto &test : cases) {
        Registers r = input;
        constexpr int lane = Lanes - 1;
        r.v[0][lane] = test.a;
        r.v[2][lane] = test.b;
        r.v[Accumulate ? 6 : 4][lane] = test.c;
        Registers before = r;
        ASSERT_EQ(call(cpu | semantics, Mask(1) << lane, 0, r.v + 6, r.v, r.v + 2, r.v + 4),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[6][lane], test.c);
        r.expect_preserved(before, 6, 1);
        for (int inactive = 0; inactive < lane; ++inactive) {
          EXPECT_EQ(r.v[6][inactive], before.v[6][inactive]);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
