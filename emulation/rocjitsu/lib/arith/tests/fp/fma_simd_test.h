// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar/SIMD agreement for exact FMA, including exceptional lanes.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>
#include <vector>

namespace rocjitsu::arith::test {

// Differential SIMD regression coverage, supplementing the independent arithmetic
// references and status fixtures. This checks optimization against the inherited
// scalar model, not the model's agreement with hardware.
template <int Width, int Lanes, typename Call>
void check_fma_simd(Call call, const std::vector<uint64_t> &modes) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  const auto cpus = supported_cpu_levels();
  if constexpr (Width == 16) {
    // Directed rounding can promote a tiny half result before output flushing.
    for (uint64_t cpu : cpus) {
      for (bool negative : {false, true}) {
        Registers<10, Lanes> r{};
        r.v[0][0] = 0x0800;
        r.v[2][0] = negative ? 0x0ab1 : 0x8ab1;
        r.v[4][0] = negative ? 0x8400 : 0x0400;
        uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
                         RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                         (negative ? RJ_ARITH_FP_ROUND_DOWN : RJ_ARITH_FP_ROUND_UP);
        ASSERT_EQ(call(flags, Mask(1), 0, r.v + 6, r.v, r.v + 2, r.v + 4, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[6][0] & 0xffffU, negative ? 0x8400U : 0x0400U);
      }
    }
  }
  std::minstd_rand rng;
  for (int sample = 0; sample < (Width == 16 ? 16 : 12); ++sample) {
    Registers<10, Lanes> input;
    for (int reg = 0; reg < 10; ++reg) {
      for (int lane = 0; lane < Lanes; ++lane) {
        // Use high engine bits for both halves. Alternate arbitrary encodings
        // with ordinary fractional inputs and one exceptional lane per operand.
        uint32_t bits = (uint32_t(rng() >> 15) << 16) | uint32_t(rng() >> 15);
        if (sample % 2 == 0) {
          bits =
              Width == 32 ? (bits & 0x807fffffU) | 0x3f000000U : (bits & 0x83ff83ffU) | 0x38003800U;
        }
        input.v[reg][lane] = bits;
      }
    }
    static const uint32_t special32[] = {0x7fc12345U, 0x7f800000U, 1U,
                                         0x80000000U, 0x00800000U, 0x7f7fffffU};
    static const uint32_t special16[] = {0x7e017e01U, 0x7c007c00U, 0x00010001U,
                                         0x80008000U, 0x04000400U, 0x7bff7bffU};
    for (int reg : {0, 2, 4, 6}) {
      input.v[reg][(sample + reg * 7) % Lanes] =
          Width == 32 ? special32[(sample / 2) % 6] : special16[(sample / 2) % 6];
    }
    if (Width == 16 && sample >= 12) {
      // Compensated AVX2 half FMA: tiny addends near a rounding boundary,
      // exact cancellation, and signed zero must match the inherited model.
      for (int lane = 0; lane < Lanes; ++lane) {
        input.v[0][lane] = sample < 14 ? 0x3c013c01U : 0x3c003c00U;
        input.v[2][lane] = sample < 14 ? 0x3bff3bffU : 0x3c003c00U;
        input.v[4][lane] = sample < 14 ? (lane & 1 ? 0x80018001U : 0x00010001U) : 0xbc00bc00U;
        input.v[6][lane] = input.v[4][lane];
      }
    }
    for (uint64_t mode : modes) {
      for (uint64_t fp :
           {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO,
            RJ_ARITH_FP_ROUND_UP | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
            RJ_ARITH_FP_ROUND_DOWN | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
            RJ_ARITH_FP16_OVFL,
            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
        for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                   RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
          int aliasing_offset = (sample % 4) * 2;
          Mask exec_mask = sample % 3 ? ~Mask(0) : Mask(0xaaaaaaaaaaaaaaaaULL);
          Registers<10, Lanes> expected = input;
          uint32_t expected_status = 0x80000000U;
          int error = call(semantics | fp, exec_mask, mode, expected.v + aliasing_offset,
                           expected.v, expected.v + 2, expected.v + 4, &expected_status);
          for (uint64_t cpu : cpus) {
            if (cpu == RJ_ARITH_CPU_BASELINE) {
              continue;
            }
            SCOPED_TRACE(::testing::Message()
                         << sample << '/' << mode << '/' << fp << '/' << semantics << '/' << cpu);
            Registers<10, Lanes> actual = input;
            uint32_t status = 0x80000000U;
            ASSERT_EQ(call(cpu | semantics | fp, exec_mask, mode, actual.v + aliasing_offset,
                           actual.v, actual.v + 2, actual.v + 4, &status),
                      error);
            EXPECT_EQ(status, expected_status);
            for (int reg = 0; reg < 10; ++reg) {
              for (int word = 0; word < Lanes + 2; ++word) {
                ASSERT_EQ(actual.words[reg][word], expected.words[reg][word])
                    << "register=" << reg << ", word=" << word;
              }
            }
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
