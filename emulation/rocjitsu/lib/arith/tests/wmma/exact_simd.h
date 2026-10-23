// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar/SIMD agreement checks for exact matrix arithmetic.

#pragma once

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "rj_arith/rj_arith.h"
#include <bit>

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Differential coverage supplements the per-instruction rocjitsu goldens.
// The scalar path retains the inherited integer model unchanged. Compare raw
// bits (including NaN payloads), all source modifiers and preserved registers.
// BF16 also exercises loose SIMD against scalar arithmetic with fuzzy compares.
template <bool Bf16, int Lanes, bool Rdna3, typename Call> void expect_wmma_exact_simd(Call call) {
  const auto levels = supported_cpu_levels();
  if (levels.size() < 2) {
    return;
  }
  std::minstd_rand random;
  constexpr uint32_t infinity = Bf16 ? 0x7f80 : 0x7c00;
  constexpr uint32_t fraction = Bf16 ? 0x7f : 0x3ff;
  constexpr uint32_t one = Bf16 ? 0x3f80 : 0x3c00;
  constexpr int dest_regs = 256 / Lanes;
  for (int sample = 0; sample < 256; ++sample) {
    Registers<32, Lanes> input;
    int kind = (sample / 8) % 8;
    for (int reg = 0; reg < 32; ++reg) {
      for (int lane = 0; lane < Lanes; ++lane) {
        uint32_t raw = (uint32_t(random()) >> 15) | ((uint32_t(random()) >> 15) << 16);
        uint32_t word = 0;
        for (int shift : {0, 16}) {
          uint32_t half = (raw >> shift) & 0xffff;
          uint32_t magnitude = half & 0x7fff;
          if (kind == 0 || kind == 5) {
            magnitude = one | (magnitude & fraction);
          }
          if (kind == 1) {
            magnitude %= infinity; // Entire finite exponent range.
          }
          if (kind == 3) {
            magnitude &= fraction; // Subnormals, including signed zero.
          }
          if (kind == 4) {
            magnitude = 0;
          }
          if (kind == 6) {
            magnitude = one; // Opposing equal products exercise cancellation.
          }
          if (kind == 7) {
            magnitude = (lane % 2) ? infinity - 1 : fraction + 1;
          }
          // Neighbors of exponent boundaries stress normalization and carry
          // in the AVX2 packing, beyond arbitrary raw encodings.
          if (sample >= 128) {
            magnitude = (1 + (reg + lane) % (Bf16 ? 254 : 30)) * (fraction + 1) +
                        (sample + reg + lane) % 3 - 1;
          }
          word |= ((half & 0x8000) | magnitude) << shift;
        }
        if (reg >= 16) {
          word = raw;
          if (kind == 0 || kind == 5) {
            word = (raw & 0x807fffffU) | 0x3f800000U;
          }
          if (kind == 1) {
            word = (raw & 0x80000000U) | ((raw & 0x7fffffffU) % 0x7f800000U);
          }
          if (kind == 3) {
            word &= 0x807fffffU;
          }
          if (kind == 4) {
            word &= 0x80000000U;
          }
          if (kind == 6) {
            word = 0;
          }
          if (kind == 7) {
            word = lane % 2 ? 0x7f7fffffU : 0x00800000U;
          }
        }
        if (sample >= 128 && reg >= 16) {
          word = (raw & 0x80000000U) |
                 ((uint32_t(1 + (sample + reg + lane) % 254) << 23) + (sample + lane) % 3 - 1);
        }
        input.v[reg][lane] = word;
      }
    }
    if (kind == 5) {
      // Isolated special columns alongside ordinary finite columns, with both
      // signaling and quiet NaNs, infinity, and diagnostic payloads.
      input.v[8][sample % 16] = infinity | (infinity << 16);
      input.v[9][(sample + 3) % 16] = (infinity + 1) | ((infinity + fraction) << 16);
      input.v[16][(sample + 5) % 16] = 0x7f800001U;
    }
    // gfx11 requires replicated source fragments; enforce that layout before comparing backends.
    if constexpr (Rdna3) {
      for (int reg = 0; reg < 16; ++reg) {
        for (int lane = 16; lane < Lanes; ++lane) {
          input.v[reg][lane] = input.v[reg][lane % 16];
        }
      }
    }

    uint64_t mode = sample % 64;

    uint64_t flags = sample % 2 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                                : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
    if ((sample / 64) % 2) {
      flags |= RJ_ARITH_FP16_OVFL;
    }
    if (sample % 3 == 0) {
      flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
               RJ_ARITH_FP_IEEE_MODE;
    }
    const int aliases[] = {0, 4, 8, 16, 24};
    int destination = aliases[sample % 5];
    // Reuse the same broad inputs, modifiers and aliases for loose FP16/BF16.
    // Include exceptional columns and the full exponent range in SIMD comparisons.
    for (bool loose : {false, true}) {
      uint64_t selected_flags = loose ? (flags & ~RJ_ARITH_SEMANTICS_MASK) : flags;
      auto expected = input;
      ASSERT_EQ(call(selected_flags, mode, expected.v + destination, expected.v, expected.v + 8,
                     expected.v + 16),
                RJ_ARITH_SUCCESS);
      for (uint64_t cpu : levels) {
        if (cpu == RJ_ARITH_CPU_BASELINE) {
          continue;
        }
        SCOPED_TRACE(::testing::Message() << loose << '/' << sample << '/' << cpu << '/' << mode);
        auto actual = input;
        ASSERT_EQ(call(selected_flags | cpu, mode, actual.v + destination, actual.v, actual.v + 8,
                       actual.v + 16),
                  RJ_ARITH_SUCCESS);
        for (int reg = destination; reg < destination + dest_regs; ++reg) {
          for (int lane = 0; lane < Lanes; ++lane) {
            if (!loose) {
              ASSERT_EQ(actual.v[reg][lane], expected.v[reg][lane]) << reg << '/' << lane;
            } else {
              uint32_t av = actual.v[reg][lane], ev = expected.v[reg][lane];

              expect_f32(av, std::bit_cast<float>(ev), RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/1e-36, /*rtol=*/1e-6);
            }
          }
        }
        actual.expect_preserved(input, destination, dest_regs);
      }
    }
  }
}

} // namespace rocjitsu::arith::test
