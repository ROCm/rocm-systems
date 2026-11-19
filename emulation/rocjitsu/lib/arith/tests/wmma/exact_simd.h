// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar/SIMD agreement checks for exact matrix arithmetic.

#pragma once

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include <bit>

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Differential coverage supplements the per-instruction rocjitsu goldens.
// The scalar path retains the inherited integer model unchanged. Compare raw
// bits (including NaN payloads), all source modifiers and preserved halfwords.
// Loose SIMD uses fuzzy comparisons for both input formats.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed, typename Call>
void expect_wmma_exact_simd(Call call) {
  const auto levels = supported_cpu_levels();
  if (levels.size() < 2) {
    return;
  }
  std::minstd_rand random;
  constexpr uint32_t infinity = Bf16 ? 0x7f80 : 0x7c00;
  constexpr uint32_t fraction = Bf16 ? 0x7f : 0x3ff;
  constexpr uint32_t one = Bf16 ? 0x3f80 : 0x3c00;
  constexpr int dest_regs = !Rdna3 && Packed ? 128 / Lanes : 256 / Lanes;
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
        if (reg >= 16 && !Packed) {
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
        if (sample >= 128 && reg >= 16 && !Packed) {
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
      input.v[16][(sample + 5) % 16] = Packed ? infinity : 0x7f800001U;
    }
    // gfx11 requires replicated source fragments; enforce that layout before comparing backends.
    if constexpr (Rdna3) {
      for (int reg = 0; reg < 16; ++reg) {
        for (int lane = 16; lane < Lanes; ++lane) {
          input.v[reg][lane] = input.v[reg][lane % 16];
        }
      }
    }

    uint64_t instruction_flags = sample % 64;
    if constexpr (Packed && Rdna3) {
      if ((sample / 64) % 2) {
        instruction_flags |= RJ_ARITH_WMMA_HIGH_C_D;
      }
    }
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
      ASSERT_EQ(call(selected_flags, instruction_flags, expected.v + destination, expected.v,
                     expected.v + 8, expected.v + 16),
                RJ_ARITH_SUCCESS);
      for (uint64_t cpu : levels) {
        if (cpu == RJ_ARITH_CPU_BASELINE) {
          continue;
        }
        SCOPED_TRACE(::testing::Message()
                     << loose << '/' << sample << '/' << cpu << '/' << instruction_flags);
        auto actual = input;
        ASSERT_EQ(call(selected_flags | cpu, instruction_flags, actual.v + destination, actual.v,
                       actual.v + 8, actual.v + 16),
                  RJ_ARITH_SUCCESS);
        for (int reg = destination; reg < destination + dest_regs; ++reg) {
          for (int lane = 0; lane < Lanes; ++lane) {
            if (!loose) {
              ASSERT_EQ(actual.v[reg][lane], expected.v[reg][lane]) << reg << '/' << lane;
            } else {
              uint32_t av = actual.v[reg][lane], ev = expected.v[reg][lane];
              if constexpr (Packed) {
                if constexpr (Rdna3) {
                  int shift = instruction_flags & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0;
                  EXPECT_EQ(av & ~(0xffffU << shift), ev & ~(0xffffU << shift));
                  av >>= shift;
                  ev >>= shift;
                } else {
                  float ah = Bf16 ? bf16_value(uint16_t(av >> 16)) : f16_value(uint16_t(av >> 16));
                  float eh = Bf16 ? bf16_value(uint16_t(ev >> 16)) : f16_value(uint16_t(ev >> 16));
                  expect_f32(std::bit_cast<uint32_t>(ah), eh, RJ_ARITH_SEMANTICS_LOOSE,
                             /*atol=*/1e-36, /*rtol=*/1e-6);
                }
                av = std::bit_cast<uint32_t>(Bf16 ? bf16_value(uint16_t(av))
                                                  : f16_value(uint16_t(av)));
                ev = std::bit_cast<uint32_t>(Bf16 ? bf16_value(uint16_t(ev))
                                                  : f16_value(uint16_t(ev)));
              }
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

// Scalar/SIMD agreement for the vectorised exact gfx12 FP8 WMMA kernels.
//
// Random factors and accumulators never drive a DOT8 result into the FP32 subnormal range or
// to overflow, so they never reach the kernels' integer-pack repair path, and they never leave
// every product zero, which is the only way the accumulation grid falls below the FP32 normal
// floor. The directed accumulators and factor bytes below cover all three, and alternate with
// random tiles so that per-group grid selection is still exercised.
template <int Lanes, typename Call> void expect_wmma_fp8_exact_simd(Call call) {
  const auto levels = supported_cpu_levels();
  if (levels.size() < 2) {
    return;
  }
  constexpr int dest_regs = 256 / Lanes;
  // Signed zero, both subnormal ends, the smallest normal, the largest finite, the infinities
  // and a quiet NaN: the accumulators that put a result on an FP32 boundary.
  static constexpr uint32_t kAccumulators[] = {
      0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000,
      0x80800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x3f800000,
      0xbf800000, 0x00000010};
  // E4M3 zero, smallest and largest subnormal, smallest normal, largest finite, NaN and one.
  static constexpr uint32_t kFactors[] = {0x00, 0x80, 0x01, 0x81, 0x07, 0x87,
                                          0x08, 0x7e, 0xfe, 0x7f, 0x38, 0xb8};
  // Every signed power of two as the accumulator, against the largest finite factors. These
  // are the tiles where the accumulator's magnitude survives the shift as exactly one: its
  // floored value is -1, but it has NOT vanished and must still contribute. Random tiles
  // essentially never land there.
  std::vector<uint32_t> accumulators(std::begin(kAccumulators), std::end(kAccumulators));
  for (uint32_t field = 1; field < 255; ++field) {
    accumulators.push_back(field << 23);
    accumulators.push_back(0x80000000u | (field << 23));
  }
  std::minstd_rand random;
  int sample = 0;
  for (uint32_t accumulator : accumulators) {
    for (uint32_t left : kFactors) {
      for (uint32_t right : kFactors) {
        Registers<32, Lanes> input;
        const int shape = sample % 4;
        for (int lane = 0; lane < Lanes; ++lane) {
          // shape 1 and 2 blank half of A or B, so a whole group of four products is zero
          // while its neighbour is not.
          const bool live_a = shape != 1 || lane < Lanes / 2;
          const bool live_b = shape != 2 || (lane % 2) == 0;
          for (int reg : {0, 1}) {
            input.v[reg][lane] = live_a ? left * 0x01010101U : 0;
          }
          for (int reg : {8, 9}) {
            input.v[reg][lane] = live_b ? right * 0x01010101U : 0;
          }
          for (int reg = 16; reg < 32; ++reg) {
            input.v[reg][lane] = accumulator;
          }
          if (shape == 3) {
            // Mixed magnitudes so the two groups of a DOT8 choose different grids.
            input.v[0][lane] = left * 0x01010101U;
            input.v[1][lane] = uint32_t(random()) & 0x7f7f7f7fU;
            input.v[8][lane] = right * 0x01010101U;
            input.v[9][lane] = uint32_t(random()) & 0x7f7f7f7fU;
          }
        }
        const uint64_t instruction_flags =
            (sample % 2 ? RJ_ARITH_WMMA_NEG_C : 0) | (sample % 3 == 0 ? RJ_ARITH_WMMA_ABS_C : 0);
        uint64_t flags = sample % 2 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                                    : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
        // Ignored for matrix operations; the result must not depend on them.
        if (sample % 3 == 0) {
          flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                   RJ_ARITH_FP_IEEE_MODE;
        }
        const int aliases[] = {0, 1, 8, 16, 24};
        const int destination = aliases[sample % 5];
        auto expected = input;
        ASSERT_EQ(call(flags, instruction_flags, expected.v + destination, expected.v,
                       expected.v + 8, expected.v + 16),
                  RJ_ARITH_SUCCESS);
        for (uint64_t cpu : levels) {
          if (cpu == RJ_ARITH_CPU_BASELINE) {
            continue;
          }
          SCOPED_TRACE(::testing::Message() << sample << '/' << cpu << '/' << accumulator << '/'
                                            << left << '/' << right);
          auto actual = input;
          ASSERT_EQ(call(flags | cpu, instruction_flags, actual.v + destination, actual.v,
                         actual.v + 8, actual.v + 16),
                    RJ_ARITH_SUCCESS);
          for (int reg = destination; reg < destination + dest_regs; ++reg) {
            for (int lane = 0; lane < Lanes; ++lane) {
              ASSERT_EQ(actual.v[reg][lane], expected.v[reg][lane]) << reg << '/' << lane;
            }
          }
          actual.expect_preserved(input, destination, dest_regs);
        }
        ++sample;
      }
    }
  }
}

} // namespace rocjitsu::arith::test
