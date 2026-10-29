// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense WMMA operand layouts and shared arithmetic, aliasing, and invalid-flag checks.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <bit>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Dense floating WMMA exposes these six independent source modifiers.
inline constexpr uint64_t kFloatingWmmaModifiers = RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_LO_B |
                                                   RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_NEG_HI_A |
                                                   RJ_ARITH_WMMA_NEG_HI_B | RJ_ARITH_WMMA_ABS_C;

// Check invalid flags with real storage (all preserved) and null operands
// (rejected before access). Offsets identify the dense WMMA D/A/B/C operands.
template <typename Call, int RegistersCount, int Lanes>
void expect_matrix_invalid_flags(Call call, const Registers<RegistersCount, Lanes> &input, int d,
                                 int a, int b, int c, uint64_t supported_modifiers) {
  for (uint64_t invalid :
       {RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16, RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16}) {
    EXPECT_EQ(call(invalid, 0, nullptr, nullptr, nullptr, nullptr), RJ_ARITH_ERROR_INVALID_FLAGS);
  }
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t modifiers) {
        auto r = input;
        int result = call(flags, modifiers, r.v + d, r.v + a, r.v + b, r.v + c);
        r.expect_preserved(input, 0, 0);
        EXPECT_EQ(call(flags, modifiers, nullptr, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return result;
      },
      supported_modifiers);
}

// Isolate each factor-half sign and accumulator modifier with exactly representable
// arithmetic. These cases supplement the varied-input per-instruction goldens.
template <bool Bf16, int Lanes, bool Gfx11, typename Call> void expect_matrix_modifiers(Call call) {
  constexpr int factors = Gfx11 ? 8 : 128 / Lanes, outputs = 256 / Lanes;
  Registers<32, Lanes> input;
  for (int reg = 0; reg < factors * 2; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] =
          reg < factors ? (Bf16 ? 0x40003f80U : 0x40003c00U) : (Bf16 ? 0x40803f80U : 0x44003c00U);
    }
  }
  for (int reg = 16; reg < 16 + outputs; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = std::bit_cast<uint32_t>(-2.0f);
    }
  }
  const struct {
    uint64_t instruction_flags;
    float negative_c, positive_c;
  } cases[] = {{0, 70, 74},
               {RJ_ARITH_WMMA_NEG_LO_A, 54, 58},
               {RJ_ARITH_WMMA_NEG_HI_A, -58, -54},
               {RJ_ARITH_WMMA_NEG_LO_B, 54, 58},
               {RJ_ARITH_WMMA_NEG_HI_B, -58, -54},
               {RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_HI_A, -74, -70},
               {RJ_ARITH_WMMA_NEG_C, 74, 70},
               {RJ_ARITH_WMMA_ABS_C, 74, 74},
               {RJ_ARITH_WMMA_ABS_C | RJ_ARITH_WMMA_NEG_C, 70, 70}};
  for (float accumulator : {-2.0f, 2.0f}) {
    for (int reg = 16; reg < 16 + outputs; ++reg) {
      std::fill_n(input.v[reg], Lanes, std::bit_cast<uint32_t>(accumulator));
    }
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : semantics_modes()) {
        for (const auto &test : cases) {
          SCOPED_TRACE(testing::Message()
                       << cpu << '/' << semantics << '/' << test.instruction_flags);
          auto r = input;
          ASSERT_EQ(
              call(cpu | semantics, test.instruction_flags, r.v + 24, r.v, r.v + factors, r.v + 16),
              RJ_ARITH_SUCCESS);
          for (int reg = 24; reg < 24 + outputs; ++reg) {
            for (int lane = 0; lane < Lanes; ++lane) {
              EXPECT_EQ(r.v[reg][lane], std::bit_cast<uint32_t>(accumulator < 0 ? test.negative_c
                                                                                : test.positive_c));
            }
          }
          r.expect_preserved(input, 24, outputs);
        }
      }
    }
  }
  for (int reg = 16; reg < 16 + outputs; ++reg) {
    std::fill_n(input.v[reg], Lanes, std::bit_cast<uint32_t>(-2.0f));
  }
  // Architecture-defined controls irrelevant to matrix arithmetic are ignored.
  for (uint64_t guest : {RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO,
                         RJ_ARITH_FP_DX10_CLAMP_OFF, RJ_ARITH_FP_IEEE_MODE,
                         RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
    auto actual = input;
    if (!Gfx11 && guest == RJ_ARITH_FP_DX10_CLAMP_OFF) {
      EXPECT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | guest, 0, actual.v + 24,
                     actual.v, actual.v + factors, actual.v + 16),
                RJ_ARITH_ERROR_INVALID_FLAGS);
      actual.expect_preserved(input, 0, 0);
      EXPECT_EQ(call(guest, 0, nullptr, nullptr, nullptr, nullptr), RJ_ARITH_ERROR_INVALID_FLAGS);
      continue;
    }
    ASSERT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | guest, 0, actual.v + 24,
                   actual.v, actual.v + factors, actual.v + 16),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(actual.v[24][0], std::bit_cast<uint32_t>(70.0f));
  }
  if constexpr (Gfx11 && !Bf16) {
    // FP16's tiny contribution rounds the gfx11 accumulator upward under every
    // guest mode: WMMA has fixed rounding, unlike ordinary FMA.
    for (int reg = 0; reg < 8; ++reg) {
      std::fill_n(input.v[reg], Lanes, reg == 0 ? 0x00000800U : 0U);
      std::fill_n(input.v[reg + 8], Lanes, 0x00001200U);
    }
    for (int reg = 16; reg < 16 + outputs; ++reg) {
      std::fill_n(input.v[reg], Lanes, 0x3f800000U);
    }
    for (uint64_t rounding :
         {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      auto actual = input;
      ASSERT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | rounding, 0,
                     actual.v + 24, actual.v, actual.v + 8, actual.v + 16),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(actual.v[24][0], 0x3f800001U);
    }
  }
}

// Write an A/B element in the gfx11 dense WMMA layout. Coordinate is the
// A row or B column; each 16-lane group repeats the complete K vector.
template <int Bits, int Lanes>
void set_wmma_gfx11(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  static_assert(Bits == 4 || Bits == 8 || Bits == 16);
  static_assert(Lanes == 32 || Lanes == 64);
  for (int group = 0; group < Lanes / 16; ++group) {
    auto &word = v[k * Bits / 32][coordinate + 16 * group];
    int shift = k * Bits % 32;
    word = (word & ~(((1U << Bits) - 1) << shift)) | (bits << shift);
  }
}

// Write a gfx12 dense WMMA FP16/BF16 A/B element. K is partitioned
// between 16-lane groups, with adjacent halves packed in each word.
template <int Lanes>
void set_wmma_gfx12_half(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  static_assert(Lanes == 32 || Lanes == 64);
  constexpr int group_k = 16 / (Lanes / 16);
  auto &word = v[(k % group_k) / 2][coordinate + 16 * (k / group_k)];
  int shift = 16 * (k % 2);
  word = (word & ~(0xffffU << shift)) | (bits << shift);
}

// Access an FP32 accumulator element in the gfx12 wave layout.
template <int Lanes> uint32_t &wmma_gfx12_acc(uint32_t *const *v, int row, int col) {
  static_assert(Lanes == 32 || Lanes == 64);
  constexpr int regs = 256 / Lanes;
  int lane = col + 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
  return v[row % regs][lane];
}

// Access a gfx11 FP32 accumulator element in the wave's ISA register layout.
template <int Lanes> uint32_t &wmma_gfx11_acc(uint32_t *const *v, int row, int col) {
  static_assert(Lanes == 32 || Lanes == 64);
  constexpr int groups = Lanes / 16;
  return v[row / groups][col + 16 * (row % groups)];
}

// Check gfx1201's negative-C floor unit (rocjitsu #13691). Wave64 applies the
// same logical matrix layout; its result was not separately captured.
template <int Lanes, typename Call> void expect_bf16_tiny_accumulator(Call call) {
  for (uint64_t cpu : supported_cpu_levels()) {
    Registers<32, Lanes> r;
    for (auto *reg : r.v) {
      std::fill_n(reg, Lanes, 0U);
    }
    for (int row = 0; row < 16; ++row) {
      set_wmma_gfx12_half<Lanes>(r.v, row, 0, 0x2900);
      for (int k = 0; k < 16; ++k) {
        set_wmma_gfx12_half<Lanes>(r.v + 4, row, k, 0x2280);
      }
      for (int col = 0; col < 16; ++col) {
        wmma_gfx12_acc<Lanes>(r.v + 8, row, col) = 0x8000082dU;
      }
    }
    ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, r.v + 8, r.v,
                   r.v + 4, r.v + 8),
              RJ_ARITH_SUCCESS);
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        EXPECT_EQ(wmma_gfx12_acc<Lanes>(r.v + 8, row, col), 0x0bffffffU);
      }
    }
  }
}

// Check mandatory gfx11 flushing of BF16 factors, FP32 accumulators, and tiny results.
template <int Lanes, typename Call> void expect_gfx11_bf16_denormal(Call call) {
  struct Case {
    int k;
    uint16_t a, b;
    uint32_t c, expected;
  };
  // The last two cases match a gfx1100 Wave32 observation; Wave64 checks the same
  // logical layout. Keep K=0 for input flushing and K=15 for result flushing:
  // intervening DOT2 steps flush the accumulator again.
  constexpr Case cases[] = {{14, 0x0001, 0x4980, 0x00000000, 0x00000000},
                            {0, 0x0080, 0xbf80, 0x00000001, 0x80800000},
                            {15, 0x0080, 0x3f80, 0x80c00000, 0x00000000}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (const auto &sample : cases) {
        SCOPED_TRACE(testing::Message() << cpu << '/' << semantics << "/K=" << sample.k);
        Registers<32, Lanes> r;
        for (auto *reg : r.v) {
          std::fill_n(reg, Lanes, 0U);
        }
        for (int coordinate = 0; coordinate < 16; ++coordinate) {
          set_wmma_gfx11<16, Lanes>(r.v, coordinate, sample.k, sample.a);
          set_wmma_gfx11<16, Lanes>(r.v + 8, coordinate, sample.k, sample.b);
          for (int col = 0; col < 16; ++col) {
            wmma_gfx11_acc<Lanes>(r.v + 16, coordinate, col) = sample.c;
          }
        }
        ASSERT_EQ(call(cpu | semantics, 0, r.v + 24, r.v, r.v + 8, r.v + 16), RJ_ARITH_SUCCESS);
        for (int row = 0; row < 16; ++row) {
          for (int col = 0; col < 16; ++col) {
            EXPECT_EQ(wmma_gfx11_acc<Lanes>(r.v + 24, row, col), sample.expected);
          }
        }
      }
    }
  }
}

// Check an FP32-output WMMA against its golden matrix across CPUs, semantics,
// and destination aliases. A/B/C use consecutive operand ranges starting at zero;
// all storage outside D is preserved. Layout selection is independent of production code.
template <bool Gfx11, typename Call, int RegistersCount, int Lanes>
void expect_wmma_f32_arithmetic(Call call, const Registers<RegistersCount, Lanes> &input,
                                const float (&golden)[16][16],
                                std::initializer_list<int> aliasing_offsets,
                                std::initializer_list<uint64_t> fp_modes = {0}) {
  constexpr int factors = Gfx11 ? 8 : 128 / Lanes;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (uint64_t fp_flags : fp_modes) {
        for (int aliasing_offset : aliasing_offsets) {
          SCOPED_TRACE(testing::Message()
                       << cpu << '/' << semantics << '/' << aliasing_offset << '/' << fp_flags);
          auto r = input;
          ASSERT_EQ(call(cpu | semantics | fp_flags, 0, r.v + aliasing_offset, r.v, r.v + factors,
                         r.v + 2 * factors),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = Gfx11 ? wmma_gfx11_acc<Lanes>(r.v + aliasing_offset, row, col)
                                      : wmma_gfx12_acc<Lanes>(r.v + aliasing_offset, row, col);
              if (semantics == RJ_ARITH_SEMANTICS_LOOSE) {
                expect_f32(actual, golden[row][col], semantics, /*atol=*/1e-5, /*rtol=*/1e-5);
              } else {
                EXPECT_EQ(actual, std::bit_cast<uint32_t>(golden[row][col]));
              }
            }
          }
          r.expect_preserved(input, aliasing_offset, 256 / Lanes);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
