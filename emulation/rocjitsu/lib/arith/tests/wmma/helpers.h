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
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check invalid flags with real storage (all preserved) and null operands
// (rejected before access). Offsets identify the dense WMMA D/A/B/C operands.
template <typename Call, int RegistersCount, int Lanes>
void expect_matrix_invalid_flags(Call call, const Registers<RegistersCount, Lanes> &input, int d,
                                 int a, int b, int c, uint64_t supported_modifiers) {
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
template <bool Bf16, int Lanes, bool Gfx11, bool Packed = false, typename Call>
void expect_matrix_modifiers(Call call) {
  constexpr int factors = Gfx11 ? 8 : 128 / Lanes,
                outputs = 256 / Lanes / (Packed && !Gfx11 ? 2 : 1);
  Registers<32, Lanes> input;
  for (int reg = 0; reg < factors * 2; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] =
          reg < factors ? (Bf16 ? 0x40003f80U : 0x40003c00U) : (Bf16 ? 0x40803f80U : 0x44003c00U);
    }
  }
  for (int reg = 16; reg < 16 + outputs; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = (Packed ? 0xc000c000U : std::bit_cast<uint32_t>(-2.0f));
    }
  }
  const struct {
    uint64_t mode;
    float expected;
  } cases[] = {{0, 70},
               {RJ_ARITH_WMMA_NEG_LO_A, 54},
               {RJ_ARITH_WMMA_NEG_HI_A, -58},
               {RJ_ARITH_WMMA_NEG_LO_B, 54},
               {RJ_ARITH_WMMA_NEG_HI_B, -58},
               {RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_HI_A, -74},
               {RJ_ARITH_WMMA_NEG_C, 74},
               {RJ_ARITH_WMMA_ABS_C, 74},
               {RJ_ARITH_WMMA_ABS_C | RJ_ARITH_WMMA_NEG_C, 70}};
  for (float accumulator : {-2.0f, 2.0f}) {
    for (int reg = 16; reg < 16 + outputs; ++reg) {
      std::fill_n(input.v[reg], Lanes,
                  (Packed ? (accumulator < 0 ? 0xc000c000U : 0x40004000U)
                          : std::bit_cast<uint32_t>(accumulator)));
    }
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : semantics_modes()) {
        for (const auto &test : cases) {
          SCOPED_TRACE(testing::Message() << cpu << '/' << semantics << '/' << test.mode);
          for (int selector = 0; selector < (Gfx11 && Packed ? 2 : 1); ++selector) {
            uint64_t high = selector ? RJ_ARITH_WMMA_HIGH_C_D : 0;
            auto r = input;
            ASSERT_EQ(
                call(cpu | semantics, test.mode | high, r.v + 24, r.v, r.v + factors, r.v + 16),
                RJ_ARITH_SUCCESS);
            float expected = test.expected + (accumulator > 0 && !(test.mode & RJ_ARITH_WMMA_ABS_C)
                                                  ? (test.mode & RJ_ARITH_WMMA_NEG_C ? -4.0f : 4.0f)
                                                  : 0.0f);
            for (int reg = 24; reg < 24 + outputs; ++reg) {
              for (int lane = 0; lane < Lanes; ++lane) {
                if constexpr (Packed) {
                  for (int half : {0, 1}) {
                    uint16_t actual = uint16_t(r.v[reg][lane] >> (16 * half));
                    if (Gfx11 && half != (high != 0)) {
                      EXPECT_EQ(actual, uint16_t(input.v[reg][lane] >> (16 * half)));
                    } else {
                      EXPECT_EQ(Bf16 ? std::bit_cast<float>(uint32_t(actual) << 16)
                                     : f16_value(actual),
                                expected);
                    }
                  }
                } else {
                  EXPECT_EQ(r.v[reg][lane], std::bit_cast<uint32_t>(expected));
                }
              }
            }
            r.expect_preserved(input, 24, outputs);
          }
        }
      }
    }
  }
  if constexpr (Packed) {
    return;
  }
  for (int reg = 16; reg < 16 + outputs; ++reg) {
    std::fill_n(input.v[reg], Lanes, (Packed ? 0xc000c000U : std::bit_cast<uint32_t>(-2.0f)));
  }
  // MODE is forwarded unchanged by callers; irrelevant controls must not reject WMMA.
  for (uint64_t guest : {RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO,
                         RJ_ARITH_FP_DX10_CLAMP_OFF, RJ_ARITH_FP_IEEE_MODE,
                         RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
    auto actual = input;
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

// Write a gfx12 dense WMMA 4/8-bit A/B element. Wave64 distributes
// the Wave32 register words into the upper 32 lanes.
template <int Bits, int Lanes>
void set_wmma_gfx12_small(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  static_assert(Bits == 4 || Bits == 8);
  static_assert(Lanes == 32 || Lanes == 64);
  int bit = (k % 8 + 8 * (k / 16)) * Bits;
  int lane = coordinate + 16 * ((k / 8) % 2);
  auto &word = v[Lanes == 32 ? bit / 32 : 0][lane + (Lanes == 64 ? 32 * (bit / 32) : 0)];
  int shift = bit % 32;
  word = (word & ~(((1U << Bits) - 1) << shift)) | (bits << shift);
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

// Replay two selected output dots from rocjitsu's packed_wmma_cases.h captures
// (gfx1100/gfx1201, both wave sizes). Broadcast those rows/columns to isolate
// the differing DOT2/DOT4 rounding boundaries without copying entire matrices.
template <bool Bf16, int Lanes, bool Gfx11, typename Call>
void expect_packed_wmma_capture(Call call) {
  const struct {
    uint16_t a[16], b[16], c, expected[2][2];
  } samples[] = {
      // FP16 fixture 0, output (0,2).
      {{0x380b, 0xcc95, 0xca7c, 0xbea9, 0x2c0b, 0x3367, 0xb997, 0xaa37, 0x2f82, 0x5015, 0xcaf3,
        0x4edc, 0xaff1, 0x2e10, 0x2600, 0xc520},
       {0x5787, 0xaca9, 0xb60b, 0xb8e6, 0xc7ca, 0xd38a, 0x29da, 0xc4af, 0xbb62, 0xb8ee, 0xade9,
        0x4c99, 0x4526, 0x4e40, 0x3a50, 0x33af},
       0xbb50,
       {{0x6036, 0x6036}, {0x6037, 0x6036}}},
      // BF16 fixture 0, output (0,3).
      {{0x3f01, 0xc192, 0xc14f, 0xbfd5, 0x3d81, 0x3e6c, 0xbf32, 0xbd46, 0x3df0, 0x4202, 0xc15e,
        0x41db, 0xbdfe, 0x3dc2, 0x3cc0, 0xc0a4},
       {0xbfc5, 0x41fc, 0xc188, 0x3e1f, 0xc19e, 0x3d60, 0x3d89, 0xc0f0, 0xc2f3, 0xc151, 0x40e7,
        0x3cda, 0x3f40, 0xc1e2, 0x4220, 0x402a},
       0x3f6f,
       {{0xc460, 0xc460}, {0xc462, 0xc462}}},
  };
  const auto &sample = samples[Bf16];
  for (uint64_t cpu : supported_cpu_levels()) {
    Registers<32, Lanes> r;
    for (int coordinate = 0; coordinate < 16; ++coordinate) {
      for (int k = 0; k < 16; ++k) {
        if constexpr (Gfx11) {
          set_wmma_gfx11<16, Lanes>(r.v, coordinate, k, sample.a[k]);
          set_wmma_gfx11<16, Lanes>(r.v + 8, coordinate, k, sample.b[k]);
        } else {
          set_wmma_gfx12_half<Lanes>(r.v, coordinate, k, sample.a[k]);
          set_wmma_gfx12_half<Lanes>(r.v + 8, coordinate, k, sample.b[k]);
        }
      }
    }
    for (int reg = 16; reg < 24; ++reg) {
      std::fill_n(r.v[reg], Lanes, uint32_t(sample.c) * 0x10001U);
    }
    ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, r.v + 24, r.v,
                   r.v + 8, r.v + 16),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(uint16_t(r.v[24][0]), sample.expected[!Gfx11][Lanes == 64]);
  }
}

// Replay a uniform FP8/BF8 gfx1201 witness from rocjitsu's RdnaWmmaFloat tests.
// Sources occupy independent eight-register ranges; only D's defined words are checked.
template <int Lanes, typename Call>
void expect_f8_wmma_capture(Call call, uint32_t a, uint32_t b, uint32_t expected) {
  for (uint64_t cpu : supported_cpu_levels()) {
    Registers<32, Lanes> r;
    for (int reg = 0; reg < 8; ++reg) {
      std::fill_n(r.v[reg], Lanes, a);
      std::fill_n(r.v[reg + 8], Lanes, b);
      std::fill_n(r.v[reg + 16], Lanes, 0xbf000000U);
    }
    ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, r.v + 24, r.v,
                   r.v + 8, r.v + 16),
              RJ_ARITH_SUCCESS);
    for (int reg = 24; reg < 24 + 256 / Lanes; ++reg) {
      for (int lane = 0; lane < Lanes; ++lane) {
        EXPECT_EQ(r.v[reg][lane], expected);
      }
    }
  }
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
