// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Matrix operand layouts, sparse indices, and shared arithmetic and API-contract checks.

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
#include <vector>

namespace rocjitsu::arith::test {

// Floating WMMA and SWMMAC expose these six independent source modifiers.
inline constexpr uint64_t kFloatingWmmaModifiers = RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_LO_B |
                                                   RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_NEG_HI_A |
                                                   RJ_ARITH_WMMA_NEG_HI_B | RJ_ARITH_WMMA_ABS_C;

// Check invalid flags with real storage (all preserved) and null operands
// (rejected before access). The last offset denotes C for WMMA or sparse
// indices for SWMMAC; SWMMAC's D is also its accumulator input.
template <typename Call, int RegistersCount, int Lanes>
void expect_matrix_invalid_flags(Call call, const Registers<RegistersCount, Lanes> &input, int d,
                                 int a, int b, int c_or_indices, uint64_t supported_modifiers) {
  for (uint64_t invalid :
       {RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16, RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16}) {
    EXPECT_EQ(call(invalid, 0, nullptr, nullptr, nullptr, nullptr), RJ_ARITH_ERROR_INVALID_FLAGS);
  }
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t modifiers) {
        auto r = input;
        int result = call(flags, modifiers, r.v + d, r.v + a, r.v + b, r.v + c_or_indices);
        r.expect_preserved(input, 0, 0);
        EXPECT_EQ(call(flags, modifiers, nullptr, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return result;
      },
      supported_modifiers);
}

// Guest controls advertised as ignored must preserve the default result. Invalid
// controls must preserve all storage and be rejected before operand access.
template <bool Floating, bool Gfx11, bool Fp16Output = false, typename Call, int RegistersCount,
          int Lanes>
void expect_matrix_guest_controls(Call call, const Registers<RegistersCount, Lanes> &input, int d,
                                  int a, int b, int c) {
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      auto reference = input;
      ASSERT_EQ(call(cpu | semantics, 0, reference.v + d, reference.v + a, reference.v + b,
                     reference.v + c),
                RJ_ARITH_SUCCESS);
      for (uint64_t guest :
           {RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO,
            RJ_ARITH_FP_DX10_CLAMP_OFF, RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS,
            RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, RJ_ARITH_FP16_OVFL}) {
        // Only FP16-output forms use finite-overflow saturation.
        if (Fp16Output && guest == RJ_ARITH_FP16_OVFL) {
          continue;
        }
        SCOPED_TRACE(testing::Message() << cpu << '/' << semantics << '/' << guest);
        auto actual = input;
        const bool accepted =
            Floating ? (Gfx11 || guest != RJ_ARITH_FP_DX10_CLAMP_OFF) : guest == RJ_ARITH_FP16_OVFL;
        EXPECT_EQ(call(cpu | semantics | guest, 0, actual.v + d, actual.v + a, actual.v + b,
                       actual.v + c),
                  accepted ? RJ_ARITH_SUCCESS : RJ_ARITH_ERROR_INVALID_FLAGS);
        actual.expect_preserved(accepted ? reference : input, 0, 0);
        if (!accepted) {
          EXPECT_EQ(call(cpu | semantics | guest, 0, nullptr, nullptr, nullptr, nullptr),
                    RJ_ARITH_ERROR_INVALID_FLAGS);
        }
      }
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

// Write one 16-bit gfx12 sparse A value; K is compressed for A.
template <int WaveSize>
void set_swmmac16_a(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  static_assert(WaveSize == 32 || WaveSize == 64);
  if constexpr (WaveSize == 32) {
    auto &word = v[2 * (k / 8) + ((k / 2) & 1)][coordinate + 16 * ((k / 4) & 1)];
    int shift = 16 * (k % 2);
    word = (word & ~(0xffffU << shift)) | (bits << shift);
  } else {
    auto &word = v[(k % 4) / 2][coordinate + 16 * (k / 4)];
    int shift = 16 * (k % 2);
    word = (word & ~(0xffffU << shift)) | (bits << shift);
  }
}

// Write one 16-bit gfx12 sparse B value; K is dense for B.
template <int WaveSize>
void set_swmmac16_b(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  static_assert(WaveSize == 32 || WaveSize == 64);
  if constexpr (WaveSize == 32) {
    auto &word = v[4 * (k / 16) + (k / 2) % 4][coordinate + 16 * ((k / 8) & 1)];
    int shift = 16 * (k % 2);
    word = (word & ~(0xffffU << shift)) | (bits << shift);
  } else {
    auto &word = v[(k % 8) / 2][coordinate + 16 * (k / 8)];
    int shift = 16 * (k % 2);
    word = (word & ~(0xffffU << shift)) | (bits << shift);
  }
}

// Write one 4/8-bit gfx12 sparse B value; K is dense for B.
// IU4 Wave64 K=64 uses a different map and has its own writer.
template <int Bits, int WaveSize>
void set_swmmac_small_b(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  static_assert(WaveSize == 32 || WaveSize == 64);
  if constexpr (WaveSize == 32) {
    auto &word = v[(16 * (k / 32) + k % 16) * Bits / 32][coordinate + 16 * ((k / 16) & 1)];
    int shift = (16 * (k / 32) + k % 16) * Bits % 32;
    word = (word & ~(((1U << Bits) - 1) << shift)) | (bits << shift);
  } else {
    auto &word = v[(k % 8) * Bits / 32][coordinate + 32 * ((k / 8) & 1) + 16 * (k / 16)];
    int shift = (k % 8) * Bits % 32;
    word = (word & ~(((1U << Bits) - 1) << shift)) | (bits << shift);
  }
}

// Replicate valid selector pairs into every metadata bank. Tests isolate the
// chosen bank per call with isolate_sparse_selector; no production decoder is used.
template <int Bits, int WaveSize, int CompressedK>
void set_swmmac_indices(uint32_t *index, const int (&selected)[16][CompressedK]) {
  std::fill_n(index, WaveSize, 0U);
  // The full-word bank is gfx12 IU4 Wave32 K=64, not a general rule
  // for arbitrary instruction shapes with compressed length 32.
  constexpr int bank_bits = WaveSize == 64 && Bits != 4           ? 8
                            : WaveSize == 32 && CompressedK == 32 ? 32
                                                                  : 16;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < CompressedK; ++k) {
      int lane, local;
      if constexpr (Bits == 16) {
        lane = row + 16 * (WaveSize == 64 ? k / 4 : (k / 4) % 2);
        local = WaveSize == 64 ? k % 4 : k % 4 + 4 * (k / 8);
      } else if constexpr (WaveSize == 32) {
        lane = row + 16 * ((k / 8) % 2);
        local = 8 * (k / 16) + k % 8;
      } else if constexpr (Bits == 8) {
        lane = row + 16 * (k / 8) + 32 * ((k / 4) % 2);
        local = k % 4;
      } else if constexpr (CompressedK == 16) {
        lane = row + 16 * (k / 8);
        local = k % 8;
      } else {
        lane = row + 32 * ((k / 8) % 2) + 16 * (k / 16);
        local = k % 8;
      }
      for (int bank = 0; bank < 32; bank += bank_bits) {
        index[lane] |= uint32_t(selected[row][k] % 4) << (bank + 2 * local);
      }
    }
  }
}

// Expected dot-plus-C results for A=(1,2), B=(1,4), C=-2.
inline constexpr struct {
  uint64_t instruction_flags;
  float negative_c, positive_c;
} kMatrixModifierCases[] = {{0, 70, 74},
                            {RJ_ARITH_WMMA_NEG_LO_A, 54, 58},
                            {RJ_ARITH_WMMA_NEG_HI_A, -58, -54},
                            {RJ_ARITH_WMMA_NEG_LO_B, 54, 58},
                            {RJ_ARITH_WMMA_NEG_HI_B, -58, -54},
                            {RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_HI_A, -74, -70},
                            {RJ_ARITH_WMMA_NEG_C, 74, 70},
                            {RJ_ARITH_WMMA_ABS_C, 74, 74},
                            {RJ_ARITH_WMMA_ABS_C | RJ_ARITH_WMMA_NEG_C, 70, 70}};

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

  for (float accumulator : {-2.0f, 2.0f}) {
    for (int reg = 16; reg < 16 + outputs; ++reg) {
      std::fill_n(input.v[reg], Lanes,
                  (Packed ? (Gfx11 ? (accumulator < 0 ? 0x4000c000U : 0xc0004000U)
                                   : (accumulator < 0 ? 0xc000c000U : 0x40004000U))
                          : std::bit_cast<uint32_t>(accumulator)));
    }
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : semantics_modes()) {
        for (const auto &test : kMatrixModifierCases) {
          SCOPED_TRACE(testing::Message()
                       << cpu << '/' << semantics << '/' << test.instruction_flags);
          for (int selector = 0; selector < (Gfx11 && Packed ? 2 : 1); ++selector) {
            uint64_t high = selector ? RJ_ARITH_WMMA_HIGH_C_D : 0;
            auto r = input;
            ASSERT_EQ(call(cpu | semantics, test.instruction_flags | high, r.v + 24, r.v,
                           r.v + factors, r.v + 16),
                      RJ_ARITH_SUCCESS);
            // gfx11 selects one accumulator half; keep the halves distinct.
            float selected_accumulator = Gfx11 && Packed && selector ? -accumulator : accumulator;
            float expected = selected_accumulator < 0 ? test.negative_c : test.positive_c;
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
  expect_matrix_guest_controls<true, Gfx11, Packed && !Bf16>(call, input, 24, 0, factors, 16);
  if constexpr (Gfx11 && !Bf16 && !Packed) {
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

// Check tied C/D aliases against a disjoint snapshot of the same effective inputs.
// Scratch registers [48, 48+outputs) must be disjoint from the operand ranges.
// Numerical correctness is checked separately by each instruction's fixture.
template <int RegistersCount, int Lanes, typename Call>
void expect_sparse_aliasing(Call call, const Registers<RegistersCount, Lanes> &input, int outputs,
                            int b, int index, std::initializer_list<int> offsets,
                            const std::vector<uint64_t> &cpu_levels) {
  static_assert(RegistersCount >= 56);
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      for (int aliasing_offset : offsets) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << aliasing_offset);
        auto r = input, reference = input;
        for (int reg = 0; reg < outputs; ++reg) {
          for (int lane = 0; lane < Lanes; ++lane) {
            reference.v[48 + reg][lane] = r.v[aliasing_offset + reg][lane];
          }
        }
        ASSERT_EQ(call(cpu | semantics, 0, reference.v + 48, reference.v, reference.v + b,
                       reference.v + index),
                  RJ_ARITH_SUCCESS);
        ASSERT_EQ(call(cpu | semantics, 0, r.v + aliasing_offset, r.v, r.v + b, r.v + index),
                  RJ_ARITH_SUCCESS);
        for (int reg = 0; reg < outputs; ++reg) {
          for (int lane = 0; lane < Lanes; ++lane) {
            EXPECT_EQ(r.v[aliasing_offset + reg][lane], reference.v[48 + reg][lane]);
          }
        }
        r.expect_preserved(input, aliasing_offset, outputs);
      }
    }
  }
}

// Clear unused metadata banks so reading the wrong selector cannot pass by
// observing the same pairs. Only the selected bank needs valid 2:4 pairs.
template <int Lanes, int Width = Lanes == 64 ? 8 : 16>
void isolate_sparse_selector(uint32_t *index, uint64_t instruction_flags) {
  static_assert(Width == 8 || Width == 16);
  constexpr int width = Width;
  int key = ((instruction_flags & RJ_ARITH_SWMMAC_INDEX_KEY_1) ? 1 : 0) |
            ((instruction_flags & RJ_ARITH_SWMMAC_INDEX_KEY_2) ? 2 : 0);
  key &= 32 / Width - 1;
  uint32_t mask = ((1U << width) - 1) << (width * key);
  for (int lane = 0; lane < Lanes; ++lane) {
    index[lane] &= mask;
  }
}

// Select two increasing K positions per group of four, using high engine bits.
// Draw order is row-major and part of the deterministic golden-input contract.
template <int K> void make_sparse_indices(std::minstd_rand &random, int (&selected)[16][K]) {
  static_assert(K % 2 == 0);
  const uint32_t pairs[] = {0x4, 0x8, 0xc, 0x9, 0xd, 0xe};
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < K; k += 2) {
      uint32_t pair = pairs[(random() >> 16) % 6];
      selected[row][k] = 2 * k + (pair & 3);
      selected[row][k + 1] = 2 * k + (pair >> 2);
    }
  }
}

// Saturate each packed FP16 accumulation stage, without saturating input infinities.
template <int WaveSize, typename Call> void expect_sparse_half_overflow(Call call) {
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (bool saturate : {false, true}) {
        for (bool infinity : {false, true}) {
          Registers<32, WaveSize> r;
          for (auto *reg : r.v) {
            std::fill_n(reg, WaveSize, 0U);
          }
          for (int coordinate = 0; coordinate < 16; ++coordinate) {
            for (int k = 0; k < 16; ++k) {
              set_swmmac16_a<WaveSize>(r.v, coordinate, k,
                                       infinity && k == 0 ? kF16PositiveInfinityBits
                                       : k < 4            ? 0x5c00U
                                       : k < 8            ? 0xbc00U
                                                          : 0U);
            }
            for (int k = 0; k < 32; ++k) {
              set_swmmac16_b<WaveSize>(r.v + 8, coordinate, k, 0x5c00U);
            }
          }
          std::fill_n(r.v[16], WaveSize, 0x44444444U);
          ASSERT_EQ(call(cpu | semantics | (saturate ? RJ_ARITH_FP16_OVFL : 0), 0, r.v + 24, r.v,
                         r.v + 8, r.v + 16),
                    RJ_ARITH_SUCCESS);
          uint32_t expected = saturate && !infinity ? 0x7bdf7bdfU : 0x7c007c00U;
          for (int reg = 24; reg < 24 + 128 / WaveSize; ++reg) {
            for (int lane = 0; lane < WaveSize; ++lane) {
              EXPECT_EQ(r.v[reg][lane], expected);
            }
          }
        }
      }
    }
  }
}

// Sparse factor signs use compressed-slot parity, not selected dense-K parity.
// Select positions (1,2), reversing the parities. All sums here are exact in both formats.
template <bool Bf16, int WaveSize, bool Packed, typename Call>
void expect_sparse_modifiers(Call call) {
  constexpr int outputs = 256 / WaveSize / (Packed ? 2 : 1);
  Registers<32, WaveSize> input;
  for (int coordinate = 0; coordinate < 16; ++coordinate) {
    for (int k = 0; k < 16; ++k) {
      set_swmmac16_a<WaveSize>(input.v, coordinate, k, k % 2 ? 0x4000U : Bf16 ? 0x3f80U : 0x3c00U);
    }
    for (int k = 0; k < 32; ++k) {
      set_swmmac16_b<WaveSize>(input.v + 8, coordinate, k,
                               k % 4 == 2 ? (Bf16 ? 0x4080U : 0x4400U)
                                          : (Bf16 ? 0x3f80U : 0x3c00U));
    }
  }
  std::fill_n(input.v[16], WaveSize, 0x99999999U); // Repeated (1,2) pairs in every selector bank.
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (const auto &test : kMatrixModifierCases) {
        for (float c : {-2.0f, 2.0f}) {
          auto actual = input;
          for (int reg = 24; reg < 24 + outputs; ++reg) {
            std::fill_n(actual.v[reg], WaveSize,
                        Packed ? (c < 0 ? 0xc000c000U : 0x40004000U) : std::bit_cast<uint32_t>(c));
          }
          ASSERT_EQ(call(cpu | semantics, test.instruction_flags, actual.v + 24, actual.v,
                         actual.v + 8, actual.v + 16),
                    RJ_ARITH_SUCCESS);
          float expected = c < 0 ? test.negative_c : test.positive_c;
          for (int reg = 24; reg < 24 + outputs; ++reg) {
            for (int lane = 0; lane < WaveSize; ++lane) {
              if constexpr (Packed) {
                for (int half : {0, 1}) {
                  uint16_t value = uint16_t(actual.v[reg][lane] >> (16 * half));
                  EXPECT_EQ(Bf16 ? std::bit_cast<float>(uint32_t(value) << 16) : f16_value(value),
                            expected);
                }
              } else {
                EXPECT_EQ(actual.v[reg][lane], std::bit_cast<uint32_t>(expected));
              }
            }
          }
        }
      }
    }
  }
}

// Access an FP32 accumulator element in the gfx12 wave layout.
template <int Lanes> uint32_t &wmma_gfx12_acc(uint32_t *const *v, int row, int col) {
  static_assert(Lanes == 32 || Lanes == 64);
  constexpr int regs = 256 / Lanes;
  int lane = col + 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
  return v[row % regs][lane];
}

// One subnormal factor times one checks FP8/BF8 scale independently of the decoder.
template <bool Bf8A, bool Bf8B, int Lanes, typename Call>
void expect_wmma_fp8_subnormals(Call call) {
  for (uint64_t cpu : supported_cpu_levels()) {
    for (bool subnormal_a : {false, true}) {
      for (uint32_t sign : {0U, 0x80U}) {
        Registers<20, Lanes> r;
        for (auto *v : r.v) {
          std::fill_n(v, Lanes, 0);
        }
        set_wmma_gfx12_small<8, Lanes>(r.v, 0, 0, subnormal_a ? (sign | 1) : (Bf8A ? 0x3c : 0x38));
        set_wmma_gfx12_small<8, Lanes>(r.v + 2, 0, 0,
                                       subnormal_a ? (Bf8B ? 0x3c : 0x38) : (sign | 1));
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE, 0, r.v + 12, r.v, r.v + 2, r.v + 4),
                  RJ_ARITH_SUCCESS);
        float expected = (subnormal_a ? Bf8A : Bf8B) ? 0x1p-16f : 0x1p-9f;
        EXPECT_EQ(wmma_gfx12_acc<Lanes>(r.v + 12, 0, 0),
                  std::bit_cast<uint32_t>(sign ? -expected : expected));
      }
    }
  }
}

// Access a gfx11 FP32 accumulator element in the wave's ISA register layout.
template <int Lanes> uint32_t &wmma_gfx11_acc(uint32_t *const *v, int row, int col) {
  static_assert(Lanes == 32 || Lanes == 64);
  constexpr int groups = Lanes / 16;
  return v[row / groups][col + 16 * (row % groups)];
}

// Distinguish fixed WMMA rounding/flushing from guest controls, plus FP16
// saturation. FP16 literals agree with reviewed gfx1100/gfx1201 Wave32 captures;
// other formats and Wave64 exercise the inherited model, not additional captures.
template <bool Bf16, int Lanes, bool Gfx11, bool Packed, typename Call>
void expect_wmma_fixed_controls(Call call) {
  auto check = [&](uint16_t a0, uint16_t a1, uint16_t b, uint32_t c, uint32_t expected,
                   uint64_t guest, bool repeat = false) {
    for (uint64_t cpu : supported_cpu_levels()) {
      for (int half = 0; half < (Packed && Gfx11 ? 2 : 1); ++half) {
        Registers<32, Lanes> r;
        for (auto *reg : r.v) {
          std::fill_n(reg, Lanes, 0U);
        }
        for (int coordinate = 0; coordinate < 16; ++coordinate) {
          for (int k = 0; k < 16; ++k) {
            uint16_t a = repeat || k == 0 ? a0 : k == 1 ? a1 : 0;
            if constexpr (Gfx11) {
              set_wmma_gfx11<16, Lanes>(r.v, coordinate, k, a);
              set_wmma_gfx11<16, Lanes>(r.v + 8, coordinate, k, b);
            } else {
              set_wmma_gfx12_half<Lanes>(r.v, coordinate, k, a);
              set_wmma_gfx12_half<Lanes>(r.v + 8, coordinate, k, b);
            }
          }
        }
        for (int reg = 16; reg < 24; ++reg) {
          std::fill_n(r.v[reg], Lanes, Packed ? c * 0x10001U : c);
        }
        uint64_t instruction_flags = half ? RJ_ARITH_WMMA_HIGH_C_D : 0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | guest,
                       instruction_flags, r.v + 24, r.v, r.v + 8, r.v + 16),
                  RJ_ARITH_SUCCESS);
        uint32_t actual = Packed ? uint16_t(r.v[24][0] >> (16 * half)) : r.v[24][0];
        EXPECT_EQ(actual, expected) << cpu << '/' << guest << '/' << half;
      }
    }
  };
  // Ignored input flushing must retain tiny factors and accumulators.
  if constexpr (!Bf16) {
    if constexpr (Packed) {
      check(0x0200, 0, 0x4000, 0, 0x0400, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
    } else {
      check(0x0001, 0, 0x3c00, 0, 0x33800000, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
    }
  } else if constexpr (Packed && !Gfx11) {
    check(0x0040, 0, 0x4000, 0, 0x0080, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  }
  if constexpr (!Gfx11 || (Packed && !Bf16)) {
    check(0, 0, 0, 1, 1, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  }
  if constexpr (Bf16 && Gfx11) {
    // Inherited-model witnesses; no hardware capture is claimed for these rows.
    if constexpr (Packed) {
      check(0x3b80, 0x3b00, 0x3f80, 0x3f80, 0x3f80, RJ_ARITH_FP_ROUND_UP);
    } else {
      for (uint64_t rounding : {RJ_ARITH_FP_ROUND_ZERO, RJ_ARITH_FP_ROUND_DOWN}) {
        check(0x3380, 0x3300, 0x3f80, 0x3f800000, 0x3f800001, rounding);
      }
    }
  }
  if constexpr (Bf16 && Packed) {
    for (uint16_t sign : {uint16_t(0), uint16_t(0x8000)}) {
      check(sign | 0x7f7f, 0, 0x3f80, 0, sign | 0x7f80, RJ_ARITH_FP16_OVFL, true);
    }
  }
  if constexpr (Packed && !Bf16) {
    for (uint16_t sign : {uint16_t(0), uint16_t(0x8000)}) {
      check(sign | 0x7bff, 0, 0x3c00, 0, sign | 0x7c00, 0, true);
      check(sign | 0x7bff, 0, 0x3c00, 0, sign | 0x7bff, RJ_ARITH_FP16_OVFL, true);
    }
    check(0x0400, 0, 0x3800, 0, 0x0200, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
    check(0x1000, 0x0c00, 0x3c00, 0x3c00, 0x3c01, RJ_ARITH_FP_ROUND_ZERO);
  } else if constexpr (!Gfx11) {
    // Selected accumulator subnormals distinguish ignored output flushing.
    check(0, 0, 0, 1, 1, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
    if constexpr (Bf16 && Packed) {
      check(0x0080, 0, 0x3f00, 0, 0x0040, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
    }
  }
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

// A subnormal BF16 factor times a large factor produces a normal FP32 result.
// Loose must decline native BF16 arithmetic that would flush the factor to zero.
template <int Lanes, bool Gfx11, typename Call> void expect_bf16_loose_subnormal(Call call) {
  constexpr int factors = Gfx11 ? 8 : 4;
  for (uint64_t cpu : supported_cpu_levels()) {
    Registers<32, Lanes> r;
    for (auto *reg : r.v) {
      std::fill_n(reg, Lanes, 0U);
    }
    for (int coordinate = 0; coordinate < 16; ++coordinate) {
      if constexpr (Gfx11) {
        set_wmma_gfx11<16, Lanes>(r.v, coordinate, 0, 0x0001);
        set_wmma_gfx11<16, Lanes>(r.v + factors, coordinate, 0, 0x7f00);
      } else {
        set_wmma_gfx12_half<Lanes>(r.v, coordinate, 0, 0x0001);
        set_wmma_gfx12_half<Lanes>(r.v + factors, coordinate, 0, 0x7f00);
      }
    }
    ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE, 0, r.v + 24, r.v, r.v + factors, r.v + 16),
              RJ_ARITH_SUCCESS);
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        uint32_t actual = Gfx11 ? wmma_gfx11_acc<Lanes>(r.v + 24, row, col)
                                : wmma_gfx12_acc<Lanes>(r.v + 24, row, col);
        expect_f32(actual, 0x1p-6f, RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/1e-5, /*rtol=*/1e-5);
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
