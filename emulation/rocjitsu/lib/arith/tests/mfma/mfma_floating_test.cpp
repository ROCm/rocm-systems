// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense floating MFMA contracts, layout, source controls and whole-wave aliasing.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_mfma_f32_16x16x16_f16_w64);

// Compare a full instruction against the established approximate reduction,
// including batched layouts and sources that overlap the destination.
template <int M, int K, int Batch, bool Bf16, bool FullF32 = false> void arithmetic(Entry entry) {
  constexpr int InputRegs = M * K * Batch / (FullF32 ? 64 : 128);
  constexpr int OutputRegs = M * M * Batch / 64;
  constexpr int Count = 2 * InputRegs + 2 * OutputRegs;
  std::array<std::array<uint32_t, 64>, Count> input;
  std::minstd_rand random;
  for (int r = 0; r < Count; ++r) {
    for (auto &word : input[r]) {
      if (r < 2 * InputRegs && !FullF32) {
        auto half = [&] { return Bf16 ? random_bf16_bits(random) : random_f16_bits(random); };
        word = uint32_t(half()) | uint32_t(half()) << 16;
      } else {
        word = random_f32_bits(random);
      }
    }
  }
  // One NaN and infinity in each factor also exercise SIMD exceptional lanes.
  for (int r : {0, InputRegs}) {
    input[r][0] = FullF32 ? kF32QuietNaNBits : Bf16 ? kBF16QuietNaNBits : kF16QuietNaNBits;
    input[r][1] = FullF32 ? kF32PositiveInfinityBits
                  : Bf16  ? kBF16PositiveInfinityBits
                          : kF16PositiveInfinityBits;
  }
  const auto levels = supported_cpu_levels();
  for (uint64_t controls :
       {0ULL, 1ULL | (1ULL << 3) | (3ULL << 7), 4ULL | (13ULL << 3) | (7ULL << 7), 7ULL}) {
    std::array<std::array<float, 64>, OutputRegs> expected{};
    for (int batch = 0; batch < Batch; ++batch) {
      for (int row = 0; row < M; ++row) {
        for (int col = 0; col < M; ++col) {
          const int groups = (64 + M * M / 4 - 1) / (M * M / 4);
          const int reg = batch * (M * M / 64) + row / (4 * (64 / M)) * 4 + row % 4;
          const int lane = batch % groups * M + row / 4 % (64 / M) * groups * M + col;
          float sum = std::bit_cast<float>(input[2 * InputRegs + reg][lane]);
          // Reference indexing follows the legacy GFX9 executor; do not call
          // the implementation's layout helpers from this test.
          auto read = [&](int operand, int index, int k) {
            const int per_lane = K / (64 / (M * Batch));
            int source_lane = batch * M + k / per_lane * M * Batch + index;
            int cbsz = int(controls & 7), blgp = int(controls >> 7 & 7);
            if (operand == 0 && cbsz > 0 && cbsz <= 4) {
              int size = 64 >> cbsz;
              source_lane = source_lane % size + size * (int(controls >> 3 & 15) % (1 << cbsz));
            }
            if (operand == 1 && blgp) {
              if (blgp <= 2) {
                source_lane = source_lane % 32 + (blgp - 1) * 32;
              } else if (blgp == 3) {
                source_lane = (source_lane + 16) % 64;
              } else {
                source_lane = source_lane % 16 + (blgp - 4) * 16;
              }
            }
            if constexpr (FullF32) {
              return std::bit_cast<float>(input[operand * InputRegs + k % per_lane][source_lane]);
            } else {
              uint16_t value = uint16_t(
                  input[operand * InputRegs + k % per_lane / 2][source_lane] >> (16 * (k % 2)));
              return Bf16 ? std::bit_cast<float>(uint32_t(value) << 16) : float(half_value(value));
            }
          };
          for (int k = 0; k < K; ++k) {
            sum = std::fma(read(0, row, k), read(1, col, k), sum);
          }
          expected[reg][lane] = sum;
        }
      }
    }
    for (uint64_t cpu : levels) {
      for (uint64_t semantics : semantics_modes()) {
        for (int aliasing_offset : {0, InputRegs, 2 * InputRegs, 2 * InputRegs + OutputRegs}) {
          auto storage = input;
          const uint32_t *a[InputRegs], *b[InputRegs], *c[OutputRegs];
          uint32_t *d[OutputRegs];
          for (int r = 0; r < InputRegs; ++r) {
            a[r] = storage[r].data();
            b[r] = storage[InputRegs + r].data();
          }
          for (int r = 0; r < OutputRegs; ++r) {
            c[r] = storage[2 * InputRegs + r].data();
            d[r] = storage[aliasing_offset + r].data();
          }
          int status = entry(cpu | semantics, controls, d, a, b, c);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            EXPECT_EQ(storage, input);
            continue;
          }
          ASSERT_EQ(status, RJ_ARITH_SUCCESS);
          // Full FP32 inputs reach magnitude 128. Cancellation exposes up to
          // 0.001 absolute error between the unfused baseline and fused reference.
          // Half-input products are exact in FP32 and retain the tighter bound.
          for (int r = 0; r < OutputRegs; ++r) {
            for (int lane = 0; lane < 64; ++lane) {
              expect_f32(d[r][lane], expected[r][lane], RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/FullF32 ? 0.002 : 1e-5f,
                         /*rtol=*/1e-5f);
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 10, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}

TEST(MfmaHalf, mfma_f32_32x32x4_2b_f16) {
  arithmetic<32, 4, 2, false>(rj_arith_v_mfma_f32_32x32x4_2b_f16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x4_4b_f16) {
  arithmetic<16, 4, 4, false>(rj_arith_v_mfma_f32_16x16x4_4b_f16_w64);
}
TEST(MfmaHalf, mfma_f32_4x4x4_16b_f16) {
  arithmetic<4, 4, 16, false>(rj_arith_v_mfma_f32_4x4x4_16b_f16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x8_f16) {
  arithmetic<32, 8, 1, false>(rj_arith_v_mfma_f32_32x32x8_f16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x16_f16) {
  arithmetic<16, 16, 1, false>(rj_arith_v_mfma_f32_16x16x16_f16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x32_f16) {
  arithmetic<16, 32, 1, false>(rj_arith_v_mfma_f32_16x16x32_f16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x16_f16) {
  arithmetic<32, 16, 1, false>(rj_arith_v_mfma_f32_32x32x16_f16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x4_2b_bf16) {
  arithmetic<32, 4, 2, true>(rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x4_4b_bf16) {
  arithmetic<16, 4, 4, true>(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_4x4x4_16b_bf16) {
  arithmetic<4, 4, 16, true>(rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x8_bf16) {
  arithmetic<32, 8, 1, true>(rj_arith_v_mfma_f32_32x32x8_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x16_bf16) {
  arithmetic<16, 16, 1, true>(rj_arith_v_mfma_f32_16x16x16_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x32_bf16) {
  arithmetic<16, 32, 1, true>(rj_arith_v_mfma_f32_16x16x32_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x16_bf16) {
  arithmetic<32, 16, 1, true>(rj_arith_v_mfma_f32_32x32x16_bf16_w64);
}

TEST(MfmaFullF32, mfma_f32_16x16x8_xf32) {
  arithmetic<16, 8, 1, false, true>(rj_arith_v_mfma_f32_16x16x8_xf32_w64);
}
TEST(MfmaFullF32, mfma_f32_32x32x4_xf32) {
  arithmetic<32, 4, 1, false, true>(rj_arith_v_mfma_f32_32x32x4_xf32_w64);
}
TEST(MfmaF32, mfma_f32_32x32x1_2b_f32_w64) {
  arithmetic<32, 1, 2, false, true>(rj_arith_v_mfma_f32_32x32x1_2b_f32_w64);
}
TEST(MfmaF32, mfma_f32_16x16x1_4b_f32_w64) {
  arithmetic<16, 1, 4, false, true>(rj_arith_v_mfma_f32_16x16x1_4b_f32_w64);
}
TEST(MfmaF32, mfma_f32_4x4x1_16b_f32_w64) {
  arithmetic<4, 1, 16, false, true>(rj_arith_v_mfma_f32_4x4x1_16b_f32_w64);
}
TEST(MfmaF32, mfma_f32_32x32x2_f32_w64) {
  arithmetic<32, 2, 1, false, true>(rj_arith_v_mfma_f32_32x32x2_f32_w64);
}
TEST(MfmaF32, mfma_f32_16x16x4_f32_w64) {
  arithmetic<16, 4, 1, false, true>(rj_arith_v_mfma_f32_16x16x4_f32_w64);
}
} // namespace
