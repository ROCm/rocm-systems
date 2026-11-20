// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse integer matrix metadata, wrapping/clamping and tied-accumulator aliasing.
#include "common/cpu.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include <array>
#include <bit>
#include <gtest/gtest.h>
#include <random>
namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_smfmac_i32_16x16x64_i8_w64);

// The reference follows the inherited scalar executor, using int64 sums.
template <int M, int K, bool Wmma = false> void arithmetic(Entry entry) {
  constexpr int L = Wmma ? 32 : 64, A = M * K / (8 * L), B = 2 * A, D = M * M / L, I = Wmma ? 2 : 1;
  constexpr int Index = A + B, Dest = Index + I, Count = Dest + 2 * D;
  std::array<std::array<uint32_t, L>, Count> input;
  std::minstd_rand random;
  for (auto &reg : input) {
    for (auto &word : reg) {
      word = uint32_t(random() >> 15) << 16 | (random() >> 15);
    }
  }
  // Each metadata nibble selects an increasing pair; both register words vary.
  constexpr uint32_t pairs[] = {4, 8, 12, 9, 13, 14};
  for (int reg = 0; reg < I; ++reg) {
    for (int lane = 0; lane < L; ++lane) {
      uint32_t word = 0;
      for (int n = 0; n < 8; ++n) {
        word |= pairs[(random() >> 16) % 6] << (4 * n);
      }
      input[Index + reg][lane] = word;
    }
  }
  for (int lane = 0; lane < L; ++lane) {
    input[Dest][lane] = lane % 2 ? 0x7fffffffU : 0x80000000U;
  }
  const auto levels = supported_cpu_levels();
  for (uint64_t controls :
       Wmma ? std::array<uint64_t, 4>{0, 3, 64, 65} : std::array<uint64_t, 4>{0, 8, 1, 127}) {
    for (int aliasing_offset : {0, A, Index, Dest}) {
      // D is an input too: the reference must use the accumulator at the selected alias.
      uint32_t expected[D][L];
      for (int row = 0; row < M; ++row) {
        for (int col = 0; col < M; ++col) {
          constexpr int groups = (64 + M * M / 4 - 1) / (M * M / 4);
          int out_reg = Wmma ? row % 8 : row / (4 * (64 / M)) * 4 + row % 4;
          int out_lane = Wmma ? col + 16 * (row / 8) : row / 4 % (64 / M) * groups * M + col;
          int64_t sum = std::bit_cast<int32_t>(input[aliasing_offset + out_reg][out_lane]);
          auto byte = [&](int base, int element, int lane, bool signed_value) {
            int value = int(input[base + element / 4][lane] >> (8 * (element % 4)) & 255);
            return signed_value && value >= 128 ? value - 256 : value;
          };
          for (int q = 0; q < K / 4; ++q) {
            for (int slot = 0; slot < 2; ++slot) {
              int ck = 2 * q + slot, al, ae, selected;
              if constexpr (Wmma) {
                al = row + 16 * (ck / 16 % 2);
                ae = ck / 32 * 16 + ck % 16;
                selected =
                    int(input[Index + (ck % 32) / 16][row + 16 * (ck / 32)] >> (2 * (ck % 16)) & 3);
              } else {
                int nibble;
                if constexpr (M * K < 2048) {
                  al = ((q % (K / 8)) / 2) * M + row;
                  nibble = 2 * (q / (K / 8)) + q % 2;
                  ae = 4 * (q / (K / 8)) + (2 * q + slot) % 4;
                } else if constexpr (M == 16) {
                  al = (2 * (q / 16) + q / 4 % 2) * 16 + row;
                  nibble = 2 * (q / 8 % 2) + 4 * (q % 4 / 2) + q % 2;
                  ae = 4 * (2 * (q / 2 % 2) + q / 8 % 2) + (2 * q + slot) % 4;
                } else {
                  al = q / 8 * 32 + row;
                  nibble = q % 2 + 2 * (q / 4 % 2) + 4 * (q / 2 % 2);
                  ae = 4 * (2 * (q / 2 % 2) + q / 4 % 2) + (2 * q + slot) % 4;
                }
                int bank = M * K < 2048 && !(controls & 3) ? 16 * int(controls >> 3 & 1) : 0;
                selected = int(input[Index][al] >> (bank + 4 * nibble + 2 * slot) & 3);
              }
              int k = 4 * q + selected;
              int bl, be;
              if constexpr (Wmma) {
                bl = col + 16 * (k / 32 % 2);
                be = k % 32 + 32 * (k / 64);
              } else {
                bl = (M == 16 ? 16 * (k % 32 / 8) : 32 * (k % 16 / 8)) + col;
                be = 8 * (k / (M == 16 ? 32 : 16)) + k % 8;
              }
              sum += byte(0, ae, al, !Wmma || (controls & 1)) *
                     byte(A, be, bl, !Wmma || (controls & 2));
            }
          }
          if (Wmma && (controls & RJ_ARITH_WMMA_CLAMP)) {
            sum = sum > 0x7fffffffLL ? 0x7fffffffLL : sum < -0x80000000LL ? -0x80000000LL : sum;
          }
          expected[out_reg][out_lane] = uint32_t(sum);
        }
      }
      for (uint64_t cpu : levels) {
        for (uint64_t semantics : semantics_modes()) {
          auto storage = input;
          const uint32_t *a[A], *b[B], *index[I];
          uint32_t *d[D];
          for (int reg = 0; reg < A; ++reg) {
            a[reg] = storage[reg].data();
          }
          for (int reg = 0; reg < B; ++reg) {
            b[reg] = storage[A + reg].data();
          }
          for (int reg = 0; reg < I; ++reg) {
            index[reg] = storage[Index + reg].data();
          }
          for (int reg = 0; reg < D; ++reg) {
            d[reg] = storage[aliasing_offset + reg].data();
          }
          ASSERT_EQ(entry(cpu | semantics, controls, d, a, b, index), RJ_ARITH_SUCCESS);
          for (int reg = 0; reg < D; ++reg) {
            for (int lane = 0; lane < L; ++lane) {
              ASSERT_EQ(d[reg][lane], expected[reg][lane]);
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
TEST(SparseIntegerMatrix, smfmac_i32_16x16x64_i8_w64) {
  arithmetic<16, 64>(rj_arith_v_smfmac_i32_16x16x64_i8_w64);
}
TEST(SparseIntegerMatrix, smfmac_i32_32x32x32_i8_w64) {
  arithmetic<32, 32>(rj_arith_v_smfmac_i32_32x32x32_i8_w64);
}
TEST(SparseIntegerMatrix, smfmac_i32_16x16x128_i8_w64) {
  arithmetic<16, 128>(rj_arith_v_smfmac_i32_16x16x128_i8_w64);
}
TEST(SparseIntegerMatrix, smfmac_i32_32x32x64_i8_w64) {
  arithmetic<32, 64>(rj_arith_v_smfmac_i32_32x32x64_i8_w64);
}
TEST(SparseIntegerMatrix, swmmac_i32_16x16x128_iu8_w32) {
  arithmetic<16, 128, true>(rj_arith_v_swmmac_i32_16x16x128_iu8_w32);
}
} // namespace
