// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense integer MFMA/WMMA: signedness, lane controls, wrap/saturation and overlap.
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
using Entry = decltype(&rj_arith_v_mfma_i32_16x16x32_i8_w64);

// Use an independent scalar int64 reference for the established integer model.
template <int M, int K, int Batch, bool Wmma = false> void arithmetic(Entry entry) {
  constexpr int Lanes = Wmma ? 32 : 64;
  constexpr int Inputs = M * K * Batch / (4 * Lanes), Outputs = M * M * Batch / Lanes;
  std::array<std::array<uint32_t, Lanes>, 2 * Inputs + 2 * Outputs> input;
  std::minstd_rand random;
  for (auto &reg : input) {
    for (auto &word : reg) {
      word = uint32_t(random() >> 15) << 16 | (random() >> 15);
    }
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    input[2 * Inputs][lane] = lane % 2 ? 0x7fffffffU : 0x80000000U;
  }
  const auto levels = supported_cpu_levels();
  std::array<uint64_t, 8> controls_set = Wmma ? std::array<uint64_t, 8>{0, 1, 2, 3, 64, 65, 66, 67}
                                              : std::array<uint64_t, 8>{0,
                                                                        1 | (1 << 3),
                                                                        2 | (3 << 3) | (1 << 7),
                                                                        3 | (7 << 3) | (2 << 7),
                                                                        4 | (13 << 3) | (3 << 7),
                                                                        4 << 7,
                                                                        5 << 7,
                                                                        7 | (7 << 7)};
  for (uint64_t controls : controls_set) {
    uint32_t expected[Outputs][Lanes];
    for (int batch = 0; batch < Batch; ++batch) {
      for (int row = 0; row < M; ++row) {
        for (int col = 0; col < M; ++col) {
          int reg, lane;
          if constexpr (Wmma) {
            reg = row % 8;
            lane = col + 16 * (row / 8);
          } else {
            constexpr int groups = (64 + M * M / 4 - 1) / (M * M / 4);
            reg = batch * (M * M / 64) + row / (4 * (64 / M)) * 4 + row % 4;
            lane = batch % groups * M + row / 4 % (64 / M) * groups * M + col;
          }
          auto read = [&](bool a, int i, int k) {
            int local, source_lane;
            if constexpr (Wmma) {
              local = k / 32 * 16 + k % 16;
              source_lane = i + 16 * (k / 16 % 2);
            } else {
              constexpr int per_lane = K / (64 / (M * Batch));
              local = k % per_lane;
              source_lane = batch * M + k / per_lane * M * Batch + i;
              int size = int(controls & 7), permutation = int(controls >> 7 & 7);
              if (a && size > 0 && size <= 4) {
                int n = 64 >> size;
                source_lane = source_lane % n + n * (int(controls >> 3 & 15) % (1 << size));
              }
              if (!a && permutation) {
                if (permutation <= 2) {
                  source_lane = source_lane % 32 + 32 * (permutation - 1);
                } else if (permutation == 3) {
                  source_lane = (source_lane + 16) % 64;
                } else {
                  source_lane = source_lane % 16 + 16 * (permutation - 4);
                }
              }
            }
            int value =
                int(input[(a ? 0 : Inputs) + local / 4][source_lane] >> (8 * (local % 4)) & 0xff);
            if ((!Wmma || (controls & (a ? 1 : 2))) && value >= 128) {
              value -= 256;
            }
            return value;
          };
          int64_t sum = std::bit_cast<int32_t>(input[2 * Inputs + reg][lane]);
          for (int k = 0; k < K; ++k) {
            sum += read(true, row, k) * read(false, col, k);
          }
          if (Wmma && (controls & RJ_ARITH_WMMA_CLAMP)) {
            sum = sum > 0x7fffffffLL ? 0x7fffffffLL : sum < -0x80000000LL ? -0x80000000LL : sum;
          }
          expected[reg][lane] = uint32_t(sum);
        }
      }
    }
    for (uint64_t cpu : levels) {
      for (uint64_t semantics : semantics_modes()) {
        for (int aliasing_offset : {0, Inputs, 2 * Inputs, 2 * Inputs + Outputs}) {
          auto storage = input;
          const uint32_t *a[Inputs], *b[Inputs], *c[Outputs];
          uint32_t *d[Outputs];
          for (int reg = 0; reg < Inputs; ++reg) {
            a[reg] = storage[reg].data();
            b[reg] = storage[Inputs + reg].data();
          }
          for (int reg = 0; reg < Outputs; ++reg) {
            c[reg] = storage[2 * Inputs + reg].data();
            d[reg] = storage[aliasing_offset + reg].data();
          }
          ASSERT_EQ(entry(cpu | semantics, controls, d, a, b, c), RJ_ARITH_SUCCESS);
          for (int reg = 0; reg < Outputs; ++reg) {
            for (int lane = 0; lane < Lanes; ++lane) {
              ASSERT_EQ(d[reg][lane], expected[reg][lane])
                  << "CPU " << cpu << " controls " << controls;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
TEST(DenseIntegerMatrix, MfmaOverflowModeSaturatesSignedResults) {
  std::array<uint32_t, 64> a, b;
  std::array<std::array<uint32_t, 64>, 4> result;
  const uint32_t *src_a[] = {a.data(), a.data()}, *src_b[] = {b.data(), b.data()};
  uint32_t *dst[] = {result[0].data(), result[1].data(), result[2].data(), result[3].data()};
  const uint32_t *acc[] = {dst[0], dst[1], dst[2], dst[3]};
  b.fill(0x01010101u);
  struct Case {
    uint32_t first, second, initial, wrapped, saturated;
  };
  for (const Case test : {
           Case{0x01010101u, 0x01010101u, 0x7fffffffu, 0x8000001fu, 0x7fffffffu},
           Case{0xffffffffu, 0xffffffffu, 0x80000000u, 0x7fffffe0u, 0x80000000u},
           // The model saturates the final sum: the last 16 products cancel the first 16.
           Case{0x01010101u, 0xffffffffu, 0x7ffffffcu, 0x7ffffffcu, 0x7ffffffcu},
           Case{0xffffffffu, 0x01010101u, 0x80000004u, 0x80000004u, 0x80000004u},
       }) {
    for (unsigned lane = 0; lane < a.size(); ++lane) {
      a[lane] = lane < 32 ? test.first : test.second;
    }
    for (uint64_t cpu : supported_cpu_levels()) {
      for (bool saturate : {false, true}) {
        for (auto &reg : result) {
          reg.fill(test.initial);
        }
        const uint64_t flags =
            cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | (saturate ? RJ_ARITH_FP16_OVFL : 0);
        ASSERT_EQ(rj_arith_v_mfma_i32_16x16x32_i8_w64(flags, 0, dst, src_a, src_b, acc),
                  RJ_ARITH_SUCCESS);
        for (const auto &reg : result) {
          for (uint32_t value : reg) {
            EXPECT_EQ(value, saturate ? test.saturated : test.wrapped);
          }
        }
      }
    }
  }
}

TEST(DenseIntegerMatrix, mfma_i32_32x32x4_2b_i8_w64) {
  arithmetic<32, 4, 2>(rj_arith_v_mfma_i32_32x32x4_2b_i8_w64);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x4_4b_i8_w64) {
  arithmetic<16, 4, 4>(rj_arith_v_mfma_i32_16x16x4_4b_i8_w64);
}
TEST(DenseIntegerMatrix, mfma_i32_4x4x4_16b_i8_w64) {
  arithmetic<4, 4, 16>(rj_arith_v_mfma_i32_4x4x4_16b_i8_w64);
}
TEST(DenseIntegerMatrix, mfma_i32_32x32x16_i8_w64) {
  arithmetic<32, 16, 1>(rj_arith_v_mfma_i32_32x32x16_i8_w64);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x32_i8_w64) {
  arithmetic<16, 32, 1>(rj_arith_v_mfma_i32_16x16x32_i8_w64);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x64_i8_w64) {
  arithmetic<16, 64, 1>(rj_arith_v_mfma_i32_16x16x64_i8_w64);
}
TEST(DenseIntegerMatrix, mfma_i32_32x32x32_i8_w64) {
  arithmetic<32, 32, 1>(rj_arith_v_mfma_i32_32x32x32_i8_w64);
}
TEST(DenseIntegerMatrix, wmma_i32_16x16x64_iu8_w32) {
  arithmetic<16, 64, 1, true>(rj_arith_v_wmma_i32_16x16x64_iu8_w32);
}
} // namespace
