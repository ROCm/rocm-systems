// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense integer MFMA/WMMA: signedness, lane controls, wrap/saturation and overlap.
#include "common/cpu.h"
#include "common/integer.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include <array>
#include <bit>
#include <gtest/gtest.h>
#include <random>
#include <vector>
namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_mfma_i32_16x16x32_i8_w64);

// Use an independent scalar int64 reference for the established integer model.
template <int M, int K, int Batch, bool Wmma = false, int ABlockLanes = 0,
          bool IgnoreSaturation = false, bool Gfx950 = false>
void arithmetic(Entry entry, bool saturate_mfma = false) {
  SCOPED_TRACE(::testing::Message() << M << 'x' << M << 'x' << K << " batches=" << Batch
                                    << " MFMA saturation=" << saturate_mfma);
  constexpr int Lanes = Wmma ? 32 : 64;
  constexpr int Inputs = M * K * Batch / (4 * Lanes), Outputs = M * M * Batch / Lanes;
  std::array<std::array<uint32_t, Lanes>, 2 * Inputs + 2 * Outputs> input;
  std::minstd_rand random;
  for (auto &reg : input) {
    for (auto &word : reg) {
      word = rocjitsu::arith::test::random_word(random);
    }
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    input[2 * Inputs][lane] = lane % 2 ? 0x7fffffffU : 0x80000000U;
  }
  if (saturate_mfma) {
    // Mix signed-limit accumulators with ordinary lanes so an incorrect source
    // selection remains visible even when some results clamp.
    for (int reg = 0; reg < Outputs; ++reg) {
      for (int lane = 0; lane < Lanes; ++lane) {
        const uint32_t offset = static_cast<uint32_t>((reg + lane / 4) % 16);
        const int corner = (reg + lane) % 4;
        if (corner == 0) {
          input[2 * Inputs + reg][lane] = 0x7fffffffu - offset;
        } else if (corner == 1) {
          input[2 * Inputs + reg][lane] = 0x80000000u + offset;
        }
      }
    }
  }
  const uint64_t overflow_mode = (saturate_mfma || IgnoreSaturation) ? RJ_ARITH_FP16_OVFL : 0;
  const std::vector<uint64_t> levels = supported_cpu_levels();
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
              uint64_t effective_controls = (Gfx950 && Batch == 1) ? 0 : controls;
              int size = int(effective_controls & 7);
              int permutation = int(effective_controls >> 7 & 7);
              if (a && size > 0 && size <= 4) {
                int n = ABlockLanes ? ABlockLanes : 64 >> size;
                if (n <= (64 >> size)) {
                  int group = n * (1 << size);
                  source_lane = source_lane / group * group + source_lane % n +
                                n * (int(effective_controls >> 3 & 15) % (1 << size));
                }
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
          if (saturate_mfma || (Wmma && (controls & RJ_ARITH_WMMA_CLAMP))) {
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
          ASSERT_EQ(entry(cpu | semantics | overflow_mode, controls, d, a, b, c), RJ_ARITH_SUCCESS);
          for (int reg = 0; reg < Outputs; ++reg) {
            for (int lane = 0; lane < Lanes; ++lane) {
              ASSERT_EQ(d[reg][lane], expected[reg][lane])
                  << "CPU " << cpu << " controls " << controls << " register " << reg << " lane "
                  << lane;
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
    uint32_t first, second, initial, wrapped, saturated_gfx942, saturated_gfx950;
  };
  for (const Case test : {
           Case{0x01010101u, 0x01010101u, 0x7fffffffu, 0x8000001fu, 0x7fffffffu, 0x7fffffffu},
           Case{0xffffffffu, 0xffffffffu, 0x80000000u, 0x7fffffe0u, 0x80000000u, 0x80000000u},
           // Unsuffixed gfx942 saturates the final sum; _gfx950 saturates after each K=16 pass.
           Case{0x01010101u, 0xffffffffu, 0x7ffffffcu, 0x7ffffffcu, 0x7ffffffcu, 0x7fffffefu},
           Case{0xffffffffu, 0x01010101u, 0x80000004u, 0x80000004u, 0x80000004u, 0x80000010u},
       }) {
    for (unsigned lane = 0; lane < a.size(); ++lane) {
      a[lane] = lane < 32 ? test.first : test.second;
    }
    for (uint64_t cpu : supported_cpu_levels()) {
      for (bool gfx950 : {false, true}) {
        for (bool saturate : {false, true}) {
          for (auto &reg : result) {
            reg.fill(test.initial);
          }
          const uint64_t flags =
              cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | (saturate ? RJ_ARITH_FP16_OVFL : 0);
          Entry fn = gfx950 ? rj_arith_v_mfma_i32_16x16x32_i8_w64_gfx950
                            : rj_arith_v_mfma_i32_16x16x32_i8_w64;
          ASSERT_EQ(fn(flags, 0, dst, src_a, src_b, acc), RJ_ARITH_SUCCESS);
          const uint32_t expected = !saturate ? test.wrapped
                                    : gfx950  ? test.saturated_gfx950
                                              : test.saturated_gfx942;
          for (const auto &reg : result) {
            for (uint32_t value : reg) {
              EXPECT_EQ(value, expected);
            }
          }
        }
      }
    }
  }
}

TEST(DenseIntegerMatrix, SingleBatchBlgpBroadcastPreservedOnGfx942Only) {
  std::array<uint32_t, 64> a0, a1, b0, b1, c0;
  std::array<std::array<uint32_t, 64>, 4> result;
  a0.fill(0x01010101u);
  a1.fill(0x01010101u);
  b0.fill(0u);
  b1.fill(0u);
  for (int lane = 0; lane < 16; ++lane) {
    b0[lane] = 0x01010101u;
    b1[lane] = 0x01010101u;
  }
  c0.fill(0u);
  const uint32_t *src_a[] = {a0.data(), a1.data()}, *src_b[] = {b0.data(), b1.data()};
  const uint32_t *acc[] = {c0.data(), c0.data(), c0.data(), c0.data()};
  uint32_t *dst[] = {result[0].data(), result[1].data(), result[2].data(), result[3].data()};
  constexpr uint64_t blgp4 = 4ULL << RJ_ARITH_MFMA_BLGP_SHIFT;
  for (uint64_t cpu : supported_cpu_levels()) {
    const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
    ASSERT_EQ(rj_arith_v_mfma_i32_16x16x32_i8_w64(flags, blgp4, dst, src_a, src_b, acc),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(result[0][0], 32u);
    ASSERT_EQ(rj_arith_v_mfma_i32_16x16x32_i8_w64_gfx950(flags, blgp4, dst, src_a, src_b, acc),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(result[0][0], 8u);
  }
}

TEST(DenseIntegerMatrix, MfmaK4OverflowModeSaturatesSignedResults) {
  arithmetic<32, 4, 2>(rj_arith_v_mfma_i32_32x32x4_2b_i8_w64, /*saturate_mfma=*/true);
  arithmetic<16, 4, 4>(rj_arith_v_mfma_i32_16x16x4_4b_i8_w64, /*saturate_mfma=*/true);
  arithmetic<4, 4, 16>(rj_arith_v_mfma_i32_4x4x4_16b_i8_w64, /*saturate_mfma=*/true);
}

TEST(DenseIntegerMatrix, MfmaK4PreservesPartiallyOverlappingAccumulators) {
  constexpr int Lanes = 64;
  constexpr int MaxRegisters = 32;
  struct Shape {
    Entry entry;
    int registers;
  };
  constexpr Shape shapes[] = {{rj_arith_v_mfma_i32_32x32x4_2b_i8_w64, 32},
                              {rj_arith_v_mfma_i32_16x16x4_4b_i8_w64, 16}};
  const std::array<uint32_t, Lanes> zero{};
  const std::array<const uint32_t *, 1> inputs{zero.data()};
  for (const Shape &shape : shapes) {
    const int adjacent = (shape.registers - 1) * Lanes + 16;
    const int layouts[][3] = {{0, adjacent, Lanes}, {adjacent, 0, Lanes}, {0, 0, Lanes - 16}};
    for (const auto &[c_start, d_start, stride] : layouts) {
      for (const uint64_t cpu : supported_cpu_levels()) {
        SCOPED_TRACE(::testing::Message()
                     << "registers=" << shape.registers << " cpu=" << cpu << " c=" << c_start
                     << " d=" << d_start << " stride=" << stride);
        std::array<uint32_t, 2 * MaxRegisters * Lanes> actual{};
        for (size_t i = 0; i < actual.size(); ++i) {
          actual[i] = static_cast<uint32_t>(i + 1);
        }
        auto expected = actual;
        std::array<const uint32_t *, MaxRegisters> c{};
        std::array<uint32_t *, MaxRegisters> d{};
        for (int reg = 0; reg < shape.registers; ++reg) {
          c[reg] = actual.data() + c_start + reg * stride;
          d[reg] = actual.data() + d_start + reg * stride;
          for (int lane = 0; lane < Lanes; ++lane) {
            expected[d_start + reg * stride + lane] = c[reg][lane];
          }
        }
        ASSERT_EQ(shape.entry(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 0, d.data(),
                              inputs.data(), inputs.data(), c.data()),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(actual, expected);
      }
    }
  }
}

TEST(DenseIntegerMatrix, MfmaGfx950PerPassSaturationWideShapes) {
  // Verify K=64 (4 passes of K=16) and 32x32x32 (2 passes of K=16) saturate after
  // each K=16 pass across all CPU tiers, matching the scalar reference bit-for-bit.
  std::array<std::array<uint32_t, 64>, 4> a, b;
  for (int r = 0; r < 4; ++r) {
    // Lanes 0..15 / 0..31 hold +1; later lanes hold -1 so pass 0 saturates INT_MAX
    // and subsequent passes step down below INT_MAX.
    for (int lane = 0; lane < 64; ++lane) {
      a[r][lane] = lane < 16 ? 0x01010101u : 0xffffffffu;
      b[r][lane] = 0x01010101u;
    }
  }
  const uint32_t *src_a[] = {a[0].data(), a[1].data(), a[2].data(), a[3].data()};
  const uint32_t *src_b[] = {b[0].data(), b[1].data(), b[2].data(), b[3].data()};
  auto check = [&]<int OutputRegs>(Entry fn) {
    std::array<std::array<uint32_t, 64>, OutputRegs> ref;
    std::array<uint32_t *, OutputRegs> ref_d;
    std::array<const uint32_t *, OutputRegs> ref_c;
    for (int r = 0; r < OutputRegs; ++r) {
      ref[r].fill(0x7ffffffcu);
      ref_d[r] = ref[r].data();
      ref_c[r] = ref[r].data();
    }
    const uint64_t base_flags = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL;
    ASSERT_EQ(fn(RJ_ARITH_CPU_BASELINE | base_flags, 0, ref_d.data(), src_a, src_b, ref_c.data()),
              RJ_ARITH_SUCCESS);
    for (uint64_t cpu : supported_cpu_levels()) {
      std::array<std::array<uint32_t, 64>, OutputRegs> out;
      std::array<uint32_t *, OutputRegs> out_d;
      std::array<const uint32_t *, OutputRegs> out_c;
      for (int r = 0; r < OutputRegs; ++r) {
        out[r].fill(0x7ffffffcu);
        out_d[r] = out[r].data();
        out_c[r] = out[r].data();
      }
      ASSERT_EQ(fn(cpu | base_flags, 0, out_d.data(), src_a, src_b, out_c.data()),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(out, ref) << "cpu=" << cpu;
    }
  };
  check.template operator()<4>(rj_arith_v_mfma_i32_16x16x64_i8_w64);
  check.template operator()<16>(rj_arith_v_mfma_i32_32x32x32_i8_w64);
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
TEST(DenseIntegerMatrix, mfma_i32_32x32x16_i8_w64_gfx950) {
  arithmetic<32, 16, 1, false, 0, false, true>(rj_arith_v_mfma_i32_32x32x16_i8_w64_gfx950);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x32_i8_w64) {
  arithmetic<16, 32, 1>(rj_arith_v_mfma_i32_16x16x32_i8_w64);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x32_i8_w64_gfx950) {
  arithmetic<16, 32, 1, false, 0, false, true>(rj_arith_v_mfma_i32_16x16x32_i8_w64_gfx950);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x64_i8_w64) {
  arithmetic<16, 64, 1, false, 0, false, true>(rj_arith_v_mfma_i32_16x16x64_i8_w64);
}
TEST(DenseIntegerMatrix, mfma_i32_32x32x32_i8_w64) {
  arithmetic<32, 32, 1, false, 0, false, true>(rj_arith_v_mfma_i32_32x32x32_i8_w64);
}
TEST(DenseIntegerMatrix, wmma_i32_16x16x64_iu8_w32) {
  arithmetic<16, 64, 1, true>(rj_arith_v_wmma_i32_16x16x64_iu8_w32);
}
TEST(DenseIntegerMatrix, mfma_i32_32x32x4_2b_i8_w64_gfx908) {
  arithmetic<32, 4, 2, false, 0, true>(rj_arith_v_mfma_i32_32x32x4_2b_i8_w64_gfx908);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x4_4b_i8_w64_gfx908) {
  arithmetic<16, 4, 4, false, 0, true>(rj_arith_v_mfma_i32_16x16x4_4b_i8_w64_gfx908);
}
TEST(DenseIntegerMatrix, mfma_i32_4x4x4_16b_i8_w64_gfx908) {
  arithmetic<4, 4, 16, false, 0, true>(rj_arith_v_mfma_i32_4x4x4_16b_i8_w64_gfx908);
}
TEST(DenseIntegerMatrix, mfma_i32_32x32x8_i8_w64_gfx908) {
  arithmetic<32, 8, 1, false, 0, true>(rj_arith_v_mfma_i32_32x32x8_i8_w64_gfx908);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x16_i8_w64_gfx908) {
  arithmetic<16, 16, 1, false, 0, true>(rj_arith_v_mfma_i32_16x16x16_i8_w64_gfx908);
}
TEST(DenseIntegerMatrix, mfma_i32_32x32x4_2b_i8_w64_gfx90a) {
  arithmetic<32, 4, 2, false, 32, true>(rj_arith_v_mfma_i32_32x32x4_2b_i8_w64_gfx90a);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x4_4b_i8_w64_gfx90a) {
  arithmetic<16, 4, 4, false, 16, true>(rj_arith_v_mfma_i32_16x16x4_4b_i8_w64_gfx90a);
}
TEST(DenseIntegerMatrix, mfma_i32_4x4x4_16b_i8_w64_gfx90a) {
  arithmetic<4, 4, 16, false, 4, true>(rj_arith_v_mfma_i32_4x4x4_16b_i8_w64_gfx90a);
}
TEST(DenseIntegerMatrix, mfma_i32_32x32x8_i8_w64_gfx90a) {
  arithmetic<32, 8, 1, false, 32, true>(rj_arith_v_mfma_i32_32x32x8_i8_w64_gfx90a);
}
TEST(DenseIntegerMatrix, mfma_i32_16x16x16_i8_w64_gfx90a) {
  arithmetic<16, 16, 1, false, 16, true>(rj_arith_v_mfma_i32_16x16x16_i8_w64_gfx90a);
}
} // namespace
