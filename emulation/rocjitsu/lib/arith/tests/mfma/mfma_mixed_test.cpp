// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Mixed-format MFMA packing, all format pairs, CPU dispatch and overlapping operands.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/matrix_formats.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

namespace {
using namespace rocjitsu::arith::test;

// Construct matrices logically, pack them into independent format layouts and compare the product.
template <int M, int K, bool Scaled = false, typename Entry>
void arithmetic(Entry entry, bool finite_fp4 = false) {
  constexpr int D = M * M / 64, Scale = 16 + 2 * D;
  const auto levels = supported_cpu_levels();
  const int first_format = finite_fp4 ? int(RJ_ARITH_MATRIX_FORMAT_FP4) : 0;
  for (int fa = first_format; fa < 5; ++fa) {
    for (int fb = first_format; fb < 5; ++fb) {
      std::array<std::array<uint32_t, 64>, Scale + D> input{};
      float left[M][K], right[M][K];
      std::minstd_rand random;
      auto fill = [&](int base, int format, auto &values) {
        int width = format < 2 ? 8 : format < 4 ? 6 : 4;
        for (int row = 0; row < M; ++row) {
          for (int k = 0; k < K; ++k) {
            int code = int(random() >> 23) & ((1 << width) - 1);
            // Keep byte magnitudes moderate; explicit special values exercise NaN/Inf separately.
            if (width == 8) {
              code = (code & 128) | (code & 63);
            }
            if (row == 0 && k == 0 && width == 8) {
              code = format == 0 ? 127 : 124;
            }
            int lane, local;
            if (width == 8) {
              lane = (k / 16 % (64 / M)) * M + row;
              local = k / (16 * (64 / M)) * 16 + k % 16;
            } else {
              lane = k / 32 * M + row;
              local = k % 32;
            }
            // Bit-by-bit packing deliberately avoids sharing the production cross-word extraction.
            for (int bit = 0; bit < width; ++bit) {
              int offset = local * width + bit;
              input[base + offset / 32][lane] |= uint32_t((code >> bit) & 1) << (offset % 32);
            }
            values[row][k] = decode_matrix_format(code, format);
          }
        }
        return M * K * width / (64 * 32);
      };
      const int a_regs = fill(0, fa, left);
      const int b_regs = fill(8, fb, right);
      for (int reg = 0; reg < D; ++reg) {
        for (auto &word : input[16 + reg]) {
          word = std::bit_cast<uint32_t>(float(int(random() >> 8) - 0x400000) * 0x1p-23f);
        }
      }
      // Select non-low bytes, with independent scales per row and reduction block.
      for (int lane = 0; lane < 64; ++lane) {
        input[Scale][lane] = uint32_t(126 + lane % 3) << 8;
        input[Scale + 1][lane] = uint32_t(126 + lane % 4) << 16;
      }
      if (!finite_fp4) {
        input[Scale][1] = 255U << 8;
      }
      uint64_t controls = uint64_t(fa) | (uint64_t(fb) << 3);
      if constexpr (Scaled) {
        controls |= (1ULL << 6) | (2ULL << 8);
        if (fa & 1) {
          controls |= RJ_ARITH_MFMA_SCALE_ABS_C;
        }
        if (fb & 1) {
          controls |= RJ_ARITH_MFMA_SCALE_NEG_C;
        }
      }
      float expected[D][64];
      for (int row = 0; row < M; ++row) {
        for (int col = 0; col < M; ++col) {
          int reg = row / (4 * (64 / M)) * 4 + row % 4, lane = row / 4 % (64 / M) * M + col;
          double sum = std::bit_cast<float>(input[16 + reg][lane]);
          if constexpr (Scaled) {
            float acc = float(sum);
            if (fa & 1) {
              acc = std::abs(acc);
            }
            if (fb & 1) {
              acc = -acc;
            }
            // The inherited model rounds each fused block before exponent scaling and C addition.
            for (int block = 0; block < K / 32; ++block) {
              float partial = 0;
              for (int k = block * 32; k < (block + 1) * 32; ++k) {
                partial = std::fma(left[row][k], right[col][k], partial);
              }
              int ea = int(input[Scale][block * M + row] >> 8);
              int eb = int(input[Scale + 1][block * M + col] >> 16);
              acc += ea == 255 || eb == 255 ? std::numeric_limits<float>::quiet_NaN()
                                            : std::ldexp(partial, ea + eb - 254);
            }
            sum = acc;
          } else {
            for (int k = 0; k < K; ++k) {
              sum += double(left[row][k]) * right[col][k];
            }
          }
          expected[reg][lane] = float(sum);
        }
      }
      for (int aliasing_offset : {0, 8, 16, 16 + D, Scale}) {
        for (uint64_t cpu : levels) {
          for (uint64_t semantics : semantics_modes()) {
            auto storage = input;
            // Compact operands provide only the VGPRs selected by their individual formats.
            const uint32_t *a[8]{}, *b[8]{}, *c[D];
            uint32_t *d[D];
            for (int i = 0; i < a_regs; ++i) {
              a[i] = storage[i].data();
            }
            for (int i = 0; i < b_regs; ++i) {
              b[i] = storage[8 + i].data();
            }
            for (int i = 0; i < D; ++i) {
              c[i] = storage[16 + i].data();
              d[i] = storage[aliasing_offset + i].data();
            }
            int status;
            if constexpr (Scaled) {
              status = entry(cpu | semantics, controls, d, a, b, c, storage[Scale].data(),
                             storage[Scale + 1].data());
            } else {
              status = entry(cpu | semantics, controls, d, a, b, c);
            }
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              EXPECT_EQ(storage, input);
              continue;
            }
            ASSERT_EQ(status, RJ_ARITH_SUCCESS);
            for (int reg = 0; reg < D; ++reg) {
              for (int lane = 0; lane < 64; ++lane) {
                expect_f32(d[reg][lane], expected[reg][lane], RJ_ARITH_SEMANTICS_LOOSE,
                           /*atol=*/1e-4f, /*rtol=*/1e-5f);
              }
            }
          }
        }
      }
    }
  }
  if constexpr (Scaled) {
    EXPECT_EQ(
        entry(RJ_ARITH_SEMANTICS_LOOSE, 5, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
        RJ_ARITH_ERROR_INVALID_FLAGS);
  } else {
    EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 5, nullptr, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_INVALID_FLAGS);
  }
}
TEST(MfmaMixed, F32_16x16x128) { arithmetic<16, 128>(rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64); }
TEST(MfmaMixed, F32_32x32x64) { arithmetic<32, 64>(rj_arith_v_mfma_f32_32x32x64_f8f6f4_w64); }
TEST(MfmaMixed, ScaleF32_16x16x128) {
  arithmetic<16, 128, true>(rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64);
  arithmetic<16, 128, true>(rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64, /*finite_fp4=*/true);
}
TEST(MfmaMixed, ScaleF32_32x32x64) {
  arithmetic<32, 64, true>(rj_arith_v_mfma_scale_f32_32x32x64_f8f6f4_w64);
  arithmetic<32, 64, true>(rj_arith_v_mfma_scale_f32_32x32x64_f8f6f4_w64, /*finite_fp4=*/true);
}

TEST(MfmaMixed, ScaleClampsFiniteOverflowAndPreservesNonfiniteInputs) {
  struct Column {
    uint8_t b, scale_b;
    uint32_t c, normal, clamped;
  };
  constexpr uint32_t nan = 0x7fc00000;
  constexpr Column columns[] = {
      {0x3c, 127, 0, 0x7f800000, 0x7f7fffff}, // BF8 +1 times a large scale
      {0xbc, 127, 0, 0xff800000, 0xff7fffff}, // BF8 -1 times a large scale
      {0x7c, 127, 0, 0x7f800000, 0x7f800000}, // BF8 +infinity
      {0xfc, 127, 0, 0xff800000, 0xff800000}, // BF8 -infinity
      {0x7e, 127, 0, nan, nan},               // BF8 NaN
      {0, 127, 0x7f800000, 0x7f800000, 0x7f800000},
      {0, 127, nan, nan, nan},
      {0x3c, 255, 0, nan, nan}, // E8M0 NaN scale
  };
  std::array<uint32_t, 64> ones, packed_b, initial, scale_a, scale_b;
  ones.fill(0x3c3c3c3c);
  scale_a.fill(254); // 2^127
  for (size_t lane = 0; lane < ones.size(); ++lane) {
    const auto &column = columns[lane % std::size(columns)];
    packed_b[lane] = uint32_t{column.b} * 0x01010101u;
    initial[lane] = column.c;
    scale_b[lane] = column.scale_b;
  }
  std::array<const uint32_t *, 8> a, b;
  std::array<const uint32_t *, 16> c;
  std::array<std::array<uint32_t, 64>, 16> result;
  std::array<uint32_t *, 16> d;
  a.fill(ones.data());
  b.fill(packed_b.data());
  c.fill(initial.data());
  for (size_t reg = 0; reg < d.size(); ++reg) {
    d[reg] = result[reg].data();
  }
  for (auto entry : {rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64,
                     rj_arith_v_mfma_scale_f32_32x32x64_f8f6f4_w64}) {
    for (uint64_t cpu : supported_cpu_levels()) {
      for (bool clamp : {false, true}) {
        const auto flags = cpu | RJ_ARITH_SEMANTICS_LOOSE | (clamp ? RJ_ARITH_FP16_OVFL : 0);
        constexpr uint64_t bf8_inputs = 1 | (1 << 3);
        ASSERT_EQ(entry(flags, bf8_inputs, d.data(), a.data(), b.data(), c.data(), scale_a.data(),
                        scale_b.data()),
                  RJ_ARITH_SUCCESS);
        for (size_t lane = 0; lane < std::size(columns); ++lane) {
          const auto expected = clamp ? columns[lane].clamped : columns[lane].normal;
          SCOPED_TRACE(::testing::Message()
                       << "cpu=" << cpu << " clamp=" << clamp << " lane=" << lane);
          if (expected == nan) {
            EXPECT_TRUE(std::isnan(std::bit_cast<float>(result.front()[lane])));
          } else {
            EXPECT_EQ(result.front()[lane], expected);
          }
        }
      }
    }
  }
}

TEST(MfmaMixed, ScaleE8m0ExtremeExponentsAndSpecialsMatchLdexpAcrossCpuLevels) {
  // Exercise extreme E8M0 scale exponents (subnormal underflow, split-exponent boundaries
  // 0/1/2/3/253/254, NaN scale 255) across BF8 inputs (normal, min subnormal, +/-0, +/-Inf, NaN).
  constexpr uint8_t kBf8Samples[16] = {
      0x3c, // +1.0
      0xbc, // -1.0
      0x01, // +2^-16 (min BF8 subnormal)
      0x81, // -2^-16
      0x7b, // +57344 (max finite BF8)
      0xfb, // -57344
      0x00, // +0.0
      0x80, // -0.0
      0x7c, // +Inf
      0xfc, // -Inf
      0x7e, // qNaN
      0x40, // +2.0
      0x3e, // +1.5
      0xbe, // -1.5
      0x04, // +2^-14
      0x84, // -2^-14
  };
  constexpr uint8_t kScalesA[16] = {0,   0,   1,   2,   3,   40,  64,  100,
                                    127, 150, 190, 220, 253, 254, 255, 126};
  constexpr uint8_t kScalesB[16] = {0,   1,   2,   3,   16,  50,  80,  127,
                                    127, 170, 210, 240, 253, 254, 255, 0};

  std::array<std::array<uint32_t, 64>, 8> a_buf{}, b_buf{};
  std::array<std::array<uint32_t, 64>, 16> c_buf{};
  std::array<uint32_t, 64> scale_a{}, scale_b{};
  // Populate block 0 only (lane 0..15 for 16x16x128) with one nonzero element per lane so the
  // 32-element block dot equals left[row][0] * right[col][0] exactly; blocks 1..3 remain 0 with
  // neutral scale 127 so each row/col tests one (dot, ea, eb) tuple cleanly.
  for (int lane = 0; lane < 64; ++lane) {
    scale_a[lane] = 127u;
    scale_b[lane] = 127u;
  }
  for (int i = 0; i < 16; ++i) {
    a_buf[0][i] = uint32_t{kBf8Samples[i]}; // k = 0 only; k = 1..31 are +0.0
    b_buf[0][i] = uint32_t{kBf8Samples[(i * 5 + 3) % 16]};
    scale_a[i] = kScalesA[i];
    scale_b[i] = kScalesB[i];
  }

  std::array<const uint32_t *, 8> a_ptrs{}, b_ptrs{};
  std::array<const uint32_t *, 16> c_ptrs{};
  for (int r = 0; r < 8; ++r) {
    a_ptrs[r] = a_buf[r].data();
    b_ptrs[r] = b_buf[r].data();
  }
  for (int r = 0; r < 16; ++r) {
    c_ptrs[r] = c_buf[r].data();
  }

  constexpr uint64_t kBf8Pair = 1u | (1u << 3);
  for (uint64_t cpu : supported_cpu_levels()) {
    std::array<std::array<uint32_t, 64>, 4> d_buf{};
    std::array<uint32_t *, 4> d_ptrs{};
    for (int r = 0; r < 4; ++r) {
      d_ptrs[r] = d_buf[r].data();
    }
    ASSERT_EQ(rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64(
                  cpu | RJ_ARITH_SEMANTICS_LOOSE, kBf8Pair, d_ptrs.data(), a_ptrs.data(),
                  b_ptrs.data(), c_ptrs.data(), scale_a.data(), scale_b.data()),
              RJ_ARITH_SUCCESS);
    for (int row = 0; row < 16; ++row) {
      const float av = decode_matrix_format(kBf8Samples[row], 1);
      for (int col = 0; col < 16; ++col) {
        const float bv = decode_matrix_format(kBf8Samples[(col * 5 + 3) % 16], 1);
        float partial = std::fma(av, bv, 0.0f);
        for (int k = 1; k < 32; ++k) {
          partial = std::fma(0.0f, 0.0f, partial);
        }
        const int ea = kScalesA[row];
        const int eb = kScalesB[col];
        float expected = ea == 255 || eb == 255 ? std::bit_cast<float>(0x7fc00000u)
                                                : std::ldexp(partial, ea + eb - 254);
        // Add the three zero blocks (blocks 1..3), matching the kernel's four block additions.
        expected = (((0.0f + expected) + 0.0f) + 0.0f) + 0.0f;
        const int reg = row % 4;
        const int lane = (row / 4) * 16 + col;
        const uint32_t actual = d_buf[reg][lane];
        SCOPED_TRACE(::testing::Message() << "cpu=" << cpu << " row=" << row << " col=" << col
                                          << " ea=" << ea << " eb=" << eb);
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(std::bit_cast<float>(actual)));
        } else {
          EXPECT_EQ(actual, std::bit_cast<uint32_t>(expected));
        }
      }
    }
  }
}
} // namespace
