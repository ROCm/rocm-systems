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
template <int M, int K, bool Scaled = false, typename Entry> void arithmetic(Entry entry) {
  constexpr int D = M * M / 64, Scale = 16 + 2 * D;
  const auto levels = supported_cpu_levels();
  for (int fa = 0; fa < 5; ++fa) {
    for (int fb = 0; fb < 5; ++fb) {
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
      };
      fill(0, fa, left);
      fill(8, fb, right);
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
      input[Scale][1] = 255U << 8;
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
            const uint32_t *a[8], *b[8], *c[D];
            uint32_t *d[D];
            for (int i = 0; i < 8; ++i) {
              a[i] = storage[i].data();
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
}
TEST(MfmaMixed, ScaleF32_32x32x64) {
  arithmetic<32, 64, true>(rj_arith_v_mfma_scale_f32_32x32x64_f8f6f4_w64);
}
} // namespace
