// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 mixed-format WMMA packing, all format pairs, CPU dispatch and overlapping operands.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/matrix_formats.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

namespace {
using namespace rocjitsu::arith::test;

// Construct matrices logically, pack them into independent format layouts and compare the product.
template <int M, int Block = 0, typename Entry> void arithmetic(Entry entry) {
  constexpr int K = 128, D = M / 2, Scale = 32 + 2 * D;
  const auto levels = supported_cpu_levels();
  for (int fa = M == 32 ? 4 : 0; fa < 5; ++fa) {
    for (int fb = M == 32 ? 4 : 0; fb < 5; ++fb) {
      std::array<std::array<uint32_t, 32>, Scale + D> input{};
      float left[M][K], right[16][K];
      std::minstd_rand random;
      auto fill = [&]<std::size_t Rows>(int base, int format, int other_format,
                                        float(&values)[Rows][K]) {
        int width = format < 2 ? 8 : format < 4 ? 6 : 4;
        for (int row = 0; row < int(Rows); ++row) {
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
              lane = row % 16 + 16 * (k / 16 % 2);
              local = k / 32 * 16 + k % 16;
            } else if (Block || other_format < 2) {
              lane = row % 16 + 16 * (k / 32 % 2);
              local = k / 64 * 32 + k % 32;
            } else {
              lane = row % 16 + 16 * (k / 4 % 2);
              local = k / 8 * 4 + k % 4;
            }
            // Bit-by-bit packing deliberately avoids sharing the production cross-word extraction.
            for (int bit = 0; bit < width; ++bit) {
              int offset = local * width + bit;
              input[base + 8 * (row / 16) + offset / 32][lane] |= uint32_t((code >> bit) & 1)
                                                                  << (offset % 32);
            }
            values[row][k] = decode_matrix_format(code, format);
          }
        }
      };
      fill(0, fa, fb, left);
      fill(16, fb, fa, right);
      for (int reg = 0; reg < D; ++reg) {
        for (auto &word : input[32 + reg]) {
          word = std::bit_cast<uint32_t>(float(int(random() >> 8) - 0x400000) * 0x1p-23f);
        }
      }
      uint64_t controls = M == 32 ? 0 : uint64_t(fa) | (uint64_t(fb) << 3);
      if (fa & 1) {
        controls |= RJ_ARITH_WMMA_MIXED_ABS_C;
      }
      if (fb & 1 || M == 32) {
        controls |= RJ_ARITH_WMMA_MIXED_NEG_C;
      }
      // Distinct packed bytes exercise block selection and both scale halfwaves.
      for (int lane = 0; lane < 32; ++lane) {
        for (int block = 0; block < 8; ++block) {
          int av = 120 + (lane + block) % 8, bv = 119 + (lane + 2 * block) % 8;
          input[Scale + block / 4][lane] |= uint32_t(av) << (8 * (block % 4));
          input[Scale + 2 + block / 4][lane] |= uint32_t(bv) << (8 * (block % 4));
        }
      }
      int asf = fa % 3, bsf = fb % 3;
      auto scale = [](uint32_t raw, int format) {
        if (format == 0) {
          return raw == 255 ? std::numeric_limits<float>::quiet_NaN()
                            : std::ldexp(1.0f, int(raw) - 127);
        }
        if (format == 2) {
          return decode_matrix_format(int(raw), 0);
        }
        if (raw == 255) {
          return std::numeric_limits<float>::quiet_NaN();
        }
        int exponent = int(raw >> 3), mantissa = int(raw & 7);
        return std::ldexp(float(exponent ? 8 + mantissa : mantissa),
                          (exponent ? exponent : 1) - 18);
      };
      if constexpr (Block != 0) {
        controls |= uint64_t(asf) << 8 | uint64_t(bsf) << 10;
        controls |= RJ_ARITH_WMMA_SCALE_ROW_A | RJ_ARITH_WMMA_SCALE_ROW_B;
      }
      float expected[D][32];
      for (int row = 0; row < M; ++row) {
        for (int col = 0; col < 16; ++col) {
          int reg = 8 * (row / 16) + row % 8, lane = 16 * (row % 16 / 8) + col;
          double sum = std::bit_cast<float>(input[32 + reg][lane]);
          if (controls & RJ_ARITH_WMMA_MIXED_ABS_C) {
            sum = std::abs(sum);
          }
          if (controls & RJ_ARITH_WMMA_MIXED_NEG_C) {
            sum = -sum;
          }
          if constexpr (Block != 0) {
            float acc = float(sum);
            for (int block = 0; block < K / Block; ++block) {
              float partial = 0;
              for (int k = block * Block; k < (block + 1) * Block; ++k) {
                partial = std::fma(left[row][k], right[col][k], partial);
              }
              auto a = input[Scale + block / 4][M == 32 ? row : row + 16];
              auto b = input[Scale + 2 + block / 4][col + 16];
              partial *= scale((a >> (8 * (block % 4))) & 255, asf);
              partial *= scale((b >> (8 * (block % 4))) & 255, bsf);
              acc += partial;
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
      for (int aliasing_offset : {0, 16, 32, 32 + D, Scale}) {
        for (uint64_t cpu : levels) {
          for (uint64_t semantics : semantics_modes()) {
            auto storage = input;
            const uint32_t *a[16], *b[16], *c[D];
            uint32_t *d[D];
            for (int i = 0; i < 16; ++i) {
              a[i] = storage[i].data();
              b[i] = storage[16 + i].data();
            }
            for (int i = 0; i < D; ++i) {
              c[i] = storage[32 + i].data();
              d[i] = storage[aliasing_offset + i].data();
            }
            int status;
            if constexpr (Block != 0) {
              const uint32_t *sa[] = {storage[Scale].data(), storage[Scale + 1].data()};
              const uint32_t *sb[] = {storage[Scale + 2].data(), storage[Scale + 3].data()};
              status = entry(cpu | semantics, controls, d, a, b, c, sa, sb);
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
              for (int lane = 0; lane < 32; ++lane) {
                expect_f32(d[reg][lane], expected[reg][lane], RJ_ARITH_SEMANTICS_LOOSE,
                           /*atol=*/1e-4f, /*rtol=*/1e-5f);
              }
            }
          }
        }
      }
    }
  }
  if constexpr (Block != 0) {
    EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 3ULL << 8, nullptr, nullptr, nullptr, nullptr,
                    nullptr, nullptr),
              RJ_ARITH_ERROR_INVALID_FLAGS);
  } else {
    EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 5, nullptr, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_INVALID_FLAGS);
  }
}
TEST(WmmaMixedGfx1250, F32_16x16x128) { arithmetic<16>(rj_arith_v_wmma_f32_16x16x128_f8f6f4_w32); }
TEST(WmmaMixedGfx1250, F32_32x16x128) { arithmetic<32>(rj_arith_v_wmma_f32_32x16x128_f4_w32); }
TEST(WmmaMixedGfx1250, scale16_16x16x128) {
  arithmetic<16, 16>(rj_arith_v_wmma_scale16_f32_16x16x128_f8f6f4_w32);
}
TEST(WmmaMixedGfx1250, scale_16x16x128) {
  arithmetic<16, 32>(rj_arith_v_wmma_scale_f32_16x16x128_f8f6f4_w32);
}
TEST(WmmaMixedGfx1250, scale16_32x16x128) {
  arithmetic<32, 16>(rj_arith_v_wmma_scale16_f32_32x16x128_f4_w32);
}
TEST(WmmaMixedGfx1250, scale_32x16x128) {
  arithmetic<32, 32>(rj_arith_v_wmma_scale_f32_32x16x128_f4_w32);
}
} // namespace
