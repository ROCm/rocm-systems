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

TEST(MfmaMixed, SpecialValuesSubnormalCAndExtremeScales) {
  // Verify independent expected values across all CPU levels (scalar, v3, v4, zen4 BF16/VNNI)
  // on extreme E8M0 scales, subnormal C preservation, and FP16_OVFL saturation.
  struct ScaleCase {
    uint8_t sa, sb;
    uint32_t c;
    bool zero_ab;
    bool clamp;
  };
  constexpr ScaleCase scale_cases[] = {
      {0, 0, 0, false, false},              // extreme underflow to +0.0
      {0, 254, 0, false, false},            // non-plain cancelling extremes (2^0 = 1)
      {254, 0, 0, false, false},            // reversed cancelling extremes (2^0 = 1)
      {1, 1, 0, false, false},              // non-plain underflow to +0.0 (2^-252)
      {1, 120, 0, false, false},            // representable FP32 subnormal scaled product (2^-133)
      {254, 254, 0, false, false},          // overflow to +Inf
      {254, 254, 0, false, true},           // overflow clamped to FLT_MAX by FP16_OVFL
      {254, 254, 0, true, false},           // extreme scale * 0.0 == +0.0 (not NaN)
      {255, 127, 0, true, false},           // NaN scale * 0.0 == NaN
      {127, 255, 0, true, true},            // NaN scale preserved under FP16_OVFL
      {127, 127, 0x00000001u, true, false}, // subnormal +C preserved
      {127, 127, 0x80000001u, true, false}, // subnormal -C preserved
  };
  constexpr std::pair<uint64_t, uint64_t> format_pairs[] = {
      {RJ_ARITH_MATRIX_FORMAT_FP4, RJ_ARITH_MATRIX_FORMAT_FP4},
      {RJ_ARITH_MATRIX_FORMAT_FP4, RJ_ARITH_MATRIX_FORMAT_FP6},
      {RJ_ARITH_MATRIX_FORMAT_FP6, RJ_ARITH_MATRIX_FORMAT_FP6},
      {RJ_ARITH_MATRIX_FORMAT_BF6, RJ_ARITH_MATRIX_FORMAT_BF6},
      {RJ_ARITH_MATRIX_FORMAT_FP8, RJ_ARITH_MATRIX_FORMAT_BF8},
  };
  auto one_bits = [](uint64_t fmt) -> uint32_t {
    switch (fmt) {
    case RJ_ARITH_MATRIX_FORMAT_FP8:
      return 0x38u;
    case RJ_ARITH_MATRIX_FORMAT_BF8:
      return 0x3cu;
    case RJ_ARITH_MATRIX_FORMAT_FP6:
      return 0x08u;
    case RJ_ARITH_MATRIX_FORMAT_BF6:
      return 0x0cu;
    default:
      return 0x02u;
    }
  };
  auto check_scaled = [&]<int OutputRegs, int M, int Blocks>(
                          decltype(&rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64) entry) {
    for (const auto &[fa, fb] : format_pairs) {
      const uint64_t controls = fa | (fb << RJ_ARITH_MATRIX_FORMAT_B_SHIFT);
      for (const ScaleCase sc : scale_cases) {
        std::array<std::array<uint32_t, 64>, 8> a_buf{}, b_buf{};
        if (!sc.zero_ab) {
          // Encode +1.0 at k=0 (reg 0, lanes 0..M-1): exactly one 1.0*1.0 product in block 0.
          for (int lane = 0; lane < M; ++lane) {
            a_buf[0][lane] = one_bits(fa);
            b_buf[0][lane] = one_bits(fb);
          }
        }
        // Independent expected value (block 0 has dot 1.0 or 0.0; remaining blocks are 0.0).
        float expected = std::bit_cast<float>(sc.c);
        for (int blk = 0; blk < Blocks; ++blk) {
          const float block_dot = (!sc.zero_ab && blk == 0) ? 1.0f : 0.0f;
          const float term = (sc.sa == 255 || sc.sb == 255)
                                 ? std::numeric_limits<float>::quiet_NaN()
                                 : std::ldexp(block_dot, int(sc.sa) + int(sc.sb) - 254);
          expected += term;
        }
        if (sc.clamp && std::isinf(expected)) {
          expected = std::copysign(std::numeric_limits<float>::max(), expected);
        }
        const uint32_t expected_bits = std::bit_cast<uint32_t>(expected);

        std::array<uint32_t, 64> sa_buf, sb_buf;
        sa_buf.fill(sc.sa);
        sb_buf.fill(sc.sb);
        std::array<std::array<uint32_t, 64>, OutputRegs> c_buf;
        for (std::array<uint32_t, 64> &r : c_buf) {
          r.fill(sc.c);
        }
        std::array<const uint32_t *, 8> a_ptrs, b_ptrs;
        for (int i = 0; i < 8; ++i) {
          a_ptrs[i] = a_buf[i].data();
          b_ptrs[i] = b_buf[i].data();
        }
        std::array<const uint32_t *, OutputRegs> c_ptrs;
        for (int i = 0; i < OutputRegs; ++i) {
          c_ptrs[i] = c_buf[i].data();
        }
        for (uint64_t cpu : supported_cpu_levels()) {
          std::array<std::array<uint32_t, 64>, OutputRegs> out{};
          std::array<uint32_t *, OutputRegs> d_ptrs;
          for (int i = 0; i < OutputRegs; ++i) {
            d_ptrs[i] = out[i].data();
          }
          const uint64_t flags =
              cpu | RJ_ARITH_SEMANTICS_LOOSE | (sc.clamp ? RJ_ARITH_FP16_OVFL : 0);
          ASSERT_EQ(entry(flags, controls, d_ptrs.data(), a_ptrs.data(), b_ptrs.data(),
                          c_ptrs.data(), sa_buf.data(), sb_buf.data()),
                    RJ_ARITH_SUCCESS);
          for (int r = 0; r < OutputRegs; ++r) {
            for (int lane = 0; lane < 64; ++lane) {
              if (std::isnan(expected)) {
                EXPECT_TRUE(std::isnan(std::bit_cast<float>(out[r][lane])));
              } else {
                EXPECT_EQ(out[r][lane], expected_bits);
              }
            }
          }
        }
      }
    }
  };
  check_scaled.template operator()<4, 16, 4>(rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64);
  check_scaled.template operator()<16, 32, 2>(rj_arith_v_mfma_scale_f32_32x32x64_f8f6f4_w64);

  // Also check unscaled subnormal-C preservation across all supported CPU levels.
  for (const auto &[fa, fb] : format_pairs) {
    const uint64_t controls = fa | (fb << RJ_ARITH_MATRIX_FORMAT_B_SHIFT);
    std::array<std::array<uint32_t, 64>, 8> zeros{};
    std::array<const uint32_t *, 8> z_ptrs;
    for (int i = 0; i < 8; ++i) {
      z_ptrs[i] = zeros[i].data();
    }
    std::array<std::array<uint32_t, 64>, 4> c_buf;
    for (int r = 0; r < 4; ++r) {
      for (int lane = 0; lane < 64; ++lane) {
        c_buf[r][lane] = (lane & 1) ? 0x80000001u : 0x00000001u;
      }
    }
    const uint32_t *c_ptrs[] = {c_buf[0].data(), c_buf[1].data(), c_buf[2].data(), c_buf[3].data()};
    for (uint64_t cpu : supported_cpu_levels()) {
      std::array<std::array<uint32_t, 64>, 4> out{};
      uint32_t *d_ptrs[] = {out[0].data(), out[1].data(), out[2].data(), out[3].data()};
      ASSERT_EQ(rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64(cpu | RJ_ARITH_SEMANTICS_LOOSE, controls,
                                                         d_ptrs, z_ptrs.data(), z_ptrs.data(),
                                                         c_ptrs),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(out, c_buf);
    }
  }
}

TEST(MfmaMixed, UnscaledFp4Fp6CompatibilityPreservesLegacyAccumulation) {
  // Compatibility preserves legacy C-first fused accumulation (including C = 2^24 absorbing unit
  // products and signed -0.0 preservation), while explicit Loose permits exact K-first VNNI dots
  // within loose tolerances.
  const uint64_t fp4_ctrl =
      RJ_ARITH_MATRIX_FORMAT_FP4 | (RJ_ARITH_MATRIX_FORMAT_FP4 << RJ_ARITH_MATRIX_FORMAT_B_SHIFT);
  const uint64_t fp6_ctrl =
      RJ_ARITH_MATRIX_FORMAT_FP6 | (RJ_ARITH_MATRIX_FORMAT_FP6 << RJ_ARITH_MATRIX_FORMAT_B_SHIFT);

  // Case 1: C = 2^24 with eight unit FP4 products (two per 32-element block across 4 blocks).
  // Legacy C-first FMA absorbs every 1.0 into 2^24 -> exact 0x4b800000 in Compatibility.
  {
    std::array<std::array<uint32_t, 64>, 8> a_buf{}, b_buf{};
    for (int lane = 0; lane < 64; ++lane) {
      a_buf[0][lane] = 0x22u; // two FP4 +1.0 nibbles in reg 0
      b_buf[0][lane] = 0x22u;
    }
    std::array<const uint32_t *, 8> a_ptrs, b_ptrs;
    for (int i = 0; i < 8; ++i) {
      a_ptrs[i] = a_buf[i].data();
      b_ptrs[i] = b_buf[i].data();
    }
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t sem : {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_LOOSE}) {
        std::array<std::array<uint32_t, 64>, 4> c_buf, d_buf{};
        for (std::array<uint32_t, 64> &r : c_buf) {
          r.fill(0x4b800000u); // 2^24
        }
        const uint32_t *c_ptrs[] = {c_buf[0].data(), c_buf[1].data(), c_buf[2].data(),
                                    c_buf[3].data()};
        uint32_t *d_ptrs[] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};
        ASSERT_EQ(rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64(cpu | sem, fp4_ctrl, d_ptrs,
                                                           a_ptrs.data(), b_ptrs.data(), c_ptrs),
                  RJ_ARITH_SUCCESS);
        for (int r = 0; r < 4; ++r) {
          for (int lane = 0; lane < 64; ++lane) {
            if (sem == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
              EXPECT_EQ(d_buf[r][lane], 0x4b800000u);
            } else {
              expect_f32(d_buf[r][lane], float(16777216.0 + 8.0), RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/1e-4f, /*rtol=*/1e-5f);
            }
          }
        }
      }
    }
  }

  // Case 2: C = -0.0 with A all +0.0 and B all -0.0 (every product -0.0) preserves -0.0 in
  // Compatibility mode across all CPU tiers.
  {
    std::array<std::array<uint32_t, 64>, 8> a_buf{}, b_buf{};
    for (int r = 0; r < 4; ++r) {
      b_buf[r].fill(0x88888888u); // FP4 -0.0 in every nibble
    }
    std::array<const uint32_t *, 8> a_ptrs, b_ptrs;
    for (int i = 0; i < 8; ++i) {
      a_ptrs[i] = a_buf[i].data();
      b_ptrs[i] = b_buf[i].data();
    }
    for (uint64_t cpu : supported_cpu_levels()) {
      std::array<std::array<uint32_t, 64>, 4> c_buf, d_buf{};
      for (std::array<uint32_t, 64> &r : c_buf) {
        r.fill(0x80000000u); // -0.0
      }
      const uint32_t *c_ptrs[] = {c_buf[0].data(), c_buf[1].data(), c_buf[2].data(),
                                  c_buf[3].data()};
      uint32_t *d_ptrs[] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};
      ASSERT_EQ(rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY,
                                                         fp4_ctrl, d_ptrs, a_ptrs.data(),
                                                         b_ptrs.data(), c_ptrs),
                RJ_ARITH_SUCCESS);
      for (int r = 0; r < 4; ++r) {
        for (int lane = 0; lane < 64; ++lane) {
          EXPECT_EQ(d_buf[r][lane], 0x80000000u);
        }
      }
    }
  }

  // Case 3: Nonzero FP6xFP6 products in both Compatibility and Loose (C = 2^24 + unit FP6).
  {
    std::array<std::array<uint32_t, 64>, 8> a_buf{}, b_buf{};
    for (int lane = 0; lane < 64; ++lane) {
      // Two FP6 +1.0 codes (0x08) at bits [0..5] and [6..11] of reg 0 -> 8 unit products total.
      a_buf[0][lane] = 0x08u | (0x08u << 6);
      b_buf[0][lane] = 0x08u | (0x08u << 6);
    }
    std::array<const uint32_t *, 8> a_ptrs, b_ptrs;
    for (int i = 0; i < 8; ++i) {
      a_ptrs[i] = a_buf[i].data();
      b_ptrs[i] = b_buf[i].data();
    }
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t sem : {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_LOOSE}) {
        std::array<std::array<uint32_t, 64>, 4> c_buf, d_buf{};
        for (std::array<uint32_t, 64> &r : c_buf) {
          r.fill(0x4b800000u);
        }
        const uint32_t *c_ptrs[] = {c_buf[0].data(), c_buf[1].data(), c_buf[2].data(),
                                    c_buf[3].data()};
        uint32_t *d_ptrs[] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};
        ASSERT_EQ(rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64(cpu | sem, fp6_ctrl, d_ptrs,
                                                           a_ptrs.data(), b_ptrs.data(), c_ptrs),
                  RJ_ARITH_SUCCESS);
        for (int r = 0; r < 4; ++r) {
          for (int lane = 0; lane < 64; ++lane) {
            if (sem == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
              EXPECT_EQ(d_buf[r][lane], 0x4b800000u);
            } else {
              expect_f32(d_buf[r][lane], float(16777216.0 + 8.0), RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/1e-4f, /*rtol=*/1e-5f);
            }
          }
        }
      }
    }
  }
}
TEST(MfmaMixed, ScaledFp4PreservesPartiallyOverlappingAccumulators) {
  constexpr int Lanes = 64;
  constexpr int MaxRegisters = 16;
  const uint64_t fp4_ctrl =
      RJ_ARITH_MATRIX_FORMAT_FP4 | (RJ_ARITH_MATRIX_FORMAT_FP4 << RJ_ARITH_MATRIX_FORMAT_B_SHIFT);
  const std::array<uint32_t, Lanes> zero{};
  std::array<uint32_t, Lanes> unit_scale;
  unit_scale.fill(127u);
  const std::array<const uint32_t *, 8> z_ptrs{zero.data(), zero.data(), zero.data(), zero.data(),
                                               zero.data(), zero.data(), zero.data(), zero.data()};
  struct Shape {
    decltype(&rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64) entry;
    int registers;
  };
  constexpr Shape shapes[] = {{rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64, 4},
                              {rj_arith_v_mfma_scale_f32_32x32x64_f8f6f4_w64, 16}};
  for (const Shape &shape : shapes) {
    const int adjacent = (shape.registers - 1) * Lanes + 16;
    const int layouts[][3] = {{0, adjacent, Lanes}, {adjacent, 0, Lanes}, {0, 0, Lanes - 16}};
    for (const auto &[c_start, d_start, stride] : layouts) {
      for (const uint64_t cpu : supported_cpu_levels()) {
        std::array<uint32_t, 2 * MaxRegisters * Lanes> actual{};
        for (size_t i = 0; i < actual.size(); ++i) {
          actual[i] = std::bit_cast<uint32_t>(static_cast<float>(i + 1));
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
        ASSERT_EQ(shape.entry(cpu | RJ_ARITH_SEMANTICS_LOOSE, fp4_ctrl, d.data(), z_ptrs.data(),
                              z_ptrs.data(), c.data(), unit_scale.data(), unit_scale.data()),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(actual, expected) << "registers=" << shape.registers << " cpu=" << cpu;
      }
    }
  }
}
} // namespace
