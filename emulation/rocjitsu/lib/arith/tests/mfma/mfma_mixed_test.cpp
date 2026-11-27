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
      constexpr uint64_t kScaleCModifiers = RJ_ARITH_MFMA_SCALE_NEG_C | RJ_ARITH_MFMA_SCALE_ABS_C;
      const bool unsupported_strict = Scaled && (controls & kScaleCModifiers) != 0;
      for (int aliasing_offset : {0, 8, 16, 16 + D, Scale}) {
        std::array<std::array<uint32_t, 64>, D> exact_reference{};
        bool have_exact_reference = false;
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
            if (unsupported_strict &&
                (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                 semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL)) {
              ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              EXPECT_EQ(storage, input);
              continue;
            }
            ASSERT_EQ(status, RJ_ARITH_SUCCESS);
            const bool exact_used = !unsupported_strict && semantics != RJ_ARITH_SEMANTICS_LOOSE &&
                                    semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY;
            for (int reg = 0; reg < D; ++reg) {
              for (int lane = 0; lane < 64; ++lane) {
                if (exact_used) {
                  // Hardware chunks align at 13 fraction bits relative to the largest product in
                  // each 8-element chunk, so cancellation error scales with sum_k |a_k * b_k|.
                  const int row = reg / 4 * (4 * (64 / M)) + lane / M * 4 + reg % 4;
                  const int col = lane % M;
                  double abs_sum = std::abs(double(std::bit_cast<float>(input[16 + reg][lane])));
                  for (int k = 0; k < K; ++k) {
                    const double prod = std::abs(double(left[row][k]) * double(right[col][k]));
                    if constexpr (Scaled) {
                      const int blk = k / 32;
                      const int ea = int(input[Scale][blk * M + row] >> 8);
                      const int eb = int(input[Scale + 1][blk * M + col] >> 16);
                      if (ea < 255 && eb < 255) {
                        abs_sum += std::ldexp(prod, ea + eb - 254);
                      }
                    } else {
                      abs_sum += prod;
                    }
                  }
                  const float bound = float(std::max(1e-4, abs_sum * 0x1p-11));
                  expect_f32(d[reg][lane], expected[reg][lane], RJ_ARITH_SEMANTICS_LOOSE, bound,
                             1e-3f);
                  if (std::isnan(expected[reg][lane])) {
                    EXPECT_EQ(d[reg][lane], 0xffc00000U);
                  }
                  if (have_exact_reference) {
                    EXPECT_EQ(d[reg][lane], exact_reference[reg][lane]);
                  }
                } else {
                  expect_f32(d[reg][lane], expected[reg][lane], RJ_ARITH_SEMANTICS_LOOSE,
                             /*atol=*/1e-4f, /*rtol=*/1e-5f);
                }
              }
            }
            if (exact_used && !have_exact_reference) {
              for (int reg = 0; reg < D; ++reg) {
                for (int lane = 0; lane < 64; ++lane) {
                  exact_reference[reg][lane] = d[reg][lane];
                }
              }
              have_exact_reference = true;
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

// Golden rows captured on an AMD Instinct MI350X (gfx950), ROCm 7.2.4 / HIP 7.2.53211
// (gfx950_mfma.bin).
TEST(MfmaMixed, Gfx950ExactMatchesCapturedMi350xCases) {
  const auto levels = supported_cpu_levels();
  const uint64_t exact_modes[] = {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL};

  auto run_32x32x64 = [](int fa, int fb, const uint8_t a_row[64], const uint8_t b_col[64],
                         uint32_t c0, uint64_t flags) {
    const int wa = fa < 2 ? 8 : fa < 4 ? 6 : 4;
    const int wb = fb < 2 ? 8 : fb < 4 ? 6 : 4;
    std::array<std::array<uint32_t, 64>, 8> a_buf{}, b_buf{};
    std::array<std::array<uint32_t, 64>, 16> c_buf{}, d_buf{};
    auto pack = [](auto &buf, int width, const uint8_t elems[64]) {
      for (int k = 0; k < 64; ++k) {
        const int lane = width == 8 ? (k / 16 % 2) * 32 : (k / 32) * 32;
        const int local = width == 8 ? (k / 32) * 16 + k % 16 : k % 32;
        for (int bit = 0; bit < width; ++bit) {
          const int off = local * width + bit;
          buf[off / 32][lane] |= uint32_t((elems[k] >> bit) & 1) << (off % 32);
        }
      }
    };
    pack(a_buf, wa, a_row);
    pack(b_buf, wb, b_col);
    c_buf[0][0] = c0;
    const uint32_t *a[8]{}, *b[8]{}, *c[16]{};
    uint32_t *d[16]{};
    for (int r = 0; r < 8; ++r) {
      a[r] = a_buf[r].data();
      b[r] = b_buf[r].data();
    }
    for (int r = 0; r < 16; ++r) {
      c[r] = c_buf[r].data();
      d[r] = d_buf[r].data();
    }
    EXPECT_EQ(rj_arith_v_mfma_f32_32x32x64_f8f6f4_w64(flags, uint64_t(fa) | (uint64_t(fb) << 3), d,
                                                      a, b, c),
              RJ_ARITH_SUCCESS);
    return d_buf[0][0];
  };

  // 1) Captured v_mfma_f32_16x16x128_f8f6f4 BF8xBF8 (16-product pass rate, row 4 col 6):
  //    host FMA yields 0x4d5905adu while MI350X silicon yields 0x4d5905e0u.
  {
    constexpr uint32_t kABf8[32] = {
        0xb059c98fu, 0x3280bcd2u, 0x4cc7b7e2u, 0x3c96bba3u, 0x333a3d57u, 0xb00d3316u, 0x36ec308eu,
        0xbc384654u, 0x4c28cd52u, 0x37a63780u, 0xc710b03fu, 0xb8ad43e8u, 0x33334462u, 0xca1ac7eau,
        0x3d18bf95u, 0x479ebbf6u, 0x42c93419u, 0xb147cf25u, 0x44face82u, 0xcc12b339u, 0xb91f47dau,
        0x40ee499fu, 0xc978cb95u, 0xc5a63b3cu, 0x4abd30e3u, 0x4c8d3246u, 0x4c1e42f3u, 0xcc804b9cu,
        0xbe3a339bu, 0x36b03d4fu, 0xbc25cd37u, 0xcb35b635u};
    constexpr uint32_t kBBf8[32] = {
        0xc571bc16u, 0x44044797u, 0x45b7381au, 0x3a6fc3dcu, 0x40cd456fu, 0xc6f44daau, 0xb714b532u,
        0xbddacb8fu, 0x4432c412u, 0x3ebc4e2eu, 0xbca7b6efu, 0x3a35b1acu, 0x388abc0au, 0xb31cc8e0u,
        0xba993c71u, 0xc82f34ddu, 0x49cfc914u, 0x4ac83befu, 0xbf0cc194u, 0xc8f6b7c8u, 0x4c09bec3u,
        0x3356c3eau, 0x4e2d418bu, 0x4b534400u, 0xb662c975u, 0xb31eb8f9u, 0xc74b4af4u, 0x3a364d58u,
        0xc453b57bu, 0xcae8bf0au, 0x4c09ce40u, 0x3a1db3a9u};
    std::array<std::array<uint32_t, 64>, 8> a_buf{}, b_buf{};
    std::array<std::array<uint32_t, 64>, 4> c_buf{}, d_buf{};
    for (int g = 0; g < 4; ++g) {
      for (int r = 0; r < 8; ++r) {
        a_buf[r][g * 16 + 4] = kABf8[g * 8 + r];
        b_buf[r][g * 16 + 6] = kBBf8[g * 8 + r];
      }
    }
    const uint32_t *a[8], *b[8], *c[4];
    uint32_t *d[4];
    for (int r = 0; r < 8; ++r) {
      a[r] = a_buf[r].data();
      b[r] = b_buf[r].data();
    }
    for (int r = 0; r < 4; ++r) {
      c[r] = c_buf[r].data();
      d[r] = d_buf[r].data();
    }
    for (uint64_t cpu : levels) {
      for (uint64_t sem : exact_modes) {
        ASSERT_EQ(rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64(cpu | sem, 1 | (1 << 3), d, a, b, c),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(d_buf[0][22], 0x4d5905e0u);
      }
    }
  }

  // 2) Captured v_mfma_f32_32x32x64_f8f6f4 FP6xFP6 (32-product subbyte pass rate, w=7 row=0 col=7):
  //    32-product passes yield MI350X's 0xc2ec503bu, whereas 16-product passes yield 0xc2ec503cu.
  {
    constexpr uint8_t kAFp6[64] = {0x0e, 0x0d, 0x33, 0x04, 0x32, 0x1a, 0x22, 0x30, 0x29, 0x1b, 0x3c,
                                   0x16, 0x31, 0x12, 0x12, 0x0d, 0x30, 0x12, 0x0c, 0x3f, 0x33, 0x3a,
                                   0x00, 0x33, 0x14, 0x11, 0x2c, 0x1d, 0x09, 0x31, 0x0e, 0x0e, 0x12,
                                   0x06, 0x3c, 0x02, 0x08, 0x39, 0x33, 0x10, 0x05, 0x02, 0x1b, 0x24,
                                   0x08, 0x2f, 0x3a, 0x2e, 0x17, 0x0d, 0x03, 0x24, 0x35, 0x0c, 0x00,
                                   0x33, 0x0c, 0x12, 0x2c, 0x14, 0x03, 0x21, 0x27, 0x13};
    constexpr uint8_t kBFp6[64] = {0x27, 0x18, 0x13, 0x37, 0x00, 0x3b, 0x33, 0x2f, 0x3a, 0x3a, 0x24,
                                   0x34, 0x33, 0x24, 0x38, 0x12, 0x03, 0x0b, 0x2c, 0x2b, 0x0b, 0x27,
                                   0x28, 0x12, 0x22, 0x04, 0x14, 0x15, 0x37, 0x02, 0x36, 0x31, 0x33,
                                   0x0e, 0x2c, 0x31, 0x3e, 0x12, 0x19, 0x0e, 0x08, 0x20, 0x34, 0x20,
                                   0x36, 0x2c, 0x31, 0x30, 0x2b, 0x24, 0x3c, 0x2d, 0x04, 0x1d, 0x0b,
                                   0x0e, 0x1d, 0x19, 0x2b, 0x21, 0x31, 0x26, 0x11, 0x30};
    for (uint64_t cpu : levels) {
      for (uint64_t sem : exact_modes) {
        EXPECT_EQ(run_32x32x64(2, 2, kAFp6, kBFp6, 0xc24cc076u, cpu | sem), 0xc2ec503bu);
      }
    }
  }

  // 3) Captured accumulator cutoff boundaries (ce < te - 25 at mixed_exact.h:101):
  //    - te - ce == 26 (v_mfma_f32_32x32x64_f8f6f4 FP8xBF8 w=7 row=30 col=25):
  //      MI350X silicon and ce < te - 25 yield 0x49dd5fdau, whereas ce < te - 26 yields
  //      0x49dd5fd6u.
  //    - te - ce == 25 (v_mfma_f32_32x32x64_f8f6f4 BF8xBF8 w=11 row=9 col=13):
  //      MI350X silicon and ce < te - 25 yield 0xc7efd300u, whereas ce < te - 24 yields
  //      0xc7efd2f8u.
  {
    constexpr uint8_t kACut26[64] = {
        0xb9, 0xca, 0x7e, 0x30, 0x5a, 0xb4, 0x89, 0x31, 0x07, 0x33, 0x06, 0x3b, 0x08,
        0xc0, 0xbf, 0x39, 0x95, 0xcb, 0x97, 0x4d, 0x34, 0xb7, 0x2c, 0x3f, 0x64, 0xc8,
        0x3b, 0xbe, 0x9f, 0x4a, 0x11, 0xb8, 0xb9, 0x43, 0x00, 0x3b, 0xd6, 0xb9, 0x2a,
        0xbe, 0x8a, 0x33, 0x01, 0xcd, 0x23, 0xbd, 0x25, 0x31, 0xf1, 0x3c, 0x5d, 0x3b,
        0x73, 0x46, 0xa4, 0xcd, 0x10, 0x36, 0x25, 0xb7, 0x10, 0xc1, 0x2a, 0x4c};
    constexpr uint8_t kBCut26[64] = {
        0x2b, 0xb7, 0x72, 0x3a, 0xd6, 0xcb, 0x05, 0x43, 0x47, 0x31, 0xbc, 0x45, 0xb0,
        0x3e, 0xae, 0x31, 0xc4, 0x38, 0x7a, 0xca, 0xa6, 0x3f, 0xfa, 0x48, 0xc4, 0xcb,
        0x37, 0x3d, 0x1e, 0x3b, 0x12, 0x37, 0x7b, 0xc8, 0xc6, 0xb7, 0xa3, 0xc7, 0xad,
        0x30, 0xaf, 0x4d, 0x3e, 0xb3, 0x09, 0x36, 0xa3, 0x4b, 0xb1, 0x36, 0x50, 0xc3,
        0xf5, 0xbd, 0x5f, 0x34, 0x15, 0x49, 0xbf, 0x37, 0xd3, 0xbd, 0xd2, 0xb6};
    constexpr uint8_t kACut25[64] = {
        0xd8, 0x3d, 0x5c, 0xb1, 0x7b, 0xba, 0x60, 0xc9, 0xf3, 0x36, 0xac, 0xc6, 0xec,
        0xce, 0x77, 0x47, 0x93, 0xcf, 0x50, 0x47, 0x8c, 0x3f, 0xce, 0x46, 0x76, 0xcd,
        0xdf, 0xc3, 0x45, 0xb5, 0x4f, 0xb3, 0x91, 0xc4, 0x03, 0xcf, 0xad, 0xb7, 0x79,
        0xc9, 0x9f, 0xc3, 0xce, 0x4d, 0xad, 0x44, 0x58, 0xb8, 0xb9, 0xbc, 0xc0, 0xcb,
        0xc6, 0x40, 0xad, 0xc2, 0x92, 0x4b, 0xd4, 0x38, 0x15, 0x32, 0x46, 0x37};
    constexpr uint8_t kBCut25[64] = {
        0x8c, 0x41, 0xe9, 0xb3, 0xc2, 0xcf, 0xda, 0x48, 0x9f, 0x47, 0xc0, 0x42, 0x25,
        0xc5, 0x24, 0xb3, 0x67, 0xc0, 0x3b, 0x35, 0x0a, 0xc8, 0xbe, 0xc8, 0x9a, 0xc6,
        0x3e, 0xc0, 0x47, 0x4e, 0x48, 0x33, 0xdc, 0x37, 0x43, 0xc8, 0xc6, 0xbd, 0x4d,
        0xcf, 0x02, 0xc4, 0xbe, 0x38, 0x00, 0xcd, 0x18, 0xb8, 0x05, 0xb5, 0x70, 0xbd,
        0x47, 0xbb, 0x69, 0xb5, 0x56, 0x4a, 0x10, 0x42, 0x48, 0xc4, 0xd5, 0x3d};
    for (uint64_t cpu : levels) {
      for (uint64_t sem : exact_modes) {
        EXPECT_EQ(run_32x32x64(0, 1, kACut26, kBCut26, 0xbd17ac34u, cpu | sem), 0x49dd5fdau);
        EXPECT_EQ(run_32x32x64(1, 1, kACut25, kBCut25, 0xbcf716cbu, cpu | sem), 0xc7efd300u);
      }
    }
  }

  // 4) Captured exact half-way RNE ties (rem == half at mixed_exact.h:119) & per-pass FP16_OVFL:
  //    - Even tie (FP8xFP8 w=3 row=0 col=9): MI350X and RNE yield 0x47e81106u (half-up:
  //    0x47e81107u).
  //    - Odd tie  (FP8xFP8 w=3 row=4 col=4): MI350X and RNE yield 0x463c2308u (half-down:
  //    0x463c2307u).
  //    - Special-value lanes verified on MI350X (e.g. w=1 reg=1 lane=0 C=0x7ffffff0 -> 0xffc00000 /
  //      0x00000000, w=2 reg=1 lane=11 C=0x7f800000 -> 0x7f800000 / 0x7f7fffff, w=3 reg=1 lane=27
  //      BF8 +Inf -> 0x7f800000 / 0x7f7fffff, and w=4 scale_16x16x128 C=0x7ffffff0 -> 0xffc00000 /
  //      0x460fd4d6).
  {
    constexpr uint8_t kAW3R0[64] = {
        0x0d, 0xc5, 0x39, 0x3a, 0x02, 0xbf, 0xbc, 0x34, 0x39, 0xb1, 0x67, 0x43, 0xc2,
        0x49, 0x0b, 0xb4, 0x4b, 0x39, 0x49, 0xc0, 0xba, 0x37, 0x47, 0x37, 0x54, 0x4c,
        0x69, 0x45, 0x8f, 0xb6, 0xc2, 0x4f, 0x45, 0xbf, 0x74, 0x36, 0x62, 0x48, 0x56,
        0x32, 0x21, 0x3a, 0xed, 0x46, 0xcb, 0x3f, 0xec, 0xbf, 0x56, 0x48, 0xb2, 0x44,
        0x5d, 0xb1, 0x79, 0xbc, 0x76, 0x3e, 0x00, 0x37, 0xc6, 0x31, 0x06, 0xbd};
    constexpr uint8_t kBTieEven[64] = {
        0xac, 0x47, 0x65, 0x31, 0x3e, 0xcc, 0xd5, 0xc1, 0xd1, 0xbc, 0xe7, 0x44, 0x21,
        0xc0, 0xa0, 0xb3, 0x87, 0x33, 0xbf, 0xc2, 0xbe, 0x46, 0x86, 0xb6, 0x87, 0x3d,
        0x84, 0xcf, 0xc4, 0x4d, 0x15, 0x4e, 0xbe, 0x46, 0x7a, 0xc3, 0x8e, 0x3a, 0x3d,
        0x4b, 0x66, 0x4b, 0xb6, 0x3c, 0xa0, 0x4a, 0xd9, 0xb8, 0xa8, 0x31, 0xf8, 0x43,
        0x59, 0x49, 0x44, 0x31, 0x78, 0xcc, 0x5d, 0x43, 0xe5, 0xc9, 0xed, 0xb1};
    constexpr uint8_t kAOddTie[64] = {
        0x79, 0x30, 0x40, 0x30, 0x2c, 0x3c, 0xc6, 0x39, 0x83, 0xc0, 0x92, 0x43, 0x92,
        0x39, 0x59, 0x39, 0x47, 0xbc, 0x07, 0x34, 0x35, 0x40, 0x52, 0x42, 0xf9, 0x30,
        0x55, 0x40, 0xb8, 0x47, 0xba, 0x46, 0x5f, 0x4b, 0x42, 0x31, 0x85, 0x35, 0xd3,
        0xcd, 0x88, 0x3f, 0x1b, 0x4d, 0xe7, 0x4f, 0x1f, 0xb8, 0x2b, 0xb6, 0x2a, 0x3b,
        0x4f, 0xcf, 0xda, 0x44, 0xd4, 0x3f, 0x1f, 0x32, 0x07, 0xcf, 0x0e, 0xb3};
    constexpr uint8_t kBOddTie[64] = {
        0x37, 0x38, 0xea, 0xce, 0x0b, 0xbf, 0x8d, 0xc0, 0x02, 0xcb, 0x4d, 0xcd, 0x2c,
        0xc5, 0xf7, 0xb9, 0x05, 0xba, 0x83, 0x4d, 0xbc, 0xb4, 0x2b, 0xba, 0x1f, 0x41,
        0x47, 0xcc, 0x0d, 0x49, 0x3b, 0xc6, 0xca, 0x33, 0xa6, 0x3b, 0x0a, 0x3c, 0x4c,
        0x4e, 0xe8, 0x3e, 0xbf, 0xc4, 0xf8, 0xb1, 0x87, 0x33, 0x06, 0x3c, 0xa1, 0x3e,
        0x72, 0xb8, 0x83, 0x4e, 0x24, 0xc8, 0x44, 0x35, 0x6f, 0x4d, 0xb4, 0xb9};
    constexpr uint8_t kBCNan[64] = {
        0x20, 0x4f, 0x72, 0x48, 0x2f, 0x35, 0x72, 0xbc, 0x28, 0xc6, 0x0f, 0xb1, 0xf5,
        0x33, 0x67, 0xcc, 0xf4, 0x42, 0xb5, 0xca, 0xaf, 0xcd, 0x48, 0xcc, 0x93, 0x3f,
        0xb3, 0xca, 0x0b, 0x4c, 0xf6, 0xcd, 0x66, 0x3c, 0xac, 0xb2, 0x77, 0xc8, 0xb0,
        0xbc, 0xa5, 0xc3, 0x92, 0x3e, 0x4c, 0x43, 0xfb, 0x43, 0x70, 0x31, 0xc5, 0xc6,
        0xf2, 0xb3, 0xd4, 0xcd, 0xbb, 0xb8, 0x13, 0xbe, 0xcf, 0xc4, 0x5a, 0xb7};
    constexpr uint8_t kBBf8Inf[64] = {
        0x16, 0xbe, 0x56, 0xc6, 0x64, 0xb1, 0xe9, 0x4c, 0xb9, 0x4a, 0x3d, 0x30, 0xd5,
        0xcb, 0x60, 0xce, 0xaa, 0xce, 0x5e, 0x3a, 0x60, 0x3d, 0xdd, 0xbf, 0xa5, 0xc7,
        0xb8, 0x34, 0xed, 0x4f, 0xda, 0xc2, 0x07, 0x40, 0x76, 0x40, 0xc3, 0xc8, 0x1c,
        0xc2, 0x6c, 0xbd, 0x81, 0xc6, 0x4d, 0x4f, 0xfc, 0xc2, 0x1b, 0x47, 0xed, 0x3f,
        0x3c, 0x3f, 0x58, 0xca, 0x81, 0x4d, 0x01, 0xbc, 0x49, 0x4d, 0x85, 0xc0};
    uint8_t neg_zeros[64];
    std::fill_n(neg_zeros, 64, uint8_t{0x80});
    for (uint64_t cpu : levels) {
      for (uint64_t sem : exact_modes) {
        EXPECT_EQ(run_32x32x64(0, 0, kAW3R0, kBTieEven, 0x80000000u, cpu | sem), 0x47e81106u);
        EXPECT_EQ(run_32x32x64(0, 0, kAOddTie, kBOddTie, 0x00000000u, cpu | sem), 0x463c2308u);
        EXPECT_EQ(run_32x32x64(0, 0, kAW3R0, kBCNan, 0x7fc00009u, cpu | sem), 0xffc00000u);
        EXPECT_EQ(run_32x32x64(0, 0, kAW3R0, kBCNan, 0x7fc00009u, cpu | sem | RJ_ARITH_FP16_OVFL),
                  0x470f923cu);
        EXPECT_EQ(run_32x32x64(0, 1, kAW3R0, kBBf8Inf, 0x7f7fffffu, cpu | sem), 0x7f800000u);
        EXPECT_EQ(run_32x32x64(0, 1, kAW3R0, kBBf8Inf, 0x7f7fffffu, cpu | sem | RJ_ARITH_FP16_OVFL),
                  0x7f7fffffu);
        EXPECT_EQ(run_32x32x64(0, 0, neg_zeros, neg_zeros, 0x7f800000u, cpu | sem), 0x7f800000u);
        EXPECT_EQ(
            run_32x32x64(0, 0, neg_zeros, neg_zeros, 0x7f800000u, cpu | sem | RJ_ARITH_FP16_OVFL),
            0x7f7fffffu);
      }
    }
  }

  // 5) Captured v_mfma_scale_f32_16x16x128_f8f6f4 FP4xFP4 (row 12 col 4):
  //    block 3 cancels non-zero products to 0 at scale_a=0x88, scale_b=0xfd (combined +135),
  //    shifting C=0x7f7fffff down by 11 bits to 0x7f7ff800 on MI350X, whereas an all-zero
  //    product block at the same scales leaves row 0 col 0's C=0x7f7fffff untouched.
  {
    constexpr uint32_t kAFp4[16] = {0x007fffffu, 0x7f7fe000u, 0x7f801fffu, 0xff7fe000u,
                                    0x7fc04000u, 0x3f801000u, 0x80000000u, 0xbf802000u,
                                    0xff7fe000u, 0xbf802000u, 0xbf802000u, 0x7fc04000u,
                                    0x7f802000u, 0xff7fe000u, 0xff800000u, 0x00000001u};
    constexpr uint32_t kBFp4[16] = {0x3f801000u, 0x3f801000u, 0x7f801fffu, 0xff800000u,
                                    0x7f801fffu, 0x7fc04000u, 0x7f801fffu, 0x7f800001u,
                                    0x7f800001u, 0x7fc04000u, 0x7fc04000u, 0x7f800001u,
                                    0x80000000u, 0x7f800000u, 0x7fc04000u, 0x7f7fe000u};
    constexpr uint8_t kSa[4] = {0x80u, 0x01u, 0x88u, 0x88u};
    constexpr uint8_t kSb[4] = {0x80u, 0x00u, 0x00u, 0xfdu};
    std::array<std::array<uint32_t, 64>, 4> a_buf{}, b_buf{}, c_buf{}, d_buf{};
    std::array<uint32_t, 64> sa{}, sb{};
    for (int g = 0; g < 4; ++g) {
      for (int r = 0; r < 4; ++r) {
        a_buf[r][g * 16 + 12] = kAFp4[g * 4 + r];
        b_buf[r][g * 16 + 4] = kBFp4[g * 4 + r];
      }
      sa[g * 16 + 12] = kSa[g];
      sb[g * 16 + 4] = kSb[g];
      sa[g * 16 + 0] = kSa[g];
      sb[g * 16 + 0] = kSb[g];
    }
    c_buf[0][52] = 0x7f7fffffu; // row 12, col 4
    c_buf[0][0] = 0x7f7fffffu;  // row 0, col 0 (all-zero products)
    const uint32_t *a[4] = {a_buf[0].data(), a_buf[1].data(), a_buf[2].data(), a_buf[3].data()};
    const uint32_t *b[4] = {b_buf[0].data(), b_buf[1].data(), b_buf[2].data(), b_buf[3].data()};
    const uint32_t *c[4] = {c_buf[0].data(), c_buf[1].data(), c_buf[2].data(), c_buf[3].data()};
    uint32_t *d[4] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};
    for (uint64_t cpu : levels) {
      for (uint64_t sem : exact_modes) {
        ASSERT_EQ(rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64(cpu | sem, 4 | (4 << 3), d, a, b,
                                                                 c, sa.data(), sb.data()),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(d_buf[0][52], 0x7f7ff800u);
        EXPECT_EQ(d_buf[0][0], 0x7f7fffffu);
      }
    }
  }

  // 6) Captured v_mfma_f32_16x16x128_f8f6f4 FP4xFP8 (row 0 col 3) under FP16_OVFL=0 vs 1:
  //    ovfl0 yields canonical 0xffc00000u, while ovfl1 flushes the NaN in k=96..111 to +0.0
  //    before accumulating k=112..127 to 0x41340000u (11.25f).
  {
    constexpr uint32_t kAFp4[16] = {0x7f800000u, 0x3f801fffu, 0x7f800001u, 0x3f800000u,
                                    0xff7fe000u, 0x7f801fffu, 0x7f800001u, 0x00000001u,
                                    0xff7fe000u, 0x7f800001u, 0xff7fe000u, 0x7f800000u,
                                    0x7f800001u, 0x7f800001u, 0x80000000u, 0x7f802000u};
    constexpr uint32_t kBFp8[32] = {
        0xbf802000u, 0x00000001u, 0x3f801000u, 0x80000000u, 0x7f7fe000u, 0x3f801000u, 0xff7fe000u,
        0x3f801fffu, 0x7fc04000u, 0x3f801000u, 0xff800000u, 0x7f801fffu, 0xff7fe000u, 0x7f7fe000u,
        0x3f801fffu, 0x00000001u, 0x3f801fffu, 0x3f800000u, 0x7f802000u, 0x00000000u, 0x00000001u,
        0x7f801fffu, 0x3f800000u, 0x3f801fffu, 0x7f801fffu, 0x7f800000u, 0xff7fe000u, 0x7f7fe000u,
        0xbf802000u, 0x3f800000u, 0x00000001u, 0x3f800000u};
    std::array<std::array<uint32_t, 64>, 4> a_buf{}, c_buf{}, d_buf{};
    std::array<std::array<uint32_t, 64>, 8> b_buf{};
    for (int g = 0; g < 4; ++g) {
      for (int r = 0; r < 4; ++r) {
        a_buf[r][g * 16 + 0] = kAFp4[g * 4 + r];
      }
      for (int r = 0; r < 8; ++r) {
        b_buf[r][g * 16 + 3] = kBFp8[g * 8 + r];
      }
    }
    c_buf[0][3] = 0x7f7fffffu;
    const uint32_t *a[4] = {a_buf[0].data(), a_buf[1].data(), a_buf[2].data(), a_buf[3].data()};
    const uint32_t *b[8], *c[4];
    uint32_t *d[4];
    for (int r = 0; r < 8; ++r) {
      b[r] = b_buf[r].data();
    }
    for (int r = 0; r < 4; ++r) {
      c[r] = c_buf[r].data();
      d[r] = d_buf[r].data();
    }
    for (uint64_t cpu : levels) {
      for (uint64_t sem : exact_modes) {
        ASSERT_EQ(rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64(cpu | sem, 4 | (0 << 3), d, a, b, c),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(d_buf[0][3], 0xffc00000u);
        ASSERT_EQ(rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64(cpu | sem | RJ_ARITH_FP16_OVFL,
                                                           4 | (0 << 3), d, a, b, c),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(d_buf[0][3], 0x41340000u);
      }
    }
  }

  // 7) Synthetic scaled special-value interaction & strict C-modifier gate:
  //    - C=+Inf with negative finite scaled overflow (-1 * 2^254) preserves +Inf (not NaN);
  //    - Non-zero NEG_C/ABS_C rejects strict modes without modifying D, while allow-Loose falls
  //      back.
  {
    std::array<std::array<uint32_t, 64>, 8> a_buf{}, b_buf{};
    std::array<std::array<uint32_t, 64>, 4> c_buf{}, d_buf{};
    std::array<uint32_t, 64> sa{}, sb{};
    sa.fill(254u); // 2^127
    sb.fill(254u); // 2^127 -> combined 2^254
    a_buf[0][0] = 0x3c3c3c3cu;
    b_buf[0][0] = 0xbcbcbcbcu;
    c_buf[0][0] = 0x7f800000u;

    const uint32_t *a[8], *b[8], *c[4];
    uint32_t *d[4];
    for (int r = 0; r < 8; ++r) {
      a[r] = a_buf[r].data();
      b[r] = b_buf[r].data();
    }
    for (int r = 0; r < 4; ++r) {
      c[r] = c_buf[r].data();
      d[r] = d_buf[r].data();
    }
    constexpr uint64_t bf8_pair = 1 | (1 << 3);
    ASSERT_EQ(rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL,
                                                             bf8_pair, d, a, b, c, sa.data(),
                                                             sb.data()),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[0][0], 0x7f800000u);

    ASSERT_EQ(rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, bf8_pair, d, a, b, c,
                  sa.data(), sb.data()),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[0][0], 0x7f7fffffu);

    d_buf[0].fill(0xdeadbeefu);
    const auto sentinel = d_buf;
    for (uint64_t cmod : {RJ_ARITH_MFMA_SCALE_NEG_C, RJ_ARITH_MFMA_SCALE_ABS_C}) {
      d_buf = sentinel;
      EXPECT_EQ(rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64(
                    RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, bf8_pair | cmod, d, a, b,
                    c, sa.data(), sb.data()),
                RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(d_buf, sentinel);
      EXPECT_EQ(rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64(
                    RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE, bf8_pair | cmod, d, a, b, c,
                    sa.data(), sb.data()),
                RJ_ARITH_SUCCESS);
    }
  }
}
} // namespace
