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
        return int(Rows) * K * width / (32 * 32);
      };
      const int a_regs = fill(0, fa, fb, left);
      const int b_regs = fill(16, fb, fa, right);
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
            // Compact operands provide only the VGPRs selected by their individual formats.
            const uint32_t *a[16]{}, *b[16]{}, *c[D];
            uint32_t *d[D];
            for (int i = 0; i < a_regs; ++i) {
              a[i] = storage[i].data();
            }
            for (int i = 0; i < b_regs; ++i) {
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

// Verify nonfinite inputs, extreme/NaN/subnormal scales, subnormal/Inf/NaN C, and scale
// selectors against an independent reference across all CPU tiers (including baseline and Zen4).
// Note: loose mode compares +0.0f and -0.0f numerically; exact bit assertions are used only for
// subnormal C preservation when A = B = 0 and contractual status checks.
template <int M, int Block, typename Entry>
void check_fallback_and_edges(Entry entry, uint64_t controls, bool zero_ab, bool inject_ab_nonfin,
                              bool inject_scale_special) {
  constexpr int K = 128, D = M / 2;
  const int fa = M == 32 ? 4 : int(controls & 7);
  const int fb = M == 32 ? 4 : int((controls >> 3) & 7);
  std::array<std::array<uint32_t, 32>, 16> a_buf{}, b_buf{};
  std::array<std::array<uint32_t, 32>, D> c_buf{};
  std::array<std::array<uint32_t, 32>, 2> sa_buf{}, sb_buf{};
  std::minstd_rand rng(0x1250U + uint32_t(M * 17 + Block * 31 + controls));
  if (!zero_ab) {
    for (int r = 0; r < 16; ++r) {
      for (int lane = 0; lane < 32; ++lane) {
        a_buf[r][lane] = uint32_t(rng()) & 0x3f3f3f3fU;
        b_buf[r][lane] = uint32_t(rng()) & 0x3f3f3f3fU;
      }
    }
  }
  if (inject_ab_nonfin) {
    if (fa == 0) {
      a_buf[0][0] = 0x0000ff7fU; // FP8 +NaN and -NaN
    } else if (fa == 1) {
      a_buf[0][0] = 0xfefc7c00U; // BF8 +Inf, -Inf, -NaN
    }
    if (fb == 0) {
      b_buf[1][17] = 0x7f000000U; // FP8 +NaN in second half-wave
    } else if (fb == 1) {
      b_buf[1][17] = 0x7d7c0000U; // BF8 +Inf and +NaN
    }
  }
  // Populate C with normal, +/-0.0f, +/-subnormals, +/-Inf, qNaN, sNaN.
  constexpr uint32_t special_c[8] = {
      0x00000000U, 0x80000000U, 0x00000001U, 0x80000001U,
      0x7f800000U, 0xff800000U, 0x7fc00001U, 0x7f800001U,
  };
  for (int r = 0; r < D; ++r) {
    for (int lane = 0; lane < 32; ++lane) {
      c_buf[r][lane] = lane < 8 ? special_c[lane]
                                : std::bit_cast<uint32_t>(float(int(rng() >> 20) - 2048) * 0.25f);
    }
  }
  // Populate scale registers with normal, 0, subnormal, max, and NaN bytes for E8M0/E5M3/E4M3.
  constexpr uint8_t special_s[8] = {0x00, 0x01, 0x7e, 0x7f, 0x80, 0x81, 0xfe, 0xff};
  for (int r = 0; r < 2; ++r) {
    for (int lane = 0; lane < 32; ++lane) {
      uint32_t wa = 0, wb = 0;
      for (int byte = 0; byte < 4; ++byte) {
        uint8_t av = inject_scale_special ? special_s[(lane + byte + r) & 7]
                                          : uint8_t(120 + ((lane + byte) & 7));
        uint8_t bv = inject_scale_special ? special_s[(lane + 2 * byte + 3) & 7]
                                          : uint8_t(121 + ((lane + byte) & 7));
        wa |= uint32_t(av) << (8 * byte);
        wb |= uint32_t(bv) << (8 * byte);
      }
      sa_buf[r][lane] = wa;
      sb_buf[r][lane] = wb;
    }
  }

  // Independent bit-by-bit unpack and FP32 reference (never uses the production library as oracle).
  auto unpack = [](const std::array<std::array<uint32_t, 32>, 16> &buf, int row, int k, int format,
                   int other_format) {
    const int width = format < 2 ? 8 : format < 4 ? 6 : 4;
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
    int code = 0;
    for (int bit = 0; bit < width; ++bit) {
      const int offset = local * width + bit;
      code |= int((buf[8 * (row / 16) + offset / 32][lane] >> (offset % 32)) & 1U) << bit;
    }
    return decode_matrix_format(code, format);
  };
  auto decode_scale_ref = [](uint32_t raw, int format) {
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
    const int exponent = int(raw >> 3), mantissa = int(raw & 7);
    return std::ldexp(float(exponent ? 8 + mantissa : mantissa), (exponent ? exponent : 1) - 18);
  };

  float left[M][K], right[16][K];
  for (int row = 0; row < M; ++row) {
    for (int k = 0; k < K; ++k) {
      left[row][k] = unpack(a_buf, row, k, fa, fb);
    }
  }
  for (int col = 0; col < 16; ++col) {
    for (int k = 0; k < K; ++k) {
      right[col][k] = unpack(b_buf, col, k, fb, fa);
    }
  }

  std::array<std::array<float, 32>, D> expected{};
  const int asf = int((controls >> 8) & 3);
  const int bsf = int((controls >> 10) & 3);
  const int a_row_off = 16 * int((controls >> 12) & 1);
  const int b_row_off = 16 * int((controls >> 13) & 1);
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < 16; ++col) {
      const int reg = 8 * (row / 16) + row % 8;
      const int lane = 16 * (row % 16 / 8) + col;
      uint32_t c_bits = c_buf[reg][lane];
      if (controls & RJ_ARITH_WMMA_MIXED_ABS_C) {
        c_bits &= 0x7fffffffU;
      }
      if (controls & RJ_ARITH_WMMA_MIXED_NEG_C) {
        c_bits ^= 0x80000000U;
      }
      float acc = std::bit_cast<float>(c_bits);
      if constexpr (Block != 0) {
        for (int block = 0; block < K / Block; ++block) {
          float partial = 0.0f;
          for (int k = block * Block; k < (block + 1) * Block; ++k) {
            partial = std::fma(left[row][k], right[col][k], partial);
          }
          const uint32_t sa_word = sa_buf[block / 4][M == 32 ? row : row + a_row_off];
          const uint32_t sb_word = sb_buf[block / 4][col + b_row_off];
          partial *= decode_scale_ref((sa_word >> (8 * (block % 4))) & 255U, asf);
          partial *= decode_scale_ref((sb_word >> (8 * (block % 4))) & 255U, bsf);
          acc += partial;
        }
      } else {
        for (int k = 0; k < K; ++k) {
          acc += left[row][k] * right[col][k];
        }
      }
      expected[reg][lane] = acc;
    }
  }

  const uint32_t *a_ptrs[16], *b_ptrs[16], *c_ptrs[D];
  const uint32_t *sa_ptrs[2] = {sa_buf[0].data(), sa_buf[1].data()};
  const uint32_t *sb_ptrs[2] = {sb_buf[0].data(), sb_buf[1].data()};
  for (int i = 0; i < 16; ++i) {
    a_ptrs[i] = a_buf[i].data();
    b_ptrs[i] = b_buf[i].data();
  }
  for (int i = 0; i < D; ++i) {
    c_ptrs[i] = c_buf[i].data();
  }

  for (uint64_t cpu : supported_cpu_levels()) {
    std::array<std::array<uint32_t, 32>, D> got{};
    uint32_t *d_ptrs[D];
    for (int i = 0; i < D; ++i) {
      d_ptrs[i] = got[i].data();
    }
    int status;
    if constexpr (Block != 0) {
      status = entry(cpu | RJ_ARITH_SEMANTICS_LOOSE, controls, d_ptrs, a_ptrs, b_ptrs, c_ptrs,
                     sa_ptrs, sb_ptrs);
    } else {
      status = entry(cpu | RJ_ARITH_SEMANTICS_LOOSE, controls, d_ptrs, a_ptrs, b_ptrs, c_ptrs);
    }
    ASSERT_EQ(status, RJ_ARITH_SUCCESS);
    for (int r = 0; r < D; ++r) {
      for (int lane = 0; lane < 32; ++lane) {
        const float exp_val = expected[r][lane];
        if (zero_ab && std::isfinite(exp_val) && exp_val != 0.0f &&
            std::fabs(exp_val) < 0x1p-126f) {
          // Verify subnormal C is preserved bit-for-bit (never flushed by VDPBF16PS).
          EXPECT_EQ(got[r][lane], std::bit_cast<uint32_t>(exp_val));
        } else {
          expect_f32(got[r][lane], exp_val, RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/1e-4f,
                     /*rtol=*/1e-5f);
        }
      }
    }
  }
}

TEST(WmmaMixedGfx1250, FallbackAndEdgeCases) {
  // Unscaled 16x16x128: nonfinite FP8/BF8 fallback + subnormal/Inf/NaN C across format families.
  for (int fa : {0, 1, 2, 3, 4}) {
    for (int fb : {0, 1, 2, 3, 4}) {
      const uint64_t base = uint64_t(fa) | (uint64_t(fb) << 3);
      check_fallback_and_edges<16, 0>(rj_arith_v_wmma_f32_16x16x128_f8f6f4_w32, base,
                                      /*zero_ab=*/true, /*inject_ab_nonfin=*/false, false);
      check_fallback_and_edges<16, 0>(rj_arith_v_wmma_f32_16x16x128_f8f6f4_w32,
                                      base | RJ_ARITH_WMMA_MIXED_ABS_C | RJ_ARITH_WMMA_MIXED_NEG_C,
                                      /*zero_ab=*/false, /*inject_ab_nonfin=*/true, false);
    }
  }
  // Unscaled 32x16x128 FP4 with subnormal/Inf/NaN C.
  check_fallback_and_edges<32, 0>(rj_arith_v_wmma_f32_32x16x128_f4_w32, 0, true, false, false);
  check_fallback_and_edges<32, 0>(rj_arith_v_wmma_f32_32x16x128_f4_w32, RJ_ARITH_WMMA_MIXED_NEG_C,
                                  false, false, false);
  // Scaled (Block=32 and Block=16): all scale formats (E8M0/E5M3/E4M3), scale row selectors,
  // NaN/subnormal scales, nonfinite FP8/BF8 A/B, and subnormal/Inf/NaN C.
  for (int sf = 0; sf < 3; ++sf) {
    const uint64_t sc = (uint64_t(sf) << 8) | (uint64_t((sf + 1) % 3) << 10) |
                        ((sf & 1) ? RJ_ARITH_WMMA_SCALE_ROW_A : 0) |
                        ((sf & 2) ? RJ_ARITH_WMMA_SCALE_ROW_B : 0);
    for (int fmt : {0, 1, 2, 3, 4}) {
      const uint64_t ctrl = uint64_t(fmt) | (uint64_t(fmt) << 3) | sc;
      check_fallback_and_edges<16, 32>(rj_arith_v_wmma_scale_f32_16x16x128_f8f6f4_w32, ctrl, true,
                                       false, true);
      check_fallback_and_edges<16, 32>(rj_arith_v_wmma_scale_f32_16x16x128_f8f6f4_w32, ctrl, false,
                                       true, true);
      check_fallback_and_edges<16, 16>(rj_arith_v_wmma_scale16_f32_16x16x128_f8f6f4_w32, ctrl, true,
                                       false, true);
      check_fallback_and_edges<16, 16>(rj_arith_v_wmma_scale16_f32_16x16x128_f8f6f4_w32, ctrl,
                                       false, true, true);
    }
    check_fallback_and_edges<32, 32>(rj_arith_v_wmma_scale_f32_32x16x128_f4_w32, sc, true, false,
                                     true);
    check_fallback_and_edges<32, 32>(rj_arith_v_wmma_scale_f32_32x16x128_f4_w32, sc, false, false,
                                     true);
    check_fallback_and_edges<32, 16>(rj_arith_v_wmma_scale16_f32_32x16x128_f4_w32, sc, true, false,
                                     true);
    check_fallback_and_edges<32, 16>(rj_arith_v_wmma_scale16_f32_32x16x128_f4_w32, sc, false, false,
                                     true);
  }

  // Exact -0.0f preservation on unscaled 16x16x128 across all 24 non-(FP4, FP4) pairs and all
  // supported CPU tiers, plus verification that +0.0f and finite sums are unaffected by -0.0f init.
  auto pack_constant = [](std::array<std::array<uint32_t, 32>, 16> &buf, int fmt, int code) {
    for (std::array<uint32_t, 32> &reg : buf) {
      reg.fill(0);
    }
    const int width = fmt < 2 ? 8 : fmt < 4 ? 6 : 4;
    for (int reg = 0; reg < (width == 8 ? 16 : width == 6 ? 12 : 8); ++reg) {
      for (int lane = 0; lane < 32; ++lane) {
        uint32_t word = 0;
        for (int bit = 0; bit < 32; ++bit) {
          const int pos = (reg * 32 + bit) % width;
          if ((code >> pos) & 1) {
            word |= 1U << bit;
          }
        }
        buf[reg][lane] = word;
      }
    }
  };
  constexpr int kNegZeroCode[5] = {0x80, 0x80, 0x20, 0x20, 0x08};
  constexpr int kPosOneCode[5] = {0x38, 0x3c, 0x08, 0x10, 0x02};
  for (int fa : {0, 1, 2, 3, 4}) {
    for (int fb : {0, 1, 2, 3, 4}) {
      if (fa == 4 && fb == 4) {
        continue; // Exact-integer FP4xFP4 dot yields +0.0f (documented Loose variation).
      }
      const uint64_t ctrl = uint64_t(fa) | (uint64_t(fb) << 3);
      std::array<std::array<uint32_t, 32>, 16> a_neg0{}, a_pos0{}, b_one{};
      std::array<std::array<uint32_t, 32>, 8> c_neg0{};
      pack_constant(a_neg0, fa, kNegZeroCode[fa]);
      pack_constant(a_pos0, fa, 0);
      pack_constant(b_one, fb, kPosOneCode[fb]);
      for (std::array<uint32_t, 32> &reg : c_neg0) {
        reg.fill(0x80000000U);
      }
      const uint32_t *an_p[16], *ap_p[16], *b_p[16], *c_p[8];
      for (int i = 0; i < 16; ++i) {
        an_p[i] = a_neg0[i].data();
        ap_p[i] = a_pos0[i].data();
        b_p[i] = b_one[i].data();
      }
      for (int i = 0; i < 8; ++i) {
        c_p[i] = c_neg0[i].data();
      }
      for (uint64_t cpu : supported_cpu_levels()) {
        std::array<std::array<uint32_t, 32>, 8> got{};
        uint32_t *d_p[8];
        for (int i = 0; i < 8; ++i) {
          d_p[i] = got[i].data();
        }
        ASSERT_EQ(rj_arith_v_wmma_f32_16x16x128_f8f6f4_w32(cpu | RJ_ARITH_SEMANTICS_LOOSE, ctrl,
                                                           d_p, an_p, b_p, c_p),
                  RJ_ARITH_SUCCESS);
        for (int r = 0; r < 8; ++r) {
          for (int lane = 0; lane < 32; ++lane) {
            EXPECT_EQ(got[r][lane], 0x80000000U) << "fa=" << fa << " fb=" << fb << " cpu=" << cpu;
          }
        }
        ASSERT_EQ(rj_arith_v_wmma_f32_16x16x128_f8f6f4_w32(cpu | RJ_ARITH_SEMANTICS_LOOSE, ctrl,
                                                           d_p, ap_p, b_p, c_p),
                  RJ_ARITH_SUCCESS);
        for (int r = 0; r < 8; ++r) {
          for (int lane = 0; lane < 32; ++lane) {
            EXPECT_EQ(got[r][lane], 0x00000000U) << "fa=" << fa << " fb=" << fb << " cpu=" << cpu;
          }
        }
      }
    }
  }
}
} // namespace
