// SPDX-License-Identifier: MIT

// gfx942 MFMA captured outputs, exact-mode selection, architecture isolation and aliasing.
#include "common/cpu.h"
#include "common/half_reference.h"
#include "mfma/gfx942_cases.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <span>
#include <string_view>

namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_mfma_f32_16x16x16_f16_w64);
struct Form {
  const char *name;
  int m, k, batches, format_a, format_b;
  Entry entry, gfx950;
  std::span<const uint32_t> captured;
};

const Form kForms[] = {
    {"f16_32x32x8", 32, 8, 1, 0, 0, rj_arith_v_mfma_f32_32x32x8_f16_w64,
     rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx950, kf16_32x32x8},
    {"f16_16x16x16", 16, 16, 1, 0, 0, rj_arith_v_mfma_f32_16x16x16_f16_w64,
     rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx950, kf16_16x16x16},
    {"f16_32x32x4_2b", 32, 4, 2, 0, 0, rj_arith_v_mfma_f32_32x32x4_2b_f16_w64,
     rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx950, kf16_32x32x4_2b},
    {"f16_16x16x4_4b", 16, 4, 4, 0, 0, rj_arith_v_mfma_f32_16x16x4_4b_f16_w64,
     rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx950, kf16_16x16x4_4b},
    {"f16_4x4x4_16b", 4, 4, 16, 0, 0, rj_arith_v_mfma_f32_4x4x4_16b_f16_w64,
     rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx950, kf16_4x4x4_16b},
    {"bf16_32x32x8", 32, 8, 1, 1, 1, rj_arith_v_mfma_f32_32x32x8_bf16_w64,
     rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx950, kbf16_32x32x8},
    {"bf16_16x16x16", 16, 16, 1, 1, 1, rj_arith_v_mfma_f32_16x16x16_bf16_w64,
     rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx950, kbf16_16x16x16},
    {"bf16_32x32x4_2b", 32, 4, 2, 1, 1, rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64,
     rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx950, kbf16_32x32x4_2b},
    {"bf16_16x16x4_4b", 16, 4, 4, 1, 1, rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64,
     rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx950, kbf16_16x16x4_4b},
    {"bf16_4x4x4_16b", 4, 4, 16, 1, 1, rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64,
     rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx950, kbf16_4x4x4_16b},
    {"xf32_32x32x4", 32, 4, 1, 2, 2, rj_arith_v_mfma_f32_32x32x4_xf32_w64,
     rj_arith_v_mfma_f32_32x32x4_xf32_w64_gfx950, kxf32_32x32x4},
    {"xf32_16x16x8", 16, 8, 1, 2, 2, rj_arith_v_mfma_f32_16x16x8_xf32_w64,
     rj_arith_v_mfma_f32_16x16x8_xf32_w64_gfx950, kxf32_16x16x8},
    {"fp8_32x32x16", 32, 16, 1, 3, 3, rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64,
     rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64_gfx950, kfp8_32x32x16},
    {"bf8_32x32x16", 32, 16, 1, 4, 4, rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64,
     rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64_gfx950, kbf8_32x32x16},
    {"fp8_bf8_32x32x16", 32, 16, 1, 3, 4, rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64,
     rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64_gfx950, kfp8_bf8_32x32x16},
    {"bf8_fp8_32x32x16", 32, 16, 1, 4, 3, rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64,
     rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64_gfx950, kbf8_fp8_32x32x16},
    {"fp8_fp8_16x16x32", 16, 32, 1, 3, 3, rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64,
     rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950, kfp8_fp8_16x16x32},
    {"bf8_bf8_16x16x32", 16, 32, 1, 4, 4, rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64,
     rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64_gfx950, kbf8_bf8_16x16x32},
    {"fp8_bf8_16x16x32", 16, 32, 1, 3, 4, rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64,
     rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64_gfx950, kfp8_bf8_16x16x32},
    {"bf8_fp8_16x16x32", 16, 32, 1, 4, 3, rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64,
     rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64_gfx950, kbf8_fp8_16x16x32},
};

// Encode a staged capture without sharing the implementation's source decoder.
uint32_t encode(uint32_t bits, int format) {
  if (format == 0) {
    return half_bits(std::bit_cast<float>(bits));
  }
  if (format == 1) {
    return bits >> 16;
  }
  if (format == 2) {
    return bits;
  }
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return 0x80;
  }
  const int fraction = format == 4 ? 2 : 3;
  const int bias = format == 4 ? 16 : 8;
  for (int code = 0; code < 256; ++code) {
    if (code == 0x80) {
      continue;
    }
    const int exponent = (code & 0x7f) >> fraction;
    const int mantissa = code & ((1 << fraction) - 1);
    float value = std::ldexp(float(exponent ? (1 << fraction) + mantissa : mantissa),
                             (exponent ? exponent : 1) - bias - fraction);
    if (code & 0x80) {
      value = -value;
    }
    if (std::bit_cast<uint32_t>(value) == bits) {
      return uint32_t(code);
    }
  }
  ADD_FAILURE() << "Capture is not representable: " << bits;
  return 0;
}

struct Placement {
  int batch, row, col;
};

// Place the same captured dot in a selected matrix/batch; other finite outputs are zero.
void captured_wave(const Form &form, Placement position = {}, bool zero_other_outputs = true) {
  SCOPED_TRACE(::testing::Message() << form.name << " batch " << position.batch << " row "
                                    << position.row << " column " << position.col);
  const int per_word = form.format_a == 2 ? 1 : form.format_a < 2 ? 2 : 4;
  const int inputs = form.m * form.k * form.batches / (64 * per_word);
  const int outputs = form.m * form.m * form.batches / 64;
  const int per_lane = form.k * form.m * form.batches / 64;
  std::array<std::array<uint32_t, 64>, 72> input{};
  for (int depth = 0; depth < form.k; ++depth) {
    const int lane = position.batch * form.m + depth / per_lane * form.m * form.batches;
    const int reg = depth % per_lane / per_word;
    const int shift = 32 / per_word * (depth % per_word);
    input[reg][lane + position.row] |= encode(form.captured[depth], form.format_a) << shift;
    input[inputs + reg][lane + position.col] |= encode(form.captured[form.k + depth], form.format_b)
                                                << shift;
  }
  // In the 4x4 form each VGPR carries 16 batches, one per four-lane group.
  const int blocks_per_register = form.m == 4 ? 16 : 1;
  const int output_reg = position.batch * (form.m * form.m / 64) +
                         position.row / (256 / form.m) * 4 + position.row % 4;
  const int output_lane = position.batch % blocks_per_register * form.m +
                          position.row / 4 % (64 / form.m) * blocks_per_register * form.m +
                          position.col;
  input[2 * inputs + output_reg][output_lane] = form.captured[2 * form.k];
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      for (int aliasing_offset : {0, inputs, 2 * inputs, 2 * inputs + outputs}) {
        auto storage = input;
        const uint32_t *a[8], *b[8], *c[32];
        uint32_t *d[32];
        for (int reg = 0; reg < inputs; ++reg) {
          a[reg] = storage[reg].data();
          b[reg] = storage[inputs + reg].data();
        }
        for (int reg = 0; reg < outputs; ++reg) {
          c[reg] = storage[2 * inputs + reg].data();
          d[reg] = storage[aliasing_offset + reg].data();
        }
        ASSERT_EQ(form.entry(cpu | semantics, 0, d, a, b, c), RJ_ARITH_SUCCESS);
        for (int reg = 0; reg < int(storage.size()); ++reg) {
          for (int lane = 0; lane < 64; ++lane) {
            const bool written = reg >= aliasing_offset && reg < aliasing_offset + outputs;
            if (written && !zero_other_outputs &&
                (reg != aliasing_offset + output_reg || lane != output_lane)) {
              continue;
            }
            const uint32_t expected = !written ? input[reg][lane]
                                      : reg == aliasing_offset + output_reg && lane == output_lane
                                          ? form.captured[2 * form.k + 1]
                                          : 0;
            ASSERT_EQ(storage[reg][lane], expected) << "register " << reg << " lane " << lane;
          }
        }
      }
    }
  }
}

TEST(MfmaGfx942Exact, CapturedOutputsAndAliasing) {
  for (const auto &form : kForms) {
    const size_t row_words = size_t(2 * form.k + 2);
    ASSERT_EQ(form.captured.size() % row_words, 0U);
    for (size_t offset = 0; offset < form.captured.size(); offset += row_words) {
      SCOPED_TRACE(offset / row_words);
      auto test = form;
      test.captured = form.captured.subspan(offset, row_words);
      bool finite_factors = true;
      for (int i = 0; i < 2 * form.k; ++i) {
        finite_factors &= ((test.captured[i] >> 23) & 0xffU) != 0xffU;
      }
      for (Placement position :
           {Placement{0, 0, 0}, Placement{form.batches / 2, form.m / 2, form.m / 2 + 1},
            Placement{form.batches - 1, form.m - 1, form.m - 1}}) {
        captured_wave(test, position, finite_factors);
      }
    }
  }
}

TEST(MfmaGfx942Exact, SpecialValues) {
  for (const auto &form : kForms) {
    const bool f16 = form.format_a == 0 && form.m == 32 && form.k == 8;
    const bool bf8 = form.format_a == 4 && form.format_b == 4 && form.m == 32;
    if (!f16 && !bf8) {
      continue;
    }
    std::array<uint32_t, 66> values{};
    // Infinity times zero is canonical NaN; a large BF8 dot must retain an infinite C.
    values[0] = bf8 ? 0x47000000U : 0x7f800000U;
    values[form.k] = bf8 ? 0x47000000U : 0;
    values[2 * form.k] = bf8 ? 0x7f800000U : 0;
    values[2 * form.k + 1] = bf8 ? 0x7f800000U : 0xffc00000U;
    auto test = form;
    test.captured = std::span<const uint32_t>(values).first(2 * form.k + 2);
    captured_wave(test, {}, false);
  }
}

TEST(MfmaGfx942Exact, ProductsAt2To128CountAsInfinite) {
  // MI300X-captured BF16 overflow: a 2^64 * 2^64 product counts as +Inf even with C = -2^127,
  // and opposite-sign +-2^128 products produce canonical NaN.
  constexpr uint32_t kPos2To64 = 0x5f800000U;
  constexpr uint32_t kNeg2To64 = 0xdf800000U;
  for (const auto &form : kForms) {
    if (form.format_a != 1 || form.format_b != 1) {
      continue;
    }
    std::array<uint32_t, 34> single_overflow{};
    single_overflow[0] = kPos2To64;
    single_overflow[form.k] = kPos2To64;
    single_overflow[2 * form.k] = 0xff000000U;
    single_overflow[2 * form.k + 1] = 0x7f800000U;
    auto test = form;
    test.captured = std::span<const uint32_t>(single_overflow).first(2 * form.k + 2);
    captured_wave(test);

    std::array<uint32_t, 34> opposite_overflows{};
    opposite_overflows[0] = kPos2To64;
    opposite_overflows[1] = kNeg2To64;
    opposite_overflows[form.k] = kPos2To64;
    opposite_overflows[form.k + 1] = kPos2To64;
    opposite_overflows[2 * form.k + 1] = 0xffc00000U;
    test.captured = std::span<const uint32_t>(opposite_overflows).first(2 * form.k + 2);
    captured_wave(test);
  }
}

TEST(MfmaGfx942Exact, Bf16OutputDenormals) {
  // MI300X-captured subnormal BF16 product outputs, nearest-even ties, and negative
  // underflow to +0.0 at C = 0.
  struct DenormalCase {
    uint16_t a, b;
    uint32_t expected;
  };
  constexpr DenormalCase kDenormals[] = {
      {0x0080U, 0x3f00U, 0x00400000U}, {0x0001U, 0x3780U, 0x00000001U},
      {0x0001U, 0x3700U, 0x00000000U}, {0x0001U, 0x37c0U, 0x00000002U},
      {0x8001U, 0x3700U, 0x00000000U}, {0x8001U, 0x37c0U, 0x80000002U},
  };
  for (const auto &form : kForms) {
    if (form.format_a != 1 || form.format_b != 1) {
      continue;
    }
    for (const auto &denormal : kDenormals) {
      std::array<uint32_t, 34> values{};
      values[0] = uint32_t(denormal.a) << 16;
      values[form.k] = uint32_t(denormal.b) << 16;
      values[2 * form.k + 1] = denormal.expected;
      auto test = form;
      test.captured = std::span<const uint32_t>(values).first(2 * form.k + 2);
      captured_wave(test);
    }
  }
}

TEST(MfmaGfx942Exact, Xf32LowPayloadNansTruncateToInfinity) {
  // MI300X-captured XF32 pre-reduction truncation: NaNs with payload only in bits 12:0
  // truncate to +-Inf, while bit 13+ payloads remain NaN.
  struct Xf32Corner {
    uint32_t a, b, expected;
  };
  constexpr Xf32Corner kCorners[] = {
      {0x7f800001U, 0x3f800000U, 0x7f800000U}, {0xff800001U, 0x3f800000U, 0xff800000U},
      {0x7f801fffU, 0x3f800000U, 0x7f800000U}, {0x7f802000U, 0x3f800000U, 0xffc00000U},
      {0x7f800001U, 0x00000000U, 0xffc00000U},
  };
  for (const auto &form : kForms) {
    if (form.format_a != 2) {
      continue;
    }
    for (const auto &corner : kCorners) {
      std::array<uint32_t, 18> values{};
      values[0] = corner.a;
      values[form.k] = corner.b;
      values[2 * form.k + 1] = corner.expected;
      auto test = form;
      test.captured = std::span<const uint32_t>(values).first(2 * form.k + 2);
      captured_wave(test, {}, false);
    }
  }
}

TEST(MfmaGfx942Exact, QualificationIsLocalToTargetAndControls) {
  for (const auto &form : kForms) {
    SCOPED_TRACE(form.name);
    const int per_word = form.format_a == 2 ? 1 : form.format_a < 2 ? 2 : 4;
    const int inputs = form.m * form.k * form.batches / (64 * per_word);
    const int outputs = form.m * form.m * form.batches / 64;
    std::array<std::array<uint32_t, 64>, 32> a_buf{}, b_buf{}, c_buf{}, d_buf{};
    const uint32_t *a[8], *b[8], *c[32];
    uint32_t *d[32];
    for (int reg = 0; reg < inputs; ++reg) {
      a[reg] = a_buf[reg].data();
      b[reg] = b_buf[reg].data();
    }
    for (int reg = 0; reg < outputs; ++reg) {
      c[reg] = c_buf[reg].data();
      d[reg] = d_buf[reg].data();
    }
    // Reserved CBSZ (5..7) and out-of-range ABID >= 2^CBSZ remain rejected in strict modes
    // before accessing operands, leaving D untouched while Loose and ExactAllowLoose execute
    // the MI300X-aligned approximate path.
    for (uint64_t unqualified_control :
         {5ULL, 7ULL, 8ULL, 1ULL | (2ULL << 3), 1ULL | (5ULL << 3), 2ULL | (4ULL << 3),
          3ULL | (8ULL << 3), 5ULL | (3ULL << 7)}) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
        for (uint64_t ovfl : {uint64_t{0}, RJ_ARITH_FP16_OVFL}) {
          EXPECT_EQ(
              form.entry(semantics | ovfl, unqualified_control, nullptr, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
          d_buf[0][0] = 0xdeadbeefU;
          EXPECT_EQ(form.entry(semantics | ovfl, unqualified_control, d, a, b, c),
                    RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
          EXPECT_EQ(d_buf[0][0], 0xdeadbeefU);
        }
      }
      for (uint64_t ovfl : {uint64_t{0}, RJ_ARITH_FP16_OVFL}) {
        // Seed C with both +0.0 and qNaN so ExactAllowLoose with FP16_OVFL is verified to
        // preserve NaN identically to Loose rather than flushing to +0.0 via the exact path.
        for (uint32_t c_init : {0U, 0x7fc00000U}) {
          c_buf[0][0] = c_init;
          d_buf[0][0] = 0xdeadbeefU;
          EXPECT_EQ(form.entry(RJ_ARITH_SEMANTICS_LOOSE | ovfl, unqualified_control, d, a, b, c),
                    RJ_ARITH_SUCCESS);
          const uint32_t loose_out = d_buf[0][0];
          EXPECT_EQ(loose_out, c_init);
          d_buf[0][0] = 0xdeadbeefU;
          EXPECT_EQ(form.entry(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE | ovfl,
                               unqualified_control, d, a, b, c),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(d_buf[0][0], loose_out);
        }
      }
      c_buf[0][0] = 0U;
    }
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      for (uint64_t controls :
           {uint64_t{0}, uint64_t{2} | (uint64_t{3} << 3) | (uint64_t{2} << 7)}) {
        EXPECT_EQ(form.gfx950(semantics, controls, nullptr, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(form.gfx950(semantics | RJ_ARITH_FP16_OVFL, controls, nullptr, nullptr, nullptr,
                              nullptr),
                  RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
      }
    }
  }
}

uint64_t splitmix64_controls(uint64_t &state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

double unit_controls(uint64_t &state) {
  return double(splitmix64_controls(state) >> 11) * (1.0 / 9007199254740992.0);
}

uint16_t f16_bits_from_double(double x) {
  const uint64_t u = std::bit_cast<uint64_t>(x);
  const uint16_t sign = static_cast<uint16_t>((u >> 48) & 0x8000U);
  const int exp = static_cast<int>((u >> 52) & 0x7ffU);
  const uint64_t frac = u & 0x000fffffffffffffULL;
  if (exp == 0x7ff) {
    return sign | (frac ? 0x7e00U : 0x7c00U);
  }
  if (exp == 0 && frac == 0) {
    return sign;
  }
  const int unbiased = exp - 1023;
  if (unbiased > 15) {
    return sign | 0x7c00U;
  }
  const uint64_t sig = (uint64_t{1} << 52) | frac;
  int h_exp = unbiased + 15;
  int shift = 42;
  if (h_exp <= 0) {
    shift += (1 - h_exp);
    h_exp = 0;
    if (shift > 53) {
      return sign;
    }
  }
  const uint64_t kept = sig >> shift;
  const uint64_t rem = sig & ((uint64_t{1} << shift) - 1);
  const uint64_t half = uint64_t{1} << (shift - 1);
  const uint16_t round = (rem > half || (rem == half && (kept & 1))) ? 1U : 0U;
  uint16_t mag = static_cast<uint16_t>((h_exp << 10) + (h_exp ? (kept & 0x3ffU) : kept) + round);
  if (mag > 0x7c00U) {
    mag = 0x7c00U;
  }
  return sign | mag;
}

uint16_t bf16_bits_from_double(double x) {
  const uint32_t b = std::bit_cast<uint32_t>(static_cast<float>(x));
  if ((b & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<uint16_t>((b >> 16) | 0x40U);
  }
  return static_cast<uint16_t>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16);
}

void fill_controls_elem(uint8_t *dst, int elem, uint64_t &state, int wave_kind, int lane,
                        int idx_in_lane) {
  if (wave_kind == 0 || wave_kind == 1) {
    const bool tag_a = (wave_kind == 0);
    const int tag = lane * 4 + (idx_in_lane % 4) + 1;
    if (elem == 0) {
      const uint16_t v =
          f16_bits_from_double(tag_a ? double(tag) * 0.125 : double(tag % 17 + 1) * 0.25);
      std::memcpy(dst, &v, 2);
    } else if (elem == 1) {
      const uint16_t v =
          bf16_bits_from_double(tag_a ? double(tag) * 0.5 : double(tag % 17 + 1) * 0.25);
      std::memcpy(dst, &v, 2);
    } else if (elem == 2) {
      const float v = float(tag) * 0.25f + 0.0625f * float(idx_in_lane);
      std::memcpy(dst, &v, 4);
    } else {
      uint8_t code = uint8_t(1 + ((lane * 3 + idx_in_lane * 7) % 126));
      if ((lane + idx_in_lane) & 1) {
        code |= 0x80U;
      }
      if (code == 0x80U) {
        code = 0x81U;
      }
      *dst = code;
    }
    return;
  }

  const bool neg = (splitmix64_controls(state) & 1ULL) != 0;
  const double sign = neg ? -1.0 : 1.0;
  double x = 0.0;
  if (wave_kind == 2) {
    x = 2.0 * unit_controls(state) - 1.0;
  } else if (wave_kind == 3) {
    const int lo = elem == 0 ? -14 : -60;
    const int hi = elem == 0 ? 14 : 60;
    x = sign * std::ldexp(1.0 + unit_controls(state),
                          lo + int(splitmix64_controls(state) % uint64_t(hi - lo + 1)));
  } else if (wave_kind == 4) {
    if (elem == 0) {
      x = sign * (32768.0 + 32736.0 * unit_controls(state));
    } else if (elem <= 2) {
      x = sign *
          std::ldexp(1.5 + 0.4 * unit_controls(state), 63 + int(splitmix64_controls(state) % 3));
    } else {
      uint8_t code = uint8_t(0x78U + (splitmix64_controls(state) % 8));
      if (neg) {
        code |= 0x80U;
      }
      *dst = code;
      return;
    }
  } else {
    switch (splitmix64_controls(state) % 6) {
    case 0:
      x = std::bit_cast<double>(neg ? uint64_t{0xfff0000000000000ULL}
                                    : uint64_t{0x7ff0000000000000ULL});
      break;
    case 1:
      x = std::bit_cast<double>(uint64_t{0x7ff8000000000000ULL});
      break;
    case 2:
      x = neg ? -0.0 : 0.0;
      break;
    case 3:
      x = sign * (elem == 0 ? 5.9604644775390625e-08 : 1.0e-40);
      break;
    default:
      x = sign * (1.0 + unit_controls(state)) * 64.0;
      break;
    }
    if (elem >= 3) {
      uint8_t code = uint8_t(splitmix64_controls(state));
      if (splitmix64_controls(state) % 5 == 0) {
        code = 0x80U;
      }
      *dst = code;
      return;
    }
  }

  if (elem == 0) {
    const uint16_t b = f16_bits_from_double(x);
    std::memcpy(dst, &b, 2);
  } else if (elem == 1) {
    const uint16_t b = bf16_bits_from_double(x);
    std::memcpy(dst, &b, 2);
  } else if (elem == 2) {
    const float f = static_cast<float>(x);
    std::memcpy(dst, &f, 4);
  } else {
    const float f = static_cast<float>(x);
    *dst = static_cast<uint8_t>(std::bit_cast<uint32_t>(f) & 0xffU);
  }
}

const Form &find_form(std::string_view name) {
  for (const auto &form : kForms) {
    if (name == form.name) {
      return form;
    }
  }
  std::abort();
}

int capture_seed_index(std::string_view name) {
  constexpr std::string_view kCaptureOrder[] = {
      "f16_32x32x8",      "f16_16x16x16",     "f16_32x32x4_2b",   "f16_16x16x4_4b",
      "f16_4x4x4_16b",    "bf16_32x32x8",     "bf16_16x16x16",    "bf16_32x32x4_2b",
      "bf16_16x16x4_4b",  "bf16_4x4x4_16b",   "xf32_32x32x4",     "xf32_16x16x8",
      "fp8_32x32x16",     "bf8_32x32x16",     "fp8_bf8_32x32x16", "bf8_fp8_32x32x16",
      "fp8_fp8_16x16x32", "fp8_bf8_16x16x32", "bf8_fp8_16x16x32", "bf8_bf8_16x16x32",
  };
  for (size_t i = 0; i < std::size(kCaptureOrder); ++i) {
    if (name == kCaptureOrder[i]) {
      return int(i);
    }
  }
  std::abort();
}

uint64_t hash_vgprs_fnv1a(const std::array<std::array<uint32_t, 64>, 32> &regs, int outputs) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int r = 0; r < outputs; ++r) {
    for (int lane = 0; lane < 64; ++lane) {
      const uint32_t w = regs[r][lane];
      for (int b = 0; b < 4; ++b) {
        h ^= uint64_t((w >> (8 * b)) & 0xffU);
        h *= 0x100000001b3ULL;
      }
    }
  }
  return h;
}

TEST(MfmaGfx942Exact, SourceLaneControlsMatchMi300xSilicon) {
  // Captured on AMD Instinct MI300X VF (gfx942, ROCm 7.2.4 / HIP 7.2.53211, mfma_controls.bin):
  // single-batch (Batch == 1) dense MFMA forms ignore CBSZ/ABID/BLGP, while multi-batch
  // (Batch > 1) F16/BF16 forms apply BLGP to B and adjacent M-lane block broadcast (clamping to
  // lane % 16 when M << CBSZ > 64) to A across CBSZ <= 4 && ABID < 2^CBSZ.
  struct ControlSiliconCase {
    const char *form_name;
    uint32_t cbsz, abid, blgp;
    uint64_t wave0_fnv1a, wave2_fnv1a, wave5_fnv1a;
  };
  constexpr ControlSiliconCase kQualifiedCases[] = {
      {"f16_32x32x8", 2, 3, 2, 0x256c07a0b9dbac54ULL, 0x6d9c226eb84c487fULL, 0x00cb242081fb5dbeULL},
      {"f16_16x16x16", 4, 15, 7, 0x82d2299a8108c65dULL, 0x4962a4322bbbb81cULL,
       0x1a82a679fbb74b25ULL},
      {"bf16_32x32x8", 3, 5, 3, 0x8833c74b774bd77aULL, 0x2acd70f4b55d165aULL,
       0x996a9ad2d8bdc2a5ULL},
      {"bf16_16x16x16", 1, 1, 1, 0x09bd74749d55785bULL, 0xd5c3cf713c6290deULL,
       0x1a82a679fbb74b25ULL},
      {"xf32_32x32x4", 2, 3, 2, 0x239af12e175ed22cULL, 0x33cea65fc6cdb878ULL,
       0xbe24c1a949d745feULL},
      {"xf32_16x16x8", 4, 9, 6, 0x714911ff64f3a116ULL, 0xd6f940e5242a196bULL,
       0x196ed7a289ec2aeaULL},
      {"fp8_32x32x16", 4, 15, 7, 0xfdb5262a70c6e07eULL, 0x97668aa1f75ba101ULL,
       0x623bdcfa1176c325ULL},
      {"bf8_32x32x16", 2, 3, 2, 0xd2d9be59abf4c8e0ULL, 0xdc4041211cbcb6dbULL,
       0x623bdcfa1176c325ULL},
      {"fp8_bf8_32x32x16", 3, 5, 3, 0xdbc152358ae8ccf0ULL, 0xeeae72bb28bad3a6ULL,
       0x623bdcfa1176c325ULL},
      {"bf8_fp8_32x32x16", 1, 1, 1, 0x93f644bbc5a999f5ULL, 0x6a5dd9984c3d33eaULL,
       0x623bdcfa1176c325ULL},
      {"fp8_fp8_16x16x32", 4, 9, 6, 0xdca95a19ef3357a3ULL, 0x7f15c7224aff83b9ULL,
       0x1a82a679fbb74b25ULL},
      {"fp8_bf8_16x16x32", 2, 3, 2, 0xc435b61eab9d3599ULL, 0xf4cb50b72da2bb80ULL,
       0x1a82a679fbb74b25ULL},
      {"bf8_fp8_16x16x32", 3, 5, 3, 0x46abf0c0ab2e610aULL, 0x0d963fbe3e0c636fULL,
       0x1a82a679fbb74b25ULL},
      {"bf8_bf8_16x16x32", 4, 15, 7, 0xe2df9f213585b20dULL, 0x22bc9fa745f7b458ULL,
       0x1a82a679fbb74b25ULL},
      {"f16_32x32x4_2b", 1, 0, 0, 0x784453fa7c8011cfULL, 0xef4edb27e5f89292ULL,
       0x4287e7e9dec39bccULL},
      {"f16_32x32x4_2b", 1, 1, 1, 0x2c52dd1af2af9f46ULL, 0x2d4f78151143a59aULL,
       0x5f014a8b9a4131fbULL},
      {"f16_32x32x4_2b", 2, 3, 2, 0x361b97756d46bc19ULL, 0xdc485257acf3d0ccULL,
       0x160f1eed316f963bULL},
      {"f16_32x32x4_2b", 4, 9, 6, 0x0ad6a32031f33aedULL, 0x2de0ac90b522a2c9ULL,
       0x439381d2406bd7a6ULL},
      {"f16_16x16x4_4b", 1, 0, 0, 0x0f756ce635f1d575ULL, 0x170f481b547182ebULL,
       0x8a1f1fa67e2f7fddULL},
      {"f16_16x16x4_4b", 1, 1, 0, 0xc0eefca48b3cfdb8ULL, 0x9fbff02d636b1fe8ULL,
       0x1fd04a7f5f6fde32ULL},
      {"f16_16x16x4_4b", 2, 3, 2, 0xd9d18e2c809fd05aULL, 0x2246aea62fe676faULL,
       0x231784c56a0734fbULL},
      {"f16_16x16x4_4b", 3, 5, 3, 0x96087ae0909b1bd9ULL, 0x436ca38ff849d512ULL,
       0xef8c26af06727e64ULL},
      {"f16_4x4x4_16b", 1, 1, 1, 0x7804096b98b3c39bULL, 0x043f06320622120fULL,
       0x30bf29198f2a7df2ULL},
      {"f16_4x4x4_16b", 2, 3, 2, 0x33e3e17ff61ad3e3ULL, 0x0ec6016855f21c97ULL,
       0x45634eebd54692a3ULL},
      {"f16_4x4x4_16b", 3, 5, 3, 0x6cb61df41e023033ULL, 0x534e3087c88736c4ULL,
       0x286d3e2b46e75d1dULL},
      {"f16_4x4x4_16b", 4, 9, 6, 0x4bc574c92fb80468ULL, 0x88518dd5fd123e43ULL,
       0x9771ef69d7e6b018ULL},
      {"f16_4x4x4_16b", 4, 15, 7, 0x49b54194cf00570dULL, 0x8d417b4a83beb68eULL,
       0xe9d83ba5f5a7f37fULL},
      {"bf16_32x32x4_2b", 1, 1, 1, 0x6a37d3dcad7178adULL, 0x0c0a78893d1e4bddULL,
       0x3d05b7c3ea1e4d2fULL},
      {"bf16_32x32x4_2b", 3, 5, 3, 0xb2bf876f0248abe5ULL, 0x798e2de328423244ULL,
       0xdee3fe04d500829dULL},
      {"bf16_16x16x4_4b", 1, 1, 0, 0xb7364d6434a953d1ULL, 0x9b3c2a8076e49dbfULL,
       0xbfe869376f423121ULL},
      {"bf16_16x16x4_4b", 2, 2, 0, 0x0a4fd50cac300e14ULL, 0x42014435782f5ae6ULL,
       0x2088607a91df6c81ULL},
      {"bf16_16x16x4_4b", 4, 15, 7, 0xeaae1944b4f53aa8ULL, 0x595776eeeb74eb0bULL,
       0x7f36b812ec14f552ULL},
      {"bf16_4x4x4_16b", 0, 0, 1, 0x589ad4cc14fb4eddULL, 0x7b5f7cc0505ed1ccULL,
       0x0da3397ebe096db7ULL},
      {"bf16_4x4x4_16b", 0, 0, 2, 0xebf41789df7426a1ULL, 0x5622dac95d2ec234ULL,
       0xa3cff56e935d1234ULL},
      {"bf16_4x4x4_16b", 0, 0, 3, 0xce1f3b17081ef2ccULL, 0xe9197ae4c44d8b89ULL,
       0x89f3674c807584e3ULL},
      {"bf16_4x4x4_16b", 0, 0, 4, 0xe38afd3db8a43a3aULL, 0xe8e4c52418a89790ULL,
       0xefe33293431c3b99ULL},
      {"bf16_4x4x4_16b", 0, 0, 5, 0x2f8eb7711583644aULL, 0xc4a9f2d67dd8ee47ULL,
       0x26ee7330e85d899cULL},
      {"bf16_4x4x4_16b", 0, 0, 6, 0x65e29749c3af66daULL, 0xbddf813a685e5183ULL,
       0x1edc9cf574433a48ULL},
      {"bf16_4x4x4_16b", 0, 0, 7, 0xd47deb3596ed3499ULL, 0xf7a1ee2d5215fe71ULL,
       0x6825f1b2ccebc635ULL},
      {"bf16_4x4x4_16b", 4, 15, 7, 0xac618efa6bdda60dULL, 0xf69c71aa4167322fULL,
       0xaed99c8f60f54936ULL},
  };

  struct WaveBuffers {
    std::array<std::array<uint32_t, 64>, 2> a{}, b{};
    std::array<std::array<uint32_t, 64>, 32> c{};
  };
  auto generate_waves = [](const Form &form) {
    std::array<WaveBuffers, 6> waves{};
    const int outputs = form.m * form.m * form.batches / 64;
    const int cd_bytes = outputs * 4;
    const int esz_a = form.format_a <= 1 ? 2 : form.format_a == 2 ? 4 : 1;
    const int esz_b = form.format_b <= 1 ? 2 : form.format_b == 2 ? 4 : 1;
    uint64_t state = 0x2026112400ULL + uint64_t(capture_seed_index(form.name));
    for (int w = 0; w < 6; ++w) {
      for (int lane = 0; lane < 64; ++lane) {
        std::array<uint8_t, 8> a_bytes{}, b_bytes{};
        for (int off = 0, idx = 0; off < 8; off += esz_a, ++idx) {
          fill_controls_elem(&a_bytes[size_t(off)], form.format_a, state, w, lane, idx);
        }
        for (int off = 0, idx = 0; off < 8; off += esz_b, ++idx) {
          fill_controls_elem(&b_bytes[size_t(off)], form.format_b, state,
                             (w == 0   ? 1
                              : w == 1 ? 0
                                       : w),
                             lane, idx);
        }
        std::memcpy(&waves[size_t(w)].a[0][size_t(lane)], &a_bytes[0], 4);
        std::memcpy(&waves[size_t(w)].a[1][size_t(lane)], &a_bytes[4], 4);
        std::memcpy(&waves[size_t(w)].b[0][size_t(lane)], &b_bytes[0], 4);
        std::memcpy(&waves[size_t(w)].b[1][size_t(lane)], &b_bytes[4], 4);
        for (int off = 0; off < cd_bytes; off += 4) {
          float cv = 0.0f;
          if (w == 0 || w == 1) {
            cv = float(lane * 16 + off / 4 + 1) * 0.0625f;
          } else if (w == 4) {
            const int sel = (lane + off / 4) % 4;
            cv = sel == 0 ? 0.0f : sel == 1 ? 65500.0f : sel == 2 ? 3.0e38f : -3.0e38f;
          } else {
            fill_controls_elem(reinterpret_cast<uint8_t *>(&cv), 2, state, w, lane, off / 4);
          }
          std::memcpy(&waves[size_t(w)].c[size_t(off / 4)][size_t(lane)], &cv, 4);
        }
      }
    }
    return waves;
  };

  for (const auto &tc : kQualifiedCases) {
    const Form &form = find_form(tc.form_name);
    const int outputs = form.m * form.m * form.batches / 64;
    const uint64_t controls =
        uint64_t(tc.cbsz) | (uint64_t(tc.abid) << 3) | (uint64_t(tc.blgp) << 7);
    const auto waves = generate_waves(form);
    struct WaveDigest {
      int wave;
      uint64_t expected_fnv1a;
    };
    for (const WaveDigest wd : {WaveDigest{0, tc.wave0_fnv1a}, WaveDigest{2, tc.wave2_fnv1a},
                                WaveDigest{5, tc.wave5_fnv1a}}) {
      SCOPED_TRACE(::testing::Message() << tc.form_name << " cbsz=" << tc.cbsz << " abid="
                                        << tc.abid << " blgp=" << tc.blgp << " wave=" << wd.wave);
      const auto &wb = waves[size_t(wd.wave)];
      for (uint64_t cpu : supported_cpu_levels()) {
        for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                                   RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                   RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
          for (int alias_offset : {0, outputs, 2 * outputs, 3 * outputs}) {
            std::array<std::array<uint32_t, 64>, 128> storage{};
            for (int r = 0; r < outputs; ++r) {
              storage[size_t(r)] = wb.c[size_t(r)];
            }
            storage[size_t(outputs)] = wb.a[0];
            storage[size_t(outputs + 1)] = wb.a[1];
            storage[size_t(2 * outputs)] = wb.b[0];
            storage[size_t(2 * outputs + 1)] = wb.b[1];
            const uint32_t *a[] = {storage[size_t(outputs)].data(),
                                   storage[size_t(outputs + 1)].data()};
            const uint32_t *b[] = {storage[size_t(2 * outputs)].data(),
                                   storage[size_t(2 * outputs + 1)].data()};
            const uint32_t *c[32];
            uint32_t *d[32];
            for (int r = 0; r < outputs; ++r) {
              c[r] = storage[size_t(r)].data();
              d[r] = storage[size_t(alias_offset + r)].data();
            }
            ASSERT_EQ(form.entry(cpu | semantics, controls, d, a, b, c), RJ_ARITH_SUCCESS);
            std::array<std::array<uint32_t, 64>, 32> out{};
            for (int r = 0; r < outputs; ++r) {
              out[size_t(r)] = storage[size_t(alias_offset + r)];
            }
            EXPECT_EQ(hash_vgprs_fnv1a(out, outputs), wd.expected_fnv1a);
          }
        }
      }
    }
  }

  // Out-of-range ABID >= 2^CBSZ wraps modulo 2^CBSZ on MI300X hardware; strict modes reject these
  // uncalibrated encodings while Loose and ExactAllowLoose reproduce the MI300X wave-0 output.
  struct WrappedControlCase {
    const char *form_name;
    uint32_t cbsz, abid, blgp;
    uint64_t wave0_fnv1a;
  };
  constexpr WrappedControlCase kWrappedCases[] = {
      {"f16_16x16x4_4b", 1, 2, 0, 0x0f756ce635f1d575ULL},
      {"f16_16x16x4_4b", 1, 3, 0, 0xc0eefca48b3cfdb8ULL},
      {"f16_16x16x4_4b", 1, 2, 5, 0xfc086e3bf6a59510ULL},
      {"bf16_4x4x4_16b", 2, 6, 0, 0xa90acb4b09b912c4ULL},
      {"bf16_4x4x4_16b", 3, 13, 0, 0x6a1cf7de928c1c56ULL},
  };
  for (const auto &tc : kWrappedCases) {
    const Form &form = find_form(tc.form_name);
    const int outputs = form.m * form.m * form.batches / 64;
    const uint64_t controls =
        uint64_t(tc.cbsz) | (uint64_t(tc.abid) << 3) | (uint64_t(tc.blgp) << 7);
    const auto waves = generate_waves(form);
    const auto &wb = waves[0];
    const uint32_t *a[] = {wb.a[0].data(), wb.a[1].data()};
    const uint32_t *b[] = {wb.b[0].data(), wb.b[1].data()};
    const uint32_t *c[32];
    for (int r = 0; r < outputs; ++r) {
      c[r] = wb.c[size_t(r)].data();
    }
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE}) {
      std::array<std::array<uint32_t, 64>, 32> out{};
      uint32_t *d[32];
      for (int r = 0; r < outputs; ++r) {
        d[r] = out[size_t(r)].data();
      }
      ASSERT_EQ(form.entry(semantics, controls, d, a, b, c), RJ_ARITH_SUCCESS);
      EXPECT_EQ(hash_vgprs_fnv1a(out, outputs), tc.wave0_fnv1a);
    }
  }

  // Verify MODE.FP16_OVFL output clamping composes with nonzero batched source-lane controls:
  // on bf16_16x16x4_4b with cbsz=1, abid=1, blgp=1, batch 0 row 0 col 0 reads A from lane 16 and
  // B from lane 0, clamping 2^64 * 2^64 (+Inf) to +MAX_FLOAT (0x7f7fffff) under FP16_OVFL=1.
  {
    std::array<std::array<uint32_t, 64>, 2> a_buf{}, b_buf{};
    std::array<std::array<uint32_t, 64>, 16> c_buf{}, d_buf{};
    a_buf[0][16] = 0x00005f80U;
    b_buf[0][0] = 0x00005f80U;
    const uint32_t *a[] = {a_buf[0].data(), a_buf[1].data()};
    const uint32_t *b[] = {b_buf[0].data(), b_buf[1].data()};
    const uint32_t *c[16];
    uint32_t *d[16];
    for (int r = 0; r < 16; ++r) {
      c[r] = c_buf[size_t(r)].data();
      d[r] = d_buf[size_t(r)].data();
    }
    constexpr uint64_t kControls = uint64_t{1} | (uint64_t{1} << 3) | (uint64_t{1} << 7);
    for (uint64_t cpu : supported_cpu_levels()) {
      ASSERT_EQ(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64(
                    cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, kControls, d, a, b, c),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[0][0], 0x7f800000U);
      ASSERT_EQ(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64(
                    cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, kControls, d,
                    a, b, c),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[0][0], 0x7f7fffffU);
    }
  }
}

TEST(MfmaGfx942Exact, Fp16OvflSanitizesEachStepOutput) {
  // MI300X MODE.FP16_OVFL flushes NaN to +0.0 and clamps +-Inf to +-MAX_FLOAT after each K-step.
  for (const auto &form : kForms) {
    SCOPED_TRACE(form.name);
    const int per_word = form.format_a == 2 ? 1 : form.format_a < 2 ? 2 : 4;
    const int inputs = form.m * form.k * form.batches / (64 * per_word);
    const int outputs = form.m * form.m * form.batches / 64;
    std::array<std::array<uint32_t, 64>, 32> a_buf{}, b_buf{}, c_buf{}, d_buf{};
    const uint32_t *a[8], *b[8], *c[32];
    uint32_t *d[32];
    for (int reg = 0; reg < inputs; ++reg) {
      a[reg] = a_buf[reg].data();
      b[reg] = b_buf[reg].data();
    }
    for (int reg = 0; reg < outputs; ++reg) {
      c[reg] = c_buf[reg].data();
      d[reg] = d_buf[reg].data();
    }
    // Nonfinite C (+Inf, -Inf, NaN) with zero products verifies output clamping across all 20
    // forms.
    struct CCase {
      uint32_t c_bits, expected;
    };
    for (const CCase tc : {CCase{0x7f800000U, 0x7f7fffffU}, CCase{0xff800000U, 0xff7fffffU},
                           CCase{0x7fc00001U, 0x00000000U}}) {
      c_buf[0][0] = tc.c_bits;
      for (uint64_t cpu : supported_cpu_levels()) {
        SCOPED_TRACE(cpu);
        const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL;
        ASSERT_EQ(form.entry(flags, 0, d, a, b, c), RJ_ARITH_SUCCESS);
        EXPECT_EQ(d_buf[0][0], tc.expected);
      }
    }
  }

  // On two-step 16x16x16 BF16, clamping step 0's +Inf to 0x7f7fffff before step 1 adds -2^127
  // yields 0x7efffffe (whereas clamping only the final result would yield 0x7f7fffff), and
  // flushing step 0's NaN to +0.0 lets step 1's 1.0 * 3.0 survive as 3.0.
  {
    std::array<std::array<uint32_t, 64>, 4> a_buf{}, b_buf{}, c_buf{}, d_buf{};
    const uint32_t *a[] = {a_buf[0].data(), a_buf[1].data()};
    const uint32_t *b[] = {b_buf[0].data(), b_buf[1].data()};
    const uint32_t *c[] = {c_buf[0].data(), c_buf[1].data(), c_buf[2].data(), c_buf[3].data()};
    uint32_t *d[] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};
    // Step 0 (lane 0, reg 0 low): 2^64 * 2^64 -> +Inf -> 0x7f7fffff.
    // Step 1 (lane 32, reg 0 low, depth = 8): -2^64 * 2^63 = -2^127 -> 0x7efffffe.
    a_buf[0][0] = 0x00005f80U;
    b_buf[0][0] = 0x00005f80U;
    a_buf[0][32] = 0x0000df80U;
    b_buf[0][32] = 0x00005f00U;
    for (uint64_t cpu : supported_cpu_levels()) {
      SCOPED_TRACE(cpu);
      const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL;
      ASSERT_EQ(rj_arith_v_mfma_f32_16x16x16_bf16_w64(flags, 0, d, a, b, c), RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[0][0], 0x7efffffeU);
    }

    // Step 0: +Inf * 0.0 -> NaN -> +0.0; Step 1: 1.0 * 3.0 -> 3.0 (0x40400000).
    a_buf[0][0] = 0x00007f80U;
    b_buf[0][0] = 0U;
    a_buf[0][32] = 0x00003f80U;
    b_buf[0][32] = 0x00004040U;
    ASSERT_EQ(rj_arith_v_mfma_f32_16x16x16_bf16_w64(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, 0, d, a, b, c),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[0][0], 0x40400000U);

    // Nonfinite initial C = qNaN is NOT flushed before step 0: with step-0 1.0 * 2.0 and
    // zero step-1 products, step 0 produces qNaN -> +0.0 (0x00000000), whereas flushing C
    // on input would return +2.0 (0x40000000).
    a_buf = {};
    b_buf = {};
    c_buf = {};
    a_buf[0][0] = 0x00003f80U;
    b_buf[0][0] = 0x00004000U;
    c_buf[0][0] = 0x7fc00000U;
    ASSERT_EQ(rj_arith_v_mfma_f32_16x16x16_bf16_w64(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, 0, d, a, b, c),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[0][0], 0x00000000U);
  }

  // MI300X-captured rows (AMD Instinct MI300X VF, gfx942, ROCm 7.2.4 / HIP 7.2.53211,
  // mfma_fp16_ovfl.bin) with initial C = qNaN (0x7fc00000) and a single nonzero step-0 product:
  // hardware preserves C into step 0 so qNaN + product flushes to 0x00000000 under FP16_OVFL=1
  // (0xffc00000 under FP16_OVFL=0), rather than flushing C on input to yield the product.
  {
    // bf16_4x4x4_16b wave 11, batch 2, row 0, col 3 (a_lane=8, b_lane=11, out_reg=0, out_lane=11):
    // k=0 has 118.5 * -109.5 = -12975.75 (0xc64abf00) and k=1..3 are zero.
    std::array<std::array<uint32_t, 64>, 4> a_buf{}, b_buf{}, c_buf{}, d_buf{};
    const uint32_t *a[] = {a_buf[0].data(), a_buf[1].data()};
    const uint32_t *b[] = {b_buf[0].data(), b_buf[1].data()};
    const uint32_t *c[] = {c_buf[0].data(), c_buf[1].data(), c_buf[2].data(), c_buf[3].data()};
    uint32_t *d[] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};
    a_buf[0][8] = 0x800042edU;
    a_buf[1][8] = 0x42cd8000U;
    b_buf[0][11] = 0x8000c2dbU;
    b_buf[1][11] = 0x8000c2cbU;
    c_buf[0][11] = 0x7fc00000U;
    ASSERT_EQ(rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 0, d,
                                                     a, b, c),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[0][11], 0xffc00000U);
    ASSERT_EQ(rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, 0, d, a, b, c),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[0][11], 0x00000000U);
  }
  {
    // bf16_16x16x4_4b wave 23, batch 3, row 2, col 0 (a_lane=50, b_lane=48, out_reg=14,
    // out_lane=0): k=1 has -85.5 * -110.5 = +9447.75 (0x46139f00) and k=0,2,3 are zero.
    std::array<std::array<uint32_t, 64>, 16> a_buf{}, b_buf{}, c_buf{}, d_buf{};
    const uint32_t *a[] = {a_buf[0].data(), a_buf[1].data()};
    const uint32_t *b[] = {b_buf[0].data(), b_buf[1].data()};
    const uint32_t *c[16];
    uint32_t *d[16];
    for (int reg = 0; reg < 16; ++reg) {
      c[reg] = c_buf[reg].data();
      d[reg] = d_buf[reg].data();
    }
    a_buf[0][50] = 0xc2ab42dcU;
    a_buf[1][50] = 0x00000001U;
    b_buf[0][48] = 0xc2dd0000U;
    b_buf[1][48] = 0xc2b48000U;
    c_buf[14][0] = 0x7fc00000U;
    ASSERT_EQ(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 0, d,
                                                      a, b, c),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[14][0], 0xffc00000U);
    ASSERT_EQ(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, 0, d, a, b, c),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[14][0], 0x00000000U);
  }

  // Verify the same two-step behavior on sparse 16x16x32 BF16.
  {
    std::array<std::array<uint32_t, 64>, 2> a_buf{};
    std::array<std::array<uint32_t, 64>, 4> b_buf{}, d_buf{};
    std::array<uint32_t, 64> idx_buf{};
    idx_buf.fill(0x44444444U);
    const uint32_t *a[] = {a_buf[0].data(), a_buf[1].data()};
    const uint32_t *b[] = {b_buf[0].data(), b_buf[1].data(), b_buf[2].data(), b_buf[3].data()};
    const uint32_t *index[] = {idx_buf.data()};
    uint32_t *d[] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};
    // t=0 (q=0, s=0 -> A reg 0 low lane 0, B depth 0 -> reg 0 low lane 0): 2^64 * 2^64.
    // t=8 (q=4, s=0 -> A reg 0 low lane 32, B depth 16 -> reg 0 low lane 32): -2^64 * 2^63.
    a_buf[0][0] = 0x00005f80U;
    b_buf[0][0] = 0x00005f80U;
    a_buf[0][32] = 0x0000df80U;
    b_buf[0][32] = 0x00005f00U;
    for (uint64_t cpu : supported_cpu_levels()) {
      SCOPED_TRACE(cpu);
      const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL;
      ASSERT_EQ(rj_arith_v_smfmac_f32_16x16x32_bf16_w64(flags, 0, d, a, b, index),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[0][0], 0x7efffffeU);
    }
  }
}

using SparseEntry = decltype(&rj_arith_v_smfmac_f32_16x16x32_f16_w64);
struct SparseForm {
  const char *name;
  int m, k, format_a, format_b;
  SparseEntry entry, gfx950;
  std::span<const uint32_t> captured;
};

const SparseForm kSparseForms[] = {
    {"smfmac_f16_16x16x32", 16, 32, 0, 0, rj_arith_v_smfmac_f32_16x16x32_f16_w64,
     rj_arith_v_smfmac_f32_16x16x32_f16_w64_gfx950, ksmfmac_f16_16x16x32},
    {"smfmac_bf16_16x16x32", 16, 32, 1, 1, rj_arith_v_smfmac_f32_16x16x32_bf16_w64,
     rj_arith_v_smfmac_f32_16x16x32_bf16_w64_gfx950, ksmfmac_bf16_16x16x32},
    {"smfmac_f16_32x32x16", 32, 16, 0, 0, rj_arith_v_smfmac_f32_32x32x16_f16_w64,
     rj_arith_v_smfmac_f32_32x32x16_f16_w64_gfx950, ksmfmac_f16_32x32x16},
    {"smfmac_bf16_32x32x16", 32, 16, 1, 1, rj_arith_v_smfmac_f32_32x32x16_bf16_w64,
     rj_arith_v_smfmac_f32_32x32x16_bf16_w64_gfx950, ksmfmac_bf16_32x32x16},
    {"smfmac_fp8_fp8_16x16x64", 16, 64, 3, 3, rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64,
     rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64_gfx950, ksmfmac_fp8_fp8_16x16x64},
    {"smfmac_fp8_bf8_16x16x64", 16, 64, 3, 4, rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64,
     rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64_gfx950, ksmfmac_fp8_bf8_16x16x64},
    {"smfmac_bf8_fp8_16x16x64", 16, 64, 4, 3, rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64,
     rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64_gfx950, ksmfmac_bf8_fp8_16x16x64},
    {"smfmac_bf8_bf8_16x16x64", 16, 64, 4, 4, rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64,
     rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64_gfx950, ksmfmac_bf8_bf8_16x16x64},
    {"smfmac_fp8_fp8_32x32x32", 32, 32, 3, 3, rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64,
     rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64_gfx950, ksmfmac_fp8_fp8_32x32x32},
    {"smfmac_fp8_bf8_32x32x32", 32, 32, 3, 4, rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64,
     rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64_gfx950, ksmfmac_fp8_bf8_32x32x32},
    {"smfmac_bf8_fp8_32x32x32", 32, 32, 4, 3, rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64,
     rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64_gfx950, ksmfmac_bf8_fp8_32x32x32},
    {"smfmac_bf8_bf8_32x32x32", 32, 32, 4, 4, rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64,
     rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64_gfx950, ksmfmac_bf8_bf8_32x32x32},
};

void captured_sparse_wave(const SparseForm &form, int row, int col) {
  SCOPED_TRACE(::testing::Message() << form.name << " row " << row << " column " << col);
  const bool half = form.format_a < 2;
  const int per_word = half ? 2 : 4;
  const int elem_bits = 32 / per_word;
  const int outputs = form.m * form.m / 64;
  const int set_bits = form.m * form.k / 64;
  const uint32_t set_mask = (1U << set_bits) - 1U;
  const uint64_t meta = uint64_t(form.captured[0]) | (uint64_t(form.captured[1]) << 32);

  for (uint64_t controls : {0ULL, 8ULL, 16ULL, 24ULL, 9ULL}) {
    SCOPED_TRACE(controls);
    const int shift =
        (controls & 3) == 0 ? set_bits * int((controls >> 3) & (32 / set_bits - 1)) : 0;
    std::array<std::array<uint32_t, 64>, 23> input{};
    // Fill unselected metadata sets with 0xf distractors so wrong CBSZ/ABID shifts fail.
    input[6].fill(~(set_mask << shift));
    for (int q = 0; q < form.k / 4; ++q) {
      const int half_groups = form.k / 8;
      const int lane = half ? (q / 2) * form.m + row : ((q % half_groups) / 2) * form.m + row;
      const int nibble = half ? q % 2 : 2 * (q / half_groups) + (q % 2);
      input[6][lane] |= uint32_t((meta >> (4 * q)) & 0xfU) << (shift + 4 * nibble);
      for (int s = 0; s < 2; ++s) {
        const int elem = half ? (2 * q + s) % 4 : 4 * (q / half_groups) + (2 * q + s) % 4;
        input[elem / per_word][lane] |= encode(form.captured[2 + 2 * q + s], form.format_a)
                                        << (elem_bits * (elem % per_word));
      }
    }
    for (int depth = 0; depth < form.k; ++depth) {
      int lane = 0, elem = 0;
      if (half) {
        lane = form.m == 16 ? (depth / 8) * 16 + col
                            : 16 * ((col / 16) + 2 * (depth / 8)) + (col % 16);
        elem = depth % 8;
      } else {
        lane = form.m == 16 ? 16 * ((depth % 32) / 8) + col : 32 * ((depth % 16) / 8) + col;
        elem = form.m == 16 ? 8 * (depth / 32) + (depth % 8) : 8 * (depth / 16) + (depth % 8);
      }
      input[2 + elem / per_word][lane] |=
          encode(form.captured[2 + form.k / 2 + depth], form.format_b)
          << (elem_bits * (elem % per_word));
    }
    const int output_reg = row / (256 / form.m) * 4 + row % 4;
    const int output_lane = row / 4 % (64 / form.m) * form.m + col;
    input[7 + output_reg][output_lane] = form.captured[2 + 3 * form.k / 2];
    const uint32_t expected_d = form.captured[2 + 3 * form.k / 2 + 1];

    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
        auto storage = input;
        const uint32_t *a[] = {storage[0].data(), storage[1].data()};
        const uint32_t *b[] = {storage[2].data(), storage[3].data(), storage[4].data(),
                               storage[5].data()};
        const uint32_t *index[] = {storage[6].data()};
        uint32_t *d[16];
        for (int reg = 0; reg < outputs; ++reg) {
          d[reg] = storage[7 + reg].data();
        }
        ASSERT_EQ(form.entry(cpu | semantics, controls, d, a, b, index), RJ_ARITH_SUCCESS);
        for (int reg = 0; reg < 7 + outputs; ++reg) {
          for (int lane = 0; lane < 64; ++lane) {
            const uint32_t expected = reg < 7 ? input[reg][lane]
                                      : reg == 7 + output_reg && lane == output_lane ? expected_d
                                                                                     : 0U;
            ASSERT_EQ(storage[reg][lane], expected) << "register " << reg << " lane " << lane;
          }
        }
      }
    }
  }
}

TEST(MfmaGfx942Exact, SparseCapturedOutputsControlsAndAliasing) {
  for (const auto &form : kSparseForms) {
    const size_t row_words = size_t(3 * form.k / 2 + 4);
    ASSERT_EQ(form.captured.size() % row_words, 0U);
    for (size_t offset = 0; offset < form.captured.size(); offset += row_words) {
      SCOPED_TRACE(offset / row_words);
      auto test = form;
      test.captured = form.captured.subspan(offset, row_words);
      captured_sparse_wave(test, 0, 0);
      captured_sparse_wave(test, form.m / 2, form.m / 2 + 1);
      captured_sparse_wave(test, form.m - 1, form.m - 1);
    }
  }
}

TEST(MfmaGfx942Exact, SparseQualificationIsLocalToTarget) {
  for (const auto &form : kSparseForms) {
    SCOPED_TRACE(form.name);
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      EXPECT_EQ(form.gfx950(semantics, 0, nullptr, nullptr, nullptr, nullptr),
                RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(form.gfx950(semantics | RJ_ARITH_FP16_OVFL, 0, nullptr, nullptr, nullptr, nullptr),
                RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
    }
  }
}
} // namespace
