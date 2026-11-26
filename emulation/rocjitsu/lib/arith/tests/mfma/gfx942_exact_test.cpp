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
#include <gtest/gtest.h>
#include <span>

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
    // Dense source-lane controls (CBSZ, ABID, BLGP, including wrapping ABID >= 2^CBSZ)
    // remain gated in strict modes pending simulator A-lane read-mask alignment, leaving D
    // untouched while Loose and ExactAllowLoose execute the approximate fallback.
    for (uint64_t unqualified_control : {1ULL, 8ULL, 1ULL | (5ULL << 3), 4ULL, 1ULL << 7}) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
        for (uint64_t ovfl : {uint64_t{0}, RJ_ARITH_FP16_OVFL}) {
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
      EXPECT_EQ(form.gfx950(semantics, 0, nullptr, nullptr, nullptr, nullptr),
                RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(form.gfx950(semantics | RJ_ARITH_FP16_OVFL, 0, nullptr, nullptr, nullptr, nullptr),
                RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
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
      ASSERT_EQ(
          form.entry(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, 0, d, a, b, c),
          RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[0][0], tc.expected);
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
    ASSERT_EQ(rj_arith_v_mfma_f32_16x16x16_bf16_w64(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, 0, d, a, b, c),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[0][0], 0x7efffffeU);

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
    ASSERT_EQ(rj_arith_v_smfmac_f32_16x16x32_bf16_w64(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, 0, d, a, b, index),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(d_buf[0][0], 0x7efffffeU);
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
