// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for signed and unsigned normalized 16-bit conversions.

#include "common/floating_point.h"
#include "conversion/helpers.h"
#include "conversion/normalized_reference.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <bit>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

enum class Normalized {
  SignedPackedFloat,
  UnsignedPackedFloat,
  SignedPackedHalf,
  UnsignedPackedHalf,
  SignedHalf,
  UnsignedHalf
};

// Check normalized conversion results and destination preservation for the supplied entry point.
// Exact is false for entry points that stay loose-only until their captures are reviewed.
template <int Lanes, int Operands, bool Cdna = false, bool Exact = true, typename Call>
void check_normalized(Call call, Normalized operation) {
  int op = int(operation);
  auto input = conversion_inputs<Lanes>(op < 2 ? 32 : 16);
  // Retain edge classes, but spend the other lanes inside the nonsaturated
  // interval. Fractional products exercise nearest-even rounding.
  std::minstd_rand random;
  for (int reg : {0, 2}) {
    for (int lane = 16; lane < Lanes; ++lane) {
      if (op < 2) {
        input.v[reg][lane] =
            std::bit_cast<uint32_t>(float(int(random() >> 7) - 0x800000) * 0x1p-23f);
      } else {
        input.v[reg][lane] = random_f16_pair(random);
      }
    }
  }

  if (op == 2 || op == 3) {
    // Largest FP16 subnormals produce nonzero normalized integers unless
    // input flushing is enabled.
    input.v[0][16] = 0x83ff03ffU;
    input.v[2][17] = 0x03ff83ffU;
  }

  uint64_t modifiers =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP |
      (op >= 2 ? RJ_ARITH_ALU_HIGH_A : 0) |
      (op >= 4 ? RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_OMOD_HALF
               : RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B | (op >= 2 ? RJ_ARITH_ALU_HIGH_B : 0));
  check_conversion<Lanes, Operands, 1, 1, true>(
      call, input,
      [=](uint64_t a, uint32_t b, uint32_t, uint32_t d, uint64_t flags, uint32_t mode) {
        if (op >= 2 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
          auto flush = [](uint32_t word) {
            for (int shift : {0, 16}) {
              if (((word >> shift) & 0x7fff) < 0x400) {
                word &= ~(0x7fffU << shift);
              }
            }
            return word;
          };
          a = flush(uint32_t(a));
          b = flush(b);
        }
        uint32_t result = normalized_reference(op, uint32_t(a), b, d, mode);
        if constexpr (Cdna) {
          if (!(mode & RJ_ARITH_ALU_HIGH_D)) {
            result &= 0xffffU;
          }
        }
        return result;
      },
      conversion_bits, modifiers, Exact,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
      /*separately_tested_modifiers=*/0, /*status_free=*/false);
}

} // namespace

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F32W32, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_i16_f32_w32, Normalized::SignedPackedFloat);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F32W32, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_u16_f32_w32, Normalized::UnsignedPackedFloat);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F16W32Gfx11, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_i16_f16_w32, Normalized::SignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F16W32Gfx12, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_i16_f16_w32_gfx12, Normalized::SignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F16W32Gfx11, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_u16_f16_w32, Normalized::UnsignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F16W32Gfx12, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_u16_f16_w32_gfx12, Normalized::UnsignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtNormI16F16W32, Arithmetic) {
  check_normalized<32, 1>(rj_arith_v_cvt_norm_i16_f16_w32, Normalized::SignedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtNormU16F16W32, Arithmetic) {
  check_normalized<32, 1>(rj_arith_v_cvt_norm_u16_f16_w32, Normalized::UnsignedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F32W64, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_i16_f32_w64, Normalized::SignedPackedFloat);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F32W64, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_u16_f32_w64, Normalized::UnsignedPackedFloat);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F16W64Gfx11, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_i16_f16_w64, Normalized::SignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F16W64Gfx12, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_i16_f16_w64_gfx12, Normalized::SignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F16W64Gfx11, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_u16_f16_w64, Normalized::UnsignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F16W64Gfx12, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_u16_f16_w64_gfx12, Normalized::UnsignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtNormI16F16W64, Arithmetic) {
  check_normalized<64, 1>(rj_arith_v_cvt_norm_i16_f16_w64, Normalized::SignedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtNormU16F16W64, Arithmetic) {
  check_normalized<64, 1>(rj_arith_v_cvt_norm_u16_f16_w64, Normalized::UnsignedHalf);
}

TEST(ConversionCvtNormI16F16W32Gfx942, Arithmetic) {
  check_normalized<32, 1, true, true>(rj_arith_v_cvt_norm_i16_f16_w32_gfx942,
                                      Normalized::SignedHalf);
}

TEST(ConversionCvtNormI16F16W64Gfx942, Arithmetic) {
  check_normalized<64, 1, true, true>(rj_arith_v_cvt_norm_i16_f16_w64_gfx942,
                                      Normalized::SignedHalf);
}

TEST(ConversionCvtNormU16F16W32Gfx942, Arithmetic) {
  check_normalized<32, 1, true, true>(rj_arith_v_cvt_norm_u16_f16_w32_gfx942,
                                      Normalized::UnsignedHalf);
}

TEST(ConversionCvtNormU16F16W64Gfx942, Arithmetic) {
  check_normalized<64, 1, true, true>(rj_arith_v_cvt_norm_u16_f16_w64_gfx942,
                                      Normalized::UnsignedHalf);
}

// gfx950 shares the gfx942 unary model and matches MI350X captures.
TEST(ConversionCvtNormI16F16W32Gfx950, Arithmetic) {
  check_normalized<32, 1, true, true>(rj_arith_v_cvt_norm_i16_f16_w32_gfx950,
                                      Normalized::SignedHalf);
}

TEST(ConversionCvtNormI16F16W64Gfx950, Arithmetic) {
  check_normalized<64, 1, true, true>(rj_arith_v_cvt_norm_i16_f16_w64_gfx950,
                                      Normalized::SignedHalf);
}

TEST(ConversionCvtNormU16F16W32Gfx950, Arithmetic) {
  check_normalized<32, 1, true, true>(rj_arith_v_cvt_norm_u16_f16_w32_gfx950,
                                      Normalized::UnsignedHalf);
}

TEST(ConversionCvtNormU16F16W64Gfx950, Arithmetic) {
  check_normalized<64, 1, true, true>(rj_arith_v_cvt_norm_u16_f16_w64_gfx950,
                                      Normalized::UnsignedHalf);
}

// gfx1201 captures: normalized results round once, saturate, and map NaNs to zero; the unary
// form keeps the high D half.
TEST(ConversionNormalizedGfx12, ExactMatchesHardware) {
  constexpr uint64_t d = 0xa5a5a5a5U;
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_norm_i16_f16_w32, 0, 0xa5fe3c00, 0, 0, d,
                                   0xa5a57fffU);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_norm_i16_f16_w32, 0, 0xc5e03800, 0, 0, d,
                                   0xa5a54000U);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_norm_i16_f16_w32, 0, 0xc7adb800, 0, 0, d,
                                   0xa5a5c000U);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_norm_i16_f16_w32, 0, 0xfdfe7e00, 0, 0, d,
                                   0xa5a50000U);
  expect_captured_conversion<2, 1>(rj_arith_v_cvt_pk_norm_u16_f32_w32, 0, 0x3f000000, 0, 0, 0,
                                   0x8000);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkNormI16F16W32Gfx1250, Arithmetic) {
  check_normalized<32, 2, false, false>(rj_arith_v_cvt_pk_norm_i16_f16_w32_gfx1250,
                                        Normalized::SignedPackedHalf);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkNormI16F16W64Gfx1250, Arithmetic) {
  check_normalized<64, 2, false, false>(rj_arith_v_cvt_pk_norm_i16_f16_w64_gfx1250,
                                        Normalized::SignedPackedHalf);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkNormU16F16W32Gfx1250, Arithmetic) {
  check_normalized<32, 2, false, false>(rj_arith_v_cvt_pk_norm_u16_f16_w32_gfx1250,
                                        Normalized::UnsignedPackedHalf);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkNormU16F16W64Gfx1250, Arithmetic) {
  check_normalized<64, 2, false, false>(rj_arith_v_cvt_pk_norm_u16_f16_w64_gfx1250,
                                        Normalized::UnsignedPackedHalf);
}

// gfx1250 inherits the shared RDNA model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtNormI16F16W32Gfx1250, Arithmetic) {
  check_normalized<32, 1, false, false>(rj_arith_v_cvt_norm_i16_f16_w32_gfx1250,
                                        Normalized::SignedHalf);
}

// gfx1250 inherits the shared RDNA model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtNormI16F16W64Gfx1250, Arithmetic) {
  check_normalized<64, 1, false, false>(rj_arith_v_cvt_norm_i16_f16_w64_gfx1250,
                                        Normalized::SignedHalf);
}

// gfx1250 inherits the shared RDNA model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtNormU16F16W32Gfx1250, Arithmetic) {
  check_normalized<32, 1, false, false>(rj_arith_v_cvt_norm_u16_f16_w32_gfx1250,
                                        Normalized::UnsignedHalf);
}

// gfx1250 inherits the shared RDNA model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtNormU16F16W64Gfx1250, Arithmetic) {
  check_normalized<64, 1, false, false>(rj_arith_v_cvt_norm_u16_f16_w64_gfx1250,
                                        Normalized::UnsignedHalf);
}

// gfx1250 inherits the shared RDNA model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkNormI16F32W32Gfx1250, Arithmetic) {
  check_normalized<32, 2, false, false>(rj_arith_v_cvt_pk_norm_i16_f32_w32_gfx1250,
                                        Normalized::SignedPackedFloat);
}

// gfx1250 inherits the shared RDNA model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkNormI16F32W64Gfx1250, Arithmetic) {
  check_normalized<64, 2, false, false>(rj_arith_v_cvt_pk_norm_i16_f32_w64_gfx1250,
                                        Normalized::SignedPackedFloat);
}

// gfx1250 inherits the shared RDNA model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkNormU16F32W32Gfx1250, Arithmetic) {
  check_normalized<32, 2, false, false>(rj_arith_v_cvt_pk_norm_u16_f32_w32_gfx1250,
                                        Normalized::UnsignedPackedFloat);
}

// gfx1250 inherits the shared RDNA model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkNormU16F32W64Gfx1250, Arithmetic) {
  check_normalized<64, 2, false, false>(rj_arith_v_cvt_pk_norm_u16_f32_w64_gfx1250,
                                        Normalized::UnsignedPackedFloat);
}

// gfx942 shares the RDNA3 packed normalized model and matches MI300X captures.
TEST(ConversionCvtPkNormI16F16W32Gfx942, Arithmetic) {
  check_normalized<32, 2, false, true>(rj_arith_v_cvt_pk_norm_i16_f16_w32_gfx942,
                                       Normalized::SignedPackedHalf);
}

TEST(ConversionCvtPkNormI16F16W64Gfx942, Arithmetic) {
  check_normalized<64, 2, false, true>(rj_arith_v_cvt_pk_norm_i16_f16_w64_gfx942,
                                       Normalized::SignedPackedHalf);
}

TEST(ConversionCvtPkNormU16F16W32Gfx942, Arithmetic) {
  check_normalized<32, 2, false, true>(rj_arith_v_cvt_pk_norm_u16_f16_w32_gfx942,
                                       Normalized::UnsignedPackedHalf);
}

TEST(ConversionCvtPkNormU16F16W64Gfx942, Arithmetic) {
  check_normalized<64, 2, false, true>(rj_arith_v_cvt_pk_norm_u16_f16_w64_gfx942,
                                       Normalized::UnsignedPackedHalf);
}

TEST(ConversionCvtPkNormI16F32W32Gfx942, Arithmetic) {
  check_normalized<32, 2, false, true>(rj_arith_v_cvt_pk_norm_i16_f32_w32_gfx942,
                                       Normalized::SignedPackedFloat);
}

TEST(ConversionCvtPkNormI16F32W64Gfx942, Arithmetic) {
  check_normalized<64, 2, false, true>(rj_arith_v_cvt_pk_norm_i16_f32_w64_gfx942,
                                       Normalized::SignedPackedFloat);
}

TEST(ConversionCvtPkNormU16F32W32Gfx942, Arithmetic) {
  check_normalized<32, 2, false, true>(rj_arith_v_cvt_pk_norm_u16_f32_w32_gfx942,
                                       Normalized::UnsignedPackedFloat);
}

TEST(ConversionCvtPkNormU16F32W64Gfx942, Arithmetic) {
  check_normalized<64, 2, false, true>(rj_arith_v_cvt_pk_norm_u16_f32_w64_gfx942,
                                       Normalized::UnsignedPackedFloat);
}

// gfx950 shares the RDNA3/gfx942 model and matches MI350X captures.
TEST(ConversionCvtPkNormI16F16W32Gfx950, Arithmetic) {
  check_normalized<32, 2, false, true>(rj_arith_v_cvt_pk_norm_i16_f16_w32_gfx950,
                                       Normalized::SignedPackedHalf);
}

TEST(ConversionCvtPkNormI16F16W64Gfx950, Arithmetic) {
  check_normalized<64, 2, false, true>(rj_arith_v_cvt_pk_norm_i16_f16_w64_gfx950,
                                       Normalized::SignedPackedHalf);
}

TEST(ConversionCvtPkNormU16F16W32Gfx950, Arithmetic) {
  check_normalized<32, 2, false, true>(rj_arith_v_cvt_pk_norm_u16_f16_w32_gfx950,
                                       Normalized::UnsignedPackedHalf);
}

TEST(ConversionCvtPkNormU16F16W64Gfx950, Arithmetic) {
  check_normalized<64, 2, false, true>(rj_arith_v_cvt_pk_norm_u16_f16_w64_gfx950,
                                       Normalized::UnsignedPackedHalf);
}

TEST(ConversionCvtPkNormI16F32W32Gfx950, Arithmetic) {
  check_normalized<32, 2, false, true>(rj_arith_v_cvt_pk_norm_i16_f32_w32_gfx950,
                                       Normalized::SignedPackedFloat);
}

TEST(ConversionCvtPkNormI16F32W64Gfx950, Arithmetic) {
  check_normalized<64, 2, false, true>(rj_arith_v_cvt_pk_norm_i16_f32_w64_gfx950,
                                       Normalized::SignedPackedFloat);
}

TEST(ConversionCvtPkNormU16F32W32Gfx950, Arithmetic) {
  check_normalized<32, 2, false, true>(rj_arith_v_cvt_pk_norm_u16_f32_w32_gfx950,
                                       Normalized::UnsignedPackedFloat);
}

TEST(ConversionCvtPkNormU16F32W64Gfx950, Arithmetic) {
  check_normalized<64, 2, false, true>(rj_arith_v_cvt_pk_norm_u16_f32_w64_gfx950,
                                       Normalized::UnsignedPackedFloat);
}

// gfx10/gfx115x/gfx908/gfx90a share the RDNA3/gfx942 models via loose-only _gfx10 wrappers.
TEST(ConversionCvtNormI16F16W32Gfx10, Arithmetic) {
  check_normalized<32, 1, true, false>(rj_arith_v_cvt_norm_i16_f16_w32_gfx10,
                                       Normalized::SignedHalf);
}

TEST(ConversionCvtNormI16F16W64Gfx10, Arithmetic) {
  check_normalized<64, 1, true, false>(rj_arith_v_cvt_norm_i16_f16_w64_gfx10,
                                       Normalized::SignedHalf);
}

TEST(ConversionCvtNormU16F16W32Gfx10, Arithmetic) {
  check_normalized<32, 1, true, false>(rj_arith_v_cvt_norm_u16_f16_w32_gfx10,
                                       Normalized::UnsignedHalf);
}

TEST(ConversionCvtNormU16F16W64Gfx10, Arithmetic) {
  check_normalized<64, 1, true, false>(rj_arith_v_cvt_norm_u16_f16_w64_gfx10,
                                       Normalized::UnsignedHalf);
}

TEST(ConversionCvtPkNormI16F16W32Gfx10, Arithmetic) {
  check_normalized<32, 2, false, false>(rj_arith_v_cvt_pk_norm_i16_f16_w32_gfx10,
                                        Normalized::SignedPackedHalf);
}

TEST(ConversionCvtPkNormI16F16W64Gfx10, Arithmetic) {
  check_normalized<64, 2, false, false>(rj_arith_v_cvt_pk_norm_i16_f16_w64_gfx10,
                                        Normalized::SignedPackedHalf);
}

TEST(ConversionCvtPkNormU16F16W32Gfx10, Arithmetic) {
  check_normalized<32, 2, false, false>(rj_arith_v_cvt_pk_norm_u16_f16_w32_gfx10,
                                        Normalized::UnsignedPackedHalf);
}

TEST(ConversionCvtPkNormU16F16W64Gfx10, Arithmetic) {
  check_normalized<64, 2, false, false>(rj_arith_v_cvt_pk_norm_u16_f16_w64_gfx10,
                                        Normalized::UnsignedPackedHalf);
}

TEST(ConversionCvtPkNormI16F32W32Gfx10, Arithmetic) {
  check_normalized<32, 2, false, false>(rj_arith_v_cvt_pk_norm_i16_f32_w32_gfx10,
                                        Normalized::SignedPackedFloat);
}

TEST(ConversionCvtPkNormI16F32W64Gfx10, Arithmetic) {
  check_normalized<64, 2, false, false>(rj_arith_v_cvt_pk_norm_i16_f32_w64_gfx10,
                                        Normalized::SignedPackedFloat);
}

TEST(ConversionCvtPkNormU16F32W32Gfx10, Arithmetic) {
  check_normalized<32, 2, false, false>(rj_arith_v_cvt_pk_norm_u16_f32_w32_gfx10,
                                        Normalized::UnsignedPackedFloat);
}

TEST(ConversionCvtPkNormU16F32W64Gfx10, Arithmetic) {
  check_normalized<64, 2, false, false>(rj_arith_v_cvt_pk_norm_u16_f32_w64_gfx10,
                                        Normalized::UnsignedPackedFloat);
}

// MI300X (2026-11-24) captures: unary normalized forms zero-extend low-half results and preserve
// bit[15:0] when HIGH_D is selected.
TEST(ConversionNormalizedGfx942, ExactMatchesHardware) {
  constexpr uint64_t d = 0xa5a5a5a5U;
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_norm_i16_f16_w32_gfx942, 0, 0xa5fe3c00U, 0, 0, d,
                                   0x00007fffU);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_norm_u16_f16_w32_gfx942, 0, 0xa6af7bffU, 0, 0, d,
                                   0x0000ffffU);
  expect_captured_conversion_lane<1, 1>(rj_arith_v_cvt_norm_i16_f16_w32_gfx942,
                                        RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_ALU_HIGH_D, 0xa5fe3c00U, 0,
                                        0, d, 0x7fffa5a5U);
}

// gfx1100 (W7900, mkm2) hw1100-noieee lanes (2026-09-25, MODE.IEEE clear): NaN sources
// normalize to zero, saturation clamps at the limit, min normal scales below one half, and the
// FP16 min normal flushes to zero only with input denormals flushed (0x100). Unary forms keep
// the captured 0xa5a5 high half.
TEST(ConversionNormalizedGfx11, ExactMatchesHardware) {
  constexpr uint64_t d = 0xa5a5a5a5U;
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_norm_u16_f16_w32, 0x1f0, 0, 0xa6af7bff, 0, 0, d, 0xa5a5ffffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_norm_u16_f16_w32, 0x1f0, 0, 0x91a07c00, 0, 0, d, 0xa5a5ffffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_norm_i16_f16_w32, 0x100, 0, 0xdfdd0001, 0, 0, d, 0xa5a50000U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 32>(
      rj_arith_v_cvt_pk_norm_i16_f32_w32, 0x1f0, 0, 0x00800000, 0x7fc00000, 0, 0, 0);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 32>(
      rj_arith_v_cvt_pk_norm_u16_f32_w32, 0x1f0, 0, 0x00800000, 0x00800000, 0, 0, 0);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 16>(
      rj_arith_v_cvt_pk_norm_i16_f16_w32, 0x100, 0, 0xfcca0001, 0x59ad0400, 0, 0, 0x20000U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 16>(
      rj_arith_v_cvt_pk_norm_u16_f16_w32, 0x1f0, 0, 0xb37c0400, 0xa4bb7e00, 0, 0, 4);
}
