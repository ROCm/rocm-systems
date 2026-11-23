// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for conversions between FP16, FP32, and 16-bit integers.

#include "common/cpu.h"
#include "common/floating_point.h"
#include "conversion/conversion16_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <bit>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

enum class Conversion16 {
  SignedToHalf,
  UnsignedToHalf,
  HalfToSigned,
  HalfToUnsigned,
  FloatToHalf,
  HalfToFloat
};

// Check FP16 conversion results, modifiers, and aliasing for the supplied entry point.
// Exact is false for entry points that stay loose-only until their captures are reviewed.
template <int Lanes, bool Gfx12, bool Cdna = false, bool Exact = true, typename Call>
void check_float16(Call call, Conversion16 operation) {
  int op = int(operation);
  auto input = conversion_inputs<Lanes>(op == 4 ? 32 : 16);
  if (op == 4) {
    // FP32 values at and beside half rounding ties, tininess and overflow.
    const uint32_t edges[] = {0x3f801000U, 0x3f800fffU, 0x3f801001U, 0x3f803000U,
                              0x387fefffU, 0x387ff000U, 0x38800000U, 0x477ff000U};
    for (int lane = 0; lane < 8; ++lane) {
      input.v[0][16 + lane] = edges[lane];
    }
  }
  uint64_t modifiers = RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP |
                       (op < 2 ? 0 : RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A) |
                       (op == 4 ? 0 : RJ_ARITH_ALU_HIGH_A) | (op == 5 ? 0 : RJ_ARITH_ALU_HIGH_D);
  check_conversion<Lanes, 1, 1, 1, true>(
      call, input,
      [=](uint64_t a, uint32_t, uint32_t, uint32_t old, uint64_t flags, uint32_t mode) {
        uint32_t raw = uint32_t(a);
        if (op != 2 && op != 3) {
          mode = conversion_effective_mode(flags, mode, Gfx12);
        }
        int shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
        uint16_t half = uint16_t(raw >> shift);
        if (op == 5 && (half & 0x7fff) > 0x7c00 && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
          mode &= ~RJ_ARITH_ALU_CLAMP;
        }
        if (op == 5 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (half & 0x7fff) < 0x400) {
          raw = uint32_t(half & 0x8000) << shift;
        }
        // The inherited conversion model flushes a tiny destination before
        // subnormal rounding; the midpoint here is at full FP16 precision.
        if (op == 4 && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
            (raw & 0x7fffffffU) < 0x387ff000U) {
          raw &= 0x80000000U;
        }
        uint32_t result = conversion16_reference(op, raw, old, mode, flags & RJ_ARITH_FP16_OVFL);
        // Inherited gfx1100 widening behavior: IEEE=0 preserves sNaN payload bits
        // through OMOD. CLAMP replaces NaN with zero unless DX10_CLAMP_OFF is set.
        if (op == 5 && !Gfx12 && !(flags & RJ_ARITH_FP_IEEE_MODE) && !(mode & RJ_ARITH_ALU_CLAMP) &&
            (half & 0x7fff) > 0x7c00) {
          result = (uint32_t(half & 0x8000) << 16) | 0x7f800000U | (uint32_t(half & 0x3ff) << 13);
          if (mode & RJ_ARITH_ALU_ABS_A) {
            result &= 0x7fffffffU;
          }
          if (mode & RJ_ARITH_ALU_NEG_A) {
            result ^= 0x80000000U;
          }
        }
        if constexpr (Cdna) {
          if (op != 5 && !(mode & RJ_ARITH_ALU_HIGH_D)) {
            result &= 0xffffU;
          }
        }
        return result;
      },
      [=](uint64_t actual, uint64_t expected, uint64_t flags, uint32_t mode) {
        if (op == 2 || op == 3 || (op == 5 && !Gfx12 && !(flags & RJ_ARITH_FP_IEEE_MODE))) {
          EXPECT_EQ(actual, expected);
        } else if (op == 5) {
          conversion_float(actual, expected, 32);
        } else {
          int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
          uint32_t keep = ~(0xffffU << shift);
          EXPECT_EQ(uint32_t(actual) & keep, uint32_t(expected) & keep);
          float a = f16_value(uint16_t(actual >> shift)),
                e = f16_value(uint16_t(expected >> shift));
          conversion_float(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(e), 32);
        }
      },
      modifiers, Exact,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | (op == 5 && !Gfx12 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0),
      Cdna && (op < 2 || op == 4) ? RJ_ARITH_CVT_SDWA : 0);
}

} // namespace

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16I16W32Gfx11, Arithmetic) {
  check_float16<32, false>(rj_arith_v_cvt_f16_i16_w32, Conversion16::SignedToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16I16W32Gfx12, Arithmetic) {
  check_float16<32, true>(rj_arith_v_cvt_f16_i16_w32_gfx12, Conversion16::SignedToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16U16W32Gfx11, Arithmetic) {
  check_float16<32, false>(rj_arith_v_cvt_f16_u16_w32, Conversion16::UnsignedToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16U16W32Gfx12, Arithmetic) {
  check_float16<32, true>(rj_arith_v_cvt_f16_u16_w32_gfx12, Conversion16::UnsignedToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtI16F16W32Gfx11, Arithmetic) {
  check_float16<32, false>(rj_arith_v_cvt_i16_f16_w32, Conversion16::HalfToSigned);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtI16F16W32Gfx12, Arithmetic) {
  check_float16<32, true>(rj_arith_v_cvt_i16_f16_w32_gfx12, Conversion16::HalfToSigned);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtU16F16W32Gfx11, Arithmetic) {
  check_float16<32, false>(rj_arith_v_cvt_u16_f16_w32, Conversion16::HalfToUnsigned);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtU16F16W32Gfx12, Arithmetic) {
  check_float16<32, true>(rj_arith_v_cvt_u16_f16_w32_gfx12, Conversion16::HalfToUnsigned);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16F32W32Gfx11, Arithmetic) {
  check_float16<32, false>(rj_arith_v_cvt_f16_f32_w32, Conversion16::FloatToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16F32W32Gfx12, Arithmetic) {
  check_float16<32, true>(rj_arith_v_cvt_f16_f32_w32_gfx12, Conversion16::FloatToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32F16W32Gfx11, Arithmetic) {
  check_float16<32, false>(rj_arith_v_cvt_f32_f16_w32, Conversion16::HalfToFloat);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32F16W32Gfx12, Arithmetic) {
  check_float16<32, true>(rj_arith_v_cvt_f32_f16_w32_gfx12, Conversion16::HalfToFloat);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16I16W64Gfx11, Arithmetic) {
  check_float16<64, false>(rj_arith_v_cvt_f16_i16_w64, Conversion16::SignedToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16I16W64Gfx12, Arithmetic) {
  check_float16<64, true>(rj_arith_v_cvt_f16_i16_w64_gfx12, Conversion16::SignedToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16U16W64Gfx11, Arithmetic) {
  check_float16<64, false>(rj_arith_v_cvt_f16_u16_w64, Conversion16::UnsignedToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16U16W64Gfx12, Arithmetic) {
  check_float16<64, true>(rj_arith_v_cvt_f16_u16_w64_gfx12, Conversion16::UnsignedToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtI16F16W64Gfx11, Arithmetic) {
  check_float16<64, false>(rj_arith_v_cvt_i16_f16_w64, Conversion16::HalfToSigned);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtI16F16W64Gfx12, Arithmetic) {
  check_float16<64, true>(rj_arith_v_cvt_i16_f16_w64_gfx12, Conversion16::HalfToSigned);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtU16F16W64Gfx11, Arithmetic) {
  check_float16<64, false>(rj_arith_v_cvt_u16_f16_w64, Conversion16::HalfToUnsigned);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtU16F16W64Gfx12, Arithmetic) {
  check_float16<64, true>(rj_arith_v_cvt_u16_f16_w64_gfx12, Conversion16::HalfToUnsigned);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16F32W64Gfx11, Arithmetic) {
  check_float16<64, false>(rj_arith_v_cvt_f16_f32_w64, Conversion16::FloatToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF16F32W64Gfx12, Arithmetic) {
  check_float16<64, true>(rj_arith_v_cvt_f16_f32_w64_gfx12, Conversion16::FloatToHalf);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32F16W64Gfx11, Arithmetic) {
  check_float16<64, false>(rj_arith_v_cvt_f32_f16_w64, Conversion16::HalfToFloat);
}

// Check FP16 conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32F16W64Gfx12, Arithmetic) {
  check_float16<64, true>(rj_arith_v_cvt_f32_f16_w64_gfx12, Conversion16::HalfToFloat);
}

TEST(ConversionCvtF16I16W32Gfx942, Arithmetic) {
  check_float16<32, false, true, false>(rj_arith_v_cvt_f16_i16_w32_gfx942,
                                        Conversion16::SignedToHalf);
}

TEST(ConversionCvtF16I16W64Gfx942, Arithmetic) {
  check_float16<64, false, true, false>(rj_arith_v_cvt_f16_i16_w64_gfx942,
                                        Conversion16::SignedToHalf);
}

TEST(ConversionCvtF16U16W32Gfx942, Arithmetic) {
  check_float16<32, false, true, false>(rj_arith_v_cvt_f16_u16_w32_gfx942,
                                        Conversion16::UnsignedToHalf);
}

TEST(ConversionCvtF16U16W64Gfx942, Arithmetic) {
  check_float16<64, false, true, false>(rj_arith_v_cvt_f16_u16_w64_gfx942,
                                        Conversion16::UnsignedToHalf);
}

TEST(ConversionCvtI16F16W32Gfx942, Arithmetic) {
  check_float16<32, false, true, false>(rj_arith_v_cvt_i16_f16_w32_gfx942,
                                        Conversion16::HalfToSigned);
}

TEST(ConversionCvtI16F16W64Gfx942, Arithmetic) {
  check_float16<64, false, true, false>(rj_arith_v_cvt_i16_f16_w64_gfx942,
                                        Conversion16::HalfToSigned);
}

TEST(ConversionCvtU16F16W32Gfx942, Arithmetic) {
  check_float16<32, false, true, false>(rj_arith_v_cvt_u16_f16_w32_gfx942,
                                        Conversion16::HalfToUnsigned);
}

TEST(ConversionCvtU16F16W64Gfx942, Arithmetic) {
  check_float16<64, false, true, false>(rj_arith_v_cvt_u16_f16_w64_gfx942,
                                        Conversion16::HalfToUnsigned);
}

TEST(ConversionCvtF16F32W32Gfx942, Arithmetic) {
  check_float16<32, false, true, false>(rj_arith_v_cvt_f16_f32_w32_gfx942,
                                        Conversion16::FloatToHalf);
}

TEST(ConversionCvtF16F32W64Gfx942, Arithmetic) {
  check_float16<64, false, true, false>(rj_arith_v_cvt_f16_f32_w64_gfx942,
                                        Conversion16::FloatToHalf);
}

// The inherited SDWA prescale keeps 65535/2 finite; VOP3 rounds 65535 to Inf first.
TEST(ConversionCvtF16F32Gfx942, SdwaPrescaling) {
  using namespace rocjitsu::arith::test;
  const uint32_t source_bits[] = {0x477fff00U, 0x7f800000U, 0x7fc00000U, 0x80000000U};
  const uint32_t expected[] = {0x7800U, 0x7c00U, 0x7e00U, 0};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      uint32_t source[64], result[64];
      for (int lane = 0; lane < 64; ++lane) {
        source[lane] = source_bits[lane % 4];
        result[lane] = 0xdeadbeefU;
      }
      const uint32_t *a = source;
      uint32_t *d = result;
      uint32_t status = 0x10;
      ASSERT_EQ(rj_arith_v_cvt_f16_f32_w64_gfx942(
                    cpu | semantics | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, ~0ULL,
                    RJ_ARITH_CVT_SDWA | RJ_ARITH_ALU_OMOD_HALF, &d, &a, &status),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 64; ++lane) {
        EXPECT_EQ(result[lane], expected[lane % 4]);
      }
      EXPECT_EQ(status, 0x10U);
    }
  }
}

// Native widening must preserve the scalar model for every half encoding and NaN policy.
TEST(ConversionCvtF32F16W32Gfx11, AllEncodings) {
  uint32_t input[32], expected[32], actual[32];
  const uint32_t *a = input;
  const auto levels = supported_cpu_levels();
  for (uint64_t ieee : {uint64_t{0}, RJ_ARITH_FP_IEEE_MODE}) {
    for (int base = 0; base < 65536; base += 32) {
      for (int lane = 0; lane < 32; ++lane) {
        input[lane] = uint32_t(base + lane);
      }
      uint32_t *d = expected;
      ASSERT_EQ(
          rj_arith_v_cvt_f32_f16_w32(RJ_ARITH_SEMANTICS_LOOSE | ieee, ~0U, 0, &d, &a, nullptr),
          RJ_ARITH_SUCCESS);
      for (uint64_t cpu : levels) {
        d = actual;
        ASSERT_EQ(rj_arith_v_cvt_f32_f16_w32(cpu | RJ_ARITH_SEMANTICS_LOOSE | ieee, ~0U, 0, &d, &a,
                                             nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          ASSERT_EQ(actual[lane], expected[lane]) << "half=" << input[lane] << " cpu=" << cpu;
        }
      }
    }
  }
}

// gfx1201 captures: guest rounding and FP16_OVFL saturation on narrowing, NaNs to zero on
// integer conversion, and quieted widening NaN payloads. FP16 results keep the high D half.
TEST(ConversionFloat16Gfx12, ExactMatchesHardware) {
  constexpr uint64_t d = 0xa5a5a5a5U;
  for (uint64_t guest : {uint64_t(0), RJ_ARITH_FP_ROUND_ZERO, RJ_ARITH_FP16_OVFL}) {
    const uint64_t expected = guest ? 0xa5a57bffU : 0xa5a57c00U;
    expect_captured_conversion<1, 1>(rj_arith_v_cvt_f16_f32_w32_gfx12, guest, 0x7f7fffff, 0, 0, d,
                                     expected);
    expect_captured_conversion<1, 1>(rj_arith_v_cvt_f16_u16_w32_gfx12, guest, 0xed09ffff, 0, 0, d,
                                     expected);
  }
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_u16_f16_w32_gfx12, 0, 0xc0e0fc00, 0, 0, d,
                                   0xa5a50000U);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_u16_f16_w32_gfx12, 0, 0x93c17e00, 0, 0, d,
                                   0xa5a50000U);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_f32_f16_w32_gfx12, 0, 0xd70f7c01, 0, 0, d,
                                   0x7fc02000);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtF16F32W32Gfx1250, Arithmetic) {
  check_float16<32, true, false, false>(rj_arith_v_cvt_f16_f32_w32_gfx1250,
                                        Conversion16::FloatToHalf);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtF16F32W64Gfx1250, Arithmetic) {
  check_float16<64, true, false, false>(rj_arith_v_cvt_f16_f32_w64_gfx1250,
                                        Conversion16::FloatToHalf);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtF16I16W32Gfx1250, Arithmetic) {
  check_float16<32, true, false, false>(rj_arith_v_cvt_f16_i16_w32_gfx1250,
                                        Conversion16::SignedToHalf);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtF16I16W64Gfx1250, Arithmetic) {
  check_float16<64, true, false, false>(rj_arith_v_cvt_f16_i16_w64_gfx1250,
                                        Conversion16::SignedToHalf);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtF16U16W32Gfx1250, Arithmetic) {
  check_float16<32, true, false, false>(rj_arith_v_cvt_f16_u16_w32_gfx1250,
                                        Conversion16::UnsignedToHalf);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtF16U16W64Gfx1250, Arithmetic) {
  check_float16<64, true, false, false>(rj_arith_v_cvt_f16_u16_w64_gfx1250,
                                        Conversion16::UnsignedToHalf);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtF32F16W32Gfx1250, Arithmetic) {
  check_float16<32, true, false, false>(rj_arith_v_cvt_f32_f16_w32_gfx1250,
                                        Conversion16::HalfToFloat);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtF32F16W64Gfx1250, Arithmetic) {
  check_float16<64, true, false, false>(rj_arith_v_cvt_f32_f16_w64_gfx1250,
                                        Conversion16::HalfToFloat);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtI16F16W32Gfx1250, Arithmetic) {
  check_float16<32, true, false, false>(rj_arith_v_cvt_i16_f16_w32_gfx1250,
                                        Conversion16::HalfToSigned);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtI16F16W64Gfx1250, Arithmetic) {
  check_float16<64, true, false, false>(rj_arith_v_cvt_i16_f16_w64_gfx1250,
                                        Conversion16::HalfToSigned);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtU16F16W32Gfx1250, Arithmetic) {
  check_float16<32, true, false, false>(rj_arith_v_cvt_u16_f16_w32_gfx1250,
                                        Conversion16::HalfToUnsigned);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtU16F16W64Gfx1250, Arithmetic) {
  check_float16<64, true, false, false>(rj_arith_v_cvt_u16_f16_w64_gfx1250,
                                        Conversion16::HalfToUnsigned);
}

// gfx942/gfx950 inherit the RDNA3 model but stay loose-only until the CDNA captures are reviewed.
TEST(ConversionCvtF32F16W32Gfx942, Arithmetic) {
  check_float16<32, false, false, false>(rj_arith_v_cvt_f32_f16_w32_gfx942,
                                         Conversion16::HalfToFloat);
}

// gfx942/gfx950 inherit the RDNA3 model but stay loose-only until the CDNA captures are reviewed.
TEST(ConversionCvtF32F16W64Gfx942, Arithmetic) {
  check_float16<64, false, false, false>(rj_arith_v_cvt_f32_f16_w64_gfx942,
                                         Conversion16::HalfToFloat);
}

// gfx1100 (W7900, mkm2) lanes, MODE words as captured (bit 8 DX10_CLAMP set, bit 9 IEEE as
// noted): hw1100-noieee (2026-09-25) for 0x1f0/0x100/0x8001f0, the Track C hw1100-wide corpus
// (2026-11-18) for round toward +inf (0x1f5) and -inf (0x1fa), and hw2-gfx1100 (2026-10-06) for
// IEEE set (0x3f0). FP16_OVFL saturates FP32 overflow to the largest finite half; directed
// rounding never overflows toward the opposite sign; RDNA3 applies mul:2 only when output
// denormals are flushed and IEEE is clear (0x100 scales 1.0 to 2.0, 0x1f0 does not); widening a
// signaling NaN keeps it signaling with IEEE clear and quiets it with IEEE set; F16 integer
// results saturate, map negative and NaN inputs to zero in the unsigned form, and write the low
// half, keeping the captured 0xa5a5 high half.
TEST(ConversionFloat16Gfx11, ExactMatchesHardware) {
  constexpr uint64_t d = 0xa5a5a5a5U;
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_f32_w32, 0x8001f0, 0, 0x7f7fffff, 0, 0, d, 0xa5a57bffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_f32_w32, 0x8001f0, 0, 0xff7fffff, 0, 0, d, 0xa5a5fbffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_f32_w32, 0x1f0, 0, 0x7f7fffff, 0, 0, d, 0xa5a57c00U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_f32_w32, 0x1fa, 0, 0x7f7fffff, 0, 0, d, 0xa5a57bffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_f32_w32, 0x1f5, 0, 0xff7fffff, 0, 0, d, 0xa5a5fbffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_f32_w32, 0x100, RJ_ARITH_ALU_OMOD_2, 0x3f800000, 0, 0, d, 0xa5a54000U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_f32_w32, 0x1f0, RJ_ARITH_ALU_OMOD_2, 0x3f800000, 0, 0, d, 0xa5a53c00U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_f32_w32, 0x300, RJ_ARITH_ALU_OMOD_2, 0x3f800000, 0, 0, d, 0xa5a53c00U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 32>(
      rj_arith_v_cvt_f32_f16_w32, 0x1f0, 0, 0xd70f7c01, 0, 0, d, 0x7f802000U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 32>(
      rj_arith_v_cvt_f32_f16_w32, 0x3f0, 0, 0xd70f7c01, 0, 0, d, 0x7fc02000U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 32>(
      rj_arith_v_cvt_f32_f16_w32, 0x100, 0, 0xba530001, 0, 0, d, 0);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_i16_f16_w32, 0x1f0, 0, 0x1ec97bff, 0, 0, d, 0xa5a57fffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_i16_f16_w32, 0x1f0, 0, 0xb6877c00, 0, 0, d, 0xa5a57fffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_u16_f16_w32, 0x1f0, 0, 0x457a7bff, 0, 0, d, 0xa5a5ffe0U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_u16_f16_w32, 0x1f0, 0, 0xb6987c00, 0, 0, d, 0xa5a5ffffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_u16_f16_w32, 0x1f0, 0, 0xfaadbc00, 0, 0, d, 0xa5a50000U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_u16_f16_w32, 0x1f0, 0, 0x93c17e00, 0, 0, d, 0xa5a50000U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_i16_w32, 0x1f5, 0, 0xd6a37fff, 0, 0, d, 0xa5a57800U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<1, 1, 16>(
      rj_arith_v_cvt_f16_i16_w32, 0x1fa, 0, 0xd6a37fff, 0, 0, d, 0xa5a577ffU);
}
