// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for conversions involving FP64 and split-word VGPR results.

#include "conversion/conversion64_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

enum class Conversion64 {
  SignedToDouble,
  UnsignedToDouble,
  DoubleToSigned,
  DoubleToUnsigned,
  FloatToDouble,
  DoubleToFloat
};

// Check wide conversions and split-word destination aliases for the supplied entry point.
template <int Lanes, int InputWords, int OutputWords, bool Gfx12, class Call>
void check_float64(Call call, Conversion64 operation) {
  int op = int(operation);
  auto input = conversion_inputs<Lanes>(InputWords == 2 ? 64 : 32);
  if (op < 2) {
    input.v[0][0] = 3;
    input.v[0][1] = 0x7fffffffU;
    input.v[0][2] = 0xffffffffU;
  }
  uint64_t modifiers = RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP |
                       (op < 2 ? 0 : RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A);
  check_conversion<Lanes, 1, InputWords, OutputWords, false>(
      call, input,
      [=](uint64_t a, uint32_t, uint32_t, uint32_t, uint64_t flags, uint32_t mode) {
        if (op != 2 && op != 3) {
          mode = conversion_effective_mode(flags, mode, Gfx12);
        }
        return conversion64_reference(op, a, mode);
      },
      [=](uint64_t actual, uint64_t expected, uint64_t flags, uint32_t mode) {
        if (op == 2 || op == 3) {
          conversion_bits(actual, expected, flags, mode);
        } else {
          conversion_float(actual, expected, OutputWords * 32);
        }
      },
      modifiers, false,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

} // namespace

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64I32W32, Arithmetic) {
  check_float64<32, 1, 2, false>(rj_arith_v_cvt_f64_i32_w32, Conversion64::SignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64I32W32Gfx12, Arithmetic) {
  check_float64<32, 1, 2, true>(rj_arith_v_cvt_f64_i32_w32_gfx12, Conversion64::SignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64U32W32, Arithmetic) {
  check_float64<32, 1, 2, false>(rj_arith_v_cvt_f64_u32_w32, Conversion64::UnsignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64U32W32Gfx12, Arithmetic) {
  check_float64<32, 1, 2, true>(rj_arith_v_cvt_f64_u32_w32_gfx12, Conversion64::UnsignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtI32F64W32, Arithmetic) {
  check_float64<32, 2, 1, false>(rj_arith_v_cvt_i32_f64_w32, Conversion64::DoubleToSigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtI32F64W32Gfx12, Arithmetic) {
  check_float64<32, 2, 1, true>(rj_arith_v_cvt_i32_f64_w32_gfx12, Conversion64::DoubleToSigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtU32F64W32, Arithmetic) {
  check_float64<32, 2, 1, false>(rj_arith_v_cvt_u32_f64_w32, Conversion64::DoubleToUnsigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtU32F64W32Gfx12, Arithmetic) {
  check_float64<32, 2, 1, true>(rj_arith_v_cvt_u32_f64_w32_gfx12, Conversion64::DoubleToUnsigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64F32W32, Arithmetic) {
  check_float64<32, 1, 2, false>(rj_arith_v_cvt_f64_f32_w32, Conversion64::FloatToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64F32W32Gfx12, Arithmetic) {
  check_float64<32, 1, 2, true>(rj_arith_v_cvt_f64_f32_w32_gfx12, Conversion64::FloatToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF32F64W32, Arithmetic) {
  check_float64<32, 2, 1, false>(rj_arith_v_cvt_f32_f64_w32, Conversion64::DoubleToFloat);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF32F64W32Gfx12, Arithmetic) {
  check_float64<32, 2, 1, true>(rj_arith_v_cvt_f32_f64_w32_gfx12, Conversion64::DoubleToFloat);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64I32W64, Arithmetic) {
  check_float64<64, 1, 2, false>(rj_arith_v_cvt_f64_i32_w64, Conversion64::SignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64I32W64Gfx12, Arithmetic) {
  check_float64<64, 1, 2, true>(rj_arith_v_cvt_f64_i32_w64_gfx12, Conversion64::SignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64U32W64, Arithmetic) {
  check_float64<64, 1, 2, false>(rj_arith_v_cvt_f64_u32_w64, Conversion64::UnsignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64U32W64Gfx12, Arithmetic) {
  check_float64<64, 1, 2, true>(rj_arith_v_cvt_f64_u32_w64_gfx12, Conversion64::UnsignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtI32F64W64, Arithmetic) {
  check_float64<64, 2, 1, false>(rj_arith_v_cvt_i32_f64_w64, Conversion64::DoubleToSigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtI32F64W64Gfx12, Arithmetic) {
  check_float64<64, 2, 1, true>(rj_arith_v_cvt_i32_f64_w64_gfx12, Conversion64::DoubleToSigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtU32F64W64, Arithmetic) {
  check_float64<64, 2, 1, false>(rj_arith_v_cvt_u32_f64_w64, Conversion64::DoubleToUnsigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtU32F64W64Gfx12, Arithmetic) {
  check_float64<64, 2, 1, true>(rj_arith_v_cvt_u32_f64_w64_gfx12, Conversion64::DoubleToUnsigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64F32W64, Arithmetic) {
  check_float64<64, 1, 2, false>(rj_arith_v_cvt_f64_f32_w64, Conversion64::FloatToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64F32W64Gfx12, Arithmetic) {
  check_float64<64, 1, 2, true>(rj_arith_v_cvt_f64_f32_w64_gfx12, Conversion64::FloatToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF32F64W64, Arithmetic) {
  check_float64<64, 2, 1, false>(rj_arith_v_cvt_f32_f64_w64, Conversion64::DoubleToFloat);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF32F64W64Gfx12, Arithmetic) {
  check_float64<64, 2, 1, true>(rj_arith_v_cvt_f32_f64_w64_gfx12, Conversion64::DoubleToFloat);
}
