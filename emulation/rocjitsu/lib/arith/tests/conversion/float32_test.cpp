// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for conversions between FP32 and 32-bit integers.

#include "conversion/conversion32_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

enum class Conversion32 {
  SignedToFloat,
  UnsignedToFloat,
  FloatToSigned,
  FloatToUnsigned,
  Nearest,
  Floor
};

// Check FP32/integer conversion results, modifiers, and aliasing for the supplied entry point.
template <int Lanes, bool Gfx12, class Call> void check_float32(Call call, Conversion32 operation) {
  int op = int(operation);
  auto input = conversion_inputs<Lanes>(32);
  if (op < 2) {
    const uint32_t edges[] = {0, 1, 3, 16777217, 16777219, 0xffffffffU, 0x80000000U, 0x7fffffffU};
    for (int lane = 0; lane < 8; ++lane) {
      input.v[0][lane] = edges[lane];
    }
  }
  uint64_t modifiers = RJ_ARITH_ALU_CLAMP | (op < 4 ? RJ_ARITH_ALU_OMOD_HALF : 0) |
                       (op < 2 ? 0 : RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A);
  check_conversion<Lanes, 1, 1, 1, true>(
      call, input,
      [=](uint64_t a, uint32_t, uint32_t, uint32_t, uint64_t flags, uint32_t mode) {
        if (op < 2) {
          mode = conversion_effective_mode(flags, mode, Gfx12);
        }
        return conversion32_reference(op, uint32_t(a), mode);
      },
      [=](uint64_t actual, uint64_t expected, uint64_t flags, uint32_t mode) {
        if (op < 2) {
          conversion_float(actual, expected, 32);
        } else {
          conversion_bits(actual, expected, flags, mode);
        }
      },
      modifiers, false,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

} // namespace

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32I32W32, Arithmetic) {
  check_float32<32, false>(rj_arith_v_cvt_f32_i32_w32, Conversion32::SignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32I32W32Gfx12, Arithmetic) {
  check_float32<32, true>(rj_arith_v_cvt_f32_i32_w32_gfx12, Conversion32::SignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32U32W32, Arithmetic) {
  check_float32<32, false>(rj_arith_v_cvt_f32_u32_w32, Conversion32::UnsignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32U32W32Gfx12, Arithmetic) {
  check_float32<32, true>(rj_arith_v_cvt_f32_u32_w32_gfx12, Conversion32::UnsignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtI32F32W32, Arithmetic) {
  check_float32<32, false>(rj_arith_v_cvt_i32_f32_w32, Conversion32::FloatToSigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtI32F32W32Gfx12, Arithmetic) {
  check_float32<32, true>(rj_arith_v_cvt_i32_f32_w32_gfx12, Conversion32::FloatToSigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtU32F32W32, Arithmetic) {
  check_float32<32, false>(rj_arith_v_cvt_u32_f32_w32, Conversion32::FloatToUnsigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtU32F32W32Gfx12, Arithmetic) {
  check_float32<32, true>(rj_arith_v_cvt_u32_f32_w32_gfx12, Conversion32::FloatToUnsigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtNearestI32F32W32, Arithmetic) {
  check_float32<32, false>(rj_arith_v_cvt_nearest_i32_f32_w32, Conversion32::Nearest);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtFloorI32F32W32, Arithmetic) {
  check_float32<32, false>(rj_arith_v_cvt_floor_i32_f32_w32, Conversion32::Floor);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32I32W64, Arithmetic) {
  check_float32<64, false>(rj_arith_v_cvt_f32_i32_w64, Conversion32::SignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32I32W64Gfx12, Arithmetic) {
  check_float32<64, true>(rj_arith_v_cvt_f32_i32_w64_gfx12, Conversion32::SignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32U32W64, Arithmetic) {
  check_float32<64, false>(rj_arith_v_cvt_f32_u32_w64, Conversion32::UnsignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32U32W64Gfx12, Arithmetic) {
  check_float32<64, true>(rj_arith_v_cvt_f32_u32_w64_gfx12, Conversion32::UnsignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtI32F32W64, Arithmetic) {
  check_float32<64, false>(rj_arith_v_cvt_i32_f32_w64, Conversion32::FloatToSigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtI32F32W64Gfx12, Arithmetic) {
  check_float32<64, true>(rj_arith_v_cvt_i32_f32_w64_gfx12, Conversion32::FloatToSigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtU32F32W64, Arithmetic) {
  check_float32<64, false>(rj_arith_v_cvt_u32_f32_w64, Conversion32::FloatToUnsigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtU32F32W64Gfx12, Arithmetic) {
  check_float32<64, true>(rj_arith_v_cvt_u32_f32_w64_gfx12, Conversion32::FloatToUnsigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtNearestI32F32W64, Arithmetic) {
  check_float32<64, false>(rj_arith_v_cvt_nearest_i32_f32_w64, Conversion32::Nearest);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtFloorI32F32W64, Arithmetic) {
  check_float32<64, false>(rj_arith_v_cvt_floor_i32_f32_w64, Conversion32::Floor);
}
