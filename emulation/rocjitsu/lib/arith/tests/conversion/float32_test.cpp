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
template <int Lanes, bool Gfx12, typename Call>
void check_float32(Call call, Conversion32 operation) {
  int op = int(operation);
  auto input = conversion_inputs<Lanes>(32);
  if (op < 2) {
    const uint32_t edges[] = {0, 1, 3, 16777217, 16777219, 0xffffffffU, 0x80000000U, 0x7fffffffU};
    for (int lane = 0; lane < 8; ++lane) {
      input.v[0][lane] = edges[lane];
    }
  }
  if (op < 2) {
    const uint32_t *a[] = {input.v[0]};
    uint32_t *d[] = {input.v[6]};
    const auto exec_mask = WaveMask<Lanes>(1) << (Lanes - 1);
    input.v[0][Lanes - 1] = 0x01000001;
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t rounding : {uint64_t{0}, RJ_ARITH_FP_ROUND_UP}) {
        ASSERT_EQ(invoke_conversion(call, cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | rounding,
                                    exec_mask, 0, d, nullptr, a),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(input.v[6][Lanes - 1], rounding ? 0x4b800001U : 0x4b800000U);
      }
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
        if (op >= 2 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) &&
            (a & 0x7fffffffU) < 0x800000U) {
          a &= 0x80000000U;
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
      modifiers, true,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
      /*status_free=*/false);
}

} // namespace

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32I32W32Gfx11, Arithmetic) {
  check_float32<32, false>(rj_arith_v_cvt_f32_i32_w32, Conversion32::SignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32I32W32Gfx12, Arithmetic) {
  check_float32<32, true>(rj_arith_v_cvt_f32_i32_w32_gfx12, Conversion32::SignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32U32W32Gfx11, Arithmetic) {
  check_float32<32, false>(rj_arith_v_cvt_f32_u32_w32, Conversion32::UnsignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32U32W32Gfx12, Arithmetic) {
  check_float32<32, true>(rj_arith_v_cvt_f32_u32_w32_gfx12, Conversion32::UnsignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtI32F32W32Gfx11, Arithmetic) {
  check_float32<32, false>(rj_arith_v_cvt_i32_f32_w32, Conversion32::FloatToSigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtI32F32W32Gfx12, Arithmetic) {
  check_float32<32, true>(rj_arith_v_cvt_i32_f32_w32_gfx12, Conversion32::FloatToSigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtU32F32W32Gfx11, Arithmetic) {
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
TEST(ConversionCvtF32I32W64Gfx11, Arithmetic) {
  check_float32<64, false>(rj_arith_v_cvt_f32_i32_w64, Conversion32::SignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32I32W64Gfx12, Arithmetic) {
  check_float32<64, true>(rj_arith_v_cvt_f32_i32_w64_gfx12, Conversion32::SignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32U32W64Gfx11, Arithmetic) {
  check_float32<64, false>(rj_arith_v_cvt_f32_u32_w64, Conversion32::UnsignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtF32U32W64Gfx12, Arithmetic) {
  check_float32<64, true>(rj_arith_v_cvt_f32_u32_w64_gfx12, Conversion32::UnsignedToFloat);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtI32F32W64Gfx11, Arithmetic) {
  check_float32<64, false>(rj_arith_v_cvt_i32_f32_w64, Conversion32::FloatToSigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtI32F32W64Gfx12, Arithmetic) {
  check_float32<64, true>(rj_arith_v_cvt_i32_f32_w64_gfx12, Conversion32::FloatToSigned);
}

// Check FP32/integer conversion results, modifiers, and aliasing.
TEST(ConversionCvtU32F32W64Gfx11, Arithmetic) {
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

// gfx1201 captures: guest rounding of wide integers, saturation of infinities and finite
// overflow, and the half-up rounding of CVT_NEAREST and the floor of CVT_FLOOR.
TEST(ConversionFloat32Gfx12, ExactMatchesHardware) {
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_f32_i32_w32_gfx12, 0, 0x7fffffff, 0, 0, 0,
                                   0x4f000000);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_f32_i32_w32_gfx12, RJ_ARITH_FP_ROUND_ZERO,
                                   0x7fffffff, 0, 0, 0, 0x4effffff);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_f32_u32_w32_gfx12, 0, 0xffffffff, 0, 0, 0,
                                   0x4f800000);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_f32_u32_w32_gfx12, RJ_ARITH_FP_ROUND_ZERO,
                                   0xffffffff, 0, 0, 0, 0x4f7fffff);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_i32_f32_w32_gfx12, 0, 0x7f7fffff, 0, 0, 0,
                                   0x7fffffff);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_i32_f32_w32_gfx12, 0, 0xff800000, 0, 0, 0,
                                   0x80000000);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_nearest_i32_f32_w32, 0, 0x3fc00000, 0, 0, 0, 2);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_nearest_i32_f32_w32, 0, 0x40200000, 0, 0, 0, 3);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_nearest_i32_f32_w32, 0, 0xc0200000, 0, 0, 0,
                                   0xfffffffe);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_floor_i32_f32_w32, 0, 0xbf000000, 0, 0, 0,
                                   0xffffffff);
  expect_captured_conversion<1, 1>(rj_arith_v_cvt_floor_i32_f32_w32, 0, 0xc0200000, 0, 0, 0,
                                   0xfffffffd);
}
