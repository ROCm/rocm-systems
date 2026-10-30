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
template <int Lanes, int InputWords, int OutputWords, bool Gfx12, typename Call>
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
        if (op == 4 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) &&
            (a & 0x7fffffffU) < 0x800000U) {
          a &= 0x80000000U;
        }
        if (op == 5 && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
            (a & 0x7fffffffffffffffULL) < 0x380ffffff0000000ULL) {
          a &= 0x8000000000000000ULL;
        }
        uint64_t result = conversion64_reference(op, a, mode);
        if (op == 4 && (a & 0x7fffffffU) > 0x7f800000U) {
          uint32_t bits = uint32_t(a);
          if (mode & RJ_ARITH_ALU_ABS_A) {
            bits &= 0x7fffffffU;
          }
          if (mode & RJ_ARITH_ALU_NEG_A) {
            bits ^= 0x80000000U;
          }
          result = (uint64_t(bits & 0x80000000U) << 32) | 0x7ff0000000000000ULL |
                   (uint64_t(bits & 0x7fffffU) << 29);
          if (Gfx12 || (flags & RJ_ARITH_FP_IEEE_MODE)) {
            result |= 1ULL << 51;
          }
          if (mode & RJ_ARITH_ALU_CLAMP) {
            result = 0;
          }
        }
        return result;
      },
      [=](uint64_t actual, uint64_t expected, uint64_t flags, uint32_t mode) {
        if (op == 2 || op == 3) {
          conversion_bits(actual, expected, flags, mode);
        } else if (op == 4 && !Gfx12 && !(flags & RJ_ARITH_FP_IEEE_MODE)) {
          EXPECT_EQ(actual, expected);
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
TEST(ConversionCvtF64I32W32Gfx11, Arithmetic) {
  check_float64<32, 1, 2, false>(rj_arith_v_cvt_f64_i32_w32, Conversion64::SignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64I32W32Gfx12, Arithmetic) {
  check_float64<32, 1, 2, true>(rj_arith_v_cvt_f64_i32_w32_gfx12, Conversion64::SignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64U32W32Gfx11, Arithmetic) {
  check_float64<32, 1, 2, false>(rj_arith_v_cvt_f64_u32_w32, Conversion64::UnsignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64U32W32Gfx12, Arithmetic) {
  check_float64<32, 1, 2, true>(rj_arith_v_cvt_f64_u32_w32_gfx12, Conversion64::UnsignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtI32F64W32Gfx11, Arithmetic) {
  check_float64<32, 2, 1, false>(rj_arith_v_cvt_i32_f64_w32, Conversion64::DoubleToSigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtI32F64W32Gfx12, Arithmetic) {
  check_float64<32, 2, 1, true>(rj_arith_v_cvt_i32_f64_w32_gfx12, Conversion64::DoubleToSigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtU32F64W32Gfx11, Arithmetic) {
  check_float64<32, 2, 1, false>(rj_arith_v_cvt_u32_f64_w32, Conversion64::DoubleToUnsigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtU32F64W32Gfx12, Arithmetic) {
  check_float64<32, 2, 1, true>(rj_arith_v_cvt_u32_f64_w32_gfx12, Conversion64::DoubleToUnsigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64F32W32Gfx11, Arithmetic) {
  check_float64<32, 1, 2, false>(rj_arith_v_cvt_f64_f32_w32, Conversion64::FloatToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64F32W32Gfx12, Arithmetic) {
  check_float64<32, 1, 2, true>(rj_arith_v_cvt_f64_f32_w32_gfx12, Conversion64::FloatToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF32F64W32Gfx11, Arithmetic) {
  check_float64<32, 2, 1, false>(rj_arith_v_cvt_f32_f64_w32, Conversion64::DoubleToFloat);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF32F64W32Gfx12, Arithmetic) {
  check_float64<32, 2, 1, true>(rj_arith_v_cvt_f32_f64_w32_gfx12, Conversion64::DoubleToFloat);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64I32W64Gfx11, Arithmetic) {
  check_float64<64, 1, 2, false>(rj_arith_v_cvt_f64_i32_w64, Conversion64::SignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64I32W64Gfx12, Arithmetic) {
  check_float64<64, 1, 2, true>(rj_arith_v_cvt_f64_i32_w64_gfx12, Conversion64::SignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64U32W64Gfx11, Arithmetic) {
  check_float64<64, 1, 2, false>(rj_arith_v_cvt_f64_u32_w64, Conversion64::UnsignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64U32W64Gfx12, Arithmetic) {
  check_float64<64, 1, 2, true>(rj_arith_v_cvt_f64_u32_w64_gfx12, Conversion64::UnsignedToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtI32F64W64Gfx11, Arithmetic) {
  check_float64<64, 2, 1, false>(rj_arith_v_cvt_i32_f64_w64, Conversion64::DoubleToSigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtI32F64W64Gfx12, Arithmetic) {
  check_float64<64, 2, 1, true>(rj_arith_v_cvt_i32_f64_w64_gfx12, Conversion64::DoubleToSigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtU32F64W64Gfx11, Arithmetic) {
  check_float64<64, 2, 1, false>(rj_arith_v_cvt_u32_f64_w64, Conversion64::DoubleToUnsigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtU32F64W64Gfx12, Arithmetic) {
  check_float64<64, 2, 1, true>(rj_arith_v_cvt_u32_f64_w64_gfx12, Conversion64::DoubleToUnsigned);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64F32W64Gfx11, Arithmetic) {
  check_float64<64, 1, 2, false>(rj_arith_v_cvt_f64_f32_w64, Conversion64::FloatToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF64F32W64Gfx12, Arithmetic) {
  check_float64<64, 1, 2, true>(rj_arith_v_cvt_f64_f32_w64_gfx12, Conversion64::FloatToDouble);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF32F64W64Gfx11, Arithmetic) {
  check_float64<64, 2, 1, false>(rj_arith_v_cvt_f32_f64_w64, Conversion64::DoubleToFloat);
}

// Check wide conversions and split-word destination aliases.
TEST(ConversionCvtF32F64W64Gfx12, Arithmetic) {
  check_float64<64, 2, 1, true>(rj_arith_v_cvt_f32_f64_w64_gfx12, Conversion64::DoubleToFloat);
}

// Ties, tiny values, and finite overflow distinguish all four guest rounding modes.
// These small binary cases are exact; no host rounding-mode changes are involved.
TEST(ConversionRounding, GuestModes) {
  struct Case {
    uint64_t input;
    uint32_t expected[4]; // nearest, upward, downward, toward zero
  };
  const Case half_cases[] = {
      {0x3f801000U, {0x3c00, 0x3c01, 0x3c00, 0x3c00}},
      {0xbf801000U, {0xbc00, 0xbc00, 0xbc01, 0xbc00}},
      {0x33000000U, {0, 1, 0, 0}},
      {0x47800000U, {0x7c00, 0x7c00, 0x7bff, 0x7bff}},
  };
  const Case float_cases[] = {
      {0x3ff0000010000000ULL, {0x3f800000, 0x3f800001, 0x3f800000, 0x3f800000}},
      {0xbff0000010000000ULL, {0xbf800000, 0xbf800000, 0xbf800001, 0xbf800000}},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int direction = 0; direction < 4; ++direction) {
      auto check = [&](const Case &test, bool half) {
        Registers<3, 32> r{};
        for (int lane = 0; lane < 32; ++lane) {
          r.v[0][lane] = uint32_t(test.input);
          r.v[1][lane] = uint32_t(test.input >> 32);
          r.v[2][lane] = 0;
        }
        uint64_t flags = cpu | RJ_ARITH_SEMANTICS_LOOSE | (uint64_t(direction) << 24);
        auto call = half ? rj_arith_v_cvt_f16_f32_w32_gfx12 : rj_arith_v_cvt_f32_f64_w32_gfx12;
        ASSERT_EQ(call(flags, 0xffffffffU, 0, r.v + 2, r.v, nullptr), RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(r.v[2][lane], test.expected[direction]) << cpu << '/' << direction;
        }
      };
      for (const auto &test : half_cases) {
        check(test, true);
      }
      for (const auto &test : float_cases) {
        check(test, false);
      }
    }
  }
}
