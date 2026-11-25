// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wide low-precision widening: cross-register fields, gfx950 exponent scales and
// gfx1250 E8M0 byte scales, preserved lanes, and overlapping destinations.

#include "common/contracts.h"
#include "common/conversion64_reference.h"
#include "common/cpu.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Decode finite fields mathematically, supplying OCP special encodings explicitly.
// Scale in FP32 and round to the output format using the independent reference.
uint32_t reference(int source, int destination, uint32_t code, int scale, bool saturate) {
  if (scale == 255) {
    return destination == 0 ? 0x7fc00000U : destination == 1 ? 0x7e00U : 0x7fc0U;
  }
  int width = source == 0 ? 4 : source >= 3 ? 6 : 8;
  int fraction = source == 0 ? 1 : (source == 1 || source == 3) ? 3 : 2;
  int bias = source == 1 ? 7 : source == 2 ? 15 : source == 4 ? 3 : 1;
  int sign = (code >> (width - 1)) & 1;
  int magnitude_bits = code & ((1 << (width - 1)) - 1);
  int exponent = magnitude_bits >> fraction, mantissa = code & ((1 << fraction) - 1);
  double magnitude = std::ldexp(double(exponent ? (1 << fraction) + mantissa : mantissa),
                                (exponent ? exponent : 1) - bias - fraction);
  float decoded = float(sign ? -magnitude : magnitude);
  if ((source == 1 && magnitude_bits == 127) || (source == 2 && magnitude_bits > 124)) {
    decoded = std::bit_cast<float>((uint32_t(sign) << 31) | 0x7fc00000U);
  } else if (source == 2 && magnitude_bits == 124) {
    decoded = std::bit_cast<float>((uint32_t(sign) << 31) | 0x7f800000U);
  }
  float multiplier =
      scale == 255 ? std::bit_cast<float>(0x7fc00000U) : float(std::ldexp(1.0, scale - 127));
  float value = decoded * multiplier;
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if (!destination) {
    return bits;
  }
  uint32_t result = uint32_t(
      conversion_reencode(bits, 23, 127, destination == 1 ? 10 : 7, destination == 1 ? 15 : 127));
  uint32_t infinity = destination == 1 ? 0x7c00U : 0x7f80U;
  if (saturate && (bits & 0x7fffffffU) < 0x7f800000U && (result & 0x7fff) == infinity) {
    --result;
  }
  return result;
}

// Share each entry point's API contract, deterministic data, exact reference
// for this inherited model, and a small set of EXEC/aliasing combinations.
template <int Lanes, int Source, int Dst, bool ByteScale = false, typename Call>
void check(Call call) {
  using Mask = WaveMask<Lanes>;
  constexpr int width = Source == 0 ? 4 : Source >= 3 ? 6 : 8;
  constexpr int elements = ByteScale ? (width == 6 ? 16 : 8) : 32;
  constexpr int words = Dst == 0 ? elements : elements / 2;
  Registers<40, Lanes> input;
  std::minstd_rand random;
  constexpr int exponents[] = {0, 1, 100, 110, 126, 127, 128, 140, 150, 240, 254, 255};
  for (int reg = 0; reg < 40; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = random_word(random);
    }
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if constexpr (ByteScale) {
      input.v[6][lane] = 0;
      for (int byte = 0; byte < 4; ++byte) {
        input.v[6][lane] |= uint32_t(exponents[(lane + byte * 3) % 12]) << (byte * 8);
      }
    } else {
      input.v[6][lane] = (uint32_t(exponents[lane % 12]) << 23) | (input.v[6][lane] & 0x807fffffU);
    }
  }
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t mode) {
        return call(flags, ~Mask(0), mode, nullptr, nullptr, nullptr);
      },
      ByteScale ? RJ_ARITH_CVT_BYTE_3 : 0);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 5; ++scenario) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario);
        uint64_t flags = cpu | semantics | (scenario % 2 ? RJ_ARITH_FP16_OVFL : 0);
        int destination = scenario == 0 ? 8 : scenario == 2 ? 0 : scenario == 3 ? 3 : 6;
        int shift = ByteScale ? (scenario % 4) * 8 : 23;
        uint64_t mode = ByteScale ? uint64_t(scenario % 4) << 16 : 0;
        auto actual = input;
        int error = call(flags, exec_mask, mode, actual.v + destination, actual.v, actual.v + 6);
        bool exact = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                     semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
        ASSERT_EQ(error, exact ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
        auto expected = input;
        if (!exact) {
          for (int lane = 0; lane < Lanes; ++lane) {
            int scale = (input.v[6][lane] >> shift) & 255;
            if (!((exec_mask >> lane) & 1) || (!ByteScale && scale == 255)) {
              continue;
            }
            for (int element = 0; element < elements; ++element) {
              // Assemble bits independently of the implementation's word-crossing extraction.
              uint32_t code = 0;
              for (int bit = 0; bit < width; ++bit) {
                int offset = element * width + bit;
                code |= ((input.v[offset / 32][lane] >> (offset % 32)) & 1) << bit;
              }
              uint32_t value = reference(Source, Dst, code, scale, flags & RJ_ARITH_FP16_OVFL);
              int reg = destination + (Dst == 0 ? element : element / 2);
              if (Dst == 0 || element % 2 == 0) {
                expected.v[reg][lane] = value;
              } else {
                expected.v[reg][lane] |= value << 16;
              }
            }
          }
        }
        for (int reg = 0; reg < words; ++reg) {
          for (int lane = 0; lane < Lanes; ++lane) {
            EXPECT_EQ(actual.v[destination + reg][lane], expected.v[destination + reg][lane])
                << "word=" << reg << ", lane=" << lane;
          }
        }
        actual.expect_preserved(input, destination, exact ? 0 : words);
      }
    }
  }
}
} // namespace

TEST(ScaledWideFP6F32W32, Arithmetic) {
  check<32, 3, 0>(rj_arith_v_cvt_scalef32_pk32_f32_fp6_w32_gfx950);
}

TEST(ScaledWideFP6F32W64, Arithmetic) {
  check<64, 3, 0>(rj_arith_v_cvt_scalef32_pk32_f32_fp6_w64_gfx950);
}

TEST(ScaledWideFP6F16W32, Arithmetic) {
  check<32, 3, 1>(rj_arith_v_cvt_scalef32_pk32_f16_fp6_w32_gfx950);
}

TEST(ScaledWideFP6F16W64, Arithmetic) {
  check<64, 3, 1>(rj_arith_v_cvt_scalef32_pk32_f16_fp6_w64_gfx950);
}

TEST(ScaledWideFP6BF16W32, Arithmetic) {
  check<32, 3, 2>(rj_arith_v_cvt_scalef32_pk32_bf16_fp6_w32_gfx950);
}

TEST(ScaledWideFP6BF16W64, Arithmetic) {
  check<64, 3, 2>(rj_arith_v_cvt_scalef32_pk32_bf16_fp6_w64_gfx950);
}

TEST(ScaledWideBF6F32W32, Arithmetic) {
  check<32, 4, 0>(rj_arith_v_cvt_scalef32_pk32_f32_bf6_w32_gfx950);
}

TEST(ScaledWideBF6F32W64, Arithmetic) {
  check<64, 4, 0>(rj_arith_v_cvt_scalef32_pk32_f32_bf6_w64_gfx950);
}

TEST(ScaledWideBF6F16W32, Arithmetic) {
  check<32, 4, 1>(rj_arith_v_cvt_scalef32_pk32_f16_bf6_w32_gfx950);
}

TEST(ScaledWideBF6F16W64, Arithmetic) {
  check<64, 4, 1>(rj_arith_v_cvt_scalef32_pk32_f16_bf6_w64_gfx950);
}

TEST(ScaledWideBF6BF16W32, Arithmetic) {
  check<32, 4, 2>(rj_arith_v_cvt_scalef32_pk32_bf16_bf6_w32_gfx950);
}

TEST(ScaledWideBF6BF16W64, Arithmetic) {
  check<64, 4, 2>(rj_arith_v_cvt_scalef32_pk32_bf16_bf6_w64_gfx950);
}

TEST(ScaledWideFP6F32W32Gfx1250, Arithmetic) {
  check<32, 3, 0, true>(rj_arith_v_cvt_scale_pk16_f32_fp6_w32_gfx1250);
}

TEST(ScaledWideFP6F16W32Gfx1250, Arithmetic) {
  check<32, 3, 1, true>(rj_arith_v_cvt_scale_pk16_f16_fp6_w32_gfx1250);
}

TEST(ScaledWideFP6BF16W32Gfx1250, Arithmetic) {
  check<32, 3, 2, true>(rj_arith_v_cvt_scale_pk16_bf16_fp6_w32_gfx1250);
}

TEST(ScaledWideBF6F32W32Gfx1250, Arithmetic) {
  check<32, 4, 0, true>(rj_arith_v_cvt_scale_pk16_f32_bf6_w32_gfx1250);
}

TEST(ScaledWideBF6F16W32Gfx1250, Arithmetic) {
  check<32, 4, 1, true>(rj_arith_v_cvt_scale_pk16_f16_bf6_w32_gfx1250);
}

TEST(ScaledWideBF6BF16W32Gfx1250, Arithmetic) {
  check<32, 4, 2, true>(rj_arith_v_cvt_scale_pk16_bf16_bf6_w32_gfx1250);
}

TEST(ScaledWideFP4F32W32Gfx1250, Arithmetic) {
  check<32, 0, 0, true>(rj_arith_v_cvt_scale_pk8_f32_fp4_w32_gfx1250);
}

TEST(ScaledWideFP4F16W32Gfx1250, Arithmetic) {
  check<32, 0, 1, true>(rj_arith_v_cvt_scale_pk8_f16_fp4_w32_gfx1250);
}

TEST(ScaledWideFP4BF16W32Gfx1250, Arithmetic) {
  check<32, 0, 2, true>(rj_arith_v_cvt_scale_pk8_bf16_fp4_w32_gfx1250);
}

TEST(ScaledWideFP8F32W32Gfx1250, Arithmetic) {
  check<32, 1, 0, true>(rj_arith_v_cvt_scale_pk8_f32_fp8_w32_gfx1250);
}

TEST(ScaledWideFP8F16W32Gfx1250, Arithmetic) {
  check<32, 1, 1, true>(rj_arith_v_cvt_scale_pk8_f16_fp8_w32_gfx1250);
}

TEST(ScaledWideFP8BF16W32Gfx1250, Arithmetic) {
  check<32, 1, 2, true>(rj_arith_v_cvt_scale_pk8_bf16_fp8_w32_gfx1250);
}

TEST(ScaledWideBF8F32W32Gfx1250, Arithmetic) {
  check<32, 2, 0, true>(rj_arith_v_cvt_scale_pk8_f32_bf8_w32_gfx1250);
}

TEST(ScaledWideBF8F16W32Gfx1250, Arithmetic) {
  check<32, 2, 1, true>(rj_arith_v_cvt_scale_pk8_f16_bf8_w32_gfx1250);
}

TEST(ScaledWideBF8BF16W32Gfx1250, Arithmetic) {
  check<32, 2, 2, true>(rj_arith_v_cvt_scale_pk8_bf16_bf8_w32_gfx1250);
}
