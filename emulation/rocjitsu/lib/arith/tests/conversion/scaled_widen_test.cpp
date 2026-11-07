// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx950 scaled widening: inherited scaling/rounding, selectors, EXEC and
// overlapping storage, shared across FP4/FP8/BF8 and destination formats.

#include "common/conversion64_reference.h"
#include "common/fp8_conversion_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Independently decode OCP narrow values, retaining the input NaN sign.
float decode(int source, uint32_t raw) {
  if (source == 0) {
    int magnitude = raw & 7;
    float value = magnitude < 2 ? float(magnitude) / 2
                                : std::ldexp(float(2 + (magnitude & 1)), (magnitude >> 1) - 2);
    return raw & 8 ? -value : value;
  }
  uint32_t bits = fp8_conversion_reference(source == 2, uint8_t(raw));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    bits = ((raw & 0x80) << 24) | 0x7fc00000U;
  }
  return std::bit_cast<float>(bits);
}

// Preserve the old generator's double scaling, followed by FP32 rounding and
// optional 16-bit narrowing. The independent integer encoder supplies RNE.
uint32_t reference(int source, int destination, uint32_t raw, uint32_t scale, bool saturate) {
  int exponent = (scale >> 23) & 255;
  if (exponent == 255) {
    return destination == 0 ? 0x7fc00000U : destination == 1 ? 0x7e00U : 0x7fc0U;
  }
  float value = float(double(decode(source, raw)) * std::ldexp(1.0, exponent - 127));
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if (!destination) {
    return bits;
  }
  int fraction = destination == 1 ? 10 : 7, bias = destination == 1 ? 15 : 127;
  uint32_t result = uint32_t(conversion_reencode(bits, 23, 127, fraction, bias));
  uint32_t infinity = destination == 1 ? 0x7c00U : 0x7f80U;
  if (saturate && (bits & 0x7fffffffU) < 0x7f800000U && (result & 0x7fff) == infinity) {
    --result;
  }
  return result;
}

// Exercise finite range boundaries and special scale exponents, then share
// API, semantics, selector, inactive-lane and overlapping-register checks.
template <int Lanes, int Src, int Dst, bool Packed, typename Call> void check(Call call) {
  auto input = conversion_inputs<Lanes>(32);
  constexpr uint32_t codes[] = {0x807f7c00U, 0xfffe7e01U, 0x01807fffU, 0x7c007e80U};
  constexpr int exponents[] = {0, 1, 2, 100, 110, 126, 127, 128, 140, 150, 240, 254, 255};
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[2][lane] = (uint32_t(exponents[lane % 13]) << 23) | (input.v[2][lane] & 0x807fffffU);
    if (lane < 4 || lane % 13 == 12) {
      input.v[0][lane] = codes[lane % 4];
    }
  }
  constexpr uint64_t modifiers = (Packed && Src != 0 ? RJ_ARITH_ALU_HIGH_A : RJ_ARITH_CVT_BYTE_3) |
                                 (!Packed && Dst == 1 ? RJ_ARITH_ALU_HIGH_D : 0);
  check_conversion<Lanes, 2, 1, Packed && Dst == 0 ? 2 : 1, false>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t, uint32_t old, uint64_t flags, uint32_t mode) {
        int shift =
            Packed && Src != 0 ? (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0) : 8 * int((mode >> 16) & 3);
        uint32_t raw = uint32_t(a >> shift);
        uint64_t result = reference(Src, Dst, raw, b, flags & RJ_ARITH_FP16_OVFL);
        if constexpr (Packed) {
          result |= uint64_t(reference(Src, Dst, raw >> (Src == 0 ? 4 : 8), b,
                                       flags & RJ_ARITH_FP16_OVFL))
                    << (Dst == 0 ? 32 : 16);
        } else if constexpr (Dst == 1) {
          int destination_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
          result = (old & ~(0xffffU << destination_shift)) | (result << destination_shift);
        }
        return result;
      },
      conversion_bits, modifiers, false);
}
} // namespace

// Check scaled FP8 to F32 pairs.
TEST(ScaledWidenF32FP8PackedW32, Arithmetic) {
  check<32, 1, 0, true>(rj_arith_v_cvt_scalef32_pk_f32_fp8_w32_gfx950);
}

// Check scaled FP8 to F32 pairs.
TEST(ScaledWidenF32FP8PackedW64, Arithmetic) {
  check<64, 1, 0, true>(rj_arith_v_cvt_scalef32_pk_f32_fp8_w64_gfx950);
}

// Check scaled FP8 to F32 values.
TEST(ScaledWidenF32FP8SingleW32, Arithmetic) {
  check<32, 1, 0, false>(rj_arith_v_cvt_scalef32_f32_fp8_w32_gfx950);
}

// Check scaled FP8 to F32 values.
TEST(ScaledWidenF32FP8SingleW64, Arithmetic) {
  check<64, 1, 0, false>(rj_arith_v_cvt_scalef32_f32_fp8_w64_gfx950);
}

// Check scaled BF8 to F32 pairs.
TEST(ScaledWidenF32BF8PackedW32, Arithmetic) {
  check<32, 2, 0, true>(rj_arith_v_cvt_scalef32_pk_f32_bf8_w32_gfx950);
}

// Check scaled BF8 to F32 pairs.
TEST(ScaledWidenF32BF8PackedW64, Arithmetic) {
  check<64, 2, 0, true>(rj_arith_v_cvt_scalef32_pk_f32_bf8_w64_gfx950);
}

// Check scaled BF8 to F32 values.
TEST(ScaledWidenF32BF8SingleW32, Arithmetic) {
  check<32, 2, 0, false>(rj_arith_v_cvt_scalef32_f32_bf8_w32_gfx950);
}

// Check scaled BF8 to F32 values.
TEST(ScaledWidenF32BF8SingleW64, Arithmetic) {
  check<64, 2, 0, false>(rj_arith_v_cvt_scalef32_f32_bf8_w64_gfx950);
}

// Check scaled FP4 to F32 pairs.
TEST(ScaledWidenF32FP4PackedW32, Arithmetic) {
  check<32, 0, 0, true>(rj_arith_v_cvt_scalef32_pk_f32_fp4_w32_gfx950);
}

// Check scaled FP4 to F32 pairs.
TEST(ScaledWidenF32FP4PackedW64, Arithmetic) {
  check<64, 0, 0, true>(rj_arith_v_cvt_scalef32_pk_f32_fp4_w64_gfx950);
}

// Check scaled FP8 to F16 pairs.
TEST(ScaledWidenF16FP8PackedW32, Arithmetic) {
  check<32, 1, 1, true>(rj_arith_v_cvt_scalef32_pk_f16_fp8_w32_gfx950);
}

// Check scaled FP8 to F16 pairs.
TEST(ScaledWidenF16FP8PackedW64, Arithmetic) {
  check<64, 1, 1, true>(rj_arith_v_cvt_scalef32_pk_f16_fp8_w64_gfx950);
}

// Check scaled FP8 to F16 values.
TEST(ScaledWidenF16FP8SingleW32, Arithmetic) {
  check<32, 1, 1, false>(rj_arith_v_cvt_scalef32_f16_fp8_w32_gfx950);
}

// Check scaled FP8 to F16 values.
TEST(ScaledWidenF16FP8SingleW64, Arithmetic) {
  check<64, 1, 1, false>(rj_arith_v_cvt_scalef32_f16_fp8_w64_gfx950);
}

// Check scaled BF8 to F16 pairs.
TEST(ScaledWidenF16BF8PackedW32, Arithmetic) {
  check<32, 2, 1, true>(rj_arith_v_cvt_scalef32_pk_f16_bf8_w32_gfx950);
}

// Check scaled BF8 to F16 pairs.
TEST(ScaledWidenF16BF8PackedW64, Arithmetic) {
  check<64, 2, 1, true>(rj_arith_v_cvt_scalef32_pk_f16_bf8_w64_gfx950);
}

// Check scaled BF8 to F16 values.
TEST(ScaledWidenF16BF8SingleW32, Arithmetic) {
  check<32, 2, 1, false>(rj_arith_v_cvt_scalef32_f16_bf8_w32_gfx950);
}

// Check scaled BF8 to F16 values.
TEST(ScaledWidenF16BF8SingleW64, Arithmetic) {
  check<64, 2, 1, false>(rj_arith_v_cvt_scalef32_f16_bf8_w64_gfx950);
}

// Check scaled FP4 to F16 pairs.
TEST(ScaledWidenF16FP4PackedW32, Arithmetic) {
  check<32, 0, 1, true>(rj_arith_v_cvt_scalef32_pk_f16_fp4_w32_gfx950);
}

// Check scaled FP4 to F16 pairs.
TEST(ScaledWidenF16FP4PackedW64, Arithmetic) {
  check<64, 0, 1, true>(rj_arith_v_cvt_scalef32_pk_f16_fp4_w64_gfx950);
}

// Check scaled FP8 to BF16 pairs.
TEST(ScaledWidenBF16FP8PackedW32, Arithmetic) {
  check<32, 1, 2, true>(rj_arith_v_cvt_scalef32_pk_bf16_fp8_w32_gfx950);
}

// Check scaled FP8 to BF16 pairs.
TEST(ScaledWidenBF16FP8PackedW64, Arithmetic) {
  check<64, 1, 2, true>(rj_arith_v_cvt_scalef32_pk_bf16_fp8_w64_gfx950);
}

// Check scaled BF8 to BF16 pairs.
TEST(ScaledWidenBF16BF8PackedW32, Arithmetic) {
  check<32, 2, 2, true>(rj_arith_v_cvt_scalef32_pk_bf16_bf8_w32_gfx950);
}

// Check scaled BF8 to BF16 pairs.
TEST(ScaledWidenBF16BF8PackedW64, Arithmetic) {
  check<64, 2, 2, true>(rj_arith_v_cvt_scalef32_pk_bf16_bf8_w64_gfx950);
}

// Check scaled FP4 to BF16 pairs.
TEST(ScaledWidenBF16FP4PackedW32, Arithmetic) {
  check<32, 0, 2, true>(rj_arith_v_cvt_scalef32_pk_bf16_fp4_w32_gfx950);
}

// Check scaled FP4 to BF16 pairs.
TEST(ScaledWidenBF16FP4PackedW64, Arithmetic) {
  check<64, 0, 2, true>(rj_arith_v_cvt_scalef32_pk_bf16_fp4_w64_gfx950);
}
