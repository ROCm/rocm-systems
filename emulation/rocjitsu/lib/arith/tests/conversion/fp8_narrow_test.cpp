// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP16/FP32 narrowing to FP8/BF8, including stochastic rounding.

#include "conversion/fp8_narrow_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Check FP8/BF8 narrowing, packing, and destination preservation for the supplied entry point.
template <int Lanes, bool Stochastic, bool Bf8, int Arch = 12, bool Half = false, typename Call>
void check_fp8_narrow(Call call) {
  auto input = conversion_inputs<Lanes>(Half ? 16 : 32);
  // Midpoints and neighbors exercise discarded mantissa bits; seed extremes
  // distinguish truncation-before-seed-addition from ordinary stochastic rounding.
  const uint32_t edges[] = {0x3f880000U, 0x3f87ffffU, 0x3f880001U, 0x3f980000U,
                            0x3fa00000U, 0x3f9fffffU, 0x3fa00001U, 0x3b000001U,
                            0x37800001U, 0x43e80000U, 0x47700000U};
  for (int lane = 0; lane < 11; ++lane) {
    if constexpr (!Half) {
      input.v[0][lane + 16] = edges[lane];
    }
  }
  if constexpr (Half) {
    const uint32_t half_edges[] = {0x3c403c40U, 0x3c3f3c41U, 0x7c007bffU, 0x00010001U, 0xfc017c01U};
    for (int lane = 0; lane < 5; ++lane) {
      input.v[0][lane + 16] = half_edges[lane];
    }
  }
  if constexpr (Stochastic) {
    for (int lane = 0; lane < 16; ++lane) {
      input.v[2][lane] = lane % 2 ? UINT32_MAX : 0;
    }
  }
  uint64_t modifiers = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A |
                       (Stochastic ? RJ_ARITH_CVT_BYTE_3
                                   : RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_HIGH_D);
  if constexpr (Arch != 12) {
    modifiers |=
        RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B;
  }
  if constexpr (Half) {
    modifiers |= RJ_ARITH_ALU_HIGH_A;
  }
  check_conversion<Lanes, Half && !Stochastic ? 1 : 2, 1, 1, true>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t, uint32_t old, uint64_t flags, uint32_t mode) {
        if constexpr (Half) {
          if constexpr (!Stochastic) {
            b = std::bit_cast<uint32_t>(f16_value(uint16_t(a >> 16)));
            mode = (mode & ~(RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B)) |
                   ((mode & RJ_ARITH_ALU_ABS_A) ? RJ_ARITH_ALU_ABS_B : 0) |
                   ((mode & RJ_ARITH_ALU_NEG_A) ? RJ_ARITH_ALU_NEG_B : 0);
          }
          int shift = Stochastic && (mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0;
          a = std::bit_cast<uint32_t>(f16_value(uint16_t(a >> shift)));
        }
        return fp8_narrow_result((Stochastic ? 2 : 0) + (Bf8 ? 1 : 0), uint32_t(a), b, old, mode,
                                 !Half && (flags & RJ_ARITH_FP16_OVFL), Arch, Half);
      },
      conversion_bits, modifiers, true);
}

} // namespace

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtPkFp8F32W32, Arithmetic) {
  check_fp8_narrow<32, false, false>(rj_arith_v_cvt_pk_fp8_f32_w32);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtPkBf8F32W32, Arithmetic) {
  check_fp8_narrow<32, false, true>(rj_arith_v_cvt_pk_bf8_f32_w32);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtSrFp8F32W32, Arithmetic) {
  check_fp8_narrow<32, true, false>(rj_arith_v_cvt_sr_fp8_f32_w32);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtSrBf8F32W32, Arithmetic) {
  check_fp8_narrow<32, true, true>(rj_arith_v_cvt_sr_bf8_f32_w32);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtPkFp8F32W64, Arithmetic) {
  check_fp8_narrow<64, false, false>(rj_arith_v_cvt_pk_fp8_f32_w64);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtPkBf8F32W64, Arithmetic) {
  check_fp8_narrow<64, false, true>(rj_arith_v_cvt_pk_bf8_f32_w64);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtSrFp8F32W64, Arithmetic) {
  check_fp8_narrow<64, true, false>(rj_arith_v_cvt_sr_fp8_f32_w64);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtSrBf8F32W64, Arithmetic) {
  check_fp8_narrow<64, true, true>(rj_arith_v_cvt_sr_bf8_f32_w64);
}

TEST(ConversionCvtPkFp8F32W32Gfx942, Arithmetic) {
  check_fp8_narrow<32, false, false, 942>(rj_arith_v_cvt_pk_fp8_f32_w32_gfx942);
}

TEST(ConversionCvtPkFp8F32W64Gfx942, Arithmetic) {
  check_fp8_narrow<64, false, false, 942>(rj_arith_v_cvt_pk_fp8_f32_w64_gfx942);
}

TEST(ConversionCvtSrFp8F32W32Gfx942, Arithmetic) {
  check_fp8_narrow<32, true, false, 942>(rj_arith_v_cvt_sr_fp8_f32_w32_gfx942);
}

TEST(ConversionCvtSrFp8F32W64Gfx942, Arithmetic) {
  check_fp8_narrow<64, true, false, 942>(rj_arith_v_cvt_sr_fp8_f32_w64_gfx942);
}

TEST(ConversionCvtPkBf8F32W32Gfx942, Arithmetic) {
  check_fp8_narrow<32, false, true, 942>(rj_arith_v_cvt_pk_bf8_f32_w32_gfx942);
}

TEST(ConversionCvtPkBf8F32W64Gfx942, Arithmetic) {
  check_fp8_narrow<64, false, true, 942>(rj_arith_v_cvt_pk_bf8_f32_w64_gfx942);
}

TEST(ConversionCvtSrBf8F32W32Gfx942, Arithmetic) {
  check_fp8_narrow<32, true, true, 942>(rj_arith_v_cvt_sr_bf8_f32_w32_gfx942);
}

TEST(ConversionCvtSrBf8F32W64Gfx942, Arithmetic) {
  check_fp8_narrow<64, true, true, 942>(rj_arith_v_cvt_sr_bf8_f32_w64_gfx942);
}

TEST(ConversionCvtPkFp8F32W32Gfx950, Arithmetic) {
  check_fp8_narrow<32, false, false, 950>(rj_arith_v_cvt_pk_fp8_f32_w32_gfx950);
}

TEST(ConversionCvtPkFp8F32W64Gfx950, Arithmetic) {
  check_fp8_narrow<64, false, false, 950>(rj_arith_v_cvt_pk_fp8_f32_w64_gfx950);
}

TEST(ConversionCvtSrFp8F32W32Gfx950, Arithmetic) {
  check_fp8_narrow<32, true, false, 950>(rj_arith_v_cvt_sr_fp8_f32_w32_gfx950);
}

TEST(ConversionCvtSrFp8F32W64Gfx950, Arithmetic) {
  check_fp8_narrow<64, true, false, 950>(rj_arith_v_cvt_sr_fp8_f32_w64_gfx950);
}

TEST(ConversionCvtPkBf8F32W32Gfx950, Arithmetic) {
  check_fp8_narrow<32, false, true, 950>(rj_arith_v_cvt_pk_bf8_f32_w32_gfx950);
}

TEST(ConversionCvtPkBf8F32W64Gfx950, Arithmetic) {
  check_fp8_narrow<64, false, true, 950>(rj_arith_v_cvt_pk_bf8_f32_w64_gfx950);
}

TEST(ConversionCvtSrBf8F32W32Gfx950, Arithmetic) {
  check_fp8_narrow<32, true, true, 950>(rj_arith_v_cvt_sr_bf8_f32_w32_gfx950);
}

TEST(ConversionCvtSrBf8F32W64Gfx950, Arithmetic) {
  check_fp8_narrow<64, true, true, 950>(rj_arith_v_cvt_sr_bf8_f32_w64_gfx950);
}

TEST(ConversionCvtPkFp8F32W32Gfx1250, Arithmetic) {
  check_fp8_narrow<32, false, false, 1250>(rj_arith_v_cvt_pk_fp8_f32_w32_gfx1250);
}

TEST(ConversionCvtPkFp8F32W64Gfx1250, Arithmetic) {
  check_fp8_narrow<64, false, false, 1250>(rj_arith_v_cvt_pk_fp8_f32_w64_gfx1250);
}

TEST(ConversionCvtSrFp8F32W32Gfx1250, Arithmetic) {
  check_fp8_narrow<32, true, false, 1250>(rj_arith_v_cvt_sr_fp8_f32_w32_gfx1250);
}

TEST(ConversionCvtSrFp8F32W64Gfx1250, Arithmetic) {
  check_fp8_narrow<64, true, false, 1250>(rj_arith_v_cvt_sr_fp8_f32_w64_gfx1250);
}

TEST(ConversionCvtPkBf8F32W32Gfx1250, Arithmetic) {
  check_fp8_narrow<32, false, true, 1250>(rj_arith_v_cvt_pk_bf8_f32_w32_gfx1250);
}

TEST(ConversionCvtPkBf8F32W64Gfx1250, Arithmetic) {
  check_fp8_narrow<64, false, true, 1250>(rj_arith_v_cvt_pk_bf8_f32_w64_gfx1250);
}

TEST(ConversionCvtSrBf8F32W32Gfx1250, Arithmetic) {
  check_fp8_narrow<32, true, true, 1250>(rj_arith_v_cvt_sr_bf8_f32_w32_gfx1250);
}

TEST(ConversionCvtSrBf8F32W64Gfx1250, Arithmetic) {
  check_fp8_narrow<64, true, true, 1250>(rj_arith_v_cvt_sr_bf8_f32_w64_gfx1250);
}

TEST(ConversionCvtPkFp8F16W32, Arithmetic) {
  check_fp8_narrow<32, false, false, 1250, true>(rj_arith_v_cvt_pk_fp8_f16_w32);
}

TEST(ConversionCvtPkFp8F16W64, Arithmetic) {
  check_fp8_narrow<64, false, false, 1250, true>(rj_arith_v_cvt_pk_fp8_f16_w64);
}

TEST(ConversionCvtSrFp8F16W32, Arithmetic) {
  check_fp8_narrow<32, true, false, 1250, true>(rj_arith_v_cvt_sr_fp8_f16_w32);
}

TEST(ConversionCvtSrFp8F16W64, Arithmetic) {
  check_fp8_narrow<64, true, false, 1250, true>(rj_arith_v_cvt_sr_fp8_f16_w64);
}

TEST(ConversionCvtPkBf8F16W32, Arithmetic) {
  check_fp8_narrow<32, false, true, 1250, true>(rj_arith_v_cvt_pk_bf8_f16_w32);
}

TEST(ConversionCvtPkBf8F16W64, Arithmetic) {
  check_fp8_narrow<64, false, true, 1250, true>(rj_arith_v_cvt_pk_bf8_f16_w64);
}

TEST(ConversionCvtSrBf8F16W32, Arithmetic) {
  check_fp8_narrow<32, true, true, 1250, true>(rj_arith_v_cvt_sr_bf8_f16_w32);
}

TEST(ConversionCvtSrBf8F16W64, Arithmetic) {
  check_fp8_narrow<64, true, true, 1250, true>(rj_arith_v_cvt_sr_bf8_f16_w64);
}
