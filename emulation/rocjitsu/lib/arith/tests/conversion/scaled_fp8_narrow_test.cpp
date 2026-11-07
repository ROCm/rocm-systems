// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx950 scaled narrowing: exponent extremes, seeded rounding, selectors,
// inactive lanes and aliases, with one shared reference for every source type.

#include "common/conversion64_reference.h"
#include "conversion/fp8_narrow_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Widen one input using an independent bit encoder before inherited scaling.
float decode(int format, uint32_t raw, uint64_t flags) {
  if (format == 0 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (raw & 0x7f800000U) == 0) {
    raw &= 0x80000000U;
  }
  uint32_t bits = format == 0   ? raw
                  : format == 1 ? uint32_t(conversion_reencode(uint16_t(raw), 10, 15, 23, 127))
                                : raw << 16;
  return std::bit_cast<float>(bits);
}

// Exercise API contracts and layouts with nonuniform inputs and extreme scale
// exponents. Double division and numeric quantization are independent of the implementation.
template <int Lanes, int Src, bool Bf8, bool Stochastic, typename Call> void check(Call call) {
  auto input = conversion_inputs<Lanes>(Src == 0 ? 32 : 16);
  constexpr int scale_reg = Stochastic || Src == 0 ? 4 : 2;
  const int exponents[] = {0, 1, 10, 110, 126, 127, 128, 140, 244, 253, 254, 255};
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[scale_reg][lane] =
        (uint32_t(exponents[lane % 12]) << 23) | (input.v[scale_reg][lane] & 0x807fffffU);
    if constexpr (Stochastic) {
      if (lane < 8) {
        input.v[2][lane] = lane & 1 ? ~0U : 0;
      }
    }
  }
  if constexpr (Src == 0) {
    input.v[0][1] = 0x00400000U;
    input.v[scale_reg][1] = 0;
  }
  check_conversion<Lanes, Stochastic || Src == 0 ? 3 : 2, 1, 1, false>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t c, uint32_t old, uint64_t flags, uint32_t mode) {
        int exponent = ((Stochastic || Src == 0 ? c : b) >> 23) & 255;
        if (exponent == 255) {
          return old;
        }
        double scale = std::ldexp(1.0, exponent - 127);
        auto narrow = [&](uint32_t raw) {
          float scaled = float(double(decode(Src, raw, flags)) / scale);
          return fp8_narrow_reference(Bf8, Stochastic, std::bit_cast<uint32_t>(scaled),
                                      Stochastic ? b : 0, flags & RJ_ARITH_FP16_OVFL, 950);
        };
        uint32_t value = narrow(uint32_t(a));
        if constexpr (!Stochastic) {
          value |= narrow(Src == 0 ? b : uint32_t(a >> 16)) << 8;
        }
        int shift = Stochastic ? 8 * int((mode >> 16) & 3) : mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
        uint32_t selected = (Stochastic ? 0xffU : 0xffffU) << shift;
        return (old & ~selected) | (value << shift);
      },
      conversion_bits, Stochastic ? RJ_ARITH_CVT_BYTE_3 : RJ_ARITH_ALU_HIGH_D, false,
      Src == 0 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
}
} // namespace

// Check scaled F32 narrowing and partial-destination preservation.
TEST(ScaledNarrowF32FP8PKW32, Arithmetic) {
  check<32, 0, false, false>(rj_arith_v_cvt_scalef32_pk_fp8_f32_w32_gfx950);
}

// Check scaled F32 narrowing and partial-destination preservation.
TEST(ScaledNarrowF32FP8PKW64, Arithmetic) {
  check<64, 0, false, false>(rj_arith_v_cvt_scalef32_pk_fp8_f32_w64_gfx950);
}

// Check scaled F32 narrowing and partial-destination preservation.
TEST(ScaledNarrowF32FP8SRW32, Arithmetic) {
  check<32, 0, false, true>(rj_arith_v_cvt_scalef32_sr_fp8_f32_w32_gfx950);
}

// Check scaled F32 narrowing and partial-destination preservation.
TEST(ScaledNarrowF32FP8SRW64, Arithmetic) {
  check<64, 0, false, true>(rj_arith_v_cvt_scalef32_sr_fp8_f32_w64_gfx950);
}

// Check scaled F32 narrowing and partial-destination preservation.
TEST(ScaledNarrowF32BF8PKW32, Arithmetic) {
  check<32, 0, true, false>(rj_arith_v_cvt_scalef32_pk_bf8_f32_w32_gfx950);
}

// Check scaled F32 narrowing and partial-destination preservation.
TEST(ScaledNarrowF32BF8PKW64, Arithmetic) {
  check<64, 0, true, false>(rj_arith_v_cvt_scalef32_pk_bf8_f32_w64_gfx950);
}

// Check scaled F32 narrowing and partial-destination preservation.
TEST(ScaledNarrowF32BF8SRW32, Arithmetic) {
  check<32, 0, true, true>(rj_arith_v_cvt_scalef32_sr_bf8_f32_w32_gfx950);
}

// Check scaled F32 narrowing and partial-destination preservation.
TEST(ScaledNarrowF32BF8SRW64, Arithmetic) {
  check<64, 0, true, true>(rj_arith_v_cvt_scalef32_sr_bf8_f32_w64_gfx950);
}

// Check scaled F16 narrowing and partial-destination preservation.
TEST(ScaledNarrowF16FP8PKW32, Arithmetic) {
  check<32, 1, false, false>(rj_arith_v_cvt_scalef32_pk_fp8_f16_w32_gfx950);
}

// Check scaled F16 narrowing and partial-destination preservation.
TEST(ScaledNarrowF16FP8PKW64, Arithmetic) {
  check<64, 1, false, false>(rj_arith_v_cvt_scalef32_pk_fp8_f16_w64_gfx950);
}

// Check scaled F16 narrowing and partial-destination preservation.
TEST(ScaledNarrowF16FP8SRW32, Arithmetic) {
  check<32, 1, false, true>(rj_arith_v_cvt_scalef32_sr_fp8_f16_w32_gfx950);
}

// Check scaled F16 narrowing and partial-destination preservation.
TEST(ScaledNarrowF16FP8SRW64, Arithmetic) {
  check<64, 1, false, true>(rj_arith_v_cvt_scalef32_sr_fp8_f16_w64_gfx950);
}

// Check scaled F16 narrowing and partial-destination preservation.
TEST(ScaledNarrowF16BF8PKW32, Arithmetic) {
  check<32, 1, true, false>(rj_arith_v_cvt_scalef32_pk_bf8_f16_w32_gfx950);
}

// Check scaled F16 narrowing and partial-destination preservation.
TEST(ScaledNarrowF16BF8PKW64, Arithmetic) {
  check<64, 1, true, false>(rj_arith_v_cvt_scalef32_pk_bf8_f16_w64_gfx950);
}

// Check scaled F16 narrowing and partial-destination preservation.
TEST(ScaledNarrowF16BF8SRW32, Arithmetic) {
  check<32, 1, true, true>(rj_arith_v_cvt_scalef32_sr_bf8_f16_w32_gfx950);
}

// Check scaled F16 narrowing and partial-destination preservation.
TEST(ScaledNarrowF16BF8SRW64, Arithmetic) {
  check<64, 1, true, true>(rj_arith_v_cvt_scalef32_sr_bf8_f16_w64_gfx950);
}

// Check scaled BF16 narrowing and partial-destination preservation.
TEST(ScaledNarrowBF16FP8PKW32, Arithmetic) {
  check<32, 2, false, false>(rj_arith_v_cvt_scalef32_pk_fp8_bf16_w32_gfx950);
}

// Check scaled BF16 narrowing and partial-destination preservation.
TEST(ScaledNarrowBF16FP8PKW64, Arithmetic) {
  check<64, 2, false, false>(rj_arith_v_cvt_scalef32_pk_fp8_bf16_w64_gfx950);
}

// Check scaled BF16 narrowing and partial-destination preservation.
TEST(ScaledNarrowBF16FP8SRW32, Arithmetic) {
  check<32, 2, false, true>(rj_arith_v_cvt_scalef32_sr_fp8_bf16_w32_gfx950);
}

// Check scaled BF16 narrowing and partial-destination preservation.
TEST(ScaledNarrowBF16FP8SRW64, Arithmetic) {
  check<64, 2, false, true>(rj_arith_v_cvt_scalef32_sr_fp8_bf16_w64_gfx950);
}

// Check scaled BF16 narrowing and partial-destination preservation.
TEST(ScaledNarrowBF16BF8PKW32, Arithmetic) {
  check<32, 2, true, false>(rj_arith_v_cvt_scalef32_pk_bf8_bf16_w32_gfx950);
}

// Check scaled BF16 narrowing and partial-destination preservation.
TEST(ScaledNarrowBF16BF8PKW64, Arithmetic) {
  check<64, 2, true, false>(rj_arith_v_cvt_scalef32_pk_bf8_bf16_w64_gfx950);
}

// Check scaled BF16 narrowing and partial-destination preservation.
TEST(ScaledNarrowBF16BF8SRW32, Arithmetic) {
  check<32, 2, true, true>(rj_arith_v_cvt_scalef32_sr_bf8_bf16_w32_gfx950);
}

// Check scaled BF16 narrowing and partial-destination preservation.
TEST(ScaledNarrowBF16BF8SRW64, Arithmetic) {
  check<64, 2, true, true>(rj_arith_v_cvt_scalef32_sr_bf8_bf16_w64_gfx950);
}
