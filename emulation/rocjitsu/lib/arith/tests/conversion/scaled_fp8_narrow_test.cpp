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

// Widen one input using an independent bit encoder before scaling.
float decode(int format, uint32_t raw, uint64_t flags) {
  bool flush = (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) != 0;
  if (format == 0) {
    return std::bit_cast<float>(flush && (raw & 0x7f800000U) == 0 ? (raw & 0x80000000U) : raw);
  }
  if (format == 1) {
    uint16_t h = uint16_t(raw);
    if (flush && (h & 0x7c00U) == 0) {
      h &= 0x8000U;
    }
    return std::bit_cast<float>(uint32_t(conversion_reencode(h, 10, 15, 23, 127)));
  }
  uint32_t b = uint32_t(uint16_t(raw)) << 16;
  return std::bit_cast<float>(flush && (b & 0x7f800000U) == 0 ? (b & 0x80000000U) : b);
}

uint32_t scaled_fp8_ref(bool bf8, bool stochastic, int src_fraction, uint32_t raw, uint32_t seed,
                        bool saturate) {
  unsigned sign = (raw >> 24) & 128, magnitude = raw & 0x7fffffffU;
  unsigned terminal = bf8 ? 124 : 127;
  unsigned limit = terminal - saturate;
  if (magnitude > 0x7f800000U) {
    return bf8 ? 0xfeU : 0xffU;
  }
  if (magnitude == 0x7f800000U) {
    return sign | limit;
  }
  double x = std::bit_cast<float>(magnitude);
  if (x > (bf8 ? 65536.0 : 512.0)) {
    return sign | limit;
  }
  int fraction_bits = bf8 ? 2 : 3;
  int exponent;
  std::frexp(x, &exponent);
  int power = exponent - 1 - fraction_bits;
  int minimum_power = bf8 ? -16 : -9;
  if (power < minimum_power || x == 0) {
    power = minimum_power;
  }
  double units = std::ldexp(x, -power);
  unsigned rounded;
  if (stochastic) {
    int precision = src_fraction - fraction_bits;
    double fixed = std::floor(std::ldexp(units, precision));
    rounded = unsigned(std::floor(std::ldexp(fixed + (seed >> (32 - precision)), -precision)));
  } else {
    rounded = unsigned(std::floor(units));
    double fraction = units - rounded;
    rounded += fraction > 0.5 || (fraction == 0.5 && (rounded & 1));
  }
  unsigned code = rounded + (unsigned(power - minimum_power) << fraction_bits);
  return sign | (code < limit ? code : limit);
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
  constexpr int src_fraction = Src == 0 ? 23 : Src == 1 ? 10 : 7;
  check_conversion<Lanes, Stochastic || Src == 0 ? 3 : 2, 1, 1, false>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t c, uint32_t old, uint64_t flags, uint32_t mode) {
        int exponent = ((Stochastic || Src == 0 ? c : b) >> 23) & 255;
        double scale = std::ldexp(1.0, exponent - 127);
        auto narrow = [&](uint32_t raw) -> uint32_t {
          if (exponent == 255) {
            return Bf8 ? 0xfeU : 0xffU;
          }
          float scaled = float(double(decode(Src, raw, flags)) / scale);
          return scaled_fp8_ref(Bf8, Stochastic, src_fraction, std::bit_cast<uint32_t>(scaled),
                                Stochastic ? b : 0, flags & RJ_ARITH_FP16_OVFL);
        };
        uint32_t value = narrow(uint32_t(a));
        if constexpr (!Stochastic) {
          value |= narrow(Src == 0 ? b : uint32_t(a >> 16)) << 8;
        }
        int shift = Stochastic ? 8 * int((mode >> 16) & 3) : mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
        uint32_t selected = (Stochastic ? 0xffU : 0xffffU) << shift;
        return (old & ~selected) | (value << shift);
      },
      conversion_bits, Stochastic ? RJ_ARITH_CVT_BYTE_3 : RJ_ARITH_ALU_HIGH_D, true,
      RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
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

// Pin MI350X (gfx950, ROCm 7.2.4) captured rows for SR +/-Inf saturation under
// FP16_OVFL and PK scale exponent 0 (E8M0 divisor 2^-127) across strict modes.
TEST(ScaledFp8NarrowCapturedCorners, Mi350xInfOvflAndZeroScaleExponent) {
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t sem : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                         RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      // MI350X gfx950_cvt_valu.bin v_cvt_scalef32_sr_fp8_f32_b0:
      // idx=327: D=0x605fc89d, A=+Inf, seed=1, scale=0x3f000000 -> 0x7f (OVFL=0), 0x7e (OVFL=1).
      // idx=328: D=0x5a5aa5a5, A=-Inf, seed=1, scale=0x3f800000 -> 0xff (OVFL=0), 0xfe (OVFL=1).
      struct SrRow {
        uint32_t d_init, a, seed, scale, expect_ovfl0, expect_ovfl1;
      };
      const SrRow sr_rows[] = {
          {0x605fc89dU, 0x7f800000U, 0U, 0x3f800000U, 0x605fc87fU, 0x605fc87eU},
          {0x5a5aa5a5U, 0xff800000U, 0U, 0x3f800000U, 0x5a5aa5ffU, 0x5a5aa5feU},
          {0x605fc89dU, 0x7f800000U, 1U, 0x3f000000U, 0x605fc87fU, 0x605fc87eU},
          {0x5a5aa5a5U, 0xff800000U, 1U, 0x3f800000U, 0x5a5aa5ffU, 0x5a5aa5feU},
      };
      for (const auto &row : sr_rows) {
        for (bool ovfl : {false, true}) {
          uint32_t d[64]{}, a[64]{}, b[64]{}, s[64]{};
          d[0] = row.d_init;
          a[0] = row.a;
          b[0] = row.seed;
          s[0] = row.scale;
          uint32_t *dp = d;
          const uint32_t *ap = a, *bp = b, *sp = s;
          uint64_t flags = cpu | sem | (ovfl ? RJ_ARITH_FP16_OVFL : 0);
          ASSERT_EQ(
              rj_arith_v_cvt_scalef32_sr_fp8_f32_w64_gfx950(flags, 1ULL, 0, &dp, &ap, &bp, &sp),
              RJ_ARITH_SUCCESS);
          EXPECT_EQ(d[0], ovfl ? row.expect_ovfl1 : row.expect_ovfl0);
        }
      }

      // MI350X gfx950_cvt_valu.bin v_cvt_scalef32_pk_fp8_f32_lo with scale exponent 0 (2^-127):
      // (+0, +0) -> 0x0000; (2^-127, 2^-127) -> 0x3838 preserved, 0x0000 flushed;
      // idx=280: D=0x5a5aa5a5, A=-Inf, B=0x00000001, S=0x00000000 -> 0x5a5a00ff;
      // idx=309: D=0x490d81cd, A=-1.0, B=0x00008000, S=0x80000000 -> 0x490d04ff / 0x490d00ff.
      struct PkRow {
        uint32_t d_init, a, b, scale, expect_keep, expect_flush;
      };
      const PkRow pk_rows[] = {
          {0xa5a50000U, 0x00000000U, 0x00000000U, 0x00000000U, 0xa5a50000U, 0xa5a50000U},
          {0xa5a50000U, 0x00400000U, 0x00400000U, 0x00000000U, 0xa5a53838U, 0xa5a50000U},
          {0x5a5aa5a5U, 0xff800000U, 0x00000001U, 0x00000000U, 0x5a5a00ffU, 0x5a5a00ffU},
          {0x490d81cdU, 0xbf800000U, 0x00008000U, 0x80000000U, 0x490d04ffU, 0x490d00ffU},
      };
      for (const auto &row : pk_rows) {
        for (bool flush_in : {false, true}) {
          uint32_t d[64]{}, a[64]{}, b[64]{}, s[64]{};
          d[0] = row.d_init;
          a[0] = row.a;
          b[0] = row.b;
          s[0] = row.scale;
          uint32_t *dp = d;
          const uint32_t *ap = a, *bp = b, *sp = s;
          uint64_t flags = cpu | sem | (flush_in ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
          ASSERT_EQ(
              rj_arith_v_cvt_scalef32_pk_fp8_f32_w64_gfx950(flags, 1ULL, 0, &dp, &ap, &bp, &sp),
              RJ_ARITH_SUCCESS);
          EXPECT_EQ(d[0], flush_in ? row.expect_flush : row.expect_keep);
        }
      }
    }
  }
}
