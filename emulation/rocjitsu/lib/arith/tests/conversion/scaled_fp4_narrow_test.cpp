// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scaled FP4 packing: inherited tiny-value behavior, paired seeds, layouts,
// source/destination aliases and partial writes, across FP32/FP16/BF16 sources.
#include "common/conversion64_reference.h"
#include "common/finite_narrow_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Independently decode one source element before scaling.
float decode(int source, uint32_t raw, uint64_t flags) {
  bool flush = (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) != 0;
  if (source == 0) {
    return std::bit_cast<float>(flush && (raw & 0x7f800000U) == 0 ? (raw & 0x80000000U) : raw);
  }
  if (source == 1) {
    uint16_t h = uint16_t(raw);
    if (flush && (h & 0x7c00U) == 0) {
      h &= 0x8000U;
    }
    return std::bit_cast<float>(uint32_t(conversion_reencode(h, 10, 15, 23, 127)));
  }
  uint32_t b = uint32_t(uint16_t(raw)) << 16;
  return std::bit_cast<float>(flush && (b & 0x7f800000U) == 0 ? (b & 0x80000000U) : b);
}

// Exercise every API with selectors, masks, seed extremes and overlap. The
// paired FP32 stochastic form consumes two registers in its first operand.
template <int Lanes, int Src, bool Stochastic, typename Call> void check(Call call) {
  auto input = conversion_inputs<Lanes>(Src == 0 ? 32 : 16);
  constexpr int scale_reg = Stochastic || Src == 0 ? 4 : 2;
  const int exponents[] = {127, 0, 254, 255, 126, 128, 129, 125};
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[scale_reg][lane] =
        (uint32_t(exponents[lane % 8]) << 23) | (input.v[scale_reg][lane] & 0x807fffffU);
    if constexpr (Stochastic) {
      const uint32_t seeds[] = {0, ~0U, 1, 0x80000000U};
      input.v[2][lane] = seeds[lane % 4];
    }
  }
  // Place adjacent subnormal values at scale one to exercise subnormal SR and RNE ties.
  constexpr uint32_t quarter = Src == 0 ? 0x3e800000U : Src == 1 ? 0x3400U : 0x3e80U;
  input.v[0][0] = quarter - 1;
  input.v[0][8] = quarter;
  if constexpr (Stochastic) {
    input.v[2][0] = input.v[2][8] = ~0U;
  }
  if constexpr (Src == 0) {
    input.v[0][1] = 0x00400000U;
    input.v[scale_reg][1] = 0;
  }
  check_conversion<Lanes, Stochastic || Src == 0 ? 3 : 2, Stochastic && Src == 0 ? 2 : 1, 1, false>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t c, uint32_t old, uint64_t flags, uint32_t mode) {
        constexpr int src_fraction = Src == 0 ? 23 : Src == 1 ? 10 : 7;
        int exponent = ((Stochastic || Src == 0 ? c : b) >> 23) & 255;
        double scale = std::ldexp(1.0, exponent - 127);
        uint32_t first_seed = Stochastic ? b : 0;
        uint32_t second_seed = (first_seed << 1) ^ ((first_seed >> 31) ? 197U : 0U);
        uint32_t second = Src == 0 ? (Stochastic ? uint32_t(a >> 32) : b) : uint32_t(a >> 16);
        auto eval = [&](uint32_t raw_elem, uint32_t seed) {
          float decoded = decode(Src, raw_elem, flags);
          float scaled = exponent == 255
                             ? std::bit_cast<float>(
                                   (std::bit_cast<uint32_t>(decoded) & 0x80000000U) | 0x7fc00000U)
                             : float(double(decoded) / scale);
          return rocjitsu::arith::test::quantize_finite<2, 1>(scaled, Stochastic, seed, true,
                                                              src_fraction);
        };
        uint32_t lo = eval(uint32_t(a), first_seed);
        uint32_t hi = eval(second, second_seed);
        int shift = 8 * int((mode >> 16) & 3);
        return (old & ~(0xffU << shift)) | ((lo | (hi << 4)) << shift);
      },
      conversion_bits, RJ_ARITH_CVT_BYTE_3, true, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  if constexpr (Stochastic && Src == 0) {
    // Compare disjoint and upper-source-aliased destinations with identical
    // initial bits. The shared contract checks above cover the other aliases.
    for (uint64_t cpu : supported_cpu_levels()) {
      auto registers = input;
      for (int lane = 0; lane < Lanes; ++lane) {
        registers.v[6][lane] = registers.v[1][lane];
      }
      uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY;
      ASSERT_EQ(call(flags, ~WaveMask<Lanes>(0), RJ_ARITH_CVT_BYTE_2, registers.v + 6, registers.v,
                     registers.v + 2, registers.v + 4),
                RJ_ARITH_SUCCESS);
      ASSERT_EQ(call(flags, ~WaveMask<Lanes>(0), RJ_ARITH_CVT_BYTE_2, registers.v + 1, registers.v,
                     registers.v + 2, registers.v + 4),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        EXPECT_EQ(registers.v[1][lane], registers.v[6][lane]);
      }
    }
  }
}
} // namespace

// Check paired F32 conversion into one FP4 destination byte.
TEST(ScaledFp4F32RNEW32, Arithmetic) {
  check<32, 0, false>(rj_arith_v_cvt_scalef32_pk_fp4_f32_w32_gfx950);
}

// Check paired F32 conversion into one FP4 destination byte.
TEST(ScaledFp4F32RNEW64, Arithmetic) {
  check<64, 0, false>(rj_arith_v_cvt_scalef32_pk_fp4_f32_w64_gfx950);
}

// Check paired F32 conversion into one FP4 destination byte.
TEST(ScaledFp4F32SRW32, Arithmetic) {
  check<32, 0, true>(rj_arith_v_cvt_scalef32_sr_pk_fp4_f32_w32_gfx950);
}

// Check paired F32 conversion into one FP4 destination byte.
TEST(ScaledFp4F32SRW64, Arithmetic) {
  check<64, 0, true>(rj_arith_v_cvt_scalef32_sr_pk_fp4_f32_w64_gfx950);
}

// Check paired F16 conversion into one FP4 destination byte.
TEST(ScaledFp4F16RNEW32, Arithmetic) {
  check<32, 1, false>(rj_arith_v_cvt_scalef32_pk_fp4_f16_w32_gfx950);
}

// Check paired F16 conversion into one FP4 destination byte.
TEST(ScaledFp4F16RNEW64, Arithmetic) {
  check<64, 1, false>(rj_arith_v_cvt_scalef32_pk_fp4_f16_w64_gfx950);
}

// Check paired F16 conversion into one FP4 destination byte.
TEST(ScaledFp4F16SRW32, Arithmetic) {
  check<32, 1, true>(rj_arith_v_cvt_scalef32_sr_pk_fp4_f16_w32_gfx950);
}

// Check paired F16 conversion into one FP4 destination byte.
TEST(ScaledFp4F16SRW64, Arithmetic) {
  check<64, 1, true>(rj_arith_v_cvt_scalef32_sr_pk_fp4_f16_w64_gfx950);
}

// Check paired BF16 conversion into one FP4 destination byte.
TEST(ScaledFp4BF16RNEW32, Arithmetic) {
  check<32, 2, false>(rj_arith_v_cvt_scalef32_pk_fp4_bf16_w32_gfx950);
}

// Check paired BF16 conversion into one FP4 destination byte.
TEST(ScaledFp4BF16RNEW64, Arithmetic) {
  check<64, 2, false>(rj_arith_v_cvt_scalef32_pk_fp4_bf16_w64_gfx950);
}

// Check paired BF16 conversion into one FP4 destination byte.
TEST(ScaledFp4BF16SRW32, Arithmetic) {
  check<32, 2, true>(rj_arith_v_cvt_scalef32_sr_pk_fp4_bf16_w32_gfx950);
}

// Check paired BF16 conversion into one FP4 destination byte.
TEST(ScaledFp4BF16SRW64, Arithmetic) {
  check<64, 2, true>(rj_arith_v_cvt_scalef32_sr_pk_fp4_bf16_w64_gfx950);
}

// Pin MI350X (gfx950, ROCm 7.2.4) captured rows verifying exponent-only E8M0
// scale extraction and signed-max saturation on scale exponent 255.
TEST(ScaledFp4NarrowCapturedCorners, Mi350xExponentOnlyScaleAndSpecialExponent) {
  struct Row {
    uint32_t d_init, a, b, scale, expected;
  };
  // MI350X gfx950_cvt_valu.bin v_cvt_scalef32_pk_fp4_f32_b0:
  // idx=273: D=0x490d81cd, A=-0.0, B=+0.0, S=0xbf801234 (-1.0) -> 0x490d8108 (scale sign ignored).
  // idx=293: D=0x490d81cd, A=-1.0, B=+3.0, S=0xffc01234 (-NaN) -> 0x490d817f.
  // idx=321: D=0x490d81cd, A=-0.0, B=+0.0, S=0xffc01234 (-NaN) -> 0x490d817f.
  // With scale -Inf (0xff800000), (+0, -0) and (+1, -1) both pack to 0xf7.
  const Row rows[] = {
      {0x490d8100U, 0x00000000U, 0x80000000U, 0xff800000U, 0x490d81f7U},
      {0x490d8100U, 0x3f800000U, 0xbf800000U, 0xff800000U, 0x490d81f7U},
      {0x490d81cdU, 0x80000000U, 0x00000000U, 0xbf801234U, 0x490d8108U},
      {0x490d81cdU, 0xbf800000U, 0x40400000U, 0xffc01234U, 0x490d817fU},
      {0x490d81cdU, 0x80000000U, 0x00000000U, 0xffc01234U, 0x490d817fU},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t sem : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                         RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      for (const auto &row : rows) {
        uint32_t d[64]{}, a[64]{}, b[64]{}, s[64]{};
        d[0] = row.d_init;
        a[0] = row.a;
        b[0] = row.b;
        s[0] = row.scale;
        uint32_t *dp = d;
        const uint32_t *ap = a, *bp = b, *sp = s;
        ASSERT_EQ(
            rj_arith_v_cvt_scalef32_pk_fp4_f32_w64_gfx950(cpu | sem, 1ULL, 0, &dp, &ap, &bp, &sp),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(d[0], row.expected);
      }
    }
  }
}
