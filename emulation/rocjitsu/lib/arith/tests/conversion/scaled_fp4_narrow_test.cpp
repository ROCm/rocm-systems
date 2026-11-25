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

// Independently decode one source element before the old double scale division.
float decode(int source, uint32_t raw, uint64_t flags) {
  if (source == 0 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (raw & 0x7f800000U) == 0) {
    raw &= 0x80000000U;
  }
  uint32_t bits = source == 0   ? raw
                  : source == 1 ? uint32_t(conversion_reencode(uint16_t(raw), 10, 15, 23, 127))
                                : raw << 16;
  return std::bit_cast<float>(bits);
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
  // The inherited SR model flushes below 0.25 even with a maximal seed.
  // Place adjacent values at scale one to distinguish that cutoff from ties.
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
        int exponent = ((Stochastic || Src == 0 ? c : b) >> 23) & 255;
        if (exponent == 255) {
          return old;
        }
        double scale = std::ldexp(1.0, exponent - 127);
        uint32_t first_seed = Stochastic ? b : 0;
        uint32_t second_seed = (first_seed << 1) ^ ((first_seed >> 31) ? 197U : 0U);
        uint32_t second = Src == 0 ? (Stochastic ? uint32_t(a >> 32) : b) : uint32_t(a >> 16);
        uint32_t lo = quantize_finite<2, 1>(float(double(decode(Src, uint32_t(a), flags)) / scale),
                                            Stochastic, first_seed);
        uint32_t hi = quantize_finite<2, 1>(float(double(decode(Src, second, flags)) / scale),
                                            Stochastic, second_seed);
        int shift = 8 * int((mode >> 16) & 3);
        return (old & ~(0xffU << shift)) | ((lo | (hi << 4)) << shift);
      },
      conversion_bits, RJ_ARITH_CVT_BYTE_3, false,
      Src == 0 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
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
