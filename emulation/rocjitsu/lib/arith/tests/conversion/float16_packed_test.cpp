// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 conversion contracts, seeded rounding, selectors, aliases and inactive lanes.
#include "conversion/helpers.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Numerical quantization for RNE; SR uses the inherited alignment-before-increment
// convention. Special encodings match util/data_types.h, not host NaN conversion.
uint32_t narrow_reference(uint32_t raw, uint32_t seed, bool sr, bool saturate) {
  uint32_t magnitude = raw & 0x7fffffff, sign = (raw >> 16) & 0x8000;
  if (magnitude >= 0x7f800000) {
    uint32_t payload = (raw & 0x7fffff) >> 13;
    return sign | (magnitude == 0x7f800000 ? 0x7c00
                   : sr                    ? 0x7fff
                                           : 0x7c00 | (payload ? payload : 1));
  }
  if (!magnitude) {
    return sign;
  }
  int exponent;
  double fraction = std::frexp(double(std::bit_cast<float>(magnitude)), &exponent) * 2;
  --exponent;
  uint32_t encoded;
  if (exponent > 15) {
    encoded = 0x7c00;
  } else if (sr) {
    double mantissa = (fraction - 1) * 0x800000;
    if (exponent < -14) {
      mantissa = std::floor(std::ldexp(fraction, exponent + 37));
    }
    auto rounded = uint32_t(mantissa + (seed >> 19));
    encoded = ((exponent < -14 ? 0 : uint32_t(exponent + 15)) << 10) + (rounded >> 13);
  } else {
    double scaled = std::ldexp(fraction, exponent < -14 ? exponent + 24 : 10);
    auto rounded = uint32_t(std::nearbyint(scaled));
    encoded = exponent < -14 ? rounded : (uint32_t(exponent + 14) << 10) + rounded;
  }
  if (encoded >= 0x7c00) {
    encoded = saturate ? 0x7bff : 0x7c00;
  }
  return sign | encoded;
}

// Exercise one inherited FP16 instruction using shared deterministic operands.
template <int Lanes, int Op, typename Call> void check(Call call) {
  auto input = conversion_inputs<Lanes>(32);
  const uint32_t edges[] = {0x33000000U, 0x33000001U, 0x387fc000U, 0x38800000U,
                            0x3f801000U, 0x3f803000U, 0x477ff000U, 0xc77ff000U,
                            0x7f800001U, 0x7fc01234U, 0xff800001U, 0x7f800000U,
                            0xff800000U, 0x80000000U, 0xb5800000U, 0x3f800000U};
  for (int lane = 0; lane < 16; ++lane) {
    input.v[0][lane] = edges[lane];
  }
  if constexpr (Op == 2 || Op == 3) {
    for (int lane = 0; lane < Lanes; ++lane) {
      constexpr uint32_t seeds[] = {0, ~0U, 0x00001000U, 0x80008000U};
      input.v[Op == 2 ? 2 : 4][lane] = seeds[lane % 4];
    }
  }
  uint64_t mode = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                  RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                  RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_MASK | (Op == 2 ? RJ_ARITH_ALU_HIGH_D : 0);
  check_conversion<Lanes, Op == 3 ? 3 : 2, 1, 1, true>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t c, uint32_t old, uint64_t flags, uint32_t mode) {
        if constexpr (Op == 1 || Op == 2) {
          if (mode & RJ_ARITH_ALU_ABS_A) {
            a &= 0x7fffffffU;
          }
          if (mode & RJ_ARITH_ALU_NEG_A) {
            a ^= 0x80000000U;
          }
          if constexpr (Op == 1) {
            if (mode & RJ_ARITH_ALU_ABS_B) {
              b &= 0x7fffffffU;
            }
            if (mode & RJ_ARITH_ALU_NEG_B) {
              b ^= 0x80000000U;
            }
          }
        }
        bool saturate = flags & RJ_ARITH_FP16_OVFL;
        if constexpr (Op == 2) {
          int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
          return (old & ~(0xffffU << shift)) |
                 (narrow_reference(uint32_t(a), b, true, saturate) << shift);
        }
        if constexpr (Op == 3) {
          return narrow_reference(uint32_t(a), c << 16, true, saturate) |
                 (narrow_reference(b, c & 0xffff0000u, true, saturate) << 16);
        }
        auto rounded = [&](uint32_t raw) {
          const uint32_t half = narrow_reference(raw, 0, false, saturate);
          if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (half & 0x7c00u) == 0) {
            return half & 0x8000u;
          }
          return half;
        };
        return rounded(uint32_t(a)) | (rounded(b) << 16);
      },
      conversion_bits, mode, Op == 2, Op == 1 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0);
}
} // namespace

TEST(ConversionF16PackW32, Arithmetic) { check<32, 1>(rj_arith_v_cvt_pk_f16_f32_w32); }

TEST(ConversionF16PackW64, Arithmetic) { check<64, 1>(rj_arith_v_cvt_pk_f16_f32_w64); }

TEST(ConversionF16StochasticW32, Arithmetic) { check<32, 2>(rj_arith_v_cvt_sr_f16_f32_w32); }

TEST(ConversionF16StochasticW64, Arithmetic) { check<64, 2>(rj_arith_v_cvt_sr_f16_f32_w64); }

TEST(ConversionF16StochasticPackW32, Arithmetic) { check<32, 3>(rj_arith_v_cvt_sr_pk_f16_f32_w32); }

TEST(ConversionF16StochasticPackW64, Arithmetic) { check<64, 3>(rj_arith_v_cvt_sr_pk_f16_f32_w64); }
