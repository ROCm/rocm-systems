// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// BF16 conversion contracts, seeded rounding, selectors, aliases and inactive lanes.
#include "conversion/helpers.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <gtest/gtest.h>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Quantize the discarded low word independently of the production packed addition.
// Special encodings and the SR subnormal convention are inherited from util/data_types.h.
uint32_t narrow_reference(uint32_t raw, uint32_t seed, bool sr, bool saturate) {
  uint32_t top = raw >> 16, tail = raw & 0xffff, magnitude = raw & 0x7fffffff;
  if (magnitude >= 0x7f800000) {
    if (sr && magnitude > 0x7f800000) {
      return top | 0x7fff;
    }
    return top | (!sr && tail ? 1U : 0U);
  }
  if (sr) {
    if (!magnitude) {
      return top;
    }
    top += tail + (seed >> 16) >= 0x10000;
  } else {
    top += tail > 0x8000 || (tail == 0x8000 && (top & 1));
  }
  return saturate && (top & 0x7fff) == 0x7f80 ? (top & 0x8000) | 0x7f7f : top;
}

// Exercise one inherited BF16 instruction using shared deterministic operands.
template <int Lanes, int Op, typename Call> void check(Call call) {
  auto input = conversion_inputs<Lanes>(32);
  const uint32_t edges[] = {0x00000001U, 0x80000001U, 0x007fffffU, 0x807fffffU,
                            0x3f808000U, 0x3f818000U, 0x7f7fffffU, 0xff7fffffU,
                            0x7f800001U, 0x7fc01234U, 0xff800001U, 0x7f800000U,
                            0xff800000U, 0x80000000U, 0,           0x3f807fffU};
  for (int lane = 0; lane < 16; ++lane) {
    input.v[0][lane] = edges[lane];
  }
  if constexpr (Op == 2 || Op == 3) {
    for (int lane = 0; lane < Lanes; ++lane) {
      constexpr uint32_t seeds[] = {0, ~0U, 0x00001000U, 0x80000000U};
      input.v[Op == 2 ? 2 : 4][lane] = seeds[lane % 4];
    }
  }
  uint64_t mode = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                  RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                  RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_MASK |
                  (Op == 0   ? RJ_ARITH_ALU_HIGH_A
                   : Op == 2 ? RJ_ARITH_ALU_HIGH_D
                             : 0);
  check_conversion<Lanes,
                   Op == 0   ? 1
                   : Op == 3 ? 3
                             : 2,
                   1, 1, true>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t c, uint32_t old, uint64_t flags, uint32_t mode) {
        if constexpr (Op == 0) {
          return uint32_t(a >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) << 16;
        }
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
        return narrow_reference(uint32_t(a), Op == 3 ? c << 16 : 0, Op == 3, saturate) |
               (narrow_reference(b, Op == 3 ? c & 0xffff0000U : 0, Op == 3, saturate) << 16);
      },
      conversion_bits, mode, true);
}
} // namespace

TEST(ConversionBf16WidenW32, Arithmetic) { check<32, 0>(rj_arith_v_cvt_f32_bf16_w32); }

TEST(ConversionBf16WidenW64, Arithmetic) { check<64, 0>(rj_arith_v_cvt_f32_bf16_w64); }

TEST(ConversionBf16PackW32, Arithmetic) { check<32, 1>(rj_arith_v_cvt_pk_bf16_f32_w32); }

TEST(ConversionBf16PackW64, Arithmetic) { check<64, 1>(rj_arith_v_cvt_pk_bf16_f32_w64); }

TEST(ConversionBf16StochasticW32, Arithmetic) { check<32, 2>(rj_arith_v_cvt_sr_bf16_f32_w32); }

TEST(ConversionBf16StochasticW64, Arithmetic) { check<64, 2>(rj_arith_v_cvt_sr_bf16_f32_w64); }

TEST(ConversionBf16StochasticPackW32, Arithmetic) {
  check<32, 3>(rj_arith_v_cvt_sr_pk_bf16_f32_w32);
}

TEST(ConversionBf16StochasticPackW64, Arithmetic) {
  check<64, 3>(rj_arith_v_cvt_sr_pk_bf16_f32_w64);
}
