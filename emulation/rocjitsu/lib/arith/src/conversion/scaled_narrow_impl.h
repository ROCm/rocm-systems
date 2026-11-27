// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited format rounding shared by short and wide scaled conversions.
#pragma once

#include "conversion/finite_narrow.h"
#include "conversion/fp8_narrow_impl.h"
#include "conversion/scaled_narrow.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Narrow scaled FP32 bits to OCP FP8/BF8 with MI350X (gfx950) SCALEF32 rules:
// unsigned canonical NaN (0xff FP8 / 0xfe BF8), +/-Inf saturation under FP16_OVFL,
// and fixed-width stochastic rounding in the source format's significand precision.
template <bool Bf8, bool Stochastic>
inline uint32_t scaled_narrow_fp8_gfx950(uint32_t raw, uint32_t seed, bool saturate,
                                         uint32_t sig_mask) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  constexpr int bias = Bf8 ? 15 : 7;
  constexpr unsigned terminal = Bf8 ? 124 : 127;
  uint32_t magnitude = raw & 0x7fffffffU;
  uint32_t sign = (raw >> 24) & 128U;
  if (magnitude > 0x7f800000U) {
    return Bf8 ? 0xfeU : 0xffU;
  }
  unsigned limit = saturate ? (terminal - 1) : terminal;
  int exponent = int(magnitude >> 23) - 127 + bias;
  if (exponent >= (Bf8 ? 31 : 16)) {
    return sign | limit;
  }
  uint32_t significand = (magnitude & 0x7fffffU) | 0x800000U;
  unsigned alignment = exponent < 1 ? unsigned(1 - exponent) : 0;
  uint32_t result;
  if constexpr (Stochastic) {
    uint32_t aligned = alignment < 25 ? ((significand >> alignment) & sig_mask) : 0;
    result = (aligned + (seed >> (9 + fraction_bits))) >> (23 - fraction_bits);
  } else {
    unsigned shift = 23 - fraction_bits + alignment;
    result = shift > 24
                 ? 0
                 : (significand + (1U << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
  }
  if (exponent > 0) {
    result += unsigned(exponent - 1) << fraction_bits;
  }
  result = result < limit ? result : limit;
  return sign | result;
}

// Apply the selected format's rounding policy to one scaled value.
template <ScaledNarrowDestination Dst, bool Stochastic, bool Gfx950 = false>
inline uint32_t scaled_narrow_encode(uint32_t raw, uint32_t seed, bool saturate,
                                     uint32_t sig_mask = 0xffffffffU) {
  if constexpr (Dst == ScaledNarrowDestination::Fp4) {
    return narrow_finite<2, 1, Stochastic, Gfx950>(raw, seed, sig_mask);
  } else if constexpr (Dst == ScaledNarrowDestination::Fp6 || Dst == ScaledNarrowDestination::Bf6) {
    constexpr bool bf6 = Dst == ScaledNarrowDestination::Bf6;
    return narrow_finite<(bf6 ? 3 : 2), (bf6 ? 2 : 3), Stochastic, Gfx950>(raw, seed, sig_mask);
  } else if constexpr (Gfx950) {
    return scaled_narrow_fp8_gfx950<Dst == ScaledNarrowDestination::Bf8, Stochastic>(
        raw, seed, saturate, sig_mask);
  } else {
    return narrow_fp8<Dst == ScaledNarrowDestination::Bf8, Stochastic>(raw, seed, saturate,
                                                                       Fp8NarrowPolicy::Ocp);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
