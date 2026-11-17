// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local scalar helpers for fp8 narrow.
#pragma once

#include "common/isa.h"
#include "conversion/fp8_narrow.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Convert FP32 bits to the selected FP8/BF8 format using nearest-even or the
// supplied stochastic seed. The policy specifies NaNs, tiny values and whether
// saturation includes infinities.
// Uses integer operations and preserves host floating-point state.
// Rounding logic adapted from rocjitsu util/data_types.h, with RDNA4 NaN,
// infinity and subnormal stochastic behavior established by GPU captures.
// CDNA policies retain the existing util models, including their tiny-value cutoff.
template <bool Bf8, bool Stochastic>
inline uint32_t narrow_fp8(uint32_t raw, uint32_t seed, bool saturate,
                           Fp8NarrowPolicy policy = Fp8NarrowPolicy::Gfx12) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  bool fnuz = policy == Fp8NarrowPolicy::Fnuz;
  bool unsigned_format = policy == Fp8NarrowPolicy::E5M3;
  bool rdna = policy == Fp8NarrowPolicy::Gfx12;
  unsigned terminal = fnuz ? 128 : unsigned_format ? 255 : Bf8 ? 124 : 127;
  uint32_t magnitude = raw & 0x7fffffff;
  uint32_t sign = unsigned_format ? 0 : (raw >> 24) & 128;
  if (magnitude > 0x7f800000) {
    return fnuz ? 0x80 : unsigned_format ? 0xff : rdna ? (Bf8 ? 0xfe : 0xff) : sign | 0x7f;
  }
  unsigned limit = terminal - (saturate && (!rdna || magnitude != 0x7f800000));
  int bias = unsigned_format ? 15 : (Bf8 ? 15 : 7) + int(fnuz);
  int exponent = int(magnitude >> 23) - 127 + bias;
  uint32_t result;
  // Overflow is decided after rounding near the boundary. Large exponents and
  // infinities are already beyond it, without needing oversized integer shifts.
  if (exponent >= (unsigned_format || (fnuz && Bf8) ? 32 : Bf8 ? 31 : 16)) {
    result = limit;
  } else {
    uint32_t significand = (magnitude & 0x7fffff) | 0x800000;
    unsigned alignment = exponent < 1 ? unsigned(1 - exponent) : 0;
    unsigned shift = 23 - fraction_bits + alignment;
    if constexpr (Stochastic) {
      if (rdna) {
        // gfx12 truncates alignment bits before adding a fixed-width seed.
        significand = alignment < 32 ? significand >> alignment : 0;
        result = (significand + (seed >> (9 + fraction_bits))) >> (23 - fraction_bits);
      } else {
        // Inherited CDNA rounding retains discarded bits and scales the seed
        // with the subnormal shift. Values below that model's cutoff become zero.
        result = shift > 24 ? 0 : (significand + (seed >> (32 - shift))) >> shift;
      }
    } else {
      result =
          shift > 24
              ? 0
              : (significand + (1U << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
    }
    if (exponent > 0) {
      result += unsigned(exponent - 1) << fraction_bits;
    }
    result = result < limit ? result : limit;
  }
  if (fnuz) {
    if (result >= terminal) {
      return 0x80;
    }
    if (!result) {
      return 0;
    }
  }
  return sign | result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
