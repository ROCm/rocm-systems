// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar gfx12 FP16 FMA with default guest controls and optional finite saturation.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/fma_integer.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Round finite FP64 to nearest-even FP16, optionally saturating finite overflow.
inline uint16_t half_fma_narrow(uint64_t bits, bool saturate) {
  // Adapted from rocjitsu mixed_fma_simd.h: round_finite_f16_simd.
  int exponent = (bits >> 52) & 2047;
  uint16_t sign = uint16_t((bits >> 48) & 0x8000);
  if (exponent < 998) {
    return sign;
  }
  if (exponent > 1038) {
    return sign | (saturate ? 0x7bff : 0x7c00);
  }
  int shift = exponent < 1009 ? 1051 - exponent : 42;
  uint64_t significand = (bits & 0x000fffffffffffffULL) | (1ULL << 52);
  uint64_t rounded = significand >> shift;
  if (exponent >= 1009) {
    rounded += (exponent - 1009) << 10;
  }
  uint64_t remainder = significand & ((1ULL << shift) - 1);
  uint64_t halfway = 1ULL << (shift - 1);
  rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
  if (rounded >= 0x7c00) {
    rounded = saturate ? 0x7bff : 0x7c00;
  }
  return sign | uint16_t(rounded);
}

// Adapted from rocjitsu shared/fp_mode.h: fma_f16 and finish_fma_f16.
// Default guest rounding and denormal policy; preserve architecture-specific NaN results.
inline uint16_t half_fma_value(uint16_t a, uint16_t b, uint16_t c, bool saturate) {
  uint16_t ma = a & 0x7fff, mb = b & 0x7fff, mc = c & 0x7fff;

  uint16_t exceptional = 0;
  if ((ma == 0 && mb == 0x7c00) || (mb == 0 && ma == 0x7c00)) {
    exceptional = 0xfe00;
  } else if (ma > 0x7c00 || mb > 0x7c00 || mc > 0x7c00) {
    exceptional = (ma > 0x7c00 ? a : mb > 0x7c00 ? b : c) | 0x0200;
  } else if (ma == 0x7c00 || mb == 0x7c00) {
    exceptional = ((a ^ b) & 0x8000) | 0x7c00;
    if (mc == 0x7c00 && ((exceptional ^ c) & 0x8000)) {
      exceptional = 0xfe00;
    }
  } else if (mc == 0x7c00) {
    exceptional = c;
  }
  if (exceptional) {
    return exceptional;
  }

  uint64_t value = fma_finite_round_odd(as_bits(f16_to_float(a)), as_bits(f16_to_float(b)),
                                        as_bits(f16_to_float(c)));
  uint16_t result = half_fma_narrow(value, saturate);

  return result;
}

} // namespace rocjitsu::arith
