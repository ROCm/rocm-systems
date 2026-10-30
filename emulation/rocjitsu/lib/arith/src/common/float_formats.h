// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar FP16/BF16 and FP8/BF8 encoding conversions.
// Kept independent of util conversions: GPU rounding and special-value policies
// can differ by architecture, and CPU SIMD implementations share these contracts.
// Internal helpers; not part of the public C API.

#ifndef RJ_ARITH_FLOAT_FORMATS_H_
#define RJ_ARITH_FLOAT_FORMATS_H_

#include "common/bits.h"
#include "common/internal.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Widen a BF16 encoding to FP32, retaining its sign, payload, and subnormal bits.
inline float bf16_to_float(uint16_t bits) { return as_float(uint32_t(bits) << 16); }

// Widen an FP16 encoding to FP32, including zeros, subnormals, infinities, and NaNs.
inline float f16_to_float(uint16_t bits) {
  uint32_t sign = uint32_t(bits & 0x8000) << 16;
  int exponent = (bits >> 10) & 31;
  uint32_t fraction = bits & 1023;
  if (exponent == 31) {
    return as_float(sign | 0x7f800000 | (fraction << 13));
  }
  if (exponent == 0) {
    if (!fraction) {
      return as_float(sign);
    }
    exponent = 1;
    while (!(fraction & 1024)) {
      fraction <<= 1;
      --exponent;
    }
    fraction &= 1023;
  }
  return as_float(sign | (uint32_t(exponent + 112) << 23) | (fraction << 13));
}

// Nearest-even FP32-to-FP16 conversion, adapted from rocjitsu util/data_types.h.
// Retains NaN payload bits; finite overflow can saturate to the largest value.
inline uint16_t float_to_f16(float value, bool saturate = false) {
  uint32_t bits = as_bits(value), sign = (bits >> 16) & 0x8000;
  uint32_t exponent = (bits >> 23) & 255, fraction = bits & 0x7fffff;
  if (exponent == 255) {
    uint32_t payload = fraction >> 13;
    return uint16_t(sign | 0x7c00 | (fraction ? (payload ? payload : 1) : 0));
  }
  int adjusted = int(exponent) - 112;
  if (adjusted <= 0) {
    if (adjusted < -10) {
      return uint16_t(sign);
    }
    uint32_t significand = fraction | 0x800000;
    int shift = 14 - adjusted;
    uint32_t rounded = significand >> shift;
    uint32_t tail = significand & ((1u << shift) - 1), half = 1u << (shift - 1);
    rounded += tail > half || (tail == half && (rounded & 1));
    return uint16_t(sign | rounded);
  }
  if (adjusted >= 31) {
    return uint16_t(sign | (saturate ? 0x7bff : 0x7c00));
  }
  uint32_t rounded = fraction >> 13, tail = fraction & 8191;
  rounded += tail > 4096 || (tail == 4096 && (rounded & 1));
  if (rounded == 1024) {
    rounded = 0;
    ++adjusted;
  }
  if (adjusted >= 31) {
    return uint16_t(sign | (saturate ? 0x7bff : 0x7c00));
  }
  return uint16_t(sign | (uint32_t(adjusted) << 10) | rounded);
}

// OCP FP8 E4M3FN and BF8 E5M2, adapted from rocjitsu util/data_types.h.
// E4M3FN's top exponent remains finite except codes 0x7f/0xff; E5M2 has
// infinities. Neither uses the FNUZ encoding. All finite values widen exactly.
template <bool Bf8> inline float fp8_to_float(uint8_t bits) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  constexpr int bias = Bf8 ? 15 : 7;
  constexpr int max_exponent = Bf8 ? 31 : 15;
  uint32_t sign = uint32_t(bits & 0x80) << 24;
  int exponent = (bits & 0x7f) >> fraction_bits;
  uint32_t fraction = bits & ((1u << fraction_bits) - 1);
  if (exponent == max_exponent) {
    if ((Bf8 && fraction) || (!Bf8 && fraction == 7)) {
      return as_float(sign | 0x7fc00000);
    }
    if constexpr (Bf8) {
      return as_float(sign | 0x7f800000);
    }
  }
  if (!exponent) {
    if (!fraction) {
      return as_float(sign);
    }
    exponent = 1;
    while (!(fraction & (1u << fraction_bits))) {
      fraction <<= 1;
      --exponent;
    }
    fraction &= (1u << fraction_bits) - 1;
  }
  return as_float(sign | (uint32_t(exponent + 127 - bias) << 23) |
                  (fraction << (23 - fraction_bits)));
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE

#endif
