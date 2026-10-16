// SPDX-License-Identifier: MIT

// Scalar FP16, BF16, FP8, and BF8 encoding conversions.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_FLOAT_FORMATS_H_
#define RJ_ARITH_FLOAT_FORMATS_H_

#include "common/internal.h"

#include <stdint.h>

namespace rocjitsu::arith {

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

} // namespace rocjitsu::arith

#endif
