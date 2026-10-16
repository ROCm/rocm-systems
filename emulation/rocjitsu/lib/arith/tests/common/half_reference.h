// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Test-side FP16 encoding, decoding, and NaN canonicalization.

#pragma once

#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Replace a NaN in the selected FP16 destination half with positive quiet NaN.
// Preserve the other half and all non-NaN encodings, including signed zeros.
inline uint32_t canonical_half_nan(uint32_t word, uint32_t mode) {
  unsigned shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  if (((word >> shift) & 0x7fff) > 0x7c00) {
    return (word & ~(65535U << shift)) | (0x7e00U << shift);
  }
  return word;
}

// Numeric half reference independent of the implementation's bit conversions.
inline double half_value(uint16_t bits) {
  int exponent = (bits >> 10) & 31, fraction = bits & 1023;
  double magnitude = exponent == 31  ? (fraction ? std::numeric_limits<double>::quiet_NaN()
                                                 : std::numeric_limits<double>::infinity())
                     : exponent == 0 ? std::ldexp(double(fraction), -24)
                                     : std::ldexp(double(1024 + fraction), exponent - 25);
  return bits & 0x8000 ? -magnitude : magnitude;
}

// Encode a double as nearest-even FP16, with optional finite-overflow saturation.
inline uint16_t half_bits(double value, bool saturate = false) {
  uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  if (std::isnan(value)) {
    return 0x7e00;
  }
  if (std::isinf(value)) {
    return sign | 0x7c00;
  }
  double magnitude = std::abs(value);
  if (magnitude >= 65520) {
    return sign | (saturate ? 0x7bff : 0x7c00);
  }
  if (magnitude == 0) {
    return sign;
  }
  int exponent;
  std::frexp(magnitude, &exponent);
  --exponent;
  double scaled = std::ldexp(magnitude, -std::max(exponent - 10, -24));
  unsigned rounded = unsigned(std::floor(scaled));
  double tail = scaled - rounded;
  rounded += tail > 0.5 || (tail == 0.5 && (rounded & 1));
  if (exponent < -14) {
    return sign | uint16_t(rounded);
  }
  return sign | uint16_t(((exponent + 14) << 10) + rounded);
}

} // namespace rocjitsu::arith::test
