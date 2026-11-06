// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 and 32-bit integer conversion rules.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Return the selected FP32/integer conversion result with source and output modifiers.
inline uint32_t conversion32_reference(int op, uint32_t raw, uint32_t mode) {
  // Independent integer-only IEEE-754 encoding/decoding, including nearest-even
  // integer-to-float rounding and exact fractional comparisons on float inputs.
  if (op < 2) {
    bool negative = op == 0 && (raw >> 31);
    uint32_t magnitude = negative ? uint32_t(0) - raw : raw;
    if (!magnitude) {
      return 0;
    }
    int exponent = 0;
    for (uint32_t copy = magnitude; copy >>= 1;) {
      ++exponent;
    }
    uint32_t significand;
    if (exponent <= 23) {
      significand = magnitude << (23 - exponent);
    } else {
      int shift = exponent - 23;
      significand = magnitude >> shift;
      uint32_t remainder = magnitude & ((uint32_t(1) << shift) - 1);
      uint32_t half = uint32_t(1) << (shift - 1);
      if (remainder > half || (remainder == half && (significand & 1))) {
        ++significand;
      }
      if (significand == 0x1000000) {
        significand >>= 1;
        ++exponent;
      }
    }
    int scale[] = {0, 1, 2, -1};
    uint32_t encoded = (negative ? 0x80000000 : 0) |
                       (uint32_t(int(exponent) + 127 + scale[(mode >> 6) & 3]) << 23) |
                       (significand & 0x7fffff);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      return negative ? 0 : encoded > 0x3f800000 ? 0x3f800000 : encoded;
    }
    return encoded;
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    raw ^= 0x80000000;
  }
  bool negative = raw >> 31;
  uint32_t fraction = raw & 0x7fffff;
  int biased = (raw >> 23) & 255;
  if (biased == 255 && fraction) {
    return op >= 4 ? (negative ? 0x80000000 : 0x7fffffff) : 0;
  }
  if (op == 3 && negative) {
    return 0;
  }
  int exponent = int(biased) - 127;
  if (exponent >= 32) {
    return op == 3 ? UINT32_MAX : negative ? 0x80000000 : 0x7fffffff;
  }
  uint64_t magnitude = 0;
  bool nonzero_fraction = (raw & 0x7fffffff) != 0;
  bool above_half = false, at_half = false;
  if (exponent >= 23) {
    magnitude = uint64_t(0x800000 | fraction) << (exponent - 23);
    nonzero_fraction = false;
  } else if (exponent >= -1) {
    int shift = unsigned(23 - exponent);
    uint32_t significand = 0x800000 | fraction;
    magnitude = significand >> shift;
    uint32_t remainder = significand & ((uint32_t(1) << shift) - 1);
    uint32_t half = uint32_t(1) << (shift - 1);
    nonzero_fraction = remainder != 0;
    above_half = remainder > half;
    at_half = remainder == half;
  }
  if ((op == 4 && (above_half || (at_half && !negative))) ||
      (op == 5 && negative && nonzero_fraction)) {
    ++magnitude;
  }
  if (op == 3) {
    return magnitude > UINT32_MAX ? UINT32_MAX : uint32_t(magnitude);
  }
  uint64_t limit = negative ? 0x80000000ULL : 0x7fffffffULL;
  if (magnitude > limit) {
    magnitude = limit;
  }
  return negative ? uint32_t(0) - uint32_t(magnitude) : uint32_t(magnitude);
}

} // namespace rocjitsu::arith::test
