// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Signed and unsigned normalized 16-bit conversions.

#pragma once

#include "conversion/conversion64_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Convert one selected source value to a normalized 16-bit encoding.
inline uint16_t normalized_half_reference(int op, uint32_t raw, uint32_t mode) {
  if (op >= 2) {
    raw = uint32_t(conversion_reencode(uint16_t(raw >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)), 10,
                                       15, 23, 127));
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    raw ^= 0x80000000;
  }
  unsigned magnitude = raw & 0x7fffffff;
  bool negative = raw >> 31;
  if (magnitude > 0x7f800000 || ((op & 1) && negative)) {
    return 0;
  }
  uint32_t scale = op & 1 ? 65535 : 32767;
  uint32_t result;
  if (magnitude >= 0x3f800000) {
    result = scale;
  } else {
    int exponent = magnitude >> 23;
    uint64_t significand = (magnitude & 0x7fffff) | (exponent ? 0x800000 : 0);
    // Multiply the exact integer significand, then perform a single tie-even
    // right shift. No host FP multiplication or intermediate rounding.
    result =
        uint32_t(conversion_round_right(significand * scale, exponent ? 150 - int(exponent) : 149));
  }
  return uint16_t(negative ? 0u - result : result);
}

// Build the normalized conversion result, preserving any unselected destination half.
inline uint32_t normalized_reference(int op, uint32_t a, uint32_t b, uint32_t original,
                                     uint32_t mode) {
  uint32_t low = normalized_half_reference(op, a, mode);
  if (op < 4) {
    return low | (uint32_t(normalized_half_reference(op, b, mode >> 1)) << 16);
  }
  int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  return (original & ~(65535u << shift)) | (low << shift);
}

} // namespace rocjitsu::arith::test
