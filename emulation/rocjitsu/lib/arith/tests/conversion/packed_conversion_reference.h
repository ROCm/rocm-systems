// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 conversion into two packed 16-bit results.

#pragma once

#include "conversion/conversion64_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Convert one FP32 source to the selected 16-bit packed result format.
inline uint16_t packed_conversion_half_reference(int op, uint32_t raw, uint32_t mode) {
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    raw ^= 0x80000000;
  }
  unsigned magnitude = raw & 0x7fffffff;
  bool negative = raw >> 31;
  if (op == 0) {
    uint16_t sign = negative ? 0x8000 : 0;
    if (magnitude >= 0x7f800000) {
      return uint16_t(sign | 0x7c00 |
                      (magnitude > 0x7f800000 ? ((raw & 0x7fffff) >> 13) | 0x200 : 0));
    }
    // Find the greatest positive half no larger than the input. FP32 bit
    // ordering and integer-only widening make this independent of host FP.
    unsigned low = 0, high = 0x7bff;
    while (low < high) {
      unsigned middle = (low + high + 1) / 2;
      if (conversion_reencode(middle, 10, 15, 23, 127) <= magnitude) {
        low = middle;
      } else {
        high = middle - 1;
      }
    }
    return uint16_t(sign | low);
  }
  if (magnitude > 0x7f800000 || magnitude < 0x3f800000 || (op == 2 && negative)) {
    return 0;
  }
  uint32_t limit = op == 2 ? 65535 : negative ? 32768 : 32767;
  int exponent = magnitude >> 23;
  uint32_t integer = exponent >= 143 ? limit : (0x800000 | (raw & 0x7fffff)) >> (150 - exponent);
  if (integer > limit) {
    integer = limit;
  }
  return uint16_t(negative ? 0u - integer : integer);
}

// Convert and pack two sources into one destination word.
inline uint32_t packed_conversion_reference(int op, uint32_t a, uint32_t b, uint32_t mode) {
  return packed_conversion_half_reference(op, a, mode) |
         (uint32_t(packed_conversion_half_reference(op, b, mode >> 1)) << 16);
}

} // namespace rocjitsu::arith::test
