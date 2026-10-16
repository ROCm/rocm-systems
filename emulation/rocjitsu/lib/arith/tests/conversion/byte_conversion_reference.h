// SPDX-License-Identifier: MIT

// Byte widening, signed nibble offsets, and byte packing.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent reference inherited from tests/conversion/byte_conversion_reference.h.
// Retain its rounding and special-value model; do not call production helpers.

// Widen the selected unsigned byte to FP32 with output scaling and clamping.
inline uint32_t byte_conversion_reference(int byte, uint32_t raw, uint32_t mode) {
  // Encode twice the mathematical result as an integer; even div:2 is exact.
  const unsigned shifts[] = {1, 2, 3, 0};
  unsigned twice = ((raw >> (8 * byte)) & 255) << shifts[(mode >> 6) & 3];
  if ((mode & RJ_ARITH_ALU_CLAMP) && twice > 2) {
    twice = 2;
  }
  if (!twice) {
    return 0;
  }
  int exponent = 0;
  for (unsigned remaining = twice; remaining >>= 1;) {
    ++exponent;
  }
  return ((126 + exponent) << 23) | ((twice << (23 - exponent)) & 0x7fffff);
}

// Compute the signed nibble-offset conversion with its modifiers.
inline uint32_t nibble_offset_reference(uint32_t raw, uint32_t mode) {
  unsigned nibble = raw & 15;
  bool negative = nibble >= 8;
  unsigned magnitude = negative ? 16 - nibble : nibble;
  const int powers[] = {-4, -3, -2, -5};
  int power = powers[(mode >> 6) & 3];
  if (!magnitude || ((mode & RJ_ARITH_ALU_CLAMP) && negative)) {
    return 0;
  }
  if ((mode & RJ_ARITH_ALU_CLAMP) && magnitude > (1u << -power)) {
    return 0x3f800000;
  }
  int top = 0;
  for (int copy = magnitude; copy >>= 1;) {
    ++top;
  }
  return (negative ? 0x80000000u : 0) | (uint32_t(127 + int(top) + power) << 23) |
         ((magnitude << (23 - top)) & 0x7fffff);
}

// Convert FP32 to the selected destination byte, preserving the other bytes.
inline uint32_t byte_pack_reference(uint32_t raw, uint32_t selection, uint32_t original,
                                    uint32_t mode) {
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    raw ^= 0x80000000;
  }
  uint32_t magnitude = raw & 0x7fffffff, byte = 0;
  if (!(raw >> 31) && magnitude <= 0x7f800000) {
    if (magnitude >= 0x437f0000) {
      byte = 255;
    } else if (magnitude >= 0x3f000000) {
      int shift = 150 - (magnitude >> 23);
      uint32_t significand = 0x800000 | (magnitude & 0x7fffff);
      byte = significand >> shift;
      uint32_t tail = significand & ((1u << shift) - 1), half = 1u << (shift - 1);
      byte += tail > half || (tail == half && (byte & 1));
    }
  }
  int shift = (selection & 3) * 8;
  return (original & ~(255u << shift)) | (byte << shift);
}

} // namespace rocjitsu::arith::test
