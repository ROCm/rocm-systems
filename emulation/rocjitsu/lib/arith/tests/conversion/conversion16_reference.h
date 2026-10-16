// SPDX-License-Identifier: MIT

// FP16/FP32 and 16-bit integer conversion rules.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "conversion/conversion64_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent reference inherited from tests/conversion/conversion16_reference.h.
// Retain its rounding and special-value model; do not call production helpers.

// Round FP32 bits to FP16 with optional finite-overflow saturation.
inline uint16_t conversion16_narrow(uint32_t value, bool saturate) {
  uint16_t result = uint16_t(conversion_reencode(value, 23, 127, 10, 15));
  if (saturate && (value & 0x7fffffff) < 0x7f800000 && (result & 0x7fff) == 0x7c00) {
    --result;
  }
  return result;
}

// Return the selected FP16/FP32/integer conversion result, retaining any unselected destination
// half.
inline uint32_t conversion16_reference(int op, uint32_t raw, uint32_t original, uint32_t mode,
                                       bool saturate) {
  uint16_t half = uint16_t(raw >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
  uint32_t value;
  if (op < 2) {
    bool negative = op == 0 && (half >> 15);
    uint16_t magnitude = negative ? uint16_t(0u - half) : half;
    value = uint32_t(conversion_encode(negative, magnitude, 0, 23, 127));
  } else {
    value = op == 4 ? raw : uint32_t(conversion_reencode(half, 10, 15, 23, 127));
    if (mode & RJ_ARITH_ALU_ABS_A) {
      value &= 0x7fffffff;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      value ^= 0x80000000;
    }
  }
  uint32_t result;
  if (op == 2 || op == 3) {
    // Decode the exact binary integer part; no host floating-point casts.
    int exponent = (value >> 23) & 255;
    bool negative = value >> 31;
    uint32_t fraction = value & 0x7fffff;
    uint32_t limit = op == 3 ? 65535 : negative ? 32768 : 32767;
    if ((exponent == 255 && fraction) || (op == 3 && negative) || exponent < 127) {
      result = 0;
    } else {
      uint32_t magnitude = exponent >= 143 ? limit : (0x800000 | fraction) >> (150 - exponent);
      if (magnitude > limit) {
        magnitude = limit;
      }
      result = negative ? uint16_t(0u - magnitude) : magnitude;
    }
  } else {
    const int scales[] = {0, 1, 2, -1};
    int omod = (mode >> 6) & 3;
    if (op == 5) {
      result = uint32_t(conversion_reencode(value, 23, 127, 23, 127, scales[omod]));
      if (omod && (result & 0x7fffffff) == 0) {
        result = 0;
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        result = (result & 0x80000000) || result > 0x7f800000 ? 0
                 : result > 0x3f800000                        ? 0x3f800000
                                                              : result;
      }
    } else {
      result = conversion16_narrow(value, saturate);
      if (omod) {
        unsigned magnitude = result & 0x7fff;
        if (magnitude < 0x400 || (magnitude == 0x400 && (value & 0x7fffffff) < 0x387ff000)) {
          result = 0;
        } else if (omod == 3 && magnitude < 0x800) {
          result &= 0x8000;
        } else {
          uint32_t scaled = uint32_t(conversion_reencode(result, 10, 15, 23, 127, scales[omod]));
          result = conversion16_narrow(scaled, saturate);
        }
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        result = (result & 0x8000) || result > 0x7c00 ? 0 : result > 0x3c00 ? 0x3c00 : result;
      }
    }
  }
  if (op == 5) {
    return result;
  }
  int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  return (original & ~(uint32_t(65535) << shift)) | (result << shift);
}

} // namespace rocjitsu::arith::test
