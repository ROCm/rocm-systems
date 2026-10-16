// SPDX-License-Identifier: MIT

// Integer-grid rounding and re-encoding for FP64 conversions.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent reference inherited from tests/conversion/conversion64_reference.h.
// Retain its rounding and special-value model; do not call production helpers.

// Return whether this conversion produces two destination words.
inline bool conversion64_wide_output(int op) { return op < 2 || op == 4; }

// Scale a significand by a binary shift, rounding right shifts to nearest-even.
inline uint64_t conversion_round_right(uint64_t value, int shift) {
  if (shift <= 0) {
    return value << -shift;
  }
  if (shift >= 64) {
    return 0; // Inputs have at most 53 significant bits.
  }
  uint64_t quotient = value >> shift, remainder = value & ((1ULL << shift) - 1);
  uint64_t half = 1ULL << (shift - 1);
  return quotient + (remainder > half || (remainder == half && (quotient & 1)));
}

// Encode a signed integer significand and binary scale in the requested floating-point format.
inline uint64_t conversion_encode(bool negative, uint64_t significand, int scale, int fraction,
                                  int bias) {
  // Integer-only packing: significand * 2^scale, rounded to nearest-even at
  // the destination precision, with separate normal/subnormal exponent ranges.
  uint64_t sign = negative ? (uint64_t(2 * bias + 2) << fraction) : 0;
  uint64_t infinity = uint64_t(2 * bias + 1) << fraction;
  if (!significand) {
    return sign;
  }
  int top = 0;
  for (uint64_t copy = significand; copy >>= 1;) {
    ++top;
  }
  int exponent = top + scale;
  if (exponent > bias) {
    return sign | infinity;
  }
  if (exponent < 1 - bias) {
    exponent = 1 - bias;
  }
  uint64_t rounded = conversion_round_right(significand, exponent - int(fraction) - scale);
  if (rounded >= (2ULL << fraction)) {
    rounded >>= 1;
    ++exponent;
  }
  if (exponent > bias) {
    return sign | infinity;
  }
  uint64_t encoded_exp = rounded < (1ULL << fraction) ? 0 : uint64_t(exponent + bias);
  return sign | (encoded_exp << fraction) | (rounded & ((1ULL << fraction) - 1));
}

// Convert an encoded floating-point value between formats, optionally scaling by a power of two.
inline uint64_t conversion_reencode(uint64_t raw, int source_fraction, int source_bias,
                                    int destination_fraction, int destination_bias,
                                    int scaling = 0) {
  uint64_t significand = raw & ((1ULL << source_fraction) - 1);
  int exponent = unsigned((raw >> source_fraction) & unsigned(2 * source_bias + 1));
  bool negative = (raw >> source_fraction) >= unsigned(2 * source_bias + 2);
  if (exponent == 2 * source_bias + 1) {
    uint64_t sign = negative ? (uint64_t(2 * destination_bias + 2) << destination_fraction) : 0;
    uint64_t special = sign | (uint64_t(2 * destination_bias + 1) << destination_fraction);
    return significand ? special | (1ULL << (destination_fraction - 1)) : special;
  }
  int scale = (exponent ? int(exponent) - source_bias : 1 - source_bias) - int(source_fraction);
  if (exponent) {
    significand |= 1ULL << source_fraction;
  }
  return conversion_encode(negative, significand, scale + scaling, destination_fraction,
                           destination_bias);
}

// Return the selected conversion result as raw bits, including FP64 word pairs.
inline uint64_t conversion64_reference(int op, uint64_t raw, uint32_t mode) {
  uint64_t output;
  if (op < 2) {
    bool negative = op == 0 && (uint32_t(raw) >> 31);
    uint32_t magnitude = negative ? uint32_t(0) - uint32_t(raw) : uint32_t(raw);
    output = conversion_encode(negative, magnitude, 0, 52, 1023);
  } else {
    uint64_t sign = op == 4 ? 0x80000000ULL : 0x8000000000000000ULL;
    if (mode & RJ_ARITH_ALU_ABS_A) {
      raw &= ~sign;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      raw ^= sign;
    }
    // Classify NaN and range before converting to integer, avoiding undefined out-of-range C++
    // casts.
    if (op == 2 || op == 3) {
      bool negative = raw >> 63;
      int exponent = unsigned((raw >> 52) & 2047);
      uint64_t fraction = raw & 0xfffffffffffffULL;
      if ((exponent == 2047 && fraction) || (op == 3 && negative) || exponent < 1023) {
        return 0;
      }
      uint64_t limit = op == 3 ? 0xffffffffULL : negative ? 0x80000000ULL : 0x7fffffffULL;
      uint64_t magnitude =
          exponent >= 1055 ? limit : (fraction | (1ULL << 52)) >> (1075 - exponent);
      if (magnitude > limit) {
        magnitude = limit;
      }
      return negative ? uint32_t(0) - uint32_t(magnitude) : uint32_t(magnitude);
    }
    output = op == 4 ? conversion_reencode(raw, 23, 127, 52, 1023)
                     : conversion_reencode(raw, 52, 1023, 23, 127);
  }
  int fraction = conversion64_wide_output(op) ? 52 : 23;
  int bias = conversion64_wide_output(op) ? 1023 : 127;
  if (op == 5 && (mode & RJ_ARITH_ALU_OMOD_HALF)) {
    uint32_t magnitude = uint32_t(output) & 0x7fffffff;
    if ((raw & 0x7fffffffffffffffULL) < 0x3810000000000000ULL || magnitude < 0x00800000) {
      output = 0;
    } else if ((mode & RJ_ARITH_ALU_OMOD_HALF) == RJ_ARITH_ALU_OMOD_HALF &&
               magnitude < 0x01000000) {
      output &= 0x80000000;
    }
  }
  if (conversion64_wide_output(op) && (mode & RJ_ARITH_ALU_OMOD_HALF) &&
      !(output & 0x7fffffffffffffffULL)) {
    output = 0;
  }
  const int scales[] = {0, 1, 2, -1};
  output = conversion_reencode(output, fraction, bias, fraction, bias, scales[(mode >> 6) & 3]);
  if (mode & RJ_ARITH_ALU_CLAMP) {
    uint64_t one = uint64_t(bias) << fraction;
    uint64_t infinity = uint64_t(2 * bias + 1) << fraction;
    if (output >= (uint64_t(2 * bias + 2) << fraction) || output > infinity) {
      output = 0;
    } else if (output > one) {
      output = one;
    }
  }
  return output;
}

} // namespace rocjitsu::arith::test
