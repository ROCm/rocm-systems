// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer-grid rounding and re-encoding for FP64 conversions.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

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

} // namespace rocjitsu::arith::test
