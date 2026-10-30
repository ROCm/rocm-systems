// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP8/BF8 widening into FP32 encodings.

#pragma once

#include "conversion/conversion64_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent reference inherited from tests/conversion/fp8_conversion_reference.h.
// Retain its rounding and special-value model; do not call production helpers.

// Decode an FP8/BF8 byte to its FP32 representation.
inline uint32_t fp8_conversion_reference(bool bf8, uint8_t code) {
  bool negative = code >> 7;
  unsigned fraction_bits = bf8 ? 2 : 3;
  unsigned magnitude = code & 127;
  if (bf8 && magnitude == 124) {
    return (negative ? 0x80000000u : 0) | 0x7f800000;
  }
  if ((bf8 && magnitude > 124) || (!bf8 && magnitude == 127)) {
    return 0xffc00000;
  }
  int exponent = magnitude >> fraction_bits;
  unsigned significand = magnitude & ((1u << fraction_bits) - 1);
  if (exponent) {
    significand += 1u << fraction_bits;
  }
  int power = int(exponent ? exponent : 1) - (bf8 ? 15 : 7) - int(fraction_bits);
  return uint32_t(conversion_encode(negative, significand, power, 23, 127));
}

} // namespace rocjitsu::arith::test
