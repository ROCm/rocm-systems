// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP8/BF8 widening into FP32 encodings.

#pragma once

#include "common/conversion64_reference.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent integer decoding of sign, exponent and significand, including
// special values. No production conversion helpers are used.

// Decode an FP8/BF8 byte to its FP32 representation.
inline uint32_t fp8_conversion_reference(bool bf8, uint8_t code, bool fnuz = false,
                                         bool preserve_nan_sign = false) {
  if (fnuz && code == 0x80U) {
    return 0xffc00000U;
  }
  bool negative = code >> 7;
  unsigned fraction_bits = bf8 ? 2 : 3;
  unsigned magnitude = code & 127;
  if (!fnuz && bf8 && magnitude == 124) {
    return (negative ? 0x80000000u : 0) | 0x7f800000;
  }
  if (!fnuz && ((bf8 && magnitude > 124) || (!bf8 && magnitude == 127))) {
    return preserve_nan_sign ? (negative ? 0xffc00000U : 0x7fc00000U) : 0xffc00000U;
  }
  int exponent = magnitude >> fraction_bits;
  unsigned significand = magnitude & ((1u << fraction_bits) - 1);
  if (exponent) {
    significand += 1u << fraction_bits;
  }
  int power = int(exponent ? exponent : 1) - (bf8 ? 15 : 7) - int(fnuz) - int(fraction_bits);
  return uint32_t(conversion_encode(negative, significand, power, 23, 127));
}

} // namespace rocjitsu::arith::test
