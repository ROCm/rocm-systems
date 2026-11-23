// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Numeric reference decoding for compact matrix formats, independent of bit widening.
#pragma once
#include <cmath>
#include <limits>
namespace rocjitsu::arith::test {
// Decode numerically, independently of the implementation's exponent-bit construction.
inline float decode_matrix_format(int code, int format) {
  int width = format < 2 ? 8 : format < 4 ? 6 : 4;
  int fraction_bits = format == 1 || format == 3 ? 2 : format == 4 ? 1 : 3;
  int bias = format == 0 ? 7 : format == 1 ? 15 : format == 3 ? 3 : 1;
  int magnitude = code & ((1 << (width - 1)) - 1), exponent = magnitude >> fraction_bits;
  int fraction = magnitude & ((1 << fraction_bits) - 1);
  if (format == 0 && magnitude == 127) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (format == 1 && exponent == 31) {
    if (fraction) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return code & 128 ? -std::numeric_limits<float>::infinity()
                      : std::numeric_limits<float>::infinity();
  }
  float value = std::ldexp(float(exponent ? (1 << fraction_bits) + fraction : fraction),
                           (exponent ? exponent : 1) - bias - fraction_bits);
  return code & (1 << (width - 1)) ? -value : value;
}

} // namespace rocjitsu::arith::test
