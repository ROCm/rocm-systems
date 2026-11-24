// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Numeric reference for inherited finite-only FP4/FP6/BF6 rounding.
#pragma once
#include <bit>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {
// Independently quantize against the finite format's positive values. The stochastic
// seed contributes a fractional offset at the source mantissa's precision.
template <int ExponentBits, int FractionBits>
uint32_t quantize_finite(float value, bool stochastic, uint32_t seed, bool gfx950 = false,
                         int source_fraction = 23) {
  constexpr int bias = (1 << (ExponentBits - 1)) - 1;
  constexpr int count = 1 << (ExponentBits + FractionBits);
  uint32_t raw = std::bit_cast<uint32_t>(value),
           sign = (raw >> (31 - ExponentBits - FractionBits)) & count;
  double values[count];
  for (int code = 0; code < count; ++code) {
    int exponent = code >> FractionBits, mantissa = code & ((1 << FractionBits) - 1);
    values[code] = std::ldexp(double(exponent ? (1 << FractionBits) + mantissa : mantissa),
                              (exponent ? exponent : 1) - bias - FractionBits);
  }
  double x = std::fabs(double(value));
  if (std::isnan(x)) {
    return gfx950 ? (sign | (count - 1)) : 0;
  }
  if (x >= values[count - 1]) {
    return sign | (count - 1);
  }
  if (!(gfx950 && stochastic) && x < values[1] / 2) {
    return sign;
  }
  int lower = 0;
  while (values[lower + 1] <= x) {
    ++lower;
  }
  double spacing = values[lower + 1] - values[lower];
  double fraction = (x - values[lower]) / spacing;
  bool up;
  if (gfx950 && stochastic) {
    int precision = source_fraction - FractionBits;
    double fixed = std::floor(std::ldexp(fraction, precision));
    up = (fixed + double(seed >> (32 - precision))) >= double(1U << precision);
  } else if (stochastic) {
    int exponent, spacing_exponent;
    std::frexp(x, &exponent);
    std::frexp(spacing, &spacing_exponent);
    int precision = 23 + spacing_exponent - exponent;
    up = fraction + std::ldexp(double(seed >> (32 - precision)), -precision) >= 1;
  } else {
    up = fraction > 0.5 || (fraction == 0.5 && (lower & 1));
  }
  return sign | uint32_t(lower + up);
}

} // namespace rocjitsu::arith::test
