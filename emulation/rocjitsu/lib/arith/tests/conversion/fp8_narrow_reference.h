// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 narrowing to FP8/BF8, including stochastic rounding.

#pragma once

#include "rj_arith/rj_arith.h"

#include <bit>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent numeric reference: quantize in units of the output spacing.
inline uint32_t fp8_narrow_reference(bool bf8, bool stochastic, uint32_t raw, uint32_t seed,
                                     bool saturate) {
  unsigned sign = (raw >> 24) & 128, magnitude = raw & 0x7fffffff;
  unsigned terminal = bf8 ? 124 : 127;
  if (magnitude > 0x7f800000) {
    return bf8 ? 0xfe : 0xff;
  }
  if (magnitude == 0x7f800000) {
    return sign | terminal;
  }
  float input = std::bit_cast<float>(magnitude);
  double x = input;
  unsigned limit = terminal - saturate;
  if (x > (bf8 ? 65536.0 : 512.0)) {
    return sign | limit;
  }
  unsigned fraction_bits = bf8 ? 2 : 3;
  int exponent;
  std::frexp(x, &exponent);
  int power = exponent - 1 - int(fraction_bits);
  int minimum_power = bf8 ? -16 : -9;
  if (power < minimum_power || x == 0) {
    power = minimum_power;
  }
  double units = std::ldexp(x, -power);
  unsigned rounded;
  // The seed supplies a fractional rounding offset; it is an instruction input, not a new random
  // draw.
  if (stochastic) {
    unsigned precision = 23 - fraction_bits;
    double fixed = std::floor(std::ldexp(units, int(precision)));
    rounded = unsigned(std::floor(std::ldexp(fixed + (seed >> (32 - precision)), -int(precision))));
  } else {
    rounded = unsigned(std::floor(units));
    double fraction = units - rounded;
    rounded += fraction > 0.5 || (fraction == 0.5 && (rounded & 1));
  }
  unsigned code = rounded + (unsigned(power - minimum_power) << fraction_bits);
  return sign | (code < limit ? code : limit);
}

// Return a packed FP8/BF8 narrowing result, preserving destination bytes not written by the
// operation.
inline uint32_t fp8_narrow_result(int op, uint32_t a, uint32_t b, uint32_t old, uint32_t mode,
                                  bool saturate) {
  if (mode & RJ_ARITH_ALU_ABS_A) {
    a &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    a ^= 0x80000000;
  }
  uint32_t value = fp8_narrow_reference(op & 1, op >= 2, a, b, saturate);
  int shift = ((mode >> 16) & 3) * 8;
  uint32_t selected = 255;
  if (op < 2) {
    if (mode & RJ_ARITH_ALU_ABS_B) {
      b &= 0x7fffffff;
    }
    if (mode & RJ_ARITH_ALU_NEG_B) {
      b ^= 0x80000000;
    }
    value |= fp8_narrow_reference(op & 1, false, b, 0, saturate) << 8;
    shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
    selected = 65535;
  }
  return (old & ~(selected << shift)) | (value << shift);
}

} // namespace rocjitsu::arith::test
