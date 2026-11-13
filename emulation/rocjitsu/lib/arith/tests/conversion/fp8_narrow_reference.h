// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 narrowing to FP8/BF8, including stochastic rounding.

#pragma once

#include "rj_arith/detail/rj_arith_common.h"

#include <bit>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent numeric reference: quantize in units of the output spacing.
inline uint32_t fp8_narrow_reference(bool bf8, bool stochastic, uint32_t raw, uint32_t seed,
                                     bool saturate, int arch = 12, bool e5m3 = false,
                                     bool half = false) {
  bool fnuz = arch == 942;
  const bool rdna = (arch == 12 || arch == 1250) && !e5m3 && !half;
  unsigned sign = e5m3 ? 0 : (raw >> 24) & 128, magnitude = raw & 0x7fffffff;
  unsigned terminal = fnuz ? 128 : e5m3 ? 255 : bf8 ? 124 : 127;
  unsigned limit = terminal - saturate;
  auto encode = [&](unsigned code) {
    if (fnuz && (!code || code >= 128)) {
      return code ? 128U : 0U;
    }
    return sign | code;
  };
  if (magnitude > 0x7f800000) {
    return fnuz ? 128 : e5m3 ? 255 : rdna ? (bf8 ? 0xfe : 0xff) : sign | 127;
  }
  if (magnitude == 0x7f800000) {
    return encode(rdna ? terminal : limit);
  }
  float input = std::bit_cast<float>(magnitude);
  double x = input;
  if (x > (e5m3 ? 131072.0 : bf8 ? 65536.0 : fnuz ? 256.0 : 512.0)) {
    return encode(limit);
  }
  unsigned fraction_bits = bf8 ? 2 : 3;
  int exponent;
  std::frexp(x, &exponent);
  int power = exponent - 1 - int(fraction_bits);
  int minimum_power = e5m3 ? -17 : (bf8 ? -16 : -9) - int(fnuz);
  if (power < minimum_power || x == 0) {
    power = minimum_power;
  }
  double units = std::ldexp(x, -power);
  unsigned rounded;
  // The seed supplies a fractional rounding offset; it is an instruction input, not a new random
  // draw.
  if (stochastic) {
    unsigned precision = 23 - fraction_bits;
    if (!rdna && power == minimum_power) {
      int subnormal_precision = minimum_power - exponent + 24;
      if (x == 0 || subnormal_precision > 24) {
        return encode(0);
      }
      if (subnormal_precision > int(precision)) {
        precision = subnormal_precision;
      }
    }
    double fixed = std::floor(std::ldexp(units, int(precision)));
    rounded = unsigned(std::floor(std::ldexp(fixed + (seed >> (32 - precision)), -int(precision))));
  } else {
    rounded = unsigned(std::floor(units));
    double fraction = units - rounded;
    rounded += fraction > 0.5 || (fraction == 0.5 && (rounded & 1));
  }
  unsigned code = rounded + (unsigned(power - minimum_power) << fraction_bits);
  return encode(code < limit ? code : limit);
}

// Return a packed FP8/BF8 narrowing result, preserving destination bytes not written by the
// operation.
inline uint32_t fp8_narrow_result(int op, uint32_t a, uint32_t b, uint32_t old, uint32_t mode,
                                  bool saturate, int arch = 12, bool half = false) {
  bool e5m3 = arch == 1250 && !(op & 1) && (mode & RJ_ARITH_ALU_CLAMP);
  if (e5m3 || (half && op >= 2)) {
    mode &= ~(RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B);
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    a &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    a ^= 0x80000000;
  }
  uint32_t value = fp8_narrow_reference(op & 1, op >= 2, a, b, saturate, arch, e5m3, half);
  int shift = ((mode >> 16) & 3) * 8;
  uint32_t selected = 255;
  if (op < 2) {
    if (mode & RJ_ARITH_ALU_ABS_B) {
      b &= 0x7fffffff;
    }
    if (mode & RJ_ARITH_ALU_NEG_B) {
      b ^= 0x80000000;
    }
    value |= fp8_narrow_reference(op & 1, false, b, 0, saturate, arch, e5m3, half) << 8;
    shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
    selected = 65535;
  }
  uint32_t mask = selected << shift;
  return (old & ~mask) | (value << shift);
}

} // namespace rocjitsu::arith::test
