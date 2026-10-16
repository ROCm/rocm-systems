// SPDX-License-Identifier: MIT

// Division fixup, exceptional operands, and FP-status expectations.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <cmath>
#include <limits>
#include <stdint.h>

namespace rocjitsu::arith::test {

inline long double fixup_decode(int width, uint64_t raw) {
  unsigned fraction = width == 16   ? 10
                      : width == 32 ? 23
                                    : 52,
           bias = width == 16   ? 15
                  : width == 32 ? 127
                                : 1023;
  uint64_t sign = 1ULL << (width - 1), mantissa = raw & ((1ULL << fraction) - 1);
  unsigned exponent = unsigned((raw & (sign - 1)) >> fraction);
  long double value;
  if (exponent == 2 * bias + 1) {
    value = mantissa ? std::numeric_limits<long double>::quiet_NaN()
                     : std::numeric_limits<long double>::infinity();
  } else {
    value = std::ldexp(static_cast<long double>(mantissa + (exponent ? (1ULL << fraction) : 0)),
                       int(exponent ? exponent : 1) - int(bias) - int(fraction));
  }
  return raw & sign ? -value : value;
}

// Numeric reference with independent FP classification and long-double scaling.
inline uint64_t fixup_reference(int width, uint64_t a, uint64_t b, uint64_t c, uint32_t mode,
                                bool saturate) {
  unsigned fraction = width == 16   ? 10
                      : width == 32 ? 23
                                    : 52,
           bias = width == 16   ? 15
                  : width == 32 ? 127
                                : 1023;
  uint64_t sign_bit = 1ULL << (width - 1), infinity = uint64_t(2 * bias + 1) << fraction,
           quiet = 1ULL << (fraction - 1);
  uint64_t raw[] = {a, b, c};
  for (int i = 0; i < 3; ++i) {
    if (mode & (RJ_ARITH_ALU_ABS_A << i)) {
      raw[i] &= sign_bit - 1;
    }
    if (mode & (RJ_ARITH_ALU_NEG_A << i)) {
      raw[i] ^= sign_bit;
    }
  }
  long double p = fixup_decode(width, raw[0]), d = fixup_decode(width, raw[1]),
              n = fixup_decode(width, raw[2]);
  uint64_t sign = (std::signbit(d) != std::signbit(n)) ? sign_bit : 0;
  uint64_t result;
  if (std::isnan(n)) {
    result = raw[2] | quiet;
  } else if (std::isnan(d)) {
    result = raw[1] | quiet;
  } else if ((d == 0 && n == 0) || (std::isinf(d) && std::isinf(n))) {
    result = sign_bit | infinity | quiet;
  } else if (d == 0 || std::isinf(n)) {
    result = sign | infinity;
  } else if (n == 0 || std::isinf(d)) {
    result = sign;
  } else {
    int de, ne;
    std::frexp(std::fabs(d), &de);
    std::frexp(std::fabs(n), &ne);
    de = de + int(bias) - 1;
    ne = ne + int(bias) - 1;
    if (de < 0) {
      de = 0;
    }
    if (ne < 0) {
      ne = 0;
    }
    if (width != 16 && ne - de < -int(bias + fraction)) {
      result = sign;
    } else if (!std::isfinite(p)) {
      result = sign | (infinity - (width == 16 && saturate));
    } else {
      result = sign | (raw[0] & (sign_bit - 1));
    }
  }
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    long double input = fixup_decode(width, result);
    if (std::isfinite(input)) {
      long double minimum = std::ldexp(1.0L, 1 - int(bias));
      if (std::fabs(input) < minimum) {
        result = 0;
      } else {
        const int powers[] = {0, 1, 2, -1};
        long double output = std::ldexp(input, powers[omod]);
        uint64_t output_sign = result & sign_bit;
        if (std::fabs(output) < minimum) {
          result = output_sign;
        } else if (std::fabs(output) > fixup_decode(width, infinity - 1)) {
          result = output_sign | (infinity - (width == 16 && saturate));
        } else {
          int exponent;
          long double mantissa = std::frexp(std::fabs(output), &exponent);
          uint64_t significand = uint64_t(std::ldexp(mantissa, int(fraction) + 1));
          result = output_sign | (uint64_t(exponent + int(bias) - 1) << fraction) |
                   (significand - (1ULL << fraction));
        }
      }
    }
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    long double value = fixup_decode(width, result);
    if (!(value > 0)) {
      result = 0;
    } else if (value > 1) {
      result = uint64_t(bias) << fraction;
    }
  }
  return result;
}

} // namespace rocjitsu::arith::test
