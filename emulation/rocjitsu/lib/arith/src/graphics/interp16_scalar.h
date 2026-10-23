// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar interpolation arithmetic with round-to-odd and FP16 narrowing.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/internal.h"
#include "fp/fma_integer.h"
#include "fp/fp64.h"
#include "fp/mixed_fma_scalar.h"

#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith {

// Preserve discarded bits without using host floating-point arithmetic.
inline double interp16_round_odd(uint32_t a, uint32_t b, uint32_t c) {
  return as_double(fma_finite_round_odd(a, b, c));
}

// Evaluate the interpolation stage with round-to-zero FP32 intermediate arithmetic.
inline float interp16_rtz_float(uint32_t a, uint32_t b, uint32_t c) {
  if ((a & 0x7fffffff) >= 0x7f800000 || (b & 0x7fffffff) >= 0x7f800000 ||
      (c & 0x7fffffff) >= 0x7f800000) {
    return std::fma(as_float(a), as_float(b), as_float(c));
  }
  uint64_t bits = fma_finite_round_odd(a, b, c);
  uint32_t sign = uint32_t(bits >> 32) & 0x80000000;
  int exponent = int((bits >> 52) & 2047) - 1023;
  uint64_t significand = (bits & 0xfffffffffffffULL) | (1ULL << 52);
  if (exponent > 127) {
    return as_float(sign | 0x7f7fffff);
  }
  if (exponent < -149) {
    return as_float(sign);
  }
  if (exponent < -126) {
    return as_float(sign | uint32_t(significand >> (-exponent - 97)));
  }
  return as_float(sign | (uint32_t(exponent + 127) << 23) |
                  (uint32_t(significand >> 29) & 0x7fffff));
}

// Evaluate the interpolation stage with round-to-zero FP16 output.
inline uint16_t interp16_rtz_half(uint32_t a, uint32_t b, uint32_t c, bool clamp) {
  uint16_t result;
  if ((a & 0x7fffffff) >= 0x7f800000 || (b & 0x7fffffff) >= 0x7f800000 ||
      (c & 0x7fffffff) >= 0x7f800000) {
    result = mixed_fma_special(a, b, c);
  } else {
    uint64_t bits = double_bits(interp16_round_odd(a, b, c));
    unsigned exponent = unsigned((bits >> 52) & 2047);
    uint16_t sign = uint16_t((bits >> 48) & 0x8000);
    uint64_t significand = (bits & 0xfffffffffffffULL) | (1ULL << 52);
    if (exponent < 999) {
      result = sign;
    } else if (exponent > 1038) {
      result = sign | 0x7bff;
    } else if (exponent < 1009) {
      result = sign | uint16_t(significand >> (1051 - exponent));
    } else {
      result = sign | uint16_t(((exponent - 1009) << 10) + (significand >> 42));
    }
  }
  return clamp ? half_fma_clamp(result) : result;
}

} // namespace rocjitsu::arith
