// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar FP32 FMA under default guest controls, including exceptional values.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "fp/fma_integer.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Round a finite round-to-odd FP64 intermediate to nearest-even FP32, preserving
// subnormals. A carried significand can promote a subnormal to the smallest normal.
inline uint32_t fma_narrow_f32(uint64_t bits) {
  uint32_t sign = uint32_t(bits >> 32) & 0x80000000;
  uint64_t magnitude = bits & 0x7fffffffffffffffULL;
  if (!magnitude) {
    return sign;
  }
  const uint32_t overflow = sign | 0x7f800000;
  int exponent = int(magnitude >> 52) - 1023;
  if (exponent > 127) {
    return overflow;
  }

  if (exponent < -150) {
    return sign;
  }
  int shift = 29 + (exponent < -126 ? -126 - exponent : 0);
  uint64_t significand = (magnitude & 0xfffffffffffffULL) | (1ULL << 52);
  uint32_t rounded = uint32_t(significand >> shift);
  uint64_t unit = 1ULL << shift, remainder = significand & (unit - 1);
  rounded += remainder > unit / 2 || (remainder == unit / 2 && (rounded & 1));
  if (exponent < -126) {
    return sign | rounded;
  }
  if (rounded == (1U << 24)) {
    rounded >>= 1;
    ++exponent;
  }
  if (exponent > 127) {
    return overflow;
  }
  return sign | (uint32_t(exponent + 127) << 23) | (rounded & 0x7fffff);
}

// rocjitsu shared/fp_mode.h FMA NaN ordering and nearest-even rounding,
// using the existing integer intermediate instead of host FP-control changes.
// Accept raw FP32 encodings, preserving subnormals. No FP status is produced.
inline uint32_t fma_value_f32(uint32_t a, uint32_t b, uint32_t c, bool quiet_nan) {
  uint32_t ma = a & 0x7fffffff, mb = b & 0x7fffffff, mc = c & 0x7fffffff;

  uint32_t result;
  if ((ma == 0 && mb == 0x7f800000) || (mb == 0 && ma == 0x7f800000)) {
    result = 0xffc00000;
  } else if (ma > 0x7f800000 || mb > 0x7f800000 || mc > 0x7f800000) {
    result = (ma > 0x7f800000 ? a : mb > 0x7f800000 ? b : c) | (quiet_nan ? 0x00400000 : 0);
  } else if (ma == 0x7f800000 || mb == 0x7f800000) {
    result = ((a ^ b) & 0x80000000) | 0x7f800000;
    if (mc == 0x7f800000 && ((result ^ c) & 0x80000000)) {
      result = 0xffc00000;
    }
  } else if (mc == 0x7f800000) {
    result = c;
  } else {
    uint64_t value = fma_finite_round_odd(a, b, c);

    result = fma_narrow_f32(value);
  }
  return result;
}

} // namespace rocjitsu::arith
