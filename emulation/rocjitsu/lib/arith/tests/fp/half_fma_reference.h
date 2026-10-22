// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Independent integer reference for FP16 fused arithmetic numeric results.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include <algorithm>
#include <bit>
#include <stdint.h>

namespace rocjitsu::arith::test::half_fma_reference {

// Return whether the FP16 encoding is a NaN.
inline bool nan(uint16_t bits) { return (bits & 0x7fff) > 0x7c00; }

// Independent exact integer oracle: magnitude * 2^exponent. All finite half
// products and aligned half addends fit in uint64_t, including their sum.
struct Number {
  uint64_t magnitude;
  int exponent;
  bool negative;
};

// Decode finite FP16 bits into an integer significand, binary scale, and sign.
inline Number decode(uint16_t bits) {
  int e = (bits >> 10) & 31;
  return {uint64_t((bits & 1023) + (e ? 1024 : 0)), (e ? e : 1) - 25, bool(bits & 0x8000)};
}

// Scale an unsigned significand by a binary shift, rounding right shifts to nearest-even.
inline uint64_t rounded_shift(uint64_t bits, int shift) {
  if (shift <= 0) {
    return bits << -shift;
  }
  uint64_t tail = bits & ((1ULL << shift) - 1);
  uint64_t midpoint = 1ULL << (shift - 1);
  return (bits >> shift) + (tail > midpoint || (tail == midpoint && ((bits >> shift) & 1)));
}

// Round a signed integer-grid value to FP16, optionally saturating finite overflow.
inline uint16_t pack(Number value, bool saturate) {
  uint16_t sign = value.negative ? 0x8000 : 0;
  if (!value.magnitude) {
    return sign;
  }
  int top = int(std::bit_width(value.magnitude)) - 1;
  int exponent = top + value.exponent;
  if (exponent < -14) {
    return sign | uint16_t(rounded_shift(value.magnitude, -24 - value.exponent));
  }
  uint64_t sig = rounded_shift(value.magnitude, top - 10);
  if (sig == 2048) {
    sig = 1024;
    ++exponent;
  }
  if (exponent > 15) {
    return sign | (saturate ? 0x7bff : 0x7c00);
  }
  return sign | uint16_t(((exponent + 15) << 10) + sig - 1024);
}

// Compute FP16 FMA under default guest controls with optional finite saturation.
inline uint16_t evaluate(uint16_t a, uint16_t b, uint16_t c, bool saturate) {
  uint16_t ma = a & 0x7fff, mb = b & 0x7fff, mc = c & 0x7fff;
  uint16_t result;
  if ((!ma && mb == 0x7c00) || (!mb && ma == 0x7c00)) {
    result = 0xfe00;
  } else if (nan(a) || nan(b) || nan(c)) {
    result = (nan(a) ? a : nan(b) ? b : c) | 0x200;
  } else if (ma == 0x7c00 || mb == 0x7c00) {
    result = ((a ^ b) & 0x8000) | 0x7c00;
    if (mc == 0x7c00 && ((result ^ c) & 0x8000)) {
      result = 0xfe00;
    }
  } else if (mc == 0x7c00) {
    result = c;
  } else {
    Number x = decode(a);
    Number y = decode(b);
    Number z = decode(c);
    Number p = {x.magnitude * y.magnitude, x.exponent + y.exponent, x.negative != y.negative};
    int grid = std::min(p.exponent, z.exponent);
    uint64_t pm = p.magnitude << (p.exponent - grid), zm = z.magnitude << (z.exponent - grid);
    Number sum = {0, grid, false};
    if (p.negative == z.negative) {
      sum.magnitude = pm + zm;
      sum.negative = p.negative;
    } else {
      sum.magnitude = pm > zm ? pm - zm : zm - pm;
      sum.negative = pm > zm ? p.negative : zm > pm ? z.negative : false;
    }
    result = pack(sum, saturate);
  }
  return result;
}

} // namespace rocjitsu::arith::test::half_fma_reference
