// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Finite IEEE fused arithmetic with guest rounding, inherited from Rocjitsu division.
// Input/output flushing, exceptional operands and instruction modifiers belong to callers.
#pragma once

#include "common/uint128.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// IEEE binary64 fields shared by finite ADD, MUL and FMA rounding.
struct FmaF64Format {
  using Bits = uint64_t;
  static constexpr int fraction = 52, bias = 1023;
  static constexpr uint64_t sign = 0x8000000000000000ULL;
  static constexpr uint64_t infinity = 0x7ff0000000000000ULL;
};

// Encode finite overflow using the guest's direction and the result sign.
inline uint64_t fma_overflow_ieee(bool negative, unsigned rounding) {
  return FmaF64Format::infinity - !(rounding == 0 || rounding == (negative ? 2u : 1u));
}

// Return the index of the highest set bit. Value must be nonzero.
inline int fma_top_bit128(Uint128 value) {
  uint64_t high = uint64_t(value >> 64), word = high ? high : uint64_t(value);
  int top = high ? 64 : 0;
  if (word >> 32) {
    word >>= 32;
    top += 32;
  }
  if (word >> 16) {
    word >>= 16;
    top += 16;
  }
  if (word >> 8) {
    word >>= 8;
    top += 8;
  }
  if (word >> 4) {
    word >>= 4;
    top += 4;
  }
  if (word >> 2) {
    word >>= 2;
    top += 2;
  }
  return top + int(word >> 1);
}

// Shift right, retaining a sticky bit for every discarded nonzero bit.
inline Uint128 fma_shift_jam128(Uint128 value, int distance) {
  if (distance <= 0) {
    return value;
  }
  if (distance >= 128) {
    return value != 0;
  }
  return (value >> distance) | Uint128((value << (128 - distance)) != 0);
}

// Round significand * 2^exponent to IEEE bits in the selected guest rounding mode.
inline uint64_t fma_round_ieee(Uint128 significand, int exponent, bool negative,
                               unsigned rounding) {
  using F = FmaF64Format;
  using T = typename F::Bits;
  T sign = negative ? F::sign : 0;
  if (significand == 0) {
    return sign;
  }
  int highest = fma_top_bit128(significand);
  if (highest + exponent > F::bias) {
    return sign | fma_overflow_ieee(negative, rounding);
  }
  int normal_shift = highest - F::fraction, subnormal_shift = 1 - F::bias - F::fraction - exponent;
  int shift = normal_shift > subnormal_shift ? normal_shift : subnormal_shift;
  Uint128 kept;
  bool guard = false, sticky = false;
  if (shift <= 0) {
    kept = significand << -shift;
  } else {
    kept = shift >= 128 ? Uint128(0) : significand >> shift;
    guard = shift <= 128 && ((significand >> (shift - 1)) & 1) != 0;
    sticky = shift > 128 ? significand != 0 : shift > 1 && (significand << (129 - shift)) != 0;
  }
  if ((rounding == 0 && guard && (sticky || (kept & 1) != 0)) ||
      ((guard || sticky) && rounding == (negative ? 2U : 1U))) {
    ++kept;
  }
  int result_exponent = exponent + shift + F::fraction;
  if (!((Uint128(1) << (F::fraction + 1)) > kept)) {
    kept >>= 1;
    ++result_exponent;
  }
  if (result_exponent > F::bias) {
    return sign | fma_overflow_ieee(negative, rounding);
  }
  if ((Uint128(1) << F::fraction) > kept) {
    return sign | T(kept);
  }
  return sign | (T(result_exponent + F::bias) << F::fraction) |
         (T(kept) & ((T(1) << F::fraction) - 1));
}

// Unrounded A*B+C as significand * 2^exponent, with sticky bits standing in for discarded
// low-order bits. A zero significand stands for the signed zero chosen by the guest rounding.
struct FmaSumF64 {
  Uint128 significand;
  int exponent;
  bool negative;
};

// Sum A*B+C exactly enough for one rounding. All three encodings must be finite.
inline FmaSumF64 fma_sum_f64(uint64_t a, uint64_t b, uint64_t c, unsigned rounding) {
  using F = FmaF64Format;
  using T = uint64_t;
  constexpr T fraction_mask = (T(1) << F::fraction) - 1;
  T am = a & ~F::sign, bm = b & ~F::sign, cm = c & ~F::sign;
  bool pn = ((a ^ b) & F::sign) != 0, cn = (c & F::sign) != 0;
  int ae = int(am >> F::fraction), be = int(bm >> F::fraction), ce = int(cm >> F::fraction);
  T as = (a & fraction_mask) | (ae ? T(1) << F::fraction : 0),
    bs = (b & fraction_mask) | (be ? T(1) << F::fraction : 0),
    cs = (c & fraction_mask) | (ce ? T(1) << F::fraction : 0);
  Uint128 product = Uint128(as) * bs, addend = cs;
  int pe = (ae ? ae : 1) + (be ? be : 1) - 2 * (F::bias + F::fraction),
      se = (ce ? ce : 1) - F::bias - F::fraction;
  if (product == 0 && addend == 0) {
    return {0, 0, pn == cn ? pn : rounding == 2};
  }
  if (product == 0) {
    return {addend, se, cn};
  }
  if (addend == 0) {
    return {product, pe, pn};
  }
  int pt = fma_top_bit128(product), ct = fma_top_bit128(addend);
  int exponent = pe + pt > se + ct ? pe + pt : se + ct;
  product = fma_shift_jam128(product << (126 - pt), exponent - pe - pt);
  addend = fma_shift_jam128(addend << (126 - ct), exponent - se - ct);
  bool negative = pn == cn ? pn : product > addend ? pn : addend > product ? cn : rounding == 2;
  Uint128 sum = pn == cn           ? product + addend
                : product > addend ? product - addend
                                   : addend - product;
  return {sum, exponent - 126, negative};
}

// Return A*B+C with one guest-directed rounding. All three encodings must be finite.
inline uint64_t fma_finite_f64(uint64_t a, uint64_t b, uint64_t c, unsigned rounding) {
  FmaSumF64 sum = fma_sum_f64(a, b, c, rounding);
  return fma_round_ieee(sum.significand, sum.exponent, sum.negative, rounding);
}

// Whether A*B+C is tiny after rounding: rounded to full precision in the guest mode with an
// unbounded exponent range, it lies below the smallest normal. A result that reaches the
// smallest normal only on the subnormal grid is tiny. Scaling by 2^64 moves every result
// near the smallest normal into the normal range, where rounding keeps all 53 bits.
// Encodings must be finite.
inline bool fma_tiny_f64(uint64_t a, uint64_t b, uint64_t c, unsigned rounding) {
  using F = FmaF64Format;
  FmaSumF64 sum = fma_sum_f64(a, b, c, rounding);
  constexpr uint64_t scaled_smallest_normal = uint64_t(1 + 64) << F::fraction;
  return sum.significand == 0 ||
         (fma_round_ieee(sum.significand, sum.exponent + 64, sum.negative, rounding) & ~F::sign) <
             scaled_smallest_normal;
}

// Replace a rounded A*B+C equal to the smallest normal by a zero of its sign when the sum is
// tiny after rounding, for callers about to flush or apply OMOD. Other results are unchanged:
// only the smallest normal can hide a tiny sum.
inline uint64_t fma_flush_tiny_f64(uint64_t result, uint64_t a, uint64_t b, uint64_t c,
                                   unsigned rounding) {
  using F = FmaF64Format;
  return (result & ~F::sign) == (uint64_t(1) << F::fraction) && fma_tiny_f64(a, b, c, rounding)
             ? result & F::sign
             : result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
