// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Division FMA with conditional scaling and guest rounding/status handling.
// Internal helpers shared by instruction implementations; not part of the public C API.

// Fused post-scaling adapted from rocjitsu shared/division.h.

#pragma once

#include "common/uint128.h"
#include "fp/division.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Return the index of the highest set bit. Value must be nonzero.
inline int division_top_bit(Uint128 value) {
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
inline Uint128 division_shift_jam(Uint128 value, int distance) {
  if (distance <= 0) {
    return value;
  }
  if (distance >= 128) {
    return value != 0;
  }
  return (value >> distance) | Uint128((value << (128 - distance)) != 0);
}

// Round significand * 2^exponent with rocjitsu div_round guest rounding. With OMOD, round first at
// normal precision without gradual underflow, then flush a tiny result to +0.
template <int Width>
inline typename DivisionFormat<Width>::Bits
division_round(Uint128 significand, int exponent, bool negative, bool omod,
               uint32_t *exceptions = nullptr, unsigned rounding = 0) {
  using F = DivisionFormat<Width>;
  using T = typename F::Bits;
  T sign = negative ? F::sign : 0;
  if (significand == 0) {
    return sign;
  }
  int highest = division_top_bit(significand);
  if (highest + exponent > F::bias) {
    if (exceptions) {
      *exceptions |= RJ_ARITH_EXCEPTION_OVERFLOW | (omod ? 0 : RJ_ARITH_EXCEPTION_INEXACT);
    }
    return sign | division_overflow<Width>(negative, rounding);
  }
  int normal_shift = highest - F::fraction, subnormal_shift = 1 - F::bias - F::fraction - exponent;
  int shift = omod || normal_shift > subnormal_shift ? normal_shift : subnormal_shift;
  Uint128 kept;
  bool guard = false, sticky = false;
  if (shift <= 0) {
    kept = significand << -shift;
  } else {
    kept = shift >= 128 ? Uint128(0) : significand >> shift;
    guard = shift <= 128 && ((significand >> (shift - 1)) & 1) != 0;
    sticky = shift > 128 ? significand != 0 : shift > 1 && (significand << (129 - shift)) != 0;
  }
  if (exceptions && !omod && (guard || sticky)) {
    *exceptions |= RJ_ARITH_EXCEPTION_INEXACT;

    // Test tininess after rounding at normal precision, before restricting
    // the exponent. A value at the midpoint below the smallest normal rounds up.
    bool tiny = highest + exponent < 1 - F::bias;
    if (highest + exponent == -F::bias && normal_shift > 0) {
      Uint128 leading = Uint128(1) << highest;
      Uint128 midpoint = leading + (leading - (Uint128(1) << (normal_shift - 1)));
      if (!(midpoint > significand)) {
        tiny = false;
      }
    }
    if (tiny) {
      *exceptions |= RJ_ARITH_EXCEPTION_UNDERFLOW;
    }
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
    if (exceptions) {
      *exceptions |= RJ_ARITH_EXCEPTION_OVERFLOW | (omod ? 0 : RJ_ARITH_EXCEPTION_INEXACT);
    }
    return sign | division_overflow<Width>(negative, rounding);
  }
  if (omod && result_exponent < 1 - F::bias) {
    return 0;
  }
  if ((Uint128(1) << F::fraction) > kept) {
    return sign | T(kept);
  }
  return sign | (T(result_exponent + F::bias) << F::fraction) |
         (T(kept) & ((T(1) << F::fraction) - 1));
}

// Compute (A*B+C)*2^adjustment with one guest-directed rounding, preserving input
// denormals. If post_scale is set, adjustment is +64/+128 when C's encoded
// exponent exceeds its bias, and -64/-128 otherwise. Applies ABS/NEG and OMOD/
// CLAMP, preserves host FP state, and returns the resulting IEEE bits.
template <int Width>
inline typename DivisionFormat<Width>::Bits
division_fmas(typename DivisionFormat<Width>::Bits a, typename DivisionFormat<Width>::Bits b,
              typename DivisionFormat<Width>::Bits c, bool post_scale, uint32_t mode,
              uint32_t *exceptions = nullptr, unsigned rounding = 0) {
  using F = DivisionFormat<Width>;
  using T = typename F::Bits;

  // DIV_FMAS suppresses invalid and input-denormal exceptions, even for sNaNs
  // and invalid infinity operations. CLAMP suppresses all remaining flags.
  if (mode & RJ_ARITH_ALU_CLAMP) {
    exceptions = nullptr;
  }
  constexpr T fraction_mask = (T(1) << F::fraction) - 1;
  T inputs[] = {a, b, c};
  for (int i = 0; i < 3; ++i) {
    if (mode & (RJ_ARITH_ALU_ABS_A << i)) {
      inputs[i] &= ~F::sign;
    }
    if (mode & (RJ_ARITH_ALU_NEG_A << i)) {
      inputs[i] ^= F::sign;
    }
  }
  a = inputs[0];
  b = inputs[1];
  c = inputs[2];
  T am = a & ~F::sign, bm = b & ~F::sign, cm = c & ~F::sign, result;
  bool pn = ((a ^ b) & F::sign) != 0, cn = (c & F::sign) != 0;
  if (am > F::infinity) {
    result = a | F::quiet;
  } else if (bm > F::infinity) {
    result = b | F::quiet;
  } else if ((am == F::infinity && bm == 0) || (bm == F::infinity && am == 0) ||
             ((am == F::infinity || bm == F::infinity) && cm == F::infinity && pn != cn)) {
    result = F::sign | F::infinity | F::quiet;
  } else if (cm > F::infinity) {
    result = c | F::quiet;
  } else if (am == F::infinity || bm == F::infinity) {
    result = (pn ? F::sign : 0) | F::infinity;
  } else if (cm == F::infinity) {
    result = c;
  } else {
    int ae = int(am >> F::fraction), be = int(bm >> F::fraction), ce = int(cm >> F::fraction);
    T as = (a & fraction_mask) | (ae ? T(1) << F::fraction : 0),
      bs = (b & fraction_mask) | (be ? T(1) << F::fraction : 0),
      cs = (c & fraction_mask) | (ce ? T(1) << F::fraction : 0);
    Uint128 product = Uint128(as) * bs, addend = cs;
    int pe = (ae ? ae : 1) + (be ? be : 1) - 2 * (F::bias + F::fraction),
        se = (ce ? ce : 1) - F::bias - F::fraction;
    int adjustment = post_scale ? (ce > F::bias ? 1 : -1) * (Width == 32 ? 64 : 128) : 0;
    bool omod = (mode & RJ_ARITH_ALU_OMOD_HALF) != 0;
    if (product == 0 && addend == 0) {
      result = (pn == cn ? pn : rounding == 2) ? F::sign : 0;
    } else if (product == 0) {
      result = division_round<Width>(addend, se + adjustment, cn, omod, exceptions, rounding);
    } else if (addend == 0) {
      result = division_round<Width>(product, pe + adjustment, pn, omod, exceptions, rounding);
    } else {
      int pt = division_top_bit(product), ct = division_top_bit(addend);
      int exponent = pe + pt > se + ct ? pe + pt : se + ct;
      product = division_shift_jam(product << (126 - pt), exponent - pe - pt);
      addend = division_shift_jam(addend << (126 - ct), exponent - se - ct);
      bool negative = pn == cn ? pn : product > addend ? pn : addend > product ? cn : rounding == 2;
      Uint128 sum = pn == cn           ? product + addend
                    : product > addend ? product - addend
                                       : addend - product;
      result = division_round<Width>(sum, exponent - 126 + adjustment, negative, omod, exceptions,
                                     rounding);
    }
  }
  unsigned omod = (mode >> 6) & 3;
  T magnitude = result & ~F::sign;
  if (exceptions && (omod == 1 || omod == 2) && magnitude < F::infinity &&
      (magnitude >> F::fraction) + omod >= unsigned(2 * F::bias + 1)) {
    *exceptions |= RJ_ARITH_EXCEPTION_OVERFLOW;
  }
  return division_output<Width>(result, mode,
                                division_overflow<Width>((result & F::sign) != 0, rounding));
}

// Evaluate division FMA with conditional scaling. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
void div_fmas_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t condition, bool flush_output,
                        uint32_t *state);

// Evaluate division FMA with conditional scaling. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
void div_fmas_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t condition, bool flush_output,
                        uint32_t *state);

} // namespace rocjitsu::arith
