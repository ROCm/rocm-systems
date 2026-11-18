// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file division.h
/// @brief Division instruction semantics with explicit guest FP controls.
/// Rounding: 0 nearest-even, 1 positive infinity, 2 negative infinity, 3 zero.
/// Denorm bits: bit 0 preserves inputs; bit 1 preserves outputs. FMAS always
/// preserves its inputs, and FIXUP preserves its provisional quotient.

#include "util/big_int.h"

#include <bit>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu {

/// @brief Format fields used by division numerics and exception classification.
/// Pre-scaling tests encoded exponents, including zero for subnormals, rather
/// than the normalized exponents returned by frexp.
template <typename Float> struct DivisionFormat {
  static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>);
  using Bits = std::conditional_t<std::is_same_v<Float, float>, uint32_t, uint64_t>;
  static constexpr int fraction = sizeof(Float) == 4 ? 23 : 52;
  static constexpr int bias = sizeof(Float) == 4 ? 127 : 1023;
  static constexpr int threshold = sizeof(Float) == 4 ? 96 : 768;
  static constexpr int scale = sizeof(Float) == 4 ? 64 : 128;
  static constexpr Bits sign = Bits{1} << (sizeof(Float) * 8 - 1);
  static constexpr Bits infinity = ((Bits{1} << (sizeof(Float) == 4 ? 8 : 11)) - 1) << fraction;
  static constexpr Bits quiet = Bits{1} << (fraction - 1);
  static constexpr Bits fraction_mask = (Bits{1} << fraction) - 1;
  static int exponent(Bits bits) { return static_cast<int>((bits & infinity) >> fraction); }
};

inline int div_top_bit(util::uint128_t value) {
  const uint64_t high = static_cast<uint64_t>(value >> 64);
  return high ? 127 - std::countl_zero(high) : 63 - std::countl_zero(static_cast<uint64_t>(value));
}

inline util::uint128_t div_shift_right_jam(util::uint128_t value, int distance) {
  if (distance <= 0)
    return value;
  if (distance >= 128)
    return value != 0;
  return (value >> distance) | ((value << (128 - distance)) != 0);
}

template <typename Float> inline Float div_overflow(bool negative, uint32_t rounding) {
  using F = DivisionFormat<Float>;
  using Bits = typename F::Bits;
  const bool to_infinity =
      rounding == 0 || (rounding == 1 && !negative) || (rounding == 2 && negative);
  return std::bit_cast<Float>(Bits((negative ? F::sign : 0) + F::infinity - !to_infinity));
}

// Round an integer significand times 2^exponent exactly once. No floating-point
// arithmetic or host rounding/flush controls participate, including at underflow.
template <typename Float>
inline Float div_round(util::uint128_t significand, int exponent, bool negative, uint32_t rounding,
                       uint32_t denorm) {
  using F = DivisionFormat<Float>;
  using Bits = typename F::Bits;
  const Bits sign = negative ? F::sign : 0;
  if (significand == 0)
    return std::bit_cast<Float>(sign);
  const int highest = div_top_bit(significand);
  if (highest + exponent > F::bias)
    return div_overflow<Float>(negative, rounding);
  const int minimum = 1 - F::bias - F::fraction;
  const int normal_shift = highest - F::fraction;
  const int subnormal_shift = minimum - exponent;
  const int shift = normal_shift > subnormal_shift ? normal_shift : subnormal_shift;
  util::uint128_t kept;
  bool guard = false, sticky = false;
  if (shift <= 0) {
    kept = significand << -shift;
  } else {
    kept = shift >= 128 ? 0 : significand >> shift;
    guard = shift <= 128 && ((significand >> (shift - 1)) & 1) != 0;
    sticky = shift > 128 ? significand != 0 : shift > 1 && (significand << (129 - shift)) != 0;
  }
  const bool inexact = guard || sticky;
  if ((rounding == 0 && guard && (sticky || (kept & 1))) ||
      (rounding == 1 && !negative && inexact) || (rounding == 2 && negative && inexact))
    ++kept;
  int result_exponent = exponent + shift + F::fraction;
  if (kept >= (util::uint128_t{1} << (F::fraction + 1))) {
    kept >>= 1;
    ++result_exponent;
  }
  if (result_exponent > F::bias)
    return div_overflow<Float>(negative, rounding);
  if (kept < (util::uint128_t{1} << F::fraction)) {
    if (!(denorm & 2u))
      kept = 0;
    return std::bit_cast<Float>(Bits(sign | static_cast<Bits>(kept)));
  }
  return std::bit_cast<Float>(Bits(sign | (Bits(result_exponent + F::bias) << F::fraction) |
                                   (static_cast<Bits>(kept) & F::fraction_mask)));
}

/// @brief Pre-scaled operand and the flag consumed by DIV_FMAS.
template <typename Float> struct DivisionScaleResult {
  Float value;
  bool post_scale;
};

/// @brief Pre-scale a numerator or denominator for the division macro.
/// The finite-input rules and source-2 FMAS selector were checked on physical
/// gfx1100/gfx1201. The public pseudocode's DENORM predicates and fixed FMAS
/// factors do not reproduce these instructions or LLVM division macros.
/// SCALE requires source 0 to equal the numerator or denominator, as in the ISA.
template <typename Float>
inline DivisionScaleResult<Float> div_scale(Float value, Float denominator, Float numerator,
                                            uint32_t rounding = 0, uint32_t denorm = 3,
                                            bool quiet_nan = true) {
  using F = DivisionFormat<Float>;
  using Bits = typename F::Bits;
  Bits vbits = std::bit_cast<Bits>(value), dbits = std::bit_cast<Bits>(denominator),
       nbits = std::bit_cast<Bits>(numerator);
  if (!(denorm & 1u)) {
    if (F::exponent(vbits) == 0)
      vbits &= F::sign;
    if (F::exponent(dbits) == 0)
      dbits &= F::sign;
    if (F::exponent(nbits) == 0)
      nbits &= F::sign;
  }
  const Bits d = dbits & ~F::sign, n = nbits & ~F::sign;
  const int de = F::exponent(d), ne = F::exponent(n);
  const int delta = ne - de;
  int adjustment = 0;
  bool post_scale = false;
  const bool is_denominator = vbits == dbits;
  if (delta >= F::threshold) {
    adjustment = is_denominator ? F::scale : 0;
    post_scale = true;
  } else if (de == 0) {
    adjustment = F::scale;
  } else if (de >= 2 * F::bias - 1) {
    post_scale = delta <= -F::threshold;
    adjustment = post_scale && !is_denominator ? 0 : -F::scale;
  } else if (delta <= -F::threshold) {
    adjustment = is_denominator ? 0 : F::scale;
    post_scale = true;
  } else if (ne <= (sizeof(Float) == 4 ? 24 : 53)) {
    adjustment = F::scale;
  }
  if (d == 0 || n == 0)
    return {std::bit_cast<Float>(Bits(F::sign | F::infinity | F::quiet)), post_scale};
  if ((vbits & ~F::sign) >= F::infinity)
    return {std::bit_cast<Float>(
                Bits(vbits | (quiet_nan && (vbits & F::fraction_mask) ? F::quiet : 0))),
            post_scale};
  const int ve = F::exponent(vbits);
  const Bits significand = (vbits & F::fraction_mask) | (ve ? Bits{1} << F::fraction : 0);
  return {div_round<Float>(significand, (ve ? ve : 1) - F::bias - F::fraction + adjustment,
                           (vbits & F::sign) != 0, rounding, denorm),
          post_scale};
}

/// @brief Fuse the multiply-add and post-scale with one final rounding.
/// Retain the complete product (up to 106 bits) and enough alignment bits for
/// exact cancellation. Widely separated terms contribute a sticky bit. Scaling
/// is fused with the final rounding, avoiding premature overflow and double
/// rounding when the quotient becomes subnormal. FMAS preserves input denormals.
template <typename Float>
inline Float div_fmas(Float first, Float second, Float third, bool post_scale, uint32_t rounding,
                      uint32_t denorm) {
  using F = DivisionFormat<Float>;
  using Bits = typename F::Bits;
  const Bits a = std::bit_cast<Bits>(first), b = std::bit_cast<Bits>(second),
             c = std::bit_cast<Bits>(third);
  const Bits am = a & ~F::sign, bm = b & ~F::sign, cm = c & ~F::sign;
  if (am > F::infinity)
    return std::bit_cast<Float>(Bits(a | F::quiet));
  if (bm > F::infinity)
    return std::bit_cast<Float>(Bits(b | F::quiet));
  const bool product_negative = ((a ^ b) & F::sign) != 0;
  const bool addend_negative = (c & F::sign) != 0;
  if ((am == F::infinity && bm == 0) || (bm == F::infinity && am == 0) ||
      ((am == F::infinity || bm == F::infinity) && cm == F::infinity &&
       product_negative != addend_negative))
    return std::bit_cast<Float>(Bits(F::sign | F::infinity | F::quiet));
  if (cm > F::infinity)
    return std::bit_cast<Float>(Bits(c | F::quiet));
  if (am == F::infinity || bm == F::infinity)
    return std::bit_cast<Float>(Bits((product_negative ? F::sign : 0) | F::infinity));
  if (cm == F::infinity)
    return third;
  const int ae = F::exponent(a), be = F::exponent(b), ce = F::exponent(c);
  const Bits as = (a & F::fraction_mask) | (ae ? Bits{1} << F::fraction : 0);
  const Bits bs = (b & F::fraction_mask) | (be ? Bits{1} << F::fraction : 0);
  const Bits cs = (c & F::fraction_mask) | (ce ? Bits{1} << F::fraction : 0);
  util::uint128_t product = util::uint128_t{as} * bs, addend = cs;
  const int pe = (ae ? ae : 1) + (be ? be : 1) - 2 * (F::bias + F::fraction);
  const int se = (ce ? ce : 1) - F::bias - F::fraction;
  const int adjustment = post_scale ? (ce > F::bias ? F::scale : -F::scale) : 0;
  if (product == 0 && addend == 0) {
    const bool negative = product_negative == addend_negative ? product_negative : rounding == 2;
    return std::bit_cast<Float>(Bits(negative ? F::sign : 0));
  }
  if (product == 0)
    return div_round<Float>(addend, se + adjustment, addend_negative, rounding, denorm);
  if (addend == 0)
    return div_round<Float>(product, pe + adjustment, product_negative, rounding, denorm);
  const int pt = div_top_bit(product), ct = div_top_bit(addend);
  const int exponent = pe + pt > se + ct ? pe + pt : se + ct;
  product = div_shift_right_jam(product << (126 - pt), exponent - pe - pt);
  addend = div_shift_right_jam(addend << (126 - ct), exponent - se - ct);
  const bool negative = product_negative == addend_negative ? product_negative
                        : product > addend                  ? product_negative
                        : product < addend                  ? addend_negative
                                                            : rounding == 2;
  const util::uint128_t sum = product_negative == addend_negative ? product + addend
                              : product >= addend                 ? product - addend
                                                                  : addend - product;
  return div_round<Float>(sum, exponent - 126 + adjustment, negative, rounding, denorm);
}

/// @brief Repair quotient sign and exceptional cases using the original inputs.
template <typename Float>
inline Float div_fixup(Float quotient, Float denominator, Float numerator, uint32_t rounding,
                       uint32_t denorm) {
  using F = DivisionFormat<Float>;
  using Bits = typename F::Bits;
  const Bits p = std::bit_cast<Bits>(quotient);
  Bits d = std::bit_cast<Bits>(denominator), n = std::bit_cast<Bits>(numerator);
  if (!(denorm & 1u)) {
    if (F::exponent(d) == 0)
      d &= F::sign;
    if (F::exponent(n) == 0)
      n &= F::sign;
  }
  const Bits dm = d & ~F::sign, nm = n & ~F::sign;
  const Bits sign = (d ^ n) & F::sign;
  if (nm > F::infinity)
    return std::bit_cast<Float>(Bits(n | F::quiet));
  if (dm > F::infinity)
    return std::bit_cast<Float>(Bits(d | F::quiet));
  if ((dm == 0 && nm == 0) || (dm == F::infinity && nm == F::infinity))
    return std::bit_cast<Float>(Bits(F::sign | F::infinity | F::quiet));
  if (dm == 0 || nm == F::infinity)
    return std::bit_cast<Float>(Bits(sign | F::infinity));
  if (nm == 0 || dm == F::infinity)
    return std::bit_cast<Float>(sign);
  if (F::exponent(n) - F::exponent(d) < -(F::bias + F::fraction))
    return div_round<Float>(1, -2 * F::bias - 2 * F::fraction, sign != 0, rounding, denorm);
  if ((p & ~F::sign) >= F::infinity)
    return div_overflow<Float>(sign != 0, rounding);
  return std::bit_cast<Float>(Bits(sign | (p & ~F::sign)));
}

/// @brief Half-precision FIXUP in the exact promoted F32 domain.
/// Nonfinite provisional values use the guest overflow result for finite
/// operands. The F32/F64 extreme-exponent underflow shortcut does not apply.
inline float div_fixup_f16(float quotient, float denominator, float numerator,
                           uint32_t rounding = 0, uint32_t denorm = 3) {
  uint32_t denominator_bits = std::bit_cast<uint32_t>(denominator);
  uint32_t numerator_bits = std::bit_cast<uint32_t>(numerator);
  if (!(denorm & 1u)) {
    // Inputs are promoted F16, whose minimum normal has F32 bits 0x38800000.
    if ((denominator_bits & 0x7fffffffu) < 0x38800000u)
      denominator_bits &= 0x80000000u;
    if ((numerator_bits & 0x7fffffffu) < 0x38800000u)
      numerator_bits &= 0x80000000u;
  }
  const uint32_t denominator_magnitude = denominator_bits & 0x7fffffffu;
  const uint32_t numerator_magnitude = numerator_bits & 0x7fffffffu;
  const uint32_t sign = (denominator_bits ^ numerator_bits) & 0x80000000u;
  if (numerator_magnitude > 0x7f800000u)
    return std::bit_cast<float>(numerator_bits | 0x00400000u);
  if (denominator_magnitude > 0x7f800000u)
    return std::bit_cast<float>(denominator_bits | 0x00400000u);
  if ((denominator_magnitude == 0 && numerator_magnitude == 0) ||
      (denominator_magnitude == 0x7f800000u && numerator_magnitude == 0x7f800000u))
    return std::bit_cast<float>(0xffc00000u);
  if (denominator_magnitude == 0 || numerator_magnitude == 0x7f800000u)
    return std::bit_cast<float>(sign | 0x7f800000u);
  if (numerator_magnitude == 0 || denominator_magnitude == 0x7f800000u)
    return std::bit_cast<float>(sign);
  const uint32_t magnitude = std::bit_cast<uint32_t>(quotient) & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) {
    const bool to_infinity = rounding == 0 || (rounding == 1 && !sign) || (rounding == 2 && sign);
    return std::bit_cast<float>(sign | (to_infinity ? 0x7f800000u : 0x477fe000u));
  }
  return std::bit_cast<float>(magnitude | sign);
}

/// @brief Apply FIXUP OMOD using guest rounding only when scaling overflows.
/// Hardware flushes a subnormal provisional result before scaling. Scaling a
/// normal into the subnormal range flushes before rounding and preserves sign;
/// an existing zero or subnormal instead becomes positive zero. Checked on gfx1201.
template <typename Float>
inline Float div_apply_omod(Float value, uint32_t rounding, uint32_t omod) {
  using F = DivisionFormat<Float>;
  using Bits = typename F::Bits;
  const Bits bits = std::bit_cast<Bits>(value);
  if (omod == 0 || (bits & ~F::sign) >= F::infinity)
    return value;
  const int exponent = F::exponent(bits);
  if (exponent == 0)
    return Float{0};
  const int adjusted = exponent + (omod == 3 ? -1 : static_cast<int>(omod));
  if (adjusted <= 0)
    return std::bit_cast<Float>(Bits(bits & F::sign));
  if (adjusted >= 2 * F::bias + 1)
    return div_overflow<Float>((bits & F::sign) != 0, rounding);
  return std::bit_cast<Float>(Bits((bits & ~F::infinity) | (Bits(adjusted) << F::fraction)));
}

} // namespace rocjitsu::amdgpu
