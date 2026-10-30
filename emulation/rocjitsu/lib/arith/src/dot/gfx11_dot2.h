// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The borrowed gfx11 DOT2 integer-grid model for FP32 results.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

// Adapted from rocjitsu 39fd0ea559a shared/gfx11_dot2.h.
// Empirical evidence: rocjitsu/tests/fixtures/float_dot/README.md.

#include "common/bits.h"
#include <cstdint>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
namespace gfx11_dot2_detail {

inline constexpr uint32_t kFactorNan = 0xffc00a3d;
inline constexpr uint32_t kInvalidProductNan = 0xffc00000;
inline constexpr uint32_t kQuietNanBit = 0x00400000;
inline constexpr int kEmptyExponent = -1024;

// A nonnegative integer significand times 2^exponent. Keeping products in this
// representation also preserves BF16 products outside the FP32 exponent range.
class Term {
public:
  uint64_t significand;
  int exponent;
  bool negative;

  // Return the leading binary exponent, or the model's empty-exponent sentinel for zero.
  int leading_exponent() const {
    return significand ? exponent + int(rocjitsu::arith::bit_width(significand)) - 1
                       : kEmptyExponent;
  }
};

template <bool Bf16> class Factor {
public:
  static constexpr int fraction_bits = Bf16 ? 7 : 10;
  static constexpr int bias = Bf16 ? 127 : 15;
  static constexpr uint16_t fraction_mask = (1u << fraction_bits) - 1;
  static constexpr uint16_t infinity = Bf16 ? 0x7f80 : 0x7c00;
  uint16_t bits;

  // Return whether the encoded value is a NaN.
  bool nan() const { return (bits & 0x7fff) > infinity; }

  // Return whether the encoded value is an infinity.
  bool inf() const { return (bits & 0x7fff) == infinity; }

  // Return whether the encoded value has its sign bit set.
  bool negative() const { return bits >> 15; }

  // Extract the encoded exponent field.
  unsigned exponent_field() const { return (bits & 0x7fff) >> fraction_bits; }

  // Return the integer significand, including the implicit leading bit for normal values.
  uint64_t significand() const {
    if (exponent_field()) {
      return (1u << fraction_bits) | (bits & fraction_mask);
    }
    return Bf16 ? 0 : bits & fraction_mask;
  }

  // Return the binary exponent used to align this factor on the accumulation grid.
  int alignment_exponent() const {
    return int(rocjitsu::arith::maximum(exponent_field(), 1u)) - bias;
  }

  // Form a gfx11 product term from two decoded factors, retaining its grid exponent.
  Term product(Factor other) const {
    return {significand() * other.significand(),
            alignment_exponent() + other.alignment_exponent() - 2 * fraction_bits,
            negative() != other.negative()};
  }
};

// Scale an unsigned integer by a binary shift, rounding discarded bits to nearest-even.
inline uint64_t shift_round_even(uint64_t value, int shift) {
  if (shift <= 0) {
    return value << -shift;
  }
  if (shift >= 64) {
    return 0;
  }
  const uint64_t tail = value & ((uint64_t{1} << shift) - 1);
  const uint64_t halfway = uint64_t{1} << (shift - 1);
  return (value >> shift) + (tail > halfway || (tail == halfway && ((value >> shift) & 1)));
}

// Pack signed grid units into FP32 using gfx11 rounding and the supplied zero-sign frame.
inline uint32_t pack(int64_t units, int grid, bool frame_negative) {
  if (!units) {
    return 0;
  }
  const uint32_t sign = uint32_t((units < 0) != frame_negative) << 31;
  const uint64_t magnitude = units < 0 ? uint64_t(-units) : uint64_t(units);
  const int top = int(rocjitsu::arith::bit_width(magnitude)) - 1;
  int exponent = top + grid;
  uint64_t significand = shift_round_even(magnitude, top - 23);
  if (significand == 0x1000000) {
    significand >>= 1;
    ++exponent;
  }
  // Round a normalized 24-bit significand before checking the exponent.
  // Gradual-underflow rounding can incorrectly round a tiny value up to the
  // smallest normal; simply flushing before rounding also disagrees at that
  // boundary when normalization's rounding itself carries into exponent -126.
  if (exponent < -126) {
    return 0;
  }
  if (exponent > 127) {
    return sign | 0x7f800000u;
  }
  return sign | (uint32_t(exponent + 127) << 23) | (uint32_t(significand) & 0x7fffff);
}
} // namespace gfx11_dot2_detail

// Return the empirical gfx11 FP32 result of two FP16/BF16 products plus acc.
// In C's sign frame, an opposing term contributes ~m rather than -m after
// grid truncation; signed zeros therefore matter. This policy is gfx11-specific.
template <bool Bf16>
inline uint32_t gfx11_dot2_f32(uint16_t a0_bits, uint16_t b0_bits, uint16_t a1_bits,
                               uint16_t b1_bits, uint32_t acc) {
  using namespace gfx11_dot2_detail;
  const Factor<Bf16> a0{a0_bits}, b0{b0_bits}, a1{a1_bits}, b1{b1_bits};

  // Lane NaNs and invalid products take precedence over an accumulator NaN.
  if (a0.nan() || b0.nan() || a1.nan() || b1.nan()) {
    return kFactorNan;
  }
  if ((a0.inf() && !b0.significand()) || (b0.inf() && !a0.significand()) ||
      (a1.inf() && !b1.significand()) || (b1.inf() && !a1.significand())) {
    return kInvalidProductNan;
  }
  bool positive_inf = false;
  bool negative_inf = false;
  const auto add_inf = [&](Factor<Bf16> left, Factor<Bf16> right) {
    if (left.inf() || right.inf()) {
      if (left.negative() != right.negative()) {
        negative_inf = true;
      } else {
        positive_inf = true;
      }
    }
  };
  add_inf(a0, b0);
  add_inf(a1, b1);
  if (positive_inf && negative_inf) {
    return kInvalidProductNan;
  }
  if ((acc & 0x7fffffff) > 0x7f800000) {
    return acc | kQuietNanBit;
  }
  positive_inf |= acc == 0x7f800000;
  negative_inf |= acc == 0xff800000;
  if (positive_inf || negative_inf) {
    return positive_inf && negative_inf ? kInvalidProductNan
           : negative_inf               ? 0xff800000
                                        : 0x7f800000;
  }

  const Term p0 = a0.product(b0), p1 = a1.product(b1);
  const unsigned acc_exp = (acc >> 23) & 255;
  const Term c{acc_exp ? (acc & 0x7fffff) | 0x800000u : 0, int(acc_exp) - 127 - 23,
               bool(acc >> 31)};
  if (!(c.significand | p0.significand | p1.significand)) {
    return 0;
  }
  const int product_grid =
      rocjitsu::arith::maximum(
          p0.significand ? a0.alignment_exponent() + b0.alignment_exponent() : kEmptyExponent,
          p1.significand ? a1.alignment_exponent() + b1.alignment_exponent() : kEmptyExponent) -
      24;
  const int grid = rocjitsu::arith::maximum(
      product_grid, rocjitsu::arith::maximum(
                        c.leading_exponent(),
                        rocjitsu::arith::maximum(p0.leading_exponent(), p1.leading_exponent())) -
                        26);
  int64_t total = 0;
  const Term terms[] = {c, p0, p1};
  for (const Term t : terms) {
    const int shift = t.exponent - grid;
    const uint64_t magnitude = !t.significand || shift <= -64 ? 0
                               : shift < 0                    ? t.significand >> -shift
                                                              : t.significand << shift;
    total += t.negative != c.negative ? -int64_t(magnitude) - 1 : int64_t(magnitude);
  }
  return pack(total, grid, c.negative);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
