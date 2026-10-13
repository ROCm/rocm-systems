// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file gfx11_dot2.h
/// @brief Hardware-characterized GFX11 DOT2 arithmetic with F32 or packed outputs.

#include "dot_packed16.h"

#include <algorithm>
#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu {
namespace gfx11_dot2_detail {

inline constexpr uint32_t kFactorNan = 0xffc00a3d;
inline constexpr uint32_t kInvalidProductNan = 0xffc00000;
inline constexpr uint32_t kQuietNanBit = 0x00400000;
inline constexpr int kEmptyExponent = -1024;

// A nonnegative integer significand times 2^exponent. Keeping products in this
// representation also preserves BF16 products outside the FP32 exponent range.
struct Term {
  uint64_t significand;
  int exponent;
  bool negative;

  int leading_exponent() const {
    return significand ? exponent + int(std::bit_width(significand)) - 1 : kEmptyExponent;
  }
};

template <bool Bf16> struct Factor {
  static constexpr int fraction_bits = Bf16 ? 7 : 10;
  static constexpr int bias = Bf16 ? 127 : 15;
  static constexpr uint16_t fraction_mask = (1u << fraction_bits) - 1;
  static constexpr uint16_t infinity = Bf16 ? 0x7f80 : 0x7c00;
  uint16_t bits;

  bool nan() const { return (bits & 0x7fff) > infinity; }
  bool inf() const { return (bits & 0x7fff) == infinity; }
  bool negative() const { return bits >> 15; }
  unsigned exponent_field() const { return (bits & 0x7fff) >> fraction_bits; }
  uint64_t significand() const {
    if (exponent_field())
      return (1u << fraction_bits) | (bits & fraction_mask);
    return Bf16 ? 0 : bits & fraction_mask;
  }
  int alignment_exponent() const { return int(std::max(exponent_field(), 1u)) - bias; }
  Term product(Factor other) const {
    return {significand() * other.significand(),
            alignment_exponent() + other.alignment_exponent() - 2 * fraction_bits,
            negative() != other.negative()};
  }
};

inline uint64_t shift_round_even(uint64_t value, int shift) {
  if (shift <= 0)
    return value << -shift;
  if (shift >= 64)
    return 0;
  const uint64_t tail = value & ((uint64_t{1} << shift) - 1);
  const uint64_t halfway = uint64_t{1} << (shift - 1);
  return (value >> shift) + (tail > halfway || (tail == halfway && ((value >> shift) & 1)));
}

inline uint32_t pack(int64_t units, int grid, bool frame_negative) {
  if (!units)
    return 0;
  const uint32_t sign = uint32_t((units < 0) != frame_negative) << 31;
  const uint64_t magnitude = units < 0 ? uint64_t(-units) : uint64_t(units);
  const int top = int(std::bit_width(magnitude)) - 1;
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
  if (exponent < -126)
    return 0;
  if (exponent > 127)
    return sign | 0x7f800000u;
  return sign | (uint32_t(exponent + 127) << 23) | (uint32_t(significand) & 0x7fffff);
}
} // namespace gfx11_dot2_detail

/// @brief GFX11 DOT2 arithmetic with F32 or packed outputs, characterized on gfx1100.
/// @details Used by VOP3P DOT2, VOP2/VOPD DOT2ACC and F32/packed-output WMMA. See
/// https://github.com/ROCm/rocm-systems/issues/12056 for the finite arithmetic
/// model. In S2's sign frame an opposing term contributes ~m, not -m, after
/// truncation onto the alignment grid. In particular, signed zeros matter.
/// Integer arithmetic makes rounding and NaN bits independent of the host FP
/// environment. This helper is not the arithmetic policy for other targets.
/// Packed F16 uses a -14 floor for C alignment even when C is subnormal.
/// BF16 narrows the ordinary FP32-rounded result after every DOT2 step.
template <bool Bf16, bool Packed = false>
inline uint32_t gfx11_dot2_bits(uint16_t a0_bits, uint16_t b0_bits, uint16_t a1_bits,
                                uint16_t b1_bits, uint32_t acc, bool fp16_ovfl = false) {
  using namespace gfx11_dot2_detail;
  const auto special = [](uint32_t bits) -> uint32_t {
    if constexpr (Packed)
      return dot_packed16::special<Bf16>(bits);
    return bits;
  };
  const Factor<Bf16> a0{a0_bits}, b0{b0_bits}, a1{a1_bits}, b1{b1_bits};
  // Lane NaNs and invalid products take precedence over an accumulator NaN.
  if (a0.nan() || b0.nan() || a1.nan() || b1.nan())
    return special(kFactorNan);
  if ((a0.inf() && !b0.significand()) || (b0.inf() && !a0.significand()) ||
      (a1.inf() && !b1.significand()) || (b1.inf() && !a1.significand()))
    return special(kInvalidProductNan);
  bool positive_inf = false;
  bool negative_inf = false;
  const auto add_inf = [&](Factor<Bf16> left, Factor<Bf16> right) {
    if (left.inf() || right.inf()) {
      if (left.negative() != right.negative())
        negative_inf = true;
      else
        positive_inf = true;
    }
  };
  add_inf(a0, b0);
  add_inf(a1, b1);
  if (positive_inf && negative_inf)
    return special(kInvalidProductNan);
  if ((acc & 0x7fffffff) > 0x7f800000)
    return special(acc | kQuietNanBit);
  positive_inf |= acc == 0x7f800000;
  negative_inf |= acc == 0xff800000;
  if (positive_inf || negative_inf)
    return special(positive_inf && negative_inf ? kInvalidProductNan
                   : negative_inf               ? 0xff800000
                                                : 0x7f800000);

  const Term p0 = a0.product(b0), p1 = a1.product(b1);
  const unsigned acc_exp = (acc >> 23) & 255;
  const Term c{acc_exp ? (acc & 0x7fffff) | 0x800000u : 0, int(acc_exp) - 127 - 23,
               bool(acc >> 31)};
  if (!(c.significand | p0.significand | p1.significand))
    return 0;
  const int product_grid =
      std::max(p0.significand ? a0.alignment_exponent() + b0.alignment_exponent() : kEmptyExponent,
               p1.significand ? a1.alignment_exponent() + b1.alignment_exponent()
                              : kEmptyExponent) -
      24;
  const int grid = std::max(
      product_grid,
      std::max({Packed && !Bf16 ? std::max(-14, c.leading_exponent()) : c.leading_exponent(),
                p0.leading_exponent(), p1.leading_exponent()}) -
          26);
  int64_t total = 0;
  for (const Term t : {c, p0, p1}) {
    const int shift = t.exponent - grid;
    const uint64_t magnitude = !t.significand || shift <= -64 ? 0
                               : shift < 0                    ? t.significand >> -shift
                                                              : t.significand << shift;
    total += t.negative != c.negative ? -int64_t(magnitude) - 1 : int64_t(magnitude);
  }
  if constexpr (Packed) {
    if constexpr (Bf16)
      return pack(total, grid, c.negative) >> 16;
    else
      return dot_packed16::pack_f16(total, grid, c.negative, fp16_ovfl);
  }
  return pack(total, grid, c.negative);
}

template <bool Bf16>
inline uint32_t gfx11_dot2_f32(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t acc) {
  return gfx11_dot2_bits<Bf16>(a0, b0, a1, b1, acc);
}

/// @brief Widen packed C, execute one GFX11 DOT2 step and return packed result bits.
template <bool Bf16>
inline uint16_t gfx11_dot2_packed16(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1,
                                    uint16_t acc, bool fp16_ovfl = false) {
  return uint16_t(
      gfx11_dot2_bits<Bf16, true>(a0, b0, a1, b1, dot_packed16::widen<Bf16>(acc), fp16_ovfl));
}
} // namespace rocjitsu::amdgpu
