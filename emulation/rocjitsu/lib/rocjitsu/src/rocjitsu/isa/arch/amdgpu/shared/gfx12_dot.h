// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file gfx12_dot.h
/// @brief GFX12 dot arithmetic with F32 or packed outputs.
/// See tests/fixtures/float_dot/README.md for the model's empirical basis.

#include "dot_packed16.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu {
namespace gfx12_dot_detail {

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

enum class InputFormat { F16, Bf16, Fp8, Bf8 };

// Decode raw input fields without converting through a host floating-point type.
template <InputFormat Format> struct Factor {
  static constexpr bool is_f8 = Format == InputFormat::Fp8 || Format == InputFormat::Bf8;
  static constexpr bool is_bfloat = Format == InputFormat::Bf16 || Format == InputFormat::Bf8;
  static constexpr int fraction_bits = is_f8 ? (is_bfloat ? 2 : 3) : (is_bfloat ? 7 : 10);
  static constexpr int bias = is_f8 ? (is_bfloat ? 15 : 7) : (is_bfloat ? 127 : 15);
  static constexpr uint16_t fraction_mask = (1u << fraction_bits) - 1;
  static constexpr uint16_t infinity = is_f8 ? 0x7c : (is_bfloat ? 0x7f80 : 0x7c00);
  static constexpr uint16_t magnitude_mask = is_f8 ? 0x7f : 0x7fff;
  uint16_t bits;

  // FP8 E4M3 reserves only its largest magnitude for NaN and has no infinity.
  bool nan() const {
    if constexpr (Format == InputFormat::Fp8)
      return (bits & magnitude_mask) == 0x7f;
    return (bits & magnitude_mask) > infinity;
  }
  bool inf() const {
    if constexpr (Format == InputFormat::Fp8)
      return false;
    return (bits & magnitude_mask) == infinity;
  }
  bool negative() const { return bits >> (is_f8 ? 7 : 15); }
  unsigned exponent_field() const { return (bits & magnitude_mask) >> fraction_bits; }
  uint64_t significand() const {
    if (exponent_field())
      return (1u << fraction_bits) | (bits & fraction_mask);
    return bits & fraction_mask;
  }
  int alignment_exponent() const { return int(std::max(exponent_field(), 1u)) - bias; }
  template <InputFormat OtherFormat> Term product(Factor<OtherFormat> other) const {
    return {significand() * other.significand(),
            alignment_exponent() + other.alignment_exponent() - fraction_bits - other.fraction_bits,
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

inline uint32_t pack(int64_t units, int grid) {
  if (!units)
    return 0;
  const uint32_t sign = uint32_t(units < 0) << 31;
  const uint64_t magnitude = units < 0 ? uint64_t(-units) : uint64_t(units);
  const int top = int(std::bit_width(magnitude)) - 1;
  int exponent = top + grid;
  // Round directly on the subnormal grid to avoid double rounding.
  if (exponent < -126) {
    const uint64_t fraction = shift_round_even(magnitude, -149 - grid);
    return fraction ? sign | uint32_t(fraction) : 0;
  }
  uint64_t significand = shift_round_even(magnitude, top - 23);
  if (significand == 0x1000000) {
    significand >>= 1;
    ++exponent;
  }
  if (exponent > 127)
    return sign | 0x7f800000u;
  return sign | (uint32_t(exponent + 127) << 23) | (uint32_t(significand) & 0x7fffff);
}

// Alignment discards low bits by arithmetic floor. An accumulator too small
// to retain any magnitude bits contributes zero, including when negative.
inline int64_t align(Term term, int grid, bool accumulator = false) {
  const int shift = term.exponent - grid;
  const uint64_t magnitude = !term.significand || shift <= -64 ? 0
                             : shift < 0                       ? term.significand >> -shift
                                                               : term.significand << shift;
  const bool tail = term.significand && shift < 0 &&
                    (shift <= -64 || (term.significand & ((uint64_t{1} << -shift) - 1)));
  return term.negative ? -int64_t(magnitude) - (tail && (!accumulator || magnitude))
                       : int64_t(magnitude);
}

// Return the raw dot-product result, preserving format-specific special values
// and rounding. Packed results occupy the low 16 bits.
template <InputFormat LeftFormat, InputFormat RightFormat, bool Packed, std::size_t N, typename Raw>
inline uint32_t dot_bits(const std::array<Raw, N> &a, const std::array<Raw, N> &b, uint32_t acc,
                         bool fp16_ovfl = false) {
  constexpr bool is_f8 = Factor<LeftFormat>::is_f8;
  constexpr bool is_bf16 = LeftFormat == InputFormat::Bf16;
  static_assert(std::is_same_v<Raw, std::conditional_t<is_f8, uint8_t, uint16_t>>);
  static_assert(is_f8 == Factor<RightFormat>::is_f8);
  static_assert(is_f8 ? (N == 8 && !Packed) : (N == 2 || N == 4));
  static_assert(is_f8 || LeftFormat == RightFormat);
  const auto special = [](uint32_t bits) -> uint32_t {
    if constexpr (Packed)
      return dot_packed16::special<is_bf16>(bits);
    return bits;
  };
  std::array<Term, N> products;
  std::array<int, N> product_grids;
  int product_grid = kEmptyExponent;
  bool positive_inf = false;
  bool negative_inf = false;
  bool invalid = false;
  for (std::size_t i = 0; i < N; ++i) {
    const Factor<LeftFormat> left{a[i]};
    const Factor<RightFormat> right{b[i]};
    if (left.nan() || right.nan())
      return special(kFactorNan);
    invalid |= (left.inf() && !right.significand()) || (right.inf() && !left.significand());
    if (left.inf() || right.inf()) {
      if (left.negative() != right.negative())
        negative_inf = true;
      else
        positive_inf = true;
    }
    products[i] = left.product(right);
    product_grids[i] = products[i].significand
                           ? left.alignment_exponent() + right.alignment_exponent() - 24
                           : kEmptyExponent;
    product_grid = std::max(product_grid, product_grids[i]);
  }
  // Factor NaNs, then invalid products, precede an accumulator NaN.
  if (invalid || (positive_inf && negative_inf))
    return special(is_f8 ? kFactorNan : kInvalidProductNan);
  if ((acc & 0x7fffffff) > 0x7f800000)
    return special(acc | kQuietNanBit);
  positive_inf |= acc == 0x7f800000;
  negative_inf |= acc == 0xff800000;
  if (positive_inf || negative_inf)
    return special(positive_inf && negative_inf ? (is_f8 ? kFactorNan : kInvalidProductNan)
                   : negative_inf               ? 0xff800000
                                                : 0x7f800000);

  const unsigned acc_exp = (acc >> 23) & 255;
  const Term c{acc_exp ? (acc & 0x7fffff) | 0x800000u : acc & 0x7fffff,
               int(std::max(acc_exp, 1u)) - 127 - 23, bool(acc >> 31)};
  // Packed F16 floors C's alignment exponent at -14. A negative subnormal C
  // contributes an arithmetic-floor unit if the grid is at or below -14,
  // even when no magnitude bits survive. Coarser grids drop that tiny C.
  const int grid =
      std::max(product_grid, std::max(Packed && !is_bf16 ? -14 : -126, c.leading_exponent()) - 26);
  int64_t total = align(c, grid, !(Packed && !is_bf16 && grid <= -14));
  // Align each group on its own grid before combining it with C: pairs for
  // F16/BF16, groups of four for FP8/BF8. Flattening this loses rounding steps.
  constexpr std::size_t group_size = is_f8 ? 4 : 2;
  for (std::size_t i = 0; i < N; i += group_size) {
    int group_grid = kEmptyExponent;
    for (std::size_t j = i; j < i + group_size; ++j)
      group_grid = std::max(group_grid, product_grids[j]);
    int64_t group = 0;
    for (std::size_t j = i; j < i + group_size; ++j)
      group += align(products[j], group_grid);
    const Term sum{uint64_t(group < 0 ? -group : group), group_grid, group < 0};
    total += align(sum, grid);
  }
  // Group alignment retains at most 28 magnitude bits; the final sum and C
  // fit comfortably in int64_t. Zero, including underflow to zero, is positive.
  if constexpr (Packed) {
    // BF16 truncates a rounded F32 result; F16 rounds directly to its own grid.
    if constexpr (is_bf16)
      return pack(total, grid) >> 16;
    else
      return dot_packed16::pack_f16(total, grid, false, fp16_ovfl);
  }
  return pack(total, grid);
}

} // namespace gfx12_dot_detail

/// @brief Evaluate two or four F16/BF16 products with an F32 or packed accumulator.
/// @details Products are aligned in pairs before adding C, with nearest-even
/// rounding. Return raw F32 bits, or the packed result in the low 16 bits.
/// Subnormals and the model's NaN payload rules are preserved; fp16_ovfl controls
/// finite F16 overflow. Guest and host rounding controls do not affect the result.
template <bool Bf16, bool Packed, std::size_t N>
inline uint32_t gfx12_dot_bits(const std::array<uint16_t, N> &a, const std::array<uint16_t, N> &b,
                               uint32_t acc, bool fp16_ovfl = false) {
  using gfx12_dot_detail::InputFormat;
  constexpr auto format = Bf16 ? InputFormat::Bf16 : InputFormat::F16;
  return gfx12_dot_detail::dot_bits<format, format, Packed>(a, b, acc, fp16_ovfl);
}

/// @brief Evaluate eight native FP8/BF8 products plus F32 C, returning raw F32 bits.
/// @details Align products in groups of four before adding C and round to nearest-even
/// independently of guest MODE and host rounding controls. Factor NaNs and invalid
/// products yield 0xffc00a3d;
/// otherwise an accumulator NaN retains its payload. See gfx12_dot_bits for the
/// corresponding F16/BF16 grouped-arithmetic contract.
template <bool A_FP8, bool B_FP8>
inline uint32_t gfx12_dot8_f32(const std::array<uint8_t, 8> &a, const std::array<uint8_t, 8> &b,
                               uint32_t acc) {
  using gfx12_dot_detail::InputFormat;
  constexpr auto left = A_FP8 ? InputFormat::Fp8 : InputFormat::Bf8;
  constexpr auto right = B_FP8 ? InputFormat::Fp8 : InputFormat::Bf8;
  return gfx12_dot_detail::dot_bits<left, right, false>(a, b, acc);
}

template <bool Bf16, std::size_t N>
inline uint32_t gfx12_dot_f32(const std::array<uint16_t, N> &a, const std::array<uint16_t, N> &b,
                              uint32_t acc) {
  return gfx12_dot_bits<Bf16, false>(a, b, acc);
}

/// @brief Widen packed C, execute one GFX12 DOT4 step and return packed result bits.
template <bool Bf16>
inline uint16_t gfx12_dot4_packed16(const std::array<uint16_t, 4> &a,
                                    const std::array<uint16_t, 4> &b, uint16_t acc,
                                    bool fp16_ovfl = false) {
  return uint16_t(gfx12_dot_bits<Bf16, true>(a, b, dot_packed16::widen<Bf16>(acc), fp16_ovfl));
}

/// @brief Two-product wrapper for GFX12 DOT2 and VOPD DOT2ACC.
template <bool Bf16>
inline uint32_t gfx12_dot2_f32(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t acc) {
  return gfx12_dot_f32<Bf16>(std::array{a0, a1}, std::array{b0, b1}, acc);
}

} // namespace rocjitsu::amdgpu
