// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file gfx12_dot.h
/// @brief Best-effort GFX12 DOT2 and WMMA arithmetic inferred from local observations.
/// @details Based solely on local gfx1201 input/output captures. The grouping,
/// alignment, rounding and precedence rules describe this simulator model, not
/// known hardware internals or a vendor specification. Matching saved outputs
/// does not guarantee untested inputs, flags, modes or devices. See
/// tests/fixtures/float_dot/README.md for capture provenance and scope.

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

template <InputFormat Format> struct Factor {
  static constexpr bool is_f8 = Format == InputFormat::Fp8 || Format == InputFormat::Bf8;
  static constexpr bool is_bfloat = Format == InputFormat::Bf16 || Format == InputFormat::Bf8;
  static constexpr int fraction_bits = is_f8 ? (is_bfloat ? 2 : 3) : (is_bfloat ? 7 : 10);
  static constexpr int bias = is_f8 ? (is_bfloat ? 15 : 7) : (is_bfloat ? 127 : 15);
  static constexpr uint16_t fraction_mask = (1u << fraction_bits) - 1;
  static constexpr uint16_t infinity = is_f8 ? 0x7c : (is_bfloat ? 0x7f80 : 0x7c00);
  static constexpr uint16_t magnitude_mask = is_f8 ? 0x7f : 0x7fff;
  uint16_t bits;

  bool nan() const {
    if constexpr (is_f8 && !is_bfloat)
      return (bits & magnitude_mask) == 0x7f;
    return (bits & magnitude_mask) > infinity;
  }
  bool inf() const {
    if constexpr (is_f8 && !is_bfloat)
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
    if constexpr (is_bf16)
      return pack(total, grid) >> 16;
    else
      return dot_packed16::pack_f16(total, grid, false, fp16_ovfl);
  }
  return pack(total, grid);
}

} // namespace gfx12_dot_detail

/// @brief Best-effort GFX12 dot model with F32 or packed outputs.
/// @details In this model, DOT2 uses two products and each F16/BF16 WMMA step
/// uses four. These steps are inferred from local gfx1201 outputs. Pairs first
/// align and add products independently, then align those sums with C. F32 outputs
/// round once to FP32 using RNE. Packed F16 rounds the integer sum directly to F16
/// using RNE, with a C-alignment exponent floor of -14 and finite overflow controlled
/// by fp16_ovfl. For packed F16, a negative C with discarded bits contributes a
/// floor unit even when its aligned magnitude is zero, provided the grid is at or
/// below -14. Packed BF16 instead truncates the rounded FP32 result. Packed WMMA
/// applies this narrowing after every step. Subnormal inputs and outputs are
/// preserved. Integer arithmetic preserves observed NaN payloads and makes results
/// independent of the host FP state.
/// See tests/fixtures/float_dot/README.md for local observations related to
/// https://github.com/ROCm/rocm-systems/issues/12056 on RDNA4.
template <bool Bf16, bool Packed, std::size_t N>
inline uint32_t gfx12_dot_bits(const std::array<uint16_t, N> &a, const std::array<uint16_t, N> &b,
                               uint32_t acc, bool fp16_ovfl = false) {
  using gfx12_dot_detail::InputFormat;
  constexpr auto format = Bf16 ? InputFormat::Bf16 : InputFormat::F16;
  return gfx12_dot_detail::dot_bits<format, format, Packed>(a, b, acc, fp16_ovfl);
}

/// @brief Best-effort model for eight native FP8/BF8 products with F32 C and output.
/// @details Inferred solely from local gfx1201 outputs, not internal hardware
/// observations. In this model, two groups of four products align and add
/// independently, then align
/// with C. The result rounds once to F32 using RNE, independently of MODE.
/// Native subnormal exponent floors are preserved. Factor NaNs and invalid
/// operations involving BF8 infinities produce 0xffc00a3d; otherwise C's NaN
/// payload survives. See tests/fixtures/float_dot/README.md for qualification.
template <bool ABf8, bool BBf8>
inline uint32_t gfx12_dot8_f32(const std::array<uint8_t, 8> &a, const std::array<uint8_t, 8> &b,
                               uint32_t acc) {
  using gfx12_dot_detail::InputFormat;
  constexpr auto left = ABf8 ? InputFormat::Bf8 : InputFormat::Fp8;
  constexpr auto right = BBf8 ? InputFormat::Bf8 : InputFormat::Fp8;
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
