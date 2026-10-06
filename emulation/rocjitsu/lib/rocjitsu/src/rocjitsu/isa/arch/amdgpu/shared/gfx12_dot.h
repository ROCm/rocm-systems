// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file gfx12_dot.h
/// @brief Hardware-characterized GFX12 DOT2 and WMMA arithmetic with F32 or packed outputs.

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

/// @brief Factor encoding: exponent and fraction widths. OCP E4M3 has no
/// infinity; S.1111.111 is its only NaN.
template <int ExponentBits, int FractionBits, bool OcpE4m3 = false> struct FactorFormat {
  static constexpr int exponent_bits = ExponentBits;
  static constexpr int fraction_bits = FractionBits;
  static constexpr int bias = (1 << (ExponentBits - 1)) - 1;
  static constexpr bool ocp_e4m3 = OcpE4m3;
};
using F16Factor = FactorFormat<5, 10>;
using Bf16Factor = FactorFormat<8, 7>;
using Fp8Factor = FactorFormat<4, 3, true>;
using Bf8Factor = FactorFormat<5, 2>;
template <bool Bf16> using HalfFactor = std::conditional_t<Bf16, Bf16Factor, F16Factor>;

template <typename Format> struct Factor {
  static constexpr int fraction_bits = Format::fraction_bits;
  static constexpr unsigned fraction_mask = (1u << fraction_bits) - 1;
  static constexpr unsigned exponent_max = (1u << Format::exponent_bits) - 1;
  uint16_t bits;

  unsigned exponent_field() const { return (bits >> fraction_bits) & exponent_max; }
  unsigned fraction() const { return bits & fraction_mask; }
  bool negative() const { return (bits >> (Format::exponent_bits + fraction_bits)) & 1; }
  bool nan() const {
    if (exponent_field() != exponent_max)
      return false;
    return Format::ocp_e4m3 ? fraction() == fraction_mask : fraction() != 0;
  }
  bool inf() const {
    return !Format::ocp_e4m3 && exponent_field() == exponent_max && fraction() == 0;
  }
  uint64_t significand() const {
    return exponent_field() ? (1u << fraction_bits) | fraction() : fraction();
  }
  int alignment_exponent() const { return int(std::max(exponent_field(), 1u)) - Format::bias; }
  template <typename Other> Term product(Factor<Other> other) const {
    return {significand() * other.significand(),
            alignment_exponent() + other.alignment_exponent() - fraction_bits -
                Factor<Other>::fraction_bits,
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

// Alignment discards low bits by arithmetic floor. With `accumulator` set, a
// term too small to retain any magnitude bits contributes zero, even when
// negative.
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
} // namespace gfx12_dot_detail

/// @brief GFX12 dot arithmetic with F32 or packed outputs, characterized on gfx1201.
/// @details DOT2 uses two products; each F16/BF16 WMMA step uses four. Pairs first
/// align and add products independently, then align those sums with C. F32 outputs
/// round once to FP32 using RNE. Packed F16 rounds the integer sum directly to F16
/// using RNE, with a C-alignment exponent floor of -14 and finite overflow controlled
/// by fp16_ovfl. A negative C with discarded bits contributes a floor unit even
/// when its aligned magnitude is zero, provided the grid is at or below the C
/// alignment floor (-14 for packed F16, -126 otherwise). Packed BF16 instead
/// truncates the rounded FP32 result. Packed WMMA applies this narrowing after
/// every step. Subnormal inputs and outputs are preserved. Integer arithmetic
/// preserves observed NaN payloads and makes results independent of the host FP
/// state. FP8/BF8 DOT4 differs in two ways: its four products align together
/// on one grid, and an invalid product returns the factor-NaN payload.
/// See tests/fixtures/float_dot/README.md for hardware qualification of
/// https://github.com/ROCm/rocm-systems/issues/12056 on RDNA4.
template <typename A, typename B, bool Packed, std::size_t N>
inline uint32_t gfx12_dot_terms(const std::array<uint16_t, N> &a, const std::array<uint16_t, N> &b,
                                uint32_t acc, bool fp16_ovfl = false) {
  static_assert(N == 2 || N == 4);
  using namespace gfx12_dot_detail;
  // 8-bit factors align all products on one grid, and an invalid product
  // returns the factor-NaN payload. 16-bit factors align products in pairs.
  constexpr bool kByteFactors = A::exponent_bits + A::fraction_bits < 8;
  constexpr std::size_t kGroup = kByteFactors ? N : 2;
  constexpr uint32_t kInvalid = kByteFactors ? kFactorNan : kInvalidProductNan;
  constexpr bool Bf16 = std::is_same_v<A, Bf16Factor>;
  static_assert(!Packed || (std::is_same_v<A, B> && !kByteFactors));
  const auto special = [](uint32_t bits) -> uint32_t {
    if constexpr (Packed)
      return dot_packed16::special<Bf16>(bits);
    return bits;
  };
  std::array<Term, N> products;
  std::array<int, N> product_grids;
  int product_grid = kEmptyExponent;
  bool positive_inf = false;
  bool negative_inf = false;
  bool invalid = false;
  for (std::size_t i = 0; i < N; ++i) {
    const Factor<A> left{a[i]};
    const Factor<B> right{b[i]};
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
    return special(kInvalid);
  if ((acc & 0x7fffffff) > 0x7f800000)
    return special(acc | kQuietNanBit);
  positive_inf |= acc == 0x7f800000;
  negative_inf |= acc == 0xff800000;
  if (positive_inf || negative_inf)
    return special(positive_inf && negative_inf ? kInvalid
                   : negative_inf               ? 0xff800000
                                                : 0x7f800000);

  const unsigned acc_exp = (acc >> 23) & 255;
  const Term c{acc_exp ? (acc & 0x7fffff) | 0x800000u : acc & 0x7fffff,
               int(std::max(acc_exp, 1u)) - 127 - 23, bool(acc >> 31)};
  // C aligns no lower than -14 for packed F16 and -126 otherwise. When the
  // grid is at or below that floor, a negative C contributes an
  // arithmetic-floor unit even when no magnitude bits survive (seen on
  // gfx1201 for subnormal C in F32 DOT2 and packed F16). Coarser grids drop
  // that tiny C.
  const int c_floor = Packed && !Bf16 ? -14 : -126;
  const int grid = std::max(product_grid, std::max(c_floor, c.leading_exponent()) - 26);
  int64_t total = align(c, grid, grid > c_floor);
  for (std::size_t i = 0; i < N; i += kGroup) {
    const int group_grid =
        *std::max_element(product_grids.begin() + i, product_grids.begin() + i + kGroup);
    int64_t group = 0;
    for (std::size_t j = i; j < i + kGroup; ++j)
      group += align(products[j], group_grid);
    total += align(Term{uint64_t(group < 0 ? -group : group), group_grid, group < 0}, grid);
  }
  // Group alignment retains at most 28 magnitude bits; the final sum and C
  // fit comfortably in int64_t. Zero, including underflow to zero, is positive.
  if constexpr (Packed) {
    if constexpr (Bf16)
      return pack(total, grid) >> 16;
    else
      return dot_packed16::pack_f16(total, grid, false, fp16_ovfl);
  }
  return pack(total, grid);
}

template <bool Bf16, bool Packed, std::size_t N>
inline uint32_t gfx12_dot_bits(const std::array<uint16_t, N> &a, const std::array<uint16_t, N> &b,
                               uint32_t acc, bool fp16_ovfl = false) {
  using Half = gfx12_dot_detail::HalfFactor<Bf16>;
  return gfx12_dot_terms<Half, Half, Packed>(a, b, acc, fp16_ovfl);
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

/// @brief Widen packed C, execute one GFX12 DOT2 step and return packed result bits.
/// @details This is also the arithmetic of VOP3 V_DOT2_F16_F16 and V_DOT2_BF16_BF16.
template <bool Bf16>
inline uint16_t gfx12_dot2_packed16(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1,
                                    uint16_t acc, bool fp16_ovfl = false) {
  return uint16_t(gfx12_dot_bits<Bf16, true>(std::array{a0, a1}, std::array{b0, b1},
                                             dot_packed16::widen<Bf16>(acc), fp16_ovfl));
}

/// @brief GFX12 V_DOT4_F32_{FP8,BF8}_{FP8,BF8}: four OCP FP8 (E4M3) or BF8 (E5M2)
/// products from packed bytes plus an F32 accumulator.
template <bool Bf8A, bool Bf8B>
inline uint32_t gfx12_dot4_f32_fp8(uint32_t a, uint32_t b, uint32_t acc) {
  using namespace gfx12_dot_detail;
  const auto bytes = [](uint32_t v) {
    return std::array<uint16_t, 4>{uint16_t(v & 0xff), uint16_t((v >> 8) & 0xff),
                                   uint16_t((v >> 16) & 0xff), uint16_t(v >> 24)};
  };
  using A = std::conditional_t<Bf8A, Bf8Factor, Fp8Factor>;
  using B = std::conditional_t<Bf8B, Bf8Factor, Fp8Factor>;
  return gfx12_dot_terms<A, B, false>(bytes(a), bytes(b), acc);
}
} // namespace rocjitsu::amdgpu
