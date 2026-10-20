// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file gfx12_dot.h
/// @brief GFX12 DOT2, FP8/BF8 DOT4 and WMMA arithmetic with F32 or packed outputs.
/// See tests/fixtures/float_dot/README.md for the model's empirical basis.

#include "dot_packed16.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"

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

/// @brief Factor encoding, named as in fp_format. OCP E4M3 has no infinity;
/// S.1111.111 is its only NaN.
template <unsigned ExponentBits, unsigned MantissaBits, bool OcpE4m3 = false> struct FactorFormat {
  static constexpr unsigned kExponentBits = ExponentBits;
  static constexpr unsigned kMantissaBits = MantissaBits;
  static constexpr int kBias = (1 << (ExponentBits - 1)) - 1;
  static constexpr bool kOcpE4m3 = OcpE4m3;
};
template <typename Fmt> using IeeeFactor = FactorFormat<Fmt::kExponentBits, Fmt::kMantissaBits>;
using F16Factor = IeeeFactor<fp_format::F16>;
using Bf16Factor = IeeeFactor<fp_format::BF16>;
using Fp8Factor = FactorFormat<4, 3, true>;
using Bf8Factor = FactorFormat<5, 2>;
template <bool Bf16> using HalfFactor = std::conditional_t<Bf16, Bf16Factor, F16Factor>;

template <typename Format> struct Factor {
  static constexpr unsigned kMantissaBits = Format::kMantissaBits;
  static constexpr unsigned kMantissaMask = (1u << kMantissaBits) - 1;
  static constexpr unsigned kExponentMax = (1u << Format::kExponentBits) - 1;
  uint16_t bits;

  unsigned exponent_field() const { return (bits >> kMantissaBits) & kExponentMax; }
  unsigned mantissa() const { return bits & kMantissaMask; }
  bool negative() const { return (bits >> (Format::kExponentBits + kMantissaBits)) & 1; }
  bool nan() const {
    if (exponent_field() != kExponentMax)
      return false;
    return Format::kOcpE4m3 ? mantissa() == kMantissaMask : mantissa() != 0;
  }
  bool inf() const {
    return !Format::kOcpE4m3 && exponent_field() == kExponentMax && mantissa() == 0;
  }
  uint64_t significand() const {
    return exponent_field() ? (1u << kMantissaBits) | mantissa() : mantissa();
  }
  int alignment_exponent() const { return int(std::max(exponent_field(), 1u)) - Format::kBias; }
  template <typename Other> Term product(Factor<Other> other) const {
    return {significand() * other.significand(),
            alignment_exponent() + other.alignment_exponent() - int(kMantissaBits) -
                int(Factor<Other>::kMantissaBits),
            negative() != other.negative()};
  }
};

/// @brief How an instruction groups its products before adding C, and the NaN
/// it returns for an invalid product (an infinity times zero, or opposite
/// infinities). F16/BF16 DOT2 and WMMA align products in pairs; gfx1201
/// FP8/BF8 DOT4 aligns all four together, and each eight-product FP8/BF8 WMMA
/// step aligns two groups of four. Both FP8/BF8 forms return the factor-NaN
/// payload.
struct ProductPolicy {
  std::size_t group;
  uint32_t invalid_nan;
};
inline constexpr ProductPolicy kPairedProducts{2, kInvalidProductNan};
inline constexpr ProductPolicy kFp8Dot4Products{4, kFactorNan};
inline constexpr ProductPolicy kFp8WmmaProducts{4, kFactorNan};

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
/// state. Product grouping and the invalid-product NaN come from `Policy`; see
/// gfx12_dot_detail::ProductPolicy.
/// See tests/fixtures/float_dot/README.md for hardware qualification of
/// https://github.com/ROCm/rocm-systems/issues/12056 on RDNA4.
template <typename Src0Format, typename Src1Format, gfx12_dot_detail::ProductPolicy Policy,
          bool Packed, std::size_t N>
inline uint32_t gfx12_dot_terms(const std::array<uint16_t, N> &src0,
                                const std::array<uint16_t, N> &src1, uint32_t acc,
                                bool fp16_ovfl = false) {
  static_assert(N == 2 || N == 4 || N == 8);
  static_assert(Policy.group > 0 && N % Policy.group == 0);
  using namespace gfx12_dot_detail;
  constexpr bool Bf16 = std::is_same_v<Src0Format, Bf16Factor>;
  static_assert(!Packed || (std::is_same_v<Src0Format, Src1Format> &&
                            (Bf16 || std::is_same_v<Src0Format, F16Factor>)));
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
    const Factor<Src0Format> left{src0[i]};
    const Factor<Src1Format> right{src1[i]};
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
    return special(Policy.invalid_nan);
  if ((acc & 0x7fffffff) > 0x7f800000)
    return special(acc | kQuietNanBit);
  positive_inf |= acc == 0x7f800000;
  negative_inf |= acc == 0xff800000;
  if (positive_inf || negative_inf)
    return special(positive_inf && negative_inf ? Policy.invalid_nan
                   : negative_inf               ? 0xff800000
                                                : 0x7f800000);

  const unsigned acc_exp = (acc >> 23) & 255;
  const Term c{acc_exp ? (acc & 0x7fffff) | 0x800000u : acc & 0x7fffff,
               int(std::max(acc_exp, 1u)) - 127 - 23, bool(acc >> 31)};
  // C aligns no lower than -14 for packed F16 and -126 otherwise. When the
  // grid is at or below that floor, a negative C contributes an
  // arithmetic-floor unit even when no magnitude bits survive; coarser grids
  // drop that tiny C. At the -126 floor only BF16 products are small enough
  // for this to change a result; the README lists which callers were measured.
  const int c_floor = Packed && !Bf16 ? -14 : -126;
  const int grid = std::max(product_grid, std::max(c_floor, c.leading_exponent()) - 26);
  int64_t total = align(c, grid, grid > c_floor);
  for (std::size_t i = 0; i < N; i += Policy.group) {
    const int group_grid =
        *std::max_element(product_grids.begin() + i, product_grids.begin() + i + Policy.group);
    int64_t group = 0;
    for (std::size_t j = i; j < i + Policy.group; ++j)
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
inline uint32_t gfx12_dot_bits(const std::array<uint16_t, N> &src0,
                               const std::array<uint16_t, N> &src1, uint32_t acc,
                               bool fp16_ovfl = false) {
  using Half = gfx12_dot_detail::HalfFactor<Bf16>;
  return gfx12_dot_terms<Half, Half, gfx12_dot_detail::kPairedProducts, Packed>(src0, src1, acc,
                                                                                fp16_ovfl);
}

template <bool Bf16, std::size_t N>
inline uint32_t gfx12_dot_f32(const std::array<uint16_t, N> &src0,
                              const std::array<uint16_t, N> &src1, uint32_t acc) {
  return gfx12_dot_bits<Bf16, false>(src0, src1, acc);
}

/// @brief Widen packed C, execute one GFX12 DOT4 step and return packed result bits.
template <bool Bf16>
inline uint16_t gfx12_dot4_packed16(const std::array<uint16_t, 4> &src0,
                                    const std::array<uint16_t, 4> &src1, uint16_t acc,
                                    bool fp16_ovfl = false) {
  return uint16_t(
      gfx12_dot_bits<Bf16, true>(src0, src1, dot_packed16::widen<Bf16>(acc), fp16_ovfl));
}

/// @brief Two-product wrapper for GFX12 DOT2 and VOPD DOT2ACC.
template <bool Bf16>
inline uint32_t gfx12_dot2_f32(uint16_t src0_lo, uint16_t src1_lo, uint16_t src0_hi,
                               uint16_t src1_hi, uint32_t acc) {
  return gfx12_dot_f32<Bf16>(std::array{src0_lo, src0_hi}, std::array{src1_lo, src1_hi}, acc);
}

/// @brief Widen packed C, execute one GFX12 DOT2 step and return packed result bits.
/// @details This is also the arithmetic of VOP3 V_DOT2_F16_F16 and V_DOT2_BF16_BF16.
template <bool Bf16>
inline uint16_t gfx12_dot2_packed16(uint16_t src0_lo, uint16_t src1_lo, uint16_t src0_hi,
                                    uint16_t src1_hi, uint16_t acc, bool fp16_ovfl = false) {
  return uint16_t(gfx12_dot_bits<Bf16, true>(std::array{src0_lo, src0_hi},
                                             std::array{src1_lo, src1_hi},
                                             dot_packed16::widen<Bf16>(acc), fp16_ovfl));
}

/// @brief GFX12 V_DOT4_F32_{FP8,BF8}_{FP8,BF8}: four OCP FP8 (E4M3) or BF8 (E5M2)
/// products from packed bytes plus an F32 accumulator.
template <bool Src0Bf8, bool Src1Bf8>
inline uint32_t gfx12_dot4_f32_fp8(uint32_t src0, uint32_t src1, uint32_t acc) {
  using namespace gfx12_dot_detail;
  const auto bytes = [](uint32_t packed_bits) {
    return std::array<uint16_t, 4>{
        uint16_t(packed_bits & 0xff), uint16_t((packed_bits >> 8) & 0xff),
        uint16_t((packed_bits >> 16) & 0xff), uint16_t(packed_bits >> 24)};
  };
  using Src0Format = std::conditional_t<Src0Bf8, Bf8Factor, Fp8Factor>;
  using Src1Format = std::conditional_t<Src1Bf8, Bf8Factor, Fp8Factor>;
  return gfx12_dot_terms<Src0Format, Src1Format, kFp8Dot4Products, false>(bytes(src0), bytes(src1),
                                                                          acc);
}

/// @brief One GFX12 FP8/BF8 WMMA step: eight native FP8 (E4M3) or BF8 (E5M2)
/// products plus F32 C, returning raw F32 bits.
template <bool Src0Fp8, bool Src1Fp8>
inline uint32_t gfx12_dot8_f32(const std::array<uint8_t, 8> &src0,
                               const std::array<uint8_t, 8> &src1, uint32_t acc) {
  using namespace gfx12_dot_detail;
  const auto widen = [](const std::array<uint8_t, 8> &bytes) {
    std::array<uint16_t, 8> factors;
    std::copy(bytes.begin(), bytes.end(), factors.begin());
    return factors;
  };
  using Src0Format = std::conditional_t<Src0Fp8, Fp8Factor, Bf8Factor>;
  using Src1Format = std::conditional_t<Src1Fp8, Fp8Factor, Bf8Factor>;
  return gfx12_dot_terms<Src0Format, Src1Format, kFp8WmmaProducts, false>(widen(src0), widen(src1),
                                                                          acc);
}
} // namespace rocjitsu::amdgpu
