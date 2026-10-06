// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file frexp.h
/// @brief V_FREXP_MANT and V_FREXP_EXP: split x into mantissa * 2^exponent.
///
/// Both operations flush the source under MODE input control, then read the
/// encoding. A subnormal source that MODE keeps is normalized first.
///
/// | Source              | Mantissa                      | Exponent         |
/// |---------------------|-------------------------------|------------------|
/// | NaN                 | The source, quieted           | 0                |
/// | +-infinity          | The source                    | 0                |
/// | +-0                 | The source                    | 0                |
/// | Otherwise           | +-[0.5, 1.0), the source sign | floor(log2|x|) + 1 |
///
/// The mantissa is exact and never subnormal; the floating_operation.h wrapper
/// applies ABS/NEG before it and OMOD/CLAMP after. The exponent is a
/// two's-complement integer of the source format's width (I16 for F16, I32
/// for F32, I64 for F64, which V_FREXP_EXP_I32_F64 narrows); it takes no
/// output modifiers. gfx1201 captures match these rules bit for bit in F16,
/// F32 and F64, VOP1 and VOP3, under every MODE denormal setting probed. Other
/// targets use them without hardware verification.
///
/// Scalar and SIMD callers use the same implementation on unsigned encodings.

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"

namespace rocjitsu::amdgpu::frexp {

namespace detail {

using comparison::detail::choose;

/// @brief A finite magnitude rewritten with its leading one in the implicit bit.
template <typename V> struct Normalized {
  /// Biased exponent, in two's complement: w - m for a subnormal of fraction width w.
  V exponent;
  /// Left shift that moves a subnormal fraction's leading one to the implicit bit.
  V shift;
};

template <typename Fmt, typename V> constexpr Normalized<V> normalize(V magnitude) {
  using Lane = typename Fmt::Lane;
  const V exponent = magnitude >> Fmt::kMantissaBits;
  const auto subnormal = exponent == Lane{0};
  // A subnormal with fraction width w moves left by m + 1 - w; its exponent
  // field would then be 1 - (m + 1 - w) = w - m.
  const V shift = V(Lane{Fmt::kMantissaBits + 1}) - fp_format::bit_width(magnitude);
  return {choose(subnormal, V(Lane{1}) - shift, exponent), choose(subnormal, shift, V(Lane{0}))};
}

} // namespace detail

/// @brief The mantissa of an input-flushed source with ABS/NEG already applied.
template <typename Fmt, typename V>
constexpr V mantissa(V bits, const input_denormal::Policy &policy) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  using Lane = typename Fmt::Lane;
  using detail::choose;
  bits = input_denormal::prepare<Fmt>(bits, policy);
  const V magnitude = bits & Fmt::kMagnitude;
  const auto normalized = detail::normalize<Fmt>(magnitude);
  const V half_exponent(Lane{Fmt::kBias - 1} << Fmt::kMantissaBits);
  const V numeric =
      (bits & Fmt::kSign) | half_exponent | ((magnitude << normalized.shift) & Fmt::kFraction);
  // ISA discrepancy: the ISA expects a NaN source to be returned unchanged, but
  // gfx1201 quiets it.
  const V nan_result = choose(magnitude > Fmt::kInfinity, bits | Fmt::kQuiet, bits);
  return choose(magnitude == Lane{0} || magnitude >= Fmt::kInfinity, nan_result, numeric);
}

/// @brief The exponent of an input-flushed source, in the format's width.
template <typename Fmt, typename V>
constexpr V exponent(V bits, const input_denormal::Policy &policy) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  using Lane = typename Fmt::Lane;
  using detail::choose;
  bits = input_denormal::prepare<Fmt>(bits, policy);
  const V magnitude = bits & Fmt::kMagnitude;
  // |x| = 1.f * 2^(e - bias) = 0.1f * 2^(e - bias + 1).
  const V numeric = (detail::normalize<Fmt>(magnitude).exponent - (Fmt::kBias - 1)) & Fmt::kBits;
  return choose(magnitude == Lane{0} || magnitude >= Fmt::kInfinity, V(Lane{0}), numeric);
}

/// @brief V_FREXP_MANT with its MODE policy, independent of instruction modifiers.
template <typename Fmt> struct Mantissa {
  input_denormal::Policy policy;

  template <typename V> constexpr V operator()(V bits) const { return mantissa<Fmt>(bits, policy); }
};

/// @brief V_FREXP_EXP with its MODE policy. ABS/NEG cannot change the result.
template <typename Fmt> struct Exponent {
  input_denormal::Policy policy;

  template <typename V> constexpr V operator()(V bits) const { return exponent<Fmt>(bits, policy); }
};

} // namespace rocjitsu::amdgpu::frexp
