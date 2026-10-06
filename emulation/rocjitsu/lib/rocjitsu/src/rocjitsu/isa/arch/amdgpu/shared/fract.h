// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file fract.h
/// @brief V_FRACT_F16/F32/F64: the fractional part x - floor(x), below 1.0.
///
/// The operation flushes the source under MODE input control, computes the
/// fractional part on the raw encoding, rounds it under MODE.FP_ROUND, and
/// flushes a subnormal result under MODE output control. The
/// floating_operation.h wrapper applies ABS/NEG before it and OMOD/CLAMP after.
///
/// | Source              | Result                                   |
/// |---------------------|------------------------------------------|
/// | NaN                 | The source, quieted                      |
/// | +-infinity          | The negative quiet NaN, e.g. 0xffc00000  |
/// | +-0 or an integer   | +0                                       |
/// | Positive, below 1.0 | The source                               |
/// | Otherwise           | x - floor(x), rounded; at most 1.0 - ulp |
///
/// Only a negative source above -0.5 needs rounding: its fractional part
/// 1 - |x| has fewer fraction bits than |x|. Every other result is exact.
/// gfx1201 captures match these rules bit for bit in F16, F32 and F64, VOP1
/// and VOP3, under every MODE rounding and denormal setting probed. Other
/// targets use them without hardware verification.
///
/// Scalar and SIMD callers use the same implementation on unsigned encodings.

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rounding.h"

#include <cstdint>

namespace rocjitsu::amdgpu::fract {

/// @brief MODE settings of the source and result format, fixed before any lane is evaluated.
struct Policy {
  input_denormal::Policy input;
  output_denormal::Policy output;
  /// MODE.FP_ROUND field: 0 nearest even, 1 toward +inf, 2 toward -inf, 3 toward zero.
  uint32_t round_mode = 0;

  /// @brief Read the format's MODE.FP_DENORM and MODE.FP_ROUND fields.
  static constexpr Policy make(uint32_t denorm_mode, uint32_t round_mode) {
    return {input_denormal::Policy::make(denorm_mode), output_denormal::Policy::make(denorm_mode),
            round_mode};
  }
};

namespace detail {

using comparison::detail::choose;

/// @brief Encode significand * 2^-shift, exact for 0 < significand < 2^shift <= 2^(m+1).
/// @details The leading one of a nonzero significand becomes the implicit bit.
/// The result is at least 2^-(m+1), a normal number in every format.
template <typename Fmt, typename V> constexpr V encode_fraction(V significand, V shift) {
  using Lane = typename Fmt::Lane;
  const V width = fp_format::bit_width(significand);
  const V exponent = width + V(Lane{Fmt::kBias - 1}) - shift;
  const V normalized = significand << (V(Lane{Fmt::kMantissaBits + 1}) - width);
  return (exponent << Fmt::kMantissaBits) | (normalized & Fmt::kFraction);
}

} // namespace detail

/// @brief The rounded fractional part of an input-flushed finite source.
template <typename Fmt, typename V> constexpr V fractional_part(V bits, uint32_t round_mode) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  using Lane = typename Fmt::Lane;
  using detail::choose;
  constexpr Lane kMantissaBits = Fmt::kMantissaBits;
  constexpr Lane kIntegral = Fmt::kBias + kMantissaBits;
  constexpr Lane kTwo = Lane{2} << kMantissaBits;
  const V zero(Lane{0});
  const V one(Lane{1});

  const V magnitude = bits & Fmt::kMagnitude;
  const V exponent = magnitude >> kMantissaBits;
  const auto negative = (bits & Fmt::kSign) != Lane{0};
  const auto subnormal = exponent == Lane{0};
  const V significand = (magnitude & Fmt::kFraction) | choose(subnormal, zero, V(Fmt::kMinNormal));
  // |x| = significand * 2^-shift: shift counts the significand bits below the
  // binary point, and is zero when x is an integer.
  const V effective_exponent = choose(subnormal, one, exponent);
  const V shift =
      V(kIntegral) - choose(effective_exponent > kIntegral, V(kIntegral), effective_exponent);

  // |x| >= 0.5: the fractional part, or 1 minus it for a negative source, needs
  // no more fraction bits than x. Bound the shift in lanes this case does not use.
  const auto exact = shift <= kMantissaBits + 1;
  const V exact_shift = choose(exact, shift, V(kMantissaBits + 1));
  const V unit = one << exact_shift;
  const V fraction_bits = significand & (unit - Lane{1});
  const V exact_result = detail::encode_fraction<Fmt>(
      choose(negative, unit - fraction_bits, fraction_bits), exact_shift);

  // -0.5 < x < 0: the result 1 - |x| lies in (0.5, 1), in steps of 2^-(m+1).
  // Scaled by 2^(m+1), it is the integer 2^(m+1) plus x * 2^(m+1), so it rounds
  // as that negative term does; toward zero means toward -inf for the positive
  // result. Dropping m + 2 or more bits rounds the same way, so bound the count.
  const V dropped_bits = choose(exact, one, shift - (kMantissaBits + 1));
  const V dropped = choose(dropped_bits > kMantissaBits + 2, V(kMantissaBits + 2), dropped_bits);
  const uint32_t result_round_mode = round_mode == 3 ? 2 : round_mode;
  const V rounded =
      V(kTwo) - rounding::shift_right(significand, dropped, negative, result_round_mode);
  // Rounding 1 - |x| up to 1.0 gives the largest value below 1.0 instead.
  const V capped = choose(rounded == kTwo, V(kTwo - Lane{1}), rounded);
  const V rounded_result = (V(Lane{Fmt::kBias - 1}) << kMantissaBits) | (capped & Fmt::kFraction);

  // A positive source below 0.5 is its own fractional part.
  const V small_result = choose(negative, rounded_result, bits);
  const V numeric = choose(exact, exact_result, small_result);
  // ISA discrepancy: the ISA expects S0 + -floor(S0), which is -0 for an
  // integer or -0 source when rounding toward -inf, but gfx1201 returns +0 in
  // every rounding mode.
  const auto integer = (exact && fraction_bits == Lane{0}) || magnitude == Lane{0};
  return choose(integer, zero, numeric);
}

/// @brief Apply input flushing, the fractional part and output flushing to a
/// source with ABS/NEG already applied.
template <typename Fmt, typename V> constexpr V evaluate(V bits, const Policy &policy) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  using detail::choose;
  bits = input_denormal::prepare<Fmt>(bits, policy.input);
  const V magnitude = bits & Fmt::kMagnitude;
  const V numeric = output_denormal::flush_output<Fmt>(
      fractional_part<Fmt>(bits, policy.round_mode), policy.output);
  // ISA discrepancy: the ISA expects S0 + -floor(S0), an invalid operation for
  // an infinite source, but gfx1201 returns the negative quiet NaN for either
  // sign rather than a NaN derived from the source.
  const V infinity_result(Fmt::kSign | Fmt::kInfinity | Fmt::kQuiet);
  const V special = choose(magnitude == Fmt::kInfinity, infinity_result, bits | Fmt::kQuiet);
  return choose(magnitude >= Fmt::kInfinity, special, numeric);
}

/// @brief Fractional part with MODE policies, independent of instruction modifiers.
template <typename Fmt> struct Operation {
  Policy policy;

  template <typename V> constexpr V operator()(V bits) const { return evaluate<Fmt>(bits, policy); }
};

} // namespace rocjitsu::amdgpu::fract
