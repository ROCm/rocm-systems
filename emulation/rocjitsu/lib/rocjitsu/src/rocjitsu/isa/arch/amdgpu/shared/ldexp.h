// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file ldexp.h
/// @brief V_LDEXP: scale a floating source by 2^exponent on raw encodings.
///
/// The source is flushed under MODE input control; the integer exponent is
/// never flushed or modified. The exact scaled value is rounded once by
/// rounding.h: a normal result is exact, a smaller one rounds under
/// MODE.FP_ROUND, and with output denormals disabled a value that stays below
/// the smallest normal at full precision flushes to a signed zero, even when its
/// subnormal encoding would round up to the smallest normal. Overflow gives
/// infinity or the largest finite value as the rounding direction selects, or
/// the largest finite value when saturating (FP16_OVFL for F16). NaN sources
/// are quieted; infinities and zeros pass through.
///
/// The floating_operation.h wrapper applies ABS/NEG to the floating source
/// only, then OMOD/CLAMP to the result. gfx1201 audit captures of V_LDEXP_F16,
/// F32 and F64 match these rules bit for bit, including the true16 F16 halves,
/// under every MODE rounding, denormal and FP16_OVFL setting probed. The
/// tininess rule comes from a hand-run gfx1201 probe of all three formats under
/// every MODE setting; the audit captures do not cover it. Other targets use
/// these rules without hardware verification.
///
/// Scalar and SIMD callers use the same implementation on unsigned encodings.

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rounding.h"

#include <cstdint>

namespace rocjitsu::amdgpu::ldexp {

/// @brief MODE settings of the source and result format, fixed before any lane is evaluated.
struct Policy {
  input_denormal::Policy input;
  rounding::Policy rounding;

  /// @brief Read the format's MODE.FP_DENORM and MODE.FP_ROUND fields.
  /// @param saturate Overflow gives the largest finite value (FP16_OVFL for F16).
  static constexpr Policy make(uint32_t denorm_mode, uint32_t round_mode, bool saturate) {
    return {input_denormal::Policy::make(denorm_mode),
            {round_mode, output_denormal::Policy::make(denorm_mode).flush_outputs, saturate}};
  }
};

/// @brief Scale a source with ABS/NEG already applied by 2^exponent.
/// @tparam ExponentWidth Width of the two's-complement exponent in the low bits
/// of its lane (16 for V_LDEXP_F16); higher lane bits are ignored.
template <typename Fmt, unsigned ExponentWidth = Fmt::kWidth, typename V>
constexpr V evaluate(V bits, V exponent, const Policy &policy) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  using Lane = typename Fmt::Lane;
  using comparison::detail::choose;
  static_assert(ExponentWidth < 8 * sizeof(Lane) || ExponentWidth == Fmt::kWidth);
  constexpr Lane kMantissaBits = Fmt::kMantissaBits;
  constexpr Lane kExponentSign = Lane{1} << (ExponentWidth - 1);
  constexpr Lane kExponentBits = kExponentSign | (kExponentSign - 1);
  // Every adjustment beyond this bound overflows or drops the whole significand.
  constexpr Lane kBound = Fmt::kExponentMax + kMantissaBits + 3;
  constexpr Lane kOrigin = rounding::kExponentOrigin;
  const V zero(Lane{0});

  bits = input_denormal::prepare<Fmt>(bits, policy.input);
  const V magnitude = bits & Fmt::kMagnitude;
  const V sign = bits & Fmt::kSign;
  const auto negative = sign != Lane{0};

  // Move a subnormal's leading one to the implicit bit, lowering its exponent
  // field from one by the same number of steps.
  const V field = magnitude >> kMantissaBits;
  const auto subnormal = field == Lane{0};
  const V steps =
      choose(subnormal, V(Lane{kMantissaBits + 1}) - fp_format::bit_width(magnitude), zero);
  const V significand =
      choose(subnormal, magnitude << steps, (magnitude & Fmt::kFraction) | Fmt::kMinNormal);
  const V source_exponent = choose(subnormal, V(Lane{1}), field) + V(kOrigin) - steps;

  // Offset binary orders the signed exponent as unsigned; bound it there, then
  // keep the result exponent between fully tiny and overflowing.
  V adjustment = (exponent & kExponentBits) ^ kExponentSign;
  adjustment = choose(adjustment < kExponentSign - kBound, V(kExponentSign - kBound), adjustment);
  adjustment = choose(adjustment > kExponentSign + kBound, V(kExponentSign + kBound), adjustment);
  V offset_exponent = source_exponent + adjustment;
  constexpr Lane kLowest = kExponentSign + kOrigin - (kMantissaBits + 3);
  constexpr Lane kHighest = kExponentSign + kOrigin + Fmt::kExponentMax;
  offset_exponent = choose(offset_exponent < kLowest, V(kLowest), offset_exponent);
  offset_exponent = choose(offset_exponent > kHighest, V(kHighest), offset_exponent);

  // Two guard positions below the fraction let rounding.h see every dropped bit.
  constexpr unsigned kTopBit = Fmt::kMantissaBits + 2;
  const V numeric =
      sign | rounding::round_significand<Fmt, kTopBit>(
                 significand << 2, offset_exponent - kExponentSign, negative, policy.rounding);
  const V special = choose(magnitude > Fmt::kInfinity, bits | Fmt::kQuiet, bits);
  return choose(magnitude == Lane{0} || magnitude >= Fmt::kInfinity, special, numeric);
}

/// @brief LDEXP with its MODE policy, independent of instruction modifiers.
template <typename Fmt, unsigned ExponentWidth = Fmt::kWidth> struct Operation {
  Policy policy;

  template <typename V> constexpr V operator()(V bits, V exponent) const {
    return evaluate<Fmt, ExponentWidth>(bits, exponent, policy);
  }
};

} // namespace rocjitsu::amdgpu::ldexp
