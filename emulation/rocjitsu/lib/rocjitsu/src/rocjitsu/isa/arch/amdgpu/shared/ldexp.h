// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file ldexp.h
/// @brief V_LDEXP_F16/F32/F64: scale a floating source by 2^exponent on raw encodings.
///
/// The source is flushed under MODE input control; the integer exponent is
/// never flushed or modified. The exact scaled value is rounded once: a normal
/// result is exact, a smaller one rounds under MODE.FP_ROUND, and with output
/// denormals disabled a value that stays below the smallest normal at full
/// precision flushes to a signed zero, even when its subnormal encoding would
/// round up to the smallest normal. Overflow gives infinity or the largest
/// finite value as the rounding direction selects, or the largest finite value
/// when saturating (FP16_OVFL for F16). NaN sources are quieted; infinities and zeros
/// pass through.
///
/// The floating_operation.h wrapper applies ABS/NEG to the floating source
/// only, then OMOD/CLAMP to the result. With effective OMOD a tiny result
/// flushes even when output denormals are kept, and OMOD then gives +0.
///
/// Evidence: gfx1201 audit captures of all three formats, including true16
/// halves and OMOD/CLAMP, and a gfx1201 boundary probe of every FP_ROUND and
/// FP_DENORM byte with FP16_OVFL clear and set, without and with mul:2/div:2;
/// IEEE and DX10 kept their launch values. A gfx1030 F16 probe matches for
/// e32, VOP3 and SDWA with every OMOD under IEEE=0 and IEEE=1, except that
/// IEEE=0 keeps signaling NaNs. Other targets use these rules without
/// hardware verification.
///
/// Scalar and SIMD callers use the same implementation on unsigned encodings.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/lane_select.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"

#include <cstdint>

namespace rocjitsu::amdgpu::ldexp {

/// @brief MODE settings of the source and result format, fixed before any lane is evaluated.
struct Policy {
  input_denormal::Policy input;
  /// MODE.FP_ROUND field: 0 nearest even, 1 toward +inf, 2 toward -inf, 3 toward zero.
  uint32_t round_mode = 0;
  /// Replace a result that is tiny at full precision with a zero of the same sign.
  bool flush_tiny = false;
  /// Finite overflow gives the largest finite value in every rounding mode.
  bool saturate = false;

  /// @brief Read the format's MODE.FP_DENORM and MODE.FP_ROUND fields.
  /// @param saturate Overflow gives the largest finite value (FP16_OVFL for F16).
  /// @param omod Effective OMOD is enabled: a tiny result flushes even with output
  /// denormals kept, so OMOD then gives +0 (gfx1201 probe, all three formats).
  static constexpr Policy make(uint32_t denorm_mode, uint32_t round_mode, bool saturate,
                               bool omod = false) {
    return {input_denormal::Policy::make(denorm_mode), round_mode,
            omod || output_denormal::Policy::make(denorm_mode).flush_outputs, saturate};
  }
};

namespace detail {

using lane::choose;

/// @brief Offset added to biased exponents so values below the subnormal range stay unsigned.
inline constexpr unsigned kExponentOrigin = 1024;

/// @brief Whether discarding `rest` from `kept` rounds the magnitude up.
/// @param half The weight of the most significant discarded bit.
template <typename Fmt, typename V, typename Mask>
constexpr Mask rounds_up(uint32_t round_mode, const Mask &negative, V kept, V rest, V half) {
  using Lane = typename Fmt::Lane;
  const V zero(Lane{0});
  switch (round_mode & 3u) {
  case 0:
    return rest > half || (rest == half && (kept & Lane{1}) != zero);
  case 1:
    return rest != zero && !negative;
  case 2:
    return rest != zero && negative;
  default:
    return rest != rest;
  }
}

/// @brief Round a finite nonzero magnitude, significand * 2^exponent, to `Fmt`.
/// @details `significand` has its leading one at bit TopBit, above the format's
/// fraction, and `exponent` is the biased exponent of that bit plus
/// kExponentOrigin. Returns the rounded magnitude; the caller applies the sign.
template <typename Fmt, unsigned TopBit, typename V, typename Mask>
constexpr V round_significand(V significand, V exponent, const Mask &negative,
                              const Policy &policy) {
  using Lane = typename Fmt::Lane;
  constexpr unsigned kM = Fmt::kMantissaBits;
  static_assert(TopBit > kM && TopBit + 2 < 8 * sizeof(Lane));
  constexpr Lane kOrigin = kExponentOrigin;
  const V zero(Lane{0});
  const V one(Lane{1});
  const auto normal = exponent > V(kOrigin);

  // Round at full precision, as if a lane below the normal range had exponent
  // one; the tininess check below reads that result. A carry out of the
  // fraction increments the exponent field.
  constexpr unsigned kDrop = TopBit - kM;
  const V normal_kept = significand >> kDrop;
  const V normal_rest = significand & V((Lane{1} << kDrop) - 1);
  const auto normal_up = rounds_up<Fmt>(policy.round_mode, negative, normal_kept, normal_rest,
                                        V(Lane{1} << (kDrop - 1)));
  const V field = (choose(normal, exponent - V(kOrigin), one) - one) << kM;
  const V normal_result = field + normal_kept + choose(normal_up, one, zero);

  // A subnormal result keeps fewer bits: one fewer per exponent step below the minimum.
  // Beyond kM + 2 extra positions every significand bit is below the rounding bit.
  V extra = V(kOrigin + 1) - choose(normal, V(kOrigin + 1), exponent);
  extra = choose(extra > V(Lane{kM + 2}), V(Lane{kM + 2}), extra);
  const V drop = V(Lane{kDrop}) + extra;
  const V kept = significand >> drop;
  const V rest = significand & ((one << drop) - one);
  const auto up = rounds_up<Fmt>(policy.round_mode, negative, kept, rest, one << (drop - one));
  // Rounding up the largest subnormal carries into the smallest normal encoding.
  const V subnormal_result = kept + choose(up, one, zero);

  V result = choose(normal, normal_result, subnormal_result);
  if (policy.flush_tiny) {
    // Only a value one exponent step below the minimum can round up to it at
    // full precision. Rounded with exponent one, that value carries into exponent two.
    const auto reaches_normal =
        exponent == V(kOrigin) && normal_result == V(Lane{Fmt::kMinNormal << 1});
    result = choose(normal || reaches_normal, result, zero);
  }

  // Nearest-even overflows to infinity; directed rounding only toward the value's sign.
  const auto overflows = result >= V(Fmt::kInfinity);
  const V largest(Lane{Fmt::kInfinity - 1});
  V overflow = largest;
  if (!policy.saturate) {
    switch (policy.round_mode & 3u) {
    case 0:
      overflow = V(Fmt::kInfinity);
      break;
    case 1:
      overflow = choose(negative, largest, V(Fmt::kInfinity));
      break;
    case 2:
      overflow = choose(negative, V(Fmt::kInfinity), largest);
      break;
    default:
      break;
    }
  }
  return choose(overflows, overflow, result);
}

} // namespace detail

/// @brief Scale a source with ABS/NEG already applied by 2^exponent.
/// @tparam ExponentWidth Width of the two's-complement exponent in the low bits
/// of its lane (16 for V_LDEXP_F16); higher lane bits are ignored.
template <typename Fmt, unsigned ExponentWidth = Fmt::kWidth, typename V>
constexpr V evaluate(V bits, V exponent, const Policy &policy) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  using Lane = typename Fmt::Lane;
  using detail::choose;
  static_assert(ExponentWidth < 8 * sizeof(Lane) || ExponentWidth == Fmt::kWidth);
  constexpr Lane kMantissaBits = Fmt::kMantissaBits;
  constexpr Lane kExponentSign = Lane{1} << (ExponentWidth - 1);
  constexpr Lane kExponentBits = kExponentSign | (kExponentSign - 1);
  // Every adjustment beyond this bound overflows or drops the whole significand.
  constexpr Lane kBound = Fmt::kExponentMax + kMantissaBits + 3;
  constexpr Lane kOrigin = detail::kExponentOrigin;
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

  // Two guard positions below the fraction keep every dropped bit visible to rounding.
  constexpr unsigned kTopBit = Fmt::kMantissaBits + 2;
  const V numeric = sign | detail::round_significand<Fmt, kTopBit>(
                               significand << 2, offset_exponent - kExponentSign, negative, policy);
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
