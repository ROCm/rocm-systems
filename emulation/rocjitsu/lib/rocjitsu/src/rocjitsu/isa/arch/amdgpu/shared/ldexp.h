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
/// Evidence, all three formats unless noted:
/// - gfx1201 boundary probe: every FP_ROUND and FP_DENORM byte with FP16_OVFL
///   clear and set, unmodified and with mul:2, mul:4, div:2, ABS, NEG and
///   CLAMP each; IEEE and DX10 kept their launch values.
/// - gfx1201 audit captures: unmodified forms under 84 MODE values; modifier
///   and true16 forms under 7, with nearest-even or toward-zero rounding and
///   one denormal setting for every format.
/// - gfx1100 IEEE=0 audit captures: match except that hardware keeps
///   signaling NaNs, which the emulator quiets.
/// - gfx1030 F16 probe: e32, VOP3 and SDWA with every OMOD under IEEE=0 and
///   IEEE=1 match in the result half. Hardware also keeps signaling NaNs at
///   IEEE=0 and keeps the upper half of e32/VOP3 destinations, which the
///   emulator zeroes for every F16 operation.
/// Other targets use these rules without hardware verification.
///
/// Scalar and SIMD callers use the same implementation on unsigned encodings.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/lane_select.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/round_increment.h"

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

  /// @brief Build the LDEXP policy from MODE, overflow saturation, and effective OMOD.
  /// @details Reads the format's MODE.FP_DENORM and MODE.FP_ROUND fields. Either
  /// effective OMOD or disabled output denormals sets flush_tiny, which replaces a
  /// result that is tiny at full precision with a zero of the source's sign.
  /// Output flushing alone keeps that sign; only when OMOD is enabled does the
  /// scaling stage turn the signed zero into +0.
  /// @param saturate Overflow gives the largest finite value (FP16_OVFL for F16).
  /// @param omod Effective OMOD is enabled (gfx1201 probe, all three formats).
  /// Callers must pass it: the OMOD stage sees only the rounded result, and a tiny
  /// value that rounds up to the smallest normal no longer looks tiny there.
  static constexpr Policy make(uint32_t denorm_mode, uint32_t round_mode, bool saturate,
                               bool omod) {
    return {input_denormal::Policy::make(denorm_mode), round_mode,
            omod || output_denormal::Policy::make(denorm_mode).flush_outputs, saturate};
  }
};

namespace detail {

using lane::choose;

/// @brief Offset added to biased exponents so values below the subnormal range stay unsigned.
inline constexpr unsigned kExponentOrigin = 1024;

/// @brief Round a finite nonzero magnitude, held as a significand and an offset
/// exponent, to `Fmt`.
/// @details `significand` has its leading one at bit TopBit and at most
/// kMantissaBits + 1 significant bits, as an LDEXP source does, so its bits
/// below TopBit - kMantissaBits are zero. `exponent` is the biased exponent of
/// the leading bit plus kExponentOrigin. Returns the rounded magnitude; the caller
/// applies the sign.
template <typename Fmt, unsigned TopBit, typename V, typename Mask>
constexpr V round_significand(V significand, V exponent, const Mask &negative,
                              const Policy &policy) {
  using Lane = typename Fmt::Lane;
  constexpr unsigned kMantissaBits = Fmt::kMantissaBits;
  static_assert(TopBit > kMantissaBits && TopBit + 2 < 8 * sizeof(Lane));
  constexpr Lane kOrigin = kExponentOrigin;
  const V zero(Lane{0});
  const V one(Lane{1});
  const auto normal = exponent > V(kOrigin);

  // A result in the normal range is exact: the source has no more significant
  // bits than the format keeps.
  constexpr unsigned kDrop = TopBit - kMantissaBits;
  const V field = (choose(normal, exponent - V(kOrigin), one) - one) << kMantissaBits;
  const V normal_result = field + (significand >> kDrop);

  // A subnormal result keeps fewer bits: one fewer per exponent step below the
  // minimum. Beyond kMantissaBits + 2 extra positions every significand bit is
  // below the rounding bit.
  V extra = V(kOrigin + 1) - choose(normal, V(kOrigin + 1), exponent);
  extra = choose(extra > V(Lane{kMantissaBits + 2}), V(Lane{kMantissaBits + 2}), extra);
  const V drop = V(Lane{kDrop}) + extra;
  const V kept = significand >> drop;
  const V rest = significand & ((one << drop) - one);
  const auto up =
      round_increment::rounds_up<Fmt>(policy.round_mode, negative, kept, rest, one << (drop - one));
  // Rounding up the largest subnormal carries into the smallest normal encoding.
  const V subnormal_result = kept + choose(up, one, zero);

  V result = choose(normal, normal_result, subnormal_result);
  // A value below the normal range is exact at full precision, so it is tiny
  // even when its subnormal encoding rounds up to the smallest normal.
  if (policy.flush_tiny)
    result = choose(normal, result, zero);

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

/// @brief LDEXP with its MODE and effective-OMOD policy.
/// @details Independent of ABS/NEG and CLAMP, and applies no OMOD scaling; the
/// policy's flush_tiny, set by effective OMOD or disabled output denormals, only
/// decides whether a tiny result flushes.
template <typename Fmt, unsigned ExponentWidth = Fmt::kWidth> struct Operation {
  Policy policy;

  template <typename V> constexpr V operator()(V bits, V exponent) const {
    return evaluate<Fmt, ExponentWidth>(bits, exponent, policy);
  }
};

} // namespace rocjitsu::amdgpu::ldexp
