// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file rounding.h
/// @brief Round exact values to a floating-point format under MODE control.
///
/// The helpers take raw encodings or integers and return raw destination
/// encodings, so host rounding, DAZ and FTZ never affect the result. Every
/// value is rounded exactly once, in the destination format:
///
///   1. MODE.FP_ROUND selects nearest-even, toward +infinity, toward
///      -infinity or toward zero.
///   2. Tininess is judged after rounding: a value is tiny when, rounded to
///      the destination precision with an unbounded exponent, it stays below
///      the smallest normal. With flush_tiny set a tiny result becomes a zero
///      of the same sign, even when its subnormal encoding would round up to
///      the smallest normal.
///   3. A finite value above the largest finite rounds to infinity under
///      nearest-even and in the direction of its sign, and to the largest
///      finite value otherwise. Saturation (FP16_OVFL for F16) always selects
///      the largest finite value.
///   4. A NaN is quieted and keeps its sign and leading payload bits.
///
/// Input flushing, source and output modifiers belong to the caller.
/// The same templates accept a scalar lane or a std::experimental::simd of
/// lanes; F16 occupies the low half of a 32-bit lane.

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"

#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu::rounding {

/// @brief MODE.FP_ROUND encodings.
enum Mode : uint32_t {
  NEAREST_EVEN = 0,
  TOWARD_POSITIVE = 1,
  TOWARD_NEGATIVE = 2,
  TOWARD_ZERO = 3,
};

/// @brief Destination rounding settings, fixed before any lane is evaluated.
struct Policy {
  /// MODE.FP_ROUND field of the destination format.
  uint32_t mode = NEAREST_EVEN;
  /// Replace a result that is tiny after rounding with a zero of the same sign.
  bool flush_tiny = false;
  /// Finite overflow gives the largest finite value in every rounding mode.
  bool saturate = false;
};

namespace detail {

using comparison::detail::choose;

template <typename V> struct LaneOf {
  using type = V;
};
template <typename V>
  requires requires { typename V::value_type; }
struct LaneOf<V> {
  using type = typename V::value_type;
};

template <typename V> using Lane = typename LaneOf<V>::type;

/// @brief Offset added to biased exponents so values below the subnormal range stay unsigned.
inline constexpr unsigned kExponentOrigin = 1024;

/// @brief Whether discarding `rest` from `kept` rounds the magnitude up.
/// @param half The weight of the most significant discarded bit.
template <typename V, typename Mask>
constexpr Mask rounds_up(uint32_t mode, const Mask &negative, V kept, V rest, V half) {
  const V zero(Lane<V>{0});
  switch (mode & 3u) {
  case NEAREST_EVEN:
    return rest > half || (rest == half && (kept & Lane<V>{1}) != zero);
  case TOWARD_POSITIVE:
    return rest != zero && !negative;
  case TOWARD_NEGATIVE:
    return rest != zero && negative;
  default:
    return rest != rest;
  }
}

/// @brief Index of the highest set bit of a nonzero lane.
template <typename V> constexpr V top_bit(V value) {
  using L = Lane<V>;
  V index(L{0});
  for (unsigned step = 4 * sizeof(L); step != 0; step /= 2) {
    const V upper = value >> step;
    const auto has_upper = upper != V(L{0});
    index = choose(has_upper, index + V(L(step)), index);
    value = choose(has_upper, upper, value);
  }
  return index;
}

/// @brief Round a normal-range magnitude to `Fmt`, without the overflow check.
/// @details `significand` has its leading one at bit TopBit, and `exponent` is the
/// biased exponent of that bit in `Fmt`, at least one. A carry out of the
/// fraction increments the exponent field.
template <typename Fmt, unsigned TopBit, typename V, typename Mask>
constexpr V round_normal(V significand, V exponent, const Mask &negative, uint32_t mode) {
  using L = Lane<V>;
  static_assert(TopBit > Fmt::kMantissaBits && TopBit < 8 * sizeof(L));
  constexpr unsigned kDrop = TopBit - Fmt::kMantissaBits;
  const V kept = significand >> kDrop;
  const V rest = significand & V((L{1} << kDrop) - 1);
  const auto up = rounds_up(mode, negative, kept, rest, V(L{1} << (kDrop - 1)));
  // The kept leading one adds one to the exponent field.
  const V field = (exponent - V(L{1})) << Fmt::kMantissaBits;
  return field + kept + choose(up, V(L{1}), V(L{0}));
}

/// @brief Round a magnitude whose exponent may fall below `Fmt`'s normal range.
/// @details `exponent` is the biased exponent of bit TopBit plus kExponentOrigin.
/// Returns the rounded magnitude, with tiny results replaced by zero when the
/// policy flushes them. Overflow is left to the caller.
template <typename Fmt, unsigned TopBit, typename V, typename Mask>
constexpr V round_magnitude(V significand, V exponent, const Mask &negative, const Policy &policy) {
  using L = Lane<V>;
  constexpr unsigned kM = Fmt::kMantissaBits;
  // Beyond kM + 2 extra positions every significand bit is below the rounding bit.
  static_assert(TopBit + 2 < 8 * sizeof(L));
  constexpr L kOrigin = kExponentOrigin;
  const auto normal = exponent > V(kOrigin);
  // Lanes below the normal range round at full precision as if their exponent were
  // one; the tininess check below reads that result.
  const V normal_result = round_normal<Fmt, TopBit>(
      significand, choose(normal, exponent - V(kOrigin), V(L{1})), negative, policy.mode);

  // A subnormal result keeps fewer bits: one fewer per exponent step below the minimum.
  V extra = V(kOrigin + 1) - choose(normal, V(kOrigin + 1), exponent);
  extra = choose(extra > V(L{kM + 2}), V(L{kM + 2}), extra);
  const V drop = V(L{TopBit - kM}) + extra;
  const V kept = significand >> drop;
  const V rest = significand & ((V(L{1}) << drop) - V(L{1}));
  const V half = V(L{1}) << (drop - V(L{1}));
  const auto up = rounds_up(policy.mode, negative, kept, rest, half);
  // Rounding up the largest subnormal carries into the smallest normal encoding.
  const V subnormal_result = kept + choose(up, V(L{1}), V(L{0}));

  V result = choose(normal, normal_result, subnormal_result);
  if (policy.flush_tiny) {
    // Only a value one exponent step below the minimum can round up to it at
    // full precision. Rounded with exponent one, that value carries into exponent two.
    const auto reaches_normal =
        exponent == V(kOrigin) && normal_result == V(L{Fmt::kMinNormal << 1});
    result = choose(normal || reaches_normal, result, V(L{0}));
  }
  return result;
}

/// @brief Replace a magnitude at or above infinity with the overflow result.
template <typename Fmt, typename V, typename Mask>
constexpr V resolve_overflow(V magnitude, const Mask &negative, const Policy &policy) {
  using L = Lane<V>;
  const auto overflows = magnitude >= V(L{Fmt::kInfinity});
  if (policy.saturate)
    return choose(overflows, V(L{Fmt::kInfinity - 1}), magnitude);
  // Nearest-even reaches infinity; directed rounding only toward the value's sign.
  const L infinity = Fmt::kInfinity;
  const L largest = Fmt::kInfinity - 1;
  V overflow(largest);
  switch (policy.mode & 3u) {
  case NEAREST_EVEN:
    overflow = V(infinity);
    break;
  case TOWARD_POSITIVE:
    overflow = choose(negative, V(largest), V(infinity));
    break;
  case TOWARD_NEGATIVE:
    overflow = choose(negative, V(infinity), V(largest));
    break;
  default:
    break;
  }
  return choose(overflows, overflow, magnitude);
}

} // namespace detail

/// @brief Convert to a narrower format (F32 to F16, F64 to F32).
/// @details `bits` holds a `From` encoding in a lane of `From::Lane`; the result is a
/// `To` encoding in the same lane type. A `From` subnormal lies below half of the
/// smallest `To` subnormal, so it rounds to zero or, when rounding away from
/// zero, to that subnormal.
template <typename From, typename To, typename V> constexpr V narrow(V bits, const Policy &policy) {
  using L = detail::Lane<V>;
  using detail::choose;
  static_assert(fp_format::is_lane_v<From, V>);
  static_assert(From::kMantissaBits > To::kMantissaBits);
  // Keep the origin offset below the smallest source exponent.
  constexpr L kRebias =
      detail::kExponentOrigin + (To::kExponentMax >> 1) - (From::kExponentMax >> 1);
  static_assert((From::kExponentMax >> 1) - (To::kExponentMax >> 1) < detail::kExponentOrigin);
  static_assert((From::kExponentMax >> 1) > (To::kExponentMax >> 1) + To::kMantissaBits + 2);

  const V magnitude = bits & V(From::kMagnitude);
  const auto negative = (bits & V(From::kSign)) != V(L{0});
  const V field = magnitude >> From::kMantissaBits;
  const V fraction = magnitude & V(From::kMinNormal - 1);
  const auto subnormal = field == V(L{0});
  const V significand = fraction | choose(subnormal, V(L{0}), V(From::kMinNormal));
  const V exponent = choose(subnormal, V(L{1}), field) + V(kRebias);

  V result =
      detail::round_magnitude<To, From::kMantissaBits>(significand, exponent, negative, policy);
  result = detail::resolve_overflow<To>(result, negative, policy);

  // Infinity stays infinite; a NaN is quieted and keeps its leading payload bits.
  constexpr unsigned kDropped = From::kMantissaBits - To::kMantissaBits;
  const V nan = V(L{To::kInfinity | To::kQuiet}) | (fraction >> kDropped);
  result = choose(field == V(L{From::kExponentMax}),
                  choose(fraction == V(L{0}), V(L{To::kInfinity}), nan), result);
  return result | choose(negative, V(L{To::kSign}), V(L{0}));
}

/// @brief Convert to a wider format exactly (F32 to F64, F16 to F32).
/// @details `bits` holds a `From` encoding in a lane of `To::Lane`. A subnormal source
/// becomes a normal result. A NaN is quieted and keeps its payload.
template <typename From, typename To, typename V> constexpr V widen(V bits) {
  using L = detail::Lane<V>;
  using detail::choose;
  static_assert(fp_format::is_lane_v<To, V>);
  static_assert(From::kMantissaBits < To::kMantissaBits);
  constexpr unsigned kShift = To::kMantissaBits - From::kMantissaBits;
  constexpr L kRebias = (To::kExponentMax >> 1) - (From::kExponentMax >> 1);

  const V magnitude = bits & V(L{From::kMagnitude});
  const V field = magnitude >> From::kMantissaBits;
  const V fraction = magnitude & V(L{From::kMinNormal - 1});
  const V normal = ((field + V(kRebias)) << To::kMantissaBits) | (fraction << kShift);

  // Shift a subnormal's leading one into the hidden-bit position; each step lowers
  // the exponent by one from that of the smallest normal.
  const V top = detail::top_bit(fraction | V(L{1}));
  const V steps = V(L{From::kMantissaBits}) - top;
  const V normalized = ((fraction << steps) & V(L{From::kMinNormal - 1})) << kShift;
  const V subnormal = ((V(kRebias + 1) - steps) << To::kMantissaBits) | normalized;

  V result = choose(field == V(L{0}), choose(fraction == V(L{0}), V(L{0}), subnormal), normal);
  const V special = V(L{To::kInfinity}) |
                    choose(fraction == V(L{0}), V(L{0}), V(L{To::kQuiet}) | (fraction << kShift));
  result = choose(field == V(L{From::kExponentMax}), special, result);
  return result | ((bits & V(L{From::kSign})) << (To::kWidth - From::kWidth));
}

/// @brief Convert an integer magnitude and sign to `To`.
/// @details `magnitude` uses the full lane width; zero converts to +0 or -0.
template <typename To, typename V, typename Mask>
constexpr V from_integer(V magnitude, const Mask &negative, const Policy &policy) {
  using L = detail::Lane<V>;
  using detail::choose;
  static_assert(fp_format::is_lane_v<To, V>);
  constexpr unsigned kTopBit = 8 * sizeof(L) - 1;
  // Integers are at least one, so results are never tiny.
  const V top = detail::top_bit(magnitude | V(L{1}));
  const V significand = magnitude << (V(L{kTopBit}) - top);
  const V exponent = top + V(L{To::kExponentMax >> 1});
  V result = detail::round_normal<To, kTopBit>(significand, exponent, negative, policy.mode);
  result = detail::resolve_overflow<To>(result, negative, policy);
  result = choose(magnitude == V(L{0}), V(L{0}), result);
  return result | choose(negative, V(L{To::kSign}), V(L{0}));
}

} // namespace rocjitsu::amdgpu::rounding
