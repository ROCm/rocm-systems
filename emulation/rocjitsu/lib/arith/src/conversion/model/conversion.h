// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Scalar/SIMD conversion model inherited from rocjitsu.
// Works on raw encodings; guest FP controls never change the host FP environment.

#include "common/bits.h"
#include "conversion/model/fp_format.h"
#include "conversion/model/input_denormal.h"
#include "conversion/model/lane_select.h"
#include "conversion/model/output_modifier.h"
#include "conversion/model/round_increment.h"
#include "conversion/model/source_modifier.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_model::conversion {

/// @brief How a conversion rounds a value its destination cannot represent.
/// @details The first four values are the MODE.FP_ROUND encodings.
enum class Rounding : uint8_t {
  /// Halves round to the even neighbour.
  NEAREST_EVEN = 0,
  /// Toward +infinity.
  CEIL = 1,
  /// Toward -infinity (V_CVT_FLOOR_I32_F32).
  FLOOR = 2,
  TOWARD_ZERO = 3,
  /// floor(x + 0.5): halves round toward +infinity (V_CVT_NEAREST_I32_F32).
  /// Integer destinations only.
  NEAREST_UP,
};

/// @brief The rounding a MODE.FP_ROUND field selects.
constexpr Rounding mode_rounding(uint32_t round_mode) {
  return static_cast<Rounding>(round_mode & 3u);
}

/// @brief The integer a NaN source converts to.
enum class NanResult : uint8_t {
  ZERO,
  /// The bound matching the NaN's sign bit: +NaN gives the largest value.
  SATURATE_BY_SIGN,
};

namespace detail {

using lane::choose;

template <typename V> struct LaneOf {
  using type = V;
};
template <typename V>
  requires requires { typename V::value_type; }
struct LaneOf<V> {
  using type = typename V::value_type;
};

/// @brief The unsigned element type of a scalar lane or a SIMD vector.
template <typename V> using Lane = typename LaneOf<V>::type;

/// @brief Whether discarding `rest` from the magnitude `kept` rounds it up.
/// @details MODE roundings defer to round_increment. When nothing is discarded,
/// `rest` and `half` are both zero and no rounding increments.
/// @param half The weight of the most significant discarded bit; zero when nothing is discarded.
template <typename Fmt, typename V, typename Mask>
constexpr Mask rounds_up(Rounding rounding, const Mask &negative, V kept, V rest, V half) {
  const V zero(Lane<V>{0});
  if (rounding == Rounding::NEAREST_UP) {
    return rest > half || (rest == half && rest != zero && !negative);
  }
  return rest != zero && round_increment::rounds_up<Fmt>(static_cast<uint32_t>(rounding), negative,
                                                         kept, rest, half);
}

} // namespace detail

/// @brief Convert an F32 encoding to a saturated integer of type Int.
/// @details Returns the integer's two's-complement bits in a 32-bit lane. Values
/// outside Int's range, including infinities, saturate to its bounds; a negative
/// value converts to zero for an unsigned Int. Input flushing belongs to the caller.
template <typename Int, typename V>
constexpr V f32_to_integer(V bits, Rounding rounding, NanResult nan_result) {
  using F32 = fp_format::F32;
  using L = typename F32::Lane;
  using detail::choose;
  static_assert(fp_format::is_lane_v<F32, V>);
  static_assert(std::numeric_limits<Int>::digits <= 32);
  constexpr L kHighest = static_cast<L>(std::numeric_limits<Int>::max());
  constexpr L kNegativeLimit = L{0} - static_cast<L>(std::numeric_limits<Int>::min());
  // The value is significand * 2^(field - kUnitField).
  constexpr L kUnitField = (F32::kExponentMax >> 1) + F32::kMantissaBits;

  const V magnitude = bits & V(F32::kMagnitude);
  const auto negative = (bits & V(F32::kSign)) != V(L{0});
  const V field = magnitude >> F32::kMantissaBits;
  const V significand =
      (magnitude & V(F32::kMinNormal - 1)) | choose(field == V(L{0}), V(L{0}), V(F32::kMinNormal));

  // A significand below 2^24 shifted right by 25 or more keeps nothing.
  const auto integral = field >= V(kUnitField);
  V shift = choose(integral, V(L{0}), V(kUnitField) - field);
  shift = choose(shift > V(L{25}), V(L{25}), shift);
  const V kept = significand >> shift;
  const V rest = significand & ((V(L{1}) << shift) - V(L{1}));
  const V half = (V(L{1}) << shift) >> 1;
  const V rounded =
      kept + choose(detail::rounds_up<F32>(rounding, negative, kept, rest, half), V(L{1}), V(L{0}));
  // Saturate before a left shift can overflow the destination integer.
  const auto huge = field >= V(kUnitField + (std::is_same_v<Int, uint32_t> ? 9 : 8));
  const V left = choose(integral && !huge, field - V(kUnitField), V(L{0}));
  const V units = choose(integral, significand << left, rounded);

  const V positive = choose(units > V(kHighest) || huge, V(kHighest), units);
  const V negative_units = choose(units > V(kNegativeLimit) || huge, V(kNegativeLimit), units);
  V result = choose(negative, V(L{0}) - negative_units, positive);

  // ISA manual discrepancy: the manual converts a NaN to zero, but gfx1201 and
  // gfx1100 saturate a V_CVT_FLOOR_I32_F32 or V_CVT_NEAREST_I32_F32 NaN by its
  // sign bit, with MODE.IEEE set or clear. Other targets assume the same.
  const V nan = nan_result == NanResult::ZERO
                    ? V(L{0})
                    : choose(negative, V(L{0} - kNegativeLimit), V(kHighest));
  return choose(magnitude > V(F32::kInfinity), nan, result);
}

/// @brief V_CVT_PK_U8_F32: an F32 source rounded in MODE to an unsigned byte.
/// @details The byte replaces the one that bits [1:0] of the selector pick in
/// the third source; the other bytes pass through. A NaN gives zero.
class PackU8 {
public:
  /// VOP3 ABS and NEG fields; bit 0 applies to the F32 source.
  uint32_t abs = 0;
  uint32_t neg = 0;
  input_denormal::Policy input;
  Rounding rounding = Rounding::NEAREST_EVEN;

  template <typename V> constexpr V operator()(V value, V selector, V packed) const {
    using L = typename fp_format::F32::Lane;
    value = source_modifier::apply<fp_format::F32>(value, 0, abs, neg);
    value = input_denormal::flush_input<fp_format::F32>(value, input);
    const V byte = f32_to_integer<uint8_t>(value, rounding, NanResult::ZERO);
    const V shift = (selector & V(L{3})) << 3;
    return (packed & ~(V(L{0xff}) << shift)) | (byte << shift);
  }
};

/// @brief Rounding into a floating destination, fixed before any lane is evaluated.
/// @details Every value is rounded once, in the destination format:
///
///   1. `direction` comes from the destination's MODE.FP_ROUND field, so it is
///      one of the four MODE roundings; NEAREST_UP is rejected.
///   2. A value is tiny when, rounded to the destination precision with an
///      unbounded exponent, it stays below the smallest normal. With
///      `flush_tiny` a tiny result becomes a zero of the same sign, even when
///      its subnormal encoding would round up to the smallest normal.
///   3. A finite value above the largest finite gives infinity under
///      nearest-even and when rounding toward its sign, and the largest finite
///      value otherwise. `saturate` (FP16_OVFL) always gives the largest finite.
///   4. A NaN is quieted and keeps its sign and leading payload bits.
struct FloatRounding {
  Rounding direction = Rounding::NEAREST_EVEN;
  bool flush_tiny = false;
  bool saturate = false;
};

namespace detail {

/// @brief Reject a rounding direction that only integer destinations define.
constexpr void check_float_rounding([[maybe_unused]] const FloatRounding &policy) {
  assert(policy.direction != Rounding::NEAREST_UP && "NEAREST_UP is for integer destinations");
}

} // namespace detail

namespace detail {

/// @brief Offset added to biased exponents so values below the subnormal range stay unsigned.
inline constexpr unsigned kExponentOrigin = 1024;

/// @brief Round a normal-range magnitude to `Fmt`, without the overflow check.
/// @details `significand` has its leading one at bit TopBit, and `exponent` is the
/// biased exponent of that bit in `Fmt`, at least one. A carry out of the
/// fraction increments the exponent field.
template <typename Fmt, unsigned TopBit, typename V, typename Mask>
constexpr V round_normal(V significand, V exponent, const Mask &negative, Rounding direction) {
  using L = Lane<V>;
  static_assert(TopBit > Fmt::kMantissaBits && TopBit < 8 * sizeof(L));
  constexpr unsigned kDrop = TopBit - Fmt::kMantissaBits;
  const V kept = significand >> kDrop;
  const V rest = significand & V((L{1} << kDrop) - 1);
  const auto up = rounds_up<Fmt>(direction, negative, kept, rest, V(L{1} << (kDrop - 1)));
  // The kept leading one adds one to the exponent field.
  const V field = (exponent - V(L{1})) << Fmt::kMantissaBits;
  return field + kept + choose(up, V(L{1}), V(L{0}));
}

/// @brief Round a magnitude whose exponent may fall below `Fmt`'s normal range.
/// @details `exponent` is the biased exponent of bit TopBit plus kExponentOrigin.
/// Returns the rounded magnitude, with tiny results replaced by zero when the
/// policy flushes them. Overflow is left to the caller.
template <typename Fmt, unsigned TopBit, typename V, typename Mask>
constexpr V round_magnitude(V significand, V exponent, const Mask &negative,
                            const FloatRounding &policy) {
  using L = Lane<V>;
  constexpr unsigned kM = Fmt::kMantissaBits;
  // Beyond kM + 2 extra positions every significand bit is below the rounding bit.
  static_assert(TopBit + 2 < 8 * sizeof(L));
  constexpr L kOrigin = kExponentOrigin;
  const auto normal = exponent > V(kOrigin);
  // Lanes below the normal range round at full precision as if their exponent were
  // one; the tininess check below reads that result.
  const V normal_result = round_normal<Fmt, TopBit>(
      significand, choose(normal, exponent - V(kOrigin), V(L{1})), negative, policy.direction);

  // A subnormal result keeps fewer bits: one fewer per exponent step below the minimum.
  V extra = V(kOrigin + 1) - choose(normal, V(kOrigin + 1), exponent);
  extra = choose(extra > V(L{kM + 2}), V(L{kM + 2}), extra);
  const V drop = V(L{TopBit - kM}) + extra;
  const V kept = significand >> drop;
  const V rest = significand & ((V(L{1}) << drop) - V(L{1}));
  const V half = V(L{1}) << (drop - V(L{1}));
  const auto up = rounds_up<Fmt>(policy.direction, negative, kept, rest, half);
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
constexpr V resolve_overflow(V magnitude, const Mask &negative, const FloatRounding &policy) {
  using L = Lane<V>;
  const auto overflows = magnitude >= V(L{Fmt::kInfinity});
  const V infinity(L{Fmt::kInfinity});
  const V largest(L{Fmt::kInfinity - 1});
  V overflow = largest;
  if (!policy.saturate) {
    switch (policy.direction) {
    case Rounding::NEAREST_EVEN:
      overflow = infinity;
      break;
    case Rounding::CEIL:
      overflow = choose(negative, largest, infinity);
      break;
    case Rounding::FLOOR:
      overflow = choose(negative, infinity, largest);
      break;
    default:
      break;
    }
  }
  return choose(overflows, overflow, magnitude);
}

} // namespace detail

/// @brief Round the magnitude of a finite `From` encoding to `To`'s layout.
/// @details `bits` holds a `From` encoding in a lane of `From::Lane`; the result is
/// a `To` magnitude in the same lane type. It may exceed `To`'s largest finite
/// encoding: overflow, NaN, infinity and the sign are left to the caller. A
/// `From` subnormal lies below half of the smallest `To` subnormal, so it rounds
/// to zero or, when rounding away from zero, to that subnormal.
template <typename From, typename To, typename V>
constexpr V round_finite(V bits, const FloatRounding &policy) {
  using L = detail::Lane<V>;
  using detail::choose;
  static_assert(fp_format::is_lane_v<From, V>);
  static_assert(From::kMantissaBits > To::kMantissaBits);
  // Keep the origin offset below the smallest source exponent.
  constexpr L kRebias =
      detail::kExponentOrigin + (To::kExponentMax >> 1) - (From::kExponentMax >> 1);
  static_assert((From::kExponentMax >> 1) - (To::kExponentMax >> 1) < detail::kExponentOrigin);
  static_assert((From::kExponentMax >> 1) > (To::kExponentMax >> 1) + To::kMantissaBits + 2);
  detail::check_float_rounding(policy);

  const V magnitude = bits & V(From::kMagnitude);
  const auto negative = (bits & V(From::kSign)) != V(L{0});
  const V field = magnitude >> From::kMantissaBits;
  const V fraction = magnitude & V(From::kMinNormal - 1);
  const auto subnormal = field == V(L{0});
  const V significand = fraction | choose(subnormal, V(L{0}), V(From::kMinNormal));
  const V exponent = choose(subnormal, V(L{1}), field) + V(kRebias);
  return detail::round_magnitude<To, From::kMantissaBits>(significand, exponent, negative, policy);
}

/// @brief Convert to a narrower format (F32 to F16, F64 to F32).
/// @details `bits` holds a `From` encoding in a lane of `From::Lane`; the result is
/// a `To` encoding in the same lane type.
template <typename From, typename To, typename V>
constexpr V narrow(V bits, const FloatRounding &policy) {
  using L = detail::Lane<V>;
  using detail::choose;
  const auto negative = (bits & V(From::kSign)) != V(L{0});
  V result = detail::resolve_overflow<To>(round_finite<From, To>(bits, policy), negative, policy);

  // Infinity stays infinite; a NaN is quieted and keeps its leading payload bits.
  constexpr unsigned kDropped = From::kMantissaBits - To::kMantissaBits;
  const V field = (bits & V(From::kMagnitude)) >> From::kMantissaBits;
  const V fraction = bits & V(From::kMinNormal - 1);
  const V nan = V(L{To::kInfinity | To::kQuiet}) | (fraction >> kDropped);
  result = choose(field == V(L{From::kExponentMax}),
                  choose(fraction == V(L{0}), V(L{To::kInfinity}), nan), result);
  return result | choose(negative, V(L{To::kSign}), V(L{0}));
}

/// @brief Convert to a wider format exactly (F16 to F32, F32 to F64).
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
  const V top = fp_format::bit_width(fraction | V(L{1})) - V(L{1});
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
constexpr V from_integer(V magnitude, const Mask &negative, const FloatRounding &policy) {
  using L = detail::Lane<V>;
  using detail::choose;
  static_assert(fp_format::is_lane_v<To, V>);
  detail::check_float_rounding(policy);
  constexpr unsigned kTopBit = 8 * sizeof(L) - 1;
  // Integers are at least one, so results are never tiny.
  const V top = fp_format::bit_width(magnitude | V(L{1})) - V(L{1});
  const V significand = magnitude << (V(L{kTopBit}) - top);
  const V exponent = top + V(L(To::kExponentMax >> 1));
  V result = detail::round_normal<To, kTopBit>(significand, exponent, negative, policy.direction);
  result = detail::resolve_overflow<To>(result, negative, policy);
  result = choose(magnitude == V(L{0}), V(L{0}), result);
  return result | choose(negative, V(L{To::kSign}), V(L{0}));
}

/// @brief An integer source format: the low Width bits of a 32-bit register.
template <typename Int> struct Integer {
  using Lane = uint32_t;
  static constexpr unsigned kWidth = 8 * sizeof(Int);
  static constexpr bool kSigned = std::is_signed_v<Int>;
};

using I16 = Integer<int16_t>;
using U16 = Integer<uint16_t>;
using I32 = Integer<int32_t>;
using U32 = Integer<uint32_t>;

template <typename Fmt> inline constexpr bool is_integer_v = false;
template <typename Int> inline constexpr bool is_integer_v<Integer<Int>> = true;

/// @brief Convert to a floating format, from an integer or another floating format.
/// @details Holds every stage's settings, resolved before any lane is evaluated:
///
///   ABS/NEG (floating sources) -> input flush -> one rounding -> OMOD -> CLAMP
///
/// Usable directly as a scalar or SIMD lane operation on raw register bits. Lanes
/// use the wider of the two formats' lane types and hold the narrower encoding in
/// their low bits; SIMD callers convert lane widths, scalar lanes convert here.
template <typename From, typename To> class ToFloat {
public:
  /// The lane type that holds both encodings.
  using Lane = std::conditional_t<(sizeof(typename From::Lane) > sizeof(typename To::Lane)),
                                  typename From::Lane, typename To::Lane>;

  /// VOP3 ABS and NEG fields; bit 0 applies to the source.
  uint32_t abs = 0;
  uint32_t neg = 0;
  input_denormal::Policy input;
  FloatRounding rounding;
  output_modifier::Policy output;
  /// Whether widening quiets a signaling NaN. Narrowing always quiets it.
  bool quiet_nan = true;

  template <typename V> constexpr auto operator()(V bits) const {
    if constexpr (std::is_arithmetic_v<V>) {
      return static_cast<typename To::Lane>(evaluate(static_cast<Lane>(bits)));
    } else {
      return evaluate(bits);
    }
  }

private:
  // The destination layout in the shared lane type.
  using Destination = fp_format::Format<Lane, To::kExponentBits, To::kMantissaBits>;

  template <typename V> constexpr V evaluate(V bits) const {
    static_assert(fp_format::is_lane_v<Destination, V>);
    return output_modifier::apply<Destination>(convert(bits), output);
  }

  template <typename V> constexpr V convert(V bits) const {
    using detail::choose;
    if constexpr (is_integer_v<From>) {
      assert(abs == 0 && neg == 0 && "an integer source takes no ABS/NEG");
      constexpr Lane kMask = Lane(~Lane{0}) >> (8 * sizeof(Lane) - From::kWidth);
      constexpr Lane kTop = Lane{1} << (From::kWidth - 1);
      const V value = bits & V(kMask);
      if constexpr (From::kSigned) {
        // A negative value's magnitude is its two's complement within the width.
        const auto negative = (value & V(kTop)) != V(Lane{0});
        const V magnitude = choose(negative, (V(Lane{0}) - value) & V(kMask), value);
        return from_integer<Destination>(magnitude, negative, rounding);
      } else {
        return from_integer<Destination>(value, value != value, rounding);
      }
    } else {
      using Source = fp_format::Format<Lane, From::kExponentBits, From::kMantissaBits>;
      bits = source_modifier::apply<Source>(bits & V(Source::kBits), 0, abs, neg);
      bits = input_denormal::flush_input<Source>(bits, input);
      if constexpr (From::kMantissaBits > To::kMantissaBits) {
        return narrow<Source, Destination>(bits, rounding);
      } else {
        const V widened = widen<Source, Destination>(bits);
        if (quiet_nan) {
          return widened;
        }
        // A signaling NaN keeps its quiet bit clear; its payload survives widening.
        const auto signaling = (bits & V(Source::kMagnitude)) > V(Source::kInfinity) &&
                               (bits & V(Source::kQuiet)) == V(Lane{0});
        return choose(signaling, widened & ~V(Destination::kQuiet), widened);
      }
    }
  }
};

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_model::conversion
