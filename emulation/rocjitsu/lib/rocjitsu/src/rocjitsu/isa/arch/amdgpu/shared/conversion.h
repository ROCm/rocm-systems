// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file conversion.h
/// @brief VALU data conversions on raw encodings, in hardware stage order.
///
/// Each conversion passes its source through the same stages:
///
///   1. Source modifiers: ABS clears, then NEG flips, the sign bit
///      (source_modifier.h).
///   2. Input flush: when the source format's MODE field disables input
///      denormals, a subnormal becomes a zero of the same sign.
///   3. Conversion: one rounding into the destination (rounding.h for
///      floating destinations).
///   4. OMOD, then CLAMP, on a floating result (output_modifier.h).
///
/// The callers supply stages 1 and 4; the operations here start at stage 2.
/// Every stage works on raw encodings in unsigned integer lanes, so host
/// rounding, DAZ and FTZ cannot change a result. The same templates accept a
/// scalar lane or a std::experimental::simd of lanes. Differences between
/// gfx1201 results and the ISA pseudocode are marked "ISA discrepancy" below.

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_modifier.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rounding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace rocjitsu::amdgpu::conversion {

/// @brief How a float-to-integer conversion rounds a value with a fraction.
enum class IntegerRounding : uint8_t {
  TOWARD_ZERO,
  /// Toward -infinity (V_CVT_FLOOR_I32_F32).
  FLOOR,
  /// floor(x + 0.5): halves round toward +infinity (V_CVT_NEAREST_I32_F32).
  NEAREST_UP,
};

/// @brief The integer a NaN source converts to.
enum class NanResult : uint8_t {
  ZERO,
  /// The bound matching the NaN's sign bit: +NaN gives the largest value.
  SATURATE_BY_SIGN,
};

namespace detail {

using comparison::detail::choose;

/// @brief Whether discarding `rest` from an integer magnitude rounds it up.
/// @param half The weight of the most significant discarded bit; zero when nothing is discarded.
template <typename V, typename Mask>
constexpr Mask rounds_up(IntegerRounding rounding, const Mask &negative, V rest, V half) {
  const V zero(typename fp_format::F32::Lane{0});
  switch (rounding) {
  case IntegerRounding::FLOOR:
    return rest != zero && negative;
  case IntegerRounding::NEAREST_UP:
    return rest > half || (rest == half && rest != zero && !negative);
  default:
    return rest != rest;
  }
}

} // namespace detail

/// @brief Convert an F32 encoding to a saturated integer of type Int.
/// @details Returns the integer's two's-complement bits in a 32-bit lane. Values
/// outside Int's range, including infinities, saturate to its bounds; a negative
/// value converts to zero for an unsigned Int. Input flushing belongs to the caller.
template <typename Int, typename V>
constexpr V f32_to_integer(V bits, IntegerRounding rounding, NanResult nan_result) {
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
      kept + choose(detail::rounds_up(rounding, negative, rest, half), V(L{1}), V(L{0}));
  // At or above 2^31 every Int saturates; below it the left shift is at most 7 bits.
  const auto huge = field >= V(kUnitField + 8);
  const V left = choose(integral && !huge, field - V(kUnitField), V(L{0}));
  const V units = choose(integral, significand << left, rounded);

  const V positive = choose(units > V(kHighest) || huge, V(kHighest), units);
  const V negative_units = choose(units > V(kNegativeLimit) || huge, V(kNegativeLimit), units);
  V result = choose(negative, V(L{0}) - negative_units, positive);

  // ISA discrepancy: the ISA converts a NaN to zero, but gfx1201 saturates a
  // V_CVT_FLOOR_I32_F32 or V_CVT_NEAREST_I32_F32 NaN by its sign bit.
  const V nan = nan_result == NanResult::ZERO
                    ? V(L{0})
                    : choose(negative, V(L{0} - kNegativeLimit), V(kHighest));
  return choose(magnitude > V(F32::kInfinity), nan, result);
}

/// @brief An F32-to-integer conversion with the instruction's input-flush policy.
/// @details Usable directly as a scalar or SIMD lane operation on raw F32 bits.
template <typename Int, IntegerRounding Rounding, NanResult Nan> struct F32ToInteger {
  input_denormal::Policy input;

  template <typename V> constexpr V operator()(V bits) const {
    return f32_to_integer<Int>(input_denormal::flush_input<fp_format::F32>(bits, input), Rounding,
                               Nan);
  }
};

/// V_CVT_FLOOR_I32_F32 (V_CVT_FLR_I32_F32).
using FloorI32 = F32ToInteger<int32_t, IntegerRounding::FLOOR, NanResult::SATURATE_BY_SIGN>;
/// V_CVT_NEAREST_I32_F32 (V_CVT_RPI_I32_F32).
using NearestI32 = F32ToInteger<int32_t, IntegerRounding::NEAREST_UP, NanResult::SATURATE_BY_SIGN>;

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
template <typename From, typename To> struct ToFloat {
  /// The lane type that holds both encodings.
  using Lane = std::conditional_t<(sizeof(typename From::Lane) > sizeof(typename To::Lane)),
                                  typename From::Lane, typename To::Lane>;

  /// VOP3 ABS and NEG fields; bit 0 applies to the source.
  uint32_t abs = 0;
  uint32_t neg = 0;
  input_denormal::Policy input;
  rounding::Policy rounding;
  output_modifier::Policy output;

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
      constexpr Lane kMask = Lane(~Lane{0}) >> (8 * sizeof(Lane) - From::kWidth);
      constexpr Lane kTop = Lane{1} << (From::kWidth - 1);
      const V value = bits & V(kMask);
      if constexpr (From::kSigned) {
        // A negative value's magnitude is its two's complement within the width.
        const auto negative = (value & V(kTop)) != V(Lane{0});
        const V magnitude = choose(negative, (V(Lane{0}) - value) & V(kMask), value);
        return rounding::from_integer<Destination>(magnitude, negative, rounding);
      } else {
        return rounding::from_integer<Destination>(value, value != value, rounding);
      }
    } else {
      using Source = fp_format::Format<Lane, From::kExponentBits, From::kMantissaBits>;
      bits = source_modifier::apply<Source>(bits & V(Source::kBits), 0, abs, neg);
      bits = input_denormal::flush_input<Source>(bits, input);
      if constexpr (From::kMantissaBits > To::kMantissaBits)
        return rounding::narrow<Source, Destination>(bits, rounding);
      else
        return rounding::widen<Source, Destination>(bits);
    }
  }
};

} // namespace rocjitsu::amdgpu::conversion
