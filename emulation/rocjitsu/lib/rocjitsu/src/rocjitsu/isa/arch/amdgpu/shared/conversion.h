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

#include <cstdint>
#include <limits>

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

} // namespace rocjitsu::amdgpu::conversion
