// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file minmax.h
/// @brief AMD minimum/maximum and *_NUM instructions from the IEEE 754-2019 family.
///
/// These instructions select a source after applying ABS/NEG and MODE input
/// flushing. Selection orders -0 below +0 and does not flush the result.
/// The floating_operation.h wrapper applies ABS/NEG before the operation and
/// OMOD/CLAMP afterward, for both scalar and SIMD callers.
///
/// The operation template chooses the result ordering (minimum or maximum)
/// and NaN rule. Three-source forms compose two binary selections; V_MED3_NUM
/// selects the median, or V_MIN3_NUM if any source is NaN. Differences between
/// gfx1201 results and the ISA pseudocode are marked "ISA discrepancy" below.
/// The RDNA4 and CDNA5 callers use these rules. Hardware measurements cover
/// gfx1201 only; applying its ISA discrepancies to CDNA5 remains an assumption
/// requiring hardware verification. SALU S_MIN_NUM, S_MAX_NUM, S_MINIMUM and
/// S_MAXIMUM share them: the ISA defines them to match their VALU counterparts,
/// and gfx1201 captures agree.
///
/// Scalar and SIMD callers use the same implementation on raw encodings in
/// unsigned lanes, so selection preserves signs and NaN payloads.

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"

#include <cstdint>
#include <functional>
#include <type_traits>

namespace rocjitsu::amdgpu::minmax {

/// @brief NaN handling for a binary selection.
enum class Nan : uint8_t {
  /// V_MINIMUM/V_MAXIMUM: return the first NaN, quieted, preserving sign and payload.
  PROPAGATE,
  /// *_NUM: prefer the number; if both sources are NaN, return the first one quieted.
  NUMBER,
};

/// @brief A binary selection's result ordering and NaN rule.
/// @details Ordering is std::less for minimum or std::greater for maximum.
template <typename Ordering, Nan Rule> struct Selection {
  using ResultOrdering = Ordering;
  static constexpr Nan kRule = Rule;
  static constexpr unsigned kSources = 2;
};

/// @brief A three-operand form: Second(First(a, b), c).
template <typename First, typename Second> struct Nested {
  static_assert(First::kSources == 2 && Second::kSources == 2);
  static_assert(First::kRule == Second::kRule);
  using Inner = First;
  using Outer = Second;
  static constexpr unsigned kSources = 3;
};

/// @brief Median of three sources, falling back to Min3Num if any source is NaN.
struct Med3Num {
  static constexpr unsigned kSources = 3;
};

using Minimum = Selection<std::less<>, Nan::PROPAGATE>;
using Maximum = Selection<std::greater<>, Nan::PROPAGATE>;
using MinNum = Selection<std::less<>, Nan::NUMBER>;
using MaxNum = Selection<std::greater<>, Nan::NUMBER>;

// Names follow evaluation order: MinimumMaximum is maximum(minimum(a, b), c).
using Minimum3 = Nested<Minimum, Minimum>;
using Maximum3 = Nested<Maximum, Maximum>;
using MinimumMaximum = Nested<Minimum, Maximum>;
using MaximumMinimum = Nested<Maximum, Minimum>;
using Min3Num = Nested<MinNum, MinNum>;
using Max3Num = Nested<MaxNum, MaxNum>;
using MinMaxNum = Nested<MinNum, MaxNum>;
using MaxMinNum = Nested<MaxNum, MinNum>;

namespace detail {

using comparison::detail::choose;

/// @brief Recover the source encoding from comparison::total_order_key.
template <typename Fmt, typename V> constexpr V from_total_order_key(V key) {
  using Lane = typename Fmt::Lane;
  const V negative = Lane{0} - ((key >> (Fmt::kWidth - 1)) ^ Lane{1});
  return key ^ ((negative & Fmt::kBits) | Fmt::kSign);
}

/// @brief A NaN encoding with its quiet bit set, keeping its sign and payload.
template <typename Fmt, typename V> constexpr V quiet(V nan) { return nan | Fmt::kQuiet; }

/// @brief Select between two sources after modifiers and input flushing.
///
/// | Inputs         | PROPAGATE: minimum/maximum | NUMBER: min_num/max_num |
/// |----------------|----------------------------|-------------------------|
/// | Neither is NaN | Numeric winner             | Numeric winner          |
/// | Only a is NaN  | Quieted a                  | b                       |
/// | Only b is NaN  | Quieted b                  | a                       |
/// | Both are NaN   | Quieted a                  | Quieted a               |
template <typename Fmt, typename Sel, typename V> constexpr V pick(V a, V b) {
  // Select b if it comes before a in the result ordering; ties keep a.
  // The keys distinguish -0 from +0. NaN handling below overrides this result.
  constexpr typename Sel::ResultOrdering result_ordering{};
  const V numeric_result = choose(
      result_ordering(comparison::total_order_key<Fmt>(b), comparison::total_order_key<Fmt>(a)), b,
      a);
  const auto a_is_nan = comparison::is_nan<Fmt>(a);
  const auto b_is_nan = comparison::is_nan<Fmt>(b);
  const auto any_nan = a_is_nan || b_is_nan;

  if constexpr (Sel::kRule == Nan::PROPAGATE) {
    // ISA discrepancy: the ISA expects signaling NaNs to take priority, but
    // we return the first NaN in source order, as observed on gfx1201.
    const V first_nan = choose(a_is_nan, a, b);
    return choose(any_nan, quiet<Fmt>(first_nan), numeric_result);
  } else { // Nan::NUMBER
    // One NaN: take the other source. Two NaNs: quiet the first source.
    const V single_nan_result = choose(a_is_nan, b, a);
    const auto both_nan = a_is_nan && b_is_nan;
    const V nan_result = choose(both_nan, quiet<Fmt>(a), single_nan_result);
    return choose(any_nan, nan_result, numeric_result);
  }
}

/// @brief Apply a two-source operation to prepared sources.
template <typename Fmt, typename Op, typename V> constexpr V apply(V a, V b) {
  return pick<Fmt, Op>(a, b);
}

/// @brief Apply a three-source operation to prepared sources.
template <typename Fmt, typename Op, typename V> constexpr V apply(V a, V b, V c) {
  if constexpr (std::is_same_v<Op, Med3Num>) {
    // The median is max(min(a, b), min(max(a, b), c)), with -0 below +0.
    // ISA discrepancy: the ISA expects floating-point == when matching the
    // largest source, but we distinguish signed zeros, as observed on gfx1201.
    // For (-0, +0, -0), we return -0; the pseudocode would return +0.
    const auto lower = [](V x, V y) { return choose(y < x, y, x); };
    const auto upper = [](V x, V y) { return choose(y > x, y, x); };
    const V key_a = comparison::total_order_key<Fmt>(a);
    const V key_b = comparison::total_order_key<Fmt>(b);
    const V key_c = comparison::total_order_key<Fmt>(c);
    // lower_ab = min(a, b), upper_ab = max(a, b), using order keys.
    const V lower_ab = lower(key_a, key_b);
    const V upper_ab = upper(key_a, key_b);
    // median = max(lower_ab, min(upper_ab, c)).
    const V median_key = upper(lower_ab, lower(upper_ab, key_c));
    const V median = from_total_order_key<Fmt>(median_key);
    const auto any_nan =
        comparison::is_nan<Fmt>(a) || comparison::is_nan<Fmt>(b) || comparison::is_nan<Fmt>(c);
    // Any NaN makes V_MED3_NUM use the V_MIN3_NUM result instead.
    return choose(any_nan, apply<Fmt, Min3Num>(a, b, c), median);
  } else {
    // Outer(Inner(a, b), c): e.g. Minimum3 is min(min(a, b), c),
    // and MinimumMaximum is max(min(a, b), c).
    const V inner_result = pick<Fmt, typename Op::Inner>(a, b);
    return pick<Fmt, typename Op::Outer>(inner_result, c);
  }
}

} // namespace detail

/// @brief Apply input flushing and selection to sources with ABS/NEG already applied.
/// @details Accepts unsigned encodings, scalar or SIMD.
template <typename Fmt, typename Op, typename V, typename... Vs>
  requires(1 + sizeof...(Vs) == Op::kSources && (std::is_same_v<V, Vs> && ...) &&
           fp_format::is_lane_v<Fmt, V>)
constexpr V evaluate(const input_denormal::Policy &policy, V a, Vs... rest) {
  return detail::apply<Fmt, Op>(input_denormal::prepare<Fmt>(a, policy),
                                input_denormal::prepare<Fmt>(rest, policy)...);
}

/// @brief Input flushing and selection, independent of instruction modifiers.
template <typename Fmt, typename Op> struct Operation {
  input_denormal::Policy policy;

  template <typename V, typename... Vs> constexpr V operator()(V a, Vs... rest) const {
    return evaluate<Fmt, Op>(policy, a, rest...);
  }
};

} // namespace rocjitsu::amdgpu::minmax
