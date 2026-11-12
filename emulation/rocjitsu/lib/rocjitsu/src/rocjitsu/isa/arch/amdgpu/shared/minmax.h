// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file minmax.h
/// @brief AMD floating min/max selection, including legacy and IEEE 754-2019 forms.
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
/// The RDNA4 and CDNA5 callers, and CDNA4's V_MINIMUM3_F32/V_MAXIMUM3_F32, use
/// these rules. Hardware measurements cover gfx1201 only; applying its ISA
/// discrepancies to CDNA4 and CDNA5 remains an assumption requiring hardware
/// verification. SALU S_MIN_NUM, S_MAX_NUM, S_MINIMUM and S_MAXIMUM share them:
/// the ISA defines them to match their VALU counterparts, and gfx1201 captures
/// agree.
///
/// CDNA5 VOPD3 V_DUAL_MIN_NUM_F64/V_DUAL_MAX_NUM_F64 still use host selection.
/// Legacy forms share the encoding selector, with architecture-specific NaN
/// priority and MODE.IEEE signaling-NaN handling. Packed F16 MIN/MAX and *_NUM
/// use this selector through fp_mode::packed_minmax_f16. Packed F16 MINIMUM/MAXIMUM,
/// BF16 and F64 retain separate helpers (packed_binary_f16, packed_select_bf16
/// and binary_f64).
///
/// Scalar and SIMD callers use the same implementation on raw encodings in
/// unsigned lanes, so selection preserves signs and NaN payloads.

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/lane_select.h"

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
  /// Older CDNA/RDNA selection, with architecture-specific priority and MODE.IEEE.
  LEGACY,
};

/// @brief Legacy NaN priority and median ordering, independent of modifiers.
struct LegacyPolicy {
  bool prefer_first_nan = false;
  bool ieee = false;
  // RDNA1-3 hardware orders -0 below +0 in MED3, unlike the manual's floating
  // equality. RDNA3.5 follows this rule; CDNA retains the ISA pseudocode.
  bool ordered_zero_median = false;
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

/// @brief Legacy median uses the target's zero ordering and MIN3 for any NaN.
struct LegacyMed3 {
  static constexpr unsigned kSources = 3;
};

using Minimum = Selection<std::less<>, Nan::PROPAGATE>;
using Maximum = Selection<std::greater<>, Nan::PROPAGATE>;
using MinNum = Selection<std::less<>, Nan::NUMBER>;
using MaxNum = Selection<std::greater<>, Nan::NUMBER>;
using LegacyMin = Selection<std::less<>, Nan::LEGACY>;
using LegacyMax = Selection<std::greater<>, Nan::LEGACY>;

// Names follow evaluation order: MinimumMaximum is maximum(minimum(a, b), c).
using Minimum3 = Nested<Minimum, Minimum>;
using Maximum3 = Nested<Maximum, Maximum>;
using MinimumMaximum = Nested<Minimum, Maximum>;
using MaximumMinimum = Nested<Maximum, Minimum>;
using Min3Num = Nested<MinNum, MinNum>;
using Max3Num = Nested<MaxNum, MaxNum>;
using MinMaxNum = Nested<MinNum, MaxNum>;
using MaxMinNum = Nested<MaxNum, MinNum>;
using LegacyMin3 = Nested<LegacyMin, LegacyMin>;
using LegacyMax3 = Nested<LegacyMax, LegacyMax>;
using LegacyMinMax = Nested<LegacyMin, LegacyMax>;
using LegacyMaxMin = Nested<LegacyMax, LegacyMin>;

namespace detail {

using lane::choose;

/// @brief Whether an operation requires the target's explicit legacy policy.
template <typename Op>
inline constexpr bool is_legacy_v = [] {
  if constexpr (requires { Op::kRule; })
    return Op::kRule == Nan::LEGACY;
  else if constexpr (requires { typename Op::Inner; })
    return Op::Inner::kRule == Nan::LEGACY;
  else
    return std::is_same_v<Op, LegacyMed3>;
}();

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
///
/// LEGACY prefers the numeric source for one NaN and the policy-selected source
/// for two NaNs. Under MODE.IEEE, a signaling NaN takes priority (a before b)
/// and is quieted; otherwise the selected encoding is unchanged.
template <typename Fmt, typename Sel, typename V> constexpr V pick(V a, V b, LegacyPolicy legacy) {
  // Select b if it comes before a in the result ordering; ties keep a.
  // The keys distinguish -0 from +0. NaN handling below overrides this result.
  constexpr typename Sel::ResultOrdering result_ordering{};
  const V numeric_result = choose(
      result_ordering(comparison::total_order_key<Fmt>(b), comparison::total_order_key<Fmt>(a)), b,
      a);
  const auto a_is_nan = comparison::is_nan<Fmt>(a);
  const auto b_is_nan = comparison::is_nan<Fmt>(b);
  const auto any_nan = a_is_nan || b_is_nan;

  if constexpr (Sel::kRule == Nan::LEGACY) {
    const V two_nans = legacy.prefer_first_nan ? a : b;
    const V nan_result = choose(a_is_nan && b_is_nan, two_nans, choose(a_is_nan, b, a));
    V result = choose(any_nan, nan_result, numeric_result);
    if (legacy.ieee) {
      const auto a_is_signaling = a_is_nan && ((a & Fmt::kQuiet) == 0);
      const auto b_is_signaling = b_is_nan && ((b & Fmt::kQuiet) == 0);
      result = choose(b_is_signaling, quiet<Fmt>(b), result);
      result = choose(a_is_signaling, quiet<Fmt>(a), result);
    }
    return result;
  } else if constexpr (Sel::kRule == Nan::PROPAGATE) {
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
template <typename Fmt, typename Op, typename V> constexpr V apply(V a, V b, LegacyPolicy legacy) {
  return pick<Fmt, Op>(a, b, legacy);
}

/// @brief Apply a three-source operation to prepared sources.
template <typename Fmt, typename Op, typename V>
constexpr V apply(V a, V b, V c, LegacyPolicy legacy) {
  if constexpr (std::is_same_v<Op, LegacyMed3>) {
    const auto equal = [legacy](V x, V y) {
      if (legacy.ordered_zero_median)
        return x == y;
      return (x == y) || (((x | y) & ~Fmt::kSign) == 0);
    };
    const V largest = pick<Fmt, LegacyMax>(pick<Fmt, LegacyMax>(a, b, legacy), c, legacy);
    const V median = choose(equal(largest, a), pick<Fmt, LegacyMax>(b, c, legacy),
                            choose(equal(largest, b), pick<Fmt, LegacyMax>(a, c, legacy),
                                   pick<Fmt, LegacyMax>(a, b, legacy)));
    const auto any_nan =
        comparison::is_nan<Fmt>(a) || comparison::is_nan<Fmt>(b) || comparison::is_nan<Fmt>(c);
    const V minimum = pick<Fmt, LegacyMin>(pick<Fmt, LegacyMin>(a, b, legacy), c, legacy);
    return choose(any_nan, minimum, median);
  } else if constexpr (std::is_same_v<Op, Med3Num>) {
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
    return choose(any_nan, apply<Fmt, Min3Num>(a, b, c, legacy), median);
  } else {
    // Outer(Inner(a, b), c): e.g. Minimum3 is min(min(a, b), c),
    // and MinimumMaximum is max(min(a, b), c).
    const V inner_result = pick<Fmt, typename Op::Inner>(a, b, legacy);
    return pick<Fmt, typename Op::Outer>(inner_result, c, legacy);
  }
}

} // namespace detail

/// @brief Apply input flushing and selection to sources with ABS/NEG already applied.
/// @details Accepts unsigned encodings, scalar or SIMD.
template <typename Fmt, typename Op, typename V, typename... Vs>
  requires(!detail::is_legacy_v<Op> && 1 + sizeof...(Vs) == Op::kSources &&
           (std::is_same_v<V, Vs> && ...) && fp_format::is_lane_v<Fmt, V>)
constexpr V evaluate(const input_denormal::Policy &policy, V a, Vs... rest) {
  return detail::apply<Fmt, Op>(input_denormal::prepare<Fmt>(a, policy),
                                input_denormal::prepare<Fmt>(rest, policy)..., LegacyPolicy{});
}

/// @brief Input flushing and selection, independent of instruction modifiers.
template <typename Fmt, typename Op>
  requires(!detail::is_legacy_v<Op>)
struct Operation {
  input_denormal::Policy policy;

  template <typename V, typename... Vs> constexpr V operator()(V a, Vs... rest) const {
    return evaluate<Fmt, Op>(policy, a, rest...);
  }
};

/// @brief Legacy selection on unsigned scalar or SIMD lanes after MODE input flushing.
template <typename Fmt, typename Op>
  requires(detail::is_legacy_v<Op>)
struct LegacyOperation {
  input_denormal::Policy input_policy;
  LegacyPolicy selection_policy;

  constexpr LegacyOperation(input_denormal::Policy input, LegacyPolicy selection)
      : input_policy(input), selection_policy(selection) {}

  template <typename V, typename... Vs>
    requires(1 + sizeof...(Vs) == Op::kSources && (std::is_same_v<V, Vs> && ...) &&
             fp_format::is_lane_v<Fmt, V>)
  constexpr V operator()(V a, Vs... rest) const {
    return detail::apply<Fmt, Op>(input_denormal::prepare<Fmt>(a, input_policy),
                                  input_denormal::prepare<Fmt>(rest, input_policy)...,
                                  selection_policy);
  }
};

} // namespace rocjitsu::amdgpu::minmax
