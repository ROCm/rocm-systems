// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file comparison.h
/// @brief Floating-point VOPC relations evaluated in hardware stage order.
///
/// Each source passes through the same stages before the relation sees it:
///
///   1. Source modifiers: ABS clears, then NEG flips, the sign bit.
///   2. Input flush: when MODE disables input denormals, a subnormal becomes a
///      zero of the same sign. The output is a lane mask, so nothing is flushed
///      afterwards.
///   3. Order key: NaNs are set aside and the remaining encodings map to
///      unsigned keys that order like their values, with -0 equal to +0.
///   4. Relation: the operation template parameter, called directly on the keys.
///
/// The VOPC encoding lists the sixteen float relations as eight ordered
/// predicates followed by their negations in reverse order: opcode 15 - k is
/// the negation of opcode k. An ordered predicate is false when either source
/// is NaN, so its negation is true there:
///
///   F  LT  EQ  LE  GT  LG  GE  O     (0-7, ordered)
///   T  NLT NEQ NLE NGT NLG NGE U     (15-8, negated)
///
/// Every stage works on raw encodings in unsigned integer lanes, never on host
/// floats, so host DAZ cannot flush a subnormal the guest MODE preserves. The
/// same templates accept a scalar lane or a std::experimental::simd of lanes;
/// F16 occupies the low half of a 32-bit lane as in the VOPC SIMD path. Lane
/// values above the format width are ignored. CLASS tests read the raw
/// encoding and do not use these stages.
/// Input flushing uses input_denormal.h; gfx1201 captures confirm the compare
/// behavior across every MODE.FP_DENORM setting.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"

#include <cstdint>
#include <functional>
#include <type_traits>

namespace rocjitsu::amdgpu::comparison {

/// @brief A relation: an operation on order keys, optionally negated.
template <typename Op, bool Negated = false> struct Relation {
  using Operation = Op;
  static constexpr bool kNegated = Negated;
};

namespace detail {

/// @brief All ones where the magnitude is nonzero, zero for either signed zero.
template <typename Fmt, typename V> constexpr V nonzero(V bits) {
  const V magnitude = bits & Fmt::kMagnitude;
  return typename Fmt::Lane{0} - ((magnitude + Fmt::kMagnitude) >> (Fmt::kWidth - 1));
}

/// @brief The ordered predicate F: false on every lane, in the key comparison's result type.
struct Never {
  template <typename V> constexpr auto operator()(const V &a, const V &) const { return a != a; }
};

/// @brief The ordered predicate O: true on every lane that reaches the operation.
struct Always {
  template <typename V> constexpr auto operator()(const V &a, const V &) const { return a == a; }
};

} // namespace detail

using F = Relation<detail::Never>;
using Lt = Relation<std::less<>>;
using Eq = Relation<std::equal_to<>>;
using Le = Relation<std::less_equal<>>;
using Gt = Relation<std::greater<>>;
using Lg = Relation<std::not_equal_to<>>;
using Ge = Relation<std::greater_equal<>>;
using O = Relation<detail::Always>;
using U = Relation<detail::Always, true>;
using Nge = Relation<std::greater_equal<>, true>;
using Nlg = Relation<std::not_equal_to<>, true>;
using Ngt = Relation<std::greater<>, true>;
using Nle = Relation<std::less_equal<>, true>;
using Neq = Relation<std::equal_to<>, true>;
using Nlt = Relation<std::less<>, true>;
using T = Relation<detail::Never, true>;

/// @brief Whether a source encoding is NaN.
template <typename Fmt, typename V> constexpr auto is_nan(V bits) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  return (bits & Fmt::kMagnitude) > Fmt::kInfinity;
}

/// @brief Map a non-NaN encoding to a key that orders like its value, with -0 below +0.
/// @details A positive encoding sets the sign bit; a negative one inverts every
/// bit of the format, reversing magnitude order.
template <typename Fmt, typename V> constexpr V total_order_key(V bits) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  bits = bits & Fmt::kBits;
  const V negative = typename Fmt::Lane{0} - (bits >> (Fmt::kWidth - 1));
  return bits ^ ((negative & Fmt::kBits) | Fmt::kSign);
}

/// @brief Stage 3: map a non-NaN encoding to a key that orders like its value.
/// @details Both zeros map to the same key.
template <typename Fmt, typename V> constexpr V order_key(V bits) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  bits = bits & Fmt::kBits;
  return total_order_key<Fmt>(bits & detail::nonzero<Fmt>(bits));
}

/// @brief Evaluate a relation on two sources that already carry their modifiers.
/// @returns bool for scalar lanes, or the key comparison's mask for SIMD lanes.
template <typename Fmt, typename Rel, typename V>
constexpr auto evaluate(V a, V b, const input_denormal::Policy &policy) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  a = input_denormal::prepare<Fmt>(a, policy);
  b = input_denormal::prepare<Fmt>(b, policy);
  const auto ordered = !(is_nan<Fmt>(a) || is_nan<Fmt>(b));
  const auto holds = ordered && typename Rel::Operation{}(order_key<Fmt>(a), order_key<Fmt>(b));
  if constexpr (Rel::kNegated)
    return !holds;
  else
    return holds;
}

/// @brief Evaluate a relation after applying per-source VOP3 modifiers.
/// @param abs VOP3 ABS field; bit 0 applies to src0 and bit 1 to src1.
/// @param neg VOP3 NEG field, with the same bit assignment.
template <typename Fmt, typename Rel, typename V>
constexpr auto evaluate(V a, V b, uint32_t abs, uint32_t neg,
                        const input_denormal::Policy &policy) {
  return evaluate<Fmt, Rel>(source_modifier::apply<Fmt>(a, 0, abs, neg),
                            source_modifier::apply<Fmt>(b, 1, abs, neg), policy);
}

} // namespace rocjitsu::amdgpu::comparison
