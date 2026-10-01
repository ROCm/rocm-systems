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

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <functional>
#include <type_traits>

namespace rocjitsu::amdgpu::comparison {

/// @brief Binary interchange format carried in an unsigned lane type.
template <typename LaneType, unsigned ExponentBits, unsigned MantissaBits> struct Format {
  using Lane = LaneType;
  static constexpr unsigned kExponentBits = ExponentBits;
  static constexpr unsigned kMantissaBits = MantissaBits;
  static constexpr unsigned kWidth = 1 + ExponentBits + MantissaBits;
  static constexpr Lane kSign = Lane{1} << (kWidth - 1);
  static constexpr Lane kMagnitude = kSign - 1;
  static constexpr Lane kBits = kSign | kMagnitude;
  static constexpr Lane kExponentMax = (Lane{1} << ExponentBits) - 1;
  static constexpr Lane kInfinity = kExponentMax << MantissaBits;

  static_assert(std::is_unsigned_v<Lane> && kWidth <= 8 * sizeof(Lane));
};

using F16 = Format<uint32_t, 5, 10>;
using F32 = Format<uint32_t, 8, 23>;
using F64 = Format<uint64_t, 11, 52>;

/// @brief Whether an architecture uses the measured compare rules.
/// @details gfx1201 captures across every MODE.FP_DENORM setting, and gfx1100
/// captures that agree with them on every compare variant. Other targets keep
/// comparing their inputs unflushed.
constexpr bool measured_rules(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA4;
}

/// @brief Per-instruction compare policy, fixed before any lane is evaluated.
struct Policy {
  bool flush_inputs = false;

  /// @param denorm_mode MODE.FP_DENORM field of the source format; bit 0 allows input denormals.
  static constexpr Policy make(rj_code_arch_t arch, uint32_t denorm_mode) {
    return {measured_rules(arch) && (denorm_mode & 1u) == 0};
  }
};

/// @brief A relation: an operation on order keys, optionally negated.
template <typename Op, bool Negated = false> struct Relation {
  using Operation = Op;
  static constexpr bool kNegated = Negated;
};

namespace detail {

/// @brief Whether V is the format's lane type or a SIMD vector of it.
template <typename Fmt, typename V>
inline constexpr bool is_lane_v = std::is_same_v<V, typename Fmt::Lane> || requires {
  requires std::is_same_v<typename V::value_type, typename Fmt::Lane>;
};

/// @brief All ones where the exponent field is nonzero, zero elsewhere.
/// @details The field plus its maximum carries into the next bit exactly when
/// the field is nonzero, which avoids a mask type.
template <typename Fmt, typename V> constexpr V normal_or_special(V bits) {
  const V field = (bits & Fmt::kInfinity) >> Fmt::kMantissaBits;
  return typename Fmt::Lane{0} - ((field + Fmt::kExponentMax) >> Fmt::kExponentBits);
}

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

/// @brief Stage 1: apply VOP3 or DPP source modifiers to one source.
template <typename Fmt, typename V> constexpr V modify(V bits, bool absolute, bool negate) {
  static_assert(detail::is_lane_v<Fmt, V>);
  if (absolute)
    bits = bits & Fmt::kMagnitude;
  if (negate)
    bits = bits ^ Fmt::kSign;
  return bits;
}

/// @brief Stage 2: flush a subnormal source to a zero of the same sign.
/// @details NaN, infinity, zero and normal encodings pass through unchanged.
template <typename Fmt, typename V> constexpr V flush_input(V bits, const Policy &policy) {
  static_assert(detail::is_lane_v<Fmt, V>);
  if (!policy.flush_inputs)
    return bits;
  return bits & (detail::normal_or_special<Fmt>(bits) | Fmt::kSign);
}

/// @brief Whether a source encoding is NaN.
template <typename Fmt, typename V> constexpr auto is_nan(V bits) {
  static_assert(detail::is_lane_v<Fmt, V>);
  return (bits & Fmt::kMagnitude) > Fmt::kInfinity;
}

/// @brief Stage 3: map a non-NaN encoding to a key that orders like its value.
/// @details Both zeros map to the same key. A positive encoding sets the sign
/// bit; a negative one inverts every bit of the format, reversing magnitude order.
template <typename Fmt, typename V> constexpr V order_key(V bits) {
  static_assert(detail::is_lane_v<Fmt, V>);
  bits = bits & Fmt::kBits;
  bits = bits & detail::nonzero<Fmt>(bits);
  const V negative = typename Fmt::Lane{0} - (bits >> (Fmt::kWidth - 1));
  return bits ^ ((negative & Fmt::kBits) | Fmt::kSign);
}

/// @brief Evaluate a relation on two sources that already carry their modifiers.
/// @returns bool for scalar lanes, or the key comparison's mask for SIMD lanes.
template <typename Fmt, typename Rel, typename V>
constexpr auto evaluate(V a, V b, const Policy &policy) {
  static_assert(detail::is_lane_v<Fmt, V>);
  a = flush_input<Fmt>(a & Fmt::kBits, policy);
  b = flush_input<Fmt>(b & Fmt::kBits, policy);
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
constexpr auto evaluate(V a, V b, uint32_t abs, uint32_t neg, const Policy &policy) {
  return evaluate<Fmt, Rel>(modify<Fmt>(a, abs & 1u, neg & 1u),
                            modify<Fmt>(b, (abs & 2u) != 0, (neg & 2u) != 0), policy);
}

} // namespace rocjitsu::amdgpu::comparison
