// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file output_modifier.h
/// @brief Apply OMOD, then CLAMP, to destination-format results.
/// @details Disabled stages preserve the input. Callers resolve effective OMOD
/// and MODE through output_modifier_policy in simd_glue.h.
///
/// Scope:
/// - Shared scalar/SIMD stage for migrated VOP3 results on every target.
/// - Older targets enable OMOD only with IEEE=0 and output denormals flushed.
///
/// Paths outside the shared stage:
/// - Not yet migrated:
///   - Host-result scalar lowering and SIMD destination helpers: separate
///     OMOD/CLAMP implementations for remaining F32/F64 operations.
///   - Specialized generators using vop3_dst_mod: F32 intermediate modifiers,
///     including before F16 narrowing in DIV_FIXUP.
///   - SDWA F16 (dpp_sdwa_ops.h): separate arithmetic and rounded-TRANS
///     helpers; no captures.
/// - Intentional exception:
///   - Pseudo-scalar F32 transcendentals: scale the exact value before one F32
///     rounding; match gfx1201 captures.
///
/// Note: Migration must preserve each operation's rounding and flushing:
/// - SDWA F16 arithmetic and F16 TRANS flush outputs via output_denormal.h.
/// - FMA tininess and OMOD zero/subnormal rules keep their own handling.
/// - F32 arithmetic with effective OMOD flushes output denormals before scaling.
///
/// Hardware evidence:
/// - gfx1201 (RDNA4): every F16/F32/F64 min/max matches under all probed MODEs.
///   OMOD overflow checked both signs, four rounding modes, FP16_OVFL off/on.
///   Captures also establish round-before-modifiers for affected arithmetic,
///   F16 unary/TRANS, conversion and division results.
/// - gfx1100 (RDNA3, W7900): IEEE=0 F16 MUL/ADD div:2 captures match gfx1201
///   for overflow before scaling and the sign of a halved negative normal.
/// - gfx1030 (RDNA2, RX 6800 XT) / gfx1010 (RDNA1, RX 5700 XT): IEEE=0 with
///   output denormals flushed matches gfx1201 on all non-NaN OMOD lanes;
///   gfx1010 matches gfx1030 bit for bit.
/// - RDNA3.5 / CDNA1-5: extrapolated; no captures, hardware verification required.
///
/// CDNA4 V_MINIMUM3_F32/V_MAXIMUM3_F32 retain the emulator's wave-IEEE-based
/// OMOD policy. The ISA specifies forced IEEE=1 for these instructions and
/// disables OMOD under IEEE=1; whether that override also suppresses their
/// OMOD needs hardware verification and a separate instruction-policy follow-up.
///
/// legacy_rounded_result_modifier_cases in valu_fp_mode_test.cpp decodes
/// gfx1100/gfx1030 captures, including IEEE=1 controls.
/// Encodings: F16 in the low half of uint32; F32/F64 in uint32/uint64.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/lane_select.h"

#include <cstdint>

namespace rocjitsu::amdgpu::output_modifier {

/// @brief Per-instruction output-modifier settings, fixed before any lane is evaluated.
struct Policy {
  /// Effective OMOD: 0 none, 1 multiply by 2, 2 multiply by 4, 3 multiply by 0.5.
  uint32_t omod = 0;
  bool clamp = false;
  /// Whether CLAMP turns a NaN into +0.
  bool clamp_nan_to_zero = false;
  /// MODE.FP_ROUND field of the result format: 0 nearest even, 1 toward +inf,
  /// 2 toward -inf, 3 toward zero.
  uint32_t round_mode = 0;
  /// Saturate F16 overflow to the largest finite value; ignored for F32/F64.
  bool fp16_ovfl = false;
};

namespace detail {

using lane::choose;

/// @brief The encoding of 1.0: a biased exponent and zero fraction bits.
template <typename Fmt>
inline constexpr typename Fmt::Lane kOne = (Fmt::kExponentMax >> 1) << Fmt::kMantissaBits;

/// @brief Choose signed infinity or the largest finite value on overflow.
template <typename Fmt> constexpr typename Fmt::Lane overflow(bool negative, const Policy &policy) {
  const bool saturate_f16 = Fmt::kWidth == 16 && policy.fp16_ovfl;
  // Nearest-even produces infinity; directed rounding does so only toward this sign.
  const bool rounds_to_infinity =
      policy.round_mode == 0 || policy.round_mode == (negative ? 2u : 1u);
  const typename Fmt::Lane magnitude =
      rounds_to_infinity && !saturate_f16 ? Fmt::kInfinity : Fmt::kInfinity - 1;
  return negative ? Fmt::kSign | magnitude : magnitude;
}

} // namespace detail

/// @brief Stage 1: scale a result by OMOD.
///
/// | Input or scaling outcome | Result when OMOD is enabled                |
/// |--------------------------|--------------------------------------------|
/// | NaN or infinity          | Original bits                              |
/// | Zero or subnormal        | +0, including for negative inputs          |
/// | Halving underflows       | Zero with the input's sign                 |
/// | Multiplication overflows | Infinity or max finite, per rounding mode  |
/// | Otherwise                | Exact input * 2, input * 4, or input / 2   |
///
/// FP16_OVFL overrides the overflow rule for F16, selecting max finite.
template <typename Fmt, typename V> constexpr V scale(V bits, const Policy &policy) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  using Lane = typename Fmt::Lane;
  using detail::choose;
  if (policy.omod == 0)
    return bits;
  const V magnitude = bits & Fmt::kMagnitude;
  const V exponent = magnitude >> Fmt::kMantissaBits;
  V scaled;
  if (policy.omod == 3) {
    // input / 2: subtract one exponent step. Every normal with exponent 1
    // would become subnormal, so return a signed zero for those inputs.
    const auto underflows = exponent == Lane{1};
    const V signed_zero = bits & Fmt::kSign;
    const V halved = bits - Fmt::kMinNormal;
    scaled = choose(underflows, signed_zero, halved);
  } else {
    // input * 2 or input * 4: add one or two exponent steps.
    const Lane exponent_steps = policy.omod;
    const auto overflows = exponent >= Fmt::kExponentMax - exponent_steps;
    const auto negative = (bits & Fmt::kSign) != Lane{0};
    const V overflow_result = choose(negative, V(detail::overflow<Fmt>(true, policy)),
                                     V(detail::overflow<Fmt>(false, policy)));
    const V multiplied = bits + (exponent_steps << Fmt::kMantissaBits);
    scaled = choose(overflows, overflow_result, multiplied);
  }
  // Override the numeric candidate for inputs outside the normal range.
  const auto zero_or_subnormal = exponent == Lane{0};
  const auto nan_or_infinity = exponent == Fmt::kExponentMax;
  scaled = choose(zero_or_subnormal, V(Lane{0}), scaled);
  return choose(nan_or_infinity, bits, scaled);
}

/// @brief Stage 2: clamp a result to [+0, 1.0] when CLAMP is set.
///
/// | Input             | Result when CLAMP is enabled         |
/// |-------------------|--------------------------------------|
/// | NaN               | +0 if clamp_nan_to_zero, else input  |
/// | Negative or -0    | +0                                   |
/// | Greater than 1.0  | 1.0                                  |
/// | Otherwise         | Original bits                        |
template <typename Fmt, typename V> constexpr V clamp(V bits, const Policy &policy) {
  static_assert(fp_format::is_lane_v<Fmt, V>);
  using Lane = typename Fmt::Lane;
  using detail::choose;
  if (!policy.clamp)
    return bits;
  const V magnitude = bits & Fmt::kMagnitude;
  const auto negative = (bits & Fmt::kSign) != Lane{0};
  const auto is_nan = magnitude > Fmt::kInfinity;
  const V zero(Lane{0});
  const V one(detail::kOne<Fmt>);
  // min(max(input, +0), 1.0), with the NaN rule applied separately below.
  const V capped_result = choose(magnitude > detail::kOne<Fmt>, one, bits);
  const V numeric_result = choose(negative, zero, capped_result);
  const V nan_result = policy.clamp_nan_to_zero ? zero : bits;
  return choose(is_nan, nan_result, numeric_result);
}

/// @brief Apply OMOD, then CLAMP, to a result in its destination format.
template <typename Fmt, typename V> constexpr V apply(V bits, const Policy &policy) {
  const V scaled = scale<Fmt>(bits, policy);
  return clamp<Fmt>(scaled, policy);
}

} // namespace rocjitsu::amdgpu::output_modifier
