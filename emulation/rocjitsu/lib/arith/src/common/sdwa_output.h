// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited SDWA FP32 output scaling shared by scalar and SIMD lanes.
// Operates on the rounded arithmetic result, after FP-status classification.
#pragma once

#include "common/isa.h"
#include "conversion/model/output_modifier.h"
#include "conversion/model/round_increment.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Build an output policy from architecture-resolved SDWA modifier fields.
inline conversion_model::output_modifier::Policy sdwa_output_policy(uint64_t flags, uint32_t mode) {
  return {(mode >> 6) & 3, bool(mode & RJ_ARITH_ALU_CLAMP), !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF),
          uint32_t((flags >> 24) & 3), bool(flags & RJ_ARITH_FP16_OVFL)};
}

// Scale FP32 encodings with the old SDWA rounding, NaN and tiny-output rules.
// All candidates use defined unsigned shifts, including inactive SIMD candidates.
template <typename V>
V sdwa_scale_f32(V bits, const conversion_model::output_modifier::Policy &policy) {
  using F32 = conversion_model::fp_format::F32;
  static_assert(conversion_model::fp_format::is_lane_v<F32, V>);
  using conversion_model::lane::choose;
  if (!policy.omod) {
    return bits;
  }
  const V sign = bits & V(F32::kSign);
  const V field = (bits >> 23) & V(0xffU);
  const V fraction = bits & V(F32::kFraction);
  const auto subnormal = field == V(0U);
  // A bias of 128 keeps even a normalized minimum subnormal's exponent positive.
  const V shift = V(24U) - conversion_model::fp_format::bit_width(fraction | V(1U));
  const V significand = choose(subnormal, fraction << shift, fraction | V(F32::kMinNormal));
  V exponent = choose(subnormal, V(129U) - shift, field + V(128U));
  exponent = policy.omod == 3 ? exponent - V(1U) : exponent + V(policy.omod);
  const V normal = sign | ((exponent - V(128U)) << 23) | (significand & V(F32::kFraction));
  const auto negative = sign != V(0U);
  const V overflow =
      choose(negative, V(conversion_model::output_modifier::detail::overflow<F32>(true, policy)),
             V(conversion_model::output_modifier::detail::overflow<F32>(false, policy)));

  // Round first: a value below minimum normal can round up to minimum normal.
  // Other tiny outputs become +0, including negative and exactly representable ones.
  const auto tiny = exponent < V(129U);
  const V right = choose(tiny, V(129U) - exponent, V(1U));
  const V kept = significand >> right;
  const V discarded = significand & ((V(1U) << right) - V(1U));
  const V halfway = V(1U) << (right - V(1U));
  const auto increment = conversion_model::round_increment::rounds_up<F32>(
      policy.round_mode, negative, kept, discarded, halfway);
  const V rounded = kept + choose(increment, V(1U), V(0U));
  const V underflow = choose(rounded == V(F32::kMinNormal), sign | V(F32::kMinNormal), V(0U));
  const V scaled = choose(exponent >= V(383U), overflow, choose(tiny, underflow, normal));

  // Scaling quiets NaNs but preserves infinity; either sign of zero becomes +0.
  const V special = choose(fraction == V(0U), bits, bits | V(F32::kQuiet));
  const auto zero = subnormal && fraction == V(0U);
  return choose(field == V(0xffU), special, choose(zero, V(0U), scaled));
}

// Apply inherited floating output controls before the VM places destination bytes.
template <typename V>
V sdwa_output_f32(V bits, const conversion_model::output_modifier::Policy &policy) {
  return conversion_model::output_modifier::clamp<conversion_model::fp_format::F32>(
      sdwa_scale_f32(bits, policy), policy);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
