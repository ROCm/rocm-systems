// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited SDWA output processing shared by scalar and SIMD lanes.
// Callers select scaling or finalization of an already-rounded floating result.
#pragma once

#include "common/isa.h"
#include "common/sdwa_dispatch.h"
#include "conversion/model/fp_format.h"
#include "conversion/model/output_modifier.h"
#include "conversion/model/round_increment.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Build an output policy from architecture-resolved SDWA modifier fields.
inline conversion_model::output_modifier::Policy
sdwa_output_policy(uint64_t flags, uint32_t mode, SdwaOutput output = SdwaOutput::F32Mode) {
  return {(mode >> 6) & 3, bool(mode & RJ_ARITH_ALU_CLAMP), !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF),
          output == SdwaOutput::F32Nearest ? 0U : uint32_t((flags >> 24) & 3),
          bool(flags & RJ_ARITH_FP16_OVFL)};
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

// Scale a rounded output word, then clamp, preserving an unselected half when present.
template <SdwaOutput Output, typename V>
V sdwa_scale_word(V bits, uint32_t mode, const conversion_model::output_modifier::Policy &policy) {
  if constexpr (Output == SdwaOutput::F32Nearest) {
    return conversion_model::output_modifier::apply<conversion_model::fp_format::F32>(bits, policy);
  } else if constexpr (Output == SdwaOutput::F32Mode) {
    return sdwa_output_f32(bits, policy);
  } else {
    const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
    const V half = (bits >> shift) & V(0xffffU);
    const V field =
        conversion_model::output_modifier::apply<conversion_model::fp_format::F16>(half, policy);
    return (bits & ~V(0xffffU << shift)) | (field << shift);
  }
}

// Finalize an already-rounded half: active OMOD canonicalizes tiny values and zeros.
// The producer owns scaling; this stage only applies final encoding rules and clamp.
template <typename V>
V sdwa_finalize_f16(V bits, const conversion_model::output_modifier::Policy &policy) {
  using F16 = conversion_model::fp_format::F16;
  static_assert(conversion_model::fp_format::is_lane_v<F16, V>);
  if (policy.omod) {
    bits = conversion_model::lane::choose((bits & V(0x7c00U)) == V(0U), V(0U), bits);
  }
  return conversion_model::output_modifier::clamp<F16>(bits, policy);
}

// Finalize one selected half while preserving the other destination half's encoding.
template <typename V>
V sdwa_finalize_half_word(V bits, uint32_t mode,
                          const conversion_model::output_modifier::Policy &policy) {
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const V field = sdwa_finalize_f16((bits >> shift) & V(0xffffU), policy);
  return (bits & ~V(0xffffU << shift)) | (field << shift);
}

// Finalize both halves of a packed result before the VM selects destination bytes.
template <typename V>
V sdwa_finalize_packed_f16(V bits, const conversion_model::output_modifier::Policy &policy) {
  return sdwa_finalize_f16(bits & V(0xffffU), policy) |
         (sdwa_finalize_f16(bits >> 16, policy) << 16);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
