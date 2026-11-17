// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited packed-half output finalization shared by scalar and SIMD lanes.
// Arithmetic owns scaling; these helpers only finalize encodings and clamp.
#pragma once

#include "common/isa.h"
#include "conversion/model/fp_format.h"
#include "conversion/model/output_modifier.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Build an output policy from architecture-resolved SDWA modifier fields.
inline conversion_model::output_modifier::Policy sdwa_output_policy(uint64_t flags, uint32_t mode) {
  return {(mode >> 6) & 3, bool(mode & RJ_ARITH_ALU_CLAMP), !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF),
          uint32_t((flags >> 24) & 3), bool(flags & RJ_ARITH_FP16_OVFL)};
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

// Finalize both halves of a packed result before the VM selects destination bytes.
template <typename V>
V sdwa_finalize_packed_f16(V bits, const conversion_model::output_modifier::Policy &policy) {
  return sdwa_finalize_f16(bits & V(0xffffU), policy) |
         (sdwa_finalize_f16(bits >> 16, policy) << 16);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
