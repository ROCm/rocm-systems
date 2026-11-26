// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA FP16 math reuses Rocjitsu's existing models, half narrowing and guest-mode policies.
// RDNA validation does not establish empirical bit-exactness on these architectures.

#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <stdint.h>

namespace {

// Forward the inherited model while rejecting strict requests before touching storage.
// Compatibility retains the implementation's numeric and FP-status policy.
template <typename Call, typename Mask>
int inherited_half_math(Call call, uint64_t flags, Mask exec_mask, uint64_t mode,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *state) {
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  if (semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
    if (int error = rocjitsu::arith::validate_semantics(flags)) {
      return error;
    }
  }
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  return call(flags, exec_mask, mode, d, a, state);
}

} // namespace

// Evaluate the inherited FP16 rcp model for active lanes and preserve other output bits.
int rj_arith_v_rcp_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_rcp_f16_w32, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 rcp model for active lanes and preserve other output bits.
int rj_arith_v_rcp_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_rcp_f16_w64, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 rcp model for active lanes and preserve other output bits.
int rj_arith_v_rcp_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_rcp_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 rsq model for active lanes and preserve other output bits.
int rj_arith_v_rsq_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_rsq_f16_w32, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 rsq model for active lanes and preserve other output bits.
int rj_arith_v_rsq_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_rsq_f16_w64, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 rsq model for active lanes and preserve other output bits.
int rj_arith_v_rsq_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_rsq_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 sqrt model for active lanes and preserve other output bits.
int rj_arith_v_sqrt_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_sqrt_f16_w32, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 sqrt model for active lanes and preserve other output bits.
int rj_arith_v_sqrt_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_sqrt_f16_w64, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 sqrt model for active lanes and preserve other output bits.
int rj_arith_v_sqrt_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_sqrt_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 exp model for active lanes and preserve other output bits.
int rj_arith_v_exp_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_exp_f16_w32, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 exp model for active lanes and preserve other output bits.
int rj_arith_v_exp_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_exp_f16_w64, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 exp model for active lanes and preserve other output bits.
int rj_arith_v_exp_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_exp_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 log model for active lanes and preserve other output bits.
int rj_arith_v_log_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_log_f16_w32, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 log model for active lanes and preserve other output bits.
int rj_arith_v_log_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_log_f16_w64, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 log model for active lanes and preserve other output bits.
int rj_arith_v_log_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_log_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 sin model for active lanes and preserve other output bits.
int rj_arith_v_sin_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_sin_f16_w32, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 sin model for active lanes and preserve other output bits.
int rj_arith_v_sin_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_sin_f16_w64, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 sin model for active lanes and preserve other output bits.
int rj_arith_v_sin_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_sin_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 cos model for active lanes and preserve other output bits.
int rj_arith_v_cos_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_cos_f16_w32, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 cos model for active lanes and preserve other output bits.
int rj_arith_v_cos_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math(rj_arith_v_cos_f16_w64, flags, exec_mask, instruction_flags, d, a,
                             trapsts);
}

// Evaluate the inherited FP16 cos model for active lanes and preserve other output bits.
int rj_arith_v_cos_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_cos_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}
