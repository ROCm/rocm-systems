// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA FP32 math reuses the numeric models already used by Rocjitsu's CDNA handlers.
// Compatibility retains those models and their status policies. Their validation on
// RDNA hardware does not establish empirical bit-exactness on these architectures.

#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <stdint.h>

namespace {

// Forward the inherited model without advertising new hardware-validated semantics.
// Reject strict requests before any destination or FP-status access.
template <typename Call, typename Mask>
int inherited_math(Call call, uint64_t flags, Mask exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *state) {
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  // The selected implementation resolves Compatibility to its own numeric/status policy.
  if (semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
    if (int error = rocjitsu::arith::validate_semantics(flags)) {
      return error;
    }
  }
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  return call(flags, exec_mask, instruction_flags, d, a, state);
}

} // namespace

// Evaluate the inherited rcp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rcp_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_rcp_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited rcp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rcp_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_rcp_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited rcp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rcp_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_rcp_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited rsq model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rsq_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_rsq_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited rsq model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rsq_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_rsq_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited rsq model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rsq_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_rsq_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited sqrt model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sqrt_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return inherited_math(rj_arith_v_sqrt_f32_w32, flags, exec_mask, instruction_flags, d, a,
                        trapsts);
}

// Evaluate the inherited sqrt model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sqrt_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return inherited_math(rj_arith_v_sqrt_f32_w64, flags, exec_mask, instruction_flags, d, a,
                        trapsts);
}

// Evaluate the inherited sqrt model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sqrt_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_sqrt_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited exp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_exp_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_exp_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited exp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_exp_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_exp_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited exp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_exp_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_exp_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited log model for the selected lanes, preserving inactive outputs.
int rj_arith_v_log_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_log_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited log model for the selected lanes, preserving inactive outputs.
int rj_arith_v_log_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_log_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited log model for the selected lanes, preserving inactive outputs.
int rj_arith_v_log_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_log_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited sin model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sin_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_sin_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited sin model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sin_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_sin_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited sin model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sin_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_sin_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited cos model for the selected lanes, preserving inactive outputs.
int rj_arith_v_cos_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_cos_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited cos model for the selected lanes, preserving inactive outputs.
int rj_arith_v_cos_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_math(rj_arith_v_cos_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited cos model for the selected lanes, preserving inactive outputs.
int rj_arith_v_cos_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_cos_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Legacy encoding retains the same model and policy as ordinary CDNA exp.
int rj_arith_v_exp_legacy_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  return rj_arith_v_exp_f32_w32_gfx942(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Legacy encoding retains the same model and policy as ordinary CDNA exp.
int rj_arith_v_exp_legacy_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  return rj_arith_v_exp_f32_w64_gfx942(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Legacy encoding retains the same model and policy as ordinary CDNA log.
int rj_arith_v_log_legacy_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  return rj_arith_v_log_f32_w32_gfx942(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Legacy encoding retains the same model and policy as ordinary CDNA log.
int rj_arith_v_log_legacy_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  return rj_arith_v_log_f32_w64_gfx942(flags, exec_mask, instruction_flags, d, a, trapsts);
}
