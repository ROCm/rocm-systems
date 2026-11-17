// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited gfx908/gfx942 predicates reuse the shared numeric comparators. Compatibility also
// preserves the legacy CLAMP-only INVALID policy, including constant predicates.

#include "common/validation.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

using Compare = int (*)(uint64_t, uint64_t, uint64_t, uint64_t *, const uint32_t *const *,
                        const uint32_t *const *, uint32_t *);

// Compute masks before publishing either output, so sources may overlap D or
// TRAPSTS. Loose and numeric-only modes leave TRAPSTS untouched.
int compare_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                   const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts,
                   Compare numeric, Compare unordered) {
  const bool signal = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY &&
                      (mode & RJ_ARITH_ALU_CLAMP);
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  // No classification or scratch result is needed on the ordinary numeric path.
  if (!signal) {
    return numeric(flags, exec_mask, mode, d, a, b, nullptr);
  }
  uint64_t result = 0, nans = 0;
  if (int error = numeric(flags, exec_mask, mode, &result, a, b, nullptr)) {
    return error;
  }
  // The unordered predicate supplies the same selected/DPP-filtered NaN mask
  // through the existing scalar/AVX2/AVX-512 arithmetic path.
  if (int error = unordered(flags, exec_mask, mode, &nans, a, b, nullptr)) {
    return error;
  }
  __builtin_memcpy(static_cast<void *>(d), &result, sizeof result);
  if (nans) {
    *trapsts |= RJ_ARITH_EXCEPTION_INVALID;
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_f_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_f_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_lt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_lt_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_eq_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_eq_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_le_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_le_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_gt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_gt_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_lg_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_lg_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_ge_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_ge_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_o_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_o_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_u_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_u_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nge_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nge_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nlg_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nlg_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_ngt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_ngt_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nle_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nle_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_neq_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_neq_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nlt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nlt_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_t_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_t_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_f_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_f_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_lt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_lt_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_eq_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_eq_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_le_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_le_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_gt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_gt_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_lg_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_lg_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_ge_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_ge_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_o_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_o_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_u_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_u_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nge_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nge_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nlg_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nlg_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_ngt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_ngt_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nle_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nle_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_neq_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_neq_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nlt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nlt_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_t_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_t_f16_w64, rj_arith_v_cmp_u_f16_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_f_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_f_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_lt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_lt_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_eq_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_eq_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_le_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_le_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_gt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_gt_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_lg_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_lg_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_ge_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_ge_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_o_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_o_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_u_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_u_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nge_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nge_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nlg_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nlg_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_ngt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_ngt_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nle_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nle_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_neq_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_neq_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nlt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nlt_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_t_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_t_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_f_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_f_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_lt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_lt_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_eq_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_eq_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_le_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_le_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_gt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_gt_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_lg_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_lg_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_ge_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_ge_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_o_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_o_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_u_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_u_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nge_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nge_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nlg_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nlg_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_ngt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_ngt_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nle_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nle_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_neq_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_neq_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nlt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nlt_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_t_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_t_f32_w64, rj_arith_v_cmp_u_f32_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_f_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_f_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_lt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_lt_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_eq_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_eq_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_le_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_le_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_gt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_gt_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_lg_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_lg_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_ge_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_ge_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_o_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_o_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_u_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_u_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nge_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nge_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nlg_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nlg_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_ngt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_ngt_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nle_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nle_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_neq_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_neq_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_nlt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_nlt_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmp_t_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmp_t_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_f_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_f_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_lt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_lt_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_eq_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_eq_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_le_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_le_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_gt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_gt_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_lg_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_lg_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_ge_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_ge_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_o_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_o_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_u_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_u_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nge_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nge_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nlg_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nlg_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_ngt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_ngt_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nle_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nle_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_neq_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_neq_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_nlt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_nlt_f64_w64, rj_arith_v_cmp_u_f64_w64);
}

// Write the predicate mask and preserve gfx908's Compatibility signaling policy.
int rj_arith_v_cmpx_t_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts) {
  return compare_gfx908(flags, exec_mask, instruction_flags, d, a, b, trapsts,
                        rj_arith_v_cmpx_t_f64_w64, rj_arith_v_cmp_u_f64_w64);
}
