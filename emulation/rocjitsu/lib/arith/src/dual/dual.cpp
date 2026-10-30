// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Individual decoded VOPD operations delegated to ordinary Wave32 arithmetic entry points.
// Owns public entry points and their validation/dispatch policy.

// One decoded VOPD operation. Arithmetic follows the corresponding ordinary
// instruction; VOPD has no instruction-modifier fields and is Wave32 only.

#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_fmac_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_fmac_f32_w32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_fmac_f32_w32(flags, exec_mask, 0, d, a, b, trapsts);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_fmac_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_fmac_f32_w32_gfx12(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_fmac_f32_w32_gfx12(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_fmaak_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t literal, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_fmaak_f32_w32(flags, 0, 0, d, a, b, literal, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_fmaak_f32_w32(flags, exec_mask, 0, d, a, b, literal, trapsts);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_fmaak_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t literal, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_fmaak_f32_w32_gfx12(flags, 0, 0, d, a, b, literal, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_fmaak_f32_w32_gfx12(flags, exec_mask, 0, d, a, b, literal, excp_flag_user);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_fmamk_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                  const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_fmamk_f32_w32(flags, 0, 0, d, a, literal, b, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_fmamk_f32_w32(flags, exec_mask, 0, d, a, literal, b, trapsts);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_fmamk_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t literal,
                                        const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_fmamk_f32_w32_gfx12(flags, 0, 0, d, a, literal, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_fmamk_f32_w32_gfx12(flags, exec_mask, 0, d, a, literal, b, excp_flag_user);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_mul_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_mul_f32_w32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_mul_f32_w32(flags, exec_mask, 0, d, a, b, trapsts);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_mul_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_mul_f32_w32_gfx12(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_mul_f32_w32_gfx12(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_add_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_add_f32_w32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_add_f32_w32(flags, exec_mask, 0, d, a, b, trapsts);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_add_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_add_f32_w32_gfx12(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_add_f32_w32_gfx12(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_sub_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_sub_f32_w32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_sub_f32_w32(flags, exec_mask, 0, d, a, b, trapsts);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_sub_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_sub_f32_w32_gfx12(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_sub_f32_w32_gfx12(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_subrev_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_subrev_f32_w32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_subrev_f32_w32(flags, exec_mask, 0, d, a, b, trapsts);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_subrev_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_subrev_f32_w32_gfx12(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_subrev_f32_w32_gfx12(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

// Multiply the operands with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_mul_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_mul_dx9_zero_f32_w32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_mul_dx9_zero_f32_w32(flags, exec_mask, 0, d, a, b, trapsts);
}

// Multiply the operands with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b,
                                               uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_mul_dx9_zero_f32_w32_gfx12(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_mul_dx9_zero_f32_w32_gfx12(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

// Copy the selected source value for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dual_mov_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_mov_b32_w32(flags, 0, 0, d, a);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_mov_b32_w32(flags, exec_mask, 0, d, a);
}

// Select between A and B using the per-lane condition for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dual_cndmask_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t condition) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_cndmask_b32_w32(flags, 0, 0, d, a, b, condition);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_cndmask_b32_w32(flags, exec_mask, 0, d, a, b, condition);
}

// Select the maximum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_max_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_max_num_f32_w32(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_max_num_f32_w32(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

// Select the minimum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_min_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_min_num_f32_w32(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_min_num_f32_w32(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_max_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_max_f32_w32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_max_f32_w32(flags, exec_mask, 0, d, a, b, trapsts);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_min_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_min_f32_w32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_min_f32_w32(flags, exec_mask, 0, d, a, b, trapsts);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dual_add_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_add_nc_u32_w32(flags, 0, 0, d, a, b);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_add_nc_u32_w32(flags, exec_mask, 0, d, a, b);
}

// Shift B left by A for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dual_lshlrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_lshlrev_b32_w32(flags, 0, 0, d, a, b);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_lshlrev_b32_w32(flags, exec_mask, 0, d, a, b);
}

// Compute bitwise AND for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dual_and_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_and_b32_w32(flags, 0, 0, d, a, b);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_and_b32_w32(flags, exec_mask, 0, d, a, b);
}

// Accumulate two packed products into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_dot2acc_f32_f16_w32(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_dot2_f32_f16_w32(flags, 0, 0, d, a, b, d, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_dot2_f32_f16_w32(flags, exec_mask, 0, d, a, b, d, trapsts);
}

// Accumulate two packed products into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_dot2acc_f32_bf16_w32(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_dot2_f32_bf16_w32(flags, 0, 0, d, a, b, d, trapsts);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_dot2_f32_bf16_w32(flags, exec_mask, 0, d, a, b, d, trapsts);
}

// Accumulate two packed products into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_dot2acc_f32_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                              uint64_t instruction_flags, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_dot2_f32_f16_w32_gfx12(flags, 0, 0, d, a, b, d, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_dot2_f32_f16_w32_gfx12(flags, exec_mask, 0, d, a, b, d, excp_flag_user);
}

// Accumulate two packed products into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dual_dot2acc_f32_bf16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b,
                                               uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags) {
    int error = rj_arith_v_dot2_f32_bf16_w32_gfx12(flags, 0, 0, d, a, b, d, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_dot2_f32_bf16_w32_gfx12(flags, exec_mask, 0, d, a, b, d, excp_flag_user);
}
