// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 dual-slot arithmetic delegates to the ordinary instruction models.
// The decoder owns pair staging and register-bank constraints.

#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_dual.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_integer.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <stdint.h>

namespace {

// Reject modifiers absent from the slot encoding before evaluating arithmetic.
// Probe the ordinary instruction with empty EXEC to preserve its error precedence.
template <auto Function, typename... Args>
int execute(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint64_t allowed_modifiers,
            Args... args) {
  if (modifiers & ~allowed_modifiers) {
    const int error = Function(flags, 0, 0, args...);
    return error ? error : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return Function(flags, exec_mask, modifiers, args...);
}

} // namespace

// Add A and B for active lanes, preserving inactive destination words.
int rj_arith_v_dual_add_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user) {
  return execute<rj_arith_v_add_f64_w32_gfx12>(flags, exec_mask, instruction_flags,
                                               RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                               excp_flag_user);
}

// Multiply A and B for active lanes, preserving inactive destination words.
int rj_arith_v_dual_mul_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user) {
  return execute<rj_arith_v_mul_f64_w32_gfx1250>(flags, exec_mask, instruction_flags,
                                                 RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                                 excp_flag_user);
}

// Select the numeric minimum of A and B for active lanes, preserving inactive destination words.
int rj_arith_v_dual_min_num_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user) {
  return execute<rj_arith_v_min_num_f64_w32>(flags, exec_mask, instruction_flags,
                                             RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                             excp_flag_user);
}

// Select the numeric maximum of A and B for active lanes, preserving inactive destination words.
int rj_arith_v_dual_max_num_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user) {
  return execute<rj_arith_v_max_num_f64_w32>(flags, exec_mask, instruction_flags,
                                             RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                             excp_flag_user);
}

// Compute fused A*B+C for active lanes, preserving inactive destination words.
int rj_arith_v_dual_fma_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *excp_flag_user) {
  return execute<rj_arith_v_fma_f64_w32_gfx1250>(
      flags, exec_mask, instruction_flags,
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C, d, a, b, c, excp_flag_user);
}

// Execute the decoded ADD slot with guest FP controls and source negation.
int rj_arith_v_dual_add_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user) {
  return execute<rj_arith_v_add_f32_w32_gfx12>(flags, exec_mask, instruction_flags,
                                               RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                               excp_flag_user);
}

// Execute the decoded SUB slot with guest FP controls and source negation.
int rj_arith_v_dual_sub_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user) {
  return execute<rj_arith_v_sub_f32_w32_gfx12>(flags, exec_mask, instruction_flags,
                                               RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                               excp_flag_user);
}

// Execute the decoded SUBREV slot with guest FP controls and source negation.
int rj_arith_v_dual_subrev_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           uint32_t *excp_flag_user) {
  return execute<rj_arith_v_subrev_f32_w32_gfx12>(flags, exec_mask, instruction_flags,
                                                  RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                                  excp_flag_user);
}

// Execute the decoded MUL slot with guest FP controls and source negation.
int rj_arith_v_dual_mul_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user) {
  // gfx1201 MUL status captures do not qualify gfx1250 FP-status reporting.
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    const int error = execute<rj_arith_v_mul_f32_w32_gfx12>(flags, 0, instruction_flags,
                                                            RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B,
                                                            d, a, b, excp_flag_user);
    return error ? error : RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return execute<rj_arith_v_mul_f32_w32_gfx12>(flags, exec_mask, instruction_flags,
                                               RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                               excp_flag_user);
}

// Execute the decoded MUL_DX9_ZERO slot with guest FP controls and source negation.
int rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                 uint64_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a, const uint32_t *const *b,
                                                 uint32_t *excp_flag_user) {
  return execute<rj_arith_v_mul_dx9_zero_f32_w32_gfx1250>(flags, exec_mask, instruction_flags,
                                                          RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B,
                                                          d, a, b, excp_flag_user);
}

// Execute the decoded MIN_NUM slot with guest FP controls and source negation.
int rj_arith_v_dual_min_num_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user) {
  return execute<rj_arith_v_min_num_f32_w32>(flags, exec_mask, instruction_flags,
                                             RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                             excp_flag_user);
}

// Execute the decoded MAX_NUM slot with guest FP controls and source negation.
int rj_arith_v_dual_max_num_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user) {
  return execute<rj_arith_v_max_num_f32_w32>(flags, exec_mask, instruction_flags,
                                             RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                             excp_flag_user);
}

// Execute the decoded FMAC slot with guest FP controls and source negation.
int rj_arith_v_dual_fmac_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *excp_flag_user) {
  return execute<rj_arith_v_fmac_f32_w32_gfx12>(flags, exec_mask, instruction_flags,
                                                RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                                excp_flag_user);
}

// Execute the decoded FMA slot with guest FP controls and source negation.
int rj_arith_v_dual_fma_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *excp_flag_user) {
  return execute<rj_arith_v_fma_f32_w32_gfx12>(
      flags, exec_mask, instruction_flags,
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C, d, a, b, c, excp_flag_user);
}

// Execute the integer SUB_NC_U32 slot; no arithmetic modifiers are encoded.
int rj_arith_v_dual_sub_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  return execute<rj_arith_v_sub_nc_u32_w32>(flags, exec_mask, instruction_flags, 0, d, a, b);
}

// Execute the integer LSHRREV_B32 slot; no arithmetic modifiers are encoded.
int rj_arith_v_dual_lshrrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  return execute<rj_arith_v_lshrrev_b32_w32>(flags, exec_mask, instruction_flags, 0, d, a, b);
}

// Execute the integer ASHRREV_I32 slot; no arithmetic modifiers are encoded.
int rj_arith_v_dual_ashrrev_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  return execute<rj_arith_v_ashrrev_i32_w32>(flags, exec_mask, instruction_flags, 0, d, a, b);
}

// Execute the integer MIN_I32 slot; no arithmetic modifiers are encoded.
int rj_arith_v_dual_min_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return execute<rj_arith_v_min_i32_w32>(flags, exec_mask, instruction_flags, 0, d, a, b);
}

// Execute the integer MAX_I32 slot; no arithmetic modifiers are encoded.
int rj_arith_v_dual_max_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return execute<rj_arith_v_max_i32_w32>(flags, exec_mask, instruction_flags, 0, d, a, b);
}

// Evaluate BITOP3's encoded truth table with C=0, as specified for BITOP2.
int rj_arith_v_dual_bitop2_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  const uint32_t zero[32] = {};
  const uint32_t *c = zero;
  return execute<rj_arith_v_bitop3_b32_w32>(flags, exec_mask, instruction_flags,
                                            RJ_ARITH_BITOP3_TABLE_MASK, d, a, b, &c);
}

// Select words by condition after the VOPD3 source-sign flips.
int rj_arith_v_dual_cndmask_b32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t condition) {
  return execute<rj_arith_v_cndmask_b32_w32>(flags, exec_mask, instruction_flags,
                                             RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B, d, a, b,
                                             condition);
}
