// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_DUAL_H_
#define RJ_ARITH_DUAL_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One decoded VOPD operation, Wave32 only. Operand layouts, numerical semantics,
// CPU paths follow the corresponding ordinary instruction. Unimplemented
// FP-status reporting returns UNSUPPORTED_FP_STATUS; loose calls skip it.
// instruction_flags must be zero: VOPD encodes no arithmetic modifiers or DPP.
// FMAC/DOT2ACC read old D as the accumulator. Literal FMA parameters contain raw
// FP32 bits. CNDMASK's condition is the wave's VCC mask. Source operands must
// already be decoded, including scalar/inline-constant replication for DOT2ACC.
// Legal VOPD register-bank, destination-parity and pair-dependency constraints
// are preconditions. Each call executes one operation, not the whole X/Y pair.
// Supply pre-pair source values when either slot overwrites the other's source;
// RDNA3 callers must stage such inputs. Legal RDNA4 pairs may execute X then Y
// because X may not overwrite a Y source.
// No Wave64 forms exist. Shared floating arithmetic uses RDNA3 APIs without _gfx12
// with TRAPSTS and _gfx12 APIs with EXCP_FLAG_USER, including architecture-specific
// NaN/IEEE behavior. DOT2ACC additionally has different accumulation models.
// FMAC and literal FMA retain exact fused results on both architectures but reject
// FP status reporting in mode 3.
// DUAL_MUL inherits ordinary MUL's nearest-even FP-status reporting contract.
// Invalid calls preserve the outputs.
int rj_arith_v_dual_fmac_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_dual_fmac_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *excp_flag_user);

int rj_arith_v_dual_fmaak_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t literal, uint32_t *trapsts);
int rj_arith_v_dual_fmaak_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t literal, uint32_t *excp_flag_user);

int rj_arith_v_dual_fmamk_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                  const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_dual_fmamk_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t literal,
                                        const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_dual_mul_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_dual_mul_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);

int rj_arith_v_dual_add_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_dual_add_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);

int rj_arith_v_dual_sub_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_dual_sub_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);

int rj_arith_v_dual_subrev_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_dual_subrev_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *excp_flag_user);

int rj_arith_v_dual_mul_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *trapsts);
int rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b,
                                               uint32_t *excp_flag_user);

int rj_arith_v_dual_mov_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_dual_cndmask_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t condition);

int rj_arith_v_dual_max_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_dual_min_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_dual_max_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_dual_min_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_dual_add_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

int rj_arith_v_dual_lshlrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

int rj_arith_v_dual_and_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

int rj_arith_v_dual_dot2acc_f32_f16_w32(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);

int rj_arith_v_dual_dot2acc_f32_bf16_w32(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *trapsts);

int rj_arith_v_dual_dot2acc_f32_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                              uint64_t instruction_flags, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              uint32_t *excp_flag_user);

int rj_arith_v_dual_dot2acc_f32_bf16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b,
                                               uint32_t *excp_flag_user);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
