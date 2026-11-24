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
// These entry points model RDNA VOPDXY, not CDNA5 VOPD3.
// instruction_flags must be zero: RDNA VOPDXY encodes no modifiers or DPP.
// FMAC/DOT2ACC read old D as the accumulator. Literal FMA parameters contain raw
// FP32 bits. RDNA VOPDXY CNDMASK's condition is the wave's VCC mask. Source operands must
// already be decoded, including scalar/inline-constant replication for DOT2ACC.
// Legal VOPD register-bank, destination-parity and pair-dependency constraints
// are preconditions. Each call executes one operation, not the whole X/Y pair.
// Supply pre-pair source values when either slot overwrites the other's source;
// This applies to both RDNA3 and RDNA4: X-then-Y execution against live registers
// is valid only after checking that X does not overwrite any Y source.
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

// Compatibility uses the ordinary MUL underflow model: flush before rounding.
// The minimum-normal boundary matches gfx1201, improving on legacy host rounding.
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

// gfx1250 decoded FP64 dual slots. Each operand occupies two VGPRs; only
// NEG_A/NEG_B (and NEG_C for FMA) are encoded. Guest rounding and denormal
// controls follow the ordinary FP64 instructions. Stage both slots before writes.
int rj_arith_v_dual_add_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

int rj_arith_v_dual_mul_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

int rj_arith_v_dual_min_num_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user);

int rj_arith_v_dual_max_num_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user);

int rj_arith_v_dual_fma_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *excp_flag_user);

// gfx1250 FP32 slots accept encoded source NEG flags, but no ABS/CLAMP/OMOD.
// MUL returns UNSUPPORTED_FP_STATUS in mode 3 pending gfx1250 qualification.
// The additional integer slots have no modifiers. BITOP2 uses bits 0..7 of
// instruction_flags as the BITOP3 truth table evaluated with C=0.
int rj_arith_v_dual_add_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

int rj_arith_v_dual_sub_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

int rj_arith_v_dual_subrev_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           uint32_t *excp_flag_user);

int rj_arith_v_dual_mul_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

int rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                 uint64_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a, const uint32_t *const *b,
                                                 uint32_t *excp_flag_user);

int rj_arith_v_dual_min_num_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user);

int rj_arith_v_dual_max_num_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user);

int rj_arith_v_dual_fmac_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *excp_flag_user);

int rj_arith_v_dual_fma_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_dual_sub_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

int rj_arith_v_dual_lshrrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

int rj_arith_v_dual_ashrrev_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

int rj_arith_v_dual_min_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

int rj_arith_v_dual_max_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

int rj_arith_v_dual_bitop2_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

// VOPD3 CNDMASK applies NEG_A/NEG_B as sign-bit flips before selection.
int rj_arith_v_dual_cndmask_b32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t condition);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
