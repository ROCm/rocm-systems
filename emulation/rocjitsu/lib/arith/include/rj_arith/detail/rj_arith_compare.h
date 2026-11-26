// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_COMPARE_H_
#define RJ_ARITH_COMPARE_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Floating comparisons write a scalar condition mask (CMP) or replacement EXEC
// mask (CMPX) to d. Inactive bits are zero, including for empty EXEC; d must
// always be writable and may alias any input word. Zero EXEC permits null VGPR
// pointers. Errors leave d unchanged. A/B use one VGPR for FP16/FP32 or a low/high
// pair for FP64. Supports ABS_A/B, NEG_A/B and FP16 HIGH_A/B. ABS precedes NEG.
// FP16/FP32 support DPP8/DPP16 on A before modifiers and half selection; B
// stays in its original lane, and filtered result bits are zero. FP64 rejects DPP.
// Supports loose and empirical-exact semantics. RJ_ARITH_FP_FLUSH_INPUT_DENORMALS
// flushes input subnormals to signed zero after modifiers; otherwise they are
// preserved. Signed zeros compare equal. NaNs make ordered relations false and
// their negations true. O/U test ordered/unordered. Host FP controls are never changed; arithmetic
// may raise sticky host flags. CLAMP selects signaling comparison: any NaN raises INVALID when
// reporting. Without CLAMP, only signaling NaNs raise INVALID. Comparison results are unchanged.
// RDNA4 exact reporting accumulates INPUT_DENORM for
// finite operand pairs containing a preserved subnormal. Only participating
// lanes contribute, after DPP filtering. RDNA3 entry points take
// trapsts; mode 4 FP status reporting is unsupported, even with empty
// EXEC. Both architectures support exact predicate results in mode 3.
// gfx12 Compatibility with CLAMP preserves the legacy wave-wide reporting gate:
// causes accumulate only when an active selected operand is nonfinite. Then all
// participating lanes contribute, including finite denormal pairs elsewhere in
// a Wave64. Without CLAMP, Compatibility leaves status unused. Reporting calls
// require valid status storage; numeric-only calls leave it unused.
// IEEE_MODE and output flushing do not affect predicate results. Loose mode
// leaves optional architectural state unchanged on both architectures.
int rj_arith_v_cmp_eq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_eq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_eq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_eq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_eq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_eq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_eq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_eq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_eq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_eq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_eq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_eq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_ge_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_ge_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_ge_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_ge_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_ge_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_ge_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_ge_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_ge_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_ge_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_ge_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_ge_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_ge_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_gt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_gt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_gt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_gt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_gt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_gt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_gt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_gt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_gt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_gt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_gt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_gt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_le_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_le_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_le_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_le_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_le_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_le_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_le_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_le_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_le_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_le_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_le_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_le_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lg_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lg_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lg_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lg_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lg_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lg_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lg_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lg_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lg_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lg_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lg_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lg_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_lt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmp_lt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmp_neq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_neq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_neq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_neq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_neq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_neq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_neq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_neq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_neq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_neq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_neq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_neq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nge_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nge_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nge_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nge_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nge_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nge_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nge_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nge_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nge_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nge_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nge_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nge_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_ngt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_ngt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_ngt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_ngt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_ngt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_ngt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nle_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nle_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nle_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nle_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nle_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nle_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nle_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nle_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nle_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nle_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nle_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nle_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlg_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlg_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlg_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlg_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlg_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlg_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_nlt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmp_o_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_o_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_o_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_o_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_o_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_o_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_o_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_o_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_o_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_o_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_o_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_o_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_u_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_u_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_u_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_u_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_u_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_u_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_u_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_u_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_u_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_u_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmp_u_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_u_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cmpx_eq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_eq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_eq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_eq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_eq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_eq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ge_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ge_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ge_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ge_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ge_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ge_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_gt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_gt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_gt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_gt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_gt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_gt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_le_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_le_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_le_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_le_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_le_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_le_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_le_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_le_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_le_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_le_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_le_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_le_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lg_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lg_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lg_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lg_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lg_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lg_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_lt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                               uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_cmpx_neq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_neq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_neq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_neq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_neq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_neq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nge_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nge_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nge_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nge_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nge_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nge_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ngt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ngt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ngt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ngt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ngt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_ngt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nle_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nle_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nle_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nle_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nle_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nle_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlg_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlg_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlg_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlg_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlg_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlg_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_nlt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);
int rj_arith_v_cmpx_o_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_o_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_o_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_o_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_o_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_o_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_o_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_o_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_o_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_o_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_o_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_o_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_u_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_u_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_u_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_u_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_u_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_u_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_u_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_u_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_u_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_u_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cmpx_u_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_u_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *excp_flag_user);

// Floating constant comparisons. F returns zero; T returns active lanes,
// after DPP16 row/bank/boundary filtering. CMPX returns the resulting EXEC mask
// in d, with no separate VCC write. Operand values do not affect the result;
// a/b may be NULL. F16 half selectors, ABS/NEG, CLAMP, guest IEEE mode and
// input/output flushing are accepted without changing the result. F64 rejects DPP.
// Result masks are exact in both semantics. Exact-mode exception reporting is
// not implemented: mode 4 returns RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS before
// modifying any output. Other modes leave FP status storage unchanged. The ISA still specifies NaN
// exceptions for these predicates; an exception-free result must not be inferred from F/T.
int rj_arith_v_cmp_f_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_f_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_f_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_f_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_f_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_f_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_t_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_t_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_t_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_t_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_t_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmp_t_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);
int rj_arith_v_cmpx_f_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_f_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_f_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_f_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_f_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_f_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_t_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_t_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_t_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_t_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_t_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);
int rj_arith_v_cmpx_t_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                              uint32_t *trapsts);

// Constant I16/U16 predicates ignore both input values and halfword selectors.
// F writes zero; T writes EXEC after DPP filtering. CMPX returns replacement EXEC.
// Output storage is required; input pointers may be null. Errors preserve outputs.
int rj_arith_v_cmp_f_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);

// RDNA3 constant integer comparisons. F writes zero; T writes participating
// EXEC after DPP filtering. CMP produces the condition mask; CMPX produces
// replacement EXEC in d. A/B values are ignored and their pointers may be null.
// Both semantics are exact. I32/U32 support DPP8/DPP16; I64/U64 reject DPP.
// All other instruction_flags are rejected. d is required even for zero EXEC.
int rj_arith_v_cmp_f_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_f_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_t_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_f_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_t_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);

// Integer comparisons write a scalar condition mask (CMP) or replacement EXEC
// mask (CMPX) to d. Inactive bits are zero, including for empty EXEC. d is always
// required and may alias any input word. Zero EXEC permits null VGPR pointers.
// A/B use one VGPR for 16/32 bits or a low/high pair for 64 bits. The 16-bit
// forms support HIGH_A/HIGH_B. The 16/32-bit forms support DPP8/DPP16 on A
// before half selection, with B in its original lane; filtered output bits are
// zero. The 64-bit forms reject DPP. Other instruction flags are rejected. Errors
// leave d unchanged. Supports loose and empirical-exact semantics. All host FP
// state is preserved; inputs are interpreted as two's-complement or unsigned.
int rj_arith_v_cmp_eq_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_eq_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ge_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_gt_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_le_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_lt_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_ne_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_eq_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ge_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_gt_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_le_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_lt_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_ne_u64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);

// Floating-point class-mask bits; any combination may be supplied in B.
constexpr uint32_t RJ_ARITH_CLASS_SNAN = 1U << 0;
constexpr uint32_t RJ_ARITH_CLASS_QNAN = 1U << 1;
constexpr uint32_t RJ_ARITH_CLASS_NEG_INF = 1U << 2;
constexpr uint32_t RJ_ARITH_CLASS_NEG_NORMAL = 1U << 3;
constexpr uint32_t RJ_ARITH_CLASS_NEG_SUBNORMAL = 1U << 4;
constexpr uint32_t RJ_ARITH_CLASS_NEG_ZERO = 1U << 5;
constexpr uint32_t RJ_ARITH_CLASS_POS_ZERO = 1U << 6;
constexpr uint32_t RJ_ARITH_CLASS_POS_SUBNORMAL = 1U << 7;
constexpr uint32_t RJ_ARITH_CLASS_POS_NORMAL = 1U << 8;
constexpr uint32_t RJ_ARITH_CLASS_POS_INF = 1U << 9;

// Classify A's raw encoding and test the corresponding bit of B's class mask.
// CMP writes a scalar condition mask to d; CMPX writes the replacement EXEC
// mask to d. Inactive bits are zero, including for zero EXEC. d must always be
// writable and may alias any input word; all inputs are read before that write.
// Zero EXEC permits null VGPR pointers. Errors leave d unchanged.
// A uses one VGPR for FP16/FP32 or a low/high pair for FP64; B uses one VGPR.
// Supports ABS_A/NEG_A. FP16 also supports HIGH_A and HIGH_B, selecting source
// halves. Only B's low ten selected bits matter. Signaling NaNs are not quieted.
// FP16/FP32 support DPP8/DPP16 on A before modifiers and half selection;
// B stays in its lane and filtered result bits are zero. FP64 rejects DPP.
// Supports loose and empirical-exact semantics, independently of denormal
// controls and guest IEEE mode. CLAMP is accepted and ignored.
// Host FP controls are never changed; arithmetic may raise sticky host flags.
// Unsuffixed APIs reject mode 4 pending qualification of CLASS FP status.
// This also applies to the gfx908/gfx90a/gfx1010/gfx1030/gfx1150 adapters;
// Compatibility retains classification masks. Gfx12 mode 4 succeeds without FP status:
// observed gfx1201 CLASS does not raise flags, including for signaling NaNs.
int rj_arith_v_cmp_class_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmp_class_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint64_t *d, const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cmpx_class_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b);

// Gfx1150 predicates retain the unsuffixed numeric mask, VGPR layout and
// instruction controls. DX10_CLAMP_OFF is additionally accepted. Compatibility
// accumulates INVALID only with that flag set: CLAMP signals on any NaN,
// otherwise only signaling NaNs contribute. No INPUT_DENORM is produced.
// Numeric-only modes leave status unused; mode 4 is unsupported. All input reads
// finish before mask/status stores. Constants need inputs only when reporting
// Compatibility status with DX10_CLAMP_OFF; empty EXEC needs no inputs.
// This preserves the existing emulator model, without new empirical qualification.
int rj_arith_v_cmp_f_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_f_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_lt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_lt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_eq_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_eq_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_le_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_le_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_gt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_gt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_lg_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_lg_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_ge_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_ge_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_o_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_o_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_u_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_u_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nge_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nge_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nle_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nle_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_neq_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_neq_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_t_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_t_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_f_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_f_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_le_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_le_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_o_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_o_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_u_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_u_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_t_f16_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_t_f16_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_f_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_f_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_lt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_lt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_eq_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_eq_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_le_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_le_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_gt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_gt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_lg_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_lg_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_ge_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_ge_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_o_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_o_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_u_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_u_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nge_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nge_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nle_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nle_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_neq_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_neq_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_t_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_t_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_f_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_f_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_le_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_le_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_o_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_o_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_u_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_u_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_t_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_t_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_f_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_f_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_lt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_lt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_eq_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_eq_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_le_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_le_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_gt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_gt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_lg_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_lg_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_ge_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_ge_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_o_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_o_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_u_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_u_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nge_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nge_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nle_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nle_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_neq_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_neq_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmp_t_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_t_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_f_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_f_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_le_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_le_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_o_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_o_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_u_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_u_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint64_t *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts);
int rj_arith_v_cmpx_t_f64_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_t_f64_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
// Gfx908/gfx942 floating predicates have the same numeric mask and operand layout as
// their unsuffixed counterparts, for Wave64 only. In Compatibility, CLAMP also
// accumulates INVALID in TRAPSTS for any participating NaN (quiet or signaling),
// including F/T predicates; no other FP causes are produced. Numeric-only modes
// leave TRAPSTS untouched. Mode 4 rejects before accessing operands or outputs.
// Numeric masks and NaN classification complete before output stores; D or
// TRAPSTS may overlap sources. With Compatibility+CLAMP, even F/T need A/B.
// Otherwise constants and empty EXEC do not read inputs. No new empirical
// qualification is implied by reusing the existing numerical model.
int rj_arith_v_cmp_f_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_lt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_eq_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_le_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_gt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_lg_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_ge_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_o_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_u_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_nge_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_nle_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_neq_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_t_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_f_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_le_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_o_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_u_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_t_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_f_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_lt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_eq_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_le_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_gt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_lg_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_ge_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_o_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_u_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_nge_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_nle_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_neq_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_t_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_f_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_le_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_o_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_u_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_t_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_f_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_lt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_eq_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_le_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_gt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_lg_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_ge_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_o_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_u_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_nge_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_nle_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_neq_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmp_t_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_f_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_le_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_o_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_u_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint64_t *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);
int rj_arith_v_cmpx_t_f64_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);

// Gfx10 FP32/FP64 predicates use the same numeric masks, operand layout and
// controls as the unsuffixed forms, for both wave sizes. Compatibility+CLAMP
// accumulates INVALID for participating quiet/signaling NaNs, including F/T.
// Constant F/T predicates also support strict FP-status semantics with this rule;
// other predicates reject that mode. Numeric-only modes leave TRAPSTS unused.
// Classification and numeric evaluation finish before output stores, allowing source/output
// overlap. Signaling constants need A/B; other constants and empty EXEC do not.
// This carries the existing emulator model, without new empirical qualification.
int rj_arith_v_cmp_f_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_f_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_lt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_lt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_eq_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_eq_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_le_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_le_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_gt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_gt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_lg_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_lg_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_ge_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_ge_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_o_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_o_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_u_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_u_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_nge_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nge_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nle_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nle_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_neq_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_neq_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_t_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_t_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmpx_f_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_f_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_le_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_le_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_o_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_o_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_u_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_u_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_t_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_t_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_f_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_f_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_lt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_lt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_eq_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_eq_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_le_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_le_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_gt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_gt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_lg_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_lg_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_ge_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_ge_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmp_o_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_o_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_u_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_u_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_nge_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nge_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nlg_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_ngt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nle_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nle_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_neq_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_neq_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_nlt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmp_t_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmp_t_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                   uint32_t *trapsts);
int rj_arith_v_cmpx_f_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_f_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_lt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_eq_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_le_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_le_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_gt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_lg_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_ge_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_cmpx_o_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_o_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_u_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_u_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nge_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nlg_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_ngt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nle_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_neq_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_nlt_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint64_t *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);
int rj_arith_v_cmpx_t_f64_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);
int rj_arith_v_cmpx_t_f64_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint64_t *d, const uint32_t *const *a, const uint32_t *const *b,
                                    uint32_t *trapsts);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
