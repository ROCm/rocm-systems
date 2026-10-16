// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_FP_H_
#define RJ_ARITH_FP_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shared FP16/FP32 FMA controls (all wave sizes).
// ABS/NEG, OMOD, CLAMP and DPP8/DPP16 are supported; FP16 also supports half selection.
// Guest rounding, input/output flushing and IEEE controls are accepted. These are translated
// API flags: FP16 uses MODE FP_ROUND[3:2]/FP_DENORM[3:2]; FP32 uses their [1:0] fields.
// gfx11 additionally accepts DX10_CLAMP_OFF; FP16_OVFL has no effect on FP32.
// Compatibility selects mode 2 for all FMA variants.
// gfx12 accepts IEEE_MODE without changing NaN quieting or OMOD gating.
// gfx11 disables OMOD when IEEE_MODE is set or output denormals are preserved.

// Fused multiply-add for selected lanes, preserving inactive lanes and
// unselected halfwords. OPSEL maps to RJ_ARITH_ALU_HIGH_A/B/C/D; zero selects the low half. FP
// status pointers are unused in modes 0..2. gfx11 supports modes 0..2; mode 3 rejects unsupported
// TRAPSTS reporting without writes.
int rj_arith_v_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

// Mode 3 ORs active-lane FP exception causes into excp_flag_user, preserving existing bits.
// Reporting requires nearest-even guest rounding and preserved denormals; other
// guest policies return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS without writes.
int rj_arith_v_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// gfx11 FP32 has the same controls and status-mode rejection as gfx12 below.
// FP16/FP32 NaN results retain the first source NaN. gfx11 preserves signaling NaNs
// with IEEE_MODE clear and quiets them when IEEE_MODE is set; gfx12 always quiets them.
int rj_arith_v_fma_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

// FP32 FMA preserves inactive lanes and supports whole-register aliases.
// Modes 0..2 are supported; mode 3 returns RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
// without changing D or FP status. The status pointer is unused in modes 0..2.
int rj_arith_v_fma_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// Same gfx11 FP32 contract as Wave32, with 64-lane operands and EXEC.
int rj_arith_v_fma_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_fma_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// Same gfx11 FP16 contract as Wave32, with 64-lane operands and EXEC.
int rj_arith_v_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

// Wave64 FP16 FMA has the same control and accumulating-status contract as Wave32.
int rj_arith_v_fma_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
