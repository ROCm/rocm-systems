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

// Reads low 16-bit A/B/C and writes low 16-bit D, preserving its high half and inactive lanes.
// Accepted controls match gfx12 FP16 below. MODE.IEEE=0 retains source signaling NaNs;
// RJ_ARITH_FP_IEEE_MODE is rejected. FP status pointers are unused in modes 0..2.
// gfx11 supports modes 0..2; mode 3 rejects unsupported TRAPSTS reporting without writes.
int rj_arith_v_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

// gfx12 reads/writes low halfwords and preserves the high half and inactive lanes.
// Instruction flags must be zero. Only CPU-level and semantics fields (with supported
// values) and RJ_ARITH_FP16_OVFL are accepted; all other flag bits are rejected.
// RJ_ARITH_FP16_OVFL saturates finite FP16 overflow; infinities remain infinite.
// Mode 3 ORs active-lane FP exception causes into excp_flag_user, preserving existing bits.
int rj_arith_v_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// gfx11 FP32 has the same controls and status-mode rejection as gfx12 below.
// Under MODE.IEEE=0, NaN results retain the first source NaN without quieting it.
// RJ_ARITH_FP_IEEE_MODE is rejected.
int rj_arith_v_fma_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

// FP32 FMA preserves inactive lanes and supports whole-register aliases.
// Modes 0..2 are supported; mode 3 returns RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
// without changing D or FP status. The status pointer is unused in modes 0..2.
// Instruction flags must be zero. Only supported CPU/semantics fields and
// RJ_ARITH_FP16_OVFL are accepted; the latter has no effect on FP32 results.
int rj_arith_v_fma_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// Same gfx11 FP32 contract as Wave32, with 64-lane operands and EXEC.
int rj_arith_v_fma_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

// Same FP32 contract as Wave32, with 64-lane operands and 64-bit EXEC.
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
