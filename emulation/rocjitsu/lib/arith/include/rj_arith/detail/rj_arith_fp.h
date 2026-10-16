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

// Fused multiply-add reads the low 16 bits of A/B/C and writes the low 16 bits of D,
// preserving D's high half and inactive lanes. FP status pointers are unused in modes 0..2.
// Instruction flags must be zero. Only CPU-level and semantics fields (with supported
// values) and RJ_ARITH_FP16_OVFL are accepted; all other flag bits are rejected.
// RJ_ARITH_FP16_OVFL saturates finite FP16 overflow; infinities remain infinite.
int rj_arith_v_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
