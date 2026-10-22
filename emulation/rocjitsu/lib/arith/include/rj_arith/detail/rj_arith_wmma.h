// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_WMMA_H_
#define RJ_ARITH_WMMA_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Dense matrix multiply-accumulate on 32 lanes. A/B each use four packed-FP16
// VGPRs; C/D each use eight FP32 VGPRs. C/D[row][col] maps to
// [row % 8][col + 16 * (row / 8)]. See lib/arith/README.md for a packing example.
// Results are staged before writes, permitting whole-VGPR aliases.
// Instruction flags must be zero. Only CPU-level and semantics fields (with supported
// values) and RJ_ARITH_FP16_OVFL are accepted; all other flag bits are rejected.
// RJ_ARITH_FP16_OVFL is accepted but has no effect on the FP32 result.
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
