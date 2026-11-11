// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense Wave64 MFMA on gfx942/gfx950. Inputs and results use the GFX9 matrix layout.
// IWYU pragma: private, include "rj_arith/rj_arith.h"
#ifndef RJ_ARITH_MFMA_H_
#define RJ_ARITH_MFMA_H_

#include "rj_arith_common.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Encoded A broadcast block size/source and B lane permutation, matching CBSZ/ABID/BLGP.
constexpr uint64_t RJ_ARITH_MFMA_CBSZ_MASK = 7ULL;
constexpr uint64_t RJ_ARITH_MFMA_ABID_SHIFT = 3ULL;
constexpr uint64_t RJ_ARITH_MFMA_BLGP_SHIFT = 7ULL;

// Compute D = A*B+C with FP32 accumulation on all 64 lanes, ignoring EXEC.
// Shapes with a batch suffix compute that many independent matrices. A and B use
// M*K*batch/128 VGPRs each, C and D use M*M*batch/64; each pointer addresses 64 words.
// CBSZ 0..4 broadcasts A blocks; reserved 5..7 retain identity, and ABID wraps
// within the chosen block count. BLGP 0..7 selects the legacy B lane permutation.
// Guest FP controls are accepted but ignored, as in the existing MFMA executor.
// Loose, allow-Loose and Compatibility use approximate arithmetic; strict exact
// modes return UNSUPPORTED_SEMANTICS. All inputs are consumed before D is stored
// in register order. Aliasing is supported; errors leave D unchanged.
int rj_arith_v_mfma_f32_32x32x4_2b_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x4_4b_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_4x4x4_16b_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x8_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x32_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                            uint32_t *const *d, const uint32_t *const *a,
                                            const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                            uint32_t *const *d, const uint32_t *const *a,
                                            const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x8_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x32_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
