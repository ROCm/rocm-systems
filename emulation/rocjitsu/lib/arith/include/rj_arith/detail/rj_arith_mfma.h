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

// FP64 MFMA: A/B use two VGPRs each, C/D eight (16x16) or two (4x4 batches).
// controls bits 0/1/2 negate A/B/C (the FP64 BLGP interpretation). All lanes
// participate regardless of EXEC; overlap is supported. Guest FP controls are
// ignored, Compatibility is approximate, and strict exact modes are unsupported.
int rj_arith_v_mfma_f64_16x16x4_f64_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
int rj_arith_v_mfma_f64_4x4x4_4b_f64_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

// Signed-byte MFMA: A/B use M*K*batch/256 VGPRs, C/D use M*M*batch/64.
// Accumulation wraps at 32 bits by default. On gfx942 and gfx950 RJ_ARITH_FP16_OVFL
// saturates the final signed sum. CBSZ/ABID/BLGP select the MFMA source lanes.
// All semantics share exact integer arithmetic.
// All 64 lanes participate, EXEC is ignored, and operands may overlap D.
int rj_arith_v_mfma_i32_32x32x4_2b_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x4_4b_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c);
int rj_arith_v_mfma_i32_4x4x4_16b_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);
int rj_arith_v_mfma_i32_32x32x16_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x32_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x64_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
int rj_arith_v_mfma_i32_32x32x32_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);

// FP32 factors use M*K*batch/64 VGPRs each; outputs use M*M*batch/64.
// For gfx950, set RJ_ARITH_FP16_OVFL to saturate finite FP32 overflow at the final result.
// Controls, semantics and overlap follow the dense MFMA contract below.
int rj_arith_v_mfma_f32_32x32x1_2b_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x1_4b_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c);

int rj_arith_v_mfma_f32_4x4x1_16b_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x2_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x4_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);

// Compute D = A*B+C with FP32 accumulation on all 64 lanes, ignoring EXEC.
// Shapes with a batch suffix compute that many independent matrices. A and B use
// M*K*batch/128 VGPRs each, C and D use M*M*batch/64; each pointer addresses 64 words.
// CBSZ 0..4 broadcasts A blocks; reserved 5..7 retain identity, and ABID wraps
// within the chosen block count. BLGP 0..7 selects the legacy B lane permutation.
// For gfx950 BF16 instructions, set RJ_ARITH_FP16_OVFL to saturate finite FP32
// overflow. Infinities from source operands are preserved.
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

// gfx942 XF32 approximation: A/B two FP32 VGPRs, C/D four for 16x16 or sixteen
// for 32x32. Both inputs are truncated to ten fraction bits before the FP32 reduction.
// Loose, allow-Loose and Compatibility are supported.
// Source controls, whole-wave participation, aliasing and error rules above apply.
int rj_arith_v_mfma_f32_16x16x8_xf32_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x4_xf32_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

// FP8/BF8 MFMA: A/B two packed VGPRs, C/D four for 16x16 or sixteen for 32x32.
// Unsuffixed forms use gfx942 FNUZ (bias 8/16, sole NaN 0x80); gfx950 forms use
// OCP E4M3FN/E5M2 (bias 7/15). The suffix distinguishes different arithmetic
// behind the same ISA mnemonic. Only Loose/allow-Loose/Compatibility are supported.
// Source controls, whole-wave participation, aliasing and error rules above apply.
int rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
