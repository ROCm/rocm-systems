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
// Accumulation wraps at 32 bits. CBSZ/ABID/BLGP use the floating MFMA lane controls.
// All semantics share exact integer arithmetic; guest FP flags are ignored.
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

// gfx942 XF32 approximation: A/B two FP32 VGPRs, C/D four for 16x16 or sixteen
// for 32x32. Like the legacy emulator, this uses full FP32 input precision rather
// than claiming the GPU's XF32 rounding model. Loose/allow-Loose/Compatibility only.
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

// Independent mixed matrix formats; A occupies controls bits 0..2 and B bits 3..5.
constexpr uint64_t RJ_ARITH_MATRIX_FORMAT_FP8 = 0ULL;
constexpr uint64_t RJ_ARITH_MATRIX_FORMAT_BF8 = 1ULL;
constexpr uint64_t RJ_ARITH_MATRIX_FORMAT_FP6 = 2ULL;
constexpr uint64_t RJ_ARITH_MATRIX_FORMAT_BF6 = 3ULL;
constexpr uint64_t RJ_ARITH_MATRIX_FORMAT_FP4 = 4ULL;
constexpr uint64_t RJ_ARITH_MATRIX_FORMAT_B_SHIFT = 3ULL;

// gfx950 D=A*B+C, Wave64, ignoring EXEC. A/B independently use 8,6,4 VGPRs
// for 8,6,4-bit formats; each pointer addresses 64 words. C/D use M*M/64 VGPRs.
// Reserved formats/controls reject before operand access. Aliases are supported.
// Compatibility retains legacy scalar unfused / SIMD fused accumulation; strict
// empirical modes are unsupported. FP modes are ignored and FP status is untouched.
int rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x64_f8f6f4_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *c);

// Scale byte selectors in controls, plus absolute-value-before-negate on C.
constexpr uint64_t RJ_ARITH_MFMA_SCALE_A_BYTE_SHIFT = 6ULL;
constexpr uint64_t RJ_ARITH_MFMA_SCALE_B_BYTE_SHIFT = 8ULL;
constexpr uint64_t RJ_ARITH_MFMA_SCALE_NEG_C = 1ULL << 10;
constexpr uint64_t RJ_ARITH_MFMA_SCALE_ABS_C = 1ULL << 11;

// gfx950 block-scaled MFMA, with the same format/layout and overlap contract above.
// scale_a/scale_b each address 64 packed words (one VGPR); byte selectors choose
// the E8M0 exponent for each lane. Scale lane M*block+row/column applies to one
// fused 32-product dot, followed by combined-exponent scaling and addition to C.
// Scale 0xff produces NaN. Inline scale sourcing is the caller's responsibility.
// Loose and Compatibility supported; strict empirical modes remain unsupported.
int rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c,
                                                   const uint32_t *scale_a,
                                                   const uint32_t *scale_b);
int rj_arith_v_mfma_scale_f32_32x32x64_f8f6f4_w64(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c, const uint32_t *scale_a,
                                                  const uint32_t *scale_b);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
