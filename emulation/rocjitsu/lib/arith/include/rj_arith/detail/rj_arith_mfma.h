// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense and sparse Wave64 matrix arithmetic with the GFX9 register layout.
// IWYU pragma: private, include "rj_arith/rj_arith.h"
#ifndef RJ_ARITH_MFMA_H_
#define RJ_ARITH_MFMA_H_

#include "rj_arith_common.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// Sparse signed-byte MFMA: A/B/D use 4/8/16 VGPRs, index one Wave64 VGPR.
// CBSZ/ABID select metadata; wide forms ignore them. Accumulation wraps by default.
// All semantics are exact. RJ_ARITH_FP16_OVFL saturates signed accumulation after
// each 8-group (K=32) hardware pass; other FP controls and EXEC are ignored.
// D is the tied accumulator; D may overlap A, B or index. Errors leave all operands unchanged.
int rj_arith_v_smfmac_i32_32x32x64_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *index);

// Sparse signed-byte MFMA: A/B/D use 4/8/4 VGPRs, index one Wave64 VGPR.
// CBSZ/ABID select metadata; wide forms ignore them. Accumulation wraps by default.
// All semantics are exact. RJ_ARITH_FP16_OVFL saturates signed accumulation after
// each 8-group (K=32) hardware pass; other FP controls and EXEC are ignored.
// D is the tied accumulator; D may overlap A, B or index. Errors leave all operands unchanged.
int rj_arith_v_smfmac_i32_16x16x128_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index);

// Sparse signed-byte MFMA: A/B/D use 2/4/16 VGPRs, index one Wave64 VGPR.
// CBSZ/ABID select metadata; wide forms ignore them. Accumulation wraps by default.
// All semantics are exact. Under RJ_ARITH_FP16_OVFL, the unsuffixed entry point
// saturates the final sum while _gfx950 saturates after each 8-group (K=32) pass;
// other FP controls and EXEC are ignored.
// D is the tied accumulator; D may overlap A, B or index. Errors leave all operands unchanged.
int rj_arith_v_smfmac_i32_32x32x32_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *index);
int rj_arith_v_smfmac_i32_32x32x32_i8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *index);

// Sparse signed-byte MFMA: A/B/D use 2/4/4 VGPRs, index one Wave64 VGPR.
// CBSZ/ABID select metadata; wide forms ignore them. Accumulation wraps by default.
// All semantics are exact. Under RJ_ARITH_FP16_OVFL, the unsuffixed entry point
// saturates the final sum while _gfx950 saturates after each 8-group (K=32) pass;
// other FP controls and EXEC are ignored.
// D is the tied accumulator; D may overlap A, B or index. Errors leave all operands unchanged.
int rj_arith_v_smfmac_i32_16x16x64_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *index);
int rj_arith_v_smfmac_i32_16x16x64_i8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *index);

// Encoded A broadcast block size/source and B lane permutation, matching CBSZ/ABID/BLGP.
constexpr uint64_t RJ_ARITH_MFMA_CBSZ_MASK = 7ULL;
constexpr uint64_t RJ_ARITH_MFMA_ABID_SHIFT = 3ULL;
constexpr uint64_t RJ_ARITH_MFMA_BLGP_SHIFT = 7ULL;

// gfx908/gfx90a signed-byte MFMA: A/B use M*K*batch/256 VGPRs each,
// C/D use M*M*batch/64. CBSZ/ABID/BLGP controls follow the encoded fields above.
// gfx908 broadcasts across Wave64; gfx90a selects adjacent M-lane block groups.
// All semantics use wrapping integer arithmetic; FP controls (including FP16_OVFL)
// and EXEC are ignored. D may overlap any source; errors leave storage unchanged.
int rj_arith_v_mfma_i32_32x32x4_2b_i8_w64_gfx908(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x4_4b_i8_w64_gfx908(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
int rj_arith_v_mfma_i32_4x4x4_16b_i8_w64_gfx908(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_i32_32x32x8_i8_w64_gfx908(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x16_i8_w64_gfx908(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_i32_32x32x4_2b_i8_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x4_4b_i8_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
int rj_arith_v_mfma_i32_4x4x4_16b_i8_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_i32_32x32x8_i8_w64_gfx90a(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x16_i8_w64_gfx90a(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

// Older BF16 and gfx90a FP16 MFMA: A/B use M*K*batch/128 VGPRs; C/D use M*M*batch/64.
// CBSZ/ABID/BLGP fields are packed as above. gfx908 broadcasts across Wave64
// and preserves denormals. gfx90a selects adjacent M-lane groups and always
// flushes input-format A/B denormals, FP32 C denormals and final FP32 denormals.
// Other FP controls and EXEC are ignored. Loose, allow-Loose and Compatibility
// are supported; the inherited models are not empirically qualified. D may
// overlap any source. Errors leave storage unchanged.

int rj_arith_v_mfma_f32_32x32x2_2b_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x2_4b_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
int rj_arith_v_mfma_f32_4x4x2_16b_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x4_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);
int rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x2_2b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x2_4b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
int rj_arith_v_mfma_f32_4x4x2_16b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x4_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
int rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);

// Older FP32 MFMA: A/B use M*K*batch/64 VGPRs; C/D use M*M*batch/64.
// CBSZ/ABID/BLGP fields are packed as above. gfx908 broadcasts across Wave64;
// gfx90a selects adjacent M-lane groups and honors input/output denormal-flush
// flags on A/B/C and the final result. gfx908 preserves denormals. Other FP
// controls (including overflow saturation) and EXEC are ignored. Only Loose,
// allow-Loose and Compatibility are supported. D may overlap any source;
// errors leave storage unchanged. These are inherited, unqualified FP models.

int rj_arith_v_mfma_f32_32x32x1_2b_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x1_4b_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);
int rj_arith_v_mfma_f32_4x4x1_16b_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x2_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x1_2b_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x1_4b_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);
int rj_arith_v_mfma_f32_4x4x1_16b_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
int rj_arith_v_mfma_f32_32x32x2_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

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

// FP64 MFMA: A/B use two VGPRs each, C/D eight (16x16) or two (4x4 batches).
// At zero controls this also implements gfx90a. Bits 0/1/2 negate A/B/C
// on gfx942/gfx950; gfx90a has no such sign controls. All lanes
// participate regardless of EXEC; overlap is supported. Guest FP controls are
// ignored, Compatibility is approximate, and strict exact modes are unsupported.
int rj_arith_v_mfma_f64_16x16x4_f64_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
int rj_arith_v_mfma_f64_4x4x4_4b_f64_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

// Signed-byte MFMA: A/B use M*K*batch/256 VGPRs, C/D use M*M*batch/64.
// Accumulation wraps at 32 bits by default. Unsuffixed shared forms preserve the
// gfx942 rules: CBSZ/ABID/BLGP permute source lanes on all shapes and
// RJ_ARITH_FP16_OVFL saturates the final sum. On _gfx950 and the gfx950-only
// 16x16x64/32x32x32 forms, single-batch shapes ignore CBSZ/ABID/BLGP and
// RJ_ARITH_FP16_OVFL saturates signed accumulation after each K<=16 hardware pass.
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
int rj_arith_v_mfma_i32_32x32x16_i8_w64_gfx950(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x32_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x32_i8_w64_gfx950(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mfma_i32_16x16x64_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
int rj_arith_v_mfma_i32_32x32x32_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);

// FP32 factors use M*K*batch/64 VGPRs each; outputs use M*M*batch/64.
// For gfx950, set RJ_ARITH_FP16_OVFL to saturate finite FP32 overflow at the final result.
// Controls, semantics and overlap follow the dense MFMA contract below.

// Compute D = A*B+C with FP32 accumulation on all 64 lanes, ignoring EXEC.
// Shapes with a batch suffix compute that many independent matrices. A and B use
// M*K*batch/128 VGPRs each, C and D use M*M*batch/64; each pointer addresses 64 words.
// CBSZ 0..4 broadcasts A blocks; reserved 5..7 retain identity, and ABID wraps
// within the chosen block count. BLGP 0..7 selects the legacy B lane permutation.
// For gfx950 BF16 instructions, set RJ_ARITH_FP16_OVFL to saturate finite FP32
// overflow. Infinities from source operands are preserved.
// gfx942 approximate arithmetic ignores FP16_OVFL, preserving its legacy policy.
// Compatibility and Loose retain approximate arithmetic. On gfx942, the five
// original FP16/BF16 shapes and both XF32 shapes with zero CBSZ/ABID/BLGP
// support exact modes for both FP16_OVFL=0 and FP16_OVFL=1 (which flushes NaN
// to +0.0 and clamps +-Inf to +-MAX_FLOAT after each internal K-step);
// exact-allow-Loose falls back when source controls are non-zero.
// The wider gfx950-only forms and the _gfx950 variants remain approximate;
// strict requests return UNSUPPORTED_SEMANTICS. All inputs are consumed before D is stored
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

// gfx942 XF32 MFMA: A/B two FP32 VGPRs, C/D four for 16x16 or sixteen
// for 32x32. Both inputs are truncated to ten fraction bits before the FP32 reduction,
// turning NaNs whose payload lies only in bits 12:0 into infinities.
// Exact-mode qualification follows the single-block gfx942 contract above.
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
// behind the same ISA mnemonic. gfx942 exact-mode qualification follows the
// half-input contract above; gfx950 strict modes remain unsupported.
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
// Compatibility and Loose retain fast approximate accumulation. Empirical exact
// modes are MI350X-calibrated across all 25 format pairs: RJ_ARITH_FP16_OVFL=0
// emits canonical 0xffc00000 on NaN and preserves +-Inf, whereas only
// RJ_ARITH_FP16_OVFL=1 flushes NaN to +0.0 and clamps +-Inf to +-MAX_FLOAT after
// each 16- or 32-product hardware pass. Other FP modes are ignored and FP status
// is untouched.
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
// the E8M0 exponent for each lane. Scale lane M*block+row/column scales the
// corresponding 32-element K block, and scale 0xff produces NaN. Inline scale
// sourcing is the caller's responsibility. Compatibility and Loose retain fast
// approximate accumulation (where RJ_ARITH_FP16_OVFL saturates finite-input
// overflow). Empirical exact modes are MI350X-calibrated across all 25 format
// pairs and scale byte selectors when C modifiers are zero, following the same
// RJ_ARITH_FP16_OVFL=0 vs RJ_ARITH_FP16_OVFL=1 per-pass NaN/infinity rule above;
// non-zero NEG_C/ABS_C controls fall back in allow-Loose and reject in strict
// exact modes.
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
// Compute tied D += sparse(A)*B in Wave64, ignoring EXEC. A uses M*K*bits/4096
// VGPRs, B twice that count, D M*M/64, and index one VGPR; each has 64 words.
// Controls encode CBSZ/ABID as above. Compact forms with CBSZ[1:0]=0 select
// an index byte by ABID[1:0] for F16/BF16, or halfword by ABID[0] for FP8/BF8.
// Wide F16/BF16 forms likewise select a halfword by ABID[0]. Otherwise the
// first set is used; wide FP8/BF8 forms consume all 32 bits and ignore ABID.
// Compatibility preserves the legacy fused reduction and NaN priority.
// The 12 compact gfx942 forms (unsuffixed 16x16x32/32x32x16 F16/BF16 and
// 16x16x64/32x32x32 FNUZ FP8/BF8) support empirical bit-exact modes for both
// RJ_ARITH_FP16_OVFL=0 and 1. Wide and *_gfx950 forms remain loose-only.
// FP status is untouched.
// FP8 unsuffixed compact forms use gfx942 FNUZ; gfx950 forms use OCP encodings.
// RJ_ARITH_FP16_OVFL saturates finite-input overflow on gfx950, and applies
// MI300X per-step no-Inf/NaN clamping in gfx942 exact modes while approximate
// gfx942 paths continue to ignore it.
int rj_arith_v_smfmac_f32_16x16x32_f16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x32_f16_w64_gfx950(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x32_bf16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x32_bf16_w64_gfx950(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x16_f16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x16_f16_w64_gfx950(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x16_bf16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x16_bf16_w64_gfx950(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_f16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_bf16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_f16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_bf16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x128_fp8_fp8_w64(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x128_fp8_bf8_w64(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x128_bf8_fp8_w64(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_smfmac_f32_16x16x128_bf8_bf8_w64(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x64_fp8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x64_fp8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x64_bf8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);
int rj_arith_v_smfmac_f32_32x32x64_bf8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);

// gfx950 variants of the shared half/XF32 mnemonics retain approximate arithmetic.
// They do not inherit gfx942's empirical qualification.
int rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

int rj_arith_v_mfma_f32_16x16x8_xf32_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x4_xf32_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

int rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);

// gfx908 retains approximate FP16 arithmetic without gfx942 exact qualification.
// These forms preserve denormals, broadcast A across Wave64 and ignore FP16_OVFL.
int rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

// gfx908 retains approximate FP16 arithmetic without gfx942 exact qualification.
// These forms preserve denormals, broadcast A across Wave64 and ignore FP16_OVFL.
int rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

// gfx908 retains approximate FP16 arithmetic without gfx942 exact qualification.
// These forms preserve denormals, broadcast A across Wave64 and ignore FP16_OVFL.
int rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);

// gfx908 retains approximate FP16 arithmetic without gfx942 exact qualification.
// These forms preserve denormals, broadcast A across Wave64 and ignore FP16_OVFL.
int rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

// gfx908 retains approximate FP16 arithmetic without gfx942 exact qualification.
// These forms preserve denormals, broadcast A across Wave64 and ignore FP16_OVFL.
int rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
