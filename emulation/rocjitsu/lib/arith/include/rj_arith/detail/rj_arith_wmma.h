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

// Matrix factor signs and accumulator modifiers.
constexpr uint32_t RJ_ARITH_WMMA_NEG_LO_A = 1U << 0;
constexpr uint32_t RJ_ARITH_WMMA_NEG_LO_B = 1U << 1;
constexpr uint32_t RJ_ARITH_WMMA_NEG_C = 1U << 2;
constexpr uint32_t RJ_ARITH_WMMA_NEG_HI_A = 1U << 3;
constexpr uint32_t RJ_ARITH_WMMA_NEG_HI_B = 1U << 4;
constexpr uint32_t RJ_ARITH_WMMA_ABS_C = 1U << 5;

// gfx11 dense FP16 matrix multiply-accumulate. A/B each use eight packed VGPRs;
// C/D use eight FP32 VGPRs in Wave32 and four in Wave64. A[row][k] and B[k][col]
// map to [k / 2][row or col + 16 * group], half k % 2, replicated across every
// 16-lane group. Results are unspecified without this replication.
// C/D[row][col] maps to [row / 2][col + 16 * (row % 2)] in Wave32 and
// [row / 4][col + 16 * (row % 4)] in Wave64. Whole-VGPR aliases are supported.
// These gfx11 entry points support modes 0..3 without an FP-status output.
// Mixed-format FP16 input-denormal overrides are rejected.
// The six source/accumulator modifiers above are supported. Common guest denormal and
// IEEE, rounding and DX10 controls are accepted but ignored; compatibility resolves to mode 2.
// FP16_OVFL has no effect on these FP32 results.
int rj_arith_v_wmma_f32_16x16x16_f16_w32(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c);

// gfx12 Wave32 dense FP16 matrix multiply-accumulate: four A/B VGPRs and eight C/D.
// A[row][k] and B[k][col] map to [(k % 8) / 2][row or col + 16 * (k / 8)], half k % 2.
// C/D[row][col] maps to [row % 8][col + 16 * (row / 8)]. Whole-VGPR aliases are supported.
// The six source/accumulator modifiers above are accepted; other instruction bits are rejected.
// Mixed-format FP16 input-denormal overrides are rejected.
// Guest rounding, common denormal and IEEE controls are accepted but ignored by the fixed
// matrix model. DX10_CLAMP_OFF is rejected on gfx12. FP16_OVFL has no effect on FP32 results.
// Compatibility selects mode 2. There is no FP-status output: mode 3 has the same numeric contract
// as mode 2.
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

// BF16 factors use the same register layout and control/status contract as FP16 above.
int rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

// Wave64: two A/B VGPRs and four C/D VGPRs, each with 64 lane words.
// A[row][k] and B[k][col] map to [(k % 4) / 2][row or col + 16 * (k / 4)], half k % 2.
// Controls and status follow the gfx12 Wave32 contract above.
// C/D[row][col] maps to [row % 4][col + 16 * (row / 8) + 32 * ((row / 4) % 2)].
int rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
