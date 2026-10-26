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

// gfx11 dense FP16 matrix multiply-accumulate. A/B each use eight packed VGPRs;
// C/D use eight FP32 VGPRs in Wave32 and four in Wave64. A[row][k] and B[k][col]
// map to [k / 2][row or col + 16 * group], half k % 2, replicated across every
// 16-lane group. Results are unspecified without this replication.
// C/D[row][col] maps to [row / 2][col + 16 * (row % 2)] in Wave32 and
// [row / 4][col + 16 * (row % 4)] in Wave64. Whole-VGPR aliases are supported.
// These gfx11 entry points support modes 0..3 without FP status. Instruction flags
// must be zero; only supported CPU/semantics fields and FP16_OVFL are accepted.
// FP16_OVFL has no effect on FP32 results.
int rj_arith_v_wmma_f32_16x16x16_f16_w32(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

// gfx11 Wave64 uses the layout, replication and controls documented above.
int rj_arith_v_wmma_f32_16x16x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

// BF16 uses the gfx11 Wave32 layout, replication and controls above.
int rj_arith_v_wmma_f32_16x16x16_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c);

// BF16 uses the gfx11 Wave64 layout, replication and controls above.
int rj_arith_v_wmma_f32_16x16x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c);

// gfx12 Wave32: A/B each use four packed VGPRs; C/D use eight FP32 VGPRs.
// A[row][k] and B[k][col] map to [(k % 8) / 2][row or col + 16 * (k / 8)],
// half k % 2. C/D[row][col] maps to [row % 8][col + 16 * (row / 8)].
// See lib/arith/README.md for a packing example. Whole-VGPR aliases are supported.
// Instruction flags must be zero. Only CPU-level and semantics fields (with supported
// values) and RJ_ARITH_FP16_OVFL are accepted; all other flag bits are rejected.
// RJ_ARITH_FP16_OVFL is accepted but has no effect on the FP32 result.
// This instruction has no FP-status output: mode 3 accepts the same numeric contract as mode 2.
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

// BF16 factors use the same register layout and control/status contract as FP16 above.
int rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

// Wave64 uses two A/B VGPRs and four C/D VGPRs, each with 64 lane words.
// C/D[row][col] maps to [row % 4][col + 16 * (row / 8) + 32 * ((row / 4) % 2)].
// A/B element (coordinate, k) maps to [(k % 4) / 2][coordinate + 16 * (k / 4)], half k % 2.
// Controls and FP-status behavior match Wave32 above.
int rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

// BF16 uses the same Wave64 layout and control/status contract as FP16.
int rj_arith_v_wmma_f32_16x16x16_bf16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
