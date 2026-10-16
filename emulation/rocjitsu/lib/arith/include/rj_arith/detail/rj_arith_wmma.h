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

// Sparse metadata selector bits; valid bits depend on format and wave size.
// The high bit is used only by Wave64 forms with four index sets.
constexpr uint32_t RJ_ARITH_SWMMAC_INDEX_KEY_1 = 1U << 7;
constexpr uint32_t RJ_ARITH_SWMMAC_INDEX_KEY_2 = 1U << 8;

// Sparse integer 2:4 WMMA with signed 32-bit in/out accumulator D (8 VGPRs).
// A/B use 2/4 VGPRs for K=32 IU8 and K=64 IU4, or 1/2 for K=32 IU4;
// index uses one VGPR. Metadata pairs must select strictly increasing positions (unchecked;
// otherwise results are unspecified). SIGNED_A/B select signed factors. CLAMP saturates after each
// of two product groups: compressed positions 0..7 then 8..15 for K=32; positions 0..7 and 16..23,
// then 8..15 and 24..31 for K=64. Without CLAMP results wrap modulo 2^32. K=32 accepts
// RJ_ARITH_SWMMAC_INDEX_KEY_1; K=64 consumes all 32 metadata bits and rejects it. Supports loose
// and empirical exact semantics; host FP state is preserved. RJ_ARITH_FP16_OVFL has no effect. All
// inputs and D are read before ascending destination-register stores; last store wins aliases. EXEC
// is ignored; every lane is read and written, and operand pointers must be valid. Errors leave
// destinations unchanged.
int rj_arith_v_swmmac_i32_16x16x32_iu8_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_i32_16x16x32_iu4_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_i32_16x16x64_iu4_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

// Wave64 integer SWMMAC: A one VGPR, B two (one for K=32 IU4), D four,
// index one. IU8 selects four metadata bytes with KEY_1/KEY_2; IU4 selects
// two metadata halfwords with KEY_1. K=32 IU4 uses only A/index lanes 0..31.
// Saturation applies after the first and second halves of compressed K.
// Both semantics and the Wave32 alias/error guarantees are preserved.
int rj_arith_v_swmmac_i32_16x16x32_iu8_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_i32_16x16x32_iu4_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_i32_16x16x64_iu4_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

// Sparse FP8/BF8 2:4 matrix multiply-accumulate into FP32 D. A uses two
// VGPRs, B four, index one, and in/out D eight. FP8 is E4M3FN; BF8 is E5M2.
// Metadata pairs must contain strictly increasing positions in each group of
// four (unchecked; otherwise results are unspecified). RJ_ARITH_SWMMAC_INDEX_KEY_1 selects the
// upper 16 metadata bits per lane; no other instruction modifiers apply. Loose semantics use FP32
// FMA with host nearest-even rounding and denormals enabled. Exception flags may change. Exact
// semantics use two native DOT8 steps from the borrowed gfx1201 model, assuming the default host FP
// environment. RJ_ARITH_FP16_OVFL has no effect. All inputs and D are read before ascending
// destination-register stores; last store wins aliases. EXEC is ignored; every lane is read and
// written, and operand pointers must be valid. Errors leave destinations unchanged.
int rj_arith_v_swmmac_f32_16x16x32_fp8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);

int rj_arith_v_swmmac_f32_16x16x32_fp8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);

int rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);

int rj_arith_v_swmmac_f32_16x16x32_bf8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);

// Wave64 FP8/BF8 SWMMAC: A one VGPR, B two, D four, index one.
// KEY_1/KEY_2 combine to select one of four consecutive metadata bytes per
// lane. Four compressed elements use four 2-bit indices from that byte.
// Both semantics and alias/error guarantees match the Wave32 forms.
int rj_arith_v_swmmac_f32_16x16x32_fp8_fp8_w64(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);

int rj_arith_v_swmmac_f32_16x16x32_fp8_bf8_w64(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);

int rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w64(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);

int rj_arith_v_swmmac_f32_16x16x32_bf8_bf8_w64(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index);

// Sparse 2:4 matrix multiply-accumulate. D is both the initial accumulator and
// destination; A holds 16 compressed elements per row, B the dense 32x16 matrix,
// and index holds packed 2-bit positions. Within each four-element K group the
// two selected positions must be strictly increasing (unchecked; otherwise results
// are unspecified). A uses four VGPRs, B eight,
// index one, and D eight for FP32 or four for packed FP16/BF16 output.
// RJ_ARITH_SWMMAC_INDEX_KEY_1 selects the upper instead of lower 16 metadata bits in
// each lane. Supports RJ_ARITH_WMMA_NEG_LO_A/B and RJ_ARITH_WMMA_NEG_HI_A/B; these negate
// the first/second member of each selected pair, including B after selection.
// RJ_ARITH_WMMA_ABS_C then RJ_ARITH_WMMA_NEG_C modify the tied accumulator. CLAMP does
// not apply. Loose semantics accumulate selected products
// with FP32 FMA, narrowing packed outputs after each four products (FP16 RNE,
// BF16 truncation). Host FP state must
// provide nearest-even rounding with denormals enabled; exception flags may change.
// RJ_ARITH_FP16_OVFL saturates finite overflow when narrowing to FP16; otherwise it
// has no effect. Exact semantics use four borrowed gfx1201 DOT4 steps, narrowing
// after each step (FP16 RNE or BF16 truncation). All inputs and D are read
// before ascending destination-register stores; the last store wins aliases.
// EXEC is ignored; every lane is read and written, and operand pointers must
// be valid. Errors leave all destinations unchanged.
int rj_arith_v_swmmac_f32_16x16x32_f16_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_f32_16x16x32_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                            uint32_t *const *d, const uint32_t *const *a,
                                            const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_f16_16x16x32_f16_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_bf16_16x16x32_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b,
                                             const uint32_t *const *index);

// Wave64 FP16/BF16 SWMMAC: A two VGPRs, B four, index one. D holds
// four VGPRs for FP32 or two for packed FP16/BF16. KEY_1/KEY_2 select one
// of four metadata bytes per lane. Sign flags, both semantics,
// packed rounding, aliasing and errors follow the Wave32 contracts above.
int rj_arith_v_swmmac_f32_16x16x32_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_f32_16x16x32_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                            uint32_t *const *d, const uint32_t *const *a,
                                            const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_f16_16x16x32_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index);

int rj_arith_v_swmmac_bf16_16x16x32_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b,
                                             const uint32_t *const *index);

// Floating WMMA and SWMMAC accept but ignore guest rounding, input/output denormal flush,
// IEEE and DX10 controls in all semantics. Packed FP16 still honors RJ_ARITH_FP16_OVFL.
// All semantics assume the default host FP environment.
//
// WMMA modifier layout follows neg_lo[0:2], then neg_hi[0:2]. For C,
// neg_hi means absolute value, applied before neg_lo negation.
constexpr uint32_t RJ_ARITH_WMMA_NEG_LO_A = 1U << 0;
constexpr uint32_t RJ_ARITH_WMMA_NEG_LO_B = 1U << 1;
constexpr uint32_t RJ_ARITH_WMMA_NEG_C = 1U << 2;
constexpr uint32_t RJ_ARITH_WMMA_NEG_HI_A = 1U << 3;
constexpr uint32_t RJ_ARITH_WMMA_NEG_HI_B = 1U << 4;
constexpr uint32_t RJ_ARITH_WMMA_ABS_C = 1U << 5;

// RDNA3 packed WMMA OPSEL selects the same half of C and D. The other half
// of D is preserved. This is not a packed adjacent-row output as on RDNA4.
constexpr uint32_t RJ_ARITH_WMMA_HIGH_C_D = 1U << 8;

// Integer WMMA modifiers: NEG[0:1] select signed interpretation of A/B;
// CLAMP saturates signed accumulation at instruction-specific stage boundaries;
// without CLAMP, results wrap modulo 2^32.
constexpr uint32_t RJ_ARITH_WMMA_SIGNED_A = 1U << 0;
constexpr uint32_t RJ_ARITH_WMMA_SIGNED_B = 1U << 1;
constexpr uint32_t RJ_ARITH_WMMA_CLAMP = 1U << 6;

// gfx11 dense FP16 matrix multiply-accumulate. A/B each use eight packed VGPRs;
// C/D use eight FP32 VGPRs in Wave32 and four in Wave64. A[row][k] and B[k][col]
// map to [k / 2][row or col + 16 * group], half k % 2, replicated across every
// 16-lane group. Results are unspecified without this replication.
// C/D[row][col] maps to [row / 2][col + 16 * (row % 2)] in Wave32 and
// [row / 4][col + 16 * (row % 4)] in Wave64. Whole-VGPR aliases are supported.
// These gfx11 entry points support modes 0..3 without an FP-status output.
// The six source/accumulator modifiers above are supported. Guest denormal and
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

// RDNA3 packed-output and integer WMMA use the same interleaved C/D layout
// above: 8 VGPRs in Wave32, 4 in Wave64. Packed A/B use 8 VGPRs each;
// IU8 uses 4 each and IU4 uses 2 each. All lanes participate and whole VGPRs may alias.
// Packed forms support all six NEG/NEG_HI bits and RJ_ARITH_WMMA_HIGH_C_D, narrow
// after each DOT2 step, and honor RJ_ARITH_FP16_OVFL for finite FP16 overflow.
// Integer forms support SIGNED_A/B and CLAMP (signed saturation after each
// two IU8 or four IU4 products); without CLAMP they wrap modulo 2^32. Both semantics are
// supported. Packed and integer paths preserve the host FP environment.
int rj_arith_v_wmma_f16_16x16x16_f16_w32(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_bf16_16x16x16_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_bf16_16x16x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x16_iu8_w32(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x16_iu8_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x16_iu4_w32(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x16_iu4_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

// gfx12 Wave32 dense FP16 matrix multiply-accumulate: four A/B VGPRs and eight C/D.
// A[row][k] and B[k][col] map to [(k % 8) / 2][row or col + 16 * (k / 8)], half k % 2.
// C/D[row][col] maps to [row % 8][col + 16 * (row / 8)]. Whole-VGPR aliases are supported.
// The six source/accumulator modifiers above are accepted; other instruction bits are rejected.
// Guest rounding, denormal, IEEE and DX10 controls are accepted but ignored by the fixed
// matrix model. FP16_OVFL has no effect on FP32 results. Compatibility selects mode 2.
// There is no FP-status output: mode 3 has the same numeric contract as mode 2.
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

// Wave64: two A/B VGPRs and four C/D VGPRs, each with 64 lane words.
// A[row][k] and B[k][col] map to [(k % 4) / 2][row or col + 16 * (k / 4)], half k % 2.
// Controls and status follow the gfx12 Wave32 contract above.
// C/D[row][col] maps to [row % 4][col + 16 * (row / 8) + 32 * ((row / 4) % 2)].
// Both semantics and all WMMA modifiers are supported. Loose uses scalar or AVX2.
int rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

// Packed-output WMMA: A/B hold 4 VGPRs (wave32) or 2 (wave64);
// C/D hold 4 or 2 VGPRs respectively, with adjacent rows in low/high halves.
// Both semantics and all six floating WMMA modifiers are supported. Packed
// results narrow after each four-product step; RJ_ARITH_FP16_OVFL controls FP16 overflow.
int rj_arith_v_wmma_f16_16x16x16_f16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_bf16_16x16x16_bf16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x16_f16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_bf16_16x16x16_bf16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);

// Wave32 FP8/BF8 WMMA: A/B each hold 2 VGPRs, C/D each hold 8.
// FP8 is OCP E4M3FN; BF8 is OCP E5M2. Both semantics are implemented.
// Exact uses two native DOT8 steps from the borrowed gfx1201 empirical model.
// RJ_ARITH_WMMA_NEG_C and RJ_ARITH_WMMA_ABS_C are accepted but ignored in both semantics,
// matching captured raw encodings. A/B sign modifiers remain unsupported.
int rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

// Wave64 FP8/BF8 WMMA: A/B each hold one 64-lane VGPR, C/D four.
// K groups 0, 8, 4, 12 occupy successive groups of 16 source lanes.
// C/D rows 0..3, 8..11, 4..7, 12..15 occupy those lane groups.
// Modifiers and semantics match the Wave32 forms above.
int rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

// Wave32 integer WMMA: C/D each hold 8 VGPRs. A/B each hold 2 VGPRs for
// IU8 and K=32 IU4, or 1 VGPR for K=16 IU4. Supports loose and exact semantics,
// signed/unsigned factors, and CLAMP. Accumulators are signed 32-bit integers.
// CLAMP follows each source-lane group: first lanes 0..15, then 16..31,
// consuming every A/B VGPR in that group. This applies to both K=16 and K=32.
int rj_arith_v_wmma_i32_16x16x16_iu8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x16_iu4_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x32_iu4_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

// Wave64 integer WMMA: A/B each hold one VGPR, C/D four. IU4 K=16
// reads only source lanes 0..31; other shapes read all 64 source lanes.
// Source groups 0+2 accumulate before the first CLAMP, then groups 1+3
// before the second. C/D layout matches the Wave64 FP8 forms. Both semantics,
// signedness flags and whole-VGPR aliases are supported.
int rj_arith_v_wmma_i32_16x16x16_iu8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x16_iu4_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x32_iu4_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
