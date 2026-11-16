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
// gfx1250 sparse IU8 WMMA: A/B/D use 8/16/8 VGPRs; index uses two Wave32 VGPRs.
// SIGNED_A/B select signedness; CLAMP saturates once after dot+D.
// All semantics are exact; FP flags and EXEC are ignored. D is the tied accumulator;
// D may overlap A, B or index. Errors leave all operands unchanged.
int rj_arith_v_swmmac_i32_16x16x128_iu8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index);

// gfx1250 byte WMMA: A/B/C/D each use eight Wave32 VGPRs. SIGNED_A/B select
// signed factors; CLAMP saturates once after dot+C, otherwise addition wraps.
// All semantics share exact integer arithmetic; guest FP flags are ignored.
// All lanes participate regardless of EXEC; inputs may overlap D.
int rj_arith_v_wmma_i32_16x16x64_iu8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

// gfx1250 FP32 16x16x4 WMMA on all 32 lanes, ignoring EXEC. A/B each use two
// VGPRs, C/D eight; NEG_C/ABS_C modify C, while unused source modifiers are ignored.
// Guest FP controls are ignored. Loose, allow-Loose and Compatibility are approximate;
// strict modes return UNSUPPORTED_SEMANTICS. Inputs may overlap D.
int rj_arith_v_wmma_f32_16x16x4_f32_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);

// gfx1250 Wave32 16x16x32 half-input WMMA with FP32 C/D: eight VGPRs per operand.
// Supports Loose, allow-Loose and Compatibility; strict exact modes are unsupported.
// BF16 adds C after reducing the dot; FP16 starts the reduction at C. ABS_C then
// NEG_C modify C; A/B NEG bits are accepted but ignored, as in the legacy executor.
// Guest FP controls are ignored. EXEC is irrelevant: all 32 lanes participate.
// Inputs are read before ascending D stores, permitting overlap. Errors leave D unchanged.
int rj_arith_v_wmma_f32_16x16x32_f16_w32(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x32_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c);

// gfx1250 packed-result forms: A/B eight VGPRs, D four. Same-type C uses four
// packed VGPRs; BF16F32 C uses eight FP32 VGPRs. Narrow only after FP32 accumulation:
// Loose rounds to nearest-even; Compatibility retains BF16 truncation. Loose FP16/BF16
// saturate overflow and signed infinities to signed max finite when FP16_OVFL is
// set, including infinite C inputs; Compatibility ignores that bit.
// Same-type forms start at C; Loose
// applies ABS_C then NEG_C, while Compatibility ignores them. BF16F32 reduces
// the dot before C, honors ABS_C/NEG_C and
// retains the legacy deterministic NaN priority. Semantics/alias/error rules above apply.
// For packed C forms, INLINE_C broadcasts c[0][0]: a low-half F16/BF16
// encoding in Loose/allow-Loose, or the inherited FP32 encoding in Compatibility.
constexpr uint32_t RJ_ARITH_WMMA_INLINE_C = 1U << 10;
int rj_arith_v_wmma_f16_16x16x32_f16_w32(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_bf16_16x16x32_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_bf16f32_16x16x32_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

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
// and empirical exact semantics; host FP state is preserved. Guest rounding, denormal,
// IEEE and DX10 controls are rejected; RJ_ARITH_FP16_OVFL is ignored. All
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
// and IEEE controls in all semantics. DX10_CLAMP_OFF is accepted and ignored only on gfx11;
// gfx12 rejects it. Packed FP16 still honors RJ_ARITH_FP16_OVFL.
// The mixed-format FP16 input-denormal override flags are rejected.
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

// RDNA3 packed-output and integer WMMA use the same interleaved C/D layout
// above: 8 VGPRs in Wave32, 4 in Wave64. Packed A/B use 8 VGPRs each;
// IU8 uses 4 each and IU4 uses 2 each. All lanes participate and whole VGPRs may alias.
// Packed forms support all six NEG/NEG_HI bits and RJ_ARITH_WMMA_HIGH_C_D, narrow
// after each DOT2 step, and honor RJ_ARITH_FP16_OVFL for finite FP16 overflow.
// Integer forms support SIGNED_A/B and CLAMP (signed saturation after each
// two IU8 or four IU4 products); without CLAMP they wrap modulo 2^32. Both semantics are
// supported. Integer forms reject guest rounding, denormal, IEEE and DX10 controls;
// FP16_OVFL is accepted and ignored. Packed and integer paths preserve the host FP environment.
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
// Mixed-format FP16 input-denormal overrides are rejected.
// Guest rounding, common denormal and IEEE controls are accepted but ignored by the fixed
// matrix model. DX10_CLAMP_OFF is rejected on gfx12. FP16_OVFL has no effect on FP32 results.
// Compatibility selects mode 2. There is no FP-status output: mode 3 has the same numeric contract
// as mode 2.
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
// Guest rounding, denormal, IEEE and DX10 controls are rejected; FP16_OVFL is ignored.
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

// gfx1250 OCP FP8/BF8 Wave32 WMMA: A/B each consume K/8 VGPRs.
// FP32 C/D consume eight VGPRs; packed FP16 C/D consume four. No EXEC or FP-status output.
// Supports Loose, allow-Loose and Compatibility; strict exact modes are unsupported.
// FP32 C accepts ABS then NEG. Packed C in Loose/allow-Loose also applies ABS then NEG;
// INLINE_C uses the low-half F16 bits of c[0][0]. Compatibility ignores packed C
// modifiers and uses the inherited full FP32 inline word instead.
// FP16_OVFL saturates finite overflow and infinite results.
// Guest rounding, denormal, IEEE and DX10 controls are ignored. Operand VGPRs may alias.
int rj_arith_v_wmma_f32_16x16x64_fp8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x64_fp8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x64_fp8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x64_fp8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x64_bf8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x64_bf8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x64_bf8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x64_bf8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x128_fp8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x128_fp8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x128_fp8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x128_fp8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x128_bf8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x128_bf8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x128_bf8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f16_16x16x128_bf8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

// gfx1250 mixed WMMA controls: formats use bits 0..2 / 3..5 (RJ_ARITH_MATRIX_FORMAT_*).
constexpr uint64_t RJ_ARITH_WMMA_MIXED_NEG_C = 1ULL << 6;
constexpr uint64_t RJ_ARITH_WMMA_MIXED_ABS_C = 1ULL << 7;

// Compute D=A*B+C across Wave32, ignoring EXEC. Each pointer addresses 32 words.
// 16-row A/B consume 16/12/8 VGPRs for 8/6/4-bit formats; C/D consume 8.
// The 32-row FP4 form consumes A=16, B=8, C/D=16 VGPRs and accepts only C modifiers.
// ABS_C precedes NEG_C. All inputs precede stores; aliases are supported.
// Loose/allow-Loose/Compatibility retain scalar unfused and SIMD fused arithmetic;
// strict empirical modes are unsupported. FP modes are ignored; errors leave D unchanged.
int rj_arith_v_wmma_f32_16x16x128_f8f6f4_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c);
int rj_arith_v_wmma_f32_32x16x128_f4_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

// Scale formats are enums: E8M0=0, unsigned E5M3=1, OCP E4M3=2; 3 is reserved.
constexpr uint64_t RJ_ARITH_WMMA_SCALE_FORMAT_A_SHIFT = 8ULL;
constexpr uint64_t RJ_ARITH_WMMA_SCALE_FORMAT_B_SHIFT = 10ULL;
constexpr uint64_t RJ_ARITH_WMMA_SCALE_ROW_A = 1ULL << 12;
constexpr uint64_t RJ_ARITH_WMMA_SCALE_ROW_B = 1ULL << 13;

// Scaled gfx1250 WMMA shares the preceding formats, C modifiers, policies and sizes.
// A/B use contiguous K groups per lane (16 byte values or 32 subbyte values).
// Scale operands contain one Wave32 VGPR (SCALE) or two (SCALE16), four bytes per word.
// Byte b scales the contiguous 32- or 16-product block b. ROW_A/B select the upper
// halfwave; 32-row FP4 A instead uses all32lanes, ignoring ROW_A. Each fused block
// is multiplied by scale A then scale B before addition to C. Scale storage may alias D.
int rj_arith_v_wmma_scale_f32_16x16x128_f8f6f4_w32(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c,
                                                   const uint32_t *const *scale_a,
                                                   const uint32_t *const *scale_b);
int rj_arith_v_wmma_scale16_f32_16x16x128_f8f6f4_w32(uint64_t flags, uint64_t controls,
                                                     uint32_t *const *d, const uint32_t *const *a,
                                                     const uint32_t *const *b,
                                                     const uint32_t *const *c,
                                                     const uint32_t *const *scale_a,
                                                     const uint32_t *const *scale_b);
int rj_arith_v_wmma_scale_f32_32x16x128_f4_w32(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c,
                                               const uint32_t *const *scale_a,
                                               const uint32_t *const *scale_b);
int rj_arith_v_wmma_scale16_f32_32x16x128_f4_w32(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b, const uint32_t *const *c,
                                                 const uint32_t *const *scale_a,
                                                 const uint32_t *const *scale_b);
// INDEX_KEY_1 follows LLVM's metadata >> 16 folding: shift each metadata word
// logically by 16 before decoding positions. Other instruction controls are unchanged.
// gfx1250 sparse D += A*B, Wave32, ignoring EXEC. A uses eight VGPRs, B sixteen;
// index uses one VGPR for half inputs and two for FP8. Each VGPR has 32 lanes.
// F32 D occupies eight VGPRs; packed D four. BF16F32 requires eight D pointers,
// reads all eight as FP32 and writes only the first four as BF16. The only control
// is RJ_ARITH_SWMMAC_INDEX_KEY_1, described above.
// Compatibility preserves ordered FMA and NaN priority. Packed outputs round once
// to nearest-even and honor FP16_OVFL, including infinite inputs. Other FP modes
// are ignored; strict empirical semantics are unsupported. Aliases are supported.
int rj_arith_v_swmmac_f32_16x16x64_f16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index);
int rj_arith_v_swmmac_f32_16x16x64_bf16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index);
int rj_arith_v_swmmac_f16_16x16x64_f16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index);
int rj_arith_v_swmmac_bf16_16x16x64_bf16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *index);
int rj_arith_v_swmmac_bf16f32_16x16x64_bf16_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_swmmac_f32_16x16x128_fp8_fp8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_swmmac_f16_16x16x128_fp8_fp8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_swmmac_f32_16x16x128_fp8_bf8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_swmmac_f16_16x16x128_fp8_bf8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_swmmac_f32_16x16x128_bf8_fp8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_swmmac_f16_16x16x128_bf8_fp8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_swmmac_f32_16x16x128_bf8_bf8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);
int rj_arith_v_swmmac_f16_16x16x128_bf8_bf8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
