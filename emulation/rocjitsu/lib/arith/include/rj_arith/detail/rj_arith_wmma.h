// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_WMMA_H_
#define RJ_ARITH_WMMA_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Sparse integer 2:4 WMMA with signed 32-bit in/out accumulator D (8 VGPRs).
// A/B use 2/4 VGPRs for K=32 IU8 and K=64 IU4, or 1/2 for K=32 IU4;
// index uses one VGPR. Metadata pairs must select strictly increasing positions.
// SIGNED_A/B select signed factors. CLAMP saturates after each of two product
// groups: compressed positions 0..7 then 8..15 for K=32; positions 0..7 and
// 16..23, then 8..15 and 24..31 for K=64. Without CLAMP results wrap modulo 2^32.
// K=32 accepts RJ_ARITH_SWMMAC_INDEX_KEY_1; K=64 consumes all 32 metadata bits and
// rejects it. Supports loose and empirical exact semantics; host FP state is
// preserved. RJ_ARITH_FP16_OVFL has no effect. All inputs and D are read before
// ascending destination-register stores; last store wins aliases. EXEC is ignored;
// every lane is read and written, and operand pointers must be valid.
// Errors leave destinations unchanged.
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
// four. RJ_ARITH_SWMMAC_INDEX_KEY_1 selects the upper 16 metadata bits per lane;
// no other instruction modifiers apply. Loose semantics use FP32 FMA with
// host nearest-even rounding and denormals enabled. Exception flags may change.
// Exact semantics use two native DOT8 steps from the borrowed gfx1201 model,
// independently of host FP state. RJ_ARITH_FP16_OVFL has no effect. All inputs and
// D are read before ascending destination-register stores; last store wins
// aliases. EXEC is ignored; every lane is read and written, and operand pointers
// must be valid. Errors leave destinations unchanged.
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
// two selected positions must be strictly increasing. A uses four VGPRs, B eight,
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
// of four metadata bytes per lane. Sign flags, loose-only arithmetic,
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

// Floating WMMA and SWMMAC accept but ignore guest input/output denormal flush
// and IEEE flags in both semantics. Packed FP16 still honors RJ_ARITH_FP16_OVFL.
// All semantics assume the default host FP environment.
//
// WMMA modifier layout follows neg_lo[0:2], then neg_hi[0:2]. For C,
// neg_hi means absolute value, applied before neg_lo negation.
static const uint32_t RJ_ARITH_WMMA_NEG_LO_A = 1U << 0;
static const uint32_t RJ_ARITH_WMMA_NEG_LO_B = 1U << 1;
static const uint32_t RJ_ARITH_WMMA_NEG_C = 1U << 2;
static const uint32_t RJ_ARITH_WMMA_NEG_HI_A = 1U << 3;
static const uint32_t RJ_ARITH_WMMA_NEG_HI_B = 1U << 4;
static const uint32_t RJ_ARITH_WMMA_ABS_C = 1U << 5;

// Sparse WMMA selector low bit: selects the upper 16-bit set in Wave32,
// or advances by one 8-bit set in Wave64 forms with four index sets.
static const uint32_t RJ_ARITH_SWMMAC_INDEX_KEY_1 = 1U << 7;

// High selector bit for Wave64 sparse forms with four index sets.
static const uint32_t RJ_ARITH_SWMMAC_INDEX_KEY_2 = 1U << 8;

// RDNA3 packed WMMA OPSEL selects the same half of C and D. The other half
// of D is preserved. This is not a packed adjacent-row output as on RDNA4.
static const uint32_t RJ_ARITH_WMMA_HIGH_C_D = 1U << 8;

// RDNA3 FP16/BF16 WMMA with FP32 accumulators. A/B each hold 8 VGPRs,
// with the full 16-element K vector in every lane. Replicate A/B across the
// two (Wave32) or four (Wave64) groups of 16 lanes. C/D hold 8 or 4 VGPRs:
// row = reg * (wave_size / 16) + lane / 16, column = lane % 16.
// EXEC is ignored and all operands must be valid. Whole-VGPR aliasing is allowed.
// Supports loose and empirical exact semantics and all six NEG/NEG_HI modifiers.
// Exact semantics use integer arithmetic and preserve the host FP environment.
// Loose FP32 arithmetic requires host nearest-even rounding and enabled denormals.
// RJ_ARITH_FP16_OVFL has no effect on FP32 output.
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

// RDNA4 Wave32 16x16x16 WMMA: A/B each contain 4 VGPRs of packed 16-bit
// elements; C/D each contain 8 VGPRs of FP32 elements. WMMA and SWMMAC ignore
// architectural EXEC, including zero, and write every destination lane.
// Both loose and empirical exact semantics are supported for these WMMA forms.
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b, const uint32_t *const *c);

// Wave64 variants: 64 words per VGPR; 2 VGPRs for A/B, 4 for C/D.
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
int rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

// Wave64 FP8/BF8 WMMA: A/B each hold one 64-lane VGPR, C/D four.
// K groups 0, 8, 4, 12 occupy successive groups of 16 source lanes.
// C/D rows 0..3, 8..11, 4..7, 12..15 occupy those lane groups.
// Modifiers and loose-only semantics match the Wave32 forms above.
int rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

// Integer WMMA modifiers: NEG[0:1] select signed interpretation of A/B;
// CLAMP saturates signed accumulation at instruction-specific stage boundaries;
// without CLAMP, results wrap modulo 2^32.
static const uint32_t RJ_ARITH_WMMA_SIGNED_A = 1U << 0;
static const uint32_t RJ_ARITH_WMMA_SIGNED_B = 1U << 1;
static const uint32_t RJ_ARITH_WMMA_CLAMP = 1U << 6;

// Wave32 integer WMMA: C/D each hold 8 VGPRs. A/B each hold 2 VGPRs for
// IU8 and K=32 IU4, or 1 VGPR for K=16 IU4. Supports loose and exact semantics,
// signed/unsigned factors, and CLAMP. Accumulators are signed 32-bit integers.
// CLAMP applies after products with (k / 8) even, then after those with
// (k / 8) odd: K=16 uses 0..7 then 8..15; K=32 uses 0..7 plus 16..23,
// then 8..15 plus 24..31.
int rj_arith_v_wmma_i32_16x16x16_iu8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x16_iu4_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_wmma_i32_16x16x32_iu4_w32(uint64_t flags, uint64_t instruction_flags,
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

int rj_arith_v_wmma_i32_16x16x32_iu4_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
