// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_CONVERSION_H_
#define RJ_ARITH_CONVERSION_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Saturate both signed I16 halves of A to U8, pack them low byte first, and
// write the selected half of D, preserving the other half. HIGH_D selects the
// upper destination half. CLAMP is accepted and ignored. DPP8/DPP16 permutes A
// before saturation; no other instruction modifiers apply. A/D use one VGPR each. Exact result
// semantics; host FP controls are never changed.
int rj_arith_v_sat_pk_u8_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a);

// Pack selected FP16 bits from A into D's low half and B into its high half.
// HIGH_A/B select source halves; ABS_A/B clear their sign bits before NEG_A/B
// toggles them. Signaling NaNs are quieted on RDNA4, or on RDNA3 with IEEE set.
// Other payload bits are preserved; input-denormal flushing applies before packing.
// DPP8/DPP16 permutes A before half selection; B stays in its original lane.
// No other instruction modifiers apply. Each operand uses one VGPR. Exact
// result semantics are supported on RDNA4; RDNA3 currently supports loose semantics
// only. Host FP controls are never changed; arithmetic may raise sticky host flags. Output-denormal
// flags have no effect. RDNA3 APIs take trapsts; RDNA4 APIs take excp_flag_user. Mode 3 FP status
// reporting is unsupported on both.
int rj_arith_v_pack_b32_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pack_b32_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);

// Wave64 packing uses the same operand layouts and modifiers, with 64 lanes
// per VGPR. SAT supports exact bits; PACK has the same architecture-specific
// numerical and FP status contracts as Wave32.
int rj_arith_v_sat_pk_u8_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a);
int rj_arith_v_pack_b32_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pack_b32_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user);

// DPP8/DPP16 permute A before modifiers; B remains in its original lane,
// including when B is the stochastic seed.
// Narrow FP32 to OCP E4M3FN (FP8) / E5M2 (BF8). Each operand is one VGPR.
// PK rounds A/B to nearest-even into the low/high bytes of the destination half
// selected by HIGH_D, preserving the other half. Supports A/B ABS/NEG.
// SR converts A using B as the stochastic seed, replaces D's RJ_ARITH_CVT_BYTE_*
// byte and preserves the others. Supports A ABS/NEG. Subnormal SR alignment
// discards low significand bits before adding the seed's high 20/21 bits.
// RJ_ARITH_FP16_OVFL saturates finite overflow to signed max finite; infinities
// remain signed FP8 NaNs / BF8 infinities. Input NaNs become 0xff / 0xfe.
// Supports both semantics, full EXEC masking and every whole-register alias.
// Results do not depend on host rounding and preserve all host FP state.
int rj_arith_v_cvt_pk_fp8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_cvt_pk_bf8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_cvt_sr_fp8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_cvt_sr_bf8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// Wave64 narrowing uses the same exact rounding, seed alignment, selectors
// and preserved destination bytes, with 64 lanes per VGPR.
int rj_arith_v_cvt_pk_fp8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_cvt_pk_bf8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_cvt_sr_fp8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_cvt_sr_bf8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// DPP8/DPP16 permute A before nibble selection.
// Convert A's signed low nibble to FP32 divided by 16, then apply OMOD/CLAMP.
// Each operand holds one VGPR; higher source bits are ignored. Supports both
// semantics, full EXEC masking and whole-register aliasing. Results are exact
// under all host rounding modes, preserving host FP state. The form without _gfx12 selects
// RDNA3: OMOD is ignored when IEEE is set or output denormals are preserved.
// _gfx12 applies OMOD independently. Guest IEEE/input/output-denormal flags are
// accepted; there are no subnormal inputs/results, so flushing has no other effect.
int rj_arith_v_cvt_off_f32_i4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_off_f32_i4_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

// DPP8/DPP16 permute A; byte selection B and preserved bytes C stay in their
// original lanes. Convert FP32 A to an unsigned byte with guest FP32 rounding, saturation
// to [0,255] and NaN-to-zero. Insert it into byte (B & 3) of C and write D;
// each operand holds one VGPR. Supports A ABS/NEG; CLAMP is accepted without
// numeric effect. Loose semantics, full EXEC masking and all whole-register
// aliases are supported. Host rounding does not affect results; exceptions may
// change. Input-denormal flushing applies before rounding. RJ_ARITH_FP16_OVFL,
// guest IEEE and output-denormal flags have no numerical effect.
// RDNA3 APIs take trapsts; RDNA4 APIs take excp_flag_user.
// Mode 3 FP status reporting is unsupported on both architectures.
int rj_arith_v_cvt_pk_u8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);

int rj_arith_v_cvt_pk_u8_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

// DPP8/DPP16 permute A before half selection or saturation; B stays in its
// original lane. Integer widening and saturating packing use one VGPR per
// operand. Widening selects the low/high A half with HIGH_A, then sign- or zero-extends to D.
// Packing saturates each 32-bit source to the signed/unsigned 16-bit range,
// placing A in D's low half and B in its high half; no numeric modifiers
// are accepted.
// Both semantics are exact, with full EXEC masking and whole-register aliases.
// Never changes host FP controls. RJ_ARITH_FP16_OVFL has no effect.

int rj_arith_v_cvt_i32_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_u32_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_pk_i16_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_cvt_pk_u16_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// Wave64 integer conversions use the same layouts and modifiers, with 64
// lanes per VGPR and DPP row enables spanning the whole wave.
int rj_arith_v_cvt_i32_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_u32_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_pk_i16_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_cvt_pk_u16_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// DPP8/DPP16 permute A before modifiers and half selection; B stays in its
// original lane. Normalized conversions scale by 32767 (signed) or 65535 (unsigned), round
// once to nearest-even, and saturate to [-32767,32767] or [0,65535]. NaNs map
// to zero. All operands hold one VGPR. Packed forms put A/B in D's low/high
// halves; unary forms preserve the unselected D half. Floating sources support
// ABS/NEG; FP16 sources support HIGH_A/B, unary destinations support HIGH_D.
// CLAMP is accepted without numeric effect; unary forms also accept and ignore
// OMOD. RJ_ARITH_FP16_OVFL has no effect. Supports loose semantics, all EXEC masks
// and whole-register aliases. Host FP exception flags may change.

// Packed-half forms use RDNA3 TRAPSTS by default and RDNA4 EXCP_FLAG_USER with
// _gfx12. Other normalized forms use the same API without an architecture suffix on both
// architectures. Half sources honor input-denormal flushing. Guest IEEE and
// output-denormal flags have no numerical effect.
// Loose calls leave FP status untouched; exact reporting is unsupported.

int rj_arith_v_cvt_pk_norm_i16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_cvt_pk_norm_u16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

// Packed FP16 normalization honors RJ_ARITH_FP_FLUSH_INPUT_DENORMALS.
int rj_arith_v_cvt_pk_norm_i16_f16_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);

int rj_arith_v_cvt_pk_norm_i16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                             uint64_t instruction_flags, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             uint32_t *excp_flag_user);

int rj_arith_v_cvt_pk_norm_u16_f16_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);

int rj_arith_v_cvt_pk_norm_u16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                             uint64_t instruction_flags, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             uint32_t *excp_flag_user);

int rj_arith_v_cvt_norm_i16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_norm_u16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a);

// Wave64 normalized conversions use the same VGPR layouts, modifiers and
// loose semantics. Packed-half FP status reporting remains unsupported.
int rj_arith_v_cvt_pk_norm_i16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cvt_pk_norm_u16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);
int rj_arith_v_cvt_pk_norm_i16_f16_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);

int rj_arith_v_cvt_pk_norm_i16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                             uint64_t instruction_flags, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             uint32_t *excp_flag_user);

int rj_arith_v_cvt_pk_norm_u16_f16_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       uint32_t *trapsts);

int rj_arith_v_cvt_pk_norm_u16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                             uint64_t instruction_flags, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             uint32_t *excp_flag_user);

int rj_arith_v_cvt_norm_i16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a);
int rj_arith_v_cvt_norm_u16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a);

// DPP8/DPP16 permute A before conversion and modifiers; B stays in its
// original lane. Pack two FP32 source VGPRs into one 16-bit-pair destination: A goes to the
// low half, B to the high half. Supports ABS/NEG on A/B, full EXEC masking and
// whole-register aliases. Loose semantics only. CLAMP is accepted without
// numeric effect. The FP16 RTZ form also accepts and ignores OMOD, truncates
// toward zero, saturates finite overflow and quiets NaNs while retaining payload
// bits. Integer forms truncate and saturate to the destination range, with NaNs
// mapping to zero. RJ_ARITH_FP16_OVFL has no effect. Results do not depend on host
// rounding mode; host FP exception flags may change.

// Packed APIs without _gfx12 use RDNA3 TRAPSTS; _gfx12 APIs use EXCP_FLAG_USER.
// The FP16 RTZ form honors input and output-denormal flushing; integer forms
// accept those controls without numerical effect. Guest IEEE has no effect.
// Loose calls leave FP status untouched; exact reporting is unsupported.

int rj_arith_v_cvt_pk_rtz_f16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);

int rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user);

int rj_arith_v_cvt_pk_i16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_cvt_pk_i16_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

int rj_arith_v_cvt_pk_u16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_cvt_pk_u16_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

// Wave64 packed conversions use the same rounding, modifiers and one VGPR
// per operand. Loose semantics and optional-state restrictions are unchanged.
int rj_arith_v_cvt_pk_rtz_f16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts);

int rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            uint32_t *excp_flag_user);

int rj_arith_v_cvt_pk_i16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_cvt_pk_i16_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

int rj_arith_v_cvt_pk_u16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_cvt_pk_u16_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

// FP8/BF8 single-result conversion byte selector, an enumeration encoded in
// instruction_flags bits 16-17. Byte 0 is the least significant byte of A for
// widening, or of D for stochastic narrowing.
constexpr uint32_t RJ_ARITH_CVT_BYTE_0 = 0U << 16;
constexpr uint32_t RJ_ARITH_CVT_BYTE_1 = 1U << 16;
constexpr uint32_t RJ_ARITH_CVT_BYTE_2 = 2U << 16;
constexpr uint32_t RJ_ARITH_CVT_BYTE_3 = 3U << 16;

// Single-result forms support DPP8/DPP16, permuting A before byte selection.
// Packed two-result forms have no RDNA4 DPP encoding and reject DPP flags.
// FP8 (OCP E4M3FN) / BF8 (OCP E5M2) to FP32, exact in both semantics. A holds one
// VGPR. Single-result forms select one byte with RJ_ARITH_CVT_BYTE_* and write D[0].
// Packed forms select the low/high A half with RJ_ARITH_ALU_HIGH_A and write its
// two bytes to D[0]/D[1], respectively. If D halves alias, D[1] wins. Supports
// whole-register source/destination aliases and EXEC masking. No ABS/NEG,
// OMOD or CLAMP. Finite results are exact; NaNs become 0xffc00000. All host
// rounding modes give the same bits and preserve FP state.

int rj_arith_v_cvt_f32_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_pk_f32_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_pk_f32_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

// Wave64 FP8/BF8 widening has the same VGPR layout, selectors and NaN rules.
// DPP row enables cover all 64 lanes in the single-result forms.
int rj_arith_v_cvt_f32_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_pk_f32_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_pk_f32_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

// DPP8/DPP16 permute A before byte selection.
// Unsigned byte-to-FP32 conversions: one VGPR per operand. The mnemonic's
// byte index selects bits [8*index, 8*index+7] of A. Supports OMOD and CLAMP;
// source ABS/NEG and half selectors are invalid. All results are exactly
// representable, independent of host rounding, and preserve host FP state.
// Supports both semantics, full EXEC masking and whole-register A/D aliasing.
// The form without _gfx12 selects RDNA3: OMOD is ignored when IEEE is set or output denormals
// are preserved. _gfx12 applies it independently. Guest IEEE/input/output-denormal
// flags are accepted; flushing has no other effect on these finite normal results.

int rj_arith_v_cvt_f32_ubyte0_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte0_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte1_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte1_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte2_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte2_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte3_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte3_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

// Wave64 byte/nibble conversions use the same operand layouts and modifiers.
// Integer-to-float results support exact semantics at both widths. Byte packing
// retains loose semantics and rejects mode 3 FP status reporting.
int rj_arith_v_cvt_f32_ubyte0_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte0_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte1_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte1_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte2_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte2_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte3_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_ubyte3_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

int rj_arith_v_cvt_off_f32_i4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_off_f32_i4_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a);

int rj_arith_v_cvt_pk_u8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);

int rj_arith_v_cvt_pk_u8_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

// FP16 conversions, one VGPR per operand; loose semantics only. HIGH_A selects
// a 16-bit source half, HIGH_D selects a 16-bit destination half, preserving the
// other half. The corresponding selector is invalid for a full FP32 operand.
// Floating inputs support ABS/NEG; integer inputs reject them. Floating outputs
// support OMOD/CLAMP. FP16 results round before OMOD, using RJ_ARITH_FP_ROUND_MASK and
// FP16_OVFL at both rounding stages. Nonzero OMOD flushes initially tiny
// results and either initial zero to +0; newly tiny scaled results become
// signed zero. APIs without _gfx12 use RDNA3 TRAPSTS; _gfx12 APIs use EXCP_FLAG_USER.
// FP status reporting is unavailable; loose calls leave it untouched.
// RDNA3 ignores OMOD when IEEE is set or output denormals are preserved.
// RDNA4 applies OMOD independently. Guest IEEE/input/output-denormal flags are
// accepted. Input flushing precedes conversion and output flushing uses tininess
// before subnormal rounding. Widening F16 to F32 flushes half subnormals when input flush is set;
// it quiets NaNs on RDNA4, or on RDNA3 with IEEE set. RDNA3 IEEE=0 widening
// preserves signaling NaNs through OMOD. RDNA3 widening accepts
// DX10_CLAMP_OFF to preserve NaNs through CLAMP; otherwise CLAMP maps them
// to zero. Other conversions retain
// their existing denormal behavior; integer results ignore these guest flags.
// Other floating results quiet NaNs. Integer outputs
// truncate, saturate, map NaNs to zero, and ignore CLAMP/OMOD numerically.
// All six forms support DPP8/DPP16 on A before source-half selection.

int rj_arith_v_cvt_f16_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f16_i16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f16_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f16_u16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_i16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_i16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_u16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_u16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f16_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f32_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f32_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

// Wave64 FP16 conversions use the same half selectors, destination preservation,
// overflow behavior and loose semantics. Optional-state restrictions are unchanged.
int rj_arith_v_cvt_f16_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f16_i16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f16_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f16_u16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_i16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_i16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_u16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_u16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f16_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f32_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f32_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

// FP64 conversions: FP64 operands use two VGPRs, low word first; FP32 and
// integer operands use one. Loose semantics only. Integer inputs reject ABS/NEG;
// floating inputs support ABS/NEG. Floating outputs round to the destination
// format before OMOD and CLAMP. Host nearest-even rounding and enabled denormals
// are required. Integer outputs truncate, saturate overflow, map NaNs to zero,
// and accept CLAMP/OMOD without numeric effect; GPU exceptions are not modeled.
// APIs without _gfx12 select RDNA3 and use TRAPSTS where applicable; _gfx12 APIs
// use EXCP_FLAG_USER. Mode 3 FP status reporting is unsupported; loose calls
// leave FP status untouched. RDNA3 ignores OMOD when IEEE is set or output
// denormals are preserved. RDNA4 applies it independently. Guest IEEE and
// input/output-denormal and destination rounding flags are accepted. Narrowing
// quiets NaNs and retains leading payload bits; widening quiets on RDNA4 and
// on RDNA3 with IEEE set. Source denormals flush before conversion; destination
// tininess is evaluated before subnormal rounding.
// RJ_ARITH_FP_DX10_CLAMP_OFF preserves NaNs through CLAMP on RDNA3; RDNA4
// always clamps NaNs to zero. Destination halves may alias each other; the high word wins.

int rj_arith_v_cvt_f64_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f64_i32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f64_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f64_u32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_i32_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_i32_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_u32_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_u32_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f64_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f64_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f32_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f32_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

// Wave64 FP64 conversions use the same one/two-VGPR layouts, loose semantics
// and optional-state restrictions. DPP is unsupported at both widths.
int rj_arith_v_cvt_f64_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f64_i32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f64_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f64_u32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_i32_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_i32_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_u32_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_u32_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f64_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f64_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_f32_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_f32_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

// FP32/integer conversions, one VGPR per operand. Loose semantics only.
// Integer-to-FP32 conversion uses RJ_ARITH_FP_ROUND_MASK, followed by OMOD
// and CLAMP; source ABS/NEG are invalid. Float-to-integer conversion saturates
// to the destination range and supports ABS/NEG. Truncating forms map NaNs to
// zero; nearest/floor forms map them to the signed limit selected by the NaN sign
// after source modifiers. CLAMP is accepted but does not affect integer results.
// Truncating forms also accept OMOD without numeric scaling. GPU exceptions are
// not modeled. Mode 3 FP status reporting is unsupported on both architectures;
// loose calls leave FP status untouched. Truncating forms without _gfx12 use
// RDNA3 TRAPSTS, while _gfx12 forms use EXCP_FLAG_USER.
// Integer-to-FP32 forms without _gfx12 select RDNA3: OMOD is ignored with IEEE mode
// or preserved output denormals. _gfx12 forms apply it independently. Guest IEEE
// and input/output-denormal flags are accepted. Floating inputs are flushed
// before conversion when requested. Host nearest-even rounding
// and enabled denormals remain required for loose floating arithmetic.
// All six forms support DPP8/DPP16 on A.

int rj_arith_v_cvt_f32_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_i32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_u32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);

// Convert toward zero.
int rj_arith_v_cvt_i32_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_i32_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

int rj_arith_v_cvt_u32_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_u32_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

// Round to nearest integer, breaking ties toward positive infinity.
int rj_arith_v_cvt_nearest_i32_f32_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a);

// Round toward negative infinity.
int rj_arith_v_cvt_floor_i32_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);

// Wave64 FP32/integer conversions use the same layouts, rounding, modifiers
// and loose semantics. Optional-state restrictions remain unchanged.
int rj_arith_v_cvt_f32_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_i32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);
int rj_arith_v_cvt_f32_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cvt_f32_u32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);
int rj_arith_v_cvt_i32_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_i32_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);
int rj_arith_v_cvt_u32_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cvt_u32_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);
int rj_arith_v_cvt_nearest_i32_f32_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a);
int rj_arith_v_cvt_floor_i32_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
