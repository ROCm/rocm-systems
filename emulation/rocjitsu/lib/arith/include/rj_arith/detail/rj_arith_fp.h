// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_FP_H_
#define RJ_ARITH_FP_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shared ordinary FP16/FP32 FMA controls (all wave sizes).
// ABS/NEG, OMOD, CLAMP and DPP8/DPP16 are supported; FP16 also supports half selection.
// The common input/output denormal controls are accepted; mixed-format FP16
// input-denormal overrides are rejected.
// Guest rounding, input/output flushing and IEEE controls are accepted. These are translated
// API flags: FP16 uses MODE FP_ROUND[3:2]/FP_DENORM[3:2]; FP32 uses their [1:0] fields.
// gfx11 additionally accepts DX10_CLAMP_OFF. FP16 accepts FP16_OVFL: finite overflow
// saturates to the signed maximum finite value before and after OMOD; true infinities
// are preserved. FP16_OVFL has no effect on FP32.
// Compatibility selects mode 2 for FP16/FP32 FMA variants; FP64 uses Loose.
// gfx12 accepts IEEE_MODE without changing NaN quieting or OMOD gating.
// gfx11 disables OMOD when IEEE_MODE is set or output denormals are preserved.

// Wave64 forms retain their Wave32 contracts with 64 lanes and a 64-bit EXEC mask.
// Required FP status accumulates across active lanes; loose semantics leaves it untouched.

// FP16 fused multiply-add: selected halves of A/B/C/D, all ALU source/output
// modifiers and RJ_ARITH_FP16_OVFL. Preserves the other D half and inactive lanes.
// Arithmetic rounds to FP16 before OMOD; active OMOD flushes tiny arithmetic
// results to +0 and newly tiny scaled results to signed zero. CLAMP is last.
// Exact empirical semantics follow rocjitsu's guest-rounding model,
// including NaN payload priority.
// FMA/FMAC/FMAMK/FMAAK without _gfx12 model RDNA3; _gfx12 variants model RDNA4.
// Both accept guest rounding, input/output-denormal flushing and IEEE mode. RDNA3 gates
// OMOD on flushed outputs and IEEE being clear, and its exact path preserves
// selected signaling NaNs when IEEE is clear. RDNA4 always quiets them and
// applies OMOD independently of these guest settings. Literal forms have no OMOD.
// RDNA3 also accepts RJ_ARITH_FP_DX10_CLAMP_OFF; loose NaN preservation retains SIMD.
// RDNA3 reporting, and RDNA4 reporting with guest flushing or directed rounding, currently
// return UNSUPPORTED_FP_STATUS for mode 3 requests. Opting out keeps
// the adapted rocjitsu exact result model available. Loose calls skip reporting.
// All semantics require the default host FP environment. Host controls are never
// explicitly changed; ordinary arithmetic may raise sticky host flags.
// OPSEL maps to RJ_ARITH_ALU_HIGH_A/B/C/D; zero selects the low half.
// FP flags are translated from MODE: FP16 uses FP_ROUND[3:2]/FP_DENORM[3:2].
// Mode 3 ORs active-lane FP exception causes into excp_flag_user, preserving existing bits.
// OMOD suppresses UNDERFLOW/INEXACT, retains INVALID/INPUT_DENORM and reports
// overflow of the scaled result. CLAMP suppresses all new causes.
// Reporting requires nearest-even guest rounding and preserved denormals; other
// guest policies return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS without writes.
int rj_arith_v_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);
int rj_arith_v_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);
int rj_arith_v_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);
int rj_arith_v_fma_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// FP32 FMA/FMAC support RJ_ARITH_ALU source/output modifiers and guest input/output
// denormal flushing. APIs without _gfx12 model RDNA3: OMOD is ignored if output
// denormals are preserved or RJ_ARITH_FP_IEEE_MODE is set. The _gfx12 variants apply
// OMOD independently of these guest settings. FMAC_DX9_ZERO is RDNA3-only.
// DX9-zero forms always flush input
// and output denormals, while retaining the architecture's OMOD gating rule.
// FMA/FMAC and DX9 forms support exact RJ_ARITH_FP_ROUND_MASK semantics, including NaN payload
// priority, DPP and all source/output modifiers. RDNA3 accepts
// RJ_ARITH_FP_DX10_CLAMP_OFF to preserve arithmetic NaNs through CLAMP in exact and loose paths.
// RDNA4 rejects that flag. Mode 3 FP status reporting is unsupported on both architectures. Loose
// calls skip FP status output. All semantics require the default host FP environment. DX9 forms
// have no FP status output. Guest settings never explicitly change host FP controls; ordinary
// arithmetic may raise sticky host flags.
// Exact and Compatibility FP16/FP32 follow guest NaN priority and quieting rules.
// gfx11 preserves signaling NaNs with IEEE_MODE clear and quiets them when set;
// gfx12 always quiets them. Loose FP16/FP32 may use native NaN handling, differing
// in quieting, payload selection, and invalid-product precedence across CPU paths.
// Loose paths may ignore guest rounding as permitted by the common semantics contract.
// FP32 flags use MODE FP_ROUND[1:0]/FP_DENORM[1:0].
int rj_arith_v_fma_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);
int rj_arith_v_fma_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);
int rj_arith_v_fma_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

// Same gfx12 FP32 contract as Wave32, with 64-lane operands and EXEC.
int rj_arith_v_fma_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// DX9 FMA: one VGPR per operand, all ALU source/output modifiers; exact guest rounding.
// Flush source/result subnormals. If either modified factor is signed zero,
// add a positive-zero product to modified C before output scaling/CLAMP;
// otherwise compute a fused multiply-add. DPP8/DPP16 permute A before source
// modifiers; B and C retain their lanes.
int rj_arith_v_fma_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_fma_dx9_zero_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c);
int rj_arith_v_fma_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_fma_dx9_zero_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c);

// RDNA3 DX9 fused multiply-accumulate into D, with the same flushing and
// zero-product rules as DX9 FMA. Supports A/B ABS/NEG, OMOD and CLAMP.
// DPP, C modifiers and half selectors are invalid. Supports loose and exact guest rounding.
int rj_arith_v_fmac_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b);
int rj_arith_v_fmac_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b);

// FP32 fused multiply-accumulate into D. Supports A/B ABS/NEG, OMOD and CLAMP.
// C modifiers and half selectors are invalid. Exact guest rounding and DPP on A
// follow the FP32 FMA contract. Mode 3 FP status reporting is unsupported.
int rj_arith_v_fmac_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *trapsts);
int rj_arith_v_fmac_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_fmac_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *trapsts);
int rj_arith_v_fmac_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

// Literal FP32 FMA: FMAMK computes A * literal + B; FMAAK computes A * B +
// literal. The literal is a raw FP32 encoding. No instruction flags are valid.
// gfx11 accepts and ignores DX10_CLAMP_OFF; gfx12 rejects it.
// Forms without _gfx12 are RDNA3; _gfx12 forms are RDNA4. Guest input/output
// denormal flushing, rounding and IEEE mode are accepted. Loose semantics permit approximate
// NaN payload/quieting and require host nearest-even rounding and denormals enabled.
// Exact semantics honor guest rounding. Mode 3 FP status reporting is unsupported
// on both architectures.
// Loose calls skip FP-status updates.
int rj_arith_v_fmamk_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                             const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_fmamk_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_fmamk_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                             const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_fmamk_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_fmaak_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t literal, uint32_t *trapsts);
int rj_arith_v_fmaak_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t literal,
                                   uint32_t *excp_flag_user);
int rj_arith_v_fmaak_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t literal, uint32_t *trapsts);
int rj_arith_v_fmaak_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t literal,
                                   uint32_t *excp_flag_user);

// FP16 fused multiply-accumulate into D. Supports A/B ABS/NEG, OMOD, CLAMP,
// HIGH_A/B/D and RJ_ARITH_FP16_OVFL, with the same rounding and exact-semantics
// contract as FP16 FMA. HIGH_D selects both the accumulator and result half;
// preserves the other half. C modifiers (including HIGH_C) are invalid.
int rj_arith_v_fmac_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *trapsts);
int rj_arith_v_fmac_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_fmac_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *trapsts);
int rj_arith_v_fmac_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

// Literal FP16 FMA: FMAMK computes A * literal + B; FMAAK computes A * B +
// literal. The literal is a raw FP16 encoding. HIGH_A/B/D select the two VGPR
// inputs and destination half; the other D half is preserved. Other instruction
// flags are invalid. Supports RJ_ARITH_FP16_OVFL and empirical exact semantics with
// the FP16 FMA rounding and host-environment contract.
int rj_arith_v_fmamk_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                             const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_fmamk_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_fmamk_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                             const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_fmamk_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_fmaak_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint16_t literal, uint32_t *trapsts);
int rj_arith_v_fmaak_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint16_t literal,
                                   uint32_t *excp_flag_user);
int rj_arith_v_fmaak_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint16_t literal, uint32_t *trapsts);
int rj_arith_v_fmaak_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint16_t literal,
                                   uint32_t *excp_flag_user);

// Packed FP16 FMA (RDNA3 without _gfx12; RDNA4 with _gfx12): two independent fused results per
// lane. All RJ_ARITH_PK_* flags below are supported. With no flags, corresponding input halves are
// multiplied and added. Each result rounds to FP16; CLAMP applies last. Supports RJ_ARITH_FP16_OVFL
// and empirical exact semantics with guest rounding, input/output flushing and IEEE mode. Directed
// rounding uses the scalar integer-intermediate path. RDNA3 supports RJ_ARITH_FP_DX10_CLAMP_OFF;
// that mode also uses the scalar path to preserve NaNs under CLAMP. RDNA3 preserves selected
// signaling NaNs with IEEE mode off; RDNA4 quiets them. Both halves use the original inputs,
// including when D aliases A/B/C. RDNA3 FP status reporting is unsupported;
// RDNA4 reporting requires nearest-even rounding and preserved denormals.
int rj_arith_v_pk_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);
int rj_arith_v_pk_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user);
int rj_arith_v_pk_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);
int rj_arith_v_pk_fma_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user);

// Packed FP16 FMAC: component-wise A * B + D. Its e32 encoding has no
// modifiers or selectors; instruction_flags must be zero.
// Supports RJ_ARITH_FP16_OVFL and the same loose/exact semantics as packed FMA.
int rj_arith_v_pk_fmac_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_pk_fmac_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);
int rj_arith_v_pk_fmac_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_pk_fmac_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);

// Packed FP16 binary arithmetic: supports RJ_ARITH_PK_* negation and half selectors
// for A/B, plus RJ_ARITH_PK_CLAMP and RJ_ARITH_FP16_OVFL. Flags for C are invalid. Each
// result rounds to FP16; the two results use original inputs even with aliases.
// Packed ADD/MUL do not support DPP.
// Loose semantics require host nearest-even with denormals enabled.
// ADD/MUL accept guest IEEE and input/output denormal flags. Flushing applies
// to each selected input half and each rounded output half. In the RDNA3 scalar
// guest-mode path, IEEE controls NaN quieting. Directed guest rounding uses
// scalar FP64 arithmetic and integer narrowing. RDNA3 RJ_ARITH_FP_DX10_CLAMP_OFF uses
// that scalar path to preserve NaNs under CLAMP. Loose and Compatibility are
// supported; mode 1 may fall back to Loose. Explicit exact modes are unsupported.
// Compatibility preserves selected gfx11 input NaN bits with IEEE clear in both
// scalar and SIMD paths; IEEE set quiets them. Loose permits approximate NaN results.
// ADD/MUL without _gfx12 take RDNA3 trapsts; RDNA4 variants
// take excp_flag_user. Both reject mode 3 FP status reporting and skip loose state.
int rj_arith_v_pk_add_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_add_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_add_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_add_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_mul_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_mul_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_mul_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_mul_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

// Binary FP32 arithmetic: one VGPR per operand. APIs without _gfx12 model RDNA3:
// OMOD is ignored unless output flushing is enabled and RJ_ARITH_FP_IEEE_MODE is clear.
// The _gfx12 APIs apply OMOD regardless of these flags. Both architectures support
// RJ_ARITH_FP_FLUSH_INPUT_DENORMALS and RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS. Active OMOD flushes
// tiny results to +0; halving a normal result below twice minimum normal gives signed zero.
// Supports ABS/NEG for A/B, OMOD, CLAMP, and DPP8/DPP16. Flags for C and half selection are
// invalid. ADD/MUL support exact results through rocjitsu's FMA model with RJ_ARITH_FP_ROUND_MASK
// guest rounding. SUB/SUBREV negate the corresponding source while preserving original
// operand order for NaN priority, using the same architecture-specific FMA model.
// MUL_DX9_ZERO also supports exact multiplication with its zero override.
// Loose binary paths accept but ignore
// guest arithmetic rounding flags, retaining SIMD dispatch. OMOD scaling honors
// guest rounding for overflow in both exact and loose paths.
// gfx12 MUL supports FP status for nearest-even rounding, preserved denormals
// and no OMOD/CLAMP: INVALID, INPUT_DENORM, OVERFLOW, UNDERFLOW and INEXACT.
// Exact subnormal products report INPUT_DENORM without UNDERFLOW. Other guest
// controls and RDNA3 reporting return UNSUPPORTED_FP_STATUS. Compatibility uses
// numeric-only exact semantics for MUL in every configuration. Rocjitsu retains
// its existing MUL status classifier when integrating Compatibility mode.
// Other binary operations do not support mode 3 FP status reporting.
int rj_arith_v_add_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_add_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_sub_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_sub_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_subrev_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_subrev_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_mul_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

// Binary FP16 ADD/SUB/SUBREV/MUL: one VGPR per operand, with independently
// selected source and destination halves. Preserves the other destination half
// and inactive lanes; D may alias a whole source VGPR. Supports source ABS/NEG,
// OMOD, CLAMP, RJ_ARITH_FP16_OVFL, and guest input/output flushing.
// APIs without _gfx12 are RDNA3; _gfx12 variants are RDNA4. RDNA3 ignores OMOD
// when output denormals are preserved or IEEE is set; RDNA4 applies it regardless.
// Loose and Compatibility are supported; mode 1 may fall back to Loose.
// Mode 2 returns UNSUPPORTED_SEMANTICS; mode 3 returns UNSUPPORTED_FP_STATUS.
// Loose may combine arithmetic and OMOD rounding; Compatibility rounds to FP16
// before OMOD, preserving rocjitsu behavior (including intermediate overflow).
// Loose semantics require host nearest-even arithmetic with denormals enabled.
// ADD/SUB/SUBREV/MUL accept RJ_ARITH_FP_ROUND_MASK. Directed guest rounding uses
// exact FP64 intermediates and integer narrowing on a scalar path; nearest-even
// retains SIMD dispatch. RDNA3 also accepts RJ_ARITH_FP_DX10_CLAMP_OFF.
int rj_arith_v_add_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_add_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_sub_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_sub_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_subrev_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_subrev_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_mul_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

// DX9 multiplication: one VGPR per operand, with A/B ABS/NEG, OMOD and CLAMP.
// Either signed-zero input forces a positive-zero product, including with NaN
// or infinity as the other input. Exact results use FMA(A, B, signed zero)
// after the zero override and honor RJ_ARITH_FP_ROUND_MASK.
// DX9 multiply without _gfx12 uses RDNA3's conditional OMOD rule; _gfx12 applies OMOD
// regardless of guest IEEE/output-denormal mode. Both accept guest input/output
// flushing. Loose execution retains SIMD. All modes assume the default host FP
// environment. FP status reporting remains unsupported.
int rj_arith_v_mul_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_mul_dx9_zero_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          uint32_t *excp_flag_user);

// Wave64 FP32 binary forms follow the corresponding Wave32 contracts, including
// source/output modifiers, DPP routing, exact results and gfx12 MUL status support.
int rj_arith_v_add_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_add_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_sub_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_sub_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_subrev_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_subrev_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_mul_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_mul_dx9_zero_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          uint32_t *excp_flag_user);

// Wave64 ordinary FP16 binary forms retain the corresponding Wave32 layout,
// modifier, numerical-semantics and FP-status contracts.
int rj_arith_v_add_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_add_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_sub_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_sub_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_subrev_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_subrev_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_mul_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

// FP64 arithmetic: two VGPRs per operand, holding each lane's low/high words.
// Supports ALU ABS/NEG on present sources, OMOD and CLAMP, with loose semantics.
// Mode 2 returns UNSUPPORTED_SEMANTICS; mode 3 returns UNSUPPORTED_FP_STATUS.
// Compatibility resolves to Loose; dropping status alone does not enable exact FP64 arithmetic.
// Nonzero RJ_ARITH_FP_ROUND_* flags are rejected with INVALID_FLAGS, including
// in Loose and Compatibility. Rejection preserves all outputs.
// Host nearest-even rounding and enabled denormals are required. Whole VGPR
// aliases may cross operand halves; sources are read before destination writes.
// ADD/MUL/FMA use APIs without _gfx12 for RDNA3 and _gfx12 variants for RDNA4.
// RDNA3 ignores OMOD when output denormals are preserved or RJ_ARITH_FP_IEEE_MODE is
// set; RDNA4 applies it regardless. These three families accept guest input/output
// denormal flushing, also in Wave64. They remain loose-only: IEEE-mode support
// does not promise exact NaN behavior or exception reporting.
// CLAMP maps NaNs to +0; RDNA3 preserves NaNs with RJ_ARITH_FP_DX10_CLAMP_OFF.
int rj_arith_v_add_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_add_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_add_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_add_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_mul_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_mul_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_fma_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);
int rj_arith_v_fma_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);
int rj_arith_v_fma_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);
int rj_arith_v_fma_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// Unary FP64 arithmetic uses the same low/high VGPR layout, Compatibility
// resolution, and guest-rounding flag rejection as FP64 FMA above. Guest
// input/output-denormal flags control flushing. RDNA3 disables OMOD when IEEE
// is set or output denormals are preserved; RDNA4 applies OMOD independently.
// Supports A ABS/NEG,
// OMOD and CLAMP, with loose semantics. RNDNE rounds ties to even; FRACT computes
// x - floor(x), capped at 0x3fefffffffffffff before output modifiers.
// These RDNA3 unary forms accept DX10_CLAMP_OFF to preserve NaNs under CLAMP.
int rj_arith_v_trunc_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_trunc_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);
int rj_arith_v_trunc_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_trunc_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_ceil_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_ceil_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);
int rj_arith_v_ceil_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_ceil_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rndne_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rndne_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);
int rj_arith_v_rndne_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rndne_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_floor_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_floor_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);
int rj_arith_v_floor_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_floor_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_fract_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_fract_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);
int rj_arith_v_fract_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_fract_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_sqrt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sqrt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);
int rj_arith_v_sqrt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sqrt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rcp_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);
int rj_arith_v_rcp_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_rsq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rsq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);
int rj_arith_v_rsq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rsq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_frexp_mant_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_frexp_mant_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_frexp_mant_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user);

int rj_arith_v_frexp_mant_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user);

// FP16 FREXP exponent: read the selected A half and write a signed int16_t
// exponent into the selected D half. Zero, infinity and NaN return zero.
// Preserves the other D half, inactive lanes and the host FP environment;
// whole-register A/D aliasing is allowed. ABS_A/NEG_A, OMOD and CLAMP are
// accepted but do not change the result. Input flushing makes subnormal inputs return zero;
// IEEE and output flushing do not affect the exponent. Supports DPP8/DPP16 on A
// and both loose and exact semantics.
// RDNA3 also accepts and ignores RJ_ARITH_FP_DX10_CLAMP_OFF.
// Exact FP status reporting is not implemented.
int rj_arith_v_frexp_exp_i16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

// Wave64 follows the corresponding Wave32 exponent contract, with 64-lane operands and EXEC.
int rj_arith_v_frexp_exp_i16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

// Binary exponent extraction into one signed 32-bit VGPR. A uses one FP32
// VGPR or an FP64 low/high pair. Returns zero for zeros, infinities and NaNs;
// preserved finite nonzero inputs satisfy A = FREXP_MANT(A) * 2^D.
// A ABS/NEG, OMOD and CLAMP are accepted but do not change the integer result.
// D may alias either whole source VGPR. FP64 supports loose semantics only
// and honors guest input flushing. IEEE and output flushing do not change the integer result.
// FP32 supports exact semantics and guest input flushing; output flushing and
// IEEE mode do not change its integer result. Flushed subnormals return zero.
// Both RDNA3 widths accept DX10_CLAMP_OFF without changing the integer result.
// Mode 3 FP status reporting is unsupported on both architectures.
// FP status pointers are unused in modes 0..2; a null pointer does not enable mode 3.
// The FP32 form supports DPP8/DPP16 on A and preserves the host FP environment.
int rj_arith_v_frexp_exp_i32_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i32_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

int rj_arith_v_frexp_exp_i32_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i32_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

int rj_arith_v_frexp_exp_i32_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i32_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

int rj_arith_v_frexp_exp_i32_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i32_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

// FP16 LDEXP: scale the selected half of A by 2 raised to the signed int16_t
// exponent in the selected half of B. Supports A ABS/NEG, HIGH_A/B/D, OMOD,
// CLAMP, RJ_ARITH_FP16_OVFL, and DPP8/DPP16 on A. Other source modifiers are invalid.
// Preserves the other D half and inactive lanes; D may alias a whole source VGPR.
// Accepts guest IEEE/input/output-denormal flags. RDNA3 disables OMOD when IEEE
// is set or output denormals are preserved; RDNA4 does not. Supports guest
// rounding; RDNA3 also accepts RJ_ARITH_FP_DX10_CLAMP_OFF to retain NaNs under CLAMP.
// The instruction rounds to FP16 before OMOD. Output flushing and active OMOD
// test tininess before that rounding; OMOD flushes tiny results to +0.
// Exact FP status reporting is not implemented. Exact and directed-round paths
// are scalar; loose nearest-even retains SIMD. All require the default host FP environment.
int rj_arith_v_ldexp_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

// Wave64 follows the corresponding Wave32 exponent contract, with 64-lane operands and EXEC.
int rj_arith_v_ldexp_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

// Scale A by 2^B, with a signed 32-bit integer exponent in one B VGPR.
// FP32 A/D each use one VGPR; FP64 A/D use low/high pairs. Supports A ABS/NEG,
// OMOD and CLAMP. Guest IEEE and input/output denormal
// controls are accepted. RDNA3 disables OMOD when IEEE is set or output
// denormals are preserved; RDNA4 applies OMOD independently. Integer B has no
// sign modifiers. D may alias any whole source VGPR; FP64 writes low then high.
// Both widths support exact numerical semantics and all guest rounding
// modes using borrowed integer rounding. Loose nearest-even keeps SIMD dispatch;
// directed rounding and exact calls use scalar integer arithmetic. Exact optional
// FP status reporting remains unsupported.
// Output flushing and active OMOD test tininess before rounding.
// Both RDNA3 widths accept DX10_CLAMP_OFF to preserve NaNs under CLAMP.
// The FP32 form supports DPP8/DPP16 on A; B is not permuted.
int rj_arith_v_ldexp_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_ldexp_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

// FP64 forms follow the same LDEXP contract, with two A/D VGPRs per lane.
int rj_arith_v_ldexp_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_ldexp_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

// Wave32 mixed FMA: one VGPR per operand; supports source ABS/NEG, source
// format/half selectors, and CLAMP. DPP8/DPP16 permute A before format/half
// selection and modifiers; B and C retain their lanes. OMOD and HIGH_D are
// invalid. MIX_F32
// produces FP32; MIXLO/MIXHI round directly to FP16 and preserve the other
// destination half. RJ_ARITH_FP16_OVFL saturates finite FP16 overflow only.
// Loose semantics require host nearest-even arithmetic with denormals enabled.
// All result types support empirical-exact scalar guest rounding via
// RJ_ARITH_FP_ROUND_MASK without changing host FP controls. Loose SIMD uses host
// nearest-even arithmetic even when a guest rounding flag is supplied. RDNA3 accepts
// RJ_ARITH_FP_DX10_CLAMP_OFF to preserve NaNs under CLAMP; RDNA4 rejects that flag. EXEC masks and
// whole-register aliases are supported. Forms without _gfx12 select RDNA3; _gfx12 forms select
// RDNA4. MIX_F32 applies guest flushing using each source's selected FP16/FP32 type and its FP32
// result. MIXLO/MIXHI also flush by source type, then round directly to FP16 with guest output
// flushing. RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK optionally overrides input flushing for FP16
// sources, independently of the common FP32 input policy. IEEE-dependent NaN handling in loose
// MIX_F32 remains approximate. RDNA3 FP status reporting is unsupported. RDNA4 MIX_F32 also rejects
// exact reporting; RDNA4 half-output forms retain their no-op reporting.
// MIX_F32 maps rounding/output preservation from MODE FP_ROUND[1:0]/FP_DENORM[1];
// MIXLO/MIXHI use FP_ROUND[3:2]/FP_DENORM[3]. FP32 input preservation comes from
// FP_DENORM[0]; FP16 input preservation comes from FP_DENORM[2]. Translate differing
// input policies using RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK; without an override,
// FP16 inputs inherit the common FP32 input policy.
int rj_arith_v_fma_mix_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *trapsts);
int rj_arith_v_fma_mix_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     uint32_t *excp_flag_user);
int rj_arith_v_fma_mix_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *trapsts);
int rj_arith_v_fma_mix_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     uint32_t *excp_flag_user);

int rj_arith_v_fma_mixlo_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);
int rj_arith_v_fma_mixlo_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);
int rj_arith_v_fma_mixlo_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);
int rj_arith_v_fma_mixlo_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_fma_mixhi_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);
int rj_arith_v_fma_mixhi_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);
int rj_arith_v_fma_mixhi_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);
int rj_arith_v_fma_mixhi_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
