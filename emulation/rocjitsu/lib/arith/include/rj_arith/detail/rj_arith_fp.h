// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_FP_H_
#define RJ_ARITH_FP_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
int rj_arith_v_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);
int rj_arith_v_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
int rj_arith_v_fma_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);
int rj_arith_v_fma_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// Wave64 FP32 FMA forms retain the corresponding Wave32 modifier, literal,
// DX9-zero, semantics and FP status contracts. DPP routes only A.
int rj_arith_v_fma_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);
int rj_arith_v_fma_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

int rj_arith_v_fma_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_fma_dx9_zero_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c);

int rj_arith_v_fmac_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b);

int rj_arith_v_fmac_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *trapsts);
int rj_arith_v_fmac_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_fmamk_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                             const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_fmamk_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

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

// FP32 fused multiply-accumulate into D. Supports A/B ABS/NEG, OMOD and CLAMP.
// C modifiers and half selectors are invalid. Loose semantics only.
int rj_arith_v_fmac_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *trapsts);
int rj_arith_v_fmac_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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

int rj_arith_v_fmaak_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint16_t literal, uint32_t *trapsts);
int rj_arith_v_fmaak_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint16_t literal,
                                   uint32_t *excp_flag_user);

// Wave64 FP16 FMA forms retain the Wave32 exact/loose semantics, modifiers
// and half selectors. Exact optional exceptions accumulate across both halves,
// after all source reads and destination writes. Loose reporting is skipped.
int rj_arith_v_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);
int rj_arith_v_fma_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

int rj_arith_v_fmac_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *trapsts);
int rj_arith_v_fmac_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_fmamk_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                             const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_fmamk_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_fmaak_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint16_t literal, uint32_t *trapsts);
int rj_arith_v_fmaak_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint16_t literal,
                                   uint32_t *excp_flag_user);

// Literal FP32 FMA: FMAMK computes A * literal + B; FMAAK computes A * B +
// literal. The literal is a raw FP32 encoding. No instruction flags are valid.
// Forms without _gfx12 are RDNA3; _gfx12 forms are RDNA4. Guest input/output
// denormal flushing and IEEE mode are accepted. Loose semantics permit approximate
// NaN payload/quieting and require host nearest-even rounding and denormals enabled.
// Exact nearest-even results use integer arithmetic and do not depend on host
// FP controls. Mode 3 FP status reporting is unsupported on both architectures.
// Loose calls skip optional state updates.
int rj_arith_v_fmamk_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                             const uint32_t *const *b, uint32_t *trapsts);
int rj_arith_v_fmamk_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_fmaak_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t literal, uint32_t *trapsts);
int rj_arith_v_fmaak_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t literal,
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

// RDNA3 DX9 fused multiply-accumulate into D, with the same flushing and
// zero-product rules as DX9 FMA. Supports A/B ABS/NEG, OMOD and CLAMP.
// DPP, C modifiers and half selectors are invalid. Supports loose and exact RNE.
int rj_arith_v_fmac_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b);

// Packed FP16 FMA (RDNA3 without _gfx12; RDNA4 with _gfx12): two independent fused results per
// lane. All RJ_ARITH_PK_* flags below are supported. With no flags, corresponding input halves are
// multiplied and added. Each result rounds to FP16; CLAMP applies last. Supports RJ_ARITH_FP16_OVFL
// and empirical exact semantics with guest rounding, input/output flushing and IEEE mode. Directed
// rounding uses the scalar integer-intermediate path. RDNA3 supports RJ_ARITH_FP_DX10_CLAMP_OFF;
// that mode also uses the scalar path to preserve NaNs under CLAMP. RDNA3 preserves selected
// signaling NaNs with IEEE mode off; RDNA4 quiets them. Both halves use the original inputs,
// including when D aliases A/B/C.
int rj_arith_v_pk_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user);

// Packed FP16 FMAC: component-wise A * B + D. Supports RJ_ARITH_ALU_ABS_A/B,
// RJ_ARITH_ALU_NEG_A/B and RJ_ARITH_ALU_CLAMP, applied to both components. RJ_ARITH_ALU_OMOD_*
// is accepted but ignored by both architectures. C modifiers and selectors are invalid.
// Supports RJ_ARITH_FP16_OVFL and the same loose/exact semantics as packed FMA.
int rj_arith_v_pk_fmac_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);

// Wave64 packed FMA forms retain the Wave32 selectors and exact/loose contracts.
// Exact exceptions accumulate from both components and wave halves; the global
// output is updated after source reads and destination writes. Loose reporting is skipped.
int rj_arith_v_pk_fma_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user);

int rj_arith_v_pk_fmac_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user);

// RDNA3 FP status reporting is not yet supported. RDNA4 exact
// reporting supports nearest-even with preserved denormals only; use mode 2 to omit FP status
// reporting.
int rj_arith_v_pk_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_pk_fmac_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_pk_fmac_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *trapsts);

// RDNA3 FP64 MIN/MAX accept DX10_CLAMP_OFF, preserving a NaN result under CLAMP.
int rj_arith_v_min_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_max_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

// Wave64 FP64 forms retain two-VGPR low/high operand layouts and Wave32
// modifiers, loose semantics and FP status contracts. DPP is unsupported.
// Each lane's two result words are computed before stores; the high word wins
// when both destination VGPRs share storage.
int rj_arith_v_add_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_add_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_mul_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_fma_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_fma_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

int rj_arith_v_trunc_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_trunc_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_ceil_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_ceil_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rndne_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rndne_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_floor_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_floor_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_fract_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_fract_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_sqrt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_sqrt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rcp_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rcp_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_rsq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rsq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_min_num_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_max_num_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_minimum_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_maximum_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_frexp_mant_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_frexp_mant_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user);

int rj_arith_v_min_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_max_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

// FP64 arithmetic: two VGPRs per operand, holding each lane's low/high words.
// Supports ALU ABS/NEG on present sources, OMOD and CLAMP, with loose semantics.
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

int rj_arith_v_add_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_mul_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_fma_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_fma_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user);

// FP64 min/max uses two VGPRs per operand, with A/B ABS/NEG, OMOD and CLAMP.
// Number variants prefer numeric operands over NaNs; minimum/maximum propagate
// NaNs, preferring signaling NaNs and quieting them. Both order -0 below +0.
// Loose semantics and the same FP64 alias/host-FP-state contract apply.
// FP64 min/max accept guest input flushing; output flushing is ignored except
// through OMOD. RDNA4 number/propagating variants accept IEEE mode without effect.
// RDNA3 MIN/MAX use IEEE-mode sNaN priority and conditional OMOD, as in FP32.
int rj_arith_v_min_num_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_max_num_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_minimum_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_maximum_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

// Unary FP64 arithmetic uses the same low/high VGPR layout. Guest IEEE and
// input/output-denormal flags are accepted. They only affect RDNA3 OMOD:
// scaling is disabled when IEEE is set or output denormals are preserved.
// RDNA4 applies OMOD independently. These unary operations do not otherwise
// flush guest denormals. Supports A ABS/NEG,
// OMOD and CLAMP, with loose semantics. RNDNE rounds ties to even; FRACT computes
// x - floor(x), capped at 0x3fefffffffffffff before output modifiers.
// These RDNA3 unary forms accept DX10_CLAMP_OFF to preserve NaNs under CLAMP.
int rj_arith_v_trunc_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_trunc_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_ceil_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_ceil_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rndne_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rndne_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_floor_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_floor_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_fract_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_fract_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_sqrt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_sqrt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rcp_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rcp_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_rsq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rsq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_frexp_mant_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_frexp_mant_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user);

// Mixed FMA source formats: unset means FP32; set means FP16 in the half
// selected by RJ_ARITH_ALU_HIGH_A/B/C. Half selectors are ignored for FP32 sources.
static const uint32_t RJ_ARITH_MIX_F16_A = 1U << 13;
static const uint32_t RJ_ARITH_MIX_F16_B = 1U << 14;
static const uint32_t RJ_ARITH_MIX_F16_C = 1U << 15;

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
int rj_arith_v_fma_mix_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *trapsts);
int rj_arith_v_fma_mix_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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

int rj_arith_v_fma_mixhi_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);
int rj_arith_v_fma_mixhi_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

// Wave64 mixed FMA retains the Wave32 source formats, modifiers and DPP.
// MIXLO/MIXHI support exact results and leave exception state unchanged, as in
// the empirical model. MIX_F32 retains loose semantics and rejects exact reporting.
int rj_arith_v_fma_mix_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *trapsts);
int rj_arith_v_fma_mix_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     uint32_t *excp_flag_user);

int rj_arith_v_fma_mixlo_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);
int rj_arith_v_fma_mixlo_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
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
