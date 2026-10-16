// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_FP_H_
#define RJ_ARITH_FP_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Fused division post-scaling: compute A*B+C and apply a power-of-two scale
// before the single guest rounding (RJ_ARITH_FP_ROUND_MASK). A set lane bit in condition (the
// implicit wave32 VCC input) selects +64/+128 when C's modified encoded exponent
// exceeds its bias, and -64/-128 otherwise, for FP32/FP64 respectively.
// Supports all source ABS/NEG, OMOD and CLAMP, loose and empirical exact semantics.
// Inputs preserve denormals even with RJ_ARITH_FP_FLUSH_INPUT_DENORMALS.
// RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS flushes rounded subnormals to signed zero.
// APIs without _gfx12 use RDNA3 TRAPSTS; _gfx12 APIs use EXCP_FLAG_USER.
// RDNA3 ignores OMOD when output denormals are preserved or IEEE mode is set;
// RDNA4 applies OMOD independently. RDNA3 strict exact with active OMOD is
// unsupported. Mode 3 FP status reporting is unsupported for RDNA3 and for RDNA4
// with output flushing or directed rounding. Loose semantics leave optional state unchanged.
// Active OMOD rounds at normal precision before
// flushing tiny results to +0, then applies the output scale and clamp.
// FP64 operands use low/high VGPR pairs, with D1 winning when D0/D1 alias.
// Supports every EXEC mask and whole-register alias; zero effective EXEC permits
// null VGPR pointers. Requires the default host FP environment; host controls
// are never explicitly changed. Ordinary arithmetic may raise sticky host flags.
// RJ_ARITH_FP16_OVFL has no effect.
int rj_arith_v_div_fmas_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t condition, uint32_t *trapsts);

int rj_arith_v_div_fmas_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t condition, uint32_t *trapsts);

// Wave64 uses one condition bit per lane and retains the same numerical,
// modifier and optional exception-accumulation contracts.
int rj_arith_v_div_fmas_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint64_t condition, uint32_t *trapsts);
int rj_arith_v_div_fmas_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint64_t condition, uint32_t *trapsts);

int rj_arith_v_div_fmas_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t condition,
                                      uint32_t *excp_flag_user);

int rj_arith_v_div_fmas_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t condition,
                                      uint32_t *excp_flag_user);

int rj_arith_v_div_fmas_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint64_t condition,
                                      uint32_t *excp_flag_user);

int rj_arith_v_div_fmas_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint64_t condition,
                                      uint32_t *excp_flag_user);

// Division pre-scaling. A must equal B (denominator) or C (numerator) after
// source NEG modifiers. Writes the pre-scaled value to D and the per-lane
// post-scaling condition to condition (the wave32 SDST operand).
// Supports source NEG, OMOD and CLAMP; ABS is not encoded by these instructions.
// Condition bits for inactive lanes are cleared. Even zero EXEC writes zero to
// condition, so that pointer must always be writable; VGPR pointers may then be
// null. The condition is written after VGPR outputs. Errors leave both unchanged.
// FP32 uses one VGPR per operand; FP64 uses low/high pairs, with D1 winning if
// D0/D1 alias. Supports all whole-register aliases, loose and empirical exact
// semantics, RJ_ARITH_FP_ROUND_MASK rounding and guest input/output denormal controls.
// Nearest-even retains SIMD paths; directed rounding uses the integer scalar path.
// Results preserve host FP state
// and do not depend on it. RJ_ARITH_FP16_OVFL has no effect.
// FP64 preserves source signaling NaNs on RDNA3; the _gfx12 variants quiet them.
// RDNA3 ignores OMOD unless output denormals are flushed and IEEE mode is clear.
// RDNA4 applies OMOD independently of these guest settings. Active RDNA3 OMOD
// supports loose semantics only; strict exact returns unsupported semantics.
int rj_arith_v_div_scale_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_div_scale_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       uint32_t *condition, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_div_scale_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_div_scale_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       uint32_t *condition, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c);

// Wave64 retains the same contract with a 64-bit condition result. Inactive
// condition bits are zero; empty EXEC writes zero. Condition is committed after
// all VGPR writes and may share their storage, including unaligned storage.
int rj_arith_v_div_scale_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *condition, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_div_scale_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       uint64_t *condition, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_div_scale_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *condition, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_div_scale_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       uint64_t *condition, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c);

// Division fixup: A is a provisional quotient, B the original denominator,
// C the original numerator. Repairs sign, propagates C/B NaNs in that order,
// and handles zero/infinity cases and extreme FP32/FP64 exponent underflow.
// FP16/FP32 operands hold one VGPR each; FP64 operands use low/high VGPR pairs.
// FP16 supports HIGH_A/B/C/D and preserves the unselected D half. FP16 also
// supports DPP8/DPP16 on A before half selection and source modifiers; B/C
// retain their lanes. FP32/FP64 reject DPP. FP64 writes
// D[0] before D[1], so D[1] wins if both destination pointers alias.
// Supports all source ABS/NEG, OMOD and CLAMP. Nonzero OMOD flushes subnormals
// before scaling, maps existing zeros to +0 and preserves signed underflow zero.
// RDNA4 FP16_OVFL saturates provisional-quotient overflow before OMOD and finite
// scaling overflow; exceptional B/C cases retain their infinity/NaN behavior.
// RDNA3 FP16 preserves nearest-rounded provisional infinities, scales in the promoted domain,
// then narrows and saturates finite overflow; active OMOD flushes rounded
// subnormals and signed zeros to +0. Subnormal inputs can scale to normal results.
// All formats accept RJ_ARITH_FP_ROUND_MASK for overflow recovery; FP16 output
// narrowing remains nearest-even. Nearest-even retains SIMD; directed rounding
// uses integer scalar recovery and scaling. Denormals are preserved by default.
// All formats accept guest input/output-denormal flags and IEEE mode. Input flushing applies to
// B/C, never A; output flushing does not flush the provisional quotient. RDNA3 ignores OMOD with
// preserved output denormals or IEEE mode, while RDNA4 applies it independently. APIs without
// _gfx12 use RDNA3 TRAPSTS; _gfx12 uses EXCP_FLAG_USER. Mode 3 FP status reporting is unsupported
// for RDNA3 and directed FP16 rounding. RDNA4 input flushing classifies the flushed
// denominator/numerator and suppresses input-denormal flags. Loose calls leave optional state
// unchanged. All EXEC masks and whole-register aliases are supported. Results are independent of
// host FP controls, which are preserved; ordinary instructions may raise sticky host exception
// flags. RDNA3 CLAMP models DX10_CLAMP=1 unless RJ_ARITH_FP_DX10_CLAMP_OFF is supplied to the RDNA3
// FP16 API.
int rj_arith_v_div_fixup_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);

int rj_arith_v_div_fixup_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_div_fixup_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);

int rj_arith_v_div_fixup_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);

// Wave64 retains each format's operand layout, modifiers and exact exception
// accumulation. FP16 supports four independent DPP rows.
int rj_arith_v_div_fixup_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);

int rj_arith_v_div_fixup_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);
int rj_arith_v_div_fixup_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);
int rj_arith_v_div_fixup_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);

int rj_arith_v_div_fixup_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_div_fixup_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_div_fixup_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_div_fixup_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

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

// Packed FP16 binary arithmetic: supports RJ_ARITH_PK_* negation and half selectors
// for A/B, plus RJ_ARITH_PK_CLAMP and RJ_ARITH_FP16_OVFL. Flags for C are invalid. Each
// result rounds to FP16; the two results use original inputs even with aliases.
// Number min/max ignore a lone NaN; minimum/maximum propagate NaNs. Both order
// -0 below +0. Loose semantics require host nearest-even with denormals enabled.
// ADD/MUL accept guest IEEE and input/output denormal flags. Flushing applies
// to each selected input half and each rounded output half. In the RDNA3 scalar
// guest-mode path, IEEE controls NaN quieting. Directed guest rounding uses
// scalar FP64 arithmetic and integer narrowing. RDNA3 RJ_ARITH_FP_DX10_CLAMP_OFF uses
// that scalar path to preserve NaNs under CLAMP. These remain loose-only APIs.
// Packed selection honors guest input/output denormal flags and IEEE mode;
// rounding flags have no effect on selection. RDNA3 MIN/MAX support
// RJ_ARITH_FP_DX10_CLAMP_OFF, preserving NaNs under CLAMP without leaving SIMD.
// ADD/MUL without _gfx12 take RDNA3 trapsts; RDNA4 variants
// take excp_flag_user. Both reject mode 3 FP status reporting and skip loose state.
int rj_arith_v_pk_add_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_add_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_mul_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_mul_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_min_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_max_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_minimum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_maximum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

// Binary FP32 arithmetic: one VGPR per operand. APIs without _gfx12 model RDNA3:
// OMOD is ignored unless output flushing is enabled and RJ_ARITH_FP_IEEE_MODE is clear.
// The _gfx12 APIs apply OMOD regardless of these flags. Both architectures support
// RJ_ARITH_FP_FLUSH_INPUT_DENORMALS and RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS. Active OMOD flushes
// tiny results to +0; halving a normal result below twice minimum normal gives signed zero.
// Supports ABS/NEG for A/B, OMOD, CLAMP, and DPP8/DPP16. Flags for C and half selection are
// invalid. ADD/MUL support exact results through rocjitsu's FMA model with RJ_ARITH_FP_ROUND_MASK
// guest rounding. SUB/SUBREV also support exact IEEE subtraction through the integer sum model.
// MUL_DX9_ZERO also supports exact multiplication with its zero override.
// Loose binary paths accept but ignore
// guest arithmetic rounding flags, retaining SIMD dispatch. OMOD scaling honors
// guest rounding for overflow in both exact and loose paths.
// MUL supports FP status reporting for nearest-even guest rounding using
// rocjitsu's inherited product classifier: input-denormal, overflow, underflow
// and inexact, with RDNA4 OMOD suppressing underflow/inexact. Classification
// precedes guest flushing and CLAMP; INVALID is not modeled. This reproduces
// the legacy classifier under nearest-even host arithmetic, not independently
// validated hardware completeness. Directed-rounding reporting is unsupported.
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

// FP32 min/max: one VGPR per operand; supports A/B ABS/NEG, OMOD and CLAMP.
// Loose semantics only. Number variants prefer numeric operands over NaNs;
// minimum/maximum propagate NaNs, preferring signaling NaNs and quieting them.
// Both families order -0 below +0. Guest input flushing applies before selection;
// output flushing is ignored except as required by OMOD. RJ_ARITH_FP_IEEE_MODE is
// accepted without changing RDNA4's selection or OMOD rules.
int rj_arith_v_min_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_max_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_minimum_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_maximum_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

// RDNA3 FP16/FP32/FP64 MIN/MAX use guest IEEE mode: signaling NaNs take priority when set;
// otherwise a lone NaN is ignored and two NaNs select A unchanged. OMOD is ignored
// when IEEE is set or output denormals are preserved. Input flushing precedes
// selection; output flushing otherwise has no effect. Both widths use SIMD.
//
// Packed FP16 MIN/MAX apply the same NaN and input-denormal selection rules to
// both halves, with their independent selectors/negations. Packed forms have no
// OMOD; output-denormal flushing applies after narrowing. FP16 narrowing
// preserves the selected signaling bit when quieting is not requested.
// Binary FP16 selection accepts and ignores RJ_ARITH_FP_ROUND_MASK, retaining its
// fixed-nearest output policy and SIMD dispatch. RDNA3 binary FP16 also accepts
// RJ_ARITH_FP_DX10_CLAMP_OFF to preserve NaNs under CLAMP.
//
// RDNA3 FP16/FP32 MIN3/MAX3/MINMAX/MAXMIN compose the binary rules above. MED3
// uses MIN3 for any NaN, otherwise removes the first maximum. Input flushing
// and conditional OMOD apply in both widths, including the eight-lane SIMD path.
// Ternary selection in both architectures accepts and ignores RJ_ARITH_FP_ROUND_MASK,
// retaining its fixed-nearest output policy. RDNA3 accepts RJ_ARITH_FP_DX10_CLAMP_OFF
// to preserve NaNs under CLAMP, including in SIMD paths.
//
// These selection APIs remain loose-only. Mode 2 returns UNSUPPORTED_SEMANTICS;
// mode 3 returns UNSUPPORTED_FP_STATUS. Modes 0 and 1 leave FP status unchanged. Binary F64 and
// packed forms reject DPP; the other forms route A with DPP8/DPP16.
int rj_arith_v_min_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_max_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_min_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_max_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

// Wave64 FP32 binary forms preserve the Wave32 source/output modifiers, DPP
// routing, loose semantics and optional-state contract, including RDNA3 MIN/MAX policy.
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

int rj_arith_v_min_num_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_max_num_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_minimum_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_maximum_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_mul_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_mul_dx9_zero_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          uint32_t *excp_flag_user);

int rj_arith_v_min_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_max_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

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

int rj_arith_v_min3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_max3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_minmax_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_maxmin_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_med3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

// Wave64 three-input F16 min/max forms retain Wave32 operand ordering,
// modifiers, DPP routing, loose semantics and FP status contracts.
int rj_arith_v_min3_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_max3_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_minmax_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user);

int rj_arith_v_maxmin_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user);

int rj_arith_v_minimum3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_maximum3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_minimummaximum_f16_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_maximumminimum_f16_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_med3_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_min3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_max3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_minmax_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_maxmin_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_med3_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_min3_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_max3_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_minmax_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_maxmin_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_med3_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

// Wave64 three-input F32 min/max forms retain Wave32 operand ordering,
// modifiers, DPP routing, loose semantics and FP status contracts.
int rj_arith_v_min3_num_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_max3_num_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_minmax_num_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user);

int rj_arith_v_maxmin_num_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user);

int rj_arith_v_minimum3_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_maximum3_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_minimummaximum_f32_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_maximumminimum_f32_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_med3_num_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_min3_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_max3_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_minmax_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_maxmin_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_med3_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts);

int rj_arith_v_pk_min_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_max_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

// Wave64 ordinary/packed FP16 binary forms retain Wave32 half selectors,
// modifiers, loose semantics and FP status contracts. Only ordinary forms
// support DPP; packed forms compute both output components from original inputs.
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

int rj_arith_v_min_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_max_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_minimum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_maximum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_add_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_add_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_mul_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_mul_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_min_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_max_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_minimum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_pk_maximum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_min_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_max_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *trapsts);

int rj_arith_v_pk_min_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_pk_max_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts);

// Three-input FP32 min/max: one VGPR per operand; supports all A/B/C ABS/NEG,
// OMOD and CLAMP, with loose semantics. First select between A/B, then between
// that result and C; output modifiers apply only after both selections.
// MIN3/MAX3 repeat the same selection; MINMAX/MAXMIN apply opposite selections.
// Number/propagating variants follow the binary NaN and signed-zero rules above.
// Guest input flushing precedes selection; output denormals survive selection.
// RDNA4 OMOD applies independently of guest IEEE/output-denormal mode.
int rj_arith_v_min3_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_max3_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_minmax_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user);

int rj_arith_v_maxmin_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user);

int rj_arith_v_minimum3_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_maximum3_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_minimummaximum_f32_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_maximumminimum_f32_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

// DX9 multiplication: one VGPR per operand, with A/B ABS/NEG, OMOD and CLAMP.
// Either signed-zero input forces a positive-zero product, including with NaN
// or infinity as the other input. Exact results use the inherited integer
// multiply model after the zero override and honor RJ_ARITH_FP_ROUND_MASK.
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

// FP32 median: one VGPR per operand; all A/B/C ABS/NEG, OMOD and CLAMP, loose
// semantics. Any NaN selects minimumNumber across all three inputs. Otherwise
// remove the first input numerically equal to the maximum and select the maximum
// of the remaining two inputs, following the ISA's signed-zero tie behavior.
int rj_arith_v_med3_num_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

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

// Binary exponent extraction into one signed 32-bit VGPR. A uses one FP32
// VGPR or an FP64 low/high pair. Returns zero for zeros, infinities and NaNs;
// preserved finite nonzero inputs satisfy A = FREXP_MANT(A) * 2^D.
// A ABS/NEG, OMOD and CLAMP are accepted but do not change the integer result.
// D may alias either whole source VGPR. FP64 supports loose semantics only
// and accepts guest IEEE/input/output-denormal flags without numeric effect.
// FP32 supports exact semantics and guest input flushing; output flushing and
// IEEE mode do not change its integer result. Flushed subnormals return zero.
// RDNA3 FP32 also accepts DX10_CLAMP_OFF without changing the integer result.
// Mode 3 FP status reporting is unsupported on both architectures.
// Null opts out; loose calls leave optional state untouched.
// The FP32 form supports DPP8/DPP16 on A and preserves the host FP environment.
int rj_arith_v_frexp_exp_i32_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);
int rj_arith_v_frexp_exp_i32_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

int rj_arith_v_frexp_exp_i32_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i32_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

// Wave64 exponent extraction uses the same operand layouts and semantics,
// with 64 lanes per VGPR. Optional state reporting is unchanged.
int rj_arith_v_frexp_exp_i32_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);
int rj_arith_v_frexp_exp_i32_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

int rj_arith_v_frexp_exp_i32_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i32_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

// Scale A by 2^B, with a signed 32-bit integer exponent in one B VGPR.
// FP32 A/D each use one VGPR; FP64 A/D use low/high pairs. Supports A ABS/NEG,
// OMOD and CLAMP with loose semantics. Guest IEEE and input/output denormal
// controls are accepted. RDNA3 disables OMOD when IEEE is set or output
// denormals are preserved; RDNA4 applies OMOD independently. Integer B has no
// sign modifiers. D may alias any whole source VGPR; FP64 writes low then high.
// Both widths support exact numerical semantics and all guest rounding
// modes using borrowed integer rounding. Loose nearest-even keeps SIMD dispatch;
// directed rounding and exact calls use scalar integer arithmetic. Exact optional
// FP status reporting remains unsupported.
// Both RDNA3 widths accept DX10_CLAMP_OFF to preserve NaNs under CLAMP.
// The FP32 form supports DPP8/DPP16 on A; B is not permuted.
int rj_arith_v_ldexp_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

// Wave64 retains the same operand layout, modifiers and loose semantics.
int rj_arith_v_ldexp_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_ldexp_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

// Wave64 retains the same operand layout, modifiers and loose semantics.
int rj_arith_v_ldexp_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

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

// Ordinary ADD/SUB/SUBREV/MUL use APIs without _gfx12 for RDNA3 and _gfx12 variants
// for RDNA4, after _w32 or _w64. RDNA3 ignores OMOD when output denormals are preserved
// or RJ_ARITH_FP_IEEE_MODE is set; RDNA4 applies it regardless. These four families
// accept guest input/output flushing. They remain loose-only; IEEE-mode support
// does not promise exact signaling-NaN behavior or exception reporting.
//
// Binary FP16 arithmetic: one VGPR per operand, with independently selected
// source and destination halves. Preserves the other destination half and all
// inactive lanes; D may alias a whole source VGPR. Supports source ABS/NEG,
// OMOD then CLAMP before FP16 narrowing, and RJ_ARITH_FP16_OVFL.
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

// Binary FP16 selection uses fixed-nearest narrowing and ignores guest rounding.
// Input flushing precedes selection; output-denormal mode alone has no effect.
// Modifiers, rounding flags and masks retain SIMD dispatch in loose semantics.
int rj_arith_v_min_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_max_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_minimum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_maximum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user);

// Three-input FP16 min/max and median: independently selected halves of A/B/C/D,
// with all source ABS/NEG, OMOD, CLAMP and RJ_ARITH_FP16_OVFL. Output modifiers apply
// after the final selection. Preserves the other D half and inactive lanes;
// any whole-VGPR aliases are allowed. MINMAX/MAXMIN select A/B first, then C.
// Number variants ignore lone NaNs; MINIMUM/MAXIMUM propagate NaNs. MED3 uses
// minimumNumber(A,B,C) if any input is NaN. Loose semantics require host
// nearest-even arithmetic with denormals enabled.
int rj_arith_v_min3_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_max3_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_minmax_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user);

int rj_arith_v_maxmin_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user);

int rj_arith_v_minimum3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_maximum3_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

int rj_arith_v_minimummaximum_f16_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_maximumminimum_f16_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_med3_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user);

// FP16 LDEXP: scale the selected half of A by 2 raised to the signed int16_t
// exponent in the selected half of B. Supports A ABS/NEG, HIGH_A/B/D, OMOD,
// CLAMP, RJ_ARITH_FP16_OVFL, and DPP8/DPP16 on A. Other source modifiers are invalid.
// Preserves the other D half and inactive lanes; D may alias a whole source VGPR.
// Accepts guest IEEE/input/output-denormal flags. RDNA3 disables OMOD when IEEE
// is set or output denormals are preserved; RDNA4 does not. Supports guest
// rounding; RDNA3 also accepts RJ_ARITH_FP_DX10_CLAMP_OFF to retain NaNs under CLAMP.
// Exact semantics use the borrowed rocjitsu model: scale before one final half
// rounding, then flush subnormal/zero outputs to +0 when OMOD is active. Exact
// FP status reporting is not implemented. Exact and directed-round
// paths are scalar and do not depend on host rounding/denormal controls. Loose
// nearest-even retains SIMD and requires host nearest-even with denormals enabled;
// its OMOD path may round the unscaled intermediate before output scaling.
int rj_arith_v_ldexp_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

// FP16 FREXP exponent: read the selected A half and write a signed int16_t
// exponent into the selected D half. Zero, infinity and NaN return zero.
// Preserves the other D half, inactive lanes and the host FP environment;
// whole-register A/D aliasing is allowed. ABS_A/NEG_A, OMOD and CLAMP are
// accepted but do not change the result, as do guest IEEE/input/output-denormal
// flags. Supports DPP8/DPP16 on A and both loose and exact semantics.
// RDNA3 also accepts and ignores RJ_ARITH_FP_DX10_CLAMP_OFF.
// Exact FP status reporting is not implemented.
int rj_arith_v_frexp_exp_i16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

// Wave64 forms retain the same half selectors, DPP routing, loose semantics,
// and optional-state contract as the Wave32 forms above.
int rj_arith_v_ldexp_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts);

int rj_arith_v_ldexp_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user);

int rj_arith_v_frexp_exp_i16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts);

int rj_arith_v_frexp_exp_i16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, uint32_t *excp_flag_user);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
