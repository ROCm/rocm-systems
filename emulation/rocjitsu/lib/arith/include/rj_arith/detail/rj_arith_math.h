// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_MATH_H_
#define RJ_ARITH_MATH_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Reciprocal with sticky integer divide-by-zero status. Supports ABS_A, NEG_A,
// OMOD and CLAMP. Input/output subnormals always flush, independently of guest
// flush flags. RDNA3 OMOD applies only when guest output flushing is enabled
// and RJ_ARITH_FP_IEEE_MODE is clear; RDNA4 OMOD applies independently.
// RDNA3 supports RJ_ARITH_FP_DX10_CLAMP_OFF to preserve NaNs under CLAMP; RDNA4
// rejects this flag. RDNA4 active zero/subnormal inputs generate INT_DIV0 unless clamped.
// Existing excp_flag_user bits, including a pre-existing INT_DIV0, survive.
// DPP8/DPP16 permutes A before source modifiers. DPP-filtered destination lanes
// add no exception; a zero-filled source in an active lane can set INT_DIV0.
// FP status is accumulated in mode 3; modes 0..2 do not access the status pointer.
// Unlike the general no-overlap requirement, this
// instruction also permits it to alias any operand word: its initial value is
// read before VGPR stores and its final write takes precedence on overlap.
// Zero EXEC permits null VGPR pointers and leaves all outputs unchanged.
// Errors leave all outputs unchanged. Loose and empirical exact results are
// supported; exact RCP_IFLAG always quiets NaNs, including on RDNA3.
// RDNA4 supports mode 3 exception accumulation; loose calls leave FP status unchanged.
// RDNA3 exact reporting inherits rocjitsu's signed-zero-only INT_DIV0 classifier,
// including under CLAMP; nonzero subnormals do not raise it. This is a borrowed
// model, not a hardware-validated RDNA3 exception capture. RDNA3 loose calls skip
// TRAPSTS updates.
int rj_arith_v_rcp_iflag_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_iflag_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, uint32_t *excp_flag_user);

// Wave64 retains the same numerical and FP-status accumulation behavior,
// with four independent DPP rows. Status is committed after all VGPR writes.
int rj_arith_v_rcp_iflag_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_iflag_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, uint32_t *excp_flag_user);

// Trigonometric range-reduction table lookup. A is an FP64 VGPR pair, B is
// one integer VGPR (only its low five bits select the segment), and D is an
// FP64 VGPR pair. Only A's encoded exponent affects the lookup, including
// for infinities and NaNs. ABS_A/NEG_A are accepted and have no effect.
// Supports OMOD and CLAMP. RDNA3 uses rocjitsu's 1201-bit table; RDNA4 uses
// the captured 1184-bit table. RDNA3 scales before flushing the rounded result;
// RDNA4 OMOD flushes tiny results before and after scaling. RDNA3 disables OMOD
// when output denormals are preserved or IEEE mode is set. All three guest FP
// flags are accepted; input/output flushing otherwise does not affect lookup.
// APIs without _gfx12 use TRAPSTS; _gfx12 uses EXCP_FLAG_USER. Mode 3
// reporting is unsupported; loose calls leave FP status unchanged.
// Supports loose and empirical-exact semantics. Never changes host FP controls.
// If destination VGPRs alias, the high word is written last.
int rj_arith_v_trig_preop_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_trig_preop_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

// Wave64 retains the same operand layout, modifiers and numerical contract.
int rj_arith_v_trig_preop_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts);

int rj_arith_v_trig_preop_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *excp_flag_user);

// FP32 sine/cosine of inputs measured in turns: sin(2*pi*A), cos(2*pi*A).
// Supports ABS_A, NEG_A, OMOD, CLAMP and DPP8/DPP16. Empirical exact semantics
// use rocjitsu's RDNA3/4 integer model. Guest input/output denormal flushing
// applies before output scaling. RDNA3 quiets NaNs only with IEEE mode enabled;
// IEEE also disables RDNA3 OMOD independently of guest output-denormal mode.
// RDNA3 OMOD flushes tiny inputs before scaling, and halving a normal result
// into the subnormal range produces signed zero. RDNA4 retains RNE scaling
// followed by flushing tiny results and signed zeros to +0. CLAMP models
// DX10_CLAMP=1 by default; RDNA3 accepts RJ_ARITH_FP_DX10_CLAMP_OFF to preserve
// NaNs through CLAMP. All three guest FP flags are accepted.
// Exact arithmetic never changes host FP controls. Loose SIMD arithmetic requires
// host nearest-even with denormals preserved; it never changes host controls.
// APIs without _gfx12 take RDNA3 trapsts; _gfx12 takes excp_flag_user. Mode 3
// reporting is unsupported on RDNA3 and with guest flushing on RDNA4. Existing
// RDNA4 reporting with denormals preserved is retained; loose reporting is skipped.
int rj_arith_v_sin_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_sin_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);
int rj_arith_v_cos_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cos_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

// Wave64 retains the same contract, with four independent DPP rows.
int rj_arith_v_sin_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_sin_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);
int rj_arith_v_cos_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cos_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

// FP16 sine/cosine of inputs measured in turns. Supports ABS_A, NEG_A, OMOD,
// CLAMP, HIGH_A/D and DPP8/DPP16; preserves the unselected D half. Both
// architectures share rocjitsu's integer trig model and RNE half rounding.
// Guest input flushing precedes evaluation; rounded-output flushing precedes
// OMOD. RDNA3 disables OMOD when output denormals are preserved or IEEE is set,
// and quiets NaNs only in IEEE mode. RDNA4 quiets NaNs and honors OMOD regardless
// of those settings. Active OMOD flushes tiny inputs/scaled results and signed
// zeros to +0. CLAMP models DX10_CLAMP=1 by default; RDNA3 accepts
// RJ_ARITH_FP_DX10_CLAMP_OFF to preserve NaNs through CLAMP. All three guest FP flags are accepted;
// FP16_OVFL has no effect on these bounded results. Exact semantics preserve
// host FP state. Loose SIMD requires host nearest-even with denormals enabled.
// APIs without _gfx12 take RDNA3 trapsts; _gfx12 takes excp_flag_user. Mode 3
// reporting is unsupported on RDNA3 and with guest flushing on RDNA4. Existing
// RDNA4 reporting with denormals preserved is retained; loose reporting is skipped.
int rj_arith_v_sin_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_sin_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_cos_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cos_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

// Wave64 retains the same contract, with four independent DPP rows.
int rj_arith_v_sin_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_sin_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_cos_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_cos_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

// FP32 fractional part: x - floor(x), capped below one before OMOD/CLAMP.
// Supports A ABS/NEG, OMOD, CLAMP and guest input/output flushing in loose
// semantics. One VGPR per operand; infinite inputs produce NaN.
// RDNA3 ignores OMOD with IEEE mode or preserved output denormals; RDNA4 does not.
// Required exact results and FP status reporting are unsupported.
int rj_arith_v_fract_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_fract_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

// Binary-significand extraction: finite nonzero inputs produce magnitude in
// [0.5,1); zeros and infinities pass through before output modifiers. FP32
// quiets signaling NaNs in exact semantics and supports guest input flushing.
// Supports A ABS/NEG, OMOD and CLAMP. FP32 uses one VGPR per operand and supports
// both semantics; FP64 uses low/high VGPR pairs with loose semantics and the
// FP64 alias contract. RDNA3 FP32 suppresses OMOD when IEEE mode is enabled or
// output denormals are preserved; RDNA4 scales independently. Non-null exact
// reporting is unsupported on both architectures; loose calls leave state alone.
int rj_arith_v_frexp_mant_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_frexp_mant_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user);

// Wave64 F32 unary forms retain Wave32 modifiers, DPP routing,
// numerical behavior, destination preservation and FP status contracts.
int rj_arith_v_trunc_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_trunc_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_ceil_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_ceil_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rndne_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rndne_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_floor_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_floor_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_sqrt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sqrt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rcp_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_rsq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rsq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_exp_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_exp_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_log_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_log_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_fract_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_fract_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_frexp_mant_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_frexp_mant_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user);

// Unary FP16: one independently selected half per A/D VGPR; supports ABS_A,
// NEG_A, HIGH_A/D, OMOD, CLAMP and RJ_ARITH_FP16_OVFL. Preserves the unselected D
// half and inactive lanes; whole-register A/D aliasing is allowed. EXP/LOG
// use base two, RNDNE rounds ties to even, and FRACT is capped below one.
// Guest ROUND controls FRACT narrowing and finite OMOD overflow; fixed-rounding
// integral instructions retain their named rounding rule for the core operation.
// Supports DPP8/DPP16 on A. Loose semantics require host nearest-even arithmetic
// with denormals enabled. SFU operations and FRACT round to FP16 before OMOD.
// All forms except FRACT support exact results. EXP/LOG use rocjitsu's FP64 libm
// model with integer narrowing and explicit NaN policy; no host controls are changed.
// All forms accept
// guest input/output-denormal flags and IEEE. EXP/LOG flush rounded half results
// before OMOD when output flushing is requested. APIs without _gfx12 select
// RDNA3: OMOD is disabled when IEEE is set or output denormals are preserved.
// RDNA4 OMOD is independent of these modes. Exact RDNA3 RCP/RSQ/SQRT retain
// signaling NaNs unless IEEE is set; RDNA4 quiets them. FP16_OVFL saturates
// zero-input RCP/RSQ infinities. Mode 3 FP status reporting is unsupported on
// both architectures. Loose paths leave FP status unchanged.
// RDNA3 accepts RJ_ARITH_FP_DX10_CLAMP_OFF to preserve arithmetic NaNs through
// CLAMP, after the operation's quieting policy. By default CLAMP maps them to
// +0. RDNA4 rejects this flag and always clamps NaNs to +0.
int rj_arith_v_trunc_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_trunc_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_ceil_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_ceil_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rndne_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rndne_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_floor_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_floor_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_sqrt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_sqrt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rcp_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rcp_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_rsq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rsq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_exp_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_exp_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_log_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_log_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_fract_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_fract_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

// FREXP preserves half subnormals independently of guest denormal flags,
// following the borrowed widen-to-FP32 model. Exact NaNs retain their sign
// and payload with the quiet bit set. Output modifiers apply afterward.
int rj_arith_v_frexp_mant_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_frexp_mant_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user);

// Wave64 F16 unary forms retain Wave32 modifiers, DPP routing,
// numerical behavior, destination preservation and FP status contracts.
int rj_arith_v_trunc_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_trunc_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_ceil_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_ceil_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rndne_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rndne_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_floor_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_floor_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_sqrt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_sqrt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rcp_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rcp_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_rsq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_rsq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_exp_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_exp_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_log_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_log_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_fract_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_fract_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_frexp_mant_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);

int rj_arith_v_frexp_mant_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user);

// Unary FP32: one VGPR each for A/D. Supports NEG_A, ABS_A, OMOD and CLAMP;
// modifiers for absent operands are invalid. CLAMP maps NaNs to +0 by default
// and clamps to [0, 1]. RDNA3 unary FP32 operations accept
// RJ_ARITH_FP_DX10_CLAMP_OFF to preserve arithmetic NaNs through CLAMP; RDNA4
// rejects that flag. The operation's NaN quieting policy still applies.
// EXP and LOG use base 2; RSQ computes reciprocal square root.
// RNDNE rounds ties to even. SQRT/RCP/RSQ/EXP/LOG also implement exact results.
// Guest ROUND controls FRACT subtraction and finite OMOD overflow.
// TRUNC/CEIL/RNDNE/FLOOR accept guest IEEE and denormal flags. Their RDNA3
// forms suppress OMOD when IEEE mode is enabled or output denormals are
// preserved; _gfx12 forms scale independently. Guest input flushing applies
// before rounding. Both architectures implement exact numerical results with
// quieted signaling NaNs. Mode 3 FP status reporting is unsupported on both
// architectures; loose calls leave FP status alone.
int rj_arith_v_trunc_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_trunc_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_ceil_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_ceil_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rndne_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rndne_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_floor_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_floor_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

// SQRT/RCP/RSQ/EXP/LOG force input/output flushing regardless of guest flush
// flags. RDNA3 forms disable OMOD when RJ_ARITH_FP_IEEE_MODE is set;
// _gfx12 forms apply OMOD independently. Exact SQRT reporting borrows rocjitsu's
// negative-input classifier: active negative nonzero inputs (including subnormals
// and infinity) raise INVALID after ABS/NEG, before flushing or CLAMP. NaNs and
// signed zeros do not contribute. This inherited model is not a claim of complete
// hardware exception validation. Other four operations do not support exact
// FP status reporting. Loose calls leave state untouched.
// All five retain exact numerical results borrowed from rocjitsu.
// Exact SQRT/RCP/RSQ/EXP/LOG use captured gfx1100/gfx1201 mappings. RDNA3 preserves
// signaling NaNs with IEEE mode off; RDNA4 always quiets them.
int rj_arith_v_sqrt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sqrt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user);

int rj_arith_v_rcp_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_rsq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rsq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_exp_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_exp_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

int rj_arith_v_log_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_log_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
