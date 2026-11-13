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

// gfx942/gfx950 legacy EXP/LOG encodings share the ordinary CDNA EXP/LOG contracts.
int rj_arith_v_exp_legacy_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_exp_legacy_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_log_legacy_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_log_legacy_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts);

// CDNA TRIG_PREOP: A/D each span two VGPRs; B is one selector VGPR (low five bits).
// Compatibility retains the emulator's 1201-bit table, RTZ construction, then guest-rounded
// OMOD and final flushing. gfx942/gfx950 gate OMOD on output flushing and !IEEE;
// gfx1250 always applies it. With no OMOD, tiny pieces remain even under guest output flushing.
// ABS_A/NEG_A are accepted but cannot change A's exponent. CLAMP caps results at one.
// Loose shares this integer model; strict empirical modes are unsupported, FP status unused.
// Inputs may alias destinations. Errors/inactive lanes preserve outputs; zero EXEC permits nulls.
int rj_arith_v_trig_preop_f64_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *state);
int rj_arith_v_trig_preop_f64_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *state);
int rj_arith_v_trig_preop_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          uint32_t *state);

// gfx1250 Wave32 BF16 math: one VGPR per operand, selected by HIGH_A/HIGH_D.
// Supports ABS_A, NEG_A, OMOD and CLAMP; other destination bits are preserved.
// Compatibility uses inherited FP32 models, scales/clamps before truncating to BF16,
// and canonicalizes tiny/zero OMOD results to +0. Guest rounding/flush/IEEE and
// FP16_OVFL controls are accepted but do not change these inherited policies.
// Loose may use the same models. Strict empirical modes are unsupported.
// FP status is unused; errors and inactive lanes leave all output storage unchanged.
// Operands may overlap; zero EXEC permits null operand pointers.
int rj_arith_v_rcp_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);
int rj_arith_v_rsq_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);
int rj_arith_v_sqrt_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);
int rj_arith_v_exp_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);
int rj_arith_v_log_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);
int rj_arith_v_sin_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);
int rj_arith_v_cos_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);
int rj_arith_v_tanh_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *excp_flag_user);

// gfx1250 Wave32 TANH, one VGPR per operand. Supports ABS_A, NEG_A, OMOD and
// CLAMP; F16 additionally supports HIGH_A/HIGH_D and preserves the other half.
// Compatibility uses host float TANH with NaNs quieted. F16 narrows to nearest
// before applying output modifiers. The bounded result cannot overflow under OMOD;
// guest rounding/denormal/IEEE settings do not change this inherited policy. Loose
// can use a vector approximation. Strict empirical modes are unsupported; FP status is unused.
// Operands may alias. Errors/inactive lanes preserve storage; zero EXEC permits null operands.
int rj_arith_v_tanh_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);
int rj_arith_v_tanh_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);

// Reciprocal with sticky integer divide-by-zero status. Supports ABS_A, NEG_A,
// OMOD and CLAMP. Input/output subnormals always flush, independently of guest
// flush flags. RDNA3 OMOD applies only when RJ_ARITH_FP_IEEE_MODE is clear,
// whatever the guest flush flags; RDNA4 OMOD applies independently.
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
// supported; exact RCP_IFLAG preserves signaling NaNs on RDNA3 with IEEE clear
// and quiets them otherwise.
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
// Supports OMOD and CLAMP. Both architectures use the captured 1184-bit table, and OMOD flushes
// tiny results before and after scaling. RDNA3 disables OMOD when output denormals are preserved
// or IEEE mode is set. All three guest FP flags are accepted; output flushing applies even without
// OMOD. Input flushing does not affect the exponent lookup. APIs without _gfx12 use TRAPSTS; _gfx12
// uses EXCP_FLAG_USER. Mode 3 reporting is unsupported; loose calls leave FP status unchanged.
// Supports loose and empirical-exact semantics. Never changes host FP controls. If destination
// VGPRs alias, the high word is written last.
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
// IEEE disables RDNA3 OMOD; SIN additionally disables it when output denormals are preserved.
// RDNA3 OMOD flushes tiny inputs before scaling, and halving a normal result
// into the subnormal range produces signed zero. RDNA4 flushes tiny results
// before and after RNE output scaling; SIN div:2 retains the sign of a newly
// underflowing result, while other flushed results become +0. CLAMP models
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
// of those settings. Active OMOD flushes already-tiny results to +0; halving
// a normal result into the subnormal range retains its sign when flushing. CLAMP models
// DX10_CLAMP=1 by default; RDNA3 accepts RJ_ARITH_FP_DX10_CLAMP_OFF to preserve NaNs through CLAMP.
// All three guest FP flags are accepted; FP16_OVFL has no effect on these bounded results. Exact
// semantics preserve host FP state. Loose SIMD requires host nearest-even with denormals enabled.
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
// quiets signaling NaNs in exact semantics on gfx12; gfx11 quiets only with IEEE set.
// Guest input flushing is supported.
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
// Guest ROUND controls FRACT narrowing and integral OMOD overflow; TRANS
// overflow ignores ROUND (FP16_OVFL still saturates). Fixed-rounding
// integral instructions retain their named rounding rule for the core operation.
// Supports DPP8/DPP16 on A. Loose semantics require host nearest-even arithmetic
// with denormals enabled. RCP, RSQ, SQRT, EXP, LOG and FRACT round to FP16 before OMOD.
// All forms except FRACT support exact results. EXP/LOG use rocjitsu's FP64 libm
// model with integer narrowing and explicit NaN policy; no host controls are changed.
// All forms accept
// guest input/output-denormal flags and IEEE. EXP/LOG flush rounded half results
// before OMOD when output flushing is requested. APIs without _gfx12 select
// RDNA3: OMOD is disabled when IEEE is set or output denormals are preserved.
// RDNA4 OMOD is independent of these modes. Exact RDNA3 unary operations retain
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

// FREXP flushes half subnormal inputs when guest input flushing is enabled.
// Exact NaNs retain their sign/payload; gfx11 preserves signaling NaNs with
// IEEE clear, while gfx12 quiets them. Output modifiers apply afterward.
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
// Guest ROUND controls FRACT subtraction and integral OMOD overflow; TRANS
// output scaling overflows to infinity independently of ROUND.
// TRUNC/CEIL/RNDNE/FLOOR accept guest IEEE and denormal flags. Their RDNA3
// forms suppress OMOD when IEEE mode is enabled or output denormals are
// preserved; _gfx12 forms scale independently. Guest input flushing applies
// before rounding. Both architectures implement exact numerical results with
// signaling NaNs preserved on RDNA3 with IEEE clear, and quieted otherwise.
// Mode 3 FP status reporting is unsupported on both
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

// gfx942/gfx950 integral rounding and mantissa extraction share the existing
// integer models, but always quiet NaNs independently of MODE.IEEE. OMOD remains
// enabled only with IEEE clear and output denormals flushed. All non-reporting
// semantics share the same arithmetic and SIMD paths; FP-status reporting is
// unsupported. D may alias A; inactive lanes and the unselected FP16 half survive.
int rj_arith_v_trunc_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_trunc_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_ceil_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_ceil_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_floor_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_floor_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_rndne_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_rndne_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_frexp_mant_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_frexp_mant_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_trunc_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_trunc_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_ceil_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_ceil_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_floor_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_floor_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_rndne_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_rndne_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_frexp_mant_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_frexp_mant_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts);

// gfx942/gfx950 fractional part retains guest rounding and always quiets NaNs.
// One VGPR per operand; FP16 HIGH_A/HIGH_D select halves. D may alias A.
// Compatibility uses the inherited arithmetic without promising empirical exactness;
// strict empirical semantics and FP-status reporting remain unsupported.
int rj_arith_v_fract_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_fract_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_fract_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);
int rj_arith_v_fract_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts);

// Inherited CDNA FP32 math models, also used for gfx950 through the gfx942 APIs.
// Compatibility preserves Rocjitsu's existing numeric and FP-status policies;
// strict empirical semantics are unsupported pending validation on these architectures.
// Loose and allow-Loose requests retain the corresponding existing math paths.
// Each operand occupies one VGPR. Controls and DPP follow the corresponding
// gfx11 (gfx942/gfx950) or gfx12 (gfx1250) math entry point. RCP_IFLAG instead
// retains the CDNA legacy rule: always quiet NaNs; in Compatibility, signed zero
// raises INT_DIV0 even with CLAMP, while subnormal inputs do not raise it.
int rj_arith_v_rcp_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);
int rj_arith_v_rcp_iflag_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_iflag_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_iflag_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *excp_flag_user);
int rj_arith_v_rsq_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rsq_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rsq_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);
int rj_arith_v_sqrt_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sqrt_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sqrt_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);
int rj_arith_v_exp_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_exp_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_exp_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);
int rj_arith_v_log_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_log_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_log_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);
int rj_arith_v_sin_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sin_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sin_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);
int rj_arith_v_cos_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_cos_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_cos_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

// Inherited CDNA FP16 math; gfx950 shares the gfx942 entry points.
// Compatibility retains the legacy numeric model without reporting FP status.
// Strict empirical modes are unsupported pending validation on these architectures.
// One VGPR per operand; HIGH_A/HIGH_D select halves, preserving the other destination half.
// Controls and DPP follow the corresponding gfx11 (gfx942/gfx950) or gfx12 (gfx1250) APIs.
int rj_arith_v_rcp_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rcp_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_rsq_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rsq_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_rsq_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_sqrt_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sqrt_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sqrt_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user);

int rj_arith_v_exp_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_exp_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_exp_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_log_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_log_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_log_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_sin_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sin_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_sin_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

int rj_arith_v_cos_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_cos_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts);
int rj_arith_v_cos_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
