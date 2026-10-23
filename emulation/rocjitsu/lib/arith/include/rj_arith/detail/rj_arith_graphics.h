// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_GRAPHICS_H_
#define RJ_ARITH_GRAPHICS_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Interpolation wait-count field (0..7). Accepted for every interpolation
// instruction; it has no effect on synchronous CPU execution.
constexpr uint32_t RJ_ARITH_INTERP_WAIT_EXP_SHIFT = 13;
constexpr uint32_t RJ_ARITH_INTERP_WAIT_EXP_MASK = 7U << 13;

// Quad-local interpolation. For lane L and Q=L&~3, P10 computes
// fma(A[Q+1],B[L],C[Q]); P2 computes fma(A[Q+2],B[L],C[L]). Source lanes are
// read even if inactive in EXEC. Each operand uses one VGPR; whole VGPRs may
// alias. Supports NEG_A/B/C, CLAMP and WAIT_EXP, with no ABS or OMOD.
// Loose semantics only. Requires host nearest-even rounding and denormals
// enabled; host exception flags may change. NaN payloads are unspecified.
// Both architectures specify that interpolation reports no GPU exceptions.
int rj_arith_v_interp_p10_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_interp_p2_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

// Wave64 retains the same quad broadcasts, modifiers and loose numerical
// semantics. Source reads include inactive lanes within each quad.
int rj_arith_v_interp_p10_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_interp_p2_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

// Mixed FP16 interpolation uses the same quad broadcasts as the FP32 forms.
// A is FP16, B is FP32. P10 reads FP16 C and writes FP32 D; P2 reads FP32 C
// and writes FP16 D, preserving the other destination half. Supports NEG_A/B/C,
// CLAMP, WAIT_EXP and HIGH_A; also HIGH_C for P10 or HIGH_D for P2. No ABS/OMOD.
// RTZ forms round toward zero; other forms round nearest-even. RJ_ARITH_FP16_OVFL
// saturates finite P2 overflow; RTZ P2 already saturates finite overflow.
// Input infinities remain infinite unless clamped. Loose semantics only;
// host nearest-even rounding and enabled denormals are required. Host rounding
// mode is preserved; host exception flags may change. NaN payloads are unspecified.
// Both architectures specify that interpolation reports no GPU exceptions.
int rj_arith_v_interp_p10_f16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c);
int rj_arith_v_interp_p2_f16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_interp_p10_rtz_f16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c);
int rj_arith_v_interp_p2_rtz_f16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

// Wave64 retains the same quad broadcasts, modifiers and loose numerical
// semantics. Source reads include inactive lanes within each quad.
int rj_arith_v_interp_p10_f16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c);
int rj_arith_v_interp_p2_f16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_interp_p10_rtz_f16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c);
int rj_arith_v_interp_p2_rtz_f16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

// Lighting multiply. After ABS/NEG source modifiers, return -FLT_MAX if B is
// -FLT_MAX, -infinity or NaN, or C is nonpositive or NaN. Otherwise return +0
// if either factor is zero, or A*B. RDNA3 OMOD scales before flushing tiny
// results and signed zero to +0; it is disabled when output denormals are
// preserved or IEEE mode is set. RDNA4 OMOD flushes tiny unscaled values to
// +0 and tiny scaled values to signed zero. CLAMP maps to [0,1], including
// NaN/-0 to +0; RDNA3 RJ_ARITH_FP_DX10_CLAMP_OFF instead preserves a NaN result.
// Each operand uses one VGPR. Guest input/output flushing applies to all
// inputs/final results. IEEE mode and guest rounding flags are accepted; loose
// arithmetic uses host rounding rather than implementing guest-directed rounding. APIs without
// _gfx12 use RDNA3 TRAPSTS, _gfx12 uses EXCP_FLAG_USER; loose calls skip FP-status output,
// exact FP status reporting in mode 3 is unsupported. Supports all A/B/C ABS/NEG, OMOD and CLAMP
// modifiers; loose semantics only. DPP8/DPP16 permute A before source modifiers. B and C retain
// their lanes; inactive or DPP-filtered destinations remain unchanged. Host nearest-even rounding
// with denormals enabled is required; exception flags may change. NaN payloads are unspecified.
int rj_arith_v_mullit_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_mullit_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user);

// Wave64 retains the same operand layout, modifiers and numerical contract.
int rj_arith_v_mullit_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *trapsts);

int rj_arith_v_mullit_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user);

// Cube-map face ID, S/T coordinates and signed doubled major axis, respectively.
// A/B/C contain X/Y/Z in one VGPR each; D holds one VGPR. Z wins magnitude ties,
// then Y, then X. Comparisons flush subnormals; SC/TC copy selected source bits
// and preserve selected NaN bits on RDNA3 unless IEEE mode is enabled; RDNA4
// always quiets selected NaNs. A zero or unordered major axis is nonnegative.
// MA doubles the signed major axis with guest-directed overflow; round-down
// retains a negative zero major axis, while other rounding modes produce +0.
// DPP8/DPP16 permutes A before source modifiers; B/C retain their lanes.
// Supports all source ABS/NEG, OMOD and CLAMP. Nonzero OMOD flushes subnormal
// inputs/outputs, maps input zeros to +0 and preserves NaN sign/payload.
// Guest input/output denormal settings do not change those fixed rules.
// RDNA3 disables OMOD with IEEE enabled; RDNA4 applies OMOD independently.
// CLAMP converts NaNs to +0 by default. RDNA3 accepts RJ_ARITH_FP_DX10_CLAMP_OFF
// to preserve their sign/payload after the IEEE quieting policy. RDNA4 rejects
// that flag and always converts clamped NaNs to +0.
// Supports loose and empirical exact semantics. RJ_ARITH_FP_ROUND_MASK controls
// MA and OMOD overflow; neither modifies the host rounding mode.
// Full EXEC masking and all whole-register aliases are supported. Host FP state
// is preserved and does not affect results; RJ_ARITH_FP16_OVFL has no effect.
int rj_arith_v_cubeid_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubeid_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubesc_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubesc_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubetc_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubetc_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubema_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubema_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

// Wave64 retains the same exact selection/modifier rules and supports four
// independent DPP rows. Y and Z retain their original lanes.
int rj_arith_v_cubeid_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubeid_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_cubesc_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubesc_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_cubetc_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubetc_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_cubema_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_cubema_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
