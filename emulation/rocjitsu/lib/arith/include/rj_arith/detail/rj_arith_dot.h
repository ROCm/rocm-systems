// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_DOT_H_
#define RJ_ARITH_DOT_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// DPP8/DPP16 permute the complete A word before modifiers; B/C stay in their
// original lanes.
// True16 DOT2: A/B each hold two packed factors; C supplies one selected half.
// D replaces only its selected half, preserving the other half. Supports all
// six RJ_ARITH_ALU ABS/NEG flags and HIGH_C/HIGH_D, without OMOD or CLAMP.
// Loose semantics use FP32 products and sums followed by nearest-even narrowing.
// Both architectures use the same numerical implementation. F16 preserves
// denormals and honors RJ_ARITH_FP16_OVFL; BF16 flushes input/output denormals and
// ignores that flag. Guest input/output-denormal and IEEE flags are accepted
// without changing this fixed policy. BF16 requires host nearest-even with
// denormals preserved. Host controls are never changed, but
// ordinary arithmetic may raise sticky flags. NaN payloads are unspecified.
// RDNA3 APIs take trapsts; _gfx12 APIs take excp_flag_user. Loose
// calls leave FP status unchanged. Required exact semantics and FP status
// reporting remain unsupported.
int rj_arith_v_dot2_f16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *trapsts);

int rj_arith_v_dot2_f16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_dot2_bf16_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *trapsts);

int rj_arith_v_dot2_bf16_bf16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *excp_flag_user);

// Wave64 retains the same packed inputs, modifiers and loose numerical contract.
int rj_arith_v_dot2_f16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *trapsts);

int rj_arith_v_dot2_f16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);
int rj_arith_v_dot2_bf16_bf16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *trapsts);

int rj_arith_v_dot2_bf16_bf16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *excp_flag_user);

// Floating DOT2 sign modifiers act after selecting each packed half.
static const uint32_t RJ_ARITH_DOT_NEG_LO_A = 1U << 0;
static const uint32_t RJ_ARITH_DOT_NEG_LO_B = 1U << 1;
static const uint32_t RJ_ARITH_DOT_NEG_C = 1U << 2;
static const uint32_t RJ_ARITH_DOT_NEG_HI_A = 1U << 3;
static const uint32_t RJ_ARITH_DOT_NEG_HI_B = 1U << 4;

// Floating DOT2 half selection: defaults are low for term 0 and high for term 1.
// These flags override those defaults; unlike raw op_sel_hi, zero means default.
static const uint32_t RJ_ARITH_DOT_LO_A_HIGH = 1U << 7;
static const uint32_t RJ_ARITH_DOT_LO_B_HIGH = 1U << 8;
static const uint32_t RJ_ARITH_DOT_HI_A_LOW = 1U << 9;
static const uint32_t RJ_ARITH_DOT_HI_B_LOW = 1U << 10;

// DPP8/DPP16 permute the complete A word before sign and half selection;
// B/C remain in their original lanes. Supported in loose and exact semantics.
// RDNA4 DOT2: A/B each hold two packed 16-bit factors in one VGPR;
// C/D each hold one FP32 value per lane. Loose and exact modes are supported.
// Supports all floating DOT2 sign/half-selection flags. RJ_ARITH_DOT_CLAMP is
// accepted but has no effect on these floating DOT2 forms (as in rocjitsu).
int rj_arith_v_dot2_f32_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_dot2_f32_bf16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

// RDNA3 DOT2 has the same register layout, sign/half selection and ignored
// CLAMP, but different empirical accumulation. Exact result bits are supported;
// mode 3 requests return UNSUPPORTED_FP_STATUS.
// Loose semantics share the RDNA4 SIMD path and leave optional state unchanged.
int rj_arith_v_dot2_f32_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *trapsts);

int rj_arith_v_dot2_f32_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);

// RDNA3 DOT2ACC adds the packed FP16 dot product to old D (one FP32 VGPR).
// Supports DPP8/DPP16 routing of A, but no sign or half-selection modifiers.
// Exact arithmetic and reporting restrictions match RDNA3 DOT2 above. Whole
// VGPR aliasing is allowed; all results use the original accumulator/source bits.
int rj_arith_v_dot2acc_f32_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts);

// Wave64 DOT2/DOT2ACC: same layouts, modifiers and architectural semantics
// as above, with 64 lanes and four independent DPP16 row enables.
int rj_arith_v_dot2_f32_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_dot2_f32_bf16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_dot2_f32_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *trapsts);

int rj_arith_v_dot2_f32_bf16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts);

int rj_arith_v_dot2acc_f32_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts);

// FP8/BF8 DOT4 accepts NEG_C and ABS_C; ABS precedes NEG. A/B modifiers,
// half selection, output scaling and CLAMP are not supported.
static const uint32_t RJ_ARITH_DOT_ABS_C = 1U << 5;

// Wave32 DOT4: one VGPR each for A/B/C/D, four packed bytes per A/B lane;
// C/D are FP32. FP8 is OCP E4M3FN, BF8 is OCP E5M2. Loose semantics only.
// Both wave widths support DPP8/DPP16 on A; B and C remain in their own lanes.
int rj_arith_v_dot4_f32_fp8_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_dot4_f32_fp8_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_dot4_f32_bf8_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_dot4_f32_bf8_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

// Wave64 retains the same packed inputs, modifiers and loose numerical contract.
int rj_arith_v_dot4_f32_fp8_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_dot4_f32_fp8_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_dot4_f32_bf8_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_dot4_f32_bf8_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

// Integer DOT modifiers: SIGNED selects signed factors for I32_IU forms.
// U32_U forms accept only CLAMP. CLAMP saturates the final accumulator to its
// signed/unsigned 32-bit range; otherwise arithmetic wraps modulo 2^32.
static const uint32_t RJ_ARITH_DOT_SIGNED_A = 1U << 0;
static const uint32_t RJ_ARITH_DOT_SIGNED_B = 1U << 1;
static const uint32_t RJ_ARITH_DOT_CLAMP = 1U << 6;

// Integer DOT wave32: one VGPR each for A/B/C/D. A/B contain four packed
// bytes or eight packed nibbles. C/D are signed for I32_IU, unsigned for U32_U.
// Loose and exact semantics return the same integer result.
int rj_arith_v_dot4_i32_iu8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_dot4_u32_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_dot8_i32_iu4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_dot8_u32_u4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

// Wave64 integer DOT uses the same packed operands, modifiers and exact
// arithmetic, with 64 lanes per VGPR. DPP is unsupported at both widths.
int rj_arith_v_dot4_i32_iu8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_dot4_u32_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_dot8_i32_iu4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_dot8_u32_u4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
