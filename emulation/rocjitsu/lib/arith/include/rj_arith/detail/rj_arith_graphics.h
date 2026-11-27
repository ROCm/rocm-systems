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

// Prefetched ray/node operands for IMAGE_BVH_INTERSECT_RAY. Node types are
// tags 0..3 (triangles), 4 (FP16 boxes), 5 (FP32 boxes), 6/7 (not intersected).
// Direction and inverse are separate ISA operands; do not recompute either.
typedef struct rj_arith_bvh_ray {
  float origin[3], direction[3], inverse[3];
  float extent;
  uint32_t flags, node_type;
} rj_arith_bvh_ray;

typedef struct rj_arith_bvh_controls {
  uint32_t grow, sort, sort_enabled, barycentrics, pointer_flags;
} rj_arith_bvh_controls;

// Compute IMAGE_BVH/IMAGE_BVH64 intersection arithmetic from already fetched nodes.
// Pointer-width differences are resolved during fetching, outside this function.
// NODES holds one NODE_STRIDE-byte record per lane (at least 64 bytes, or 128
// for Box32). Unsupported node tags return invalid words. FETCH_MASK identifies successful fetches;
// other active lanes return four invalid words. EXEC selects destination writes. D uses four VGPRs
// and may overlap inputs; all operands are captured before writes. Controls are the decoded
// resource fields; grow is 0..255, sort is 0..3 and other fields are 0/1. Compatibility/Loose
// inherit rocjitsu's gfx11 model. Strict empirical modes are unsupported; guest FP controls do not
// apply and no FP status is reported.
int rj_arith_image_bvh_intersect_ray_w32(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const void *nodes, uint32_t node_stride,
                                         uint32_t fetch_mask, const rj_arith_bvh_ray *rays,
                                         const rj_arith_bvh_controls *controls);
int rj_arith_image_bvh_intersect_ray_w64(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const void *nodes, uint32_t node_stride,
                                         uint64_t fetch_mask, const rj_arith_bvh_ray *rays,
                                         const rj_arith_bvh_controls *controls);

// gfx10 interpolation after LDS fetching: A contains the selected coefficient,
// B the coordinate, and C the origin/previous accumulator. Each is one VGPR.
// MOV consumes only A. P1LL/P1LV decode FP16 A/C, P2_F16 decodes FP16 A and
// writes a selected FP16 half. All other inputs/results are FP32. HIGH_A/D
// select the shared FP16 source half and destination half respectively.
// ABS/NEG_B modify coordinates; ABS/NEG_C apply only to explicit P1LV/P2_F16
// addends and are ignored by the other stages. OMOD/CLAMP follow the instruction.
// Guest rounding and flushing use destination precision, except FP32 input
// flushing and the independent FP16 input override. P2_F16 ignores OMOD.
// Compatibility/Loose retain the inherited model; strict empirical modes are
// unsupported. No FP status is produced. Inactive lanes and the unselected
// destination half are preserved; inputs may overlap destinations.
int rj_arith_v_interp_mov_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
int rj_arith_v_interp_mov_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
int rj_arith_v_interp_p1_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);
int rj_arith_v_interp_p1_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);
int rj_arith_v_interp_p2_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);
int rj_arith_v_interp_p2_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);
int rj_arith_v_interp_p1ll_f16_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);
int rj_arith_v_interp_p1ll_f16_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);
int rj_arith_v_interp_p1lv_f16_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);
int rj_arith_v_interp_p1lv_f16_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);
int rj_arith_v_interp_p2_f16_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);
int rj_arith_v_interp_p2_f16_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);

// Quad-local interpolation. For lane L and Q=L&~3, P10 computes
// fma(A[Q+1],B[L],C[Q]); P2 computes fma(A[Q+2],B[L],C[L]). Source lanes are
// read even if inactive in EXEC. Each operand uses one VGPR; whole VGPRs may
// alias. Supports NEG_A/B/C, CLAMP, with no ABS or OMOD.
// Compatibility resolves to bit-exact semantics and reports no FP status. Every
// semantics shares one dispatch: without a guest rounding or denormal control the specialized
// kernels run and hand back only the lanes the guest model must produce, so Loose, Compatibility
// and bit-exact return the same bits at the same cost. All semantics require the
// default host FP environment (nearest-even rounding, denormals enabled); host exception flags may
// change. A selected NaN keeps its sign and payload. IEEE_MODE quiets FP32 results; DX10_CLAMP_OFF
// preserves NaNs under CLAMP. For RDNA4, set IEEE_MODE and leave DX10_CLAMP_OFF clear. Guest
// rounding and precision-specific input/output denormal controls are honored. WAIT_EXP belongs to
// the scheduler and is not an arithmetic modifier. Both architectures specify that interpolation
// reports no GPU exceptions.
int rj_arith_v_interp_p10_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_interp_p2_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

// Wave64 retains the quad broadcasts, modifiers and semantics of the Wave32
// forms above. Source reads include inactive lanes within each quad.
int rj_arith_v_interp_p10_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_interp_p2_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

// Mixed FP16 interpolation uses the same quad broadcasts as the FP32 forms.
// A is FP16, B is FP32. P10 reads FP16 C and writes FP32 D; P2 reads FP32 C
// and writes FP16 D, preserving the other destination half. Supports NEG_A/B/C,
// CLAMP and HIGH_A; also HIGH_C for P10 or HIGH_D for P2. No ABS/OMOD.
// RTZ forms override guest rounding toward zero; other forms use guest rounding.
// Guest denormal controls apply to both forms. RJ_ARITH_FP16_OVFL
// saturates finite P2 overflow; RTZ P2 already saturates finite overflow.
// Input infinities remain infinite unless clamped. Semantics, including the
// default host FP environment requirement, follow the FP32 forms. Host rounding
// mode is preserved; host exception flags may change. P10 uses the FP32 NaN
// controls above. P2 always quiets a NaN when narrowing to FP16 and honors DX10_CLAMP_OFF.
// Guest rounding and precision-specific input/output denormal controls are honored.
// WAIT_EXP belongs to the scheduler and is not an arithmetic modifier.
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

// Wave64 retains the quad broadcasts, modifiers and semantics of the Wave32
// forms above. Source reads include inactive lanes within each quad.
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
// arithmetic uses host rounding rather than implementing guest-directed rounding.
// Exact modes and Compatibility use the exact MUL backend for guest rounding, tininess,
// and OMOD after applying MULLIT source predicates; the gfx1100 and gfx1201 captures match
// this model, so Compatibility resolves to mode 2. FP status is still omitted. APIs without
// _gfx12 use RDNA3 TRAPSTS, _gfx12 uses EXCP_FLAG_USER; loose calls skip FP-status output,
// exact FP status reporting in mode 3 is unsupported. Supports all A/B/C ABS/NEG, OMOD and CLAMP
// modifiers. DPP8/DPP16 permute A before source modifiers. B and C retain
// their lanes; inactive or DPP-filtered destinations remain unchanged. Host nearest-even rounding
// with denormals enabled is required; exception flags may change. NaN payloads are unspecified.
// Loose may differ in intermediate flushing and signed zero under OMOD;
// exact modes retain the hardware-matching intermediate flush ordering.
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

// gfx1250 shares the gfx12 MULLIT model; strict exact is unsupported pending gfx1250
// captures. Compatibility keeps the inherited model.
int rj_arith_v_mullit_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);
int rj_arith_v_mullit_f32_w64_gfx1250(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

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

// Inherited older RDNA model; strict empirical qualification is unavailable.
int rj_arith_v_mullit_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user);

int rj_arith_v_mullit_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user);

int rj_arith_v_mullit_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

int rj_arith_v_mullit_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t *excp_flag_user);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
