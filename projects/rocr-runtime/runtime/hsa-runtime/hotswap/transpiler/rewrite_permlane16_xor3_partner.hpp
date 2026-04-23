// Post-PromoteMemToReg pass that rewrites the Triton-on-gfx1250
// cross-16 bitonic-merge idiom.
//
// STATUS: TRANSITIONAL (2026-04-22).  Remove when either of the
// two conditions below is met — see the block comment below for
// the full justification.
//
//   (a) AMD publishes gfx1250 ISA documentation confirming the
//       silicon semantic of `v_permlane16_swap_b32` under the
//       `vdst_in == src0_in` initializer.  If the silicon
//       differs from the gfx950-documented cross-wire (produces
//       `partner` instead of `self` after the xor3), move the
//       fix UP the stack: update `emitPermLaneSwapEmulation` in
//       `handle_valu_cross_lane.cpp` to model the gfx1250
//       silicon semantic.  This pass then becomes dead code
//       (harmless — it'll still match, substituting an already-
//       correct partner value for itself).
//
//   (b) Triton's gfx1250 codegen changes to stop emitting this
//       idiom (e.g. switches to `ds_swizzle_b32 swap:16`, which
//       gfx942 codegen already uses).  The fingerprint stops
//       appearing in lifted IR and the pass becomes dead code.
//
// Under either condition we also delete the `--enable-/--disable-
// permlane16-xor3-partner` raise_cli flag and the
// `enablePermLane16Xor3PartnerRewrite` parameter threaded through
// `raiser.hpp` + `pipeline.hpp`.
//
// Regression pins:
//   * `Gfx1250Gpu.BitonicXor3TritonState` — hardware probe whose
//     per-lane summary flips from "32 self" (pre-rewrite) to
//     "32 partner" (post-rewrite).
//   * `canary_tl_sort_fp32_deterministic` — end-to-end sort
//     canary that graduates WRONG 16384/16384 → MATCH under
//     the rewrite.
//   * `Gfx1250Gpu.Permlane16Swap{,Wave32,Wave32WaveNative}` and
//     `Gfx1250Gpu.BitonicCross16Probe` — pin that the swap's
//     standalone semantic is unaffected (these probes' inputs
//     are distinct enough to NOT trigger the idiom).
//
// Background
// ==========
//
// Triton's `tl.sort` / `tl.topk` codegen for gfx1250 emits the
// following 4-instruction sequence at the cross-16 stage of a
// bitonic merge:
//
//   v_dual_mov_b32 v_a, v_c :: v_dual_mov_b32 v_b, v_c
//   v_permlane16_swap_b32 v_a, v_b
//   v_xor3_b32           v_a, v_a, v_b, v_c
//
// The ISA-documented (gfx950-style) semantics of
// `v_permlane16_swap_b32` cross-wire the two outputs:
//   new_vdst[L]      = src0_in[L XOR 16]
//   new_src0_out[L]  = vdst_in[L XOR 16]
// Under Triton's `v_a_in == v_b_in == v_c` initialiser, both
// outputs end up holding `partner_v_c`.  The subsequent xor3
// `v_a = v_a ^ v_b ^ v_c = partner ^ partner ^ self = self`
// LITERALLY collapses v_a back to `self_v_c` on a faithful
// register-machine evaluation.
//
// Salmon faithfully reproduces that literal evaluation via the
// `emitPermLaneSwapEmulation` ds_bpermute pair plus the standard
// `V_XOR3_B32` 3-way xor lift.  The `BitonicXor3TritonState` GTest
// confirms — under WaveNativeProjection AND under
// ModuloReplicationProjection — that v_a post-xor3 is `self_v_c`
// for every lane.  Yet the gfx942-NATIVE Triton compile of the
// same Python source produces a correctly-sorted output, which
// rules out "the algorithm doesn't need a swap here".  Triton's
// gfx942 codegen uses `ds_swizzle_b32 swap:16` to put partner
// directly in a register, then a plain `v_cmp_gt_f32` against
// self — clean.  The gfx1250 codegen uses the
// `v_permlane16_swap + v_xor3` pair instead.
//
// We have no gfx1250 hardware to verify whether the gfx1250
// silicon's `v_permlane16_swap_b32` truly cross-wires (matching
// gfx950) or has a different semantic that makes the xor3
// produce `partner` instead of `self`.  Two possibilities:
//
//   (a) gfx1250 silicon's swap differs from gfx950's — the xor3
//       genuinely yields `partner_v_c`, and Triton's codegen is
//       correct.  Our salmon emulation (which mirrors gfx950)
//       produces `self_v_c`, so the lift is the bug.
//
//   (b) gfx1250 silicon matches gfx950, the xor3 really does
//       yield `self_v_c`, and Triton's gfx1250 codegen has a bug.
//       In that case salmon is faithful but reproduces a Triton
//       bug.
//
// In BOTH cases the right thing for salmon to do is detect this
// idiom and substitute `partner_v_c` for the xor3 result — that's
// what the sort algorithm needs to make the surrounding
// `v_cmp_*` and `cndmask` chain swap correctly.  Under (a) the
// fix matches the silicon.  Under (b) the fix produces a result
// that matches the gfx942-native compile (the only ground truth
// we have access to without gfx1250 hardware) and "papers over"
// Triton's bug in a way that's principled at the salmon-lift
// boundary (we're emitting what the algorithm intends, not what
// the literal bytes encode).
//
// Pattern (after PromoteMemToReg)
// ===============================
//
// `emitPermLaneSwapEmulation` plus `V_XOR3_B32` produce, after
// alloca-promotion:
//
//   %addr  = ...                        ; (lane_id ^ 16) << 2
//   %seed  = ...                        ; v_c — same SSA across all
//                                       ; three xor inputs at
//                                       ; this site
//   %bp_a  = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr, i32 %seed)
//   %bp_b  = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr, i32 %seed)
//   %xor3a = xor i32 %bp_a, %bp_b
//   %xor3b = xor i32 %xor3a, %seed     ; named %vxor3 in salmon
//
// The pass walks every `xor i32 %xor3a, %seed` whose first
// operand is `xor i32 %bp_a, %bp_b` with both bpermutes sharing
// the same address argument and both bpermutes' data argument
// equal to `%seed`.  When matched, RAUWs %xor3b with %bp_a
// (which holds `partner_seed`).
//
// Conservative match
// ------------------
//
// The match requires structural identity (same SSA values, no
// looking through arbitrary computation).  This makes false
// matches outside the Triton idiom essentially impossible —
// the conjunction of "two ds.bpermute calls with identical
// operands" plus "outer xor with the seed" plus "the inner xor
// is exactly those two bpermutes" is a fingerprint salmon would
// not emit through any other handler.  The
// `BitonicXor3TritonState` GTest pins the rewrite; the
// `Permlane16Swap*` GTests pin that the swap's standalone
// semantic is unaffected (their xor3-less inputs don't trigger
// the rewrite).

#pragma once

#include "llvm/IR/Function.h"

namespace transpiler {

struct Permlane16Xor3PartnerRewriteReport {
  unsigned matchedSites = 0;
};

Permlane16Xor3PartnerRewriteReport
rewritePermLane16Xor3Partner(llvm::Function &F);

} // namespace transpiler
