// ARCH: gfx1250
//
// Regression fixture for the VOPD `v_dual_cndmask_b32` SGPR-condition
// bug (hotswap/docs/modrep-predicate-chain.md §9.7, 2026-04-22 fix).
//
// PRE-FIX BUG. `handle_vopd.cpp::v_cndmask_b32` hardcoded
// `ctx.regs.loadVCC(ctx.B)` as the condition for BOTH halves of a
// `v_dual_cndmask_b32 ... :: v_dual_cndmask_b32 ...` pair, ignoring
// the explicit scalar condition operand gfx1250's VOPD encoding
// carries. The mis-lift was invisible when both halves read VCC, and
// silently miscompiled whenever the two halves took DIFFERENT scalar
// conditions — precisely the shape Triton's `tl.cumsum` Kogge-Stone
// scan emits at distances 8 and 16:
//
//   v_dual_cndmask_b32 v9, v9, v1, s0           (stage-3 selector
//                                                advance: s0 = tid < 8)
//   :: v_dual_cndmask_b32 v2, v2, v7, vcc_lo    (stage-2 value update:
//                                                vcc = tid > 3)
//
// Hardcoding VCC for the first half inherited `(tid > 3)` from the
// paired V_CMP_LT writer, making lanes 8..31 pick "self" instead of
// "lane - 8" on the stage-3 bpermute selector. The scan's fadd then
// doubled those lanes' stage-2 partial sums (observed: salmon
// output[8] = 2 × (native[8] − native[0]) = 3.4694, native = 1.4988,
// delta 1.9706). Flagged end-to-end as `canary_bpermute_scan_fp32`
// WRONG 4/4 AND `corpus_layernorm_fp32` WRONG on N=128/256/512 (same
// bug on LayerNorm's mean-of-squares reduction).
//
// POST-FIX. The handler now parses `operands[3]` for each half and
// dispatches:
//
//   * `vcc_lo` / `vcc` → `loadVCC` (the old default, now explicit).
//   * `sN` → prefer the fresh per-BB V_CMP shadow `i1` from
//     `lookupSgprWaveMaskI1(N)`; fall back to the projection's
//     `extractLaneBitFromWaveMask` on the raw SGPR alloca.
//   * absent / unrecognised → `loadVCC` (safety).
//
// This mirrors the non-VOPD `V_CNDMASK_B32` handler in
// `handle_valu_vop3p.cpp` — the two handlers now share the exact
// same SGPR-condition semantics.
//
// The fixture forces a paired `v_dual_cndmask_b32 ..., s0 ::
// v_dual_cndmask_b32 ..., vcc_lo` shape via inline asm, so the
// assembler emits a real VOPD encoding (rather than two single-
// issue VOPs the auto-pairer might or might not bundle). Both
// halves consume a value derived from a preceding `v_cmp_*_e64`
// that writes `s0` (fresh-shadow path) and a separate
// `v_cmp_*_e32 vcc_lo` (VCC path) so both handler branches fire
// in a single kernel.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_dual_cndmask_b32_sgpr_cond_kernel(int *out,
                                                                const int *in) {
  int lane = (int)threadIdx.x;
  int val_a = in[lane];
  int val_b = in[(lane + 1) & 31];
  int val_c = in[(lane + 2) & 31];
  int val_d = in[(lane + 3) & 31];

  // Set up the two scalar conditions:
  //
  //   * `s0` via `v_cmp_lt_u32_e64 s0, 8, tid` — (tid > 8) into an
  //     explicit SGPR. This is the VOPD-CNDMASK SGPR-condition
  //     operand shape the fix targets.
  //   * `vcc_lo` via `v_cmp_lt_u32_e32 vcc_lo, 3, tid` — (tid > 3)
  //     into the implicit VCC register. This is the VOPD-CNDMASK
  //     VCC-condition operand shape that always worked.
  //
  // The VOPD pair then consumes both: first half guarded by s0,
  // second half guarded by vcc_lo. Pre-fix the handler ignored the
  // SGPR operand and the first half silently read VCC = (tid > 3),
  // reproducing the `canary_bpermute_scan_fp32` miscompile shape.
  // Post-fix the first half correctly reads s0 = (tid > 8) and the
  // two halves diverge on lanes 4..8 (where tid > 3 but tid <= 8) —
  // pinned below as a per-lane output asymmetry.
  int first_half = 0;
  int second_half = 0;
  asm volatile(
      // Set up s0 = (tid > 8).
      "v_cmp_lt_u32_e64 s0, 8, %[tid]\n\t"
      // Set up vcc_lo = (tid > 3).
      "v_cmp_lt_u32_e32 vcc_lo, 3, %[tid]\n\t"
      // VOPD: paired cndmask with DIFFERENT scalar conditions.
      "v_dual_cndmask_b32 %[d0], %[a], %[b], s0 :: "
      "v_dual_cndmask_b32 %[d1], %[c], %[d], vcc_lo\n\t"
      : [d0] "=v"(first_half), [d1] "=v"(second_half)
      : [tid] "v"(lane),
        [a] "v"(val_a), [b] "v"(val_b),
        [c] "v"(val_c), [d] "v"(val_d)
      : "s0", "vcc_lo");

  out[lane * 2 + 0] = first_half;
  out[lane * 2 + 1] = second_half;
}
