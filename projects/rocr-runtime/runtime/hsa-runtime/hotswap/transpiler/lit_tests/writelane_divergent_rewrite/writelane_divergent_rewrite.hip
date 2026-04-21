// ARCH: gfx1250
//
// Fixture for the cross-widen-divergent `v_writelane_b32` rewrite
// (lit_tests/writelane_divergent_rewrite). Pins the IR shape that
// `rewrite_cross_lane_divergent.cpp` emits when the rewrite pass's
// forward use-chain classifier clears a writelane site for symmetric
// rewriting under `gfx1250 → gfx942` cross-widening.
//
// The kernel deliberately exercises BFE-only + writelane — NO WMMA —
// so the `WaveIdLiftScalarized` classifier in
// `wave_size_obstruction.cpp::buildObstructionReport` does NOT refuse
// this kernel (the refusal fires only on the three-way co-occurrence
// BFE + writelane/readlane + WMMA; see
// `hotswap/docs/wave-size-translation.md` §6 Class 1 table). That
// lets the rewrite pass actually run and gives us a stable anchor
// for the rewritten IR.
//
// Two RUN lines cover the two sides of the `--enable-writelane-rewrite`
// flag's contract:
//
//   * REWRITE: flag on -> writelane is gone, replaced by a `select`
//     keyed on `mbcnt_{lo,hi}`-derived lane_id. Pins
//     `rewrite_cross_lane_divergent.cpp::rewriteWritelaneCall`.
//
//   * UNCHANGED: flag off -> `llvm.amdgcn.writelane` stays in place
//     (commit 1's "no behaviour change" contract). Pins the default-
//     off flag plumbing in `raiser.cpp` / `raise_cli.cpp`.

#include <hip/hip_runtime.h>

extern "C" __global__ void writelane_divergent_rewrite_kernel(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // (1) Canonical `wave_id_in_workgroup` extraction, bits [29:25] of
  //     ttmp8. `handle_sop2.cpp::handleSOP2` rescues this exact shape
  //     into `(workitem.id.x >> 5) & 0x1F`, so the destination SGPR
  //     carries a divergent VGPR-backed value in the raised IR. Under
  //     the symmetric-rewrite rule the rewrite pass rewrites every
  //     writelane regardless of operand divergence, but this fixture's
  //     writelane is divergent in both operands anyway so the pre-
  //     symmetry anchor points still hold.
  int wave_id;
  asm volatile("s_bfe_u32 %[wid], ttmp8, 0x50019\n\t"
               : [wid] "=s"(wave_id));

  // (2) `v_writelane_b32 vOUT, sDST, <imm lane>` consumes the divergent
  //     scalar as its val operand. Under cross-widening the backend
  //     would insert an implicit `v_readfirstlane_b32` on the SSA-scalar
  //     operand and collapse per-source-wave divergence, unless the
  //     rewrite pass converts the call to a per-lane `select` that
  //     keeps the value in a VGPR. The `=v` constraint on the output
  //     forces hipcc to emit the writelane shape verbatim; a `+v`
  //     tied output (register allocator preferred form) would let the
  //     front-end fold the op away.
  int lane_slot;
  asm volatile("v_writelane_b32 %[out], %[val], 0\n\t"
               : [out] "=v"(lane_slot)
               : [val] "s"(wave_id));

  // Keep both inline-asm outputs live so front-end DCE does not
  // strip either piece out of the compiled code object.
  out[tid] = wave_id ^ lane_slot;
}
