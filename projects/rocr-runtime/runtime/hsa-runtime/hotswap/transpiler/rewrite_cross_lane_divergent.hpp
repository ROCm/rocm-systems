#ifndef HOTSWAP_TRANSPILER_REWRITE_CROSS_LANE_DIVERGENT_HPP
#define HOTSWAP_TRANSPILER_REWRITE_CROSS_LANE_DIVERGENT_HPP

namespace llvm {
class Function;
} // namespace llvm

namespace transpiler {

// ============================================================================
// Post-raise rewrite: per-source-wave writelane/readlane under cross-
// widen divergence. See hotswap/docs/wave-size-translation.md §5.6.3
// for the principled derivation of the rewrite shapes below.
// ============================================================================
//
// PROBLEM. The AMDGPU backend lowers `llvm.amdgcn.writelane(val, lane,
// old)` / `llvm.amdgcn.readlane(src, lane)` through the
// `v_writelane_b32` / `v_readlane_b32` machine opcodes. Those opcodes
// accept the value / src as an *SGPR*, and the backend inserts an
// implicit `v_readfirstlane_b32` on any divergent-in-a-VGPR input to
// scalarise it. Under cross-widening (source wave32 -> target wave64),
// a value that is *intentionally* per-lane-divergent within a target
// wave — e.g. the `wave_id_in_workgroup` divergent-VGPR lifted out of
// `s_bfe_u32 ttmp8, 0x50019` (handle_sop2.cpp §5.6.2 rescue) — gets
// collapsed by the readfirstlane to one lane's value, and the other
// target lanes lose their per-source-wave meaning. Every downstream
// computation keyed on the collapsed value (tile-column address, EXEC
// predicate, scalar broadcast) then reads the wrong source wave's data,
// which miscompiles every Matmul128x128-class kernel on gfx1250 ->
// gfx942 translation.
//
// REWRITE. Replace the cross-lane primitive with a per-source-wave
// shape that keeps the scalar operand in a VGPR and preserves per-
// source-wave state across the target wave:
//
//   * `writelane(val_div, lane_idx, old)` ->
//       `select ((lane_id & (W_s-1)) == lane_idx), val_div, old`
//     Semantics: within each virtual source wave (each contiguous
//     W_s lanes of the target wave), the lane whose source-wave
//     relative index equals `lane_idx` receives `val_div`'s per-lane
//     value; the other W_s-1 lanes keep `old`'s per-lane value. This
//     matches the source kernel's single-wave writelane exactly
//     under modulo-replication / wave-native projection.
//
//   * `readlane(src_div, lane_idx)` ->
//       `ds_bpermute(selector = ((lane_id & ~(W_s-1)) | lane_idx) << 2,
//                    src_div)`
//     Semantics: broadcast lane `lane_idx` of the current source wave
//     to all lanes in that source wave. The selector `<< 2` multiplier
//     is `ds_bpermute`'s byte-to-lane offset convention
//     (IntrinsicsAMDGPU.td comment on `int_amdgcn_ds_bpermute`); the
//     base mask `~(W_s-1)` aligns the selector to the source-wave
//     boundary so each source wave broadcasts only within itself.
//
// UNIFORM SITES. When the scalar operand is provably uniform (common
// case — lane-index constants, kernel-arg SGPRs, tile-layout scalars),
// the rewrite is a no-op: the original `amdgcn.writelane` / `readlane`
// is kept verbatim and the backend's canonical
// `v_writelane_b32` / `v_readlane_b32` lowering fires unchanged. This
// is essential for corpus stability — wave-uniform writelane is the
// dominant shape in Tensile / rocBLAS and any conversion to
// `select` / `ds_bpermute` on that path would regress codegen
// quality without a correctness benefit.
//
// PRECONDITION. Run AFTER `PromoteMemToReg` so the divergence oracle
// operates on post-mem2reg SSA: scratch-addrspace allocas that would
// otherwise obscure divergence through a round-trip load/store are
// gone. The rewrite assumes `writelane` / `readlane` are overloaded on
// `i32` (the only shape the raiser emits today — see
// `handle_valu_cross_lane.cpp`'s `V_WRITELANE_B32` / `V_READLANE_B32`
// branches).
//
// DIRECTION GATE. Only runs when `targetWaveSize > sourceWaveSize`
// (cross-widening). Same-wave and narrowing directions leave every
// site unchanged — the implicit readfirstlane does not collapse
// per-source-wave state when a source wave *is* a target wave.

struct CrossLaneDivergentRewriteReport {
  // Number of `amdgcn.writelane` calls that were rewritten to a
  // per-lane `select` because their `val` or `old` operand was
  // cross-widen-divergent.
  unsigned writelaneRewritten = 0;

  // Number of `amdgcn.readlane` calls that were rewritten to a
  // `ds_bpermute` because their `src` operand was cross-widen-
  // divergent.
  unsigned readlaneRewritten = 0;

  // Number of writelane / readlane sites left untouched (operand
  // proven uniform). Reported for diagnostic symmetry with the
  // rewritten counts.
  unsigned uniformPreserved = 0;

  // Total sites rewritten. Callers use this to decide whether to
  // run the rewrite pass's safety-net abort (commit 2): when the
  // classifier flagged a `WaveIdLiftScalarized` site but the rewrite
  // rewrote nothing, the classifier's approximation disagrees with
  // the oracle and we should refuse rather than emit silently
  // unchanged IR.
  unsigned totalRewritten() const {
    return writelaneRewritten + readlaneRewritten;
  }
};

// Rewrite every cross-lane primitive site in `F` whose scalar operand
// is cross-widen-divergent. No-op when `targetWaveSize <=
// sourceWaveSize`.
CrossLaneDivergentRewriteReport rewriteCrossLaneDivergent(
    llvm::Function &F, unsigned sourceWaveSize, unsigned targetWaveSize);

} // namespace transpiler

#endif
