#ifndef HOTSWAP_TRANSPILER_REWRITE_CROSS_LANE_DIVERGENT_HPP
#define HOTSWAP_TRANSPILER_REWRITE_CROSS_LANE_DIVERGENT_HPP

#include <string>

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
//   * `writelane(val, lane_idx, old)` ->
//       `select ((lane_id & (W_s-1)) == lane_idx), val, old`
//     Semantics: within each virtual source wave (each contiguous
//     W_s lanes of the target wave), the lane whose source-wave
//     relative index equals `lane_idx` receives `val`'s per-lane
//     value; the other W_s-1 lanes keep `old`'s per-lane value. This
//     matches the source kernel's single-wave writelane exactly
//     under modulo-replication / wave-native projection.
//
//   * `readlane(src, lane_idx)` ->
//       `ds_bpermute(selector = ((lane_id & ~(W_s-1)) | lane_idx) << 2,
//                    src)`
//     Semantics: broadcast lane `lane_idx` of the current source wave
//     to all lanes in that source wave. The selector `<< 2` multiplier
//     is `ds_bpermute`'s byte-to-lane offset convention
//     (IntrinsicsAMDGPU.td comment on `int_amdgcn_ds_bpermute`); the
//     base mask `~(W_s-1)` aligns the selector to the source-wave
//     boundary so each source wave broadcasts only within itself.
//
// WRITELANE / READLANE SYMMETRY. ALL writelane and ALL readlane sites
// are rewritten under cross-widening, independent of whether operands
// are divergent or uniform. The `select` / `ds_bpermute` forms are
// semantically equivalent to the source opcodes for every combination
// of operand divergence (uniform operands agree on lanes `N` and
// `N + W_s`; divergent operands carry their per-source-wave split),
// and unconditional rewriting is the only rule that keeps a
// writelane/readlane pair on a shared VGPR self-consistent.
//
// Asymmetric rewriting (the pre-symmetry rule that preserved
// "uniform" sites as native `v_writelane_b32` / `v_readlane_b32`) is
// silently unsound: a preserved uniform writelane writes hardware
// lane `N` only, leaving lane `N + W_s` at its prior value (undef for
// a fresh spill slot), while a sibling `ds_bpermute`-rewritten
// readlane on the same VGPR reads BOTH `N` and `N + W_s`. The upper
// replica returns undef, which flows into downstream address math and
// faults the kernel with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
// (observed on the Gfx1250Gpu.Matmul128x128* gtests after graduation;
// see hotswap/docs/learnings.md).
//
// USE-CHAIN CONSTRAINT. The rewrite is correct for any kernel whose
// writelane / readlane results flow only into VGPR-safe consumers
// (global / scratch / LDS memory ops, VALU arithmetic, other
// rewritten cross-lane primitives, branches, etc.). If any site's
// result transitively reaches an SGPR-constrained operand —
// `llvm.amdgcn.s.buffer.load` rsrc, `llvm.amdgcn.s.sendmsg` message,
// an explicit `llvm.amdgcn.readfirstlane`, a `load` from
// addrspace(4), inline asm with `"s"` constraint, or any unknown
// sink the classifier cannot prove safe — the backend will insert
// `v_readfirstlane` on the `ds_bpermute` result and re-introduce the
// cross-source-wave collapse that the rewrite was designed to
// eliminate. We detect this pre-rewrite via `classifyUseChain`
// (internal to `rewrite_cross_lane_divergent.cpp`) and refuse the
// whole function by returning a populated
// `refusal.sgprForcedDetail`. The refusal is principled per the
// project's no-silent-miscompile contract: if we cannot prove every
// consumer is VGPR-safe, we do not emit a best-effort rewrite.
//
// PRECONDITION. Run AFTER `PromoteMemToReg` so the use-chain
// classifier operates on post-mem2reg SSA: scratch-addrspace allocas
// that would otherwise obscure a use chain through a round-trip
// load/store are gone. The rewrite assumes `writelane` / `readlane`
// are overloaded on `i32` (the only shape the raiser emits today —
// see `handle_valu_cross_lane.cpp`'s `V_WRITELANE_B32` /
// `V_READLANE_B32` branches).
//
// DIRECTION GATE. Only runs when `targetWaveSize > sourceWaveSize`
// (cross-widening). Same-wave and narrowing directions leave every
// site unchanged — the implicit readfirstlane does not collapse
// per-source-wave state when a source wave *is* a target wave.

struct CrossLaneDivergentRewriteReport {
  // Number of `amdgcn.writelane` calls rewritten to a per-lane
  // `select`. Under cross-widening with a VGPR-safe use chain this
  // equals the total number of writelane sites in the function.
  unsigned writelaneRewritten = 0;

  // Number of `amdgcn.readlane` calls rewritten to a `ds_bpermute`.
  // Under cross-widening with a VGPR-safe use chain this equals the
  // total number of readlane sites in the function.
  unsigned readlaneRewritten = 0;

  // Non-empty iff the use-chain classifier rejected at least one
  // writelane or readlane site because a transitive consumer requires
  // an SGPR (e.g. `s_buffer_load` rsrc, `s_sendmsg` message,
  // `readfirstlane`, addrspace(4) load, inline asm `"s"`, or any
  // unknown sink). When populated: no sites were rewritten (refusal
  // is all-or-nothing to avoid the asymmetric-rewrite trap), the
  // raiser translates this into a
  // `RaiseFailure::crossWaveRewriteOracleDisagreement` refusal, and
  // the detail string names the offending intrinsic / memory op so
  // the next investigation knows where to start.
  std::string sgprForcedDetail;

  // Total sites rewritten. Callers use this only for diagnostic
  // formatting — under the symmetry contract every reachable site is
  // rewritten (or the whole function refuses), so `totalRewritten()
  // == 0` on a non-empty-site kernel always implies
  // `!sgprForcedDetail.empty()`.
  unsigned totalRewritten() const {
    return writelaneRewritten + readlaneRewritten;
  }

  // True iff the classifier refused. When true, the rewrite pass
  // performed zero rewrites and the caller must surface
  // `sgprForcedDetail` as a raise-time refusal diagnostic.
  bool refusedSgprForced() const { return !sgprForcedDetail.empty(); }
};

// Rewrite every cross-lane primitive site in `F` under cross-widening
// IF the forward use-chain classifier proves every site VGPR-safe.
// Otherwise perform zero rewrites and return the refusal detail in
// `sgprForcedDetail`. No-op when `targetWaveSize <= sourceWaveSize`.
CrossLaneDivergentRewriteReport rewriteCrossLaneDivergent(
    llvm::Function &F, unsigned sourceWaveSize, unsigned targetWaveSize);

} // namespace transpiler

#endif
