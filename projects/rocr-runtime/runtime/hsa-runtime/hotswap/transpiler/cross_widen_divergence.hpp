#ifndef HOTSWAP_TRANSPILER_CROSS_WIDEN_DIVERGENCE_HPP
#define HOTSWAP_TRANSPILER_CROSS_WIDEN_DIVERGENCE_HPP

#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {
class Function;
class Value;
} // namespace llvm

namespace transpiler {

// ============================================================================
// Cross-widen divergence oracle.
// ============================================================================
//
// Identifies SSA values that carry different per-lane values within a
// single target wave under cross-widening (source wave32 -> target
// wave64 via `WaveNativeProjection` or `ModuloReplicationProjection`).
// Consumed by the writelane/readlane rewrite pass
// (rewrite_cross_lane_divergent.cpp) to decide whether a scalar
// operand entering `v_writelane_b32` / `v_readlane_b32` would be
// implicitly scalarised by the backend's `v_readfirstlane_b32`,
// collapsing per-source-wave distinctions. See
// hotswap/docs/wave-size-translation.md §5.6.3 for the rewrite
// contract this oracle underwrites.
//
// Contract: sound, over-approximating.
//   * Any value transitively derived from a known-divergent leaf is
//     marked divergent.
//   * Arithmetic / logical / compare / select / phi / cast / freeze
//     / gep / extract / insert / trunc are divergence-transparent
//     (divergent iff any operand is).
//   * A `load` is divergent iff its pointer is divergent (standard
//     LLVM divergence rule). Scratch-space loads are expected to be
//     gone post-mem2reg; if any survive they come in through a
//     `workitem.id.x`-derived address and the pointer-divergence rule
//     catches them.
//   * A `store` is not a producer, so it is never in the divergent
//     set.
//   * Function arguments are uniform (kernel args are workgroup-
//     uniform in the AMDGPU ABI; the raiser does not synthesise
//     per-lane function parameters).
//   * Constants / globals are uniform by definition.
//
// Known-divergent leaves:
//   * `llvm.amdgcn.workitem.id.{x,y,z}`
//   * `llvm.amdgcn.mbcnt.{lo,hi}`
//   * `llvm.amdgcn.ds.bpermute` / `ds.permute` / `ds.swizzle`
//   * `llvm.amdgcn.permlane{16,x16,16.swap,32.swap,64}`
//   * `llvm.amdgcn.update.dpp` (any variant)
//   * `llvm.amdgcn.ballot.{i32,i64}`
//   * `llvm.amdgcn.readlane(src, lane)` — iff `src` is divergent
//     (per-lane data traversing a lane selector becomes divergent).
//   * `llvm.amdgcn.writelane(val, lane, old)` result — iff either
//     `val` or `old` is divergent (non-selected lanes retain `old`'s
//     per-lane value; selected lane carries `val`'s per-lane value).
//   * `llvm.amdgcn.readfirstlane` is always uniform (that is its
//     purpose) — treated as a divergence sink.
//
// False positives (a uniform value flagged divergent) are benign:
// the rewrite pass will emit an extra `select` or `ds_bpermute`
// that computes a correct result but costs a handful of extra
// instructions. False negatives (a divergent value missed) silently
// miscompile on entry to any cross-lane primitive, so the oracle
// biases conservatively — unknown intrinsics / operand shapes are
// treated as divergence-transparent and propagate through.

class CrossWidenDivergenceAnalysis {
 public:
  // Build the cache for `F`. Walks each instruction in program order;
  // iterates to fixed point (typically 2-3 passes on matmul-scale
  // kernels).
  explicit CrossWidenDivergenceAnalysis(llvm::Function &F);

  // Returns true iff `V` is cross-widen-divergent within the function
  // passed at construction time. `nullptr` -> false (safe default for
  // callers that may feed a trivially-uniform operand).
  bool isDivergent(const llvm::Value *V) const;

 private:
  // Pointer set keyed on the original SSA value. LLVM caps the
  // SmallPtrSet inline buffer at 32 entries; the set grows to the
  // heap beyond that (still O(1) amortised membership).
  llvm::SmallPtrSet<const llvm::Value *, 32> divergent;

  void seedLeaves(llvm::Function &F);
  void propagate(llvm::Function &F);
};

} // namespace transpiler

#endif
