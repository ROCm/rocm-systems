#include "cross_widen_divergence.hpp"

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

namespace transpiler {

namespace {

// Is `id` a known source of per-lane divergence at the target-hardware
// level? These intrinsics introduce a value that *must* differ between
// at least two lanes of the same wave at runtime; any value derived
// from their output is divergent-or-worse. See the header for the
// rationale and the wave-size-translation.md §5.6.3 cross-ref.
bool isDivergentLeafIntrinsic(Intrinsic::ID id) {
  switch (id) {
  case Intrinsic::amdgcn_workitem_id_x:
  case Intrinsic::amdgcn_workitem_id_y:
  case Intrinsic::amdgcn_workitem_id_z:
  case Intrinsic::amdgcn_mbcnt_lo:
  case Intrinsic::amdgcn_mbcnt_hi:
  case Intrinsic::amdgcn_ds_bpermute:
  case Intrinsic::amdgcn_ds_permute:
  case Intrinsic::amdgcn_ds_swizzle:
  case Intrinsic::amdgcn_permlane16:
  case Intrinsic::amdgcn_permlanex16:
  case Intrinsic::amdgcn_permlane16_swap:
  case Intrinsic::amdgcn_permlane32_swap:
  case Intrinsic::amdgcn_permlane64:
  case Intrinsic::amdgcn_update_dpp:
  case Intrinsic::amdgcn_mov_dpp:
  case Intrinsic::amdgcn_mov_dpp8:
  case Intrinsic::amdgcn_ballot:
    return true;
  default:
    return false;
  }
}

} // namespace

CrossWidenDivergenceAnalysis::CrossWidenDivergenceAnalysis(Function &F) {
  seedLeaves(F);
  propagate(F);
}

void CrossWidenDivergenceAnalysis::seedLeaves(Function &F) {
  for (Instruction &I : instructions(F)) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI)
      continue;
    Function *callee = CI->getCalledFunction();
    if (!callee)
      continue;
    if (isDivergentLeafIntrinsic(callee->getIntrinsicID()))
      divergent.insert(&I);
  }
}

void CrossWidenDivergenceAnalysis::propagate(Function &F) {
  // Simple fixed-point over the function body. Typical matmul-scale
  // kernels converge in 2-4 sweeps; the SmallPtrSet membership test
  // is O(1) amortised so the overall cost is O(|insts|^2 / k) in the
  // pathological case, which remains negligible next to raise/llc
  // time for any kernel we care about.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Instruction &I : instructions(F)) {
      if (divergent.count(&I))
        continue;

      // Cross-lane primitives: the result's divergence is not a
      // straightforward union of operand divergence — walk them
      // explicitly so the rewrite pass sees the right output
      // classification on a single query.
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (Function *callee = CI->getCalledFunction()) {
          Intrinsic::ID id = callee->getIntrinsicID();
          if (id == Intrinsic::amdgcn_readfirstlane) {
            // By construction `readfirstlane` produces a uniform value
            // (that is exactly the opcode's hardware semantics). Never
            // mark it divergent, regardless of operand divergence.
            continue;
          }
          if (id == Intrinsic::amdgcn_readlane) {
            // readlane(src, lane) returns the value held by lane
            // `lane` of `src`. If `src` is divergent the per-source-
            // wave meaning of "lane L" differs across waves and the
            // *input* to `readlane` carries per-lane state the sink
            // needs — so we must mark the call divergent so that the
            // rewrite pass converts it to a ds_bpermute that preserves
            // per-source-wave semantics.
            if (divergent.count(CI->getArgOperand(0))) {
              divergent.insert(&I);
              changed = true;
            }
            continue;
          }
          if (id == Intrinsic::amdgcn_writelane) {
            // writelane(val, lane, old) produces a VGPR. Non-selected
            // lanes keep `old`'s per-lane value; the selected lane
            // carries `val`'s per-lane value. Either being divergent
            // makes the result divergent (mixing a uniform value into
            // one lane of a divergent wave is still divergent; mixing
            // a divergent value into any lane of a uniform wave is
            // divergent at that lane).
            if (divergent.count(CI->getArgOperand(0)) ||
                divergent.count(CI->getArgOperand(2))) {
              divergent.insert(&I);
              changed = true;
            }
            continue;
          }
        }
      }

      // PHIs join values from predecessor blocks; divergence at any
      // incoming edge is enough.
      if (auto *PN = dyn_cast<PHINode>(&I)) {
        for (Value *V : PN->incoming_values()) {
          if (divergent.count(V)) {
            divergent.insert(&I);
            changed = true;
            break;
          }
        }
        continue;
      }

      // Default rule: divergent iff any operand is divergent. Covers
      // arithmetic / logical / compare / select / cast / freeze / gep /
      // extract / insert / trunc / load (via pointer operand) and
      // every other SSA producer the raiser emits today.
      bool any = false;
      for (const Use &U : I.operands()) {
        if (divergent.count(U.get())) {
          any = true;
          break;
        }
      }
      if (any) {
        divergent.insert(&I);
        changed = true;
      }
    }
  }
}

bool CrossWidenDivergenceAnalysis::isDivergent(const Value *V) const {
  if (!V)
    return false;
  return divergent.count(V) > 0;
}

} // namespace transpiler
