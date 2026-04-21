#include "rewrite_cross_lane_divergent.hpp"

#include "cross_widen_divergence.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace transpiler {

namespace {

// Build (once per function) the target-wave absolute lane id as the
// standard two-step mbcnt idiom. Returned value dominates every use
// site because it is emitted at the head of the function's entry
// block, immediately after the terminator of `allocas-and-setup`
// prelude (we insert at the entry's first insertion point).
//
// Wave64-correct derivation (AMDGPU Instruction Set Architecture
// chapter, v_mbcnt_* section):
//   * `mbcnt_lo(-1, 0)`: popcount((-1) & ((1 << min(laneId, 32)) - 1))
//                       = min(laneId, 32).
//   * `mbcnt_hi(-1, prev)`: prev + popcount((-1) & ((1 <<
//                              max(laneId - 32, 0)) - 1))
//                        = prev + max(laneId - 32, 0).
// Summed: the absolute target-wave lane id in [0, targetWaveSize).
//
// On a wave32-target host the `mbcnt_hi` leg is a no-op (every lane
// has `max(laneId - 32, 0) == 0`) so the same emission is correct; we
// never enter this pass for wave32 targets per the direction gate in
// `rewriteCrossLaneDivergent`, but the form stays portable.
Value *buildTargetLaneId(Function &F) {
  Module *M = F.getParent();
  LLVMContext &C = F.getContext();
  Type *i32Ty = Type::getInt32Ty(C);
  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  Function *mbcntLo = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_mbcnt_lo);
  Function *mbcntHi = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_mbcnt_hi);
  // `ConstantInt::get(IntegerType*, uint64_t V, bool IsSigned=false)`
  // asserts `V < 2^BitWidth` when `!IsSigned`; implicit (int64_t)-1 ->
  // uint64_t produces `0xFFFF'FFFF'FFFF'FFFF` which blows that assert
  // for a 32-bit type. Use the unsigned 32-bit all-ones bit pattern
  // (2^32 - 1), which mbcnt hardware interprets as the wave-wide
  // exec-all mask — the standard idiom behind the two-step lane_id
  // construction.
  Value *minusOne = ConstantInt::get(i32Ty, 0xFFFFFFFFu);
  Value *zero = ConstantInt::get(i32Ty, 0);
  Value *laneLo = B.CreateCall(mbcntLo, {minusOne, zero},
                                "cwd_lane_id_lo");
  Value *laneId = B.CreateCall(mbcntHi, {minusOne, laneLo},
                                "cwd_lane_id");
  return laneId;
}

// Rewrite one `amdgcn.writelane(val, lane, old)` call to
// `select ((lane_id & (W_s-1)) == lane), val, old` in-place. Preserves
// debug locations and original `val` / `lane` / `old` types (the
// writelane intrinsic is overloaded; we stay on i32 because that is
// the only overload the raiser emits today — see
// `handle_valu_cross_lane.cpp::V_WRITELANE_B32`).
void rewriteWritelaneCall(CallInst *CI, Value *laneId,
                          unsigned sourceWaveSize) {
  IRBuilder<> B(CI);
  B.SetCurrentDebugLocation(CI->getDebugLoc());
  Type *i32Ty = B.getInt32Ty();
  Value *val = CI->getArgOperand(0);
  Value *laneIdx = CI->getArgOperand(1);
  Value *oldVal = CI->getArgOperand(2);
  Value *modMask = ConstantInt::get(i32Ty, sourceWaveSize - 1);
  Value *laneMod = B.CreateAnd(laneId, modMask, "cwd_wl_lane_mod");
  Value *selMask = B.CreateICmpEQ(laneMod, laneIdx, "cwd_wl_mask");
  Value *newVal = B.CreateSelect(selMask, val, oldVal,
                                  "cwd_writelane_rewritten");
  CI->replaceAllUsesWith(newVal);
  CI->eraseFromParent();
}

// Rewrite one `amdgcn.readlane(src, lane)` call to
// `ds_bpermute(((lane_id & ~(W_s-1)) | lane) << 2, src)`. The `<< 2`
// byte-offset convention is `int_amdgcn_ds_bpermute`'s selector
// encoding (IntrinsicsAMDGPU.td docstring); the base-mask alignment
// keeps each source wave's broadcast inside its own W_s lanes so
// source_wave[k]'s `readlane(…, lane_idx=j)` delivers the value held
// by lane `(k*W_s + j)` of the pre-bpermute src — exactly the source
// kernel's single-wave readlane under cross-widening.
void rewriteReadlaneCall(CallInst *CI, Value *laneId,
                         unsigned sourceWaveSize) {
  IRBuilder<> B(CI);
  B.SetCurrentDebugLocation(CI->getDebugLoc());
  Module *M = CI->getModule();
  Type *i32Ty = B.getInt32Ty();
  Value *src = CI->getArgOperand(0);
  Value *laneIdx = CI->getArgOperand(1);

  // Two's-complement sign-extends the sourceWaveSize-1 mask when
  // we flip it — casting through a uint32_t keeps the negation on a
  // well-defined 32-bit domain and avoids UB on platforms where
  // `unsigned` is wider than 32 bits.
  uint32_t baseMaskImm =
      ~(static_cast<uint32_t>(sourceWaveSize) - 1u);
  Value *baseMask = ConstantInt::get(i32Ty, baseMaskImm);
  Value *srcWaveBase = B.CreateAnd(laneId, baseMask,
                                    "cwd_rl_src_wave_base");
  Value *bcastLane = B.CreateOr(srcWaveBase, laneIdx,
                                 "cwd_rl_bcast_lane");
  Value *selector = B.CreateShl(bcastLane, ConstantInt::get(i32Ty, 2),
                                 "cwd_rl_selector");
  Function *bpermute = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ds_bpermute);
  Value *broadcast = B.CreateCall(bpermute, {selector, src},
                                   "cwd_readlane_rewritten");
  CI->replaceAllUsesWith(broadcast);
  CI->eraseFromParent();
}

} // namespace

CrossLaneDivergentRewriteReport rewriteCrossLaneDivergent(
    Function &F, unsigned sourceWaveSize, unsigned targetWaveSize) {
  CrossLaneDivergentRewriteReport report;

  // Direction gate. Same-wave / narrowing skip the rewrite entirely:
  // the backend's implicit readfirstlane would not collapse any per-
  // source-wave state that the target wave does not also hold.
  if (targetWaveSize <= sourceWaveSize)
    return report;

  // Pre-collect candidate call sites. Iterating the function while
  // rewriting would mutate the CFG under the iterator; the two-phase
  // shape keeps the walk O(n) and the rewrite-phase linear in the
  // number of matched sites.
  SmallVector<CallInst *, 16> writelaneSites;
  SmallVector<CallInst *, 16> readlaneSites;
  for (Instruction &I : instructions(F)) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI)
      continue;
    Function *callee = CI->getCalledFunction();
    if (!callee)
      continue;
    switch (callee->getIntrinsicID()) {
    case Intrinsic::amdgcn_writelane:
      writelaneSites.push_back(CI);
      break;
    case Intrinsic::amdgcn_readlane:
      readlaneSites.push_back(CI);
      break;
    default:
      break;
    }
  }

  if (writelaneSites.empty() && readlaneSites.empty())
    return report;

  // Build the divergence oracle once for the whole function.
  CrossWidenDivergenceAnalysis divAna(F);

  // Materialise the lane-id helper lazily: only emit the mbcnt pair
  // if at least one site actually needs rewriting. Avoids perturbing
  // uniform-only functions (the dominant corpus case) with an
  // otherwise-dead lane_id in the entry block.
  Value *laneIdCached = nullptr;
  auto getLaneId = [&]() -> Value * {
    if (!laneIdCached)
      laneIdCached = buildTargetLaneId(F);
    return laneIdCached;
  };

  // Rewrite divergent writelane sites. `val` divergent OR `old`
  // divergent is the trigger: the output must preserve per-source-wave
  // state on both the selected lane (from `val`) and the unselected
  // lanes (from `old`). Uniform on both operands -> backend's canonical
  // `v_writelane_b32` lowering is correct; leave in place.
  for (CallInst *CI : writelaneSites) {
    Value *val = CI->getArgOperand(0);
    Value *oldVal = CI->getArgOperand(2);
    bool divergentOperand =
        divAna.isDivergent(val) || divAna.isDivergent(oldVal);
    if (!divergentOperand) {
      ++report.uniformPreserved;
      continue;
    }
    rewriteWritelaneCall(CI, getLaneId(), sourceWaveSize);
    ++report.writelaneRewritten;
  }

  // Rewrite divergent readlane sites. `src` divergent is the trigger:
  // the backend's readfirstlane on the src would collapse per-source-
  // wave state. Uniform src -> the per-source-wave readlane semantics
  // reduce to the trivial "every lane reads the same SGPR value";
  // leave the canonical `v_readlane_b32` lowering in place.
  for (CallInst *CI : readlaneSites) {
    Value *src = CI->getArgOperand(0);
    if (!divAna.isDivergent(src)) {
      ++report.uniformPreserved;
      continue;
    }
    rewriteReadlaneCall(CI, getLaneId(), sourceWaveSize);
    ++report.readlaneRewritten;
  }

  return report;
}

} // namespace transpiler
