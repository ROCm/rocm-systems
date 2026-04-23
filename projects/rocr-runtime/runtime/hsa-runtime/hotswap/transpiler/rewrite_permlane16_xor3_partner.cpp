#include "rewrite_permlane16_xor3_partner.hpp"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <vector>

using namespace llvm;

namespace transpiler {

namespace {

// `bpermuteCall(V)` returns the `@llvm.amdgcn.ds.bpermute` call
// reached by walking through `emitUnderExec`-shaped phi pairs.
// Salmon's per-instruction lift wraps SPE-relevant writes in:
//
//   spe_skip:
//     %v.next = phi i32 [ %active_value, %spe_do ],
//                       [ %v.prev, %spe_skip_pred ]
//
// where `%active_value` is the bpermute call.  Mem2reg can't fold
// these phis because the two incoming arms are genuinely different
// (the active arm holds the bpermute result, the inactive arm
// preserves the prior register state).  We walk through phis up
// to `maxSteps` deep, picking the FIRST incoming arm that is (or
// reaches) a bpermute call.  Self-references and nullptr arms are
// skipped.  The match is intentionally narrow: any phi whose arms
// don't contain a bpermute call returns nullptr — we don't try to
// be smart about other phi shapes (loop headers, control-flow
// joins outside SPE).
CallInst *bpermuteCall(Value *V, unsigned maxSteps = 6) {
  for (unsigned i = 0; i < maxSteps; ++i) {
    if (!V)
      return nullptr;
    if (auto *CI = dyn_cast<CallInst>(V)) {
      Function *F = CI->getCalledFunction();
      if (F && F->getIntrinsicID() == Intrinsic::amdgcn_ds_bpermute)
        return CI;
      return nullptr;
    }
    auto *Phi = dyn_cast<PHINode>(V);
    if (!Phi)
      return nullptr;
    Value *next = nullptr;
    for (Value *inc : Phi->incoming_values()) {
      if (!inc || inc == Phi)
        continue;
      if (auto *CI = dyn_cast<CallInst>(inc)) {
        Function *F = CI->getCalledFunction();
        if (F && F->getIntrinsicID() == Intrinsic::amdgcn_ds_bpermute) {
          next = CI;
          break;
        }
      }
      if (auto *nestedPhi = dyn_cast<PHINode>(inc)) {
        // One phi-hop deeper for nested SPE wrappers.
        for (Value *inc2 : nestedPhi->incoming_values()) {
          if (!inc2 || inc2 == nestedPhi)
            continue;
          if (auto *CI = dyn_cast<CallInst>(inc2)) {
            Function *F = CI->getCalledFunction();
            if (F &&
                F->getIntrinsicID() == Intrinsic::amdgcn_ds_bpermute) {
              next = CI;
              break;
            }
          }
        }
        if (next)
          break;
      }
    }
    if (!next)
      return nullptr;
    V = next;
  }
  return nullptr;
}

// Tests whether `value` and `target` are the same SSA after
// walking SPE phi-pair wrappers.  Two scenarios that we accept as
// "same" for the cross-16 idiom match:
//
//   1. Direct SSA identity (`value == target`).
//   2. `value` is a SPE-shaped phi where the "active" arm is
//      `target`, and the "preserve" arm is the same as the
//      target's preceding-stage state (i.e. `target` itself,
//      reached through some chain of preserve-arm phi
//      hops).  Under SPE wrapping with saved_exec=all-ones,
//      every active lane sees `target` and every inactive lane
//      sees a value that wouldn't matter for the rewrite (we'd
//      not swap a lane that wasn't active anyway).
//
// We implement this with a bounded BFS over phi incoming arms:
// `seedReachableFrom(value, target)` returns true iff `target`
// is reachable from `value` by walking phi incoming-value edges,
// only stepping through phis (not through arithmetic / calls /
// other instructions).  This catches both shapes uniformly.
bool seedReachableFrom(Value *value, Value *target,
                        unsigned maxSteps = 6) {
  if (value == target)
    return true;
  for (unsigned i = 0; i < maxSteps; ++i) {
    auto *Phi = dyn_cast<PHINode>(value);
    if (!Phi)
      return false;
    Value *next = nullptr;
    for (Value *inc : Phi->incoming_values()) {
      if (!inc || inc == Phi)
        continue;
      if (inc == target)
        return true;
      if (isa<PHINode>(inc) && !next)
        next = inc;  // queue one phi-hop deeper
    }
    if (!next)
      return false;
    value = next;
  }
  return false;
}

// Match the outer `xor X, seed` where `X = xor bp_a, bp_b` and
// `bp_a / bp_b` are bpermutes with identical address operand and
// both reading `seed` as data.  Returns the bpermute whose result
// equals `partner_seed` (either bpermute will do — they read
// identical data, just the call sites are different).
CallInst *tryMatch(BinaryOperator *outerXor) {
  if (outerXor->getOpcode() != Instruction::Xor)
    return nullptr;
  Value *outerLhs = outerXor->getOperand(0);
  Value *seed = outerXor->getOperand(1);
  // The inner xor can be on either operand of outerXor (xor is
  // commutative; salmon's CreateXor canonicalises one ordering
  // but match both for robustness against IR transforms).
  auto tryShape = [&](Value *innerCandidate, Value *seedCandidate)
      -> CallInst * {
    auto *innerXor = dyn_cast<BinaryOperator>(innerCandidate);
    if (!innerXor || innerXor->getOpcode() != Instruction::Xor)
      return nullptr;
    // Walk through SPE phi wrappers to find the bpermute call for
    // each inner operand.
    CallInst *bp0 = bpermuteCall(innerXor->getOperand(0));
    CallInst *bp1 = bpermuteCall(innerXor->getOperand(1));
    if (!bp0 || !bp1)
      return nullptr;
    // Address must match by SSA identity.  bpermute address is
    // `(lane_id ^ 16) << 2`, emitted once per
    // `emitPermLaneSwapEmulation` call site and reused by both
    // bpermute calls in the same site.
    if (bp0->getArgOperand(0) != bp1->getArgOperand(0))
      return nullptr;
    // The data arguments of both bpermutes should reach `seed`
    // through SPE phi-pair walks.  See `seedReachableFrom` for
    // the precise SSA shape we accept (active arm is seed, with
    // preserve arms allowed to be anything).
    if (!seedReachableFrom(bp0->getArgOperand(1), seedCandidate))
      return nullptr;
    if (!seedReachableFrom(bp1->getArgOperand(1), seedCandidate))
      return nullptr;
    return bp0;
  };
  if (CallInst *bp = tryShape(outerLhs, seed))
    return bp;
  if (CallInst *bp = tryShape(seed, outerLhs))
    return bp;
  return nullptr;
}

} // namespace

Permlane16Xor3PartnerRewriteReport
rewritePermLane16Xor3Partner(Function &F) {
  Permlane16Xor3PartnerRewriteReport report;
  bool dbg = std::getenv("SALMON_DEBUG_XOR3_IDIOM") != nullptr;

  // Two-phase rewrite (collect then mutate) so the iterator over
  // `instructions(F)` doesn't see the IR change underneath us.
  std::vector<std::pair<BinaryOperator *, CallInst *>> sites;
  for (Instruction &I : instructions(F)) {
    auto *bo = dyn_cast<BinaryOperator>(&I);
    if (!bo || bo->getOpcode() != Instruction::Xor)
      continue;
    if (CallInst *partner = tryMatch(bo))
      sites.emplace_back(bo, partner);
  }

  for (auto &site : sites) {
    BinaryOperator *outerXor = site.first;
    CallInst *partner = site.second;
    if (dbg)
      errs() << "[xor3-idiom] rewriting %"
             << (outerXor->hasName() ? outerXor->getName() : "<anon>")
             << " -> %"
             << (partner->hasName() ? partner->getName() : "<bpermute>")
             << "\n";
    outerXor->replaceAllUsesWith(partner);
    outerXor->eraseFromParent();
    report.matchedSites++;
  }

  if (dbg)
    errs() << "[xor3-idiom] " << report.matchedSites
           << " site(s) rewritten in '" << F.getName() << "'\n";

  return report;
}

} // namespace transpiler
