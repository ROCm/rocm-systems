// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Loop Address Promotion Pass
//
// Promotes loop-carried LDS read address computation from per-iteration
// V_ADD_U32 operations to precomputed rotating VGPR iter_args.
//
// Motivation: in double/triple-buffered loops, SGPR pointers rotate each
// iteration (e.g. s0->s1, s1->s0). Each iteration computes an LDS address
// as v_add_u32(tid, s_current), but since tid is loop-invariant and only N
// distinct SGPR values ever appear, all N results can be precomputed.
//
// Algorithm:
//
//   1. findRotationGroups: build permutation sigma where sigma[i] = j iff
//      condIterArgs[i] == blockArgs[j]. Find cycles of length > 1 among
//      SGPR-typed args — these are rotation groups.
//
//   2. analyzeLoopAddressPromotion: scan loop body for v_add_u32 ops that
//      (a) feed a ds_read, (b) have one loop-invariant VGPR operand and
//      one SGPR block arg belonging to a rotation group.
//
//   3. applyLoopAddressPromotion:
//      a) Before the loop, emit N v_add_u32(invariant, init_args[slot])
//         for each rotation slot — these are the precomputed addresses.
//      b) Add them as new VGPR iter_args to an expanded loop.
//      c) Clone the loop body, replacing each promoted v_add_u32 result
//         with the corresponding VGPR iter_arg. Skip the original v_add.
//      d) In condIterArgs, rotate the new VGPR iter_args by 1 position
//         (shift left) so next iteration uses the next precomputed address.
//      e) Erase the old loop.
//
// Net effect: zero VALU address computation in the loop body.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMLOOPADDRESSPROMOTION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

struct PromotableAddressAdd {
  V_ADD_U32 addOp;
  Value invariantVGPR;
  unsigned groupIdx;   // index into rotationGroups
  unsigned posInGroup; // position inside that rotation group
};

struct LoopAddressPromotionPlan {
  LoopOp loopOp;
  ConditionOp condOp;
  Block *body = nullptr;
  unsigned numArgs = 0;
  SmallVector<Value> condIterArgs;
  SmallVector<SmallVector<unsigned>> rotationGroups;
  SmallVector<PromotableAddressAdd> promotableAdds;
};

// True if val is defined outside the loop (not a block arg or op in the body).
static bool isLoopInvariantForLoop(Value val, LoopOp loopOp,
                                   Region *loopRegion) {
  if (auto *defOp = val.getDefiningOp())
    return defOp->getParentRegion() != loopRegion;
  if (auto ba = dyn_cast<BlockArgument>(val))
    return ba.getOwner()->getParentOp() != loopOp.getOperation();
  return false;
}

// True if op is any LDS read variant.
static bool isLDSReadOp(Operation *op) {
  return isa<DS_READ_B128, DS_READ_B64, DS_READ_B32, DS_READ_U8, DS_READ_I8,
             DS_READ_U16, DS_READ_I16>(op);
}

// True if any user of v is an LDS read (profitability filter).
static bool feedsLDSRead(Value v) {
  for (auto *user : v.getUsers())
    if (isLDSReadOp(user))
      return true;
  return false;
}

// Build the iter_arg permutation and return cycles of length > 1 among
// SGPR-typed block args. Each cycle is a rotation group.
static SmallVector<SmallVector<unsigned>>
findRotationGroups(Block &body, ValueRange condIterArgs) {
  auto blockArgs = body.getArguments();
  unsigned numArgs = blockArgs.size();

  SmallVector<int> sigma(numArgs, -1);
  for (unsigned i : llvm::seq(numArgs)) {
    for (unsigned j : llvm::seq(numArgs)) {
      if (condIterArgs[i] == blockArgs[j]) {
        sigma[i] = j;
        break;
      }
    }
  }

  llvm::DenseSet<unsigned> visited;
  SmallVector<SmallVector<unsigned>> rotationGroups;
  for (unsigned i : llvm::seq(numArgs)) {
    if (visited.count(i) || sigma[i] < 0)
      continue;
    if (!isSGPRType(blockArgs[i].getType()))
      continue;

    SmallVector<unsigned> group;
    unsigned cur = i;
    while (!visited.count(cur)) {
      visited.insert(cur);
      group.push_back(cur);
      cur = sigma[cur];
    }

    if (group.size() > 1 && cur == i)
      rotationGroups.push_back(std::move(group));
  }
  return rotationGroups;
}

// Find v_add_u32(invariant_vgpr, rotating_sgpr) ops that feed LDS reads.
// Returns a plan if any promotable adds are found, nullopt otherwise.
static std::optional<LoopAddressPromotionPlan>
analyzeLoopAddressPromotion(LoopOp loopOp) {
  Block &body = loopOp.getBodyBlock();
  auto condOp = dyn_cast<ConditionOp>(body.getTerminator());
  if (!condOp)
    return std::nullopt;

  auto blockArgs = body.getArguments();
  ValueRange condIterArgs = condOp.getIterArgs();
  unsigned numArgs = blockArgs.size();
  if (numArgs == 0 || condIterArgs.size() != numArgs)
    return std::nullopt;

  auto rotationGroups = findRotationGroups(body, condIterArgs);
  if (rotationGroups.empty())
    return std::nullopt;

  SmallVector<std::pair<int, int>> argToGroupAndPos(numArgs,
                                                    std::make_pair(-1, -1));
  for (auto [gi, grp] : llvm::enumerate(rotationGroups))
    for (auto [pi, argIdx] : llvm::enumerate(grp))
      argToGroupAndPos[argIdx] = {static_cast<int>(gi), static_cast<int>(pi)};

  Region *loopRegion = &loopOp.getBodyRegion();
  SmallVector<PromotableAddressAdd> promotableAdds;
  for (Operation &op : body) {
    auto addOp = dyn_cast<V_ADD_U32>(&op);
    if (!addOp || !feedsLDSRead(addOp.getResult()))
      continue;

    auto tryMatch =
        [&](Value maybeVGPR,
            Value maybeSGPR) -> std::optional<PromotableAddressAdd> {
      if (!isLoopInvariantForLoop(maybeVGPR, loopOp, loopRegion))
        return std::nullopt;
      auto ba = dyn_cast<BlockArgument>(maybeSGPR);
      if (!ba || ba.getOwner() != &body || !isSGPRType(ba.getType()))
        return std::nullopt;

      unsigned argIdx = ba.getArgNumber();
      auto [groupIdx, posInGroup] = argToGroupAndPos[argIdx];
      if (groupIdx < 0 || posInGroup < 0)
        return std::nullopt;

      return PromotableAddressAdd{addOp, maybeVGPR,
                                  static_cast<unsigned>(groupIdx),
                                  static_cast<unsigned>(posInGroup)};
    };

    if (auto match = tryMatch(addOp.getSrc0(), addOp.getSrc1())) {
      promotableAdds.push_back(*match);
      continue;
    }
    if (auto match = tryMatch(addOp.getSrc1(), addOp.getSrc0())) {
      promotableAdds.push_back(*match);
      continue;
    }
  }

  if (promotableAdds.empty())
    return std::nullopt;

  LoopAddressPromotionPlan plan;
  plan.loopOp = loopOp;
  plan.condOp = condOp;
  plan.body = &body;
  plan.numArgs = numArgs;
  plan.condIterArgs.assign(condIterArgs.begin(), condIterArgs.end());
  plan.rotationGroups = std::move(rotationGroups);
  plan.promotableAdds = std::move(promotableAdds);
  return plan;
}

// Precompute addresses before the loop, build an expanded loop with VGPR
// iter_args replacing the promoted v_add_u32 ops, and rotate them per
// iteration.
static void applyLoopAddressPromotion(LoopOp loopOp) {
  auto planOpt = analyzeLoopAddressPromotion(loopOp);
  if (!planOpt)
    return;
  LoopAddressPromotionPlan &plan = *planOpt;

  OpBuilder builder(plan.loopOp);
  auto loc = plan.loopOp.getLoc();
  ValueRange initArgs = plan.loopOp.getInitArgs();

  SmallVector<SmallVector<Value>> precomputedByPromotable;
  precomputedByPromotable.reserve(plan.promotableAdds.size());
  for (auto &pa : plan.promotableAdds) {
    auto &grp = plan.rotationGroups[pa.groupIdx];
    SmallVector<Value> precomputed;
    precomputed.reserve(grp.size());
    for (unsigned k : llvm::seq<unsigned>(grp.size())) {
      unsigned rotIdx = (pa.posInGroup + k) % grp.size();
      Value sInit = initArgs[grp[rotIdx]];
      auto preAdd =
          V_ADD_U32::create(builder, loc, pa.addOp.getResult().getType(),
                            pa.invariantVGPR, sInit);
      precomputed.push_back(preAdd.getResult());
    }
    precomputedByPromotable.push_back(std::move(precomputed));
  }

  SmallVector<Value> expandedInit(initArgs.begin(), initArgs.end());
  for (auto &precomputed : precomputedByPromotable)
    expandedInit.append(precomputed.begin(), precomputed.end());

  auto newLoop = LoopOp::create(builder, loc, expandedInit);
  Block &newBody = newLoop.getBodyBlock();

  IRMapping mapping;
  auto oldBlockArgs = plan.body->getArguments();
  for (unsigned i : llvm::seq(plan.numArgs))
    mapping.map(oldBlockArgs[i], newBody.getArgument(i));

  unsigned newArgBase = plan.numArgs;
  llvm::DenseSet<Operation *> promotedOps;
  for (auto &pa : plan.promotableAdds) {
    promotedOps.insert(pa.addOp.getOperation());
    mapping.map(pa.addOp.getResult(), newBody.getArgument(newArgBase));
    newArgBase += plan.rotationGroups[pa.groupIdx].size();
  }

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&newBody);
  for (Operation &op : *plan.body) {
    if (isa<ConditionOp>(&op) || promotedOps.count(&op))
      continue;
    bodyBuilder.clone(op, mapping);
  }

  Value newCond = mapping.lookup(plan.condOp.getCondition());
  SmallVector<Value> newCondIterArgs;
  newCondIterArgs.reserve(plan.condIterArgs.size() + expandedInit.size() -
                          plan.numArgs);
  for (Value v : plan.condIterArgs)
    newCondIterArgs.push_back(mapping.lookup(v));

  newArgBase = plan.numArgs;
  for (auto &pa : plan.promotableAdds) {
    unsigned N = plan.rotationGroups[pa.groupIdx].size();
    for (unsigned k : llvm::seq(1u, N + 1))
      newCondIterArgs.push_back(newBody.getArgument(newArgBase + (k % N)));
    newArgBase += N;
  }

  ConditionOp::create(bodyBuilder, loc, newCond, newCondIterArgs);

  for (unsigned i : llvm::seq(plan.numArgs))
    plan.loopOp.getResult(i).replaceAllUsesWith(newLoop.getResult(i));
  plan.loopOp.erase();
}

struct LoopAddressPromotionPass
    : public waveasm::impl::WAVEASMLoopAddressPromotionBase<
          LoopAddressPromotionPass> {
  using WAVEASMLoopAddressPromotionBase::WAVEASMLoopAddressPromotionBase;

  void runOnOperation() override {
    // Post-order so inner loops are processed before outer ones.
    // Collect first since transformation invalidates walk iterator.
    SmallVector<LoopOp> loops;
    getOperation()->walk<WalkOrder::PostOrder>(
        [&](LoopOp loopOp) { loops.push_back(loopOp); });
    for (auto loopOp : loops)
      applyLoopAddressPromotion(loopOp);
  }
};

} // namespace
