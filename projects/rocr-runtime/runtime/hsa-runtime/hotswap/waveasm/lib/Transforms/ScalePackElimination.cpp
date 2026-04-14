// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Scale Pack Elimination Pass
//
// Eliminates redundant BFE->iter_arg->LSHL_OR round-trips for B-scale values
// in scaled MFMA loops.
//
// TODO: This should ideally be done at the vector dialect level (before
// lowering to waveasm), where the byte-extract/repack pattern is more
// explicit and easier to canonicalize away.
//
// Pattern:
//   Before the loop:
//     %init_dword = buffer_load_dword ...
//     %b0 = v_bfe_u32(%init_dword, 0, 8)
//     %b1 = v_bfe_u32(%init_dword, 8, 8)
//     %b2 = v_bfe_u32(%init_dword, 16, 8)
//     %b3 = v_bfe_u32(%init_dword, 24, 8)
//
//   Loop body (top):
//     %t1     = v_lshl_or_b32(%arg_b1, 8, %arg_b0)
//     %t2     = v_lshl_or_b32(%arg_b2, 16, %t1)
//     %packed = v_lshl_or_b32(%arg_b3, 24, %t2)
//     ... use %packed as B-scale in v_mfma_scale ...
//
//   Loop body (bottom):
//     %new_dword = buffer_load_dword ...
//     %nb0 = v_bfe_u32(%new_dword, 0, 8)
//     %nb1 = v_bfe_u32(%new_dword, 8, 8)
//     %nb2 = v_bfe_u32(%new_dword, 16, 8)
//     %nb3 = v_bfe_u32(%new_dword, 24, 8)
//     yield ..., %nb0, %nb1, %nb2, %nb3, ...
//
// After optimization:
//   Loop carries the dword directly. The LSHL_OR chain and BFE extractions
//   are eliminated, saving 3 LSHL_OR + 4 BFE = 7 VALU instructions per
//   chain per iteration.
//   This matches AITER's pattern where buffer_load_dword feeds MFMA opsel
//   directly.
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "waveasm-scale-pack-elimination"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMSCALEPACKELIMINATION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

struct PackChain {
  V_LSHL_OR_B32 outerOp;  // shift = 24.
  V_LSHL_OR_B32 middleOp; // shift = 16.
  V_LSHL_OR_B32 innerOp;  // shift = 8.
  unsigned byteArgIdx[4]; // Block arg indices for bytes 0-3.
  Value initDword;        // Init arg source dword (before loop).
  Value yieldDword;       // Yield source dword (bottom of loop).
};

/// Match a v_bfe_u32(src, offset, 8) and return (src, offset).
static std::optional<std::pair<Value, int64_t>> matchBFE8(Value v) {
  auto bfe = v.getDefiningOp<V_BFE_U32>();
  if (!bfe)
    return std::nullopt;

  auto width = getConstantValue(bfe.getSrc2());
  if (!width || *width != 8)
    return std::nullopt;

  auto offset = getConstantValue(bfe.getSrc1());
  if (!offset)
    return std::nullopt;

  return std::make_pair(bfe.getSrc0(), *offset);
}

/// Try to find a LSHL_OR pack chain starting from the outermost op.
/// Returns the chain if all 4 byte sources are block arguments of the loop.
static std::optional<PackChain> findPackChain(V_LSHL_OR_B32 outerOp,
                                              Block &body) {
  // Outer: v_lshl_or_b32(byte3, 24, middle_result).
  auto shiftOuter = getConstantValue(outerOp.getSrc1());
  if (!shiftOuter || *shiftOuter != 24)
    return std::nullopt;

  auto byte3 = dyn_cast<BlockArgument>(outerOp.getSrc0());
  if (!byte3 || byte3.getOwner() != &body)
    return std::nullopt;

  // Middle: v_lshl_or_b32(byte2, 16, inner_result).
  auto middleOp = outerOp.getSrc2().getDefiningOp<V_LSHL_OR_B32>();
  if (!middleOp)
    return std::nullopt;

  auto shiftMiddle = getConstantValue(middleOp.getSrc1());
  if (!shiftMiddle || *shiftMiddle != 16)
    return std::nullopt;

  auto byte2 = dyn_cast<BlockArgument>(middleOp.getSrc0());
  if (!byte2 || byte2.getOwner() != &body)
    return std::nullopt;

  // Inner: v_lshl_or_b32(byte1, 8, byte0).
  auto innerOp = middleOp.getSrc2().getDefiningOp<V_LSHL_OR_B32>();
  if (!innerOp)
    return std::nullopt;

  auto shiftInner = getConstantValue(innerOp.getSrc1());
  if (!shiftInner || *shiftInner != 8)
    return std::nullopt;

  auto byte1 = dyn_cast<BlockArgument>(innerOp.getSrc0());
  if (!byte1 || byte1.getOwner() != &body)
    return std::nullopt;

  auto byte0 = dyn_cast<BlockArgument>(innerOp.getSrc2());
  if (!byte0 || byte0.getOwner() != &body)
    return std::nullopt;

  // All 4 bytes must be distinct block arguments.
  llvm::SmallDenseSet<unsigned, 4> indices;
  indices.insert(byte0.getArgNumber());
  indices.insert(byte1.getArgNumber());
  indices.insert(byte2.getArgNumber());
  indices.insert(byte3.getArgNumber());
  if (indices.size() != 4)
    return std::nullopt;

  PackChain chain;
  chain.outerOp = outerOp;
  chain.middleOp = middleOp;
  chain.innerOp = innerOp;
  chain.byteArgIdx[0] = byte0.getArgNumber();
  chain.byteArgIdx[1] = byte1.getArgNumber();
  chain.byteArgIdx[2] = byte2.getArgNumber();
  chain.byteArgIdx[3] = byte3.getArgNumber();
  return chain;
}

/// Check if 4 values (at the given arg indices) are BFE extractions of
/// bytes 0,8,16,24 from the same source dword.
static std::optional<Value> verifyBFEGroup(llvm::ArrayRef<unsigned> argIndices,
                                           ValueRange values) {
  if (argIndices.size() != 4)
    return std::nullopt;

  // Expected offsets for each byte position.
  static const int64_t expectedOffsets[] = {0, 8, 16, 24};

  Value commonSource;
  for (unsigned i : llvm::seq(4u)) {
    unsigned idx = argIndices[i];
    if (idx >= values.size())
      return std::nullopt;

    auto match = matchBFE8(values[idx]);
    if (!match)
      return std::nullopt;

    if (match->second != expectedOffsets[i])
      return std::nullopt;

    if (i == 0) {
      commonSource = match->first;
    } else if (commonSource != match->first) {
      return std::nullopt;
    }
  }

  return commonSource;
}

/// Check that the byte block arguments and intermediate chain results have
/// no uses outside the pack chain. This ensures removing them is safe.
static bool chainOnlyUsedInternally(PackChain &chain, Block &body) {
  llvm::SmallDenseSet<Operation *, 4> packOps;
  packOps.insert(chain.innerOp.getOperation());
  packOps.insert(chain.middleOp.getOperation());
  packOps.insert(chain.outerOp.getOperation());

  // Byte block args must only feed into the pack chain.
  for (unsigned i : llvm::seq(4u)) {
    Value arg = body.getArgument(chain.byteArgIdx[i]);
    for (OpOperand &use : arg.getUses()) {
      if (!packOps.contains(use.getOwner()))
        return false;
    }
  }

  // Intermediate results (inner, middle) must only feed the next op in chain.
  if (!chain.innerOp.getResult().hasOneUse() ||
      !chain.middleOp.getResult().hasOneUse())
    return false;

  return true;
}

static void eliminateScalePackChains(LoopOp loopOp) {
  Block &body = loopOp.getBodyBlock();
  auto condOp = dyn_cast<ConditionOp>(body.getTerminator());
  if (!condOp)
    return;

  ValueRange initArgs = loopOp.getInitArgs();
  ValueRange condIterArgs = condOp.getIterArgs();
  unsigned numArgs = body.getNumArguments();

  if (numArgs == 0 || condIterArgs.size() != numArgs)
    return;

  // Step 1: Find all pack chains.
  SmallVector<PackChain> chains;
  for (Operation &op : body) {
    auto lshlOr = dyn_cast<V_LSHL_OR_B32>(&op);
    if (!lshlOr)
      continue;

    auto chain = findPackChain(lshlOr, body);
    if (!chain)
      continue;

    // Step 2: Verify init args are BFE from same source.
    auto initDword = verifyBFEGroup(chain->byteArgIdx, initArgs);
    if (!initDword)
      continue;
    chain->initDword = *initDword;

    // Step 3: Verify yield args are BFE from same source.
    auto yieldDword = verifyBFEGroup(chain->byteArgIdx, condIterArgs);
    if (!yieldDword)
      continue;
    chain->yieldDword = *yieldDword;

    // Step 4: Verify byte args and intermediate results have no other uses.
    if (!chainOnlyUsedInternally(*chain, body))
      continue;

    chains.push_back(*chain);
  }

  if (chains.empty())
    return;

  LDBG() << "found " << chains.size() << " pack chains to eliminate ("
         << chains.size() * 4 << " iter_args -> " << chains.size()
         << " dword iter_args)";

  // Step 5: Collect indices of byte args to remove.
  llvm::SmallDenseSet<unsigned, 8> removedArgIndices;
  for (auto &chain : chains)
    for (unsigned i : llvm::seq(4u))
      removedArgIndices.insert(chain.byteArgIdx[i]);

  // Collect operations to skip during cloning (pack chain ops + dead BFEs).
  llvm::SmallDenseSet<Operation *, 8> opsToSkip;
  for (auto &chain : chains) {
    opsToSkip.insert(chain.innerOp.getOperation());
    opsToSkip.insert(chain.middleOp.getOperation());
    opsToSkip.insert(chain.outerOp.getOperation());
  }

  // Also mark yield-side BFE ops as skippable (if single-use for the yield).
  for (auto &chain : chains) {
    for (unsigned i : llvm::seq(4u)) {
      Value yieldVal = condIterArgs[chain.byteArgIdx[i]];
      if (auto bfe = yieldVal.getDefiningOp<V_BFE_U32>())
        if (bfe.getResult().hasOneUse())
          opsToSkip.insert(bfe.getOperation());
    }
  }

  // Step 6: Build new init args (old args minus removed + new dword args).
  SmallVector<Value> newInitArgs;
  // Map from old arg index to new arg index.
  SmallVector<int> oldToNewArgIdx(numArgs, -1);
  unsigned newIdx = 0;
  for (unsigned i : llvm::seq(numArgs)) {
    if (!removedArgIndices.contains(i)) {
      newInitArgs.push_back(initArgs[i]);
      oldToNewArgIdx[i] = newIdx++;
    }
  }
  // Append dword init args for each chain.
  SmallVector<unsigned> chainNewArgIdx;
  for (auto &chain : chains) {
    chainNewArgIdx.push_back(newIdx++);
    newInitArgs.push_back(chain.initDword);
  }

  // Step 7: Build new LoopOp.
  OpBuilder builder(loopOp);
  auto loc = loopOp.getLoc();
  auto newLoop = LoopOp::create(builder, loc, newInitArgs);
  Block &newBody = newLoop.getBodyBlock();

  // Step 8: Build IRMapping from old block args to new.
  IRMapping mapping;
  for (unsigned i : llvm::seq(numArgs)) {
    if (oldToNewArgIdx[i] >= 0)
      mapping.map(body.getArgument(i), newBody.getArgument(oldToNewArgIdx[i]));
  }

  // Map pack chain outputs to the new dword block args.
  for (auto [ci, chain] : llvm::enumerate(chains)) {
    Value dwordArg = newBody.getArgument(chainNewArgIdx[ci]);
    mapping.map(chain.outerOp.getResult(), dwordArg);
  }

  // Step 9: Clone body (skipping removed ops).
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&newBody);
  for (Operation &op : body) {
    if (isa<ConditionOp>(&op))
      continue;
    if (opsToSkip.contains(&op))
      continue;
    bodyBuilder.clone(op, mapping);
  }

  // Step 10: Build new ConditionOp.
  Value newCond = mapping.lookup(condOp.getCondition());
  SmallVector<Value> newCondIterArgs;
  for (unsigned i : llvm::seq(numArgs)) {
    if (removedArgIndices.contains(i))
      continue;
    newCondIterArgs.push_back(mapping.lookup(condIterArgs[i]));
  }
  // Append dword yield values for each chain.
  for (auto &chain : chains)
    newCondIterArgs.push_back(mapping.lookup(chain.yieldDword));

  ConditionOp::create(bodyBuilder, loc, newCond, newCondIterArgs);

  // Step 11: Replace old loop results with new loop results.
  // For removed byte args with post-loop uses, emit BFE extractions.
  OpBuilder postBuilder(builder.getContext());
  postBuilder.setInsertionPointAfter(newLoop);

  auto emitPostLoopBFE = [&](unsigned argIdx) -> Value {
    for (auto [ci, chain] : llvm::enumerate(chains)) {
      for (unsigned bi : llvm::seq(4u)) {
        if (chain.byteArgIdx[bi] != argIdx)
          continue;
        int64_t offset = bi * 8;
        auto vregType = postBuilder.getType<VRegType>(1, 1);
        auto offsetImm = postBuilder.getType<ImmType>(offset);
        auto offsetConst =
            ConstantOp::create(postBuilder, loc, offsetImm, offset);
        auto widthImm = postBuilder.getType<ImmType>(8);
        auto widthConst = ConstantOp::create(postBuilder, loc, widthImm, 8);
        Value dwordResult = newLoop.getResult(chainNewArgIdx[ci]);
        return V_BFE_U32::create(postBuilder, loc, vregType, dwordResult,
                                 offsetConst, widthConst);
      }
    }
    llvm_unreachable("removed arg not found in any chain");
  };

  for (unsigned i : llvm::seq(numArgs)) {
    if (oldToNewArgIdx[i] >= 0) {
      loopOp.getResult(i).replaceAllUsesWith(
          newLoop.getResult(oldToNewArgIdx[i]));
    } else if (!loopOp.getResult(i).use_empty()) {
      loopOp.getResult(i).replaceAllUsesWith(emitPostLoopBFE(i));
    }
  }

  loopOp.erase();
}

struct ScalePackEliminationPass
    : public waveasm::impl::WAVEASMScalePackEliminationBase<
          ScalePackEliminationPass> {
  using WAVEASMScalePackEliminationBase::WAVEASMScalePackEliminationBase;

  void runOnOperation() override {
    // Post-order so inner loops are processed before outer ones.
    // Collect first since transformation invalidates walk iterator.
    SmallVector<LoopOp> loops;
    getOperation()->walk<WalkOrder::PostOrder>(
        [&](LoopOp loopOp) { loops.push_back(loopOp); });
    for (auto loopOp : loops)
      eliminateScalePackChains(loopOp);
  }
};

} // namespace
