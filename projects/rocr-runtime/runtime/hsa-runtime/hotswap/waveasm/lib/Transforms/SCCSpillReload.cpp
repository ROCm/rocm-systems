// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SCC Spill/Reload Pass
//
// After CSE merges identical s_cmp ops, the SCC live range may extend
// across SCC-clobbering ops (s_and, s_or, s_add, etc.).  This pass
// inserts s_cselect_b32 (spill: SCC -> SGPR) right after the s_cmp and
// s_cmp_ne_u32 (reload: SGPR -> SCC) right before each consumer that
// has an SCC clobber between it and the definition.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "waveasm-scc-spill-reload"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMSCCSPILLRELOAD
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

/// Return true if the operation writes the SCC flag on hardware.
static bool writesSCC(Operation *op) { return op->hasTrait<OpTrait::SCCDef>(); }

/// Check if there is any SCC-clobbering op between \p producer and
/// \p consumer in the same block.  Returns true if a clobber exists.
static bool hasSCCClobberBetween(Operation *producer, Operation *consumer) {
  if (!producer || !consumer)
    return false;
  // Different blocks: conservatively assume clobber.
  if (producer->getBlock() != consumer->getBlock())
    return true;
  for (Operation *op = producer->getNextNode(); op && op != consumer;
       op = op->getNextNode()) {
    if (writesSCC(op))
      return true;
  }
  return false;
}

struct SCCSpillReloadPass
    : public waveasm::impl::WAVEASMSCCSpillReloadBase<SCCSpillReloadPass> {
  using WAVEASMSCCSpillReloadBase::WAVEASMSCCSpillReloadBase;

  void runOnOperation() override {
    getOperation()->walk(
        [&](ProgramOp program) { processBlock(program.getBodyBlock()); });
  }

private:
  void processBlock(Block &block) {
    // Collect SCC-typed results that need spill/reload.
    // We process each SCC-typed value and check if any of its users
    // have an SCC-clobbering op between the definition and the use.
    //
    // Key: the SCC-typed value (from s_cmp or carry result).
    // Value: the spill SGPR (created lazily, shared across all users).
    llvm::DenseMap<Value, Value> spillMap;

    // First pass: identify values that need spilling.  Do not modify
    // the IR during this pass to avoid iterator invalidation.
    struct SpillInfo {
      Value sccValue;
      Operation *producer;
      llvm::SmallVector<OpOperand *, 4> usesNeedingReload;
    };
    llvm::SmallVector<SpillInfo> spills;

    for (Operation &op : block) {
      // Recurse into nested regions first.
      for (Region &region : op.getRegions())
        for (Block &nestedBlock : region)
          processBlock(nestedBlock);

      for (Value result : op.getResults()) {
        if (!isa<SCCType>(result.getType()))
          continue;

        // Check each use of this SCC value.
        llvm::SmallVector<OpOperand *, 4> needsReload;
        for (OpOperand &use : result.getUses()) {
          Operation *consumer = use.getOwner();
          if (hasSCCClobberBetween(&op, consumer))
            needsReload.push_back(&use);
        }

        if (!needsReload.empty()) {
          spills.push_back({result, &op, std::move(needsReload)});
        }
      }
    }

    if (spills.empty())
      return;

    // Second pass: insert spill/reload ops.
    auto *ctx = block.getParent()->getParentOp()->getContext();
    auto sregTy = SRegType::get(ctx, 1, 1);
    auto sccTy = SCCType::get(ctx);
    auto immTy1 = ImmType::get(ctx, 1);
    auto immTy0 = ImmType::get(ctx, 0);

    for (auto &info : spills) {
      OpBuilder builder(ctx);

      // Insert spill right after the producer: s_cselect_b32 sN, 1, 0.
      builder.setInsertionPointAfter(info.producer);
      Value oneConst =
          ConstantOp::create(builder, info.producer->getLoc(), immTy1, 1);
      Value zeroConst =
          ConstantOp::create(builder, info.producer->getLoc(), immTy0, 0);
      // s_cselect_b32 reads SCC explicitly: dst = SCC ? src0 : src1.
      Value spillSgpr =
          S_CSELECT_B32::create(builder, info.producer->getLoc(), sregTy,
                                info.sccValue, oneConst, zeroConst);

      LDBG() << "SCC spill: inserted s_cselect_b32 after "
             << info.producer->getName();

      // Insert reload before each consumer that needs it.
      for (OpOperand *use : info.usesNeedingReload) {
        Operation *consumer = use->getOwner();
        builder.setInsertionPoint(consumer);

        // s_cmp_ne_u32 sN, 0: reloads the spilled boolean back into SCC.
        Value reloadZero =
            ConstantOp::create(builder, consumer->getLoc(), immTy0, 0);
        Value reloadSCC = S_CMP_NE_U32::create(builder, consumer->getLoc(),
                                               sccTy, spillSgpr, reloadZero);

        // Replace the consumer's SCC operand with the reload result.
        use->set(reloadSCC);

        LDBG() << "SCC reload: inserted s_cmp_ne_u32 before "
               << consumer->getName();
      }
    }
  }
};

} // namespace
