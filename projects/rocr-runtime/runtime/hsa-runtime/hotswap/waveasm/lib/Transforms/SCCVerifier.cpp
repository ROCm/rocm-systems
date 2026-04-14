// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SCC Verifier Pass
//
// Verifies that no SCC-clobbering SALU instruction is placed between an
// SCC-producing op and its consumer.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "waveasm-scc-verifier"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMSCCVERIFIER
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

/// Returns true if the operation writes the SCC flag on hardware.
static bool writesSCC(Operation *op) { return op->hasTrait<OpTrait::SCCDef>(); }

struct SCCVerifierPass
    : public waveasm::impl::WAVEASMSCCVerifierBase<SCCVerifierPass> {
  using WAVEASMSCCVerifierBase::WAVEASMSCCVerifierBase;

  void runOnOperation() override {
    Operation *module = getOperation();
    unsigned errorCount = 0;
    module->walk([&](ProgramOp program) {
      for (Block &block : program.getBody())
        errorCount += verifyBlock(block);
    });
    if (errorCount > 0) {
      LDBG() << "SCC verifier: found " << errorCount << " SCC hazard(s)";
      signalPassFailure();
    }
  }

private:
  static SmallVector<Operation *> findSCCClobbersBetween(Operation *producer,
                                                         Operation *consumer) {
    SmallVector<Operation *> clobbers;
    if (!producer || !consumer || producer->getBlock() != consumer->getBlock())
      return clobbers;
    for (Operation *op = producer->getNextNode(); op && op != consumer;
         op = op->getNextNode()) {
      if (writesSCC(op))
        clobbers.push_back(op);
    }
    return clobbers;
  }

  static void emitSCCClobberError(Operation *consumer, Operation *producer,
                                  ArrayRef<Operation *> clobbers) {
    auto diag = consumer->emitError()
                << "SCC hazard: " << clobbers.size()
                << " SCC-clobbering op(s) between SCC producer '"
                << producer->getName() << "' and consumer '"
                << consumer->getName() << "'";
    for (Operation *c : clobbers)
      diag.attachNote(c->getLoc())
          << "SCC clobbered here by '" << c->getName() << "'";
    diag.attachNote(producer->getLoc()) << "SCC defined here";
  }

  unsigned verifyBlock(Block &block) {
    unsigned errors = 0;
    for (Operation &op : block) {
      // Check every SCC-typed operand: its producer must not be separated
      // from this consumer by an SCC-clobbering op.
      for (Value operand : op.getOperands()) {
        if (!isa<SCCType>(operand.getType()))
          continue;
        Operation *producer = operand.getDefiningOp();
        if (!producer)
          continue;
        auto clobbers = findSCCClobbersBetween(producer, &op);
        if (!clobbers.empty()) {
          emitSCCClobberError(&op, producer, clobbers);
          ++errors;
        }
      }
      for (Region &region : op.getRegions())
        for (Block &nestedBlock : region)
          errors += verifyBlock(nestedBlock);
    }
    return errors;
  }
};

} // namespace
