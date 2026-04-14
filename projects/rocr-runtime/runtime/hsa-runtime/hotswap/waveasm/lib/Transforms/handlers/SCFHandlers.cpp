// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SCF Dialect Handlers
//===----------------------------------------------------------------------===//
//
// This file implements handlers for SCF dialect operations:
//   - scf.for
//   - scf.if
//   - scf.yield
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"
#include "waveasm/Transforms/RegionBuilder.h"

#include "waveasm/Transforms/TranslateFromMLIR.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace waveasm;

namespace waveasm {

LogicalResult handleSCFIf(Operation *op, TranslationContext &ctx) {
  auto ifOp = cast<scf::IfOp>(op);

  RegionBuilder regionBuilder(ctx);
  auto waveIfOp = regionBuilder.buildIfFromSCFIf(ifOp);

  if (!waveIfOp)
    return failure();

  return success();
}

LogicalResult handleSCFFor(Operation *op, TranslationContext &ctx) {
  auto forOp = cast<scf::ForOp>(op);

  RegionBuilder regionBuilder(ctx);
  auto loopOp = regionBuilder.buildLoopFromSCFFor(forOp);

  if (!loopOp)
    return failure();

  return success();
}

/// Handle scf.yield - typically a no-op
LogicalResult handleSCFYield(Operation *op, TranslationContext &ctx) {
  // Yield values are handled by the parent op
  return success();
}

} // namespace waveasm
