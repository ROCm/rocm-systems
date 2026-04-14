// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WAVEASM_TRANSFORMS_REGIONBUILDER_H
#define WAVEASM_TRANSFORMS_REGIONBUILDER_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "waveasm/Transforms/TranslateFromMLIR.h"

namespace waveasm {

/// Helper for building structured control flow regions during translation
class RegionBuilder {
public:
  RegionBuilder(TranslationContext &ctx) : ctx(ctx) {}

  /// Build a loop from SCF for
  /// Maps scf.for to waveasm.loop with proper block arguments
  LoopOp buildLoopFromSCFFor(mlir::scf::ForOp forOp);

  /// Build an if-then-else from SCF if
  /// Maps scf.if to waveasm.if with proper regions
  IfOp buildIfFromSCFIf(mlir::scf::IfOp ifOp);

private:
  TranslationContext &ctx;

  /// Helper: Translate an operation (forward declaration for recursion)
  mlir::LogicalResult translateOp(mlir::Operation *op);
};

} // namespace waveasm

#endif // WAVEASM_TRANSFORMS_REGIONBUILDER_H
