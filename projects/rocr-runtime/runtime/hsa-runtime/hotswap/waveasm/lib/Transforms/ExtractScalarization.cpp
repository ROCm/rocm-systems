// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Extract Scalarization Pass
//
// Pre-translation pass that rewrites vector.extract[k] from
//   arith.addi(vector.broadcast(%x), dense<[...]>)
// (optionally wrapped in arith.index_cast / arith.select) into scalar
// operations so that downstream translation handlers only see ordinary
// scalar IR.
//
// This eliminates the need for special-case lane-index reasoning in
// the translator's arith.addi handler.
//===----------------------------------------------------------------------===//

#include "waveasm/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "waveasm-extract-scalarization"

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMEXTRACTSCALARIZATION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

using namespace mlir;

namespace {

/// Read element `index` from a DenseElementsAttr with integer elements,
/// returning its value as int64_t.
static std::optional<int64_t> getIntElement(DenseElementsAttr dense,
                                            int64_t index) {
  auto vecType = dyn_cast<VectorType>(dense.getType());
  if (!vecType || index < 0 || index >= vecType.getNumElements())
    return std::nullopt;
  auto values = dense.getValues<APInt>();
  return (*(values.begin() + index)).getSExtValue();
}

/// Peel through an optional arith.index_cast, returning the cast input
/// and recording whether we peeled.
static Value peelIndexCast(Value v, bool &peeled) {
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    peeled = true;
    return castOp.getIn();
  }
  peeled = false;
  return v;
}

/// Peel through an optional arith.select, returning both branches and
/// the condition.  Returns failure if `v` is not a select.
static LogicalResult peelSelect(Value v, Value &cond, Value &trueVal,
                                Value &falseVal) {
  if (auto selOp = v.getDefiningOp<arith::SelectOp>()) {
    cond = selOp.getCondition();
    trueVal = selOp.getTrueValue();
    falseVal = selOp.getFalseValue();
    return success();
  }
  return failure();
}

/// Try to match and scalarize:
///   vector.extract[k] (
///     index_cast? (
///       select?(
///         arith.addi(vector.broadcast(%x), dense<[...]>)
///       )
///     )
///   )
/// Returns the scalar replacement value, or nullptr on failure.
static Value tryScalarize(vector::ExtractOp extractOp, OpBuilder &rewriter) {
  auto staticPos = extractOp.getStaticPosition();
  if (staticPos.size() != 1 || staticPos[0] == ShapedType::kDynamic)
    return nullptr;
  int64_t k = staticPos[0];

  Value src = extractOp.getSource();
  Location loc = extractOp.getLoc();

  // Peel optional index_cast wrapping the extract source.
  bool hadOuterIndexCast = false;
  Value preIndexCast = peelIndexCast(src, hadOuterIndexCast);

  // Try select path: extract[k]( index_cast?( select(cond, T, F) ) )
  Value cond, selTrue, selFalse;
  if (succeeded(peelSelect(preIndexCast, cond, selTrue, selFalse))) {
    // Peel optional index_cast inside select branches.
    bool trueHadCast = false, falseHadCast = false;
    Value trueInner = peelIndexCast(selTrue, trueHadCast);
    Value falseInner = peelIndexCast(selFalse, falseHadCast);

    // Try to scalarize the true branch (addi(broadcast, dense)).
    auto scalarizeBranch = [&](Value inner) -> Value {
      auto addOp = inner.getDefiningOp<arith::AddIOp>();
      if (!addOp)
        return nullptr;

      Value broadcastSide = nullptr;
      DenseElementsAttr dense;

      for (int swap = 0; swap < 2; ++swap) {
        Value lhs = swap ? addOp.getRhs() : addOp.getLhs();
        Value rhs = swap ? addOp.getLhs() : addOp.getRhs();

        auto bcast = lhs.getDefiningOp<vector::BroadcastOp>();
        auto constOp = rhs.getDefiningOp<arith::ConstantOp>();
        if (bcast && constOp) {
          auto d = dyn_cast<DenseElementsAttr>(constOp.getValue());
          if (d && !d.isSplat()) {
            broadcastSide = bcast.getSource();
            dense = d;
            break;
          }
        }
      }
      if (!broadcastSide)
        return nullptr;

      auto elemVal = getIntElement(dense, k);
      if (!elemVal)
        return nullptr;

      Type scalarType = broadcastSide.getType();
      Value elemConst = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(scalarType, *elemVal));
      return arith::AddIOp::create(rewriter, loc, scalarType, broadcastSide,
                                   elemConst);
    };

    Value scalarTrue = scalarizeBranch(trueInner);
    if (!scalarTrue)
      return nullptr;

    // The false branch can be either another addi(broadcast, dense) or
    // a splat dense constant.
    Value scalarFalse = scalarizeBranch(falseInner);
    if (!scalarFalse) {
      // Try to extract element k from a splat dense constant.
      auto constOp = falseInner.getDefiningOp<arith::ConstantOp>();
      if (!constOp)
        return nullptr;
      auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
      if (!denseAttr)
        return nullptr;
      auto elemVal = getIntElement(denseAttr, k);
      if (!elemVal)
        return nullptr;
      Type scalarType = scalarTrue.getType();
      scalarFalse = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(scalarType, *elemVal));
    }

    // The original select condition may be vector<Nxi1> (e.g. from a
    // broadcast of a scalar i1).  Scalarize it for the new scalar select.
    Value scalarCond = cond;
    if (isa<VectorType>(cond.getType())) {
      if (auto bcast = cond.getDefiningOp<vector::BroadcastOp>()) {
        scalarCond = bcast.getSource();
      } else {
        scalarCond = vector::ExtractOp::create(rewriter, loc, cond, k);
      }
    }

    return arith::SelectOp::create(rewriter, loc, scalarCond, scalarTrue,
                                   scalarFalse);
  }

  // Non-select path: extract[k]( index_cast?( addi(broadcast, dense) ) )
  auto addOp = preIndexCast.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return nullptr;

  Value broadcastSide = nullptr;
  DenseElementsAttr dense;

  for (int swap = 0; swap < 2; ++swap) {
    Value lhs = swap ? addOp.getRhs() : addOp.getLhs();
    Value rhs = swap ? addOp.getLhs() : addOp.getRhs();

    auto bcast = lhs.getDefiningOp<vector::BroadcastOp>();
    auto constOp = rhs.getDefiningOp<arith::ConstantOp>();
    if (bcast && constOp) {
      auto d = dyn_cast<DenseElementsAttr>(constOp.getValue());
      if (d && !d.isSplat()) {
        broadcastSide = bcast.getSource();
        dense = d;
        break;
      }
    }
  }
  if (!broadcastSide)
    return nullptr;

  auto elemVal = getIntElement(dense, k);
  if (!elemVal)
    return nullptr;

  Type scalarType = broadcastSide.getType();
  Value elemConst = arith::ConstantOp::create(
      rewriter, loc, rewriter.getIntegerAttr(scalarType, *elemVal));
  return arith::AddIOp::create(rewriter, loc, scalarType, broadcastSide,
                               elemConst);
}

struct ExtractScalarizationPass
    : public waveasm::impl::WAVEASMExtractScalarizationBase<
          ExtractScalarizationPass> {
  using WAVEASMExtractScalarizationBase::WAVEASMExtractScalarizationBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    SmallVector<vector::ExtractOp> candidates;

    getOperation()->walk(
        [&](vector::ExtractOp op) { candidates.push_back(op); });

    for (auto op : candidates) {
      rewriter.setInsertionPoint(op);
      Value scalar = tryScalarize(op, rewriter);
      if (!scalar)
        continue;

      Type resultType = op.getResult().getType();
      if (scalar.getType() != resultType) {
        scalar = arith::IndexCastOp::create(rewriter, op.getLoc(), resultType,
                                            scalar);
      }

      rewriter.replaceOp(op, scalar);
    }

    // Erase vector ops that became dead after the rewrites above.
    // Only target vector-typed ops and dense vector constants -- never
    // erase scalar arith.constant or other scalar ops that may be
    // used elsewhere.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> deadOps;
      getOperation()->walk([&](Operation *op) {
        if (!op->use_empty())
          return;
        if (isDeadVectorOp(op))
          deadOps.push_back(op);
      });
      for (auto *op : llvm::reverse(deadOps)) {
        if (op->use_empty()) {
          rewriter.eraseOp(op);
          changed = true;
        }
      }
    }
  }

private:
  static bool isDeadVectorOp(Operation *op) {
    if (isa<vector::BroadcastOp, vector::ExtractOp>(op))
      return true;
    // Vector-typed arith ops (addi, index_cast, select on vectors).
    if (op->getNumResults() == 1 &&
        isa<VectorType>(op->getResult(0).getType()) &&
        isa<arith::AddIOp, arith::IndexCastOp, arith::SelectOp>(op))
      return true;
    // Dense vector constants.
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      if (isa<VectorType>(constOp.getResult().getType()))
        return true;
    }
    return false;
  }
};

} // namespace
