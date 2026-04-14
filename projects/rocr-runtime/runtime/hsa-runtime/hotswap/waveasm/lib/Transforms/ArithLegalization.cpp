// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Arithmetic Legalization Pass
//
// Lowers generic arithmetic pseudo-ops (arith.add, arith.mul, arith.cmp,
// arith.select, arith.trunc, arith.sext, arith.zext) to concrete SALU or
// VALU machine ops based on operand register files and widths (i32/i64).
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMARITHLEGALIZATION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

using namespace mlir;
using namespace waveasm;

//===----------------------------------------------------------------------===//
// Width and register-file helpers
//===----------------------------------------------------------------------===//

/// Return register width in 32-bit units (1 = i32, 2 = i64).
/// Returns 0 for unsupported types.
static int64_t getRegWidth(Value v) {
  return TypeSwitch<Type, int64_t>(v.getType())
      .Case<SRegType, VRegType, PSRegType, PVRegType>(
          [](auto t) { return t.getSize(); })
      .Case<ImmType>([](ImmType) { return int64_t(1); })
      .Default([](Type) { return int64_t(0); });
}

/// Return true if any operand is a VGPR (divergent context).
static bool anyVGPR(ValueRange operands) {
  return llvm::any_of(operands,
                      [](Value v) { return isVGPRType(v.getType()); });
}

/// Validate that width is exactly 1 (i32) or 2 (i64).
/// Returns failure and emits an error on the op otherwise.
static LogicalResult checkWidth(Operation *op, int64_t width) {
  if (width == 1 || width == 2)
    return success();
  op->emitError("unsupported operand width (expected i32 or i64, got ")
      << width << " dwords)";
  return failure();
}

//===----------------------------------------------------------------------===//
// i64 split/merge helpers
//===----------------------------------------------------------------------===//

/// Split an i64 (size-2) register into {lo, hi} i32 halves.
static std::pair<Value, Value> splitI64(Value v, OpBuilder &builder,
                                        Location loc) {
  // Look through pack ops to avoid extract/pack round-trips.
  // The register allocator does not insert copies for pack, so extracting
  // from a pack whose inputs are at different physical registers would
  // read stale data.
  if (auto pack = v.getDefiningOp<PackOp>()) {
    auto operands = pack.getOperands();
    assert(operands.size() == 2 && "expected 2-element pack for i64");
    return {operands[0], operands[1]};
  }
  if (isSGPRType(v.getType())) {
    // For precolored SGPRs, create precolored extracts at known indices.
    if (auto psreg = v.getDefiningOp<PrecoloredSRegOp>()) {
      auto sregTy = SRegType::get(builder.getContext(), 1, 1);
      Value lo = PrecoloredSRegOp::create(builder, loc, sregTy,
                                          psreg.getIndex(), /*size=*/1);
      Value hi = PrecoloredSRegOp::create(builder, loc, sregTy,
                                          psreg.getIndex() + 1, /*size=*/1);
      return {lo, hi};
    }
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    Value lo = ExtractOp::create(builder, loc, sregTy, v, 0);
    Value hi = ExtractOp::create(builder, loc, sregTy, v, 1);
    return {lo, hi};
  }
  auto vregTy = VRegType::get(builder.getContext());
  Value lo = ExtractOp::create(builder, loc, vregTy, v, 0);
  Value hi = ExtractOp::create(builder, loc, vregTy, v, 1);
  return {lo, hi};
}

/// Merge {lo, hi} i32 values into an i64 (size-2) register.
static Value mergeI64(Value lo, Value hi, OpBuilder &builder, Location loc) {
  if (isSGPRType(lo.getType())) {
    auto sregTy = SRegType::get(builder.getContext(), 2, 2);
    return PackOp::create(builder, loc, sregTy, ValueRange{lo, hi});
  }
  auto vregTy = VRegType::get(builder.getContext(), 2);
  return PackOp::create(builder, loc, vregTy, ValueRange{lo, hi});
}

//===----------------------------------------------------------------------===//
// Register file conversion helpers
//===----------------------------------------------------------------------===//

/// Move an SGPR i32 value to a VGPR via v_mov_b32.
static Value sgprToVgpr(Value v, OpBuilder &builder, Location loc) {
  if (!isSGPRType(v.getType()))
    return v;
  auto vregTy = VRegType::get(builder.getContext());
  return V_MOV_B32::create(builder, loc, vregTy, v);
}

//===----------------------------------------------------------------------===//
// i32 legalization
//===----------------------------------------------------------------------===//

static void legalizeAddI32(Value lhs, Value rhs, ArithAddOp op,
                           OpBuilder &builder) {
  Location loc = op.getLoc();
  Value result;
  if (anyVGPR({lhs, rhs})) {
    lhs = sgprToVgpr(lhs, builder, loc);
    auto vregTy = VRegType::get(builder.getContext());
    result = V_ADD_U32::create(builder, loc, vregTy, lhs, rhs);
  } else {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    auto sccTy = SCCType::get(builder.getContext());
    result = S_ADD_U32::create(builder, loc, sregTy, sccTy, lhs, rhs).getDst();
  }
  op.replaceAllUsesWith(result);
}

static void legalizeMulI32(Value lhs, Value rhs, ArithMulOp op,
                           OpBuilder &builder) {
  Location loc = op.getLoc();
  Value result;
  if (anyVGPR({lhs, rhs})) {
    lhs = sgprToVgpr(lhs, builder, loc);
    auto vregTy = VRegType::get(builder.getContext());
    result = V_MUL_LO_U32::create(builder, loc, vregTy, lhs, rhs);
  } else {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    result = S_MUL_I32::create(builder, loc, sregTy, lhs, rhs);
  }
  op.replaceAllUsesWith(result);
}

//===----------------------------------------------------------------------===//
// i64 legalization
//===----------------------------------------------------------------------===//

/// i64 add via carry chain: s_add_u32 + s_addc_u32 (SALU) or
/// v_add_co_u32 + v_addc_co_u32 (VALU).
/// NOTE: The carry between the two ops is implicit (SCC/VCC). They must
/// remain adjacent -- do not schedule or insert ops between them.
static void legalizeAddI64(Value lhs, Value rhs, ArithAddOp op,
                           OpBuilder &builder) {
  Location loc = op.getLoc();
  auto [lhsLo, lhsHi] = splitI64(lhs, builder, loc);
  auto [rhsLo, rhsHi] = splitI64(rhs, builder, loc);

  Value loResult, hiResult;
  if (anyVGPR({lhs, rhs})) {
    lhsLo = sgprToVgpr(lhsLo, builder, loc);
    lhsHi = sgprToVgpr(lhsHi, builder, loc);
    auto vregTy = VRegType::get(builder.getContext());
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    // v_add_co_u32: lo + lo, carry out to VCC.
    auto addLo =
        V_ADD_CO_U32::create(builder, loc, vregTy, sregTy, lhsLo, rhsLo);
    loResult = addLo.getDst();
    // v_addc_co_u32: hi + hi + carry in from VCC.
    auto addHi =
        V_ADDC_CO_U32::create(builder, loc, vregTy, sregTy, lhsHi, rhsHi);
    hiResult = addHi.getDst();
  } else {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    auto sccTy = SCCType::get(builder.getContext());
    // s_add_u32: lo + lo, carry out to SCC.
    auto addLo = S_ADD_U32::create(builder, loc, sregTy, sccTy, lhsLo, rhsLo);
    loResult = addLo.getDst();
    // s_addc_u32: hi + hi + carry in from SCC.
    auto addHi = S_ADDC_U32::create(builder, loc, sregTy, sccTy, addLo.getScc(),
                                    lhsHi, rhsHi);
    hiResult = addHi.getDst();
  }

  Value result = mergeI64(loResult, hiResult, builder, loc);
  op.replaceAllUsesWith(result);
}

/// i64 multiply via schoolbook decomposition:
///   result_lo = mul_lo(a_lo, b_lo)
///   result_hi = mul_hi(a_lo, b_lo) + mul_lo(a_lo, b_hi) + mul_lo(a_hi, b_lo)
static void legalizeMulI64(Value lhs, Value rhs, ArithMulOp op,
                           OpBuilder &builder) {
  Location loc = op.getLoc();
  auto [aLo, aHi] = splitI64(lhs, builder, loc);
  auto [bLo, bHi] = splitI64(rhs, builder, loc);

  Value loResult, hiResult;
  if (anyVGPR({lhs, rhs})) {
    aLo = sgprToVgpr(aLo, builder, loc);
    aHi = sgprToVgpr(aHi, builder, loc);
    auto vregTy = VRegType::get(builder.getContext());
    // lo = mul_lo(a_lo, b_lo).
    loResult = V_MUL_LO_U32::create(builder, loc, vregTy, aLo, bLo);
    // hi = mul_hi(a_lo, b_lo) + mul_lo(a_lo, b_hi) + mul_lo(a_hi, b_lo).
    Value hiPartial = V_MUL_HI_U32::create(builder, loc, vregTy, aLo, bLo);
    Value cross1 = V_MUL_LO_U32::create(builder, loc, vregTy, aLo, bHi);
    Value cross2 = V_MUL_LO_U32::create(builder, loc, vregTy, aHi, bLo);
    // Accumulate with v_add3_u32 (3-input add, no carry needed since we
    // discard bits above 64).
    hiResult =
        V_ADD3_U32::create(builder, loc, vregTy, hiPartial, cross1, cross2);
  } else {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    // lo = mul_lo(a_lo, b_lo).
    loResult = S_MUL_I32::create(builder, loc, sregTy, aLo, bLo);
    // hi = mul_hi(a_lo, b_lo) + mul_lo(a_lo, b_hi) + mul_lo(a_hi, b_lo).
    Value hiPartial = S_MUL_HI_U32::create(builder, loc, sregTy, aLo, bLo);
    Value cross1 = S_MUL_I32::create(builder, loc, sregTy, aLo, bHi);
    Value cross2 = S_MUL_I32::create(builder, loc, sregTy, aHi, bLo);
    // Accumulate (carry discarded -- computing mod 2^64).
    auto sccTy = SCCType::get(builder.getContext());
    Value hiTemp =
        S_ADD_U32::create(builder, loc, sregTy, sccTy, hiPartial, cross1)
            .getDst();
    hiResult =
        S_ADD_U32::create(builder, loc, sregTy, sccTy, hiTemp, cross2).getDst();
  }

  Value result = mergeI64(loResult, hiResult, builder, loc);
  op.replaceAllUsesWith(result);
}

/// Emit a VCC-setting compare and return a VCC placeholder.
static Value emitVCmp(CmpPredicate pred, Value lhs, Value rhs,
                      OpBuilder &builder, Location loc) {
  switch (pred) {
  case CmpPredicate::eq:
    V_CMP_EQ_I32::create(builder, loc, lhs, rhs);
    break;
  case CmpPredicate::ne:
    V_CMP_NE_I32::create(builder, loc, lhs, rhs);
    break;
  case CmpPredicate::slt:
    V_CMP_LT_I32::create(builder, loc, lhs, rhs);
    break;
  case CmpPredicate::sle:
    V_CMP_LE_I32::create(builder, loc, lhs, rhs);
    break;
  case CmpPredicate::sgt:
    V_CMP_GT_I32::create(builder, loc, lhs, rhs);
    break;
  case CmpPredicate::sge:
    V_CMP_GE_I32::create(builder, loc, lhs, rhs);
    break;
  case CmpPredicate::ult:
    V_CMP_LT_U32::create(builder, loc, lhs, rhs);
    break;
  case CmpPredicate::ule:
    V_CMP_LE_U32::create(builder, loc, lhs, rhs);
    break;
  case CmpPredicate::ugt:
    V_CMP_GT_U32::create(builder, loc, lhs, rhs);
    break;
  case CmpPredicate::uge:
    V_CMP_GE_U32::create(builder, loc, lhs, rhs);
    break;
  }
  auto ty = ImmType::get(builder.getContext(), 1);
  return ConstantOp::create(builder, loc, ty, 1);
}

/// Emit an SCC-setting compare and return the sreg result.
static Value emitSCmp(CmpPredicate pred, Value lhs, Value rhs,
                      OpBuilder &builder, Location loc) {
  auto sregTy = SCCType::get(builder.getContext());
  switch (pred) {
  case CmpPredicate::eq:
    return S_CMP_EQ_I32::create(builder, loc, sregTy, lhs, rhs);
  case CmpPredicate::ne:
    return S_CMP_NE_I32::create(builder, loc, sregTy, lhs, rhs);
  case CmpPredicate::slt:
    return S_CMP_LT_I32::create(builder, loc, sregTy, lhs, rhs);
  case CmpPredicate::sle:
    return S_CMP_LE_I32::create(builder, loc, sregTy, lhs, rhs);
  case CmpPredicate::sgt:
    return S_CMP_GT_I32::create(builder, loc, sregTy, lhs, rhs);
  case CmpPredicate::sge:
    return S_CMP_GE_I32::create(builder, loc, sregTy, lhs, rhs);
  case CmpPredicate::ult:
    return S_CMP_LT_U32::create(builder, loc, sregTy, lhs, rhs);
  case CmpPredicate::ule:
    return S_CMP_LE_U32::create(builder, loc, sregTy, lhs, rhs);
  case CmpPredicate::ugt:
    return S_CMP_GT_U32::create(builder, loc, sregTy, lhs, rhs);
  case CmpPredicate::uge:
    return S_CMP_GE_U32::create(builder, loc, sregTy, lhs, rhs);
  }
  llvm_unreachable("unhandled CmpPredicate");
}

/// Get hi/lo predicates for ordered i64 comparison.
/// Hi uses the strict less/greater variant with same signedness.
/// Lo always uses unsigned (lo halves have no sign meaning).
static std::pair<CmpPredicate, CmpPredicate>
getOrderedI64Preds(CmpPredicate pred) {
  switch (pred) {
  case CmpPredicate::slt:
    return {CmpPredicate::slt, CmpPredicate::ult};
  case CmpPredicate::sle:
    return {CmpPredicate::slt, CmpPredicate::ule};
  case CmpPredicate::sgt:
    return {CmpPredicate::sgt, CmpPredicate::ugt};
  case CmpPredicate::sge:
    return {CmpPredicate::sgt, CmpPredicate::uge};
  case CmpPredicate::ult:
    return {CmpPredicate::ult, CmpPredicate::ult};
  case CmpPredicate::ule:
    return {CmpPredicate::ult, CmpPredicate::ule};
  case CmpPredicate::ugt:
    return {CmpPredicate::ugt, CmpPredicate::ugt};
  case CmpPredicate::uge:
    return {CmpPredicate::ugt, CmpPredicate::uge};
  default:
    llvm_unreachable("not an ordered predicate");
  }
}

/// i64 compare: eq/ne via XOR+OR, ordered via hi/lo decomposition.
///
/// Ordered strategy: result = hiPred(hi_a, hi_b) ||
///                            (hi_a == hi_b && loPred(lo_a, lo_b))
/// Hi comparison uses same signedness as original predicate (strict variant).
/// Lo comparison always uses unsigned (lo halves have no sign meaning).
static LogicalResult legalizeCmpI64(Value lhs, Value rhs, CmpPredicate pred,
                                    ArithCmpOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  auto [lhsLo, lhsHi] = splitI64(lhs, builder, loc);
  auto [rhsLo, rhsHi] = splitI64(rhs, builder, loc);

  bool isEqNe = pred == CmpPredicate::eq || pred == CmpPredicate::ne;

  if (anyVGPR({lhs, rhs})) {
    lhsLo = sgprToVgpr(lhsLo, builder, loc);
    lhsHi = sgprToVgpr(lhsHi, builder, loc);
    auto vregTy = VRegType::get(builder.getContext());

    if (isEqNe) {
      // eq/ne: XOR each half, OR, compare to zero.
      // Materialize to vreg so VCC can be re-established by the consumer.
      Value xorLo = V_XOR_B32::create(builder, loc, vregTy, lhsLo, rhsLo);
      Value xorHi = V_XOR_B32::create(builder, loc, vregTy, lhsHi, rhsHi);
      Value combined = V_OR_B32::create(builder, loc, vregTy, xorLo, xorHi);
      auto zeroTy = ImmType::get(builder.getContext(), 0);
      Value zero = ConstantOp::create(builder, loc, zeroTy, 0);
      auto oneTy = ImmType::get(builder.getContext(), 1);
      Value one = ConstantOp::create(builder, loc, oneTy, 1);
      if (pred == CmpPredicate::eq)
        V_CMP_EQ_I32::create(builder, loc, combined, zero);
      else
        V_CMP_NE_I32::create(builder, loc, combined, zero);
      auto vccPlaceholder = ConstantOp::create(builder, loc, oneTy, 1);
      Value boolResult = V_CNDMASK_B32::create(builder, loc, vregTy, zero, one,
                                               vccPlaceholder);
      op.replaceAllUsesWith(boolResult);
      op.erase();
      return success();
    } else {
      // Ordered: materialize hi/lo results, select based on hi equality.
      auto [hiPred, loPred] = getOrderedI64Preds(pred);
      auto zeroTy = ImmType::get(builder.getContext(), 0);
      Value zero = ConstantOp::create(builder, loc, zeroTy, 0);
      auto oneTy = ImmType::get(builder.getContext(), 1);
      Value one = ConstantOp::create(builder, loc, oneTy, 1);

      // Materialize hi comparison to vreg.
      Value hiVcc = emitVCmp(hiPred, lhsHi, rhsHi, builder, loc);
      Value hiRes =
          V_CNDMASK_B32::create(builder, loc, vregTy, zero, one, hiVcc);

      // Materialize lo comparison to vreg.
      Value loVcc = emitVCmp(loPred, lhsLo, rhsLo, builder, loc);
      Value loRes =
          V_CNDMASK_B32::create(builder, loc, vregTy, zero, one, loVcc);

      // Select: if hi equal, use lo result; else use hi result.
      Value eqVcc = emitVCmp(CmpPredicate::eq, lhsHi, rhsHi, builder, loc);
      Value finalBool =
          V_CNDMASK_B32::create(builder, loc, vregTy, hiRes, loRes, eqVcc);

      // Return the materialized boolean vreg so the consumer (select) can
      // re-establish VCC right before its v_cndmask_b32.  Setting VCC here
      // is unsafe because intervening i64 ops (v_add_co_u32) may clobber it.
      op.replaceAllUsesWith(finalBool);
      op.erase();
      return success();
    }
  } else {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);

    if (isEqNe) {
      // eq/ne: XOR each half, OR, compare to zero.
      auto sccTy = SCCType::get(builder.getContext());
      Value xorLo =
          S_XOR_B32::create(builder, loc, sregTy, sccTy, lhsLo, rhsLo).getDst();
      Value xorHi =
          S_XOR_B32::create(builder, loc, sregTy, sccTy, lhsHi, rhsHi).getDst();
      Value combined =
          S_OR_B32::create(builder, loc, sregTy, sccTy, xorLo, xorHi).getDst();
      auto immTy = ImmType::get(builder.getContext(), 0);
      Value zero = ConstantOp::create(builder, loc, immTy, 0);
      Value result;
      if (pred == CmpPredicate::eq)
        result = S_CMP_EQ_I32::create(builder, loc, sccTy, combined, zero);
      else
        result = S_CMP_NE_I32::create(builder, loc, sccTy, combined, zero);
      op.replaceAllUsesWith(result);
    } else {
      // Ordered: result = hiPred(hi) | (hiEq(hi) & loPred(lo)).
      // Each S_CMP sets SCC (returns SCCType). Materialize each to a
      // 0/1 SGPR via s_cselect_b32 before combining with AND/OR.
      auto [hiPred, loPred] = getOrderedI64Preds(pred);
      auto immTy0 = ImmType::get(builder.getContext(), 0);
      auto immTy1 = ImmType::get(builder.getContext(), 1);
      Value zero = ConstantOp::create(builder, loc, immTy0, 0);
      Value one = ConstantOp::create(builder, loc, immTy1, 1);
      Value oneSgpr = S_MOV_B32::create(builder, loc, sregTy, one);

      Value hiSCC = emitSCmp(hiPred, lhsHi, rhsHi, builder, loc);
      Value hiCmp =
          S_CSELECT_B32::create(builder, loc, sregTy, hiSCC, oneSgpr, zero);

      Value eqSCC = emitSCmp(CmpPredicate::eq, lhsHi, rhsHi, builder, loc);
      Value hiEq =
          S_CSELECT_B32::create(builder, loc, sregTy, eqSCC, oneSgpr, zero);

      Value loSCC = emitSCmp(loPred, lhsLo, rhsLo, builder, loc);
      Value loCmp =
          S_CSELECT_B32::create(builder, loc, sregTy, loSCC, oneSgpr, zero);

      auto sccTy2 = SCCType::get(builder.getContext());
      Value eqAndLo =
          S_AND_B32::create(builder, loc, sregTy, sccTy2, hiEq, loCmp).getDst();
      Value result =
          S_OR_B32::create(builder, loc, sregTy, sccTy2, hiCmp, eqAndLo)
              .getDst();
      op.replaceAllUsesWith(result);
    }
  }
  op.erase();
  return success();
}

/// If the condition is a materialized boolean VGPR (from an i64 compare),
/// re-establish VCC right before the v_cndmask_b32 that consumes it.
/// Returns a VCC placeholder suitable for v_cndmask_b32's condition operand.
static Value ensureVCC(Value cond, OpBuilder &builder, Location loc) {
  if (isSGPRType(cond.getType()))
    cond = sgprToVgpr(cond, builder, loc);
  if (isVGPRType(cond.getType())) {
    auto zeroTy = ImmType::get(builder.getContext(), 0);
    Value zero = ConstantOp::create(builder, loc, zeroTy, 0);
    V_CMP_NE_I32::create(builder, loc, cond, zero);
  }
  auto placeholderTy = ImmType::get(builder.getContext(), 1);
  return ConstantOp::create(builder, loc, placeholderTy, 1);
}

/// i64 select: split both operands, select each half, merge.
static void legalizeSelectI64(Value trueVal, Value falseVal, Value cond,
                              ArithSelectOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  auto [trueLo, trueHi] = splitI64(trueVal, builder, loc);
  auto [falseLo, falseHi] = splitI64(falseVal, builder, loc);

  auto vregTy = VRegType::get(builder.getContext());
  falseLo = sgprToVgpr(falseLo, builder, loc);
  trueLo = sgprToVgpr(trueLo, builder, loc);
  falseHi = sgprToVgpr(falseHi, builder, loc);
  trueHi = sgprToVgpr(trueHi, builder, loc);

  Value vccCond = ensureVCC(cond, builder, loc);
  Value selLo =
      V_CNDMASK_B32::create(builder, loc, vregTy, falseLo, trueLo, vccCond);
  Value selHi =
      V_CNDMASK_B32::create(builder, loc, vregTy, falseHi, trueHi, vccCond);
  Value result = mergeI64(selLo, selHi, builder, loc);
  op.replaceAllUsesWith(result);
}

//===----------------------------------------------------------------------===//
// Dispatch functions (i32 vs i64)
//===----------------------------------------------------------------------===//

static LogicalResult legalizeAdd(ArithAddOp op, OpBuilder &builder) {
  int64_t width = getRegWidth(op.getLhs());
  if (failed(checkWidth(op, width)))
    return failure();
  if (width == 2)
    legalizeAddI64(op.getLhs(), op.getRhs(), op, builder);
  else
    legalizeAddI32(op.getLhs(), op.getRhs(), op, builder);
  op.erase();
  return success();
}

static LogicalResult legalizeMul(ArithMulOp op, OpBuilder &builder) {
  int64_t width = getRegWidth(op.getLhs());
  if (failed(checkWidth(op, width)))
    return failure();
  if (width == 2)
    legalizeMulI64(op.getLhs(), op.getRhs(), op, builder);
  else
    legalizeMulI32(op.getLhs(), op.getRhs(), op, builder);
  op.erase();
  return success();
}

/// Legalize a generic bitwise op (or, and) to SALU/VALU concrete ops.
/// Template parameters: the ArithPseudoOp, and the four machine ops
/// (SALU i32, SALU i64, VALU i32, VALU i64).
template <typename ArithOp, typename SALUOp32, typename SALUOp64,
          typename VALUOp32, typename VALUOp64>
static LogicalResult legalizeBitwiseOp(ArithOp op, OpBuilder &builder) {
  int64_t width = getRegWidth(op.getLhs());
  if (failed(checkWidth(op, width)))
    return failure();

  Location loc = op.getLoc();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value result;

  if (width == 2) {
    // i64: bitwise ops have native s_*_b64 / v_*_b64 instructions.
    if (anyVGPR({lhs, rhs})) {
      lhs = sgprToVgpr(lhs, builder, loc);
      auto vregTy = VRegType::get(builder.getContext(), 2);
      result = VALUOp64::create(builder, loc, vregTy, lhs, rhs);
    } else {
      auto sregTy = SRegType::get(builder.getContext(), 2, 2);
      auto sccTy = SCCType::get(builder.getContext());
      result = SALUOp64::create(builder, loc, sregTy, sccTy, lhs, rhs).getDst();
    }
  } else {
    if (anyVGPR({lhs, rhs})) {
      lhs = sgprToVgpr(lhs, builder, loc);
      auto vregTy = VRegType::get(builder.getContext());
      result = VALUOp32::create(builder, loc, vregTy, lhs, rhs);
    } else {
      auto sregTy = SRegType::get(builder.getContext(), 1, 1);
      auto sccTy = SCCType::get(builder.getContext());
      result = SALUOp32::create(builder, loc, sregTy, sccTy, lhs, rhs).getDst();
    }
  }
  op.replaceAllUsesWith(result);
  op.erase();
  return success();
}

static LogicalResult legalizeCmp(ArithCmpOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  int64_t width = getRegWidth(op.getLhs());
  if (failed(checkWidth(op, width)))
    return failure();

  if (width == 2)
    return legalizeCmpI64(op.getLhs(), op.getRhs(), op.getPredicate(), op,
                          builder);

  // i32 path.
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  auto pred = op.getPredicate();

  if (anyVGPR({lhs, rhs})) {
    if (isSGPRType(lhs.getType()))
      lhs = sgprToVgpr(lhs, builder, loc);

    switch (pred) {
    case CmpPredicate::eq:
      V_CMP_EQ_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::ne:
      V_CMP_NE_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::slt:
      V_CMP_LT_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::sle:
      V_CMP_LE_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::sgt:
      V_CMP_GT_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::sge:
      V_CMP_GE_I32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::ult:
      V_CMP_LT_U32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::ule:
      V_CMP_LE_U32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::ugt:
      V_CMP_GT_U32::create(builder, loc, lhs, rhs);
      break;
    case CmpPredicate::uge:
      V_CMP_GE_U32::create(builder, loc, lhs, rhs);
      break;
    }
    // VCC-setting compares have no explicit result. Create a placeholder
    // constant for uses (v_cndmask_b32 reads VCC implicitly).
    auto immTy = ImmType::get(builder.getContext(), 1);
    auto placeholder = ConstantOp::create(builder, loc, immTy, 1);
    op.replaceAllUsesWith(placeholder.getResult());
  } else {
    auto sccTy = SCCType::get(builder.getContext());
    Value result;
    switch (pred) {
    case CmpPredicate::eq:
      result = S_CMP_EQ_I32::create(builder, loc, sccTy, lhs, rhs);
      break;
    case CmpPredicate::ne:
      result = S_CMP_NE_I32::create(builder, loc, sccTy, lhs, rhs);
      break;
    case CmpPredicate::slt:
      result = S_CMP_LT_I32::create(builder, loc, sccTy, lhs, rhs);
      break;
    case CmpPredicate::sle:
      result = S_CMP_LE_I32::create(builder, loc, sccTy, lhs, rhs);
      break;
    case CmpPredicate::sgt:
      result = S_CMP_GT_I32::create(builder, loc, sccTy, lhs, rhs);
      break;
    case CmpPredicate::sge:
      result = S_CMP_GE_I32::create(builder, loc, sccTy, lhs, rhs);
      break;
    case CmpPredicate::ult:
      result = S_CMP_LT_U32::create(builder, loc, sccTy, lhs, rhs);
      break;
    case CmpPredicate::ule:
      result = S_CMP_LE_U32::create(builder, loc, sccTy, lhs, rhs);
      break;
    case CmpPredicate::ugt:
      result = S_CMP_GT_U32::create(builder, loc, sccTy, lhs, rhs);
      break;
    case CmpPredicate::uge:
      result = S_CMP_GE_U32::create(builder, loc, sccTy, lhs, rhs);
      break;
    }
    op.replaceAllUsesWith(result);
  }
  op.erase();
  return success();
}

static LogicalResult legalizeSelect(ArithSelectOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Value falseVal = op.getFalseVal();
  Value trueVal = op.getTrueVal();
  Value cond = op.getCondition();
  int64_t width = getRegWidth(trueVal);
  if (failed(checkWidth(op, width)))
    return failure();

  if (width == 2) {
    legalizeSelectI64(trueVal, falseVal, cond, op, builder);
  } else {
    auto vregTy = VRegType::get(builder.getContext());
    falseVal = sgprToVgpr(falseVal, builder, loc);
    trueVal = sgprToVgpr(trueVal, builder, loc);
    Value vccCond = ensureVCC(cond, builder, loc);
    auto sel =
        V_CNDMASK_B32::create(builder, loc, vregTy, falseVal, trueVal, vccCond);
    op.replaceAllUsesWith(sel.getResult());
  }
  op.erase();
  return success();
}

static LogicalResult legalizeReadFirstLane(ArithReadFirstLaneOp op,
                                           OpBuilder &builder) {
  Value src = op.getSrc();
  Location loc = op.getLoc();

  // Already scalar -- just forward.
  if (isSGPRType(src.getType())) {
    op.replaceAllUsesWith(src);
    op.erase();
    return success();
  }

  int64_t width = getRegWidth(src);
  if (width == 1) {
    SRegType sregTy = SRegType::get(builder.getContext(), /*size=*/1,
                                    /*alignment=*/1);
    Value lane = V_READFIRSTLANE_B32::create(builder, loc, sregTy, src);
    op.replaceAllUsesWith(lane);
    op.erase();
    return success();
  }

  if (width == 2) {
    auto [lo, hi] = splitI64(src, builder, loc);
    SRegType sregTy = SRegType::get(builder.getContext(), /*size=*/1,
                                    /*alignment=*/1);
    Value loS = V_READFIRSTLANE_B32::create(builder, loc, sregTy, lo);
    Value hiS = V_READFIRSTLANE_B32::create(builder, loc, sregTy, hi);
    SRegType sregPairTy = SRegType::get(builder.getContext(), /*size=*/2,
                                        /*alignment=*/2);
    Value packed =
        PackOp::create(builder, loc, sregPairTy, ValueRange{loS, hiS});
    op.replaceAllUsesWith(packed);
    op.erase();
    return success();
  }

  return op.emitError("unsupported readfirstlane width (got ")
         << width << " dwords)";
}

static LogicalResult legalizeTrunc(ArithTruncOp op, OpBuilder &builder) {
  Value src = op.getSrc();
  int64_t width = getRegWidth(src);
  if (width < 2) {
    // Already i32 or narrower -- pass through.
    op.replaceAllUsesWith(src);
    op.erase();
    return success();
  }
  if (failed(checkWidth(op, width)))
    return failure();

  Location loc = op.getLoc();
  // For precolored SGPRs, create a precolored reference to the lo half.
  if (auto psreg = src.getDefiningOp<PrecoloredSRegOp>()) {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    Value lo = PrecoloredSRegOp::create(builder, loc, sregTy, psreg.getIndex(),
                                        /*size=*/1);
    op.replaceAllUsesWith(lo);
    op.erase();
    return success();
  }
  auto [lo, hi] = splitI64(src, builder, loc);
  (void)hi;
  op.replaceAllUsesWith(lo);
  op.erase();
  return success();
}

static LogicalResult legalizeSExt(ArithSExtOp op, OpBuilder &builder) {
  Value src = op.getSrc();
  int64_t srcWidth = getRegWidth(src);
  if (srcWidth != 1) {
    op.emitError("sext source must be i32 (got ") << srcWidth << " dwords)";
    return failure();
  }

  Location loc = op.getLoc();
  // hi = arithmetic shift right by 31 (sign-fill).
  if (isSGPRType(src.getType())) {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    auto immTy = ImmType::get(builder.getContext(), 31);
    Value shift = ConstantOp::create(builder, loc, immTy, 31);
    auto sccTy = SCCType::get(builder.getContext());
    Value hi =
        S_ASHR_I32::create(builder, loc, sregTy, sccTy, src, shift).getDst();
    Value result = mergeI64(src, hi, builder, loc);
    op.replaceAllUsesWith(result);
  } else {
    auto vregTy = VRegType::get(builder.getContext());
    auto immTy = ImmType::get(builder.getContext(), 31);
    Value shift = ConstantOp::create(builder, loc, immTy, 31);
    // v_ashrrev_i32: dst = src >> shift (reversed operand order).
    Value hi = V_ASHRREV_I32::create(builder, loc, vregTy, shift, src);
    Value result = mergeI64(src, hi, builder, loc);
    op.replaceAllUsesWith(result);
  }
  op.erase();
  return success();
}

static LogicalResult legalizeZExt(ArithZExtOp op, OpBuilder &builder) {
  Value src = op.getSrc();
  int64_t srcWidth = getRegWidth(src);
  if (srcWidth != 1) {
    op.emitError("zext source must be i32 (got ") << srcWidth << " dwords)";
    return failure();
  }

  Location loc = op.getLoc();
  // hi = 0.
  if (isSGPRType(src.getType())) {
    auto sregTy = SRegType::get(builder.getContext(), 1, 1);
    auto immTy = ImmType::get(builder.getContext(), 0);
    Value zero = ConstantOp::create(builder, loc, immTy, 0);
    Value hi = S_MOV_B32::create(builder, loc, sregTy, zero);
    Value result = mergeI64(src, hi, builder, loc);
    op.replaceAllUsesWith(result);
  } else {
    auto vregTy = VRegType::get(builder.getContext());
    auto immTy = ImmType::get(builder.getContext(), 0);
    Value zero = ConstantOp::create(builder, loc, immTy, 0);
    Value hi = V_MOV_B32::create(builder, loc, vregTy, zero);
    Value result = mergeI64(src, hi, builder, loc);
    op.replaceAllUsesWith(result);
  }
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

struct ArithLegalizationPass
    : waveasm::impl::WAVEASMArithLegalizationBase<ArithLegalizationPass> {

  void runOnOperation() override {
    // Post-order walk is safe for in-place erasure of the current op.
    WalkResult walkResult = getOperation()->walk([&](Operation *op) {
      if (!isa<ArithAddOp, ArithMulOp, ArithOrOp, ArithAndOp, ArithCmpOp,
               ArithSelectOp, ArithReadFirstLaneOp, ArithTruncOp, ArithSExtOp,
               ArithZExtOp>(op))
        return WalkResult::advance();

      OpBuilder builder(op);
      LogicalResult result =
          TypeSwitch<Operation *, LogicalResult>(op)
              .Case([&](ArithAddOp o) { return legalizeAdd(o, builder); })
              .Case([&](ArithMulOp o) { return legalizeMul(o, builder); })
              .Case([&](ArithOrOp o) {
                return legalizeBitwiseOp<ArithOrOp, S_OR_B32, S_OR_B64,
                                         V_OR_B32, V_OR_B64>(o, builder);
              })
              .Case([&](ArithAndOp o) {
                return legalizeBitwiseOp<ArithAndOp, S_AND_B32, S_AND_B64,
                                         V_AND_B32, V_AND_B64>(o, builder);
              })
              .Case([&](ArithCmpOp o) { return legalizeCmp(o, builder); })
              .Case([&](ArithSelectOp o) { return legalizeSelect(o, builder); })
              .Case([&](ArithReadFirstLaneOp o) {
                return legalizeReadFirstLane(o, builder);
              })
              .Case([&](ArithTruncOp o) { return legalizeTrunc(o, builder); })
              .Case([&](ArithSExtOp o) { return legalizeSExt(o, builder); })
              .Case([&](ArithZExtOp o) { return legalizeZExt(o, builder); })
              .Default([](Operation *) { return success(); });
      if (mlir::failed(result))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
