// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Peephole Optimization Pass
//
// This pass implements local pattern-based peephole optimizations for WAVEASM
// IR: instruction fusion, strength reduction, constant folding, and redundant
// move elimination.  Region-level optimizations (LICM, M0 redundancy) live in
// their own passes.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMPEEPHOLE
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

using namespace mlir;
using namespace waveasm;

namespace {

//===----------------------------------------------------------------------===//
// Pattern: lshlrev + or -> v_lshl_or_b32
//
// Fuses:
//   %shifted = v_lshlrev_b32 %shift_amt, %src
//   %result = v_or_b32 %shifted, %orend
// Into:
//   %result = v_lshl_or_b32 %src, %shift_amt, %orend
//
// This is a common pattern in address calculations where bit ranges
// don't overlap, allowing OR to be used instead of ADD.
//===----------------------------------------------------------------------===//

struct LshlOrFusePattern : public OpRewritePattern<V_OR_B32> {
  using OpRewritePattern<V_OR_B32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_OR_B32 orOp,
                                PatternRewriter &rewriter) const override {
    // Check if either operand is the result of a left shift
    auto checkOperand = [&](Value shifted, Value orend) -> LogicalResult {
      auto lshlOp = shifted.getDefiningOp<V_LSHLREV_B32>();
      if (!lshlOp)
        return failure();

      // Check that the shift has only one use (this or)
      if (!lshlOp.getResult().hasOneUse())
        return failure();

      // Get the shift amount - must be a constant for efficiency
      auto constOp = lshlOp.getSrc0().getDefiningOp<ConstantOp>();
      if (!constOp)
        return failure();

      // v_lshl_or_b32 operands: src (value to shift), shift_amt, orend
      // In v_lshlrev_b32: src0 = shift_amt, src1 = value to shift
      Value shiftSrc = lshlOp.getSrc1(); // value being shifted
      Value shiftAmt = lshlOp.getSrc0(); // shift amount

      // VOP3 constant bus restriction: at most one distinct SGPR source.
      if (isSGPRType(shiftSrc.getType()) && isSGPRType(orend.getType()) &&
          shiftSrc != orend)
        return failure();

      auto loc = orOp.getLoc();
      auto resultType = orOp.getResult().getType();

      // Create v_lshl_or_b32: dst = (src << shift_amt) | orend
      auto fusedOp = V_LSHL_OR_B32::create(rewriter, loc, resultType, shiftSrc,
                                           shiftAmt, orend);

      rewriter.replaceOp(orOp, fusedOp.getResult());
      return success();
    };

    // Try both operand orderings
    if (succeeded(checkOperand(orOp.getSrc0(), orOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(orOp.getSrc1(), orOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: lshl + add -> v_lshl_add_u32
//
// Fuses:
//   %shifted = v_lshlrev_b32 %shift_amt, %base
//   %result = v_add_u32 %shifted, %addend
// Into:
//   %result = v_lshl_add_u32 %base, %shift_amt, %addend
//
// This matches Python's use of the fused shift-add instruction.
//===----------------------------------------------------------------------===//

struct LshlAddPattern : public OpRewritePattern<V_ADD_U32> {
  using OpRewritePattern<V_ADD_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_ADD_U32 addOp,
                                PatternRewriter &rewriter) const override {
    // Check if either operand is the result of a left shift
    auto checkOperand = [&](Value shifted, Value other) -> LogicalResult {
      auto lshlOp = shifted.getDefiningOp<V_LSHLREV_B32>();
      if (!lshlOp)
        return failure();

      // Check that the shift has only one use (this add)
      if (!lshlOp.getResult().hasOneUse())
        return failure();

      // Get the shift amount - must be a constant for v_lshl_add_u32
      auto constOp = lshlOp.getSrc0().getDefiningOp<ConstantOp>();
      if (!constOp)
        return failure();

      int64_t shiftAmount = constOp.getValue();

      // Shift amount must be non-negative and <= 31 for 32-bit operations.
      if (shiftAmount < 0 || shiftAmount > 31)
        return failure();

      // If the shifted base is scalar (SGPR) and the result feeds into a
      // buffer_load_lds voffset chain, skip fusion — let the
      // BufferLoadLDSSoffsetPattern extract the shift into soffset instead.
      if (isSGPRType(lshlOp.getSrc1().getType())) {
        auto isBufferLoadLDS = [](Operation *op) {
          return isa<BUFFER_LOAD_DWORDX4_LDS, BUFFER_LOAD_DWORD_LDS>(op);
        };
        auto feedsBufferLoadLDS = [&](Value v) {
          for (Operation *user : v.getUsers()) {
            if (isBufferLoadLDS(user))
              return true;
            if (auto add = dyn_cast<V_ADD_U32>(user)) {
              for (Operation *u2 : add.getResult().getUsers())
                if (isBufferLoadLDS(u2))
                  return true;
            }
          }
          return false;
        };
        if (feedsBufferLoadLDS(addOp.getResult()))
          return failure();
      }

      // VOP3 constant bus restriction: at most one distinct SGPR source.
      // If both base and addend are SGPRs (and not the same value), the
      // fused instruction would need two constant bus slots → illegal.
      if (isSGPRType(lshlOp.getSrc1().getType()) &&
          isSGPRType(other.getType()) && lshlOp.getSrc1() != other)
        return failure();

      auto loc = addOp.getLoc();

      // Get the base value being shifted
      Value base = lshlOp.getSrc1();

      // Get or create the shift amount constant
      Value shiftConst = lshlOp.getSrc0();

      // Create v_lshl_add_u32: dst = (base << shift) + other
      auto resultType = addOp.getResult().getType();
      auto lshlAddOp = V_LSHL_ADD_U32::create(rewriter, loc, resultType, base,
                                              shiftConst, other);

      rewriter.replaceOp(addOp, lshlAddOp.getResult());
      return success();
    };

    // Try both operand orderings
    if (succeeded(checkOperand(addOp.getSrc0(), addOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(addOp.getSrc1(), addOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Extract SGPR shift from buffer_load_lds voffset into soffset.
//
// Matches:
//   %shifted = v_lshlrev_b32 <const>, <sgpr>    ; single-use
//   %combined = v_add_u32 %shifted, %vgpr_base  ; or reversed operands
//   %final = v_add_u32 %combined, %row          ; (optional second add)
//   buffer_load_dword[x4]_lds %final, %srd, 0
//
// Also matches when LshlAddPattern has already fused a VGPR shift+add,
// burying the scalar shift as the addend of v_lshl_add_u32:
//   %shifted = v_lshlrev_b32 <const>, <sgpr>    ; single-use
//   %fused = v_lshl_add_u32 %vgpr, <const2>, %shifted
//   %final = v_add_u32 %fused, %row
//   buffer_load_dword[x4]_lds %final, %srd, %soff
//
// Rewrites to:
//   %soff = s_lshl_b32 <sgpr>, <const>
//   %voff = v_add_u32 %vgpr_base, %row          ; loop-invariant, hoistable
//   buffer_load_dword[x4]_lds %voff, %srd, %soff
//===----------------------------------------------------------------------===//

template <typename BufferLoadOp>
struct BufferLoadLDSSoffsetPattern : public OpRewritePattern<BufferLoadOp> {
  using OpRewritePattern<BufferLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Only apply when soffset is currently 0.
    auto soffsetConst =
        loadOp.getSoffset().template getDefiningOp<ConstantOp>();
    if (!soffsetConst || soffsetConst.getValue() != 0)
      return failure();

    Value voffset = loadOp.getVoffset();
    auto outerAdd = voffset.template getDefiningOp<V_ADD_U32>();
    if (!outerAdd)
      return failure();

    // Check if a value is V_LSHLREV_B32(const, sgpr) with one use.
    auto extractScalarShift =
        [](Value v) -> std::optional<std::pair<Value, Value>> {
      auto lshl = v.getDefiningOp<V_LSHLREV_B32>();
      if (!lshl || !lshl.getResult().hasOneUse())
        return std::nullopt;
      if (!lshl.getSrc0().getDefiningOp<ConstantOp>())
        return std::nullopt;
      if (!isSGPRType(lshl.getSrc1().getType()))
        return std::nullopt;
      // Return (base_sgpr, shift_amt_const).
      return std::make_pair(lshl.getSrc1(), lshl.getSrc0());
    };

    // Try to find a scalar shift in an add's operands.
    // Returns (base_sgpr, shift_amt, other_operand).
    auto findShiftInAdd =
        [&](V_ADD_U32 add) -> std::optional<std::tuple<Value, Value, Value>> {
      for (int i = 0; i < 2; ++i) {
        Value op = i == 0 ? add.getSrc0() : add.getSrc1();
        Value other = i == 0 ? add.getSrc1() : add.getSrc0();
        if (auto shift = extractScalarShift(op))
          return std::make_tuple(shift->first, shift->second, other);
      }
      return std::nullopt;
    };

    Location loc = loadOp.getLoc();

    // Return the i-th operand pair of outerAdd (commutative).
    auto outerOperands = [&](int idx) -> std::pair<Value, Value> {
      Value lhs = outerAdd.getSrc0(), rhs = outerAdd.getSrc1();
      return idx == 0 ? std::make_pair(lhs, rhs) : std::make_pair(rhs, lhs);
    };

    // Case 1: voffset = V_ADD_U32(row, V_LSHLREV_B32(const, sgpr)).
    if (auto found = findShiftInAdd(outerAdd)) {
      auto [base, shiftAmt, other] = *found;
      auto sccTy = SCCType::get(rewriter.getContext());
      auto sLshl = S_LSHL_B32::create(rewriter, loc, base.getType(), sccTy,
                                      base, shiftAmt);
      rewriter.modifyOpInPlace(loadOp, [&]() {
        loadOp.getVoffsetMutable().assign(other);
        loadOp.getSoffsetMutable().assign(sLshl.getDst());
      });
      return success();
    }

    // Case 2: voffset = V_ADD_U32(row, V_ADD_U32(thread, V_LSHLREV(...))).
    for (int i = 0; i < 2; ++i) {
      auto [candidate, row] = outerOperands(i);
      auto innerAdd = candidate.template getDefiningOp<V_ADD_U32>();
      if (!innerAdd)
        continue;
      std::optional<std::tuple<Value, Value, Value>> found =
          findShiftInAdd(innerAdd);
      if (!found)
        continue;

      auto [base, shiftAmt, threadBase] = *found;
      auto sccTy = SCCType::get(rewriter.getContext());
      auto sLshl = S_LSHL_B32::create(rewriter, loc, base.getType(), sccTy,
                                      base, shiftAmt);
      // Rebuild voffset without the scalar shift: v_add_u32(threadBase, row).
      auto newAdd = V_ADD_U32::create(
          rewriter, loc, outerAdd.getResult().getType(), threadBase, row);
      rewriter.modifyOpInPlace(loadOp, [&]() {
        loadOp.getVoffsetMutable().assign(newAdd);
        loadOp.getSoffsetMutable().assign(sLshl.getDst());
      });
      return success();
    }

    // Case 3: v_lshl_add_u32(vgpr, const2, V_LSHLREV_B32(const, sgpr)).
    // LshlAddPattern fused a VGPR shift+add, burying the SGPR shift as the
    // addend of v_lshl_add_u32.
    for (int i = 0; i < 2; ++i) {
      auto [candidate, row] = outerOperands(i);
      auto lshlAdd = candidate.template getDefiningOp<V_LSHL_ADD_U32>();
      if (!lshlAdd)
        continue;
      // Check if src2 (the addend) is a scalar shift.
      std::optional<std::pair<Value, Value>> shift =
          extractScalarShift(lshlAdd.getSrc2());
      if (!shift)
        continue;

      auto [base, shiftAmt] = *shift;
      auto sccTy = SCCType::get(rewriter.getContext());
      auto sLshl = S_LSHL_B32::create(rewriter, loc, base.getType(), sccTy,
                                      base, shiftAmt);
      // Create v_lshl_add_u32(src0, src1, 0) — drop the scalar shift.
      auto immType = rewriter.getType<ImmType>(0);
      auto zeroConst = ConstantOp::create(rewriter, loc, immType, 0);
      auto newLshlAdd = V_LSHL_ADD_U32::create(
          rewriter, loc, lshlAdd.getResult().getType(), lshlAdd.getSrc0(),
          lshlAdd.getSrc1(), zeroConst);
      // Build new voffset: v_add_u32(newLshlAdd, row).
      auto newAdd = V_ADD_U32::create(
          rewriter, loc, outerAdd.getResult().getType(), newLshlAdd, row);
      rewriter.modifyOpInPlace(loadOp, [&]() {
        loadOp.getVoffsetMutable().assign(newAdd);
        loadOp.getSoffsetMutable().assign(sLshl.getDst());
      });
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Redundant move elimination
//
// Removes:
//   %dst = v_mov_b32 %src
// When %src is already in a VGPR and can be used directly.
//===----------------------------------------------------------------------===//

struct RedundantMovePattern : public OpRewritePattern<V_MOV_B32> {
  using OpRewritePattern<V_MOV_B32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_MOV_B32 movOp,
                                PatternRewriter &rewriter) const override {
    Value src = movOp.getSrc();

    // Skip if source is an immediate (move is necessary)
    if (isa<ImmType>(src.getType()))
      return failure();

    // Skip if source is an SGPR (move to VGPR is necessary for broadcast)
    if (isa<SRegType, PSRegType>(src.getType()))
      return failure();

    // Source is already a VGPR, so the move is redundant
    rewriter.replaceOp(movOp, src);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Multiply by power of 2 -> shift
//
// Transforms:
//   %dst = v_mul_lo_u32 %src, %const  (where const is power of 2)
// Into:
//   %dst = v_lshlrev_b32 log2(const), %src
//===----------------------------------------------------------------------===//

struct MulPow2ToShiftPattern : public OpRewritePattern<V_MUL_LO_U32> {
  using OpRewritePattern<V_MUL_LO_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_MUL_LO_U32 mulOp,
                                PatternRewriter &rewriter) const override {
    auto checkOperand = [&](Value constVal, Value other) -> LogicalResult {
      auto valueOpt = getConstantValue(constVal);
      if (!valueOpt)
        return failure();

      int64_t value = *valueOpt;

      // Check if value is a positive power of 2
      if (value <= 0 || (value & (value - 1)) != 0)
        return failure();

      int64_t shiftAmount =
          static_cast<int64_t>(llvm::Log2_64(static_cast<uint64_t>(value)));

      // Create the shift constant
      auto loc = mulOp.getLoc();
      auto immType = rewriter.getType<ImmType>(shiftAmount);
      auto shiftConst = ConstantOp::create(rewriter, loc, immType, shiftAmount);

      // Create lshlrev: dst = other << shiftAmount
      auto resultType = mulOp.getResult().getType();
      auto lshlOp =
          V_LSHLREV_B32::create(rewriter, loc, resultType, shiftConst, other);

      rewriter.replaceOp(mulOp, lshlOp.getResult());
      return success();
    };

    // Try both operand orderings
    if (succeeded(checkOperand(mulOp.getSrc0(), mulOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(mulOp.getSrc1(), mulOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Multiply by negative power of 2 -> shift + negate
//
// Transforms:
//   %dst = v_mul_lo_u32 %src, %const  (where const is negative power of 2)
// Into:
//   %shifted = v_lshlrev_b32 log2(-const), %src
//   %dst = v_sub_u32 0, %shifted
//
// This is common in index calculations where negative strides appear.
//===----------------------------------------------------------------------===//

struct MulNegPow2ToShiftNegPattern : public OpRewritePattern<V_MUL_LO_U32> {
  using OpRewritePattern<V_MUL_LO_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_MUL_LO_U32 mulOp,
                                PatternRewriter &rewriter) const override {
    auto checkOperand = [&](Value constVal, Value other) -> LogicalResult {
      auto valueOpt = getConstantValue(constVal);
      if (!valueOpt)
        return failure();

      int64_t value = *valueOpt;

      // Check if value is a negative power of 2
      if (value >= 0)
        return failure();

      // Guard against INT64_MIN (negation would overflow)
      if (value == std::numeric_limits<int64_t>::min())
        return failure();

      int64_t absValue = -value;
      if ((absValue & (absValue - 1)) != 0)
        return failure(); // Not a power of 2

      int64_t shiftAmount =
          static_cast<int64_t>(llvm::Log2_64(static_cast<uint64_t>(absValue)));

      auto loc = mulOp.getLoc();

      // Create shift constant
      auto immType = rewriter.getType<ImmType>(shiftAmount);
      auto shiftConst = ConstantOp::create(rewriter, loc, immType, shiftAmount);

      // Create zero constant for negation
      auto zeroImm = rewriter.getType<ImmType>(0);
      auto zeroConst = ConstantOp::create(rewriter, loc, zeroImm, 0);

      // Shift the value: shifted = src << log2(abs(const))
      auto resultType = mulOp.getResult().getType();
      auto lshlOp =
          V_LSHLREV_B32::create(rewriter, loc, resultType, shiftConst, other);

      // Negate the result: dst = 0 - shifted
      auto subOp =
          V_SUB_U32::create(rewriter, loc, resultType, zeroConst, lshlOp);

      rewriter.replaceOp(mulOp, subOp.getResult());
      return success();
    };

    // Try both operand orderings
    if (succeeded(checkOperand(mulOp.getSrc0(), mulOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(mulOp.getSrc1(), mulOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Add constant simplification
//
// 1. Both operands constant -> fold to v_mov_b32 of computed sum.
// 2. Either operand zero    -> identity elimination.
//===----------------------------------------------------------------------===//

struct AddZeroPattern : public OpRewritePattern<V_ADD_U32> {
  using OpRewritePattern<V_ADD_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_ADD_U32 addOp,
                                PatternRewriter &rewriter) const override {
    auto val0 = getConstantValue(addOp.getSrc0());
    auto val1 = getConstantValue(addOp.getSrc1());

    // Fold: both operands constant -> compute result at compile time.
    if (val0 && val1) {
      int64_t result = static_cast<int64_t>(
          (static_cast<uint64_t>(*val0) + static_cast<uint64_t>(*val1)) &
          0xFFFFFFFF);
      auto loc = addOp.getLoc();
      auto immType = rewriter.getType<ImmType>(result);
      auto constOp = ConstantOp::create(rewriter, loc, immType, result);
      auto vregType = rewriter.getType<VRegType>(1, 1);
      auto movOp = V_MOV_B32::create(rewriter, loc, vregType, constOp);
      rewriter.replaceOp(addOp, movOp.getResult());
      return success();
    }

    // Fold: either operand is zero -> identity elimination.
    // Guard: the surviving operand must be a VGPR; replacing a VGPR result
    // with an SGPR/immediate would create a type mismatch for downstream users.
    if (val0 && *val0 == 0 && isVGPRType(addOp.getSrc1().getType())) {
      rewriter.replaceOp(addOp, addOp.getSrc1());
      return success();
    }
    if (val1 && *val1 == 0 && isVGPRType(addOp.getSrc0().getType())) {
      rewriter.replaceOp(addOp, addOp.getSrc0());
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Multiply by one elimination
//
// Removes:
//   %dst = v_mul_lo_u32 %src, 1
// Replacing with %src directly.
//===----------------------------------------------------------------------===//

struct MulOnePattern : public OpRewritePattern<V_MUL_LO_U32> {
  using OpRewritePattern<V_MUL_LO_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_MUL_LO_U32 mulOp,
                                PatternRewriter &rewriter) const override {
    auto checkOperand = [&](Value constVal, Value other) -> LogicalResult {
      auto val = getConstantValue(constVal);
      if (!val || *val != 1)
        return failure();
      if (!isVGPRType(other.getType()))
        return failure();

      rewriter.replaceOp(mulOp, other);
      return success();
    };

    if (succeeded(checkOperand(mulOp.getSrc0(), mulOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(mulOp.getSrc1(), mulOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Multiply constant simplification
//
// 1. Both operands constant -> fold to v_mov_b32 of computed product.
// 2. Either operand zero    -> result is zero.
//===----------------------------------------------------------------------===//

struct MulZeroPattern : public OpRewritePattern<V_MUL_LO_U32> {
  using OpRewritePattern<V_MUL_LO_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_MUL_LO_U32 mulOp,
                                PatternRewriter &rewriter) const override {
    auto val0 = getConstantValue(mulOp.getSrc0());
    auto val1 = getConstantValue(mulOp.getSrc1());

    // Fold: both operands constant -> compute result at compile time.
    if (val0 && val1) {
      int64_t result = static_cast<int64_t>(
          (static_cast<uint64_t>(*val0) * static_cast<uint64_t>(*val1)) &
          0xFFFFFFFF);
      auto loc = mulOp.getLoc();
      auto immType = rewriter.getType<ImmType>(result);
      auto constOp = ConstantOp::create(rewriter, loc, immType, result);
      auto vregType = rewriter.getType<VRegType>(1, 1);
      auto movOp = V_MOV_B32::create(rewriter, loc, vregType, constOp);
      rewriter.replaceOp(mulOp, movOp.getResult());
      return success();
    }

    // Fold: either operand is zero -> result is zero.
    if ((val0 && *val0 == 0) || (val1 && *val1 == 0)) {
      auto loc = mulOp.getLoc();
      auto immType = rewriter.getType<ImmType>(0);
      auto zeroConst = ConstantOp::create(rewriter, loc, immType, 0);
      auto vregType = rewriter.getType<VRegType>(1, 1);
      auto movOp = V_MOV_B32::create(rewriter, loc, vregType, zeroConst);
      rewriter.replaceOp(mulOp, movOp.getResult());
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: (x >> N) << N -> x & ~((1 << N) - 1)
//
// The affine handler compiles floor(x / 2^N) * 2^N as:
//   %shr = v_lshrrev_b32 N, x
//   %shl = v_lshlrev_b32 N, %shr
// This is equivalent to masking off the low N bits:
//   %result = v_and_b32 x, ~((1 << N) - 1)
// Saves 1 VALU instruction per occurrence.
//===----------------------------------------------------------------------===//

struct ShiftRightLeftToAndPattern : public OpRewritePattern<V_LSHLREV_B32> {
  using OpRewritePattern<V_LSHLREV_B32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_LSHLREV_B32 lshlOp,
                                PatternRewriter &rewriter) const override {
    // Check if src1 (value being shifted left) comes from a right shift
    auto lshrOp = lshlOp.getSrc1().getDefiningOp<V_LSHRREV_B32>();
    if (!lshrOp)
      return failure();

    // Both shift amounts must be the same constant
    auto lshlAmt = getConstantValue(lshlOp.getSrc0());
    auto lshrAmt = getConstantValue(lshrOp.getSrc0());
    if (!lshlAmt || !lshrAmt || *lshlAmt != *lshrAmt)
      return failure();

    // Shift amount must be valid
    if (*lshlAmt <= 0 || *lshlAmt >= 32)
      return failure();

    // Compute the AND mask: ~((1 << N) - 1) = all bits set except low N
    int64_t mask = ~((1LL << *lshlAmt) - 1) & 0xFFFFFFFF;

    auto loc = lshlOp.getLoc();
    auto resultType = lshlOp.getResult().getType();

    // Create the mask constant
    auto maskImm = rewriter.getType<ImmType>(mask);
    auto maskConst = ConstantOp::create(rewriter, loc, maskImm, mask);

    // Create: result = x & mask
    // NOTE: constant must be src0 (first operand) for VOP2 encoding.
    // src1 must be a VGPR on AMDGCN.
    auto andOp = V_AND_B32::create(rewriter, loc, resultType, maskConst,
                                   lshrOp.getSrc1());

    rewriter.replaceOp(lshlOp, andOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: a + (0 - b) -> a - b
//
// The affine handler produces negate-then-add sequences:
//   %neg = v_sub_u32 0, %val
//   %result = v_add_u32 %other, %neg
// This simplifies to:
//   %result = v_sub_u32 %other, %val
// Saves 1 VALU instruction per occurrence.
//===----------------------------------------------------------------------===//

struct AddNegToSubPattern : public OpRewritePattern<V_ADD_U32> {
  using OpRewritePattern<V_ADD_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_ADD_U32 addOp,
                                PatternRewriter &rewriter) const override {
    auto checkOperand = [&](Value negated, Value other) -> LogicalResult {
      auto subOp = negated.getDefiningOp<V_SUB_U32>();
      if (!subOp)
        return failure();

      // Check that src0 is constant 0 (i.e., this is 0 - x = -x)
      auto zeroVal = getConstantValue(subOp.getSrc0());
      if (!zeroVal || *zeroVal != 0)
        return failure();

      // No single-use constraint: if the negation has multiple uses that are
      // all add(y, neg) patterns, the greedy rewriter will fire this pattern
      // for each use. Once all uses are rewritten, the negation becomes dead
      // and is cleaned up by DCE or the next CSE pass.

      auto loc = addOp.getLoc();
      auto resultType = addOp.getResult().getType();

      // Replace a + (0 - b) with a - b
      auto newSub =
          V_SUB_U32::create(rewriter, loc, resultType, other, subOp.getSrc1());

      rewriter.replaceOp(addOp, newSub.getResult());
      return success();
    };

    // Try both operand orderings
    if (succeeded(checkOperand(addOp.getSrc0(), addOp.getSrc1())))
      return success();
    if (succeeded(checkOperand(addOp.getSrc1(), addOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: MFMA with zeroed accumulator -> MFMA with inline 0
//
// When the accumulator input to an MFMA is a zero constant (via v_mov_b32),
// replace it with an inline 0 constant directly. This:
// 1. Eliminates the zeroing mov instruction
// 2. Uses hardware's native inline constant support
//===----------------------------------------------------------------------===//

struct MFMAZeroAccumPattern
    : public OpInterfaceRewritePattern<MFMAOpInterface> {
  using OpInterfaceRewritePattern<MFMAOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(MFMAOpInterface mfmaOp,
                                PatternRewriter &rewriter) const override {
    // Get the accumulator operand using the named accessor
    Value accum = mfmaOp.getAcc();

    // Check if accumulator comes from a v_mov_b32 of zero
    auto movOp = accum.getDefiningOp<V_MOV_B32>();
    if (!movOp)
      return failure();

    // Don't optimize away accumulator initializations marked with no_cse
    // These are loop-carried accumulators that need to exist as registers
    if (movOp->hasAttr("no_cse"))
      return failure();

    // Check if the mov source is a zero constant
    auto constOp = movOp.getSrc().getDefiningOp<ConstantOp>();
    if (!constOp || constOp.getValue() != 0)
      return failure();

    // Check that the mov result is only used by this MFMA
    if (!accum.hasOneUse())
      return failure();

    // Replace the accumulator with an inline zero constant
    Operation *op = mfmaOp.getOperation();
    auto loc = op->getLoc();
    auto immType = rewriter.getType<ImmType>(0);
    auto zeroConst = ConstantOp::create(rewriter, loc, immType, 0);

    // Update the MFMA's accumulator operand (operand index from interface)
    rewriter.modifyOpInPlace(
        op, [&]() { op->setOperand(mfmaOp.getTiedOperandIndex(), zeroConst); });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Fold v_mov_b32 of constant into v_add_u32
//
// Transforms:
//   %tmp = v_mov_b32 %const  (where %const is a ConstantOp)
//   %dst = v_add_u32 %tmp, %x
// Into:
//   %dst = v_add_u32 %const, %x
//
// On GFX9, v_add_u32 (VOP2) supports a 32-bit literal constant as src0.
// This eliminates the v_mov_b32 instruction and frees its VGPR output,
// reducing both instruction count and register pressure.
//===----------------------------------------------------------------------===//

struct FoldMovConstIntoAddPattern : public OpRewritePattern<V_ADD_U32> {
  using OpRewritePattern<V_ADD_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_ADD_U32 addOp,
                                PatternRewriter &rewriter) const override {
    // Check if either operand is a v_mov_b32 of a constant
    auto tryFold = [&](Value movVal, Value other) -> LogicalResult {
      auto movOp = movVal.getDefiningOp<V_MOV_B32>();
      if (!movOp)
        return failure();

      // The mov source must be a ConstantOp (immediate)
      auto constOp = movOp.getSrc().getDefiningOp<ConstantOp>();
      if (!constOp)
        return failure();

      // Only fold if the mov result is used only by this add.
      // If other users exist, the mov is still needed.
      if (!movOp.getResult().hasOneUse())
        return failure();

      // Replace: v_add_u32(v_mov_b32(CONST), X) -> v_add_u32(CONST, X)
      auto loc = addOp.getLoc();
      auto resultType = addOp.getResult().getType();
      auto newAdd =
          V_ADD_U32::create(rewriter, loc, resultType, movOp.getSrc(), other);
      rewriter.replaceOp(addOp, newAdd.getResult());
      return success();
    };

    // Try both operand orderings (V_ADD_U32 is commutative)
    if (succeeded(tryFold(addOp.getSrc0(), addOp.getSrc1())))
      return success();
    if (succeeded(tryFold(addOp.getSrc1(), addOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Fold v_mov_b32 of constant into v_sub_u32
//
// Transforms:
//   %tmp = v_mov_b32 %const
//   %dst = v_sub_u32 %x, %tmp     (x - const)
// Into:
//   %dst = v_sub_u32 %x, %const
//===----------------------------------------------------------------------===//

struct FoldMovConstIntoSubPattern : public OpRewritePattern<V_SUB_U32> {
  using OpRewritePattern<V_SUB_U32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_SUB_U32 subOp,
                                PatternRewriter &rewriter) const override {
    // Check each operand for a v_mov_b32 of a constant
    auto tryFoldSrc0 = [&]() -> LogicalResult {
      auto movOp = subOp.getSrc0().getDefiningOp<V_MOV_B32>();
      if (!movOp || !movOp.getSrc().getDefiningOp<ConstantOp>())
        return failure();
      if (!movOp.getResult().hasOneUse())
        return failure();
      auto loc = subOp.getLoc();
      auto resultType = subOp.getResult().getType();
      auto newSub = V_SUB_U32::create(rewriter, loc, resultType, movOp.getSrc(),
                                      subOp.getSrc1());
      rewriter.replaceOp(subOp, newSub.getResult());
      return success();
    };

    auto tryFoldSrc1 = [&]() -> LogicalResult {
      auto movOp = subOp.getSrc1().getDefiningOp<V_MOV_B32>();
      if (!movOp || !movOp.getSrc().getDefiningOp<ConstantOp>())
        return failure();
      if (!movOp.getResult().hasOneUse())
        return failure();
      auto loc = subOp.getLoc();
      auto resultType = subOp.getResult().getType();
      auto newSub = V_SUB_U32::create(rewriter, loc, resultType,
                                      subOp.getSrc0(), movOp.getSrc());
      rewriter.replaceOp(subOp, newSub.getResult());
      return success();
    };

    if (succeeded(tryFoldSrc0()))
      return success();
    if (succeeded(tryFoldSrc1()))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Fold v_mov_b32 of constant into v_lshlrev_b32
//
// Transforms:
//   %tmp = v_mov_b32 %const
//   %dst = v_lshlrev_b32 %tmp, %x    (x << const)
// Into:
//   %dst = v_lshlrev_b32 %const, %x
//===----------------------------------------------------------------------===//

struct FoldMovConstIntoLshlPattern : public OpRewritePattern<V_LSHLREV_B32> {
  using OpRewritePattern<V_LSHLREV_B32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_LSHLREV_B32 lshlOp,
                                PatternRewriter &rewriter) const override {
    // src0 is the shift amount - check if it's a v_mov_b32 of a constant
    auto movOp = lshlOp.getSrc0().getDefiningOp<V_MOV_B32>();
    if (!movOp)
      return failure();
    auto constOp = movOp.getSrc().getDefiningOp<ConstantOp>();
    if (!constOp)
      return failure();
    if (!movOp.getResult().hasOneUse())
      return failure();

    auto loc = lshlOp.getLoc();
    auto resultType = lshlOp.getResult().getType();
    auto newLshl = V_LSHLREV_B32::create(rewriter, loc, resultType,
                                         movOp.getSrc(), lshlOp.getSrc1());
    rewriter.replaceOp(lshlOp, newLshl.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Fold v_mov_b32 of constant into v_lshrrev_b32
//===----------------------------------------------------------------------===//

struct FoldMovConstIntoLshrPattern : public OpRewritePattern<V_LSHRREV_B32> {
  using OpRewritePattern<V_LSHRREV_B32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_LSHRREV_B32 lshrOp,
                                PatternRewriter &rewriter) const override {
    auto movOp = lshrOp.getSrc0().getDefiningOp<V_MOV_B32>();
    if (!movOp)
      return failure();
    auto constOp = movOp.getSrc().getDefiningOp<ConstantOp>();
    if (!constOp)
      return failure();
    if (!movOp.getResult().hasOneUse())
      return failure();

    auto loc = lshrOp.getLoc();
    auto resultType = lshrOp.getResult().getType();
    auto newLshr = V_LSHRREV_B32::create(rewriter, loc, resultType,
                                         movOp.getSrc(), lshrOp.getSrc1());
    rewriter.replaceOp(lshrOp, newLshr.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Fold v_mov_b32 of constant into v_and_b32
//===----------------------------------------------------------------------===//

struct FoldMovConstIntoAndPattern : public OpRewritePattern<V_AND_B32> {
  using OpRewritePattern<V_AND_B32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_AND_B32 andOp,
                                PatternRewriter &rewriter) const override {
    auto tryFold = [&](Value movVal, Value other) -> LogicalResult {
      auto movOp = movVal.getDefiningOp<V_MOV_B32>();
      if (!movOp)
        return failure();
      auto constOp = movOp.getSrc().getDefiningOp<ConstantOp>();
      if (!constOp)
        return failure();
      if (!movOp.getResult().hasOneUse())
        return failure();

      auto loc = andOp.getLoc();
      auto resultType = andOp.getResult().getType();
      auto newAnd =
          V_AND_B32::create(rewriter, loc, resultType, movOp.getSrc(), other);
      rewriter.replaceOp(andOp, newAnd.getResult());
      return success();
    };

    if (succeeded(tryFold(andOp.getSrc0(), andOp.getSrc1())))
      return success();
    if (succeeded(tryFold(andOp.getSrc1(), andOp.getSrc0())))
      return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: BFE → pack identity elimination.
//
// Recognizes:
//   v_lshl_or_b32(bfe(x,24,8), 24,
//     v_lshl_or_b32(bfe(x,16,8), 16,
//       v_lshl_or_b32(bfe(x,8,8), 8, bfe(x,0,8))))
// All four BFEs extract consecutive bytes from the same source dword.
// The pack reconstructs the original dword exactly, so replace with x.
//===----------------------------------------------------------------------===//

struct BFEPackIdentityPattern : public OpRewritePattern<V_LSHL_OR_B32> {
  using OpRewritePattern<V_LSHL_OR_B32>::OpRewritePattern;

  LogicalResult matchAndRewrite(V_LSHL_OR_B32 outerOp,
                                PatternRewriter &rewriter) const override {
    // Outer: v_lshl_or_b32(byte3, 24, middle_result).
    std::optional<int64_t> shiftOuter = getConstantValue(outerOp.getSrc1());
    if (!shiftOuter || *shiftOuter != 24)
      return failure();

    auto bfe3 = outerOp.getSrc0().getDefiningOp<V_BFE_U32>();
    if (!bfe3)
      return failure();

    // Middle: v_lshl_or_b32(byte2, 16, inner_result).
    auto middleOp = outerOp.getSrc2().getDefiningOp<V_LSHL_OR_B32>();
    if (!middleOp)
      return failure();

    std::optional<int64_t> shiftMiddle = getConstantValue(middleOp.getSrc1());
    if (!shiftMiddle || *shiftMiddle != 16)
      return failure();

    auto bfe2 = middleOp.getSrc0().getDefiningOp<V_BFE_U32>();
    if (!bfe2)
      return failure();

    // Inner: v_lshl_or_b32(byte1, 8, byte0).
    auto innerOp = middleOp.getSrc2().getDefiningOp<V_LSHL_OR_B32>();
    if (!innerOp)
      return failure();

    std::optional<int64_t> shiftInner = getConstantValue(innerOp.getSrc1());
    if (!shiftInner || *shiftInner != 8)
      return failure();

    auto bfe1 = innerOp.getSrc0().getDefiningOp<V_BFE_U32>();
    if (!bfe1)
      return failure();

    auto bfe0 = innerOp.getSrc2().getDefiningOp<V_BFE_U32>();
    if (!bfe0)
      return failure();

    // All 4 BFEs must extract from the same source with width 8.
    auto checkBFE = [](V_BFE_U32 bfe,
                       int64_t expectedOffset) -> std::optional<Value> {
      std::optional<int64_t> width = getConstantValue(bfe.getSrc2());
      if (!width || *width != 8)
        return std::nullopt;
      std::optional<int64_t> offset = getConstantValue(bfe.getSrc1());
      if (!offset || *offset != expectedOffset)
        return std::nullopt;
      return bfe.getSrc0();
    };

    std::optional<Value> src0 = checkBFE(bfe0, 0);
    std::optional<Value> src1 = checkBFE(bfe1, 8);
    std::optional<Value> src2 = checkBFE(bfe2, 16);
    std::optional<Value> src3 = checkBFE(bfe3, 24);

    if (!src0 || !src1 || !src2 || !src3)
      return failure();

    // All must extract from the same source register.
    if (*src0 != *src1 || *src0 != *src2 || *src0 != *src3)
      return failure();

    // The source must be a VGPR — replacing a VGPR result (from v_lshl_or_b32)
    // with an SGPR or immediate would create a type mismatch for downstream
    // users.
    if (!isVGPRType(src0->getType()))
      return failure();

    // The packed result equals the original source dword.
    rewriter.replaceOp(outerOp, *src0);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Peephole Optimization Pass
//===----------------------------------------------------------------------===//

struct PeepholePass : public waveasm::impl::WAVEASMPeepholeBase<PeepholePass> {
  using WAVEASMPeepholeBase::WAVEASMPeepholeBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    RewritePatternSet patterns(ctx);

    // Add all peephole patterns.
    patterns.add<
        // clang-format off
        AddNegToSubPattern,
        AddZeroPattern,
        BFEPackIdentityPattern,
        BufferLoadLDSSoffsetPattern<BUFFER_LOAD_DWORD_LDS>,
        BufferLoadLDSSoffsetPattern<BUFFER_LOAD_DWORDX4_LDS>,
        FoldMovConstIntoAddPattern,
        FoldMovConstIntoAndPattern,
        FoldMovConstIntoLshlPattern,
        FoldMovConstIntoLshrPattern,
        FoldMovConstIntoSubPattern,
        LshlAddPattern,
        LshlOrFusePattern,
        MFMAZeroAccumPattern,
        MulNegPow2ToShiftNegPattern,
        MulOnePattern,
        MulPow2ToShiftPattern,
        MulZeroPattern,
        RedundantMovePattern,
        ShiftRightLeftToAndPattern
        // clang-format on
        >(ctx);

    // Apply patterns greedily.
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
