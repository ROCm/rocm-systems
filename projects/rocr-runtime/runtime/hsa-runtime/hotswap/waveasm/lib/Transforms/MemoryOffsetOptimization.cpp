// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Memory Offset Optimization Pass
//
// This pass folds constant address components into memory instruction offset
// fields, reducing VALU instruction count. It handles:
//
// 1. Top-level constant add: V_ADD_U32(base, K) -> mem_op(base, offset:K)
// 2. Constant through V_MOV_B32: V_ADD_U32(base, V_MOV_B32(K)) -> offset:K
// 3. Constant through shift (algebraic distribution):
//    V_LSHLREV_B32(N, V_ADD_U32(base, K)) -> V_LSHLREV_B32(N, base) +
//    offset:K<<N
// 4. Multi-level combinations of the above
//
// After folding, dead instructions are removed by a DCE sweep.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-memory-offset-opt"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMMEMORYOFFSETOPT
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

//===----------------------------------------------------------------------===//
// Hardware offset limits
//===----------------------------------------------------------------------===//

/// Maximum offset for DS (LDS) instructions: unsigned 16-bit
static constexpr int64_t DS_MAX_OFFSET = 65535;

/// Maximum offset for buffer (VMEM) instructions: unsigned 12-bit
static constexpr int64_t BUFFER_MAX_OFFSET = 4095;

//===----------------------------------------------------------------------===//
// Constant extraction from address expression trees
//===----------------------------------------------------------------------===//

/// Result of recursive constant extraction from an address expression tree.
struct AddrAnalysis {
  Value base;          // Address without extracted constants
  int64_t constOffset; // Accumulated constant offset
};

/// Helper: check if an op is V_ADD_U32 or V_OR_B32 and return its two src
/// operands. V_OR_B32 is treated the same as ADD when the affine handler
/// uses it for non-overlapping bit ranges (x | K == x + K when bits don't
/// overlap).
///
/// INVARIANT: The AffineHandlers only emit V_OR_B32 when bit ranges provably
/// don't overlap (via BitRange tracking).  The shift distribution
/// `(a|K)<<N = (a<<N)|(K<<N)` is always correct for OR, but the extracted
/// constant is only a valid *additive* offset when bits don't overlap.
///
/// This invariant is enforced at runtime by `checkOrOverlap()`:
///   - Proven safe   -> proceed with the fold
///   - Proven overlap -> emit a warning (upstream bug) and skip
///   - Unknown        -> conservatively skip the fold
struct BinaryAddLike {
  Value src0, src1;
  Type resultType;
  bool isOr; // true if this came from V_OR_B32 (not V_ADD_U32)
};

static std::optional<BinaryAddLike> getAddLikeOp(Value v) {
  if (auto addOp = v.getDefiningOp<V_ADD_U32>())
    return BinaryAddLike{addOp.getSrc0(), addOp.getSrc1(),
                         addOp.getResult().getType(), /*isOr=*/false};
  if (auto orOp = v.getDefiningOp<V_OR_B32>())
    return BinaryAddLike{orOp.getSrc0(), orOp.getSrc1(),
                         orOp.getResult().getType(), /*isOr=*/true};
  return std::nullopt;
}

/// Result of checking bit overlap for V_OR_B32 treated as addition.
enum class OrOverlapCheck { Safe, Overlap, Unknown };

/// Check whether treating V_OR_B32(nonConst, K) as V_ADD_U32(nonConst, K) is
/// safe - i.e., the constant K only sets bits that are provably zero in the
/// non-constant operand.
///
/// Returns:
///   Safe    - non-overlap proven (e.g., nonConst is left-shifted by N and
///             K fits entirely within the low N bits)
///   Overlap - bits definitely overlap (upstream invariant violated)
///   Unknown - cannot determine locally; caller should conservatively skip
static OrOverlapCheck checkOrOverlap(Value nonConstOperand, int64_t constVal) {
  if (constVal == 0)
    return OrOverlapCheck::Safe;

  // If the non-constant operand is a left shift by N, its low N bits are zero.
  if (auto shiftOp = nonConstOperand.getDefiningOp<V_LSHLREV_B32>()) {
    if (auto shiftAmt = getConstantValue(shiftOp.getSrc0())) {
      if (*shiftAmt > 0 && *shiftAmt < 32) {
        int64_t lowMask = (1LL << *shiftAmt) - 1;
        if ((constVal & ~lowMask) == 0)
          return OrOverlapCheck::Safe;  // K only uses low N bits
        return OrOverlapCheck::Overlap; // K sets bits above shift range
      }
    }
  }

  return OrOverlapCheck::Unknown;
}

//===----------------------------------------------------------------------===//
// Overflow-safe arithmetic helpers
//===----------------------------------------------------------------------===//

/// Shift left with overflow check. Returns std::nullopt if the result would
/// not be representable as int64_t.
static std::optional<int64_t> safeShiftLeft(int64_t val, int64_t amt) {
  if (val == 0)
    return int64_t(0);
  if (amt <= 0)
    return (amt == 0) ? std::optional<int64_t>(val) : std::nullopt;
  if (amt >= 63)
    return std::nullopt;
  // val must fit in (64 - amt) signed bits, i.e. lie in [-limit, limit).
  int64_t limit = int64_t(1) << (63 - amt);
  if (val >= limit || val < -limit)
    return std::nullopt;
  // Use unsigned shift to avoid C++17 UB on signed left-shift of negatives.
  return static_cast<int64_t>(static_cast<uint64_t>(val) << amt);
}

/// Add with overflow check. Returns std::nullopt if the result would not be
/// representable as int64_t.
static std::optional<int64_t> safeAdd(int64_t a, int64_t b) {
  if (b > 0 && a > INT64_MAX - b)
    return std::nullopt;
  if (b < 0 && a < INT64_MIN - b)
    return std::nullopt;
  return a + b;
}

/// Recursively extract constant components from an address expression tree.
/// Returns the base address (without constants) and the accumulated constant.
///
/// Patterns handled:
///   V_ADD_U32(base, K) or V_OR_B32(base, K) -> base, K
///   V_ADD_U32(base, V_MOV_B32(K)) -> base, K
///   V_LSHLREV_B32(N, V_ADD_U32(base, K)) -> V_LSHLREV_B32(N, base), K << N
///   V_LSHLREV_B32(N, V_OR_B32(base, K))  -> V_LSHLREV_B32(N, base), K << N
///   V_ADD_U32(X, Y) where X or Y has extractable constants through shifts
static AddrAnalysis extractConstant(Value addr, OpBuilder &builder,
                                    Location loc) {
  // Pattern 1: V_ADD_U32(base, const) or V_OR_B32(base, const)
  if (auto addLike = getAddLikeOp(addr)) {
    // Check src1 for constant
    if (auto c = getConstantValue(addLike->src1)) {
      // For V_OR_B32, verify bit non-overlap before treating as addition
      if (addLike->isOr) {
        auto check = checkOrOverlap(addLike->src0, *c);
        if (check == OrOverlapCheck::Overlap) {
          addr.getDefiningOp()->emitWarning()
              << "V_OR_B32 treated as ADD but constant " << *c
              << " overlaps with non-constant operand; "
              << "skipping offset fold (upstream invariant violation)";
          return {addr, 0};
        }
        if (check == OrOverlapCheck::Unknown) {
          LLVM_DEBUG(llvm::dbgs()
                     << "MemoryOffsetOpt: skipping V_OR_B32 - cannot prove "
                     << "non-overlapping bits for constant " << *c << "\n");
          return {addr, 0};
        }
      }
      // Recurse on the non-constant operand to find deeper patterns
      auto inner = extractConstant(addLike->src0, builder, loc);
      auto sum = safeAdd(inner.constOffset, *c);
      if (!sum)
        return {addr, 0};
      return {inner.base, *sum};
    }
    // Check src0 for constant (commutative)
    if (auto c = getConstantValue(addLike->src0)) {
      // For V_OR_B32, verify bit non-overlap before treating as addition
      if (addLike->isOr) {
        auto check = checkOrOverlap(addLike->src1, *c);
        if (check == OrOverlapCheck::Overlap) {
          addr.getDefiningOp()->emitWarning()
              << "V_OR_B32 treated as ADD but constant " << *c
              << " overlaps with non-constant operand; "
              << "skipping offset fold (upstream invariant violation)";
          return {addr, 0};
        }
        if (check == OrOverlapCheck::Unknown) {
          LLVM_DEBUG(llvm::dbgs()
                     << "MemoryOffsetOpt: skipping V_OR_B32 - cannot prove "
                     << "non-overlapping bits for constant " << *c << "\n");
          return {addr, 0};
        }
      }
      auto inner = extractConstant(addLike->src1, builder, loc);
      auto sum = safeAdd(inner.constOffset, *c);
      if (!sum)
        return {addr, 0};
      return {inner.base, *sum};
    }

    // Neither operand is a direct constant, but check if either is a shift
    // with an extractable constant inside:
    // V_ADD_U32(V_LSHLREV_B32(N, V_ADD_U32(base, K)), other)
    // -> V_ADD_U32(V_LSHLREV_B32(N, base), other) + K << N
    auto tryShiftOperand = [&](Value shiftedVal,
                               Value otherVal) -> std::optional<AddrAnalysis> {
      if (auto shiftOp = shiftedVal.getDefiningOp<V_LSHLREV_B32>()) {
        auto shiftAmt = getConstantValue(shiftOp.getSrc0());
        if (!shiftAmt || *shiftAmt < 0 || *shiftAmt >= 32)
          return std::nullopt;

        auto inner = extractConstant(shiftOp.getSrc1(), builder, loc);
        if (inner.constOffset == 0)
          return std::nullopt;

        // Check shift overflow before creating any new ops
        auto shiftedConst = safeShiftLeft(inner.constOffset, *shiftAmt);
        if (!shiftedConst)
          return std::nullopt;

        // Create new shift of the stripped base
        auto newShift =
            V_LSHLREV_B32::create(builder, loc, shiftOp.getResult().getType(),
                                  shiftOp.getSrc0(), inner.base);

        // Recurse on the other operand too
        auto otherAnalysis = extractConstant(otherVal, builder, loc);

        // Check addition overflow
        auto totalConst = safeAdd(*shiftedConst, otherAnalysis.constOffset);
        if (!totalConst)
          return std::nullopt;

        // Create new add with the stripped shift and other operand
        Value newBase =
            V_ADD_U32::create(builder, loc, addLike->resultType,
                              newShift.getResult(), otherAnalysis.base);

        return AddrAnalysis{newBase, *totalConst};
      }
      return std::nullopt;
    };

    // Try src0 as shifted, src1 as other
    if (auto result = tryShiftOperand(addLike->src0, addLike->src1))
      return *result;
    // Try src1 as shifted, src0 as other
    if (auto result = tryShiftOperand(addLike->src1, addLike->src0))
      return *result;

    // Recursive case: neither operand is a direct constant or shift, but one
    // may contain an extractable constant deeper in its expression tree.
    // V_ADD_U32(X, Y) where extractConstant(X) = (X_base, K), K != 0
    //   -> V_ADD_U32(X_base, Y) + K
    // This handles patterns like:
    //   V_ADD_U32(V_ADD_U32(V_LSHLREV_B32(N, base+K), col), loop_offset)
    // where the constant K is inside a nested shift within an add chain.
    auto trySrcRecurse = [&](Value src,
                             Value other) -> std::optional<AddrAnalysis> {
      // Only recurse into add-like ops to avoid infinite recursion on
      // non-add trees (shifts are already handled above).
      if (!getAddLikeOp(src))
        return std::nullopt;
      auto srcAnalysis = extractConstant(src, builder, loc);
      if (srcAnalysis.constOffset == 0)
        return std::nullopt;
      // Also recurse on the other operand
      auto otherAnalysis = extractConstant(other, builder, loc);
      auto totalConst =
          safeAdd(srcAnalysis.constOffset, otherAnalysis.constOffset);
      if (!totalConst)
        return std::nullopt;
      Value newBase = V_ADD_U32::create(builder, loc, addLike->resultType,
                                        srcAnalysis.base, otherAnalysis.base);
      return AddrAnalysis{newBase, *totalConst};
    };

    if (auto result = trySrcRecurse(addLike->src0, addLike->src1))
      return *result;
    if (auto result = trySrcRecurse(addLike->src1, addLike->src0))
      return *result;
  }

  // Pattern 2: V_LSHLREV_B32(N, add_like(base, K))
  // Distributes shift over add/or: (base + K) << N = (base << N) + (K << N)
  if (auto shiftOp = addr.getDefiningOp<V_LSHLREV_B32>()) {
    auto shiftAmt = getConstantValue(shiftOp.getSrc0());
    // Guard against invalid/large shift amounts that could cause UB
    if (shiftAmt && *shiftAmt >= 0 && *shiftAmt < 32) {
      auto inner = extractConstant(shiftOp.getSrc1(), builder, loc);
      if (inner.constOffset != 0) {
        // Check shift overflow before creating any new ops
        auto shiftedConst = safeShiftLeft(inner.constOffset, *shiftAmt);
        if (shiftedConst) {
          // Create new shift of the stripped base
          auto newShift =
              V_LSHLREV_B32::create(builder, loc, shiftOp.getResult().getType(),
                                    shiftOp.getSrc0(), inner.base);
          return {newShift.getResult(), *shiftedConst};
        }
      }
    }
  }

  // No constant found
  return {addr, 0};
}

//===----------------------------------------------------------------------===//
// Memory Op Classification and Offset Access
//===----------------------------------------------------------------------===//

enum class MemOpKind { DS, Buffer, Unknown };

/// Classify a memory operation and return its kind.
static MemOpKind getMemOpKind(Operation *op) {
  StringRef name = op->getName().getStringRef();
  // Exclude gather-to-LDS ops (buffer_load_*_lds): the hardware may not
  // honor the instOffset field when the MUBUF LDS bit is set on GFX9/GFX950,
  // causing silent address corruption.
  if (name.contains("_lds"))
    return MemOpKind::Unknown;
  if (name.contains("ds_read") || name.contains("ds_write"))
    return MemOpKind::DS;
  if (name.contains("buffer_store") || name.contains("buffer_load") ||
      name.contains("global_store") || name.contains("global_load") ||
      name.contains("flat_store") || name.contains("flat_load"))
    return MemOpKind::Buffer;
  return MemOpKind::Unknown;
}

/// Get the maximum allowed offset for a memory op kind.
static int64_t getMaxOffset(MemOpKind kind) {
  switch (kind) {
  case MemOpKind::DS:
    return DS_MAX_OFFSET;
  case MemOpKind::Buffer:
    return BUFFER_MAX_OFFSET;
  case MemOpKind::Unknown:
    return 0;
  }
  return 0;
}

/// Get the address operand index for a memory op.
/// DS ops: vaddr is operand 0 (ds_read) or 1 (ds_write: data, vaddr)
/// Buffer loads: voffset is operand 1 (saddr, voffset)
/// Buffer load LDS (gather): voffset is operand 0 (voffset, saddr, soffset)
/// Buffer stores: voffset is operand 2 (data, saddr, voffset)
static int getAddrOperandIndex(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name.contains("ds_read") || name.contains("ds_write"))
    return name.contains("ds_write") ? 1 : 0; // ds_write: (data, vaddr)
  // buffer_load_dwordx4_lds has layout: (voffset, saddr, soffset)
  if (name.contains("_lds"))
    return 0;
  if (name.contains("buffer_load") || name.contains("global_load") ||
      name.contains("flat_load"))
    return 1; // (saddr, voffset)
  if (name.contains("buffer_store") || name.contains("global_store") ||
      name.contains("flat_store"))
    return 2; // (data, saddr, voffset)
  return -1;
}

/// Get the current offset value from a memory op.
static int64_t getCurrentOffset(Operation *op) {
  // DS ops use "offset" attribute
  if (auto offsetAttr = op->getAttrOfType<IntegerAttr>("offset"))
    return offsetAttr.getInt();
  // Buffer ops use "instOffset" attribute
  if (auto offsetAttr = op->getAttrOfType<IntegerAttr>("instOffset"))
    return offsetAttr.getInt();
  return 0;
}

/// Set the offset on a memory op.
static void setOffset(Operation *op, int64_t offset, MemOpKind kind) {
  OpBuilder builder(op->getContext());
  if (kind == MemOpKind::DS) {
    op->setAttr("offset", builder.getI64IntegerAttr(offset));
  } else {
    op->setAttr("instOffset", builder.getI64IntegerAttr(offset));
  }
}

//===----------------------------------------------------------------------===//
// Memory Offset Optimization Pass
//===----------------------------------------------------------------------===//

struct MemoryOffsetOptPass
    : public waveasm::impl::WAVEASMMemoryOffsetOptBase<MemoryOffsetOptPass> {
  using WAVEASMMemoryOffsetOptBase::WAVEASMMemoryOffsetOptBase;

  void runOnOperation() override {
    Operation *module = getOperation();
    unsigned totalFolded = 0;

    module->walk([&](ProgramOp program) {
      OpBuilder builder(program.getBody().front().getParentOp());

      // Collect memory ops to process (avoid modifying while iterating)
      SmallVector<Operation *, 32> memOps;
      program.walk([&](Operation *op) {
        if (getMemOpKind(op) != MemOpKind::Unknown)
          memOps.push_back(op);
      });

      for (auto *op : memOps) {
        MemOpKind kind = getMemOpKind(op);
        int addrIdx = getAddrOperandIndex(op);
        if (addrIdx < 0 || addrIdx >= static_cast<int>(op->getNumOperands()))
          continue;

        Value addr = op->getOperand(addrIdx);
        int64_t existingOffset = getCurrentOffset(op);
        int64_t maxOffset = getMaxOffset(kind);

        // Set insertion point right before the memory op for any new
        // instructions
        builder.setInsertionPoint(op);
        Location loc = op->getLoc();

        // Extract constants from the address tree
        AddrAnalysis analysis = extractConstant(addr, builder, loc);

        if (analysis.constOffset == 0)
          continue;

        int64_t newOffset = existingOffset + analysis.constOffset;

        if (newOffset < 0)
          continue;

        if (newOffset <= maxOffset) {
          // Constant fits in hardware offset field: fold into offset:N
          op->setOperand(addrIdx, analysis.base);
          setOffset(op, newOffset, kind);
          totalFolded++;
        } else {
          // Constant exceeds hardware offset limit. Still apply the shift
          // distribution to simplify the address tree, but leave the constant
          // as an explicit v_add_u32 with a literal. This enables CSE to
          // deduplicate the shared base (analysis.base) across multiple
          // memory ops that differ only by their constant offset.
          //
          // Check if constant fits in 32-bit integer (V_ADD_U32 limitation)
          if (analysis.constOffset > std::numeric_limits<int32_t>::max() ||
              analysis.constOffset < std::numeric_limits<int32_t>::min()) {
            continue;
          }

          // Before: (base + K) << N + col  [3 ops, K<<N > maxOffset]
          // After:  (base << N + col) + K<<N  [1 op + shared base via CSE]
          auto constImm = builder.getType<ImmType>(analysis.constOffset);
          auto constOp =
              ConstantOp::create(builder, loc, constImm, analysis.constOffset);
          auto vregType = builder.getType<VRegType>(1, 1);
          // NOTE: constant must be src0 (first operand) for VOP2 encoding.
          // src1 must be a VGPR on AMDGCN.
          auto newAddr =
              V_ADD_U32::create(builder, loc, vregType, constOp, analysis.base);
          op->setOperand(addrIdx, newAddr.getResult());
          totalFolded++;
        }
      }

      // Remove dead instructions created by the folding
      // totalDead += removeDeadOps(program);
      // NOTE: Dead code elimination is delegated to the standard Canonicalizer
      // or CSE passes that should run after this pass.
    });

    LLVM_DEBUG(if (totalFolded > 0) {
      llvm::dbgs() << "MemoryOffsetOpt: folded " << totalFolded
                   << " constant address components into offset fields\n";
    });
  }
};

} // namespace
