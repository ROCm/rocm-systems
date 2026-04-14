// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Buffer Load Strength Reduction Pass
//
// Replaces per-iteration buffer_load voffset recomputation with precomputed
// voffsets and SGPR soffset bumping. For each buffer_load in a loop whose
// voffset depends on the induction variable:
//
//   1. Symbolically compute the constant stride per candidate by
//      differentiating the address chain w.r.t. the induction variable.
//   2. Group candidates by (SRD, stride); each group shares one soffset.
//   3. Precompute each voffset at iv=initial_value (loop-invariant).
//   4. Carry one soffset per group as SGPR iter_arg (starts at 0).
//   5. Each iteration: soffset += stride (one s_add_u32 per group).
//   6. Set each buffer_load's soffset to the group's soffset.
//
// This eliminates ALL VALU address computation from the loop body.
//
// Why buffer-load-specific and not a general loop strength reduction:
//
// Buffer instructions have effective_address = SRD_base + voffset + soffset,
// where soffset is an SGPR-only scalar offset added at zero VALU cost by the
// hardware. The voffset captures the thread-specific part (loop-invariant),
// and soffset captures the iteration-dependent part (uniform across threads,
// lives in SGPR). This lets us go from N VALU/iter → 1 SALU/SRD-group/iter.
//
// A general strength reduction would produce v_add(base, accumulated_stride)
// — still 1 VALU per iteration — which a subsequent buffer-load peephole
// could split into voffset/soffset. But the only consumers of IV-dependent
// address chains in GEMM loops are buffer_loads and LDS ops (handled by
// LoopAddressPromotion with a different strategy: rotating precomputed VGPRs).
// Splitting into two passes adds an abstraction boundary for one consumer.
// If non-buffer IV-dependent VALU chains appear later, factor out the stride
// computation (symbolic differentiation of address chain) as shared utility.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "waveasm-buffer-load-strength-reduction"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMBUFFERLOADSTRENGTHREDUCTION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

static bool isAddressVALU(Operation *op) {
  return isa<V_LSHLREV_B32, V_LSHL_OR_B32, V_LSHL_ADD_U32, V_ADD_U32, V_SUB_U32,
             V_AND_B32, V_OR_B32, V_LSHRREV_B32, V_MUL_LO_U32, V_MOV_B32,
             V_XOR_B32, V_BFE_U32,
             // Scalar address ops in buffer_load address chains (e.g.
             // s_mul_i32 computing iv * element_bytes).
             S_MUL_I32, S_ADD_U32, S_SUB_U32, S_MOV_B32>(op);
}

static bool isBufferLoad(Operation *op) {
  return isa<BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3,
             BUFFER_LOAD_DWORDX4, BUFFER_LOAD_UBYTE, BUFFER_LOAD_USHORT>(op);
}

static bool isBufferLoadLDS(Operation *op) {
  return isa<BUFFER_LOAD_DWORD_LDS, BUFFER_LOAD_DWORDX4_LDS>(op);
}

// Get voffset from load operation using interface.
static Value getVoffset(Operation *op) {
  if (auto vmemLoad = dyn_cast<VMEMLoadOpInterface>(op))
    return vmemLoad.getVoffset();
  if (auto ldsLoad = dyn_cast<VMEMToLDSLoadOpInterface>(op))
    return ldsLoad.getVoffset();
  return nullptr;
}

// Get SRD/saddr from load operation using interface.
static Value getSrd(Operation *op) {
  if (auto vmemLoad = dyn_cast<VMEMLoadOpInterface>(op))
    return vmemLoad.getSaddr();
  if (auto ldsLoad = dyn_cast<VMEMToLDSLoadOpInterface>(op))
    return ldsLoad.getSrd();
  return nullptr;
}

// Get voffset operand index for setOperand.
static unsigned getVoffsetIdx(Operation *op) {
  return isBufferLoadLDS(op) ? 0 : 1;
}

static bool isDefinedInLoop(Value val, Region *loopRegion) {
  if (auto *defOp = val.getDefiningOp())
    return defOp->getParentRegion() == loopRegion;
  if (auto ba = dyn_cast<BlockArgument>(val))
    return ba.getOwner()->getParent() == loopRegion;
  return false;
}

static void collectVoffsetDeps(Value voffset, Region *loopRegion,
                               llvm::SetVector<Operation *> &deps) {
  SmallVector<Value> worklist;
  worklist.push_back(voffset);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    auto *defOp = v.getDefiningOp();
    if (!defOp)
      continue;
    if (defOp->getParentRegion() != loopRegion)
      continue;
    if (!deps.insert(defOp))
      continue;
    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
}

static std::optional<int64_t> findIVStep(ConditionOp condOp, Block &body) {
  ValueRange condArgs = condOp.getIterArgs();
  if (condArgs.empty())
    return std::nullopt;
  Value nextIV = condArgs[0];
  auto addOp = nextIV.getDefiningOp<S_ADD_U32>();
  if (!addOp)
    return std::nullopt;
  Value iv = body.getArgument(0);
  auto tryExtract = [&](Value maybeSrc,
                        Value maybeConst) -> std::optional<int64_t> {
    if (maybeSrc != iv)
      return std::nullopt;
    return getConstantValue(maybeConst);
  };
  if (auto step = tryExtract(addOp.getSrc0(), addOp.getSrc1()))
    return step;
  if (auto step = tryExtract(addOp.getSrc1(), addOp.getSrc0()))
    return step;
  return std::nullopt;
}

/// Try to extract the loop iteration count from the condition.
/// Looks for s_cmp_lt_u32(next_iv, limit) and computes
/// (limit - iv_init) / iv_step.  Returns nullopt if any part
/// is non-constant or the division is inexact.
static std::optional<int64_t>
estimateMaxIterations(ConditionOp condOp, Value ivInit, int64_t ivStep) {
  Value cond = condOp.getCondition();
  auto cmpOp = cond.getDefiningOp<S_CMP_LT_U32>();
  if (!cmpOp)
    return std::nullopt;
  auto limit = getConstantValue(cmpOp.getSrc1());
  auto init = getConstantValue(ivInit);
  if (!limit || !init || ivStep == 0)
    return std::nullopt;
  int64_t range = *limit - *init;
  if (range <= 0 || range % ivStep != 0)
    return std::nullopt;
  return range / ivStep;
}

static bool dependsOnIV(const llvm::SetVector<Operation *> &deps, Value iv) {
  for (Operation *op : deps)
    for (Value operand : op->getOperands())
      if (operand == iv)
        return true;
  return false;
}

static Value cloneChainBeforeLoop(const llvm::SetVector<Operation *> &deps,
                                  Value targetVoffset, Value ivValue,
                                  LoopOp loopOp, Block &body,
                                  OpBuilder &builder) {
  IRMapping mapping;
  ValueRange initArgs = loopOp.getInitArgs();
  for (unsigned i : llvm::seq(body.getNumArguments()))
    mapping.map(body.getArgument(i), initArgs[i]);
  mapping.map(body.getArgument(0), ivValue);
  for (Operation &op : body)
    if (deps.contains(&op))
      builder.clone(op, mapping);
  return mapping.lookupOrDefault(targetVoffset);
}

/// Run local CSE on ops in [from, to) within a block.  Merges structurally
/// identical ops that share the same opcode, operands, attributes, and result
/// types.  For commutative ops, operand order is canonicalized so that
/// V_ADD_U32(a, b) and V_ADD_U32(b, a) produce the same key.
///
/// This is used after cloneChainBeforeLoop to merge the independently cloned
/// address chains for different buffer_load candidates.  Because all clones
/// share the same leaf operands (init args, precolored registers), the CSE
/// cascades from leaves upward, collapsing N identical chains into one.
///
/// O(N*M) where N = ops in range, M = canonical set size.  Acceptable for
/// typical pre-loop regions (< 300 ops); switch to hash-based lookup if
/// pre-loop regions grow significantly.
static void localCSERange(Block *block, Operation *from, Operation *to) {
  struct CSEEntry {
    Operation *op;
    StringRef opName;
    SmallVector<Value, 4> operands;
    SmallVector<NamedAttribute, 2> attrs;
    SmallVector<Type, 2> resultTypes;
  };
  SmallVector<CSEEntry> canonical;
  SmallVector<Operation *, 16> toErase;

  auto matches = [](const CSEEntry &a, const CSEEntry &b) -> bool {
    if (a.opName != b.opName)
      return false;
    if (a.operands.size() != b.operands.size())
      return false;
    for (unsigned i = 0; i < a.operands.size(); ++i)
      if (a.operands[i] != b.operands[i])
        return false;
    if (a.attrs.size() != b.attrs.size())
      return false;
    for (unsigned i = 0; i < a.attrs.size(); ++i)
      if (a.attrs[i] != b.attrs[i])
        return false;
    if (a.resultTypes.size() != b.resultTypes.size())
      return false;
    for (unsigned i = 0; i < a.resultTypes.size(); ++i)
      if (a.resultTypes[i] != b.resultTypes[i])
        return false;
    return true;
  };

  auto startIt = from ? std::next(Block::iterator(from)) : block->begin();
  for (auto it = startIt; it != Block::iterator(to);) {
    Operation *op = &*it;
    ++it; // Advance before potential erase.

    // Only CSE pure ops (no side effects, no regions).
    if (op->getNumRegions() > 0)
      continue;
    if (!op->hasTrait<OpTrait::IsTerminator>() && !isa<ConstantOp>(op) &&
        !isAddressVALU(op))
      continue;

    CSEEntry entry;
    entry.op = op;
    entry.opName = op->getName().getStringRef();
    entry.operands.append(op->operand_begin(), op->operand_end());
    // Canonicalize commutative binary ops by sorting operand pointers.
    if (entry.operands.size() == 2 && op->hasTrait<OpTrait::IsCommutative>()) {
      if (entry.operands[0].getAsOpaquePointer() >
          entry.operands[1].getAsOpaquePointer())
        std::swap(entry.operands[0], entry.operands[1]);
    }
    for (NamedAttribute attr : op->getAttrs())
      entry.attrs.push_back(attr);
    for (Type ty : op->getResultTypes())
      entry.resultTypes.push_back(ty);

    // Search for a match in existing canonical ops.
    Operation *match = nullptr;
    for (auto &prev : canonical) {
      if (matches(prev, entry)) {
        match = prev.op;
        break;
      }
    }

    if (match) {
      op->replaceAllUsesWith(match);
      toErase.push_back(op);
    } else {
      canonical.push_back(std::move(entry));
    }
  }

  for (Operation *op : llvm::reverse(toErase))
    op->erase();
}

/// Decomposition of a precomputed voffset into shared VGPR base, per-load
/// SGPR addends, and compile-time constant (for instOffset).
struct VoffsetDecomp {
  Value vgprBase;                  // Shared VGPR base (thread-dependent).
  SmallVector<Value, 2> sgprParts; // Per-load SGPR addends (summed in soffset).
  int64_t constAddend;             // Compile-time constant for instOffset.
};

/// Walk V_ADD_U32 chains on a precomputed voffset, stripping both constant
/// addends (foldable into instOffset) and SGPR addends (foldable into a
/// per-load soffset computation).  Stops at the first non-V_ADD_U32 op or
/// when neither operand is a constant or SGPR.
///
/// Example: V_ADD_U32(V_ADD_U32(vgpr_base, sgpr_n_delta), const_2048)
///   → {vgprBase=vgpr_base, sgprAddend=sgpr_n_delta, constAddend=2048}
/// Combine multiple SGPR addend values into one via S_ADD_U32 chain.
/// The combined value is emitted at the builder's current insertion point
/// (before the loop), so it is loop-invariant and only computed once.
/// Zero-valued parts (e.g. s_mul_i32(0, stride) from iv_init=0) are skipped.
static Value _combineSGPRParts(ArrayRef<Value> parts, OpBuilder &builder,
                               Location loc, SRegType sregTy) {
  Value combined = nullptr;
  for (Value part : parts) {
    // Skip provably-zero SGPRs — adding 0 wastes an SALU and extends
    // live ranges for no benefit.  Catches s_mov_b32(0) directly and
    // s_mul_i32(0, x) / s_mul_i32(x, 0) from iv_init=0.
    auto cval = getConstantValue(part);
    if (cval && *cval == 0)
      continue;
    if (auto mulOp = part.getDefiningOp<S_MUL_I32>()) {
      auto c0 = getConstantValue(mulOp.getSrc0());
      auto c1 = getConstantValue(mulOp.getSrc1());
      if ((c0 && *c0 == 0) || (c1 && *c1 == 0))
        continue;
    }
    if (!combined)
      combined = part;
    else
      combined = S_ADD_U32::create(builder, loc, sregTy, sregTy, combined, part)
                     .getDst();
  }
  return combined;
}

static VoffsetDecomp decomposeVoffset(Value voff) {
  VoffsetDecomp result = {voff, {}, 0};

  // Phase 1: strip constant addends from the outermost V_ADD_U32 chain.
  Value current = voff;
  int64_t totalConst = 0;
  while (auto addOp = current.getDefiningOp<V_ADD_U32>()) {
    auto c1 = getConstantValue(addOp.getSrc1());
    if (c1) {
      totalConst += *c1;
      current = addOp.getSrc0();
      continue;
    }
    auto c0 = getConstantValue(addOp.getSrc0());
    if (c0) {
      totalConst += *c0;
      current = addOp.getSrc1();
      continue;
    }
    break;
  }
  result.constAddend = totalConst;

  // Phase 2: iteratively peel SGPR addends from V_ADD_U32(vgpr, sgpr)
  // chains.  Handles multi-level patterns like
  //   V_ADD_U32(V_ADD_U32(base, sgpr1), sgpr2)
  // where both sgpr1 and sgpr2 are folded into per-load soffset.
  while (auto addOp = current.getDefiningOp<V_ADD_U32>()) {
    Value src0 = addOp.getSrc0();
    Value src1 = addOp.getSrc1();
    if (isVGPRType(src0.getType()) && isSGPRType(src1.getType())) {
      result.sgprParts.push_back(src1);
      current = src0;
    } else if (isSGPRType(src0.getType()) && isVGPRType(src1.getType())) {
      result.sgprParts.push_back(src0);
      current = src1;
    } else {
      break;
    }
  }
  result.vgprBase = current;

  return result;
}

// Symbolically compute the stride of the voffset chain w.r.t. the IV.
// Returns the constant stride if determinable, nullopt otherwise.
// Works by computing the "derivative" of each op: IV -> ivStep,
// loop-invariant -> 0, add(a,b) -> da+db, lshl(a,c) -> da<<c, etc.
// The deps SetVector is in reverse topological order (DFS from voffset),
// so we iterate in reverse to process inputs before outputs.
//
// Because the result must be a compile-time integer, non-uniform strides
// are automatically rejected. E.g. v_mul_lo_u32(tid, iv) would need
// getConstantValue(tid) to succeed, which it cannot for a VGPR — so
// the candidate is skipped. No runtime readfirstlane needed.
static std::optional<int64_t>
computeStaticStride(const llvm::SetVector<Operation *> &deps, Value voffset,
                    Value iv, int64_t ivStep) {
  // Maps each Value to its per-IV-step delta (symbolic derivative).
  // Populated in topological order; absent entries are loop-invariant
  // (delta=0).
  llvm::DenseMap<Value, int64_t> delta;
  delta[iv] = ivStep;

  // Values not in the map are loop-invariant (delta=0): defined outside the
  // loop (constants, precolored VGPRs like tid) or non-IV block args
  // (accumulators). Delta=0 means "does not change across iterations", NOT
  // "has value 0" — so tid correctly gets delta=0 even though its per-thread
  // value is nonzero. For linear ops (add, sub, shift) the tid contribution
  // cancels in the derivative; for mul, we require getConstantValue on the
  // loop-invariant operand, which rejects VGPRs like tid.
  auto getDelta = [&](Value v) -> int64_t {
    auto it = delta.find(v);
    return it != delta.end() ? it->second : 0;
  };

  for (Operation *op : llvm::reverse(deps)) {

    if (isa<ConstantOp>(op)) {
      for (Value r : op->getResults())
        delta[r] = 0;
      continue;
    }

    if (auto movOp = dyn_cast<V_MOV_B32>(op)) {
      for (Value r : op->getResults())
        delta[r] = getDelta(movOp.getSrc());
      continue;
    }

    // Validate shift amount is in [0, 31] (32-bit GPU ops) to avoid UB.
    auto validShift = [](std::optional<int64_t> amt) -> std::optional<int64_t> {
      if (!amt || *amt < 0 || *amt > 31)
        return std::nullopt;
      return amt;
    };

    // Overflow-safe multiply for delta propagation.  Returns nullopt if
    // the product would overflow int64_t (e.g. large stride * large
    // element size in a deeply nested address chain).
    auto safeMul = [](int64_t a, int64_t b) -> std::optional<int64_t> {
      if (a == 0 || b == 0)
        return int64_t(0);
      if (a > 0 && b > 0 && a > INT64_MAX / b)
        return std::nullopt;
      if (a < 0 && b < 0 && a < INT64_MAX / b)
        return std::nullopt;
      if (a > 0 && b < 0 && b < INT64_MIN / a)
        return std::nullopt;
      if (a < 0 && b > 0 && a < INT64_MIN / b)
        return std::nullopt;
      return a * b;
    };

    if (auto lshlAddOp = dyn_cast<V_LSHL_ADD_U32>(op)) {
      // v_lshl_add_u32(a, b, c) = (a << b) + c.
      // For lshl_add: treat as add(lshl(a, b), c) — but b must be constant
      // and IV-independent for the shift to be linear.
      int64_t dSrc = getDelta(lshlAddOp.getSrc0());
      int64_t dShift = getDelta(lshlAddOp.getSrc1());
      int64_t dAdd = getDelta(lshlAddOp.getSrc2());
      if (dShift != 0)
        return std::nullopt;
      auto shiftAmt = validShift(getConstantValue(lshlAddOp.getSrc1()));
      if (!shiftAmt)
        return std::nullopt;
      for (Value r : op->getResults())
        delta[r] = (dSrc << *shiftAmt) + dAdd;
      continue;
    }

    if (auto addOp = dyn_cast<V_ADD_U32>(op)) {
      // v_add_u32(a, b) = a + b.
      int64_t d = getDelta(addOp.getSrc0()) + getDelta(addOp.getSrc1());
      for (Value r : op->getResults())
        delta[r] = d;
      continue;
    }

    if (auto subOp = dyn_cast<V_SUB_U32>(op)) {
      int64_t d = getDelta(subOp.getSrc0()) - getDelta(subOp.getSrc1());
      for (Value r : op->getResults())
        delta[r] = d;
      continue;
    }

    if (auto lshlrevOp = dyn_cast<V_LSHLREV_B32>(op)) {
      // lshlrev(amt, src) = src << amt.
      int64_t dAmt = getDelta(lshlrevOp.getSrc0());
      int64_t dSrc = getDelta(lshlrevOp.getSrc1());
      if (dAmt != 0)
        return std::nullopt;
      auto shiftAmt = validShift(getConstantValue(lshlrevOp.getSrc0()));
      if (!shiftAmt)
        return std::nullopt;
      for (Value r : op->getResults())
        delta[r] = dSrc << *shiftAmt;
      continue;
    }

    if (auto lshlOrOp = dyn_cast<V_LSHL_OR_B32>(op)) {
      // lshl_or(src, amt, or_val) = (src << amt) | or_val.
      // Treating OR as ADD is only safe when the OR value is
      // IV-independent (delta=0, bits are disjoint by construction).
      // If the OR operand varies with IV, the bit-overlap semantics
      // diverge from addition and the delta is undefined.
      int64_t dSrc = getDelta(lshlOrOp.getSrc0());
      int64_t dShift = getDelta(lshlOrOp.getSrc1());
      int64_t dOr = getDelta(lshlOrOp.getSrc2());
      if (dShift != 0 || dOr != 0)
        return std::nullopt;
      auto shiftAmt = validShift(getConstantValue(lshlOrOp.getSrc1()));
      if (!shiftAmt)
        return std::nullopt;
      for (Value r : op->getResults())
        delta[r] = dSrc << *shiftAmt;
      continue;
    }

    if (auto mulOp = dyn_cast<V_MUL_LO_U32>(op)) {
      // Linear only if exactly one operand depends on IV.
      int64_t d0 = getDelta(mulOp.getSrc0());
      int64_t d1 = getDelta(mulOp.getSrc1());
      if (d0 != 0 && d1 != 0)
        return std::nullopt;
      if (d0 == 0 && d1 == 0) {
        for (Value r : op->getResults())
          delta[r] = 0;
        continue;
      }
      Value constOperand = d0 == 0 ? mulOp.getSrc0() : mulOp.getSrc1();
      int64_t dVar = d0 != 0 ? d0 : d1;
      auto constVal = getConstantValue(constOperand);
      if (!constVal)
        return std::nullopt;
      auto product = safeMul(dVar, *constVal);
      if (!product)
        return std::nullopt;
      for (Value r : op->getResults())
        delta[r] = *product;
      continue;
    }

    if (auto lshrrevOp = dyn_cast<V_LSHRREV_B32>(op)) {
      // lshrrev(amt, src) = src >> amt.
      // Safe when delta(src) is exactly divisible by 2^amt — no bits lost.
      // Example: address chain does lshl 7 then lshr 8 with IV step 2:
      //   delta(src) = 2*128 = 256, shift = 8, 256 % 256 == 0 → delta = 1.
      int64_t dAmt = getDelta(lshrrevOp.getSrc0());
      int64_t dSrc = getDelta(lshrrevOp.getSrc1());
      if (dAmt != 0)
        return std::nullopt;
      auto shiftAmt = validShift(getConstantValue(lshrrevOp.getSrc0()));
      if (!shiftAmt)
        return std::nullopt;
      int64_t divisor = int64_t(1) << *shiftAmt;
      if (dSrc % divisor != 0)
        return std::nullopt;
      for (Value r : op->getResults())
        delta[r] = dSrc / divisor;
      continue;
    }

    // Scalar multiply: same as V_MUL_LO_U32 but on SGPRs.
    if (auto smulOp = dyn_cast<S_MUL_I32>(op)) {
      int64_t d0 = getDelta(smulOp.getSrc0());
      int64_t d1 = getDelta(smulOp.getSrc1());
      if (d0 != 0 && d1 != 0)
        return std::nullopt;
      if (d0 == 0 && d1 == 0) {
        for (Value r : op->getResults())
          delta[r] = 0;
        continue;
      }
      Value constOperand = d0 == 0 ? smulOp.getSrc0() : smulOp.getSrc1();
      int64_t dVar = d0 != 0 ? d0 : d1;
      auto constVal = getConstantValue(constOperand);
      if (!constVal)
        return std::nullopt;
      auto product = safeMul(dVar, *constVal);
      if (!product)
        return std::nullopt;
      for (Value r : op->getResults())
        delta[r] = *product;
      continue;
    }

    // Scalar add/sub: same as VALU variants.
    if (auto saddOp = dyn_cast<S_ADD_U32>(op)) {
      int64_t d = getDelta(saddOp.getSrc0()) + getDelta(saddOp.getSrc1());
      for (Value r : op->getResults())
        delta[r] = d;
      continue;
    }
    if (auto ssubOp = dyn_cast<S_SUB_U32>(op)) {
      int64_t d = getDelta(ssubOp.getSrc0()) - getDelta(ssubOp.getSrc1());
      for (Value r : op->getResults())
        delta[r] = d;
      continue;
    }

    // Scalar move: pass through delta.
    if (isa<S_MOV_B32>(op)) {
      for (Value r : op->getResults())
        delta[r] = getDelta(op->getOperand(0));
      continue;
    }

    // Bitwise ops (AND, OR, XOR, BFE): nonlinear if IV-dependent.
    bool hasIVDep = false;
    for (Value operand : op->getOperands())
      if (getDelta(operand) != 0)
        hasIVDep = true;
    if (hasIVDep)
      return std::nullopt;
    for (Value r : op->getResults())
      delta[r] = 0;
  }

  auto found = delta.find(voffset);
  return found != delta.end() ? std::optional<int64_t>(found->second)
                              : std::nullopt;
}

struct BufferLoadInfo {
  Operation *loadOp;
  Value voffset;
  Value srd;
  llvm::SetVector<Operation *> deps;
};

static void applyStrengthReduction(LoopOp loopOp) {
  Block &body = loopOp.getBodyBlock();
  auto condOp = dyn_cast<ConditionOp>(body.getTerminator());
  if (!condOp)
    return;

  unsigned numArgs = body.getNumArguments();
  ValueRange condIterArgs = condOp.getIterArgs();
  if (numArgs == 0 || condIterArgs.size() != numArgs)
    return;

  auto ivStep = findIVStep(condOp, body);
  if (!ivStep)
    return;

  Region *loopRegion = &loopOp.getBodyRegion();
  Value iv = body.getArgument(0);

  SmallVector<BufferLoadInfo> candidates;
  llvm::SetVector<Operation *> allDeps;

  for (Operation &op : body) {
    if (!isBufferLoad(&op) && !isBufferLoadLDS(&op))
      continue;
    if (op.getNumOperands() < 3)
      continue;

    Value voffset = getVoffset(&op);
    Value srd = getSrd(&op);
    if (!voffset || !srd)
      continue;

    if (!isDefinedInLoop(voffset, loopRegion))
      continue;

    llvm::SetVector<Operation *> deps;
    collectVoffsetDeps(voffset, loopRegion, deps);

    if (!dependsOnIV(deps, iv))
      continue;

    bool allPure = true;
    for (Operation *dep : deps) {
      if (!isAddressVALU(dep) && !isa<ConstantOp>(dep)) {
        allPure = false;
        break;
      }
    }
    if (!allPure)
      continue;

    allDeps.insert(deps.begin(), deps.end());

    BufferLoadInfo info;
    info.loadOp = &op;
    info.voffset = voffset;
    info.srd = srd;
    info.deps = std::move(deps);
    candidates.push_back(std::move(info));
  }

  if (candidates.empty())
    return;

  LDBG() << "found " << candidates.size() << " buffer_loads to optimize";

  OpBuilder builder(loopOp);
  auto loc = loopOp.getLoc();
  ValueRange initArgs = loopOp.getInitArgs();
  Value ivInit = initArgs[0];
  auto sregType = builder.getType<SRegType>();

  // Compute static stride for each candidate by symbolically differentiating
  // the address chain w.r.t. IV. Drop candidates with non-constant strides
  // (the voffset delta may be non-uniform across iterations).
  SmallVector<int64_t> candidateStrides;
  {
    SmallVector<BufferLoadInfo> filtered;
    for (auto &info : candidates) {
      auto stride = computeStaticStride(info.deps, info.voffset, iv, *ivStep);
      // Reject stride == 0: the voffset chain depends on the IV (checked
      // earlier by dependsOnIV), but the symbolic derivative collapsed to
      // zero — typically because two IV-dependent sub-expressions cancel
      // in the delta while the actual values still vary per iteration
      // (e.g. via right-shift truncation / staircase patterns). Hoisting
      // the voffset and bumping soffset by 0 would freeze the address.
      if (stride && *stride != 0) {
        // Guard against soffset overflow: if the accumulated soffset
        // (stride * iterations) exceeds 32 bits, the buffer address wraps
        // and produces incorrect results.
        auto maxIter = estimateMaxIterations(condOp, ivInit, *ivStep);
        if (maxIter) {
          int64_t maxSoff = std::abs(*stride) * *maxIter;
          if (maxSoff > INT32_MAX) {
            LDBG() << "skipping candidate: soffset would overflow "
                   << "(stride=" << *stride << " * iters=" << *maxIter << " = "
                   << maxSoff << ")";
            continue;
          }
        }
        candidateStrides.push_back(*stride);
        filtered.push_back(std::move(info));
      } else {
        LDBG() << "skipping candidate: cannot determine constant stride";
      }
    }
    candidates = std::move(filtered);
  }

  if (candidates.empty())
    return;

  // Group by (SRD, stride). Loads sharing the same SRD and same constant
  // stride share one soffset iter_arg; different strides get separate ones.
  struct SRDGroup {
    Value srd;
    int64_t stride;
  };
  SmallVector<SRDGroup> groups;
  SmallVector<unsigned> candidateGroupIdx;

  for (auto [i, info] : llvm::enumerate(candidates)) {
    std::optional<unsigned> matchIdx;
    for (auto [g, group] : llvm::enumerate(groups)) {
      if (group.srd == info.srd && group.stride == candidateStrides[i]) {
        matchIdx = g;
        break;
      }
    }
    if (matchIdx) {
      candidateGroupIdx.push_back(*matchIdx);
    } else {
      candidateGroupIdx.push_back(groups.size());
      groups.push_back({info.srd, candidateStrides[i]});
    }
  }

  // Precompute all initial voffsets (at iv=initial_value, soffset=0).
  // Record the last op before cloning so we can CSE the newly cloned ops.
  Operation *lastBeforeClone = nullptr;
  {
    auto it = Block::iterator(loopOp);
    if (it != loopOp->getBlock()->begin())
      lastBeforeClone = &*std::prev(it);
  }

  SmallVector<Value> initialVoffsets;
  for (auto &info : candidates) {
    Value voff = cloneChainBeforeLoop(info.deps, info.voffset, ivInit, loopOp,
                                      body, builder);
    initialVoffsets.push_back(voff);
  }

  // CSE the cloned chains to merge structurally identical address ops.
  // Each cloneChainBeforeLoop call creates a fresh IRMapping, so two
  // candidates sharing the same base expression get independent SSA
  // clones.  Since all clones share the same leaf operands (init args,
  // precolored registers), the CSE cascades from leaves upward and
  // collapses N identical chains into one.
  localCSERange(loopOp->getBlock(), lastBeforeClone, loopOp);

  // Decompose voffsets within each SRD group into (vgprBase, sgprAddend,
  // constAddend).  Constants fold into instOffset; SGPR addends fold into
  // per-load soffset computations (s_add_u32 in the loop body).  This
  // reduces unique voffset VGPRs to the number of distinct VGPR bases —
  // typically 1–2 for B-data loads (matching AITER's pattern).
  constexpr int64_t kMaxInstOffset = 4095; // 12-bit unsigned.
  SmallVector<int64_t> instOffsetDeltas(candidates.size(), 0);
  SmallVector<Value> sgprAddends(candidates.size());

  for (unsigned g = 0; g < groups.size(); ++g) {
    SmallVector<unsigned> groupMembers;
    for (unsigned i = 0; i < candidates.size(); ++i)
      if (candidateGroupIdx[i] == g)
        groupMembers.push_back(i);

    if (groupMembers.size() <= 1)
      continue;

    // Decompose each candidate's precomputed voffset.
    SmallVector<VoffsetDecomp> decomps(candidates.size());
    for (unsigned idx : groupMembers)
      decomps[idx] = decomposeVoffset(initialVoffsets[idx]);

    // Find the canonical VGPR base: pick the first member's base, then
    // verify all members match.  After localCSERange, structurally
    // identical bases share the same SSA Value.
    Value canonicalBase = decomps[groupMembers[0]].vgprBase;

    for (unsigned idx : groupMembers) {
      auto &d = decomps[idx];

      // If the VGPR base doesn't match the canonical one, this candidate
      // has a genuinely different thread-dependent expression.  Fall back
      // to keeping its original voffset unchanged.
      if (d.vgprBase != canonicalBase) {
        LDBG() << "voffset dedup: base mismatch for candidate " << idx;
        continue;
      }

      // For LDS loads, do not fold constants into instOffset.
      // The inst_offset field may not interact correctly with the SRD
      // swizzle stride on GFX950, causing incorrect address computation.
      bool isLDS = isBufferLoadLDS(candidates[idx].loadOp);

      // Validate the constant fits in instOffset budget.
      int64_t existingOffset = 0;
      if (auto attr =
              candidates[idx].loadOp->getAttrOfType<IntegerAttr>("instOffset"))
        existingOffset = attr.getInt();
      if (isLDS || d.constAddend + existingOffset > kMaxInstOffset ||
          d.constAddend < 0) {
        // Constant too large for instOffset (or LDS load).
        // Keep the original voffset but still try SGPR splitting
        // (without the constant part).
        if (!d.sgprParts.empty() && d.constAddend == 0) {
          initialVoffsets[idx] = d.vgprBase;
          sgprAddends[idx] =
              _combineSGPRParts(d.sgprParts, builder, loc, sregType);
        }
        continue;
      }

      // Apply decomposition: shared base + instOffset + sgpr addend.
      initialVoffsets[idx] = canonicalBase;
      instOffsetDeltas[idx] = d.constAddend;
      sgprAddends[idx] = _combineSGPRParts(d.sgprParts, builder, loc, sregType);
    }
  }

  // Build expanded init args: old args + soffset per SRD group (starts at 0).
  SmallVector<Value> expandedInit(initArgs.begin(), initArgs.end());
  unsigned soffsetArgBase = expandedInit.size();
  auto zeroImm = builder.getType<ImmType>(0);
  auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
  auto zeroSoff = S_MOV_B32::create(builder, loc, sregType, zeroConst);
  expandedInit.append(groups.size(), zeroSoff);

  // Build new loop.
  auto newLoop = LoopOp::create(builder, loc, expandedInit);
  Block &newBody = newLoop.getBodyBlock();

  // Map old block args to new block args.
  IRMapping mapping;
  for (unsigned i : llvm::seq(numArgs))
    mapping.map(body.getArgument(i), newBody.getArgument(i));

  // Clone loop body, tracking Operation* mapping for result-less ops.
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&newBody);
  DenseMap<Operation *, Operation *> opMapping;
  for (Operation &op : body) {
    if (isa<ConditionOp>(&op))
      continue;
    Operation *cloned = bodyBuilder.clone(op, mapping);
    opMapping[&op] = cloned;
  }

  // Pre-compute soffset + sgpr_addend for each unique (group, addend) pair
  // at the top of the loop body, so loads sharing the same addend reuse one
  // s_add_u32 instead of each emitting their own.
  DenseMap<std::pair<unsigned, Value>, Value> soffsetCache;
  auto getSoffsetWithAddend = [&](unsigned groupIdx, Value addend) -> Value {
    Value base = newBody.getArgument(soffsetArgBase + groupIdx);
    if (!addend)
      return base;
    auto key = std::make_pair(groupIdx, addend);
    auto it = soffsetCache.find(key);
    if (it != soffsetCache.end())
      return it->second;
    OpBuilder topBuilder = OpBuilder::atBlockBegin(&newBody);
    Value result =
        S_ADD_U32::create(topBuilder, loc, sregType,
                          SCCType::get(topBuilder.getContext()), base, addend)
            .getDst();
    soffsetCache[key] = result;
    return result;
  };

  // Patch buffer_loads: set voffset to precomputed value, soffset to
  // iter_arg (possibly adjusted by per-load SGPR addend), and apply
  // instOffset deltas from voffset deduplication.
  for (auto [i, info] : llvm::enumerate(candidates)) {
    Operation *clonedLoad = opMapping.lookup(info.loadOp);
    assert(clonedLoad && "cloned load not found in op mapping");

    // Replace voffset with precomputed (possibly deduplicated) value.
    clonedLoad->setOperand(getVoffsetIdx(clonedLoad), initialVoffsets[i]);

    // Replace soffset with the group's soffset iter_arg, folding in any
    // per-load SGPR addend extracted during voffset deduplication.
    unsigned groupIdx = candidateGroupIdx[i];
    Value soffsetVal = getSoffsetWithAddend(groupIdx, sgprAddends[i]);
    clonedLoad->setOperand(2, soffsetVal);

    // Apply instOffset delta from voffset deduplication.
    if (instOffsetDeltas[i] != 0) {
      int64_t existingOffset = 0;
      if (auto attr = clonedLoad->getAttrOfType<IntegerAttr>("instOffset"))
        existingOffset = attr.getInt();
      clonedLoad->setAttr(
          "instOffset",
          IntegerAttr::get(IntegerType::get(clonedLoad->getContext(), 64),
                           existingOffset + instOffsetDeltas[i]));
    }
  }

  // Compute next soffsets BEFORE the condition (s_add_u32 clobbers SCC,
  // and s_cmp -> s_cbranch must have no SCC-clobbering ops between them).
  // Find the cloned s_cmp/s_add for IV (the ops that produce the condition)
  // and insert soffset updates before them.
  // s_add_u32 wraps at 2^32. When the loop bound is a compile-time
  // constant, estimateMaxIterations rejects candidates whose accumulated
  // soffset would overflow INT32_MAX. For dynamic bounds (non-constant
  // limit), the check is skipped and we rely on the typical GEMM
  // assumption (stride < 2^20, iterations < 2^12).
  Value newCond = mapping.lookup(condOp.getCondition());

  // Insert soffset updates before the s_cmp that produces the condition.
  Operation *condProducer = newCond.getDefiningOp();
  if (condProducer) {
    OpBuilder preCondBuilder(condProducer);
    SmallVector<Value> nextSoffs;
    for (auto [g, group] : llvm::enumerate(groups)) {
      Value currentSoff = newBody.getArgument(soffsetArgBase + g);
      auto strideImm = preCondBuilder.getType<ImmType>(group.stride);
      Value strideConst =
          ConstantOp::create(preCondBuilder, loc, strideImm, group.stride);
      Value nextSoff = S_ADD_U32::create(preCondBuilder, loc, sregType,
                                         SCCType::get(builder.getContext()),
                                         currentSoff, strideConst)
                           .getDst();
      nextSoffs.push_back(nextSoff);
    }

    SmallVector<Value> newCondIterArgs;
    for (Value v : condIterArgs)
      newCondIterArgs.push_back(mapping.lookup(v));
    for (Value s : nextSoffs)
      newCondIterArgs.push_back(s);

    ConditionOp::create(bodyBuilder, loc, newCond, newCondIterArgs);
  } else {
    SmallVector<Value> newCondIterArgs;
    for (Value v : condIterArgs)
      newCondIterArgs.push_back(mapping.lookup(v));
    for (auto [g, group] : llvm::enumerate(groups)) {
      Value currentSoff = newBody.getArgument(soffsetArgBase + g);
      auto strideImm = bodyBuilder.getType<ImmType>(group.stride);
      Value strideConst =
          ConstantOp::create(bodyBuilder, loc, strideImm, group.stride);
      Value nextSoff = S_ADD_U32::create(bodyBuilder, loc, sregType,
                                         SCCType::get(builder.getContext()),
                                         currentSoff, strideConst)
                           .getDst();
      newCondIterArgs.push_back(nextSoff);
    }
    ConditionOp::create(bodyBuilder, loc, newCond, newCondIterArgs);
  }

  // Replace old loop results.
  for (unsigned i : llvm::seq(numArgs))
    loopOp.getResult(i).replaceAllUsesWith(newLoop.getResult(i));

  // Verify no cross-references.
  bool hasCrossRefs = false;
  for (Operation &op : body) {
    if (isa<ConditionOp>(&op))
      continue;
    for (Value result : op.getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner()->getParentRegion() != &loopOp.getBodyRegion()) {
          hasCrossRefs = true;
          break;
        }
      }
      if (hasCrossRefs)
        break;
    }
    if (hasCrossRefs)
      break;
  }

  if (hasCrossRefs) {
    LDBG() << "cross-references detected, reverting";
    for (unsigned i : llvm::seq(numArgs))
      newLoop.getResult(i).replaceAllUsesWith(loopOp.getResult(i));
    newLoop.erase();
    return;
  }

  loopOp.erase();
}

// Peephole: when a buffer_load has voffset = V_ADD_U32(vgpr, sgpr) and
// soffset = 0, fold the SGPR addend into soffset. This avoids a VALU
// instruction per load by using the hardware scalar offset field.
static void peepholeSoffsetFold(Operation *root) {
  root->walk([&](Operation *op) {
    if (!isBufferLoad(op) && !isBufferLoadLDS(op))
      return;
    if (op->getNumOperands() < 3)
      return;

    unsigned soffsetIdx = 2;
    auto soffsetConst = getConstantValue(op->getOperand(soffsetIdx));
    if (!soffsetConst || *soffsetConst != 0)
      return;

    unsigned voffsetIdx = getVoffsetIdx(op);
    Value voffset = op->getOperand(voffsetIdx);
    auto addOp = voffset.getDefiningOp<V_ADD_U32>();
    if (!addOp)
      return;

    Value src0 = addOp.getSrc0();
    Value src1 = addOp.getSrc1();
    Value vgprPart = nullptr;
    Value sgprPart = nullptr;

    if (isVGPRType(src0.getType()) && isSGPRType(src1.getType())) {
      vgprPart = src0;
      sgprPart = src1;
    } else if (isSGPRType(src0.getType()) && isVGPRType(src1.getType())) {
      vgprPart = src1;
      sgprPart = src0;
    }
    if (!vgprPart || !sgprPart)
      return;

    // Only fold if the V_ADD_U32 is used exclusively by buffer_loads.
    for (Operation *user : addOp->getUsers()) {
      if (!isBufferLoad(user) && !isBufferLoadLDS(user))
        return;
    }

    op->setOperand(voffsetIdx, vgprPart);
    op->setOperand(soffsetIdx, sgprPart);
  });
}

struct BufferLoadStrengthReductionPass
    : public waveasm::impl::WAVEASMBufferLoadStrengthReductionBase<
          BufferLoadStrengthReductionPass> {
  using WAVEASMBufferLoadStrengthReductionBase::
      WAVEASMBufferLoadStrengthReductionBase;

  void runOnOperation() override {
    Operation *module = getOperation();
    SmallVector<LoopOp> loops;
    module->walk([&](LoopOp loopOp) { loops.push_back(loopOp); });
    for (auto loopOp : loops)
      applyStrengthReduction(loopOp);
    peepholeSoffsetFold(module);
  }
};

} // namespace
