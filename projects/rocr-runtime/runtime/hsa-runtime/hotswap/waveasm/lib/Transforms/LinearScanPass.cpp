// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Linear Scan Register Allocation Pass
//
// This pass runs the linear scan register allocator and transforms virtual
// register types to physical register types in the IR.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/RegAlloc.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMLINEARSCAN
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

/// Get a value's physical register index from the mapping, falling back to
/// the type's index for already-physical (precolored) values.
/// Returns -1 if the value has no physical register assignment.
static int64_t getEffectivePhysReg(Value value,
                                   const PhysicalMapping &mapping) {
  int64_t physReg = mapping.getPhysReg(value);
  if (physReg >= 0)
    return physReg;
  Type ty = value.getType();
  if (auto pvreg = dyn_cast<PVRegType>(ty))
    return pvreg.getIndex();
  if (auto pareg = dyn_cast<PARegType>(ty))
    return pareg.getIndex();
  if (auto psreg = dyn_cast<PSRegType>(ty))
    return psreg.getIndex();
  return -1;
}

/// Convert a virtual register type to a physical register type.
/// Also handles re-indexing an already-physical type to a new physReg.
/// Returns the original type unchanged if it's not a register type
/// or if physReg < 0.
static Type makePhysicalType(MLIRContext *ctx, Type virtualType,
                             int64_t physReg) {
  if (physReg < 0)
    return virtualType;
  if (auto vreg = dyn_cast<VRegType>(virtualType))
    return PVRegType::get(ctx, physReg, vreg.getSize());
  if (auto sreg = dyn_cast<SRegType>(virtualType))
    return PSRegType::get(ctx, physReg, sreg.getSize());
  if (auto areg = dyn_cast<ARegType>(virtualType))
    return PARegType::get(ctx, physReg, areg.getSize());
  if (auto pvreg = dyn_cast<PVRegType>(virtualType))
    return PVRegType::get(ctx, physReg, pvreg.getSize());
  if (auto psreg = dyn_cast<PSRegType>(virtualType))
    return PSRegType::get(ctx, physReg, psreg.getSize());
  if (auto pareg = dyn_cast<PARegType>(virtualType))
    return PARegType::get(ctx, physReg, pareg.getSize());
  return virtualType;
}

//===----------------------------------------------------------------------===//
// Rematerialization
//===----------------------------------------------------------------------===//

/// Return true when \p op sits inside a LoopOp body at any nesting depth.
static bool isInsideLoopBody(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (isa<LoopOp>(p))
      return true;
  }
  return false;
}

/// Check if an operation can be cheaply rematerialized (cloned near each
/// use to shorten its live range).
///
/// Accepted patterns:
///   - V_MOV_B32 with immediate operands  (accumulator zero-init)
///   - V_MOV_B32 with SGPR source         (scalar-to-vector address copy)
///   - S_MOV_B32 with immediate operands  (scalar constant materialisation)
static bool isRematerializableOp(Operation *op) {
  if (op->getNumResults() != 1)
    return false;

  if (isa<V_MOV_B32>(op)) {
    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp)
        return false;
      if (isa<ConstantOp>(defOp))
        continue;
      // SGPR source: the scalar value dominates the original def, so it
      // also dominates every use site where we might place a clone.
      if (isSGPRType(operand.getType()))
        continue;
      return false;
    }
    return true;
  }

  if (isa<S_MOV_B32>(op)) {
    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || !isa<ConstantOp>(defOp))
        return false;
    }
    return true;
  }

  return false;
}

/// Rematerialize cheap-to-compute VGPR values by cloning their defining ops
/// near each use site, shortening live ranges and reducing peak register
/// pressure at the cost of slightly increased code size.
///
/// A `v_mov_b32 %v, 0` defined at instruction 5 and used at instruction 100
/// holds a VGPR for 95 instructions. After rematerialization the clone
/// appears at instruction 99 with a 1-instruction live range, freeing the
/// VGPR(s) for the other 94 instructions. For multi-register results
/// (e.g. 4-wide accumulators) this frees 4 VGPRs across that span.
static void rematerializeCheapOps(ProgramOp program) {
  // Only rematerialize when the live range exceeds this threshold.
  // Short ranges don't benefit from cloning (the clone itself takes
  // one instruction slot) and would increase code size for no gain.
  // 10 is empirical — long enough that the cloned op meaningfully
  // reduces pressure, short enough to catch real GEMM zero-init patterns.
  // TODO: A future improvement could make this pressure-aware: only remat when
  // current peak pressure exceeds the register budget, avoiding unnecessary
  // code bloat in kernels that already fit.
  constexpr int64_t kMinRematRangeLength = 10;

  LivenessInfo liveness = computeLiveness(program);

  llvm::SmallVector<Operation *> candidates;
  program.walk([&](Operation *op) {
    if (!isRematerializableOp(op))
      return;
    Value result = op->getResult(0);
    // Accept both VGPR and SGPR results (V_MOV_B32 -> VGPR, S_MOV_B32 -> SGPR).
    if (!isVGPRType(result.getType()) && !isSGPRType(result.getType()))
      return;
    const LiveRange *range = liveness.getRange(result);
    if (!range || range->length() <= kMinRematRangeLength)
      return;
    candidates.push_back(op);
  });

  for (Operation *op : candidates) {
    Value result = op->getResult(0);
    bool defOutsideLoop = !isInsideLoopBody(op);
    bool resultIsVGPR = isVGPRType(result.getType());

    llvm::SmallVector<OpOperand *> uses;
    for (OpOperand &use : result.getUses())
      uses.push_back(&use);

    if (uses.empty())
      continue;

    // Clone once per unique user operation to avoid redundant copies when
    // the same op references the value in multiple operand slots.
    llvm::DenseMap<Operation *, Value> cloneCache;

    for (OpOperand *use : uses) {
      Operation *user = use->getOwner();

      // Preserve VALU-free loop bodies: when a VGPR-producing op is defined
      // outside a loop but a use is inside, skip that use.  The value's live
      // range already spans the entire loop body (Pass 2b extends it), so
      // cloning inside wouldn't reduce in-loop pressure — it would only add
      // a VALU instruction to the critical loop path.
      // SALU ops (S_MOV_B32) don't use the VALU pipeline, so they're fine.
      if (defOutsideLoop && resultIsVGPR && isInsideLoopBody(user))
        continue;

      auto it = cloneCache.find(user);
      if (it != cloneCache.end()) {
        use->set(it->second);
      } else {
        OpBuilder builder(user);
        Operation *clone = builder.clone(*op);
        Value cloned = clone->getResult(0);
        use->set(cloned);
        cloneCache[user] = cloned;
      }
    }

    if (result.use_empty())
      op->erase();
  }
}

namespace {

//===----------------------------------------------------------------------===//
// Linear Scan Pass
//===----------------------------------------------------------------------===//

struct LinearScanPass
    : public waveasm::impl::WAVEASMLinearScanBase<LinearScanPass> {
  using WAVEASMLinearScanBase::WAVEASMLinearScanBase;

  void runOnOperation() override {
    Operation *module = getOperation();

    // Process each program.
    module->walk([&](ProgramOp program) {
      if (failed(processProgram(program))) {
        signalPassFailure();
      }
    });
  }

private:
  /// Create a fresh zero-initialized copy of a duplicate init arg to ensure
  /// unique physical registers. This is used when CSE merges identical
  /// zero-initialized accumulators, causing multiple loop block args to be
  /// tied to the same init value. Each block arg needs its own physical
  /// register, so we create a new v_mov_b32/s_mov_b32 from zero.
  ///
  /// PRECONDITION: This should only be called for zero-initialized init args
  /// (e.g., v_mov_b32 %vreg, 0). Calling it for non-zero init args will
  /// produce incorrect zero values silently.
  Value createZeroInitCopy(LoopOp loopOp, Value initArg) {
    OpBuilder copyBuilder(loopOp);
    auto loc = loopOp.getLoc();

    // Create a zero immediate. We always use 0 because this function is
    // only called for duplicate init args produced by CSE merging identical
    // zero-initialized values (e.g., v_mov_b32 vN, 0).
    auto immType = ImmType::get(loopOp->getContext(), 0);
    Value zeroImm = ConstantOp::create(copyBuilder, loc, immType, 0);

    if (isAGPRType(initArg.getType())) {
      // AGPR zero-init: V_MOV_B32 with ARegType destination.
      // The assembly emitter will produce v_accvgpr_write_b32 aN, 0.
      auto aregType = cast<ARegType>(initArg.getType());
      return V_MOV_B32::create(copyBuilder, loc, aregType, zeroImm);
    }
    if (isVGPRType(initArg.getType())) {
      auto vregType = cast<VRegType>(initArg.getType());
      return V_MOV_B32::create(copyBuilder, loc, vregType, zeroImm);
    }
    if (isSGPRType(initArg.getType())) {
      auto sregType = cast<SRegType>(initArg.getType());
      return S_MOV_B32::create(copyBuilder, loc, sregType, zeroImm);
    }
    return nullptr;
  }

  /// Get the accumulator operand from an MFMA op using the interface.
  /// Returns nullptr if the operation is not an MFMA.
  Value getMFMAAccumulator(Operation *op) {
    if (auto mfmaOp = dyn_cast<MFMAOpInterface>(op)) {
      return mfmaOp.getAcc();
    }
    return nullptr;
  }

  LogicalResult processProgram(ProgramOp program) {
    // Collect precolored values from precolored.vreg and precolored.sreg ops
    llvm::DenseMap<Value, int64_t> precoloredValues;
    llvm::DenseSet<int64_t> reservedVGPRs;
    llvm::DenseSet<int64_t> reservedSGPRs;
    llvm::DenseSet<int64_t> reservedAGPRs;

    // Collect tied operand pairs from MFMA ops
    // tiedPairs[result] = accumulator (result should get same phys reg as acc)
    llvm::DenseMap<Value, Value> tiedPairs;

    // Reserve v15 as scratch VGPR for literal materialization in assembly
    // emitter. See AssemblyEmitter.h kScratchVGPR. VOP3 instructions like
    // v_mul_lo_u32 don't support large literal operands, so the emitter
    // generates v_mov_b32 v15, <literal> before such instructions.
    reservedVGPRs.insert(15);

    // ABI SGPRs (kernarg ptr, preload regs, workgroup IDs, SRDs) are
    // reserved via PrecoloredSRegOp ops emitted during translation. The
    // collection loop below picks those up and adds their indices to
    // reservedSGPRs automatically -- no manual reservation needed here.
    //
    // Scalar kernel arg SGPRs (loaded via RawOp s_load_dword) may have had
    // their PrecoloredSRegOps DCE'd by canonicalize. Reserve them explicitly.
    if (auto baseAttr =
            program->getAttrOfType<IntegerAttr>("scalar_sgpr_base")) {
      int64_t base = baseAttr.getInt();
      int64_t count = 0;
      if (auto countAttr =
              program->getAttrOfType<IntegerAttr>("scalar_sgpr_count"))
        count = countAttr.getInt();
      for (int64_t i = 0; i < count; ++i)
        reservedSGPRs.insert(base + i);
    }

    bool collectFailed = false;
    program.walk([&](Operation *op) {
      if (collectFailed)
        return;
      if (auto precoloredVReg = dyn_cast<PrecoloredVRegOp>(op)) {
        int64_t physIdx = precoloredVReg.getIndex();
        int64_t size = precoloredVReg.getSize();
        precoloredValues[precoloredVReg.getResult()] = physIdx;
        for (int64_t i = 0; i < size; ++i) {
          reservedVGPRs.insert(physIdx + i);
        }
      } else if (auto precoloredSReg = dyn_cast<PrecoloredSRegOp>(op)) {
        int64_t physIdx = precoloredSReg.getIndex();
        int64_t size = precoloredSReg.getSize();
        precoloredValues[precoloredSReg.getResult()] = physIdx;
        for (int64_t i = 0; i < size; ++i) {
          reservedSGPRs.insert(physIdx + i);
        }
      } else if (auto precoloredAReg = dyn_cast<PrecoloredARegOp>(op)) {
        int64_t physIdx = precoloredAReg.getIndex();
        int64_t size = precoloredAReg.getSize();
        precoloredValues[precoloredAReg.getResult()] = physIdx;
        for (int64_t i = 0; i < size; ++i) {
          reservedAGPRs.insert(physIdx + i);
        }
      } else if (auto mfmaOp = dyn_cast<MFMAOpInterface>(op)) {
        // For MFMA with VGPR accumulator, tie result to accumulator
        Value acc = mfmaOp.getAcc();
        if (!acc) {
          op->emitError() << "MFMA operation must have at least 3 operands "
                          << "(A, B, accumulator), but found "
                          << op->getNumOperands();
          collectFailed = true;
          return;
        }
        if ((isVGPRType(acc.getType()) &&
             isVGPRType(op->getResult(0).getType())) ||
            (isAGPRType(acc.getType()) &&
             isAGPRType(op->getResult(0).getType()))) {
          // Result should be allocated to same physical register as accumulator
          tiedPairs[op->getResult(0)] = acc;
        }
      }
    });

    if (collectFailed)
      return failure();

    // Handle duplicate init args: if CSE merged identical zero-initialized
    // accumulators, multiple block args may be tied to the same init value.
    // Each block arg needs its own physical register, so insert copies.
    // This must run before liveness analysis since it modifies the IR.
    program.walk([&](LoopOp loopOp) {
      Block &bodyBlock = loopOp.getBodyBlock();
      llvm::DenseSet<Value> usedInitArgs;

      for (unsigned i = 0; i < bodyBlock.getNumArguments(); ++i) {
        if (i < loopOp.getInitArgs().size()) {
          Value initArg = loopOp.getInitArgs()[i];
          if (usedInitArgs.contains(initArg)) {
            Value copy = createZeroInitCopy(loopOp, initArg);
            if (copy) {
              loopOp.getInitArgsMutable()[i].set(copy);
            }
          }
          usedInitArgs.insert(loopOp.getInitArgs()[i]);
        }
      }
    });

    // Rematerialize cheap VGPR ops (v_mov_b32 from immediates) near their
    // use sites to shorten live ranges and reduce peak register pressure.
    // This must run after duplicate-init-arg handling (which creates new
    // V_MOV_B32 ops) and before allocation (which consumes the IR).
    rematerializeCheapOps(program);

    // Create allocator with precolored values and tied operands.
    // MFMA ties come from the local tiedPairs map; loop ties come from
    // the TiedValueClasses built during liveness analysis (see below).
    LinearScanRegAlloc allocator(maxVGPRs, maxSGPRs, maxAGPRs, reservedVGPRs,
                                 reservedSGPRs, reservedAGPRs);
    for (const auto &[value, physIdx] : precoloredValues) {
      allocator.precolorValue(value, physIdx);
    }
    // Add MFMA accumulator ties (these aren't loop ties -- keep them separate)
    for (const auto &[result, acc] : tiedPairs) {
      allocator.addTiedOperand(result, acc);
    }

    // Use bidirectional allocation for VGPRs: long-lived multi-register
    // ranges are allocated from the top, short-lived from the bottom.
    if (bidirectionalThreshold > 0)
      allocator.setVGPRStrategy(
          std::make_unique<BidirectionalStrategy>(bidirectionalThreshold));

    // Run allocation
    auto result = allocator.allocate(program);
    if (failed(result)) {
      return failure();
    }

    auto [mapping, stats] = *result;

    // Handle waveasm.extract ops: result = source[offset].
    // Set the extract result's physical register = source's physReg + offset.
    program.walk([&](ExtractOp extractOp) {
      int64_t sourcePhysReg =
          getEffectivePhysReg(extractOp.getVector(), mapping);
      if (sourcePhysReg >= 0)
        mapping.setPhysReg(extractOp.getResult(),
                           sourcePhysReg + extractOp.getIndex());
    });

    // Handle waveasm.insert ops: result aliases the source vector.
    // Walk InsertOp chains to find the root source with a physical register.
    // The inserted value was independently allocated; emit a mov into the
    // target slot at the InsertOp's position.
    WalkResult insertResult = program.walk([&](InsertOp insertOp) {
      Value source = insertOp.getVector();
      int64_t sourcePhysReg = getEffectivePhysReg(source, mapping);
      while (sourcePhysReg < 0) {
        auto defOp = source.getDefiningOp<InsertOp>();
        if (!defOp)
          break;
        source = defOp.getVector();
        sourcePhysReg = getEffectivePhysReg(source, mapping);
      }
      if (sourcePhysReg < 0)
        return WalkResult::advance();
      mapping.setPhysReg(insertOp.getResult(), sourcePhysReg);

      // Only single-word inserts are lowered for now (one mov).
      if (getRegSize(insertOp.getValue().getType()) != 1) {
        insertOp.emitError("multi-word insert is not yet supported");
        return WalkResult::interrupt();
      }

      // Emit a mov from the inserted value into the target slot.
      int64_t targetSlot = sourcePhysReg + insertOp.getIndex();
      int64_t valuePhysReg = getEffectivePhysReg(insertOp.getValue(), mapping);
      // Skip the mov if the value is already at the target slot.
      if (valuePhysReg == targetSlot)
        return WalkResult::advance();
      OpBuilder movBuilder(insertOp);
      Value val = insertOp.getValue();
      Location loc = insertOp.getLoc();
      MLIRContext *ctx = movBuilder.getContext();
      Value mov;
      if (isSGPRType(val.getType())) {
        auto dstType = PSRegType::get(ctx, targetSlot, 1);
        mov = S_MOV_B32::create(movBuilder, loc, dstType, val);
      } else if (isVGPRType(val.getType())) {
        auto dstType = PVRegType::get(ctx, targetSlot, 1);
        mov = V_MOV_B32::create(movBuilder, loc, dstType, val);
      } else {
        insertOp.emitError("insert with AGPR operands is not yet supported");
        return WalkResult::interrupt();
      }
      DCEProtectOp::create(movBuilder, loc, mov);
      return WalkResult::advance();
    });
    if (insertResult.wasInterrupted())
      return failure();

    // Handle waveasm.pack ops: input[i] gets result's physReg + i.
    // Pack inputs were excluded from the allocation worklists during liveness
    // analysis, so they have no mapping yet. Assign them here from the pack
    // result's contiguous allocation.
    WalkResult packResult = program.walk([&](PackOp packOp) {
      int64_t resultPhysReg = getEffectivePhysReg(packOp.getResult(), mapping);
      if (resultPhysReg < 0) {
        packOp.emitError(
            "pack result has no physical register; cannot assign inputs");
        return WalkResult::interrupt();
      }
      for (auto [i, input] : llvm::enumerate(packOp.getElements())) {
#ifndef NDEBUG
        for (unsigned j = 0; j < i; ++j)
          assert(packOp.getElements()[j] != input &&
                 "duplicate pack input survived liveness copy insertion");
#endif
        mapping.setPhysReg(input, resultPhysReg + static_cast<int64_t>(i));
      }
      return WalkResult::advance();
    });
    if (packResult.wasInterrupted())
      return failure();

    // Transform the IR: replace virtual register types with physical types
    OpBuilder builder(program.getContext());

    // Track values that need type updates
    llvm::DenseMap<Value, Value> valueReplacements;

    // For each operation, update result types from virtual to physical
    program.walk([&](Operation *op) {
      // Skip program op itself
      if (isa<ProgramOp>(op))
        return;

      bool needsUpdate = false;
      SmallVector<Type> newResultTypes;

      for (Value result : op->getResults()) {
        Type ty = result.getType();
        int64_t physReg = mapping.getPhysReg(result);
        Type newTy = makePhysicalType(op->getContext(), ty, physReg);
        newResultTypes.push_back(newTy);
        if (newTy != ty)
          needsUpdate = true;
      }

      // Update the operation's result types if needed
      // Note: MLIR operations typically require replacement, but we can
      // modify in place for dialect ops that support it
      if (needsUpdate && !newResultTypes.empty()) {
        for (size_t i = 0; i < op->getNumResults(); ++i) {
          op->getResult(i).setType(newResultTypes[i]);
        }
      }
    });

    // Update block arguments and result types for region-based control flow.
    // After the walk above, operation results inside loop/if bodies have
    // physical register types, but block arguments and the parent op's
    // result types still have virtual types. We need to propagate the
    // physical types to maintain consistency.
    //
    // Strategy: Use the allocation mapping to update block argument types
    // directly to their assigned physical register types. Then propagate
    // to loop result types and init arg types.
    program.walk([&](LoopOp loopOp) {
      Block &bodyBlock = loopOp.getBodyBlock();

      // Get the condition op (terminator of body)
      auto condOp = dyn_cast<ConditionOp>(bodyBlock.getTerminator());
      if (!condOp)
        return;

      // Update block argument types from the allocation mapping.
      // Block arguments are tied to init args, so they should get the same
      // physical register. Using the mapping directly is more robust than
      // inferring from condition iter_arg types (which may themselves be
      // block arguments not yet updated, e.g. in cross-swap patterns).
      for (unsigned i = 0; i < bodyBlock.getNumArguments(); ++i) {
        BlockArgument blockArg = bodyBlock.getArgument(i);
        Type ty = blockArg.getType();
        int64_t physReg = mapping.getPhysReg(blockArg);

        if (physReg >= 0) {
          blockArg.setType(makePhysicalType(loopOp->getContext(), ty, physReg));
        } else if (i < condOp.getIterArgs().size()) {
          // Fallback: infer from condition iter_arg type (for values not in
          // the mapping, e.g. precolored values)
          Type condType = condOp.getIterArgs()[i].getType();
          if (isa<PVRegType>(condType) || isa<PSRegType>(condType)) {
            blockArg.setType(condType);
          }
        }
      }

      // --- Back-edge register bookkeeping for pipelined loops ---
      //
      // Problem: In pipelined (double-buffered) loops, the liveness pass
      // may "untie" an iter_arg from its block_arg so they get different
      // physical registers.  This happens in two cases:
      //
      //   (a) Swap pattern – the iter_arg at position i is block_arg[j]
      //       (j != i), implementing LDS double-buffer ping-pong.
      //
      //   (b) WAR hazard – a buffer_load iter_arg is interleaved with
      //       MFMAs that still consume the old block_arg value.  Tying
      //       them would make the MFMA read the new load instead of the
      //       old value.
      //
      // After register allocation the LoopLikeOpInterface verifier
      // requires init/blockArg/iterArg types to be compatible.  Blindly
      // coercing iter_arg types to match block_arg types would silently
      // overwrite the physical register the allocator chose, breaking
      // case (b).
      //
      // Solution (two steps):
      //   1. Snapshot each iter_arg's physical register index into the
      //      "_iterArgPhysRegs" attribute *before* any coercion.
      //   2. Coerce only when it's safe: skip swap-pattern block args
      //      and WAR-hazard-separated registers.
      //
      // The AssemblyEmitter reads "_iterArgPhysRegs" to emit the correct
      // back-edge copies/swaps at the loop latch.
      SmallVector<int64_t> origPhysRegs;
      for (unsigned i = 0; i < condOp.getIterArgs().size(); ++i) {
        Type ty = condOp.getIterArgs()[i].getType();
        int64_t idx = -1;
        if (auto psreg = dyn_cast<PSRegType>(ty))
          idx = psreg.getIndex();
        else if (auto pvreg = dyn_cast<PVRegType>(ty))
          idx = pvreg.getIndex();
        origPhysRegs.push_back(idx);
      }
      condOp->setAttr(
          "_iterArgPhysRegs",
          DenseI64ArrayAttr::get(loopOp->getContext(), origPhysRegs));

      // Step 2: Coerce types for LoopLikeOpInterface verifier.
      for (unsigned i = 0; i < bodyBlock.getNumArguments(); ++i) {
        Type blockArgType = bodyBlock.getArgument(i).getType();

        if (i < condOp.getIterArgs().size()) {
          Value iterArg = condOp.getIterArgs()[i];
          if (iterArg.getType() != blockArgType) {
            // Case (a): swap-pattern — iter_arg IS a block_arg of this
            // loop at a different position.  Leave as-is.
            if (auto ba = dyn_cast<BlockArgument>(iterArg);
                ba && ba.getOwner() == &bodyBlock) {
            } else {
              // Case (b): check for WAR-hazard separation.
              int64_t iterPhys = origPhysRegs[i];
              int64_t blockPhys = -1;
              if (auto pvreg = dyn_cast<PVRegType>(blockArgType))
                blockPhys = pvreg.getIndex();
              else if (auto psreg = dyn_cast<PSRegType>(blockArgType))
                blockPhys = psreg.getIndex();

              if (iterPhys >= 0 && blockPhys >= 0 && iterPhys != blockPhys) {
                // Deliberately different registers — do not coerce.
              } else {
                iterArg.setType(blockArgType);
              }
            }
          }
        }

        if (i < loopOp.getInitArgs().size()) {
          Value initArg = loopOp.getInitArgs()[i];
          if (initArg.getType() != blockArgType) {
            // When the allocator assigned init arg and block arg to different
            // physical registers (because the init arg has post-loop uses),
            // skip coercion. Overwriting the init arg's type would corrupt the
            // post-loop value by making it reference the block arg's register,
            // which the loop body mutates. The assembly emitter inserts a copy
            // from the init arg register to the block arg register before the
            // loop entry.
            int64_t initPhys = mapping.getPhysReg(initArg);
            int64_t blockPhys = -1;
            if (auto pvreg = dyn_cast<PVRegType>(blockArgType))
              blockPhys = pvreg.getIndex();
            else if (auto psreg = dyn_cast<PSRegType>(blockArgType))
              blockPhys = psreg.getIndex();

            if (initPhys < 0 || blockPhys < 0 || initPhys == blockPhys)
              initArg.setType(blockArgType);
          }
        }
      }

      for (unsigned i = 0; i < loopOp->getNumResults(); ++i) {
        if (i < bodyBlock.getNumArguments()) {
          loopOp->getResult(i).setType(bodyBlock.getArgument(i).getType());
        }
      }
    });

    // Also update if op result types.
    // Prefer the allocation mapping (which respects loop ties) over the
    // then-yield operand type.  When an if result feeds a loop init arg,
    // the allocator ties it to the loop block arg and both receive the
    // same physical register.  The then-yield operand may carry a
    // *different* physical register (from the inner loop), so copying it
    // blindly would break the LoopLikeOpInterface verifier which requires
    // exact type equality between init args and region iter_args.
    program.walk([&](IfOp ifOp) {
      auto &thenBlock = ifOp.getThenBlock();
      auto yieldOp = dyn_cast<YieldOp>(thenBlock.getTerminator());
      if (!yieldOp)
        return;
      for (unsigned i = 0; i < ifOp->getNumResults(); ++i) {
        Value res = ifOp->getResult(i);
        int64_t physReg = mapping.getPhysReg(res);
        if (physReg >= 0) {
          res.setType(
              makePhysicalType(ifOp->getContext(), res.getType(), physReg));
        } else if (i < yieldOp.getResults().size()) {
          res.setType(yieldOp.getResults()[i].getType());
        }
      }
    });

    return success();
  }
};

} // namespace
