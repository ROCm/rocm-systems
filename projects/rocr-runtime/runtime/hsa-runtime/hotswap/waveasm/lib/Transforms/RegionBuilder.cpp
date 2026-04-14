// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/RegionBuilder.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-region-builder"

using namespace mlir;
using namespace waveasm;

/// Insert a move instruction to coerce `src` into a register of the target
/// type's register class.  Returns the coerced value, or a null Value if the
/// target type is not a recognized register type.
static Value coerceToRegType(OpBuilder &builder, Location loc,
                             TranslationContext &ctx, Type targetType,
                             Value src) {
  auto regClass = getRegClass(targetType);
  if (!regClass)
    return {};
  int64_t size = getRegSize(targetType);
  Type regType;
  switch (*regClass) {
  case RegClass::AGPR:
    regType = ctx.createARegType(size);
    break;
  case RegClass::VGPR:
    regType = ctx.createVRegType(size);
    break;
  case RegClass::SGPR:
    regType = ctx.createSRegType(size);
    break;
  default:
    llvm_unreachable("unhandled register class");
  }
  if (*regClass == RegClass::SGPR)
    return S_MOV_B32::create(builder, loc, regType, src);
  return V_MOV_B32::create(builder, loc, regType, src);
}

/// Check if a value is a memref in LDS (workgroup address space).
/// Used to detect memref iter_args that carry LDS buffer offsets.
static bool isLDSMemRefValue(Value val) {
  auto memrefType = dyn_cast<MemRefType>(val.getType());
  if (!memrefType)
    return false;
  return isLDSMemRef(memrefType);
}

/// Resolve the LDS base offset for a memref value.
/// Returns the SGPR value holding the byte offset, or nullptr.
static Value resolveLDSOffset(Value memref, TranslationContext &ctx,
                              OpBuilder &builder, Location loc) {
  // First check if we have a tracked LDS base offset
  if (auto baseOffset = ctx.getLDSBaseOffset(memref)) {
    Value offsetVal = *baseOffset;
    // If the offset is an immediate, materialize it as an SGPR
    if (isa<ImmType>(offsetVal.getType())) {
      auto sregType = ctx.createSRegType();
      return S_MOV_B32::create(builder, loc, sregType, offsetVal);
    }
    // If already an SGPR, use it directly
    if (isa<SRegType>(offsetVal.getType()) ||
        isa<PSRegType>(offsetVal.getType())) {
      return offsetVal;
    }
    // Otherwise, it's a VGPR - convert to SGPR via readfirstlane
    auto sregType = ctx.createSRegType();
    return V_READFIRSTLANE_B32::create(builder, loc, sregType, offsetVal);
  }
  // No tracked LDS base offset for this memref. Return nullptr so the caller
  // can report a proper error rather than silently emitting offset 0 (which
  // would produce a kernel that reads from the wrong LDS address).
  return nullptr;
}

/// Check if an scf.for region iter arg is used exclusively as a scaled MFMA
/// accumulator (destC operand). Used to decide whether the init arg
/// should be an AGPR on gfx950+ targets.
///
/// Only matches ScaledMFMAOp -- regular MFMAOp uses VGPR accumulators in
/// the C++ backend, so AGPR init would cause a type mismatch.
static bool isAccumulatorIterArg(scf::ForOp forOp, unsigned iterArgIdx) {
  auto regionIterArgs = forOp.getRegionIterArgs();
  if (iterArgIdx >= regionIterArgs.size())
    return false;
  Value regionArg = regionIterArgs[iterArgIdx];

  for (OpOperand &use : regionArg.getUses()) {
    Operation *user = use.getOwner();
    // Only scaled MFMA uses AGPR accumulators in this backend
    if (!isa<amdgpu::ScaledMFMAOp>(user))
      return false;
    if (use.getOperandNumber() != 2)
      return false;
  }
  return !regionArg.use_empty();
}

// Forward declaration
LogicalResult translateOperation(Operation *op, TranslationContext &ctx);

LogicalResult RegionBuilder::translateOp(Operation *op) {
  return translateOperation(op, ctx);
}

LoopOp RegionBuilder::buildLoopFromSCFFor(scf::ForOp forOp) {
  auto &builder = ctx.getBuilder();
  auto loc = forOp.getLoc();

  // Collect loop-carried values: [induction_var, iter_args...]
  SmallVector<Value> initArgs;
  auto lowerBound = ctx.getMapper().getMapped(forOp.getLowerBound());
  if (!lowerBound) {
    forOp.emitError("lower bound not mapped");
    return nullptr;
  }

  // The loop induction variable must be an SGPR (loop increment and comparison
  // use scalar ALU: s_add_u32, s_cmp_lt_u32).  Convert from immediate or VGPR
  // as needed.
  Value lowerBoundValue = *lowerBound;
  if (isa<ImmType>(lowerBoundValue.getType())) {
    auto sregType = ctx.createSRegType();
    lowerBoundValue =
        S_MOV_B32::create(builder, loc, sregType, lowerBoundValue);
  } else if (isVGPRType(lowerBoundValue.getType())) {
    // SAFETY: v_readfirstlane_b32 reads only lane 0. This is correct because
    // scf.for loop bounds must be uniform across all lanes in a wavefront
    // (the loop branch is scalar). If a divergent value reached here, only
    // lane 0's value would be used, silently producing incorrect results.
    auto sregType = ctx.createSRegType();
    lowerBoundValue =
        V_READFIRSTLANE_B32::create(builder, loc, sregType, lowerBoundValue);
  }
  initArgs.push_back(lowerBoundValue);

  // Track which iter_arg indices (0-based, relative to iter_args not including
  // induction var) are memref iter_args. We need this to propagate LDS offsets
  // to block arguments and to handle yield correctly.
  SmallVector<bool> isMemrefIterArg;

  for (auto [iterIdx, arg] : llvm::enumerate(forOp.getInitArgs())) {
    bool isLDSMemref = isLDSMemRefValue(arg);
    isMemrefIterArg.push_back(isLDSMemref);

    if (isLDSMemref) {
      // Memref iter_arg: resolve its LDS base offset and carry it as an SGPR.
      // This implements the ping-pong double-buffering pattern where each
      // iteration swaps which LDS buffer is "current" vs "next".
      Value offsetSgpr = resolveLDSOffset(arg, ctx, builder, loc);
      if (!offsetSgpr) {
        forOp.emitError("LDS memref iter_arg has no tracked base offset");
        return nullptr;
      }
      initArgs.push_back(offsetSgpr);
    } else if (auto mapped = ctx.getMapper().getMapped(arg)) {
      Value mappedValue = *mapped;
      // Convert immediate iter args to register types
      if (isa<ImmType>(mappedValue.getType())) {
        // Infer register size from the original SCF type.
        // For vector types (e.g. vector<4xf32> for MFMA accumulators),
        // we need a register with matching element count and alignment.
        int64_t regSize = 1;
        if (auto vecType = dyn_cast<VectorType>(arg.getType())) {
          regSize = vecType.getNumElements();
        }
        int64_t regAlign = regSize > 1 ? regSize : 1;

        // On gfx950, use AGPRs for MFMA accumulator iter_args.
        // This keeps accumulators in the upper half of the unified register
        // file (a0, a1, ...) so that VGPR indices stay within the 256
        // hardware limit. Without AGPRs, large tiles would need v256+ which
        // the assembler rejects.
        bool useAGPR = llvm::isa<GFX950TargetAttr>(ctx.getTarget());
        if (useAGPR && isAccumulatorIterArg(forOp, iterIdx)) {
          auto aregType = ctx.createARegType(regSize, regAlign);
          mappedValue = V_MOV_B32::create(builder, loc, aregType, mappedValue);
        } else {
          auto vregType = ctx.createVRegType(regSize, regAlign);
          mappedValue = V_MOV_B32::create(builder, loc, vregType, mappedValue);
        }
      }
      initArgs.push_back(mappedValue);
    } else {
      forOp.emitError("init arg not mapped");
      return nullptr;
    }
  }

  // Create loop op (result types inferred from initArgs)
  auto loopOp = LoopOp::create(builder, loc, initArgs);
  Block &bodyBlock = loopOp.getBodyBlock();

  // Map loop induction variable and iter args to block arguments
  ctx.getMapper().mapValue(forOp.getInductionVar(), bodyBlock.getArgument(0));

  size_t argIdx = 1;
  size_t iterIdx = 0;
  for (Value origArg : forOp.getRegionIterArgs()) {
    if (argIdx < bodyBlock.getNumArguments()) {
      Value blockArg = bodyBlock.getArgument(argIdx++);
      ctx.getMapper().mapValue(origArg, blockArg);

      // For memref iter_args, propagate the LDS base offset to the block
      // argument. The block argument is an SGPR carrying the LDS byte offset,
      // and we need vector.load/store/gather_to_lds handlers to be able to
      // look up this offset when they encounter the memref.
      //
      // INVARIANT: LDS base offsets are keyed on the *original SCF Values*
      // (not the mapped waveasm Values). Handlers look up the offset using
      // the unmapped SCF key (origArg), which the value mapper maps to the
      // waveasm block arg. If a remapping step is inserted between
      // RegionBuilder and the handlers, this association will break.
      if (iterIdx < isMemrefIterArg.size() && isMemrefIterArg[iterIdx]) {
        // The block argument itself IS the LDS offset (as an SGPR).
        // Set it as the LDS base offset for the original memref SSA value
        // so that operations inside the loop body can find it.
        ctx.setLDSBaseOffset(origArg, blockArg);
      }
      iterIdx++;
    }
  }

  // Translate loop body
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (failed(translateOp(&op))) {
      // Non-critical failures (e.g. mask computation on vector<Nxindex>) leave
      // results unmapped.  Downstream consumers that need them (maskedload)
      // handle the absence gracefully.  Only abort if the yield operands end
      // up unmapped (checked below).
      op.emitWarning("loop body: skipping untranslated op");
      LLVM_DEBUG(llvm::dbgs()
                 << "loop body: skipping untranslated op: " << op << "\n");
    }
  }

  // Build loop increment and condition
  Value inductionVar = bodyBlock.getArgument(0);

  auto step = ctx.getMapper().getMapped(forOp.getStep());
  auto upperBound = ctx.getMapper().getMapped(forOp.getUpperBound());

  if (!step || !upperBound) {
    forOp.emitError("step or upper bound not mapped");
    return nullptr;
  }

  auto sregType = ctx.createSRegType();
  auto sccType = ctx.createSCCType();
  Value nextIV =
      S_ADD_U32::create(builder, loc, sregType, sccType, inductionVar, *step)
          .getDst();
  // S_CMP only accepts SGPR operands. If upper bound is a VGPR (e.g. from
  // v_min_i32 in split-K trip count computation), convert to SGPR first.
  Value ub = *upperBound;
  if (isVGPRType(ub.getType())) {
    ub = V_READFIRSTLANE_B32::create(builder, loc, sregType, ub);
  }
  Value cond = S_CMP_LT_U32::create(builder, loc, sccType, nextIV, ub);

  // Collect iter args from yield
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> iterArgs;
  iterArgs.push_back(nextIV);

  size_t yieldIdx = 0;
  for (Value result : yieldOp.getResults()) {
    bool isMemref =
        yieldIdx < isMemrefIterArg.size() && isMemrefIterArg[yieldIdx];
    yieldIdx++;

    if (isMemref) {
      // For memref yield values, resolve the LDS base offset of the yielded
      // memref. This is the key to the double-buffer swap: the yield swaps
      // which memref is passed to which iter_arg position, so the SGPR
      // carrying the LDS offset also swaps.
      Value offsetSgpr = resolveLDSOffset(result, ctx, builder, loc);
      if (!offsetSgpr) {
        yieldOp.emitError("yielded LDS memref has no tracked base offset");
        return nullptr;
      }
      iterArgs.push_back(offsetSgpr);
    } else if (auto mapped = ctx.getMapper().getMapped(result)) {
      iterArgs.push_back(*mapped);
    } else {
      yieldOp.emitError("yield result not mapped");
      return nullptr;
    }
  }

  // Create condition terminator
  ConditionOp::create(builder, loc, cond, iterArgs);

  // Map loop results to for results.
  // scf.for results correspond 1:1 with iter_args (no induction variable).
  // waveasm.loop results include the induction variable at index 0,
  // then iter_args starting at index 1.
  assert(forOp.getResults().size() + 1 == loopOp.getResults().size() &&
         "result count mismatch between scf.for and waveasm.loop");
  size_t resIdx = 0;
  for (Value forRes : forOp.getResults()) {
    Value loopResult = loopOp.getResults()[resIdx + 1];
    ctx.getMapper().mapValue(forRes, loopResult);

    // For memref iter_args, propagate the LDS base offset to the loop result.
    // After the loop, the epilogue may use these results in vector.load/store
    // operations that need to know the LDS base offset. The loop result SGPR
    // carries the final (post-swap) LDS offset, so we set it as the LDS base
    // offset for the corresponding scf.for result memref.
    //
    // Index alignment: resIdx corresponds to iter_arg index because scf.for
    // results exclude the induction variable (same as isMemrefIterArg
    // indexing).
    if (resIdx < isMemrefIterArg.size() && isMemrefIterArg[resIdx]) {
      ctx.setLDSBaseOffset(forRes, loopResult);
    }
    resIdx++;
  }

  return loopOp;
}

IfOp RegionBuilder::buildIfFromSCFIf(scf::IfOp ifOp) {
  auto &builder = ctx.getBuilder();
  auto loc = ifOp.getLoc();

  // Get mapped condition
  auto condition = ctx.getMapper().getMapped(ifOp.getCondition());
  if (!condition) {
    ifOp.emitError("condition not mapped");
    return nullptr;
  }

  // Convert condition to sreg if it's an immediate
  // (arith.cmpi maps result to immediate placeholder)
  Value conditionValue = *condition;
  if (isa<ImmType>(conditionValue.getType())) {
    auto sregType = ctx.createSRegType();
    conditionValue = S_MOV_B32::create(builder, loc, sregType, conditionValue);
  }

  // Infer result types by peeking at what the then region will yield
  // This requires translating yield operands first to get their mapped types
  SmallVector<Type> resultTypes;
  auto thenYield =
      cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
  for (Value yieldVal : thenYield.getOperands()) {
    // If already mapped, use its type
    if (auto mapped = ctx.getMapper().getMapped(yieldVal)) {
      resultTypes.push_back(mapped->getType());
    } else {
      // Use a default vreg type if not yet mapped
      // (will be set properly during body translation)
      resultTypes.push_back(ctx.createVRegType());
    }
  }

  // Create if operation with inferred result types
  bool hasElse = !ifOp.getElseRegion().empty();
  auto waveIfOp =
      IfOp::create(builder, loc, resultTypes, conditionValue, hasElse);

  // Translate then region body (yield created later, after type coercion).
  SmallVector<Value> thenYieldVals;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&waveIfOp.getThenBlock());

    for (Operation &op : ifOp.getThenRegion().front().without_terminator()) {
      if (failed(translateOp(&op))) {
        ifOp.emitError("failed to translate then region");
        return nullptr;
      }
    }

    auto scfYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    for (Value res : scfYield.getResults()) {
      if (auto mapped = ctx.getMapper().getMapped(res)) {
        thenYieldVals.push_back(*mapped);
      } else {
        scfYield.emitError("yield result not mapped");
        return nullptr;
      }
    }
  }

  // Translate else region body (yield created later, after type coercion).
  SmallVector<Value> elseYieldVals;
  if (hasElse) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(waveIfOp.getElseBlock());

    for (Operation &op : ifOp.getElseRegion().front().without_terminator()) {
      if (failed(translateOp(&op))) {
        ifOp.emitError("failed to translate else region");
        return nullptr;
      }
    }

    auto scfYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
    for (Value res : scfYield.getResults()) {
      if (auto mapped = ctx.getMapper().getMapped(res)) {
        elseYieldVals.push_back(*mapped);
      } else {
        scfYield.emitError("yield result not mapped");
        return nullptr;
      }
    }
  }

  // Coerce yield values so both branches produce type-compatible results.
  // One branch may produce register values (AGPR/VGPR/SGPR) while the
  // other yields bare immediates (e.g. zero constants).  The waveasm.if
  // verifier requires both branches to yield type-compatible values.
  if (hasElse) {
    for (unsigned i = 0, numYields = thenYieldVals.size(); i < numYields; ++i) {
      Type thenType = thenYieldVals[i].getType();
      Type elseType = elseYieldVals[i].getType();
      if (typesCompatible(thenType, elseType))
        continue;
      // Coerce the immediate side to the register side.
      if (getRegClass(thenType)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(waveIfOp.getElseBlock());
        Value coerced =
            coerceToRegType(builder, loc, ctx, thenType, elseYieldVals[i]);
        if (!coerced) {
          ifOp.emitError("unsupported type coercion for yield result ")
              << i << ": then type " << thenType << ", else type " << elseType;
          return nullptr;
        }
        elseYieldVals[i] = coerced;
      } else if (getRegClass(elseType)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(&waveIfOp.getThenBlock());
        Value coerced =
            coerceToRegType(builder, loc, ctx, elseType, thenYieldVals[i]);
        if (!coerced) {
          ifOp.emitError("unsupported type coercion for yield result ")
              << i << ": then type " << thenType << ", else type " << elseType;
          return nullptr;
        }
        thenYieldVals[i] = coerced;
      } else {
        ifOp.emitError("unsupported type coercion for yield result ")
            << i << ": then type " << thenType << ", else type " << elseType;
        return nullptr;
      }
    }
  }

  // Create then yield and update if-op result types.
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&waveIfOp.getThenBlock());
    YieldOp::create(builder, loc, thenYieldVals);
    for (unsigned i = 0, e = waveIfOp->getNumResults(); i < e; ++i) {
      if (i < thenYieldVals.size())
        waveIfOp->getResult(i).setType(thenYieldVals[i].getType());
    }
  }

  // Create else yield.
  if (hasElse) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(waveIfOp.getElseBlock());
    YieldOp::create(builder, loc, elseYieldVals);
  }

  // Map if results
  for (auto [ifRes, waveRes] :
       llvm::zip(ifOp.getResults(), waveIfOp.getResults())) {
    ctx.getMapper().mapValue(ifRes, waveRes);
  }

  return waveIfOp;
}
