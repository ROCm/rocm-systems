// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// AMDGPU, ROCDL, and IREE Dialect Handlers
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-amdgpu-handlers"

using namespace mlir;
using namespace waveasm;

namespace waveasm {

//===----------------------------------------------------------------------===//
// AMDGPU Dialect Handlers
//===----------------------------------------------------------------------===//

LogicalResult handleAMDGPULdsBarrier(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  S_WAITCNT_LGKMCNT::create(builder, loc, 0);
  S_BARRIER::create(builder, loc);

  return success();
}

/// Handle amdgpu.memory_counter_wait - emit s_waitcnt vmcnt(N)
/// This op waits for N outstanding VMEM loads to complete.  It is used by the
/// double-buffer schedule to synchronize global-to-LDS prefetch with the next
/// stage of the pipeline.
LogicalResult handleAMDGPUMemoryCounterWait(Operation *op,
                                            TranslationContext &ctx) {
  auto waitOp = cast<amdgpu::MemoryCounterWaitOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  if (auto loadAttr = waitOp.getLoadAttr()) {
    int64_t vmcnt = loadAttr.getInt();
    S_WAITCNT_VMCNT::create(builder, loc, vmcnt);
  }
  if (auto dsAttr = waitOp.getDsAttr()) {
    int64_t lgkmcnt = dsAttr.getInt();
    S_WAITCNT_LGKMCNT::create(builder, loc, lgkmcnt);
  }
  return success();
}

/// Handle rocdl.s.barrier - emit s_barrier
LogicalResult handleROCDLSBarrier(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  S_BARRIER::create(builder, loc);
  return success();
}

/// Handle rocdl.s.setprio - emit s_setprio <priority>
LogicalResult handleROCDLSetPrio(Operation *op, TranslationContext &ctx) {
  auto setPrioOp = cast<ROCDL::SetPrioOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  int32_t priority = setPrioOp.getPriority();
  S_SETPRIO::create(builder, loc, priority);

  return success();
}

LogicalResult handleAMDGPUMfma(Operation *op, TranslationContext &ctx) {
  auto mfmaOp = cast<amdgpu::MFMAOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  int64_t m = mfmaOp.getM();
  int64_t n = mfmaOp.getN();
  int64_t k = mfmaOp.getK();

  auto srcA = ctx.getMapper().getMapped(mfmaOp.getSourceA());
  auto srcB = ctx.getMapper().getMapped(mfmaOp.getSourceB());
  auto srcC = ctx.getMapper().getMapped(mfmaOp.getDestC());

  if (!srcA || !srcB || !srcC) {
    return op->emitError("MFMA operands not mapped");
  }

  Type srcAType = mfmaOp.getSourceA().getType();
  Type elemType;
  if (auto vecType = dyn_cast<VectorType>(srcAType)) {
    elemType = vecType.getElementType();
  } else {
    elemType = srcAType;
  }

  int64_t accSize = 4;
  if (m == 32 && n == 32) {
    accSize = 16;
  } else if (m == 4 && n == 4) {
    accSize = 4;
  } else if (m == 16 && n == 16 && k == 4) {
    accSize = 16;
  }

  if (elemType.isF64()) {
    if (m == 16 && n == 16)
      accSize = 8;
    else if (m == 4 && n == 4)
      accSize = 2;
  }

  auto vregType = ctx.createVRegType(accSize, 4);
  Value result;

  auto isF16 = [&]() { return elemType.isF16(); };
  auto isBF16 = [&]() { return elemType.isBF16(); };
  auto isI8 = [&]() { return elemType.isInteger(8); };
  auto isF32 = [&]() { return elemType.isF32(); };
  auto isF64 = [&]() { return elemType.isF64(); };

  if (isF16()) {
    if (m == 16 && n == 16 && k == 16) {
      result = V_MFMA_F32_16X16X16_F16::create(builder, loc, vregType, *srcA,
                                               *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 32) {
      result = V_MFMA_F32_16X16X32_F16::create(builder, loc, vregType, *srcA,
                                               *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 8) {
      result = V_MFMA_F32_32X32X8_F16::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 4) {
      auto largeVregType = ctx.createVRegType(16, 4);
      result = V_MFMA_F32_16X16X4_F16::create(builder, loc, largeVregType,
                                              *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 4) {
      auto largeVregType = ctx.createVRegType(32, 4);
      result = V_MFMA_F32_32X32X4_F16::create(builder, loc, largeVregType,
                                              *srcA, *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      result = V_MFMA_F32_4X4X4_F16::create(builder, loc, vregType, *srcA,
                                            *srcB, *srcC);
    } else {
      result = V_MFMA_F32_16X16X16_F16::create(builder, loc, vregType, *srcA,
                                               *srcB, *srcC);
    }
  } else if (isBF16()) {
    if (m == 16 && n == 16 && k == 16) {
      result = V_MFMA_F32_16X16X16_BF16::create(builder, loc, vregType, *srcA,
                                                *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 32) {
      result = V_MFMA_F32_16X16X32_BF16::create(builder, loc, vregType, *srcA,
                                                *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 8) {
      result = V_MFMA_F32_32X32X8_BF16::create(builder, loc, vregType, *srcA,
                                               *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 4) {
      auto largeVregType = ctx.createVRegType(16, 4);
      result = V_MFMA_F32_16X16X4_BF16::create(builder, loc, largeVregType,
                                               *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 4) {
      auto largeVregType = ctx.createVRegType(32, 4);
      result = V_MFMA_F32_32X32X4_BF16::create(builder, loc, largeVregType,
                                               *srcA, *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      result = V_MFMA_F32_4X4X4_BF16::create(builder, loc, vregType, *srcA,
                                             *srcB, *srcC);
    } else {
      result = V_MFMA_F32_16X16X16_BF16::create(builder, loc, vregType, *srcA,
                                                *srcB, *srcC);
    }
  } else if (isI8()) {
    if (m == 16 && n == 16 && k == 16) {
      result = V_MFMA_I32_16X16X16_I8::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 8) {
      result = V_MFMA_I32_32X32X8_I8::create(builder, loc, vregType, *srcA,
                                             *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 4) {
      auto largeVregType = ctx.createVRegType(16, 4);
      result = V_MFMA_I32_16X16X4_I8::create(builder, loc, largeVregType, *srcA,
                                             *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 4) {
      auto largeVregType = ctx.createVRegType(32, 4);
      result = V_MFMA_I32_32X32X4_I8::create(builder, loc, largeVregType, *srcA,
                                             *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      result = V_MFMA_I32_4X4X4_I8::create(builder, loc, vregType, *srcA, *srcB,
                                           *srcC);
    } else {
      result = V_MFMA_I32_16X16X16_I8::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    }
  } else if (isF32()) {
    if (m == 16 && n == 16 && k == 4) {
      result = V_MFMA_F32_16X16X4_F32::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 2) {
      result = V_MFMA_F32_32X32X2_F32::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 1) {
      result = V_MFMA_F32_4X4X1_F32::create(builder, loc, vregType, *srcA,
                                            *srcB, *srcC);
    } else {
      result = V_MFMA_F32_16X16X4_F32::create(builder, loc, vregType, *srcA,
                                              *srcB, *srcC);
    }
  } else if (isF64()) {
    if (m == 16 && n == 16 && k == 4) {
      auto f64VregType = ctx.createVRegType(8, 4);
      result = V_MFMA_F64_16X16X4_F64::create(builder, loc, f64VregType, *srcA,
                                              *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      auto f64VregType = ctx.createVRegType(2, 2);
      result = V_MFMA_F64_4X4X4_F64::create(builder, loc, f64VregType, *srcA,
                                            *srcB, *srcC);
    } else {
      auto f64VregType = ctx.createVRegType(8, 4);
      result = V_MFMA_F64_16X16X4_F64::create(builder, loc, f64VregType, *srcA,
                                              *srcB, *srcC);
    }
  } else {
    result = V_MFMA_F32_16X16X16_F16::create(builder, loc, vregType, *srcA,
                                             *srcB, *srcC);
  }

  ctx.getMapper().mapValue(mfmaOp.getDestD(), result);
  return success();
}

/// Get the scaled MFMA format code from an MLIR element type.
/// These codes map to the cbsz/blgp fields on the hardware instruction:
///   0 = fp8 (E4M3FN), 1 = bf8 (E5M2), 2 = fp6 (E2M3FN),
///   3 = fp6 (E3M2FN), 4 = fp4 (E2M1FN)
static int32_t getScaledMFMAFormatCode(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  if (isa<Float4E2M1FNType>(elemType))
    return 4;
  if (isa<Float6E2M3FNType>(elemType))
    return 2;
  if (isa<Float6E3M2FNType>(elemType))
    return 3;
  if (isa<Float8E4M3FNType>(elemType))
    return 0;
  if (isa<Float8E5M2Type>(elemType))
    return 1;
  // Default to FP4 if type is unknown (e.g., after bitcast to i-type).
  // This is a fallback - if you add a new float type, add an explicit case
  // above.
  LLVM_DEBUG(llvm::dbgs() << "getScaledMFMAFormatCode: unrecognized type, "
                          << "defaulting to FP4 (code 4)\n");
  return 4;
}

LogicalResult handleAMDGPUScaledMfma(Operation *op, TranslationContext &ctx) {
  auto scaledMfmaOp = cast<amdgpu::ScaledMFMAOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Extract M, N, K dimensions from attributes
  int64_t m = scaledMfmaOp.getM();
  int64_t n = scaledMfmaOp.getN();
  int64_t k = scaledMfmaOp.getK();

  // Map operands: sourceA (data), sourceB (data), destC (acc), scaleA, scaleB
  // Note: In amdgpu.scaled_mfma, operands are ordered differently than in the
  // IR The Python implementation showed that operands come in this order: 0:
  // sourceA (data), 1: sourceB (data), 2: destC (acc), 3: scaleA, 4: scaleB
  auto srcA = ctx.getMapper().getMapped(scaledMfmaOp.getSourceA());
  auto srcB = ctx.getMapper().getMapped(scaledMfmaOp.getSourceB());
  auto srcC = ctx.getMapper().getMapped(scaledMfmaOp.getDestC());
  auto scaleA = ctx.getMapper().getMapped(scaledMfmaOp.getScalesA());
  auto scaleB = ctx.getMapper().getMapped(scaledMfmaOp.getScalesB());

  if (!srcA || !srcB || !srcC || !scaleA || !scaleB) {
    return op->emitError("Scaled MFMA operands not mapped");
  }

  // For MXFP4 16x16x128: accumulator is 4 VGPRs
  int64_t accSize = 4;
  if (m == 32 && n == 32) {
    accSize = 16;
  }

  // Use the accumulator input's type to determine the output type.
  // When the accumulator comes from a loop iter_arg initialized as an AGPR
  // (by RegionBuilder for gfx950 ScaledMFMA patterns), the MFMA result
  // should also be an AGPR to maintain the loop-carried type. Otherwise,
  // use VGPRs. This avoids unnecessarily forcing AGPRs on standalone MFMAs
  // (not inside a loop) where they cause accum_offset overhead.
  Type accRegType;
  if (isAGPRType(srcC->getType())) {
    accRegType = ctx.createARegType(accSize, 4);
  } else {
    accRegType = ctx.createVRegType(accSize, 4);
  }
  Value result;

  // Generate the appropriate scaled MFMA instruction based on dimensions
  if (m == 16 && n == 16 && k == 128) {
    result = V_MFMA_SCALE_F32_16X16X128_F8F6F4::create(
        builder, loc, accRegType, *srcA, *srcB, *srcC, *scaleA, *scaleB);
  } else if (m == 32 && n == 32 && k == 64) {
    result = V_MFMA_SCALE_F32_32X32X64_F8F6F4::create(
        builder, loc, accRegType, *srcA, *srcB, *srcC, *scaleA, *scaleB);
  } else {
    return op->emitError("Unsupported scaled MFMA dimensions: ")
           << m << "x" << n << "x" << k;
  }

  // Set cbsz/blgp format code attributes based on source data types
  int32_t aTypeCode =
      getScaledMFMAFormatCode(scaledMfmaOp.getSourceA().getType());
  int32_t bTypeCode =
      getScaledMFMAFormatCode(scaledMfmaOp.getSourceB().getType());
  result.getDefiningOp()->setAttr("cbsz", builder.getI32IntegerAttr(aTypeCode));
  result.getDefiningOp()->setAttr("blgp", builder.getI32IntegerAttr(bTypeCode));

  // Propagate scalesIdxA/scalesIdxB as op_sel/op_sel_hi attributes.
  // These select which element of the packed scale vector is used.
  // Hardware encoding: op_sel[0] = scalesIdxA bit 0,
  //                    op_sel[1] = scalesIdxB bit 0,
  //                    op_sel_hi[0] = scalesIdxA bit 1,
  //                    op_sel_hi[1] = scalesIdxB bit 1.
  int32_t scalesIdxA = scaledMfmaOp.getScalesIdxA();
  int32_t scalesIdxB = scaledMfmaOp.getScalesIdxB();
  if (scalesIdxA != 0 || scalesIdxB != 0) {
    result.getDefiningOp()->setAttr("op_sel_a",
                                    builder.getI32IntegerAttr(scalesIdxA));
    result.getDefiningOp()->setAttr("op_sel_b",
                                    builder.getI32IntegerAttr(scalesIdxB));
  }

  ctx.getMapper().mapValue(scaledMfmaOp.getDestD(), result);
  return success();
}

/// Create a constant SGPR value: ConstantOp + S_MOV_B32.
static Value buildSrdConstantWord(OpBuilder &builder, Location loc,
                                  TranslationContext &ctx, int64_t value) {
  auto sregTy = ctx.createSRegType();
  auto imm = ctx.createImmType(value);
  auto immVal = ConstantOp::create(builder, loc, imm, value);
  return S_MOV_B32::create(builder, loc, sregTy, immVal);
}

/// Build the SRD NUM_RECORDS field (word 2) as a free-register SSA value.
/// Used by PackOp-based SRD construction. Returns an SGPR value suitable
/// for feeding into PackOp.
static Value buildSrdWord2(OpBuilder &builder, Location loc, Operation *op,
                           TranslationContext &ctx) {
  auto castOp = cast<amdgpu::FatRawBufferCastOp>(op);
  Value validBytesVal = castOp.getValidBytes();

  if (validBytesVal) {
    if (auto constVal = getArithConstantValue(validBytesVal)) {
      int64_t clamped = std::min(*constVal, (int64_t)0xFFFFFFFF);
      return buildSrdConstantWord(builder, loc, ctx, clamped);
    }

    auto mapped = ctx.getMapper().getMapped(validBytesVal);
    if (mapped) {
      Value src = *mapped;
      auto sregTy = ctx.createSRegType();
      if (isVGPRType(src.getType()))
        return V_READFIRSTLANE_B32::create(builder, loc, sregTy, src);
      return S_MOV_B32::create(builder, loc, sregTy, src);
    }
  }

  return buildSrdConstantWord(builder, loc, ctx, 0xFFFFFFFF);
}

/// Patch the SRD NUM_RECORDS field (word 2) in place at a known physical
/// register. Used for the suppress-swizzle path where we modify an existing
/// prologue SRD without allocating a new one.
static void patchSrdWord2InPlace(OpBuilder &builder, Location loc,
                                 int64_t srdBase, Operation *op,
                                 TranslationContext &ctx) {
  auto castOp = cast<amdgpu::FatRawBufferCastOp>(op);
  Value validBytesVal = castOp.getValidBytes();
  auto dstType = PSRegType::get(builder.getContext(), srdBase + 2, 1);

  if (validBytesVal) {
    if (auto constVal = getArithConstantValue(validBytesVal)) {
      int64_t clamped = std::min(*constVal, (int64_t)0xFFFFFFFF);
      auto imm = ctx.createImmType(clamped);
      auto immVal = ConstantOp::create(builder, loc, imm, clamped);
      auto result = S_MOV_B32::create(builder, loc, dstType, immVal);
      DCEProtectOp::create(builder, loc, result);
      return;
    }

    auto mapped = ctx.getMapper().getMapped(validBytesVal);
    if (mapped) {
      Value src = *mapped;
      if (isVGPRType(src.getType())) {
        auto result = V_READFIRSTLANE_B32::create(builder, loc, dstType, src);
        DCEProtectOp::create(builder, loc, result);
      } else {
        auto sccType = ctx.createSRegType();
        auto zeroImm = ctx.createImmType(0);
        auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
        S_ADD_U32::create(builder, loc, dstType, sccType, src, zeroConst);
      }
      return;
    }
  }

  auto imm = ctx.createImmType(0xFFFFFFFF);
  auto immVal = ConstantOp::create(builder, loc, imm, 0xFFFFFFFF);
  auto result = S_MOV_B32::create(builder, loc, dstType, immVal);
  DCEProtectOp::create(builder, loc, result);
}

LogicalResult handleFatRawBufferCast(Operation *op, TranslationContext &ctx) {
  if (op->getNumOperands() < 1) {
    return success();
  }

  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto srcMapped = ctx.getMapper().getMapped(op->getOperand(0));
  if (!srcMapped) {
    return success();
  }

  auto castOp = cast<amdgpu::FatRawBufferCastOp>(op);
  Value validBytesVal = castOp.getValidBytes();

  bool hasCacheSwizzle = false;
  int64_t swizzleStride = 0;
  if (op->getNumOperands() >= 3) {
    // Enable cache swizzle whenever the stride operand exists, even if
    // the value is dynamic (not a compile-time constant). The SRD
    // encoding (0x40400000 in word 1, 0x27000 in word 3) is constant
    // regardless of the actual stride value.
    hasCacheSwizzle = true;
    if (auto swizzleVal = getArithConstantValue(op->getOperand(2))) {
      swizzleStride = *swizzleVal;
    }
  }

  // Check whether validBytes requires a new SRD (non-max value).
  // The hardware max is (2^31 - 1 - elem_bytes) which varies by element type.
  // Any value >= 0x70000000 is treated as "hardware max" (a safe threshold).
  bool hasNonMaxValidBytes = false;
  if (validBytesVal) {
    if (auto constVal = getArithConstantValue(validBytesVal)) {
      hasNonMaxValidBytes = (*constVal < 0x70000000);
    } else {
      hasNonMaxValidBytes = true;
    }
  }

  // Check if the source (or its memref.cast source) has a pending SRD base
  // adjustment from a linearized reinterpret_cast (use_buffer_ops path).
  Value src = op->getOperand(0);
  auto *adj = ctx.getPendingSRDBaseAdjust(src);
  if (!adj) {
    if (auto castOp2 = src.getDefiningOp<memref::CastOp>())
      adj = ctx.getPendingSRDBaseAdjust(castOp2.getSource());
  }

  // When adj exists AND the cast doesn't need non-max validBytes, emit
  // the SRD adjustment eagerly.  The fat_raw_buffer_cast is typically
  // outside any scf.for loop while the loads are inside; deferring to
  // load-time would place the SRD init RawOps inside the loop body.
  if (adj && !hasNonMaxValidBytes) {
    Value srd = emitSRDBaseAdjustment(*adj, op->getResult(0), ctx, loc);
    ctx.getMapper().mapValue(op->getResult(0), srd);
    return success();
  }

  // Suppress cache swizzle for SRDs whose consumers are exclusively
  // direct buffer loads (vector.load).  Swizzle is only meaningful for
  // gather_to_lds which writes into swizzled LDS.  When the handler
  // passes through (hasNonMaxValidBytes == false), the source SRD
  // naturally lacks swizzle.  We must replicate that behaviour when
  // constructing a new SRD for non-max validBytes.
  bool suppressWord3Swizzle = false;
  if (hasCacheSwizzle) {
    bool hasGatherUser = false;
    for (auto &use : op->getResult(0).getUses()) {
      if (use.getOwner()->getName().getStringRef() == "amdgpu.gather_to_lds") {
        hasGatherUser = true;
        break;
      }
      // Also check through fat_raw_buffer_cast chains (loop body re-casts).
      if (isa<amdgpu::FatRawBufferCastOp>(use.getOwner())) {
        for (auto &innerUse : use.getOwner()->getResult(0).getUses()) {
          if (innerUse.getOwner()->getName().getStringRef() ==
              "amdgpu.gather_to_lds") {
            hasGatherUser = true;
            break;
          }
        }
        if (hasGatherUser)
          break;
      }
    }
    if (!hasGatherUser)
      suppressWord3Swizzle = true;
  }

  // For direct-load-only SRDs (no gather_to_lds consumers), pass through
  // and let the downstream handler construct the SRD with proper base
  // adjustment.  The source SRD carries a tight NUM_RECORDS from
  // computeBufferSizeFromMemRef, so OOB prefetches in epilogue-eliminated
  // loop iterations silently return zero.
  //
  // When hasNonMaxValidBytes and no pending base adjustment, the MLIR
  // specifies a tighter bound (e.g. the real buffer size for epilogue
  // elimination).  Overwrite the source SRD's NUM_RECORDS in place so
  // that downstream vector.loads get hardware OOB protection without
  // allocating a new SRD (which would risk SGPR overflow).
  if (suppressWord3Swizzle) {
    if (hasNonMaxValidBytes && !adj) {
      // Patch word 2 (NUM_RECORDS) in the source SRD. For prologue SRDs
      // with known physical base, write directly. For PackOp-built SRDs,
      // use InsertOp to replace the single word without a full rebuild.
      int64_t srdBase = -1;
      if (auto psreg = dyn_cast<PSRegType>(srcMapped->getType()))
        srdBase = psreg.getIndex();
      else if (auto defOp = srcMapped->getDefiningOp())
        if (defOp->getName().getStringRef() == "waveasm.precolored.sreg")
          if (auto indexAttr = defOp->getAttrOfType<IntegerAttr>("index"))
            srdBase = indexAttr.getInt();
      if (srdBase >= 0) {
        patchSrdWord2InPlace(builder, loc, srdBase, op, ctx);
        ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
        return success();
      }
      // PackOp-built SRD: use InsertOp to replace word 2 in place.
      Value newWord2 = buildSrdWord2(builder, loc, op, ctx);
      auto srdType = srcMapped->getType();
      Value updated =
          InsertOp::create(builder, loc, srdType, newWord2, *srcMapped, 2);
      ctx.getMapper().mapValue(op->getResult(0), updated);
      return success();
    } else {
      ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
      if (adj) {
        Value numRecordsOverride;
        if (hasNonMaxValidBytes && validBytesVal) {
          if (auto mapped = ctx.getMapper().getMapped(validBytesVal))
            numRecordsOverride = *mapped;
        }
        ctx.setPendingSRDBaseAdjust(op->getResult(0), adj->elementOffset,
                                    adj->srcSrdBase, adj->elementBytes,
                                    numRecordsOverride, adj->srcSrdValue);
      }
      return success();
    }
  }

  if (!hasCacheSwizzle && !hasNonMaxValidBytes) {
    ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
    return success();
  }

  // Resolve the source SRD as an SSA value. Prefer previously-built
  // PackOp SRD, then index-based lookup, then mapped precolored value.
  Value srcSrd;

  // Prefer previously-built SRD value (from PackOp construction).
  if (auto srcSrdVal = ctx.getSRDValue(op->getOperand(0))) {
    srcSrd = *srcSrdVal;
  }

  if (!srcSrd) {
    int64_t srcSrdBase = -1;

    // When we have a pending SRD base adjustment, use its source SRD base
    // (the prologue-allocated SRD).
    if (adj)
      srcSrdBase = adj->srcSrdBase;

    // Prefer previously-built swizzle SRD over the kernel-arg SRD, since
    // the kernel-arg register may be repurposed as a loop counter.
    if (srcSrdBase < 0) {
      if (auto srcSrdIdx = ctx.getSRDIndex(op->getOperand(0)))
        srcSrdBase = *srcSrdIdx;
    }

    if (srcSrdBase < 0) {
      if (auto defOp = srcMapped->getDefiningOp())
        if (defOp->getName().getStringRef() == "waveasm.precolored.sreg")
          if (auto indexAttr = defOp->getAttrOfType<IntegerAttr>("index"))
            srcSrdBase = indexAttr.getInt();
    }

    if (srcSrdBase < 0) {
      if (auto psreg = dyn_cast<PSRegType>(srcMapped->getType()))
        srcSrdBase = psreg.getIndex();
    }

    if (srcSrdBase >= 0) {
      auto srcSrdType = ctx.createSRegType(4, 4);
      srcSrd =
          PrecoloredSRegOp::create(builder, loc, srcSrdType, srcSrdBase, 4);
    }
  }

  // If all lookups failed but srcMapped is already a 4-wide SRD (e.g.
  // a PackOp result from a prior fat_raw_buffer_cast), use it directly.
  if (!srcSrd) {
    if (auto sregTy = dyn_cast<SRegType>(srcMapped->getType())) {
      if (sregTy.getSize() == 4)
        srcSrd = *srcMapped;
    }
  }

  if (!srcSrd) {
    ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
    return success();
  }

  // Extract individual SRD words from the source.
  auto sregTy = ctx.createSRegType();
  Value srcWord0 = ExtractOp::create(builder, loc, sregTy, srcSrd, 0);
  Value srcWord1 = ExtractOp::create(builder, loc, sregTy, srcSrd, 1);
  Value srcWord2 = ExtractOp::create(builder, loc, sregTy, srcSrd, 2);

  // Word 0: copy base address lo.
  Value word0 = S_MOV_B32::create(builder, loc, sregTy, srcWord0);

  // Word 1: optionally apply cache swizzle stride.
  Value word1;
  if (hasCacheSwizzle && !suppressWord3Swizzle) {
    int64_t srdWord1Bits = static_cast<int64_t>(swizzleStride | 0x4000) << 16;
    auto maskImm = ctx.createImmType(0xffff);
    auto maskVal = ConstantOp::create(builder, loc, maskImm, 0xffff);
    Value cleared = S_AND_B32::create(builder, loc, sregTy, ctx.createSCCType(),
                                      srcWord1, maskVal)
                        .getDst();
    auto swizzleImm = ctx.createImmType(srdWord1Bits);
    auto swizzleVal =
        ConstantOp::create(builder, loc, swizzleImm, srdWord1Bits);
    word1 = S_OR_B32::create(builder, loc, sregTy, ctx.createSCCType(), cleared,
                             swizzleVal)
                .getDst();
  } else {
    word1 = S_MOV_B32::create(builder, loc, sregTy, srcWord1);
  }

  // Word 2: num_records.
  Value word2;
  if (hasNonMaxValidBytes) {
    word2 = buildSrdWord2(builder, loc, op, ctx);
  } else {
    // Copy num_records from source SRD so sentinel offsets remain OOB.
    word2 = S_MOV_B32::create(builder, loc, sregTy, srcWord2);
  }

  // Word 3: flags.
  uint64_t word3Flags =
      (hasCacheSwizzle && !suppressWord3Swizzle) ? 0x27000 : 0x20000;
  Value word3 = buildSrdConstantWord(builder, loc, ctx, word3Flags);

  // Pack into a 4-wide SGPR SRD.
  auto srdType = ctx.createSRegType(4, 4);
  auto newSrd = PackOp::create(builder, loc, srdType,
                               ValueRange{word0, word1, word2, word3});
  ctx.getMapper().mapValue(op->getResult(0), newSrd);
  if (!suppressWord3Swizzle)
    ctx.setCacheSwizzleStride(op->getResult(0), swizzleStride);

  // Record the SRD value for the SOURCE memref so that subsequent
  // fat_raw_buffer_cast ops on the same source copy from this stable
  // SRD rather than from the kernel-arg SRD whose register may be
  // repurposed as a loop counter.
  ctx.setSRDValue(op->getOperand(0), newSrd);

  return success();
}

/// Handle amdgpu.gather_to_lds - full implementation with dynamic M0
/// computation
LogicalResult handleGatherToLds(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  Value srcBuffer = op->getOperand(0);
  auto srcMapped = ctx.getMapper().getMapped(srcBuffer);
  if (!srcMapped) {
    return op->emitError("source buffer not mapped");
  }

  int64_t elemBytes = 2;
  if (auto transferTypeAttr = op->getAttrOfType<TypeAttr>("transferType")) {
    if (auto vecType = dyn_cast<VectorType>(transferTypeAttr.getValue())) {
      auto elemType = vecType.getElementType();
      elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    }
  }

  // Create voffset for global memory access
  // When the srcIndex is base + constant, we compute the byte-scaled voffset
  // as base_voff + (constant * elemBytes) to enable CSE of base_voff across
  // multiple gathers that share the same base index.
  Value voff;
  if (op->getNumOperands() > 1) {
    Value srcIndex = op->getOperand(1);
    auto srcIndexMapped = ctx.getMapper().getMapped(srcIndex);
    if (srcIndexMapped) {
      auto vregType = ctx.createVRegType();
      if (elemBytes != 1 && llvm::isPowerOf2_64(elemBytes)) {
        int64_t shiftAmt = llvm::Log2_64(elemBytes);

        // Check if srcIndex is V_ADD_U32(base, constant) - if so, distribute
        // the byte scaling: (base + K) * elemBytes = base * elemBytes + K *
        // elemBytes This enables CSE of base*elemBytes across gathers.
        Value idxVal = *srcIndexMapped;
        auto addOp = idxVal.getDefiningOp<V_ADD_U32>();
        Value baseOperand;
        int64_t constVal = 0;
        bool hasConstAdd = false;

        if (addOp) {
          // Check if either operand is a constant
          if (auto constOp = addOp.getSrc1().getDefiningOp<ConstantOp>()) {
            constVal = constOp.getValue();
            baseOperand = addOp.getSrc0();
            hasConstAdd = true;
          } else if (auto constOp =
                         addOp.getSrc0().getDefiningOp<ConstantOp>()) {
            constVal = constOp.getValue();
            baseOperand = addOp.getSrc1();
            hasConstAdd = true;
          }
        }

        if (hasConstAdd) {
          // Distribute: (base + K) << shift = (base << shift) + (K << shift)
          auto shiftImm = ctx.createImmType(shiftAmt);
          auto shiftConst =
              ConstantOp::create(builder, loc, shiftImm, shiftAmt);
          Value baseVoff = V_LSHLREV_B32::create(builder, loc, vregType,
                                                 shiftConst, baseOperand);
          int64_t scaledConst = constVal * elemBytes;
          auto scaledImm = ctx.createImmType(scaledConst);
          auto scaledConstOp =
              ConstantOp::create(builder, loc, scaledImm, scaledConst);
          // NOTE: constant must be src0 (first operand) for VOP2 encoding.
          // src1 must be a VGPR on AMDGCN.
          voff = V_ADD_U32::create(builder, loc, vregType, scaledConstOp,
                                   baseVoff);
        } else {
          // Simple case: just shift
          auto shiftImm = ctx.createImmType(shiftAmt);
          auto shiftConst =
              ConstantOp::create(builder, loc, shiftImm, shiftAmt);
          voff = V_LSHLREV_B32::create(builder, loc, vregType, shiftConst,
                                       *srcIndexMapped);
        }
      } else if (elemBytes != 1) {
        auto vregType = ctx.createVRegType();
        auto scaleImm = ctx.createImmType(elemBytes);
        auto scaleConst = ConstantOp::create(builder, loc, scaleImm, elemBytes);
        voff = V_MUL_LO_U32::create(builder, loc, vregType, scaleConst,
                                    *srcIndexMapped);
      } else {
        auto vregType = ctx.createVRegType();
        voff = V_MOV_B32::create(builder, loc, vregType, *srcIndexMapped);
      }
    }
  }
  if (!voff) {
    auto immType = ctx.createImmType(0);
    auto zeroConst = ConstantOp::create(builder, loc, immType, 0);
    auto vregType = ctx.createVRegType();
    voff = V_MOV_B32::create(builder, loc, vregType, zeroConst);
  }

  // Get LDS info.
  // The LDS base offset can be:
  //   (a) A constant (from memref.view with static byte offset) -> ImmType
  //   (b) An SGPR (from memref iter_arg in a pipelined loop) ->
  //   SRegType/PSRegType
  // We track both cases and handle them appropriately when computing m0.
  int64_t ldsBaseOffsetConst = 0;
  int64_t ldsRowStride = 0;
  bool hasLdsBaseOffset = false;
  bool hasLdsDynamicBaseOffset = false;
  Value ldsDynamicBaseOffset;

  if (op->getNumOperands() > 2) {
    mlir::Value ldsMemref = op->getOperand(2);
    if (auto baseOffset = ctx.getLDSBaseOffset(ldsMemref)) {
      if (auto immType = dyn_cast<ImmType>(baseOffset->getType())) {
        ldsBaseOffsetConst = immType.getValue();
        hasLdsBaseOffset = true;
      } else {
        // Dynamic offset: SGPR-carried from a memref iter_arg (pipelined loop)
        hasLdsDynamicBaseOffset = true;
        ldsDynamicBaseOffset = *baseOffset;
      }
    }
    if (auto memrefType = dyn_cast<MemRefType>(ldsMemref.getType())) {
      auto shape = memrefType.getShape();
      int64_t elementBytes = 2;
      if (auto elemType = memrefType.getElementType()) {
        elementBytes = elemType.getIntOrFloatBitWidth() / 8;
      }
      if (shape.size() >= 2) {
        ldsRowStride = shape[1] * elementBytes;
      }
    }
  }

  int64_t ldsElemBytes = 2;
  if (op->getNumOperands() > 2) {
    if (auto memrefType = dyn_cast<MemRefType>(op->getOperand(2).getType())) {
      if (auto elemType = memrefType.getElementType()) {
        ldsElemBytes = elemType.getIntOrFloatBitWidth() / 8;
      }
    }
  }

  int64_t m0Const = ldsBaseOffsetConst;
  // Can only use immediate m0 if the base offset is constant (not dynamic SGPR)
  bool canUseImmediateM0 = hasLdsBaseOffset && !hasLdsDynamicBaseOffset;
  int64_t colIdxConst = 0;
  bool hasConstCol = false;

  if (op->getNumOperands() > 4) {
    mlir::Value ldsColIndex = op->getOperand(4);
    auto ldsColMapped = ctx.getMapper().getMapped(ldsColIndex);
    if (ldsColMapped) {
      if (auto immType = dyn_cast<ImmType>(ldsColMapped->getType())) {
        colIdxConst = immType.getValue();
        hasConstCol = true;
      }
    }
  }

  if (op->getNumOperands() > 3) {
    mlir::Value ldsRowIndex = op->getOperand(3);
    auto ldsRowMapped = ctx.getMapper().getMapped(ldsRowIndex);
    if (ldsRowMapped) {
      if (auto immType = dyn_cast<ImmType>(ldsRowMapped->getType())) {
        int64_t rowIdxConst = immType.getValue();
        if (ldsRowStride > 0) {
          m0Const = ldsBaseOffsetConst + rowIdxConst * ldsRowStride;
          if (hasConstCol) {
            m0Const += colIdxConst * ldsElemBytes;
          }
        } else {
          m0Const = ldsBaseOffsetConst + rowIdxConst;
        }
        // Can use immediate only if base is static and col is const
        canUseImmediateM0 =
            !hasLdsDynamicBaseOffset && (hasConstCol || (colIdxConst == 0));
      } else {
        canUseImmediateM0 = false;
      }
    }
  }

  // Set M0 for LDS destination
  if (canUseImmediateM0) {
    auto m0Imm = ctx.createImmType(m0Const);
    auto m0ConstVal = ConstantOp::create(builder, loc, m0Imm, m0Const);
    S_MOV_B32_M0::create(builder, loc, m0ConstVal);
  } else {
    // Need to compute m0 dynamically.
    // This path handles both:
    //   (1) Dynamic row/col indices (original path)
    //   (2) Dynamic base offset from SGPR-carried memref iter_args (new)
    Value m0Val;

    // Start from the dynamic base offset if we have one (pipelined loop)
    if (hasLdsDynamicBaseOffset) {
      // Start m0 computation from the SGPR-carried LDS base offset.
      // We'll use SALU when possible since the base is already an SGPR.
      // (ldsDynamicBaseOffset is guaranteed non-null when the flag is set)
      m0Val = ldsDynamicBaseOffset;
    }

    if (op->getNumOperands() > 3) {
      mlir::Value ldsRowIndex = op->getOperand(3);
      auto ldsRowMapped = ctx.getMapper().getMapped(ldsRowIndex);
      if (ldsRowMapped) {
        // Lambda to convert SGPR to VGPR if needed
        auto convertToVgpr = [&](Value val) -> Value {
          if (!isVGPRType(val.getType())) {
            auto zeroImm = ctx.createImmType(0);
            auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
            Value zeroVgpr = V_MOV_B32::create(builder, loc,
                                               ctx.createVRegType(), zeroConst);
            return V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                     zeroVgpr, val);
          }
          return val;
        };

        // Compute row*stride contribution
        Value rowContrib = *ldsRowMapped;
        bool rowIsConst = isa<ImmType>(rowContrib.getType());
        int64_t rowConstBytes = 0;
        if (ldsRowStride > 1) {
          // If the row index is a constant, we can stay in SGPR domain
          if (auto immType = dyn_cast<ImmType>(rowContrib.getType())) {
            rowConstBytes = immType.getValue() * ldsRowStride;
            rowIsConst = true;
          } else if (!isVGPRType(rowContrib.getType())) {
            // Row index is an SGPR (e.g., from subgroup_broadcast via
            // v_readfirstlane). Keep computation in SALU to avoid
            // unnecessary VGPR pressure.
            auto sregType = ctx.createSRegType();
            if (llvm::isPowerOf2_64(ldsRowStride)) {
              int64_t shiftAmt = llvm::Log2_64(ldsRowStride);
              auto shiftImm = ctx.createImmType(shiftAmt);
              auto shiftConst =
                  ConstantOp::create(builder, loc, shiftImm, shiftAmt);
              rowContrib = S_LSHL_B32::create(builder, loc, sregType,
                                              ctx.createSCCType(), rowContrib,
                                              shiftConst)
                               .getDst();
            } else {
              auto strideImm = ctx.createImmType(ldsRowStride);
              auto strideConst =
                  ConstantOp::create(builder, loc, strideImm, ldsRowStride);
              rowContrib = S_MUL_I32::create(builder, loc, sregType, rowContrib,
                                             strideConst);
            }
            rowIsConst = false;
          } else {
            // Row index is a VGPR: use VALU multiply
            auto strideImm = ctx.createImmType(ldsRowStride);
            auto strideConst =
                ConstantOp::create(builder, loc, strideImm, ldsRowStride);
            rowContrib = convertToVgpr(rowContrib);
            rowContrib = V_MUL_LO_U32::create(
                builder, loc, ctx.createVRegType(), rowContrib, strideConst);
            rowIsConst = false;
          }
        } else if (rowIsConst) {
          rowConstBytes = dyn_cast<ImmType>(rowContrib.getType()).getValue();
        }

        // Check if column offset is also a constant (for folding)
        int64_t colConstBytes = 0;
        bool colIsConst = false;
        Value colDynamic;
        if (op->getNumOperands() > 4) {
          mlir::Value ldsColIndex = op->getOperand(4);
          auto ldsColMapped = ctx.getMapper().getMapped(ldsColIndex);
          if (ldsColMapped) {
            if (auto immType = dyn_cast<ImmType>(ldsColMapped->getType())) {
              colConstBytes = immType.getValue() * ldsElemBytes;
              colIsConst = true;
            } else {
              colDynamic = *ldsColMapped;
            }
          }
        }

        // Optimization: if both row and col are constants and we have a
        // dynamic base, fold them into a single constant add to save one
        // SALU instruction.
        if (rowIsConst && colIsConst) {
          int64_t staticContrib = rowConstBytes + colConstBytes;
          if (!m0Val) {
            // All static: just materialize the total offset
            auto totalImm = ctx.createImmType(staticContrib);
            m0Val = ConstantOp::create(builder, loc, totalImm, staticContrib);
          } else if (staticContrib != 0) {
            // Dynamic base + single static add
            auto totalImm = ctx.createImmType(staticContrib);
            auto totalConst =
                ConstantOp::create(builder, loc, totalImm, staticContrib);
            bool m0IsSgpr = !isVGPRType(m0Val.getType());
            if (m0IsSgpr) {
              auto sregType = ctx.createSRegType();
              m0Val = S_ADD_U32::create(builder, loc, sregType,
                                        ctx.createSCCType(), m0Val, totalConst)
                          .getDst();
            } else {
              m0Val = V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                        m0Val, totalConst);
            }
          }
          // else: staticContrib == 0, m0Val already has the dynamic base
        } else {
          // Non-constant row or col: add row contribution
          if (rowIsConst && rowConstBytes != 0) {
            auto rowImm = ctx.createImmType(rowConstBytes);
            rowContrib =
                ConstantOp::create(builder, loc, rowImm, rowConstBytes);
          }
          if (!rowIsConst || rowConstBytes != 0) {
            if (!m0Val) {
              m0Val = rowContrib;
            } else {
              bool m0IsSgpr = !isVGPRType(m0Val.getType());
              bool rowIsSgpr = !isVGPRType(rowContrib.getType());
              if (m0IsSgpr && rowIsSgpr) {
                auto sregType = ctx.createSRegType();
                m0Val =
                    S_ADD_U32::create(builder, loc, sregType,
                                      ctx.createSCCType(), m0Val, rowContrib)
                        .getDst();
              } else {
                m0Val = convertToVgpr(m0Val);
                rowContrib = convertToVgpr(rowContrib);
                m0Val = V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                          m0Val, rowContrib);
              }
            }
          }

          // Add column offset (non-constant or constant with non-const row)
          if (colIsConst && colConstBytes != 0) {
            auto colImm = ctx.createImmType(colConstBytes);
            auto colConst =
                ConstantOp::create(builder, loc, colImm, colConstBytes);
            bool m0IsSgpr = m0Val && !isVGPRType(m0Val.getType());
            if (m0IsSgpr) {
              auto sregType = ctx.createSRegType();
              m0Val = S_ADD_U32::create(builder, loc, sregType,
                                        ctx.createSCCType(), m0Val, colConst)
                          .getDst();
            } else if (m0Val) {
              m0Val = V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                        m0Val, colConst);
            }
          } else if (colDynamic) {
            bool colIsSgpr = !isVGPRType(colDynamic.getType()) &&
                             !isa<ImmType>(colDynamic.getType());
            bool m0IsSgpr = m0Val && !isVGPRType(m0Val.getType());
            if (colIsSgpr && m0IsSgpr) {
              // Both SGPR: stay in SALU domain
              Value colVal = colDynamic;
              if (ldsElemBytes > 1) {
                auto sregType = ctx.createSRegType();
                if (llvm::isPowerOf2_64(ldsElemBytes)) {
                  int64_t shiftAmt = llvm::Log2_64(ldsElemBytes);
                  auto shiftImm = ctx.createImmType(shiftAmt);
                  auto shiftConst =
                      ConstantOp::create(builder, loc, shiftImm, shiftAmt);
                  colVal = S_LSHL_B32::create(builder, loc, sregType,
                                              ctx.createSCCType(), colVal,
                                              shiftConst)
                               .getDst();
                } else {
                  auto scaleImm = ctx.createImmType(ldsElemBytes);
                  auto scaleConst =
                      ConstantOp::create(builder, loc, scaleImm, ldsElemBytes);
                  colVal = S_MUL_I32::create(builder, loc, sregType, colVal,
                                             scaleConst);
                }
              }
              auto sregType = ctx.createSRegType();
              m0Val = S_ADD_U32::create(builder, loc, sregType,
                                        ctx.createSCCType(), m0Val, colVal)
                          .getDst();
            } else {
              // At least one is VGPR: fall back to VALU
              m0Val = convertToVgpr(m0Val);
              Value colVgpr = convertToVgpr(colDynamic);
              if (ldsElemBytes > 1) {
                auto scaleImm = ctx.createImmType(ldsElemBytes);
                auto scaleConst =
                    ConstantOp::create(builder, loc, scaleImm, ldsElemBytes);
                colVgpr = V_MUL_LO_U32::create(
                    builder, loc, ctx.createVRegType(), colVgpr, scaleConst);
              }
              m0Val = V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                        m0Val, colVgpr);
            }
          }
        }

        // Add static base offset (if we have a constant base AND a dynamic
        // row/col making us land in this branch)
        if (hasLdsBaseOffset && !hasLdsDynamicBaseOffset &&
            ldsBaseOffsetConst != 0) {
          auto baseImm = ctx.createImmType(ldsBaseOffsetConst);
          auto baseConst =
              ConstantOp::create(builder, loc, baseImm, ldsBaseOffsetConst);
          bool m0IsSgpr = !isVGPRType(m0Val.getType());
          if (m0IsSgpr) {
            auto sregType = ctx.createSRegType();
            m0Val = S_ADD_U32::create(builder, loc, sregType,
                                      ctx.createSCCType(), m0Val, baseConst)
                        .getDst();
          } else {
            Value baseVgpr = V_MOV_B32::create(builder, loc,
                                               ctx.createVRegType(), baseConst);
            m0Val = V_ADD_U32::create(builder, loc, ctx.createVRegType(), m0Val,
                                      baseVgpr);
          }
        }
      }
    }

    if (m0Val) {
      Value m0Src = m0Val;
      // Convert VGPR to SGPR using v_readfirstlane_b32
      if (isVGPRType(m0Src.getType())) {
        auto sregType = ctx.createSRegType();
        m0Src = V_READFIRSTLANE_B32::create(builder, loc, sregType, m0Src);
      }
      S_MOV_B32_M0::create(builder, loc, m0Src);
    } else {
      auto zeroImm = ctx.createImmType(0);
      auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
      S_MOV_B32_M0::create(builder, loc, zeroConst);
    }
  }

  // Determine transfer size
  int64_t transferBytes = 4;
  if (auto transferTypeAttr = op->getAttrOfType<TypeAttr>("transferType")) {
    if (auto vecType = dyn_cast<VectorType>(transferTypeAttr.getValue())) {
      int64_t numElems = vecType.getNumElements();
      auto elemType = vecType.getElementType();
      int64_t elemSize = elemType.getIntOrFloatBitWidth() / 8;
      transferBytes = numElems * elemSize;
    }
  }

  // Emit buffer_load instruction
  auto soffImm = ctx.createImmType(0);
  auto soffConst = ConstantOp::create(builder, loc, soffImm, 0);
  if (transferBytes == 16) {
    BUFFER_LOAD_DWORDX4_LDS::create(builder, loc, voff, *srcMapped, soffConst);
  } else if (transferBytes == 4) {
    BUFFER_LOAD_DWORD_LDS::create(builder, loc, voff, *srcMapped, soffConst);
  } else {
    return op->emitError("unsupported transfer size for gather_to_lds: " +
                         std::to_string(transferBytes) + " bytes");
  }

  // NOTE: We do NOT emit s_waitcnt here. The Ticketing pass handles
  // synchronization by inserting s_waitcnt vmcnt(0) before every s_barrier.
  // Since gather-to-lds is always followed by a barrier before any ds_read,
  // this ensures correctness without redundant per-gather waits.

  return success();
}

LogicalResult handleRawBufferLoad(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  if (op->getNumOperands() < 1) {
    return op->emitError("raw_buffer_load requires at least a memref operand");
  }

  Value memref = op->getOperand(0);
  auto srcMapped = ctx.getMapper().getMapped(memref);
  if (!srcMapped) {
    return op->emitError("buffer descriptor not mapped");
  }

  Value voffset;
  if (op->getNumOperands() > 1) {
    auto voffMapped = ctx.getMapper().getMapped(op->getOperand(1));
    if (voffMapped) {
      voffset = *voffMapped;
    }
  }
  if (!voffset) {
    auto immZero = ctx.createImmType(0);
    voffset = ConstantOp::create(builder, loc, immZero, 0);
  }

  Type resultType = op->getResult(0).getType();
  int64_t numElements = 1;
  if (auto vecType = dyn_cast<VectorType>(resultType)) {
    numElements = vecType.getNumElements();
  }

  auto zeroImm = builder.getType<ImmType>(0);
  auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);

  Operation *loadInstr;
  if (numElements == 1) {
    auto vregType = ctx.createVRegType(1);
    loadInstr = BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{vregType},
                                          *srcMapped, voffset, zeroConst);
  } else if (numElements == 2) {
    auto vregType = ctx.createVRegType(2, 2);
    loadInstr = BUFFER_LOAD_DWORDX2::create(builder, loc, TypeRange{vregType},
                                            *srcMapped, voffset, zeroConst);
  } else if (numElements == 4) {
    auto vregType = ctx.createVRegType(4, 4);
    loadInstr = BUFFER_LOAD_DWORDX4::create(builder, loc, TypeRange{vregType},
                                            *srcMapped, voffset, zeroConst);
  } else {
    return op->emitError("unsupported buffer load width");
  }

  ctx.getMapper().mapValue(op->getResult(0), loadInstr->getResult(0));
  return success();
}

LogicalResult handleRawBufferStore(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  if (op->getNumOperands() < 2) {
    return op->emitError("raw_buffer_store requires value and memref operands");
  }

  Value storeValue = op->getOperand(0);
  Value memref = op->getOperand(1);

  auto valueMapped = ctx.getMapper().getMapped(storeValue);
  auto srd = ctx.getMapper().getMapped(memref);

  if (!valueMapped) {
    return op->emitError("store value not mapped");
  }
  if (!srd) {
    return op->emitError("buffer descriptor not mapped");
  }

  Value voffset;
  if (op->getNumOperands() > 2) {
    auto voffMapped = ctx.getMapper().getMapped(op->getOperand(2));
    if (voffMapped) {
      voffset = *voffMapped;
    }
  }
  if (!voffset) {
    auto immZero = ctx.createImmType(0);
    voffset = ConstantOp::create(builder, loc, immZero, 0);
  }

  Type valueType = storeValue.getType();
  int64_t numElements = 1;
  if (auto vecType = dyn_cast<VectorType>(valueType)) {
    numElements = vecType.getNumElements();
  }

  if (numElements == 1) {
    BUFFER_STORE_DWORD::create(builder, loc, *valueMapped, *srd, voffset);
  } else if (numElements == 2) {
    BUFFER_STORE_DWORDX2::create(builder, loc, *valueMapped, *srd, voffset);
  } else if (numElements == 4) {
    BUFFER_STORE_DWORDX4::create(builder, loc, *valueMapped, *srd, voffset);
  } else {
    return op->emitError("unsupported buffer store width");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ROCDL Dialect Handlers
//===----------------------------------------------------------------------===//

LogicalResult handleReadFirstLane(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto sregType = ctx.createSRegType();

  if (op->getNumOperands() < 1) {
    return op->emitError("readfirstlane requires an operand");
  }

  auto src = ctx.getMapper().getMapped(op->getOperand(0));
  if (!src) {
    return op->emitError("source operand not mapped");
  }

  auto result = V_READFIRSTLANE_B32::create(builder, loc, sregType, *src);
  ctx.getMapper().mapValue(op->getResult(0), result);

  return success();
}

LogicalResult handleMemRefAtomicRMW(Operation *op, TranslationContext &ctx) {
  auto atomicOp = cast<memref::AtomicRMWOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto valueMapped = ctx.getMapper().getMapped(atomicOp.getValue());
  if (!valueMapped) {
    return op->emitError("atomic_rmw value not mapped");
  }

  auto memrefType = atomicOp.getMemRefType();
  Type elementType = memrefType.getElementType();

  auto kind = atomicOp.getKind();
  if (kind != arith::AtomicRMWKind::addf &&
      kind != arith::AtomicRMWKind::addi) {
    return op->emitError(
        "unsupported atomic_rmw kind; only addf/addi supported");
  }

  auto [voffset, instOffset] = computeVOffsetFromIndices(
      memrefType, atomicOp.getIndices(), ctx, loc, atomicOp.getMemref());
  Value srd = lookupSRD(atomicOp.getMemref(), ctx, loc);

  if (elementType.isBF16()) {
    auto vregType = ctx.createVRegType();

    if (isAGPRType(valueMapped->getType())) {
      valueMapped =
          V_ACCVGPR_READ_B32::create(builder, loc, vregType, *valueMapped);
    }
    Value bf16Value =
        V_CVT_BF16_F32::create(builder, loc, vregType, *valueMapped);
    valueMapped = bf16Value;

    int64_t positionFromInstOffset = instOffset & 2;
    int64_t alignedInstOffset = instOffset & ~3;

    auto immTwo = ConstantOp::create(builder, loc, ctx.createImmType(2), 2);
    Value position = V_AND_B32::create(builder, loc, vregType, voffset, immTwo);

    auto immNeg4 = ConstantOp::create(builder, loc, ctx.createImmType(-4), -4);
    Value alignedVoffset =
        V_AND_B32::create(builder, loc, vregType, voffset, immNeg4);

    if (positionFromInstOffset != 0) {
      auto immPos = ConstantOp::create(
          builder, loc, ctx.createImmType(positionFromInstOffset),
          positionFromInstOffset);
      position = V_ADD_U32::create(builder, loc, vregType, position, immPos);
      position = V_AND_B32::create(builder, loc, vregType, position, immTwo);
    }

    auto immSixteen =
        ConstantOp::create(builder, loc, ctx.createImmType(16), 16);
    Value shifted =
        V_LSHLREV_B32::create(builder, loc, vregType, immSixteen, *valueMapped);

    V_CMP_EQ_U32::create(builder, loc, position, immTwo);

    auto sregType = ctx.createSRegType(2, 2);
    auto vcc = PrecoloredSRegOp::create(builder, loc, sregType, 106, 2);

    Value packed = V_CNDMASK_B32::create(builder, loc, vregType, *valueMapped,
                                         shifted, vcc);

    BUFFER_ATOMIC_PK_ADD_BF16::create(builder, loc, packed, srd, alignedVoffset,
                                      alignedInstOffset);
  } else if (elementType.isF32()) {
    BUFFER_ATOMIC_ADD_F32::create(builder, loc, *valueMapped, srd, voffset,
                                  instOffset);
  } else {
    return op->emitError("unsupported element type for atomic_rmw: ")
           << elementType;
  }

  // NOTE: memref.atomic_rmw semantically returns the old value at the memory
  // location, but our buffer_atomic instructions are configured without the
  // GLC (return data) bit, so no value is returned from hardware. We map the
  // result to the input value as a placeholder. This is safe as long as no
  // consumer actually uses the old-value result. If a consumer does, the
  // value will be the *new* value written, not the old one.
  ctx.getMapper().mapValue(atomicOp.getResult(), *valueMapped);
  return success();
}

LogicalResult handleSWaitcnt(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  int64_t vmcnt = -1, lgkmcnt = -1;

  if (auto bitfieldAttr = op->getAttrOfType<IntegerAttr>("bitfield")) {
    int64_t bitfield = bitfieldAttr.getInt();
    int64_t vmcnt_lo = bitfield & 0xF;
    int64_t vmcnt_hi = (bitfield >> 14) & 0x3;
    int64_t vmcnt_full = vmcnt_lo | (vmcnt_hi << 4);
    int64_t lgkmcnt_full = (bitfield >> 8) & 0xF;

    if (vmcnt_full < 63) {
      vmcnt = vmcnt_full;
    }
    if (lgkmcnt_full < 15) {
      lgkmcnt = lgkmcnt_full;
    }
  }

  if (vmcnt < 0 && lgkmcnt < 0) {
    return success();
  }

  if (vmcnt >= 0 && lgkmcnt >= 0) {
    auto vmcntAttr = builder.getI32IntegerAttr(vmcnt);
    auto lgkmcntAttr = builder.getI32IntegerAttr(lgkmcnt);
    S_WAITCNT::create(builder, loc, vmcntAttr, lgkmcntAttr, IntegerAttr());
  } else if (vmcnt >= 0) {
    S_WAITCNT_VMCNT::create(builder, loc, vmcnt);
  } else {
    S_WAITCNT_LGKMCNT::create(builder, loc, lgkmcnt);
  }
  return success();
}

} // namespace waveasm
