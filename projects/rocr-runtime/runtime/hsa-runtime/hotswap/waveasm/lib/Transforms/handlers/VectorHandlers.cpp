// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Vector Dialect Handlers
//===----------------------------------------------------------------------===//
//
// This file implements handlers for Vector dialect operations:
//   - vector.extract_strided_slice
//   - vector.broadcast
//   - vector.extract
//   - vector.insert
//   - vector.shape_cast
//   - vector.fma
//   - vector.reduction
//
// Note: Complex operations (vector.load, vector.store, vector.transfer_read,
// vector.transfer_write) remain in TranslateFromMLIR.cpp due to their
// complexity involving SRD setup and multiple instruction variants.
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-vector-handlers"

using namespace mlir;

namespace waveasm {

LogicalResult handleVectorBroadcast(Operation *op, TranslationContext &ctx) {
  auto broadcastOp = cast<vector::BroadcastOp>(op);

  // For GPU, broadcast is typically a no-op (value is already lane-uniform
  // or will be handled by register allocation)
  auto src = ctx.getMapper().getMapped(broadcastOp.getSource());
  if (src) {
    ctx.getMapper().mapValue(broadcastOp.getResult(), *src);
  }
  return success();
}

LogicalResult handleVectorExtract(Operation *op, TranslationContext &ctx) {
  auto extractOp = cast<vector::ExtractOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto src = ctx.getMapper().getMapped(extractOp.getSource());
  if (!src) {
    return op->emitError("source value not mapped");
  }

  // Get the extraction index (position in the source vector)
  auto staticPos = extractOp.getStaticPosition();
  int64_t index = 0;
  if (!staticPos.empty()) {
    index = staticPos[0];
  }

  // Get the source register type to find the base physical register
  Type srcType = src->getType();
  int64_t baseIdx = 0;

  if (auto pvreg = dyn_cast<PVRegType>(srcType)) {
    // Physical VGPR - extract element at offset
    baseIdx = pvreg.getIndex() + index;
    auto elemType = PVRegType::get(builder.getContext(), baseIdx, 1);
    auto elemReg = PrecoloredVRegOp::create(builder, loc, elemType, baseIdx, 1);
    ctx.getMapper().mapValue(extractOp.getResult(), elemReg);
  } else {
    // Virtual VGPR or other type - use waveasm.extract op
    // This will be lowered to proper register offset during register allocation
    Type elemType;
    if (isAGPRType(srcType)) {
      elemType = ctx.createARegType(1, 1);
    } else {
      elemType = ctx.createVRegType(1, 1);
    }
    auto extractWaveOp = ExtractOp::create(builder, loc, elemType, *src,
                                           builder.getI64IntegerAttr(index));
    ctx.getMapper().mapValue(extractOp.getResult(), extractWaveOp.getResult());
  }
  return success();
}

LogicalResult handleVectorInsert(Operation *op, TranslationContext &ctx) {
  auto insertOp = cast<vector::InsertOp>(op);

  // Pass through the destination (modification happens via register offset)
  auto dest = ctx.getMapper().getMapped(insertOp.getDest());
  if (dest) {
    ctx.getMapper().mapValue(insertOp.getResult(), *dest);
  }
  return success();
}

LogicalResult handleVectorShapeCast(Operation *op, TranslationContext &ctx) {
  auto castOp = cast<vector::ShapeCastOp>(op);

  // Shape cast is a no-op at the register level
  auto src = ctx.getMapper().getMapped(castOp.getSource());
  if (src) {
    ctx.getMapper().mapValue(castOp.getResult(), *src);
  }
  return success();
}

LogicalResult handleVectorBitCast(Operation *op, TranslationContext &ctx) {
  auto castOp = cast<vector::BitCastOp>(op);

  // Bit cast is a no-op at the register level (reinterpret cast)
  // The data stays in the same registers, just interpreted differently
  auto src = ctx.getMapper().getMapped(castOp.getSource());
  if (src) {
    ctx.getMapper().mapValue(castOp.getResult(), *src);
  }
  return success();
}

LogicalResult handleVectorFma(Operation *op, TranslationContext &ctx) {
  auto fmaOp = cast<vector::FMAOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto lhs = ctx.getMapper().getMapped(fmaOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(fmaOp.getRhs());
  auto acc = ctx.getMapper().getMapped(fmaOp.getAcc());

  if (!lhs || !rhs || !acc) {
    return op->emitError("FMA operands not mapped");
  }

  auto resultType = fmaOp.getResult().getType();
  Type elemType;
  int64_t numElements = 1;
  if (auto vecType = dyn_cast<VectorType>(resultType)) {
    elemType = vecType.getElementType();
    numElements = vecType.getNumElements();
  } else {
    elemType = resultType;
  }

  // Create result register
  auto vregType = ctx.createVRegType(numElements, 1);

  Value result;
  if (elemType.isF32()) {
    // v_fma_f32 dst, src0, src1, src2 : dst = src0 * src1 + src2
    result = V_FMA_F32::create(builder, loc, vregType, *lhs, *rhs, *acc);
  } else if (elemType.isF16()) {
    // v_fma_f16 for f16 types
    result = V_FMA_F16::create(builder, loc, vregType, *lhs, *rhs, *acc);
  } else {
    // Fall back to mul + add for other types
    auto mulResult = V_MUL_F32::create(builder, loc, vregType, *lhs, *rhs);
    result = V_ADD_F32::create(builder, loc, vregType, mulResult, *acc);
  }

  ctx.getMapper().mapValue(fmaOp.getResult(), result);
  return success();
}

LogicalResult handleVectorReduction(Operation *op, TranslationContext &ctx) {
  // vector.reduction has operands: vector to reduce, optional accumulator
  if (op->getNumOperands() < 1) {
    return op->emitError("reduction requires vector operand");
  }

  Value vector = op->getOperand(0);
  auto vectorMapped = ctx.getMapper().getMapped(vector);
  if (!vectorMapped) {
    return op->emitError("vector operand not mapped");
  }

  // For now, emit a comment - full reduction requires wave-level operations
  // like DPP or permute instructions
  ctx.emitComment("vector.reduction - wave-level reduction");

  // Simple fallback: just map the first element
  ctx.getMapper().mapValue(op->getResult(0), *vectorMapped);
  return success();
}

LogicalResult handleVectorExtractStridedSlice(Operation *op,
                                              TranslationContext &ctx) {
  auto extractOp = cast<vector::ExtractStridedSliceOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto src = ctx.getMapper().getMapped(extractOp.getSource());
  if (!src) {
    return op->emitError("source value not mapped");
  }

  auto offsets = extractOp.getOffsets();
  int64_t offset = 0;
  if (!offsets.empty()) {
    offset = cast<IntegerAttr>(offsets[0]).getInt();
  }

  auto sizes = extractOp.getSizes();
  int64_t size = 1;
  if (!sizes.empty()) {
    size = cast<IntegerAttr>(sizes[0]).getInt();
  }

  // Check for sub-register packed extraction: when all elements fit in a
  // single 32-bit VGPR (e.g. vector<4xi8>), extracting at offset>0 requires
  // a bitfield operation rather than a register-index offset.
  // Only handle the single-register case; multi-register packed vectors
  // (e.g. vector<8xi8> = 2 VGPRs) would need per-register selection first.
  auto sourceVecType = dyn_cast<VectorType>(extractOp.getSource().getType());
  if (sourceVecType) {
    int64_t elemBitWidth =
        sourceVecType.getElementType().getIntOrFloatBitWidth();
    int64_t numElems = sourceVecType.getNumElements();
    int64_t totalBits = numElems * elemBitWidth;
    int64_t srcRegWidth = (totalBits + 31) / 32;

    if (srcRegWidth == 1 && srcRegWidth < numElems) {
      int64_t bitOffset = offset * elemBitWidth;
      int64_t extractWidth = size * elemBitWidth;
      auto vregType = ctx.createVRegType(1, 1);
      auto bitOffsetImm = builder.getType<ImmType>(bitOffset);
      auto bitOffsetConst =
          ConstantOp::create(builder, loc, bitOffsetImm, bitOffset);
      auto widthImm = builder.getType<ImmType>(extractWidth);
      auto widthConst =
          ConstantOp::create(builder, loc, widthImm, extractWidth);
      auto bfe = V_BFE_U32::create(builder, loc, vregType, *src, bitOffsetConst,
                                   widthConst);
      ctx.getMapper().mapValue(extractOp.getResult(), bfe);
      return success();
    }
  }

  Type srcType = src->getType();

  if (auto pvreg = dyn_cast<PVRegType>(srcType)) {
    int64_t baseIdx = pvreg.getIndex() + offset;
    auto elemType = PVRegType::get(builder.getContext(), baseIdx, size);
    auto elemReg =
        PrecoloredVRegOp::create(builder, loc, elemType, baseIdx, size);
    ctx.getMapper().mapValue(extractOp.getResult(), elemReg);
  } else if (auto pareg = dyn_cast<PARegType>(srcType)) {
    int64_t baseIdx = pareg.getIndex() + offset;
    auto elemType = PARegType::get(builder.getContext(), baseIdx, size);
    auto elemReg =
        PrecoloredARegOp::create(builder, loc, elemType, baseIdx, size);
    ctx.getMapper().mapValue(extractOp.getResult(), elemReg);
  } else if (isAGPRType(srcType)) {
    auto elemType = ctx.createARegType(size, 1);
    auto extractWaveOp = ExtractOp::create(builder, loc, elemType, *src,
                                           builder.getI64IntegerAttr(offset));
    ctx.getMapper().mapValue(extractOp.getResult(), extractWaveOp.getResult());
  } else {
    auto elemType = ctx.createVRegType(size, 1);
    auto extractWaveOp = ExtractOp::create(builder, loc, elemType, *src,
                                           builder.getI64IntegerAttr(offset));
    ctx.getMapper().mapValue(extractOp.getResult(), extractWaveOp.getResult());
  }
  return success();
}

} // namespace waveasm
