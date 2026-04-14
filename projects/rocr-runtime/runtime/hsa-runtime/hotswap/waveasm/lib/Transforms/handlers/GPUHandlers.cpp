// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// GPU Dialect Handlers
//===----------------------------------------------------------------------===//
//
// This file implements handlers for GPU dialect operations:
//   - gpu.thread_id
//   - gpu.block_id
//   - gpu.barrier
//   - gpu.block_dim
//   - gpu.grid_dim
//   - gpu.lane_id
//   - gpu.subgroup_broadcast
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

namespace waveasm {

LogicalResult handleGPUThreadId(Operation *op, TranslationContext &ctx) {
  auto threadIdOp = cast<gpu::ThreadIdOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  Value result;

  // Get the dimension being requested
  gpu::Dimension dim = threadIdOp.getDimension();

  // Check if this is a multi-wave kernel
  // On gfx9/gfx950 hardware, all thread IDs are packed into v0:
  //   v0[9:0]   = workitem_id_x
  //   v0[19:10] = workitem_id_y
  //   v0[29:20] = workitem_id_z
  // We use v_bfe_u32 / v_and_b32 to extract the appropriate field.
  if (ctx.isMultiWaveKernel()) {
    ctx.setUsesWorkitemId(true);

    auto v0 = PrecoloredVRegOp::create(builder, loc, vregType, 0, 1);

    switch (dim) {
    case gpu::Dimension::x: {
      auto immMask = ctx.createImmType(0x3FF);
      auto maskConst = ConstantOp::create(builder, loc, immMask, 0x3FF);
      result = V_AND_B32::create(builder, loc, vregType, maskConst, v0);
      break;
    }
    case gpu::Dimension::y: {
      auto imm10 = ctx.createImmType(10);
      auto shift10 = ConstantOp::create(builder, loc, imm10, 10);
      result = V_BFE_U32::create(builder, loc, vregType, v0, shift10, shift10);
      break;
    }
    case gpu::Dimension::z: {
      auto imm20 = ctx.createImmType(20);
      auto imm10 = ctx.createImmType(10);
      auto shift20 = ConstantOp::create(builder, loc, imm20, 20);
      auto width10 = ConstantOp::create(builder, loc, imm10, 10);
      result = V_BFE_U32::create(builder, loc, vregType, v0, shift20, width10);
      break;
    }
    }
  } else {
    // Single-wave: compute lane ID using v_mbcnt
    // Note: for single-wave, tid_x == lane_id, and tid_y/tid_z are always 0

    if (dim == gpu::Dimension::x) {
      // Create constant -1 for the mask
      auto immType = ctx.createImmType(-1);
      auto maskConst = ConstantOp::create(builder, loc, immType, -1);

      // Create constant 0
      auto immZero = ctx.createImmType(0);
      auto zeroConst = ConstantOp::create(builder, loc, immZero, 0);

      // v_mbcnt_lo_u32_b32 - count bits in lower 32 lanes
      auto mbcntLo = V_MBCNT_LO_U32_B32::create(builder, loc, vregType,
                                                maskConst, zeroConst);

      // v_mbcnt_hi_u32_b32 - count bits in upper 32 lanes, add to low
      auto mbcntHi = V_MBCNT_HI_U32_B32::create(builder, loc, vregType,
                                                maskConst, mbcntLo);

      result = mbcntHi;
    } else {
      // For single-wave, tid_y and tid_z are always 0
      auto immZero = ctx.createImmType(0);
      result = ConstantOp::create(builder, loc, immZero, 0);
    }

    // Note: for single-wave, we don't set usesWorkitemId since we use v_mbcnt
    // The kernel descriptor should have amdhsa_system_vgpr_workitem_id = 0
  }

  // Map the result
  ctx.getMapper().mapValue(threadIdOp.getResult(), result);

  // Track upper bound for affine simplification
  // The upper_bound attribute tells us the range of this thread ID
  if (auto upperBoundAttr = threadIdOp.getUpperBoundAttr()) {
    int64_t upperBound = upperBoundAttr.getInt();
    // Track this thread ID value with its upper bound
    ctx.setThreadIdUpperBound(threadIdOp.getResult(), upperBound);

    // Also set bit range for OR vs ADD optimization
    // Thread ID is in range [0, upperBound-1], so we need bits
    // 0..log2(upperBound-1)
    ctx.setBitRange(result, BitRange::fromMaxValue(upperBound - 1));
  } else {
    // Default for 64-lane wavefront: [0, 63] needs 6 bits
    ctx.setBitRange(result, BitRange(0, 5));
  }

  return success();
}

LogicalResult handleGPUBlockId(Operation *op, TranslationContext &ctx) {
  auto blockIdOp = cast<gpu::BlockIdOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Block IDs are passed in system SGPRs via ABI
  // System SGPRs come after user SGPRs (kernarg ptr + preloaded args)
  gpu::Dimension dim = blockIdOp.getDimension();
  int dimIndex = 0;
  switch (dim) {
  case gpu::Dimension::x:
    dimIndex = 0;
    ctx.setUsesWorkgroupIdX(true);
    break;
  case gpu::Dimension::y:
    dimIndex = 1;
    ctx.setUsesWorkgroupIdY(true);
    break;
  case gpu::Dimension::z:
    dimIndex = 2;
    ctx.setUsesWorkgroupIdZ(true);
    break;
  }

  int64_t sgprIndex = ctx.getWorkgroupIdSgprIndex(dimIndex);

  auto sregType = ctx.createSRegType();
  auto blockId = PrecoloredSRegOp::create(builder, loc, sregType, sgprIndex, 1);

  ctx.getMapper().mapValue(blockIdOp.getResult(), blockId);
  return success();
}

LogicalResult handleGPUBarrier(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  S_BARRIER::create(builder, loc);
  return success();
}

LogicalResult handleGPUBlockDim(Operation *op, TranslationContext &ctx) {
  auto blockDimOp = cast<gpu::BlockDimOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Block dimensions are available from the dispatch packet or as kernel args
  // For now, use precolored SGPRs at standard ABI locations
  int64_t sgprIndex = 0;
  switch (blockDimOp.getDimension()) {
  case gpu::Dimension::x:
    sgprIndex = 6;
    break; // workgroup_size_x
  case gpu::Dimension::y:
    sgprIndex = 7;
    break; // workgroup_size_y
  case gpu::Dimension::z:
    sgprIndex = 8;
    break; // workgroup_size_z
  }

  auto sregType = ctx.createSRegType();
  auto blockDim =
      PrecoloredSRegOp::create(builder, loc, sregType, sgprIndex, 1);

  ctx.getMapper().mapValue(blockDimOp.getResult(), blockDim);
  return success();
}

LogicalResult handleGPUGridDim(Operation *op, TranslationContext &ctx) {
  auto gridDimOp = cast<gpu::GridDimOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Grid dimensions come from dispatch packet
  int64_t sgprIndex = 0;
  switch (gridDimOp.getDimension()) {
  case gpu::Dimension::x:
    sgprIndex = 9;
    break; // grid_size_x
  case gpu::Dimension::y:
    sgprIndex = 10;
    break; // grid_size_y
  case gpu::Dimension::z:
    sgprIndex = 11;
    break; // grid_size_z
  }

  auto sregType = ctx.createSRegType();
  auto gridDim = PrecoloredSRegOp::create(builder, loc, sregType, sgprIndex, 1);

  ctx.getMapper().mapValue(gridDimOp.getResult(), gridDim);
  return success();
}

LogicalResult handleGPULaneId(Operation *op, TranslationContext &ctx) {
  auto laneIdOp = cast<gpu::LaneIdOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  // Lane ID is the thread's position within the wavefront
  // Use v_mbcnt_lo + v_mbcnt_hi pattern
  auto immType = ctx.createImmType(-1);
  auto maskConst = ConstantOp::create(builder, loc, immType, -1);

  auto immZero = ctx.createImmType(0);
  auto zeroConst = ConstantOp::create(builder, loc, immZero, 0);

  auto mbcntLo =
      V_MBCNT_LO_U32_B32::create(builder, loc, vregType, maskConst, zeroConst);
  auto mbcntHi =
      V_MBCNT_HI_U32_B32::create(builder, loc, vregType, maskConst, mbcntLo);

  ctx.getMapper().mapValue(laneIdOp.getResult(), mbcntHi);
  return success();
}

LogicalResult handleGPUSubgroupBroadcast(Operation *op,
                                         TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // gpu.subgroup_broadcast can have either:
  // - 2 operands: value and lane_id
  // - 1 operand with lane_id as attribute (newer MLIR versions)
  if (op->getNumOperands() < 1) {
    return op->emitError(
        "subgroup_broadcast requires at least a value operand");
  }

  Value srcValue = op->getOperand(0);
  auto srcMapped = ctx.getMapper().getMapped(srcValue);
  if (!srcMapped) {
    return op->emitError("source value not mapped");
  }

  // If the source is an immediate (constant), broadcasting is a no-op
  // Just map the result to the same immediate value
  if (isa<ImmType>(srcMapped->getType())) {
    ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
    return success();
  }

  // Check for broadcast_type attribute first (newer MLIR with gpu.broadcast
  // enum) #gpu<broadcast first_active_lane> means use v_readfirstlane_b32
  bool usesFirstLane = false;
  if (auto broadcastTypeAttr = op->getAttr("broadcast_type")) {
    // The attribute is an enum like #gpu<broadcast first_active_lane>
    // For gpu.subgroup_broadcast with broadcast_type = first_active_lane,
    // we should use v_readfirstlane_b32
    // The attribute prints as "#gpu<broadcast first_active_lane>"
    // TODO: Use mlir::gpu::BroadcastTypeAttr when available to avoid string
    // matching
    std::string attrDump;
    llvm::raw_string_ostream os(attrDump);
    broadcastTypeAttr.print(os);
    if (attrDump.find("first_active_lane") != std::string::npos) {
      usesFirstLane = true;
    }
  }

  // Check for lane_id - either as operand or attribute
  std::optional<Value> laneMapped;
  int64_t laneIdValue = 0;
  bool hasLaneId = false;

  if (op->getNumOperands() >= 2) {
    // Lane ID is operand 1
    Value laneIdx = op->getOperand(1);
    laneMapped = ctx.getMapper().getMapped(laneIdx);
    hasLaneId = true;
  } else if (auto laneAttr = op->getAttrOfType<IntegerAttr>("lane_id")) {
    // Lane ID is an attribute (newer MLIR)
    laneIdValue = laneAttr.getInt();
    hasLaneId = true;
  } else if (auto subgroupIdAttr =
                 op->getAttrOfType<IntegerAttr>("subgroup_id")) {
    // Alternative attribute name
    laneIdValue = subgroupIdAttr.getInt();
    hasLaneId = true;
  }

  auto sregType = ctx.createSRegType();

  // If lane is constant 0, also use v_readfirstlane_b32
  if (!usesFirstLane) {
    if (laneMapped) {
      if (auto constOp = laneMapped->getDefiningOp<ConstantOp>()) {
        if (constOp.getValue() == 0) {
          usesFirstLane = true;
        }
      }
    } else if (hasLaneId && laneIdValue == 0) {
      usesFirstLane = true;
    }
  }

  Value result;
  if (usesFirstLane) {
    result = V_READFIRSTLANE_B32::create(builder, loc, sregType, *srcMapped);
  } else {
    // Use v_readlane_b32 with lane index
    if (!laneMapped) {
      auto immType = ctx.createImmType(laneIdValue);
      laneMapped = ConstantOp::create(builder, loc, immType, laneIdValue);
    }
    result =
        V_READLANE_B32::create(builder, loc, sregType, *srcMapped, *laneMapped);
  }

  ctx.getMapper().mapValue(op->getResult(0), result);
  return success();
}

} // namespace waveasm
