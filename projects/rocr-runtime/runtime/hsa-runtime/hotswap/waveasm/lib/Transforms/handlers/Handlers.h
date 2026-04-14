// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Operation Handlers Header
//===----------------------------------------------------------------------===//
//
// This file declares all operation handlers for the MLIR to WaveASM
// translation. Handlers are organized by source dialect:
//   - GPUHandlers.cpp: gpu.thread_id, gpu.block_id, gpu.barrier, etc.
//   - ArithHandlers.cpp: arith.addi, arith.muli, arith.constant, etc.
//   - VectorHandlers.cpp: vector.load, vector.store, vector.fma, etc.
//   - SCFHandlers.cpp: scf.for, scf.if, scf.yield
//   - AMDGPUHandlers.cpp: amdgpu.mfma, amdgpu.lds_barrier, etc.
//   - MemRefHandlers.cpp: memref.load, memref.store, memref.alloc, etc.
//   - MathHandlers.cpp: math.fma, etc.
//   - AffineHandlers.cpp: affine.apply
//
//===----------------------------------------------------------------------===//

#ifndef WAVEASM_TRANSFORMS_HANDLERS_HANDLERS_H
#define WAVEASM_TRANSFORMS_HANDLERS_HANDLERS_H

#include "waveasm/Transforms/TranslateFromMLIR.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstring>
#include <optional>
#include <utility>

namespace waveasm {

//===----------------------------------------------------------------------===//
// GPU Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleGPUThreadId(mlir::Operation *op,
                                      TranslationContext &ctx);
mlir::LogicalResult handleGPUBlockId(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleGPUBarrier(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleGPUBlockDim(mlir::Operation *op,
                                      TranslationContext &ctx);
mlir::LogicalResult handleGPUGridDim(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleGPULaneId(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleGPUSubgroupBroadcast(mlir::Operation *op,
                                               TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// Arith Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleArithConstant(mlir::Operation *op,
                                        TranslationContext &ctx);
mlir::LogicalResult handleArithAddI(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithSubI(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithMulI(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithDivUI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithRemUI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithIndexCast(mlir::Operation *op,
                                         TranslationContext &ctx);
mlir::LogicalResult handleArithAndI(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithOrI(mlir::Operation *op,
                                   TranslationContext &ctx);
mlir::LogicalResult handleArithXorI(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithShLI(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithShRUI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithShRSI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithExtUI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithExtSI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithTruncI(mlir::Operation *op,
                                      TranslationContext &ctx);
mlir::LogicalResult handleArithMinSI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithMaxSI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithMinUI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithMaxUI(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleArithCmpI(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithSelect(mlir::Operation *op,
                                      TranslationContext &ctx);
mlir::LogicalResult handleArithAddF(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithSubF(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithMulF(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithDivF(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithNegF(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithCmpF(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleArithTruncF(mlir::Operation *op,
                                      TranslationContext &ctx);
mlir::LogicalResult handleArithExtF(mlir::Operation *op,
                                    TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// Math Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleMathFma(mlir::Operation *op, TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// Affine Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleAffineApply(mlir::Operation *op,
                                      TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// Vector Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleVectorExtractStridedSlice(mlir::Operation *op,
                                                    TranslationContext &ctx);
mlir::LogicalResult handleVectorBroadcast(mlir::Operation *op,
                                          TranslationContext &ctx);
mlir::LogicalResult handleVectorExtract(mlir::Operation *op,
                                        TranslationContext &ctx);
mlir::LogicalResult handleVectorInsert(mlir::Operation *op,
                                       TranslationContext &ctx);
mlir::LogicalResult handleVectorShapeCast(mlir::Operation *op,
                                          TranslationContext &ctx);
mlir::LogicalResult handleVectorBitCast(mlir::Operation *op,
                                        TranslationContext &ctx);
mlir::LogicalResult handleVectorFma(mlir::Operation *op,
                                    TranslationContext &ctx);
mlir::LogicalResult handleVectorReduction(mlir::Operation *op,
                                          TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// AMDGPU Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleAMDGPULdsBarrier(mlir::Operation *op,
                                           TranslationContext &ctx);
mlir::LogicalResult handleAMDGPUMfma(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleAMDGPUScaledMfma(mlir::Operation *op,
                                           TranslationContext &ctx);
mlir::LogicalResult handleFatRawBufferCast(mlir::Operation *op,
                                           TranslationContext &ctx);
mlir::LogicalResult handleGatherToLds(mlir::Operation *op,
                                      TranslationContext &ctx);
mlir::LogicalResult handleRawBufferLoad(mlir::Operation *op,
                                        TranslationContext &ctx);
mlir::LogicalResult handleRawBufferStore(mlir::Operation *op,
                                         TranslationContext &ctx);
mlir::LogicalResult handleMemRefAtomicRMW(mlir::Operation *op,
                                          TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// ROCDL Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleReadFirstLane(mlir::Operation *op,
                                        TranslationContext &ctx);
mlir::LogicalResult handleSWaitcnt(mlir::Operation *op,
                                   TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// IREE/Stream Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleBindingSubspan(mlir::Operation *op,
                                         TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// MemRef Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleMemRefAlloc(mlir::Operation *op,
                                      TranslationContext &ctx);
mlir::LogicalResult handleMemRefView(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleMemRefReinterpretCast(mlir::Operation *op,
                                                TranslationContext &ctx);
mlir::LogicalResult handleMemRefSubView(mlir::Operation *op,
                                        TranslationContext &ctx);
mlir::LogicalResult handleMemRefLoad(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleMemRefStore(mlir::Operation *op,
                                      TranslationContext &ctx);
mlir::LogicalResult handleMemRefCast(mlir::Operation *op,
                                     TranslationContext &ctx);
mlir::LogicalResult handleMemRefExtractStridedMetadata(mlir::Operation *op,
                                                       TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// SCF Dialect Handlers
//===----------------------------------------------------------------------===//

mlir::LogicalResult handleSCFFor(mlir::Operation *op, TranslationContext &ctx);
mlir::LogicalResult handleSCFIf(mlir::Operation *op, TranslationContext &ctx);
mlir::LogicalResult handleSCFYield(mlir::Operation *op,
                                   TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// Helper Functions (shared across handlers)
//===----------------------------------------------------------------------===//

/// Check if a memref has LDS address space
bool isLDSMemRef(mlir::MemRefType memrefType);

/// Get element size in bytes
int64_t getElementBytes(mlir::Type type);

/// Compute buffer size from memref type
int64_t computeBufferSizeFromMemRef(mlir::MemRefType memrefType);

/// Check if value is power of 2
bool isPowerOf2(int64_t val);

/// Barrett-reduction-based unsigned integer floor division.
/// Exact for all uint32 values (unlike float-rcp which fails for x >= 2^23).
mlir::Value emitUnsignedFloordiv(mlir::Value x, mlir::Value d,
                                 mlir::OpBuilder &builder, mlir::Location loc,
                                 TranslationContext &ctx);

/// Magic-number unsigned floor division by a known constant divisor (>= 2).
/// Uses llvm::UnsignedDivisionByConstantInfo: 2-5 VALU instructions.
mlir::Value emitConstantUnsignedFloordiv(mlir::Value x, int64_t divisor,
                                         mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         TranslationContext &ctx);

/// Get log2 of power of 2
int64_t log2(int64_t val);

/// Create an immediate constant Value (combines createImmType + ConstantOp).
inline mlir::Value createImmConst(int64_t val, mlir::OpBuilder &builder,
                                  mlir::Location loc, TranslationContext &ctx) {
  auto immType = ctx.createImmType(val);
  return ConstantOp::create(builder, loc, immType, val);
}

/// Extract constant value from a Value if it is a constant
std::optional<int64_t> getArithConstantValue(mlir::Value val);

//===----------------------------------------------------------------------===//
// Type-safe bit casting (replaces C-style memcpy for type punning)
//===----------------------------------------------------------------------===//

/// Bit-cast between types of the same size (type-safe alternative to memcpy)
template <typename To, typename From> inline To bitCast(From value) {
  static_assert(sizeof(To) == sizeof(From),
                "bitCast requires types of equal size");
  To result;
  std::memcpy(&result, &value, sizeof(To));
  return result;
}

//===----------------------------------------------------------------------===//
// Error Handling Helpers
//===----------------------------------------------------------------------===//

/// Helper to check if a value is mapped and emit error if not
inline mlir::LogicalResult requireMapped(std::optional<mlir::Value> mapped,
                                         mlir::Operation *op,
                                         llvm::StringRef name) {
  if (!mapped) {
    return op->emitError() << name << " not mapped";
  }
  return mlir::success();
}

/// Helper to get and validate binary operands
template <typename OpTy>
inline std::pair<std::optional<mlir::Value>, std::optional<mlir::Value>>
getBinaryOperands(OpTy op, TranslationContext &ctx) {
  auto lhs = ctx.getMapper().getMapped(op.getLhs());
  auto rhs = ctx.getMapper().getMapped(op.getRhs());
  return {lhs, rhs};
}

/// Helper to validate both binary operands are mapped
template <typename OpTy>
inline mlir::LogicalResult
validateBinaryOperands(OpTy op, TranslationContext &ctx,
                       std::optional<mlir::Value> &lhs,
                       std::optional<mlir::Value> &rhs) {
  std::tie(lhs, rhs) = getBinaryOperands(op, ctx);
  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Binary Operation Handler Template
//===----------------------------------------------------------------------===//

/// Template for handling binary VALU operations (arith.addi, arith.muli, etc.)
/// Reduces code duplication across ~25 similar handlers.
///
/// Usage:
///   return handleBinaryVALU<arith::AddIOp, V_ADD_U32>(op, ctx);
///
template <typename OpTy, typename InstrTy>
mlir::LogicalResult handleBinaryVALU(mlir::Operation *op,
                                     TranslationContext &ctx) {
  auto typedOp = mlir::cast<OpTy>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  std::optional<mlir::Value> lhs, rhs;
  if (mlir::failed(validateBinaryOperands(typedOp, ctx, lhs, rhs))) {
    return mlir::failure();
  }

  auto result = InstrTy::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(typedOp.getResult(), result);
  return mlir::success();
}

/// Template for binary operations with reversed operand order (shift
/// instructions) v_lshlrev_b32, v_lshrrev_b32, v_ashrrev_i32 have:
/// shift_amount, value
template <typename OpTy, typename InstrTy>
mlir::LogicalResult handleBinaryVALURev(mlir::Operation *op,
                                        TranslationContext &ctx) {
  auto typedOp = mlir::cast<OpTy>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  std::optional<mlir::Value> lhs, rhs;
  if (mlir::failed(validateBinaryOperands(typedOp, ctx, lhs, rhs))) {
    return mlir::failure();
  }

  // Reversed: rhs (shift amount) comes first
  auto result = InstrTy::create(builder, loc, vregType, *rhs, *lhs);
  ctx.getMapper().mapValue(typedOp.getResult(), result);
  return mlir::success();
}

/// Template for ternary VALU operations (FMA: a*b+c)
template <typename OpTy, typename InstrTy>
mlir::LogicalResult handleTernaryVALU(mlir::Operation *op,
                                      TranslationContext &ctx) {
  auto typedOp = mlir::cast<OpTy>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto a = ctx.getMapper().getMapped(typedOp.getA());
  auto b = ctx.getMapper().getMapped(typedOp.getB());
  auto c = ctx.getMapper().getMapped(typedOp.getC());

  if (!a || !b || !c) {
    return op->emitError("operands not mapped");
  }

  auto result = InstrTy::create(builder, loc, vregType, *a, *b, *c);
  ctx.getMapper().mapValue(typedOp.getResult(), result);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MemRef Address Computation Helper
//===----------------------------------------------------------------------===//

/// Helper class for computing byte offsets from memref indices and strides.
/// Factors out common logic from vector.load/store handlers.
class MemRefAddressComputer {
public:
  /// Compute byte offset from indices for a given memref type
  static mlir::Value computeByteOffset(mlir::ArrayRef<mlir::Value> indices,
                                       mlir::MemRefType memrefType,
                                       TranslationContext &ctx);

  /// Compute constant byte offset (for statically known indices)
  static std::optional<int64_t>
  computeConstantOffset(mlir::ArrayRef<mlir::Value> indices,
                        mlir::MemRefType memrefType, TranslationContext &ctx);

  /// Get the stride for a given dimension
  static int64_t getStride(mlir::MemRefType memrefType, unsigned dim);
};

//===----------------------------------------------------------------------===//
// Vector Splitting Helper
//===----------------------------------------------------------------------===//

/// Helper class for splitting large vector operations into smaller chunks.
/// AMDGPU buffer_load/store instructions have size limits (typically 16 bytes).
class VectorSplitter {
public:
  /// Maximum bytes per single buffer_load/store instruction
  static constexpr int64_t kMaxBytesPerOp = 16;

  /// Get split ranges for a vector of given total bytes
  /// Returns pairs of (offset, size) for each split
  static llvm::SmallVector<std::pair<int64_t, int64_t>>
  getSplitRanges(int64_t totalBytes);

  /// Check if a vector needs splitting
  static bool needsSplit(int64_t totalBytes) {
    return totalBytes > kMaxBytesPerOp;
  }
};

} // namespace waveasm

#endif // WAVEASM_TRANSFORMS_HANDLERS_HANDLERS_H
