// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/TranslateFromMLIR.h"
#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Transforms/Utils.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <cctype>
#include <sstream>

#define DEBUG_TYPE "waveasm-translate"

using namespace mlir;

namespace waveasm {

// Named constants for SRD descriptor fields and limits.
// kDefaultMaxBufferSize lives in TranslationContext (header).
static constexpr int64_t kSRDStrideSwizzle = 0x20000;
static constexpr int64_t kMaxNumRecords32 = 0x7FFFFFFE;

//===----------------------------------------------------------------------===//
// TranslationContext Implementation
//===----------------------------------------------------------------------===//

TranslationContext::TranslationContext(OpBuilder &builder, ProgramOp program,
                                       TargetAttrInterface target,
                                       const TranslationOptions &options)
    : builder(builder), registry(builder.getContext()), program(program),
      target(target), options(options) {}

VRegType TranslationContext::createVRegType(int64_t size, int64_t alignment) {
  return VRegType::get(builder.getContext(), size, alignment);
}

SRegType TranslationContext::createSRegType(int64_t size, int64_t alignment) {
  return SRegType::get(builder.getContext(), size, alignment);
}

ARegType TranslationContext::createARegType(int64_t size, int64_t alignment) {
  return ARegType::get(builder.getContext(), size, alignment);
}

SCCType TranslationContext::createSCCType() {
  return SCCType::get(builder.getContext());
}

ImmType TranslationContext::createImmType(int64_t value) {
  return ImmType::get(builder.getContext(), value);
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//
// Note: Utility functions have been moved to handlers/HandlerUtils.cpp

CommentOp TranslationContext::emitComment(StringRef text) {
  return CommentOp::create(builder, builder.getUnknownLoc(), text);
}

RawOp TranslationContext::emitRaw(StringRef code) {
  return RawOp::create(builder, builder.getUnknownLoc(), code);
}

Block::iterator TranslationContext::getInsertionPoint() {
  return builder.getInsertionPoint();
}

std::optional<Value>
TranslationContext::getCachedExpr(StringRef opName, ValueRange operands,
                                  ArrayRef<int64_t> constants) const {
  // Build a cache key string
  std::string key;
  llvm::raw_string_ostream os(key);
  os << opName;
  for (Value v : operands) {
    os << "_" << v.getAsOpaquePointer();
  }
  for (int64_t c : constants) {
    os << "_c" << c;
  }

  auto it = exprCache.find(os.str());
  if (it != exprCache.end())
    return it->second;
  return std::nullopt;
}

void TranslationContext::cacheExpr(StringRef opName, ValueRange operands,
                                   ArrayRef<int64_t> constants, Value result) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << opName;
  for (Value v : operands) {
    os << "_" << v.getAsOpaquePointer();
  }
  for (int64_t c : constants) {
    os << "_c" << c;
  }
  exprCache[os.str()] = result;
}

void TranslationContext::queueSRDSetup(Value memref, int64_t argIndex,
                                       int64_t bufferSize) {
  // Check if this memref already has an SRD assigned (avoid duplicates)
  if (srdIndexMap.count(memref) > 0) {
    // Already processed - just update size if larger
    updateSRDBufferSize(memref, bufferSize);
    return;
  }

  // Allocate an SRD slot (4 consecutive SGPRs)
  int64_t srdBase = getNextSRDIndex();

  PendingSRD pending;
  pending.memref = memref;
  pending.argIndex = argIndex;
  pending.bufferSize = bufferSize;
  pending.srdBaseIndex = srdBase;

  pendingSRDs.push_back(pending);
  setSRDIndex(memref, srdBase);
}

void TranslationContext::queueScalarArgLoad(Value blockArg, int64_t argIndex) {
  PendingScalarArg pending;
  pending.blockArg = blockArg;
  pending.argIndex = argIndex;
  pendingScalarArgs.push_back(pending);
}

void TranslationContext::emitSRDPrologue() {
  if (srdPrologueEmitted || (pendingSRDs.empty() && pendingScalarArgs.empty()))
    return;

  srdPrologueEmitted = true;
  auto loc = builder.getUnknownLoc();

  bool isGFX95 = llvm::isa<GFX950TargetAttr>(target);
  bool usePreloading = isGFX95;

  // Recompute SRD base indices now that we know the total number of args.
  // SRDs must start after: user SGPRs + system SGPRs (workgroup IDs).
  int64_t userSgprCount = getUserSgprCount();
  int64_t systemSgprCount = 3; // workgroup_id_x, y, z
  int64_t srdStartIndex =
      (userSgprCount + systemSgprCount + 3) & ~3; // Align to 4

  // Update pending SRDs with correct indices and update srdIndexMap.
  for (size_t i = 0; i < pendingSRDs.size(); ++i) {
    int64_t newSrdBase = srdStartIndex + i * 4;
    pendingSRDs[i].srdBaseIndex = newSrdBase;
    srdIndexMap[pendingSRDs[i].memref] = newSrdBase;
  }

  // Compute overflow scalar count: args whose preload slot (2+argIndex*2)
  // exceeds the 16-SGPR hardware preload window.
  int64_t numOverflowScalars = 0;
  for (const auto &pending : pendingScalarArgs) {
    if (2 + pending.argIndex * 2 >= 16)
      ++numOverflowScalars;
  }

  // Overflow scalars are placed in SGPRs after all SRDs.
  int64_t afterLastSrd = srdStartIndex + pendingSRDs.size() * 4;
  int64_t overflowSgprBase = (afterLastSrd + 3) & ~3;

  // Emit comment for prologue.
  CommentOp::create(builder, loc, "SRD setup prologue");

  // s[0:1] = kernarg segment pointer.
  auto kernargPtrType = createSRegType(2, 2);
  PrecoloredSRegOp::create(builder, loc, kernargPtrType, /*index=*/0,
                           /*size=*/2);

  if (usePreloading) {
    // Reserve preload SGPR pairs within the 16-SGPR hardware window
    // so regalloc doesn't use them.
    llvm::DenseSet<int64_t> reservedPreloadBases;
    for (const auto &pending : pendingSRDs) {
      int64_t preloadBase = 2 + pending.argIndex * 2;
      if (preloadBase >= 16)
        continue;
      if (reservedPreloadBases.insert(preloadBase).second) {
        auto preloadType = createSRegType(2, 2);
        PrecoloredSRegOp::create(builder, loc, preloadType, preloadBase,
                                 /*size=*/2);
      }
    }
    for (const auto &pending : pendingScalarArgs) {
      int64_t preloadBase = 2 + pending.argIndex * 2;
      if (preloadBase >= 16)
        continue;
      if (reservedPreloadBases.insert(preloadBase).second) {
        auto preloadType = createSRegType(2, 2);
        PrecoloredSRegOp::create(builder, loc, preloadType, preloadBase,
                                 /*size=*/2);
      }
    }
  }

  if (usePreloading) {
    // GFX95* preloading path with partial preloading: args that fit in
    // s[2:15] are hardware-preloaded, overflow args are loaded via
    // explicit s_load after the aligned entry point.

    // Step 1: Load base addresses into preload locations.
    auto kernargSRegType = createSRegType(2, 2);
    auto kernargBase =
        PrecoloredSRegOp::create(builder, loc, kernargSRegType, 0, 2);

    for (const auto &pending : pendingSRDs) {
      int64_t loadBase = 2 + pending.argIndex * 2;
      if (loadBase >= 16)
        continue;
      int64_t kernargOffset = pending.argIndex * 8;

      auto loadDstType = createSRegType(2, loadBase);
      auto offsetImm = builder.getType<ImmType>(kernargOffset);
      auto offsetConst =
          ConstantOp::create(builder, loc, offsetImm, kernargOffset);
      S_LOAD_DWORDX2::create(builder, loc, TypeRange{loadDstType}, kernargBase,
                             offsetConst);
    }

    // Also load scalar args that fit in the preload window.
    for (const auto &pending : pendingScalarArgs) {
      int64_t loadBase = 2 + pending.argIndex * 2;
      if (loadBase >= 16)
        continue;
      int64_t kernargOffset = pending.argIndex * 8;

      auto loadDstType = createSRegType(2, loadBase);
      auto offsetImm = builder.getType<ImmType>(kernargOffset);
      auto offsetConst =
          ConstantOp::create(builder, loc, offsetImm, kernargOffset);
      S_LOAD_DWORDX2::create(builder, loc, TypeRange{loadDstType}, kernargBase,
                             offsetConst);
    }

    // Step 2: Branch to aligned entry point (gfx95* requirement).
    std::string kernelName = getKernelName(program).str();
    std::string mainLabel = ".L_" + kernelName + "_main";

    RawOp::create(builder, loc, "s_branch " + mainLabel);
    RawOp::create(builder, loc, ".p2align 8");
    RawOp::create(builder, loc, mainLabel + ":");

    // Step 3: Load overflow scalar args after the aligned entry point.
    for (int64_t i = 0; i < numOverflowScalars; ++i) {
      int64_t base = overflowSgprBase + i * 2;
      auto ovfType = createSRegType(2, 2);
      PrecoloredSRegOp::create(builder, loc, ovfType, base, /*size=*/2);
    }

    int64_t overflowIdx = 0;
    for (const auto &pending : pendingScalarArgs) {
      int64_t preloadBase = 2 + pending.argIndex * 2;
      if (preloadBase < 16)
        continue;
      int64_t loadBase = overflowSgprBase + overflowIdx * 2;
      overflowIdx++;
      int64_t kernargOffset = pending.argIndex * 8;

      auto loadDstType = createSRegType(2, loadBase);
      auto offsetImm = builder.getType<ImmType>(kernargOffset);
      auto offsetConst =
          ConstantOp::create(builder, loc, offsetImm, kernargOffset);
      S_LOAD_DWORDX2::create(builder, loc, TypeRange{loadDstType}, kernargBase,
                             offsetConst);
    }

    // Step 4: Wait for all scalar loads to complete.
    auto i32Type = builder.getI32Type();
    auto lgkmcntAttr = IntegerAttr::get(i32Type, 0);
    S_WAITCNT::create(builder, loc, /*vmcnt=*/IntegerAttr{}, lgkmcntAttr,
                      /*expcnt=*/IntegerAttr{});

    // Step 5: Copy from preload locations to SRD positions and fill
    // size/stride. Use typed ops with DCEProtectOp to prevent elimination.
    for (size_t i = 0; i < pendingSRDs.size(); ++i) {
      const auto &pending = pendingSRDs[i];
      int64_t srdBase = pending.srdBaseIndex;
      int64_t preloadBase = 2 + pending.argIndex * 2;

      auto srdType = createSRegType(4, 4);
      auto srdReg = PrecoloredSRegOp::create(builder, loc, srdType, srdBase, 4);

      auto preloadType = createSRegType(2, preloadBase);
      auto preloadSrc =
          PrecoloredSRegOp::create(builder, loc, preloadType, preloadBase, 2);
      auto dstB64Type = PSRegType::get(builder.getContext(), srdBase, 2);
      auto movB64 = S_MOV_B64::create(builder, loc, dstB64Type, preloadSrc);
      DCEProtectOp::create(builder, loc, movB64);

      int64_t clampedSize = std::min(pending.bufferSize, kMaxNumRecords32);
      auto sizeImm = createImmType(clampedSize);
      auto sizeVal = ConstantOp::create(builder, loc, sizeImm, clampedSize);
      auto dstW2Type = PSRegType::get(builder.getContext(), srdBase + 2, 1);
      auto movSize = S_MOV_B32::create(builder, loc, dstW2Type, sizeVal);
      DCEProtectOp::create(builder, loc, movSize);

      auto strideImm = createImmType(kSRDStrideSwizzle);
      auto strideVal =
          ConstantOp::create(builder, loc, strideImm, kSRDStrideSwizzle);
      auto dstW3Type = PSRegType::get(builder.getContext(), srdBase + 3, 1);
      auto movStride = S_MOV_B32::create(builder, loc, dstW3Type, strideVal);
      DCEProtectOp::create(builder, loc, movStride);

      mapper.mapValue(pending.memref, srdReg);
    }

    // Step 6: Move scalar args from SGPRs to VGPRs.
    // Preloaded scalars come from s[2+argIndex*2], overflow from
    // overflowSgprBase slots.
    int64_t ovfIdx = 0;
    for (const auto &pending : pendingScalarArgs) {
      int64_t preloadBase = 2 + pending.argIndex * 2;
      int64_t sgprSrc;
      if (preloadBase >= 16) {
        sgprSrc = overflowSgprBase + ovfIdx * 2;
        ovfIdx++;
      } else {
        sgprSrc = preloadBase;
      }

      auto vregType = createVRegType();
      auto vreg =
          PrecoloredVRegOp::create(builder, loc, vregType, pending.argIndex, 1);
      RawOp::create(builder, loc,
                    "v_mov_b32 v" + std::to_string(pending.argIndex) + ", s" +
                        std::to_string(sgprSrc));
      mapper.mapValue(pending.blockArg, vreg);
    }
  } else {
    // Direct-load path (non-GFX950 targets):
    // Load base addresses directly into SRD[0:1] positions, then fill SRD[2:3].

    PrecoloredSRegOp::create(builder, loc, createSRegType(2, 2), 0, 2);

    for (const auto &pending : pendingSRDs) {
      int64_t srdBase = pending.srdBaseIndex;
      int64_t kernargOffset = pending.argIndex * 8;

      auto pairType = PSRegType::get(builder.getContext(), srdBase, 2);
      PrecoloredSRegOp::create(builder, loc, pairType, srdBase, 2);
      RawOp::create(builder, loc,
                    "s_load_dwordx2 s[" + std::to_string(srdBase) + ":" +
                        std::to_string(srdBase + 1) + "], s[0:1], " +
                        std::to_string(kernargOffset));
    }

    for (size_t i = 0; i < pendingScalarArgs.size(); ++i) {
      const auto &pending = pendingScalarArgs[i];
      int64_t pairBase = overflowSgprBase + (int64_t)i * 2;
      int64_t kernargOffset = pending.argIndex * 8;

      auto pairType = PSRegType::get(builder.getContext(), pairBase, 2);
      PrecoloredSRegOp::create(builder, loc, pairType, pairBase, 2);
      RawOp::create(builder, loc,
                    "s_load_dwordx2 s[" + std::to_string(pairBase) + ":" +
                        std::to_string(pairBase + 1) + "], s[0:1], " +
                        std::to_string(kernargOffset));
    }

    auto i32Type = builder.getI32Type();
    auto lgkmcntAttr = IntegerAttr::get(i32Type, 0);
    S_WAITCNT::create(builder, loc, /*vmcnt=*/IntegerAttr{}, lgkmcntAttr,
                      /*expcnt=*/IntegerAttr{});

    for (size_t i = 0; i < pendingSRDs.size(); ++i) {
      const auto &pending = pendingSRDs[i];
      int64_t srdBase = pending.srdBaseIndex;

      auto srdType = createSRegType(4, 4);
      auto srdReg = PrecoloredSRegOp::create(builder, loc, srdType, srdBase, 4);

      int64_t clampedSize = std::min(pending.bufferSize, kMaxNumRecords32);
      auto sizeImm = createImmType(clampedSize);
      auto sizeVal = ConstantOp::create(builder, loc, sizeImm, clampedSize);
      auto dstW2Type = PSRegType::get(builder.getContext(), srdBase + 2, 1);
      auto movSize = S_MOV_B32::create(builder, loc, dstW2Type, sizeVal);
      DCEProtectOp::create(builder, loc, movSize);

      auto strideImm = createImmType(kSRDStrideSwizzle);
      auto strideVal =
          ConstantOp::create(builder, loc, strideImm, kSRDStrideSwizzle);
      auto dstW3Type = PSRegType::get(builder.getContext(), srdBase + 3, 1);
      auto movStride = S_MOV_B32::create(builder, loc, dstW3Type, strideVal);
      DCEProtectOp::create(builder, loc, movStride);

      mapper.mapValue(pending.memref, srdReg);
    }

    for (size_t i = 0; i < pendingScalarArgs.size(); ++i) {
      const auto &pending = pendingScalarArgs[i];
      int64_t srcSgpr = overflowSgprBase + (int64_t)i * 2;

      auto vregType = createVRegType();
      auto vreg =
          PrecoloredVRegOp::create(builder, loc, vregType, pending.argIndex, 1);
      RawOp::create(builder, loc,
                    "v_mov_b32 v" + std::to_string(pending.argIndex) + ", s" +
                        std::to_string(srcSgpr));
      mapper.mapValue(pending.blockArg, vreg);
    }
  }

  // For the direct-load path, record the pinned SGPR reservations as
  // attributes. Their PrecoloredSRegOps may be DCE'd by canonicalize
  // (they're Pure with no SSA users since the RawOp references registers
  // by string). The preloading path uses typed S_LOAD_DWORDX2 ops which
  // the allocator sees directly, so no extra attributes are needed.
  if (!usePreloading && !pendingScalarArgs.empty()) {
    int64_t scalarCount = (int64_t)pendingScalarArgs.size() * 2;
    int64_t scalarEnd = overflowSgprBase + scalarCount;
    program->setAttr("min_sgprs", builder.getI64IntegerAttr(scalarEnd));
    program->setAttr("scalar_sgpr_base",
                     builder.getI64IntegerAttr(overflowSgprBase));
    program->setAttr("scalar_sgpr_count",
                     builder.getI64IntegerAttr(scalarCount));
  }

  CommentOp::create(builder, loc, "End SRD setup");
}

std::optional<int64_t> TranslationContext::getSRDIndex(Value memref) const {
  auto it = srdIndexMap.find(memref);
  if (it != srdIndexMap.end())
    return it->second;
  return std::nullopt;
}

void TranslationContext::setSRDIndex(Value memref, int64_t srdBaseIndex) {
  srdIndexMap[memref] = srdBaseIndex;
}

std::optional<Value> TranslationContext::getSRDValue(Value memref) const {
  auto it = srdValueMap.find(memref);
  if (it != srdValueMap.end())
    return it->second;
  return std::nullopt;
}

void TranslationContext::setSRDValue(Value memref, Value srd) {
  srdValueMap[memref] = srd;
}

int64_t TranslationContext::getNextSRDIndex() {
  // Note: This is called during queueSRDSetup, but we don't know the
  // total number of kernel args yet. We return a placeholder that will
  // be recomputed in emitSRDPrologue.
  // For now, just use a simple incrementing scheme starting from 0.
  // The actual SGPR indices will be computed in emitSRDPrologue.
  int64_t idx = nextSRDIndex < 0 ? 0 : nextSRDIndex;
  nextSRDIndex = idx + 4; // Each SRD uses 4 consecutive SGPRs
  return idx;
}

void TranslationContext::updateSRDBufferSize(Value memref, int64_t bufferSize) {
  // Find the pending SRD for this memref and update its buffer size
  for (auto &pending : pendingSRDs) {
    if (pending.memref == memref) {
      // Only update if the new size is larger (more specific)
      if (bufferSize > pending.bufferSize) {
        pending.bufferSize = bufferSize;
      }
      return;
    }
  }
}

VOffsetResult computeVOffsetFromIndices(MemRefType memrefType,
                                        ValueRange indices,
                                        TranslationContext &ctx, Location loc,
                                        Value base) {
  auto &builder = ctx.getBuilder();
  Type elementType = memrefType.getElementType();
  int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

  Value voffset;
  int64_t instOffset = 0;

  SmallVector<int64_t, 4> strides;
  int64_t memrefOffset;
  if (succeeded(memrefType.getStridesAndOffset(strides, memrefOffset))) {
    for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
      auto idxMapped = ctx.getMapper().getMapped(indices[i]);
      if (!idxMapped)
        continue;
      Value idx = *idxMapped;

      // Handle dynamic strides: look up the runtime stride value from
      // the reinterpret_cast that created this memref.
      if (strides[i] == ShapedType::kDynamic && base) {
        auto dynStride = ctx.getDynamicStride(base, i);
        if (dynStride) {
          // dimOffset = idx * (runtimeStride * elementBytes)
          // First: compute runtimeStride * elementBytes if elementBytes > 1
          Value bytesPerElement;
          if (elementBytes > 1) {
            auto elemBytesImm = ConstantOp::create(
                builder, loc, ctx.createImmType(elementBytes), elementBytes);
            bytesPerElement = V_MUL_LO_U32::create(
                builder, loc, ctx.createVRegType(), *dynStride, elemBytesImm);
          } else {
            bytesPerElement = *dynStride;
          }
          // dimOffset = idx * bytesPerElement
          Value dimOffset = V_MUL_LO_U32::create(
              builder, loc, ctx.createVRegType(), idx, bytesPerElement);
          if (!voffset) {
            voffset = dimOffset;
          } else {
            voffset = V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                        voffset, dimOffset);
          }
          continue;
        }
      }

      int64_t strideBytes = strides[i] * elementBytes;
      if (strideBytes == 0)
        continue;
      int64_t constAddend = ctx.getConstOffset(indices[i]);
      if (constAddend != 0)
        instOffset += constAddend * strideBytes;

      Value dimOffset;
      if (strideBytes == 1) {
        dimOffset = idx;
      } else if ((strideBytes & (strideBytes - 1)) == 0) {
        int shift = 0;
        int64_t temp = strideBytes;
        while (temp > 1) {
          shift++;
          temp >>= 1;
        }
        auto shiftImm =
            ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
        dimOffset = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(),
                                          shiftImm, idx);
      } else {
        auto strideImm = ConstantOp::create(
            builder, loc, ctx.createImmType(strideBytes), strideBytes);
        dimOffset = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(),
                                         idx, strideImm);
      }
      if (!voffset) {
        voffset = dimOffset;
      } else {
        voffset = V_ADD_U32::create(builder, loc, ctx.createVRegType(), voffset,
                                    dimOffset);
      }
    }
  } else {
    if (!indices.empty()) {
      if (auto mapped = ctx.getMapper().getMapped(indices[0]))
        voffset = *mapped;
    }
  }
  if (!voffset) {
    auto immType = ctx.createImmType(0);
    voffset = ConstantOp::create(builder, loc, immType, 0);
  }
  return {voffset, instOffset};
}

Value lookupSRD(Value memref, TranslationContext &ctx, Location loc) {
  auto &builder = ctx.getBuilder();
  if (auto srdVal = ctx.getSRDValue(memref))
    return *srdVal;
  if (auto srdIdx = ctx.getSRDIndex(memref)) {
    auto sregType = ctx.createSRegType(4, 4);
    return PrecoloredSRegOp::create(builder, loc, sregType, *srdIdx, 4);
  }
  if (auto mapped = ctx.getMapper().getMapped(memref))
    return *mapped;
  auto sregType = ctx.createSRegType(4, 4);
  return PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);
}

Value emitSRDBaseAdjustment(const TranslationContext::PendingSRDBaseAdjust &adj,
                            Value memref, TranslationContext &ctx,
                            Location loc) {
  auto &builder = ctx.getBuilder();
  auto sregTy = ctx.createSRegType();

  // Extract base address words from the source SRD.
  // Prefer the SSA value captured from the prologue to maintain a def-use
  // chain, which prevents the register allocator from reusing the
  // kernel-arg SRD registers between the prologue and this adjustment.
  Value srcSrd;
  if (adj.srcSrdValue) {
    srcSrd = adj.srcSrdValue;
  } else {
    auto srcSrdType = ctx.createSRegType(4, 4);
    srcSrd =
        PrecoloredSRegOp::create(builder, loc, srcSrdType, adj.srcSrdBase, 4);
  }
  Value srcWord0 = ExtractOp::create(builder, loc, sregTy, srcSrd, 0);
  Value srcWord1 = ExtractOp::create(builder, loc, sregTy, srcSrd, 1);

  // Get element offset as SGPR.
  Value offsetVal = adj.elementOffset;
  if (isVGPRType(offsetVal.getType()))
    offsetVal = V_READFIRSTLANE_B32::create(builder, loc, sregTy, offsetVal);
  else
    offsetVal = S_MOV_B32::create(builder, loc, sregTy, offsetVal);

  // 64-bit byte offset = element_offset * elementBytes.
  auto elemSizeImm = ConstantOp::create(
      builder, loc, ctx.createImmType(adj.elementBytes), adj.elementBytes);
  auto byteOffHi =
      S_MUL_HI_I32::create(builder, loc, sregTy, offsetVal, elemSizeImm);
  auto byteOffLo =
      S_MUL_I32::create(builder, loc, sregTy, offsetVal, elemSizeImm);

  // Adjust SRD base: s_add_u32 (sets SCC) + s_addc_u32 (reads SCC).
  SCCType sccTy = ctx.createSCCType();
  auto addLo =
      S_ADD_U32::create(builder, loc, sregTy, sccTy, srcWord0, byteOffLo);
  Value adjWord0 = addLo.getDst();
  Value adjWord1 = S_ADDC_U32::create(builder, loc, sregTy, sccTy,
                                      addLo.getScc(), srcWord1, byteOffHi)
                       .getDst();

  // Build word 2 (num_records).
  Value word2;
  if (adj.numRecordsOverride) {
    Value nrVal = adj.numRecordsOverride;
    if (isVGPRType(nrVal.getType()))
      word2 = V_READFIRSTLANE_B32::create(builder, loc, sregTy, nrVal);
    else
      word2 = S_MOV_B32::create(builder, loc, sregTy, nrVal);
  } else {
    int64_t bufferSize = ctx.getBufferSizeForSRD(adj.srcSrdBase);
    int64_t clampedSize = std::min(bufferSize, kMaxNumRecords32);
    auto sizeImm = ctx.createImmType(clampedSize);
    auto sizeVal = ConstantOp::create(builder, loc, sizeImm, clampedSize);
    word2 = S_MOV_B32::create(builder, loc, sregTy, sizeVal);
  }

  // Build word 3 (stride/swizzle flags).
  auto strideImm = ctx.createImmType(kSRDStrideSwizzle);
  auto strideVal =
      ConstantOp::create(builder, loc, strideImm, kSRDStrideSwizzle);
  Value word3 = S_MOV_B32::create(builder, loc, sregTy, strideVal);

  // Pack into a 4-wide SGPR SRD.
  auto srdType = ctx.createSRegType(4, 4);
  auto srd = PackOp::create(builder, loc, srdType,
                            ValueRange{adjWord0, adjWord1, word2, word3});
  ctx.setSRDValue(memref, srd);
  ctx.clearPendingSRDBaseAdjust(memref);
  return srd;
}

SmallVector<Value, 4> emitBufferLoads(Value srd, Value voffset,
                                      int64_t instOffset, int64_t numBytes,
                                      TranslationContext &ctx, Location loc) {
  auto &builder = ctx.getBuilder();
  auto zeroImm = builder.getType<ImmType>(0);
  auto zeroSoffset = ConstantOp::create(builder, loc, zeroImm, 0);

  SmallVector<Value, 4> loadResults;
  int64_t bytesRemaining = numBytes;
  int64_t currentOffset = 0;
  while (bytesRemaining > 0) {
    int64_t loadBytes, loadDwords;
    if (bytesRemaining >= 16) {
      loadBytes = 16;
      loadDwords = 4;
    } else if (bytesRemaining >= 8) {
      loadBytes = 8;
      loadDwords = 2;
    } else if (bytesRemaining >= 4) {
      loadBytes = 4;
      loadDwords = 1;
    } else {
      loadBytes = bytesRemaining;
      loadDwords = 1;
    }

    auto loadVregType =
        ctx.createVRegType(loadDwords, loadDwords > 1 ? loadDwords : 1);
    int64_t totalOffset = instOffset + currentOffset;
    Operation *loadInstr;
    if (loadBytes == 1) {
      loadInstr =
          BUFFER_LOAD_UBYTE::create(builder, loc, TypeRange{loadVregType}, srd,
                                    voffset, zeroSoffset, totalOffset);
    } else if (loadBytes == 2) {
      loadInstr =
          BUFFER_LOAD_USHORT::create(builder, loc, TypeRange{loadVregType}, srd,
                                     voffset, zeroSoffset, totalOffset);
    } else if (loadDwords == 4) {
      loadInstr =
          BUFFER_LOAD_DWORDX4::create(builder, loc, TypeRange{loadVregType},
                                      srd, voffset, zeroSoffset, totalOffset);
    } else if (loadDwords == 2) {
      loadInstr =
          BUFFER_LOAD_DWORDX2::create(builder, loc, TypeRange{loadVregType},
                                      srd, voffset, zeroSoffset, totalOffset);
    } else {
      loadInstr =
          BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{loadVregType}, srd,
                                    voffset, zeroSoffset, totalOffset);
    }
    loadResults.push_back(loadInstr->getResult(0));
    bytesRemaining -= loadBytes;
    currentOffset += loadBytes;
  }
  return loadResults;
}

} // namespace waveasm

using namespace waveasm;

//===----------------------------------------------------------------------===//
// Forward Declarations for External Handler Files
//===----------------------------------------------------------------------===//
// These handlers are implemented in separate files under handlers/
// Forward declarations allow registerDefaultHandlers to find them.

namespace waveasm {
// GPU handlers (implemented in handlers/GPUHandlers.cpp)
LogicalResult handleGPUThreadId(Operation *op, TranslationContext &ctx);
LogicalResult handleGPUBlockId(Operation *op, TranslationContext &ctx);
LogicalResult handleGPUBarrier(Operation *op, TranslationContext &ctx);
LogicalResult handleGPUBlockDim(Operation *op, TranslationContext &ctx);
LogicalResult handleGPUGridDim(Operation *op, TranslationContext &ctx);
LogicalResult handleGPULaneId(Operation *op, TranslationContext &ctx);
LogicalResult handleGPUSubgroupBroadcast(Operation *op,
                                         TranslationContext &ctx);

// Arith handlers (implemented in handlers/ArithHandlers.cpp)
LogicalResult handleArithConstant(Operation *op, TranslationContext &ctx);
LogicalResult handleArithAddI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithSubI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithMulI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithDivUI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithRemUI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithIndexCast(Operation *op, TranslationContext &ctx);
LogicalResult handleArithAndI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithOrI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithXorI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithShLI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithShRUI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithShRSI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithExtUI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithExtSI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithTruncI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithMinSI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithMaxSI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithMinUI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithMaxUI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithCmpI(Operation *op, TranslationContext &ctx);
LogicalResult handleArithSelect(Operation *op, TranslationContext &ctx);
LogicalResult handleArithAddF(Operation *op, TranslationContext &ctx);
LogicalResult handleArithSubF(Operation *op, TranslationContext &ctx);
LogicalResult handleArithMulF(Operation *op, TranslationContext &ctx);
LogicalResult handleArithDivF(Operation *op, TranslationContext &ctx);
LogicalResult handleArithNegF(Operation *op, TranslationContext &ctx);
LogicalResult handleArithCmpF(Operation *op, TranslationContext &ctx);
LogicalResult handleArithTruncF(Operation *op, TranslationContext &ctx);
LogicalResult handleArithExtF(Operation *op, TranslationContext &ctx);

// Math handlers (implemented in handlers/MathHandlers.cpp)
LogicalResult handleMathFma(Operation *op, TranslationContext &ctx);

// Vector handlers (implemented in handlers/VectorHandlers.cpp)
// Note: handleVectorLoad, handleVectorStore, handleVectorTransferRead,
// handleVectorTransferWrite remain in this file due to their complexity.
LogicalResult handleVectorBroadcast(Operation *op, TranslationContext &ctx);
LogicalResult handleVectorExtract(Operation *op, TranslationContext &ctx);
LogicalResult handleVectorInsert(Operation *op, TranslationContext &ctx);
LogicalResult handleVectorShapeCast(Operation *op, TranslationContext &ctx);
LogicalResult handleVectorBitCast(Operation *op, TranslationContext &ctx);
LogicalResult handleVectorFma(Operation *op, TranslationContext &ctx);
LogicalResult handleVectorReduction(Operation *op, TranslationContext &ctx);
LogicalResult handleVectorExtractStridedSlice(Operation *op,
                                              TranslationContext &ctx);

} // namespace waveasm

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Check if a memref has LDS address space
bool isLDSMemRef(MemRefType memrefType) {
  auto memSpace = memrefType.getMemorySpace();
  if (!memSpace)
    return false;

  // Check for gpu.address_space<workgroup> attribute
  if (auto gpuSpace = dyn_cast<gpu::AddressSpaceAttr>(memSpace)) {
    return gpuSpace.getValue() == gpu::AddressSpace::Workgroup;
  }
  // Also check for integer address space (3 on AMDGPU)
  if (auto intAttr = dyn_cast<IntegerAttr>(memSpace)) {
    return intAttr.getInt() == 3;
  }
  return false;
}

/// Get element size in bytes
int64_t getElementBytes(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() / 8;
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8;
  return 4;
}

/// Get vector size in bytes
int64_t getVectorBytes(VectorType vecType) {
  return vecType.getNumElements() * getElementBytes(vecType.getElementType());
}

/// Get or create a mapped VGPR value for an MLIR value
[[maybe_unused]]
static Value getOrCreateVReg(Value mlirValue, TranslationContext &ctx,
                             int64_t size = 1) {
  auto &mapper = ctx.getMapper();

  // Check if already mapped
  if (auto mapped = mapper.getMapped(mlirValue))
    return *mapped;

  // Create a new VGPR - this will be a placeholder that gets resolved
  // In a real implementation, we'd need more sophisticated tracking
  [[maybe_unused]] auto vregType =
      ctx.createVRegType(size, size > 1 ? size : 1);
  auto loc = mlirValue.getLoc();

  // For block arguments (like function parameters), create a precolored reg
  if (auto blockArg = dyn_cast<BlockArgument>(mlirValue)) {
    // Function arguments are passed via SGPRs (buffer descriptors)
    // This is a simplification - real ABI handling is more complex
    auto sregType = ctx.createSRegType(2, 2); // 64-bit pointer
    auto sreg = PrecoloredSRegOp::create(ctx.getBuilder(), loc, sregType,
                                         blockArg.getArgNumber() * 2, 2);
    mapper.mapValue(mlirValue, sreg);
    return sreg;
  }

  return mlirValue; // Return as-is for now, handler will create proper op
}

// Note: GPU handlers have been moved to handlers/GPUHandlers.cpp
// Note: Arith handlers have been moved to handlers/ArithHandlers.cpp
// Note: Math handlers have been moved to handlers/MathHandlers.cpp

// Note: Simple vector handlers (handleVectorBroadcast, handleVectorExtract,
// handleVectorInsert, handleVectorShapeCast, handleVectorFma,
// handleVectorReduction, handleVectorExtractStridedSlice) moved to
// handlers/VectorHandlers.cpp

/// Handle vector.transfer_read - similar to vector.load
LogicalResult handleVectorTransferRead(Operation *op, TranslationContext &ctx) {
  auto readOp = cast<vector::TransferReadOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(readOp.getBase().getType());
  auto vectorType = readOp.getVectorType();
  int64_t numBytes = getVectorBytes(vectorType);
  int64_t numDwords = (numBytes + 3) / 4;

  auto vregType = ctx.createVRegType(numDwords, numDwords > 1 ? numDwords : 1);

  if (isLDSMemRef(memrefType)) {
    // LDS read
    Value vaddr;
    if (!readOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(readOp.getIndices()[0])) {
        vaddr = *mapped;
      }
    }
    if (!vaddr) {
      auto immType = ctx.createImmType(0);
      vaddr = ConstantOp::create(builder, loc, immType, 0);
    }

    Operation *loadInstr;
    if (numBytes == 8) {
      loadInstr = DS_READ_B64::create(builder, loc, TypeRange{vregType}, vaddr);
    } else if (numBytes == 16) {
      loadInstr =
          DS_READ_B128::create(builder, loc, TypeRange{vregType}, vaddr);
    } else {
      loadInstr = DS_READ_B32::create(builder, loc, TypeRange{vregType}, vaddr);
    }
    ctx.getMapper().mapValue(readOp.getResult(), loadInstr->getResult(0));
  } else {
    // Global read - buffer load
    auto sregType = ctx.createSRegType(4, 4);
    auto srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);

    Value voffset;
    if (!readOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(readOp.getIndices()[0])) {
        voffset = *mapped;
      }
    }
    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    auto zeroImm = builder.getType<ImmType>(0);
    auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);

    Operation *loadInstr;
    if (numDwords == 1) {
      loadInstr = BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{vregType},
                                            srd, voffset, zeroConst);
    } else if (numDwords == 2) {
      loadInstr = BUFFER_LOAD_DWORDX2::create(builder, loc, TypeRange{vregType},
                                              srd, voffset, zeroConst);
    } else {
      loadInstr = BUFFER_LOAD_DWORDX4::create(builder, loc, TypeRange{vregType},
                                              srd, voffset, zeroConst);
    }
    ctx.getMapper().mapValue(readOp.getResult(), loadInstr->getResult(0));
  }

  return success();
}

/// Handle vector.transfer_write - similar to vector.store
LogicalResult handleVectorTransferWrite(Operation *op,
                                        TranslationContext &ctx) {
  auto writeOp = cast<vector::TransferWriteOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(writeOp.getBase().getType());
  auto vectorType = writeOp.getVectorType();
  int64_t numBytes = getVectorBytes(vectorType);
  int64_t numDwords = (numBytes + 3) / 4;

  auto data = ctx.getMapper().getMapped(writeOp.getVector());
  if (!data) {
    return op->emitError("data value not mapped");
  }

  if (isLDSMemRef(memrefType)) {
    // LDS write
    Value vaddr;
    if (!writeOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(writeOp.getIndices()[0])) {
        vaddr = *mapped;
      }
    }
    if (!vaddr) {
      auto immType = ctx.createImmType(0);
      vaddr = ConstantOp::create(builder, loc, immType, 0);
    }

    if (numBytes == 0) {
      return op->emitError("zero-byte LDS write is invalid");
    } else if (numBytes == 1) {
      DS_WRITE_B8::create(builder, loc, *data, vaddr);
    } else if (numBytes == 2) {
      DS_WRITE_B16::create(builder, loc, *data, vaddr);
    } else if (numBytes <= 4) {
      DS_WRITE_B32::create(builder, loc, *data, vaddr);
    } else if (numBytes <= 8) {
      DS_WRITE_B64::create(builder, loc, *data, vaddr);
    } else if (numBytes <= 16) {
      DS_WRITE_B128::create(builder, loc, *data, vaddr);
    } else {
      return op->emitError("LDS write of ")
             << numBytes << " bytes exceeds maximum (16)";
    }
  } else {
    // Global write - buffer store
    auto sregType = ctx.createSRegType(4, 4);
    auto srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);

    Value voffset;
    if (!writeOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(writeOp.getIndices()[0])) {
        voffset = *mapped;
      }
    }
    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    if (numDwords == 1) {
      BUFFER_STORE_DWORD::create(builder, loc, *data, srd, voffset);
    } else if (numDwords == 2) {
      BUFFER_STORE_DWORDX2::create(builder, loc, *data, srd, voffset);
    } else {
      BUFFER_STORE_DWORDX4::create(builder, loc, *data, srd, voffset);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Vector Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle vector.load - emit buffer_load or ds_read based on address space
/// Splits large loads (> 16 bytes) into multiple buffer_load_dwordx4
/// instructions
LogicalResult handleVectorLoad(Operation *op, TranslationContext &ctx) {
  auto loadOp = cast<vector::LoadOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(loadOp.getBase().getType());
  auto vectorType = loadOp.getVectorType();
  int64_t numBytes = getVectorBytes(vectorType);

  // Validate vector size - zero-size vectors would cause division issues
  if (numBytes <= 0) {
    return op->emitError("vector.load with zero-size vector type is invalid");
  }

  int64_t numDwords = (numBytes + 3) / 4;

  if (isLDSMemRef(memrefType)) {
    // LDS load - ds_read_b* with proper byte address computation
    auto vregType =
        ctx.createVRegType(numDwords, numDwords > 1 ? numDwords : 1);
    auto indices = loadOp.getIndices();
    Type elementType = memrefType.getElementType();
    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

    // Compute vaddr as byte offset from indices and strides
    // Also track instruction offset from constant addends in affine expressions
    Value vaddr;
    int64_t instOffset = 0; // Instruction offset for ds_read offset:N
    SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (succeeded(memrefType.getStridesAndOffset(strides, offset))) {
      // Process each index dimension
      for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
        auto idxMapped = ctx.getMapper().getMapped(indices[i]);
        if (!idxMapped)
          continue;

        Value idx = *idxMapped;
        int64_t strideBytes = strides[i] * elementBytes;

        if (strideBytes == 0)
          continue;

        // Check if this index has a constant addend from affine.apply
        // This allows using "offset:N" in the instruction instead of VGPR
        // computation
        int64_t constAddend = ctx.getConstOffset(indices[i]);
        if (constAddend != 0) {
          // Add constant * stride to instruction offset
          instOffset += constAddend * strideBytes;
        }

        Value dimOffset;
        if (strideBytes == 1) {
          dimOffset = idx;
        } else if ((strideBytes & (strideBytes - 1)) == 0) {
          // Power of 2 - use shift
          int shift = 0;
          int64_t temp = strideBytes;
          while (temp > 1) {
            shift++;
            temp >>= 1;
          }
          auto shiftImm =
              ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          dimOffset = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(),
                                            shiftImm, idx);
        } else {
          // General case - use multiply
          auto strideImm = ConstantOp::create(
              builder, loc, ctx.createImmType(strideBytes), strideBytes);
          dimOffset = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(),
                                           idx, strideImm);
        }

        if (!vaddr) {
          vaddr = dimOffset;
        } else {
          vaddr = V_ADD_U32::create(builder, loc, ctx.createVRegType(), vaddr,
                                    dimOffset);
        }
      }
    }

    // Fallback: use first index with element size scaling
    if (!vaddr) {
      auto idxMapped = ctx.getMapper().getMapped(indices[0]);
      if (idxMapped) {
        if (elementBytes > 1) {
          int shift = 0;
          int64_t temp = elementBytes;
          while (temp > 1) {
            shift++;
            temp >>= 1;
          }
          auto shiftImm =
              ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          vaddr = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(),
                                        shiftImm, *idxMapped);
        } else {
          vaddr = *idxMapped;
        }
      } else {
        auto immType = ctx.createImmType(0);
        vaddr = ConstantOp::create(builder, loc, immType, 0);
      }
    }

    // Add the LDS base offset from memref.view (if any).
    // This handles both static offsets (from memref.view with constant byte
    // offset) and dynamic offsets (from SGPR-carried memref iter_args in
    // pipelined double-buffering loops).
    if (auto baseOffset = ctx.getLDSBaseOffset(loadOp.getBase())) {
      // If the base offset is a constant, fold it into instOffset instead of
      // emitting VALU instructions. This saves 2 VALU ops per LDS read.
      if (auto constVal = getArithConstantValue(*baseOffset)) {
        instOffset += *constVal;
      } else {
        // Non-constant base offset: add directly to vaddr via V_ADD_U32.
        // V_ADD_U32 accepts SGPR sources (VALUSrc), so SGPR-carried offsets
        // from memref iter_args work without an intermediate V_MOV_B32.
        vaddr = V_ADD_U32::create(builder, loc, ctx.createVRegType(), vaddr,
                                  *baseOffset);
      }
    }

    // Create the DS_READ operation with optional offset attribute
    Operation *readOp;
    if (numBytes == 1) {
      readOp = DS_READ_U8::create(builder, loc, TypeRange{vregType}, vaddr);
    } else if (numBytes == 2) {
      readOp = DS_READ_U16::create(builder, loc, TypeRange{vregType}, vaddr);
    } else if (numBytes <= 4) {
      readOp = DS_READ_B32::create(builder, loc, TypeRange{vregType}, vaddr);
    } else if (numBytes <= 8) {
      readOp = DS_READ_B64::create(builder, loc, TypeRange{vregType}, vaddr);
    } else if (numBytes <= 16) {
      readOp = DS_READ_B128::create(builder, loc, TypeRange{vregType}, vaddr);
    } else {
      return op->emitError("LDS read of ")
             << numBytes << " bytes exceeds maximum (16)";
    }

    // Add offset attribute if we have a non-zero instruction offset
    if (instOffset != 0) {
      readOp->setAttr("offset", builder.getI64IntegerAttr(instOffset));
    }

    ctx.getMapper().mapValue(loadOp.getResult(), readOp->getResult(0));
  } else {
    // Global load - buffer_load_dwordx* with splitting for large vectors
    auto [voffset, instOffset] = computeVOffsetFromIndices(
        memrefType, loadOp.getIndices(), ctx, loc, loadOp.getBase());

    Value srd;

    if (auto *adj = ctx.getPendingSRDBaseAdjust(loadOp.getBase())) {
      srd = emitSRDBaseAdjustment(*adj, loadOp.getBase(), ctx, loc);
    } else {
      srd = lookupSRD(loadOp.getBase(), ctx, loc);
    }

    auto loadResults =
        emitBufferLoads(srd, voffset, instOffset, numBytes, ctx, loc);

    if (!loadResults.empty()) {
      ctx.getMapper().mapValue(loadOp.getResult(), loadResults[0]);
      if (loadResults.size() > 1) {
        ctx.registerSplitResults(loadOp.getResult(), loadResults);
      }
    }
  }

  return success();
}

/// Emit exec masking: save exec, AND with VCC derived from mask VGPR.
/// Returns the saved-exec SSA value for later restoration.
static Value emitExecMask(Value maskVgpr, TranslationContext &ctx,
                          OpBuilder &builder, Location loc) {
  auto immZero = ctx.createImmType(0);
  auto zeroConst = ConstantOp::create(builder, loc, immZero, 0);
  V_CMP_NE_U32::create(builder, loc, maskVgpr, zeroConst);
  auto sregType64 = ctx.createSRegType(2, 2);
  return S_AND_SAVEEXEC_B64::create(builder, loc, sregType64);
}

/// Restore exec from saved value.
static void emitExecRestore(Value savedExec, TranslationContext &ctx,
                            OpBuilder &builder, Location loc) {
  S_MOV_B64_EXEC::create(builder, loc, savedExec);
}

/// Handle vector.maskedload with optional exec masking.
/// When the mask is available, saves exec and AND-masks it so only active lanes
/// load.  When the mask is unavailable (e.g. the mask computation chain uses
/// unsupported vector<Nxindex> ops), falls back to an unconditional
/// buffer_load. This is safe because buffer_load returns zero for out-of-bounds
/// accesses, matching the zero passthrough value used by the frontend.
LogicalResult handleVectorMaskedLoad(Operation *op, TranslationContext &ctx) {
  auto maskedLoadOp = cast<vector::MaskedLoadOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(maskedLoadOp.getBase().getType());
  auto vectorType = maskedLoadOp.getVectorType();
  int64_t numBytes = getVectorBytes(vectorType);

  if (numBytes <= 0)
    return op->emitError("vector.maskedload with zero-size vector is invalid");

  // The exec-mask path relies on hardware OOB zeroing for inactive lanes.
  // This only produces correct results when the passthrough is zero.
  if (auto cst =
          maskedLoadOp.getPassThru().getDefiningOp<arith::ConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(cst.getValue())) {
      if (denseAttr.isSplat()) {
        Attribute splatVal = denseAttr.getSplatValue<Attribute>();
        bool isZero = false;
        if (auto floatAttr = dyn_cast<FloatAttr>(splatVal))
          isZero = floatAttr.getValue().isZero();
        else if (auto intAttr = dyn_cast<IntegerAttr>(splatVal))
          isZero = intAttr.getValue().isZero();
        assert(isZero &&
               "masked load passthrough must be zero for exec-mask lowering");
      }
    }
  }

  auto mask = ctx.getMapper().getMapped(maskedLoadOp.getMask());
  Value savedExec;
  if (mask)
    savedExec = emitExecMask(*mask, ctx, builder, loc);

  auto [voffset, instOffset] = computeVOffsetFromIndices(
      memrefType, maskedLoadOp.getIndices(), ctx, loc, maskedLoadOp.getBase());

  Value srd;
  if (auto *adj = ctx.getPendingSRDBaseAdjust(maskedLoadOp.getBase())) {
    srd = emitSRDBaseAdjustment(*adj, maskedLoadOp.getBase(), ctx, loc);
  } else {
    srd = lookupSRD(maskedLoadOp.getBase(), ctx, loc);
  }
  auto loadResults =
      emitBufferLoads(srd, voffset, instOffset, numBytes, ctx, loc);

  if (mask)
    emitExecRestore(savedExec, ctx, builder, loc);

  if (!loadResults.empty()) {
    ctx.getMapper().mapValue(maskedLoadOp.getResult(), loadResults[0]);
    if (loadResults.size() > 1)
      ctx.registerSplitResults(maskedLoadOp.getResult(), loadResults);
  }

  return success();
}

/// Convert f32 data to packed bf16 for store operations.
/// Returns the converted data value and updates numBytes to reflect the
/// packed bf16 size. For a single element, emits v_cvt_bf16_f32.
/// For multiple elements, pairs adjacent f32 values and emits
/// v_cvt_pk_bf16_f32 for each pair.
static std::pair<Value, int64_t>
convertF32ToBF16ForStore(Value srcData, int64_t numElems,
                         TranslationContext &ctx, OpBuilder &builder,
                         Location loc) {
  Type srcType = srcData.getType();

  // VALU conversion instructions cannot read from AGPR. Move to VGPR first.
  if (isAGPRType(srcType)) {
    int64_t size = getRegSize(srcType);
    auto vregType = ctx.createVRegType(size, size > 1 ? size : 1);
    srcData = V_ACCVGPR_READ_B32::create(builder, loc, vregType, srcData);
    srcType = srcData.getType();
  }

  auto extractF32Elem = [&](int64_t i) -> Value {
    if (auto pvreg = dyn_cast<PVRegType>(srcType)) {
      int64_t baseIdx = pvreg.getIndex() + i;
      auto elemType = PVRegType::get(builder.getContext(), baseIdx, 1);
      return PrecoloredVRegOp::create(builder, loc, elemType, baseIdx, 1);
    }
    if (isAGPRType(srcType)) {
      auto aregType = ctx.createARegType(1, 1);
      auto aregElem = ExtractOp::create(builder, loc, aregType, srcData,
                                        builder.getI64IntegerAttr(i));
      auto vregType = ctx.createVRegType(1, 1);
      return V_ACCVGPR_READ_B32::create(builder, loc, vregType,
                                        aregElem.getResult());
    }
    auto elemType = ctx.createVRegType(1, 1);
    return ExtractOp::create(builder, loc, elemType, srcData,
                             builder.getI64IntegerAttr(i));
  };

  if (numElems == 1) {
    auto vregType = ctx.createVRegType();
    Value converted =
        V_CVT_BF16_F32::create(builder, loc, vregType, extractF32Elem(0));
    return {converted, 2};
  }

  SmallVector<Value> packedVals;
  for (int64_t i = 0; i < numElems; i += 2) {
    Value elemI = extractF32Elem(i);
    Value elemJ;
    if (i + 1 < numElems) {
      elemJ = extractF32Elem(i + 1);
    } else {
      auto immZero = ctx.createImmType(0);
      elemJ = ConstantOp::create(builder, loc, immZero, 0);
    }
    auto vregType = ctx.createVRegType();
    packedVals.push_back(
        V_CVT_PK_BF16_F32::create(builder, loc, vregType, elemI, elemJ));
  }

  Value result;
  if (packedVals.size() == 1) {
    result = packedVals[0];
  } else {
    auto packedType = ctx.createVRegType(packedVals.size(), 1);
    result = PackOp::create(builder, loc, packedType, packedVals);
  }
  int64_t newNumBytes = static_cast<int64_t>(packedVals.size()) * 4;

  // For odd element counts, the last packed dword contains a trailing zero
  // bf16 in the upper half. The caller's store loop handles subdword stores
  // so this won't corrupt adjacent memory as long as numBytes is set to the
  // exact bf16 byte count.
  int64_t exactBF16Bytes = numElems * 2;
  if (exactBF16Bytes < newNumBytes) {
    newNumBytes = exactBF16Bytes;
  }

  return {result, newNumBytes};
}

/// Handle vector.store - emit buffer_store or ds_write
/// Splits large stores (> 16 bytes) into multiple buffer_store_dwordx4
/// instructions
LogicalResult handleVectorStore(Operation *op, TranslationContext &ctx) {
  auto storeOp = cast<vector::StoreOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(storeOp.getBase().getType());
  auto vectorType = storeOp.getVectorType();
  int64_t numBytes = getVectorBytes(vectorType);

  // Validate vector size - zero-size vectors would cause issues
  if (numBytes <= 0) {
    return op->emitError("vector.store with zero-size vector type is invalid");
  }

  [[maybe_unused]] int64_t numDwords = (numBytes + 3) / 4;

  auto data = ctx.getMapper().getMapped(storeOp.getValueToStore());
  if (!data) {
    return op->emitError("data value not mapped");
  }

  if (isLDSMemRef(memrefType)) {
    // LDS store - ds_write_b* with proper byte address computation
    auto indices = storeOp.getIndices();
    Type elementType = memrefType.getElementType();
    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

    // Compute vaddr as byte offset from indices and strides
    Value vaddr;
    SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (succeeded(memrefType.getStridesAndOffset(strides, offset))) {
      // Process each index dimension
      for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
        auto idxMapped = ctx.getMapper().getMapped(indices[i]);
        if (!idxMapped)
          continue;

        Value idx = *idxMapped;
        int64_t strideBytes = strides[i] * elementBytes;

        if (strideBytes == 0)
          continue;

        Value dimOffset;
        if (strideBytes == 1) {
          dimOffset = idx;
        } else if ((strideBytes & (strideBytes - 1)) == 0) {
          // Power of 2 - use shift
          int shift = 0;
          int64_t temp = strideBytes;
          while (temp > 1) {
            shift++;
            temp >>= 1;
          }
          auto shiftImm =
              ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          dimOffset = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(),
                                            shiftImm, idx);
        } else {
          // General case - use multiply
          auto strideImm = ConstantOp::create(
              builder, loc, ctx.createImmType(strideBytes), strideBytes);
          dimOffset = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(),
                                           idx, strideImm);
        }

        if (!vaddr) {
          vaddr = dimOffset;
        } else {
          vaddr = V_ADD_U32::create(builder, loc, ctx.createVRegType(), vaddr,
                                    dimOffset);
        }
      }
    }

    // Fallback: use first index with element size scaling
    if (!vaddr) {
      auto idxMapped = ctx.getMapper().getMapped(indices[0]);
      if (idxMapped) {
        if (elementBytes > 1) {
          int shift = 0;
          int64_t temp = elementBytes;
          while (temp > 1) {
            shift++;
            temp >>= 1;
          }
          auto shiftImm =
              ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          vaddr = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(),
                                        shiftImm, *idxMapped);
        } else {
          vaddr = *idxMapped;
        }
      } else {
        auto immType = ctx.createImmType(0);
        vaddr = ConstantOp::create(builder, loc, immType, 0);
      }
    }

    // Add the LDS base offset from memref.view (if any).
    // This handles both static offsets and dynamic SGPR-carried offsets from
    // memref iter_args in pipelined double-buffering loops.
    int64_t ldsInstOffset = 0;
    if (auto baseOffset = ctx.getLDSBaseOffset(storeOp.getBase())) {
      // If the base offset is a constant, fold it into ldsInstOffset instead
      // of emitting VALU instructions. This saves 2 VALU ops per LDS write.
      if (auto constVal = getArithConstantValue(*baseOffset)) {
        ldsInstOffset += *constVal;
      } else {
        // Non-constant base offset: add directly to vaddr via V_ADD_U32.
        // V_ADD_U32 accepts SGPR sources (VALUSrc), so SGPR-carried offsets
        // from memref iter_args work without an intermediate V_MOV_B32.
        vaddr = V_ADD_U32::create(builder, loc, ctx.createVRegType(), vaddr,
                                  *baseOffset);
      }
    }

    Operation *writeOp;
    if (numBytes == 0) {
      return op->emitError("zero-byte LDS write is invalid");
    } else if (numBytes == 1) {
      writeOp = DS_WRITE_B8::create(builder, loc, *data, vaddr);
    } else if (numBytes == 2) {
      writeOp = DS_WRITE_B16::create(builder, loc, *data, vaddr);
    } else if (numBytes <= 4) {
      writeOp = DS_WRITE_B32::create(builder, loc, *data, vaddr);
    } else if (numBytes <= 8) {
      writeOp = DS_WRITE_B64::create(builder, loc, *data, vaddr);
    } else if (numBytes <= 16) {
      writeOp = DS_WRITE_B128::create(builder, loc, *data, vaddr);
    } else {
      return op->emitError("LDS write of ")
             << numBytes << " bytes exceeds maximum (16)";
    }

    // Add offset attribute if we have a non-zero LDS instruction offset
    if (ldsInstOffset != 0) {
      writeOp->setAttr("offset", builder.getI64IntegerAttr(ldsInstOffset));
    }
  } else {
    // Global store - buffer_store_dwordx* with splitting for large vectors

    // Compute voffset as byte offset from indices and strides
    // For 2D memrefs: offset = idx0 * stride0 * elementBytes + idx1 * stride1 *
    // elementBytes
    Value voffset;
    int64_t instOffset = 0; // Constant offset for buffer_store offset:N syntax
    auto indices = storeOp.getIndices();
    Type elementType = memrefType.getElementType();
    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

    // Get strides from the memref type
    SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (succeeded(memrefType.getStridesAndOffset(strides, offset))) {
      // Process each index dimension
      for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
        auto idxMapped = ctx.getMapper().getMapped(indices[i]);
        if (!idxMapped)
          continue;

        Value idx = *idxMapped;

        // Handle dynamic strides from reinterpret_cast
        if (strides[i] == ShapedType::kDynamic) {
          auto dynStride = ctx.getDynamicStride(storeOp.getBase(), i);
          if (dynStride) {
            Value bytesPerElement;
            if (elementBytes > 1) {
              auto elemBytesImm = ConstantOp::create(
                  builder, loc, ctx.createImmType(elementBytes), elementBytes);
              bytesPerElement = V_MUL_LO_U32::create(
                  builder, loc, ctx.createVRegType(), *dynStride, elemBytesImm);
            } else {
              bytesPerElement = *dynStride;
            }
            Value dimOffset = V_MUL_LO_U32::create(
                builder, loc, ctx.createVRegType(), idx, bytesPerElement);
            if (!voffset) {
              voffset = dimOffset;
            } else {
              voffset = V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                          voffset, dimOffset);
            }
            continue;
          }
        }

        int64_t strideBytes = strides[i] * elementBytes;

        if (strideBytes == 0)
          continue;

        // Check if this index has a constant addend that can be used for
        // instOffset This allows using "offset:N" in the instruction instead of
        // VGPR computation
        int64_t constAddend = ctx.getConstOffset(indices[i]);
        if (constAddend != 0) {
          // Add constant * stride to instruction offset
          instOffset += constAddend * strideBytes;
        }

        Value dimOffset;
        if (strideBytes == 1) {
          dimOffset = idx;
        } else if ((strideBytes & (strideBytes - 1)) == 0) {
          // Power of 2 - use shift
          int shift = 0;
          int64_t temp = strideBytes;
          while (temp > 1) {
            shift++;
            temp >>= 1;
          }
          auto shiftImm =
              ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          dimOffset = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(),
                                            shiftImm, idx);
        } else {
          // General case - use multiply
          auto strideImm = ConstantOp::create(
              builder, loc, ctx.createImmType(strideBytes), strideBytes);
          dimOffset = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(),
                                           idx, strideImm);
        }

        if (!voffset) {
          voffset = dimOffset;
        } else {
          // Add to existing offset
          voffset = V_ADD_U32::create(builder, loc, ctx.createVRegType(),
                                      voffset, dimOffset);
        }
      }
    } else {
      // Fallback: use first index directly if strides not available
      if (!indices.empty()) {
        if (auto mapped = ctx.getMapper().getMapped(indices[0])) {
          voffset = *mapped;
        }
      }
    }

    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    // Get SRD for this memref.
    // Emit pending base adjustment first (must happen before the store),
    // otherwise fall through to the unified lookup.
    Value srd;
    if (auto *adj = ctx.getPendingSRDBaseAdjust(storeOp.getBase())) {
      srd = emitSRDBaseAdjustment(*adj, storeOp.getBase(), ctx, loc);
    } else {
      srd = lookupSRD(storeOp.getBase(), ctx, loc);
    }

    // Check if the source value has split results from a corresponding load
    auto splitResults = ctx.getSplitResults(storeOp.getValueToStore());

    // BF16 store conversion: the arith.truncf handler defers vector f32->bf16
    // conversion, so data registers may still contain f32 values.
    if (elementType.isBF16() && data.has_value()) {
      int64_t numElems = vectorType.getNumElements();
      auto [converted, newNumBytes] =
          convertF32ToBF16ForStore(*data, numElems, ctx, builder, loc);
      data = converted;
      numBytes = newNumBytes;
    }

    // Split large stores into multiple buffer_store_dwordx4 (16 bytes each)
    // Use the same voffset for all stores, with instOffset for subsequent
    // chunks Add any constant offset from affine expressions to the base offset
    int64_t bytesRemaining = numBytes;
    int64_t currentOffset =
        instOffset; // Start with constant offset from affine expressions
    size_t splitIndex = 0;

    while (bytesRemaining > 0) {
      int64_t storeBytes;

      if (bytesRemaining >= 16) {
        storeBytes = 16;
      } else if (bytesRemaining >= 8) {
        storeBytes = 8;
      } else if (bytesRemaining >= 4) {
        storeBytes = 4;
      } else if (bytesRemaining >= 2) {
        storeBytes = 2;
      } else {
        storeBytes = 1;
      }

      // Use the correct split result if available, otherwise use mapped data
      Value storeData = *data;
      if (!splitResults.empty() && splitIndex < splitResults.size()) {
        storeData = splitResults[splitIndex];
      }

      // If the data source is an AGPR (from MFMA accumulator), we need to
      // move it to a VGPR before storing. MUBUF store instructions require
      // VGPR sources. Emit v_accvgpr_read_b32 to a temporary VGPR.
      if (isAGPRType(storeData.getType())) {
        int64_t storeDwords = (storeBytes + 3) / 4;
        auto vregType =
            ctx.createVRegType(storeDwords, storeDwords > 1 ? storeDwords : 1);
        storeData =
            V_ACCVGPR_READ_B32::create(builder, loc, vregType, storeData);
      }

      // Use instOffset attribute instead of computing new voffset
      // This generates "offset:N" modifier in assembly, saving a V_ADD_U32
      // instruction
      if (storeBytes >= 16) {
        BUFFER_STORE_DWORDX4::create(builder, loc, storeData, srd, voffset,
                                     currentOffset);
      } else if (storeBytes >= 8) {
        BUFFER_STORE_DWORDX2::create(builder, loc, storeData, srd, voffset,
                                     currentOffset);
      } else if (storeBytes >= 4) {
        BUFFER_STORE_DWORD::create(builder, loc, storeData, srd, voffset,
                                   currentOffset);
      } else if (storeBytes == 2) {
        BUFFER_STORE_SHORT::create(builder, loc, storeData, srd, voffset,
                                   currentOffset);
      } else {
        BUFFER_STORE_BYTE::create(builder, loc, storeData, srd, voffset,
                                  currentOffset);
      }

      bytesRemaining -= storeBytes;
      currentOffset += storeBytes;
      splitIndex++;
    }
  }

  return success();
}

/// Handle vector.maskedstore with exec masking.
/// Sets exec to mask lanes before the buffer_store so inactive lanes
/// do not write to memory, then restores exec.
LogicalResult handleVectorMaskedStore(Operation *op, TranslationContext &ctx) {
  auto maskedStoreOp = cast<vector::MaskedStoreOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(maskedStoreOp.getBase().getType());
  Type elementType = memrefType.getElementType();

  auto data = ctx.getMapper().getMapped(maskedStoreOp.getValueToStore());
  if (!data)
    return op->emitError("maskedstore data value not mapped");

  auto vectorType = maskedStoreOp.getValueToStore().getType();
  auto vecType = dyn_cast<VectorType>(vectorType);
  int64_t numBytes = vecType ? getVectorBytes(vecType)
                             : (vectorType.getIntOrFloatBitWidth() + 7) / 8;

  if (isLDSMemRef(memrefType))
    return op->emitError("maskedstore to LDS not yet supported");

  // Get mask VGPR
  auto mask = ctx.getMapper().getMapped(maskedStoreOp.getMask());
  if (!mask)
    return op->emitError("mask operand not mapped");

  // Save exec and apply mask
  Value savedExec = emitExecMask(*mask, ctx, builder, loc);

  // Compute address (same as handleVectorStore global path)
  auto [voffset, instOffset] =
      computeVOffsetFromIndices(memrefType, maskedStoreOp.getIndices(), ctx,
                                loc, maskedStoreOp.getBase());

  Value srd;
  if (auto *adj = ctx.getPendingSRDBaseAdjust(maskedStoreOp.getBase())) {
    srd = emitSRDBaseAdjustment(*adj, maskedStoreOp.getBase(), ctx, loc);
  } else {
    srd = lookupSRD(maskedStoreOp.getBase(), ctx, loc);
  }

  auto splitResults = ctx.getSplitResults(maskedStoreOp.getValueToStore());

  if (vecType && elementType.isBF16() && data.has_value()) {
    int64_t numElems = vecType.getNumElements();
    auto [converted, newNumBytes] =
        convertF32ToBF16ForStore(*data, numElems, ctx, builder, loc);
    data = converted;
    numBytes = newNumBytes;
  }

  // Emit buffer_store (only active lanes execute)
  int64_t bytesRemaining = numBytes;
  int64_t currentOffset = instOffset;
  size_t splitIndex = 0;

  while (bytesRemaining > 0) {
    int64_t storeBytes = (bytesRemaining >= 16)  ? 16
                         : (bytesRemaining >= 8) ? 8
                         : (bytesRemaining >= 4) ? 4
                         : (bytesRemaining >= 2) ? 2
                                                 : 1;

    Value storeData = *data;
    if (!splitResults.empty() && splitIndex < splitResults.size())
      storeData = splitResults[splitIndex];

    if (isAGPRType(storeData.getType())) {
      int64_t storeDwords = (storeBytes + 3) / 4;
      auto vregType =
          ctx.createVRegType(storeDwords, storeDwords > 1 ? storeDwords : 1);
      storeData = V_ACCVGPR_READ_B32::create(builder, loc, vregType, storeData);
    }

    if (storeBytes >= 16)
      BUFFER_STORE_DWORDX4::create(builder, loc, storeData, srd, voffset,
                                   currentOffset);
    else if (storeBytes >= 8)
      BUFFER_STORE_DWORDX2::create(builder, loc, storeData, srd, voffset,
                                   currentOffset);
    else if (storeBytes >= 4)
      BUFFER_STORE_DWORD::create(builder, loc, storeData, srd, voffset,
                                 currentOffset);
    else if (storeBytes == 2)
      BUFFER_STORE_SHORT::create(builder, loc, storeData, srd, voffset,
                                 currentOffset);
    else
      BUFFER_STORE_BYTE::create(builder, loc, storeData, srd, voffset,
                                currentOffset);

    bytesRemaining -= storeBytes;
    currentOffset += storeBytes;
    splitIndex++;
  }

  // Restore exec
  emitExecRestore(savedExec, ctx, builder, loc);

  return success();
}

//===----------------------------------------------------------------------===//
// AMDGPU Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle amdgpu.lds_barrier - emit s_waitcnt + s_barrier
//===----------------------------------------------------------------------===//
// SCF Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle scf.for - emit loop structure with iter_args support
} // namespace

namespace waveasm {
//===----------------------------------------------------------------------===//
// Forward declarations for handlers in external files
//===----------------------------------------------------------------------===//

// From AffineHandlers.cpp
LogicalResult handleAffineApply(Operation *op, TranslationContext &ctx);

// From MemRefHandlers.cpp
LogicalResult handleMemRefAlloc(Operation *op, TranslationContext &ctx);
LogicalResult handleMemRefView(Operation *op, TranslationContext &ctx);
LogicalResult handleMemRefReinterpretCast(Operation *op,
                                          TranslationContext &ctx);
LogicalResult handleMemRefSubView(Operation *op, TranslationContext &ctx);
LogicalResult handleMemRefLoad(Operation *op, TranslationContext &ctx);
LogicalResult handleMemRefStore(Operation *op, TranslationContext &ctx);
LogicalResult handleMemRefCast(Operation *op, TranslationContext &ctx);
LogicalResult handleMemRefExtractStridedMetadata(Operation *op,
                                                 TranslationContext &ctx);

// From SCFHandlers.cpp
LogicalResult handleSCFFor(Operation *op, TranslationContext &ctx);
LogicalResult handleSCFIf(Operation *op, TranslationContext &ctx);
LogicalResult handleSCFYield(Operation *op, TranslationContext &ctx);

// From AMDGPUHandlers.cpp
LogicalResult handleAMDGPULdsBarrier(Operation *op, TranslationContext &ctx);
LogicalResult handleAMDGPUMemoryCounterWait(Operation *op,
                                            TranslationContext &ctx);
LogicalResult handleAMDGPUMfma(Operation *op, TranslationContext &ctx);
LogicalResult handleAMDGPUScaledMfma(Operation *op, TranslationContext &ctx);
LogicalResult handleFatRawBufferCast(Operation *op, TranslationContext &ctx);
LogicalResult handleGatherToLds(Operation *op, TranslationContext &ctx);
LogicalResult handleRawBufferLoad(Operation *op, TranslationContext &ctx);
LogicalResult handleRawBufferStore(Operation *op, TranslationContext &ctx);
LogicalResult handleMemRefAtomicRMW(Operation *op, TranslationContext &ctx);
LogicalResult handleReadFirstLane(Operation *op, TranslationContext &ctx);
LogicalResult handleROCDLSBarrier(Operation *op, TranslationContext &ctx);
LogicalResult handleROCDLSetPrio(Operation *op, TranslationContext &ctx);
LogicalResult handleSWaitcnt(Operation *op, TranslationContext &ctx);

//===----------------------------------------------------------------------===//
// OpHandlerRegistry Implementation
//===----------------------------------------------------------------------===//

OpHandlerRegistry::OpHandlerRegistry(mlir::MLIRContext *ctx) {
  registerDefaultHandlers(ctx);
}

void OpHandlerRegistry::registerHandler(OperationName opName,
                                        OpHandler handler) {
  handlers[opName] = std::move(handler);
}

std::optional<OpHandler>
OpHandlerRegistry::getHandler(OperationName opName) const {
  auto it = handlers.find(opName);
  if (it != handlers.end())
    return it->second;
  return std::nullopt;
}

bool OpHandlerRegistry::hasHandler(OperationName opName) const {
  return handlers.contains(opName);
}

//===----------------------------------------------------------------------===//
// IREE/Stream Dialect Handlers (Unregistered Operations)
//===----------------------------------------------------------------------===//

/// Handle iree_input.binding.subspan or stream.binding.subspan
/// These operations map kernel arguments to buffer descriptors (SRDs)
LogicalResult handleBindingSubspan(Operation *op, TranslationContext &ctx) {
  // This operation connects a kernel argument to a memref
  // The result is typically consumed by a memref.reinterpret_cast
  // We track this for SRD setup during load/store operations

  // Get the binding index from the operation
  // The first operand is a !stream.binding which comes from a function argument
  int64_t bindingIdx = 0;
  if (auto bindingAttr = op->getAttrOfType<IntegerAttr>("binding")) {
    bindingIdx = bindingAttr.getInt();
  } else if (op->getNumOperands() > 0) {
    // Get binding index from the first operand (the !stream.binding argument)
    // If it's a block argument, use its argument number
    if (auto blockArg = dyn_cast<BlockArgument>(op->getOperand(0))) {
      bindingIdx = blockArg.getArgNumber();
    }
  }

  // Store the binding info
  ctx.trackBinding(op->getResult(0), bindingIdx);

  // Try to compute buffer size from result type
  int64_t bufferSize = 512; // Default buffer size
  if (auto memrefType = dyn_cast<MemRefType>(op->getResult(0).getType())) {
    bufferSize = computeBufferSizeFromMemRef(memrefType);
  }

  // Queue SRD setup for this binding
  ctx.queueSRDSetup(op->getResult(0), bindingIdx, bufferSize);

  return success();
}

//===----------------------------------------------------------------------===//
// Handler Registration
//===----------------------------------------------------------------------===//

void OpHandlerRegistry::registerDefaultHandlers(mlir::MLIRContext *ctx) {
#define REGISTER_HANDLER(OP, HANDLER)                                          \
  registerHandler(mlir::OperationName(OP::getOperationName(), ctx), HANDLER)

  // GPU dialect
  REGISTER_HANDLER(gpu::ThreadIdOp, handleGPUThreadId);
  REGISTER_HANDLER(gpu::BlockIdOp, handleGPUBlockId);
  REGISTER_HANDLER(gpu::BarrierOp, handleGPUBarrier);
  REGISTER_HANDLER(gpu::BlockDimOp, handleGPUBlockDim);
  REGISTER_HANDLER(gpu::GridDimOp, handleGPUGridDim);
  REGISTER_HANDLER(gpu::LaneIdOp, handleGPULaneId);
  REGISTER_HANDLER(gpu::SubgroupBroadcastOp, handleGPUSubgroupBroadcast);

  // Arith dialect - basic operations
  REGISTER_HANDLER(arith::ConstantOp, handleArithConstant);
  REGISTER_HANDLER(arith::AddIOp, handleArithAddI);
  REGISTER_HANDLER(arith::SubIOp, handleArithSubI);
  REGISTER_HANDLER(arith::MulIOp, handleArithMulI);
  REGISTER_HANDLER(arith::DivUIOp, handleArithDivUI);
  REGISTER_HANDLER(arith::RemUIOp, handleArithRemUI);
  REGISTER_HANDLER(arith::IndexCastOp, handleArithIndexCast);

  // Arith dialect - bitwise operations
  REGISTER_HANDLER(arith::AndIOp, handleArithAndI);
  REGISTER_HANDLER(arith::OrIOp, handleArithOrI);
  REGISTER_HANDLER(arith::XOrIOp, handleArithXorI);

  // Arith dialect - shift operations
  REGISTER_HANDLER(arith::ShLIOp, handleArithShLI);
  REGISTER_HANDLER(arith::ShRUIOp, handleArithShRUI);
  REGISTER_HANDLER(arith::ShRSIOp, handleArithShRSI);

  // Arith dialect - type conversions
  REGISTER_HANDLER(arith::ExtUIOp, handleArithExtUI);
  REGISTER_HANDLER(arith::ExtSIOp, handleArithExtSI);
  REGISTER_HANDLER(arith::TruncIOp, handleArithTruncI);
  REGISTER_HANDLER(arith::MinSIOp, handleArithMinSI);
  REGISTER_HANDLER(arith::MaxSIOp, handleArithMaxSI);
  REGISTER_HANDLER(arith::MinUIOp, handleArithMinUI);
  REGISTER_HANDLER(arith::MaxUIOp, handleArithMaxUI);

  // Arith dialect - comparison and select
  REGISTER_HANDLER(arith::CmpIOp, handleArithCmpI);
  REGISTER_HANDLER(arith::SelectOp, handleArithSelect);

  // Arith dialect - floating-point operations
  REGISTER_HANDLER(arith::AddFOp, handleArithAddF);
  REGISTER_HANDLER(arith::SubFOp, handleArithSubF);
  REGISTER_HANDLER(arith::MulFOp, handleArithMulF);
  REGISTER_HANDLER(arith::DivFOp, handleArithDivF);
  REGISTER_HANDLER(arith::NegFOp, handleArithNegF);
  REGISTER_HANDLER(arith::CmpFOp, handleArithCmpF);
  REGISTER_HANDLER(arith::TruncFOp, handleArithTruncF);
  REGISTER_HANDLER(arith::ExtFOp, handleArithExtF);

  // Math dialect
  REGISTER_HANDLER(math::FmaOp, handleMathFma);

  // Affine dialect
  REGISTER_HANDLER(affine::AffineApplyOp, handleAffineApply);

  // Vector dialect
  REGISTER_HANDLER(vector::LoadOp, handleVectorLoad);
  REGISTER_HANDLER(vector::MaskedLoadOp, handleVectorMaskedLoad);
  REGISTER_HANDLER(vector::StoreOp, handleVectorStore);
  REGISTER_HANDLER(vector::MaskedStoreOp, handleVectorMaskedStore);
  REGISTER_HANDLER(vector::ExtractStridedSliceOp,
                   handleVectorExtractStridedSlice);
  REGISTER_HANDLER(vector::BroadcastOp, handleVectorBroadcast);
  REGISTER_HANDLER(vector::ExtractOp, handleVectorExtract);
  REGISTER_HANDLER(vector::InsertOp, handleVectorInsert);
  REGISTER_HANDLER(vector::ShapeCastOp, handleVectorShapeCast);
  REGISTER_HANDLER(vector::BitCastOp, handleVectorBitCast);
  REGISTER_HANDLER(vector::TransferReadOp, handleVectorTransferRead);
  REGISTER_HANDLER(vector::TransferWriteOp, handleVectorTransferWrite);
  REGISTER_HANDLER(vector::FMAOp, handleVectorFma);
  REGISTER_HANDLER(vector::ReductionOp, handleVectorReduction);

  // AMDGPU dialect
  REGISTER_HANDLER(amdgpu::LDSBarrierOp, handleAMDGPULdsBarrier);
  REGISTER_HANDLER(amdgpu::MemoryCounterWaitOp, handleAMDGPUMemoryCounterWait);
  REGISTER_HANDLER(amdgpu::MFMAOp, handleAMDGPUMfma);
  REGISTER_HANDLER(amdgpu::ScaledMFMAOp, handleAMDGPUScaledMfma);
  REGISTER_HANDLER(amdgpu::FatRawBufferCastOp, handleFatRawBufferCast);
  REGISTER_HANDLER(amdgpu::GatherToLDSOp, handleGatherToLds);
  REGISTER_HANDLER(amdgpu::RawBufferLoadOp, handleRawBufferLoad);
  REGISTER_HANDLER(amdgpu::RawBufferStoreOp, handleRawBufferStore);
  REGISTER_HANDLER(memref::AtomicRMWOp, handleMemRefAtomicRMW);

  // ROCDL dialect
  REGISTER_HANDLER(ROCDL::ReadfirstlaneOp, handleReadFirstLane);
  REGISTER_HANDLER(ROCDL::SBarrierOp, handleROCDLSBarrier);
  REGISTER_HANDLER(ROCDL::SetPrioOp, handleROCDLSetPrio);
  REGISTER_HANDLER(ROCDL::SWaitcntOp, handleSWaitcnt);

  // IREE/Stream dialect (unregistered operations)
  registerHandler(mlir::OperationName("stream.binding.subspan", ctx),
                  handleBindingSubspan);
  registerHandler(mlir::OperationName("iree_input.binding.subspan", ctx),
                  handleBindingSubspan);

  // MemRef dialect
  REGISTER_HANDLER(memref::AllocOp, handleMemRefAlloc);
  REGISTER_HANDLER(memref::ViewOp, handleMemRefView);
  REGISTER_HANDLER(memref::ReinterpretCastOp, handleMemRefReinterpretCast);
  REGISTER_HANDLER(memref::SubViewOp, handleMemRefSubView);
  REGISTER_HANDLER(memref::LoadOp, handleMemRefLoad);
  REGISTER_HANDLER(memref::StoreOp, handleMemRefStore);
  REGISTER_HANDLER(memref::CastOp, handleMemRefCast);
  REGISTER_HANDLER(memref::ExtractStridedMetadataOp,
                   handleMemRefExtractStridedMetadata);

  // SCF dialect
  REGISTER_HANDLER(scf::ForOp, handleSCFFor);
  REGISTER_HANDLER(scf::IfOp, handleSCFIf);
  REGISTER_HANDLER(scf::YieldOp, handleSCFYield);
}

//===----------------------------------------------------------------------===//
// Translation Functions
//===----------------------------------------------------------------------===//

LogicalResult translateOperation(Operation *op, TranslationContext &ctx) {
  OperationName opName = op->getName();

  // Skip terminators (handled by parent)
  if (op->hasTrait<OpTrait::IsTerminator>())
    return success();

  // Look up handler
  if (auto handler = ctx.getRegistry().getHandler(opName)) {
    return (*handler)(op, ctx);
  }

  // No handler - emit comment for debugging
  LLVM_DEBUG(llvm::dbgs() << "No handler for: " << opName << "\n");
  ctx.emitComment(("unhandled: " + opName.getStringRef()).str());
  return success();
}

LogicalResult translateModule(ModuleOp module, StringRef targetId) {
  // Get target
  auto target = getTargetKindAttr(module.getContext(), targetId);
  if (!target) {
    return module.emitError() << "Unknown target: " << targetId;
  }

  // Collect GPU functions to translate (can't erase during walk)
  SmallVector<gpu::GPUFuncOp> gpuFuncsToTranslate;
  module.walk([&](gpu::GPUFuncOp gpuFunc) {
    if (!gpuFunc.isDeclaration())
      gpuFuncsToTranslate.push_back(gpuFunc);
  });

  // Translate each GPU function
  for (auto gpuFunc : gpuFuncsToTranslate) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    // Create a program for this GPU function
    auto program = createProgramFromGPUFunc(gpuFunc, builder, targetId);
    if (!program)
      continue;

    // Set up translation context
    builder.setInsertionPointToStart(&program.getBodyBlock());
    TranslationContext ctx(builder, program, target);

    // Map function arguments
    for (auto arg : gpuFunc.getBody().getArguments()) {
      int64_t argIdx = arg.getArgNumber();
      // Check if argument is a memref type
      if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
        // Queue SRD setup for this binding
        int64_t bufferSize = computeBufferSizeFromMemRef(memrefType);
        ctx.queueSRDSetup(arg, argIdx, bufferSize);
      } else if (arg.getType().isIndex() || arg.getType().isInteger()) {
        // Scalar kernel arg (index, i32, i64) - load from kernarg buffer
        ctx.queueScalarArgLoad(arg, argIdx);
      } else {
        // Other args (stream.binding, etc.) - placeholder VGPR
        auto vregType = ctx.createVRegType();
        auto vreg = PrecoloredVRegOp::create(builder, gpuFunc.getLoc(),
                                             vregType, argIdx, 1);
        ctx.getMapper().mapValue(arg, vreg);
      }
    }

    // First pass: handle binding.subspan operations to queue SRD setups
    for (Operation &op : gpuFunc.getBody().front()) {
      StringRef opName = op.getName().getStringRef();
      if (opName == "stream.binding.subspan" ||
          opName == "iree_input.binding.subspan" ||
          opName == "memref.reinterpret_cast") {
        (void)translateOperation(&op, ctx);
      }
    }

    // Emit SRD prologue (s_load, s_waitcnt, s_mov instructions)
    ctx.emitSRDPrologue();

    // Second pass: translate all operations
    for (Operation &op : gpuFunc.getBody().front()) {
      if (failed(translateOperation(&op, ctx)))
        return failure();
    }

    // Emit endpgm
    S_ENDPGM::create(builder, gpuFunc.getLoc());

    // Set the number of kernel arguments attribute on the program
    size_t numKernelArgs = ctx.getNumKernelArgs();
    program->setAttr(
        "num_kernel_args",
        builder.getI64IntegerAttr(static_cast<int64_t>(numKernelArgs)));
    // Set the LDS size attribute if any LDS was allocated
    int64_t ldsSize = ctx.getTotalLDSSize();
    if (ldsSize > 0) {
      program->setAttr("lds_size", builder.getI64IntegerAttr(ldsSize));
    }

    // Erase the original GPU function to avoid symbol collision
    gpuFunc.erase();
  }

  // Clean up empty gpu.module containers
  SmallVector<gpu::GPUModuleOp> gpuModulesToErase;
  module.walk([&](gpu::GPUModuleOp gpuModule) {
    // Check if the module is now empty (only contains gpu.module_end
    // terminator)
    auto *body = gpuModule.getBody();
    if (body) {
      // If only one op (the terminator) remains, erase the module
      if (body->getOperations().size() <= 1)
        gpuModulesToErase.push_back(gpuModule);
    }
  });
  for (auto gpuModule : gpuModulesToErase)
    gpuModule.erase();

  // Collect func.func ops to translate (can't erase during walk)
  SmallVector<func::FuncOp> funcsToTranslate;
  module.walk([&](func::FuncOp funcOp) {
    // Skip if already handled as GPU func
    if (funcOp->getParentOfType<gpu::GPUModuleOp>())
      return;
    // Skip function declarations (no body)
    if (funcOp.isDeclaration())
      return;
    funcsToTranslate.push_back(funcOp);
  });

  // Translate each func.func
  for (auto funcOp : funcsToTranslate) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    // Create program from func.func
    auto program = createProgramFromFunc(funcOp, builder, targetId);
    if (!program)
      continue;

    // Set up translation context
    builder.setInsertionPointToStart(&program.getBodyBlock());
    // Create target instance and keep it alive for the duration of ctx
    auto targetInstance = getTargetKindAttr(module.getContext(), targetId);
    TranslationContext ctx(builder, program, targetInstance);

    // Map function arguments
    for (auto arg : funcOp.getBody().getArguments()) {
      int64_t argIdx = arg.getArgNumber();
      // Check if argument is a memref type
      if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
        // Queue SRD setup for this binding
        int64_t bufferSize = computeBufferSizeFromMemRef(memrefType);
        ctx.queueSRDSetup(arg, argIdx, bufferSize);
      } else if (arg.getType().isIndex() || arg.getType().isInteger()) {
        // Scalar kernel arg (index, i32, i64) - load from kernarg buffer
        ctx.queueScalarArgLoad(arg, argIdx);
      } else {
        // Other args (stream.binding, etc.) - placeholder VGPR
        auto vregType = ctx.createVRegType();
        auto vreg = PrecoloredVRegOp::create(builder, funcOp.getLoc(), vregType,
                                             argIdx, 1);
        ctx.getMapper().mapValue(arg, vreg);
      }
    }

    // First pass: handle binding.subspan operations to queue SRD setups
    for (Operation &op : funcOp.getBody().front()) {
      StringRef opName = op.getName().getStringRef();
      if (opName == "stream.binding.subspan" ||
          opName == "iree_input.binding.subspan" ||
          opName == "memref.reinterpret_cast") {
        (void)translateOperation(&op, ctx);
      }
    }

    // Emit SRD prologue (s_load, s_waitcnt, s_mov instructions)
    ctx.emitSRDPrologue();

    // Second pass: translate all operations
    for (Operation &op : funcOp.getBody().front()) {
      (void)translateOperation(&op, ctx);
    }

    // Emit endpgm
    S_ENDPGM::create(builder, funcOp.getLoc());

    // Set the number of kernel arguments attribute on the program
    size_t numKernelArgs = ctx.getNumKernelArgs();
    program->setAttr(
        "num_kernel_args",
        builder.getI64IntegerAttr(static_cast<int64_t>(numKernelArgs)));
    // Set the LDS size attribute if any LDS was allocated
    int64_t ldsSize = ctx.getTotalLDSSize();
    if (ldsSize > 0) {
      program->setAttr("lds_size", builder.getI64IntegerAttr(ldsSize));
    }

    // Erase the original func.func to avoid symbol name collision
    funcOp.erase();
  }

  return success();
}

LogicalResult translateModule(ModuleOp module,
                              const TranslationOptions &options) {
  // Get target
  auto target = getTargetKindAttr(module.getContext(), options.targetId);
  if (!target) {
    return module.emitError() << "Unknown target: " << options.targetId;
  }

  // Collect func.func ops to translate (can't erase during walk)
  SmallVector<func::FuncOp> funcsToTranslate;
  module.walk([&](func::FuncOp funcOp) {
    // Skip function declarations (no body)
    if (funcOp.isDeclaration())
      return;
    funcsToTranslate.push_back(funcOp);
  });

  // Translate each func.func
  for (auto funcOp : funcsToTranslate) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    // Create target attribute
    auto *ctx = builder.getContext();
    auto loc = funcOp.getLoc();
    auto targetAttr =
        TargetAttr::get(ctx, getTargetKindAttr(ctx, options.targetId), 5);

    // Create ABI attribute
    auto abiAttr =
        KernelABIAttr::get(ctx, 0, 0, std::nullopt, std::nullopt, std::nullopt);

    // Determine workgroup size - use explicit options if provided
    ArrayAttr workgroupSizeAttr;
    if (options.hasExplicitWorkgroupSize()) {
      auto [wgX, wgY, wgZ] = options.getWorkgroupSize();
      SmallVector<Attribute, 3> sizes = {builder.getI64IntegerAttr(wgX),
                                         builder.getI64IntegerAttr(wgY),
                                         builder.getI64IntegerAttr(wgZ)};
      workgroupSizeAttr = builder.getArrayAttr(sizes);
    } else {
      // Fall back to extraction from MLIR (translation_info or gpu.thread_id)
      if (auto translationInfo = funcOp->getAttr("translation_info")) {
        // Parse workgroup_size from translation_info
        std::string attrStr;
        llvm::raw_string_ostream os(attrStr);
        translationInfo.print(os);
        os.flush();
        auto pos = attrStr.find("workgroup_size");
        if (pos != std::string::npos) {
          auto bracketStart = attrStr.find('[', pos);
          auto bracketEnd = attrStr.find(']', bracketStart);
          if (bracketStart != std::string::npos &&
              bracketEnd != std::string::npos) {
            std::string arrayStr =
                attrStr.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
            SmallVector<Attribute, 3> sizes;
            std::stringstream ss(arrayStr);
            std::string token;
            while (std::getline(ss, token, ',')) {
              size_t start = token.find_first_not_of(" \t");
              size_t end = token.find_last_not_of(" \t");
              if (start != std::string::npos && end != std::string::npos) {
                std::string numStr = token.substr(start, end - start + 1);
                bool isValid = !numStr.empty();
                for (size_t i = 0; i < numStr.size(); ++i) {
                  char c = numStr[i];
                  if (i == 0 && c == '-')
                    continue;
                  if (!std::isdigit(static_cast<unsigned char>(c))) {
                    isValid = false;
                    break;
                  }
                }
                if (isValid) {
                  int64_t val = std::stoll(numStr);
                  sizes.push_back(builder.getI64IntegerAttr(val));
                }
              }
            }
            if (sizes.size() >= 3) {
              workgroupSizeAttr = builder.getArrayAttr(sizes);
            }
          }
        }
      }
      // Try gpu.thread_id upper_bound if still not found
      if (!workgroupSizeAttr) {
        int64_t wgSizeX = 64, wgSizeY = 1, wgSizeZ = 1;
        funcOp.walk([&](gpu::ThreadIdOp threadIdOp) {
          if (auto upperBoundAttr =
                  threadIdOp->getAttrOfType<IntegerAttr>("upper_bound")) {
            int64_t bound = upperBoundAttr.getInt();
            switch (threadIdOp.getDimension()) {
            case gpu::Dimension::x:
              wgSizeX = bound;
              break;
            case gpu::Dimension::y:
              wgSizeY = bound;
              break;
            case gpu::Dimension::z:
              wgSizeZ = bound;
              break;
            }
          }
        });
        SmallVector<Attribute, 3> sizes = {builder.getI64IntegerAttr(wgSizeX),
                                           builder.getI64IntegerAttr(wgSizeY),
                                           builder.getI64IntegerAttr(wgSizeZ)};
        workgroupSizeAttr = builder.getArrayAttr(sizes);
      }
    }

    // Create program with workgroup size
    auto program =
        ProgramOp::create(builder, loc, funcOp.getName(), targetAttr, abiAttr,
                          /*vgprs=*/int64_t{256},
                          /*sgprs=*/int64_t{104},
                          /*workgroup_size=*/workgroupSizeAttr,
                          /*lds_size=*/IntegerAttr{});

    if (program.getBody().empty())
      program.getBody().emplaceBlock();

    // Set up translation context
    builder.setInsertionPointToStart(&program.getBodyBlock());
    auto targetInstance = getTargetKindAttr(ctx, options.targetId);
    TranslationContext transCtx(builder, program, targetInstance);

    // Map function arguments
    for (auto arg : funcOp.getBody().getArguments()) {
      int64_t argIdx = arg.getArgNumber();
      if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
        int64_t bufferSize = computeBufferSizeFromMemRef(memrefType);
        transCtx.queueSRDSetup(arg, argIdx, bufferSize);
      } else if (arg.getType().isIndex() || arg.getType().isInteger()) {
        transCtx.queueScalarArgLoad(arg, argIdx);
      } else {
        auto vregType = transCtx.createVRegType();
        auto vreg = PrecoloredVRegOp::create(builder, funcOp.getLoc(), vregType,
                                             argIdx, 1);
        transCtx.getMapper().mapValue(arg, vreg);
      }
    }

    // First pass: handle binding.subspan operations
    for (Operation &op : funcOp.getBody().front()) {
      StringRef opName = op.getName().getStringRef();
      if (opName == "stream.binding.subspan" ||
          opName == "iree_input.binding.subspan" ||
          opName == "memref.reinterpret_cast") {
        (void)translateOperation(&op, transCtx);
      }
    }

    // Emit SRD prologue
    transCtx.emitSRDPrologue();

    // Second pass: translate all operations
    for (Operation &op : funcOp.getBody().front()) {
      (void)translateOperation(&op, transCtx);
    }

    // Emit endpgm
    S_ENDPGM::create(builder, funcOp.getLoc());

    // Set kernel arguments count
    size_t numKernelArgs = transCtx.getNumKernelArgs();
    program->setAttr(
        "num_kernel_args",
        builder.getI64IntegerAttr(static_cast<int64_t>(numKernelArgs)));
    // Set LDS size if used
    int64_t ldsSize = transCtx.getTotalLDSSize();
    if (ldsSize > 0) {
      program->setAttr("lds_size", builder.getI64IntegerAttr(ldsSize));
    }

    // Erase original function
    funcOp.erase();
  }

  return success();
}

ProgramOp createProgramFromGPUFunc(gpu::GPUFuncOp gpuFunc, OpBuilder &builder,
                                   StringRef targetId) {
  auto *ctx = builder.getContext();
  auto loc = gpuFunc.getLoc();

  // Create target attribute
  auto targetAttr = TargetAttr::get(ctx, getTargetKindAttr(ctx, targetId), 5);

  // Create ABI attribute with default bindings
  auto abiAttr =
      KernelABIAttr::get(ctx, 0, 0, std::nullopt, std::nullopt, std::nullopt);

  // Create program
  auto program =
      ProgramOp::create(builder, loc, gpuFunc.getName(), targetAttr, abiAttr,
                        /*vgprs=*/int64_t{256},
                        /*sgprs=*/int64_t{104},
                        /*workgroup_size=*/ArrayAttr{},
                        /*lds_size=*/IntegerAttr{});

  // Ensure the body region has a block
  if (program.getBody().empty())
    program.getBody().emplaceBlock();

  return program;
}

ProgramOp createProgramFromFunc(func::FuncOp funcOp, OpBuilder &builder,
                                StringRef targetId) {
  auto *ctx = builder.getContext();
  auto loc = funcOp.getLoc();

  // Create target attribute
  auto targetAttr = TargetAttr::get(ctx, getTargetKindAttr(ctx, targetId), 5);

  // Create ABI attribute
  auto abiAttr =
      KernelABIAttr::get(ctx, 0, 0, std::nullopt, std::nullopt, std::nullopt);

  // Try to extract workgroup size from translation_info attribute
  // The attribute looks like: #iree_codegen.translation_info<... workgroup_size
  // = [64, 1, 1] ...> When parsed with unregistered dialects, it becomes an
  // opaque attribute that we need to parse from its string representation.
  ArrayAttr workgroupSizeAttr;
  if (auto translationInfo = funcOp->getAttr("translation_info")) {
    // Try as dictionary first (in case it's a registered dialect)
    if (auto dictAttr = dyn_cast<DictionaryAttr>(translationInfo)) {
      if (auto wgSize = dictAttr.get("workgroup_size")) {
        workgroupSizeAttr = dyn_cast<ArrayAttr>(wgSize);
      }
    }
    // If not a dictionary, parse from string representation
    if (!workgroupSizeAttr) {
      std::string attrStr;
      llvm::raw_string_ostream os(attrStr);
      translationInfo.print(os);
      os.flush();

      // Parse "workgroup_size = [X, Y, Z]" from the string
      // Look for pattern: workgroup_size = [num, num, num]
      auto pos = attrStr.find("workgroup_size");
      if (pos != std::string::npos) {
        auto bracketStart = attrStr.find('[', pos);
        auto bracketEnd = attrStr.find(']', bracketStart);
        if (bracketStart != std::string::npos &&
            bracketEnd != std::string::npos) {
          std::string arrayStr =
              attrStr.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
          SmallVector<Attribute, 3> sizes;
          std::stringstream ss(arrayStr);
          std::string token;
          while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
              std::string numStr = token.substr(start, end - start + 1);
              // Check if it's a valid integer
              bool isValid = !numStr.empty();
              for (size_t i = 0; i < numStr.size(); ++i) {
                char c = numStr[i];
                if (i == 0 && c == '-')
                  continue; // Allow leading minus
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                  isValid = false;
                  break;
                }
              }
              if (isValid) {
                int64_t val = std::stoll(numStr);
                sizes.push_back(builder.getI64IntegerAttr(val));
              }
            }
          }
          if (sizes.size() >= 3) {
            workgroupSizeAttr = builder.getArrayAttr(sizes);
          }
        }
      }
    }
  }

  // If no translation_info, try to extract from gpu.thread_id upper_bound attrs
  if (!workgroupSizeAttr) {
    int64_t wgSizeX = 64, wgSizeY = 1, wgSizeZ = 1; // defaults
    funcOp.walk([&](gpu::ThreadIdOp threadIdOp) {
      if (auto upperBoundAttr =
              threadIdOp->getAttrOfType<IntegerAttr>("upper_bound")) {
        int64_t bound = upperBoundAttr.getInt();
        switch (threadIdOp.getDimension()) {
        case gpu::Dimension::x:
          wgSizeX = bound;
          break;
        case gpu::Dimension::y:
          wgSizeY = bound;
          break;
        case gpu::Dimension::z:
          wgSizeZ = bound;
          break;
        }
      }
    });
    SmallVector<Attribute, 3> sizes = {builder.getI64IntegerAttr(wgSizeX),
                                       builder.getI64IntegerAttr(wgSizeY),
                                       builder.getI64IntegerAttr(wgSizeZ)};
    workgroupSizeAttr = builder.getArrayAttr(sizes);
  }

  // Create program
  auto program =
      ProgramOp::create(builder, loc, funcOp.getName(), targetAttr, abiAttr,
                        /*vgprs=*/int64_t{256},
                        /*sgprs=*/int64_t{104},
                        /*workgroup_size=*/workgroupSizeAttr,
                        /*lds_size=*/IntegerAttr{});

  // Ensure the body region has a block
  if (program.getBody().empty())
    program.getBody().emplaceBlock();

  return program;
}

} // namespace waveasm
