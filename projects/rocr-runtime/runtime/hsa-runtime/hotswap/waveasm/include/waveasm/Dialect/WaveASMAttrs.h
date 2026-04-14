// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMATTRS_H
#define WaveASM_DIALECT_WAVEASMATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "waveasm/Dialect/WaveASMTypes.h"

namespace waveasm {

//===----------------------------------------------------------------------===//
// Target Feature Flags
//===----------------------------------------------------------------------===//

enum class TargetFeature : uint32_t {
  None = 0,
  HasMFMA = 1 << 0,          // Matrix fused multiply-add
  HasFP8 = 1 << 1,           // FP8 support
  HasPackedFP32 = 1 << 2,    // Packed FP32 operations
  HasWave32 = 1 << 3,        // Wave32 mode support
  HasWave64 = 1 << 4,        // Wave64 mode support
  HasXF32 = 1 << 5,          // Extended FP32 (TF32)
  HasScaledMFMA = 1 << 6,    // Scaled MFMA instructions
  HasAtomicFAdd = 1 << 7,    // Atomic float add
  HasGlobalLoadLDS = 1 << 8, // Global load to LDS
  HasFlatScratch = 1 << 9,   // Flat scratch support
  HasAGPRs = 1 << 10,        // Accumulator GPRs
};

inline TargetFeature operator|(TargetFeature a, TargetFeature b) {
  return static_cast<TargetFeature>(static_cast<uint32_t>(a) |
                                    static_cast<uint32_t>(b));
}

inline TargetFeature operator&(TargetFeature a, TargetFeature b) {
  return static_cast<TargetFeature>(static_cast<uint32_t>(a) &
                                    static_cast<uint32_t>(b));
}

inline bool hasFeature(TargetFeature features, TargetFeature query) {
  return (features & query) == query;
}

//===----------------------------------------------------------------------===//
// Wave Size
//===----------------------------------------------------------------------===//

enum class WaveSize : int64_t {
  Wave32 = 32,
  Wave64 = 64,
};
} // namespace waveasm

#include "waveasm/Dialect/WaveASMAttrEnums.h.inc"
#include "waveasm/Dialect/WaveASMAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "waveasm/Dialect/WaveASMAttrs.h.inc"

namespace waveasm {
// Create a context-unique attribute describing a target for the specified
// architecture. If the architecture is not supported, return nullptr.
TargetAttrInterface getTargetKindAttr(mlir::MLIRContext *ctx,
                                      TargetKind targetKind);
static inline TargetAttrInterface getTargetKindAttr(mlir::MLIRContext *ctx,
                                                    llvm::StringRef targetId) {
  std::optional<TargetKind> targetKind = symbolizeTargetKind(targetId);
  if (!targetKind) {
    return nullptr;
  }
  return getTargetKindAttr(ctx, *targetKind);
}
} // namespace waveasm

#endif // WaveASM_DIALECT_WAVEASMATTRS_H
