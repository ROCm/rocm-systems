// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WAVEASM_TRANSFORMS_UTILS_H
#define WAVEASM_TRANSFORMS_UTILS_H

#include "mlir/IR/Value.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include <optional>

namespace waveasm {

/// Extract an integer constant from a Value.
/// Handles ConstantOp directly, or V_MOV_B32(ConstantOp), or ImmType.
inline std::optional<int64_t> getConstantValue(mlir::Value v) {
  if (auto constOp = v.getDefiningOp<ConstantOp>())
    return constOp.getValue();
  if (auto movOp = v.getDefiningOp<V_MOV_B32>()) {
    if (auto constOp = movOp.getSrc().getDefiningOp<ConstantOp>())
      return constOp.getValue();
  }
  if (auto immType = llvm::dyn_cast<ImmType>(v.getType()))
    return immType.getValue();
  return std::nullopt;
}

} // namespace waveasm

#endif // WAVEASM_TRANSFORMS_UTILS_H
