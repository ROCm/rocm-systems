// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Math Dialect Handlers
//===----------------------------------------------------------------------===//
//
// This file implements handlers for Math dialect operations:
//   - math.fma
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"

#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;

namespace waveasm {

LogicalResult handleMathFma(Operation *op, TranslationContext &ctx) {
  auto fmaOp = cast<math::FmaOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto a = ctx.getMapper().getMapped(fmaOp.getA());
  auto b = ctx.getMapper().getMapped(fmaOp.getB());
  auto c = ctx.getMapper().getMapped(fmaOp.getC());

  if (!a || !b || !c) {
    return op->emitError("operands not mapped");
  }

  // v_fma_f32 dst, a, b, c computes a*b+c
  auto fma = V_FMA_F32::create(builder, loc, vregType, *a, *b, *c);
  ctx.getMapper().mapValue(fmaOp.getResult(), fma);
  return success();
}

} // namespace waveasm
