// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Dialect/WaveASMAttrs.h"

using namespace mlir;
using namespace waveasm;

#include "waveasm/Dialect/WaveASMAttrEnums.cpp.inc"
#include "waveasm/Dialect/WaveASMAttrInterfaces.cpp.inc"

TargetAttrInterface waveasm::getTargetKindAttr(mlir::MLIRContext *ctx,
                                               TargetKind targetKind) {
  switch (targetKind) {
  case TargetKind::GFX942:
    return GFX942TargetAttr::get(ctx);
  case TargetKind::GFX950:
    return GFX950TargetAttr::get(ctx);
  case TargetKind::GFX1250:
    return GFX1250TargetAttr::get(ctx);
  }
  llvm_unreachable("Invalid target kind");
}

// Note: Attribute definitions (GET_ATTRDEF_CLASSES) are included in
// WaveASMDialect.cpp to ensure storage types are complete when registering
// attributes.
