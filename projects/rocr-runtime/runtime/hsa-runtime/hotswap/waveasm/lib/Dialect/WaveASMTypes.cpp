// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Dialect/WaveASMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace waveasm;

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMEnums.cpp.inc"

// Note: Type definitions (GET_TYPEDEF_CLASSES) are included in
// WaveASMDialect.cpp to ensure storage types are complete when registering
// types.
