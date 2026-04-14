// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace waveasm;

#include "waveasm/Dialect/WaveASMDialect.cpp.inc"

// Include OpInterface definitions
#include "waveasm/Dialect/WaveASMOpInterfaces.cpp.inc"

// Include full type and attribute definitions to make storage types complete
#define GET_TYPEDEF_CLASSES
#include "waveasm/Dialect/WaveASMTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_ATTRDEF_CLASSES
#include "waveasm/Dialect/WaveASMAttrs.cpp.inc"
#undef GET_ATTRDEF_CLASSES

//===----------------------------------------------------------------------===//
// WAVEASM Dialect
//===----------------------------------------------------------------------===//

void WaveASMDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "waveasm/Dialect/WaveASMOps.cpp.inc"
      >();
}

void WaveASMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "waveasm/Dialect/WaveASMTypes.cpp.inc"
      >();
}

void WaveASMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "waveasm/Dialect/WaveASMAttrs.cpp.inc"
      >();
}
