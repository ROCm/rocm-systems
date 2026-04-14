// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Shared Helper Functions for Operation Handlers
//===----------------------------------------------------------------------===//
//
// This file implements utility functions declared in Handlers.h that are
// shared across multiple handler files (ArithHandlers, AffineHandlers,
// MemRefHandlers, AMDGPUHandlers, etc.).
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// isLDSMemRef
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// getElementBytes
//===----------------------------------------------------------------------===//

int64_t getElementBytes(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() / 8;
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8;
  return 4;
}

//===----------------------------------------------------------------------===//
// computeBufferSizeFromMemRef
//===----------------------------------------------------------------------===//

int64_t computeBufferSizeFromMemRef(MemRefType memrefType) {
  // Use (1 << 31) - 2 = 0x7FFFFFFE as num_records. The Wave Python frontend
  // emits OOB sentinel index values at (valid_bytes + elem_bytes) / elem_bytes
  // which lands at byte offset 0x7FFFFFFF. With num_records = 0x7FFFFFFE the
  // sentinel is one byte past the SRD range, so hardware returns 0 for OOB
  // lanes. Using 0xFFFFFFFF would make the sentinel "in bounds" and cause a
  // real access to unmapped memory, triggering HSA page faults.
  (void)memrefType;
  return 0x7FFFFFFE;
}

//===----------------------------------------------------------------------===//
// isPowerOf2 / log2
//===----------------------------------------------------------------------===//

bool isPowerOf2(int64_t val) { return val > 0 && (val & (val - 1)) == 0; }

int64_t log2(int64_t val) {
  int64_t result = 0;
  while ((1LL << result) < val)
    ++result;
  return result;
}

//===----------------------------------------------------------------------===//
// getArithConstantValue
//===----------------------------------------------------------------------===//

std::optional<int64_t> getArithConstantValue(Value val) {
  IntegerAttr attr;
  if (mlir::matchPattern(val, mlir::m_Constant(&attr)))
    return attr.getInt();
  return std::nullopt;
}

} // namespace waveasm
