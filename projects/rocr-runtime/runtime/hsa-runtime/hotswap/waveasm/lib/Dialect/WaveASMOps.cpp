// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace waveasm;

//===----------------------------------------------------------------------===//
// ProgramOp
//===----------------------------------------------------------------------===//

// Note: ProgramOp now uses the new attribute format from WAVEASMAttrs.td
// Verification is handled by TableGen-generated code for basic structure.
// Custom verification can be added here if needed.

//===----------------------------------------------------------------------===//
// MFMA Operation Verifiers
//===----------------------------------------------------------------------===//

/// Helper to get expected accumulator size from MFMA mnemonic.
/// Returns the expected number of 32-bit registers for the accumulator.
static int64_t getMFMAAccumulatorSize(llvm::StringRef mnemonic) {
  // Parse mnemonic to extract dimensions: v_mfma_f32_MxNxK_type
  // Accumulator sizes must match the AccumulatorSize in WaveASMOps.td.
  // More specific checks (e.g. 4x4x4_f64) must come before generic ones
  // (e.g. 4x4x4) to avoid false matches.

  // F64 variants (check before generic patterns)
  if (mnemonic.contains("4x4x4_f64"))
    return 2;
  if (mnemonic.contains("16x16x4_f64"))
    return 8;

  // Scaled MFMA variants (gfx950+) - check before generic dimension patterns
  // to avoid false matches (e.g., "16x16x128" should not match "16x16x16")
  if (mnemonic.contains("16x16x128"))
    return 4;
  if (mnemonic.contains("32x32x64"))
    return 16;

  // 4-register accumulators
  if (mnemonic.contains("16x16x16") || mnemonic.contains("16x16x32") ||
      mnemonic.contains("16x16x4_f32") || mnemonic.contains("4x4x4") ||
      mnemonic.contains("4x4x1"))
    return 4;

  // 16-register accumulators
  if (mnemonic.contains("32x32x8") || mnemonic.contains("32x32x16") ||
      mnemonic.contains("32x32x2") || mnemonic.contains("16x16x4_f16") ||
      mnemonic.contains("16x16x4_bf16") || mnemonic.contains("16x16x4_i8"))
    return 16;

  // 32-register accumulators
  if (mnemonic.contains("32x32x4"))
    return 32;

  // Default fallback
  return 4;
}

/// Helper to get register vector width for VGPR/AGPR virtual/physical types.
static std::optional<int64_t> getVectorRegSize(Type type) {
  if (auto vregType = dyn_cast<VRegType>(type))
    return vregType.getSize();
  if (auto pvregType = dyn_cast<PVRegType>(type))
    return pvregType.getSize();
  if (auto aregType = dyn_cast<ARegType>(type))
    return aregType.getSize();
  if (auto paregType = dyn_cast<PARegType>(type))
    return paregType.getSize();
  return std::nullopt;
}

/// Verifier for MFMA operations - validates accumulator size matches expected.
/// This is a template that works for all MFMAOp subclasses.
template <typename MFMAOpType>
static LogicalResult verifyMFMAOp(MFMAOpType op) {
  // Get the accumulator operand
  Value acc = op.getAcc();
  Type accType = acc.getType();

  // If accumulator is an immediate (constant 0 for initialization), skip check
  if (isa<ImmType>(accType))
    return success();

  // Get register size from type
  int64_t accSize = getVectorRegSize(accType).value_or(1);

  // Get expected accumulator size from the operation name
  llvm::StringRef mnemonic = op->getName().getStringRef();
  int64_t expectedSize = getMFMAAccumulatorSize(mnemonic);

  // Verify accumulator size matches expected
  if (accSize != expectedSize && accSize != 1) {
    // Size 1 is allowed (will be broadcast or is a placeholder)
    return op.emitOpError()
           << "accumulator size mismatch: expected " << expectedSize
           << " registers but got " << accSize << " for " << mnemonic;
  }

  // Verify result type matches accumulator type (tied operand constraint)
  Value dst = op.getDst();
  Type dstType = dst.getType();
  int64_t dstSize = getVectorRegSize(dstType).value_or(1);

  if (dstSize != expectedSize) {
    return op.emitOpError()
           << "result size mismatch: expected " << expectedSize
           << " registers but got " << dstSize << " for " << mnemonic;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Extract Operation Verifier
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify() {
  Type vectorType = getVector().getType();
  Type resultType = getResult().getType();
  int64_t index = getIndex();

  if (index < 0) {
    return emitOpError() << "index must be non-negative, got " << index;
  }

  // Preserve register bank (VGPR <-> VGPR, AGPR <-> AGPR, SGPR <-> SGPR).
  bool bankMismatch = false;
  if (isSGPRType(vectorType) != isSGPRType(resultType))
    bankMismatch = true;
  else if (isAGPRType(vectorType) != isAGPRType(resultType))
    bankMismatch = true;
  if (bankMismatch) {
    return emitOpError()
           << "result register class must match source register class: source "
           << vectorType << ", result " << resultType;
  }

  int64_t vectorSize = getRegSize(vectorType);
  int64_t resultSize = getRegSize(resultType);
  if (resultSize > vectorSize) {
    return emitOpError() << "result width (" << resultSize
                         << ") exceeds source width (" << vectorSize << ")";
  }
  if (index > vectorSize - resultSize) {
    return emitOpError() << "extract range [" << index << ", "
                         << (index + resultSize) << ") exceeds source width "
                         << vectorSize;
  }

  return success();
}

LogicalResult InsertOp::verify() {
  Type vectorType = getVector().getType();
  Type valueType = getValue().getType();
  Type resultType = getResult().getType();
  int64_t index = getIndex();

  if (index < 0)
    return emitOpError() << "index must be non-negative, got " << index;

  // Result must match the source vector type.
  if (vectorType != resultType)
    return emitOpError() << "result type " << resultType
                         << " must match source type " << vectorType;

  // Preserve register bank.
  bool bankMismatch = false;
  if (isSGPRType(vectorType) != isSGPRType(valueType))
    bankMismatch = true;
  else if (isAGPRType(vectorType) != isAGPRType(valueType))
    bankMismatch = true;
  if (bankMismatch)
    return emitOpError()
           << "inserted value register class must match source register class: "
              "source "
           << vectorType << ", value " << valueType;

  int64_t vectorSize = getRegSize(vectorType);
  int64_t valueSize = getRegSize(valueType);
  if (valueSize > vectorSize)
    return emitOpError() << "inserted value width (" << valueSize
                         << ") exceeds source width (" << vectorSize << ")";
  if (index > vectorSize - valueSize)
    return emitOpError() << "insert range [" << index << ", "
                         << (index + valueSize) << ") exceeds source width "
                         << vectorSize;

  return success();
}

//===----------------------------------------------------------------------===//
// VMEM Load Operation Verifiers
//===----------------------------------------------------------------------===//

/// Verifier for VMEM load operations - validates SRD size and result count.
template <typename VMEMLoadOpType>
static LogicalResult verifyVMEMLoadOp(VMEMLoadOpType op) {
  // Verify SRD (saddr) is a 4-SGPR quad for buffer operations
  Value saddr = op.getSaddr();
  Type saddrType = saddr.getType();

  int64_t srdSize = 1;
  if (auto sregType = dyn_cast<SRegType>(saddrType)) {
    srdSize = sregType.getSize();
  } else if (auto psregType = dyn_cast<PSRegType>(saddrType)) {
    srdSize = psregType.getSize();
  }

  // Buffer operations require a 4-SGPR SRD
  llvm::StringRef mnemonic = op->getName().getStringRef();
  if (mnemonic.contains("buffer_") && srdSize != 4) {
    return op.emitOpError()
           << "buffer load requires 4-SGPR SRD but got " << srdSize << " SGPRs";
  }

  // Verify instruction offset is within valid range (0-4095 for AMDGCN)
  int64_t instOffset = op.getInstOffset();
  if (instOffset < 0 || instOffset > 4095) {
    return op.emitOpError() << "instruction offset " << instOffset
                            << " is out of valid range [0, 4095]";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VMEM Store Operation Verifiers
//===----------------------------------------------------------------------===//

/// Verifier for VMEM store operations - validates SRD size.
template <typename VMEMStoreOpType>
static LogicalResult verifyVMEMStoreOp(VMEMStoreOpType op) {
  // Verify SRD (saddr) is a 4-SGPR quad for buffer operations
  Value saddr = op.getSaddr();
  Type saddrType = saddr.getType();

  int64_t srdSize = 1;
  if (auto sregType = dyn_cast<SRegType>(saddrType)) {
    srdSize = sregType.getSize();
  } else if (auto psregType = dyn_cast<PSRegType>(saddrType)) {
    srdSize = psregType.getSize();
  }

  // Buffer operations require a 4-SGPR SRD
  llvm::StringRef mnemonic = op->getName().getStringRef();
  if (mnemonic.contains("buffer_") && srdSize != 4) {
    return op.emitOpError() << "buffer store requires 4-SGPR SRD but got "
                            << srdSize << " SGPRs";
  }

  // Verify instruction offset is within valid range (0-4095 for AMDGCN)
  int64_t instOffset = op.getInstOffset();
  if (instOffset < 0 || instOffset > 4095) {
    return op.emitOpError() << "instruction offset " << instOffset
                            << " is out of valid range [0, 4095]";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Explicit Verifier Definitions for MFMA Operations
//===----------------------------------------------------------------------===//

// F32 output, F16 input
LogicalResult V_MFMA_F32_16X16X16_F16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_16X16X32_F16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_32X32X8_F16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_16X16X4_F16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_32X32X4_F16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_4X4X4_F16::verify() { return verifyMFMAOp(*this); }

// F32 output, BF16 input
LogicalResult V_MFMA_F32_16X16X16_BF16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_16X16X32_BF16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_32X32X8_BF16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_16X16X4_BF16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_32X32X4_BF16::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_4X4X4_BF16::verify() { return verifyMFMAOp(*this); }

// I32 output, I8 input
LogicalResult V_MFMA_I32_16X16X16_I8::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_I32_32X32X8_I8::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_I32_16X16X4_I8::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_I32_32X32X4_I8::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_I32_4X4X4_I8::verify() { return verifyMFMAOp(*this); }

// F32 output, F32 input
LogicalResult V_MFMA_F32_16X16X4_F32::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_32X32X2_F32::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F32_4X4X1_F32::verify() { return verifyMFMAOp(*this); }

// F64 output, F64 input
LogicalResult V_MFMA_F64_16X16X4_F64::verify() { return verifyMFMAOp(*this); }
LogicalResult V_MFMA_F64_4X4X4_F64::verify() { return verifyMFMAOp(*this); }

// FP8 variants
LogicalResult V_MFMA_F32_16X16X32_FP8_FP8::verify() {
  return verifyMFMAOp(*this);
}
LogicalResult V_MFMA_F32_32X32X16_FP8_FP8::verify() {
  return verifyMFMAOp(*this);
}
LogicalResult V_MFMA_F32_16X16X32_FP8_BF8::verify() {
  return verifyMFMAOp(*this);
}
LogicalResult V_MFMA_F32_32X32X16_FP8_BF8::verify() {
  return verifyMFMAOp(*this);
}
LogicalResult V_MFMA_F32_16X16X32_BF8_FP8::verify() {
  return verifyMFMAOp(*this);
}
LogicalResult V_MFMA_F32_32X32X16_BF8_FP8::verify() {
  return verifyMFMAOp(*this);
}
LogicalResult V_MFMA_F32_16X16X32_BF8_BF8::verify() {
  return verifyMFMAOp(*this);
}
LogicalResult V_MFMA_F32_32X32X16_BF8_BF8::verify() {
  return verifyMFMAOp(*this);
}

// Scaled MFMA variants (MXFP4/FP6/FP8 with per-group scales)
LogicalResult V_MFMA_SCALE_F32_16X16X128_F8F6F4::verify() {
  return verifyMFMAOp(*this);
}
LogicalResult V_MFMA_SCALE_F32_32X32X64_F8F6F4::verify() {
  return verifyMFMAOp(*this);
}

//===----------------------------------------------------------------------===//
// Explicit Verifier Definitions for VMEM Load Operations
//===----------------------------------------------------------------------===//

// Buffer loads
LogicalResult BUFFER_LOAD_DWORD::verify() { return verifyVMEMLoadOp(*this); }
LogicalResult BUFFER_LOAD_DWORDX2::verify() { return verifyVMEMLoadOp(*this); }
LogicalResult BUFFER_LOAD_DWORDX3::verify() { return verifyVMEMLoadOp(*this); }
LogicalResult BUFFER_LOAD_DWORDX4::verify() { return verifyVMEMLoadOp(*this); }
LogicalResult BUFFER_LOAD_UBYTE::verify() { return verifyVMEMLoadOp(*this); }
LogicalResult BUFFER_LOAD_SBYTE::verify() { return verifyVMEMLoadOp(*this); }
LogicalResult BUFFER_LOAD_USHORT::verify() { return verifyVMEMLoadOp(*this); }
LogicalResult BUFFER_LOAD_SSHORT::verify() { return verifyVMEMLoadOp(*this); }

// Global loads (no SRD requirement)
LogicalResult GLOBAL_LOAD_DWORD::verify() { return success(); }
LogicalResult GLOBAL_LOAD_DWORDX2::verify() { return success(); }
LogicalResult GLOBAL_LOAD_DWORDX3::verify() { return success(); }
LogicalResult GLOBAL_LOAD_DWORDX4::verify() { return success(); }
LogicalResult GLOBAL_LOAD_UBYTE::verify() { return success(); }
LogicalResult GLOBAL_LOAD_SBYTE::verify() { return success(); }
LogicalResult GLOBAL_LOAD_USHORT::verify() { return success(); }
LogicalResult GLOBAL_LOAD_SSHORT::verify() { return success(); }

// Flat loads (no SRD requirement)
LogicalResult FLAT_LOAD_DWORD::verify() { return success(); }
LogicalResult FLAT_LOAD_DWORDX2::verify() { return success(); }
LogicalResult FLAT_LOAD_DWORDX3::verify() { return success(); }
LogicalResult FLAT_LOAD_DWORDX4::verify() { return success(); }

//===----------------------------------------------------------------------===//
// Explicit Verifier Definitions for VMEM Store Operations
//===----------------------------------------------------------------------===//

// Buffer stores
LogicalResult BUFFER_STORE_DWORD::verify() { return verifyVMEMStoreOp(*this); }
LogicalResult BUFFER_STORE_DWORDX2::verify() {
  return verifyVMEMStoreOp(*this);
}
LogicalResult BUFFER_STORE_DWORDX3::verify() {
  return verifyVMEMStoreOp(*this);
}
LogicalResult BUFFER_STORE_DWORDX4::verify() {
  return verifyVMEMStoreOp(*this);
}
LogicalResult BUFFER_STORE_BYTE::verify() { return verifyVMEMStoreOp(*this); }
LogicalResult BUFFER_STORE_SHORT::verify() { return verifyVMEMStoreOp(*this); }

// Global stores (no SRD requirement)
LogicalResult GLOBAL_STORE_DWORD::verify() { return success(); }
LogicalResult GLOBAL_STORE_DWORDX2::verify() { return success(); }
LogicalResult GLOBAL_STORE_DWORDX3::verify() { return success(); }
LogicalResult GLOBAL_STORE_DWORDX4::verify() { return success(); }
LogicalResult GLOBAL_STORE_BYTE::verify() { return success(); }
LogicalResult GLOBAL_STORE_SHORT::verify() { return success(); }

// Flat stores (no SRD requirement)
LogicalResult FLAT_STORE_DWORD::verify() { return success(); }
LogicalResult FLAT_STORE_DWORDX2::verify() { return success(); }
LogicalResult FLAT_STORE_DWORDX3::verify() { return success(); }
LogicalResult FLAT_STORE_DWORDX4::verify() { return success(); }

//===----------------------------------------------------------------------===//
// Explicit Verifier Definitions for VMEM Atomic Operations
//===----------------------------------------------------------------------===//

LogicalResult BUFFER_ATOMIC_PK_ADD_BF16::verify() {
  return verifyVMEMStoreOp(*this);
}
LogicalResult BUFFER_ATOMIC_ADD_F32::verify() {
  return verifyVMEMStoreOp(*this);
}

//===----------------------------------------------------------------------===//
// TableGen'd Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "waveasm/Dialect/WaveASMOps.cpp.inc"
