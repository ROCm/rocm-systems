// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// AMDGCN Literal Materialization
//
// Handles immediate operands outside the inline constant range [-16, 64]
// for VALU instructions. VOP2 instructions require the literal in src0;
// VOP3+ instructions need a scratch VGPR via v_mov_b32. SALU instructions
// accept 32-bit literals natively.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Target/AMDGCN/RegisterInfo.h"
#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Transforms/RegAlloc.h"

#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace waveasm {

// AMDGCN inline constants: -16 to 64 (inclusive), plus special float values.
static bool isInlineConstant(int64_t val) {
  if (val >= -16 && val <= 64)
    return true;
  return false;
}

static const llvm::StringSet<> &getVOP2Instructions() {
  static const llvm::StringSet<> kVOP2 = {
      "v_add_u32",     "v_sub_u32", "v_subrev_u32",  "v_and_b32",
      "v_or_b32",      "v_xor_b32", "v_lshlrev_b32", "v_lshrrev_b32",
      "v_ashrrev_i32", "v_max_u32", "v_min_u32",     "v_add_i32",
      "v_sub_i32",
  };
  return kVOP2;
}

static bool needsLiteralMaterialization(llvm::StringRef mnemonic) {
  if (!mnemonic.starts_with("v_"))
    return false;
  if (mnemonic == "v_mov_b32")
    return false;
  if (mnemonic.starts_with("v_cmp_"))
    return false;
  // v_cndmask_b32 has a dedicated handler in AssemblyEmitter that drops
  // the condition operand and materializes non-inline literals as needed.
  if (mnemonic == "v_cndmask_b32")
    return false;
  if (getVOP2Instructions().count(mnemonic))
    return false;
  return true;
}

llvm::SmallVector<std::string>
KernelGenerator::generateOpWithLiteralHandling(Operation *op) {
  llvm::SmallVector<std::string> lines;

  llvm::StringRef opName = op->getName().getStringRef();
  llvm::StringRef mnemonic = opName;
  if (opName.starts_with("waveasm.")) {
    mnemonic = opName.drop_front(8);
  }

  bool hasNonInlineLiteral = false;
  int64_t literalValue = 0;
  int literalOperandIdx = -1;

  for (int i = 0; i < static_cast<int>(op->getNumOperands()); ++i) {
    auto [isLiteral, val] = getLiteralValue(op->getOperand(i));
    if (isLiteral && !isInlineConstant(val)) {
      hasNonInlineLiteral = true;
      literalValue = val;
      literalOperandIdx = i;
      break;
    }
  }

  if (!hasNonInlineLiteral) {
    if (auto line = generateOp(op)) {
      lines.push_back(*line);
    }
    return lines;
  }

  // SALU instructions support 32-bit literals natively
  if (mnemonic.starts_with("s_")) {
    if (auto line = generateOp(op)) {
      lines.push_back(*line);
    }
    return lines;
  }

  // Carry ops have dedicated handlers in generateOp that emit VCC
  // operands and materialize literals as needed.
  if (isa<V_ADD_CO_U32, V_SUB_CO_U32, V_ADDC_CO_U32, V_SUBB_CO_U32>(op)) {
    if (auto line = generateOp(op))
      lines.push_back(*line);
    return lines;
  }

  // VOP3+ instructions need literal materialization into scratch VGPR.
  if (needsLiteralMaterialization(mnemonic)) {
    emitMaterializedLiteral(lines, op, mnemonic, literalOperandIdx,
                            literalValue);
    return lines;
  }

  // v_cndmask_b32 has a dedicated Case handler in generateOp that
  // materializes literals AND drops the implicit VCC condition operand.
  if (mnemonic == "v_cndmask_b32") {
    if (auto line = generateOp(op)) {
      lines.push_back(*line);
    }
    return lines;
  }

  // VOP2: literal MUST be in src0. Swap for commutative ops, materialize
  // otherwise.
  bool isCommutative = op->hasTrait<mlir::OpTrait::IsCommutative>();

  if (literalOperandIdx == 0) {
    if (auto line = generateOp(op)) {
      lines.push_back(*line);
    }
    return lines;
  }

  if (literalOperandIdx == 1 && isCommutative && op->getNumOperands() == 2) {
    llvm::SmallVector<std::string> operands;
    for (Value result : op->getResults()) {
      operands.push_back(resolveValue(result));
    }
    operands.push_back(std::to_string(literalValue));
    operands.push_back(resolveValue(op->getOperand(0)));
    lines.push_back(formatter.format(mnemonic, operands));
    return lines;
  }

  emitMaterializedLiteral(lines, op, mnemonic, literalOperandIdx, literalValue);
  return lines;
}

void KernelGenerator::emitMaterializedLiteral(
    llvm::SmallVector<std::string> &lines, Operation *op,
    llvm::StringRef mnemonic, int literalOperandIdx, int64_t literalValue) {
  std::string scratchReg = formatVGPRRange(kScratchVGPR, 1);
  if (scratchVGPRValue != literalValue) {
    lines.push_back("  v_mov_b32 " + scratchReg + ", " +
                    std::to_string(literalValue));
    scratchVGPRValue = literalValue;
  }

  llvm::SmallVector<std::string> operands;
  for (Value result : op->getResults()) {
    operands.push_back(resolveValue(result));
  }
  for (int i = 0; i < static_cast<int>(op->getNumOperands()); ++i) {
    if (i == literalOperandIdx) {
      operands.push_back(scratchReg);
    } else {
      operands.push_back(resolveValue(op->getOperand(i)));
    }
  }

  lines.push_back(formatter.format(mnemonic, operands));
  peakVGPRs = std::max(peakVGPRs, kScratchVGPR + 1);
}

} // namespace waveasm
