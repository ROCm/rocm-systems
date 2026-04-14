// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TARGET_AMDGCN_INSTRUCTIONINFO_H
#define WaveASM_TARGET_AMDGCN_INSTRUCTIONINFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Instruction Categories
//===----------------------------------------------------------------------===//

enum class InstrCategory : int32_t {
  Unknown = 0,
  VectorALU, // VALU instructions
  ScalarALU, // SALU instructions
  VectorMem, // VMEM instructions (buffer, global, flat)
  ScalarMem, // SMEM instructions
  LDS,       // LDS (local data share) instructions
  Export,    // Export instructions
  MFMA,      // Matrix fused multiply-add
  Branch,    // Branch/control flow
  Misc,      // Miscellaneous (nop, waitcnt, etc.)
};

//===----------------------------------------------------------------------===//
// Operand Types
//===----------------------------------------------------------------------===//

enum class OperandType : int32_t {
  None = 0,
  VGPR,         // Vector register
  VGPR_Pair,    // 2 consecutive VGPRs
  VGPR_Quad,    // 4 consecutive VGPRs
  VGPR_Range,   // N consecutive VGPRs
  SGPR,         // Scalar register
  SGPR_Pair,    // 2 consecutive SGPRs (typically 64-bit)
  SGPR_Quad,    // 4 consecutive SGPRs
  SGPR_Range,   // N consecutive SGPRs
  AGPR,         // Accumulator register
  AGPR_Range,   // N consecutive AGPRs
  VCC,          // Vector condition code
  EXEC,         // Execution mask
  SCC,          // Scalar condition code
  M0,           // M0 special register
  Immediate,    // Immediate constant
  LiteralConst, // Literal constant (32-bit)
  Label,        // Branch target label
  Offset,       // Memory offset
};

//===----------------------------------------------------------------------===//
// Operand Descriptor
//===----------------------------------------------------------------------===//

/// Describes an instruction operand
struct OperandDesc {
  std::string name;        // Operand name (for assembly)
  OperandType type;        // Operand type
  int64_t size = 1;        // Size in registers (for ranges)
  int64_t alignment = 1;   // Alignment requirement
  bool isOptional = false; // Is this operand optional?
  bool isDef = false;      // Is this a definition (output)?
  bool isUse = true;       // Is this a use (input)?

  /// For tied operands: index of the operand this is tied to
  std::optional<int64_t> tiedTo = std::nullopt;
};

//===----------------------------------------------------------------------===//
// Instruction Descriptor
//===----------------------------------------------------------------------===//

/// Describes an instruction's properties
struct InstrDesc {
  std::string name; // Instruction name
  InstrCategory category = InstrCategory::Unknown;
  llvm::SmallVector<OperandDesc> defs; // Definition operands
  llvm::SmallVector<OperandDesc> uses; // Use operands

  // Latency and scheduling
  int64_t latency = 1;    // Execution latency in cycles
  int64_t issueSlots = 1; // Number of issue slots consumed

  // Flags
  bool mayLoad = false;
  bool mayStore = false;
  bool hasSideEffects = false;
  bool isBarrier = false;
  bool isBranch = false;
  bool isConditionalBranch = false;
  bool isTerminator = false;
  bool hasDelaySlot = false;

  // Wait count effects
  bool incrementsVmcnt = false;   // Memory ops
  bool incrementsLgkmcnt = false; // LDS/SMEM ops
  bool incrementsExpcnt = false;  // Export ops

  // Target support (bitmask of targets that support this instruction)
  uint32_t targetMask = 0xFFFFFFFF; // All targets by default

  /// Check if this instruction has a definition
  bool hasDefs() const { return !defs.empty(); }

  /// Get total number of operands
  size_t getNumOperands() const { return defs.size() + uses.size(); }
};

//===----------------------------------------------------------------------===//
// Instruction Registry
//===----------------------------------------------------------------------===//

/// Registry of instruction information
class InstructionRegistry {
public:
  InstructionRegistry();
  ~InstructionRegistry();

  /// Get instruction descriptor by name
  const InstrDesc *getInstrDesc(llvm::StringRef name) const;

  /// Check if an instruction is registered
  bool hasInstruction(llvm::StringRef name) const;

  /// Get all registered instruction names
  llvm::SmallVector<llvm::StringRef> getInstructionNames() const;

  /// Get instructions by category
  llvm::SmallVector<llvm::StringRef>
  getInstructionsByCategory(InstrCategory category) const;

  /// Register an instruction (used during initialization)
  void registerInstruction(const InstrDesc &desc);

  /// Get the global instruction registry
  static InstructionRegistry &get();

private:
  llvm::StringMap<InstrDesc> instructions;
};

//===----------------------------------------------------------------------===//
// Instruction Format Helpers
//===----------------------------------------------------------------------===//

/// Format a register name
std::string formatRegister(OperandType type, int64_t baseIndex,
                           int64_t size = 1);

/// Format an immediate value
std::string formatImmediate(int64_t value);

/// Format a branch target
std::string formatLabel(llvm::StringRef label);

/// Parse a register string to get type and index
std::optional<std::pair<OperandType, int64_t>>
parseRegister(llvm::StringRef regStr);

} // namespace waveasm

#endif // WaveASM_TARGET_AMDGCN_INSTRUCTIONINFO_H
