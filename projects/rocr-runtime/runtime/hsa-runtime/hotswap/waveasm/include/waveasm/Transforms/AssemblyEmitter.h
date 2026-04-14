// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_ASSEMBLYEMITTER_H
#define WaveASM_TRANSFORMS_ASSEMBLYEMITTER_H

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/RegAlloc.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace waveasm {

/// Return the kernel name for assembly emission.
/// Uses the "kernel_name" attribute if present, otherwise falls back to
/// sym_name. The kernel_name attribute is set when the program is given a
/// mangled sym_name to avoid symbol collisions with the original llvm.func.
inline mlir::StringRef getKernelName(ProgramOp program) {
  auto attrName = WaveASMDialect::getKernelNameAttrName();
  if (auto attr = program->getAttrOfType<mlir::StringAttr>(attrName))
    return attr.getValue();
  return program.getSymName();
}

//===----------------------------------------------------------------------===//
// Instruction Formatter
//===----------------------------------------------------------------------===//

/// Formats instructions for assembly output
class InstructionFormatter {
public:
  InstructionFormatter() = default;

  /// Format an instruction with operands
  std::string format(llvm::StringRef name,
                     llvm::ArrayRef<std::string> operands);

  /// Format a label
  std::string formatLabel(llvm::StringRef label);

  /// Format a comment
  std::string formatComment(llvm::StringRef text);

  /// Format s_waitcnt instruction
  std::string formatWaitcnt(std::optional<int64_t> vmcnt,
                            std::optional<int64_t> lgkmcnt,
                            std::optional<int64_t> expcnt = std::nullopt);

  /// Format s_barrier instruction
  std::string formatBarrier();

  /// Format s_endpgm instruction
  std::string formatEndpgm();

  /// Format a raw assembly line (passthrough)
  std::string formatRaw(llvm::StringRef text);
};

//===----------------------------------------------------------------------===//
// Kernel Metadata Emitter
//===----------------------------------------------------------------------===//

/// Emits kernel metadata for AMDGPU
class MetadataEmitter {
public:
  MetadataEmitter(ProgramOp program, TargetAttrInterface target);

  /// Emit assembly prologue (target directive, .text, kernel directive)
  llvm::SmallVector<std::string> emitPrologue();

  /// Emit assembly epilogue (kernel symbol info, metadata YAML)
  llvm::SmallVector<std::string> emitEpilogue(int64_t peakVGPRs,
                                              int64_t peakSGPRs,
                                              int64_t peakAGPRs,
                                              int64_t ldsSize);

private:
  /// Emit .amdhsa_kernel directive and its fields
  llvm::SmallVector<std::string> emitKernelDescriptor(int64_t peakVGPRs,
                                                      int64_t peakSGPRs,
                                                      int64_t peakAGPRs,
                                                      int64_t ldsSize);

  /// Emit .amdgpu_metadata section
  llvm::SmallVector<std::string> emitMetadataYAML(int64_t peakVGPRs,
                                                  int64_t peakSGPRs,
                                                  int64_t peakAGPRs,
                                                  int64_t ldsSize);

  ProgramOp program;
  TargetAttrInterface target;
};

//===----------------------------------------------------------------------===//
// Kernel Generator (Pure SSA)
//===----------------------------------------------------------------------===//

/// Generates assembly code for a kernel with pure SSA ops
class KernelGenerator {
public:
  KernelGenerator(ProgramOp program, const PhysicalMapping &mapping,
                  TargetAttrInterface target);

  /// Generate complete assembly output
  llvm::SmallVector<std::string> generate();

  /// Get peak VGPR usage from allocation
  int64_t getPeakVGPRs() const { return peakVGPRs; }

  /// Get peak SGPR usage from allocation
  int64_t getPeakSGPRs() const { return peakSGPRs; }

  /// Get peak AGPR usage from allocation
  int64_t getPeakAGPRs() const { return peakAGPRs; }

private:
  /// Scratch VGPR for loading non-inline literals (v15).
  /// VGPRCompaction has a matching constant; keep them in sync.
  static constexpr int64_t kScratchVGPR = 15;
  /// Resolve an SSA Value to its physical register string
  std::string resolveValue(mlir::Value value);
  std::string resolveScalarValue(mlir::Value value);

  /// Get the literal value if the operand is an immediate constant
  /// Returns (isLiteral, value) pair
  std::pair<bool, int64_t> getLiteralValue(mlir::Value value);

  /// Materialize a non-inline literal operand into a scratch VGPR.
  /// Returns the resolved operand string and any prefix v_mov_b32
  /// instructions needed. If the operand is inline or a register, returns
  /// it directly with an empty prefix.
  struct MaterializedOperand {
    std::string operandStr;
    std::string prefix;
  };
  MaterializedOperand materializeLiteralOperand(mlir::Value operand,
                                                int scratchIdx);

  /// Generate code for a single operation using TypeSwitch dispatch
  /// Returns (instruction_lines, needs_literal_load)
  std::optional<std::string> generateOp(mlir::Operation *op);

  /// Generate instructions with literal handling
  /// For VOP3 ops that don't support literals, emit s_mov_b32 first
  llvm::SmallVector<std::string>
  generateOpWithLiteralHandling(mlir::Operation *op);

  /// Emit a v_mov_b32 to materialize a literal into scratch VGPR, then
  /// emit the instruction with the scratch register replacing the literal.
  void emitMaterializedLiteral(llvm::SmallVector<std::string> &lines,
                               mlir::Operation *op, llvm::StringRef mnemonic,
                               int literalOperandIdx, int64_t literalValue);

  /// Generate code for a label op
  std::string generateLabel(LabelOp labelOp);

  /// Generate code for a comment op
  std::string generateComment(CommentOp commentOp);

  /// Generate code for a raw op
  std::string generateRaw(RawOp rawOp);

  //===--------------------------------------------------------------------===//
  // Helper methods for TypeSwitch-based code generation
  //===--------------------------------------------------------------------===//

  /// Emit buffer load instruction (buffer_load_dword, etc.)
  std::string emitBufferLoad(mlir::Operation *op, llvm::StringRef mnemonic);

  /// Emit buffer load to LDS (gather-to-LDS) instruction
  std::string emitBufferLoadLDS(mlir::Operation *op, llvm::StringRef mnemonic);

  /// Emit buffer store instruction (buffer_store_dword, etc.)
  std::string emitBufferStore(mlir::Operation *op, llvm::StringRef mnemonic);

  /// Emit global load instruction (global_load_dword, etc.)
  std::string emitGlobalLoad(mlir::Operation *op, llvm::StringRef mnemonic);

  /// Emit global store instruction (global_store_dword, etc.)
  std::string emitGlobalStore(mlir::Operation *op, llvm::StringRef mnemonic);

  /// Emit LDS read instruction (ds_read_b32, etc.)
  std::string emitLDSRead(mlir::Operation *op, llvm::StringRef mnemonic);

  /// Emit LDS write instruction (ds_write_b32, etc.)
  std::string emitLDSWrite(mlir::Operation *op, llvm::StringRef mnemonic);

  /// Emit default instruction format (results then operands)
  std::string emitDefaultFormat(mlir::Operation *op, llvm::StringRef mnemonic);

  /// Emit scaled MFMA instruction with cbsz/blgp format modifiers
  std::optional<std::string> emitScaledMFMA(mlir::Operation *op,
                                            llvm::StringRef mnemonic);

  ProgramOp program;
  const PhysicalMapping &mapping;
  TargetAttrInterface target;
  InstructionFormatter formatter;

  int64_t peakVGPRs = 0;
  int64_t peakSGPRs = 0;
  int64_t peakAGPRs = 0;

  /// Counter for generating unique loop labels in assembly
  int loopLabelCounter = 0;

  /// Cached value currently held in kScratchVGPR, used to avoid redundant
  /// v_mov_b32 materializations of the same literal.
  std::optional<int64_t> scratchVGPRValue;

  /// Invalidate the scratch VGPR cache (call on control flow changes or
  /// when kScratchVGPR is written with a non-literal value).
  void invalidateScratchCache() { scratchVGPRValue = std::nullopt; }
};

//===----------------------------------------------------------------------===//
// Assembly Output
//===----------------------------------------------------------------------===//

/// Write assembly to a file
mlir::LogicalResult writeAssembly(ProgramOp program,
                                  const PhysicalMapping &mapping,
                                  llvm::StringRef outputPath);

/// Write assembly to a stream
mlir::LogicalResult writeAssembly(ProgramOp program,
                                  const PhysicalMapping &mapping,
                                  llvm::raw_ostream &os);

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_ASSEMBLYEMITTER_H
