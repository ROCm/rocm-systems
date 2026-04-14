////////////////////////////////////////////////////////////////////////////////
//
// Binary Lifter: AMDGPU machine code (MCInst) -> waveasm MLIR ops
//
// Takes raw instruction bytes from an ELF .text section, disassembles them
// using LLVM MC, and lifts each instruction into the waveasm MLIR dialect
// with physical register types (pre-SSA).
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ROCR_HOTSWAP_LIFTER_HPP
#define ROCR_HOTSWAP_LIFTER_HPP

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Target/AMDGCN/InstructionInfo.h"

#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInst.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include <map>
#include <string>
#include <vector>

namespace hotswap {

struct LiftStats {
  uint64_t totalInstructions = 0;
  uint64_t liftedInstructions = 0;
  uint64_t rawFallbacks = 0;
  uint64_t failedDisassembly = 0;
};

/// Holds initialized LLVM MC state for a specific target ISA.
struct MCState {
  const llvm::Target *target = nullptr;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<const llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstrInfo> MCII;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCDisassembler> disasm;
  std::unique_ptr<llvm::MCInstPrinter> printer;
  std::string cpu;
  bool valid = false;
};

/// Initialize MC state for an AMDGPU ISA (e.g., "gfx1250", "gfx942").
MCState initMCState(llvm::StringRef isaName);

/// Lifts AMDGPU machine code bytes into a waveasm.program MLIR operation.
class Lifter {
public:
  explicit Lifter(mlir::MLIRContext &ctx);

  /// Lift raw instruction bytes for the given ISA into a waveasm.program.
  /// Returns the containing ModuleOp on success, or nullptr on failure.
  mlir::OwningOpRef<mlir::ModuleOp>
  lift(llvm::ArrayRef<uint8_t> bytes, llvm::StringRef isaName,
       llvm::StringRef kernelName = "kernel");

  const LiftStats &getStats() const { return stats; }

private:
  /// Lift a single MCInst into the current insertion block.
  /// Returns true if the instruction was lifted to a typed waveasm op,
  /// false if it fell back to waveasm.raw.
  bool liftInstruction(const llvm::MCInst &inst, uint64_t pc,
                       const MCState &mc, mlir::OpBuilder &builder,
                       mlir::Location loc);

  /// Classify an LLVM MC register into a waveasm physical register type.
  mlir::Type classifyRegister(unsigned reg, const MCState &mc) const;

  /// Get the waveasm type for an MCInst operand.
  mlir::Type getOperandType(const llvm::MCOperand &op, const MCState &mc) const;

  /// Get the mnemonic from an MCInst via the printer.
  std::string getMnemonic(const llvm::MCInst &inst, const MCState &mc) const;

  /// Create a precolored register value for a physical register.
  mlir::Value getOrCreatePhysReg(unsigned reg, const MCState &mc,
                                 mlir::OpBuilder &builder, mlir::Location loc);

  /// Create a constant op for an immediate value.
  mlir::Value createImmediate(int64_t value, mlir::OpBuilder &builder,
                              mlir::Location loc);

  /// Dispatch to typed waveasm op creation based on mnemonic.
  /// Returns true if handled, false for fallback.
  bool dispatchToWaveasmOp(llvm::StringRef mnemonic, const llvm::MCInst &inst,
                           uint64_t pc, const MCState &mc,
                           mlir::OpBuilder &builder, mlir::Location loc);

  /// Create a waveasm op dynamically using OperationState.
  mlir::Operation *
  createWaveasmOp(llvm::StringRef opName, mlir::TypeRange resultTypes,
                  mlir::ValueRange operands, mlir::OpBuilder &builder,
                  mlir::Location loc,
                  llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

  mlir::MLIRContext &ctx;
  LiftStats stats;
  llvm::DenseMap<unsigned, mlir::Value> physRegCache;
  std::map<uint64_t, std::string> branchLabels;
};

} // namespace hotswap

#endif // ROCR_HOTSWAP_LIFTER_HPP
