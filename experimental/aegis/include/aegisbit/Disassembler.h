//===-- aegisbit/Disassembler.h - AMDGPU Disassembler ----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// AMDGPU instruction disassembler and encoder using LLVM MC layer.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_DISASSEMBLER_H
#define AEGISBIT_DISASSEMBLER_H

#include "aegisbit/Types.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace llvm {
class Target;
} // namespace llvm

namespace aegisbit {

/// AMDGPU instruction disassembler and encoder
class Disassembler {
public:
  /// Create disassembler for specified GPU target
  /// \param TargetTriple Triple string (e.g., "amdgcn-amd-amdhsa")
  /// \param CPU CPU name (e.g., "gfx908", "gfx90a", "gfx942")
  /// \param Features Feature string (e.g., "+wavefrontsize64")
  static llvm::Expected<std::unique_ptr<Disassembler>>
  create(const std::string& TargetTriple = "amdgcn-amd-amdhsa",
         const std::string& CPU = "gfx942",
         const std::string& Features = "+wavefrontsize64");

  /// Disassemble a single instruction from bytes
  /// \param Bytes Input byte array
  /// \param Address PC address of instruction
  /// \param Size Output: size of instruction in bytes
  /// \return Decoded instruction or error
  llvm::Expected<DecodedInstruction> disassemble(llvm::ArrayRef<uint8_t> Bytes,
                                                  uint64_t Address,
                                                  uint64_t& Size);

  /// Disassemble entire code section
  /// \param Bytes Code section bytes
  /// \param BaseAddress Starting PC address
  /// \return Vector of decoded instructions or error
  llvm::Expected<std::vector<DecodedInstruction>>
  disassembleAll(llvm::ArrayRef<uint8_t> Bytes, uint64_t BaseAddress = 0);

  /// Encode MCInst back to bytes
  /// \param Inst Instruction to encode
  /// \return Encoded bytes or error
  llvm::Expected<std::vector<uint8_t>> encode(const llvm::MCInst& Inst);

  /// Categorize instruction into VALU/SALU/VMEM/etc.
  InstructionCategory categorize(const llvm::MCInst& Inst) const;

  /// Check if instruction is a branch (includes calls, returns, endpgm)
  bool isBranch(const llvm::MCInst& Inst) const;

  /// Check if instruction is a PC-relative branch (s_branch, s_cbranch_*)
  /// Returns false for s_endpgm, s_setpc, s_call, and other non-branch
  /// terminators.
  bool isPCRelativeBranch(const llvm::MCInst& Inst) const;

  /// Check if instruction is a memory operation
  bool isMemory(const llvm::MCInst& Inst) const;

  /// Get branch target offset for branch instructions
  /// \param Inst Branch instruction
  /// \param CurrentPC Current program counter
  /// \return Target address or error if not a branch
  llvm::Expected<int64_t> getBranchTarget(const llvm::MCInst& Inst,
                                          uint64_t CurrentPC) const;

  /// Get instruction name as string (LLVM internal opcode name)
  std::string getInstructionName(const llvm::MCInst& Inst) const;

  /// Get the assembly mnemonic for an instruction (valid for assembler input)
  std::string getAsmMnemonic(const llvm::MCInst& Inst) const;

  /// Print instruction to string (for debugging)
  std::string printInstruction(const llvm::MCInst& Inst) const;

  /// Get MCInstrInfo for direct access to instruction metadata
  const llvm::MCInstrInfo& getMCII() const { return *MCII; }

  /// Get MCRegisterInfo for register metadata
  const llvm::MCRegisterInfo& getMRI() const { return *MRI; }

  /// Get MCSubtargetInfo for subtarget features
  const llvm::MCSubtargetInfo& getSTI() const { return *STI; }

  /// Get MCAsmInfo for assembly syntax info
  const llvm::MCAsmInfo& getMAI() const { return *MAI; }

  /// Get MCContext for symbol/section management
  llvm::MCContext& getContext() { return *Ctx; }
  const llvm::MCContext& getContext() const { return *Ctx; }

  /// Get the LLVM Target (for creating MCStreamer, MCAsmBackend, etc.)
  const llvm::Target& getTarget() const { return *TheTarget; }

  /// Get MCCodeEmitter (for direct encoding)
  const llvm::MCCodeEmitter& getEmitter() const { return *Emitter; }

private:
  Disassembler(std::unique_ptr<llvm::MCContext> Ctx,
               std::unique_ptr<llvm::MCDisassembler> Disasm,
               std::unique_ptr<llvm::MCCodeEmitter> Emitter,
               std::unique_ptr<llvm::MCInstrInfo> MCII,
               std::unique_ptr<llvm::MCRegisterInfo> MRI,
               std::unique_ptr<llvm::MCSubtargetInfo> STI,
               std::unique_ptr<llvm::MCAsmInfo> MAI,
               std::unique_ptr<llvm::MCInstPrinter> Printer,
               const llvm::Target* TheTarget);

  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCDisassembler> DisasmImpl;
  std::unique_ptr<llvm::MCCodeEmitter> Emitter;
  std::unique_ptr<llvm::MCInstrInfo> MCII;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstPrinter> Printer;
  const llvm::Target* TheTarget;
};

} // namespace aegisbit

#endif // AEGISBIT_DISASSEMBLER_H
