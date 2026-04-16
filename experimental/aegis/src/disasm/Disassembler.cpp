//===-- Disassembler.cpp - AMDGPU Disassembler Implementation --*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/Disassembler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include <sstream>

// Forward declarations for LLVM initialization functions
extern "C" {
void LLVMInitializeAMDGPUTargetInfo();
void LLVMInitializeAMDGPUTarget();
void LLVMInitializeAMDGPUTargetMC();
void LLVMInitializeAMDGPUDisassembler();
}

namespace aegisbit {

llvm::Expected<std::unique_ptr<Disassembler>>
Disassembler::create(const std::string& TargetTriple,
                     const std::string& CPU,
                     const std::string& Features) {
  // Initialize AMDGPU target
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();

  // Get the target
  llvm::Triple TT(TargetTriple);
  std::string Error;
  const llvm::Target* TheTarget =
      llvm::TargetRegistry::lookupTarget(TT, Error);
  if (!TheTarget) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to lookup target: " + Error);
  }

  // Create MC components
  auto MRI = std::unique_ptr<llvm::MCRegisterInfo>(
      TheTarget->createMCRegInfo(TT));
  if (!MRI) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to create MCRegisterInfo");
  }

  llvm::MCTargetOptions MCOptions;
  auto MAI = std::unique_ptr<llvm::MCAsmInfo>(
      TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
  if (!MAI) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to create MCAsmInfo");
  }

  auto MCII = std::unique_ptr<llvm::MCInstrInfo>(TheTarget->createMCInstrInfo());
  if (!MCII) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to create MCInstrInfo");
  }

  auto STI = std::unique_ptr<llvm::MCSubtargetInfo>(
      TheTarget->createMCSubtargetInfo(TT, CPU, Features));
  if (!STI) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to create MCSubtargetInfo");
  }

  auto Ctx = std::make_unique<llvm::MCContext>(TT, MAI.get(), MRI.get(), STI.get());

  // Create disassembler
  auto DisasmImpl = std::unique_ptr<llvm::MCDisassembler>(
      TheTarget->createMCDisassembler(*STI, *Ctx));
  if (!DisasmImpl) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to create MCDisassembler");
  }

  // Create code emitter
  auto Emitter = std::unique_ptr<llvm::MCCodeEmitter>(
      TheTarget->createMCCodeEmitter(*MCII, *Ctx));
  if (!Emitter) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to create MCCodeEmitter");
  }

  // Create instruction printer
  auto Printer = std::unique_ptr<llvm::MCInstPrinter>(
      TheTarget->createMCInstPrinter(TT, 0, *MAI, *MCII, *MRI));
  if (!Printer) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to create MCInstPrinter");
  }

  return std::unique_ptr<Disassembler>(new Disassembler(
      std::move(Ctx), std::move(DisasmImpl), std::move(Emitter),
      std::move(MCII), std::move(MRI), std::move(STI), std::move(MAI),
      std::move(Printer), TheTarget));
}

Disassembler::Disassembler(std::unique_ptr<llvm::MCContext> Ctx,
                           std::unique_ptr<llvm::MCDisassembler> Disasm,
                           std::unique_ptr<llvm::MCCodeEmitter> Emitter,
                           std::unique_ptr<llvm::MCInstrInfo> MCII,
                           std::unique_ptr<llvm::MCRegisterInfo> MRI,
                           std::unique_ptr<llvm::MCSubtargetInfo> STI,
                           std::unique_ptr<llvm::MCAsmInfo> MAI,
                           std::unique_ptr<llvm::MCInstPrinter> Printer,
                           const llvm::Target* TheTarget)
    : Ctx(std::move(Ctx)), DisasmImpl(std::move(Disasm)),
      Emitter(std::move(Emitter)), MCII(std::move(MCII)), MRI(std::move(MRI)),
      STI(std::move(STI)), MAI(std::move(MAI)), Printer(std::move(Printer)),
      TheTarget(TheTarget) {}

llvm::Expected<DecodedInstruction>
Disassembler::disassemble(llvm::ArrayRef<uint8_t> Bytes, uint64_t Address,
                          uint64_t& Size) {
  llvm::MCInst Inst;
  llvm::MCDisassembler::DecodeStatus Status =
      DisasmImpl->getInstruction(Inst, Size, Bytes, Address, llvm::nulls());

  if (Status != llvm::MCDisassembler::Success) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Failed to disassemble instruction at address 0x" +
            llvm::Twine::utohexstr(Address));
  }

  DecodedInstruction DI;
  DI.Inst = Inst;
  DI.Address = Address;
  DI.Size = Size;
  DI.Category = categorize(Inst);

  return DI;
}

llvm::Expected<std::vector<DecodedInstruction>>
Disassembler::disassembleAll(llvm::ArrayRef<uint8_t> Bytes,
                             uint64_t BaseAddress) {
  std::vector<DecodedInstruction> Instructions;
  uint64_t Offset = 0;

  while (Offset < Bytes.size()) {
    uint64_t Size = 0;
    llvm::ArrayRef<uint8_t> Remaining = Bytes.slice(Offset);
    auto DI = disassemble(Remaining, BaseAddress + Offset, Size);

    if (!DI) {
      return DI.takeError();
    }

    Instructions.push_back(std::move(*DI));
    Offset += Size;
  }

  return Instructions;
}

llvm::Expected<std::vector<uint8_t>>
Disassembler::encode(const llvm::MCInst& Inst) {
  llvm::SmallVector<char, 16> Code;
  llvm::SmallVector<llvm::MCFixup, 4> Fixups;

  Emitter->encodeInstruction(Inst, Code, Fixups, *STI);

  if (!Fixups.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Instruction encoding produced fixups (relocations not supported)");
  }

  // Explicitly copy to avoid iterator aliasing bug when SmallVector uses inline storage
  std::vector<uint8_t> Result;
  Result.reserve(Code.size());
  for (char C : Code) {
    Result.push_back(static_cast<uint8_t>(C));
  }
  return Result;
}

InstructionCategory Disassembler::categorize(const llvm::MCInst& Inst) const {
  const llvm::MCInstrDesc& Desc = MCII->get(Inst.getOpcode());
  std::string Name = getInstructionName(Inst);

  // Use MCInstrDesc flags for reliable classification
  // These are stable LLVM APIs, not internal bit positions

  // 1. Check for branches first using MCInstrDesc
  if (Desc.isBranch() || Desc.isCall() || Desc.isReturn()) {
    return InstructionCategory::BRANCH;
  }

  // 2. Check for barriers and sync instructions by name
  if (Name.find("S_BARRIER") != std::string::npos ||
      Name.find("S_WAITCNT") != std::string::npos ||
      Name.find("S_WAIT_") != std::string::npos) {
    return InstructionCategory::BARRIER;
  }

  // 3. Check for MFMA (matrix operations)
  if (Name.find("V_MFMA_") == 0 || Name.find("V_SMFMAC_") == 0) {
    return InstructionCategory::MFMA;
  }

  // 4. Memory operations - check by instruction name prefix
  // Vector memory: BUFFER_, GLOBAL_, FLAT_, SCRATCH_
  if (Name.find("BUFFER_") == 0 || Name.find("GLOBAL_") == 0 ||
      Name.find("FLAT_") == 0 || Name.find("SCRATCH_") == 0 ||
      Name.find("TBUFFER_") == 0) {
    return InstructionCategory::VMEM;
  }

  // Scalar memory: S_LOAD_, S_STORE_, S_BUFFER_LOAD_
  if (Name.find("S_LOAD_") == 0 || Name.find("S_STORE_") == 0 ||
      Name.find("S_BUFFER_LOAD_") == 0 || Name.find("S_DCACHE_") == 0 ||
      Name.find("S_MEMTIME") == 0 || Name.find("S_MEMREALTIME") == 0) {
    return InstructionCategory::SMEM;
  }

  // LDS/GDS operations: DS_
  if (Name.find("DS_") == 0) {
    return InstructionCategory::LDS;
  }

  // 5. ALU operations - distinguish scalar vs vector by prefix
  // Scalar ALU: S_ prefix (but not S_LOAD, S_STORE, S_BUFFER, S_BARRIER, etc.)
  if (Name.size() >= 2 && Name[0] == 'S' && Name[1] == '_') {
    // Already handled: S_LOAD, S_STORE, S_BUFFER, S_BARRIER, S_WAIT, S_DCACHE
    // Remaining S_ instructions are scalar ALU
    return InstructionCategory::SALU;
  }

  // Vector ALU: V_ prefix (but not V_MFMA which is already handled)
  if (Name.size() >= 2 && Name[0] == 'V' && Name[1] == '_') {
    return InstructionCategory::VALU;
  }

  // Image operations are VMEM
  if (Name.find("IMAGE_") == 0) {
    return InstructionCategory::VMEM;
  }

  return InstructionCategory::OTHER;
}

bool Disassembler::isBranch(const llvm::MCInst& Inst) const {
  const llvm::MCInstrDesc& Desc = MCII->get(Inst.getOpcode());
  return Desc.isBranch() || Desc.isCall() || Desc.isReturn();
}

bool Disassembler::isPCRelativeBranch(const llvm::MCInst& Inst) const {
  const llvm::MCInstrDesc& Desc = MCII->get(Inst.getOpcode());
  // Only actual PC-relative branches (s_branch, s_cbranch_*), not
  // endpgm/return/call/setpc which are terminators without relocatable
  // PC-relative targets.
  // NOTE: Do NOT check isBarrier() here -- unconditional s_branch is a
  // barrier (no fall-through) but IS a PC-relative branch we must relocate.
  return Desc.isBranch() && !Desc.isReturn() && !Desc.isCall() &&
         Inst.getNumOperands() > 0;
}

bool Disassembler::isMemory(const llvm::MCInst& Inst) const {
  InstructionCategory Cat = categorize(Inst);
  return Cat == InstructionCategory::VMEM ||
         Cat == InstructionCategory::SMEM ||
         Cat == InstructionCategory::LDS;
}

llvm::Expected<int64_t>
Disassembler::getBranchTarget(const llvm::MCInst& Inst,
                              uint64_t CurrentPC) const {
  if (!isBranch(Inst)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Instruction is not a branch");
  }

  // AMDGPU branches typically have the offset as the last operand
  // The offset is PC-relative in units of 4-byte dwords
  if (Inst.getNumOperands() == 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Branch has no operands");
  }

  const llvm::MCOperand& OffsetOp = Inst.getOperand(Inst.getNumOperands() - 1);
  if (!OffsetOp.isImm()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Branch offset is not immediate");
  }

  // The offset is a 16-bit signed immediate, PC-relative.
  // LLVM's disassembler returns the raw 16-bit value which may appear as
  // unsigned (e.g., 0xFFFE for -2). Cast to int16_t to get signed semantics.
  // Target = (PC + 4) + (offset * 4)
  int64_t Offset = static_cast<int64_t>(static_cast<int16_t>(OffsetOp.getImm()));
  int64_t Target = CurrentPC + 4 + (Offset * 4);

  return Target;
}

std::string Disassembler::getInstructionName(const llvm::MCInst& Inst) const {
  return std::string(MCII->getName(Inst.getOpcode()));
}

std::string Disassembler::getAsmMnemonic(const llvm::MCInst& Inst) const {
  // Use the InstPrinter to get the real assembly text, then extract the
  // mnemonic (first whitespace-delimited token). This avoids LLVM-internal
  // opcode suffixes like _vi, _gfx9, etc.
  std::string Full = printInstruction(Inst);
  size_t Start = Full.find_first_not_of(" \t");
  if (Start == std::string::npos)
    return "";
  size_t End = Full.find_first_of(" \t", Start);
  if (End == std::string::npos)
    End = Full.size();
  return Full.substr(Start, End - Start);
}

std::string Disassembler::printInstruction(const llvm::MCInst& Inst) const {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  Printer->printInst(&Inst, 0, "", *STI, OS);
  return Str;
}

} // namespace aegisbit
