//===-- aegisbit/ISAEncoder.h - AMDGPU ISA Encoding ---------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Low-level AMDGPU instruction encoding via LLVM MCCodeEmitter.
/// Discovers opcodes at construction time by disassembling known byte patterns,
/// then provides named encoding methods for each instruction form.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_ISA_ENCODER_H
#define AEGISBIT_ISA_ENCODER_H

#include "aegisbit/InstructionBuilder.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

namespace aegisbit {

class Disassembler;

class ISAEncoder {
public:
  static llvm::Expected<std::unique_ptr<ISAEncoder>>
  create(llvm::StringRef GPUArch, Disassembler &Disasm);

  ~ISAEncoder();

  Disassembler &getDisassembler() { return *D; }
  const std::string &getArch() const { return Arch; }

  // --- Scalar branch / call ---
  llvm::Expected<std::vector<uint8_t>> encodeSBranch(int16_t DwordOffset);
  llvm::Expected<std::vector<uint8_t>>
  encodeSCall(unsigned SGPRPairLo, int16_t DwordOffset);

  // --- PC manipulation ---
  llvm::Expected<std::vector<uint8_t>> encodeSetPC(unsigned SGPRPairLo);
  llvm::Expected<std::vector<uint8_t>>
  encodeSwapPC(unsigned DstSGPRPairLo, unsigned SrcSGPRPairLo);
  llvm::Expected<std::vector<uint8_t>> encodeMovK(unsigned SGPR, uint16_t Imm16);

  // --- Lane save / restore ---
  llvm::Expected<std::vector<uint8_t>>
  encodeWriteLane(unsigned VGPR, unsigned SGPR, unsigned Lane);
  llvm::Expected<std::vector<uint8_t>>
  encodeReadLane(unsigned SGPR, unsigned VGPR, unsigned Lane);

  // --- NOP ---
  llvm::Expected<std::vector<uint8_t>> encodeNop();

  // --- Long jumps (4-instruction sequence: getpc + add + addc + setpc) ---
  llvm::Expected<std::vector<uint8_t>>
  encodeLongJump(unsigned SGPRPairLo, int64_t ByteOffset);
  llvm::Expected<std::vector<uint8_t>>
  encodeLongJumpVCC(int64_t ByteOffset);

  // --- Register resolution ---
  llvm::Expected<unsigned> resolveSGPRPair(unsigned EvenSGPRIndex);
  unsigned getM0Reg() const { return MCReg_M0; }

  // --- Higher-level helpers ---
  llvm::Expected<std::vector<uint8_t>>
  emitInst(const llvm::MCInst &Inst);

  llvm::Expected<std::vector<uint8_t>>
  buildAndEmit(const std::string &Mnemonic,
               const std::vector<InstructionBuilder::Operand> &Ops = {});

  llvm::Expected<std::vector<uint8_t>>
  encodeAccVGPRWrite(unsigned DstAGPR, unsigned SrcVGPR);

  llvm::Expected<std::vector<uint8_t>>
  encodeAccVGPRRead(unsigned DstVGPR, unsigned SrcAGPR);

  static void append(std::vector<uint8_t> &Dst,
                     const std::vector<uint8_t> &Src);

private:
  ISAEncoder() = default;

  Disassembler *D = nullptr;
  std::string Arch;

  unsigned OP_S_BRANCH = 0;
  unsigned OP_S_NOP = 0;
  unsigned OP_S_GETPC_B64 = 0;
  unsigned OP_S_SETPC_B64 = 0;
  unsigned OP_S_ADD_U32 = 0;
  unsigned OP_S_ADDC_U32 = 0;
  unsigned OP_S_MOV_B32 = 0;
  unsigned OP_S_CALL_B64 = 0;
  unsigned OP_S_SWAPPC_B64 = 0;
  unsigned OP_S_MOVK_I32 = 0;
  unsigned OP_V_WRITELANE_B32 = 0;
  unsigned OP_V_READLANE_B32 = 0;
  unsigned MCReg_M0 = 0;

  llvm::Error discoverOpcodes();
};

} // namespace aegisbit

#endif // AEGISBIT_ISA_ENCODER_H
