//===-- ISAEncoder.cpp - AMDGPU ISA Encoding --------------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/ISAEncoder.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/InstructionBuilder.h"
#include "aegisbit/RegisterHelper.h"

#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace aegisbit {

ISAEncoder::~ISAEncoder() = default;

//===----------------------------------------------------------------------===//
// Opcode discovery
//===----------------------------------------------------------------------===//

Error ISAEncoder::discoverOpcodes() {
  auto tryDecode = [&](ArrayRef<uint8_t> Bytes,
                       unsigned &Out) -> Error {
    uint64_t Size;
    auto DIOrErr = D->disassemble(Bytes, /*Address=*/0, Size);
    if (!DIOrErr)
      return DIOrErr.takeError();
    Out = DIOrErr->Inst.getOpcode();
    return Error::success();
  };

  // s_nop 0: 0xBF800000
  {
    uint8_t B[] = {0x00, 0x00, 0x80, 0xBF};
    if (auto E = tryDecode(B, OP_S_NOP)) return E;
  }
  // s_branch 0: 0xBF820000
  {
    uint8_t B[] = {0x00, 0x00, 0x82, 0xBF};
    if (auto E = tryDecode(B, OP_S_BRANCH)) return E;
  }
  // s_getpc_b64 s[0:1]: SOP1 encoding
  {
    uint8_t B[] = {0x00, 0x1C, 0x80, 0xBE};
    if (auto E = tryDecode(B, OP_S_GETPC_B64)) return E;
  }
  // s_setpc_b64 s[0:1]: SOP1, OP=0x1D
  {
    uint8_t B[] = {0x00, 0x1D, 0x80, 0xBE};
    if (auto E = tryDecode(B, OP_S_SETPC_B64)) return E;
  }
  // s_add_u32 s0, s0, 0: SOP2 encoding
  {
    uint8_t B[] = {0x00, 0x80, 0x00, 0x80};
    if (auto E = tryDecode(B, OP_S_ADD_U32)) return E;
  }
  // s_addc_u32 s0, s0, 0: SOP2, OP=4
  {
    uint8_t B[] = {0x00, 0x80, 0x00, 0x82};
    if (auto E = tryDecode(B, OP_S_ADDC_U32)) return E;
  }
  // s_mov_b32 s0, 0: SOP1, OP=3
  {
    uint8_t B[] = {0x80, 0x03, 0x80, 0xBE};
    if (auto E = tryDecode(B, OP_S_MOV_B32)) return E;
  }
  // s_call_b64 s[0:1], 0: SOPK encoding 0xBA800000
  {
    uint8_t B[] = {0x00, 0x00, 0x80, 0xBA};
    if (auto E = tryDecode(B, OP_S_CALL_B64)) return E;
  }
  // s_swappc_b64 s[0:1], s[0:1]
  {
    uint8_t B[] = {0x00, 0x1E, 0x80, 0xBE};
    if (auto E = tryDecode(B, OP_S_SWAPPC_B64)) return E;
  }
  // s_movk_i32 s0, 0
  {
    uint8_t B[] = {0x00, 0x00, 0x00, 0xB0};
    if (auto E = tryDecode(B, OP_S_MOVK_I32)) return E;
  }
  // v_writelane_b32 v0, s0, 0: VOP3 8-byte encoding
  {
    uint8_t B[] = {0x00, 0x00, 0x8A, 0xD2, 0x00, 0x00, 0x01, 0x00};
    if (auto E = tryDecode(B, OP_V_WRITELANE_B32)) return E;
  }
  // v_readlane_b32 s0, v0, 0: VOP3 8-byte encoding
  {
    uint8_t B[] = {0x00, 0x00, 0x89, 0xD2, 0x00, 0x01, 0x01, 0x00};
    if (auto E = tryDecode(B, OP_V_READLANE_B32)) return E;
  }

  // Discover MC register number for M0 (HW encoding = 124 = 0x7C).
  // Encode "s_mov_b32 m0, s0" (SOP1: SDST=124<<16, OP=3, SSRC=0) and
  // extract the destination register from the disassembled MCInst.
  // Non-fatal: if discovery fails (e.g. in unit tests with a limited
  // disassembler), MCReg_M0 stays 0 and callers must check.
  {
    // "s_mov_b32 m0, s0" encoding (verified via llvm-mc for gfx950)
    uint8_t B[4] = {0x00, 0x00, 0xFC, 0xBE};
    uint64_t Size;
    auto DIOrErr = D->disassemble(B, 0, Size);
    if (DIOrErr && DIOrErr->Inst.getNumOperands() >= 1 &&
        DIOrErr->Inst.getOperand(0).isReg())
      MCReg_M0 = DIOrErr->Inst.getOperand(0).getReg();
    else if (DIOrErr)
      MCReg_M0 = 0;
    else
      consumeError(DIOrErr.takeError());
  }

  return Error::success();
}

//===----------------------------------------------------------------------===//
// Instruction encoding
//===----------------------------------------------------------------------===//

Expected<std::vector<uint8_t>>
ISAEncoder::encodeSBranch(int16_t DwordOffset) {
  MCInst MI;
  MI.setOpcode(OP_S_BRANCH);
  MI.addOperand(MCOperand::createImm(DwordOffset));
  return D->encode(MI);
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeSCall(unsigned SGPRPairLo, int16_t DwordOffset) {
  MCInst MI;
  MI.setOpcode(OP_S_CALL_B64);
  MI.addOperand(MCOperand::createReg(SGPRPairLo));
  MI.addOperand(MCOperand::createImm(DwordOffset));
  return D->encode(MI);
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeSetPC(unsigned SGPRPairLo) {
  MCInst MI;
  MI.setOpcode(OP_S_SETPC_B64);
  MI.addOperand(MCOperand::createReg(SGPRPairLo));
  return D->encode(MI);
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeSwapPC(unsigned DstSGPRPairLo, unsigned SrcSGPRPairLo) {
  MCInst MI;
  MI.setOpcode(OP_S_SWAPPC_B64);
  MI.addOperand(MCOperand::createReg(DstSGPRPairLo));
  MI.addOperand(MCOperand::createReg(SrcSGPRPairLo));
  return D->encode(MI);
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeMovK(unsigned SGPR, uint16_t Imm16) {
  MCInst MI;
  MI.setOpcode(OP_S_MOVK_I32);
  MI.addOperand(MCOperand::createReg(SGPR));
  MI.addOperand(MCOperand::createImm(static_cast<int16_t>(Imm16)));
  return D->encode(MI);
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeWriteLane(unsigned VGPR, unsigned SGPR, unsigned Lane) {
  MCInst MI;
  MI.setOpcode(OP_V_WRITELANE_B32);
  MI.addOperand(MCOperand::createReg(VGPR));
  MI.addOperand(MCOperand::createReg(SGPR));
  MI.addOperand(MCOperand::createImm(Lane));
  return D->encode(MI);
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeReadLane(unsigned SGPR, unsigned VGPR, unsigned Lane) {
  MCInst MI;
  MI.setOpcode(OP_V_READLANE_B32);
  MI.addOperand(MCOperand::createReg(SGPR));
  MI.addOperand(MCOperand::createReg(VGPR));
  MI.addOperand(MCOperand::createImm(Lane));
  return D->encode(MI);
}

Expected<std::vector<uint8_t>> ISAEncoder::encodeNop() {
  MCInst MI;
  MI.setOpcode(OP_S_NOP);
  MI.addOperand(MCOperand::createImm(0));
  return D->encode(MI);
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeLongJump(unsigned SGPRPairLo, int64_t ByteOffset) {
  int64_t Addend = ByteOffset - 4;
  int32_t Lo = static_cast<int32_t>(Addend & 0xFFFFFFFF);
  int32_t Hi = static_cast<int32_t>((Addend >> 32) & 0xFFFFFFFF);

  unsigned RegLo = SGPRPairLo;
  unsigned RegHi = SGPRPairLo + 1;

  MCInst GetPC;
  GetPC.setOpcode(OP_S_GETPC_B64);
  GetPC.addOperand(MCOperand::createReg(RegLo));
  auto GetPCBytes = D->encode(GetPC);
  if (!GetPCBytes) return GetPCBytes.takeError();

  MCInst Add;
  Add.setOpcode(OP_S_ADD_U32);
  Add.addOperand(MCOperand::createReg(RegLo));
  Add.addOperand(MCOperand::createReg(RegLo));
  Add.addOperand(MCOperand::createImm(Lo));
  auto AddBytes = D->encode(Add);
  if (!AddBytes) return AddBytes.takeError();

  MCInst Addc;
  Addc.setOpcode(OP_S_ADDC_U32);
  Addc.addOperand(MCOperand::createReg(RegHi));
  Addc.addOperand(MCOperand::createReg(RegHi));
  Addc.addOperand(MCOperand::createImm(Hi));
  auto AddcBytes = D->encode(Addc);
  if (!AddcBytes) return AddcBytes.takeError();

  MCInst SetPC;
  SetPC.setOpcode(OP_S_SETPC_B64);
  SetPC.addOperand(MCOperand::createReg(RegLo));
  auto SetPCBytes = D->encode(SetPC);
  if (!SetPCBytes) return SetPCBytes.takeError();

  std::vector<uint8_t> Result;
  Result.insert(Result.end(), GetPCBytes->begin(), GetPCBytes->end());
  Result.insert(Result.end(), AddBytes->begin(), AddBytes->end());
  Result.insert(Result.end(), AddcBytes->begin(), AddcBytes->end());
  Result.insert(Result.end(), SetPCBytes->begin(), SetPCBytes->end());
  return Result;
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeLongJumpVCC(int64_t ByteOffset) {
  int64_t Addend = ByteOffset - 4;
  int32_t Lo = static_cast<int32_t>(Addend & 0xFFFFFFFF);
  int32_t Hi = static_cast<int32_t>((Addend >> 32) & 0xFFFFFFFF);

  unsigned VCCLo = InstructionBuilder::VCC_LO_REG;
  unsigned VCCHi = InstructionBuilder::VCC_HI_REG;

  MCInst GetPC;
  GetPC.setOpcode(OP_S_GETPC_B64);
  GetPC.addOperand(MCOperand::createReg(VCCLo));
  auto GetPCBytes = D->encode(GetPC);
  if (!GetPCBytes) return GetPCBytes.takeError();

  MCInst Add;
  Add.setOpcode(OP_S_ADD_U32);
  Add.addOperand(MCOperand::createReg(VCCLo));
  Add.addOperand(MCOperand::createReg(VCCLo));
  Add.addOperand(MCOperand::createImm(Lo));
  auto AddBytes = D->encode(Add);
  if (!AddBytes) return AddBytes.takeError();

  MCInst Addc;
  Addc.setOpcode(OP_S_ADDC_U32);
  Addc.addOperand(MCOperand::createReg(VCCHi));
  Addc.addOperand(MCOperand::createReg(VCCHi));
  Addc.addOperand(MCOperand::createImm(Hi));
  auto AddcBytes = D->encode(Addc);
  if (!AddcBytes) return AddcBytes.takeError();

  MCInst SetPC;
  SetPC.setOpcode(OP_S_SETPC_B64);
  SetPC.addOperand(MCOperand::createReg(VCCLo));
  auto SetPCBytes = D->encode(SetPC);
  if (!SetPCBytes) return SetPCBytes.takeError();

  std::vector<uint8_t> Result;
  Result.insert(Result.end(), GetPCBytes->begin(), GetPCBytes->end());
  Result.insert(Result.end(), AddBytes->begin(), AddBytes->end());
  Result.insert(Result.end(), AddcBytes->begin(), AddcBytes->end());
  Result.insert(Result.end(), SetPCBytes->begin(), SetPCBytes->end());
  return Result;
}

//===----------------------------------------------------------------------===//
// Register resolution
//===----------------------------------------------------------------------===//

Expected<unsigned>
ISAEncoder::resolveSGPRPair(unsigned EvenSGPRIndex) {
  uint32_t SDST = EvenSGPRIndex;
  uint32_t Encoding = 0xBA800000u | (SDST << 16);
  uint8_t Bytes[4];
  Bytes[0] = (Encoding >>  0) & 0xFF;
  Bytes[1] = (Encoding >>  8) & 0xFF;
  Bytes[2] = (Encoding >> 16) & 0xFF;
  Bytes[3] = (Encoding >> 24) & 0xFF;

  uint64_t Size;
  auto DIOrErr = D->disassemble(ArrayRef<uint8_t>(Bytes, 4), 0, Size);
  if (!DIOrErr)
    return DIOrErr.takeError();

  if (DIOrErr->Inst.getNumOperands() < 1 ||
      !DIOrErr->Inst.getOperand(0).isReg())
    return createStringError(inconvertibleErrorCode(),
                             "Failed to resolve SGPR pair for index " +
                                 std::to_string(EvenSGPRIndex));

  return DIOrErr->Inst.getOperand(0).getReg();
}

//===----------------------------------------------------------------------===//
// Higher-level helpers
//===----------------------------------------------------------------------===//

Expected<std::vector<uint8_t>>
ISAEncoder::emitInst(const llvm::MCInst &Inst) {
  auto BytesOrErr = D->encode(Inst);
  if (!BytesOrErr) return BytesOrErr.takeError();
  return *BytesOrErr;
}

Expected<std::vector<uint8_t>>
ISAEncoder::buildAndEmit(const std::string &Mnemonic,
                         const std::vector<InstructionBuilder::Operand> &Ops) {
  auto InstOrErr = InstructionBuilder::build(*D, Mnemonic, Ops);
  if (!InstOrErr) return InstOrErr.takeError();
  return emitInst(*InstOrErr);
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeAccVGPRWrite(unsigned DstAGPR, unsigned SrcVGPR) {
  using Op = InstructionBuilder::Operand;
  return buildAndEmit("V_ACCVGPR_WRITE_B32",
                      {Op::Reg(DstAGPR), Op::Reg(SrcVGPR)});
}

Expected<std::vector<uint8_t>>
ISAEncoder::encodeAccVGPRRead(unsigned DstVGPR, unsigned SrcAGPR) {
  using Op = InstructionBuilder::Operand;
  return buildAndEmit("V_ACCVGPR_READ_B32",
                      {Op::Reg(DstVGPR), Op::Reg(SrcAGPR)});
}

void ISAEncoder::append(std::vector<uint8_t> &Dst,
                        const std::vector<uint8_t> &Src) {
  Dst.insert(Dst.end(), Src.begin(), Src.end());
}

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

Expected<std::unique_ptr<ISAEncoder>>
ISAEncoder::create(StringRef GPUArch, Disassembler &Disasm) {
  auto Enc = std::unique_ptr<ISAEncoder>(new ISAEncoder());
  Enc->D = &Disasm;
  Enc->Arch = GPUArch.str();

  if (auto E = Enc->discoverOpcodes())
    return std::move(E);

  return Enc;
}

} // namespace aegisbit
