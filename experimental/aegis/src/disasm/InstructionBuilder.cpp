//===-- InstructionBuilder.cpp - Instruction Builder -----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/InstructionBuilder.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/RegisterHelper.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Error.h"
#include <unordered_map>

namespace aegisbit {

//===----------------------------------------------------------------------===//
// SGPRPair implementation
//===----------------------------------------------------------------------===//

SGPRPair SGPRPair::fromIndex(unsigned BaseIdx) {
  // Base index must be even for aligned pairs
  unsigned AlignedIdx = BaseIdx & ~1u;
  return SGPRPair{
      RegisterHelper::getSGPR(AlignedIdx),
      RegisterHelper::getSGPR(AlignedIdx + 1)
  };
}

VGPRPair VGPRPair::fromIndex(unsigned BaseIdx) {
  return VGPRPair{
      RegisterHelper::getVGPR(BaseIdx),
      RegisterHelper::getVGPR(BaseIdx + 1)
  };
}

//===----------------------------------------------------------------------===//
// Opcode Lookup Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Preferred architecture suffixes for gfx9 (gfx942/gfx950)
static const std::vector<std::string> ArchSuffixes = {
    "_vi",        // GFX9 (Vega, gfx900-gfx942/gfx950)
    "_gfx940",    // GFX940/GFX950 (CDNA3) — specific encodings
    "_gfx9",      // GFX9 alternative naming
    "_gfx10",     // GFX10 (fallback)
    "_gfx11",     // GFX11 (fallback)
    "_gfx12",     // GFX12 (fallback)
    "_gfx6_gfx7", // GFX6/GFX7 (for some legacy instructions)
};

/// Check if the suffix is a valid architecture suffix
/// Valid: "vi", "gfx10", "gfx11", "gfx12", "gfx6_gfx7", "gfx13"
/// Invalid: "ORDERED_PS_DONE_vi" (has extra prefix)
bool isValidArchSuffix(const std::string& Suffix) {
  // Simple suffix without underscores (like "vi", "gfx10")
  if (Suffix.find('_') == std::string::npos) {
    return true;
  }
  // Allow gfx6_gfx7 format
  if (Suffix.find("gfx") == 0) {
    return true;
  }
  return false;
}

/// Check if OpName is an exact architecture variant of Mnemonic
/// Returns the priority of the match (lower is better), or 0 if no match
/// Priority 1: _vi or _gfx9 (best for our target)
/// Priority 2: Other architecture suffixes
unsigned getArchVariantPriority(const std::string& OpName,
                                const std::string& Mnemonic,
                                const llvm::MCInstrDesc& Desc) {
  if (Desc.isPseudo()) {
    return 0;  // No match for pseudo instructions
  }

  for (const auto& Suffix : ArchSuffixes) {
    if (OpName == Mnemonic + Suffix) {
      // Priority 1: Best match for our target (gfx942/gfx950)
      if (Suffix == "_vi" || Suffix == "_gfx9" || Suffix == "_gfx940") {
        return 1;
      }
      // Priority 2: Other architecture variants
      return 2;
    }
  }
  return 0;
}

/// Check if OpName is a general variant match (handles less common suffixes)
/// Returns true if this is a valid general variant
bool isGeneralVariant(const std::string& OpName,
                      const std::string& Mnemonic,
                      const llvm::MCInstrDesc& Desc) {
  if (Desc.isPseudo()) {
    return false;
  }

  // Must start with Mnemonic + "_"
  std::string Prefix = Mnemonic + "_";
  if (OpName.find(Prefix) != 0) {
    return false;
  }

  // Skip SADDR variants - they have different operand counts
  if (OpName.find("_SADDR") != std::string::npos) {
    return false;
  }

  // Get the suffix part after Mnemonic + "_"
  std::string Suffix = OpName.substr(Prefix.length());
  return isValidArchSuffix(Suffix);
}

} // anonymous namespace

llvm::Expected<unsigned>
InstructionBuilder::findOpcodeByName(const llvm::MCInstrInfo& MCII,
                                     const std::string& Mnemonic) {
  // Lazy-initialized cache: mnemonic → resolved opcode.
  // The full scan is O(n) over ~40,000 AMDGPU opcodes, which is fine once
  // but unacceptable when called thousands of times during trampoline
  // construction (71 sites × ~30 instructions each).
  static std::unordered_map<std::string, unsigned> Cache;
  auto It = Cache.find(Mnemonic);
  if (It != Cache.end())
    return It->second;

  // Strategy: Find all matching names, prefer real (non-pseudo) instructions.
  // Priority order:
  //   1. Exact match with non-pseudo instruction
  //   2. Architecture variant match (_vi, _gfx9) - best for our target
  //   3. Other architecture variant
  //   4. General variant with valid suffix
  //   5. Pseudo instruction as last resort

  unsigned PseudoMatch = 0;
  bool FoundPseudo = false;
  unsigned ArchVariant = 0;
  unsigned ArchPriority = 0;  // Lower is better (1=best, 2=fallback)
  unsigned GeneralVariant = 0;
  bool FoundGeneral = false;

  unsigned NumOpcodes = MCII.getNumOpcodes();
  for (unsigned Opcode = 0; Opcode < NumOpcodes; ++Opcode) {
    const llvm::MCInstrDesc& Desc = MCII.get(Opcode);
    const char* Name = MCII.getName(Opcode).data();

    if (!Name)
      continue;

    std::string OpName(Name);

    if (OpName == Mnemonic) {
      if (!Desc.isPseudo()) {
        Cache[Mnemonic] = Opcode;
        return Opcode;
      }
      if (!FoundPseudo) {
        PseudoMatch = Opcode;
        FoundPseudo = true;
      }
      continue;
    }

    unsigned Priority = getArchVariantPriority(OpName, Mnemonic, Desc);
    if (Priority == 1) {
      Cache[Mnemonic] = Opcode;
      return Opcode;
    }
    if (Priority > 0 && (ArchPriority == 0 || Priority < ArchPriority)) {
      ArchVariant = Opcode;
      ArchPriority = Priority;
      continue;
    }

    if (!FoundGeneral && isGeneralVariant(OpName, Mnemonic, Desc)) {
      GeneralVariant = Opcode;
      FoundGeneral = true;
    }
  }

  unsigned Result = 0;
  bool Found = false;
  if (ArchPriority > 0) {
    Result = ArchVariant;
    Found = true;
  } else if (FoundGeneral) {
    Result = GeneralVariant;
    Found = true;
  } else if (FoundPseudo) {
    Result = PseudoMatch;
    Found = true;
  }

  if (Found) {
    Cache[Mnemonic] = Result;
    return Result;
  }

  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Instruction mnemonic '" + Mnemonic + "' not found");
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::build(const llvm::MCInstrInfo& MCII,
                          const std::string& Mnemonic,
                          const std::vector<Operand>& Operands) {
  // Find opcode by name
  auto OpcodeOrErr = findOpcodeByName(MCII, Mnemonic);
  if (!OpcodeOrErr)
    return OpcodeOrErr.takeError();

  unsigned Opcode = *OpcodeOrErr;

  // Create instruction with opcode
  llvm::MCInst Inst;
  Inst.setOpcode(Opcode);

  // Add operands
  for (const auto& Op : Operands) {
    switch (Op.Type) {
    case OperandType::Immediate:
      Inst.addOperand(llvm::MCOperand::createImm(Op.Value));
      break;
    case OperandType::Register:
      Inst.addOperand(llvm::MCOperand::createReg(
          static_cast<unsigned>(Op.Value)));
      break;
    case OperandType::Expression:
      // Expression operands would require MCExpr handling - not needed yet
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Expression operands not yet supported");
    }
  }

  return Inst;
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::build(const Disassembler& Disasm,
                          const std::string& Mnemonic,
                          const std::vector<Operand>& Operands) {
  return build(Disasm.getMCII(), Mnemonic, Operands);
}

//===----------------------------------------------------------------------===//
// High-level instruction builders
//===----------------------------------------------------------------------===//

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSMovB32(const Disassembler& Disasm,
                                  unsigned DstSGPR,
                                  uint32_t Imm) {
  // S_MOV_B32 sdst, ssrc0
  // For immediate source, we need to use the literal constant encoding
  return build(Disasm, "S_MOV_B32", {
      Operand::Reg(DstSGPR),
      Operand::Imm(static_cast<int64_t>(Imm))
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSAddU32(const Disassembler& Disasm,
                                  unsigned DstSGPR,
                                  unsigned Src0SGPR,
                                  uint32_t Imm) {
  // S_ADD_U32 sdst, ssrc0, ssrc1
  return build(Disasm, "S_ADD_U32", {
      Operand::Reg(DstSGPR),
      Operand::Reg(Src0SGPR),
      Operand::Imm(static_cast<int64_t>(Imm))
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSAddU32Reg(const Disassembler& Disasm,
                                     unsigned DstSGPR,
                                     unsigned Src0SGPR,
                                     unsigned Src1SGPR) {
  // S_ADD_U32 sdst, ssrc0, ssrc1 — register + register
  return build(Disasm, "S_ADD_U32", {
      Operand::Reg(DstSGPR),
      Operand::Reg(Src0SGPR),
      Operand::Reg(Src1SGPR)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSAddcU32(const Disassembler& Disasm,
                                   unsigned DstSGPR,
                                   unsigned Src0SGPR,
                                   uint32_t Imm) {
  // S_ADDC_U32 sdst, ssrc0, ssrc1 — adds with carry-in from SCC
  return build(Disasm, "S_ADDC_U32", {
      Operand::Reg(DstSGPR),
      Operand::Reg(Src0SGPR),
      Operand::Imm(static_cast<int64_t>(Imm))
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSSubU32(const Disassembler& Disasm,
                                  unsigned DstSGPR,
                                  unsigned Src0SGPR,
                                  unsigned Src1SGPR) {
  return build(Disasm, "S_SUB_U32", {
      Operand::Reg(DstSGPR),
      Operand::Reg(Src0SGPR),
      Operand::Reg(Src1SGPR)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSSubbU32(const Disassembler& Disasm,
                                   unsigned DstSGPR,
                                   unsigned Src0SGPR,
                                   uint32_t Imm) {
  return build(Disasm, "S_SUBB_U32", {
      Operand::Reg(DstSGPR),
      Operand::Reg(Src0SGPR),
      Operand::Imm(static_cast<int64_t>(Imm))
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSAndB32(const Disassembler& Disasm,
                                  unsigned DstSGPR,
                                  unsigned Src0SGPR,
                                  const Operand& Src1) {
  // S_AND_B32 sdst, ssrc0, ssrc1
  return build(Disasm, "S_AND_B32", {
      Operand::Reg(DstSGPR),
      Operand::Reg(Src0SGPR),
      Src1
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSGetRegB32(const Disassembler& Disasm,
                                     unsigned DstSGPR,
                                     uint16_t HwRegEncoding) {
  // S_GETREG_B32 sdst, hwreg(id, offset, width)
  // SOPK format: reads a hardware register field into an SGPR
  return build(Disasm, "S_GETREG_B32", {
      Operand::Reg(DstSGPR),
      Operand::Imm(static_cast<int64_t>(HwRegEncoding))
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSGetPCB64(const Disassembler& Disasm,
                                    unsigned DstSGPRPairLo) {
  // S_GETPC_B64 sdst - gets current PC into 64-bit SGPR pair
  return build(Disasm, "S_GETPC_B64", {
      Operand::Reg(DstSGPRPairLo)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSSetPCB64(const Disassembler& Disasm,
                                    unsigned SrcSGPRPairLo) {
  // S_SETPC_B64 ssrc - indirect jump to address in 64-bit SGPR pair
  return build(Disasm, "S_SETPC_B64", {
      Operand::Reg(SrcSGPRPairLo)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSWaitCnt(const Disassembler& Disasm,
                                   unsigned VmCnt,
                                   unsigned ExpCnt,
                                   unsigned LgkmCnt) {
  // S_WAITCNT uses packed encoding:
  // GFX9/CDNA: imm16 = vmcnt[3:0] | (expcnt[2:0] << 4) | (lgkmcnt[3:0] << 8) | (vmcnt[5:4] << 14)
  // WARNING: GFX10+ uses a completely different bit layout:
  //   GFX10+: vmcnt[5:0] in bits 0-5, expcnt[2:0] in bits 8-10, lgkmcnt[5:0] in bits 12-17
  //   GFX10+ also uses separate S_WAITCNT_VSCNT for vs_cnt.
  // This encoding is ONLY valid for GFX9/CDNA targets (gfx900-gfx950).
  // TODO: Add GFX10+ encoding path when RDNA support is needed.

  // Build the packed immediate (GFX9 format)
  uint16_t Imm = (VmCnt & 0xF) | ((ExpCnt & 0x7) << 4) | ((LgkmCnt & 0xF) << 8);
  if (VmCnt > 15) {
    Imm |= ((VmCnt >> 4) & 0x3) << 14;  // High bits of vmcnt
  }

  return build(Disasm, "S_WAITCNT", {Operand::Imm(Imm)});
}

//===----------------------------------------------------------------------===//
// SMEM (Scalar Memory) Instructions
//===----------------------------------------------------------------------===//

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSLoadDword(const Disassembler& Disasm,
                                     unsigned DstSGPR,
                                     unsigned BaseSGPRPairLo,
                                     uint32_t Offset) {
  // S_LOAD_DWORD sdst, sbase, offset, cpol
  // Note: LLVM uses S_LOAD_DWORD_IMM for immediate offset variant
  // cpol = cache policy (0 = default)
  return build(Disasm, "S_LOAD_DWORD_IMM", {
      Operand::Reg(DstSGPR),
      Operand::Reg(BaseSGPRPairLo),
      Operand::Imm(static_cast<int64_t>(Offset)),
      Operand::Imm(0)  // cpol: default cache policy
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSLoadDwordX2(const Disassembler& Disasm,
                                       unsigned DstSGPRPairLo,
                                       unsigned BaseSGPRPairLo,
                                       uint32_t Offset) {
  // S_LOAD_DWORDX2 sdst, sbase, offset, cpol
  return build(Disasm, "S_LOAD_DWORDX2_IMM", {
      Operand::Reg(DstSGPRPairLo),
      Operand::Reg(BaseSGPRPairLo),
      Operand::Imm(static_cast<int64_t>(Offset)),
      Operand::Imm(0)  // cpol: default cache policy
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSLoadDwordX4(const Disassembler& Disasm,
                                       unsigned DstSGPRQuadLo,
                                       unsigned BaseSGPRPairLo,
                                       uint32_t Offset) {
  // S_LOAD_DWORDX4 sdst, sbase, offset, cpol
  return build(Disasm, "S_LOAD_DWORDX4_IMM", {
      Operand::Reg(DstSGPRQuadLo),
      Operand::Reg(BaseSGPRPairLo),
      Operand::Imm(static_cast<int64_t>(Offset)),
      Operand::Imm(0)  // cpol: default cache policy
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSStoreDword(const Disassembler& Disasm,
                                      unsigned SrcSGPR,
                                      unsigned BaseSGPRPairLo,
                                      uint32_t Offset) {
  // S_STORE_DWORD sdata, sbase, offset, cpol
  // Cache policy: 0 = default (no GLC/DLC bits set)
  return build(Disasm, "S_STORE_DWORD_IMM", {
      Operand::Reg(SrcSGPR),
      Operand::Reg(BaseSGPRPairLo),
      Operand::Imm(static_cast<int64_t>(Offset)),
      Operand::Imm(0)  // Cache policy: default
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSStoreDwordX2(const Disassembler& Disasm,
                                        unsigned SrcSGPRPairLo,
                                        unsigned BaseSGPRPairLo,
                                        uint32_t Offset) {
  // S_STORE_DWORDX2 sdata, sbase, offset, cpol
  return build(Disasm, "S_STORE_DWORDX2_IMM", {
      Operand::Reg(SrcSGPRPairLo),
      Operand::Reg(BaseSGPRPairLo),
      Operand::Imm(static_cast<int64_t>(Offset)),
      Operand::Imm(0)  // Cache policy: default
  });
}

//===----------------------------------------------------------------------===//
// VALU (Vector ALU) Instructions - M4 Memory Tracing
//===----------------------------------------------------------------------===//

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildVMovB32(const Disassembler& Disasm,
                                  unsigned DstVGPR,
                                  unsigned SrcSGPR) {
  // V_MOV_B32 vdst, src
  // VOP1 format: broadcasts scalar to all lanes
  return build(Disasm, "V_MOV_B32_e32", {
      Operand::Reg(DstVGPR),
      Operand::Reg(SrcSGPR)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildVMovB32Imm(const Disassembler& Disasm,
                                     unsigned DstVGPR,
                                     uint32_t Imm) {
  // V_MOV_B32 vdst, imm
  // VOP1 format: broadcasts immediate to all lanes
  return build(Disasm, "V_MOV_B32_e32", {
      Operand::Reg(DstVGPR),
      Operand::Imm(static_cast<int64_t>(Imm))
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildVReadFirstLaneB32(const Disassembler& Disasm,
                                            unsigned DstSGPR,
                                            unsigned SrcVGPR) {
  // V_READFIRSTLANE_B32 sdst, vsrc
  // VOP1 format: copies first active lane to scalar
  return build(Disasm, "V_READFIRSTLANE_B32", {
      Operand::Reg(DstSGPR),
      Operand::Reg(SrcVGPR)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildVAddCOU32(const Disassembler& Disasm,
                                    unsigned DstVGPR,
                                    unsigned Src0SGPR,
                                    unsigned Src1VGPR) {
  // V_ADD_CO_U32 vdst, vcc, src0, src1
  // VOP2 format on GFX9: adds two values and sets VCC with carry
  //
  // On GFX9, this is V_ADD_CO_U32 (not V_ADD_I32 which is the old name)
  // The instruction implicitly writes to VCC.
  return build(Disasm, "V_ADD_CO_U32_e32", {
      Operand::Reg(DstVGPR),
      Operand::Reg(Src0SGPR),
      Operand::Reg(Src1VGPR)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildVAddcU32(const Disassembler& Disasm,
                                   unsigned DstVGPR,
                                   uint32_t Src0Imm,
                                   unsigned Src1VGPR) {
  // V_ADDC_U32 vdst, vcc, src0, src1, vcc
  // VOP2 format on GFX9: adds two values plus VCC carry, sets VCC with carry
  //
  // This instruction reads and writes VCC implicitly.
  return build(Disasm, "V_ADDC_U32_e32", {
      Operand::Reg(DstVGPR),
      Operand::Imm(static_cast<int64_t>(Src0Imm)),
      Operand::Reg(Src1VGPR)
  });
}

//===----------------------------------------------------------------------===//
// VALU Lane ID Instructions — for per-lane address computation
//===----------------------------------------------------------------------===//

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildVMbcntLoU32B32(const Disassembler& Disasm,
                                         unsigned DstVGPR,
                                         uint32_t Src0Imm,
                                         uint32_t Src1Imm) {
  // V_MBCNT_LO_U32_B32 vdst, src0, src1
  // e64-only on GFX9 (no VOP2/e32 form on GFX9, only VOP3/e64).
  // For each lane, counts the number of set bits in src0 at bit positions
  // below the current lane (within lanes 0-31), then adds src1.
  // With src0=0xFFFFFFFF, src1=0: result = lane_id for lanes 0-31.
  return build(Disasm, "V_MBCNT_LO_U32_B32_e64", {
      Operand::Reg(DstVGPR),
      Operand::Imm(static_cast<int64_t>(Src0Imm)),
      Operand::Imm(static_cast<int64_t>(Src1Imm))
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildVMbcntHiU32B32(const Disassembler& Disasm,
                                         unsigned DstVGPR,
                                         uint32_t Src0Imm,
                                         unsigned Src1VGPR) {
  // V_MBCNT_HI_U32_B32 vdst, src0, src1
  // e64-only on GFX9 (no VOP2/e32 form on GFX9, only VOP3/e64).
  // Continues the count from V_MBCNT_LO for lanes 32-63.
  // With src0=0xFFFFFFFF, src1=v_mbcnt_lo result: gives complete lane_id 0-63.
  return build(Disasm, "V_MBCNT_HI_U32_B32_e64", {
      Operand::Reg(DstVGPR),
      Operand::Imm(static_cast<int64_t>(Src0Imm)),
      Operand::Reg(Src1VGPR)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildVLshlrevB32(const Disassembler& Disasm,
                                      unsigned DstVGPR,
                                      uint32_t ShiftAmt,
                                      unsigned SrcVGPR) {
  return build(Disasm, "V_LSHLREV_B32_e32", {
      Operand::Reg(DstVGPR),
      Operand::Imm(static_cast<int64_t>(ShiftAmt)),
      Operand::Reg(SrcVGPR)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildVLshrrevB32(const Disassembler& Disasm,
                                      unsigned DstVGPR,
                                      uint32_t ShiftAmt,
                                      unsigned SrcVGPR) {
  return build(Disasm, "V_LSHRREV_B32_e32", {
      Operand::Reg(DstVGPR),
      Operand::Imm(static_cast<int64_t>(ShiftAmt)),
      Operand::Reg(SrcVGPR)
  });
}

//===----------------------------------------------------------------------===//
// EXEC Mask Manipulation
//===----------------------------------------------------------------------===//

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSMovB32FromExec(const Disassembler& Disasm,
                                          unsigned DstSGPR, bool High) {
  // S_MOV_B32 sdst, exec_lo/exec_hi
  unsigned ExecReg = High ? EXEC_HI_REG : EXEC_LO_REG;
  return build(Disasm, "S_MOV_B32", {
      Operand::Reg(DstSGPR),
      Operand::Reg(ExecReg)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSMovB32ToExecImm(const Disassembler& Disasm,
                                           uint32_t Imm, bool High) {
  // S_MOV_B32 exec_lo/exec_hi, imm
  unsigned ExecReg = High ? EXEC_HI_REG : EXEC_LO_REG;
  return build(Disasm, "S_MOV_B32", {
      Operand::Reg(ExecReg),
      Operand::Imm(static_cast<int64_t>(Imm))
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildSMovB32ToExecReg(const Disassembler& Disasm,
                                           unsigned SrcSGPR, bool High) {
  // S_MOV_B32 exec_lo/exec_hi, sgpr
  unsigned ExecReg = High ? EXEC_HI_REG : EXEC_LO_REG;
  return build(Disasm, "S_MOV_B32", {
      Operand::Reg(ExecReg),
      Operand::Reg(SrcSGPR)
  });
}

//===----------------------------------------------------------------------===//
// VMEM (Vector Memory) Instructions - M4 Memory Tracing
//===----------------------------------------------------------------------===//

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildGlobalLoadDword(const Disassembler& Disasm,
                                          unsigned VDstVGPR,
                                          unsigned VAddrVGPR,
                                          unsigned SAddrSGPRLo,
                                          int32_t Offset) {
  // GLOBAL_LOAD_DWORD_SADDR: MCInst operand order is vdst, saddr, vaddr,
  // offset, cpol (saddr before vaddr — opposite of store SADDR variants).
  return build(Disasm, "GLOBAL_LOAD_DWORD_SADDR", {
      Operand::Reg(VDstVGPR),
      Operand::Reg(SAddrSGPRLo),
      Operand::Reg(VAddrVGPR),
      Operand::Imm(Offset),
      Operand::Imm(0)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildGlobalStoreDword(const Disassembler& Disasm,
                                           unsigned VAddrVGPR,
                                           unsigned VDataVGPR,
                                           unsigned SAddrSGPRLo,
                                           int32_t Offset) {
  // GLOBAL_STORE_DWORD_SADDR vaddr, vdata, saddr, offset, cpol
  // Uses SGPR pair for base address. s_store_dword is broken on gfx950.
  return build(Disasm, "GLOBAL_STORE_DWORD_SADDR", {
      Operand::Reg(VAddrVGPR),
      Operand::Reg(VDataVGPR),
      Operand::Reg(SAddrSGPRLo),
      Operand::Imm(Offset),
      Operand::Imm(0)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildGlobalAtomicAddRtn(const Disassembler& Disasm,
                                             unsigned VDstVGPR,
                                             unsigned VAddrVGPR,
                                             unsigned VDataVGPR,
                                             unsigned SAddrSGPRLo,
                                             int32_t Offset) {
  // GLOBAL_ATOMIC_ADD_SADDR_RTN vdst, vaddr, vdata, saddr, offset, cpol
  // Atomically: old = *addr; *addr += vdata; vdst = old
  // GFX9 cpol encoding: GLC is bit 0. So cpol=1 means GLC=1.
  return build(Disasm, "GLOBAL_ATOMIC_ADD_SADDR_RTN", {
      Operand::Reg(VDstVGPR),
      Operand::Reg(VAddrVGPR),
      Operand::Reg(VDataVGPR),
      Operand::Reg(SAddrSGPRLo),
      Operand::Imm(Offset),
      Operand::Imm(1)              // cpol: GLC=1 (required for RTN)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildGlobalAtomicAddNoRtn(const Disassembler& Disasm,
                                               unsigned VAddrVGPR,
                                               unsigned VDataVGPR,
                                               unsigned SAddrSGPRLo,
                                               int32_t Offset) {
  // GLOBAL_ATOMIC_ADD_SADDR vaddr, vdata, saddr, offset, cpol
  // Fire-and-forget: atomically *addr += vdata, no return value.
  return build(Disasm, "GLOBAL_ATOMIC_ADD_SADDR", {
      Operand::Reg(VAddrVGPR),
      Operand::Reg(VDataVGPR),
      Operand::Reg(SAddrSGPRLo),
      Operand::Imm(Offset),
      Operand::Imm(0)              // cpol: GLC=0 (no return)
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildGlobalStoreDwordNoSaddr(const Disassembler& Disasm,
                                                  unsigned VAddrLo,
                                                  unsigned VDataVGPR,
                                                  int32_t Offset) {
  // GLOBAL_STORE_DWORD vaddr, vdata, off offset:N cpol:0
  // Non-SADDR variant: vaddr is a VGPR pair (64-bit per-lane address),
  // vdata is a single VGPR (32-bit data per lane).
  // Used for 32-bit LDS address capture (zero-extended to 64-bit slots).
  return build(Disasm, "GLOBAL_STORE_DWORD", {
      Operand::Reg(VAddrLo),       // vaddr pair (low VGPR of 64-bit address)
      Operand::Reg(VDataVGPR),     // vdata (single VGPR, 32-bit data)
      Operand::Imm(Offset),        // offset immediate
      Operand::Imm(0)              // cpol: default cache policy
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildGlobalStoreDwordX2(const Disassembler& Disasm,
                                             unsigned VAddrLo,
                                             unsigned VDataLo,
                                             int32_t Offset) {
  // GLOBAL_STORE_DWORDX2 vaddr, vdata, off offset:N cpol:0
  // FLAT_GLOBAL format with offset
  //
  // The instruction format for GFX9 (gfx942):
  //   GLOBAL_STORE_DWORDX2 v[addr:addr+1], v[data:data+1], off offset:N
  //
  // Operands from LLVM TableGen (FLAT_Store_Pseudo):
  //   vaddr: VReg_64 - VGPR pair for 64-bit address per lane
  //   vdata: AVLdSt_64 - VGPR pair for 64-bit data per lane
  //   offset: flat_offset - Immediate offset
  //   cpol: CPol_0 - Cache policy (0 = default)

  return build(Disasm, "GLOBAL_STORE_DWORDX2", {
      Operand::Reg(VAddrLo),      // vaddr pair (low VGPR)
      Operand::Reg(VDataLo),      // vdata pair (low VGPR)
      Operand::Imm(Offset),       // offset immediate
      Operand::Imm(0)             // cpol: default cache policy
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildGlobalLoadDwordX2(const Disassembler& Disasm,
                                            unsigned VDstLo,
                                            unsigned VAddrLo,
                                            int32_t Offset) {
  // GLOBAL_LOAD_DWORDX2 vdst, vaddr, off offset:N cpol:0
  // FLAT_GLOBAL format with offset
  //
  // Operands:
  //   vdst: VGPR pair for destination
  //   vaddr: VGPR pair for address
  //   offset: immediate offset
  //   cpol: cache policy
  return build(Disasm, "GLOBAL_LOAD_DWORDX2", {
      Operand::Reg(VDstLo),       // vdst pair (low VGPR)
      Operand::Reg(VAddrLo),      // vaddr pair (low VGPR)
      Operand::Imm(Offset),       // offset immediate
      Operand::Imm(0)             // cpol: default cache policy
  });
}

//===----------------------------------------------------------------------===//
// Scratch Memory Instructions
//===----------------------------------------------------------------------===//

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildScratchStoreDword(const Disassembler& Disasm,
                                            unsigned VDataVGPR,
                                            uint32_t Offset) {
  // SCRATCH_STORE_DWORD_ST vdata, offset:N, cpol
  // "_ST" variant: offset-only. Address = flat_scratch + sext(offset).
  // WARNING: offset is signed 13-bit (-4096..4095). For offsets > 4095,
  // use buildScratchStoreDwordSAddr with an SGPR base.
  return build(Disasm, "SCRATCH_STORE_DWORD_ST", {
      Operand::Reg(VDataVGPR),    // vdata
      Operand::Imm(Offset),       // offset immediate
      Operand::Imm(0)             // cpol: default cache policy
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildScratchStoreDwordSAddr(const Disassembler& Disasm,
                                                 unsigned VDataVGPR,
                                                 unsigned SAddrSGPR,
                                                 int32_t Offset) {
  // SCRATCH_STORE_DWORD_SADDR vdata, saddr, offset, cpol
  // Address = flat_scratch + saddr + sext(offset).
  // Use when the total offset exceeds the 13-bit immediate range.
  return build(Disasm, "SCRATCH_STORE_DWORD_SADDR", {
      Operand::Reg(VDataVGPR),    // vdata
      Operand::Reg(SAddrSGPR),    // saddr (SGPR holding base offset)
      Operand::Imm(Offset),       // small offset immediate
      Operand::Imm(0)             // cpol: default cache policy
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildScratchLoadDword(const Disassembler& Disasm,
                                           unsigned VDstVGPR,
                                           uint32_t Offset) {
  // SCRATCH_LOAD_DWORD_ST vdst, offset:N, cpol
  // "_ST" variant: offset-only. Address = flat_scratch + sext(offset).
  // WARNING: offset is signed 13-bit (-4096..4095). For offsets > 4095,
  // use buildScratchLoadDwordSAddr with an SGPR base.
  return build(Disasm, "SCRATCH_LOAD_DWORD_ST", {
      Operand::Reg(VDstVGPR),     // vdst
      Operand::Imm(Offset),       // offset immediate
      Operand::Imm(0)             // cpol: default cache policy
  });
}

llvm::Expected<llvm::MCInst>
InstructionBuilder::buildScratchLoadDwordSAddr(const Disassembler& Disasm,
                                                unsigned VDstVGPR,
                                                unsigned SAddrSGPR,
                                                int32_t Offset) {
  // SCRATCH_LOAD_DWORD_SADDR vdst, saddr, offset, cpol
  // Address = flat_scratch + saddr + sext(offset).
  // Use when the total offset exceeds the 13-bit immediate range.
  return build(Disasm, "SCRATCH_LOAD_DWORD_SADDR", {
      Operand::Reg(VDstVGPR),     // vdst
      Operand::Reg(SAddrSGPR),    // saddr (SGPR holding base offset)
      Operand::Imm(Offset),       // small offset immediate
      Operand::Imm(0)             // cpol: default cache policy
  });
}

} // namespace aegisbit
