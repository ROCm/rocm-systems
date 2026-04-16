//===-- aegisbit/InstructionBuilder.h - Instruction Builder -----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helper for building AMDGPU instructions from mnemonics.
/// This provides architecture-agnostic instruction construction for tests
/// and instrumentation code, eliminating hard-coded instruction bytes.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_INSTRUCTION_BUILDER_H
#define AEGISBIT_INSTRUCTION_BUILDER_H

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace aegisbit {

class Disassembler;

/// Register pair for 64-bit operations (e.g., SGPR[N:N+1])
struct SGPRPair {
  unsigned Lo;  ///< Lower register (even)
  unsigned Hi;  ///< Higher register (odd)

  /// Create from base index (e.g., 0 for s[0:1])
  static SGPRPair fromIndex(unsigned BaseIdx);
};

/// VGPR pair for 64-bit vector operations (e.g., VGPR[N:N+1])
struct VGPRPair {
  unsigned Lo;  ///< Lower register
  unsigned Hi;  ///< Higher register

  /// Create from base index (e.g., 0 for v[0:1])
  static VGPRPair fromIndex(unsigned BaseIdx);
};

/// Helper for constructing MCInst from instruction mnemonics
/// Usage:
///   auto Inst = InstructionBuilder::build(Disasm, "S_WAITCNT", {Imm(0)});
///   auto Bytes = Disasm.encode(*Inst);  // Get architecture-specific encoding
class InstructionBuilder {
public:
  /// Operand types for instruction construction
  enum class OperandType {
    Immediate,
    Register,
    Expression
  };

  /// Operand descriptor
  struct Operand {
    OperandType Type;
    int64_t Value;  // For immediate/register number

    static Operand Imm(int64_t Val) {
      return Operand{OperandType::Immediate, Val};
    }

    static Operand Reg(unsigned RegNum) {
      return Operand{OperandType::Register, static_cast<int64_t>(RegNum)};
    }

    /// Create a register pair operand (for 64-bit SGPR pairs)
    /// The encoding uses the pair register class (e.g., SGPR0_SGPR1)
    static Operand RegPair(const SGPRPair& Pair) {
      // For pairs, we encode the lower register - LLVM figures out the pair
      return Operand{OperandType::Register, static_cast<int64_t>(Pair.Lo)};
    }
  };

  /// Build an instruction from mnemonic and operands
  /// \param MCII Instruction info from disassembler
  /// \param Mnemonic Instruction name (e.g., "S_WAITCNT", "S_BRANCH")
  /// \param Operands List of operands (immediates, registers)
  /// \return MCInst or error if instruction not found
  static llvm::Expected<llvm::MCInst> build(
      const llvm::MCInstrInfo& MCII,
      const std::string& Mnemonic,
      const std::vector<Operand>& Operands = {});

  /// Build an instruction using a Disassembler (convenience wrapper)
  static llvm::Expected<llvm::MCInst> build(
      const Disassembler& Disasm,
      const std::string& Mnemonic,
      const std::vector<Operand>& Operands = {});

  //===--------------------------------------------------------------------===//
  // High-level instruction builders for instrumentation
  //===--------------------------------------------------------------------===//

  /// Build S_MOV_B32: Move 32-bit immediate to SGPR
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPR Destination SGPR register number (LLVM number)
  /// \param Imm 32-bit immediate value
  static llvm::Expected<llvm::MCInst> buildSMovB32(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      uint32_t Imm);

  /// Build S_ADD_U32: Add two 32-bit values
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPR Destination SGPR
  /// \param Src0SGPR First source SGPR
  /// \param Imm Immediate to add
  static llvm::Expected<llvm::MCInst> buildSAddU32(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      unsigned Src0SGPR,
      uint32_t Imm);

  /// Build S_ADD_U32 with register + register operands.
  /// \param DstSGPR Destination SGPR
  /// \param Src0SGPR First source SGPR
  /// \param Src1SGPR Second source SGPR (register, not immediate)
  static llvm::Expected<llvm::MCInst> buildSAddU32Reg(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      unsigned Src0SGPR,
      unsigned Src1SGPR);

  /// Build S_ADDC_U32: Add two 32-bit values with carry-in from SCC
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPR Destination SGPR
  /// \param Src0SGPR First source SGPR
  /// \param Imm Immediate to add (typically 0 for carry propagation)
  static llvm::Expected<llvm::MCInst> buildSAddcU32(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      unsigned Src0SGPR,
      uint32_t Imm);

  /// Build S_SUB_U32: Subtract two 32-bit values (Dst = Src0 - Src1)
  static llvm::Expected<llvm::MCInst> buildSSubU32(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      unsigned Src0SGPR,
      unsigned Src1SGPR);

  /// Build S_SUBB_U32: Subtract with borrow-in from SCC
  static llvm::Expected<llvm::MCInst> buildSSubbU32(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      unsigned Src0SGPR,
      uint32_t Imm);

  /// Build S_AND_B32: Bitwise AND of two 32-bit values
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPR Destination SGPR
  /// \param Src0SGPR First source SGPR
  /// \param Src1 Second source (SGPR or immediate)
  static llvm::Expected<llvm::MCInst> buildSAndB32(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      unsigned Src0SGPR,
      const Operand& Src1);

  /// Build S_GETREG_B32: Read hardware register to SGPR
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPR Destination SGPR register number
  /// \param HwRegEncoding Encoded hwreg value: id | (offset << 6) | ((width-1) << 11)
  ///   For HW_REG_HW_ID (full 32-bit): 4 | (0 << 6) | (31 << 11) = 0xF804
  static llvm::Expected<llvm::MCInst> buildSGetRegB32(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      uint16_t HwRegEncoding);

  /// Build S_GETPC_B64: Get current PC value
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPRPairLo Lower SGPR of destination pair (must be even)
  static llvm::Expected<llvm::MCInst> buildSGetPCB64(
      const Disassembler& Disasm,
      unsigned DstSGPRPairLo);

  /// Build S_SETPC_B64: Set PC (indirect jump)
  /// \param Disasm Disassembler for encoding
  /// \param SrcSGPRPairLo Lower SGPR of source pair holding target address
  static llvm::Expected<llvm::MCInst> buildSSetPCB64(
      const Disassembler& Disasm,
      unsigned SrcSGPRPairLo);

  /// Build S_WAITCNT: Wait for memory operations to complete
  /// \param Disasm Disassembler for encoding
  /// \param VmCnt VM (vector memory) count to wait for (0-15, 0=all complete)
  /// \param ExpCnt Export count to wait for (0-7, 0=all complete)
  /// \param LgkmCnt LGKM (scalar memory/LDS) count to wait for (0-15, 0=all complete)
  static llvm::Expected<llvm::MCInst> buildSWaitCnt(
      const Disassembler& Disasm,
      unsigned VmCnt = 0,
      unsigned ExpCnt = 0,
      unsigned LgkmCnt = 0);

  //===--------------------------------------------------------------------===//
  // SMEM (Scalar Memory) Instructions
  //===--------------------------------------------------------------------===//

  /// Build S_LOAD_DWORD: Load 1 dword from memory
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPR Destination SGPR
  /// \param BaseSGPRPairLo Base address SGPR pair (64-bit pointer)
  /// \param Offset Immediate offset in bytes
  static llvm::Expected<llvm::MCInst> buildSLoadDword(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      unsigned BaseSGPRPairLo,
      uint32_t Offset);

  /// Build S_LOAD_DWORDX2: Load 2 dwords from memory
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPRPairLo Destination SGPR pair (must be even-aligned)
  /// \param BaseSGPRPairLo Base address SGPR pair (64-bit pointer)
  /// \param Offset Immediate offset in bytes
  static llvm::Expected<llvm::MCInst> buildSLoadDwordX2(
      const Disassembler& Disasm,
      unsigned DstSGPRPairLo,
      unsigned BaseSGPRPairLo,
      uint32_t Offset);

  /// Build S_LOAD_DWORDX4: Load 4 dwords from memory
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPRQuadLo Destination SGPR quad (must be 4-aligned)
  /// \param BaseSGPRPairLo Base address SGPR pair (64-bit pointer)
  /// \param Offset Immediate offset in bytes
  static llvm::Expected<llvm::MCInst> buildSLoadDwordX4(
      const Disassembler& Disasm,
      unsigned DstSGPRQuadLo,
      unsigned BaseSGPRPairLo,
      uint32_t Offset);

  /// Build S_STORE_DWORD: Store 1 dword to memory
  /// \param Disasm Disassembler for encoding
  /// \param SrcSGPR Source SGPR to store
  /// \param BaseSGPRPairLo Base address SGPR pair (64-bit pointer)
  /// \param Offset Immediate offset in bytes
  static llvm::Expected<llvm::MCInst> buildSStoreDword(
      const Disassembler& Disasm,
      unsigned SrcSGPR,
      unsigned BaseSGPRPairLo,
      uint32_t Offset);

  /// Build S_STORE_DWORDX2: Store 2 dwords to memory
  /// \param Disasm Disassembler for encoding
  /// \param SrcSGPRPairLo Source SGPR pair
  /// \param BaseSGPRPairLo Base address SGPR pair (64-bit pointer)
  /// \param Offset Immediate offset in bytes
  static llvm::Expected<llvm::MCInst> buildSStoreDwordX2(
      const Disassembler& Disasm,
      unsigned SrcSGPRPairLo,
      unsigned BaseSGPRPairLo,
      uint32_t Offset);

  //===--------------------------------------------------------------------===//
  // VALU (Vector ALU) Instructions - M4 Memory Tracing
  //===--------------------------------------------------------------------===//

  /// Build V_MOV_B32: Move scalar value to vector register
  /// Broadcasts the scalar value to all lanes in the vector register.
  /// \param Disasm Disassembler for encoding
  /// \param DstVGPR Destination VGPR register number (LLVM number)
  /// \param SrcSGPR Source SGPR register number (LLVM number)
  static llvm::Expected<llvm::MCInst> buildVMovB32(
      const Disassembler& Disasm,
      unsigned DstVGPR,
      unsigned SrcSGPR);

  /// Build V_MOV_B32: Move immediate value to vector register
  /// Broadcasts the immediate to all lanes in the vector register.
  /// \param Disasm Disassembler for encoding
  /// \param DstVGPR Destination VGPR register number (LLVM number)
  /// \param Imm Immediate value
  static llvm::Expected<llvm::MCInst> buildVMovB32Imm(
      const Disassembler& Disasm,
      unsigned DstVGPR,
      uint32_t Imm);

  /// Build V_READFIRSTLANE_B32: Read first active lane to scalar
  /// Copies the value from the first active lane of the VGPR to an SGPR.
  /// \param Disasm Disassembler for encoding
  /// \param DstSGPR Destination SGPR register number (LLVM number)
  /// \param SrcVGPR Source VGPR register number (LLVM number)
  static llvm::Expected<llvm::MCInst> buildVReadFirstLaneB32(
      const Disassembler& Disasm,
      unsigned DstSGPR,
      unsigned SrcVGPR);

  /// Build V_ADD_CO_U32: 32-bit add with carry out to VCC
  /// Performs Dst = Src0 + Src1, VCC = carry
  /// \param Disasm Disassembler for encoding
  /// \param DstVGPR Destination VGPR
  /// \param Src0SGPR First source (SGPR - scalar broadcast to all lanes)
  /// \param Src1VGPR Second source VGPR
  static llvm::Expected<llvm::MCInst> buildVAddCOU32(
      const Disassembler& Disasm,
      unsigned DstVGPR,
      unsigned Src0SGPR,
      unsigned Src1VGPR);

  /// Build V_ADDC_U32: 32-bit add with carry in from VCC and carry out
  /// Performs Dst = Src0 + Src1 + VCC, VCC = carry
  /// \param Disasm Disassembler for encoding
  /// \param DstVGPR Destination VGPR
  /// \param Src0Imm First source (immediate, typically 0 for carry propagation)
  /// \param Src1VGPR Second source VGPR
  static llvm::Expected<llvm::MCInst> buildVAddcU32(
      const Disassembler& Disasm,
      unsigned DstVGPR,
      uint32_t Src0Imm,
      unsigned Src1VGPR);

  //===--------------------------------------------------------------------===//
  // VALU Lane ID Instructions — for per-lane address computation
  //===--------------------------------------------------------------------===//

  /// Build V_MBCNT_LO_U32_B32: Count set bits in src0 below current lane (low 32)
  /// For lanes 0-31, returns the number of set bits in src0 at positions below
  /// the current lane, plus src1. With src0=0xFFFFFFFF, src1=0, gives lane ID.
  /// \param Disasm Disassembler for encoding
  /// \param DstVGPR Destination VGPR
  /// \param Src0Imm First source (typically 0xFFFFFFFF for full lane count)
  /// \param Src1Imm Second source (typically 0, added to result)
  static llvm::Expected<llvm::MCInst> buildVMbcntLoU32B32(
      const Disassembler& Disasm,
      unsigned DstVGPR,
      uint32_t Src0Imm,
      uint32_t Src1Imm);

  /// Build V_MBCNT_HI_U32_B32: Count set bits in src0 for upper 32 lanes
  /// Continues the count from V_MBCNT_LO_U32_B32 for lanes 32-63.
  /// Result = popcount(src0 & mask_for_upper_lanes_below_current) + src1.
  /// \param Disasm Disassembler for encoding
  /// \param DstVGPR Destination VGPR
  /// \param Src0Imm First source (typically 0xFFFFFFFF)
  /// \param Src1VGPR Second source VGPR (typically result from V_MBCNT_LO)
  static llvm::Expected<llvm::MCInst> buildVMbcntHiU32B32(
      const Disassembler& Disasm,
      unsigned DstVGPR,
      uint32_t Src0Imm,
      unsigned Src1VGPR);

  /// Build V_LSHLREV_B32: Left-shift data by immediate amount
  /// Dst = Src1 << Src0 (reversed operand order: shift amount first)
  static llvm::Expected<llvm::MCInst> buildVLshlrevB32(
      const Disassembler& Disasm,
      unsigned DstVGPR,
      uint32_t ShiftAmt,
      unsigned SrcVGPR);

  /// Build V_LSHRREV_B32: Right-shift data by immediate amount
  /// Dst = Src1 >> Src0 (reversed operand order: shift amount first)
  static llvm::Expected<llvm::MCInst> buildVLshrrevB32(
      const Disassembler& Disasm,
      unsigned DstVGPR,
      uint32_t ShiftAmt,
      unsigned SrcVGPR);

  //===--------------------------------------------------------------------===//
  // EXEC Mask Manipulation — for atomic trace offset (lane 0 only)
  //===--------------------------------------------------------------------===//

  /// LLVM register numbers for special registers
  static constexpr unsigned EXEC_LO_REG = 4;
  static constexpr unsigned EXEC_HI_REG = 3;
  static constexpr unsigned VCC_LO_REG  = 47;
  static constexpr unsigned VCC_HI_REG  = 46;
  // M0_REG: use ISAEncoder::getM0Reg() instead (discovered at runtime).
  static constexpr unsigned M0_REG      = 323;

  /// Build S_MOV_B32 saving exec_lo or exec_hi to an SGPR
  static llvm::Expected<llvm::MCInst> buildSMovB32FromExec(
      const Disassembler& Disasm, unsigned DstSGPR, bool High);

  /// Build S_MOV_B32 writing an immediate to exec_lo or exec_hi
  static llvm::Expected<llvm::MCInst> buildSMovB32ToExecImm(
      const Disassembler& Disasm, uint32_t Imm, bool High);

  /// Build S_MOV_B32 restoring exec_lo or exec_hi from an SGPR
  static llvm::Expected<llvm::MCInst> buildSMovB32ToExecReg(
      const Disassembler& Disasm, unsigned SrcSGPR, bool High);

  //===--------------------------------------------------------------------===//
  // VMEM (Vector Memory) Instructions - M4 Memory Tracing
  //===--------------------------------------------------------------------===//

  /// Build GLOBAL_LOAD_DWORD with SADDR: Load 32-bit data using SGPR base address
  /// Address = saddr + vaddr + offset
  static llvm::Expected<llvm::MCInst> buildGlobalLoadDword(
      const Disassembler& Disasm,
      unsigned VDstVGPR,
      unsigned VAddrVGPR,
      unsigned SAddrSGPRLo,
      int32_t Offset = 0);

  /// Build GLOBAL_STORE_DWORD with SADDR: Store 32-bit data using SGPR base address
  /// Address = saddr + vaddr + offset
  static llvm::Expected<llvm::MCInst> buildGlobalStoreDword(
      const Disassembler& Disasm,
      unsigned VAddrVGPR,
      unsigned VDataVGPR,
      unsigned SAddrSGPRLo,
      int32_t Offset = 0);

  /// Build GLOBAL_ATOMIC_ADD with SADDR and return (RTN): Atomically add to memory
  /// Returns the OLD value before the add in VDstVGPR.
  /// Address = saddr + vaddr + offset
  /// \param VDstVGPR  Destination VGPR (receives old value)
  /// \param VAddrVGPR Single VGPR holding per-lane offset from saddr (0 for uniform)
  /// \param VDataVGPR Single VGPR holding value to add
  /// \param SAddrSGPRLo SGPR pair Lo register holding 64-bit base address
  /// \param Offset Immediate offset in bytes
  static llvm::Expected<llvm::MCInst> buildGlobalAtomicAddRtn(
      const Disassembler& Disasm,
      unsigned VDstVGPR,
      unsigned VAddrVGPR,
      unsigned VDataVGPR,
      unsigned SAddrSGPRLo,
      int32_t Offset = 0);

  /// Build GLOBAL_ATOMIC_ADD with SADDR (fire-and-forget, no return value):
  /// Atomically adds VData to memory at saddr + vaddr + offset.
  static llvm::Expected<llvm::MCInst> buildGlobalAtomicAddNoRtn(
      const Disassembler& Disasm,
      unsigned VAddrVGPR,
      unsigned VDataVGPR,
      unsigned SAddrSGPRLo,
      int32_t Offset = 0);

  /// Build GLOBAL_STORE_DWORD (non-SADDR): Store 32-bit per-lane data to global memory
  /// vaddr is a VGPR pair holding full 64-bit per-lane addresses.
  /// \param Disasm Disassembler for encoding
  /// \param VAddrLo VGPR pair low register holding per-lane 64-bit addresses
  /// \param VDataVGPR Single VGPR holding 32-bit data per lane
  /// \param Offset Immediate offset in bytes (added to address)
  static llvm::Expected<llvm::MCInst> buildGlobalStoreDwordNoSaddr(
      const Disassembler& Disasm,
      unsigned VAddrLo,
      unsigned VDataVGPR,
      int32_t Offset = 0);

  /// Build GLOBAL_STORE_DWORDX2: Store 64-bit per-lane data to global memory
  /// Each lane stores 8 bytes (2 dwords) to its computed address.
  /// \param Disasm Disassembler for encoding
  /// \param VAddrLo VGPR pair holding per-lane 64-bit addresses [VAddrLo:VAddrLo+1]
  /// \param VDataLo VGPR pair holding per-lane 64-bit data [VDataLo:VDataLo+1]
  /// \param Offset Immediate offset in bytes (added to address)
  static llvm::Expected<llvm::MCInst> buildGlobalStoreDwordX2(
      const Disassembler& Disasm,
      unsigned VAddrLo,
      unsigned VDataLo,
      int32_t Offset = 0);

  /// Build GLOBAL_LOAD_DWORDX2: Load 64-bit per-lane data from global memory
  /// Each lane loads 8 bytes (2 dwords) from its computed address.
  /// \param Disasm Disassembler for encoding
  /// \param VDstLo Destination VGPR pair [VDstLo:VDstLo+1]
  /// \param VAddrLo VGPR pair holding per-lane 64-bit addresses [VAddrLo:VAddrLo+1]
  /// \param Offset Immediate offset in bytes (added to address)
  static llvm::Expected<llvm::MCInst> buildGlobalLoadDwordX2(
      const Disassembler& Disasm,
      unsigned VDstLo,
      unsigned VAddrLo,
      int32_t Offset = 0);

  //===--------------------------------------------------------------------===//
  // Scratch Memory Instructions - for VGPR spill/restore
  //===--------------------------------------------------------------------===//

  /// Build SCRATCH_STORE_DWORD: Store VGPR to scratch memory
  /// Uses flat_scratch as base address (set by hardware).
  /// \param Disasm Disassembler for encoding
  /// \param VDataVGPR VGPR to store
  /// \param Offset Immediate offset in bytes from flat_scratch base
  static llvm::Expected<llvm::MCInst> buildScratchStoreDword(
      const Disassembler& Disasm,
      unsigned VDataVGPR,
      uint32_t Offset);

  /// Build SCRATCH_STORE_DWORD_SADDR: Store VGPR to scratch via SGPR base
  /// Address = flat_scratch + SAddrSGPR + sext(Offset).
  /// Use when offset > 4095 (exceeds signed 13-bit immediate range).
  static llvm::Expected<llvm::MCInst> buildScratchStoreDwordSAddr(
      const Disassembler& Disasm,
      unsigned VDataVGPR,
      unsigned SAddrSGPR,
      int32_t Offset);

  /// Build SCRATCH_LOAD_DWORD: Load VGPR from scratch memory
  /// Uses flat_scratch as base address (set by hardware).
  /// \param Disasm Disassembler for encoding
  /// \param VDstVGPR Destination VGPR
  /// \param Offset Immediate offset in bytes from flat_scratch base
  static llvm::Expected<llvm::MCInst> buildScratchLoadDword(
      const Disassembler& Disasm,
      unsigned VDstVGPR,
      uint32_t Offset);

  /// Build SCRATCH_LOAD_DWORD_SADDR: Load VGPR from scratch via SGPR base
  /// Address = flat_scratch + SAddrSGPR + sext(Offset).
  /// Use when offset > 4095 (exceeds signed 13-bit immediate range).
  static llvm::Expected<llvm::MCInst> buildScratchLoadDwordSAddr(
      const Disassembler& Disasm,
      unsigned VDstVGPR,
      unsigned SAddrSGPR,
      int32_t Offset);

private:
  /// Find opcode by mnemonic name
  static llvm::Expected<unsigned> findOpcodeByName(
      const llvm::MCInstrInfo& MCII,
      const std::string& Mnemonic);
};

} // namespace aegisbit

#endif // AEGISBIT_INSTRUCTION_BUILDER_H
