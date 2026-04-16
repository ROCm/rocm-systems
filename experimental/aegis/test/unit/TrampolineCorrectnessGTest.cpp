//===-- TrampolineCorrectnessGTest.cpp - Trampoline Correctness Tests -*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests that validate assumptions underlying the instrumented trampoline:
///
///   1. Encoding roundtrip: EXEC-related instructions encode to the exact
///      bytes we observe in disassembly and decode back to the same instruction.
///
///   2. Register identity: getSGPR(N) produces a register that, when used
///      as an operand in S_MOV_B32, encodes as hardware register sN.
///
///   3. SGPR conflict detection: scratch registers allocated by
///      fromDescriptorInstrumented must not alias VCC, FLAT_SCRATCH,
///      or XNACK_MASK after the descriptor bump.
///
//===----------------------------------------------------------------------===//

#include "fixtures/DisasmFixture.h"
#include "aegisbit/RegisterHelper.h"
#include "aegisbit/TrampolineBridge.h"
#include "aegisbit/CFGBuilder.h"
#include "aegisbit/Types.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <sstream>

using namespace aegisbit;
using namespace aegisbit::test;

class TrampolineCorrectnessTest : public DisasmFixture {};

//===----------------------------------------------------------------------===//
// Test 1: EXEC Instruction Encoding Roundtrip
//
// We verify that every EXEC-related instruction in the trampoline encodes
// to the EXACT bytes we observed in the llvm-objdump disassembly of the
// patched Triton kernel. If any of these fail, our EXEC_LO_REG / EXEC_HI_REG
// constants or the S_MOV_B32 encoding path has a bug.
//===----------------------------------------------------------------------===//

TEST_F(TrampolineCorrectnessTest, ExecSaveLoEncodesToExpectedBytes) {
  // s_mov_b32 s27, exec_lo  =>  expected: BE9B007E
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(RegisterHelper::getSGPR(27)),
       IB::Operand::Reg(IB::EXEC_LO_REG)});
  ASSERT_TRUE(static_cast<bool>(InstOrErr))
      << llvm::toString(InstOrErr.takeError());

  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr))
      << llvm::toString(BytesOrErr.takeError());

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);

  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  EXPECT_EQ(Word, 0xBE9B007Eu)
      << "s_mov_b32 s27, exec_lo should encode to BE9B007E, got: "
      << std::hex << Word;
}

TEST_F(TrampolineCorrectnessTest, ExecSaveHiEncodesToExpectedBytes) {
  // s_mov_b32 s28, exec_hi  =>  expected: BE9C007F
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(RegisterHelper::getSGPR(28)),
       IB::Operand::Reg(IB::EXEC_HI_REG)});
  ASSERT_TRUE(static_cast<bool>(InstOrErr))
      << llvm::toString(InstOrErr.takeError());

  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr))
      << llvm::toString(BytesOrErr.takeError());

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);

  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  EXPECT_EQ(Word, 0xBE9C007Fu)
      << "s_mov_b32 s28, exec_hi should encode to BE9C007F, got: "
      << std::hex << Word;
}

TEST_F(TrampolineCorrectnessTest, ExecSetLoEncodesToExpectedBytes) {
  // s_mov_b32 exec_lo, 1  =>  expected: BEFE0081
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(IB::EXEC_LO_REG),
       IB::Operand::Imm(1)});
  ASSERT_TRUE(static_cast<bool>(InstOrErr))
      << llvm::toString(InstOrErr.takeError());

  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr))
      << llvm::toString(BytesOrErr.takeError());

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);

  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  EXPECT_EQ(Word, 0xBEFE0081u)
      << "s_mov_b32 exec_lo, 1 should encode to BEFE0081, got: "
      << std::hex << Word;
}

TEST_F(TrampolineCorrectnessTest, ExecSetHiEncodesToExpectedBytes) {
  // s_mov_b32 exec_hi, 0  =>  expected: BEFF0080
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(IB::EXEC_HI_REG),
       IB::Operand::Imm(0)});
  ASSERT_TRUE(static_cast<bool>(InstOrErr))
      << llvm::toString(InstOrErr.takeError());

  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr))
      << llvm::toString(BytesOrErr.takeError());

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);

  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  EXPECT_EQ(Word, 0xBEFF0080u)
      << "s_mov_b32 exec_hi, 0 should encode to BEFF0080, got: "
      << std::hex << Word;
}

TEST_F(TrampolineCorrectnessTest, ExecRestoreLoEncodesToExpectedBytes) {
  // s_mov_b32 exec_lo, s27  =>  expected: BEFE001B
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(IB::EXEC_LO_REG),
       IB::Operand::Reg(RegisterHelper::getSGPR(27))});
  ASSERT_TRUE(static_cast<bool>(InstOrErr))
      << llvm::toString(InstOrErr.takeError());

  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr))
      << llvm::toString(BytesOrErr.takeError());

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);

  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  EXPECT_EQ(Word, 0xBEFE001Bu)
      << "s_mov_b32 exec_lo, s27 should encode to BEFE001B, got: "
      << std::hex << Word;
}

TEST_F(TrampolineCorrectnessTest, ExecRestoreHiEncodesToExpectedBytes) {
  // s_mov_b32 exec_hi, s28  =>  expected: BEFF001C
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(IB::EXEC_HI_REG),
       IB::Operand::Reg(RegisterHelper::getSGPR(28))});
  ASSERT_TRUE(static_cast<bool>(InstOrErr))
      << llvm::toString(InstOrErr.takeError());

  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr))
      << llvm::toString(BytesOrErr.takeError());

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);

  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  EXPECT_EQ(Word, 0xBEFF001Cu)
      << "s_mov_b32 exec_hi, s28 should encode to BEFF001C, got: "
      << std::hex << Word;
}

TEST_F(TrampolineCorrectnessTest, ExecInstructionsDecodeBackCorrectly) {
  // Verify all 6 EXEC instructions roundtrip through encode→decode
  struct ExecInstr {
    std::string Desc;
    unsigned Dst;
    unsigned Src;
    bool SrcIsImm;
    int64_t ImmVal;
  };

  std::vector<ExecInstr> Instrs = {
    {"save exec_lo to s27",  RegisterHelper::getSGPR(27), IB::EXEC_LO_REG, false, 0},
    {"save exec_hi to s28",  RegisterHelper::getSGPR(28), IB::EXEC_HI_REG, false, 0},
    {"set exec_lo = 1",      IB::EXEC_LO_REG, 0, true, 1},
    {"set exec_hi = 0",      IB::EXEC_HI_REG, 0, true, 0},
    {"restore exec_lo",      IB::EXEC_LO_REG, RegisterHelper::getSGPR(27), false, 0},
    {"restore exec_hi",      IB::EXEC_HI_REG, RegisterHelper::getSGPR(28), false, 0},
  };

  for (const auto &I : Instrs) {
    std::vector<IB::Operand> Ops;
    Ops.push_back(IB::Operand::Reg(I.Dst));
    if (I.SrcIsImm)
      Ops.push_back(IB::Operand::Imm(I.ImmVal));
    else
      Ops.push_back(IB::Operand::Reg(I.Src));

    auto InstOrErr = IB::build(*Disasm, "S_MOV_B32", Ops);
    ASSERT_TRUE(static_cast<bool>(InstOrErr))
        << "Failed to build: " << I.Desc;

    auto BytesOrErr = Disasm->encode(*InstOrErr);
    ASSERT_TRUE(static_cast<bool>(BytesOrErr))
        << "Failed to encode: " << I.Desc;

    uint64_t Size = 0;
    auto Decoded = Disasm->disassemble(*BytesOrErr, 0, Size);
    ASSERT_TRUE(static_cast<bool>(Decoded))
        << "Failed to decode: " << I.Desc;

    EXPECT_EQ(Size, BytesOrErr->size())
        << "Size mismatch after roundtrip: " << I.Desc;

    std::string Name = Disasm->getInstructionName(Decoded->Inst);
    EXPECT_NE(Name.find("S_MOV_B32"), std::string::npos)
        << "Decoded instruction is not S_MOV_B32: " << I.Desc
        << " (got " << Name << ")";
  }
}

//===----------------------------------------------------------------------===//
// Test 2: Register Identity
//
// Verify that getSGPR(N) produces a register that the hardware encodes as sN.
// We do this by encoding s_mov_b32 sN, 0 and extracting the SDST field from
// the SOP1 encoding.
//
// SOP1 encoding (GFX9): bits [22:16] = SDST (7 bits = SGPR index)
//===----------------------------------------------------------------------===//

TEST_F(TrampolineCorrectnessTest, SGPRIndexEncodesAsExpectedHardwareRegister) {
  // For each SGPR index 0..31, encode s_mov_b32 sN, 0 and verify the SDST
  // field in the binary equals N.
  for (unsigned Idx = 0; Idx < 32; ++Idx) {
    unsigned Reg = RegisterHelper::getSGPR(Idx);
    auto InstOrErr = IB::buildSMovB32(*Disasm, Reg, 0);
    ASSERT_TRUE(static_cast<bool>(InstOrErr))
        << "Failed to build s_mov_b32 s" << Idx << ", 0";

    auto BytesOrErr = Disasm->encode(*InstOrErr);
    ASSERT_TRUE(static_cast<bool>(BytesOrErr))
        << "Failed to encode s_mov_b32 s" << Idx << ", 0";

    auto &B = *BytesOrErr;
    ASSERT_EQ(B.size(), 4u)
        << "s_mov_b32 s" << Idx << ", 0 should be 4 bytes (inline constant)";

    uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
    // SOP1: bits [22:16] = SDST
    unsigned SDST = (Word >> 16) & 0x7F;
    EXPECT_EQ(SDST, Idx)
        << "s_mov_b32 s" << Idx << " should encode SDST=" << Idx
        << " but got SDST=" << SDST
        << " (word=0x" << std::hex << Word << ")";
  }
}

TEST_F(TrampolineCorrectnessTest, SGPRSourceEncodesAsExpectedHardwareRegister) {
  // Verify the SSRC0 field: encode s_mov_b32 s0, sN and extract SSRC0.
  // SOP1: bits [7:0] = SSRC0 (8 bits, 0-103 for SGPRs)
  for (unsigned Idx = 0; Idx < 32; ++Idx) {
    unsigned SrcReg = RegisterHelper::getSGPR(Idx);
    auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
        {IB::Operand::Reg(RegisterHelper::getSGPR(0)),
         IB::Operand::Reg(SrcReg)});
    ASSERT_TRUE(static_cast<bool>(InstOrErr))
        << "Failed to build s_mov_b32 s0, s" << Idx;

    auto BytesOrErr = Disasm->encode(*InstOrErr);
    ASSERT_TRUE(static_cast<bool>(BytesOrErr))
        << "Failed to encode s_mov_b32 s0, s" << Idx;

    auto &B = *BytesOrErr;
    ASSERT_EQ(B.size(), 4u);

    uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
    unsigned SSRC0 = Word & 0xFF;
    EXPECT_EQ(SSRC0, Idx)
        << "s_mov_b32 s0, s" << Idx << " should encode SSRC0=" << Idx
        << " but got SSRC0=" << SSRC0;
  }
}

TEST_F(TrampolineCorrectnessTest, ExecLoHardwareEncoding) {
  // Verify EXEC_LO encodes as hardware register 126 (SSRC0 field)
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(RegisterHelper::getSGPR(0)),
       IB::Operand::Reg(IB::EXEC_LO_REG)});
  ASSERT_TRUE(static_cast<bool>(InstOrErr));
  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr));

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);
  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  unsigned SSRC0 = Word & 0xFF;
  EXPECT_EQ(SSRC0, 126u)
      << "exec_lo should encode as SSRC0=126 (0x7E), got " << SSRC0;
}

TEST_F(TrampolineCorrectnessTest, ExecHiHardwareEncoding) {
  // Verify EXEC_HI encodes as hardware register 127 (SSRC0 field)
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(RegisterHelper::getSGPR(0)),
       IB::Operand::Reg(IB::EXEC_HI_REG)});
  ASSERT_TRUE(static_cast<bool>(InstOrErr));
  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr));

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);
  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  unsigned SSRC0 = Word & 0xFF;
  EXPECT_EQ(SSRC0, 127u)
      << "exec_hi should encode as SSRC0=127 (0x7F), got " << SSRC0;
}

TEST_F(TrampolineCorrectnessTest, ExecLoHardwareEncodingAsDestination) {
  // Verify EXEC_LO encodes as hardware SDST=126 when used as destination
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(IB::EXEC_LO_REG),
       IB::Operand::Imm(1)});
  ASSERT_TRUE(static_cast<bool>(InstOrErr));
  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr));

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);
  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  unsigned SDST = (Word >> 16) & 0x7F;
  EXPECT_EQ(SDST, 126u)
      << "exec_lo as SDST should be 126, got " << SDST;
}

TEST_F(TrampolineCorrectnessTest, ExecHiHardwareEncodingAsDestination) {
  // Verify EXEC_HI encodes as hardware SDST=127 when used as destination
  auto InstOrErr = IB::build(*Disasm, "S_MOV_B32",
      {IB::Operand::Reg(IB::EXEC_HI_REG),
       IB::Operand::Imm(0)});
  ASSERT_TRUE(static_cast<bool>(InstOrErr));
  auto BytesOrErr = Disasm->encode(*InstOrErr);
  ASSERT_TRUE(static_cast<bool>(BytesOrErr));

  auto &B = *BytesOrErr;
  ASSERT_EQ(B.size(), 4u);
  uint32_t Word = B[0] | (B[1] << 8) | (B[2] << 16) | (B[3] << 24);
  unsigned SDST = (Word >> 16) & 0x7F;
  EXPECT_EQ(SDST, 127u)
      << "exec_hi as SDST should be 127, got " << SDST;
}

//===----------------------------------------------------------------------===//
// Test 3: SGPR Conflict Detection
//
// On AMDGPU, the descriptor's SGPRCount includes "extra" SGPRs for VCC
// (and optionally FLAT_SCRATCH, XNACK_MASK). These implicit registers are
// allocated at the TOP of the SGPR range. If we bump SGPRCount and allocate
// scratch registers above the old count, they might collide with VCC's new
// position.
//
// AMDHSA ABI: SGPRCount = user_sgprs + extra_sgprs
//   extra_sgprs = 2 (VCC) + 2 (FLAT_SCRATCH if enabled) + 2 (XNACK if enabled)
//
// VCC is at s[SGPRCount-2 : SGPRCount-1] (top of allocation).
// After bumping by N, VCC moves to s[SGPRCount+N-2 : SGPRCount+N-1].
//
// Our scratch starts at s[SGPRCount]. If VCC moves to s[SGPRCount+N-2],
// then scratch registers at indices SGPRCount+N-2 and SGPRCount+N-1 alias VCC.
//===----------------------------------------------------------------------===//

// Helper: compute where VCC would be in the hardware SGPR allocation.
// The AMDHSA ABI places VCC at the top of the granulated SGPR count.
static unsigned vccLoIndex(uint32_t TotalSGPRs, uint32_t Granularity = 8) {
  uint32_t Granulated = ((TotalSGPRs + Granularity - 1) / Granularity) * Granularity;
  return Granulated - 2;
}

static unsigned flatScratchLoIndex(uint32_t TotalSGPRs, uint32_t Granularity = 8) {
  uint32_t Granulated = ((TotalSGPRs + Granularity - 1) / Granularity) * Granularity;
  return Granulated - 4;
}

TEST_F(TrampolineCorrectnessTest, ScratchSGPRsDontAliasVCC_EmptyTrampoline) {
  // Triton add_kernel: SGPRCount=24 from descriptor
  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;

  auto SR = ScratchRegisters::fromDescriptor(KD);

  // After bump: total = 24 + 2 = 26, granulated to 32
  uint32_t NewTotal = KD.SGPRCount + SR.ExtraSGPRs;  // 26
  unsigned VCCLo = vccLoIndex(NewTotal);  // granulated(26)=32, VCC at 30

  unsigned ScratchStart = KD.SGPRCount;  // 24
  unsigned ScratchEnd = ScratchStart + SR.ExtraSGPRs - 1;  // 25

  EXPECT_LT(ScratchEnd, VCCLo)
      << "Empty trampoline: scratch s[" << ScratchStart << ":" << ScratchEnd
      << "] must not overlap VCC at s[" << VCCLo << ":" << (VCCLo+1) << "]";
}

TEST_F(TrampolineCorrectnessTest, ScratchSGPRsDontAliasVCC_InstrumentedTrampoline) {
  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);

  uint32_t NewTotal = KD.SGPRCount + SR.ExtraSGPRs;
  unsigned VCCLo = vccLoIndex(NewTotal);

  // Only the 5 USED scratch registers must be below VCC.
  // ExtraSGPRs is the descriptor bump (includes padding for VCC/FS).
  unsigned LastUsedScratch = KD.SGPRCount + 4; // s[N+4] = ExecSaveSGPRHi

  EXPECT_LT(LastUsedScratch, VCCLo)
      << "Instrumented trampoline: used scratch s[" << KD.SGPRCount
      << ":" << LastUsedScratch
      << "] must not overlap VCC at s[" << VCCLo << ":" << (VCCLo+1) << "]";
}

TEST_F(TrampolineCorrectnessTest, ScratchSGPRsDontAliasFlatScratch) {
  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);

  uint32_t NewTotal = KD.SGPRCount + SR.ExtraSGPRs;
  unsigned FSLo = flatScratchLoIndex(NewTotal);

  unsigned LastUsedScratch = SR.FirstFreeSGPRIdx + 4;

  EXPECT_LT(LastUsedScratch, FSLo)
      << "Instrumented trampoline: used scratch s[" << SR.FirstFreeSGPRIdx
      << ":" << LastUsedScratch
      << "] overlaps FLAT_SCRATCH at s[" << FSLo << ":" << (FSLo+1) << "]";
}

TEST_F(TrampolineCorrectnessTest, ScratchSGPRConflictWithSmallKernel) {
  KernelDescriptor KD{};
  KD.SGPRCount = 8;
  KD.VGPRCount = 4;
  KD.VGPRGranularity = 8;

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);

  uint32_t NewTotal = KD.SGPRCount + SR.ExtraSGPRs;
  unsigned VCCLo = vccLoIndex(NewTotal);
  unsigned LastUsedScratch = KD.SGPRCount + 4; // s12

  EXPECT_LT(LastUsedScratch, VCCLo)
      << "Small kernel: used scratch ends at s" << LastUsedScratch
      << " but VCC starts at s" << VCCLo;
}

TEST_F(TrampolineCorrectnessTest, ScratchSGPRConflictAtGranularityBoundary) {
  // Edge case: SGPRCount=27. Previously this caused s[27:31] overlapping VCC.
  // After the fix, ExtraSGPRs includes headroom for VCC+FLAT_SCRATCH,
  // and scratch starts at SGPRCount - ImplicitSGPRs (reclaiming implicit slots).
  KernelDescriptor KD{};
  KD.SGPRCount = 27;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);

  uint32_t NewTotal = KD.SGPRCount + SR.ExtraSGPRs;
  unsigned VCCLo = vccLoIndex(NewTotal);
  unsigned LastUsedScratch = SR.FirstFreeSGPRIdx + 4;

  EXPECT_LT(LastUsedScratch, VCCLo)
      << "Granularity boundary: used scratch ends at s" << LastUsedScratch
      << " but VCC starts at s" << VCCLo
      << " (NewTotal=" << NewTotal << ", ExtraSGPRs=" << SR.ExtraSGPRs << ")";
}

//===----------------------------------------------------------------------===//
// Test 3b: Verify scratch never aliases VCC across a range of SGPR counts
//===----------------------------------------------------------------------===//

TEST_F(TrampolineCorrectnessTest, TritonKernelVCCConflict_GranulatedModel) {
  // Sweep SGPRCount from 8 to 96 — no configuration should alias VCC.
  for (uint32_t SGPRCount = 8; SGPRCount <= 96; SGPRCount += 1) {
    KernelDescriptor KD{};
    KD.SGPRCount = SGPRCount;
    KD.VGPRCount = 8;
    KD.VGPRGranularity = 8;

    auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
    uint32_t NewTotal = SGPRCount + SR.ExtraSGPRs;
    unsigned VCCLo = vccLoIndex(NewTotal);
    unsigned LastUsedScratch = SR.FirstFreeSGPRIdx + 4;

    EXPECT_LT(LastUsedScratch, VCCLo)
        << "SGPRCount=" << SGPRCount
        << ": scratch ends at s" << LastUsedScratch
        << " but VCC at s" << VCCLo
        << " (ExtraSGPRs=" << SR.ExtraSGPRs << ")";
  }
}

TEST_F(TrampolineCorrectnessTest, TritonKernelVCCConflict_DeclaredTotalModel) {
  // Same sweep but also check FLAT_SCRATCH doesn't overlap.
  for (uint32_t SGPRCount = 8; SGPRCount <= 96; SGPRCount += 1) {
    KernelDescriptor KD{};
    KD.SGPRCount = SGPRCount;
    KD.VGPRCount = 8;
    KD.VGPRGranularity = 8;

    auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
    uint32_t NewTotal = SGPRCount + SR.ExtraSGPRs;
    unsigned FSLo = flatScratchLoIndex(NewTotal);
    unsigned LastUsedScratch = SR.FirstFreeSGPRIdx + 4;

    EXPECT_LT(LastUsedScratch, FSLo)
        << "SGPRCount=" << SGPRCount
        << ": scratch ends at s" << LastUsedScratch
        << " but FLAT_SCRATCH at s" << FSLo
        << " (ExtraSGPRs=" << SR.ExtraSGPRs << ")";
  }
}

//===----------------------------------------------------------------------===//
// Test 3c: XNACK_MASK conflict detection (gfx940+/CDNA3+)
//
// On gfx940+, ArchitectedFlatScratch reserves 6 implicit SGPRs at the top
// of the allocation: VCC(2) + FLAT_SCRATCH(2) + XNACK_MASK(2).
// The ImplicitSGPRs=4 bug caused SwapTargetSGPR to overlap XNACK_MASK,
// corrupting the precomputed island address at runtime.
//===----------------------------------------------------------------------===//

static unsigned xnackMaskLoIndex(uint32_t TotalSGPRs, uint32_t Granularity = 8) {
  uint32_t Granulated = ((TotalSGPRs + Granularity - 1) / Granularity) * Granularity;
  return Granulated - 6;
}

TEST_F(TrampolineCorrectnessTest, ScratchSGPRsDontAliasXnackMask_Instrumented_GFX950) {
  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;
  KD.ImplicitSGPRs = 6; // gfx940+

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
  uint32_t NewTotal = KD.SGPRCount + SR.ExtraSGPRs;
  unsigned XnackLo = xnackMaskLoIndex(NewTotal);
  unsigned LastUsedScratch = SR.FirstFreeSGPRIdx + 5; // 6 scratch SGPRs

  EXPECT_LT(LastUsedScratch, XnackLo)
      << "GFX950 instrumented: scratch ends at s" << LastUsedScratch
      << " but XNACK_MASK starts at s" << XnackLo
      << " (ScratchBase=" << SR.FirstFreeSGPRIdx
      << ", NewTotal=" << NewTotal << ")";
}

TEST_F(TrampolineCorrectnessTest, ScratchSGPRsDontAliasXnackMask_SwapPC_GFX950) {
  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;
  KD.ImplicitSGPRs = 6; // gfx940+

  auto SR = ScratchRegisters::fromDescriptorSwapPC(KD);
  uint32_t NewTotal = KD.SGPRCount + SR.ExtraSGPRs;
  unsigned XnackLo = xnackMaskLoIndex(NewTotal);
  unsigned SwapTargetIdx = SR.FirstFreeSGPRIdx + 7; // 8 scratch SGPRs, last is SwapTargetSGPRHi

  EXPECT_LT(SwapTargetIdx, XnackLo)
      << "GFX950 SwapPC: SwapTargetSGPRHi at s" << SwapTargetIdx
      << " overlaps XNACK_MASK at s" << XnackLo
      << " (ScratchBase=" << SR.FirstFreeSGPRIdx
      << ", NewTotal=" << NewTotal << ")";
}

TEST_F(TrampolineCorrectnessTest, SwapPCScratchNeverAliasesImplicits_GFX950_Sweep) {
  for (uint32_t SGPRCount = 8; SGPRCount <= 96; SGPRCount += 1) {
    KernelDescriptor KD{};
    KD.SGPRCount = SGPRCount;
    KD.VGPRCount = 8;
    KD.VGPRGranularity = 8;
    KD.ImplicitSGPRs = 6; // gfx940+

    auto SR = ScratchRegisters::fromDescriptorSwapPC(KD);
    uint32_t NewTotal = SGPRCount + SR.ExtraSGPRs;
    unsigned XnackLo = xnackMaskLoIndex(NewTotal);
    unsigned FSLo = flatScratchLoIndex(NewTotal);
    unsigned VCCLo = vccLoIndex(NewTotal);
    unsigned SwapTargetIdx = SR.FirstFreeSGPRIdx + 7;

    EXPECT_LT(SwapTargetIdx, XnackLo)
        << "SGPRCount=" << SGPRCount
        << ": SwapPC scratch ends at s" << SwapTargetIdx
        << " but XNACK_MASK at s" << XnackLo;
    EXPECT_LT(SwapTargetIdx, FSLo)
        << "SGPRCount=" << SGPRCount
        << ": SwapPC scratch ends at s" << SwapTargetIdx
        << " but FLAT_SCRATCH at s" << FSLo;
    EXPECT_LT(SwapTargetIdx, VCCLo)
        << "SGPRCount=" << SGPRCount
        << ": SwapPC scratch ends at s" << SwapTargetIdx
        << " but VCC at s" << VCCLo;
  }
}

TEST_F(TrampolineCorrectnessTest, InstrumentedScratchNeverAliasesImplicits_GFX950_Sweep) {
  for (uint32_t SGPRCount = 8; SGPRCount <= 96; SGPRCount += 1) {
    KernelDescriptor KD{};
    KD.SGPRCount = SGPRCount;
    KD.VGPRCount = 8;
    KD.VGPRGranularity = 8;
    KD.ImplicitSGPRs = 6; // gfx940+

    auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
    uint32_t NewTotal = SGPRCount + SR.ExtraSGPRs;
    unsigned XnackLo = xnackMaskLoIndex(NewTotal);
    unsigned LastUsedScratch = SR.FirstFreeSGPRIdx + 5;

    EXPECT_LT(LastUsedScratch, XnackLo)
        << "SGPRCount=" << SGPRCount
        << ": instrumented scratch ends at s" << LastUsedScratch
        << " but XNACK_MASK at s" << XnackLo;
  }
}

//===----------------------------------------------------------------------===//
// Test 4: End-to-End Trampoline Verification
//
// Feed a synthetic kernel through the ACTUAL buildInstrumented pipeline,
// then disassemble every instruction in the resulting trampoline island.
// Verify the instruction sequence matches the expected pattern.
//
// This catches bugs that primitive encoding tests miss:
//   - Wrong register passed to buildAndEmit at assembly time
//   - Wrong operand order
//   - Missing or extra instructions
//   - Incorrect displaced instruction bytes
//===----------------------------------------------------------------------===//

class TrampolineE2ETest : public ::testing::Test {
protected:
  std::unique_ptr<Disassembler> Disasm;

  void SetUp() override {
    auto D = Disassembler::create();
    ASSERT_TRUE(static_cast<bool>(D))
        << "Failed to create disassembler: "
        << llvm::toString(D.takeError());
    Disasm = std::move(*D);
  }

  /// Helper: encode one instruction via InstructionBuilder.
  std::vector<uint8_t> enc(const std::string &Mnemonic,
                            std::initializer_list<IB::Operand> Ops) {
    auto I = IB::build(*Disasm, Mnemonic, std::vector<IB::Operand>(Ops));
    if (!I) { llvm::consumeError(I.takeError()); return {}; }
    auto B = Disasm->encode(*I);
    if (!B) { llvm::consumeError(B.takeError()); return {}; }
    return *B;
  }

  /// Helper: disassemble all instructions from a byte buffer, return their
  /// mnemonic names (e.g., "S_MOV_B32_gfx9", "V_WRITELANE_B32").
  std::vector<std::string> disasmNames(llvm::ArrayRef<uint8_t> Code,
                                        uint64_t BaseAddr = 0) {
    std::vector<std::string> Names;
    uint64_t Offset = 0;
    while (Offset < Code.size()) {
      uint64_t Size = 0;
      auto DI = Disasm->disassemble(
          Code.slice(Offset), BaseAddr + Offset, Size);
      if (!DI) {
        llvm::consumeError(DI.takeError());
        Names.push_back("<DECODE_ERROR>");
        break;
      }
      Names.push_back(Disasm->getInstructionName(DI->Inst));
      Offset += Size;
    }
    return Names;
  }

  /// Helper: check if a name contains a substring.
  static bool contains(const std::string &Haystack, const std::string &Needle) {
    return Haystack.find(Needle) != std::string::npos;
  }
};

TEST_F(TrampolineE2ETest, InstrumentedTrampolineInstructionSequence) {
  // --- Step 1: Build a synthetic kernel with a buffer_load_dword ---
  //
  // Minimal kernel:
  //   s_nop 0                          (padding to offset 0)
  //   buffer_load_dword v1, v0, s[0:3], 0 offen   (8 bytes)
  //   s_endpgm
  //
  // The buffer_load is the instruction we want to instrument.

  auto SNop = enc("S_NOP", {IB::Operand::Imm(0)});
  ASSERT_EQ(SNop.size(), 4u);

  // buffer_load_dword v1, v0, s[0:3], 0 offen = E0501000 80000100
  std::vector<uint8_t> BufLoad = {0x00, 0x10, 0x50, 0xE0,
                                   0x00, 0x01, 0x00, 0x80};
  auto SEndpgm = enc("S_ENDPGM", {IB::Operand::Imm(0)});
  ASSERT_FALSE(SEndpgm.empty());

  std::vector<uint8_t> KernelCode;
  KernelCode.insert(KernelCode.end(), SNop.begin(), SNop.end());       // offset 0
  KernelCode.insert(KernelCode.end(), BufLoad.begin(), BufLoad.end()); // offset 4
  KernelCode.insert(KernelCode.end(), SEndpgm.begin(), SEndpgm.end()); // offset 12

  // --- Step 2: Find memory sites via the real findMemorySites ---
  uint64_t BaseAddr = 0;
  CFGBuilder Builder(*Disasm);
  auto CFGOrErr = Builder.build(KernelCode, BaseAddr);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  auto Sites = TrampolineBridge::findMemorySites(*CFGOrErr, BaseAddr, *Disasm);
  ASSERT_EQ(Sites.size(), 1u)
      << "Expected 1 buffer_load site in synthetic kernel, got " << Sites.size();

  const auto &Site = Sites[0];
  EXPECT_EQ(Site.Offset, 4u) << "buffer_load should be at offset 4";
  EXPECT_EQ(Site.OrigInstSize, 8u) << "buffer_load_dword is 8 bytes";
  EXPECT_TRUE(Site.IsLoad) << "buffer_load should be classified as load";

  // --- Step 3: Set up ScratchRegisters and TraceConfig ---
  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;

  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;

  // --- Step 4: Build the instrumented trampoline ---
  auto BridgeOrErr = TrampolineBridge::create("gfx950", *Disasm);
  ASSERT_TRUE(static_cast<bool>(BridgeOrErr))
      << llvm::toString(BridgeOrErr.takeError());

  // Use a larger TextSectionSize so the island fits
  uint64_t TextSize = 4096;
  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      KernelCode, BaseAddr, TextSize, Sites, Scratch, Trace);
  ASSERT_TRUE(static_cast<bool>(ResultOrErr))
      << llvm::toString(ResultOrErr.takeError());

  BridgeResult &BR = *ResultOrErr;
  ASSERT_EQ(BR.PatchedCount, 1u) << "Should patch exactly 1 site";
  ASSERT_EQ(BR.Slots.size(), 1u);
  ASSERT_FALSE(BR.Islands.empty()) << "Island should contain trampoline bytes";
  ASSERT_FALSE(BR.Islands[0].Bytes.empty()) << "Island should contain trampoline bytes";

  // --- Step 5: Disassemble the full island and verify instruction sequence ---
  // In shared-body architecture, Slot.TrampolineBytes only holds the 12-byte
  // dispatch entry. The full instrumentation code is in the island bytes.
  const auto &Slot = BR.Slots[0];
  const auto &IslandBytes = BR.Islands[0].Bytes;
  auto TrampolineNames = disasmNames(IslandBytes);

  // Print the full sequence for diagnostic visibility
  std::ostringstream Dump;
  Dump << "Trampoline instruction sequence (" << TrampolineNames.size() << " instructions):\n";
  for (size_t i = 0; i < TrampolineNames.size(); ++i) {
    Dump << "  [" << i << "] " << TrampolineNames[i] << "\n";
  }
  SCOPED_TRACE(Dump.str());

  // With shared-body architecture, the island contains:
  //   [DispatchTable] [SharedBodyVMEM] [SharedBodyLDS] [ReturnTable]
  // Dispatch entries: s_mov_b32 (packed_info) + s_branch (to shared body).
  // The shared body has: save RA, unpack packed_info, save SCC/VCC/EXEC,
  //   VGPR copy jump table, cache-line computation, payload, restore,
  //   computed jump.
  // Return entries have: s_and_b32 (SCC restore) + displaced + s_setpc.

  ASSERT_GE(TrampolineNames.size(), 20u)
      << "Island should have at least 20 instructions";

  // First two instructions: dispatch entry (s_mov_b32 packed_info + s_branch)
  EXPECT_TRUE(contains(TrampolineNames[0], "S_MOV_B32"))
      << "[0] expected S_MOV_B32 (dispatch packed_info), got: " << TrampolineNames[0];
  EXPECT_TRUE(contains(TrampolineNames[1], "S_BRANCH"))
      << "[1] expected S_BRANCH (to shared body), got: " << TrampolineNames[1];

  // SCC save (s_cselect_b32) should exist somewhere in the shared body
  bool FoundSCSelect = false;
  for (const auto &N : TrampolineNames) {
    if (contains(N, "S_CSELECT_B32")) { FoundSCSelect = true; break; }
  }
  EXPECT_TRUE(FoundSCSelect) << "Shared body should contain S_CSELECT_B32 (SCC save)";

  // EXEC save: s_mov_b32 instructions with exec_lo/exec_hi as sources
  unsigned ExecSaveLoIdx = RegisterHelper::getSGPRIndex(Scratch.ExecSaveSGPRLo);
  unsigned ExecSaveHiIdx = RegisterHelper::getSGPRIndex(Scratch.ExecSaveSGPRHi);
  uint32_t ExpectedExecSaveLo = 0xBE800000u | (ExecSaveLoIdx << 16) | 0x7Eu;
  uint32_t ExpectedExecSaveHi = 0xBE800000u | (ExecSaveHiIdx << 16) | 0x7Fu;

  int ExecSaveLoPos = -1, ExecSaveHiPos = -1;
  {
    uint64_t Off = 0;
    for (size_t i = 0; i < TrampolineNames.size(); ++i) {
      uint64_t Sz = 0;
      auto DI = Disasm->disassemble(
          llvm::ArrayRef<uint8_t>(IslandBytes).slice(Off), 0, Sz);
      if (!DI) { llvm::consumeError(DI.takeError()); break; }

      if (contains(TrampolineNames[i], "S_MOV_B32") && Sz == 4) {
        uint32_t W = IslandBytes[Off]
                   | (IslandBytes[Off+1] << 8)
                   | (IslandBytes[Off+2] << 16)
                   | (IslandBytes[Off+3] << 24);
        if (W == ExpectedExecSaveLo && ExecSaveLoPos < 0)
          ExecSaveLoPos = static_cast<int>(i);
        if (W == ExpectedExecSaveHi && ExecSaveHiPos < 0)
          ExecSaveHiPos = static_cast<int>(i);
      }
      Off += Sz;
    }
  }
  EXPECT_GE(ExecSaveLoPos, 0)
      << "Island should contain s_mov_b32 s" << ExecSaveLoIdx
      << ", exec_lo (0x" << std::hex << ExpectedExecSaveLo << ")";
  EXPECT_GE(ExecSaveHiPos, 0)
      << "Island should contain s_mov_b32 s" << ExecSaveHiIdx
      << ", exec_hi (0x" << std::hex << ExpectedExecSaveHi << ")";

  // Verify payload: load/store instructions exist in the island
  bool FoundLoad = false, FoundStore = false;
  for (const auto &N : TrampolineNames) {
    if (contains(N, "GLOBAL_LOAD_DWORD") || contains(N, "S_LOAD_DWORD"))
      FoundLoad = true;
    if (contains(N, "GLOBAL_STORE_DWORD"))
      FoundStore = true;
  }
  EXPECT_TRUE(FoundLoad) << "Island should contain a load instruction (payload)";
  EXPECT_TRUE(FoundStore) << "Island should contain a store instruction (payload)";

  // EXEC restore instructions should exist
  uint32_t ExpectedExecRestoreLo = 0xBEFE0000u | ExecSaveLoIdx;
  uint32_t ExpectedExecRestoreHi = 0xBEFF0000u | ExecSaveHiIdx;
  int ExecRestoreLoPos = -1, ExecRestoreHiPos = -1;
  {
    uint64_t Off = 0;
    for (size_t i = 0; i < TrampolineNames.size(); ++i) {
      uint64_t Sz = 0;
      auto DI = Disasm->disassemble(
          llvm::ArrayRef<uint8_t>(IslandBytes).slice(Off), 0, Sz);
      if (!DI) { llvm::consumeError(DI.takeError()); break; }
      if (contains(TrampolineNames[i], "S_MOV_B32") && Sz == 4) {
        uint32_t W = IslandBytes[Off]
                   | (IslandBytes[Off+1] << 8)
                   | (IslandBytes[Off+2] << 16)
                   | (IslandBytes[Off+3] << 24);
        if (W == ExpectedExecRestoreLo && ExecRestoreLoPos < 0)
          ExecRestoreLoPos = static_cast<int>(i);
        if (W == ExpectedExecRestoreHi && ExecRestoreHiPos < 0)
          ExecRestoreHiPos = static_cast<int>(i);
      }
      Off += Sz;
    }
  }
  EXPECT_GE(ExecRestoreLoPos, 0)
      << "Island must contain s_mov_b32 exec_lo, s" << ExecSaveLoIdx;
  EXPECT_GE(ExecRestoreHiPos, 0)
      << "Island must contain s_mov_b32 exec_hi, s" << ExecSaveHiIdx;

  // Return entry tail: the island ends with return table entries.
  // Last instruction is S_SETPC_B64, and a BUFFER_LOAD_DWORD should appear
  // in the return entry (displaced instruction).
  size_t Last = TrampolineNames.size() - 1;
  EXPECT_TRUE(contains(TrampolineNames[Last], "S_SETPC_B64"))
      << "Last instruction should be S_SETPC_B64, got: " << TrampolineNames[Last];

  // The displaced BUFFER_LOAD_DWORD should exist in the return table
  bool FoundDisplaced = false;
  for (const auto &N : TrampolineNames) {
    if (contains(N, "BUFFER_LOAD_DWORD")) { FoundDisplaced = true; break; }
  }
  EXPECT_TRUE(FoundDisplaced)
      << "Island should contain displaced BUFFER_LOAD_DWORD in return entry";

  // Verify displaced instruction bytes exist in the return table portion
  // The return table starts after shared body + dispatch table.
  // Each return entry: s_and_b32(4) + displaced(8) + s_setpc(4) = 16 bytes.
  {
    size_t ReturnEntryStart = IslandBytes.size() - 16;
    // Displaced instruction at offset +4 in the return entry (after s_and_b32)
    std::vector<uint8_t> DisplacedBytes(
        IslandBytes.begin() + ReturnEntryStart + 4,
        IslandBytes.begin() + ReturnEntryStart + 4 + 8);
    EXPECT_EQ(DisplacedBytes, BufLoad)
        << "Displaced instruction bytes must be identical to original";
  }

  // In shared-body architecture, the return entry starts with S_AND_B32
  // (SCC restore from bit 0 of return address), followed by the displaced
  // instruction and S_SETPC_B64. Verify the return entry structure:
  // [Last-4] S_AND_B32 (extract SCC from bit 0)
  // [Last-3] S_ANDN2_B32 (clear SCC tag from return addr)
  // [Last-2] S_CMP_LG_U32 (restore SCC)
  // [Last-1] BUFFER_LOAD_DWORD (displaced instruction)
  // [Last]   S_SETPC_B64 (return to kernel)
  EXPECT_TRUE(contains(TrampolineNames[Last - 4], "S_AND_B32"))
      << "[" << (Last-4) << "] expected S_AND_B32 (SCC extract), got: "
      << TrampolineNames[Last - 4];
  EXPECT_TRUE(contains(TrampolineNames[Last - 3], "S_ANDN2_B32"))
      << "[" << (Last-3) << "] expected S_ANDN2_B32 (clear tag), got: "
      << TrampolineNames[Last - 3];
  EXPECT_TRUE(contains(TrampolineNames[Last - 2], "S_CMP_LG_U32"))
      << "[" << (Last-2) << "] expected S_CMP_LG_U32 (SCC restore), got: "
      << TrampolineNames[Last - 2];

  // --- Step 6: Verify the patch site ---
  // The patch should be s_call_b64 (4 bytes) + s_nop (4 bytes) = 8 bytes
  ASSERT_EQ(Slot.PatchBytes.size(), 8u)
      << "Patch for 8-byte instruction should be s_call_b64 + s_nop = 8 bytes";

  // Decode the patch: first 4 bytes should be s_call_b64
  {
    uint64_t Sz = 0;
    auto PatchDI = Disasm->disassemble(Slot.PatchBytes, Site.Address, Sz);
    ASSERT_TRUE(static_cast<bool>(PatchDI))
        << "Failed to decode patch bytes";
    std::string PatchName = Disasm->getInstructionName(PatchDI->Inst);
    EXPECT_TRUE(contains(PatchName, "S_CALL_B64"))
        << "Patch should be S_CALL_B64, got: " << PatchName;
  }
}

//===----------------------------------------------------------------------===//
// Test 5: ZeroSGPR Instrumented Trampoline E2E
//===----------------------------------------------------------------------===//

TEST_F(TrampolineE2ETest, ZeroSGPRInstrumentedE2E) {
  auto SNop = enc("S_NOP", {IB::Operand::Imm(0)});
  ASSERT_EQ(SNop.size(), 4u);

  std::vector<uint8_t> BufLoad = {0x00, 0x10, 0x50, 0xE0,
                                   0x00, 0x01, 0x00, 0x80};
  auto SEndpgm = enc("S_ENDPGM", {IB::Operand::Imm(0)});
  ASSERT_FALSE(SEndpgm.empty());

  std::vector<uint8_t> KernelCode;
  KernelCode.insert(KernelCode.end(), SNop.begin(), SNop.end());
  KernelCode.insert(KernelCode.end(), BufLoad.begin(), BufLoad.end());
  KernelCode.insert(KernelCode.end(), SEndpgm.begin(), SEndpgm.end());

  uint64_t BaseAddr = 0;
  CFGBuilder Builder(*Disasm);
  auto CFGOrErr = Builder.build(KernelCode, BaseAddr);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  auto Sites = TrampolineBridge::findMemorySites(*CFGOrErr, BaseAddr, *Disasm);
  ASSERT_GE(Sites.size(), 1u);

  const auto &Site = Sites[0];

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto BridgeOrErr = TrampolineBridge::create("gfx950", *Disasm);
  ASSERT_TRUE(static_cast<bool>(BridgeOrErr))
      << llvm::toString(BridgeOrErr.takeError());

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      KernelCode, BaseAddr, 4096, Sites, Scratch, Trace);
  ASSERT_TRUE(static_cast<bool>(ResultOrErr))
      << llvm::toString(ResultOrErr.takeError());

  BridgeResult &BR = *ResultOrErr;
  ASSERT_GE(BR.PatchedCount, 1u);
  ASSERT_GE(BR.Slots.size(), 1u);
  ASSERT_FALSE(BR.Islands.empty());

  const auto &Slot = BR.Slots[0];
  auto TrampolineNames = disasmNames(Slot.TrampolineBytes);

  std::ostringstream Dump;
  Dump << "ZeroSGPR trampoline (" << TrampolineNames.size() << " instructions):\n";
  for (size_t i = 0; i < TrampolineNames.size(); ++i)
    Dump << "  [" << i << "] " << TrampolineNames[i] << "\n";
  SCOPED_TRACE(Dump.str());

  ASSERT_GE(TrampolineNames.size(), 15u);

  // Patch should be s_branch (not s_call_b64) for ZeroSGPR
  {
    uint64_t Sz = 0;
    auto PatchDI = Disasm->disassemble(Slot.PatchBytes, Site.Address, Sz);
    ASSERT_TRUE(static_cast<bool>(PatchDI));
    std::string PatchName = Disasm->getInstructionName(PatchDI->Inst);
    EXPECT_TRUE(contains(PatchName, "S_BRANCH"))
        << "ZeroSGPR patch should be S_BRANCH, got: " << PatchName;
  }

  // ZeroSGPR trampoline should NOT start with v_writelane for return address
  // (no s_call_b64 used, so no return address to save)
  // It should start with VCC/SCC save sequence
  EXPECT_TRUE(contains(TrampolineNames[0], "S_CSELECT_B32") ||
              contains(TrampolineNames[0], "V_WRITELANE_B32"))
      << "[0] expected SCC or VCC save, got: " << TrampolineNames[0];

  // Last instruction should be s_branch (return), not s_setpc_b64
  size_t Last = TrampolineNames.size() - 1;
  EXPECT_TRUE(contains(TrampolineNames[Last], "S_BRANCH"))
      << "ZeroSGPR trampoline should end with S_BRANCH, got: "
      << TrampolineNames[Last];
  EXPECT_FALSE(contains(TrampolineNames[Last], "S_SETPC"))
      << "ZeroSGPR trampoline must NOT end with S_SETPC_B64";
}

//===----------------------------------------------------------------------===//
// Test 6: Save/Restore Symmetry Checker
//===----------------------------------------------------------------------===//

TEST_F(TrampolineE2ETest, SaveRestoreSymmetry) {
  auto SNop = enc("S_NOP", {IB::Operand::Imm(0)});
  ASSERT_EQ(SNop.size(), 4u);

  std::vector<uint8_t> BufLoad = {0x00, 0x10, 0x50, 0xE0,
                                   0x00, 0x01, 0x00, 0x80};
  auto SEndpgm = enc("S_ENDPGM", {IB::Operand::Imm(0)});
  ASSERT_FALSE(SEndpgm.empty());

  std::vector<uint8_t> KernelCode;
  KernelCode.insert(KernelCode.end(), SNop.begin(), SNop.end());
  KernelCode.insert(KernelCode.end(), BufLoad.begin(), BufLoad.end());
  KernelCode.insert(KernelCode.end(), SEndpgm.begin(), SEndpgm.end());

  uint64_t BaseAddr = 0;
  CFGBuilder Builder(*Disasm);
  auto CFGOrErr = Builder.build(KernelCode, BaseAddr);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr));

  auto Sites = TrampolineBridge::findMemorySites(*CFGOrErr, BaseAddr, *Disasm);
  ASSERT_GE(Sites.size(), 1u);

  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;

  auto BridgeOrErr = TrampolineBridge::create("gfx950", *Disasm);
  ASSERT_TRUE(static_cast<bool>(BridgeOrErr));

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      KernelCode, BaseAddr, 4096, Sites, Scratch, Trace);
  ASSERT_TRUE(static_cast<bool>(ResultOrErr));

  ASSERT_FALSE(ResultOrErr->Islands.empty());
  const auto &IslandBytes = ResultOrErr->Islands[0].Bytes;

  // In shared-body architecture, the full instrumentation code is in the island.
  struct LaneOp {
    unsigned InstrIdx;
    std::string Name;
  };

  std::vector<LaneOp> Writes, Reads;
  auto Names = disasmNames(IslandBytes);

  for (size_t i = 0; i < Names.size(); ++i) {
    if (contains(Names[i], "V_WRITELANE_B32"))
      Writes.push_back({static_cast<unsigned>(i), Names[i]});
    else if (contains(Names[i], "V_READLANE_B32"))
      Reads.push_back({static_cast<unsigned>(i), Names[i]});
  }

  // There must be writelane saves and readlane restores for at least
  // the return address pair (RA_lo, RA_hi) and VCC save/restore.
  EXPECT_GE(Writes.size(), 2u)
      << "Expected at least 2 v_writelane saves (RA_lo, RA_hi)";
  EXPECT_GE(Reads.size(), 2u)
      << "Expected at least 2 v_readlane restores";

  // The counts should be approximately equal. The payload section may use
  // a small number of extra readlane/writelane instructions for its own
  // purposes (e.g., reading buffer addresses), so allow a small margin.
  size_t Diff = Writes.size() > Reads.size()
              ? Writes.size() - Reads.size()
              : Reads.size() - Writes.size();
  EXPECT_LE(Diff, 4u)
      << "v_writelane/v_readlane count difference should be small. "
      << "Writes=" << Writes.size() << " Reads=" << Reads.size();

  // Prologue saves (first few writelanes) should come before the last
  // epilogue restores (last few readlanes). We check that the first
  // writelane save comes before the last readlane restore.
  if (!Writes.empty() && !Reads.empty()) {
    EXPECT_LT(Writes.front().InstrIdx, Reads.back().InstrIdx)
        << "First writelane save should precede last readlane restore";
  }
}
