//===-- ScratchRegisters.cpp - Scratch Register Allocation ------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of ScratchRegisters. The factory methods describe the
/// four trampoline allocation strategies (empty, instrumented, SwapPC,
/// zero-SGPR) as a pure function of KernelDescriptor. The instance
/// methods (refineScratchVGPRs / setupAccVGPRSpill / setupScratchSpill)
/// consume a decoded CFG / disassembler and live here because they are
/// strictly analysis concerns shared between the patcher and tests.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/ScratchRegisters.h"

#include "aegisbit/Disassembler.h"
#include "aegisbit/RegisterHelper.h"

#include "llvm/MC/MCRegisterInfo.h"

#include <algorithm>
#include <bitset>
#include <vector>

using namespace llvm;

namespace aegisbit {

//===----------------------------------------------------------------------===//
// Factory methods
//===----------------------------------------------------------------------===//

ScratchRegisters
ScratchRegisters::fromDescriptor(const KernelDescriptor &KD) {
  ScratchRegisters SR;
  SR.FirstFreeSGPRIdx = KD.SGPRCount;
  SR.FirstFreeVGPRIdx = KD.VGPRCount;

  SR.ReturnAddrSGPR   = RegisterHelper::getSGPR(KD.SGPRCount);
  SR.ReturnAddrSGPRHi = RegisterHelper::getSGPR(KD.SGPRCount + 1);
  SR.ScratchVGPR      = RegisterHelper::getVGPR(KD.VGPRCount);
  SR.ExtraSGPRs = 2;
  SR.ExtraVGPRs = 1;
  return SR;
}

ScratchRegisters
ScratchRegisters::fromDescriptorInstrumented(const KernelDescriptor &KD) {
  ScratchRegisters SR;
  SR.HasAccumVGPRs = (KD.AccumOffset > 0);

  static constexpr uint32_t ScratchSGPRCount = 6;
  const uint32_t ImplicitSGPRs = KD.ImplicitSGPRs;
  static constexpr uint32_t SGPRGranularity = 8;

  // KD.SGPRCount includes implicit SGPRs at the top.  When we expand
  // the allocation, those implicit regs move up, so we can reclaim
  // their old slots.  Place scratch starting at SGPRCount - ImplicitSGPRs.
  uint32_t ScratchBase = KD.SGPRCount - ImplicitSGPRs;
  SR.FirstFreeSGPRIdx = ScratchBase;

  uint32_t MinTotal = ScratchBase + ScratchSGPRCount + ImplicitSGPRs;
  uint32_t Granulated = ((MinTotal + SGPRGranularity - 1) / SGPRGranularity)
                        * SGPRGranularity;
  uint32_t ExtraSGPRs = Granulated - KD.SGPRCount;

  SR.ReturnAddrSGPR   = RegisterHelper::getSGPR(ScratchBase);
  SR.ReturnAddrSGPRHi = RegisterHelper::getSGPR(ScratchBase + 1);
  SR.ScratchSGPR      = RegisterHelper::getSGPR(ScratchBase + 2);
  SR.ExecSaveSGPRLo   = RegisterHelper::getSGPR(ScratchBase + 3);
  SR.ExecSaveSGPRHi   = RegisterHelper::getSGPR(ScratchBase + 4);
  SR.SAddrTempSGPR    = RegisterHelper::getSGPR(ScratchBase + 5);

  if (KD.AccumOffset > 0 && KD.AccumOffset <= 256) {
    // gfx942/gfx950 with AccVGPRs: VOP/FLAT VDST is 8 bits (v0-v255 only).
    // Scratch VGPRs will first be searched for via refineScratchVGPRs().
    // If none are free, we fall back to spilling 3 "victim" VGPRs to
    // unused AccVGPR slots at trampoline entry/exit.
    SR.FirstFreeVGPRIdx = KD.AccumOffset - 3; // placeholder
    SR.ScratchVGPR = 0; // set by refineScratchVGPRs() or setupAccVGPRSpill()
    SR.LaneVGPR    = 0;
    SR.TempVGPR    = 0;
    SR.ExtraVGPRs = 0;
  } else {
    SR.FirstFreeVGPRIdx = KD.VGPRCount;
    SR.ScratchVGPR = RegisterHelper::getVGPR(KD.VGPRCount);
    SR.LaneVGPR    = RegisterHelper::getVGPR(KD.VGPRCount + 1);
    SR.TempVGPR    = RegisterHelper::getVGPR(KD.VGPRCount + 2);
    SR.ExtraVGPRs = 3;
  }

  SR.ExtraSGPRs = ExtraSGPRs;
  return SR;
}

ScratchRegisters
ScratchRegisters::fromDescriptorSwapPC(const KernelDescriptor &KD) {
  ScratchRegisters SR;
  SR.HasAccumVGPRs = (KD.AccumOffset > 0);
  SR.UseSwapPC = true;

  static constexpr uint32_t ScratchSGPRCount = 8;
  const uint32_t ImplicitSGPRs = KD.ImplicitSGPRs;
  static constexpr uint32_t SGPRGranularity = 8;

  uint32_t ScratchBase = KD.SGPRCount - ImplicitSGPRs;
  SR.FirstFreeSGPRIdx = ScratchBase;

  uint32_t MinTotal = ScratchBase + ScratchSGPRCount + ImplicitSGPRs;
  uint32_t Granulated =
      ((MinTotal + SGPRGranularity - 1) / SGPRGranularity) * SGPRGranularity;
  uint32_t ExtraSGPRs = Granulated - KD.SGPRCount;

  SR.ReturnAddrSGPR = RegisterHelper::getSGPR(ScratchBase);
  SR.ReturnAddrSGPRHi = RegisterHelper::getSGPR(ScratchBase + 1);
  SR.ScratchSGPR = RegisterHelper::getSGPR(ScratchBase + 2);
  SR.ExecSaveSGPRLo = RegisterHelper::getSGPR(ScratchBase + 3);
  SR.ExecSaveSGPRHi = RegisterHelper::getSGPR(ScratchBase + 4);
  SR.SAddrTempSGPR = RegisterHelper::getSGPR(ScratchBase + 5);
  SR.SwapTargetSGPR = RegisterHelper::getSGPR(ScratchBase + 6);
  SR.SwapTargetSGPRHi = RegisterHelper::getSGPR(ScratchBase + 7);

  if (KD.AccumOffset > 0 && KD.AccumOffset <= 256) {
    SR.FirstFreeVGPRIdx = KD.AccumOffset - 3;
    SR.ScratchVGPR = 0;
    SR.LaneVGPR = 0;
    SR.TempVGPR = 0;
    SR.ExtraVGPRs = 0;
  } else {
    SR.FirstFreeVGPRIdx = KD.VGPRCount;
    SR.ScratchVGPR = RegisterHelper::getVGPR(KD.VGPRCount);
    SR.LaneVGPR = RegisterHelper::getVGPR(KD.VGPRCount + 1);
    SR.TempVGPR = RegisterHelper::getVGPR(KD.VGPRCount + 2);
    SR.ExtraVGPRs = 3;
  }

  SR.ExtraSGPRs = ExtraSGPRs;
  return SR;
}

ScratchRegisters
ScratchRegisters::fromDescriptorZeroSGPR(const KernelDescriptor &KD) {
  ScratchRegisters SR;
  SR.HasAccumVGPRs = (KD.AccumOffset > 0);
  SR.ZeroSGPR = true;
  SR.ExtraSGPRs = 0;

  SR.ReturnAddrSGPR = 0;
  SR.ReturnAddrSGPRHi = 0;
  SR.ScratchSGPR = 0;
  SR.ExecSaveSGPRLo = 0;
  SR.ExecSaveSGPRHi = 0;
  SR.FirstFreeSGPRIdx = 0;

  if (KD.AccumOffset > 0 && KD.AccumOffset <= 256) {
    SR.FirstFreeVGPRIdx = KD.AccumOffset - 3;
    SR.ScratchVGPR = 0;
    SR.LaneVGPR    = 0;
    SR.TempVGPR    = 0;
    SR.ExtraVGPRs = 0;
  } else {
    SR.FirstFreeVGPRIdx = KD.VGPRCount;
    SR.ScratchVGPR = RegisterHelper::getVGPR(KD.VGPRCount);
    SR.LaneVGPR    = RegisterHelper::getVGPR(KD.VGPRCount + 1);
    SR.TempVGPR    = RegisterHelper::getVGPR(KD.VGPRCount + 2);
    SR.ExtraVGPRs = 3;
  }

  return SR;
}

//===----------------------------------------------------------------------===//
// CFG-aware refinement / spill setup
//===----------------------------------------------------------------------===//

bool ScratchRegisters::refineScratchVGPRs(const ControlFlowGraph &CFG,
                                          const Disassembler &Disasm,
                                          uint32_t AccumOffset) {
  std::bitset<256> Used;
  const auto &MRI = Disasm.getMRI();

  for (const auto &BB : CFG.BasicBlocks) {
    for (const auto &DI : BB.Instructions) {
      for (unsigned i = 0; i < DI.Inst.getNumOperands(); ++i) {
        const auto &Op = DI.Inst.getOperand(i);
        if (!Op.isReg())
          continue;
        unsigned Reg = Op.getReg();
        auto markIfVGPR = [&](unsigned R) {
          if (RegisterHelper::isVGPR(R)) {
            unsigned Idx = RegisterHelper::getVGPRIndex(R);
            if (Idx < AccumOffset)
              Used.set(Idx);
          }
        };
        markIfVGPR(Reg);
        for (MCRegister Sub : MRI.subregs(Reg))
          markIfVGPR(Sub);
      }
    }
  }

  // Search from the top down — high-numbered VGPRs are less likely to be used.
  std::vector<unsigned> FreeVGPRs;
  for (int Idx = static_cast<int>(AccumOffset) - 1;
       Idx >= 0 && FreeVGPRs.size() < 3; --Idx) {
    if (!Used.test(static_cast<unsigned>(Idx)))
      FreeVGPRs.push_back(static_cast<unsigned>(Idx));
  }

  if (FreeVGPRs.size() < 3)
    return false;

  std::sort(FreeVGPRs.begin(), FreeVGPRs.end());
  FirstFreeVGPRIdx = FreeVGPRs[0];
  ScratchVGPR = RegisterHelper::getVGPR(FreeVGPRs[0]);
  LaneVGPR    = RegisterHelper::getVGPR(FreeVGPRs[1]);
  TempVGPR    = RegisterHelper::getVGPR(FreeVGPRs[2]);
  ExtraVGPRs  = 0;
  return true;
}

void ScratchRegisters::setupAccVGPRSpill(uint32_t AccumOffset,
                                         uint32_t VGPRCount) {
  // Pick 3 victim VGPRs from the top of the regular range.
  // These will be saved to AccVGPR spill slots before the trampoline body
  // and restored after.
  unsigned V0 = AccumOffset - 3;
  unsigned V1 = AccumOffset - 2;
  unsigned V2 = AccumOffset - 1;

  FirstFreeVGPRIdx = V0;
  ScratchVGPR = RegisterHelper::getVGPR(V0);
  LaneVGPR    = RegisterHelper::getVGPR(V1);
  TempVGPR    = RegisterHelper::getVGPR(V2);
  ExtraVGPRs  = 0;

  uint32_t NumAccUsed = VGPRCount - AccumOffset;
  uint32_t AllocGran = (VGPRCount > 256) ? 8 : 4;
  uint32_t GranTotal = ((VGPRCount + AllocGran - 1) / AllocGran) * AllocGran;
  uint32_t TotalAcc = GranTotal - AccumOffset;

  // Spill slots start after kernel's AccVGPR usage.
  // 4 slots: 3 for victim VGPRs + 1 for AddrVGPR save during counting loop.
  SpillAGPR0 = RegisterHelper::getAGPR(NumAccUsed);
  SpillAGPR1 = RegisterHelper::getAGPR(NumAccUsed + 1);
  SpillAGPR2 = RegisterHelper::getAGPR(NumAccUsed + 2);
  SpillAGPR3 = RegisterHelper::getAGPR(NumAccUsed + 3);

  if (NumAccUsed + 4 > TotalAcc) {
    uint32_t NewTotal = AccumOffset + NumAccUsed + 4;
    uint32_t NewGran = ((NewTotal + AllocGran - 1) / AllocGran) * AllocGran;
    ExtraVGPRs = NewGran - GranTotal;
  }

  NeedsAccVGPRSpill = true;
}

void ScratchRegisters::setupScratchSpill(uint32_t AccumOffset,
                                         uint32_t CurrentScratchSize) {
  // Pick 3 victim VGPRs from the top of the regular range.
  // These will be saved to scratch memory before the trampoline body
  // and restored after. This is more robust than AccVGPR spill because
  // it doesn't interfere with MFMA accumulator state.
  unsigned V0 = AccumOffset - 3;
  unsigned V1 = AccumOffset - 2;
  unsigned V2 = AccumOffset - 1;

  FirstFreeVGPRIdx = V0;
  ScratchVGPR = RegisterHelper::getVGPR(V0);
  LaneVGPR    = RegisterHelper::getVGPR(V1);
  TempVGPR    = RegisterHelper::getVGPR(V2);
  ExtraVGPRs  = 0;

  // Scratch memory layout for spill:
  //   [CurrentScratchSize + 0]:  ScratchVGPR (4 bytes)
  //   [CurrentScratchSize + 4]:  LaneVGPR    (4 bytes)
  //   [CurrentScratchSize + 8]:  TempVGPR    (4 bytes)
  ScratchSpillOffset = CurrentScratchSize;
  ExtraScratchBytes = 12;

  NeedsScratchSpill = true;
  NeedsAccVGPRSpill = false;
}

} // namespace aegisbit
