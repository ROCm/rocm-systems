//===-- RelayEmitter.cpp - Relay Stub Generation -----------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/RelayEmitter.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/InstructionBuilder.h"
#include "aegisbit/RegisterHelper.h"
#include "aegisbit/RuntimeConfig.h"

#include <cstring>

using namespace llvm;

namespace aegisbit {

RelayEmitter::RelayEmitter(ISAEncoder &Enc) : Enc(&Enc) {}
RelayEmitter::~RelayEmitter() = default;

Expected<RelayStubs>
RelayEmitter::emitRelayStubs(const InstrumentationSite &Site,
                              const ScratchRegisters &Scratch,
                              ArrayRef<uint8_t> Code,
                              uint64_t ReturnTargetAbs,
                              uint64_t RetBranchPC,
                              bool ForceNoBodyJump) {
  using Op = InstructionBuilder::Operand;
  auto &D = Enc->getDisassembler();
  RelayStubs Result;

  const auto &TFlags = RuntimeConfig::getInstance().Transform;
  const bool MinimalRelay = TFlags.MinimalRelay;
  const bool NoBodyJumpEnv = TFlags.NoBodyJump;
  bool NoBodyJump = NoBodyJumpEnv || ForceNoBodyJump;
  if (MinimalRelay) {
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              Code.data() + Site.Offset,
                              Code.data() + Site.Offset + Site.OrigInstSize);
    constexpr int64_t SBRANCH_MIN = -32768;
    constexpr int64_t SBRANCH_MAX = 32767;
    uint64_t BranchPC = RetBranchPC + Result.ReturnStub.size();
    int64_t BackOffset = static_cast<int64_t>(ReturnTargetAbs) -
                         static_cast<int64_t>(BranchPC);
    int64_t BackDw = (BackOffset - 4) / 4;
    if (BackDw < SBRANCH_MIN || BackDw > SBRANCH_MAX)
      return RelayStubs{};
    auto RetBr = Enc->encodeSBranch(static_cast<int16_t>(BackDw));
    if (!RetBr) return RetBr.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              RetBr->begin(), RetBr->end());
    Result.FwdLongJumpOffset = 0;
    return Result;
  }

  const bool NopRelay = TFlags.NopRelay;
  if (NopRelay) {
    // Same SIZE as NO_BODY_JUMP but all NOPs instead of writelane/readlane.
    // Tests whether it's the instruction behavior or just extra code.
    for (int i = 0; i < 4; ++i) {
      auto Nop = Enc->buildAndEmit("S_NOP", {Op::Imm(0)});
      if (!Nop) return Nop.takeError();
      Result.ForwardStub.insert(Result.ForwardStub.end(), Nop->begin(), Nop->end());
    }
    Result.FwdLongJumpOffset = Result.ForwardStub.size();
    for (int i = 0; i < 5; ++i) {
      auto Nop = Enc->buildAndEmit("S_NOP", {Op::Imm(0)});
      if (!Nop) return Nop.takeError();
      Result.ReturnStub.insert(Result.ReturnStub.end(), Nop->begin(), Nop->end());
    }
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              Code.data() + Site.Offset,
                              Code.data() + Site.Offset + Site.OrigInstSize);
    constexpr int64_t SBRANCH_MIN_ = -32768;
    constexpr int64_t SBRANCH_MAX_ = 32767;
    uint64_t BranchPC_ = RetBranchPC + Result.ForwardStub.size() +
                        Result.ReturnStub.size();
    int64_t BackOff_ = static_cast<int64_t>(ReturnTargetAbs) -
                       static_cast<int64_t>(BranchPC_);
    int64_t BackDw_ = (BackOff_ - 4) / 4;
    if (BackDw_ < SBRANCH_MIN_ || BackDw_ > SBRANCH_MAX_)
      return RelayStubs{};
    auto RetBr_ = Enc->encodeSBranch(static_cast<int16_t>(BackDw_));
    if (!RetBr_) return RetBr_.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RetBr_->begin(), RetBr_->end());
    return Result;
  }

  const bool VccOnlyRelay = TFlags.VccOnlyRelay;
  if (VccOnlyRelay) {
    // VCC save/restore only (no SCC), no body island.
    auto W0 = Enc->encodeWriteLane(Scratch.ScratchVGPR,
                                    InstructionBuilder::VCC_LO_REG, 0);
    if (!W0) return W0.takeError();
    Result.ForwardStub.insert(Result.ForwardStub.end(), W0->begin(), W0->end());
    auto W1 = Enc->encodeWriteLane(Scratch.ScratchVGPR,
                                    InstructionBuilder::VCC_HI_REG, 1);
    if (!W1) return W1.takeError();
    Result.ForwardStub.insert(Result.ForwardStub.end(), W1->begin(), W1->end());
    Result.FwdLongJumpOffset = Result.ForwardStub.size();
    auto RV0 = Enc->encodeReadLane(InstructionBuilder::VCC_LO_REG,
                                    Scratch.ScratchVGPR, 0);
    if (!RV0) return RV0.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RV0->begin(), RV0->end());
    auto RV1 = Enc->encodeReadLane(InstructionBuilder::VCC_HI_REG,
                                    Scratch.ScratchVGPR, 1);
    if (!RV1) return RV1.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RV1->begin(), RV1->end());
    auto RNop = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
    if (!RNop) return RNop.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RNop->begin(), RNop->end());
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              Code.data() + Site.Offset,
                              Code.data() + Site.Offset + Site.OrigInstSize);
    constexpr int64_t SB_MIN = -32768;
    constexpr int64_t SB_MAX = 32767;
    uint64_t BPC = RetBranchPC + Result.ForwardStub.size() +
                   Result.ReturnStub.size();
    int64_t BOff = static_cast<int64_t>(ReturnTargetAbs) -
                   static_cast<int64_t>(BPC);
    int64_t BDw = (BOff - 4) / 4;
    if (BDw < SB_MIN || BDw > SB_MAX)
      return RelayStubs{};
    auto RetBr = Enc->encodeSBranch(static_cast<int16_t>(BDw));
    if (!RetBr) return RetBr.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RetBr->begin(), RetBr->end());
    return Result;
  }

  if (NoBodyJump) {
    // VCC/SCC save/restore but NO body island round-trip.
    // ForwardStub: writelane VCC/SCC into ScratchVGPR
    // ReturnStub: readlane VCC/SCC, displaced inst, s_branch back
    // The ForwardStub falls straight through to ReturnStub (no long-jump).
    auto W0 = Enc->encodeWriteLane(Scratch.ScratchVGPR,
                                    InstructionBuilder::VCC_LO_REG, 0);
    if (!W0) return W0.takeError();
    Result.ForwardStub.insert(Result.ForwardStub.end(), W0->begin(), W0->end());
    auto W1 = Enc->encodeWriteLane(Scratch.ScratchVGPR,
                                    InstructionBuilder::VCC_HI_REG, 1);
    if (!W1) return W1.takeError();
    Result.ForwardStub.insert(Result.ForwardStub.end(), W1->begin(), W1->end());
    {
      auto WrM0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Enc->getM0Reg(), 3);
      if (!WrM0) return WrM0.takeError();
      Result.ForwardStub.insert(Result.ForwardStub.end(), WrM0->begin(), WrM0->end());
      auto CSel = Enc->buildAndEmit("S_CSELECT_B32",
          {Op::Reg(Enc->getM0Reg()), Op::Imm(1), Op::Imm(0)});
      if (!CSel) return CSel.takeError();
      Result.ForwardStub.insert(Result.ForwardStub.end(), CSel->begin(), CSel->end());
      auto WrSCC = Enc->encodeWriteLane(Scratch.ScratchVGPR,
                                         Enc->getM0Reg(), 2);
      if (!WrSCC) return WrSCC.takeError();
      Result.ForwardStub.insert(Result.ForwardStub.end(), WrSCC->begin(), WrSCC->end());
    }
    Result.FwdLongJumpOffset = Result.ForwardStub.size();
    // Restore — VCC first, then SCC, so v_readlane→VCC can't clobber SCC
    auto RV0 = Enc->encodeReadLane(InstructionBuilder::VCC_LO_REG,
                                    Scratch.ScratchVGPR, 0);
    if (!RV0) return RV0.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RV0->begin(), RV0->end());
    auto RV1 = Enc->encodeReadLane(InstructionBuilder::VCC_HI_REG,
                                    Scratch.ScratchVGPR, 1);
    if (!RV1) return RV1.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RV1->begin(), RV1->end());
    auto RdSCC = Enc->encodeReadLane(Enc->getM0Reg(),
                                      Scratch.ScratchVGPR, 2);
    if (!RdSCC) return RdSCC.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RdSCC->begin(), RdSCC->end());
    auto SccNop = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
    if (!SccNop) return SccNop.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), SccNop->begin(), SccNop->end());
    auto RCmp = Enc->buildAndEmit("S_CMP_LG_U32",
        {Op::Reg(Enc->getM0Reg()), Op::Imm(0)});
    if (!RCmp) return RCmp.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RCmp->begin(), RCmp->end());
    {
      auto RdM0 = Enc->encodeReadLane(Enc->getM0Reg(), Scratch.ScratchVGPR, 3);
      if (!RdM0) return RdM0.takeError();
      Result.ReturnStub.insert(Result.ReturnStub.end(), RdM0->begin(), RdM0->end());
    }
    auto RNop = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
    if (!RNop) return RNop.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RNop->begin(), RNop->end());
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              Code.data() + Site.Offset,
                              Code.data() + Site.Offset + Site.OrigInstSize);
    constexpr int64_t SBRANCH_MIN = -32768;
    constexpr int64_t SBRANCH_MAX = 32767;
    uint64_t BranchPC = RetBranchPC + Result.ForwardStub.size() +
                        Result.ReturnStub.size();
    int64_t BackOffset = static_cast<int64_t>(ReturnTargetAbs) -
                         static_cast<int64_t>(BranchPC);
    int64_t BackDw = (BackOffset - 4) / 4;
    if (BackDw < SBRANCH_MIN || BackDw > SBRANCH_MAX)
      return RelayStubs{};
    auto RetBr = Enc->encodeSBranch(static_cast<int16_t>(BackDw));
    if (!RetBr) return RetBr.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(), RetBr->begin(), RetBr->end());
    return Result;
  }

  // --- Forward stub ---
  // When scratch spill is active, ScratchVGPR holds live kernel data.
  // Save it to scratch BEFORE using its lanes for VCC/SCC storage.
  constexpr uint32_t SCRATCH_OFFSET_IMM_MAX = 4095;
  if (Scratch.NeedsScratchSpill) {
    auto WaitPre = InstructionBuilder::buildSWaitCnt(
        D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/0);
    if (!WaitPre) return WaitPre.takeError();
    auto WaitPreB = Enc->emitInst(*WaitPre);
    if (!WaitPreB) return WaitPreB.takeError();
    ISAEncoder::append(Result.ForwardStub, *WaitPreB);

    uint32_t SVOffset = Scratch.ScratchSpillOffset + 0;
    if (SVOffset <= SCRATCH_OFFSET_IMM_MAX) {
      auto SpillSV = InstructionBuilder::buildScratchStoreDword(
          D, Scratch.ScratchVGPR, SVOffset);
      if (!SpillSV) return SpillSV.takeError();
      auto SpillSVB = Enc->emitInst(*SpillSV);
      if (!SpillSVB) return SpillSVB.takeError();
      ISAEncoder::append(Result.ForwardStub, *SpillSVB);
    } else {
      // Offset exceeds 13-bit signed immediate range; use SADDR encoding.
      // Stash VCC_LO in M0, use VCC_LO as SADDR temp, then restore.
      auto StashVCC = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(Enc->getM0Reg()), Op::Reg(InstructionBuilder::VCC_LO_REG)});
      if (!StashVCC) return StashVCC.takeError();
      ISAEncoder::append(Result.ForwardStub, *StashVCC);

      auto MovOff = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::VCC_LO_REG),
           Op::Imm(static_cast<int32_t>(SVOffset))});
      if (!MovOff) return MovOff.takeError();
      ISAEncoder::append(Result.ForwardStub, *MovOff);

      auto SpillSV = InstructionBuilder::buildScratchStoreDwordSAddr(
          D, Scratch.ScratchVGPR, InstructionBuilder::VCC_LO_REG, 0);
      if (!SpillSV) return SpillSV.takeError();
      auto SpillSVB = Enc->emitInst(*SpillSV);
      if (!SpillSVB) return SpillSVB.takeError();
      ISAEncoder::append(Result.ForwardStub, *SpillSVB);

      auto RestoreVCC = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::VCC_LO_REG), Op::Reg(Enc->getM0Reg())});
      if (!RestoreVCC) return RestoreVCC.takeError();
      ISAEncoder::append(Result.ForwardStub, *RestoreVCC);
    }

    auto WaitSV = InstructionBuilder::buildSWaitCnt(
        D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/15);
    if (!WaitSV) return WaitSV.takeError();
    auto WaitSVB = Enc->emitInst(*WaitSV);
    if (!WaitSVB) return WaitSVB.takeError();
    ISAEncoder::append(Result.ForwardStub, *WaitSVB);
  }

  // Save VCC, save SCC, long-jump placeholder
  auto W0 = Enc->encodeWriteLane(Scratch.ScratchVGPR,
                                  InstructionBuilder::VCC_LO_REG, 0);
  if (!W0) return W0.takeError();
  Result.ForwardStub.insert(Result.ForwardStub.end(), W0->begin(), W0->end());

  auto W1 = Enc->encodeWriteLane(Scratch.ScratchVGPR,
                                  InstructionBuilder::VCC_HI_REG, 1);
  if (!W1) return W1.takeError();
  Result.ForwardStub.insert(Result.ForwardStub.end(), W1->begin(), W1->end());

  // Save M0 to lane 3 (kernel may use M0; must preserve it)
  {
    auto WrM0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Enc->getM0Reg(), 3);
    if (!WrM0) return WrM0.takeError();
    Result.ForwardStub.insert(Result.ForwardStub.end(),
                               WrM0->begin(), WrM0->end());
  }

  // Snapshot SCC -> M0 -> lane 2 (avoids clobbering VCC_LO)
  {
    auto CSelM0 = Enc->buildAndEmit("S_CSELECT_B32",
        {Op::Reg(Enc->getM0Reg()), Op::Imm(1), Op::Imm(0)});
    if (!CSelM0) return CSelM0.takeError();
    Result.ForwardStub.insert(Result.ForwardStub.end(),
                               CSelM0->begin(), CSelM0->end());

    auto WrSCC = Enc->encodeWriteLane(Scratch.ScratchVGPR,
                                       Enc->getM0Reg(), 2);
    if (!WrSCC) return WrSCC.takeError();
    Result.ForwardStub.insert(Result.ForwardStub.end(),
                               WrSCC->begin(), WrSCC->end());
  }

  Result.FwdLongJumpOffset = Result.ForwardStub.size();
  const bool SBranchBody = TFlags.SBranchBody;
  Result.UseSBranchBody = SBranchBody;
  if (SBranchBody) {
    auto Br = Enc->encodeSBranch(0);
    if (!Br) return Br.takeError();
    Result.ForwardStub.insert(Result.ForwardStub.end(), Br->begin(), Br->end());
  } else {
    auto LJ = Enc->encodeLongJumpVCC(LJ_PLACEHOLDER);
    if (!LJ) return LJ.takeError();
    Result.ForwardStub.insert(Result.ForwardStub.end(), LJ->begin(), LJ->end());
  }

  // --- Return stub: restore SCC, restore VCC, optional scratch load,
  //     NOP, displaced instruction, s_branch back ---
  // When NeedsScratchSpill is false, restore VCC before SCC to avoid the
  // GFX950 erratum where v_readlane→VCC clobbers SCC.  When NeedsScratchSpill
  // is true, keep the original SCC-first order to maintain compatibility with
  // the scratch spill restore sequence.
  if (!Scratch.NeedsScratchSpill) {
    auto RV0 = Enc->encodeReadLane(InstructionBuilder::VCC_LO_REG,
                                    Scratch.ScratchVGPR, 0);
    if (!RV0) return RV0.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              RV0->begin(), RV0->end());

    auto RV1 = Enc->encodeReadLane(InstructionBuilder::VCC_HI_REG,
                                    Scratch.ScratchVGPR, 1);
    if (!RV1) return RV1.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              RV1->begin(), RV1->end());
  }

  auto RdSCC = Enc->encodeReadLane(Enc->getM0Reg(),
                                    Scratch.ScratchVGPR, 2);
  if (!RdSCC) return RdSCC.takeError();
  Result.ReturnStub.insert(Result.ReturnStub.end(),
                            RdSCC->begin(), RdSCC->end());

  // v_readlane writes an SGPR; SALU needs 5 wait states before reading it
  auto SccNop = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
  if (!SccNop) return SccNop.takeError();
  Result.ReturnStub.insert(Result.ReturnStub.end(),
                            SccNop->begin(), SccNop->end());

  auto RCmp = Enc->buildAndEmit("S_CMP_LG_U32",
      {Op::Reg(Enc->getM0Reg()), Op::Imm(0)});
  if (!RCmp) return RCmp.takeError();
  Result.ReturnStub.insert(Result.ReturnStub.end(),
                            RCmp->begin(), RCmp->end());

  if (Scratch.NeedsScratchSpill) {
    auto RV0 = Enc->encodeReadLane(InstructionBuilder::VCC_LO_REG,
                                    Scratch.ScratchVGPR, 0);
    if (!RV0) return RV0.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              RV0->begin(), RV0->end());

    auto RV1 = Enc->encodeReadLane(InstructionBuilder::VCC_HI_REG,
                                    Scratch.ScratchVGPR, 1);
    if (!RV1) return RV1.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              RV1->begin(), RV1->end());
  }

  if (Scratch.NeedsScratchSpill) {
    uint32_t SVOffset = Scratch.ScratchSpillOffset + 0;
    if (SVOffset <= SCRATCH_OFFSET_IMM_MAX) {
      auto Rest0 = InstructionBuilder::buildScratchLoadDword(
          D, Scratch.ScratchVGPR, SVOffset);
      if (!Rest0) return Rest0.takeError();
      auto Rest0B = Enc->emitInst(*Rest0);
      if (!Rest0B) return Rest0B.takeError();
      Result.ReturnStub.insert(Result.ReturnStub.end(),
                                Rest0B->begin(), Rest0B->end());
    } else {
      // VCC_LO holds restored value; stash in M0 for SADDR temp.
      auto StashVCC = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(Enc->getM0Reg()), Op::Reg(InstructionBuilder::VCC_LO_REG)});
      if (!StashVCC) return StashVCC.takeError();
      Result.ReturnStub.insert(Result.ReturnStub.end(),
                                StashVCC->begin(), StashVCC->end());

      auto MovOff = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::VCC_LO_REG),
           Op::Imm(static_cast<int32_t>(SVOffset))});
      if (!MovOff) return MovOff.takeError();
      Result.ReturnStub.insert(Result.ReturnStub.end(),
                                MovOff->begin(), MovOff->end());

      auto Rest0 = InstructionBuilder::buildScratchLoadDwordSAddr(
          D, Scratch.ScratchVGPR, InstructionBuilder::VCC_LO_REG, 0);
      if (!Rest0) return Rest0.takeError();
      auto Rest0B = Enc->emitInst(*Rest0);
      if (!Rest0B) return Rest0B.takeError();
      Result.ReturnStub.insert(Result.ReturnStub.end(),
                                Rest0B->begin(), Rest0B->end());

      auto RestoreVCC = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::VCC_LO_REG), Op::Reg(Enc->getM0Reg())});
      if (!RestoreVCC) return RestoreVCC.takeError();
      Result.ReturnStub.insert(Result.ReturnStub.end(),
                                RestoreVCC->begin(), RestoreVCC->end());
    }

    auto WaitL = InstructionBuilder::buildSWaitCnt(
        D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/15);
    if (!WaitL) return WaitL.takeError();
    auto WaitLB = Enc->emitInst(*WaitL);
    if (!WaitLB) return WaitLB.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              WaitLB->begin(), WaitLB->end());
  }

  // Restore M0 from lane 3
  {
    auto RdM0 = Enc->encodeReadLane(Enc->getM0Reg(), Scratch.ScratchVGPR, 3);
    if (!RdM0) return RdM0.takeError();
    Result.ReturnStub.insert(Result.ReturnStub.end(),
                              RdM0->begin(), RdM0->end());
  }

  auto RNop = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
  if (!RNop) return RNop.takeError();
  Result.ReturnStub.insert(Result.ReturnStub.end(),
                            RNop->begin(), RNop->end());

  // Displaced instruction
  Result.ReturnStub.insert(Result.ReturnStub.end(),
                            Code.data() + Site.Offset,
                            Code.data() + Site.Offset + Site.OrigInstSize);

  // s_branch back to kernel
  constexpr int64_t SBRANCH_MIN = -32768;
  constexpr int64_t SBRANCH_MAX = 32767;

  uint64_t BranchPC = RetBranchPC + Result.ForwardStub.size() +
                      Result.ReturnStub.size();
  int64_t BackOffset = static_cast<int64_t>(ReturnTargetAbs) -
                       static_cast<int64_t>(BranchPC);
  int64_t BackDw = (BackOffset - 4) / 4;
  if (BackDw < SBRANCH_MIN || BackDw > SBRANCH_MAX) {
    // Return empty stubs to signal the site is unreachable
    return RelayStubs{};
  }
  auto RetBr = Enc->encodeSBranch(static_cast<int16_t>(BackDw));
  if (!RetBr) return RetBr.takeError();
  Result.ReturnStub.insert(Result.ReturnStub.end(),
                            RetBr->begin(), RetBr->end());

  return Result;
}

size_t RelayEmitter::addBodyEntry(const std::vector<uint8_t> &BodyBytes) {
  size_t Offset = BodyIslandCursor;
  BodyIslandBytes.insert(BodyIslandBytes.end(),
                          BodyBytes.begin(), BodyBytes.end());
  BodyIslandCursor += BodyBytes.size();
  return Offset;
}

Expected<size_t> RelayEmitter::appendReturnLongJump() {
  size_t Offset = BodyIslandCursor;
  const bool SBranchBody = RuntimeConfig::getInstance().Transform.SBranchBody;
  if (SBranchBody) {
    auto Br = Enc->encodeSBranch(0);
    if (!Br) return Br.takeError();
    BodyIslandBytes.insert(BodyIslandBytes.end(), Br->begin(), Br->end());
    BodyIslandCursor += Br->size();
  } else {
    auto RetLJ = Enc->encodeLongJumpVCC(LJ_PLACEHOLDER);
    if (!RetLJ) return RetLJ.takeError();
    BodyIslandBytes.insert(BodyIslandBytes.end(),
                            RetLJ->begin(), RetLJ->end());
    BodyIslandCursor += RetLJ->size();
  }
  return Offset;
}

void RelayEmitter::addFixup(RelayFixup Fix) {
  Fixups.push_back(Fix);
}

Expected<TrampolineIsland>
RelayEmitter::fixupRelays(uint64_t BodyIslandStart,
                           std::vector<TrampolineIsland> &StubIslands) {
  const bool SBranchBody = RuntimeConfig::getInstance().Transform.SBranchBody;

  for (auto &Fix : Fixups) {
    uint64_t BodyEntryAbs = BodyIslandStart + Fix.BodyEntryOff;

    if (SBranchBody) {
      // s_branch fixup: patch the 4-byte s_branch in the forward stub
      int64_t FwdBrPC = Fix.FwdGetPCAbs; // s_branch address
      int64_t FwdOffset = static_cast<int64_t>(BodyEntryAbs) -
                          static_cast<int64_t>(FwdBrPC);
      int64_t FwdDw = (FwdOffset - 4) / 4;
      auto FwdBr = Enc->encodeSBranch(static_cast<int16_t>(FwdDw));
      if (!FwdBr) return FwdBr.takeError();

      bool FwdPatched = false;
      for (auto &Isl : StubIslands) {
        uint64_t IslEnd = Isl.Offset + Isl.Bytes.size();
        if (Fix.FwdGetPCAbs >= Isl.Offset && Fix.FwdGetPCAbs < IslEnd) {
          size_t off = static_cast<size_t>(Fix.FwdGetPCAbs - Isl.Offset);
          if (off + FwdBr->size() <= Isl.Bytes.size()) {
            std::memcpy(Isl.Bytes.data() + off,
                        FwdBr->data(), FwdBr->size());
            FwdPatched = true;
          }
          break;
        }
      }
      if (!FwdPatched) {
        llvm::errs() << "[aegisbit] ERROR: s_branch fwd fixup failed\n";
      }

      // Return s_branch fixup in body island
      uint64_t RetBrAbs = BodyIslandStart + Fix.RetGetPCBodyOff;
      int64_t RetOffset = static_cast<int64_t>(Fix.RelayReturnAbs) -
                          static_cast<int64_t>(RetBrAbs);
      int64_t RetDw = (RetOffset - 4) / 4;
      auto RetBr = Enc->encodeSBranch(static_cast<int16_t>(RetDw));
      if (!RetBr) return RetBr.takeError();

      if (Fix.RetGetPCBodyOff + RetBr->size() <= BodyIslandBytes.size()) {
        std::memcpy(BodyIslandBytes.data() + Fix.RetGetPCBodyOff,
                    RetBr->data(), RetBr->size());
      } else {
        llvm::errs() << "[aegisbit] ERROR: s_branch ret fixup overflow\n";
      }
      continue;
    }

    // Forward long-jump fixup (relay stub -> body)
    int64_t FwdOffset = static_cast<int64_t>(BodyEntryAbs) -
                        static_cast<int64_t>(Fix.FwdGetPCAbs);
    auto FwdLJ = Enc->encodeLongJumpVCC(FwdOffset);
    if (!FwdLJ) return FwdLJ.takeError();

    bool FwdPatched = false;
    for (auto &Isl : StubIslands) {
      uint64_t IslEnd = Isl.Offset + Isl.Bytes.size();
      if (Fix.FwdGetPCAbs >= Isl.Offset && Fix.FwdGetPCAbs < IslEnd) {
        size_t off = static_cast<size_t>(Fix.FwdGetPCAbs - Isl.Offset);
        if (off + FwdLJ->size() <= Isl.Bytes.size()) {
          std::memcpy(Isl.Bytes.data() + off,
                      FwdLJ->data(), FwdLJ->size());
          FwdPatched = true;
        }
        break;
      }
    }
    if (!FwdPatched) {
      char buf[256];
      snprintf(buf, sizeof(buf),
          "[aegisbit] ERROR: forward LJ fixup at 0x%llX not found in any "
          "island (BodyEntryOff=%zu, %zu islands)\n",
          (unsigned long long)Fix.FwdGetPCAbs,
          (size_t)Fix.BodyEntryOff, StubIslands.size());
      llvm::errs() << buf;
    }

    // Return long-jump fixup (body -> relay return)
    uint64_t RetGetPCAbs = BodyIslandStart + Fix.RetGetPCBodyOff;
    int64_t RetOffset = static_cast<int64_t>(Fix.RelayReturnAbs) -
                        static_cast<int64_t>(RetGetPCAbs);
    auto RetLJ = Enc->encodeLongJumpVCC(RetOffset);
    if (!RetLJ) return RetLJ.takeError();

    if (Fix.RetGetPCBodyOff + RetLJ->size() <= BodyIslandBytes.size()) {
      std::memcpy(BodyIslandBytes.data() + Fix.RetGetPCBodyOff,
                  RetLJ->data(), RetLJ->size());
    } else {
      char buf[256];
      snprintf(buf, sizeof(buf),
          "[aegisbit] ERROR: return LJ fixup at body offset %zu overflows "
          "body island (size=%zu, LJ size=%zu)\n",
          (size_t)Fix.RetGetPCBodyOff, BodyIslandBytes.size(),
          RetLJ->size());
      llvm::errs() << buf;
    }
  }

  TrampolineIsland BodyIsl;
  BodyIsl.Offset = BodyIslandStart;
  BodyIsl.Bytes = std::move(BodyIslandBytes);
  return BodyIsl;
}

} // namespace aegisbit
