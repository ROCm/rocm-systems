//===-- TrampolineEmitter.cpp - Trampoline Body Codegen ----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/TrampolineEmitter.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/InstructionBuilder.h"
#include "aegisbit/PayloadCompiler.h"
#include "aegisbit/RegisterHelper.h"
#include "aegisbit/RuntimeConfig.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <cstdlib>

using namespace llvm;

namespace aegisbit {

TrampolineEmitter::~TrampolineEmitter() = default;

Expected<std::unique_ptr<TrampolineEmitter>>
TrampolineEmitter::create(ISAEncoder &Enc, StringRef Arch,
                           const TraceConfig &Trace) {
  auto E = std::unique_ptr<TrampolineEmitter>(new TrampolineEmitter());
  E->Enc = &Enc;

  if (Trace.Strategy == PayloadStrategy::OnGpuReduce) {
    auto PC = PayloadCompiler::create(Arch);
    if (!PC) return PC.takeError();

    auto CountingMod = PayloadCompiler::buildCountingLoop(PC->get()->getContext());
    auto CountingOrErr = PC->get()->compile(std::move(CountingMod));
    if (!CountingOrErr) return CountingOrErr.takeError();
    E->CountingBytes = std::move(*CountingOrErr);

    auto LDSMod = PayloadCompiler::buildMaxPopCountLoop(PC->get()->getContext());
    auto LDSOrErr = PC->get()->compile(std::move(LDSMod));
    if (!LDSOrErr) return LDSOrErr.takeError();
    E->LDSCountingBytes = std::move(*LDSOrErr);

    auto AtomicMod = PayloadCompiler::buildAtomicAccumulator(
        PC->get()->getContext(), true);
    auto AtomicOrErr = PC->get()->compile(std::move(AtomicMod));
    if (!AtomicOrErr) return AtomicOrErr.takeError();
    E->AtomicBytesAtomic = std::move(*AtomicOrErr);

    auto NonAtomicMod = PayloadCompiler::buildAtomicAccumulator(
        PC->get()->getContext(), false);
    auto NonAtomicOrErr = PC->get()->compile(std::move(NonAtomicMod));
    if (!NonAtomicOrErr) return NonAtomicOrErr.takeError();
    E->AtomicBytesNonAtomic = std::move(*NonAtomicOrErr);

    if (!RuntimeConfig::getInstance().Debug.DumpBlobsDir.empty()) {
      auto DumpBlob = [](const char *Path, const std::vector<uint8_t> &B) {
        if (FILE *F = fopen(Path, "wb")) {
          fwrite(B.data(), 1, B.size(), F);
          fclose(F);
          fprintf(stderr, "[DEBUG] Dumped %zu bytes to %s\n", B.size(), Path);
        }
      };
      DumpBlob("/tmp/aegis_counting.bin", E->CountingBytes);
      DumpBlob("/tmp/aegis_lds_counting.bin", E->LDSCountingBytes);
      DumpBlob("/tmp/aegis_atomic_nonat.bin", E->AtomicBytesNonAtomic);
      DumpBlob("/tmp/aegis_atomic_at.bin", E->AtomicBytesAtomic);
    }
  }

  return E;
}

static constexpr uint32_t SCRATCH_OFFSET_IMM_MAX = 4095;

Error TrampolineEmitter::appendScratchStore(std::vector<uint8_t> &TB,
                                             unsigned VDataVGPR,
                                             uint32_t Offset,
                                             unsigned TempSGPR) {
  using Op = InstructionBuilder::Operand;
  const auto &D = Enc->getDisassembler();
  if (Offset <= SCRATCH_OFFSET_IMM_MAX) {
    auto I = InstructionBuilder::buildScratchStoreDword(D, VDataVGPR, Offset);
    if (!I) return I.takeError();
    auto B = Enc->emitInst(*I);
    if (!B) return B.takeError();
    ISAEncoder::append(TB, *B);
  } else {
    auto Mov = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(TempSGPR), Op::Imm(static_cast<int32_t>(Offset))});
    if (!Mov) return Mov.takeError();
    ISAEncoder::append(TB, *Mov);
    auto I = InstructionBuilder::buildScratchStoreDwordSAddr(
        D, VDataVGPR, TempSGPR, 0);
    if (!I) return I.takeError();
    auto B = Enc->emitInst(*I);
    if (!B) return B.takeError();
    ISAEncoder::append(TB, *B);
  }
  return Error::success();
}

Error TrampolineEmitter::appendScratchLoad(std::vector<uint8_t> &TB,
                                             unsigned VDstVGPR,
                                             uint32_t Offset,
                                             unsigned TempSGPR) {
  using Op = InstructionBuilder::Operand;
  const auto &D = Enc->getDisassembler();
  if (Offset <= SCRATCH_OFFSET_IMM_MAX) {
    auto I = InstructionBuilder::buildScratchLoadDword(D, VDstVGPR, Offset);
    if (!I) return I.takeError();
    auto B = Enc->emitInst(*I);
    if (!B) return B.takeError();
    ISAEncoder::append(TB, *B);
  } else {
    auto Mov = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(TempSGPR), Op::Imm(static_cast<int32_t>(Offset))});
    if (!Mov) return Mov.takeError();
    ISAEncoder::append(TB, *Mov);
    auto I = InstructionBuilder::buildScratchLoadDwordSAddr(
        D, VDstVGPR, TempSGPR, 0);
    if (!I) return I.takeError();
    auto B = Enc->emitInst(*I);
    if (!B) return B.takeError();
    ISAEncoder::append(TB, *B);
  }
  return Error::success();
}

Expected<std::vector<uint8_t>>
TrampolineEmitter::emitDirectBody(const InstrumentationSite &Site,
                                  const InstrumentationPlan &Plan,
                                  const ScratchRegisters &Scratch,
                                  const TraceConfig &Trace,
                                  unsigned RetAddrSGPRPair,
                                  uint32_t SiteIdx) {
  return emitBodyForPath(Site, Plan, Scratch, Trace, /*UseRelay=*/false,
                         RetAddrSGPRPair, SiteIdx);
}

Expected<std::vector<uint8_t>>
TrampolineEmitter::emitRelayBody(const InstrumentationSite &Site,
                                 const InstrumentationPlan &Plan,
                                 const ScratchRegisters &Scratch,
                                 const TraceConfig &Trace,
                                 unsigned RetAddrSGPRPair,
                                 uint32_t SiteIdx) {
  return emitBodyForPath(Site, Plan, Scratch, Trace, /*UseRelay=*/true,
                         RetAddrSGPRPair, SiteIdx);
}

Expected<std::vector<uint8_t>>
TrampolineEmitter::emitBodyForPath(const InstrumentationSite &Site,
                                   const InstrumentationPlan &Plan,
                                   const ScratchRegisters &Scratch,
                                   const TraceConfig &Trace,
                                   bool UseRelay,
                                   unsigned RetAddrSGPRPair,
                                   uint32_t SiteIdx) {
  (void)Plan;

  using Op = InstructionBuilder::Operand;
  auto &D = Enc->getDisassembler();
  std::vector<uint8_t> TB;

  const auto &Cfg = RuntimeConfig::getInstance();
  const int PayloadLevel = Cfg.Debug.DryPayloadLevel;

  // AEGISBIT_NOOP_TRAMPOLINE: emit zero-overhead trampoline (no save/restore).
  // The trampoline body is empty; only the displaced instruction + return
  // branch are emitted by the caller.  Used to isolate whether the bug is in
  // the trampoline overhead or the displaced-instruction relocation.
  if (Cfg.Transform.NoopTrampoline)
    return TB;  // return empty body

  // ---- Scratch memory spill: save victim VGPRs (if needed) ----
  //
  // In ZeroSGPR mode with large scratch offsets (> 4095), VCC_LO is the only
  // available SGPR for SADDR-based scratch ops. We stash VCC_LO into M0
  // before the stores (which clobber VCC_LO), then restore it after.
  bool VCCStashedInM0 = false;

  if (Scratch.NeedsScratchSpill) {
    {
      auto WaitSpill = InstructionBuilder::buildSWaitCnt(
          D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/0);
      if (!WaitSpill) return WaitSpill.takeError();
      auto WaitSpillB = Enc->emitInst(*WaitSpill);
      if (!WaitSpillB) return WaitSpillB.takeError();
      ISAEncoder::append(TB, *WaitSpillB);
    }

    if (!Scratch.ZeroSGPR) {
      auto SE0 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
      if (!SE0) return SE0.takeError();
      ISAEncoder::append(TB, *SE0);
      auto SE1 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
      if (!SE1) return SE1.takeError();
      ISAEncoder::append(TB, *SE1);
    }

    if (!Scratch.ZeroSGPR) {
      auto XA0 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Imm(-1)});
      if (!XA0) return XA0.takeError();
      ISAEncoder::append(TB, *XA0);
      auto XA1 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Imm(-1)});
      if (!XA1) return XA1.takeError();
      ISAEncoder::append(TB, *XA1);
    }

    // In ZeroSGPR mode, the only available SADDR temp is VCC_LO.
    // For small offsets (<= 4095) the immediate encoding is used and
    // TempSGPR is not touched, so passing VCC_LO is harmless.
    unsigned SpillTempSGPR = Scratch.ZeroSGPR
        ? InstructionBuilder::VCC_LO_REG
        : Scratch.ScratchSGPR;

    // For large offsets, the SADDR path will clobber VCC_LO via s_mov_b32.
    // Stash VCC_LO in M0 first; restored after the stores complete.
    if (Scratch.ZeroSGPR && !UseRelay &&
        Scratch.ScratchSpillOffset > SCRATCH_OFFSET_IMM_MAX) {
      auto Stash = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(Enc->getM0Reg()), Op::Reg(InstructionBuilder::VCC_LO_REG)});
      if (!Stash) return Stash.takeError();
      ISAEncoder::append(TB, *Stash);
      VCCStashedInM0 = true;
    }

    // In relay mode, the forward stub already saved ScratchVGPR to scratch
    // (before writing VCC/SCC into its lanes). Skip the redundant store.
    if (!UseRelay) {
      if (auto E = appendScratchStore(TB, Scratch.ScratchVGPR,
              Scratch.ScratchSpillOffset + 0, SpillTempSGPR))
        return std::move(E);
    }

    if (auto E = appendScratchStore(TB, Scratch.LaneVGPR,
            Scratch.ScratchSpillOffset + 4, SpillTempSGPR))
      return std::move(E);

    if (auto E = appendScratchStore(TB, Scratch.TempVGPR,
            Scratch.ScratchSpillOffset + 8, SpillTempSGPR))
      return std::move(E);

    {
      auto WaitStore = InstructionBuilder::buildSWaitCnt(
          D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/15);
      if (!WaitStore) return WaitStore.takeError();
      auto WaitStoreB = Enc->emitInst(*WaitStore);
      if (!WaitStoreB) return WaitStoreB.takeError();
      ISAEncoder::append(TB, *WaitStoreB);
    }

    if (VCCStashedInM0) {
      auto Restore = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::VCC_LO_REG), Op::Reg(Enc->getM0Reg())});
      if (!Restore) return Restore.takeError();
      ISAEncoder::append(TB, *Restore);
    }

    if (!Scratch.ZeroSGPR) {
      auto RE0 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(Scratch.ExecSaveSGPRLo)});
      if (!RE0) return RE0.takeError();
      ISAEncoder::append(TB, *RE0);
      auto RE1 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(Scratch.ExecSaveSGPRHi)});
      if (!RE1) return RE1.takeError();
      ISAEncoder::append(TB, *RE1);
    }
  }

  // ---- AccVGPR spill (DEPRECATED) ----
  if (Scratch.NeedsAccVGPRSpill) {
    {
      auto WaitSpill = InstructionBuilder::buildSWaitCnt(
          D, Site.PreSpillVmWait, /*ExpCnt=*/7, Site.PreSpillLgkmWait);
      if (!WaitSpill) return WaitSpill.takeError();
      auto WaitSpillB = Enc->emitInst(*WaitSpill);
      if (!WaitSpillB) return WaitSpillB.takeError();
      ISAEncoder::append(TB, *WaitSpillB);
    }

    auto SE0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
    if (!SE0) return SE0.takeError();
    ISAEncoder::append(TB, *SE0);

    auto SE1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
    if (!SE1) return SE1.takeError();
    ISAEncoder::append(TB, *SE1);

    auto XA0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Imm(-1)});
    if (!XA0) return XA0.takeError();
    ISAEncoder::append(TB, *XA0);

    auto XA1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Imm(-1)});
    if (!XA1) return XA1.takeError();
    ISAEncoder::append(TB, *XA1);

    auto Spill0 = Enc->encodeAccVGPRWrite(Scratch.SpillAGPR0, Scratch.ScratchVGPR);
    if (!Spill0) return Spill0.takeError();
    ISAEncoder::append(TB, *Spill0);

    auto Spill1 = Enc->encodeAccVGPRWrite(Scratch.SpillAGPR1, Scratch.LaneVGPR);
    if (!Spill1) return Spill1.takeError();
    ISAEncoder::append(TB, *Spill1);

    auto Spill2 = Enc->encodeAccVGPRWrite(Scratch.SpillAGPR2, Scratch.TempVGPR);
    if (!Spill2) return Spill2.takeError();
    ISAEncoder::append(TB, *Spill2);

    auto RE0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(Scratch.ExecSaveSGPRLo)});
    if (!RE0) return RE0.takeError();
    ISAEncoder::append(TB, *RE0);

    auto RE1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(Scratch.ExecSaveSGPRHi)});
    if (!RE1) return RE1.takeError();
    ISAEncoder::append(TB, *RE1);
  }

  // ---- Common prologue: save architectural state ----
  if (Scratch.ZeroSGPR) {
    if (!UseRelay) {
      auto Wr0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, InstructionBuilder::VCC_LO_REG, 0);
      if (!Wr0) return Wr0.takeError();
      ISAEncoder::append(TB, *Wr0);

      auto Wr1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, InstructionBuilder::VCC_HI_REG, 1);
      if (!Wr1) return Wr1.takeError();
      ISAEncoder::append(TB, *Wr1);
    }

    auto WrE0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, InstructionBuilder::EXEC_LO_REG, 5);
    if (!WrE0) return WrE0.takeError();
    ISAEncoder::append(TB, *WrE0);

    auto WrE1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, InstructionBuilder::EXEC_HI_REG, 6);
    if (!WrE1) return WrE1.takeError();
    ISAEncoder::append(TB, *WrE1);

    if (!UseRelay) {
      auto WrM0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Enc->getM0Reg(), 3);
      if (!WrM0) return WrM0.takeError();
      ISAEncoder::append(TB, *WrM0);

      auto CSel = Enc->buildAndEmit("S_CSELECT_B32",
          {Op::Reg(Enc->getM0Reg()), Op::Imm(1), Op::Imm(0)});
      if (!CSel) return CSel.takeError();
      ISAEncoder::append(TB, *CSel);

      auto WrSCC = Enc->encodeWriteLane(Scratch.ScratchVGPR, Enc->getM0Reg(), 2);
      if (!WrSCC) return WrSCC.takeError();
      ISAEncoder::append(TB, *WrSCC);
    }
  } else {
    auto Wr0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ReturnAddrSGPR, 0);
    if (!Wr0) return Wr0.takeError();
    ISAEncoder::append(TB, *Wr0);

    auto Wr1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ReturnAddrSGPRHi, 1);
    if (!Wr1) return Wr1.takeError();
    ISAEncoder::append(TB, *Wr1);

    auto CSel = Enc->buildAndEmit("S_CSELECT_B32",
        {Op::Reg(Scratch.ScratchSGPR), Op::Imm(1), Op::Imm(0)});
    if (!CSel) return CSel.takeError();
    ISAEncoder::append(TB, *CSel);

    auto WrSCC = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ScratchSGPR, 2);
    if (!WrSCC) return WrSCC.takeError();
    ISAEncoder::append(TB, *WrSCC);
  }

  // ---- Strategy-specific payload ----

  if (Trace.Strategy == PayloadStrategy::OnGpuReduce && PayloadLevel >= 1) {

  unsigned AddrVGPRLo = RegisterHelper::getVGPR(Site.AddrVGPRIndex);
  unsigned TempSGPR = Scratch.ZeroSGPR
      ? InstructionBuilder::VCC_LO_REG
      : Scratch.ScratchSGPR;

  if (!Scratch.ZeroSGPR) {
    auto W0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, InstructionBuilder::VCC_LO_REG, 3);
    if (!W0) return W0.takeError();
    ISAEncoder::append(TB, *W0);

    auto W1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, InstructionBuilder::VCC_HI_REG, 4);
    if (!W1) return W1.takeError();
    ISAEncoder::append(TB, *W1);

    auto E0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
    if (!E0) return E0.takeError();
    ISAEncoder::append(TB, *E0);

    auto E1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
    if (!E1) return E1.takeError();
    ISAEncoder::append(TB, *E1);

    auto WE0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ExecSaveSGPRLo, 5);
    if (!WE0) return WE0.takeError();
    ISAEncoder::append(TB, *WE0);

    auto WE1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ExecSaveSGPRHi, 6);
    if (!WE1) return WE1.takeError();
    ISAEncoder::append(TB, *WE1);
  }

  if (PayloadLevel >= 2) {

  for (unsigned I = 0; I < 8; ++I) {
    unsigned SGPR = RegisterHelper::getSGPR(I);
    auto W = Enc->encodeWriteLane(Scratch.ScratchVGPR, SGPR, 10 + I);
    if (!W) return W.takeError();
    ISAEncoder::append(TB, *W);
  }

  {
    auto WaitV0 = InstructionBuilder::buildSWaitCnt(
        D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/0);
    if (!WaitV0) return WaitV0.takeError();
    auto WaitV0B = Enc->emitInst(*WaitV0);
    if (!WaitV0B) return WaitV0B.takeError();
    ISAEncoder::append(TB, *WaitV0B);
  }

  {
    constexpr unsigned V0 = RegisterHelper::getVGPR(0);
    auto Sv = InstructionBuilder::buildVMovB32(D, Scratch.TempVGPR, V0);
    if (!Sv) return Sv.takeError();
    auto SvB = Enc->emitInst(*Sv);
    if (!SvB) return SvB.takeError();
    ISAEncoder::append(TB, *SvB);
  }

  {
    constexpr unsigned V1 = RegisterHelper::getVGPR(1);
    constexpr unsigned V2 = RegisterHelper::getVGPR(2);
    constexpr unsigned V3 = RegisterHelper::getVGPR(3);

    auto S1 = Enc->encodeReadLane(TempSGPR, V1, 0);
    if (!S1) return S1.takeError();
    ISAEncoder::append(TB, *S1);
    auto W1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 7);
    if (!W1) return W1.takeError();
    ISAEncoder::append(TB, *W1);

    auto S2 = Enc->encodeReadLane(TempSGPR, V2, 0);
    if (!S2) return S2.takeError();
    ISAEncoder::append(TB, *S2);
    auto W2 = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 8);
    if (!W2) return W2.takeError();
    ISAEncoder::append(TB, *W2);

    auto S3 = Enc->encodeReadLane(TempSGPR, V3, 0);
    if (!S3) return S3.takeError();
    ISAEncoder::append(TB, *S3);
    auto W3 = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 9);
    if (!W3) return W3.takeError();
    ISAEncoder::append(TB, *W3);
  }

  {
    constexpr unsigned V0 = RegisterHelper::getVGPR(0);
    constexpr unsigned S2 = RegisterHelper::getSGPR(2);
    constexpr unsigned S3 = RegisterHelper::getSGPR(3);
    bool IsLDSSite = !Site.IsGlobal;

    if (IsLDSSite) {
      if (Scratch.NeedsAccVGPRSpill || Scratch.NeedsScratchSpill) {
        if (AddrVGPRLo != V0) {
          auto Mv = InstructionBuilder::buildVMovB32(D, V0, AddrVGPRLo);
          if (!Mv) return Mv.takeError();
          auto MvB = Enc->emitInst(*Mv);
          if (!MvB) return MvB.takeError();
          ISAEncoder::append(TB, *MvB);
        }
        auto Shr = InstructionBuilder::buildVLshrrevB32(D, V0, 2, V0);
        if (!Shr) return Shr.takeError();
        auto ShrB = Enc->emitInst(*Shr);
        if (!ShrB) return ShrB.takeError();
        ISAEncoder::append(TB, *ShrB);
      } else {
        auto Shr = InstructionBuilder::buildVLshrrevB32(D, V0, 2, AddrVGPRLo);
        if (!Shr) return Shr.takeError();
        auto ShrB = Enc->emitInst(*Shr);
        if (!ShrB) return ShrB.takeError();
        ISAEncoder::append(TB, *ShrB);
      }
      auto And = Enc->buildAndEmit("V_AND_B32_e64",
          {Op::Reg(V0), Op::Imm(31), Op::Reg(V0)});
      if (!And) return And.takeError();
      ISAEncoder::append(TB, *And);
    } else if (Scratch.NeedsAccVGPRSpill || Scratch.NeedsScratchSpill) {
      if (AddrVGPRLo != V0) {
        auto Mv = InstructionBuilder::buildVMovB32(D, V0, AddrVGPRLo);
        if (!Mv) return Mv.takeError();
        auto MvB = Enc->emitInst(*Mv);
        if (!MvB) return MvB.takeError();
        ISAEncoder::append(TB, *MvB);
      }
      auto Shr = InstructionBuilder::buildVLshrrevB32(D, V0, 7, V0);
      if (!Shr) return Shr.takeError();
      auto ShrB = Enc->emitInst(*Shr);
      if (!ShrB) return ShrB.takeError();
      ISAEncoder::append(TB, *ShrB);
    } else {
      auto Shr = InstructionBuilder::buildVLshrrevB32(D, V0, 7, AddrVGPRLo);
      if (!Shr) return Shr.takeError();
      auto ShrB = Enc->emitInst(*Shr);
      if (!ShrB) return ShrB.takeError();
      ISAEncoder::append(TB, *ShrB);
    }

    {
      auto M0 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(S2), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
      if (!M0) return M0.takeError();
      ISAEncoder::append(TB, *M0);

      auto M1 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(S3), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
      if (!M1) return M1.takeError();
      ISAEncoder::append(TB, *M1);
    }

    {
      const int CountMode = Cfg.Debug.CountMode;

      if (CountMode == 0) {
        for (int i = 0; i < 15; ++i) {
          uint8_t nop[] = {0x00, 0x00, 0x80, 0xBF};
          TB.insert(TB.end(), nop, nop + 4);
        }
      } else if (CountMode == 1) {
        auto MC = Enc->buildAndEmit("S_MOV_B32",
            {Op::Reg(RegisterHelper::getSGPR(0)), Op::Imm(1)});
        if (!MC) return MC.takeError();
        ISAEncoder::append(TB, *MC);
      } else if (CountMode == 2) {
        auto FF = Enc->buildAndEmit("S_FF1_I32_B64",
            {Op::Reg(RegisterHelper::getSGPR(0)),
             Op::Reg(S2), Op::Reg(S3)});
        if (!FF) return FF.takeError();
        ISAEncoder::append(TB, *FF);

        auto RL = Enc->buildAndEmit("V_READLANE_B32",
            {Op::Reg(RegisterHelper::getSGPR(0)),
             Op::Reg(V0),
             Op::Reg(RegisterHelper::getSGPR(0))});
        if (!RL) return RL.takeError();
        ISAEncoder::append(TB, *RL);

        auto MC = Enc->buildAndEmit("S_MOV_B32",
            {Op::Reg(RegisterHelper::getSGPR(0)), Op::Imm(1)});
        if (!MC) return MC.takeError();
        ISAEncoder::append(TB, *MC);
      } else if (CountMode == 3) {
        {
          auto MZ = Enc->buildAndEmit("S_MOV_B32",
              {Op::Reg(RegisterHelper::getSGPR(0)), Op::Imm(0)});
          if (!MZ) return MZ.takeError();
          ISAEncoder::append(TB, *MZ);

          auto CMP = Enc->buildAndEmit("V_CMP_EQ_U32_e64",
              {Op::Reg(RegisterHelper::getSGPR(0)),
               Op::Reg(RegisterHelper::getSGPR(1)),
               Op::Reg(RegisterHelper::getSGPR(0)),
               Op::Reg(V0)});
          if (!CMP) return CMP.takeError();
          ISAEncoder::append(TB, *CMP);

          auto MC = Enc->buildAndEmit("S_MOV_B32",
              {Op::Reg(RegisterHelper::getSGPR(0)), Op::Imm(1)});
          if (!MC) return MC.takeError();
          ISAEncoder::append(TB, *MC);
        }
      } else if (CountMode == 5) {
        {
          auto MZ = Enc->buildAndEmit("S_MOV_B32",
              {Op::Reg(RegisterHelper::getSGPR(0)), Op::Imm(0)});
          if (!MZ) return MZ.takeError();
          ISAEncoder::append(TB, *MZ);

          uint8_t cmp[] = {0x00, 0x00, 0x94, 0x7D};
          TB.insert(TB.end(), cmp, cmp + 4);

          auto MC = Enc->buildAndEmit("S_MOV_B32",
              {Op::Reg(RegisterHelper::getSGPR(0)), Op::Imm(1)});
          if (!MC) return MC.takeError();
          ISAEncoder::append(TB, *MC);
        }
      } else if (CountMode == 4) {
        auto FF = Enc->buildAndEmit("S_FF1_I32_B64",
            {Op::Reg(RegisterHelper::getSGPR(0)),
             Op::Reg(S2), Op::Reg(S3)});
        if (!FF) return FF.takeError();
        ISAEncoder::append(TB, *FF);

        auto RL = Enc->buildAndEmit("V_READLANE_B32",
            {Op::Reg(RegisterHelper::getSGPR(0)),
             Op::Reg(V0),
             Op::Reg(RegisterHelper::getSGPR(0))});
        if (!RL) return RL.takeError();
        ISAEncoder::append(TB, *RL);

        for (int i = 0; i < 10; ++i) {
          auto NP = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
          if (!NP) return NP.takeError();
          ISAEncoder::append(TB, *NP);
        }

        uint8_t cmp[] = {0x00, 0x00, 0x94, 0x7D};
        TB.insert(TB.end(), cmp, cmp + 4);

        auto MC = Enc->buildAndEmit("S_MOV_B32",
            {Op::Reg(RegisterHelper::getSGPR(0)), Op::Imm(1)});
        if (!MC) return MC.takeError();
        ISAEncoder::append(TB, *MC);
      } else {
        const auto &Blob = IsLDSSite ? LDSCountingBytes : CountingBytes;
        TB.insert(TB.end(), Blob.begin(), Blob.end());
      }
    }

    {
      constexpr unsigned S0 = RegisterHelper::getSGPR(0);
      auto MC = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(TempSGPR), Op::Reg(S0)});
      if (!MC) return MC.takeError();
      ISAEncoder::append(TB, *MC);
    }
  }

  if (!Scratch.ZeroSGPR) {
    auto RE0 = Enc->buildAndEmit("V_READLANE_B32",
        {Op::Reg(Scratch.ExecSaveSGPRLo),
         Op::Reg(Scratch.ScratchVGPR),
         Op::Imm(5)});
    if (!RE0) return RE0.takeError();
    ISAEncoder::append(TB, *RE0);

    auto RE1 = Enc->buildAndEmit("V_READLANE_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi),
         Op::Reg(Scratch.ScratchVGPR),
         Op::Imm(6)});
    if (!RE1) return RE1.takeError();
    ISAEncoder::append(TB, *RE1);
  }

  } // PayloadLevel >= 2

  if (PayloadLevel >= 3) {
  {
    auto X0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Imm(1)});
    if (!X0) return X0.takeError();
    ISAEncoder::append(TB, *X0);

    auto X1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Imm(0)});
    if (!X1) return X1.takeError();
    ISAEncoder::append(TB, *X1);
  }

  if (Trace.SupportsGPUAtomics) {
    constexpr unsigned S2 = RegisterHelper::getSGPR(2);
    constexpr unsigned S3 = RegisterHelper::getSGPR(3);
    uint64_t SiteAddr = Trace.BufferAddr +
        static_cast<uint64_t>(SiteIdx) * 8;

    auto Wc = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 19);
    if (!Wc) return Wc.takeError();
    ISAEncoder::append(TB, *Wc);

    auto Sv0 = Enc->encodeReadLane(TempSGPR, Scratch.TempVGPR, 0);
    if (!Sv0) return Sv0.takeError();
    ISAEncoder::append(TB, *Sv0);
    auto Wv0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 18);
    if (!Wv0) return Wv0.takeError();
    ISAEncoder::append(TB, *Wv0);

    auto Rc = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 19);
    if (!Rc) return Rc.takeError();
    ISAEncoder::append(TB, *Rc);
    auto Vc = InstructionBuilder::buildVMovB32(D, Scratch.TempVGPR, TempSGPR);
    if (!Vc) return Vc.takeError();
    auto VcB = Enc->emitInst(*Vc);
    if (!VcB) return VcB.takeError();
    ISAEncoder::append(TB, *VcB);

    auto BLo = InstructionBuilder::buildSMovB32(D, S2,
        static_cast<uint32_t>(SiteAddr & 0xFFFFFFFF));
    if (!BLo) return BLo.takeError();
    auto BLoB = Enc->emitInst(*BLo);
    if (!BLoB) return BLoB.takeError();
    ISAEncoder::append(TB, *BLoB);

    auto BHi = InstructionBuilder::buildSMovB32(D, S3,
        static_cast<uint32_t>(SiteAddr >> 32));
    if (!BHi) return BHi.takeError();
    auto BHiB = Enc->emitInst(*BHi);
    if (!BHiB) return BHiB.takeError();
    ISAEncoder::append(TB, *BHiB);

    auto Vz = InstructionBuilder::buildVMovB32Imm(D, Scratch.LaneVGPR, 0);
    if (!Vz) return Vz.takeError();
    auto VzB = Enc->emitInst(*Vz);
    if (!VzB) return VzB.takeError();
    ISAEncoder::append(TB, *VzB);

    auto A1 = InstructionBuilder::buildGlobalAtomicAddNoRtn(D,
        Scratch.LaneVGPR, Scratch.TempVGPR, S2, 0);
    if (!A1) return A1.takeError();
    auto A1B = Enc->emitInst(*A1);
    if (!A1B) return A1B.takeError();
    ISAEncoder::append(TB, *A1B);

    auto V1i = InstructionBuilder::buildVMovB32Imm(D, Scratch.TempVGPR, 1);
    if (!V1i) return V1i.takeError();
    auto V1B = Enc->emitInst(*V1i);
    if (!V1B) return V1B.takeError();
    ISAEncoder::append(TB, *V1B);

    auto A2 = InstructionBuilder::buildGlobalAtomicAddNoRtn(D,
        Scratch.LaneVGPR, Scratch.TempVGPR, S2, 4);
    if (!A2) return A2.takeError();
    auto A2B = Enc->emitInst(*A2);
    if (!A2B) return A2B.takeError();
    ISAEncoder::append(TB, *A2B);

    auto Rv0L = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 18);
    if (!Rv0L) return Rv0L.takeError();
    ISAEncoder::append(TB, *Rv0L);
    auto Wv0L = Enc->encodeWriteLane(Scratch.TempVGPR, TempSGPR, 0);
    if (!Wv0L) return Wv0L.takeError();
    ISAEncoder::append(TB, *Wv0L);
  } else {
    constexpr unsigned S2 = RegisterHelper::getSGPR(2);
    constexpr unsigned S3 = RegisterHelper::getSGPR(3);
    constexpr unsigned S4 = RegisterHelper::getSGPR(4);

    uint64_t SiteAddr = Trace.BufferAddr +
        static_cast<uint64_t>(SiteIdx) * 8;

    auto BLo = InstructionBuilder::buildSMovB32(D, S2,
        static_cast<uint32_t>(SiteAddr & 0xFFFFFFFF));
    if (!BLo) return BLo.takeError();
    auto BLoB = Enc->emitInst(*BLo);
    if (!BLoB) return BLoB.takeError();
    ISAEncoder::append(TB, *BLoB);

    auto BHi = InstructionBuilder::buildSMovB32(D, S3,
        static_cast<uint32_t>(SiteAddr >> 32));
    if (!BHi) return BHi.takeError();
    auto BHiB = Enc->emitInst(*BHi);
    if (!BHiB) return BHiB.takeError();
    ISAEncoder::append(TB, *BHiB);

    auto MC = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(S4), Op::Reg(TempSGPR)});
    if (!MC) return MC.takeError();
    ISAEncoder::append(TB, *MC);

    TB.insert(TB.end(), AtomicBytesNonAtomic.begin(),
              AtomicBytesNonAtomic.end());
  }
  } // PayloadLevel >= 3

  if (PayloadLevel >= 2) {
    for (unsigned I = 0; I < 8; ++I) {
      unsigned SGPR = RegisterHelper::getSGPR(I);
      auto R = Enc->encodeReadLane(SGPR, Scratch.ScratchVGPR, 10 + I);
      if (!R) return R.takeError();
      ISAEncoder::append(TB, *R);
    }
  }

  // Restore EXEC
  if (Scratch.ZeroSGPR) {
    auto RE0 = Enc->encodeReadLane(InstructionBuilder::VCC_LO_REG, Scratch.ScratchVGPR, 5);
    if (!RE0) return RE0.takeError();
    ISAEncoder::append(TB, *RE0);
    auto ME0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(InstructionBuilder::VCC_LO_REG)});
    if (!ME0) return ME0.takeError();
    ISAEncoder::append(TB, *ME0);

    auto RE1 = Enc->encodeReadLane(InstructionBuilder::VCC_LO_REG, Scratch.ScratchVGPR, 6);
    if (!RE1) return RE1.takeError();
    ISAEncoder::append(TB, *RE1);
    auto ME1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(InstructionBuilder::VCC_LO_REG)});
    if (!ME1) return ME1.takeError();
    ISAEncoder::append(TB, *ME1);
  } else {
    auto R0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(Scratch.ExecSaveSGPRLo)});
    if (!R0) return R0.takeError();
    ISAEncoder::append(TB, *R0);

    auto R1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(Scratch.ExecSaveSGPRHi)});
    if (!R1) return R1.takeError();
    ISAEncoder::append(TB, *R1);
  }

  // Restore v0
  if (PayloadLevel >= 2) {
    constexpr unsigned V0 = RegisterHelper::getVGPR(0);
    auto Rv = InstructionBuilder::buildVMovB32(D, V0, Scratch.TempVGPR);
    if (!Rv) return Rv.takeError();
    auto RvB = Enc->emitInst(*Rv);
    if (!RvB) return RvB.takeError();
    ISAEncoder::append(TB, *RvB);
  }

  // Restore v1-v3 lane 0
  if (PayloadLevel >= 2) {
    constexpr unsigned V1 = RegisterHelper::getVGPR(1);
    constexpr unsigned V2 = RegisterHelper::getVGPR(2);
    constexpr unsigned V3 = RegisterHelper::getVGPR(3);

    auto R1 = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 7);
    if (!R1) return R1.takeError();
    ISAEncoder::append(TB, *R1);
    auto W1 = Enc->encodeWriteLane(V1, TempSGPR, 0);
    if (!W1) return W1.takeError();
    ISAEncoder::append(TB, *W1);

    auto R2 = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 8);
    if (!R2) return R2.takeError();
    ISAEncoder::append(TB, *R2);
    auto W2 = Enc->encodeWriteLane(V2, TempSGPR, 0);
    if (!W2) return W2.takeError();
    ISAEncoder::append(TB, *W2);

    auto R3 = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 9);
    if (!R3) return R3.takeError();
    ISAEncoder::append(TB, *R3);
    auto W3 = Enc->encodeWriteLane(V3, TempSGPR, 0);
    if (!W3) return W3.takeError();
    ISAEncoder::append(TB, *W3);
  }

  // Restore VCC
  if (!UseRelay) {
    unsigned VCCLane0 = Scratch.ZeroSGPR ? 0 : 3;
    unsigned VCCLane1 = Scratch.ZeroSGPR ? 1 : 4;

    auto R0 = Enc->encodeReadLane(InstructionBuilder::VCC_LO_REG, Scratch.ScratchVGPR, VCCLane0);
    if (!R0) return R0.takeError();
    ISAEncoder::append(TB, *R0);

    auto R1 = Enc->encodeReadLane(InstructionBuilder::VCC_HI_REG, Scratch.ScratchVGPR, VCCLane1);
    if (!R1) return R1.takeError();
    ISAEncoder::append(TB, *R1);

    auto NP = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
    if (!NP) return NP.takeError();
    ISAEncoder::append(TB, *NP);
  }
  } // OnGpuReduce

  else {
  // FullCapture payload -- same code, just via Enc
  if (PayloadLevel >= 1) {
    auto E0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
    if (!E0) return E0.takeError();
    ISAEncoder::append(TB, *E0);

    auto E1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
    if (!E1) return E1.takeError();
    ISAEncoder::append(TB, *E1);

    auto X0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Imm(1)});
    if (!X0) return X0.takeError();
    ISAEncoder::append(TB, *X0);

    auto X1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Imm(0)});
    if (!X1) return X1.takeError();
    ISAEncoder::append(TB, *X1);

    if (PayloadLevel >= 2) {
      auto MvLo = InstructionBuilder::buildSMovB32(D, Scratch.ReturnAddrSGPR,
          static_cast<uint32_t>(Trace.CounterAddr & 0xFFFFFFFF));
      if (!MvLo) return MvLo.takeError();
      auto MvLoB = Enc->emitInst(*MvLo);
      if (!MvLoB) return MvLoB.takeError();
      ISAEncoder::append(TB, *MvLoB);

      auto MvHi = InstructionBuilder::buildSMovB32(D, Scratch.ReturnAddrSGPRHi,
          static_cast<uint32_t>(Trace.CounterAddr >> 32));
      if (!MvHi) return MvHi.takeError();
      auto MvHiB = Enc->emitInst(*MvHi);
      if (!MvHiB) return MvHiB.takeError();
      ISAEncoder::append(TB, *MvHiB);

      auto VmovVal = InstructionBuilder::buildVMovB32Imm(D, Scratch.TempVGPR,
                                                          TraceConfig::RecordSize);
      if (!VmovVal) return VmovVal.takeError();
      auto VmovValB = Enc->emitInst(*VmovVal);
      if (!VmovValB) return VmovValB.takeError();
      ISAEncoder::append(TB, *VmovValB);

      auto VmovZero = InstructionBuilder::buildVMovB32Imm(D, Scratch.LaneVGPR, 0);
      if (!VmovZero) return VmovZero.takeError();
      auto VmovZeroB = Enc->emitInst(*VmovZero);
      if (!VmovZeroB) return VmovZeroB.takeError();
      ISAEncoder::append(TB, *VmovZeroB);

      auto Atomic = InstructionBuilder::buildGlobalAtomicAddRtn(
          D, Scratch.TempVGPR, Scratch.LaneVGPR, Scratch.TempVGPR,
          RetAddrSGPRPair, /*Offset=*/0);
      if (!Atomic) return Atomic.takeError();
      auto AtomicB = Enc->emitInst(*Atomic);
      if (!AtomicB) return AtomicB.takeError();
      ISAEncoder::append(TB, *AtomicB);

      auto Wait = InstructionBuilder::buildSWaitCnt(D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/15);
      if (!Wait) return Wait.takeError();
      auto WaitB = Enc->emitInst(*Wait);
      if (!WaitB) return WaitB.takeError();
      ISAEncoder::append(TB, *WaitB);
    }

    auto R0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(Scratch.ExecSaveSGPRLo)});
    if (!R0) return R0.takeError();
    ISAEncoder::append(TB, *R0);

    auto R1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(Scratch.ExecSaveSGPRHi)});
    if (!R1) return R1.takeError();
    ISAEncoder::append(TB, *R1);
  }

  if (PayloadLevel >= 3) {
    auto RFL = InstructionBuilder::buildVReadFirstLaneB32(
        D, Scratch.ScratchSGPR, Scratch.TempVGPR);
    if (!RFL) return RFL.takeError();
    auto RFLB = Enc->emitInst(*RFL);
    if (!RFLB) return RFLB.takeError();
    ISAEncoder::append(TB, *RFLB);

    auto BLo = InstructionBuilder::buildSMovB32(D, Scratch.ReturnAddrSGPR,
        static_cast<uint32_t>(Trace.BufferAddr & 0xFFFFFFFF));
    if (!BLo) return BLo.takeError();
    auto BLoB = Enc->emitInst(*BLo);
    if (!BLoB) return BLoB.takeError();
    ISAEncoder::append(TB, *BLoB);

    auto BHi = InstructionBuilder::buildSMovB32(D, Scratch.ReturnAddrSGPRHi,
        static_cast<uint32_t>(Trace.BufferAddr >> 32));
    if (!BHi) return BHi.takeError();
    auto BHiB = Enc->emitInst(*BHi);
    if (!BHiB) return BHiB.takeError();
    ISAEncoder::append(TB, *BHiB);

    auto Add = InstructionBuilder::buildSAddU32Reg(
        D, Scratch.ReturnAddrSGPR, Scratch.ReturnAddrSGPR, Scratch.ScratchSGPR);
    if (!Add) return Add.takeError();
    auto AddB = Enc->emitInst(*Add);
    if (!AddB) return AddB.takeError();
    ISAEncoder::append(TB, *AddB);

    auto Adc = InstructionBuilder::buildSAddcU32(
        D, Scratch.ReturnAddrSGPRHi, Scratch.ReturnAddrSGPRHi, 0);
    if (!Adc) return Adc.takeError();
    auto AdcB = Enc->emitInst(*Adc);
    if (!AdcB) return AdcB.takeError();
    ISAEncoder::append(TB, *AdcB);

    auto VmovSite = InstructionBuilder::buildVMovB32Imm(D, Scratch.TempVGPR, SiteIdx);
    if (!VmovSite) return VmovSite.takeError();
    auto VmovSiteB = Enc->emitInst(*VmovSite);
    if (!VmovSiteB) return VmovSiteB.takeError();
    ISAEncoder::append(TB, *VmovSiteB);

    auto VmovZero = InstructionBuilder::buildVMovB32Imm(D, Scratch.LaneVGPR, 0);
    if (!VmovZero) return VmovZero.takeError();
    auto VmovZeroB = Enc->emitInst(*VmovZero);
    if (!VmovZeroB) return VmovZeroB.takeError();
    ISAEncoder::append(TB, *VmovZeroB);

    auto GStore = InstructionBuilder::buildGlobalStoreDword(
        D, Scratch.LaneVGPR, Scratch.TempVGPR, RetAddrSGPRPair, /*Offset=*/0);
    if (!GStore) return GStore.takeError();
    auto GStoreB = Enc->emitInst(*GStore);
    if (!GStoreB) return GStoreB.takeError();
    ISAEncoder::append(TB, *GStoreB);

    auto Mlo = InstructionBuilder::buildVMbcntLoU32B32(
        D, Scratch.LaneVGPR, 0xFFFFFFFF, 0);
    if (!Mlo) return Mlo.takeError();
    auto MloB = Enc->emitInst(*Mlo);
    if (!MloB) return MloB.takeError();
    ISAEncoder::append(TB, *MloB);

    auto Mhi = InstructionBuilder::buildVMbcntHiU32B32(
        D, Scratch.LaneVGPR, 0xFFFFFFFF, Scratch.LaneVGPR);
    if (!Mhi) return Mhi.takeError();
    auto MhiB = Enc->emitInst(*Mhi);
    if (!MhiB) return MhiB.takeError();
    ISAEncoder::append(TB, *MhiB);

    auto Shl = InstructionBuilder::buildVLshlrevB32(
        D, Scratch.LaneVGPR, 3, Scratch.LaneVGPR);
    if (!Shl) return Shl.takeError();
    auto ShlB = Enc->emitInst(*Shl);
    if (!ShlB) return ShlB.takeError();
    ISAEncoder::append(TB, *ShlB);

    unsigned AddrVGPRLo = RegisterHelper::getVGPR(Site.AddrVGPRIndex);
    auto GStore2 = InstructionBuilder::buildGlobalStoreDword(
        D, Scratch.LaneVGPR, AddrVGPRLo, RetAddrSGPRPair, /*Offset=*/8);
    if (!GStore2) return GStore2.takeError();
    auto GStore2B = Enc->emitInst(*GStore2);
    if (!GStore2B) return GStore2B.takeError();
    ISAEncoder::append(TB, *GStore2B);

    if (Site.Addr64) {
      unsigned AddrVGPRHi = RegisterHelper::getVGPR(Site.AddrVGPRIndex + 1);
      auto GStoreHi = InstructionBuilder::buildGlobalStoreDword(
          D, Scratch.LaneVGPR, AddrVGPRHi, RetAddrSGPRPair, /*Offset=*/8 + 4);
      if (!GStoreHi) return GStoreHi.takeError();
      auto GStoreHiB = Enc->emitInst(*GStoreHi);
      if (!GStoreHiB) return GStoreHiB.takeError();
      ISAEncoder::append(TB, *GStoreHiB);
    }
  }
  } // FullCapture

  // ---- Common epilogue ----
  if (Scratch.ZeroSGPR && !UseRelay) {
    auto RdSCC = Enc->encodeReadLane(Enc->getM0Reg(), Scratch.ScratchVGPR, 2);
    if (!RdSCC) return RdSCC.takeError();
    ISAEncoder::append(TB, *RdSCC);

    auto SccNop = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
    if (!SccNop) return SccNop.takeError();
    ISAEncoder::append(TB, *SccNop);

    auto Cmp = Enc->buildAndEmit("S_CMP_LG_U32",
        {Op::Reg(Enc->getM0Reg()), Op::Imm(0)});
    if (!Cmp) return Cmp.takeError();
    ISAEncoder::append(TB, *Cmp);

    auto ReVCC = Enc->encodeReadLane(InstructionBuilder::VCC_LO_REG, Scratch.ScratchVGPR, 0);
    if (!ReVCC) return ReVCC.takeError();
    ISAEncoder::append(TB, *ReVCC);

    auto RdM0 = Enc->encodeReadLane(Enc->getM0Reg(), Scratch.ScratchVGPR, 3);
    if (!RdM0) return RdM0.takeError();
    ISAEncoder::append(TB, *RdM0);
  } else if (!Scratch.ZeroSGPR) {
    auto Rd0 = Enc->encodeReadLane(Scratch.ReturnAddrSGPR, Scratch.ScratchVGPR, 0);
    if (!Rd0) return Rd0.takeError();
    ISAEncoder::append(TB, *Rd0);

    auto Rd1 = Enc->encodeReadLane(Scratch.ReturnAddrSGPRHi, Scratch.ScratchVGPR, 1);
    if (!Rd1) return Rd1.takeError();
    ISAEncoder::append(TB, *Rd1);

    auto RdSCC = Enc->encodeReadLane(Scratch.ScratchSGPR, Scratch.ScratchVGPR, 2);
    if (!RdSCC) return RdSCC.takeError();
    ISAEncoder::append(TB, *RdSCC);

    auto SccNop = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
    if (!SccNop) return SccNop.takeError();
    ISAEncoder::append(TB, *SccNop);

    auto Cmp = Enc->buildAndEmit("S_CMP_LG_U32",
        {Op::Reg(Scratch.ScratchSGPR), Op::Imm(0)});
    if (!Cmp) return Cmp.takeError();
    ISAEncoder::append(TB, *Cmp);
  }

  // ---- Scratch memory restore ----
  if (Scratch.NeedsScratchSpill) {
    if (Scratch.ZeroSGPR) {
      // In ZeroSGPR mode, VCC_LO is the only SADDR temp for large offsets.
      // For small offsets (<= 4095) the immediate encoding is used and the
      // TempSGPR arg is ignored, so passing VCC_LO is harmless.
      // For large offsets, VCC_LO has already been restored (readlane above),
      // so we must stash it in M0 before the SADDR path clobbers it.
      unsigned RestoreTempSGPR = InstructionBuilder::VCC_LO_REG;
      bool NeedVCCStash = Scratch.ScratchSpillOffset > SCRATCH_OFFSET_IMM_MAX;

      if (NeedVCCStash) {
        auto StashVCC = Enc->buildAndEmit("S_MOV_B32",
            {Op::Reg(Enc->getM0Reg()), Op::Reg(InstructionBuilder::VCC_LO_REG)});
        if (!StashVCC) return StashVCC.takeError();
        ISAEncoder::append(TB, *StashVCC);
      }

      if (auto E = appendScratchLoad(TB, Scratch.LaneVGPR,
              Scratch.ScratchSpillOffset + 4, RestoreTempSGPR))
        return std::move(E);

      if (auto E = appendScratchLoad(TB, Scratch.TempVGPR,
              Scratch.ScratchSpillOffset + 8, RestoreTempSGPR))
        return std::move(E);

      if (!UseRelay) {
        if (auto E = appendScratchLoad(TB, Scratch.ScratchVGPR,
                Scratch.ScratchSpillOffset + 0, RestoreTempSGPR))
          return std::move(E);
      }

      {
        auto WaitLoad = InstructionBuilder::buildSWaitCnt(
            D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/15);
        if (!WaitLoad) return WaitLoad.takeError();
        auto WaitLoadB = Enc->emitInst(*WaitLoad);
        if (!WaitLoadB) return WaitLoadB.takeError();
        ISAEncoder::append(TB, *WaitLoadB);
      }

      if (NeedVCCStash) {
        auto RestoreVCC = Enc->buildAndEmit("S_MOV_B32",
            {Op::Reg(InstructionBuilder::VCC_LO_REG), Op::Reg(Enc->getM0Reg())});
        if (!RestoreVCC) return RestoreVCC.takeError();
        ISAEncoder::append(TB, *RestoreVCC);
      }
    } else {
      auto SE0 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
      if (!SE0) return SE0.takeError();
      ISAEncoder::append(TB, *SE0);

      auto SE1 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
      if (!SE1) return SE1.takeError();
      ISAEncoder::append(TB, *SE1);

      auto XA0 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Imm(-1)});
      if (!XA0) return XA0.takeError();
      ISAEncoder::append(TB, *XA0);

      auto XA1 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Imm(-1)});
      if (!XA1) return XA1.takeError();
      ISAEncoder::append(TB, *XA1);

      if (auto E = appendScratchLoad(TB, Scratch.ScratchVGPR,
              Scratch.ScratchSpillOffset + 0, Scratch.ScratchSGPR))
        return std::move(E);

      if (auto E = appendScratchLoad(TB, Scratch.LaneVGPR,
              Scratch.ScratchSpillOffset + 4, Scratch.ScratchSGPR))
        return std::move(E);

      if (auto E = appendScratchLoad(TB, Scratch.TempVGPR,
              Scratch.ScratchSpillOffset + 8, Scratch.ScratchSGPR))
        return std::move(E);

      {
        auto WaitLoad = InstructionBuilder::buildSWaitCnt(
            D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/15);
        if (!WaitLoad) return WaitLoad.takeError();
        auto WaitLoadB = Enc->emitInst(*WaitLoad);
        if (!WaitLoadB) return WaitLoadB.takeError();
        ISAEncoder::append(TB, *WaitLoadB);
      }

      auto RE0 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(Scratch.ExecSaveSGPRLo)});
      if (!RE0) return RE0.takeError();
      ISAEncoder::append(TB, *RE0);

      auto RE1 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(Scratch.ExecSaveSGPRHi)});
      if (!RE1) return RE1.takeError();
      ISAEncoder::append(TB, *RE1);
    }
  }

  // ---- AccVGPR restore (DEPRECATED) ----
  if (Scratch.NeedsAccVGPRSpill) {
    auto SE0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
    if (!SE0) return SE0.takeError();
    ISAEncoder::append(TB, *SE0);

    auto SE1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
    if (!SE1) return SE1.takeError();
    ISAEncoder::append(TB, *SE1);

    auto XA0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Imm(-1)});
    if (!XA0) return XA0.takeError();
    ISAEncoder::append(TB, *XA0);

    auto XA1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Imm(-1)});
    if (!XA1) return XA1.takeError();
    ISAEncoder::append(TB, *XA1);

    auto Rest0 = Enc->encodeAccVGPRRead(Scratch.ScratchVGPR, Scratch.SpillAGPR0);
    if (!Rest0) return Rest0.takeError();
    ISAEncoder::append(TB, *Rest0);

    auto Rest1 = Enc->encodeAccVGPRRead(Scratch.LaneVGPR, Scratch.SpillAGPR1);
    if (!Rest1) return Rest1.takeError();
    ISAEncoder::append(TB, *Rest1);

    auto Rest2 = Enc->encodeAccVGPRRead(Scratch.TempVGPR, Scratch.SpillAGPR2);
    if (!Rest2) return Rest2.takeError();
    ISAEncoder::append(TB, *Rest2);

    auto RE0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(Scratch.ExecSaveSGPRLo)});
    if (!RE0) return RE0.takeError();
    ISAEncoder::append(TB, *RE0);

    auto RE1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(Scratch.ExecSaveSGPRHi)});
    if (!RE1) return RE1.takeError();
    ISAEncoder::append(TB, *RE1);
  }

  // Drain trampoline VMEM traffic.
  // Only emit vmcnt(0) when the trampoline itself issued VMEM ops
  // (e.g., atomic accumulator at PayloadLevel >= 2/3).  With an empty
  // or counting-only payload, no trampoline VMEM is outstanding, and
  // draining the kernel's pending loads serializes them unnecessarily.
  // On gfx9, vmcnt(63) is effectively a no-op (max counter value).
  {
    unsigned DrainVm = 63;  // default: don't drain kernel's pending loads
    if (Trace.SupportsGPUAtomics &&
        Trace.Strategy == PayloadStrategy::OnGpuReduce) {
      if (Scratch.NeedsAccVGPRSpill)
        DrainVm = Site.PreSpillVmWait;
      else if (PayloadLevel >= 2)
        DrainVm = 0;  // trampoline issued VMEM (atomic), must drain
    }
    auto WaitDrain = InstructionBuilder::buildSWaitCnt(
        D, DrainVm, /*ExpCnt=*/7, /*LgkmCnt=*/15);
    if (!WaitDrain) return WaitDrain.takeError();
    auto WaitDrainB = Enc->emitInst(*WaitDrain);
    if (!WaitDrainB) return WaitDrainB.takeError();
    ISAEncoder::append(TB, *WaitDrainB);
  }

  return TB;
}

//===----------------------------------------------------------------------===//
// Shared Body Trampoline: emitSharedBody
//===----------------------------------------------------------------------===//

Expected<std::vector<uint8_t>>
TrampolineEmitter::emitSharedBody(const ScratchRegisters &Scratch,
                                   const TraceConfig &Trace,
                                   bool IsLDS,
                                   unsigned RetAddrSGPRPair,
                                   unsigned maxPreSpillVmWait,
                                   int32_t GetpcToReturnTableOffset) {
  using Op = InstructionBuilder::Operand;
  auto &D = Enc->getDisassembler();
  std::vector<uint8_t> TB;

  unsigned TempSGPR = Scratch.ScratchSGPR;

  // ---- Scratch memory spill: save victim VGPRs (if needed) ----
  if (Scratch.NeedsScratchSpill) {
    {
      auto WaitSpill = InstructionBuilder::buildSWaitCnt(
          D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/0);
      if (!WaitSpill) return WaitSpill.takeError();
      auto WaitSpillB = Enc->emitInst(*WaitSpill);
      if (!WaitSpillB) return WaitSpillB.takeError();
      ISAEncoder::append(TB, *WaitSpillB);
    }

    auto SE0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
    if (!SE0) return SE0.takeError();
    ISAEncoder::append(TB, *SE0);
    auto SE1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
    if (!SE1) return SE1.takeError();
    ISAEncoder::append(TB, *SE1);

    auto XA0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Imm(-1)});
    if (!XA0) return XA0.takeError();
    ISAEncoder::append(TB, *XA0);
    auto XA1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Imm(-1)});
    if (!XA1) return XA1.takeError();
    ISAEncoder::append(TB, *XA1);

    if (auto E = appendScratchStore(TB, Scratch.ScratchVGPR,
            Scratch.ScratchSpillOffset + 0, Scratch.SAddrTempSGPR))
      return std::move(E);

    if (auto E = appendScratchStore(TB, Scratch.LaneVGPR,
            Scratch.ScratchSpillOffset + 4, Scratch.SAddrTempSGPR))
      return std::move(E);

    if (auto E = appendScratchStore(TB, Scratch.TempVGPR,
            Scratch.ScratchSpillOffset + 8, Scratch.SAddrTempSGPR))
      return std::move(E);

    {
      auto WaitStore = InstructionBuilder::buildSWaitCnt(
          D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/15);
      if (!WaitStore) return WaitStore.takeError();
      auto WaitStoreB = Enc->emitInst(*WaitStore);
      if (!WaitStoreB) return WaitStoreB.takeError();
      ISAEncoder::append(TB, *WaitStoreB);
    }

    auto RE0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(Scratch.ExecSaveSGPRLo)});
    if (!RE0) return RE0.takeError();
    ISAEncoder::append(TB, *RE0);
    auto RE1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(Scratch.ExecSaveSGPRHi)});
    if (!RE1) return RE1.takeError();
    ISAEncoder::append(TB, *RE1);
  }

  // ---- Save kernel return addr to ScratchVGPR lanes 0-1 ----
  {
    auto Wr0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ReturnAddrSGPR, 0);
    if (!Wr0) return Wr0.takeError();
    ISAEncoder::append(TB, *Wr0);

    auto Wr1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ReturnAddrSGPRHi, 1);
    if (!Wr1) return Wr1.takeError();
    ISAEncoder::append(TB, *Wr1);
  }

  // ---- SCC snapshot (must happen before any SCC-clobbering instructions) ----
  // ReturnAddrSGPR is free (already saved to lanes 0-1), so use it as dest
  // to keep ScratchSGPR (PackedInfo from dispatch entry) intact for unpacking.
  {
    auto CSel = Enc->buildAndEmit("S_CSELECT_B32",
        {Op::Reg(Scratch.ReturnAddrSGPR), Op::Imm(1), Op::Imm(0)});
    if (!CSel) return CSel.takeError();
    ISAEncoder::append(TB, *CSel);

    auto WrSCC = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ReturnAddrSGPR, 2);
    if (!WrSCC) return WrSCC.takeError();
    ISAEncoder::append(TB, *WrSCC);
  }

  // ---- Unpack packed_info from ScratchSGPR ----
  // ScratchSGPR = (SiteIdx | AddrVGPRIdx<<16 | IsLDS<<24 | 1<<25)
  // Use ReturnAddrSGPR as temp (it's been saved to ScratchVGPR already)
  {
    auto AndIdx = Enc->buildAndEmit("S_AND_B32",
        {Op::Reg(Scratch.ReturnAddrSGPR), Op::Reg(Scratch.ScratchSGPR), Op::Imm(0xFFFF)});
    if (!AndIdx) return AndIdx.takeError();
    ISAEncoder::append(TB, *AndIdx);

    auto WrIdx = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ReturnAddrSGPR, 18);
    if (!WrIdx) return WrIdx.takeError();
    ISAEncoder::append(TB, *WrIdx);

    auto Shr = Enc->buildAndEmit("S_LSHR_B32",
        {Op::Reg(Scratch.ReturnAddrSGPR), Op::Reg(Scratch.ScratchSGPR), Op::Imm(16)});
    if (!Shr) return Shr.takeError();
    ISAEncoder::append(TB, *Shr);

    auto AndVGPR = Enc->buildAndEmit("S_AND_B32",
        {Op::Reg(Scratch.ReturnAddrSGPR), Op::Reg(Scratch.ReturnAddrSGPR), Op::Imm(0xFF)});
    if (!AndVGPR) return AndVGPR.takeError();
    ISAEncoder::append(TB, *AndVGPR);

    auto WrVGPR = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ReturnAddrSGPR, 19);
    if (!WrVGPR) return WrVGPR.takeError();
    ISAEncoder::append(TB, *WrVGPR);
  }

  // ---- VCC save ----
  {
    auto W0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, InstructionBuilder::VCC_LO_REG, 3);
    if (!W0) return W0.takeError();
    ISAEncoder::append(TB, *W0);

    auto W1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, InstructionBuilder::VCC_HI_REG, 4);
    if (!W1) return W1.takeError();
    ISAEncoder::append(TB, *W1);
  }

  // ---- EXEC save + writelane ----
  {
    auto E0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
    if (!E0) return E0.takeError();
    ISAEncoder::append(TB, *E0);

    auto E1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
    if (!E1) return E1.takeError();
    ISAEncoder::append(TB, *E1);

    auto WE0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ExecSaveSGPRLo, 5);
    if (!WE0) return WE0.takeError();
    ISAEncoder::append(TB, *WE0);

    auto WE1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, Scratch.ExecSaveSGPRHi, 6);
    if (!WE1) return WE1.takeError();
    ISAEncoder::append(TB, *WE1);
  }

  // ---- s0-s7 save ----
  for (unsigned I = 0; I < 8; ++I) {
    unsigned SGPR = RegisterHelper::getSGPR(I);
    auto W = Enc->encodeWriteLane(Scratch.ScratchVGPR, SGPR, 10 + I);
    if (!W) return W.takeError();
    ISAEncoder::append(TB, *W);
  }

  // ---- Wait for all pending VMEM/LDS ----
  {
    auto WaitV0 = InstructionBuilder::buildSWaitCnt(
        D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/0);
    if (!WaitV0) return WaitV0.takeError();
    auto WaitV0B = Enc->emitInst(*WaitV0);
    if (!WaitV0B) return WaitV0B.takeError();
    ISAEncoder::append(TB, *WaitV0B);
  }

  // ---- v0 save ----
  {
    constexpr unsigned V0 = RegisterHelper::getVGPR(0);
    auto Sv = InstructionBuilder::buildVMovB32(D, Scratch.TempVGPR, V0);
    if (!Sv) return Sv.takeError();
    auto SvB = Enc->emitInst(*Sv);
    if (!SvB) return SvB.takeError();
    ISAEncoder::append(TB, *SvB);
  }

  // ---- v1-v3 lane 0 save ----
  {
    constexpr unsigned V1 = RegisterHelper::getVGPR(1);
    constexpr unsigned V2 = RegisterHelper::getVGPR(2);
    constexpr unsigned V3 = RegisterHelper::getVGPR(3);

    auto S1 = Enc->encodeReadLane(TempSGPR, V1, 0);
    if (!S1) return S1.takeError();
    ISAEncoder::append(TB, *S1);
    auto W1 = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 7);
    if (!W1) return W1.takeError();
    ISAEncoder::append(TB, *W1);

    auto S2 = Enc->encodeReadLane(TempSGPR, V2, 0);
    if (!S2) return S2.takeError();
    ISAEncoder::append(TB, *S2);
    auto W2 = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 8);
    if (!W2) return W2.takeError();
    ISAEncoder::append(TB, *W2);

    auto S3 = Enc->encodeReadLane(TempSGPR, V3, 0);
    if (!S3) return S3.takeError();
    ISAEncoder::append(TB, *S3);
    auto W3 = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 9);
    if (!W3) return W3.takeError();
    ISAEncoder::append(TB, *W3);
  }

  // ---- Indirect VGPR access: copy v[AddrVGPR] to v0 ----
  // GFX950 (CDNA3) does not support v_movrels_b32. Use a computed-jump
  // table instead: each entry contains v_mov_b32 v0, v[N]; s_branch back.
  // s0:s1 (saved in v253 lanes 10-11) serve as the getpc pair.
  {
    constexpr unsigned V0 = RegisterHelper::getVGPR(0);
    constexpr unsigned S0 = RegisterHelper::getSGPR(0);
    constexpr unsigned S1 = RegisterHelper::getSGPR(1);
    constexpr unsigned NumVGPREntries = 256;

    auto RdIdx = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 19);
    if (!RdIdx) return RdIdx.takeError();
    ISAEncoder::append(TB, *RdIdx);

    // TempSGPR = AddrVGPRIdx * 8 (each table entry is 8 bytes)
    auto Shl = Enc->buildAndEmit("S_LSHL_B32",
        {Op::Reg(TempSGPR), Op::Reg(TempSGPR), Op::Imm(3)});
    if (!Shl) return Shl.takeError();
    ISAEncoder::append(TB, *Shl);

    // +12: skip the 3 instructions between s_getpc and table start
    auto AddFixed = Enc->buildAndEmit("S_ADD_U32",
        {Op::Reg(TempSGPR), Op::Reg(TempSGPR), Op::Imm(12)});
    if (!AddFixed) return AddFixed.takeError();
    ISAEncoder::append(TB, *AddFixed);

    auto PairOrErr = Enc->resolveSGPRPair(0);
    if (!PairOrErr) return PairOrErr.takeError();
    unsigned JumpPair = *PairOrErr;

    auto GetPC = Enc->buildAndEmit("S_GETPC_B64", {Op::Reg(JumpPair)});
    if (!GetPC) return GetPC.takeError();
    ISAEncoder::append(TB, *GetPC);

    auto AddLo = InstructionBuilder::buildSAddU32Reg(D, S0, S0, TempSGPR);
    if (!AddLo) return AddLo.takeError();
    auto AddLoB = Enc->emitInst(*AddLo);
    if (!AddLoB) return AddLoB.takeError();
    ISAEncoder::append(TB, *AddLoB);

    auto AdcHi = InstructionBuilder::buildSAddcU32(D, S1, S1, 0);
    if (!AdcHi) return AdcHi.takeError();
    auto AdcHiB = Enc->emitInst(*AdcHi);
    if (!AdcHiB) return AdcHiB.takeError();
    ISAEncoder::append(TB, *AdcHiB);

    auto SetPC = Enc->encodeSetPC(JumpPair);
    if (!SetPC) return SetPC.takeError();
    ISAEncoder::append(TB, *SetPC);

    for (unsigned I = 0; I < NumVGPREntries; ++I) {
      unsigned VI = RegisterHelper::getVGPR(I);
      auto Mov = InstructionBuilder::buildVMovB32(D, V0, VI);
      if (!Mov) return Mov.takeError();
      auto MovB = Enc->emitInst(*Mov);
      if (!MovB) return MovB.takeError();
      ISAEncoder::append(TB, *MovB);

      int16_t BrDwords = static_cast<int16_t>((NumVGPREntries - 1 - I) * 2);
      auto Br = Enc->encodeSBranch(BrDwords);
      if (!Br) return Br.takeError();
      ISAEncoder::append(TB, *Br);
    }
  }

  // ---- Cache line / bank computation ----
  {
    constexpr unsigned V0 = RegisterHelper::getVGPR(0);
    constexpr unsigned S2 = RegisterHelper::getSGPR(2);
    constexpr unsigned S3 = RegisterHelper::getSGPR(3);

    if (IsLDS) {
      auto Shr = InstructionBuilder::buildVLshrrevB32(D, V0, 2, V0);
      if (!Shr) return Shr.takeError();
      auto ShrB = Enc->emitInst(*Shr);
      if (!ShrB) return ShrB.takeError();
      ISAEncoder::append(TB, *ShrB);

      auto And = Enc->buildAndEmit("V_AND_B32_e64",
          {Op::Reg(V0), Op::Imm(31), Op::Reg(V0)});
      if (!And) return And.takeError();
      ISAEncoder::append(TB, *And);
    } else {
      auto Shr = InstructionBuilder::buildVLshrrevB32(D, V0, 7, V0);
      if (!Shr) return Shr.takeError();
      auto ShrB = Enc->emitInst(*Shr);
      if (!ShrB) return ShrB.takeError();
      ISAEncoder::append(TB, *ShrB);
    }

    // EXEC dance: save exec to s2:s3
    {
      auto M0 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(S2), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
      if (!M0) return M0.takeError();
      ISAEncoder::append(TB, *M0);

      auto M1 = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(S3), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
      if (!M1) return M1.takeError();
      ISAEncoder::append(TB, *M1);
    }

    // Counting payload blob
    {
      const auto &Blob = IsLDS ? LDSCountingBytes : CountingBytes;
      TB.insert(TB.end(), Blob.begin(), Blob.end());
    }

    // Save counting result
    {
      constexpr unsigned S0 = RegisterHelper::getSGPR(0);
      auto MC = Enc->buildAndEmit("S_MOV_B32",
          {Op::Reg(TempSGPR), Op::Reg(S0)});
      if (!MC) return MC.takeError();
      ISAEncoder::append(TB, *MC);
    }
  }

  // ---- Read back EXEC ----
  {
    auto RE0 = Enc->buildAndEmit("V_READLANE_B32",
        {Op::Reg(Scratch.ExecSaveSGPRLo),
         Op::Reg(Scratch.ScratchVGPR),
         Op::Imm(5)});
    if (!RE0) return RE0.takeError();
    ISAEncoder::append(TB, *RE0);

    auto RE1 = Enc->buildAndEmit("V_READLANE_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi),
         Op::Reg(Scratch.ScratchVGPR),
         Op::Imm(6)});
    if (!RE1) return RE1.takeError();
    ISAEncoder::append(TB, *RE1);
  }

  // ---- Atomic update: EXEC = lane 0 only ----
  {
    auto X0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Imm(1)});
    if (!X0) return X0.takeError();
    ISAEncoder::append(TB, *X0);

    auto X1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Imm(0)});
    if (!X1) return X1.takeError();
    ISAEncoder::append(TB, *X1);
  }

  // ---- Buffer address computation (dynamic from SiteIdx in lane 18) ----
  if (Trace.SupportsGPUAtomics) {
    constexpr unsigned S2 = RegisterHelper::getSGPR(2);
    constexpr unsigned S3 = RegisterHelper::getSGPR(3);

    auto Wc = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 19);
    if (!Wc) return Wc.takeError();
    ISAEncoder::append(TB, *Wc);

    auto Sv0 = Enc->encodeReadLane(TempSGPR, Scratch.TempVGPR, 0);
    if (!Sv0) return Sv0.takeError();
    ISAEncoder::append(TB, *Sv0);
    auto Wv0 = Enc->encodeWriteLane(Scratch.ScratchVGPR, TempSGPR, 20);
    if (!Wv0) return Wv0.takeError();
    ISAEncoder::append(TB, *Wv0);

    auto Rc = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 19);
    if (!Rc) return Rc.takeError();
    ISAEncoder::append(TB, *Rc);
    auto Vc = InstructionBuilder::buildVMovB32(D, Scratch.TempVGPR, TempSGPR);
    if (!Vc) return Vc.takeError();
    auto VcB = Enc->emitInst(*Vc);
    if (!VcB) return VcB.takeError();
    ISAEncoder::append(TB, *VcB);

    // Read SiteIdx from lane 18 and compute buffer address dynamically
    auto RdSite = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 18);
    if (!RdSite) return RdSite.takeError();
    ISAEncoder::append(TB, *RdSite);

    auto Shl = Enc->buildAndEmit("S_LSHL_B32",
        {Op::Reg(TempSGPR), Op::Reg(TempSGPR), Op::Imm(3)});
    if (!Shl) return Shl.takeError();
    ISAEncoder::append(TB, *Shl);

    auto BLo = InstructionBuilder::buildSMovB32(D, S2,
        static_cast<uint32_t>(Trace.BufferAddr & 0xFFFFFFFF));
    if (!BLo) return BLo.takeError();
    auto BLoB = Enc->emitInst(*BLo);
    if (!BLoB) return BLoB.takeError();
    ISAEncoder::append(TB, *BLoB);

    auto BHi = InstructionBuilder::buildSMovB32(D, S3,
        static_cast<uint32_t>(Trace.BufferAddr >> 32));
    if (!BHi) return BHi.takeError();
    auto BHiB = Enc->emitInst(*BHi);
    if (!BHiB) return BHiB.takeError();
    ISAEncoder::append(TB, *BHiB);

    auto Add = InstructionBuilder::buildSAddU32Reg(D, S2, S2, TempSGPR);
    if (!Add) return Add.takeError();
    auto AddB = Enc->emitInst(*Add);
    if (!AddB) return AddB.takeError();
    ISAEncoder::append(TB, *AddB);

    auto Adc = InstructionBuilder::buildSAddcU32(D, S3, S3, 0);
    if (!Adc) return Adc.takeError();
    auto AdcB = Enc->emitInst(*Adc);
    if (!AdcB) return AdcB.takeError();
    ISAEncoder::append(TB, *AdcB);

    auto Vz = InstructionBuilder::buildVMovB32Imm(D, Scratch.LaneVGPR, 0);
    if (!Vz) return Vz.takeError();
    auto VzB = Enc->emitInst(*Vz);
    if (!VzB) return VzB.takeError();
    ISAEncoder::append(TB, *VzB);

    auto A1 = InstructionBuilder::buildGlobalAtomicAddNoRtn(D,
        Scratch.LaneVGPR, Scratch.TempVGPR, S2, 0);
    if (!A1) return A1.takeError();
    auto A1B = Enc->emitInst(*A1);
    if (!A1B) return A1B.takeError();
    ISAEncoder::append(TB, *A1B);

    auto V1i = InstructionBuilder::buildVMovB32Imm(D, Scratch.TempVGPR, 1);
    if (!V1i) return V1i.takeError();
    auto V1B = Enc->emitInst(*V1i);
    if (!V1B) return V1B.takeError();
    ISAEncoder::append(TB, *V1B);

    auto A2 = InstructionBuilder::buildGlobalAtomicAddNoRtn(D,
        Scratch.LaneVGPR, Scratch.TempVGPR, S2, 4);
    if (!A2) return A2.takeError();
    auto A2B = Enc->emitInst(*A2);
    if (!A2B) return A2B.takeError();
    ISAEncoder::append(TB, *A2B);

    auto Rv0L = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 20);
    if (!Rv0L) return Rv0L.takeError();
    ISAEncoder::append(TB, *Rv0L);
    auto Wv0L = Enc->encodeWriteLane(Scratch.TempVGPR, TempSGPR, 0);
    if (!Wv0L) return Wv0L.takeError();
    ISAEncoder::append(TB, *Wv0L);
  } else {
    constexpr unsigned S2 = RegisterHelper::getSGPR(2);
    constexpr unsigned S3 = RegisterHelper::getSGPR(3);
    constexpr unsigned S4 = RegisterHelper::getSGPR(4);

    auto RdSite = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 18);
    if (!RdSite) return RdSite.takeError();
    ISAEncoder::append(TB, *RdSite);

    auto Shl = Enc->buildAndEmit("S_LSHL_B32",
        {Op::Reg(TempSGPR), Op::Reg(TempSGPR), Op::Imm(3)});
    if (!Shl) return Shl.takeError();
    ISAEncoder::append(TB, *Shl);

    auto BLo = InstructionBuilder::buildSMovB32(D, S2,
        static_cast<uint32_t>(Trace.BufferAddr & 0xFFFFFFFF));
    if (!BLo) return BLo.takeError();
    auto BLoB = Enc->emitInst(*BLo);
    if (!BLoB) return BLoB.takeError();
    ISAEncoder::append(TB, *BLoB);

    auto BHi = InstructionBuilder::buildSMovB32(D, S3,
        static_cast<uint32_t>(Trace.BufferAddr >> 32));
    if (!BHi) return BHi.takeError();
    auto BHiB = Enc->emitInst(*BHi);
    if (!BHiB) return BHiB.takeError();
    ISAEncoder::append(TB, *BHiB);

    auto Add = InstructionBuilder::buildSAddU32Reg(D, S2, S2, TempSGPR);
    if (!Add) return Add.takeError();
    auto AddB = Enc->emitInst(*Add);
    if (!AddB) return AddB.takeError();
    ISAEncoder::append(TB, *AddB);

    auto Adc = InstructionBuilder::buildSAddcU32(D, S3, S3, 0);
    if (!Adc) return Adc.takeError();
    auto AdcB = Enc->emitInst(*Adc);
    if (!AdcB) return AdcB.takeError();
    ISAEncoder::append(TB, *AdcB);

    auto MC = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(S4), Op::Reg(TempSGPR)});
    if (!MC) return MC.takeError();
    ISAEncoder::append(TB, *MC);

    TB.insert(TB.end(), AtomicBytesNonAtomic.begin(),
              AtomicBytesNonAtomic.end());
  }

  // ---- Restore s0-s7 ----
  for (unsigned I = 0; I < 8; ++I) {
    unsigned SGPR = RegisterHelper::getSGPR(I);
    auto R = Enc->encodeReadLane(SGPR, Scratch.ScratchVGPR, 10 + I);
    if (!R) return R.takeError();
    ISAEncoder::append(TB, *R);
  }

  // ---- Restore EXEC ----
  {
    auto R0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(Scratch.ExecSaveSGPRLo)});
    if (!R0) return R0.takeError();
    ISAEncoder::append(TB, *R0);

    auto R1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(Scratch.ExecSaveSGPRHi)});
    if (!R1) return R1.takeError();
    ISAEncoder::append(TB, *R1);
  }

  // ---- Restore v0 ----
  {
    constexpr unsigned V0 = RegisterHelper::getVGPR(0);
    auto Rv = InstructionBuilder::buildVMovB32(D, V0, Scratch.TempVGPR);
    if (!Rv) return Rv.takeError();
    auto RvB = Enc->emitInst(*Rv);
    if (!RvB) return RvB.takeError();
    ISAEncoder::append(TB, *RvB);
  }

  // ---- Restore v1-v3 lane 0 ----
  {
    constexpr unsigned V1 = RegisterHelper::getVGPR(1);
    constexpr unsigned V2 = RegisterHelper::getVGPR(2);
    constexpr unsigned V3 = RegisterHelper::getVGPR(3);

    auto R1 = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 7);
    if (!R1) return R1.takeError();
    ISAEncoder::append(TB, *R1);
    auto W1 = Enc->encodeWriteLane(V1, TempSGPR, 0);
    if (!W1) return W1.takeError();
    ISAEncoder::append(TB, *W1);

    auto R2 = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 8);
    if (!R2) return R2.takeError();
    ISAEncoder::append(TB, *R2);
    auto W2 = Enc->encodeWriteLane(V2, TempSGPR, 0);
    if (!W2) return W2.takeError();
    ISAEncoder::append(TB, *W2);

    auto R3 = Enc->encodeReadLane(TempSGPR, Scratch.ScratchVGPR, 9);
    if (!R3) return R3.takeError();
    ISAEncoder::append(TB, *R3);
    auto W3 = Enc->encodeWriteLane(V3, TempSGPR, 0);
    if (!W3) return W3.takeError();
    ISAEncoder::append(TB, *W3);
  }

  // ---- Restore VCC ----
  {
    auto R0 = Enc->encodeReadLane(InstructionBuilder::VCC_LO_REG, Scratch.ScratchVGPR, 3);
    if (!R0) return R0.takeError();
    ISAEncoder::append(TB, *R0);

    auto R1 = Enc->encodeReadLane(InstructionBuilder::VCC_HI_REG, Scratch.ScratchVGPR, 4);
    if (!R1) return R1.takeError();
    ISAEncoder::append(TB, *R1);

    auto NP = Enc->buildAndEmit("S_NOP", {Op::Imm(4)});
    if (!NP) return NP.takeError();
    ISAEncoder::append(TB, *NP);
  }

  // ---- Pre-return: read SiteIdx and SCC before scratch restore ----
  // ScratchSGPR survives scratch restore (SAddrTempSGPR is used as the
  // SADDR temp for large-offset scratch ops). ExecSaveSGPR pair does NOT.
  {
    // Read SiteIdx from lane 18 into ScratchSGPR
    auto RdSite = Enc->encodeReadLane(Scratch.ScratchSGPR, Scratch.ScratchVGPR, 18);
    if (!RdSite) return RdSite.takeError();
    ISAEncoder::append(TB, *RdSite);

    // SiteIdx * 24 (return entry stride: 12-byte preamble + 8 displaced + 4 setpc)
    auto Mul = Enc->buildAndEmit("S_MUL_I32",
        {Op::Reg(Scratch.ScratchSGPR), Op::Reg(Scratch.ScratchSGPR), Op::Imm(24)});
    if (!Mul) return Mul.takeError();
    ISAEncoder::append(TB, *Mul);

    // Read SCC snapshot from lane 2 into ExecSaveLo (temporary)
    auto RdSCC = Enc->encodeReadLane(Scratch.ExecSaveSGPRLo, Scratch.ScratchVGPR, 2);
    if (!RdSCC) return RdSCC.takeError();
    ISAEncoder::append(TB, *RdSCC);

    // Read kernel return addr from lanes 0-1
    auto Rd0 = Enc->encodeReadLane(Scratch.ReturnAddrSGPR, Scratch.ScratchVGPR, 0);
    if (!Rd0) return Rd0.takeError();
    ISAEncoder::append(TB, *Rd0);

    auto Rd1 = Enc->encodeReadLane(Scratch.ReturnAddrSGPRHi, Scratch.ScratchVGPR, 1);
    if (!Rd1) return Rd1.takeError();
    ISAEncoder::append(TB, *Rd1);

    // Encode SCC snapshot into bit 0 of ReturnAddr.
    // s_setpc_b64 ignores bottom 2 bits (code is 4-byte aligned).
    // The return entry will extract and restore SCC from this bit.
    auto OrSCC = Enc->buildAndEmit("S_OR_B32",
        {Op::Reg(Scratch.ReturnAddrSGPR), Op::Reg(Scratch.ReturnAddrSGPR),
         Op::Reg(Scratch.ExecSaveSGPRLo)});
    if (!OrSCC) return OrSCC.takeError();
    ISAEncoder::append(TB, *OrSCC);

    // Pre-add getpc-to-return-table offset into ScratchSGPR.
    // ScratchSGPR = SiteIdx*16 + GetpcToReturnTableOffset.
    // This instruction may be 4 or 8 bytes depending on the offset value,
    // but that's fine: the jump block uses only register ops, so the
    // getpc-relative distance is always 12 (= 3 instructions after getpc).
    auto AddBase = InstructionBuilder::buildSAddU32(D,
        Scratch.ScratchSGPR, Scratch.ScratchSGPR,
        static_cast<uint32_t>(GetpcToReturnTableOffset));
    if (!AddBase) return AddBase.takeError();
    auto AddBaseB = Enc->emitInst(*AddBase);
    if (!AddBaseB) return AddBaseB.takeError();
    ISAEncoder::append(TB, *AddBaseB);
  }

  // ---- Scratch memory restore ----
  if (Scratch.NeedsScratchSpill) {
    auto SE0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(InstructionBuilder::EXEC_LO_REG)});
    if (!SE0) return SE0.takeError();
    ISAEncoder::append(TB, *SE0);

    auto SE1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(InstructionBuilder::EXEC_HI_REG)});
    if (!SE1) return SE1.takeError();
    ISAEncoder::append(TB, *SE1);

    auto XA0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Imm(-1)});
    if (!XA0) return XA0.takeError();
    ISAEncoder::append(TB, *XA0);

    auto XA1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Imm(-1)});
    if (!XA1) return XA1.takeError();
    ISAEncoder::append(TB, *XA1);

    if (auto E = appendScratchLoad(TB, Scratch.ScratchVGPR,
            Scratch.ScratchSpillOffset + 0, Scratch.SAddrTempSGPR))
      return std::move(E);

    if (auto E = appendScratchLoad(TB, Scratch.LaneVGPR,
            Scratch.ScratchSpillOffset + 4, Scratch.SAddrTempSGPR))
      return std::move(E);

    if (auto E = appendScratchLoad(TB, Scratch.TempVGPR,
            Scratch.ScratchSpillOffset + 8, Scratch.SAddrTempSGPR))
      return std::move(E);

    {
      auto WaitLoad = InstructionBuilder::buildSWaitCnt(
          D, /*VmCnt=*/0, /*ExpCnt=*/7, /*LgkmCnt=*/15);
      if (!WaitLoad) return WaitLoad.takeError();
      auto WaitLoadB = Enc->emitInst(*WaitLoad);
      if (!WaitLoadB) return WaitLoadB.takeError();
      ISAEncoder::append(TB, *WaitLoadB);
    }

    auto RE0 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_LO_REG), Op::Reg(Scratch.ExecSaveSGPRLo)});
    if (!RE0) return RE0.takeError();
    ISAEncoder::append(TB, *RE0);

    auto RE1 = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(InstructionBuilder::EXEC_HI_REG), Op::Reg(Scratch.ExecSaveSGPRHi)});
    if (!RE1) return RE1.takeError();
    ISAEncoder::append(TB, *RE1);
  }

  // ---- Drain trampoline VMEM traffic ----
  {
    unsigned DrainVm = 0;
    if (Trace.SupportsGPUAtomics && Scratch.NeedsAccVGPRSpill)
      DrainVm = maxPreSpillVmWait;
    auto WaitDrain = InstructionBuilder::buildSWaitCnt(
        D, DrainVm, /*ExpCnt=*/7, /*LgkmCnt=*/15);
    if (!WaitDrain) return WaitDrain.takeError();
    auto WaitDrainB = Enc->emitInst(*WaitDrain);
    if (!WaitDrainB) return WaitDrainB.takeError();
    ISAEncoder::append(TB, *WaitDrainB);
  }

  // ---- Computed jump to per-site return entry (PC-relative) ----
  // ScratchSGPR (s14) = SiteIdx*16 + GetpcToReturnTableOffset (set above).
  // Use s[ScratchSGPR : ExecSaveSGPRLo] as the getpc pair — ScratchSGPR is
  // at ScratchBase+2 (even-aligned), which is required for 64-bit SGPR ops.
  // Save the offset to ExecSaveSGPRHi first since getpc clobbers both regs.
  // Total: 5 instructions, 20 bytes. JumpBlockTailBytes remains 12 (from
  // getpc+4 to end).
  {
    unsigned ScratchIdx = Scratch.ScratchSGPR - RegisterHelper::SGPR_BASE;
    auto PairOrErr = Enc->resolveSGPRPair(ScratchIdx);
    if (!PairOrErr) return PairOrErr.takeError();
    unsigned JumpPair = *PairOrErr;

    auto MovOff = Enc->buildAndEmit("S_MOV_B32",
        {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(Scratch.ScratchSGPR)});
    if (!MovOff) return MovOff.takeError();
    ISAEncoder::append(TB, *MovOff);

    auto GetPC = Enc->buildAndEmit("S_GETPC_B64",
        {Op::Reg(JumpPair)});
    if (!GetPC) return GetPC.takeError();
    ISAEncoder::append(TB, *GetPC);

    auto AddSite = InstructionBuilder::buildSAddU32Reg(D,
        Scratch.ScratchSGPR, Scratch.ScratchSGPR, Scratch.ExecSaveSGPRHi);
    if (!AddSite) return AddSite.takeError();
    auto AddSiteB = Enc->emitInst(*AddSite);
    if (!AddSiteB) return AddSiteB.takeError();
    ISAEncoder::append(TB, *AddSiteB);

    auto AdcSite = InstructionBuilder::buildSAddcU32(D,
        Scratch.ExecSaveSGPRLo, Scratch.ExecSaveSGPRLo, 0);
    if (!AdcSite) return AdcSite.takeError();
    auto AdcSiteB = Enc->emitInst(*AdcSite);
    if (!AdcSiteB) return AdcSiteB.takeError();
    ISAEncoder::append(TB, *AdcSiteB);

    auto Ret = Enc->encodeSetPC(JumpPair);
    if (!Ret) return Ret.takeError();
    ISAEncoder::append(TB, *Ret);
  }

  return TB;
}

//===----------------------------------------------------------------------===//
// Shared Body Trampoline: emitDispatchEntry
//===----------------------------------------------------------------------===//

Expected<std::vector<uint8_t>>
TrampolineEmitter::emitDispatchEntry(const ScratchRegisters &Scratch,
                                      uint32_t SiteIdx, uint32_t AddrVGPRIdx,
                                      bool IsGlobal,
                                      int16_t BranchDwordOffset) {
  using Op = InstructionBuilder::Operand;
  std::vector<uint8_t> Entry;

  // Bit 25 is always set to force literal-constant encoding in s_mov_b32,
  // guaranteeing every dispatch entry is exactly 12 bytes (8 + 4).
  // The shared body's unpacking (bits 0-15 for SiteIdx, 16-23 for AddrVGPRIdx)
  // is unaffected by bit 25.
  uint32_t PackedInfo = SiteIdx
                      | (AddrVGPRIdx << 16)
                      | ((IsGlobal ? 0u : 1u) << 24)
                      | (1u << 25);

  auto Mov = Enc->buildAndEmit("S_MOV_B32",
      {Op::Reg(Scratch.ScratchSGPR), Op::Imm(PackedInfo)});
  if (!Mov) return Mov.takeError();
  ISAEncoder::append(Entry, *Mov);

  auto Br = Enc->encodeSBranch(BranchDwordOffset);
  if (!Br) return Br.takeError();
  ISAEncoder::append(Entry, *Br);

  return Entry;
}

//===----------------------------------------------------------------------===//
// Shared Body Trampoline: emitReturnEntry
//===----------------------------------------------------------------------===//

Expected<std::vector<uint8_t>>
TrampolineEmitter::emitReturnEntry(ArrayRef<uint8_t> DisplacedBytes,
                                    unsigned RetAddrSGPRPair,
                                    unsigned ScratchSGPR,
                                    unsigned ReturnAddrSGPR) {
  using Op = InstructionBuilder::Operand;
  std::vector<uint8_t> Entry;

  // Extract SCC from bit 0, clear the tag, then restore SCC.
  // s_and_b32   extracts bit 0 into ScratchSGPR (sets SCC = extracted value)
  // s_andn2_b32 clears bit 0 in ReturnAddrSGPR (clobbers SCC)
  // s_cmp_lg_u32 restores SCC from the extracted value
  auto AndSCC = Enc->buildAndEmit("S_AND_B32",
      {Op::Reg(ScratchSGPR), Op::Reg(ReturnAddrSGPR), Op::Imm(1)});
  if (!AndSCC) return AndSCC.takeError();
  ISAEncoder::append(Entry, *AndSCC);

  auto ClearBit = Enc->buildAndEmit("S_ANDN2_B32",
      {Op::Reg(ReturnAddrSGPR), Op::Reg(ReturnAddrSGPR), Op::Imm(1)});
  if (!ClearBit) return ClearBit.takeError();
  ISAEncoder::append(Entry, *ClearBit);

  auto RestoreSCC = Enc->buildAndEmit("S_CMP_LG_U32",
      {Op::Reg(ScratchSGPR), Op::Imm(0)});
  if (!RestoreSCC) return RestoreSCC.takeError();
  ISAEncoder::append(Entry, *RestoreSCC);

  Entry.insert(Entry.end(), DisplacedBytes.begin(), DisplacedBytes.end());

  // Pad to 20 bytes before setpc so total is always 24 (uniform stride).
  // 12-byte preamble + 4-byte displaced = 16 → need 4-byte nop.
  // 12-byte preamble + 8-byte displaced = 20 → no padding needed.
  if (Entry.size() < 20) {
    auto Nop = Enc->encodeNop();
    if (!Nop) return Nop.takeError();
    ISAEncoder::append(Entry, *Nop);
  }

  auto Ret = Enc->encodeSetPC(RetAddrSGPRPair);
  if (!Ret) return Ret.takeError();
  ISAEncoder::append(Entry, *Ret);

  return Entry;
}

Expected<std::vector<uint8_t>>
TrampolineEmitter::emitSwapPCPreamble(const ScratchRegisters &Scratch,
                                      int32_t GetpcToDispatchTable) {
  using Op = InstructionBuilder::Operand;
  std::vector<uint8_t> TB;

  auto Shl8 = Enc->buildAndEmit(
      "S_LSHL_B32",
      {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(Scratch.ScratchSGPR), Op::Imm(3)});
  if (!Shl8)
    return Shl8.takeError();
  ISAEncoder::append(TB, *Shl8);

  auto Shl4 = Enc->buildAndEmit(
      "S_LSHL_B32",
      {Op::Reg(Scratch.ScratchSGPR), Op::Reg(Scratch.ScratchSGPR), Op::Imm(2)});
  if (!Shl4)
    return Shl4.takeError();
  ISAEncoder::append(TB, *Shl4);

  auto Mul12 = Enc->buildAndEmit(
      "S_ADD_U32",
      {Op::Reg(Scratch.ExecSaveSGPRHi), Op::Reg(Scratch.ScratchSGPR),
       Op::Reg(Scratch.ExecSaveSGPRLo)});
  if (!Mul12)
    return Mul12.takeError();
  ISAEncoder::append(TB, *Mul12);

  unsigned ScratchIdx = Scratch.ScratchSGPR - RegisterHelper::SGPR_BASE;
  auto PairOrErr = Enc->resolveSGPRPair(ScratchIdx);
  if (!PairOrErr)
    return PairOrErr.takeError();
  unsigned JumpPair = *PairOrErr;

  auto GetPC = Enc->buildAndEmit("S_GETPC_B64", {Op::Reg(JumpPair)});
  if (!GetPC)
    return GetPC.takeError();
  ISAEncoder::append(TB, *GetPC);

  int64_t Addend64 = static_cast<int64_t>(GetpcToDispatchTable);
  int32_t Lo = static_cast<int32_t>(Addend64 & 0xFFFFFFFF);
  int32_t Hi = static_cast<int32_t>((Addend64 >> 32) & 0xFFFFFFFF);

  auto AddLo = Enc->buildAndEmit(
      "S_ADD_U32",
      {Op::Reg(Scratch.ScratchSGPR), Op::Reg(Scratch.ScratchSGPR), Op::Imm(Lo)});
  if (!AddLo)
    return AddLo.takeError();
  ISAEncoder::append(TB, *AddLo);

  auto AddHi = Enc->buildAndEmit(
      "S_ADDC_U32",
      {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(Scratch.ExecSaveSGPRLo),
       Op::Imm(Hi)});
  if (!AddHi)
    return AddHi.takeError();
  ISAEncoder::append(TB, *AddHi);

  auto AddOff = Enc->buildAndEmit(
      "S_ADD_U32",
      {Op::Reg(Scratch.ScratchSGPR), Op::Reg(Scratch.ScratchSGPR),
       Op::Reg(Scratch.ExecSaveSGPRHi)});
  if (!AddOff)
    return AddOff.takeError();
  ISAEncoder::append(TB, *AddOff);

  auto AddcOff = Enc->buildAndEmit(
      "S_ADDC_U32",
      {Op::Reg(Scratch.ExecSaveSGPRLo), Op::Reg(Scratch.ExecSaveSGPRLo), Op::Imm(0)});
  if (!AddcOff)
    return AddcOff.takeError();
  ISAEncoder::append(TB, *AddcOff);

  auto Ret = Enc->encodeSetPC(JumpPair);
  if (!Ret)
    return Ret.takeError();
  ISAEncoder::append(TB, *Ret);

  return TB;
}

} // namespace aegisbit
