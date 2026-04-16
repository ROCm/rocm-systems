//===-- SwapPCSharedBodyStrategy.cpp - s_swappc_b64 shared body -*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared-body strategy using s_swappc_b64 for dispatch. Required when the
/// kernel is large enough that an s_call_b64 can't reach the island
/// in-range.
///
//===----------------------------------------------------------------------===//

#include "TrampolineStrategy.h"
#include "BridgeHelpers.h"

#include "aegisbit/ISAEncoder.h"
#include "aegisbit/InstructionBuilder.h"
#include "aegisbit/RegisterHelper.h"
#include "aegisbit/TrampolineEmitter.h"

using namespace llvm;

namespace aegisbit {

namespace {

class SwapPCSharedBodyStrategy final : public TrampolineStrategy {
public:
  Expected<BridgeResult> build(const BridgeInputs &In) override {
    BridgeResult Result;
    const auto &Sites = *In.Sites;

    uint64_t IslandStart =
        alignIslandStart(In.TextSectionSize, *In.Occupied);

    unsigned SwapTargetIdx =
        In.Scratch->SwapTargetSGPR - RegisterHelper::SGPR_BASE;
    auto SwapPairOrErr = In.Enc->resolveSGPRPair(SwapTargetIdx);
    if (!SwapPairOrErr)
      return SwapPairOrErr.takeError();
    unsigned SwapTargetPair = *SwapPairOrErr;

    SiteSummary Summary = summarizeSites(Sites);

    auto PreambleOrErr = In.Emitter->emitSwapPCPreamble(*In.Scratch, 0);
    if (!PreambleOrErr)
      return PreambleOrErr.takeError();
    uint64_t PreambleSize = PreambleOrErr->size();

    auto BodiesOrErr = emitSharedBodies(*In.Emitter, *In.Scratch, *In.Trace,
                                         In.RetAddrSGPRPair, Summary);
    if (!BodiesOrErr)
      return BodiesOrErr.takeError();
    auto &Bodies = *BodiesOrErr;

    uint64_t DispatchTableSize = Sites.size() * 12;
    uint64_t VMEMBodyAbs = IslandStart + PreambleSize + DispatchTableSize;
    uint64_t LDSBodyAbs = VMEMBodyAbs + Bodies.VMEM.size();

    auto DispatchTableOrErr =
        buildDispatchTable(*In.Emitter, *In.Scratch, Sites, IslandStart,
                           /*DispatchTableOffsetInIsland=*/PreambleSize,
                           VMEMBodyAbs, LDSBodyAbs);
    if (!DispatchTableOrErr)
      return DispatchTableOrErr.takeError();
    auto &DispatchTable = *DispatchTableOrErr;

    auto ReturnTableOrErr =
        buildReturnTable(*In.Emitter, *In.Scratch, Sites, In.Code,
                         In.RetAddrSGPRPair);
    if (!ReturnTableOrErr)
      return ReturnTableOrErr.takeError();
    auto &ReturnTable = *ReturnTableOrErr;

    int32_t GetpcToDispatch = static_cast<int32_t>(PreambleSize) - 16;
    auto PreambleFixed = In.Emitter->emitSwapPCPreamble(*In.Scratch,
                                                         GetpcToDispatch);
    if (!PreambleFixed)
      return PreambleFixed.takeError();
    if (PreambleFixed->size() != PreambleSize) {
      PreambleSize = PreambleFixed->size();
      GetpcToDispatch = static_cast<int32_t>(PreambleSize) - 16;
      PreambleFixed = In.Emitter->emitSwapPCPreamble(*In.Scratch,
                                                      GetpcToDispatch);
      if (!PreambleFixed)
        return PreambleFixed.takeError();
    }

    TrampolineIsland Isl;
    Isl.Offset = IslandStart;
    ISAEncoder::append(Isl.Bytes, *PreambleFixed);
    ISAEncoder::append(Isl.Bytes, DispatchTable);
    ISAEncoder::append(Isl.Bytes, Bodies.VMEM);
    ISAEncoder::append(Isl.Bytes, Bodies.LDS);
    ISAEncoder::append(Isl.Bytes, ReturnTable);

    uint64_t SwapTargetAbs = IslandStart;
    using Op = InstructionBuilder::Operand;
    auto GetPC = In.Enc->buildAndEmit("S_GETPC_B64",
                                       {Op::Reg(SwapTargetPair)});
    if (!GetPC)
      return GetPC.takeError();
    uint64_t GetPCSize = GetPC->size();

    uint64_t IterPrologueSize = GetPCSize + 8 + 8;
    for (int Iter = 0; Iter < 3; ++Iter) {
      int64_t GetpcResult = static_cast<int64_t>(In.BaseAddr) -
                            static_cast<int64_t>(IterPrologueSize) + 4;
      int64_t Offset = static_cast<int64_t>(SwapTargetAbs) - GetpcResult;
      int32_t OffLo = static_cast<int32_t>(Offset & 0xFFFFFFFF);
      int32_t OffHi = static_cast<int32_t>((Offset >> 32) & 0xFFFFFFFF);

      auto Add = In.Enc->buildAndEmit(
          "S_ADD_U32",
          {Op::Reg(In.Scratch->SwapTargetSGPR),
           Op::Reg(In.Scratch->SwapTargetSGPR), Op::Imm(OffLo)});
      if (!Add)
        return Add.takeError();
      auto Adc = In.Enc->buildAndEmit(
          "S_ADDC_U32",
          {Op::Reg(In.Scratch->SwapTargetSGPRHi),
           Op::Reg(In.Scratch->SwapTargetSGPRHi), Op::Imm(OffHi)});
      if (!Adc)
        return Adc.takeError();

      uint64_t NewSize = GetPCSize + Add->size() + Adc->size();
      if (NewSize == IterPrologueSize) {
        ISAEncoder::append(Result.PrologueBytes, *GetPC);
        ISAEncoder::append(Result.PrologueBytes, *Add);
        ISAEncoder::append(Result.PrologueBytes, *Adc);
        break;
      }
      IterPrologueSize = NewSize;
    }

    if (Result.PrologueBytes.empty())
      return createStringError(inconvertibleErrorCode(),
                               "SwapPC prologue size did not converge");

    for (uint32_t SiteIdx = 0; SiteIdx < Sites.size(); ++SiteIdx) {
      const auto &Site = Sites[SiteIdx];

      TrampolineSlot Slot;
      Slot.OriginalPC = Site.Address;
      Slot.DisplacedSize = Site.OrigInstSize;
      Slot.TrampolineOffset = PreambleSize + SiteIdx * 12;

      auto MovK = In.Enc->encodeMovK(In.Scratch->ScratchSGPR,
                                      static_cast<uint16_t>(SiteIdx));
      if (!MovK)
        return MovK.takeError();
      auto Swap = In.Enc->encodeSwapPC(In.RetAddrSGPRPair, SwapTargetPair);
      if (!Swap)
        return Swap.takeError();

      Slot.PatchBytes = std::move(*MovK);
      ISAEncoder::append(Slot.PatchBytes, *Swap);
      Result.Slots.push_back(std::move(Slot));
      Result.PatchedCount++;
    }

    if (!Isl.Bytes.empty())
      Result.Islands.push_back(std::move(Isl));

    return Result;
  }
};

} // namespace

std::unique_ptr<TrampolineStrategy> createSwapPCSharedBodyStrategy() {
  return std::make_unique<SwapPCSharedBodyStrategy>();
}

} // namespace aegisbit
