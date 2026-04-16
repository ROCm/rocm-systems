//===-- SharedBodyStrategy.cpp - s_call_b64 shared body ---------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared-body strategy using s_call_b64 for dispatch + s_setpc_b64 for
/// return. Suitable for kernels whose dispatch table entries fit within
/// ±128 KB of every patch site.
///
//===----------------------------------------------------------------------===//

#include "TrampolineStrategy.h"
#include "BridgeHelpers.h"

#include "aegisbit/ISAEncoder.h"
#include "aegisbit/TrampolineEmitter.h"

using namespace llvm;

namespace aegisbit {

namespace {

class SharedBodyStrategy final : public TrampolineStrategy {
public:
  Expected<BridgeResult> build(const BridgeInputs &In) override {
    BridgeResult Result;
    const auto &Sites = *In.Sites;

    uint64_t IslandStart =
        alignIslandStart(In.TextSectionSize, *In.Occupied);

    SiteSummary Summary = summarizeSites(Sites);
    auto BodiesOrErr = emitSharedBodies(*In.Emitter, *In.Scratch, *In.Trace,
                                         In.RetAddrSGPRPair, Summary);
    if (!BodiesOrErr)
      return BodiesOrErr.takeError();
    auto &Bodies = *BodiesOrErr;

    uint64_t DispatchTableSize = Sites.size() * 12;
    uint64_t VMEMBodyAbs = IslandStart + DispatchTableSize;
    uint64_t LDSBodyAbs = VMEMBodyAbs + Bodies.VMEM.size();

    auto DispatchTableOrErr =
        buildDispatchTable(*In.Emitter, *In.Scratch, Sites, IslandStart,
                           /*DispatchTableOffsetInIsland=*/0, VMEMBodyAbs,
                           LDSBodyAbs);
    if (!DispatchTableOrErr)
      return DispatchTableOrErr.takeError();
    auto &DispatchTable = *DispatchTableOrErr;

    auto ReturnTableOrErr =
        buildReturnTable(*In.Emitter, *In.Scratch, Sites, In.Code,
                         In.RetAddrSGPRPair);
    if (!ReturnTableOrErr)
      return ReturnTableOrErr.takeError();
    auto &ReturnTable = *ReturnTableOrErr;

    TrampolineIsland Isl;
    Isl.Offset = IslandStart;
    ISAEncoder::append(Isl.Bytes, DispatchTable);
    ISAEncoder::append(Isl.Bytes, Bodies.VMEM);
    ISAEncoder::append(Isl.Bytes, Bodies.LDS);
    ISAEncoder::append(Isl.Bytes, ReturnTable);

    for (uint32_t SiteIdx = 0; SiteIdx < Sites.size(); ++SiteIdx) {
      const auto &Site = Sites[SiteIdx];
      uint64_t PatchSiteAbs = In.BaseAddr + Site.Offset;
      uint64_t DispatchEntryAbs = IslandStart + SiteIdx * 12;

      int64_t ForwardByteOffset = static_cast<int64_t>(DispatchEntryAbs) -
                                  static_cast<int64_t>(PatchSiteAbs);
      int64_t BranchToDword = (ForwardByteOffset - 4) / 4;

      TrampolineSlot Slot;
      Slot.OriginalPC = Site.Address;
      Slot.DisplacedSize = Site.OrigInstSize;
      Slot.TrampolineOffset = SiteIdx * 12;

      if (BranchToDword < -32768 || BranchToDword > 32767) {
        return createStringError(
            inconvertibleErrorCode(),
            "Shared-body dispatch entry for site " + std::to_string(SiteIdx) +
                " out of s_call_b64 range (" + std::to_string(BranchToDword) +
                " dwords). Kernel too large for single-island layout.");
      }

      auto CallBytes = In.Enc->encodeSCall(In.RetAddrSGPRPair,
                                            static_cast<int16_t>(BranchToDword));
      if (!CallBytes)
        return CallBytes.takeError();
      Slot.PatchBytes = std::move(*CallBytes);
      if (Site.OrigInstSize > 4) {
        auto NopBytes = In.Enc->encodeNop();
        if (!NopBytes)
          return NopBytes.takeError();
        ISAEncoder::append(Slot.PatchBytes, *NopBytes);
      }

      ArrayRef<uint8_t> DispatchSlice(DispatchTable.data() + SiteIdx * 12, 12);
      Slot.TrampolineBytes.assign(DispatchSlice.begin(), DispatchSlice.end());

      Result.Slots.push_back(std::move(Slot));
      Result.PatchedCount++;
    }

    if (!Isl.Bytes.empty())
      Result.Islands.push_back(std::move(Isl));

    return Result;
  }
};

} // namespace

std::unique_ptr<TrampolineStrategy> createSharedBodyStrategy() {
  return std::make_unique<SharedBodyStrategy>();
}

} // namespace aegisbit
