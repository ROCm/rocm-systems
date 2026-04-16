//===-- BridgeHelpers.cpp - Shared TrampolineBridge helpers ------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "BridgeHelpers.h"

#include "aegisbit/ISAEncoder.h"
#include "aegisbit/TrampolineEmitter.h"

using namespace llvm;

namespace aegisbit {

namespace {
constexpr int32_t JumpBlockTailBytes = 12;
} // namespace

uint64_t alignIslandStart(uint64_t TextSectionSize,
                          const TrampolineBridge::OccupiedRanges &Occupied) {
  uint64_t IslandStart = (TextSectionSize + 255) & ~255ULL;
  for (const auto &R : Occupied) {
    if (IslandStart >= R.first && IslandStart < R.second)
      IslandStart = (R.second + 255) & ~255ULL;
  }
  return IslandStart;
}

SiteSummary summarizeSites(const std::vector<InstrumentationSite> &Sites) {
  SiteSummary Summary;
  for (const auto &Site : Sites) {
    if (Site.PreSpillVmWait > Summary.MaxPreSpillVmWait)
      Summary.MaxPreSpillVmWait = Site.PreSpillVmWait;
    if (Site.IsGlobal)
      Summary.HasVMEM = true;
    else
      Summary.HasLDS = true;
  }
  return Summary;
}

Expected<SharedBodies>
emitSharedBodies(TrampolineEmitter &Emitter, const ScratchRegisters &Scratch,
                 const TraceConfig &Trace, unsigned RetAddrSGPRPair,
                 const SiteSummary &Summary) {
  SharedBodies Out;

  if (Summary.HasLDS) {
    int32_t OffsetLDS = JumpBlockTailBytes;
    auto BodyOrErr = Emitter.emitSharedBody(Scratch, Trace, /*IsLDS=*/true,
                                             RetAddrSGPRPair,
                                             Summary.MaxPreSpillVmWait,
                                             OffsetLDS);
    if (!BodyOrErr)
      return BodyOrErr.takeError();
    Out.LDS = std::move(*BodyOrErr);
  }

  if (Summary.HasVMEM) {
    int32_t OffsetVMEM = JumpBlockTailBytes + static_cast<int32_t>(Out.LDS.size());
    auto BodyOrErr = Emitter.emitSharedBody(Scratch, Trace, /*IsLDS=*/false,
                                             RetAddrSGPRPair,
                                             Summary.MaxPreSpillVmWait,
                                             OffsetVMEM);
    if (!BodyOrErr)
      return BodyOrErr.takeError();
    Out.VMEM = std::move(*BodyOrErr);
  }

  return Out;
}

Expected<std::vector<uint8_t>> buildDispatchTable(
    TrampolineEmitter &Emitter, const ScratchRegisters &Scratch,
    const std::vector<InstrumentationSite> &Sites, uint64_t IslandStart,
    uint64_t DispatchTableOffsetInIsland, uint64_t VMEMBodyAbs,
    uint64_t LDSBodyAbs) {
  std::vector<uint8_t> DispatchTable;
  for (uint32_t SiteIdx = 0; SiteIdx < Sites.size(); ++SiteIdx) {
    const auto &Site = Sites[SiteIdx];

    uint64_t DispatchEntryAbs =
        IslandStart + DispatchTableOffsetInIsland + SiteIdx * 12;
    uint64_t TargetBodyAbs = Site.IsGlobal ? VMEMBodyAbs : LDSBodyAbs;

    uint64_t BranchPC = DispatchEntryAbs + 8;
    int64_t BranchByteOffset = static_cast<int64_t>(TargetBodyAbs) -
                               static_cast<int64_t>(BranchPC);
    int64_t BranchDword = (BranchByteOffset - 4) / 4;

    auto EntryOrErr = Emitter.emitDispatchEntry(
        Scratch, SiteIdx, Site.AddrVGPRIndex, Site.IsGlobal,
        static_cast<int16_t>(BranchDword));
    if (!EntryOrErr)
      return EntryOrErr.takeError();
    ISAEncoder::append(DispatchTable, *EntryOrErr);
  }
  return DispatchTable;
}

Expected<std::vector<uint8_t>> buildReturnTable(
    TrampolineEmitter &Emitter, const ScratchRegisters &Scratch,
    const std::vector<InstrumentationSite> &Sites,
    ArrayRef<uint8_t> Code, unsigned RetAddrSGPRPair) {
  std::vector<uint8_t> ReturnTable;
  for (uint32_t SiteIdx = 0; SiteIdx < Sites.size(); ++SiteIdx) {
    const auto &Site = Sites[SiteIdx];
    ArrayRef<uint8_t> DisplacedBytes(Code.data() + Site.Offset,
                                      Site.OrigInstSize);

    auto EntryOrErr = Emitter.emitReturnEntry(DisplacedBytes, RetAddrSGPRPair,
                                              Scratch.ScratchSGPR,
                                              Scratch.ReturnAddrSGPR);
    if (!EntryOrErr)
      return EntryOrErr.takeError();
    ISAEncoder::append(ReturnTable, *EntryOrErr);
  }
  return ReturnTable;
}

} // namespace aegisbit
