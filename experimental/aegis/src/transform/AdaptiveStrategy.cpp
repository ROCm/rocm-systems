//===-- AdaptiveStrategy.cpp - Per-site jump-strategy picker ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The "adaptive" trampoline strategy picks per-site between a direct body
/// (s_call_b64 / long-jump) and a relay stub with an out-of-range body. The
/// main loop splits into two helpers — `emitRelayPath` and
/// `emitDirectPath` — with retry/island-rollover bookkeeping done at the
/// dispatch site.
///
//===----------------------------------------------------------------------===//

#include "TrampolineStrategy.h"

#include "aegisbit/ISAEncoder.h"
#include "aegisbit/IslandAllocator.h"
#include "aegisbit/JumpHeuristics.h"
#include "aegisbit/RelayEmitter.h"
#include "aegisbit/RuntimeConfig.h"
#include "aegisbit/TrampolineEmitter.h"

using namespace llvm;

namespace aegisbit {

namespace {

struct AdaptiveContext {
  const BridgeInputs *In;
  IslandAllocator *Alloc;
  RelayEmitter *Relay;
  uint64_t *StandardCursor;
  uint32_t *LastRetrySiteIdx;
  uint32_t *RetryCount;
  BridgeResult *Result;
};

enum class SiteAction {
  Committed,   ///< Site produced a slot.
  Retry,       ///< Caller should decrement SiteIdx and loop again.
  Skip,        ///< Site could not be instrumented; skip.
};

/// Emit a relay-stub pair + optional out-of-range body for ZeroSGPR mode.
/// Matches the `Plan.Register == ZeroSGPR && UseRelay` branch of the
/// original `buildAdaptivePath` loop.
Expected<SiteAction> emitRelayPath(AdaptiveContext &Ctx, uint32_t SiteIdx,
                                    int64_t BranchToDword) {
  const auto &In = *Ctx.In;
  const auto &Site = (*In.Sites)[SiteIdx];
  const auto &Flags = RuntimeConfig::getInstance();

  const bool NoBodyJumpBase = Flags.Transform.NoBodyJump;
  const bool MinRelay = Flags.Transform.MinimalRelay;
  const bool NopRelay = Flags.Transform.NopRelay;
  const bool VccOnlyRelay = Flags.Transform.VccOnlyRelay;
  const unsigned MaxBodySites = Flags.Debug.MaxBodySites;
  const int OnlyBodySite = Flags.Debug.BodySiteOnly;
  bool NoBodyJump = NoBodyJumpBase || (SiteIdx >= MaxBodySites);
  if (OnlyBodySite >= 0)
    NoBodyJump = (static_cast<int>(SiteIdx) != OnlyBodySite);

  uint64_t PatchSiteAbs = In.BaseAddr + Site.Offset;
  uint64_t ReturnTargetAbs = PatchSiteAbs + Site.OrigInstSize;
  auto StubsOrErr = Ctx.Relay->emitRelayStubs(
      Site, *In.Scratch, In.Code, ReturnTargetAbs, Ctx.Alloc->getCurrentAbsolute(),
      NoBodyJump);
  if (!StubsOrErr)
    return StubsOrErr.takeError();
  auto &Stubs = *StubsOrErr;

  if (Stubs.ForwardStub.empty() && Stubs.ReturnStub.empty())
    return SiteAction::Skip;

  uint64_t RelayReturnAbs =
      Ctx.Alloc->getCurrentAbsolute() + Stubs.ForwardStub.size();

  TrampolineSlot Slot;
  Slot.OriginalPC = Site.Address;
  Slot.DisplacedSize = Site.OrigInstSize;
  Slot.TrampolineBytes.clear();
  ISAEncoder::append(Slot.TrampolineBytes, Stubs.ForwardStub);
  ISAEncoder::append(Slot.TrampolineBytes, Stubs.ReturnStub);

  if (!IslandAllocator::inBranchRange(BranchToDword))
    return SiteAction::Skip;

  uint64_t StubSize = Slot.TrampolineBytes.size();
  if (Ctx.Alloc->wouldOverlapOccupied(StubSize)) {
    if (Ctx.Alloc->getCursor() > 0) {
      Ctx.Alloc->startNewIsland();
      return SiteAction::Retry;
    }
    return SiteAction::Skip;
  }

  auto BrToRelay =
      In.Enc->encodeSBranch(static_cast<int16_t>(BranchToDword));
  if (!BrToRelay)
    return BrToRelay.takeError();
  Slot.PatchBytes = std::move(*BrToRelay);
  if (Site.OrigInstSize > 4) {
    auto NopPad = In.Enc->encodeNop();
    if (!NopPad)
      return NopPad.takeError();
    ISAEncoder::append(Slot.PatchBytes, *NopPad);
  }

  if (!NoBodyJump && !MinRelay && !NopRelay && !VccOnlyRelay) {
    RelayFixup Fix;
    Fix.FwdGetPCAbs =
        Ctx.Alloc->getCurrentAbsolute() + Stubs.FwdLongJumpOffset;
    Fix.RelayReturnAbs = RelayReturnAbs;
    Fix.BodyEntryOff = Ctx.Relay->getBodyCursor();
    Fix.RetGetPCBodyOff = 0;

    Flags.logAt(2, "[aegisbit] relay site " + std::to_string(SiteIdx) +
                       ": FwdGetPCAbs=0x" +
                       [&]() {
                         char B[32];
                         snprintf(B, sizeof(B), "%llX",
                                  (unsigned long long)Fix.FwdGetPCAbs);
                         return std::string(B);
                       }() +
                       " BodyEntryOff=" + std::to_string(Fix.BodyEntryOff) +
                       " BodyCursor=" +
                       std::to_string(Ctx.Relay->getBodyCursor()));

    auto BodyOrErr = In.Emitter->emitRelayBody(Site, *In.Plan, *In.Scratch,
                                                *In.Trace, In.RetAddrSGPRPair,
                                                SiteIdx);
    if (!BodyOrErr)
      return BodyOrErr.takeError();

    size_t BodySize = BodyOrErr->size();
    Ctx.Relay->addBodyEntry(*BodyOrErr);

    auto RetLJOff = Ctx.Relay->appendReturnLongJump();
    if (!RetLJOff)
      return RetLJOff.takeError();
    Fix.RetGetPCBodyOff = Fix.BodyEntryOff + BodySize;

    Ctx.Relay->addFixup(Fix);
  }

  Ctx.Alloc->commitRelayStub(std::move(Slot), StubSize);
  Ctx.Result->PatchedCount++;
  return SiteAction::Committed;
}

/// Emit a direct-body slot (s_branch / s_call_b64 / long-jump).
/// Matches the non-relay branch of the original `buildAdaptivePath` loop.
Expected<SiteAction> emitDirectPath(AdaptiveContext &Ctx, uint32_t SiteIdx,
                                     int64_t BranchToDword, bool NeedLongJump) {
  const auto &In = *Ctx.In;
  const auto &Site = (*In.Sites)[SiteIdx];

  TrampolineSlot Slot;
  Slot.OriginalPC = Site.Address;
  Slot.DisplacedSize = Site.OrigInstSize;

  if (In.Plan->Register == RegisterMode::ZeroSGPR) {
    auto BrBytes = In.Enc->encodeSBranch(static_cast<int16_t>(BranchToDword));
    if (!BrBytes)
      return BrBytes.takeError();
    Slot.PatchBytes = std::move(*BrBytes);
    if (Site.OrigInstSize > 4) {
      auto NopBytes = In.Enc->encodeNop();
      if (!NopBytes)
        return NopBytes.takeError();
      ISAEncoder::append(Slot.PatchBytes, *NopBytes);
    }
  } else if (!NeedLongJump) {
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
  } else {
    Slot.UsedLongJump = true;
    Ctx.Result->LongJumpCount++;
    uint64_t PatchSiteAbs = In.BaseAddr + Site.Offset;
    int64_t ForwardByteOffset = Ctx.Alloc->byteOffset(PatchSiteAbs);
    auto LJBytes =
        In.Enc->encodeLongJump(In.RetAddrSGPRPair, ForwardByteOffset);
    if (!LJBytes)
      return LJBytes.takeError();
    Slot.PatchBytes = std::move(*LJBytes);
  }

  auto BodyOrErr = In.Emitter->emitDirectBody(Site, *In.Plan, *In.Scratch,
                                                *In.Trace, In.RetAddrSGPRPair,
                                                SiteIdx);
  if (!BodyOrErr)
    return BodyOrErr.takeError();
  auto TB = std::move(*BodyOrErr);

  TB.insert(TB.end(), In.Code.data() + Site.Offset,
            In.Code.data() + Site.Offset + Site.OrigInstSize);

  if (In.Plan->Register == RegisterMode::ZeroSGPR) {
    uint64_t PatchSiteAbs = In.BaseAddr + Site.Offset;
    uint64_t ReturnTargetAbs = PatchSiteAbs + Site.OrigInstSize;
    uint64_t BranchPC = Ctx.Alloc->getCurrentAbsolute() + TB.size();
    int64_t BackOffset = static_cast<int64_t>(ReturnTargetAbs) -
                         static_cast<int64_t>(BranchPC);
    int64_t BackDword = (BackOffset - 4) / 4;

    if (!IslandAllocator::inBranchRange(BackDword))
      return SiteAction::Skip;

    auto Ret = In.Enc->encodeSBranch(static_cast<int16_t>(BackDword));
    if (!Ret)
      return Ret.takeError();
    ISAEncoder::append(TB, *Ret);
  } else {
    auto Ret = In.Enc->encodeSetPC(In.RetAddrSGPRPair);
    if (!Ret)
      return Ret.takeError();
    ISAEncoder::append(TB, *Ret);
  }

  if (Ctx.Alloc->wouldOverlapKernel(TB.size())) {
    if (SiteIdx == *Ctx.LastRetrySiteIdx) {
      (*Ctx.RetryCount)++;
    } else {
      *Ctx.LastRetrySiteIdx = SiteIdx;
      *Ctx.RetryCount = 1;
    }
    constexpr uint32_t MAX_RETRIES = 4;
    if (*Ctx.RetryCount > MAX_RETRIES)
      return SiteAction::Skip;
    if (Ctx.Alloc->getCursor() > 0) {
      Ctx.Alloc->startNewIsland();
      return SiteAction::Retry;
    }
    return SiteAction::Skip;
  }

  uint64_t BodySize = TB.size();
  Slot.TrampolineBytes = std::move(TB);

  if (In.Plan->Register == RegisterMode::ZeroSGPR) {
    Ctx.Alloc->commitSlot(std::move(Slot), BodySize);
  } else {
    Slot.TrampolineOffset = *Ctx.StandardCursor;
    *Ctx.StandardCursor += BodySize;
    Ctx.Result->Slots.push_back(std::move(Slot));
  }
  Ctx.Result->PatchedCount++;
  return SiteAction::Committed;
}

class AdaptiveStrategy final : public TrampolineStrategy {
public:
  Expected<BridgeResult> build(const BridgeInputs &In) override {
    BridgeResult Result;
    const auto &Sites = *In.Sites;

    IslandAllocator Alloc(In.BaseAddr, In.TextSectionSize, In.PreKernelSpace);
    for (const auto &R : *In.Occupied)
      Alloc.addOccupiedRegion(R.first, R.second);
    RelayEmitter Relay(*In.Enc);

    uint64_t StandardCursor = 0;
    uint32_t LastRetrySiteIdx = UINT32_MAX;
    uint32_t RetryCount = 0;
    const bool ForceAllRelay = shouldForceAllRelay(
        *In.Plan, Sites.size(), In.BaseAddr, In.TextSectionSize);
    constexpr uint32_t MAX_RETRIES = 4;

    AdaptiveContext Ctx{&In,        &Alloc, &Relay, &StandardCursor,
                        &LastRetrySiteIdx, &RetryCount, &Result};

    for (uint32_t SiteIdx = 0; SiteIdx < Sites.size(); ++SiteIdx) {
      const auto &Site = Sites[SiteIdx];
      (void)Site;
      bool UseRelay = ForceAllRelay;

      uint64_t PatchSiteAbs = In.BaseAddr + Site.Offset;
      int64_t BranchToDword;
      if (In.Plan->Register == RegisterMode::ZeroSGPR) {
        BranchToDword = Alloc.branchDword(PatchSiteAbs);
      } else {
        int64_t TrampolineAbs = static_cast<int64_t>(Alloc.getIslandStart()) +
                                static_cast<int64_t>(StandardCursor);
        BranchToDword =
            (TrampolineAbs - static_cast<int64_t>(PatchSiteAbs) - 4) / 4;
      }
      bool NeedLongJump = !IslandAllocator::inBranchRange(BranchToDword);

      if (resolveAdaptiveOverflow(*In.Plan, Alloc, PatchSiteAbs, SiteIdx,
                                  LastRetrySiteIdx, RetryCount, MAX_RETRIES,
                                  UseRelay, BranchToDword)) {
        --SiteIdx;
        continue;
      }

      auto dispatch = [&]() -> Expected<SiteAction> {
        if (In.Plan->Register == RegisterMode::ZeroSGPR && UseRelay)
          return emitRelayPath(Ctx, SiteIdx, BranchToDword);
        return emitDirectPath(Ctx, SiteIdx, BranchToDword, NeedLongJump);
      };

      auto Action = dispatch();
      if (!Action)
        return Action.takeError();
      if (*Action == SiteAction::Retry) {
        --SiteIdx;
        continue;
      }
    }

    // Compute body island start BEFORE moving islands out of the allocator,
    // so computeBodyIslandStart() can see FinalizedIslands for correct
    // placement.
    bool HasBodyIsland = Relay.hasFixups() && !Relay.getBodyBytes().empty();
    uint64_t BodyIslandStart = 0;

    if (In.Plan->Register == RegisterMode::ZeroSGPR) {
      Alloc.finalizeCurrentIsland();
      Alloc.sortIslands();

      if (HasBodyIsland) {
        BodyIslandStart =
            Alloc.computeBodyIslandStart(Relay.getBodyBytes().size());
        char B[32];
        snprintf(B, sizeof(B), "%llX", (unsigned long long)BodyIslandStart);
        RuntimeConfig::getInstance().logAt(
            2, std::string("[aegisbit] BodyIsland: computeBodyIslandStart=0x") +
                   B + " FinalizedIslands=" +
                   std::to_string(Alloc.getIslands().size()));
      }

      Result.Islands = std::move(Alloc.getIslands());
      for (auto &S : Alloc.getSlots())
        Result.Slots.push_back(std::move(S));
    } else {
      TrampolineIsland Isl;
      uint64_t IslOff = (In.TextSectionSize + 255) & ~255ULL;
      IslOff = Alloc.avoidOccupiedForward(IslOff);
      Isl.Offset = IslOff;
      for (const auto &Slot : Result.Slots) {
        Isl.Bytes.insert(Isl.Bytes.end(), Slot.TrampolineBytes.begin(),
                         Slot.TrampolineBytes.end());
      }
      if (!Isl.Bytes.empty())
        Result.Islands.push_back(std::move(Isl));
    }

    if (HasBodyIsland) {
      char B[32];
      snprintf(B, sizeof(B), "%llX", (unsigned long long)BodyIslandStart);
      RuntimeConfig::getInstance().logAt(
          2, std::string("[aegisbit] BodyIsland: final start=0x") + B +
                 " bodySize=" + std::to_string(Relay.getBodyBytes().size()));
      auto BodyIslOrErr = Relay.fixupRelays(BodyIslandStart, Result.Islands);
      if (!BodyIslOrErr)
        return BodyIslOrErr.takeError();
      Result.Islands.push_back(std::move(*BodyIslOrErr));
      Result.LongJumpCount += static_cast<uint32_t>(Relay.getFixupCount());
    }

    return Result;
  }
};

} // namespace

std::unique_ptr<TrampolineStrategy> createAdaptiveStrategy() {
  return std::make_unique<AdaptiveStrategy>();
}

} // namespace aegisbit
