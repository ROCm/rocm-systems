//===-- IslandAllocator.cpp - Island Layout Manager --------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/IslandAllocator.h"
#include "aegisbit/RuntimeConfig.h"

#include <algorithm>
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace aegisbit {

IslandAllocator::IslandAllocator(uint64_t BaseAddr, uint64_t TextSectionSize,
                                 uint64_t PreKernelSpace)
    : BaseAddr(BaseAddr), TextSectionSize(TextSectionSize),
      PreKernelSpace(PreKernelSpace) {
  IslandStartAbsolute = (TextSectionSize + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

void IslandAllocator::addOccupiedRegion(uint64_t Start, uint64_t End) {
  if (Start >= End)
    return;
  OccupiedRegions.emplace_back(Start, End);
  uint64_t OldStart = IslandStartAbsolute;
  IslandStartAbsolute = avoidOverlaps(IslandStartAbsolute, BackwardMode);
  if (IslandStartAbsolute != OldStart) {
    llvm::errs() << llvm::formatv(
        "[aegisbit] IslandAllocator: adjusted island start {0:X} -> {1:X}"
        " (avoiding occupied [{2:X}, {3:X}))\n",
        OldStart, IslandStartAbsolute, Start, End);
  }
}

int64_t IslandAllocator::branchDword(uint64_t PatchSiteAbs) const {
  int64_t ByteOff = byteOffset(PatchSiteAbs);
  return (ByteOff - 4) / 4;
}

int64_t IslandAllocator::byteOffset(uint64_t PatchSiteAbs) const {
  uint64_t TrampolineAbs = IslandStartAbsolute + IslandCursor;
  return static_cast<int64_t>(TrampolineAbs) -
         static_cast<int64_t>(PatchSiteAbs);
}

void IslandAllocator::commitSlot(TrampolineSlot Slot, uint64_t ByteSize) {
  Slot.TrampolineOffset = IslandCursor;
  IslandCursor += ByteSize;
  CurrentIslandSlots.push_back(std::move(Slot));
}

void IslandAllocator::commitRelayStub(TrampolineSlot Slot, uint64_t StubSize) {
  Slot.TrampolineOffset = IslandCursor;
  IslandCursor += StubSize;
  CurrentIslandSlots.push_back(std::move(Slot));
}

void IslandAllocator::finalizeCurrentIsland() {
  if (CurrentIslandSlots.empty() && IslandCursor == 0)
    return;
  TrampolineIsland Isl;
  Isl.Offset = IslandStartAbsolute;
  for (const auto &S : CurrentIslandSlots)
    Isl.Bytes.insert(Isl.Bytes.end(),
                      S.TrampolineBytes.begin(),
                      S.TrampolineBytes.end());
  if (!Isl.Bytes.empty()) {
    // Register the finalized island as an occupied region so subsequent
    // backward searches and body-island placement can skip past it.
    OccupiedRegions.emplace_back(Isl.Offset,
                                  Isl.Offset + Isl.Bytes.size());
    FinalizedIslands.push_back(std::move(Isl));
  }
  for (auto &S : CurrentIslandSlots)
    FinalizedSlots.push_back(std::move(S));
  CurrentIslandSlots.clear();
  IslandCursor = 0;
}

uint64_t IslandAllocator::avoidOverlaps(uint64_t candidate,
                                         bool searchDown) const {
  bool changed = true;
  while (changed) {
    changed = false;
    uint64_t prev = candidate;
    for (const auto &Isl : FinalizedIslands) {
      uint64_t IslEnd = Isl.Offset + Isl.Bytes.size();
      if (candidate >= Isl.Offset && candidate < IslEnd) {
        if (searchDown) {
          candidate = (Isl.Offset >= ALIGNMENT)
                          ? (Isl.Offset - 1) & ~(ALIGNMENT - 1)
                          : 0;
        } else {
          candidate = (IslEnd + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        }
        changed = (candidate != prev);
      }
    }
    for (const auto &R : OccupiedRegions) {
      if (candidate >= R.first && candidate < R.second) {
        if (searchDown) {
          candidate = (R.first >= ALIGNMENT)
                          ? (R.first - 1) & ~(ALIGNMENT - 1)
                          : 0;
        } else {
          candidate = (R.second + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        }
        changed = (candidate != prev);
      }
    }
  }

  // If backward search got stuck inside an occupied region (e.g. hit 0),
  // fall back to a forward search from the same position to find the
  // first clear gap above.
  if (searchDown) {
    bool overlapping = false;
    for (const auto &R : OccupiedRegions)
      if (candidate >= R.first && candidate < R.second) { overlapping = true; break; }
    if (!overlapping)
      for (const auto &Isl : FinalizedIslands) {
        uint64_t IslEnd = Isl.Offset + Isl.Bytes.size();
        if (candidate >= Isl.Offset && candidate < IslEnd) { overlapping = true; break; }
      }
    if (overlapping)
      candidate = avoidOverlaps(candidate, /*searchDown=*/false);
  }

  return candidate;
}

void IslandAllocator::startNewIsland() {
  finalizeCurrentIsland();
  if (BackwardMode) {
    uint64_t step = (IslandCursor > 0 ? IslandCursor : 65536);
    if (IslandStartAbsolute > step + ALIGNMENT - 1)
      IslandStartAbsolute =
          (IslandStartAbsolute - step - (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
    else
      IslandStartAbsolute = 0;
    IslandStartAbsolute = avoidOverlaps(IslandStartAbsolute, true);
  } else {
    IslandStartAbsolute =
        (IslandStartAbsolute + IslandCursor + ALIGNMENT - 1) &
        ~(ALIGNMENT - 1);
    IslandStartAbsolute = avoidOverlaps(IslandStartAbsolute, false);
  }
}

void IslandAllocator::switchToBackward(uint64_t SiteAbs) {
  finalizeCurrentIsland();
  BackwardMode = true;
  uint64_t target = (BaseAddr > ALIGNMENT)
                        ? (BaseAddr - ALIGNMENT) & ~(ALIGNMENT - 1)
                        : 0;
  uint64_t beforeAvoid = target;
  target = avoidOverlaps(target, true);
  {
    char buf[256];
    snprintf(buf, sizeof(buf),
        "[aegisbit] switchToBackward: BaseAddr=0x%llX initial=0x%llX"
        " afterAvoid=0x%llX OccupiedCount=%zu\n",
        (unsigned long long)BaseAddr, (unsigned long long)beforeAvoid,
        (unsigned long long)target, OccupiedRegions.size());
    llvm::errs() << buf;
    bool foundContaining = false;
    for (const auto &R : OccupiedRegions) {
      if (target >= R.first && target < R.second) {
        snprintf(buf, sizeof(buf),
            "[aegisbit]   BUG: result 0x%llX is INSIDE occupied [0x%llX, 0x%llX)!\n",
            (unsigned long long)target,
            (unsigned long long)R.first, (unsigned long long)R.second);
        llvm::errs() << buf;
        foundContaining = true;
      }
    }
    if (!foundContaining) {
      for (const auto &R : OccupiedRegions) {
        if (R.first >= target && R.first < target + 0x500) {
          snprintf(buf, sizeof(buf),
              "[aegisbit]   nearby occupied above: [0x%llX, 0x%llX)\n",
              (unsigned long long)R.first, (unsigned long long)R.second);
          llvm::errs() << buf;
        }
        if (R.second > target - 0x500 && R.second <= target) {
          snprintf(buf, sizeof(buf),
              "[aegisbit]   nearby occupied below: [0x%llX, 0x%llX)\n",
              (unsigned long long)R.first, (unsigned long long)R.second);
          llvm::errs() << buf;
        }
      }
    }
  }
  IslandStartAbsolute = target;
}

bool IslandAllocator::tryResolveOverflow(uint64_t PatchSiteAbs,
                                          uint32_t SiteIdx,
                                          uint32_t &LastRetrySiteIdx,
                                          uint32_t &RetryCount,
                                          uint32_t MaxRetries) {
  if (SiteIdx == LastRetrySiteIdx)
    RetryCount++;
  else {
    LastRetrySiteIdx = SiteIdx;
    RetryCount = 1;
  }

  if (RetryCount > MaxRetries)
    return false; // caller should use relay

  if (IslandCursor > 0) {
    startNewIsland();
    return true; // retry
  }

  if (BackwardMode) {
    uint64_t FwdCandidate =
        (TextSectionSize + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    FwdCandidate = avoidOverlaps(FwdCandidate, false);
    int64_t fwdDw = (static_cast<int64_t>(FwdCandidate) -
                     static_cast<int64_t>(PatchSiteAbs) - 4) /
                    4;
    if (inBranchRange(fwdDw)) {
      finalizeCurrentIsland();
      BackwardMode = false;
      IslandStartAbsolute = FwdCandidate;
      return true; // retry
    }
    return false; // use relay
  }

  if (PreKernelSpace > 0) {
    switchToBackward(PatchSiteAbs);
    return true; // retry
  }

  return false; // use relay
}

bool IslandAllocator::wouldOverlapKernel(uint64_t AdditionalBytes) const {
  return BackwardMode &&
         IslandStartAbsolute + IslandCursor + AdditionalBytes > BaseAddr;
}

bool IslandAllocator::wouldOverlapOccupied(uint64_t AdditionalBytes) const {
  uint64_t End = IslandStartAbsolute + IslandCursor + AdditionalBytes;
  for (const auto &R : OccupiedRegions)
    if (End > R.first && IslandStartAbsolute + IslandCursor < R.second)
      return true;
  return false;
}

void IslandAllocator::sortIslands() {
  std::sort(FinalizedIslands.begin(), FinalizedIslands.end(),
            [](const TrampolineIsland &A, const TrampolineIsland &B) {
              return A.Offset < B.Offset;
            });
}

uint64_t IslandAllocator::avoidOccupiedForward(uint64_t candidate) const {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &R : OccupiedRegions) {
      if (candidate >= R.first && candidate < R.second) {
        candidate = (R.second + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        changed = true;
      }
    }
  }
  return candidate;
}

uint64_t IslandAllocator::computeBodyIslandStart(uint64_t BodySize) const {
  uint64_t MaxEnd = 0;
  uint64_t MinEnd = UINT64_MAX;
  for (const auto &Isl : FinalizedIslands) {
    uint64_t End = Isl.Offset + static_cast<uint64_t>(Isl.Bytes.size());
    MaxEnd = std::max(MaxEnd, End);
    MinEnd = std::min(MinEnd, End);
  }
  if (MinEnd == UINT64_MAX)
    MinEnd = MaxEnd;

  // Verify the full [Start, Start+BodySize) range is clear of every
  // finalized island and occupied region. When BodySize is 0 we skip the
  // check (preserves legacy behavior for callers that don't know the size).
  auto bodyFits = [&](uint64_t Start) {
    if (BodySize == 0)
      return true;
    uint64_t End = Start + BodySize;
    for (const auto &Isl : FinalizedIslands) {
      uint64_t IslEnd = Isl.Offset + Isl.Bytes.size();
      if (End > Isl.Offset && Start < IslEnd)
        return false;
    }
    for (const auto &R : OccupiedRegions) {
      if (End > R.first && Start < R.second)
        return false;
    }
    return true;
  };

  // Prefer placing the body island near the relay stubs (in the pre-kernel
  // padding area) to keep long-jump offsets small.
  const bool ForceNear = RuntimeConfig::getInstance().Transform.BodyNearStubs;
  if (ForceNear || PreKernelSpace > 0) {
    uint64_t nearCandidate = (MinEnd + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    nearCandidate = avoidOverlaps(nearCandidate, false);
    uint64_t farCandidate = (MaxEnd + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    farCandidate = avoidOverlaps(farCandidate, false);
    if (nearCandidate < farCandidate && bodyFits(nearCandidate))
      return nearCandidate;
  }

  uint64_t candidate = (MaxEnd + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
  candidate = avoidOverlaps(candidate, false);
  // Walk forward past any occupied region / island that the body would
  // straddle until we find a slot where the full body range fits.
  while (!bodyFits(candidate)) {
    uint64_t NextStart = UINT64_MAX;
    uint64_t BodyEnd = candidate + BodySize;
    for (const auto &Isl : FinalizedIslands) {
      uint64_t IslEnd = Isl.Offset + Isl.Bytes.size();
      if (BodyEnd > Isl.Offset && candidate < IslEnd)
        NextStart = std::min(NextStart, IslEnd);
    }
    for (const auto &R : OccupiedRegions) {
      if (BodyEnd > R.first && candidate < R.second)
        NextStart = std::min(NextStart, R.second);
    }
    if (NextStart == UINT64_MAX)
      break;
    candidate = (NextStart + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
  }
  return candidate;
}

} // namespace aegisbit
