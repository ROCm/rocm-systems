//===-- aegisbit/IslandAllocator.h - Island Layout Manager ---------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Manages spatial layout of trampoline islands within the .text section.
/// Handles forward/backward island placement, chaining, overlap avoidance,
/// and relay stub positioning -- all without knowing what trampoline code
/// looks like. Only deals with sizes and offsets.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_ISLAND_ALLOCATOR_H
#define AEGISBIT_ISLAND_ALLOCATOR_H

#include "aegisbit/TrampolineTypes.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace aegisbit {

class IslandAllocator {
public:
  static constexpr int64_t SBRANCH_MAX_DWORD = 32767;
  static constexpr int64_t SBRANCH_MIN_DWORD = -32768;
  static constexpr uint64_t ALIGNMENT = 256;

  IslandAllocator(uint64_t BaseAddr, uint64_t TextSectionSize,
                  uint64_t PreKernelSpace);

  /// Register a code region that must not be overwritten (e.g. device
  /// functions in the same code object). Ranges are absolute .text offsets.
  void addOccupiedRegion(uint64_t Start, uint64_t End);

  /// Commit bytes to the current island and record a slot.
  /// Call after generating trampoline bytes for a direct (non-relay) site.
  void commitSlot(TrampolineSlot Slot, uint64_t ByteSize);

  /// Commit relay stub bytes to the current island.
  void commitRelayStub(TrampolineSlot Slot, uint64_t StubSize);

  /// Get the absolute address of the current write cursor in the island.
  uint64_t getCurrentAbsolute() const {
    return IslandStartAbsolute + IslandCursor;
  }

  /// Get the current cursor offset within the island.
  uint64_t getCursor() const { return IslandCursor; }

  /// Compute the s_branch dword offset from a patch site to the current island
  /// cursor. Returns the raw dword offset (caller checks range).
  int64_t branchDword(uint64_t PatchSiteAbs) const;

  /// Check if a dword offset is within s_branch range.
  static bool inBranchRange(int64_t DwordOffset) {
    return DwordOffset >= SBRANCH_MIN_DWORD &&
           DwordOffset <= SBRANCH_MAX_DWORD;
  }

  /// Compute the byte offset from a patch site to the current island cursor.
  int64_t byteOffset(uint64_t PatchSiteAbs) const;

  /// Try to resolve an out-of-range site by starting a new island,
  /// switching to backward mode, or switching back to forward mode.
  /// Returns true if the site should be retried (decremented SiteIdx).
  /// Returns false if the site should use relay or be skipped.
  bool tryResolveOverflow(uint64_t PatchSiteAbs, uint32_t SiteIdx,
                          uint32_t &LastRetrySiteIdx, uint32_t &RetryCount,
                          uint32_t MaxRetries);

  /// Finalize the current island (flush to results).
  void finalizeCurrentIsland();

  /// Start a new island after the current one.
  void startNewIsland();

  /// Switch to backward island placement mode.
  void switchToBackward(uint64_t SiteAbs);

  /// Check if the backward island would overlap the kernel.
  bool wouldOverlapKernel(uint64_t AdditionalBytes) const;

  /// Check if extending the current island by AdditionalBytes would
  /// overlap an occupied region.
  bool wouldOverlapOccupied(uint64_t AdditionalBytes) const;

  /// Get all finalized islands.
  std::vector<TrampolineIsland> &getIslands() { return FinalizedIslands; }

  /// Get all finalized slots.
  std::vector<TrampolineSlot> &getSlots() { return FinalizedSlots; }

  /// Sort islands by offset.
  void sortIslands();

  /// Compute the body island start address (after all existing islands).
  /// When BodySize > 0, validates that the full [Start, Start+BodySize) range
  /// does not intersect any finalized island or occupied region.
  uint64_t computeBodyIslandStart(uint64_t BodySize = 0) const;

  /// Adjust a candidate address forward to avoid occupied regions.
  uint64_t avoidOccupiedForward(uint64_t candidate) const;

  bool isBackwardMode() const { return BackwardMode; }
  uint64_t getIslandStart() const { return IslandStartAbsolute; }
  uint64_t getBaseAddr() const { return BaseAddr; }
  uint64_t getTextSectionSize() const { return TextSectionSize; }
  uint64_t getPreKernelSpace() const { return PreKernelSpace; }

private:
  uint64_t BaseAddr;
  uint64_t TextSectionSize;
  uint64_t PreKernelSpace;

  uint64_t IslandStartAbsolute;
  uint64_t IslandCursor = 0;
  bool BackwardMode = false;

  std::vector<TrampolineSlot> CurrentIslandSlots;
  std::vector<TrampolineIsland> FinalizedIslands;
  std::vector<TrampolineSlot> FinalizedSlots;

  /// Ranges of existing code that must not be overwritten: {start, end}.
  std::vector<std::pair<uint64_t, uint64_t>> OccupiedRegions;

  uint64_t avoidOverlaps(uint64_t candidate, bool searchDown) const;
};

} // namespace aegisbit

#endif // AEGISBIT_ISLAND_ALLOCATOR_H
