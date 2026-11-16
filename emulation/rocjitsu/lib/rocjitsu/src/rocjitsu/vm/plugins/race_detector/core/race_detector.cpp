// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

RaceDetector::RaceDetector(int nWaves, int vgprCount, int sgprCount, Dim3d workgroupId,
                           std::function<void(RaceViolation)> raceHandler,
                           CounterCapacities counterCapacities, size_t ldsSizeBytes)
    : ldsSizeBytes_(ldsSizeBytes), workgroupId(workgroupId), raceHandler(std::move(raceHandler)) {
  waveRaceStates.reserve(nWaves);
  for (int i = 0; i < nWaves; ++i) {
    waveRaceStates.emplace_back(vgprCount, sgprCount, WaveId{i}, this, counterCapacities);
  }
}

WaveRaceState &RaceDetector::getWaveRaceState(int waveIndex) {
  assert(waveIndex >= 0 && waveIndex < static_cast<int>(waveRaceStates.size()));
  return waveRaceStates[waveIndex];
}

EventId RaceDetector::allocateEventId(
    WaveId waveId, InstructionSnapshot instruction, MemoryEventType type,
    std::vector<uint32_t> registers, uint64_t execMask, uint8_t byteMask, IntervalSet ldsIntervals,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder, uint8_t lastRegisterByteMask, LdsAccess ldsAccess) {
  bool hasLds = !ldsIntervals.empty();
  EventId eid = events_.add(waveId, instruction, type, std::move(registers), execMask, byteMask,
                            std::move(ldsIntervals), counterObligations, memoryOrder,
                            lastRegisterByteMask, ldsAccess);
  if (hasLds) {
    const auto &ivs = events_.ldsIntervals(eid);
    const WaveId orderedWave = memoryOrder == MemoryOrderClass::LDS ? waveId : WaveId{-1};
    if (events_.writesLds(eid)) {
      ldsWriteEvents.push_back(eid);
      adjustByteCounts(ivs, byteWriteCounts, +1, orderedWave);
    }
    if (events_.readsLds(eid)) {
      ldsReadEvents.push_back(eid);
      adjustByteCounts(ivs, byteReadCounts, +1, orderedWave);
    }
  }
  return eid;
}

void RaceDetector::markEventWaveComplete(EventId eventId) { events_.markComplete(eventId); }

bool RaceDetector::satisfyEventWaitCounter(EventId eventId, amdgpu::WaitCounterType waitCounter) {
  return events_.satisfyWaitCounter(eventId, waitCounter);
}

void RaceDetector::retireEvent(EventId eventId) {
  if (events_.writesLds(eventId)) {
    removeFromUnorderedList(ldsWriteEvents, eventId);
    adjustByteCounts(events_.ldsIntervals(eventId), byteWriteCounts, -1);
  }
  if (events_.readsLds(eventId)) {
    removeFromUnorderedList(ldsReadEvents, eventId);
    adjustByteCounts(events_.ldsIntervals(eventId), byteReadCounts, -1);
  }
  events_.markRetired(eventId);
}

void RaceDetector::validateRead(int addr, WaveId wave, int lane, int nBytes,
                                MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes) const {
  if (addr < 0 || nBytes <= 0 || !validLdsRange(addr, nBytes))
    return;
  if (hasPotentialLdsConflict(addr, nBytes, byteWriteCounts, wave, currentMemoryOrder))
    reportUnorderedOverlaps(ldsWriteEvents, addr, wave, lane, nBytes, /*isWrite=*/false,
                            currentMemoryOrder, atomicBytes);
}

void RaceDetector::validateWrite(int addr, WaveId wave, int lane, int nBytes,
                                 MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes) const {
  if (addr < 0 || nBytes <= 0 || !validLdsRange(addr, nBytes))
    return;
  // An earlier write to the same bytes that is not ordered before this one can
  // land after it.
  if (hasPotentialLdsConflict(addr, nBytes, byteWriteCounts, wave, currentMemoryOrder))
    reportUnorderedOverlaps(ldsWriteEvents, addr, wave, lane, nBytes, /*isWrite=*/true,
                            currentMemoryOrder, atomicBytes);
  if (hasPotentialLdsConflict(addr, nBytes, byteReadCounts, wave, currentMemoryOrder))
    reportUnorderedOverlaps(ldsReadEvents, addr, wave, lane, nBytes, /*isWrite=*/true,
                            currentMemoryOrder, atomicBytes);
}

void RaceDetector::reportUnorderedOverlaps(const std::vector<EventId> &pending, int addr,
                                           WaveId wave, int lane, int nBytes, bool isWrite,
                                           MemoryOrderClass currentMemoryOrder,
                                           uint8_t atomicBytes) const {
  for (EventId eventId : pending) {
    if (events_.atomicLdsElementMatches(eventId, atomicBytes))
      continue;
    if (wave == events_.waveId(eventId)) {
      // Operations in the same non-UNORDERED class complete in issue order.
      // A completed event is safe for its owning wave regardless of class, but
      // remains live for cross-wave checks until a barrier retires it. A
      // pending read's destination VGPR remains independently protected by its
      // wait-counter obligations.
      const MemoryOrderClass pendingMemoryOrder = events_.memoryOrder(eventId);
      const bool orderedWithCurrent = currentMemoryOrder != MemoryOrderClass::UNORDERED &&
                                      pendingMemoryOrder == currentMemoryOrder;
      if (orderedWithCurrent || events_.status(eventId) == EventStatus::WAVE_COMPLETE)
        continue;
    }
    // Report a byte both accesses touch, not the start of this access.
    if (const auto byte = events_.ldsIntervals(eventId).firstOverlap(addr, addr + nBytes)) {
      raceHandler(
          {RaceViolation::Space::LDS, *byte, wave.value, lane, isWrite, workgroupId, eventId});
    }
  }
}

bool RaceDetector::mayConflict(const IntervalSet &accessed, WaveId wave, bool isWrite,
                               MemoryOrderClass currentMemoryOrder) const {
  // A write conflicts with pending reads and, through the write-after-write
  // check in validateWrite, with pending writes.
  return std::ranges::any_of(accessed, [&](const Interval &interval) {
    const int bytes = interval.end - interval.start;
    return hasPotentialLdsConflict(interval.start, bytes, byteWriteCounts, wave,
                                   currentMemoryOrder) ||
           (isWrite && hasPotentialLdsConflict(interval.start, bytes, byteReadCounts, wave,
                                               currentMemoryOrder));
  });
}

bool RaceDetector::hasPotentialLdsConflict(int addr, int nBytes,
                                           const std::vector<LdsChunkCounts> &counts, WaveId wave,
                                           MemoryOrderClass currentMemoryOrder) {
  if (addr < 0 || nBytes <= 0 || nBytes > std::numeric_limits<int>::max() - addr)
    return false;
  const int limit = static_cast<int>(counts.size());
  const int cStart = addr / kCountGranularity;
  const int cEnd = (addr + nBytes - 1) / kCountGranularity + 1;
  // Every overlapping event contributes to a touched chunk. A matching owner
  // proves that the full scan would skip it by the same-wave LDS ordering rule.
  for (int c = cStart; c < cEnd && c < limit; ++c) {
    if (counts[c].bytes > 0 &&
        (currentMemoryOrder != MemoryOrderClass::LDS || counts[c].orderedWave != wave))
      return true;
  }
  return false;
}

void RaceDetector::adjustByteCounts(const IntervalSet &ivs, std::vector<LdsChunkCounts> &counts,
                                    int delta, WaveId orderedWave) {
  for (const auto &iv : ivs) {
    if (iv.start < 0 || iv.end <= iv.start ||
        !validLdsRange(iv.start, static_cast<uint64_t>(iv.end) - iv.start))
      continue;
    int cStart = iv.start / kCountGranularity;
    int cEnd = (iv.end - 1) / kCountGranularity + 1;
    if (cEnd > static_cast<int>(counts.size()))
      counts.resize(cEnd);
    for (int c = cStart; c < cEnd; ++c) {
      int byteStart = std::max(iv.start, c * kCountGranularity);
      int nBytes = std::min(iv.end - byteStart, kCountGranularity - byteStart % kCountGranularity);
      auto &chunk = counts[c];
      if (delta > 0) {
        if (chunk.bytes == 0)
          chunk.orderedWave = orderedWave;
        else if (chunk.orderedWave != orderedWave)
          chunk.orderedWave = WaveId{-1};
      }
      chunk.bytes += delta * nBytes;
    }
  }
}

} // namespace rocjitsu::plugins::race_detector
