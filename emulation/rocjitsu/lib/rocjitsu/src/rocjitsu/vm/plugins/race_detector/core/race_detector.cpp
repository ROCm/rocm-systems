// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"

#include <algorithm>
#include <array>
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
                                MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes) {
  if (addr < 0 || nBytes <= 0 || !validLdsRange(addr, nBytes))
    return;
  if (hasPotentialLdsConflict(addr, nBytes, byteWriteCounts, wave, currentMemoryOrder))
    reportUnorderedOverlaps(ldsWriteEvents, LdsHazard::ReadAfterWrite, addr, wave, lane, nBytes,
                            currentMemoryOrder, atomicBytes);
  reportObservedConflicts(addr, nBytes, wave, lane);
}

void RaceDetector::validateWrite(int addr, WaveId wave, int lane, int nBytes,
                                 MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes,
                                 const uint8_t *data, const InstructionSnapshot *instruction) {
  if (addr < 0 || nBytes <= 0 || !validLdsRange(addr, nBytes))
    return;
  // The write replaces whatever two earlier unordered writes left in its
  // bytes. Where it is itself unordered with a pending write, the check below
  // records a new conflict.
  const std::vector<LdsWawConflict> contested = conflictsIn(addr, addr + nBytes);
  clearConflicts(addr, addr + nBytes);
  if (hasPotentialLdsConflict(addr, nBytes, byteWriteCounts, wave, currentMemoryOrder))
    reportUnorderedOverlaps(ldsWriteEvents, LdsHazard::WriteAfterWrite, addr, wave, lane, nBytes,
                            currentMemoryOrder, atomicBytes, data, instruction, contested);
  if (hasPotentialLdsConflict(addr, nBytes, byteReadCounts, wave, currentMemoryOrder))
    reportUnorderedOverlaps(ldsReadEvents, LdsHazard::WriteAfterRead, addr, wave, lane, nBytes,
                            currentMemoryOrder, atomicBytes);
}

void RaceDetector::differingFromLds(int start, int end, const uint8_t *data,
                                    std::vector<Interval> &spans) const {
  std::array<uint8_t, 64> held;
  for (int at = start; at < end; at += static_cast<int>(held.size())) {
    const int count = std::min(end - at, static_cast<int>(held.size()));
    ldsReader(at, count, held.data());
    if (std::equal(held.begin(), held.begin() + count, data + (at - start)))
      continue;
    for (int i = 0; i < count; ++i) {
      if (held[i] == data[at - start + i])
        continue;
      if (!spans.empty() && spans.back().end == at + i)
        ++spans.back().end;
      else
        spans.push_back({at + i, at + i + 1});
    }
  }
}

void RaceDetector::reportUnorderedOverlaps(const std::vector<EventId> &pending, LdsHazard hazard,
                                           int addr, WaveId wave, int lane, int nBytes,
                                           MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes,
                                           const uint8_t *data,
                                           const InstructionSnapshot *instruction,
                                           std::span<const LdsWawConflict> contested) {
  const int end = addr + nBytes;
  std::vector<LdsWawConflict> added;
  std::vector<Interval> differing;
  auto addConflict = [&](EventId eventId, int start, int stop) {
    added.push_back({start, stop, events_.instruction(eventId), events_.waveId(eventId),
                     instruction ? std::optional(*instruction) : std::nullopt, wave, lane});
  };
  // Only bytes whose values differ can end with either write's value.
  auto checkWrite = [&](EventId eventId, int start, int stop, bool knownValue) {
    if (!knownValue) {
      addConflict(eventId, start, stop);
      return;
    }
    differing.clear();
    differingFromLds(start, stop, data + (start - addr), differing);
    if (differing.empty()) {
      RaceViolation note{
          RaceViolation::Space::LDS, start, wave.value, lane, true, workgroupId, eventId};
      note.kind = RaceViolation::Kind::RedundantWrite;
      raceHandler(note);
    }
    for (const Interval &span : differing)
      addConflict(eventId, span.start, span.end);
  };
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
    const IntervalSet &intervals = events_.ldsIntervals(eventId);
    if (hazard != LdsHazard::WriteAfterWrite) {
      // Report a byte both accesses touch, not the start of this access.
      if (const auto byte = intervals.firstOverlap(addr, end))
        raceHandler({RaceViolation::Space::LDS, *byte, wave.value, lane,
                     hazard == LdsHazard::WriteAfterRead, workgroupId, eventId});
      continue;
    }
    // Sorted disjoint intervals also have sorted ends, so the overlapping ones
    // start at the first interval ending after addr.
    for (auto iv =
             std::ranges::upper_bound(intervals.begin(), intervals.end(), addr, {}, &Interval::end);
         iv != intervals.end() && iv->start < end; ++iv) {
      int at = std::max(iv->start, addr);
      const int stop = std::min(iv->end, end);
      // A pending store from VGPRs (DS, or FLAT to LDS) has already written
      // LDS, so LDS holds its bytes; a pending DMA write has not landed.
      //
      // LDS bytes of a conflict hold its later write's value, so they hold
      // this event's only if this event is that write.
      const bool fromVgprs =
          data && ldsReader && events_.type(eventId) == MemoryEventType::VGPR_TO_LDS;
      for (const LdsWawConflict &held : contested) {
        const int from = std::max(held.start, at);
        const int to = std::min(held.end, stop);
        if (from >= to)
          continue;
        if (at < from)
          checkWrite(eventId, at, from, fromVgprs);
        const bool laterWrite = held.second && *held.second == events_.instruction(eventId) &&
                                held.secondWave == events_.waveId(eventId);
        checkWrite(eventId, from, to, fromVgprs && laterWrite);
        at = to;
      }
      if (at < stop)
        checkWrite(eventId, at, stop, fromVgprs);
    }
  }
  // A later conflict on the same bytes replaces an earlier one. A conflict that
  // continues the one before it, between the same two writes, extends it: the
  // lanes of one instruction write adjacent ranges.
  for (const LdsWawConflict &conflict : added) {
    clearConflicts(conflict.start, conflict.end);
    auto next = wawConflicts.lower_bound(conflict.start);
    if (next != wawConflicts.begin()) {
      LdsWawConflict &prev = std::prev(next)->second;
      if (prev.end == conflict.start && prev.first == conflict.first &&
          prev.firstWave == conflict.firstWave && prev.second == conflict.second &&
          prev.secondWave == conflict.secondWave) {
        prev.end = conflict.end;
        continue;
      }
    }
    wawConflicts.emplace_hint(next, conflict.start, conflict);
  }
}

void RaceDetector::reportObservedConflicts(int addr, int nBytes, WaveId wave, int lane) {
  const int end = addr + nBytes;
  if (!overlapsConflict(addr, end))
    return;
  // Report each conflict the read overlaps once, at the first byte of the overlap.
  auto it = wawConflicts.upper_bound(addr);
  if (it != wawConflicts.begin() && std::prev(it)->second.end > addr)
    --it;
  for (; it != wawConflicts.end() && it->first < end; ++it) {
    const LdsWawConflict &conflict = it->second;
    RaceViolation violation{RaceViolation::Space::LDS,
                            std::max(conflict.start, addr),
                            wave.value,
                            lane,
                            false,
                            workgroupId,
                            EventId{}};
    violation.observedWrite = conflict.first;
    violation.observedWriteWave = conflict.firstWave.value;
    violation.observedLaterWrite = conflict.second;
    violation.observedLaterWriteWave = conflict.secondWave.value;
    raceHandler(violation);
  }
  clearConflicts(addr, end);
}

void RaceDetector::splitConflictAt(int at) {
  auto it = wawConflicts.upper_bound(at);
  if (it == wawConflicts.begin())
    return;
  --it;
  LdsWawConflict &conflict = it->second;
  if (conflict.start >= at || conflict.end <= at)
    return;
  LdsWawConflict right = conflict;
  right.start = at;
  conflict.end = at;
  wawConflicts.emplace(at, right);
}

void RaceDetector::clearConflicts(int start, int end) {
  if (!overlapsConflict(start, end))
    return;
  splitConflictAt(start);
  splitConflictAt(end);
  wawConflicts.erase(wawConflicts.lower_bound(start), wawConflicts.lower_bound(end));
}

std::vector<RaceDetector::LdsWawConflict> RaceDetector::conflictsIn(int start, int end) const {
  std::vector<LdsWawConflict> parts;
  if (!overlapsConflict(start, end))
    return parts;
  auto it = wawConflicts.upper_bound(start);
  if (it != wawConflicts.begin() && std::prev(it)->second.end > start)
    --it;
  for (; it != wawConflicts.end() && it->first < end; ++it) {
    LdsWawConflict part = it->second;
    part.start = std::max(part.start, start);
    part.end = std::min(part.end, end);
    parts.push_back(std::move(part));
  }
  return parts;
}

bool RaceDetector::overlapsConflict(int start, int end) const {
  if (wawConflicts.empty() || start >= end)
    return false;
  auto it = wawConflicts.lower_bound(end);
  if (it == wawConflicts.begin())
    return false;
  return std::prev(it)->second.end > start;
}

bool RaceDetector::mayConflict(const IntervalSet &accessed, WaveId wave, bool isWrite,
                               MemoryOrderClass currentMemoryOrder) const {
  // A write conflicts with pending reads and, through the write-after-write
  // check in validateWrite, with pending writes.
  return std::ranges::any_of(accessed, [&](const Interval &interval) {
    if (overlapsConflict(interval.start, interval.end))
      return true;
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
