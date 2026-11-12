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
  acquired_.assign(static_cast<size_t>(nWaves) * nWaves, 0);
  // Each wave starts in epoch 1, so 0 means nothing acquired.
  for (int i = 0; i < nWaves; ++i)
    acquired(WaveId{i}, WaveId{i}) = 1;
  exitClock_.assign(nWaves, 0);
  ended_.assign(nWaves, false);
  // Round membership is a bit per wave. A workgroup has at most 1024
  // work-items, so at most 32 wave32 waves (CDNA5 ISA 2.2).
  assert(nWaves <= 64);
  // With no other wave, nothing holds back retirement.
  covered_.assign(nWaves, nWaves == 1 ? std::numeric_limits<uint32_t>::max() : 0);
  atCoverage_.assign(nWaves, nWaves - 1);
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
  // An event with no counter obligation is complete now, in its wave's
  // current barrier epoch.
  EventId eid = events_.add(waveId, instruction, type, std::move(registers), execMask, byteMask,
                            std::move(ldsIntervals), counterObligations, memoryOrder,
                            lastRegisterByteMask, ldsAccess, acquired(waveId, waveId));
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

void RaceDetector::markEventWaveComplete(EventId eventId) {
  const WaveId owner = events_.waveId(eventId);
  events_.markComplete(eventId, acquired(owner, owner));
}

namespace {

/// Join `from` into `into`, element-wise maximum.
void joinClock(std::span<uint32_t> into, std::span<const uint32_t> from) {
  for (size_t index = 0; index < into.size(); ++index)
    into[index] = std::max(into[index], from[index]);
}

} // namespace

RaceDetector::BarrierRoundState *RaceDetector::findRound(BarrierRoundId round) {
  for (BarrierRoundState &state : rounds_)
    if (state.open && state.id == round)
      return &state;
  return nullptr;
}

RaceDetector::BarrierRoundState &RaceDetector::roundState(BarrierRoundId round) {
  if (BarrierRoundState *state = findRound(round))
    return *state;
  // Reuse a closed slot, and its clock's storage, before adding one.
  auto slot = std::ranges::find_if(rounds_, [](const BarrierRoundState &s) { return !s.open; });
  if (slot == rounds_.end())
    slot = rounds_.insert(rounds_.end(), BarrierRoundState{});
  slot->id = round;
  slot->open = true;
  slot->completed = false;
  slot->awaiting = 0;
  slot->clock.assign(waveCount(), 0);
  ++openRounds_;
  return *slot;
}

void RaceDetector::closeRound(BarrierRoundState &state) {
  state.open = false;
  --openRounds_;
}

void RaceDetector::barrierSignaled(WaveId wave, BarrierRoundId round) {
  BarrierRoundState &state = roundState(round);
  const auto row =
      std::span(acquired_).subspan(static_cast<size_t>(wave.value) * waveCount(), waveCount());
  joinClock(state.clock, row);
  // Events the wave completes from now on wait for its next signal.
  ++acquired(wave, wave);
}

void RaceDetector::barrierRoundCompleted(BarrierRoundId round, std::span<const WaveId> members,
                                         bool wholeWorkgroup) {
  BarrierRoundState &state = roundState(round);
  if (wholeWorkgroup)
    joinClock(state.clock, exitClock_);
  state.completed = true;
  state.awaiting = 0;
  for (WaveId member : members)
    if (!ended_[member.value])
      state.awaiting |= uint64_t{1} << member.value;
  if (state.awaiting == 0)
    closeRound(state);
}

void RaceDetector::barrierWaitPassed(WaveId wave, BarrierRoundId round) {
  if (BarrierRoundState *state = findRound(round); state && state->completed)
    acquire(wave, state->clock);
  // The wave's completion bit is now clear. It never waits for another round
  // of this barrier that has already completed: one that completed while the
  // bit was set left the bit unchanged (CDNA5 ISA 5.6.1).
  stopAwaiting(wave, round.barrier);
}

void RaceDetector::waveEnded(WaveId wave) {
  ended_[wave.value] = true;
  const auto row =
      std::span(acquired_).subspan(static_cast<size_t>(wave.value) * waveCount(), waveCount());
  joinClock(exitClock_, row);
  stopAwaiting(wave, std::nullopt);
  // The ended wave no longer holds back any other wave's retirement.
  for (int owner = 0; owner < waveCount(); ++owner)
    if (owner != wave.value && acquired(wave, WaveId{owner}) == covered_[owner])
      leaveCoverageMinimum(wave, WaveId{owner});
}

void RaceDetector::stopAwaiting(WaveId wave, std::optional<uint8_t> barrier) {
  const uint64_t bit = uint64_t{1} << wave.value;
  for (BarrierRoundState &state : rounds_) {
    if (!state.open || !state.completed || (barrier && state.id.barrier != *barrier) ||
        (state.awaiting & bit) == 0)
      continue;
    state.awaiting &= ~bit;
    if (state.awaiting == 0)
      closeRound(state);
  }
}

void RaceDetector::acquire(WaveId accessor, std::span<const uint32_t> clock) {
  for (int owner = 0; owner < waveCount(); ++owner) {
    uint32_t &entry = acquired(accessor, WaveId{owner});
    if (clock[owner] <= entry)
      continue;
    const bool wasAtCoverage = owner != accessor.value && entry == covered_[owner];
    entry = clock[owner];
    if (wasAtCoverage)
      leaveCoverageMinimum(accessor, WaveId{owner});
  }
}

void RaceDetector::leaveCoverageMinimum(WaveId /*accessor*/, WaveId owner) {
  // Coverage can rise only when the last wave at the minimum leaves it.
  if (--atCoverage_[owner.value] == 0)
    recomputeCoverage(owner);
}

void RaceDetector::recomputeCoverage(WaveId owner) {
  uint32_t covered = std::numeric_limits<uint32_t>::max();
  int atCoverage = 0;
  for (int accessor = 0; accessor < waveCount(); ++accessor) {
    if (accessor == owner.value || ended_[accessor])
      continue;
    const uint32_t entry = acquired(WaveId{accessor}, owner);
    if (entry < covered) {
      covered = entry;
      atCoverage = 1;
    } else if (entry == covered) {
      ++atCoverage;
    }
  }
  covered_[owner.value] = covered;
  atCoverage_[owner.value] = atCoverage;
  waveRaceStates[owner.value].retireReleasedThrough(covered);
}

bool RaceDetector::ordered(WaveId accessor, EventId eventId) const {
  return events_.status(eventId) == EventStatus::WAVE_COMPLETE &&
         acquired(accessor, events_.waveId(eventId)) >= events_.releaseEpoch(eventId);
}

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
  if (!hasPotentialLdsConflict(addr, nBytes, byteWriteCounts, wave, currentMemoryOrder)) {
    return;
  }

  for (EventId eventId : ldsWriteEvents) {
    if (events_.atomicLdsElementMatches(eventId, atomicBytes))
      continue;
    if (wave == events_.waveId(eventId)) {
      // Operations in the same non-UNORDERED class complete in issue order.
      // A completed event is safe for its owning wave regardless of class, but
      // other waves need a barrier to order it.
      const MemoryOrderClass pendingMemoryOrder = events_.memoryOrder(eventId);
      const bool orderedWithCurrent = currentMemoryOrder != MemoryOrderClass::UNORDERED &&
                                      pendingMemoryOrder == currentMemoryOrder;
      if (orderedWithCurrent || events_.status(eventId) == EventStatus::WAVE_COMPLETE) {
        continue;
      }
    } else if (ordered(wave, eventId)) {
      continue;
    }
    // Report a byte both accesses touch, not the start of this access.
    if (const auto byte = events_.ldsIntervals(eventId).firstOverlap(addr, addr + nBytes)) {
      raceHandler(
          {RaceViolation::Space::LDS, *byte, wave.value, lane, false, workgroupId, eventId});
    }
  }
}

void RaceDetector::validateWrite(int addr, WaveId wave, int lane, int nBytes,
                                 MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes) const {
  if (addr < 0 || nBytes <= 0 || !validLdsRange(addr, nBytes))
    return;
  if (!hasPotentialLdsConflict(addr, nBytes, byteReadCounts, wave, currentMemoryOrder)) {
    return;
  }

  for (EventId eventId : ldsReadEvents) {
    if (events_.atomicLdsElementMatches(eventId, atomicBytes))
      continue;
    if (wave == events_.waveId(eventId)) {
      // The LDS bytes are safe only when the two operations share a proven
      // FIFO completion class, or an explicit wait completed the older event.
      // The read's destination VGPR remains independently protected by its
      // wait-counter obligations. This WAR check does not cover LDS
      // write/write ordering, which is not currently checked.
      const MemoryOrderClass pendingMemoryOrder = events_.memoryOrder(eventId);
      const bool orderedWithCurrent = currentMemoryOrder != MemoryOrderClass::UNORDERED &&
                                      pendingMemoryOrder == currentMemoryOrder;
      if (orderedWithCurrent || events_.status(eventId) == EventStatus::WAVE_COMPLETE)
        continue;
    } else if (ordered(wave, eventId)) {
      continue;
    }
    // Report a byte both accesses touch, not the start of this access.
    if (const auto byte = events_.ldsIntervals(eventId).firstOverlap(addr, addr + nBytes)) {
      raceHandler({RaceViolation::Space::LDS, *byte, wave.value, lane, true, workgroupId, eventId});
    }
  }
}

bool RaceDetector::mayConflict(const IntervalSet &accessed, WaveId wave, bool isWrite,
                               MemoryOrderClass currentMemoryOrder) const {
  const std::vector<LdsChunkCounts> &counts = isWrite ? byteReadCounts : byteWriteCounts;
  return std::ranges::any_of(accessed, [&](const Interval &interval) {
    return hasPotentialLdsConflict(interval.start, interval.end - interval.start, counts, wave,
                                   currentMemoryOrder);
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
