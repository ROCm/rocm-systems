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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

RaceDetector::RaceDetector(int nWaves, int vgprCount, int sgprCount, Dim3d workgroupId,
                           std::function<void(RaceViolation)> raceHandler,
                           CounterCapacities counterCapacities, size_t ldsSizeBytes)
    : ldsSizeBytes_(ldsSizeBytes), workgroupId(workgroupId), raceHandler(std::move(raceHandler)) {
  // Barrier rounds keep a bit per wave. A workgroup has at most 1024
  // work-items (CDNA5 ISA 2.2), so at most 32 waves of 32 lanes.
  if (nWaves > kMaxWaves)
    throw std::length_error("race detector supports at most " + std::to_string(kMaxWaves) +
                            " waves in a workgroup, got " + std::to_string(nWaves));
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
  // With no other wave, nothing holds back retirement.
  covered_.assign(nWaves, nWaves == 1 ? std::numeric_limits<uint32_t>::max() : 0);
  atCoverage_.assign(nWaves, nWaves - 1);
  staleCoverage_.assign(nWaves, false);
  liveWaves_ = nWaves;
}

WaveRaceState &RaceDetector::getWaveRaceState(int waveIndex) {
  assert(waveIndex >= 0 && waveIndex < static_cast<int>(waveRaceStates.size()));
  return waveRaceStates[waveIndex];
}

EventId RaceDetector::allocateEventId(
    WaveId waveId, InstructionSnapshot instruction, MemoryEventType type,
    std::vector<uint32_t> registers, uint64_t execMask, uint8_t byteMask, IntervalSet ldsIntervals,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations, VgprWriteOrder writeOrder,
    uint8_t lastRegisterByteMask, LdsAccess ldsAccess) {
  bool hasLds = !ldsIntervals.empty();
  // An event with no counter obligation is complete now, in its wave's
  // current barrier epoch.
  EventId eid = events_.add(waveId, instruction, type, std::move(registers), execMask, byteMask,
                            std::move(ldsIntervals), counterObligations, writeOrder,
                            lastRegisterByteMask, ldsAccess);
  if (hasLds)
    indexLdsIntervals(eid);
  return eid;
}

void RaceDetector::attachLdsIntervals(EventId eid, IntervalSet intervals) {
  events_.attachLdsIntervals(eid, std::move(intervals));
  indexLdsIntervals(eid);
}

void RaceDetector::indexLdsIntervals(EventId eid) {
  const auto &ivs = events_.ldsIntervals(eid);
  const WaveId orderedWave =
      events_.ldsOrder(eid) == MemoryOrderClass::LDS ? events_.waveId(eid) : WaveId{-1};
  // The LDS access is in flight until stampReleaseEpoch.
  assert(events_.releaseEpoch(eid) == 0);
  if (events_.writesLds(eid)) {
    ldsWriteEvents.push_back(eid);
    ++liveLdsWriteEvents_;
    adjustByteCounts(ivs, byteWriteCounts, +1, orderedWave);
  }
  if (events_.readsLds(eid)) {
    ldsReadEvents.push_back(eid);
    ++liveLdsReadEvents_;
    adjustByteCounts(ivs, byteReadCounts, +1, orderedWave);
  }
}

void RaceDetector::attachSrc2Source(EventId id, IntervalSet sourceReads) {
  if (sourceReads.empty())
    return;
  events_.markLdsSourceReads(id);
  const auto owner =
      events_.ldsOrder(id) == MemoryOrderClass::LDS ? events_.waveId(id) : WaveId{-1};
  adjustByteCounts(sourceReads, byteReadCounts, +1, owner);
  src2Events.push_back({id, std::move(sourceReads)});
}

void RaceDetector::markEventWaveComplete(EventId eventId) { events_.markComplete(eventId); }

void RaceDetector::stampReleaseEpoch(EventId eventId) {
  const WaveId owner = events_.waveId(eventId);
  events_.setReleaseEpoch(eventId, acquired(owner, owner));
  // The caller appends the event to its wave's history, where the other waves
  // find it until they acquire this epoch.
  if (!events_.hasLiveLdsIntervals(eventId))
    return;
  if (events_.writesLds(eventId))
    removeFromUnorderedList(ldsWriteEvents, eventId);
  if (events_.readsLds(eventId))
    removeFromUnorderedList(ldsReadEvents, eventId);
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
  slot->signalled = 0;
  slot->clock.assign(waveCount(), 0);
  ++openRounds_;
  // A named barrier's counted signals can be discarded by S_BARRIER_INIT, and
  // its round can complete while no live wave has joined it; neither is
  // reported. A newer generation shows that an older round which has not
  // completed never will.
  if (round.namedId != 0) {
    for (BarrierRoundState &older : rounds_)
      if (older.open && !older.completed && older.id.barrier == round.barrier &&
          older.id.namedId == round.namedId && older.id.generation < round.generation)
        closeRound(older);
  }
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
  state.signalled |= uint64_t{1} << wave.value;
  // Events the wave completes from now on wait for its next signal, and so do
  // the LDS operands it reads from now on.
  ++acquired(wave, wave);
  waveRaceStates[wave.value].forgetLdsOperandReads();
}

void RaceDetector::barrierRoundCompleted(BarrierRoundId round, std::span<const WaveId> armed,
                                         bool wholeWorkgroup) {
  BarrierRoundState &state = roundState(round);
  if (wholeWorkgroup) {
    // The round could not complete until each wave had signalled or ended, so
    // it follows everything an ended wave completed, unless that wave had
    // signalled the round: then the round could complete without its end, and
    // orders only what the signal released.
    if ((state.signalled & endedWaves_) == 0) {
      joinClock(state.clock, exitClock_);
    } else {
      for (int wave = 0; wave < waveCount(); ++wave) {
        const uint64_t bit = uint64_t{1} << wave;
        if ((endedWaves_ & bit) == 0 || (state.signalled & bit) != 0)
          continue;
        joinClock(state.clock, std::span(acquired_).subspan(static_cast<size_t>(wave) * waveCount(),
                                                            waveCount()));
      }
    }
  }
  state.completed = true;
  state.awaiting = 0;
  for (WaveId wave : armed)
    if (!ended_[wave.value])
      state.awaiting |= uint64_t{1} << wave.value;
  // The round set these waves' completion bit, so the bit was clear: a wave it
  // arms no longer waits for an older round of this barrier's bit. It passed
  // that wait, or S_BARRIER_JOIN, S_BARRIER_LEAVE or S_BARRIER_INIT cleared
  // the bit without one, which is not reported. Its next wait passes for this
  // round, so the older round's clock is not acquired. Waves this round did
  // not arm keep their older waits.
  if (state.awaiting != 0) {
    for (BarrierRoundState &older : rounds_) {
      if (&older == &state || !older.open || !older.completed ||
          older.id.barrier != round.barrier || (older.awaiting & state.awaiting) == 0)
        continue;
      older.awaiting &= ~state.awaiting;
      if (older.awaiting == 0)
        closeRound(older);
    }
  }
  if (state.awaiting == 0)
    closeRound(state);
}

void RaceDetector::barrierWaitPassed(WaveId wave, BarrierRoundId round) {
  refreshStaleCoverage();
  if (BarrierRoundState *state = findRound(round); state && state->completed)
    acquire(wave, state->clock);
  // The wave's completion bit is now clear. It never waits for another round
  // of this barrier that has already completed: one that completed while the
  // bit was set left the bit unchanged (CDNA5 ISA 5.6.1).
  stopAwaiting(wave, round.barrier);
}

void RaceDetector::waveEnded(WaveId wave) {
  assert(!ended_[wave.value]);
  ended_[wave.value] = true;
  endedWaves_ |= uint64_t{1} << wave.value;
  const auto row =
      std::span(acquired_).subspan(static_cast<size_t>(wave.value) * waveCount(), waveCount());
  joinClock(exitClock_, row);
  stopAwaiting(wave, std::nullopt);
  --liveWaves_;
  // The ended wave no longer holds back any other wave's retirement. Where it
  // was the last wave at an owner's coverage, the coverage is recomputed at
  // the next barrier wait (see staleCoverage_).
  for (int owner = 0; owner < waveCount(); ++owner) {
    if (owner == wave.value || staleCoverage_[owner] ||
        acquired(wave, WaveId{owner}) != covered_[owner])
      continue;
    if (--atCoverage_[owner] == 0) {
      staleCoverage_[owner] = true;
      hasStaleCoverage_ = true;
    }
  }
}

void RaceDetector::refreshStaleCoverage() {
  if (!hasStaleCoverage_)
    return;
  hasStaleCoverage_ = false;
  for (int owner = 0; owner < waveCount(); ++owner) {
    if (!staleCoverage_[owner])
      continue;
    staleCoverage_[owner] = false;
    recomputeCoverage(WaveId{owner});
  }
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
      leaveCoverageMinimum(WaveId{owner});
  }
}

void RaceDetector::leaveCoverageMinimum(WaveId owner) {
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
  // An event has a release epoch once its LDS access has completed.
  const uint32_t releaseEpoch = events_.releaseEpoch(eventId);
  return releaseEpoch != 0 && acquired(accessor, events_.waveId(eventId)) >= releaseEpoch;
}

bool RaceDetector::satisfyEventWaitCounter(EventId eventId, amdgpu::WaitCounterType waitCounter) {
  return events_.satisfyWaitCounter(eventId, waitCounter);
}

void RaceDetector::retireLdsIntervals(EventId eventId) {
  const auto src2 = std::ranges::find(src2Events, eventId, &Src2Event::id);
  if (src2 != src2Events.end()) {
    adjustByteCounts(src2->sourceReads, byteReadCounts, -1);
    std::iter_swap(src2, src2Events.end() - 1);
    src2Events.pop_back();
  }
  const auto &bytes = events_.ldsIntervals(eventId);
  if (!events_.hasLiveLdsIntervals(eventId))
    return;
  // An event whose LDS access completed is in its wave's history, which the
  // wave trims as it retires events.
  const bool inFlight = events_.releaseEpoch(eventId) == 0;
  if (events_.writesLds(eventId)) {
    if (inFlight)
      removeFromUnorderedList(ldsWriteEvents, eventId);
    --liveLdsWriteEvents_;
    adjustByteCounts(bytes, byteWriteCounts, -1);
  }
  if (events_.readsLds(eventId)) {
    if (inFlight)
      removeFromUnorderedList(ldsReadEvents, eventId);
    --liveLdsReadEvents_;
    adjustByteCounts(bytes, byteReadCounts, -1);
  }
  events_.markLdsPublished(eventId);
}

void RaceDetector::retireEvent(EventId eventId) {
  retireLdsIntervals(eventId);
  events_.markRetired(eventId);
}

template <typename Visit>
void RaceDetector::forEachUnacquiredLdsEvent(WaveId accessor, Visit visit) const {
  for (const WaveRaceState &owner : waveRaceStates) {
    if (owner.getWaveId() == accessor)
      continue;
    const uint32_t acquiredEpoch = acquired(accessor, owner.getWaveId());
    const std::vector<EventId> &history = owner.getBarrierPendingEvents();
    for (auto event = history.rbegin();
         event != history.rend() && events_.releaseEpoch(*event) > acquiredEpoch; ++event)
      visit(*event);
  }
}

void RaceDetector::validateRead(int addr, WaveId wave, int lane, int nBytes,
                                MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes) const {
  if (addr < 0 || nBytes <= 0 || !validLdsRange(addr, nBytes))
    return;
  if (!hasPotentialLdsConflict(addr, nBytes, byteWriteCounts, wave, currentMemoryOrder)) {
    return;
  }

  const auto check = [&](EventId eventId) {
    if (!events_.writesLds(eventId) || events_.atomicLdsElementMatches(eventId, atomicBytes))
      return;
    // Report a byte both accesses touch, not the start of this access.
    if (const auto byte = events_.ldsIntervals(eventId).firstOverlap(addr, addr + nBytes)) {
      raceHandler(
          {RaceViolation::Space::LDS, *byte, wave.value, lane, false, workgroupId, eventId});
    }
  };
  for (EventId eventId : ldsWriteEvents) {
    // Operations in the same non-UNORDERED class complete in issue order.
    if (wave == events_.waveId(eventId) &&
        (events_.ldsAccessComplete(eventId) || (currentMemoryOrder != MemoryOrderClass::UNORDERED &&
                                                events_.ldsOrder(eventId) == currentMemoryOrder)))
      continue;
    check(eventId);
  }
  forEachUnacquiredLdsEvent(wave, check);
}

void RaceDetector::validateWrite(int addr, WaveId wave, int lane, int nBytes,
                                 MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes) const {
  if (addr < 0 || nBytes <= 0 || !validLdsRange(addr, nBytes))
    return;
  if (!hasPotentialLdsConflict(addr, nBytes, byteReadCounts, wave, currentMemoryOrder)) {
    return;
  }

  for (const auto &source : src2Events) {
    const auto id = source.id;
    if (events_.waveId(id) == wave) {
      if (events_.ldsAccessComplete(id) || (currentMemoryOrder != MemoryOrderClass::UNORDERED &&
                                            events_.ldsOrder(id) == currentMemoryOrder))
        continue;
    } else if (ordered(wave, id)) {
      continue;
    }
    if (const auto byte = source.sourceReads.firstOverlap(addr, addr + nBytes)) {
      RaceViolation violation{
          RaceViolation::Space::LDS, *byte, wave.value, lane, true, workgroupId, id};
      violation.conflictsWithSrc2Read = true;
      raceHandler(violation);
    }
  }
  const auto check = [&](EventId eventId) {
    if (!events_.readsLds(eventId) || events_.atomicLdsElementMatches(eventId, atomicBytes))
      return;
    // Report a byte both accesses touch, not the start of this access.
    if (const auto byte = events_.ldsIntervals(eventId).firstOverlap(addr, addr + nBytes)) {
      raceHandler({RaceViolation::Space::LDS, *byte, wave.value, lane, true, workgroupId, eventId});
    }
  };
  for (EventId eventId : ldsReadEvents) {
    // The LDS bytes are safe only when the two operations share a proven
    // FIFO completion class, or an explicit wait completed the older event.
    // The read's destination VGPR remains independently protected by its
    // wait-counter obligations. This WAR check does not cover LDS
    // write/write ordering, which is not currently checked.
    if (wave == events_.waveId(eventId) &&
        (events_.ldsAccessComplete(eventId) || (currentMemoryOrder != MemoryOrderClass::UNORDERED &&
                                                events_.ldsOrder(eventId) == currentMemoryOrder)))
      continue;
    check(eventId);
  }
  forEachUnacquiredLdsEvent(wave, check);
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
