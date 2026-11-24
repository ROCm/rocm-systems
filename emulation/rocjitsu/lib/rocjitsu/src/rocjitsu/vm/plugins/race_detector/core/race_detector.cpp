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
#include <optional>
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
  if (nWaves > kMaxWaves) {
    throw std::length_error("race detector supports at most " + std::to_string(kMaxWaves) +
                            " waves in a workgroup, got " + std::to_string(nWaves));
  }
  waveRaceStates.reserve(nWaves);
  for (int i = 0; i < nWaves; ++i) {
    waveRaceStates.emplace_back(vgprCount, sgprCount, WaveId{i}, this, counterCapacities);
  }
  acquired_.assign(static_cast<size_t>(nWaves) * clockSize(), 0);
  // Each wave starts in epoch 1, and in handoff epoch 1 of each kind, so 0
  // means nothing acquired.
  for (int i = 0; i < nWaves; ++i) {
    const WaveId wave{i};
    acquired(wave, wave) = 1;
    for (int kind = 0; kind < kArrivalKinds; ++kind) {
      acquiredRow(wave)[handoffIndex(static_cast<BarrierCellArrival>(kind), wave)] = 1;
    }
  }
  exitClock_.assign(clockSize(), 0);
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
  if (hasLds) {
    indexLdsIntervals(eid);
  }
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
  attachLdsRead(id, std::move(sourceReads), /*barrierArrive=*/false);
}

void RaceDetector::attachBarrierArrive(EventId transfer, uint32_t cell) {
  IntervalSet bytes;
  bytes.append(static_cast<int>(cell), static_cast<int>(cell) + kBarrierCellBytes);
  bytes.finalize();
  attachLdsRead(transfer, std::move(bytes), /*barrierArrive=*/true);
}

void RaceDetector::attachLdsRead(EventId id, IntervalSet bytes, bool barrierArrive) {
  if (bytes.empty()) {
    return;
  }
  events_.markLdsSourceReads(id);
  const auto owner =
      events_.ldsOrder(id) == MemoryOrderClass::LDS ? events_.waveId(id) : WaveId{-1};
  adjustByteCounts(bytes, byteReadCounts, +1, owner);
  attachedLdsReads.push_back({id, std::move(bytes), barrierArrive});
}

void RaceDetector::markEventWaveComplete(EventId eventId) { events_.markComplete(eventId); }

void RaceDetector::stampReleaseEpoch(EventId eventId) {
  const WaveId owner = events_.waveId(eventId);
  events_.setReleaseEpoch(eventId, acquired(owner, owner));
  // The caller appends the event to its wave's history, where the other waves
  // find it until they acquire this epoch.
  if (!events_.hasLiveLdsIntervals(eventId)) {
    return;
  }
  if (events_.writesLds(eventId)) {
    removeFromUnorderedList(ldsWriteEvents, eventId);
  }
  if (events_.readsLds(eventId)) {
    removeFromUnorderedList(ldsReadEvents, eventId);
  }
}

namespace {

/// Join `from` into `into`, element-wise maximum.
void joinClock(std::span<uint32_t> into, std::span<const uint32_t> from) {
  assert(into.size() == from.size());
  for (size_t index = 0; index < into.size(); ++index) {
    into[index] = std::max(into[index], from[index]);
  }
}

} // namespace

void RaceDetector::joinReleased(std::span<uint32_t> clock, WaveId wave,
                                std::optional<BarrierCellArrival> arrival) const {
  // Hold back the wave's own handoff epochs, except that of `arrival`.
  std::array<uint32_t, kArrivalKinds> held;
  for (int kind = 0; kind < kArrivalKinds; ++kind) {
    held[kind] = clock[handoffIndex(static_cast<BarrierCellArrival>(kind), wave)];
  }
  joinClock(clock, acquiredRow(wave));
  for (int kind = 0; kind < kArrivalKinds; ++kind) {
    if (static_cast<BarrierCellArrival>(kind) != arrival) {
      clock[handoffIndex(static_cast<BarrierCellArrival>(kind), wave)] = held[kind];
    }
  }
}

RaceDetector::BarrierRoundState *RaceDetector::findRound(BarrierRoundId round) {
  for (BarrierRoundState &state : rounds_) {
    if (state.open && state.id == round) {
      return &state;
    }
  }
  return nullptr;
}

RaceDetector::BarrierRoundState &RaceDetector::roundState(BarrierRoundId round) {
  if (BarrierRoundState *state = findRound(round)) {
    return *state;
  }
  // Reuse a closed slot, and its clock's storage, before adding one.
  auto slot = std::ranges::find_if(rounds_, [](const BarrierRoundState &s) { return !s.open; });
  if (slot == rounds_.end()) {
    slot = rounds_.insert(rounds_.end(), BarrierRoundState{});
  }
  slot->id = round;
  slot->open = true;
  slot->completed = false;
  slot->awaiting = 0;
  slot->signalled = 0;
  slot->clock.assign(clockSize(), 0);
  ++openRounds_;
  // A named barrier's counted signals can be discarded by S_BARRIER_INIT, and
  // its round can complete while no live wave has joined it; neither is
  // reported. A newer generation shows that an older round which has not
  // completed never will.
  if (round.namedId != 0) {
    for (BarrierRoundState &older : rounds_) {
      if (older.open && !older.completed && older.id.barrier == round.barrier &&
          older.id.namedId == round.namedId && older.id.generation < round.generation) {
        closeRound(older);
      }
    }
  }
  return *slot;
}

void RaceDetector::closeRound(BarrierRoundState &state) {
  state.open = false;
  --openRounds_;
}

void RaceDetector::barrierSignaled(WaveId wave, BarrierRoundId round) {
  BarrierRoundState &state = roundState(round);
  joinReleased(state.clock, wave);
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
        if ((endedWaves_ & bit) == 0 || (state.signalled & bit) != 0) {
          continue;
        }
        joinReleased(state.clock, WaveId{wave});
      }
    }
  }
  state.completed = true;
  state.awaiting = 0;
  for (WaveId wave : armed) {
    if (!ended_[wave.value]) {
      state.awaiting |= uint64_t{1} << wave.value;
    }
  }
  // The round set these waves' completion bit, so the bit was clear: a wave it
  // arms no longer waits for an older round of this barrier's bit. It passed
  // that wait, or S_BARRIER_JOIN or S_BARRIER_LEAVE cleared
  // the bit without one, which is not reported. Its next wait passes for this
  // round, so the older round's clock is not acquired. Waves this round did
  // not arm keep their older waits.
  if (state.awaiting != 0) {
    for (BarrierRoundState &older : rounds_) {
      if (&older == &state || !older.open || !older.completed ||
          older.id.barrier != round.barrier || (older.awaiting & state.awaiting) == 0) {
        continue;
      }
      older.awaiting &= ~state.awaiting;
      if (older.awaiting == 0) {
        closeRound(older);
      }
    }
  }
  if (state.awaiting == 0) {
    closeRound(state);
  }
}

void RaceDetector::barrierWaitPassed(WaveId wave, BarrierRoundId round) {
  refreshStaleCoverage();
  if (BarrierRoundState *state = findRound(round); state && state->completed) {
    acquire(wave, state->clock);
  }
  // The wave's completion bit is now clear. It never waits for another round
  // of this barrier that has already completed: one that completed while the
  // bit was set left the bit unchanged (CDNA5 ISA 5.6.1).
  stopAwaiting(wave, round.barrier);
}

void RaceDetector::waveEnded(WaveId wave) {
  assert(!ended_[wave.value]);
  ended_[wave.value] = true;
  endedWaves_ |= uint64_t{1} << wave.value;
  joinReleased(exitClock_, wave);
  stopAwaiting(wave, std::nullopt);
  --liveWaves_;
  // The ended wave no longer holds back any other wave's retirement. Where it
  // was the last wave at an owner's coverage, the coverage is recomputed at
  // the next barrier wait or read of an LDS barrier cell (see staleCoverage_).
  for (int owner = 0; owner < waveCount(); ++owner) {
    if (owner == wave.value || staleCoverage_[owner] ||
        acquired(wave, WaveId{owner}) != covered_[owner]) {
      continue;
    }
    if (--atCoverage_[owner] == 0) {
      staleCoverage_[owner] = true;
      hasStaleCoverage_ = true;
    }
  }
}

void RaceDetector::refreshStaleCoverage() {
  if (!hasStaleCoverage_) {
    return;
  }
  hasStaleCoverage_ = false;
  for (int owner = 0; owner < waveCount(); ++owner) {
    if (!staleCoverage_[owner]) {
      continue;
    }
    staleCoverage_[owner] = false;
    recomputeCoverage(WaveId{owner});
  }
}

void RaceDetector::stopAwaiting(WaveId wave, std::optional<uint8_t> barrier) {
  const uint64_t bit = uint64_t{1} << wave.value;
  for (BarrierRoundState &state : rounds_) {
    if (!state.open || !state.completed || (barrier && state.id.barrier != *barrier) ||
        (state.awaiting & bit) == 0) {
      continue;
    }
    state.awaiting &= ~bit;
    if (state.awaiting == 0) {
      closeRound(state);
    }
  }
}

void RaceDetector::acquire(WaveId accessor, std::span<const uint32_t> clock) {
  for (int owner = 0; owner < waveCount(); ++owner) {
    uint32_t &entry = acquired(accessor, WaveId{owner});
    if (clock[owner] <= entry) {
      continue;
    }
    const bool wasAtCoverage = owner != accessor.value && entry == covered_[owner];
    entry = clock[owner];
    if (wasAtCoverage) {
      leaveCoverageMinimum(WaveId{owner});
    }
  }
  // Handoff epochs order events but retire none, so they have no coverage.
  const auto row = acquiredRow(accessor);
  for (int index = waveCount(); index < clockSize(); ++index) {
    row[index] = std::max(row[index], clock[index]);
  }
}

void RaceDetector::leaveCoverageMinimum(WaveId owner) {
  // Coverage can rise only when the last wave at the minimum leaves it.
  if (--atCoverage_[owner.value] == 0) {
    recomputeCoverage(owner);
  }
}

void RaceDetector::recomputeCoverage(WaveId owner) {
  uint32_t covered = std::numeric_limits<uint32_t>::max();
  int atCoverage = 0;
  for (int accessor = 0; accessor < waveCount(); ++accessor) {
    if (accessor == owner.value || ended_[accessor]) {
      continue;
    }
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

bool RaceDetector::handedOffTo(WaveId accessor, EventId eventId) const {
  const WaveId owner = events_.waveId(eventId);
  const std::optional<BarrierCellArrival> kind = events_.handoffKind(eventId);
  return kind && owner != accessor &&
         acquiredRow(accessor)[handoffIndex(*kind, owner)] >= events_.handoffEpoch(eventId);
}

RaceDetector::BarrierCell &RaceDetector::barrierCell(uint32_t address) {
  const auto cell = std::ranges::find(cells_, address, &BarrierCell::address);
  if (cell != cells_.end()) {
    return *cell;
  }
  return cells_.emplace_back(BarrierCell{.address = address,
                                         .open = std::vector<uint32_t>(clockSize(), 0),
                                         .done = std::vector<uint32_t>(clockSize(), 0)});
}

void RaceDetector::barrierCellArrived(WaveId wave, uint32_t address, BarrierCellArrival kind,
                                      bool completed) {
  // With no other wave there is nothing to order, and events retire when
  // they complete.
  if (waveCount() == 1) {
    return;
  }
  uint32_t &handoffEpoch = acquiredRow(wave)[handoffIndex(kind, wave)];
  waveRaceStates[wave.value].handOffPending(kind, handoffEpoch);
  BarrierCell &cell = barrierCell(address);
  joinReleased(cell.open, wave, kind);
  // Events the wave completes from now on wait for its next release, and so
  // do the LDS operands it reads from now on. Accesses it issues from now on
  // wait for its next arrival of this kind.
  ++acquired(wave, wave);
  ++handoffEpoch;
  waveRaceStates[wave.value].forgetLdsOperandReads();
  if (completed) {
    joinClock(cell.done, cell.open);
    std::ranges::fill(cell.open, 0);
  }
  if (kind == BarrierCellArrival::Ds) {
    refreshStaleCoverage();
    acquire(wave, cell.done);
  }
}

// A cell's address fits an int: the plugin reports only cells inside the LDS
// backing, whose byte addresses the interval sets index as int.
void RaceDetector::acquireBarrierCells(WaveId wave, const IntervalSet &accessed) {
  // Indexed: acquiring retires events, which never adds a cell.
  for (size_t index = 0; index < cells_.size(); ++index) {
    const int address = static_cast<int>(cells_[index].address);
    if (!accessed.overlapsRange(address, address + kBarrierCellStateBytes)) {
      continue;
    }
    refreshStaleCoverage();
    acquire(wave, cells_[index].done);
  }
}

bool RaceDetector::satisfyEventWaitCounter(EventId eventId, amdgpu::WaitCounterType waitCounter) {
  return events_.satisfyWaitCounter(eventId, waitCounter);
}

void RaceDetector::retireLdsIntervals(EventId eventId) {
  // An event can have both a source operand and a barrier cell attached.
  for (size_t index = 0; index < attachedLdsReads.size();) {
    if (attachedLdsReads[index].id != eventId) {
      ++index;
      continue;
    }
    adjustByteCounts(attachedLdsReads[index].bytes, byteReadCounts, -1);
    std::swap(attachedLdsReads[index], attachedLdsReads.back());
    attachedLdsReads.pop_back();
  }
  const auto &bytes = events_.ldsIntervals(eventId);
  if (!events_.hasLiveLdsIntervals(eventId)) {
    return;
  }
  // An event whose LDS access completed is in its wave's history, which the
  // wave trims as it retires events.
  const bool inFlight = events_.releaseEpoch(eventId) == 0;
  if (events_.writesLds(eventId)) {
    if (inFlight) {
      removeFromUnorderedList(ldsWriteEvents, eventId);
    }
    --liveLdsWriteEvents_;
    adjustByteCounts(bytes, byteWriteCounts, -1);
  }
  if (events_.readsLds(eventId)) {
    if (inFlight) {
      removeFromUnorderedList(ldsReadEvents, eventId);
    }
    --liveLdsReadEvents_;
    adjustByteCounts(bytes, byteReadCounts, -1);
  }
  events_.markLdsPublished(eventId);
}

void RaceDetector::retireEvent(EventId eventId) {
  retireLdsIntervals(eventId);
  events_.markRetired(eventId);
}

template <typename Matches>
bool RaceDetector::anyCheckedLdsEvent(WaveId accessor, bool isWrite,
                                      MemoryOrderClass currentMemoryOrder, int start, int end,
                                      Matches matches) const {
  for (EventId eventId : isWrite ? ldsReadEvents : ldsWriteEvents) {
    // The LDS bytes are safe only when the two operations share a proven FIFO
    // completion class: operations in the same non-UNORDERED class complete in
    // issue order. An explicit wait that completed the older access also makes
    // them safe. A read's destination VGPR remains independently protected by
    // its wait-counter obligations. LDS write/write ordering is not checked.
    if (accessor == events_.waveId(eventId) &&
        (events_.ldsAccessComplete(eventId) || (currentMemoryOrder != MemoryOrderClass::UNORDERED &&
                                                events_.ldsOrder(eventId) == currentMemoryOrder))) {
      continue;
    }
    if (matches(eventId)) {
      return true;
    }
  }
  for (const WaveRaceState &owner : waveRaceStates) {
    if (owner.getWaveId() == accessor || !owner.barrierPendingMayOverlap(start, end)) {
      continue;
    }
    const uint32_t acquiredEpoch = acquired(accessor, owner.getWaveId());
    const std::vector<EventId> &history = owner.getBarrierPendingEvents();
    for (auto event = history.rbegin();
         event != history.rend() && events_.releaseEpoch(*event) > acquiredEpoch; ++event) {
      if (matches(*event)) {
        return true;
      }
    }
  }
  return false;
}

template <typename Matches>
bool RaceDetector::anyCheckedAttachedRead(WaveId accessor, MemoryOrderClass currentMemoryOrder,
                                          Matches matches) const {
  for (const AttachedLdsRead &source : attachedLdsReads) {
    const EventId id = source.id;
    if (events_.waveId(id) == accessor) {
      if (events_.ldsAccessComplete(id) || (currentMemoryOrder != MemoryOrderClass::UNORDERED &&
                                            events_.ldsOrder(id) == currentMemoryOrder)) {
        continue;
      }
    } else if (ordered(accessor, id) || handedOffTo(accessor, id)) {
      continue;
    }
    if (matches(source)) {
      return true;
    }
  }
  return false;
}

void RaceDetector::validateRead(int addr, WaveId wave, int lane, int nBytes,
                                MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes) const {
  if (addr < 0 || nBytes <= 0 || !validLdsRange(addr, nBytes)) {
    return;
  }
  if (!hasPotentialLdsConflict(addr, nBytes, byteWriteCounts, wave, currentMemoryOrder)) {
    return;
  }

  anyCheckedLdsEvent(
      wave, /*isWrite=*/false, currentMemoryOrder, addr, addr + nBytes, [&](EventId eventId) {
        if (!events_.writesLds(eventId) || events_.atomicLdsElementMatches(eventId, atomicBytes)) {
          return false;
        }
        // Report a byte both accesses touch, not the start of this access.
        if (const auto byte = events_.ldsIntervals(eventId).firstOverlap(addr, addr + nBytes);
            byte && !handedOffTo(wave, eventId)) {
          raceHandler(
              {RaceViolation::Space::LDS, *byte, wave.value, lane, false, workgroupId, eventId});
        }
        // Report every conflicting event.
        return false;
      });
}

void RaceDetector::validateWrite(int addr, WaveId wave, int lane, int nBytes,
                                 MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes) const {
  if (addr < 0 || nBytes <= 0 || !validLdsRange(addr, nBytes)) {
    return;
  }
  if (!hasPotentialLdsConflict(addr, nBytes, byteReadCounts, wave, currentMemoryOrder)) {
    return;
  }

  anyCheckedAttachedRead(wave, currentMemoryOrder, [&](const AttachedLdsRead &source) {
    // An atomic of the cell's width cannot interleave with an arrival.
    if (source.barrierArrive && atomicBytes == kBarrierCellBytes) {
      return false;
    }
    if (const auto byte = source.bytes.firstOverlap(addr, addr + nBytes)) {
      RaceViolation violation{
          RaceViolation::Space::LDS, *byte, wave.value, lane, true, workgroupId, source.id};
      (source.barrierArrive ? violation.conflictsWithBarrierArrive
                            : violation.conflictsWithSrc2Read) = true;
      raceHandler(violation);
    }
    // Report every conflicting event.
    return false;
  });
  anyCheckedLdsEvent(
      wave, /*isWrite=*/true, currentMemoryOrder, addr, addr + nBytes, [&](EventId eventId) {
        if (!events_.readsLds(eventId) || events_.atomicLdsElementMatches(eventId, atomicBytes)) {
          return false;
        }
        // Report a byte both accesses touch, not the start of this access.
        if (const auto byte = events_.ldsIntervals(eventId).firstOverlap(addr, addr + nBytes);
            byte && !handedOffTo(wave, eventId)) {
          raceHandler(
              {RaceViolation::Space::LDS, *byte, wave.value, lane, true, workgroupId, eventId});
        }
        // Report every conflicting event.
        return false;
      });
}

bool RaceDetector::mayConflict(const IntervalSet &accessed, WaveId wave, bool isWrite,
                               MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes) const {
  // Fast path: no live event has bytes in any 16-byte chunk the access touches.
  const std::vector<LdsChunkCounts> &counts = isWrite ? byteReadCounts : byteWriteCounts;
  if (!std::ranges::any_of(accessed, [&](const Interval &interval) {
        return hasPotentialLdsConflict(interval.start, interval.end - interval.start, counts, wave,
                                       currentMemoryOrder);
      })) {
    return false;
  }
  // Scan the events once for the whole instruction, by the rules the per-lane
  // checks apply, so a clean access does not repeat the scan for each lane.
  if (isWrite &&
      anyCheckedAttachedRead(wave, currentMemoryOrder, [&](const AttachedLdsRead &source) {
        return !(source.barrierArrive && atomicBytes == kBarrierCellBytes) &&
               std::ranges::any_of(accessed, [&](const Interval &interval) {
                 return source.bytes.overlapsRange(interval.start, interval.end);
               });
      })) {
    return true;
  }
  // The fast path found a chunk with live bytes, so `accessed` is not empty.
  const int start = accessed.begin()->start;
  const int end = (accessed.end() - 1)->end;
  return anyCheckedLdsEvent(wave, isWrite, currentMemoryOrder, start, end, [&](EventId eventId) {
    if (!(isWrite ? events_.readsLds(eventId) : events_.writesLds(eventId)) ||
        events_.atomicLdsElementMatches(eventId, atomicBytes)) {
      return false;
    }
    const IntervalSet &bytes = events_.ldsIntervals(eventId);
    return std::ranges::any_of(accessed,
                               [&](const Interval &interval) {
                                 return bytes.overlapsRange(interval.start, interval.end);
                               }) &&
           !handedOffTo(wave, eventId);
  });
}

bool RaceDetector::hasPotentialLdsConflict(int addr, int nBytes,
                                           const std::vector<LdsChunkCounts> &counts, WaveId wave,
                                           MemoryOrderClass currentMemoryOrder) {
  if (addr < 0 || nBytes <= 0 || nBytes > std::numeric_limits<int>::max() - addr) {
    return false;
  }
  const int limit = static_cast<int>(counts.size());
  const int cStart = addr / kCountGranularity;
  const int cEnd = (addr + nBytes - 1) / kCountGranularity + 1;
  // Every overlapping event contributes to a touched chunk. A matching owner
  // proves that the full scan would skip it by the same-wave LDS ordering rule.
  for (int c = cStart; c < cEnd && c < limit; ++c) {
    if (counts[c].bytes > 0 &&
        (currentMemoryOrder != MemoryOrderClass::LDS || counts[c].orderedWave != wave)) {
      return true;
    }
  }
  return false;
}

void RaceDetector::adjustByteCounts(const IntervalSet &ivs, std::vector<LdsChunkCounts> &counts,
                                    int delta, WaveId orderedWave) {
  for (const auto &iv : ivs) {
    if (iv.start < 0 || iv.end <= iv.start ||
        !validLdsRange(iv.start, static_cast<uint64_t>(iv.end) - iv.start)) {
      continue;
    }
    int cStart = iv.start / kCountGranularity;
    int cEnd = (iv.end - 1) / kCountGranularity + 1;
    if (cEnd > static_cast<int>(counts.size())) {
      counts.resize(cEnd);
    }
    for (int c = cStart; c < cEnd; ++c) {
      int byteStart = std::max(iv.start, c * kCountGranularity);
      int nBytes = std::min(iv.end - byteStart, kCountGranularity - byteStart % kCountGranularity);
      auto &chunk = counts[c];
      if (delta > 0) {
        if (chunk.bytes == 0) {
          chunk.orderedWave = orderedWave;
        } else if (chunk.orderedWave != orderedWave) {
          chunk.orderedWave = WaveId{-1};
        }
      }
      chunk.bytes += delta * nBytes;
    }
  }
}

} // namespace rocjitsu::plugins::race_detector
