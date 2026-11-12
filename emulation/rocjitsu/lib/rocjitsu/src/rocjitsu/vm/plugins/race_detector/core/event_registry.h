// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/isa/instruction_snapshot.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/types.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

/// Append-only event store with amortized prefix trimming.
///
/// Events are allocated with monotonically increasing IDs. Callers access
/// event data through typed accessors indexed by EventId. The registry
/// periodically trims a prefix of RETIRED events. Callers must drop references
/// from wave queues and LDS conflict lists before the next event allocation.
///
/// Trimming is attempted every kTrimAttemptInterval allocations. The
/// trimmable prefix is scanned and erased only if it covers at least half
/// the entries, amortizing the O(n) vector shift.
class EventRegistry {
  struct EventInfo {
    WaveId waveId;
    InstructionSnapshot instruction;
    LdsAccess access;
    std::array<amdgpu::MemoryCounterObligation, amdgpu::MemoryIssueInfo::MAX_COUNTER_OBLIGATIONS>
        counterObligations;
    uint8_t numCounterObligations;
    uint8_t satisfiedCounterMask;
    MemoryOrderClass memoryOrder;
    EventStatus status;
    IntervalSet ldsIntervals;
    uint8_t alignedAtomicBytes;
  };

public:
  /// Allocate an event with its producing instruction and every counter obligation.
  EventId add(WaveId waveId, InstructionSnapshot instruction, LdsAccess access,
              IntervalSet ldsIntervals,
              std::span<const amdgpu::MemoryCounterObligation> counterObligations,
              MemoryOrderClass memoryOrder, uint8_t alignedAtomicBytes = 0) {
    int id = base_offset_ + static_cast<int>(entries_.size());
    assert(access != LdsAccess::None || ldsIntervals.empty());
    assert(!counterObligations.empty() || (access == LdsAccess::Read && !ldsIntervals.empty()));
    assert(counterObligations.size() <= amdgpu::MemoryIssueInfo::MAX_COUNTER_OBLIGATIONS);
    EventInfo event{waveId,
                    instruction,
                    access,
                    {},
                    static_cast<uint8_t>(counterObligations.size()),
                    0,
                    memoryOrder,
                    counterObligations.empty() ? EventStatus::WAVE_COMPLETE : EventStatus::ACTIVE,
                    std::move(ldsIntervals),
                    alignedAtomicBytes};
    std::ranges::copy(counterObligations, event.counterObligations.begin());
    entries_.push_back(std::move(event));

    // To prevent the number of events recorded growing indefinitely, we try to
    // remove retired events from time to time.
    if (id % kTrimAttemptInterval == kTrimAttemptInterval - 1)
      tryTrimEvents();
    return EventId{id};
  }

  /// Transition ACTIVE → WAVE_COMPLETE after every counter obligation is satisfied.
  void markComplete(EventId id) {
    assert(entries_[index(id)].status == EventStatus::ACTIVE);
    assert(allWaitCountersSatisfied(id));
    entries_[index(id)].status = EventStatus::WAVE_COMPLETE;
  }

  /// End the event's lifetime once neither counter progress nor peers need it.
  void markRetired(EventId id) { entries_[index(id)].status = EventStatus::RETIRED; }

  // -- Typed accessors (all inline) --

  LdsAccess access(EventId id) const { return entries_[index(id)].access; }
  bool hasPendingWaitCounter(EventId id, amdgpu::WaitCounterType waitType) const {
    const auto &event = entries_[index(id)];
    for (uint8_t i = 0; i < event.numCounterObligations; ++i) {
      if ((event.satisfiedCounterMask & (uint8_t{1} << i)) == 0 &&
          amdgpu::wait_counter_covers(waitType, event.counterObligations[i].wait_counter_type()))
        return true;
    }
    return false;
  }
  std::optional<MemoryOrderClass> pendingCompletionClass(EventId id,
                                                         amdgpu::WaitCounterType waitType) const {
    const auto &event = entries_[index(id)];
    for (uint8_t i = 0; i < event.numCounterObligations; ++i) {
      if ((event.satisfiedCounterMask & (uint8_t{1} << i)) == 0 &&
          amdgpu::wait_counter_covers(waitType, event.counterObligations[i].wait_counter_type()))
        return event.counterObligations[i].completion_class();
    }
    return std::nullopt;
  }
  uint8_t pendingCounterIncrement(EventId id, amdgpu::WaitCounterType waitType) const {
    const auto &event = entries_[index(id)];
    for (uint8_t i = 0; i < event.numCounterObligations; ++i) {
      if ((event.satisfiedCounterMask & (uint8_t{1} << i)) == 0 &&
          amdgpu::wait_counter_covers(waitType, event.counterObligations[i].wait_counter_type()))
        return event.counterObligations[i].counter_increment();
    }
    return 0;
  }
  bool satisfyWaitCounter(EventId id, amdgpu::WaitCounterType waitType) {
    auto &event = entries_[index(id)];
    assert(event.status == EventStatus::ACTIVE);
    for (uint8_t i = 0; i < event.numCounterObligations; ++i) {
      if (amdgpu::wait_counter_covers(waitType, event.counterObligations[i].wait_counter_type()))
        event.satisfiedCounterMask |= uint8_t{1} << i;
    }
    return allWaitCountersSatisfied(id);
  }
  bool allWaitCountersSatisfied(EventId id) const {
    const auto &event = entries_[index(id)];
    return event.satisfiedCounterMask ==
           static_cast<uint8_t>((uint8_t{1} << event.numCounterObligations) - 1);
  }
  bool atomicLdsElementMatches(EventId id, uint8_t bytes) const {
    return bytes != 0 && entries_[index(id)].alignedAtomicBytes == bytes;
  }
  bool readsLds(EventId id) const {
    return access(id) == LdsAccess::Read || access(id) == LdsAccess::ReadWrite;
  }
  bool writesLds(EventId id) const {
    return access(id) == LdsAccess::Write || access(id) == LdsAccess::ReadWrite;
  }
  MemoryOrderClass memoryOrder(EventId id) const { return entries_[index(id)].memoryOrder; }
  EventStatus status(EventId id) const { return entries_[index(id)].status; }
  const InstructionSnapshot &instruction(EventId id) const {
    return entries_[index(id)].instruction;
  }
  WaveId waveId(EventId id) const { return entries_[index(id)].waveId; }

  const IntervalSet &ldsIntervals(EventId id) const { return entries_[index(id)].ldsIntervals; }

  // -- Test helpers --

  /// Number of events currently stored (live + not-yet-trimmed dead).
  int size() const { return static_cast<int>(entries_.size()); }

  /// Total events ever allocated (including trimmed).
  int totalAllocated() const { return base_offset_ + size(); }

  /// Whether @p id names an event still present in this registry.
  bool contains(EventId id) const {
    return id.isValid() && id.value >= base_offset_ && id.value < totalAllocated();
  }

  /// Number of events trimmed so far.
  int trimmedCount() const { return base_offset_; }

  static constexpr int kTrimAttemptInterval = 1000;

private:
  int index(EventId id) const { return id.value - base_offset_; }

  void tryTrimEvents() {
    int trimCount = 0;
    int size = static_cast<int>(entries_.size());
    while (trimCount < size && entries_[trimCount].status == EventStatus::RETIRED)
      trimCount++;

    if (trimCount < size / 2)
      return;

    entries_.erase(entries_.begin(), entries_.begin() + trimCount);
    base_offset_ += trimCount;
  }

  std::vector<EventInfo> entries_;
  int base_offset_ = 0;
};

} // namespace rocjitsu::plugins::race_detector
