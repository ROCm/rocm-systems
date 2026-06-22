// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/types.h"
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu::plugins::race_detector {

/// Append-only event store with amortized prefix trimming.
///
/// Events are allocated with monotonically increasing IDs. Callers access
/// event data through typed accessors indexed by EventId. The registry
/// periodically trims a prefix of dead events to bound memory growth.
///
/// An event is trimmable when it can never be accessed again:
///   - RETIRED events (explicitly retired at s_barrier via retireEvent).
///   - WAVE_COMPLETE non-LDS events. These have left all per-wave and
///     per-register lists; retireEvent is a no-op for them, so they don't
///     need to wait for s_barrier.
///   - WAVE_COMPLETE LDS events are NOT trimmable until retired, because
///     retireEvent must still remove them from ldsWriteEvents/ldsReadEvents
///     and adjust byte counts.
///
/// Trimming is attempted every kTrimAttemptInterval allocations. The
/// trimmable prefix is scanned and erased only if it covers at least half
/// the entries, amortizing the O(n) vector shift.
class EventRegistry {
  struct EventInfo {
    WaveId waveId;
    uint64_t pc;
    MemoryEventType type;
    EventStatus status;
    uint8_t byteMask;
    uint64_t execMask;
    std::vector<uint32_t> registers;
    IntervalSet ldsIntervals;
  };

public:
  /// Allocate a new event. Returns a unique EventId.
  EventId add(WaveId waveId, uint64_t pc, MemoryEventType type, std::vector<uint32_t> registers,
              uint64_t execMask, uint8_t byteMask, IntervalSet ldsIntervals) {
    int id = base_offset_ + static_cast<int>(entries_.size());
    entries_.push_back({waveId, pc, type, EventStatus::ACTIVE, byteMask, execMask,
                        std::move(registers), std::move(ldsIntervals)});

    // To prevent the number of events recorded growing indefinitely, we try to
    // remove retired events from time to time.
    if (id % kTrimAttemptInterval == kTrimAttemptInterval - 1)
      tryTrimEvents();
    return EventId{id};
  }

  /// Transition ACTIVE → WAVE_COMPLETE (s_waitcnt resolved this event).
  void markComplete(EventId id) {
    assert(entries_[index(id)].status == EventStatus::ACTIVE);
    entries_[index(id)].status = EventStatus::WAVE_COMPLETE;
  }

  /// Transition → RETIRED (s_barrier flushed this event).
  void markRetired(EventId id) { entries_[index(id)].status = EventStatus::RETIRED; }

  // -- Typed accessors (all inline) --

  MemoryEventType type(EventId id) const { return entries_[index(id)].type; }
  EventStatus status(EventId id) const { return entries_[index(id)].status; }
  uint64_t pc(EventId id) const { return entries_[index(id)].pc; }
  uint8_t byteMask(EventId id) const { return entries_[index(id)].byteMask; }
  uint64_t execMask(EventId id) const { return entries_[index(id)].execMask; }
  WaveId waveId(EventId id) const { return entries_[index(id)].waveId; }

  std::span<const uint32_t> registers(EventId id) const { return entries_[index(id)].registers; }

  const IntervalSet &ldsIntervals(EventId id) const { return entries_[index(id)].ldsIntervals; }

  bool isActiveForLane(EventId id, int lane) const {
    return (entries_[index(id)].execMask >> lane) & 1;
  }

  // -- Test helpers --

  /// Number of events currently stored (live + not-yet-trimmed dead).
  int size() const { return static_cast<int>(entries_.size()); }

  /// Total events ever allocated (including trimmed).
  int totalAllocated() const { return base_offset_ + size(); }

  /// Number of events trimmed so far.
  int trimmedCount() const { return base_offset_; }

  static constexpr int kTrimAttemptInterval = 1000;

private:
  int index(EventId id) const { return id.value - base_offset_; }

  static bool isTrimmable(const EventInfo &e) {
    if (e.status == EventStatus::RETIRED)
      return true;
    return e.status == EventStatus::WAVE_COMPLETE && isWaveLocal(e.type);
  }

  void tryTrimEvents() {
    int trimCount = 0;
    int size = static_cast<int>(entries_.size());
    while (trimCount < size && isTrimmable(entries_[trimCount]))
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
