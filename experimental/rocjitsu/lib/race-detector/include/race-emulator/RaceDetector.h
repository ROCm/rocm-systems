// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "CommonRegister.h"
#include "IntervalSet.h"
#include "Types.h"
#include "WaveRaceState.h"
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

/// Workgroup-level race detection state. Owns the event registry, live LDS
/// event lists, per-byte counters, and per-wave WaveRaceStates.
///
/// Event lifecycle:
///   1. allocateEventId() — registers a new event (ACTIVE).
///   2. markEventWaveComplete() — transitions to WAVE_COMPLETE (s_waitcnt).
///   3. retireEvent() — removes from live lists, decrements byte counts
///      (s_barrier).
///
/// LDS race validation uses a two-level approach:
///   - Fast path: per-byte counters (byteWriteCounts / byteReadCounts)
///     provide O(1) checks.
///   - Slow path: when counts are non-zero, scans live event intervals
///     with binary search (IntervalSet::overlapsRange).
class RaceDetector {
  friend class Workgroup;

  /// Per-event metadata stored in the event registry.
  struct EventInfo {
    WaveId waveId;
    int pc;
    MemoryEventType type;
    EventStatus status;
    uint8_t byteMask;
    uint64_t execMask;
    std::vector<uint32_t> registers;
    IntervalSet ldsIntervals;
  };

public:
  RaceDetector(int ldsSize, int nWaves, int vgprCount, int sgprCount,
               Dim3d workgroupId,
               std::function<void(RaceViolation)> raceHandler);

  /// Allocate a workgroup-global event ID and record its metadata.
  EventId allocateEventId(WaveId waveId, int pc, MemoryEventType type,
                          std::vector<uint32_t> registers, uint64_t execMask,
                          uint8_t byteMask = 0xF,
                          IntervalSet ldsIntervals = {});

  /// Transition an event from ACTIVE to WAVE_COMPLETE.
  void markEventWaveComplete(EventId);

  /// Retire an event at s_barrier. For LDS-touching events, removes from
  /// live lists and decrements per-byte counts. Non-LDS events are no-ops.
  void retireEvent(EventId);

  /// Check for RAW hazards: no outstanding LDS writes overlap the range.
  void validateRead(int addr, WaveId, int lane, int nBytes) const;

  /// Check for WAR hazards: no outstanding LDS reads overlap the range.
  void validateWrite(int addr, WaveId, int lane, int nBytes) const;

  // --- Event accessors ---

  const IntervalSet &getEventIntervals(EventId eventId) const {
    return eventRegistry[eventId.value].ldsIntervals;
  }

  WaveId getEventWaveId(EventId eventId) const {
    return eventRegistry[eventId.value].waveId;
  }

  MemoryEventType getEventType(EventId eventId) const {
    return eventRegistry[eventId.value].type;
  }

  EventStatus getEventStatus(EventId eventId) const {
    return eventRegistry[eventId.value].status;
  }

  uint8_t getEventByteMask(EventId eventId) const {
    return eventRegistry[eventId.value].byteMask;
  }

  int getEventPc(EventId eventId) const {
    return eventRegistry[eventId.value].pc;
  }

  uint64_t getEventExecMask(EventId eventId) const {
    return eventRegistry[eventId.value].execMask;
  }

  static inline bool isLaneActive(uint64_t execMask, int lane) {
    return (execMask >> lane) & 1;
  }

  bool isEventActiveForLane(EventId eventId, int lane) const {
    return isLaneActive(eventRegistry[eventId.value].execMask, lane);
  }

  std::span<const uint32_t> getEventRegisters(EventId eventId) const {
    return eventRegistry[eventId.value].registers;
  }

  const std::vector<EventId> &getLdsWriteEvents() const {
    return ldsWriteEvents;
  }
  const std::vector<EventId> &getLdsReadEvents() const { return ldsReadEvents; }

  Dim3d getWorkgroupId() const { return workgroupId; }
  const std::function<void(RaceViolation)> &getRaceHandler() const {
    return raceHandler;
  }

  /// Format a RaceViolation with assembly context for diagnostics.
  /// getSourceLine(i) returns the original source text for line index i.
  std::string
  decorateException(const RaceViolation &v, int wavePc,
                    WaveRaceState *waveRaceState, int numSourceLines,
                    std::function<std::string_view(int)> getSourceLine) const;

  WaveRaceState &getWaveRaceState(int waveIndex);

private:
  void setProfiler(ProfilerInterface &p);

  static void adjustByteCounts(const IntervalSet &ivs, std::vector<int> &counts,
                               int delta);

  /// Active LDS write events (for scanning during read validation).
  std::vector<EventId> ldsWriteEvents;

  /// Active LDS read events (for scanning during write validation).
  std::vector<EventId> ldsReadEvents;

  /// Per-LDS-byte outstanding write/read counts for fast-path validation.
  std::vector<int> byteWriteCounts;
  std::vector<int> byteReadCounts;

  /// All events ever allocated, indexed by EventId::value.
  std::vector<EventInfo> eventRegistry;

  Dim3d workgroupId;
  std::function<void(RaceViolation)> raceHandler;

  /// One WaveRaceState per wave, indexed by wave ID.
  std::vector<WaveRaceState> waveRaceStates;
};

} // namespace raceemulator
