// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/event_registry.h"
#include "rocjitsu/vm/plugins/race_detector/core/types.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

/// Workgroup-level race detection state. Owns the event registry, live LDS
/// event lists, per-byte counters, and per-wave WaveRaceStates.
///
/// Event lifecycle:
///   1. allocateEventId() — registers a new event (ACTIVE).
///   2. markEventWaveComplete() — transitions to WAVE_COMPLETE after a wait or
///      ordered counter-capacity backpressure satisfies all of the event's
///      wait-counter obligations.
///   3. retireEvent() — removes from live lists, decrements byte counts,
///      marks RETIRED (for LDS events, called at s_barrier via
///      flushBarrierPendingEvents).
///
/// LDS race validation uses a two-level approach:
///   - Fast path: per-byte counters (byteWriteCounts / byteReadCounts)
///     provide O(1) checks.
///   - Slow path: when counts are non-zero, scans live event intervals
///     with binary search (IntervalSet::overlapsRange).
class RaceDetector {
  friend class Workgroup;

public:
  RaceDetector(int nWaves, int vgprCount, int sgprCount, Dim3d workgroupId,
               std::function<void(RaceViolation)> raceHandler, CounterCapacities);

  /// Allocate an event using the counter and ordering implied by its type.
  EventId allocateEventId(WaveId wave, uint64_t pc, MemoryEventType type,
                          std::vector<uint32_t> registers, uint64_t execMask) {
    return allocateEventId(wave, pc, type, std::move(registers), execMask, 0xF, {},
                           defaultWaitCounterType(type), defaultMemoryOrder(type));
  }

  /// Allocate a workgroup-global event ID and record its metadata.
  EventId
  allocateEventId(WaveId, uint64_t pc, MemoryEventType, std::vector<uint32_t> registers,
                  uint64_t execMask, uint8_t byteMask, IntervalSet ldsIntervals,
                  amdgpu::WaitCounterType waitCounterType, MemoryOrderClass memoryOrder,
                  std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt);

  /// Transition an event from ACTIVE to WAVE_COMPLETE.
  void markEventWaveComplete(EventId);

  /// Satisfy the event's obligations covered by one wait-counter update.
  /// Returns true once every obligation for the event has been satisfied.
  bool satisfyEventWaitCounter(EventId, amdgpu::WaitCounterType);

  /// Retire an event. For LDS-touching events, removes from live lists
  /// and decrements per-byte counts. All events are marked RETIRED in the
  /// registry, enabling prefix trimming.
  void retireEvent(EventId);

  /// Check for RAW hazards: no outstanding LDS writes overlap the range.
  /// Same-wave operations are ordered only when both belong to the same
  /// non-UNORDERED completion class.
  void validateRead(int addr, WaveId, int lane, int nBytes,
                    MemoryOrderClass currentMemoryOrder = MemoryOrderClass::LDS) const;

  /// Check for WAR hazards: no outstanding LDS reads overlap the range.
  /// Same-wave operations are ordered only when both belong to the same
  /// non-UNORDERED completion class.
  /// TODO(newling): WAW detection (write vs outstanding writes) is not
  /// implemented.
  void validateWrite(int addr, WaveId, int lane, int nBytes,
                     MemoryOrderClass currentMemoryOrder = MemoryOrderClass::LDS) const;

  const EventRegistry &events() const { return events_; }

  const std::vector<EventId> &getLdsWriteEvents() const { return ldsWriteEvents; }
  const std::vector<EventId> &getLdsReadEvents() const { return ldsReadEvents; }

  Dim3d getWorkgroupId() const { return workgroupId; }
  const std::function<void(RaceViolation)> &getRaceHandler() const { return raceHandler; }

  /// Format a RaceViolation with assembly context for diagnostics.
  /// getSourceLine(i) returns the original source text for line index i.
  std::string decorateException(const RaceViolation &, uint64_t wavePc, int numSourceLines,
                                std::function<std::string_view(int)> getSourceLine) const;

  WaveRaceState &getWaveRaceState(int waveIndex);

private:
  void setProfiler(ProfilerInterface &);

  static void adjustByteCounts(const IntervalSet &ivs, std::vector<int> &counts, int delta);

  /// Active LDS write events (for scanning during read validation).
  std::vector<EventId> ldsWriteEvents;

  /// Active LDS read events (for scanning during write validation).
  std::vector<EventId> ldsReadEvents;

  /// Outstanding write/read counts at chunk granularity for fast-path
  /// validation. Indexed by byte address / kCountGranularity. Conservative:
  /// may over-report conflicts for sub-chunk accesses. Vectors start empty
  /// and grow dynamically in adjustByteCounts as LDS events arrive, avoiding
  /// reliance on compiler metadata for the LDS size.
  static constexpr int kCountGranularity = 16;
  std::vector<int> byteWriteCounts;
  std::vector<int> byteReadCounts;

  EventRegistry events_;

  Dim3d workgroupId;
  std::function<void(RaceViolation)> raceHandler;

  /// One WaveRaceState per wave, indexed by wave ID.
  std::vector<WaveRaceState> waveRaceStates;
};

} // namespace rocjitsu::plugins::race_detector
