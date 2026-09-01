// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/dim3d.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace rocjitsu::plugins::race_detector {

/// Identifies a wave (SIMD execution unit) within a workgroup. Wave 0 runs
/// lanes [0, waveSize), wave 1 runs [waveSize, 2*waveSize), and so on.
/// Strongly typed to prevent accidental mixing with lane indices, register
/// indices, or event IDs.
struct WaveId {
  int value;
  bool operator==(WaveId o) const { return value == o.value; }
  bool operator!=(WaveId o) const { return value != o.value; }
  bool operator<(WaveId o) const { return value < o.value; }
};

/// Number of lanes per wave (32 for RDNA, 64 for CDNA).
/// Implicitly convertible to int for use in arithmetic and loop bounds.
struct WaveSize {
  int value;
  operator int() const { return value; }
};

/// Identifies a memory event within a workgroup. Each memory instruction
/// (LDS read/write, global load/store) creates an event via
/// RaceDetector::allocateEventId(). The event ID is used to track the event
/// through its lifecycle: registration, wave-local completion, and barrier
/// retirement. Strongly typed to prevent accidental mixing with wave IDs,
/// register indices, or byte addresses.
struct EventId {
  static constexpr int kInvalidValue = -1;

  int value = kInvalidValue;

  bool isValid() const { return value != kInvalidValue; }
  bool operator==(EventId o) const { return value == o.value; }
  bool operator!=(EventId o) const { return value != o.value; }
  bool operator<(EventId o) const { return value < o.value; }
};

inline void removeFromUnorderedList(std::vector<EventId> &list, EventId eventId) {
  auto it = std::find(list.begin(), list.end(), eventId);
  if (it != list.end()) {
    std::swap(*it, list.back());
    list.pop_back();
  }
}

/// Status of a memory event in the race detection lifecycle.
enum class EventStatus {
  ACTIVE,        // Pending; in flight.
  WAVE_COMPLETE, // Owning-wave wait passed; retained until retirement if needed.
  RETIRED        // Fully retired; no longer referenced.
};

/// Describes a detected race condition. Used by the race detection layer
/// to report violations without depending on any exception type.
struct RaceViolation {
  enum class Space { VGPR, SGPR, TTMP, LDS };
  Space space;
  int index;    ///< Register index (VGPR/SGPR) or byte address (LDS).
  int wave;     ///< Wave that triggered the violation.
  int lane;     ///< Lane within the wave, or -1 for scalar.
  bool isWrite; ///< True if the violating access was a write.
  Dim3d workgroupId;
  EventId conflictingEvent; ///< Exact pending memory event that caused the violation.

  RaceViolation(Space space, int index, int wave, int lane, bool isWrite, Dim3d workgroupId,
                EventId conflictingEvent)
      : space(space), index(index), wave(wave), lane(lane), isWrite(isWrite),
        workgroupId(workgroupId), conflictingEvent(conflictingEvent) {}
};

struct WaitCounterUpdate {
  amdgpu::WaitCounterType type;
  int threshold;
};

/// Counter thresholds changed by one wait instruction. A wait can update one
/// or more counters; counters absent from this list retain their prior state.
struct PendingWaitCount {
  void add(amdgpu::WaitCounterType type, int threshold) {
    if (threshold >= 0)
      updates.push_back({type, threshold});
  }

  [[nodiscard]] bool empty() const { return updates.empty(); }

  std::vector<WaitCounterUpdate> updates;
};

} // namespace rocjitsu::plugins::race_detector
