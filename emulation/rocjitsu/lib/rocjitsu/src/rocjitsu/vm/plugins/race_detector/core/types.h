// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/dim3d.h"
#include <algorithm>
#include <cstdint>
#include <utility>
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

/// Identifies one round of one barrier within a workgroup: the signals counted
/// between two completions. Waves release into a round when they signal it and
/// acquire it when they pass the wait it released.
struct BarrierRoundId {
  uint8_t barrier = 0;     ///< The simulator's completion bit for the barrier.
  uint32_t namedId = 0;    ///< Named barrier ID, or 0.
  uint64_t generation = 0; ///< Distinguishes successive rounds of one barrier.

  auto operator<=>(const BarrierRoundId &) const = default;
};

/// A contiguous run of LDS bytes, for events described by byte ranges rather
/// than per-lane addresses.
struct LdsByteRange {
  uint32_t address = 0; ///< Absolute CU LDS byte address.
  uint32_t bytes = 0;
};

inline void removeFromUnorderedList(std::vector<EventId> &list, EventId eventId) {
  auto it = std::ranges::find(list, eventId);
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
  int lane;     ///< Lane within the wave, or -1 for no lane (scalar or tensor DMA).
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
