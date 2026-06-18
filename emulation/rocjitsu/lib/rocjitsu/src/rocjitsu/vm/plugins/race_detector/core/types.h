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
/// through its lifecycle: registration, s_waitcnt completion, and barrier
/// retirement. Strongly typed to prevent accidental mixing with wave IDs,
/// register indices, or byte addresses.
struct EventId {
  int value;
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
  ACTIVE,        // Pending. Unsafe for everyone.
  WAVE_COMPLETE, // s_waitcnt passed. Safe for owning wave, unsafe for others.
  RETIRED        // Fully retired (s_barrier). No longer referenced.
};

/// Describes a detected race condition. Used by the race detection layer
/// to report violations without depending on any exception type.
struct RaceViolation {
  enum class Space { VGPR, SGPR, LDS };
  Space space;
  int index;    ///< Register index (VGPR/SGPR) or byte address (LDS).
  int wave;     ///< Wave that triggered the violation.
  int lane;     ///< Lane within the wave, or -1 for scalar.
  bool isWrite; ///< True if the violating access was a write.
  Dim3d workgroupId;
};

/// Pending memory event data written by instruction executors. Dispatched to
/// WaveRaceState by Workgroup::run after tryExecute returns.
struct PendingMemoryEvent {
  uint64_t pc;
  MemoryEventType type;
  std::vector<uint32_t> registers;
  uint64_t execMask;
  int waveSize;
  uint8_t byteMask = 0xF;
  // LDS events:
  std::vector<uint32_t> laneBaseAddresses;
  int bytesPerLane = 0;
  // Dual-offset LDS events:
  bool isDualOffset = false;
  int32_t offset0 = 0;
  int32_t offset1 = 0;
};

/// Pending wait count data written by the s_waitcnt executor. Dispatched to
/// WaveRaceState by Workgroup::run after tryExecute returns.
struct PendingWaitCount {
  int vmcnt = -1;
  int lgkmcnt = -1;
};

} // namespace rocjitsu::plugins::race_detector
