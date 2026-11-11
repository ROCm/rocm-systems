// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/vm/plugins/race_detector/core/dim3d.h"
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

using MemoryOrderClass = amdgpu::MemoryCompletionClass;

/// Counter-only issues occupy queue positions but have no LDS effect.
enum class LdsAccess : uint8_t { None, Read, Write, ReadWrite };

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

/// An LDS conflict reported without depending on the plugin's output format.
struct RaceViolation {
  int address;  ///< Byte address shared by the conflicting LDS accesses.
  int wave;     ///< Wave that triggered the violation.
  int lane;     ///< Lane within the wave, or -1 for a tensor DMA transfer.
  bool isWrite; ///< True if the violating access was a write.
  Dim3d workgroupId;
  EventId conflictingEvent; ///< Exact retained event that caused the violation.
};

} // namespace rocjitsu::plugins::race_detector
