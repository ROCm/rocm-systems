// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/profiler_interface.h"
#include "rocjitsu/vm/plugins/race_detector/core/types.h"
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::plugins::race_detector {

class IntervalSet;
class RaceDetector;

struct CounterCapacities {
  int vmcnt;
  int lgkmcnt;
};

/// Adapt the generated architecture properties to the race core's compact
/// counter model. A zero capacity means that the architecture has no combined
/// counter of that type.
[[nodiscard]] inline constexpr CounterCapacities counterCapacitiesForArch(rj_code_arch_t arch) {
  const auto properties = isa_properties(arch);
  return {
      .vmcnt = properties.vmcnt_capacity,
      .lgkmcnt = properties.lgkmcnt_capacity,
  };
}

/// Per-wave race detection state. Owns the wave's in-flight event queue and
/// barrier-pending events, and provides event registration and waitcnt
/// resolution. Register hazards are diagnosed by the core memory-wait
/// scoreboard, so events carry only counter obligations and LDS intervals.
/// Holds a pointer to the workgroup level RaceDetector for event allocation
/// and lifecycle transitions.
class WaveRaceState {
public:
  WaveRaceState(WaveId, RaceDetector *, CounterCapacities);

  /// Apply the issue backpressure required before an instruction adds one
  /// token to a finite hardware counter. Call before operand checks when the
  /// counter is known statically, or after routing for address-dependent FLAT.
  void prepareForCounterIncrement(amdgpu::WaitCounterType, uint8_t increment = 1);

  /// Apply backpressure to every counter incremented by a resolved issue.
  void prepareForMemoryIssue(const amdgpu::MemoryIssueInfo &);

  /// Register an in-flight memory event that does not involve LDS. The event
  /// keeps partial waits and counter capacity ordered for later LDS events.
  /// \param pc The PC of the instruction that produced the event.
  /// \param type The type of the memory event.
  void registerEvent(uint64_t pc, MemoryEventType type);

  /// Register an event with its architectural wait-counter obligations.
  void
  registerEvent(uint64_t pc, MemoryEventType type, amdgpu::WaitCounterType waitCounterType,
                MemoryOrderClass memoryOrder,
                std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt);
  void registerEvent(uint64_t pc, MemoryEventType type,
                     std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                     MemoryOrderClass memoryOrder);

  /// Register an in-flight memory event that involves LDS.
  /// The LDS memory involved is defined by `laneBaseAddresses` and `bytesPerLane`. Each active lane
  /// contributes one interval of size `bytesPerLane`.
  void registerLdsEvent(uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
                        std::span<const uint32_t> laneBaseAddresses, int bytesPerLane);
  void
  registerLdsEvent(uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
                   std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
                   amdgpu::WaitCounterType waitCounterType, MemoryOrderClass memoryOrder,
                   std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt);
  /// ldsLaneMask restricts LDS intervals to a subset of execMask (for mixed
  /// FLAT). The whole instruction occupies one event with one set of counter
  /// obligations. Bits outside execMask are ignored; zero records no LDS
  /// intervals.
  /// elementBytes selects the execution granularity at the backing boundary;
  /// zero treats the whole per-lane access as one element.
  void registerLdsEvent(uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
                        std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
                        std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                        MemoryOrderClass memoryOrder, uint64_t ldsLaneMask = ~uint64_t{0},
                        int elementBytes = 0);
  /// Register two LDS address sets per lane. elementBytes selects the granularity
  /// of complete elements retained at the backing boundary; zero treats the
  /// whole per-lane access in each address set as one element.
  void registerLdsEvent(uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
                        std::span<const uint32_t> firstLaneBaseAddresses,
                        std::span<const uint32_t> secondLaneBaseAddresses, int bytesPerLane,
                        std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                        MemoryOrderClass memoryOrder, int elementBytes = 0);

  /// Register an LDS event with dual-offset intervals. Each active lane
  /// contributes two 8-byte intervals at laneBaseAddresses[lane] + offset0*8
  /// and laneBaseAddresses[lane] + offset1*8. So each active lane contributes 2 intervals of 8
  /// bytes each.
  void registerDualOffsetLdsEvent(uint64_t pc, MemoryEventType type, uint64_t execMask,
                                  int waveSize, std::span<const uint32_t> laneBaseAddresses,
                                  int32_t offset0, int32_t offset1);
  void registerDualOffsetLdsEvent(
      uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
      std::span<const uint32_t> laneBaseAddresses, int32_t offset0, int32_t offset1,
      amdgpu::WaitCounterType waitCounterType, MemoryOrderClass memoryOrder,
      std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt);
  void
  registerDualOffsetLdsEvent(uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
                             std::span<const uint32_t> laneBaseAddresses, int32_t offset0,
                             int32_t offset1,
                             std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                             MemoryOrderClass memoryOrder);

  /// Dispatch the counter thresholds changed by one wait instruction.
  void dispatch(const PendingWaitCount &);

  /// Retire non-trimmable events whose owning wave completed them before a workgroup barrier.
  void flushBarrierPendingEvents();

  const std::vector<EventId> &getWaveMemoryEvents() const { return waveMemoryEvents; }

  const std::vector<EventId> &getBarrierPendingEvents() const { return barrierPendingEvents; }

  RaceDetector *getDetector() { return detector; }
  const RaceDetector *getDetector() const { return detector; }

  void setProfiler(ProfilerInterface &p) { profiler_ = &p; }

  WaveId getWaveId() const { return waveId; }

private:
  void registerEventWithIntervals(uint64_t pc, MemoryEventType, IntervalSet ldsIntervals,
                                  std::span<const amdgpu::MemoryCounterObligation>,
                                  MemoryOrderClass memoryOrder);

  template <typename Pred>
  void satisfyOldestCounterObligations(int count, amdgpu::WaitCounterType, Pred matches);
  void applyCounterConstraint(amdgpu::WaitCounterType type, int maximumRemaining,
                              bool includeUnordered);
  void applyWaitCounter(amdgpu::WaitCounterType type, int threshold);
  int doNotWaitValue(amdgpu::WaitCounterType type) const;

  std::vector<EventId> waveMemoryEvents;
  std::vector<EventId> barrierPendingEvents;

  CounterCapacities counterCapacities;

  WaveId waveId;
  RaceDetector *detector;
  ProfilerInterface *profiler_ = &NullProfiler::instance();
};

} // namespace rocjitsu::plugins::race_detector
