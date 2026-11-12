// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/instruction_snapshot.h"
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

/// Per-wave race detection state. Owns VGPR event lists, wave-level event
/// queues, and provides event registration, waitcnt resolution, and VGPR
/// race checking. Holds a pointer to the workgroup level RaceDetector for
/// event allocation and lifecycle transitions.
///
/// Every event registration method takes the producing instruction as an
/// InstructionSnapshot; the event's PC comes from the snapshot. The encoding
/// is decoded only if a plugin report needs its disassembly.
class WaveRaceState {
public:
  WaveRaceState(int vgprCount, int sgprCount, WaveId, RaceDetector *, CounterCapacities);

  /// Apply the issue backpressure required before an instruction adds one
  /// token to a finite hardware counter. Call before operand checks when the
  /// counter is known statically, or after routing for address-dependent FLAT.
  void prepareForCounterIncrement(amdgpu::WaitCounterType, uint8_t increment = 1);

  /// Apply backpressure to every counter incremented by a resolved issue.
  void prepareForMemoryIssue(const amdgpu::MemoryIssueInfo &);

  /// Register an in-flight memory event that does not involve LDS.
  /// \param instruction Complete captured encoding and PC of the producing instruction.
  /// \param type The type of the memory event.
  /// \param registers The register IDs involved in the event.
  /// \param execMask The execution mask at the time the event was produced.
  /// \param byteMask The byte mask of the event, if this event effects only certain bytes offset
  ///                 register(s). Defaults to 0xF (all bytes).
  void registerEvent(InstructionSnapshot instruction, MemoryEventType type,
                     std::vector<uint32_t> registers, uint64_t execMask, uint8_t byteMask = 0xF);

  /// Register an event with its architectural wait-counter obligations.
  /// A nonzero lastRegisterByteMask overrides byteMask for the last destination.
  void registerEvent(InstructionSnapshot instruction, MemoryEventType type,
                     std::vector<uint32_t> registers, uint64_t execMask, uint8_t byteMask,
                     std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                     MemoryOrderClass memoryOrder, uint8_t lastRegisterByteMask = 0);

  /// Register an in-flight scalar load using its architectural destination.
  void registerScalarLoad(InstructionSnapshot instruction, RegisterRef destination,
                          uint64_t execMask,
                          std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                          MemoryOrderClass memoryOrder);

  /// Register an in-flight memory event that involves LDS.
  /// The LDS memory involved is defined by `laneBaseAddresses` and `bytesPerLane`. Each active lane
  /// contributes one interval of size `bytesPerLane`.
  void registerLdsEvent(InstructionSnapshot instruction, MemoryEventType type,
                        std::vector<uint32_t> registers, uint64_t execMask, int waveSize,
                        std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
                        uint8_t byteMask = 0xF);
  /// ldsLaneMask restricts LDS intervals to a subset of execMask (for mixed
  /// FLAT). Register dependencies still use all execMask lanes, and the whole
  /// instruction occupies one event with one set of counter obligations.
  /// Bits outside execMask are ignored; zero records no LDS intervals.
  /// elementBytes selects the execution granularity at the backing boundary;
  /// zero treats the whole per-lane access as one element.
  void registerLdsEvent(InstructionSnapshot instruction, MemoryEventType type,
                        std::vector<uint32_t> registers, uint64_t execMask, int waveSize,
                        std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
                        uint8_t byteMask,
                        std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                        MemoryOrderClass memoryOrder, uint64_t ldsLaneMask = ~uint64_t{0},
                        int elementBytes = 0);
  /// Register two LDS address sets per lane. elementBytes selects the granularity
  /// of complete elements retained at the backing boundary; zero treats the
  /// whole per-lane access in each address set as one element.
  void registerLdsEvent(InstructionSnapshot instruction, MemoryEventType type,
                        std::vector<uint32_t> registers, uint64_t execMask, int waveSize,
                        std::span<const uint32_t> firstLaneBaseAddresses,
                        std::span<const uint32_t> secondLaneBaseAddresses, int bytesPerLane,
                        uint8_t byteMask,
                        std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                        MemoryOrderClass memoryOrder, int elementBytes = 0);

  /// Register an LDS event described by byte ranges rather than lanes, such as
  /// a tensor DMA transfer. The event has no registers. Each range is retained
  /// byte by byte up to the end of the LDS backing, not rounded to whole
  /// elements, because a range can end mid-element.
  ///
  /// This only registers the event; validate its ranges first with
  /// RaceDetector::validateRead() or validateWrite(). Callers choose three
  /// orderings independently:
  /// - Partial waits use the completion class of each obligation in
  ///   counterObligations. Tensor DMA passes TENSOR on TENSORCNT, since its
  ///   transfers from one wave complete in issue order, so a partial
  ///   s_wait_tensorcnt retires the oldest.
  /// - memoryOrder is stored on the event and compared with the order a later
  ///   same-wave access validates with: only an access in the same class,
  ///   other than UNORDERED, is treated as ordered with the event. Tensor DMA
  ///   passes TENSOR.
  /// - The order passed to validation decides which of the wave's pending
  ///   accesses the new ranges are checked against. Tensor DMA passes
  ///   UNORDERED, since the ISA does not specify the timing of its LDS
  ///   accesses; passing TENSOR would skip conflicts with the wave's own
  ///   pending transfers.
  void registerLdsRangeEvent(InstructionSnapshot instruction, MemoryEventType type,
                             std::span<const LdsByteRange> ranges,
                             std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                             MemoryOrderClass memoryOrder);

  /// @brief Register an event using the caller's finalized LDS intervals.
  /// @details Intervals must contain only accessed bytes within the LDS backing
  /// and, for FLAT, the workgroup allocation, with endpoints that fit in int.
  /// This does not clip ranges or validate conflicts; validate before registering.
  /// Empty intervals retain the event's register and counter dependencies.
  void registerEventWithIntervals(InstructionSnapshot instruction, MemoryEventType,
                                  std::vector<uint32_t> registers, uint64_t execMask,
                                  uint8_t byteMask, IntervalSet ldsIntervals,
                                  std::span<const amdgpu::MemoryCounterObligation>,
                                  MemoryOrderClass memoryOrder, uint8_t lastRegisterByteMask = 0);

  /// Dispatch the counter thresholds changed by one wait instruction.
  void dispatch(const PendingWaitCount &);

  /// Retire non-trimmable events whose owning wave completed them before a workgroup barrier.
  void flushBarrierPendingEvents();

  /// Check a VGPR read by a set of lanes for races. Calls the RaceHandler on violation.
  void checkVgprReadLanes(int reg, uint64_t laneMask, uint8_t byteMask) const;

  /// Check a VGPR instruction write for conflicts with pending asynchronous
  /// loads targeting the same register bytes and lanes in laneMask.
  void checkVgprWriteLanes(int reg, uint64_t laneMask, uint8_t byteMask) const;

  /// Check an asynchronous memory destination write. On supported targets,
  /// operations in the same non-UNORDERED class cannot overtake one another.
  void checkVgprWrite(int reg, uint64_t execMask, uint8_t byteMask,
                      MemoryOrderClass currentMemoryOrder) const;

  /// Check a scalar register read for races. Calls the RaceHandler on
  /// violation (outstanding scalar load targeting the same register).
  void checkScalarRead(RegisterRef reg) const;

  /// Check a scalar instruction write for conflicts with pending scalar loads.
  /// A wide write reports each conflicting memory event at most once.
  void checkScalarWrite(RegisterRef reg) const;

  std::vector<EventId> &getVgprMemoryEvents(int reg) { return vgprMemoryEvents[reg]; }

  const std::vector<EventId> &getWaveMemoryEvents() const { return waveMemoryEvents; }

  const std::vector<EventId> &getBarrierPendingEvents() const { return barrierPendingEvents; }

  RaceDetector *getDetector() { return detector; }
  const RaceDetector *getDetector() const { return detector; }

  WaveId getWaveId() const { return waveId; }

private:
  void retireEventRegisters(EventId);
  void checkScalarAccess(RegisterRef reg, bool isWrite) const;

  template <typename Pred>
  void satisfyOldestCounterObligations(int count, amdgpu::WaitCounterType, Pred matches);
  void applyCounterConstraint(amdgpu::WaitCounterType type, int maximumRemaining,
                              bool includeUnordered);
  void applyWaitCounter(amdgpu::WaitCounterType type, int threshold);
  int doNotWaitValue(amdgpu::WaitCounterType type) const;

  std::vector<std::vector<EventId>> vgprMemoryEvents;
  std::vector<std::vector<EventId>> sgprMemoryEvents;
  std::vector<std::vector<EventId>> ttmpMemoryEvents;

  std::vector<EventId> waveMemoryEvents;
  std::vector<EventId> barrierPendingEvents;

  CounterCapacities counterCapacities;

  WaveId waveId;
  RaceDetector *detector;
};

} // namespace rocjitsu::plugins::race_detector
