// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/instruction_snapshot.h"
#include "rocjitsu/vm/amdgpu/lds_byte_range.h"
#include "rocjitsu/vm/plugins/race_detector/core/types.h"
#include "util/result.h"
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
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

/// Per-wave counter progress for LDS race detection. Completed LDS evidence
/// remains in the workgroup detector until a barrier orders it with peer waves.
/// Instruction snapshots are decoded only when a report needs disassembly.
class WaveRaceState {
public:
  WaveRaceState(WaveId, RaceDetector *, CounterCapacities);

  /// Apply the issue backpressure required before an instruction adds one
  /// token to a finite hardware counter. Call before operand checks when the
  /// counter is known statically, or after routing for address-dependent FLAT.
  /// Returns failure if the counter capacity or increment is invalid.
  util::Result prepareForCounterIncrement(amdgpu::WaitCounterType, uint8_t increment = 1);

  /// Apply backpressure to every counter incremented by a resolved issue.
  util::Result prepareForMemoryIssue(const amdgpu::MemoryIssueInfo &);

  /// Count an issue without an LDS effect, so partial waits retain their
  /// position relative to LDS operations sharing the counter.
  void registerCounterEvent(InstructionSnapshot instruction,
                            std::span<const amdgpu::MemoryCounterObligation> counterObligations);

  /// Register one LDS interval per active lane. elementBytes selects the
  /// granularity of complete elements kept at the backing/allocation boundary;
  /// zero treats the whole per-lane access as one element. Empty intervals
  /// still contribute the issue's counter obligations.
  void registerLdsEvent(InstructionSnapshot instruction, LdsAccess access, uint64_t laneMask,
                        int waveSize, std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
                        std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                        MemoryOrderClass memoryOrder, int elementBytes = 0,
                        std::optional<uint64_t> allocationEnd = std::nullopt);

  /// Register two LDS address sets per lane with the same clipping rules.
  void registerLdsEvent(InstructionSnapshot instruction, LdsAccess access, uint64_t laneMask,
                        int waveSize, std::span<const uint32_t> firstLaneBaseAddresses,
                        std::span<const uint32_t> secondLaneBaseAddresses, int bytesPerLane,
                        std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                        MemoryOrderClass memoryOrder, int elementBytes = 0);

  /// Register an LDS event described by byte ranges rather than lanes, such as
  /// a tensor DMA transfer or async LDS store. Each range is retained
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
  void registerLdsRangeEvent(InstructionSnapshot instruction, LdsAccess access,
                             std::span<const amdgpu::LdsByteRange> ranges,
                             std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                             MemoryOrderClass memoryOrder);

  /// @brief Register an event using the caller's finalized LDS intervals.
  /// @details Intervals must contain only accessed bytes within the LDS backing
  /// and, for FLAT, the workgroup allocation, with endpoints that fit in int.
  /// This does not clip ranges or validate conflicts; validate before registering.
  /// Empty intervals retain the event's counter dependencies.
  void registerEventWithIntervals(InstructionSnapshot instruction, LdsAccess access,
                                  IntervalSet ldsIntervals,
                                  std::span<const amdgpu::MemoryCounterObligation>,
                                  MemoryOrderClass memoryOrder, uint8_t alignedAtomicBytes = 0);

  /// Apply a decoded counter bound; negative values leave progress unchanged.
  /// ISA no-wait sentinels must be filtered by the decoder, not passed as bounds.
  void applyWaitCounter(amdgpu::WaitCounterType type, int threshold);

  /// Check a synchronous VALU source against older writers and remember the
  /// completed read until a barrier publishes it to peer waves.
  void recordLdsOperandRead(InstructionSnapshot instruction, uint32_t address, uint32_t bytes,
                            uint32_t lane);

  /// Retire completed LDS evidence ordered with peer waves by a workgroup barrier.
  void flushBarrierPendingEvents();

  const std::vector<EventId> &getWaveMemoryEvents() const { return waveMemoryEvents; }

  const std::vector<EventId> &getBarrierPendingEvents() const { return barrierPendingEvents; }

  RaceDetector *getDetector() { return detector; }
  const RaceDetector *getDetector() const { return detector; }

  WaveId getWaveId() const { return waveId; }

private:
  template <typename Pred>
  void satisfyOldestCounterObligations(int count, amdgpu::WaitCounterType, Pred matches);
  void applyCounterConstraint(amdgpu::WaitCounterType type, int maximumRemaining,
                              bool includeUnordered);

  std::vector<EventId> waveMemoryEvents;
  std::unordered_map<uint32_t, uint8_t> directLdsReadWidths;
  std::vector<EventId> barrierPendingEvents;

  CounterCapacities counterCapacities;

  WaveId waveId;
  RaceDetector *detector;
};

} // namespace rocjitsu::plugins::race_detector
