// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/amdgpu/lds_allocation.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {
namespace {
void appendLdsInterval(IntervalSet &intervals, uint32_t address, int bytes,
                       const RaceDetector &detector, int elementBytes = 0,
                       std::optional<uint64_t> allocationEnd = std::nullopt) {
  if (bytes <= 0 || elementBytes < 0)
    return;
  bytes = static_cast<int>(
      detector.ldsBytesInRange(address, bytes, elementBytes ? elementBytes : bytes));
  if (allocationEnd) {
    const uint32_t element = elementBytes ? elementBytes : bytes;
    bytes = static_cast<int>(
        amdgpu::whole_element_bytes_in_range(address, bytes, element, *allocationEnd));
  }
  // Intervals must fit the CU's logical LDS backing and the signed index type.
  // Rejected addresses still contribute their event's counter obligations.
  if (bytes <= 0 || !detector.validLdsRange(address, bytes) ||
      address > static_cast<uint32_t>(std::numeric_limits<int>::max() - bytes))
    return;
  const int start = static_cast<int>(address);
  intervals.append(start, start + bytes);
}
} // namespace

WaveRaceState::WaveRaceState(WaveId waveId, RaceDetector *detector,
                             CounterCapacities counterCapacities)
    : counterCapacities(counterCapacities), waveId(waveId), detector(detector) {}

void WaveRaceState::registerCounterEvent(
    InstructionSnapshot instruction,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations) {
  registerEventWithIntervals(instruction, LdsAccess::None, {}, counterObligations,
                             MemoryOrderClass::UNORDERED);
}

void WaveRaceState::registerEventWithIntervals(
    InstructionSnapshot instruction, LdsAccess access, IntervalSet ldsIntervals,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder, uint8_t alignedAtomicBytes) {
  waveMemoryEvents.push_back(detector->allocateEventId(waveId, instruction, access,
                                                       std::move(ldsIntervals), counterObligations,
                                                       memoryOrder, alignedAtomicBytes));
}

util::Result WaveRaceState::prepareForCounterIncrement(amdgpu::WaitCounterType type,
                                                       uint8_t increment) {
  // The runtime plugin calls this only for pre-GFX12 capacity models, where
  // these internal split names alias the combined architectural counters.
  if (amdgpu::wait_counter_covers(amdgpu::WaitCounterType::VMCNT, type))
    type = amdgpu::WaitCounterType::VMCNT;
  else if (amdgpu::wait_counter_covers(amdgpu::WaitCounterType::LGKMCNT, type))
    type = amdgpu::WaitCounterType::LGKMCNT;

  int capacity;
  switch (type) {
  case amdgpu::WaitCounterType::VMCNT:
    capacity = counterCapacities.vmcnt;
    break;
  case amdgpu::WaitCounterType::LGKMCNT:
    capacity = counterCapacities.lgkmcnt;
    break;
  case amdgpu::WaitCounterType::EXPCNT:
  case amdgpu::WaitCounterType::VSCNT:
  case amdgpu::WaitCounterType::LOADCNT:
  case amdgpu::WaitCounterType::STORECNT:
  case amdgpu::WaitCounterType::DSCNT:
  case amdgpu::WaitCounterType::KMCNT:
  case amdgpu::WaitCounterType::TENSORCNT:
  case amdgpu::WaitCounterType::SAMPLECNT:
  case amdgpu::WaitCounterType::ASYNCCNT:
    return util::Result::success();
  default:
    assert(false && "invalid wait counter type");
    return util::Result::success();
  }
  if (capacity <= 0 || increment == 0 || increment > capacity)
    return util::Result::failure();

  int pending_tokens = 0;
  for (const EventId event_id : waveMemoryEvents)
    pending_tokens += detector->events().pendingCounterIncrement(event_id, type);
  if (pending_tokens + increment <= capacity)
    return util::Result::success();

  // An instruction cannot issue if adding its counter token would overflow the
  // finite counter. Once participation is known, admission establishes the
  // same upper bound as an implicit partial wait.
  applyCounterConstraint(type, capacity - increment, /*includeUnordered=*/false);
  return util::Result::success();
}

util::Result WaveRaceState::prepareForMemoryIssue(const amdgpu::MemoryIssueInfo &info) {
  for (const auto obligation : info.counter_obligations()) {
    const auto counter = obligation.wait_counter_type();
    if (amdgpu::wait_counter_covers(amdgpu::WaitCounterType::VMCNT, counter) ||
        amdgpu::wait_counter_covers(amdgpu::WaitCounterType::LGKMCNT, counter))
      if (prepareForCounterIncrement(counter, obligation.counter_increment()).failed())
        return util::Result::failure();
  }
  return util::Result::success();
}

void WaveRaceState::registerLdsEvent(
    InstructionSnapshot instruction, LdsAccess access, uint64_t laneMask, int waveSize,
    std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder, int elementBytes, std::optional<uint64_t> allocationEnd) {
  IntervalSet intervals;
  forEachActiveLane(laneMask, waveSize, [&](int lane) {
    appendLdsInterval(intervals, laneBaseAddresses[lane], bytesPerLane, *detector, elementBytes,
                      allocationEnd);
  });
  intervals.finalize();
  registerEventWithIntervals(instruction, access, std::move(intervals), counterObligations,
                             memoryOrder);
}

void WaveRaceState::registerLdsEvent(
    InstructionSnapshot instruction, LdsAccess access, uint64_t laneMask, int waveSize,
    std::span<const uint32_t> firstLaneBaseAddresses,
    std::span<const uint32_t> secondLaneBaseAddresses, int bytesPerLane,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder, int elementBytes) {
  IntervalSet intervals;
  forEachActiveLane(laneMask, waveSize, [&](int lane) {
    for (const auto addresses : {firstLaneBaseAddresses, secondLaneBaseAddresses})
      appendLdsInterval(intervals, addresses[lane], bytesPerLane, *detector, elementBytes);
  });
  intervals.finalize();
  registerEventWithIntervals(instruction, access, std::move(intervals), counterObligations,
                             memoryOrder);
}

void WaveRaceState::registerLdsRangeEvent(
    InstructionSnapshot instruction, LdsAccess access, std::span<const amdgpu::LdsByteRange> ranges,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder) {
  IntervalSet intervals;
  for (const auto &range : ranges)
    appendLdsInterval(intervals, range.address, static_cast<int>(range.bytes), *detector,
                      /*elementBytes=*/1);
  intervals.finalize();
  registerEventWithIntervals(instruction, access, std::move(intervals), counterObligations,
                             memoryOrder);
}

template <typename Pred>
void WaveRaceState::satisfyOldestCounterObligations(int count, amdgpu::WaitCounterType waitCounter,
                                                    Pred matches) {
  if (count <= 0)
    return;

  int satisfied = 0;
  size_t write = 0;
  for (size_t read = 0; read < waveMemoryEvents.size(); ++read) {
    const EventId eventId = waveMemoryEvents[read];
    bool eventComplete = false;
    if (matches(eventId) && satisfied < count) {
      satisfied += detector->events().pendingCounterIncrement(eventId, waitCounter);
      eventComplete = detector->satisfyEventWaitCounter(eventId, waitCounter);
      if (eventComplete) {
        detector->markEventWaveComplete(eventId);
        // Trimmable WAVE_COMPLETE events may be removed from the registry
        // immediately. Only keep non-trimmable events for later barrier retire;
        // otherwise a later barrier could try to retire stale EventIds. A
        // completed LDS event is kept only so that other waves can still race
        // with it, so a single-wave workgroup retires it now.
        if (!detector->events().isTrimmable(eventId)) {
          if (detector->hasSingleWave())
            detector->retireEvent(eventId);
          else
            barrierPendingEvents.push_back(eventId);
        }
      }
    }
    if (!eventComplete)
      waveMemoryEvents[write++] = eventId;
  }
  waveMemoryEvents.resize(write);
}

void WaveRaceState::applyCounterConstraint(amdgpu::WaitCounterType type, int maximumRemaining,
                                           bool includeUnordered) {
  auto targetsCounter = [&](EventId eventId) {
    return detector->events().hasPendingWaitCounter(eventId, type);
  };
  auto pending_tokens = [&](auto matches) {
    int total = 0;
    for (const EventId event_id : waveMemoryEvents) {
      if (matches(event_id))
        total += detector->events().pendingCounterIncrement(event_id, type);
    }
    return total;
  };
  if (includeUnordered) {
    satisfyOldestCounterObligations(pending_tokens(targetsCounter), type, targetsCounter);
    return;
  }

  // A nonzero bound cannot select an unordered event. For each independently
  // ordered class, however, at most `maximumRemaining` members of that class
  // can still be pending after the counter constraint is satisfied.
  for (MemoryOrderClass order :
       {MemoryOrderClass::VMEM, MemoryOrderClass::LDS, MemoryOrderClass::GDS,
        MemoryOrderClass::ASYNC_LOAD, MemoryOrderClass::ASYNC_STORE, MemoryOrderClass::TENSOR}) {
    auto belongsToOrder = [&](EventId eventId) {
      return detector->events().pendingCompletionClass(eventId, type) == order;
    };
    const int pending = pending_tokens(belongsToOrder);
    satisfyOldestCounterObligations(pending - maximumRemaining, type, belongsToOrder);
  }
}

void WaveRaceState::applyWaitCounter(amdgpu::WaitCounterType type, int threshold) {
  if (threshold < 0)
    return;
  applyCounterConstraint(type, threshold, /*includeUnordered=*/threshold == 0);
}

void WaveRaceState::recordLdsOperandRead(InstructionSnapshot instruction, uint32_t address,
                                         uint32_t bytes, uint32_t lane) {
  if (!detector->validLdsRange(address, bytes))
    return;
  // Each actual read can conflict with a newer write, even when these bytes
  // are already represented in the wave's completed read history.
  detector->validateRead(address, waveId, lane, bytes, MemoryOrderClass::UNORDERED);
  auto &covered = directLdsReadWidths[address];
  if (bytes <= covered)
    return;
  // Preserve the first instruction to consume each byte. Only a wider read
  // adds to this DWORD's barrier history, without creating a counter token.
  IntervalSet intervals;
  intervals.append(address + covered, address + bytes);
  auto event = detector->allocateEventId(waveId, instruction, LdsAccess::Read, std::move(intervals),
                                         std::span<const amdgpu::MemoryCounterObligation>{},
                                         MemoryOrderClass::UNORDERED);
  barrierPendingEvents.push_back(event);
  covered = bytes;
}

void WaveRaceState::flushBarrierPendingEvents() {
  for (EventId eventId : barrierPendingEvents) {
    detector->retireEvent(eventId);
  }
  barrierPendingEvents.clear();
  directLdsReadWidths.clear();
}

} // namespace rocjitsu::plugins::race_detector
