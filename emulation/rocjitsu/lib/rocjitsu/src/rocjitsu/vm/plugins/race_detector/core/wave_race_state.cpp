// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include <array>
#include <limits>
#include <span>
#include <stdexcept>

namespace rocjitsu::plugins::race_detector {
namespace {
void appendLdsInterval(IntervalSet &intervals, uint32_t address, int bytes,
                       const RaceDetector &detector, int elementBytes = 0) {
  if (bytes <= 0 || elementBytes < 0)
    return;
  bytes = static_cast<int>(
      detector.ldsBytesInRange(address, bytes, elementBytes ? elementBytes : bytes));
  // Intervals must fit the CU's logical LDS backing and the signed index type.
  // Rejected addresses retain their event's counter obligations.
  if (bytes <= 0 || !detector.validLdsRange(address, bytes) ||
      address > static_cast<uint32_t>(std::numeric_limits<int>::max() - bytes))
    return;
  const int start = static_cast<int>(address);
  intervals.append(start, start + bytes);
}

/// RAII guard for ProfilerInterface::beginScope/endScope.
struct ProfileScope {
  ProfilerInterface &p;
  ProfileScope(ProfilerInterface &p, std::string_view key) : p(p) { p.beginScope(key); }
  ~ProfileScope() { p.endScope(); }
};
} // namespace

WaveRaceState::WaveRaceState(WaveId waveId, RaceDetector *detector,
                             CounterCapacities counterCapacities)
    : counterCapacities(counterCapacities), waveId(waveId), detector(detector) {}

void WaveRaceState::dispatch(const PendingWaitCount &wait_count) {
  for (const auto &update : wait_count.updates)
    applyWaitCounter(update.type, update.threshold);
}

void WaveRaceState::registerEvent(uint64_t pc, MemoryEventType type) {
  registerEvent(pc, type, defaultWaitCounterType(type), defaultMemoryOrder(type));
}

void WaveRaceState::registerEvent(
    uint64_t pc, MemoryEventType type, amdgpu::WaitCounterType waitCounterType,
    MemoryOrderClass memoryOrder,
    std::optional<amdgpu::WaitCounterType> additionalWaitCounterType) {
  std::array<amdgpu::MemoryCounterObligation, 2> obligations{
      amdgpu::MemoryCounterObligation{waitCounterType, memoryOrder}, {}};
  size_t count = 1;
  if (additionalWaitCounterType)
    obligations[count++] = {*additionalWaitCounterType, memoryOrder};
  registerEvent(pc, type, std::span(obligations.data(), count), memoryOrder);
}

void WaveRaceState::registerEvent(
    uint64_t pc, MemoryEventType type,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder) {
  registerEventWithIntervals(pc, type, {}, counterObligations, memoryOrder);
}

void WaveRaceState::registerEventWithIntervals(
    uint64_t pc, MemoryEventType type, IntervalSet ldsIntervals,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder) {
  ProfileScope ps(*profiler_, "registerEvent");
  waveMemoryEvents.push_back(detector->allocateEventId(waveId, pc, type, std::move(ldsIntervals),
                                                       counterObligations, memoryOrder));
}

void WaveRaceState::prepareForCounterIncrement(amdgpu::WaitCounterType type, uint8_t increment) {
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
  case amdgpu::WaitCounterType::ASYNCCNT:
    return;
  default:
    assert(false && "invalid wait counter type");
    return;
  }
  if (capacity <= 0)
    throw std::logic_error("counter capacity is not configured");
  assert(increment > 0 && increment <= capacity &&
         "counter increment is outside the configured capacity");
  if (increment == 0 || increment > capacity)
    return;

  int pending_tokens = 0;
  for (const EventId event_id : waveMemoryEvents)
    pending_tokens += detector->events().pendingCounterIncrement(event_id, type);
  if (pending_tokens + increment <= capacity)
    return;

  // An instruction cannot issue if adding its counter token would overflow the
  // finite counter. Once participation is known, admission establishes the
  // same upper bound as an implicit partial wait.
  applyCounterConstraint(type, capacity - increment, /*includeUnordered=*/false);
}

void WaveRaceState::prepareForMemoryIssue(const amdgpu::MemoryIssueInfo &info) {
  for (const auto obligation : info.counter_obligations()) {
    const auto counter = obligation.wait_counter_type();
    if (amdgpu::wait_counter_covers(amdgpu::WaitCounterType::VMCNT, counter) ||
        amdgpu::wait_counter_covers(amdgpu::WaitCounterType::LGKMCNT, counter))
      prepareForCounterIncrement(counter, obligation.counter_increment());
  }
}

void WaveRaceState::registerLdsEvent(uint64_t pc, MemoryEventType type, uint64_t execMask,
                                     int waveSize, std::span<const uint32_t> laneBaseAddresses,
                                     int bytesPerLane) {
  registerLdsEvent(pc, type, execMask, waveSize, laneBaseAddresses, bytesPerLane,
                   defaultWaitCounterType(type), defaultMemoryOrder(type));
}

void WaveRaceState::registerLdsEvent(
    uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
    std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
    amdgpu::WaitCounterType waitCounterType, MemoryOrderClass memoryOrder,
    std::optional<amdgpu::WaitCounterType> additionalWaitCounterType) {
  std::array<amdgpu::MemoryCounterObligation, 2> obligations{
      amdgpu::MemoryCounterObligation{waitCounterType, memoryOrder}, {}};
  size_t count = 1;
  if (additionalWaitCounterType)
    obligations[count++] = {*additionalWaitCounterType, memoryOrder};
  registerLdsEvent(pc, type, execMask, waveSize, laneBaseAddresses, bytesPerLane,
                   std::span(obligations.data(), count), memoryOrder);
}

void WaveRaceState::registerLdsEvent(
    uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
    std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder, uint64_t ldsLaneMask, int elementBytes) {
  IntervalSet intervals;
  forEachActiveLane(execMask & ldsLaneMask, waveSize, [&](int lane) {
    appendLdsInterval(intervals, laneBaseAddresses[lane], bytesPerLane, *detector, elementBytes);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(intervals), counterObligations, memoryOrder);
}

void WaveRaceState::registerLdsEvent(
    uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
    std::span<const uint32_t> firstLaneBaseAddresses,
    std::span<const uint32_t> secondLaneBaseAddresses, int bytesPerLane,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder, int elementBytes) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    for (const auto addresses : {firstLaneBaseAddresses, secondLaneBaseAddresses})
      appendLdsInterval(intervals, addresses[lane], bytesPerLane, *detector, elementBytes);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(intervals), counterObligations, memoryOrder);
}

void WaveRaceState::registerDualOffsetLdsEvent(uint64_t pc, MemoryEventType type, uint64_t execMask,
                                               int waveSize,
                                               std::span<const uint32_t> laneBaseAddresses,
                                               int32_t offset0, int32_t offset1) {
  registerDualOffsetLdsEvent(pc, type, execMask, waveSize, laneBaseAddresses, offset0, offset1,
                             defaultWaitCounterType(type), defaultMemoryOrder(type));
}

void WaveRaceState::registerDualOffsetLdsEvent(
    uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
    std::span<const uint32_t> laneBaseAddresses, int32_t offset0, int32_t offset1,
    amdgpu::WaitCounterType waitCounterType, MemoryOrderClass memoryOrder,
    std::optional<amdgpu::WaitCounterType> additionalWaitCounterType) {
  std::array<amdgpu::MemoryCounterObligation, 2> obligations{
      amdgpu::MemoryCounterObligation{waitCounterType, memoryOrder}, {}};
  size_t count = 1;
  if (additionalWaitCounterType)
    obligations[count++] = {*additionalWaitCounterType, memoryOrder};
  registerDualOffsetLdsEvent(pc, type, execMask, waveSize, laneBaseAddresses, offset0, offset1,
                             std::span(obligations.data(), count), memoryOrder);
}

void WaveRaceState::registerDualOffsetLdsEvent(
    uint64_t pc, MemoryEventType type, uint64_t execMask, int waveSize,
    std::span<const uint32_t> laneBaseAddresses, int32_t offset0, int32_t offset1,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    uint32_t vAddr = laneBaseAddresses[lane];
    appendLdsInterval(intervals, vAddr + static_cast<uint32_t>(offset0) * 8, 8, *detector);
    appendLdsInterval(intervals, vAddr + static_cast<uint32_t>(offset1) * 8, 8, *detector);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(intervals), counterObligations, memoryOrder);
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
        // otherwise a later barrier could try to retire stale EventIds.
        if (!detector->events().isTrimmable(eventId))
          barrierPendingEvents.push_back(eventId);
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
        MemoryOrderClass::ASYNC_LOAD, MemoryOrderClass::ASYNC_STORE}) {
    auto belongsToOrder = [&](EventId eventId) {
      return detector->events().pendingCompletionClass(eventId, type) == order;
    };
    const int pending = pending_tokens(belongsToOrder);
    satisfyOldestCounterObligations(pending - maximumRemaining, type, belongsToOrder);
  }
}

int WaveRaceState::doNotWaitValue(amdgpu::WaitCounterType type) const {
  switch (type) {
  case amdgpu::WaitCounterType::VMCNT:
    return counterCapacities.vmcnt;
  case amdgpu::WaitCounterType::LGKMCNT:
    return counterCapacities.lgkmcnt;
  case amdgpu::WaitCounterType::EXPCNT:
    return amdgpu::WaitCounters::EXPCNT_MAX;
  case amdgpu::WaitCounterType::VSCNT:
    return amdgpu::WaitCounters::VSCNT_MAX;
  case amdgpu::WaitCounterType::LOADCNT:
    return amdgpu::WaitCounters::VMCNT_MAX;
  case amdgpu::WaitCounterType::STORECNT:
    return amdgpu::WaitCounters::VSCNT_MAX;
  case amdgpu::WaitCounterType::DSCNT:
    return amdgpu::WaitCounters::DSCNT_MAX;
  case amdgpu::WaitCounterType::KMCNT:
    return amdgpu::WaitCounters::KMCNT_MAX;
  case amdgpu::WaitCounterType::TENSORCNT:
    return amdgpu::WaitCounters::TENSORCNT_MAX;
  case amdgpu::WaitCounterType::ASYNCCNT:
    return amdgpu::WaitCounters::ASYNCCNT_MAX;
  }
  assert(false && "invalid wait counter type");
  return -1;
}

void WaveRaceState::applyWaitCounter(amdgpu::WaitCounterType type, int threshold) {
  if (threshold < 0 || threshold == doNotWaitValue(type))
    return;

  if (threshold == 0) {
    applyCounterConstraint(type, 0, /*includeUnordered=*/true);
    return;
  }

  applyCounterConstraint(type, threshold, /*includeUnordered=*/false);
}

void WaveRaceState::flushBarrierPendingEvents() {
  ProfileScope ps(*profiler_, "removeEvents");
  for (EventId eventId : barrierPendingEvents) {
    detector->retireEvent(eventId);
  }
  barrierPendingEvents.clear();
}

} // namespace rocjitsu::plugins::race_detector
