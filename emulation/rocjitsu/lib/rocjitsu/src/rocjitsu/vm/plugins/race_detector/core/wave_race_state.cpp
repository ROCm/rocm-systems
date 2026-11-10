// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include <algorithm>
#include <bit>
#include <limits>
#include <span>
#include <stdexcept>

namespace rocjitsu::plugins::race_detector {
namespace {
void appendLdsInterval(IntervalSet &intervals, uint32_t address, int bytes,
                       const RaceDetector &detector, int elementBytes = 0,
                       std::optional<amdgpu::LdsAllocationRange> allocation = std::nullopt) {
  if (bytes <= 0 || elementBytes < 0)
    return;
  bytes = static_cast<int>(
      detector.ldsBytesInRange(address, bytes, elementBytes ? elementBytes : bytes));
  if (allocation) {
    const uint32_t element = elementBytes ? elementBytes : bytes;
    bytes = static_cast<int>(allocation->retained_bytes(address, bytes, element));
  }
  // Intervals must fit the CU's logical LDS backing and the signed index type.
  // Rejected addresses retain their event's register/counter dependencies.
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

WaveRaceState::WaveRaceState(int vgprCount, int sgprCount, WaveId waveId, RaceDetector *detector,
                             CounterCapacities counterCapacities)
    : counterCapacities(counterCapacities), waveId(waveId), detector(detector) {
  vgprMemoryEvents.resize(vgprCount);
  sgprMemoryEvents.resize(sgprCount);
  sgprEventCount.resize(sgprCount, 0);
  ttmpMemoryEvents.resize(REGISTER_SET_MAX_TTMPS);
  ttmpEventCount.resize(REGISTER_SET_MAX_TTMPS, 0);
  for (size_t i = 0; i != regEventCount.size(); ++i) {
    if (static_cast<MemoryEventType>(i) != MemoryEventType::DIRECT_LDS_READ)
      regEventCount[i].resize(vgprCount, 0);
  }
}

void WaveRaceState::lockGdsVgpr(InstructionSnapshot instruction, int reg, uint64_t lanes) {
  for (const auto &lock : gdsVgprLocks)
    if (lock.reg == reg)
      lanes &= ~lock.lanes;
  if (lanes)
    gdsVgprLocks.push_back({reg, lanes, instruction});
}

void WaveRaceState::lockGdsExec(InstructionSnapshot instruction) {
  if (!gdsExecLock)
    gdsExecLock = instruction;
}

void WaveRaceState::reportGdsSourceWrite(RaceViolation::Space space, int reg, int lane,
                                         const InstructionSnapshot &source) const {
  RaceViolation violation(space, reg, waveId.value, lane, true, detector->getWorkgroupId(), {});
  violation.sourceInstruction = source;
  detector->getRaceHandler()(std::move(violation));
}

void WaveRaceState::checkGdsExecWrite() const {
  if (gdsExecLock)
    reportGdsSourceWrite(RaceViolation::Space::EXEC, 0, -1, *gdsExecLock);
}

void WaveRaceState::checkGdsVgprWrite(int reg, uint64_t lanes) const {
  for (const auto &lock : gdsVgprLocks)
    if (lock.reg == reg && (lock.lanes & lanes)) {
      reportGdsSourceWrite(RaceViolation::Space::VGPR, reg, std::countr_zero(lock.lanes & lanes),
                           lock.instruction);
      return;
    }
}

void WaveRaceState::dispatch(const PendingWaitCount &wait_count) {
  for (const auto &update : wait_count.updates)
    applyWaitCounter(update.type, update.threshold);
}

void WaveRaceState::applyGfx10FullWait() {
  using amdgpu::WaitCounterType;
  for (auto counter : {WaitCounterType::VMCNT, WaitCounterType::LGKMCNT, WaitCounterType::EXPCNT,
                       WaitCounterType::VSCNT})
    applyWaitCounter(counter, 0);
}

void WaveRaceState::registerEvent(InstructionSnapshot instruction, MemoryEventType type,
                                  std::vector<uint32_t> regIds, uint64_t execMask,
                                  uint8_t byteMask) {
  registerEvent(instruction, type, std::move(regIds), execMask, byteMask,
                defaultWaitCounterType(type), defaultMemoryOrder(type));
}

void WaveRaceState::registerEvent(
    InstructionSnapshot instruction, MemoryEventType type, std::vector<uint32_t> regIds,
    uint64_t execMask, uint8_t byteMask, amdgpu::WaitCounterType waitCounterType,
    MemoryOrderClass memoryOrder,
    std::optional<amdgpu::WaitCounterType> additionalWaitCounterType) {
  std::array<amdgpu::MemoryCounterObligation, 2> obligations{
      amdgpu::MemoryCounterObligation{waitCounterType, memoryOrder}, {}};
  size_t count = 1;
  if (additionalWaitCounterType)
    obligations[count++] = {*additionalWaitCounterType, memoryOrder};
  registerEvent(instruction, type, std::move(regIds), execMask, byteMask,
                std::span(obligations.data(), count), memoryOrder);
}

void WaveRaceState::registerEvent(
    InstructionSnapshot instruction, MemoryEventType type, std::vector<uint32_t> regIds,
    uint64_t execMask, uint8_t byteMask,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations, VgprWriteOrder writeOrder,
    uint8_t lastRegisterByteMask, ConditionalVgprWrite feedback) {
  registerEventWithIntervals(instruction, type, std::move(regIds), execMask, byteMask, {},
                             counterObligations, writeOrder, lastRegisterByteMask, {}, feedback);
}

void WaveRaceState::registerScalarLoad(
    InstructionSnapshot instruction, RegisterRef destination, uint64_t execMask,
    amdgpu::WaitCounterType waitCounterType, MemoryOrderClass memoryOrder,
    std::optional<amdgpu::WaitCounterType> additionalWaitCounterType) {
  std::array<amdgpu::MemoryCounterObligation, 2> obligations{
      amdgpu::MemoryCounterObligation{waitCounterType, memoryOrder}, {}};
  size_t count = 1;
  if (additionalWaitCounterType)
    obligations[count++] = {*additionalWaitCounterType, memoryOrder};
  registerScalarLoad(instruction, destination, execMask, std::span(obligations.data(), count),
                     memoryOrder);
}

void WaveRaceState::registerScalarLoad(
    InstructionSnapshot instruction, RegisterRef destination, uint64_t execMask,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder) {
  const size_t limit = destination.cls == RegClass::SGPR   ? sgprMemoryEvents.size()
                       : destination.cls == RegClass::TTMP ? ttmpMemoryEvents.size()
                                                           : 0;
  if (destination.width == 0)
    return;
  if (limit == 0) {
    registerEvent(instruction, MemoryEventType::GLOBAL_TO_SGPR, {}, execMask, 0xF,
                  counterObligations, memoryOrder);
    return;
  }
  if (destination.index >= limit || destination.width > limit - destination.index)
    return;

  checkScalarWrite(destination);

  std::vector<uint32_t> registers(destination.width);
  for (uint32_t i = 0; i < destination.width; ++i)
    registers[i] = destination.index + i;
  registerEvent(instruction,
                destination.cls == RegClass::SGPR ? MemoryEventType::GLOBAL_TO_SGPR
                                                  : MemoryEventType::GLOBAL_TO_TTMP,
                std::move(registers), execMask, 0xF, counterObligations, memoryOrder);
}

EventId WaveRaceState::registerEventWithIntervals(
    InstructionSnapshot instruction, MemoryEventType type, std::vector<uint32_t> regIds,
    uint64_t execMask, uint8_t byteMask, IntervalSet ldsIntervals,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations, VgprWriteOrder writeOrder,
    uint8_t lastRegisterByteMask, LdsAccess ldsAccess, ConditionalVgprWrite feedback) {
  ProfileScope ps(*profiler_, "registerEvent");
  bool toSgpr = isToSgpr(type);
  bool toTtmp = isToTtmp(type);
  if (counterCapacities.compact_counter_only_issues && regIds.empty() && ldsIntervals.empty()) {
    // Unordered positions cannot prove any earlier data ready at a partial
    // wait. For an ordered position only an older event in its own FIFO matters.
    for (const auto obligation : counterObligations) {
      if (obligation.completion_class() == MemoryOrderClass::UNORDERED)
        continue;
      for (auto eventId : waveMemoryEvents | std::views::reverse)
        if (detector->appendCounterOnly(eventId, obligation))
          break;
    }
    return {};
  }
  const size_t register_limit = toSgpr   ? sgprMemoryEvents.size()
                                : toTtmp ? ttmpMemoryEvents.size()
                                         : vgprMemoryEvents.size();
  if (std::ranges::any_of(regIds, [register_limit](uint32_t reg) { return reg >= register_limit; }))
    return {};
  if (!toSgpr && !toTtmp) {
    for (auto reg : regIds) {
      regEventCountInc(type, reg);
    }
  }

  auto eventId = detector->allocateEventId(waveId, instruction, type, std::move(regIds), execMask,
                                           byteMask, std::move(ldsIntervals), counterObligations,
                                           writeOrder, lastRegisterByteMask, ldsAccess);
  if (feedback.lanes)
    detector->setConditionalVgprWrite(eventId, feedback);
  for (uint32_t reg : detector->events().registers(eventId)) {
    if (toSgpr) {
      sgprMemoryEvents[reg].push_back(eventId);
      sgprEventCount[reg]++;
    } else if (toTtmp) {
      ttmpMemoryEvents[reg].push_back(eventId);
      ttmpEventCount[reg]++;
    } else {
      vgprMemoryEvents[reg].push_back(eventId);
    }
  }
  waveMemoryEvents.push_back(eventId);
  return eventId;
}

amdgpu::WaitCounterType WaveRaceState::architecturalCounter(amdgpu::WaitCounterType type) const {
  if (!counterCapacities.split_wait_counters) {
    if (amdgpu::wait_counter_covers(amdgpu::WaitCounterType::VMCNT, type))
      return amdgpu::WaitCounterType::VMCNT;
    if (amdgpu::wait_counter_covers(amdgpu::WaitCounterType::LGKMCNT, type))
      return amdgpu::WaitCounterType::LGKMCNT;
    if (amdgpu::wait_counter_covers(amdgpu::WaitCounterType::VSCNT, type))
      return amdgpu::WaitCounterType::VSCNT;
  }
  return type;
}

bool WaveRaceState::hasPendingCounter(amdgpu::WaitCounterType counter) const {
  counter = architecturalCounter(counter);
  return std::ranges::any_of(waveMemoryEvents, [&](EventId event) {
    return detector->events().hasPendingWaitCounter(event, counter);
  });
}

void WaveRaceState::prepareForCounterIncrement(amdgpu::WaitCounterType type, uint8_t increment) {
  type = architecturalCounter(type);
  if (!counterCapacities.split_wait_counters && type != amdgpu::WaitCounterType::VMCNT &&
      type != amdgpu::WaitCounterType::LGKMCNT)
    return;
  // Split-counter targets have no aggregate VM, VS or LGKM queue.
  if (counterCapacities.split_wait_counters &&
      (type == amdgpu::WaitCounterType::VMCNT || type == amdgpu::WaitCounterType::VSCNT ||
       type == amdgpu::WaitCounterType::LGKMCNT))
    return;
  const int capacity = doNotWaitValue(type);
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
  for (const auto obligation : info.counter_obligations())
    prepareForCounterIncrement(obligation.wait_counter_type(), obligation.counter_increment());
}

void WaveRaceState::registerLdsEvent(InstructionSnapshot instruction, MemoryEventType type,
                                     std::vector<uint32_t> registers, uint64_t execMask,
                                     int waveSize, std::span<const uint32_t> laneBaseAddresses,
                                     int bytesPerLane, uint8_t byteMask) {
  registerLdsEvent(instruction, type, std::move(registers), execMask, waveSize, laneBaseAddresses,
                   bytesPerLane, byteMask, defaultWaitCounterType(type), defaultMemoryOrder(type));
}

void WaveRaceState::registerLdsEvent(
    InstructionSnapshot instruction, MemoryEventType type, std::vector<uint32_t> registers,
    uint64_t execMask, int waveSize, std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
    uint8_t byteMask, amdgpu::WaitCounterType waitCounterType, MemoryOrderClass memoryOrder,
    std::optional<amdgpu::WaitCounterType> additionalWaitCounterType) {
  std::array<amdgpu::MemoryCounterObligation, 2> obligations{
      amdgpu::MemoryCounterObligation{waitCounterType, memoryOrder}, {}};
  size_t count = 1;
  if (additionalWaitCounterType)
    obligations[count++] = {*additionalWaitCounterType, memoryOrder};
  registerLdsEvent(instruction, type, std::move(registers), execMask, waveSize, laneBaseAddresses,
                   bytesPerLane, byteMask, std::span(obligations.data(), count), memoryOrder);
}

EventId WaveRaceState::registerLdsEvent(
    InstructionSnapshot instruction, MemoryEventType type, std::vector<uint32_t> registers,
    uint64_t execMask, int waveSize, std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
    uint8_t byteMask, std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    VgprWriteOrder writeOrder, uint64_t ldsLaneMask, int elementBytes,
    std::optional<uint64_t> allocationEnd, LdsAccess ldsAccess) {
  IntervalSet intervals;
  forEachActiveLane(execMask & ldsLaneMask, waveSize, [&](int lane) {
    appendLdsInterval(intervals, laneBaseAddresses[lane], bytesPerLane, *detector, elementBytes,
                      allocationEnd ? std::optional{amdgpu::LdsAllocationRange{0, *allocationEnd}}
                                    : std::nullopt);
  });
  intervals.finalize();
  return registerEventWithIntervals(instruction, type, std::move(registers), execMask, byteMask,
                                    std::move(intervals), counterObligations, writeOrder, 0,
                                    ldsAccess);
}

EventId WaveRaceState::registerLdsEvent(
    InstructionSnapshot instruction, MemoryEventType type, std::vector<uint32_t> registers,
    uint64_t execMask, int waveSize, std::span<const uint32_t> firstLaneBaseAddresses,
    std::span<const uint32_t> secondLaneBaseAddresses, int bytesPerLane, uint8_t byteMask,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder, int elementBytes, LdsAccess ldsAccess) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    for (const auto addresses : {firstLaneBaseAddresses, secondLaneBaseAddresses})
      appendLdsInterval(intervals, addresses[lane], bytesPerLane, *detector, elementBytes);
  });
  intervals.finalize();
  return registerEventWithIntervals(instruction, type, std::move(registers), execMask, byteMask,
                                    std::move(intervals), counterObligations, memoryOrder, 0,
                                    ldsAccess);
}

void WaveRaceState::registerLdsRangeEvent(
    InstructionSnapshot instruction, MemoryEventType type, std::span<const LdsByteRange> ranges,
    std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder) {
  IntervalSet intervals;
  for (const auto &range : ranges)
    appendLdsInterval(intervals, range.address, static_cast<int>(range.bytes), *detector,
                      /*elementBytes=*/1);
  intervals.finalize();
  registerEventWithIntervals(instruction, type, {}, 0, 0xF, std::move(intervals),
                             counterObligations, memoryOrder);
}

void WaveRaceState::registerDualOffsetLdsEvent(InstructionSnapshot instruction,
                                               MemoryEventType type,
                                               std::vector<uint32_t> registers, uint64_t execMask,
                                               int waveSize,
                                               std::span<const uint32_t> laneBaseAddresses,
                                               int32_t offset0, int32_t offset1) {
  registerDualOffsetLdsEvent(instruction, type, std::move(registers), execMask, waveSize,
                             laneBaseAddresses, offset0, offset1, defaultWaitCounterType(type),
                             defaultMemoryOrder(type));
}

void WaveRaceState::registerDualOffsetLdsEvent(
    InstructionSnapshot instruction, MemoryEventType type, std::vector<uint32_t> registers,
    uint64_t execMask, int waveSize, std::span<const uint32_t> laneBaseAddresses, int32_t offset0,
    int32_t offset1, amdgpu::WaitCounterType waitCounterType, MemoryOrderClass memoryOrder,
    std::optional<amdgpu::WaitCounterType> additionalWaitCounterType) {
  std::array<amdgpu::MemoryCounterObligation, 2> obligations{
      amdgpu::MemoryCounterObligation{waitCounterType, memoryOrder}, {}};
  size_t count = 1;
  if (additionalWaitCounterType)
    obligations[count++] = {*additionalWaitCounterType, memoryOrder};
  registerDualOffsetLdsEvent(instruction, type, std::move(registers), execMask, waveSize,
                             laneBaseAddresses, offset0, offset1,
                             std::span(obligations.data(), count), memoryOrder);
}

void WaveRaceState::registerDualOffsetLdsEvent(
    InstructionSnapshot instruction, MemoryEventType type, std::vector<uint32_t> registers,
    uint64_t execMask, int waveSize, std::span<const uint32_t> laneBaseAddresses, int32_t offset0,
    int32_t offset1, std::span<const amdgpu::MemoryCounterObligation> counterObligations,
    MemoryOrderClass memoryOrder) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    uint32_t vAddr = laneBaseAddresses[lane];
    appendLdsInterval(intervals, vAddr + static_cast<uint32_t>(offset0) * 8, 8, *detector);
    appendLdsInterval(intervals, vAddr + static_cast<uint32_t>(offset1) * 8, 8, *detector);
  });
  intervals.finalize();
  registerEventWithIntervals(instruction, type, std::move(registers), execMask, 0xF,
                             std::move(intervals), counterObligations, memoryOrder);
}

void WaveRaceState::retireEventRegisters(EventId eventId) {
  ProfileScope ps(*profiler_, "retireEventRegisters");
  auto eventType = detector->events().type(eventId);
  bool toSgpr = isToSgpr(eventType);
  bool toTtmp = isToTtmp(eventType);
  for (uint32_t regId : detector->events().registers(eventId)) {
    if (toSgpr) {
      removeFromUnorderedList(sgprMemoryEvents[regId], eventId);
      sgprEventCount[regId]--;
    } else if (toTtmp) {
      removeFromUnorderedList(ttmpMemoryEvents[regId], eventId);
      ttmpEventCount[regId]--;
    } else {
      removeFromUnorderedList(getVgprMemoryEvents(regId), eventId);
      regEventCountDec(eventType, regId);
    }
  }
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
      const auto &events = detector->events();
      const bool ldsWasComplete = events.ldsAccessComplete(eventId);
      satisfied += events.pendingCounterIncrement(eventId, waitCounter);
      eventComplete = detector->satisfyEventWaitCounter(eventId, waitCounter);
      if (eventComplete) {
        retireEventRegisters(eventId);
        detector->markEventWaveComplete(eventId);
      }
      // A barrier can publish the LDS bytes as soon as their acknowledgment
      // arrives. The remaining counter may still own the VGPR destination.
      if (!ldsWasComplete && events.ldsAccessComplete(eventId) &&
          events.hasLiveLdsIntervals(eventId))
        barrierPendingEvents.push_back(eventId);
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
    if (counterCapacities.separate_vmem_types && type == amdgpu::WaitCounterType::VMCNT &&
        order == MemoryOrderClass::VMEM) {
      for (auto subtype : {amdgpu::VmemWriteback::Ordinary, amdgpu::VmemWriteback::Sample,
                           amdgpu::VmemWriteback::Bvh}) {
        const auto belongsToType = [&](EventId id) {
          return belongsToOrder(id) && detector->events().writeOrder(id).type == subtype;
        };
        satisfyOldestCounterObligations(pending_tokens(belongsToType) - maximumRemaining, type,
                                        belongsToType);
      }
      continue;
    }
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
  case amdgpu::WaitCounterType::BVHCNT:
    return amdgpu::WaitCounters::BVHCNT_MAX;
  case amdgpu::WaitCounterType::SAMPLECNT:
    return amdgpu::WaitCounters::SAMPLECNT_MAX;
  }
  assert(false && "invalid wait counter type");
  return -1;
}

void WaveRaceState::applyWaitCounter(amdgpu::WaitCounterType type, int threshold) {
  if (threshold < 0 || threshold == doNotWaitValue(type))
    return;

  if (threshold == 0) {
    if (type == amdgpu::WaitCounterType::EXPCNT) {
      gdsVgprLocks.clear();
      gdsExecLock.reset();
    }
    applyCounterConstraint(type, 0, /*includeUnordered=*/true);
    return;
  }

  applyCounterConstraint(type, threshold, /*includeUnordered=*/false);
}

void WaveRaceState::cancelRoutedLdsRead(EventId eventId) {
  // This source-only event has no register dependencies. Remove its counter positions
  // and its byte intervals together, preserving earlier requests at the same PC.
  const bool pending = std::erase(waveMemoryEvents, eventId) != 0;
  const bool awaitingBarrier = std::erase(barrierPendingEvents, eventId) != 0;
  if (pending || awaitingBarrier)
    detector->retireEvent(eventId);
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
  auto event = detector->allocateEventId(
      waveId, instruction, MemoryEventType::LDS_OPERAND_READ, {}, 0, 0, std::move(intervals),
      std::span<const amdgpu::MemoryCounterObligation>{}, MemoryOrderClass::UNORDERED);
  barrierPendingEvents.push_back(event);
  covered = bytes;
}

void WaveRaceState::flushBarrierPendingEvents() {
  ProfileScope ps(*profiler_, "removeEvents");
  for (EventId eventId : barrierPendingEvents) {
    if (detector->events().status(eventId) == EventStatus::WAVE_COMPLETE)
      detector->retireEvent(eventId);
    else
      detector->retireLdsIntervals(eventId);
  }
  barrierPendingEvents.clear();
  directLdsReadWidths.clear();
}

void WaveRaceState::checkVgprRead(int reg, int lane, uint8_t byteMask) const {
  checkVgprReadLanes(reg, uint64_t{1} << lane, byteMask);
}

void WaveRaceState::checkVgprReadLanes(int reg, uint64_t laneMask, uint8_t byteMask) const {
  if (laneMask == 0)
    return;
  for (EventId eid : vgprMemoryEvents[reg]) {
    uint64_t conflictMask = laneMask & detector->events().registerLanes(eid, reg);
    if (isToVgpr(detector->events().type(eid)) &&
        (detector->events().registerByteMask(eid, reg) & byteMask) != 0 && conflictMask != 0) {
      int lane = std::countr_zero(conflictMask);
      detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value, lane, false,
                                  detector->getWorkgroupId(), eid});
    }
  }
}

void WaveRaceState::checkVgprWrite(int reg, int lane, uint8_t byteMask) const {
  checkVgprWriteLanes(reg, uint64_t{1} << lane, byteMask);
}

void WaveRaceState::checkVgprWriteLanes(int reg, uint64_t laneMask, uint8_t byteMask) const {
  if (byteMask)
    checkGdsVgprWrite(reg, laneMask);
  checkVgprWrite(reg, laneMask, byteMask, MemoryOrderClass::UNORDERED);
}

void WaveRaceState::checkVgprWrite(int reg, uint64_t execMask, uint8_t byteMask,
                                   VgprWriteOrder writeOrder) const {
  if (!execMask)
    return;
  if (byteMask)
    checkGdsVgprWrite(reg, execMask);
  for (EventId eventId : vgprMemoryEvents[reg]) {
    const MemoryEventType pendingType = detector->events().type(eventId);
    // Same-class writeback can resolve a pending destination WAW. An export
    // still owns its source bytes until its independent EXPCNT event retires.
    const bool lockedSource = pendingType == MemoryEventType::GDS_ORDERED_COUNT_SOURCE;
    if ((!lockedSource && !isToVgpr(pendingType)) ||
        (detector->events().registerByteMask(eventId, reg) & byteMask) == 0) {
      continue;
    }

    uint64_t overlappingLanes = detector->events().registerLanes(eventId, reg) & execMask;
    if (!lockedSource)
      overlappingLanes =
          detector->events().conflictingWriteLanes(eventId, writeOrder, overlappingLanes);
    if (overlappingLanes == 0)
      continue;

    const int lane = static_cast<int>(std::countr_zero(overlappingLanes));
    detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value, lane, true,
                                detector->getWorkgroupId(), eventId});
  }
}

// Like checkVgprRead but for instructions that read all lanes (e.g. cross-lane ops).
// countr_zero picks the first active lane from the event's exec mask as the
// representative lane for the violation report.
void WaveRaceState::checkVgprReadAllLanes(int reg) const {
  if (getRegEventCount(MemoryEventType::GLOBAL_TO_VGPR, reg) != 0 ||
      getRegEventCount(MemoryEventType::LDS_TO_VGPR, reg) != 0) {
    for (EventId eid : vgprMemoryEvents[reg]) {
      if (isToVgpr(detector->events().type(eid)) &&
          (detector->events().registerByteMask(eid, reg) & 0xF) != 0) {
        int lane = std::countr_zero(detector->events().registerLanes(eid, reg));
        detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value, lane, false,
                                    detector->getWorkgroupId(), eid});
      }
    }
  }
}

bool WaveRaceState::isOutstandingFromVgpr(int lane, int reg) const {
  for (EventId eid : vgprMemoryEvents[reg]) {
    if (isFromVgpr(detector->events().type(eid)) && detector->events().isActiveForLane(eid, lane)) {
      return true;
    }
  }
  return false;
}

void WaveRaceState::checkScalarAccess(RegisterRef ref, bool isWrite) const {
  const std::vector<std::vector<EventId>> *events = nullptr;
  const std::vector<int> *counts = nullptr;
  RaceViolation::Space space;
  MemoryEventType type;
  if (ref.cls == RegClass::SGPR) {
    events = &sgprMemoryEvents;
    counts = &sgprEventCount;
    space = RaceViolation::Space::SGPR;
    type = MemoryEventType::GLOBAL_TO_SGPR;
  } else if (ref.cls == RegClass::TTMP) {
    events = &ttmpMemoryEvents;
    counts = &ttmpEventCount;
    space = RaceViolation::Space::TTMP;
    type = MemoryEventType::GLOBAL_TO_TTMP;
  } else {
    return;
  }

  if (ref.width == 0 || ref.index >= events->size() || ref.width > events->size() - ref.index)
    return;
  std::vector<EventId> reportedEvents;
  for (uint32_t offset = 0; offset < ref.width; ++offset) {
    const uint32_t index = ref.index + offset;
    if ((*counts)[index] == 0)
      continue;
    for (EventId eid : (*events)[index]) {
      if (detector->events().type(eid) != type ||
          std::ranges::find(reportedEvents, eid) != reportedEvents.end())
        continue;
      reportedEvents.push_back(eid);
      detector->getRaceHandler()({space, static_cast<int>(index), waveId.value, -1, isWrite,
                                  detector->getWorkgroupId(), eid});
    }
  }
}

void WaveRaceState::checkScalarRead(RegisterRef ref) const {
  checkScalarAccess(ref, /*isWrite=*/false);
}

void WaveRaceState::checkScalarWrite(RegisterRef ref) const {
  checkScalarAccess(ref, /*isWrite=*/true);
}

} // namespace rocjitsu::plugins::race_detector
