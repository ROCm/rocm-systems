// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include <algorithm>
#include <bit>
#include <span>

namespace rocjitsu::plugins::race_detector {
namespace {
/// RAII guard for ProfilerInterface::beginScope/endScope.
struct ProfileScope {
  ProfilerInterface &p;
  ProfileScope(ProfilerInterface &p, std::string_view key) : p(p) { p.beginScope(key); }
  ~ProfileScope() { p.endScope(); }
};
} // namespace

WaveRaceState::WaveRaceState(int vgprCount, int sgprCount, WaveId waveId, RaceDetector *detector)
    : waveId(waveId), detector(detector) {
  vgprMemoryEvents.resize(vgprCount);
  sgprMemoryEvents.resize(sgprCount);
  sgprEventCount.resize(sgprCount, 0);
  ttmpMemoryEvents.resize(REGISTER_SET_MAX_TTMPS);
  ttmpEventCount.resize(REGISTER_SET_MAX_TTMPS, 0);
  for (auto &counts : regEventCount) {
    counts.resize(vgprCount, 0);
  }
}

void WaveRaceState::dispatch(const PendingWaitCount &wait_count) {
  for (const auto &update : wait_count.updates)
    applyWaitCounter(update.type, update.threshold);
}

void WaveRaceState::registerEvent(uint64_t pc, MemoryEventType type, std::vector<uint32_t> regIds,
                                  uint64_t execMask, uint8_t byteMask) {
  registerEvent(pc, type, std::move(regIds), execMask, byteMask, defaultWaitCounterType(type));
}

void WaveRaceState::registerEvent(uint64_t pc, MemoryEventType type, std::vector<uint32_t> regIds,
                                  uint64_t execMask, uint8_t byteMask,
                                  amdgpu::WaitCounterType waitCounterType) {
  registerEventWithIntervals(pc, type, std::move(regIds), execMask, byteMask, {}, waitCounterType);
}

void WaveRaceState::registerScalarLoad(uint64_t pc, RegisterRef destination, uint64_t execMask,
                                       amdgpu::WaitCounterType waitCounterType) {
  const size_t limit = destination.cls == RegClass::SGPR   ? sgprMemoryEvents.size()
                       : destination.cls == RegClass::TTMP ? ttmpMemoryEvents.size()
                                                           : 0;
  if (destination.width == 0)
    return;
  if (limit == 0) {
    registerEvent(pc, MemoryEventType::GLOBAL_TO_SGPR, {}, execMask, 0xF, waitCounterType);
    return;
  }
  if (destination.index >= limit || destination.width > limit - destination.index)
    return;

  checkScalarWrite(destination);

  std::vector<uint32_t> registers(destination.width);
  for (uint32_t i = 0; i < destination.width; ++i)
    registers[i] = destination.index + i;
  registerEvent(pc,
                destination.cls == RegClass::SGPR ? MemoryEventType::GLOBAL_TO_SGPR
                                                  : MemoryEventType::GLOBAL_TO_TTMP,
                std::move(registers), execMask, 0xF, waitCounterType);
}

void WaveRaceState::registerEventWithIntervals(uint64_t pc, MemoryEventType type,
                                               std::vector<uint32_t> regIds, uint64_t execMask,
                                               uint8_t byteMask, IntervalSet ldsIntervals,
                                               amdgpu::WaitCounterType waitCounterType) {
  ProfileScope ps(*profiler_, "registerEvent");
  bool toSgpr = isToSgpr(type);
  bool toTtmp = isToTtmp(type);
  const size_t register_limit = toSgpr   ? sgprMemoryEvents.size()
                                : toTtmp ? ttmpMemoryEvents.size()
                                         : vgprMemoryEvents.size();
  if (std::any_of(regIds.begin(), regIds.end(),
                  [register_limit](uint32_t reg) { return reg >= register_limit; }))
    return;
  if (!toSgpr && !toTtmp) {
    for (auto reg : regIds) {
      regEventCountInc(type, reg);
    }
  }

  auto eventId = detector->allocateEventId(waveId, pc, type, std::move(regIds), execMask, byteMask,
                                           std::move(ldsIntervals), waitCounterType);
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
}

void WaveRaceState::registerLdsEvent(uint64_t pc, MemoryEventType type,
                                     std::vector<uint32_t> registers, uint64_t execMask,
                                     int waveSize, std::span<const uint32_t> laneBaseAddresses,
                                     int bytesPerLane, uint8_t byteMask) {
  registerLdsEvent(pc, type, std::move(registers), execMask, waveSize, laneBaseAddresses,
                   bytesPerLane, byteMask, defaultWaitCounterType(type));
}

void WaveRaceState::registerLdsEvent(uint64_t pc, MemoryEventType type,
                                     std::vector<uint32_t> registers, uint64_t execMask,
                                     int waveSize, std::span<const uint32_t> laneBaseAddresses,
                                     int bytesPerLane, uint8_t byteMask,
                                     amdgpu::WaitCounterType waitCounterType) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    int addr = static_cast<int>(laneBaseAddresses[lane]);
    intervals.append(addr, addr + bytesPerLane);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(registers), execMask, byteMask,
                             std::move(intervals), waitCounterType);
}

void WaveRaceState::registerDualOffsetLdsEvent(uint64_t pc, MemoryEventType type,
                                               std::vector<uint32_t> registers, uint64_t execMask,
                                               int waveSize,
                                               std::span<const uint32_t> laneBaseAddresses,
                                               int32_t offset0, int32_t offset1) {
  registerDualOffsetLdsEvent(pc, type, std::move(registers), execMask, waveSize, laneBaseAddresses,
                             offset0, offset1, defaultWaitCounterType(type));
}

void WaveRaceState::registerDualOffsetLdsEvent(uint64_t pc, MemoryEventType type,
                                               std::vector<uint32_t> registers, uint64_t execMask,
                                               int waveSize,
                                               std::span<const uint32_t> laneBaseAddresses,
                                               int32_t offset0, int32_t offset1,
                                               amdgpu::WaitCounterType waitCounterType) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    uint32_t vAddr = laneBaseAddresses[lane];
    int intAddr0 = static_cast<int>(vAddr + static_cast<uint32_t>(offset0) * 8);
    intervals.append(intAddr0, intAddr0 + 8);
    int intAddr1 = static_cast<int>(vAddr + static_cast<uint32_t>(offset1) * 8);
    intervals.append(intAddr1, intAddr1 + 8);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(registers), execMask, 0xF, std::move(intervals),
                             waitCounterType);
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

template <typename Pred> void WaveRaceState::resolveWaitCnt(int limit, Pred isTargetEvent) {
  int total = 0;
  for (auto eid : waveMemoryEvents)
    if (isTargetEvent(eid))
      total++;
  int toRetire = total - limit;
  if (toRetire <= 0)
    return;

  int retired = 0;
  size_t write = 0;
  for (size_t read = 0; read < waveMemoryEvents.size(); ++read) {
    EventId eid = waveMemoryEvents[read];
    if (isTargetEvent(eid) && retired < toRetire) {
      retired++;
      retireEventRegisters(eid);
      detector->markEventWaveComplete(eid);
      // Trimmable WAVE_COMPLETE events may be removed from the registry
      // immediately. Only keep non-trimmable events for later barrier retire;
      // otherwise a later barrier could try to retire stale EventIds.
      if (!detector->events().isTrimmable(eid))
        barrierPendingEvents.push_back(eid);
    } else {
      waveMemoryEvents[write++] = eid;
    }
  }
  waveMemoryEvents.resize(write);
}

void WaveRaceState::applyWaitCounter(amdgpu::WaitCounterType type, int threshold) {
  if (threshold < 0)
    return;

  // Scalar-memory operations are not guaranteed to complete in issue order.
  // A nonzero scalar or combined wait therefore cannot identify a particular
  // scalar destination as complete. DS operations are ordered only within an
  // operation class, so a combined LGKM wait can retire events older than its
  // remaining-operation bound independently for DS reads and DS writes.
  if (threshold != 0 &&
      (type == amdgpu::WaitCounterType::LGKMCNT || type == amdgpu::WaitCounterType::KMCNT)) {
    if (type == amdgpu::WaitCounterType::LGKMCNT)
      for (MemoryEventType ds_type : {MemoryEventType::LDS_TO_VGPR, MemoryEventType::VGPR_TO_LDS})
        resolveWaitCnt(threshold, [this, type, ds_type](EventId event_id) {
          return amdgpu::wait_counter_covers(type, detector->events().waitCounterType(event_id)) &&
                 detector->events().type(event_id) == ds_type;
        });
    return;
  }

  resolveWaitCnt(threshold, [this, type](EventId event_id) {
    return amdgpu::wait_counter_covers(type, detector->events().waitCounterType(event_id));
  });
}

void WaveRaceState::flushBarrierPendingEvents() {
  ProfileScope ps(*profiler_, "removeEvents");
  for (EventId eventId : barrierPendingEvents) {
    detector->retireEvent(eventId);
  }
  barrierPendingEvents.clear();
}

void WaveRaceState::checkVgprRead(int reg, int lane, uint8_t byteMask) const {
  checkVgprReadLanes(reg, uint64_t{1} << lane, byteMask);
}

void WaveRaceState::checkVgprReadLanes(int reg, uint64_t laneMask, uint8_t byteMask) const {
  if (laneMask == 0)
    return;
  for (EventId eid : vgprMemoryEvents[reg]) {
    uint64_t conflictMask = laneMask & detector->events().execMask(eid);
    if (isToVgpr(detector->events().type(eid)) &&
        (detector->events().byteMask(eid) & byteMask) != 0 && conflictMask != 0) {
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
  if (laneMask == 0)
    return;
  for (EventId eid : vgprMemoryEvents[reg]) {
    uint64_t conflictMask = laneMask & detector->events().execMask(eid);
    if (isToVgpr(detector->events().type(eid)) &&
        (detector->events().byteMask(eid) & byteMask) != 0 && conflictMask != 0) {
      int lane = std::countr_zero(conflictMask);
      detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value, lane, true,
                                  detector->getWorkgroupId(), eid});
    }
  }
}

// Like checkVgprRead but for instructions that read all lanes (e.g. cross-lane ops).
// countr_zero picks the first active lane from the event's exec mask as the
// representative lane for the violation report.
void WaveRaceState::checkVgprReadAllLanes(int reg) const {
  if (getRegEventCount(MemoryEventType::GLOBAL_TO_VGPR, reg) != 0 ||
      getRegEventCount(MemoryEventType::LDS_TO_VGPR, reg) != 0) {
    for (EventId eid : vgprMemoryEvents[reg]) {
      if (isToVgpr(detector->events().type(eid)) && (detector->events().byteMask(eid) & 0xF) != 0) {
        int lane = std::countr_zero(detector->events().execMask(eid));
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
          std::find(reportedEvents.begin(), reportedEvents.end(), eid) != reportedEvents.end())
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
