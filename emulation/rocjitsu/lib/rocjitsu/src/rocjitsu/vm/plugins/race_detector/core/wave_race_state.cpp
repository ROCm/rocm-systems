// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
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
  for (auto &counts : regEventCount) {
    counts.resize(vgprCount, 0);
  }
}

void WaveRaceState::dispatch(PendingMemoryEvent event) {
  if (event.isDualOffset) {
    registerDualOffsetLdsEvent(event.pc, event.type, std::move(event.registers), event.execMask,
                               event.waveSize, event.laneBaseAddresses, event.offset0,
                               event.offset1);
  } else if (!event.laneBaseAddresses.empty()) {
    registerLdsEvent(event.pc, event.type, std::move(event.registers), event.execMask,
                     event.waveSize, event.laneBaseAddresses, event.bytesPerLane, event.byteMask);
  } else {
    registerEvent(event.pc, event.type, std::move(event.registers), event.execMask, event.byteMask);
  }
}

void WaveRaceState::dispatch(PendingWaitCount waitCount) {
  if (waitCount.vmcnt >= 0) {
    sWaitCntVmcnt(waitCount.vmcnt);
  }
  if (waitCount.lgkmcnt >= 0) {
    sWaitCntLgkmcnt(waitCount.lgkmcnt);
  }
}

void WaveRaceState::registerEvent(uint64_t pc, MemoryEventType type, std::vector<uint32_t> regIds,
                                  uint64_t execMask, uint8_t byteMask) {
  registerEventWithIntervals(pc, type, std::move(regIds), execMask, byteMask, {});
}

void WaveRaceState::registerEventWithIntervals(uint64_t pc, MemoryEventType type,
                                               std::vector<uint32_t> regIds, uint64_t execMask,
                                               uint8_t byteMask, IntervalSet ldsIntervals) {
  ProfileScope ps(*profiler_, "registerEvent");
  bool toSgpr = isToSgpr(type);
  if (!toSgpr) {
    for (auto reg : regIds) {
      regEventCountInc(type, reg);
    }
  }
  auto eventId = detector->allocateEventId(waveId, pc, type, std::move(regIds), execMask, byteMask,
                                           std::move(ldsIntervals));
  for (uint32_t reg : detector->events().registers(eventId)) {
    if (toSgpr) {
      sgprMemoryEvents[reg].push_back(eventId);
      sgprEventCount[reg]++;
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
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    int addr = static_cast<int>(laneBaseAddresses[lane]);
    intervals.append(addr, addr + bytesPerLane);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(registers), execMask, byteMask,
                             std::move(intervals));
}

void WaveRaceState::registerDualOffsetLdsEvent(uint64_t pc, MemoryEventType type,
                                               std::vector<uint32_t> registers, uint64_t execMask,
                                               int waveSize,
                                               std::span<const uint32_t> laneBaseAddresses,
                                               int32_t offset0, int32_t offset1) {
  IntervalSet intervals;
  forEachActiveLane(execMask, waveSize, [&](int lane) {
    uint32_t vAddr = laneBaseAddresses[lane];
    int intAddr0 = static_cast<int>(vAddr + static_cast<uint32_t>(offset0) * 8);
    intervals.append(intAddr0, intAddr0 + 8);
    int intAddr1 = static_cast<int>(vAddr + static_cast<uint32_t>(offset1) * 8);
    intervals.append(intAddr1, intAddr1 + 8);
  });
  intervals.finalize();
  registerEventWithIntervals(pc, type, std::move(registers), execMask, 0xF, std::move(intervals));
}

void WaveRaceState::retireEventRegisters(EventId eventId) {
  ProfileScope ps(*profiler_, "retireEventRegisters");
  auto eventType = detector->events().type(eventId);
  bool toSgpr = isToSgpr(eventType);
  for (uint32_t regId : detector->events().registers(eventId)) {
    if (toSgpr) {
      removeFromUnorderedList(sgprMemoryEvents[regId], eventId);
      sgprEventCount[regId]--;
    } else {
      removeFromUnorderedList(getVgprMemoryEvents(regId), eventId);
      regEventCountDec(eventType, regId);
    }
  }
}

template <typename Pred> void WaveRaceState::resolveWaitCnt(int limit, Pred isTargetType) {
  int total = 0;
  for (auto eid : waveMemoryEvents)
    if (isTargetType(detector->events().type(eid)))
      total++;
  int toRetire = total - limit;
  if (toRetire <= 0)
    return;

  int retired = 0;
  size_t write = 0;
  for (size_t read = 0; read < waveMemoryEvents.size(); ++read) {
    EventId eid = waveMemoryEvents[read];
    if (isTargetType(detector->events().type(eid)) && retired < toRetire) {
      retired++;
      retireEventRegisters(eid);
      detector->markEventWaveComplete(eid);
      waveCompleteMemoryEvents.push_back(eid);
    } else {
      waveMemoryEvents[write++] = eid;
    }
  }
  waveMemoryEvents.resize(write);
}

void WaveRaceState::sWaitCntVmcnt(int vmcnt) {
  resolveWaitCnt(vmcnt, [](MemoryEventType type) {
    return type == MemoryEventType::GLOBAL_TO_VGPR || type == MemoryEventType::VGPR_TO_GLOBAL ||
           type == MemoryEventType::GLOBAL_TO_LDS;
  });
}

void WaveRaceState::sWaitCntLgkmcnt(int lgkmcnt) {
  resolveWaitCnt(lgkmcnt, [](MemoryEventType type) {
    return type == MemoryEventType::LDS_TO_VGPR || type == MemoryEventType::VGPR_TO_LDS ||
           type == MemoryEventType::GLOBAL_TO_SGPR;
  });
}

void WaveRaceState::flushWaveCompleteMemoryEvents() {
  ProfileScope ps(*profiler_, "removeEvents");
  for (EventId eventId : waveCompleteMemoryEvents) {
    detector->retireEvent(eventId);
  }
  waveCompleteMemoryEvents.clear();
}

void WaveRaceState::checkVgprRead(int reg, int lane, uint8_t byteMask) const {
  for (EventId eid : vgprMemoryEvents[reg]) {
    if (isToVgpr(detector->events().type(eid)) &&
        (detector->events().byteMask(eid) & byteMask) != 0 &&
        detector->events().isActiveForLane(eid, lane)) {
      detector->getRaceHandler()(
          {RaceViolation::Space::VGPR, reg, waveId.value, lane, false, detector->getWorkgroupId()});
    }
  }
}

void WaveRaceState::checkVgprReadAllLanes(int reg) const {
  if (getRegEventCount(MemoryEventType::GLOBAL_TO_VGPR, reg) != 0 ||
      getRegEventCount(MemoryEventType::LDS_TO_VGPR, reg) != 0) {
    for (EventId eid : vgprMemoryEvents[reg]) {
      if (isToVgpr(detector->events().type(eid)) && (detector->events().byteMask(eid) & 0xF) != 0) {
        int lane = std::countr_zero(detector->events().execMask(eid));
        detector->getRaceHandler()({RaceViolation::Space::VGPR, reg, waveId.value, lane, false,
                                    detector->getWorkgroupId()});
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

void WaveRaceState::checkSgprRead(int reg) const {
  if (sgprEventCount[reg] == 0) {
    return;
  }
  for (EventId eid : sgprMemoryEvents[reg]) {
    if (isToSgpr(detector->events().type(eid))) {
      detector->getRaceHandler()(
          {RaceViolation::Space::SGPR, reg, waveId.value, -1, false, detector->getWorkgroupId()});
    }
  }
}

} // namespace rocjitsu::plugins::race_detector
