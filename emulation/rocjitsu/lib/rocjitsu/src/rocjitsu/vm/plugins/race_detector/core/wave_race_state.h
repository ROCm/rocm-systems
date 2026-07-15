// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/profiler_interface.h"
#include "rocjitsu/vm/plugins/race_detector/core/types.h"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu::plugins::race_detector {

class IntervalSet;
class RaceDetector;

/// Per-wave race detection state. Owns VGPR event lists, wave-level event
/// queues, and provides event registration, waitcnt resolution, and VGPR
/// race checking. Holds a pointer to the workgroup level RaceDetector for
/// event allocation and lifecycle transitions.
class WaveRaceState {
public:
  WaveRaceState(int vgprCount, int sgprCount, WaveId waveId, RaceDetector *detector);

  /// Register an in-flight memory event that does not involve LDS.
  /// \param pc The PC of the instruction that produced the event.
  /// \param type The type of the memory event.
  /// \param registers The register IDs involved in the event.
  /// \param execMask The execution mask at the time the event was produced.
  /// \param byteMask The byte mask of the event, if this event effects only certain bytes offset
  ///                 register(s). Defaults to 0xF (all bytes).
  void registerEvent(uint64_t pc, MemoryEventType type, std::vector<uint32_t> registers,
                     uint64_t execMask, uint8_t byteMask = 0xF);

  /// Register an in-flight memory event that involves LDS.
  /// The LDS memory involved is defined by `laneBaseAddresses` and `bytesPerLane`. Each active lane
  /// contributes one interval of size `bytesPerLane`.
  void registerLdsEvent(uint64_t pc, MemoryEventType type, std::vector<uint32_t> registers,
                        uint64_t execMask, int waveSize,
                        std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
                        uint8_t byteMask = 0xF);

  /// Register an LDS event with dual-offset intervals. Each active lane
  /// contributes two 8-byte intervals at laneBaseAddresses[lane] + offset0*8
  /// and laneBaseAddresses[lane] + offset1*8. So each active lane contributes 2 intervals of 8
  /// bytes each.
  void registerDualOffsetLdsEvent(uint64_t pc, MemoryEventType type,
                                  std::vector<uint32_t> registers, uint64_t execMask, int waveSize,
                                  std::span<const uint32_t> laneBaseAddresses, int32_t offset0,
                                  int32_t offset1);

  /// Retire global memory events until `vmcnt` or fewer remain outstanding. 'Retire' here means
  /// marking the event as wave-complete.
  void sWaitCntVmcnt(int vmcnt);

  /// Retire LDS events until at `lgkmcnt` or fewer remain outstanding.
  void sWaitCntLgkmcnt(int lgkmcnt);

  /// Dispatch a pending memory event produced by an instruction executor.
  void dispatch(PendingMemoryEvent event);

  /// Dispatch a pending wait count produced by an s_waitcnt executor.
  void dispatch(PendingWaitCount waitCount);

  /// Retire non-trimmable events whose owning wave completed them before a workgroup barrier.
  void flushBarrierPendingEvents();

  /// Check a full VGPR read for races. Calls the RaceHandler on violation.
  void checkVgprRead(int reg, int lane, uint8_t byteMask) const;

  /// Check all lanes of a VGPR for races (used by bulk register reads).
  void checkVgprReadAllLanes(int reg) const;

  /// Check a scalar register read for races. Calls the RaceHandler on
  /// violation (outstanding s_load targeting this SGPR).
  void checkSgprRead(int reg) const;

  /// True if any outstanding global/LDS store reads from the given VGPR lane.
  bool isOutstandingFromVgpr(int lane, int reg) const;

  /// Number of outstanding events of a given type on a register.
  int getRegEventCount(MemoryEventType type, int reg) const {
    return regEventCount[static_cast<int>(type)][reg];
  }

  std::vector<EventId> &getVgprMemoryEvents(int reg) { return vgprMemoryEvents[reg]; }

  const std::vector<EventId> &getWaveMemoryEvents() const { return waveMemoryEvents; }

  const std::vector<EventId> &getBarrierPendingEvents() const { return barrierPendingEvents; }

  RaceDetector *getDetector() { return detector; }
  const RaceDetector *getDetector() const { return detector; }

  void setProfiler(ProfilerInterface &p) { profiler_ = &p; }

  WaveId getWaveId() const { return waveId; }

private:
  void registerEventWithIntervals(uint64_t pc, MemoryEventType type,
                                  std::vector<uint32_t> registers, uint64_t execMask,
                                  uint8_t byteMask, IntervalSet ldsIntervals);
  void retireEventRegisters(EventId);

  template <typename Pred> void resolveWaitCnt(int limit, Pred isTargetType);

  void regEventCountInc(MemoryEventType type, int reg) {
    regEventCount[static_cast<int>(type)][reg]++;
  }
  void regEventCountDec(MemoryEventType type, int reg) {
    regEventCount[static_cast<int>(type)][reg]--;
  }

  std::vector<std::vector<EventId>> vgprMemoryEvents;
  std::vector<std::vector<EventId>> sgprMemoryEvents;
  std::vector<int> sgprEventCount;

  static constexpr int kNumEventTypes = static_cast<int>(MemoryEventType::N);
  std::array<std::vector<int>, kNumEventTypes> regEventCount;

  std::vector<EventId> waveMemoryEvents;
  std::vector<EventId> barrierPendingEvents;

  WaveId waveId;
  RaceDetector *detector;
  ProfilerInterface *profiler_ = &NullProfiler::instance();
};

} // namespace rocjitsu::plugins::race_detector
