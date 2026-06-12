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
/// race checking. Holds a pointer to the shared RaceDetector for event
/// allocation and lifecycle transitions.
///
/// This class is self-contained: it has no dependency on Wave, Workgroup,
/// Emulator, or any instruction code.
class WaveRaceState {
public:
  WaveRaceState(int vgprCount, int sgprCount, WaveId waveId, RaceDetector *detector);

  /// Register an in-flight global memory event (no LDS intervals).
  void registerEvent(uint64_t pc, MemoryEventType type, std::vector<uint32_t> registers,
                     uint64_t execMask, uint8_t byteMask = 0xF);

  /// Register an LDS event with contiguous intervals built from per-lane base
  /// addresses. Each active lane contributes one interval of bytesPerLane
  /// starting at laneBaseAddresses[lane].
  void registerLdsEvent(uint64_t pc, MemoryEventType type, std::vector<uint32_t> registers,
                        uint64_t execMask, int waveSize,
                        std::span<const uint32_t> laneBaseAddresses, int bytesPerLane,
                        uint8_t byteMask = 0xF);

  /// Register an LDS event with dual-offset intervals. Each active lane
  /// contributes two 8-byte intervals at laneBaseAddresses[lane] + offset0*8
  /// and laneBaseAddresses[lane] + offset1*8.
  void registerDualOffsetLdsEvent(uint64_t pc, MemoryEventType type,
                                  std::vector<uint32_t> registers, uint64_t execMask, int waveSize,
                                  std::span<const uint32_t> laneBaseAddresses, int32_t offset0,
                                  int32_t offset1);

  /// Retire global memory events until at most vmcnt remain outstanding.
  void sWaitCntVmcnt(int vmcnt);

  /// Retire LDS events until at most lgkmcnt remain outstanding.
  void sWaitCntLgkmcnt(int lgkmcnt);

  /// Dispatch a pending memory event produced by an instruction executor.
  void dispatch(PendingMemoryEvent event);

  /// Dispatch a pending wait count produced by an s_waitcnt executor.
  void dispatch(PendingWaitCount waitCount);

  /// Discard all wave-complete events (called when all waves reach barrier).
  void flushWaveCompleteMemoryEvents();

  /// Check a full VGPR read for races. Calls the RaceHandler on violation.
  void checkVgprRead(int reg, int lane, uint8_t byteMask) const;

  /// Check all lanes of a VGPR for races (used by getVgprs bulk read).
  void checkVgprReadAllLanes(int reg) const;

  /// Check a scalar register read for races. Calls the RaceHandler on
  /// violation (outstanding s_load targeting this SGPR).
  void checkSgprRead(int reg) const;

  /// True if any outstanding store reads from the given VGPR lane.
  bool isOutstandingFromVgpr(int lane, int reg) const;

  /// Number of outstanding events of a given type on a register.
  int getRegEventCount(MemoryEventType type, int reg) const {
    return regEventCount[static_cast<int>(type)][reg];
  }

  std::vector<EventId> &getVgprMemoryEvents(int reg) { return vgprMemoryEvents[reg]; }

  const std::vector<EventId> &getWaveMemoryEvents() const { return waveMemoryEvents; }

  const std::vector<EventId> &getWaveCompleteMemoryEvents() const {
    return waveCompleteMemoryEvents;
  }

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
  std::vector<EventId> waveCompleteMemoryEvents;

  WaveId waveId;
  RaceDetector *detector;
  ProfilerInterface *profiler_ = &NullProfiler::instance();
};

} // namespace rocjitsu::plugins::race_detector
