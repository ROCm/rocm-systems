// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// RaceTestBuilder: test helper for race detection logic.
//
// Drives RaceDetector + WaveRaceState directly — no instruction parser,
// no assembly, no emulator.

#pragma once

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::plugins::race_detector {

class RaceTestBuilder {
public:
  RaceTestBuilder(int numWaves, int waveSize = 64, Dim3d wgId = Dim3d(0))
      : waveSize_(waveSize), defaultExec_(waveSize == 64 ? ~0ULL : (1ULL << waveSize) - 1) {
    detector_ = std::make_unique<RaceDetector>(
        numWaves, wgId, [this](RaceViolation v) { violations_.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4), /*ldsSizeBytes=*/64 * 1024);
    for (int w = 0; w < numWaves; ++w) {
      waves_.push_back(&detector_->getWaveRaceState(w));
    }
  }

  // -- Memory events --
  //
  // Each helper returns the event it registered, so tests can observe which
  // events a wait or counter-capacity constraint completes.

  /// Register a global load into VGPRs (tracked by vmcnt).
  EventId
  globalLoad(int wave,
             MemoryOrderClass memoryOrder = defaultMemoryOrder(MemoryEventType::GLOBAL_TO_VGPR)) {
    waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT);
    waves_[wave]->registerEvent(pc_++, MemoryEventType::GLOBAL_TO_VGPR,
                                amdgpu::WaitCounterType::VMCNT, memoryOrder);
    return lastEvent();
  }

  /// Register a generic FLAT load whose active lanes resolve uniformly to
  /// global memory. The caller supplies architecture-specific counter
  /// obligations because their domains and ordering vary by target.
  EventId flatGlobalLoad(int wave,
                         std::span<const amdgpu::MemoryCounterObligation> counterObligations) {
    waves_[wave]->registerEvent(pc_++, MemoryEventType::GLOBAL_TO_VGPR, counterObligations,
                                memoryOrderForObligations(counterObligations));
    return lastEvent();
  }

  /// Register a Direct-to-LDS global load (tracked by vmcnt).
  /// ldsAddrs is per-active-lane; padded to waveSize internally.
  EventId globalToLds(int wave, std::vector<uint32_t> ldsAddrs, int bytesPerLane,
                      uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    constexpr MemoryOrderClass memoryOrder = MemoryOrderClass::VMEM;
    waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT);
    ldsAddrs.resize(waveSize_, 0);
    waves_[wave]->registerLdsEvent(pc_++, MemoryEventType::GLOBAL_TO_LDS, exec, waveSize_, ldsAddrs,
                                   bytesPerLane, amdgpu::WaitCounterType::VMCNT, memoryOrder);
    return lastEvent();
  }

  /// Register a global store from VGPRs (tracked by vmcnt).
  EventId globalStore(int wave) {
    constexpr MemoryOrderClass memoryOrder = MemoryOrderClass::VMEM;
    waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT);
    waves_[wave]->registerEvent(pc_++, MemoryEventType::VGPR_TO_GLOBAL,
                                amdgpu::WaitCounterType::VMCNT, memoryOrder);
    return lastEvent();
  }

  /// Register a scalar load with its architecture-specific counter. A load of
  /// more than one dword contributes two counter tokens.
  EventId scalarLoad(int wave, int dwords = 1,
                     amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT) {
    const uint8_t increment = dwords > 1 ? 2 : 1;
    waves_[wave]->prepareForCounterIncrement(waitCounterType, increment);
    const std::array obligations{
        amdgpu::MemoryCounterObligation{waitCounterType, MemoryOrderClass::UNORDERED, increment}};
    waves_[wave]->registerEvent(pc_++, MemoryEventType::GLOBAL_TO_SGPR, obligations,
                                MemoryOrderClass::UNORDERED);
    return lastEvent();
  }

  /// Register a scalar store so partial waits retain counter ordering.
  EventId scalarStore(int wave,
                      amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT) {
    waves_[wave]->prepareForCounterIncrement(waitCounterType);
    waves_[wave]->registerEvent(pc_++, MemoryEventType::SCALAR_TO_GLOBAL, waitCounterType,
                                MemoryOrderClass::UNORDERED);
    return lastEvent();
  }

  /// Apply issue-time backpressure before checking an operation's accesses.
  void prepareCounterIncrement(int wave, amdgpu::WaitCounterType counter) {
    waves_[wave]->prepareForCounterIncrement(counter);
  }

  void prepareMemoryIssue(int wave, const amdgpu::MemoryIssueInfo &info) {
    waves_[wave]->prepareForMemoryIssue(info);
  }

  /// Register an LDS write and validate against outstanding reads.
  EventId
  ldsWrite(int wave, int lane, int addr, int bytes,
           MemoryOrderClass memoryOrder = MemoryOrderClass::LDS,
           amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT,
           std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt) {
    waves_[wave]->prepareForCounterIncrement(waitCounterType);
    if (additionalWaitCounterType)
      waves_[wave]->prepareForCounterIncrement(*additionalWaitCounterType);
    detector_->validateWrite(addr, WaveId{wave}, lane, bytes, memoryOrder);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    waves_[wave]->registerLdsEvent(pc_++, MemoryEventType::VGPR_TO_LDS, laneMask, waveSize_,
                                   ldsAddrs, bytes, waitCounterType, memoryOrder,
                                   additionalWaitCounterType);
    return lastEvent();
  }

  /// Register an LDS read and validate against outstanding writes.
  EventId ldsRead(int wave, int lane, int addr, int bytes,
                  amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT,
                  MemoryOrderClass memoryOrder = MemoryOrderClass::LDS,
                  std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt) {
    waves_[wave]->prepareForCounterIncrement(waitCounterType);
    if (additionalWaitCounterType)
      waves_[wave]->prepareForCounterIncrement(*additionalWaitCounterType);
    detector_->validateRead(addr, WaveId{wave}, lane, bytes, memoryOrder);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    waves_[wave]->registerLdsEvent(pc_++, MemoryEventType::LDS_TO_VGPR, laneMask, waveSize_,
                                   ldsAddrs, bytes, waitCounterType, memoryOrder,
                                   additionalWaitCounterType);
    return lastEvent();
  }

  /// Register a CDNA1-4 generic FLAT load whose active lanes resolve uniformly
  /// to LDS. Its unordered VMCNT and LGKMCNT obligations are independent of
  /// that resolved memory route.
  EventId flatLdsLoad(int wave, int lane, int addr, int bytes) {
    return ldsRead(wave, lane, addr, bytes, amdgpu::WaitCounterType::VMCNT,
                   MemoryOrderClass::UNORDERED, amdgpu::WaitCounterType::LGKMCNT);
  }

  /// Register a CDNA1-4 generic FLAT store whose active lane resolves to LDS.
  EventId flatLdsStore(int wave, int lane, int addr, int bytes) {
    return ldsWrite(wave, lane, addr, bytes, MemoryOrderClass::UNORDERED,
                    amdgpu::WaitCounterType::VMCNT, amdgpu::WaitCounterType::LGKMCNT);
  }

  // -- Sync --

  /// Dispatch s_waitcnt. -1 means "don't change this counter".
  void waitcnt(int wave, int vmcnt = -1, int lgkmcnt = -1) {
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::VMCNT, vmcnt);
    wait.add(amdgpu::WaitCounterType::LGKMCNT, lgkmcnt);
    waves_[wave]->dispatch(wait);
  }

  void waitKmcnt(int wave, int kmcnt) {
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::KMCNT, kmcnt);
    waves_[wave]->dispatch(wait);
  }

  /// Barrier: flush all waves' barrier-pending events (simulates s_barrier).
  void barrier() {
    for (auto *w : waves_) {
      w->flushBarrierPendingEvents();
    }
  }

  // -- Trigger race checks --

  void checkLdsRead(int wave, int lane, int addr, int bytes,
                    MemoryOrderClass memoryOrder = MemoryOrderClass::LDS) {
    detector_->validateRead(addr, WaveId{wave}, lane, bytes, memoryOrder);
  }

  void checkLdsWrite(int wave, int lane, int addr, int bytes,
                     MemoryOrderClass memoryOrder = MemoryOrderClass::LDS) {
    detector_->validateWrite(addr, WaveId{wave}, lane, bytes, memoryOrder);
  }

  // -- Results --

  bool hasRace() const { return !violations_.empty(); }
  int raceCount() const { return static_cast<int>(violations_.size()); }
  const std::vector<RaceViolation> &violations() const { return violations_; }

  bool hasLdsRace(int addr) const {
    for (const auto &v : violations_) {
      if (v.space == RaceViolation::Space::LDS && v.index == addr) {
        return true;
      }
    }
    return false;
  }

  void clearViolations() { violations_.clear(); }

  const EventRegistry &events() const { return detector_->events(); }

  /// True until every wait-counter obligation of the event is satisfied.
  /// Trimmed events completed before the registry dropped them.
  bool isPending(EventId event) const {
    return events().contains(event) && events().status(event) == EventStatus::ACTIVE;
  }

private:
  EventId lastEvent() const { return EventId{events().totalAllocated() - 1}; }

  std::unique_ptr<RaceDetector> detector_;
  std::vector<WaveRaceState *> waves_;
  std::vector<RaceViolation> violations_;
  int waveSize_;
  uint64_t defaultExec_;
  int pc_ = 0; // Auto-incrementing fake PC for event registration.
};

} // namespace rocjitsu::plugins::race_detector
