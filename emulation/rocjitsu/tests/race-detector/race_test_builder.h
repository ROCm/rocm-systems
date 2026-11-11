// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// RaceTestBuilder: test helper for race detection logic.
//
// Drives RaceDetector + WaveRaceState directly — no instruction parser,
// no assembly, no emulator.

#pragma once

#include "race_test_instruction.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <gtest/gtest.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
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

  EventId counterIssue(int wave, amdgpu::WaitCounterType counter, MemoryOrderClass order,
                       uint8_t increment = 1) {
    EXPECT_TRUE(waves_[wave]->prepareForCounterIncrement(counter, increment).succeeded());
    const std::array obligations{amdgpu::MemoryCounterObligation{counter, order, increment}};
    waves_[wave]->registerCounterEvent(testInstruction(pc_++), obligations);
    return waves_[wave]->getWaveMemoryEvents().back();
  }

  /// Register a Direct-to-LDS global load (tracked by vmcnt).
  /// ldsAddrs is per-active-lane; padded to waveSize internally.
  void globalToLds(int wave, std::vector<uint32_t> ldsAddrs, int bytesPerLane, uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    constexpr MemoryOrderClass memoryOrder = MemoryOrderClass::VMEM;
    ASSERT_TRUE(
        waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT).succeeded());
    ldsAddrs.resize(waveSize_, 0);
    // Validate each active lane's write against outstanding reads, as the plugin does.
    for (int lane = 0; lane < waveSize_; ++lane)
      if (exec & (1ULL << lane))
        detector_->validateWrite(static_cast<int>(ldsAddrs[lane]), WaveId{wave}, lane, bytesPerLane,
                                 memoryOrder);
    waves_[wave]->registerLdsEvent(
        testInstruction(pc_++), LdsAccess::Write, exec, waveSize_, ldsAddrs, bytesPerLane,
        CounterObligations{amdgpu::WaitCounterType::VMCNT, memoryOrder}, memoryOrder);
  }

  /// Register a tensor DMA load into LDS (tracked by tensorcnt). The load
  /// writes every range; it is validated against outstanding LDS reads.
  void tensorLoadToLds(int wave, std::vector<LdsByteRange> ranges) {
    tensorTransfer(wave, LdsAccess::Write, ranges);
  }

  /// Register a tensor DMA store from LDS (tracked by tensorcnt). The store
  /// reads every range; it is validated against outstanding LDS writes.
  void tensorStoreFromLds(int wave, std::vector<LdsByteRange> ranges) {
    tensorTransfer(wave, LdsAccess::Read, ranges);
  }

  /// Apply issue-time backpressure before checking an operation's accesses.
  void prepareCounterIncrement(int wave, amdgpu::WaitCounterType counter) {
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(counter).succeeded());
  }

  void prepareMemoryIssue(int wave, const amdgpu::MemoryIssueInfo &info) {
    ASSERT_TRUE(waves_[wave]->prepareForMemoryIssue(info).succeeded());
  }

  /// Register an LDS write and validate against outstanding reads.
  void ldsWrite(int wave, int lane, int addr, int bytes,
                MemoryOrderClass memoryOrder = MemoryOrderClass::LDS,
                amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT,
                std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt) {
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(waitCounterType).succeeded());
    if (additionalWaitCounterType) {
      ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(*additionalWaitCounterType).succeeded());
    }
    detector_->validateWrite(addr, WaveId{wave}, lane, bytes, memoryOrder);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    waves_[wave]->registerLdsEvent(
        testInstruction(pc_++), LdsAccess::Write, laneMask, waveSize_, ldsAddrs, bytes,
        CounterObligations{waitCounterType, memoryOrder, additionalWaitCounterType}, memoryOrder);
  }

  /// Register an LDS read and validate against outstanding writes.
  void ldsRead(int wave, int lane, int addr, int bytes,
               amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT,
               MemoryOrderClass memoryOrder = MemoryOrderClass::LDS,
               std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt) {
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(waitCounterType).succeeded());
    if (additionalWaitCounterType) {
      ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(*additionalWaitCounterType).succeeded());
    }
    detector_->validateRead(addr, WaveId{wave}, lane, bytes, memoryOrder);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    waves_[wave]->registerLdsEvent(
        testInstruction(pc_++), LdsAccess::Read, laneMask, waveSize_, ldsAddrs, bytes,
        CounterObligations{waitCounterType, memoryOrder, additionalWaitCounterType}, memoryOrder);
  }

  /// Record the LDS side of a mixed FLAT issue. Both counter domains remain
  /// attached to this event, with legacy CDNA's unordered completion classes.
  void mixedFlatLdsRead(int wave, int lane, int addr, int bytes) {
    ldsRead(wave, lane, addr, bytes, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::UNORDERED,
            amdgpu::WaitCounterType::LGKMCNT);
  }

  /// Record the LDS write of a mixed FLAT issue with both counter obligations.
  void mixedFlatLdsWrite(int wave, int lane, int addr, int bytes) {
    ldsWrite(wave, lane, addr, bytes, MemoryOrderClass::UNORDERED, amdgpu::WaitCounterType::VMCNT,
             amdgpu::WaitCounterType::LGKMCNT);
  }

  // -- Sync --

  /// Dispatch s_waitcnt. -1 means "don't change this counter".
  void waitcnt(int wave, int vmcnt = -1, int lgkmcnt = -1) {
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::VMCNT, vmcnt);
    wait.add(amdgpu::WaitCounterType::LGKMCNT, lgkmcnt);
    waves_[wave]->dispatch(wait);
  }

  void waitTensorcnt(int wave, int tensorcnt) {
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::TENSORCNT, tensorcnt);
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
      if (v.address == addr) {
        return true;
      }
    }
    return false;
  }

  void clearViolations() { violations_.clear(); }

  const EventRegistry &events() const { return detector_->events(); }
  const RaceDetector &detector() const { return *detector_; }

private:
  /// A transfer has no lanes; its violations report lane -1. Validates and
  /// registers the same bytes as the plugin's tensor DMA hook.
  void tensorTransfer(int wave, LdsAccess access, std::span<const LdsByteRange> ranges) {
    // Completion is in issue order (TENSOR); LDS accesses are checked as
    // unordered, a conservative rule, as in the plugin.
    constexpr MemoryOrderClass memoryOrder = MemoryOrderClass::TENSOR;
    constexpr MemoryOrderClass ldsAccessOrder = MemoryOrderClass::UNORDERED;
    for (const auto &range : ranges) {
      const auto bytes = static_cast<int>(
          detector_->ldsBytesInRange(range.address, range.bytes, /*elementBytes=*/1));
      if (bytes == 0)
        continue;
      const int addr = static_cast<int>(range.address);
      if (access == LdsAccess::Write)
        detector_->validateWrite(addr, WaveId{wave}, /*lane=*/-1, bytes, ldsAccessOrder);
      else
        detector_->validateRead(addr, WaveId{wave}, /*lane=*/-1, bytes, ldsAccessOrder);
    }
    const std::array obligations{
        amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::TENSORCNT, memoryOrder}};
    waves_[wave]->registerLdsRangeEvent(testInstruction(pc_++), access, ranges, obligations,
                                        memoryOrder);
  }

  std::unique_ptr<RaceDetector> detector_;
  std::vector<WaveRaceState *> waves_;
  std::vector<RaceViolation> violations_;
  int waveSize_;
  uint64_t defaultExec_;
  int pc_ = 0; // Auto-incrementing fake PC for event registration.
};

} // namespace rocjitsu::plugins::race_detector
