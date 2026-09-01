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

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocjitsu::plugins::race_detector {

class RaceTestBuilder {
public:
  RaceTestBuilder(int numWaves, int vgprs, int sgprs, int waveSize = 64, Dim3d wgId = Dim3d(0),
                  bool modelCounterBackpressure = true)
      : waveSize_(waveSize), defaultExec_(waveSize == 64 ? ~0ULL : (1ULL << waveSize) - 1) {
    detector_ = std::make_unique<RaceDetector>(
        numWaves, vgprs, sgprs, wgId, [this](RaceViolation v) { violations_.push_back(v); },
        /*vmcntNoWait=*/63, /*lgkmcntNoWait=*/15, modelCounterBackpressure);
    for (int w = 0; w < numWaves; ++w) {
      waves_.push_back(&detector_->getWaveRaceState(w));
    }
  }

  // -- Memory events --

  /// Register a global load into VGPRs (tracked by vmcnt).
  void
  globalLoad(int wave, int vgprBase, int numRegs, uint64_t exec = 0, uint8_t byteMask = 0xF,
             MemoryOrdering ordering = defaultMemoryOrdering(MemoryEventType::GLOBAL_TO_VGPR)) {
    if (!exec) {
      exec = defaultExec_;
    }
    waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT);
    std::vector<uint32_t> regs(numRegs);
    for (int i = 0; i < numRegs; ++i) {
      regs[i] = vgprBase + i;
      waves_[wave]->checkVgprWrite(vgprBase + i, exec, byteMask, ordering.writeback);
    }
    waves_[wave]->registerEvent(pc_++, MemoryEventType::GLOBAL_TO_VGPR, std::move(regs), exec,
                                byteMask, amdgpu::WaitCounterType::VMCNT, ordering);
  }

  /// Register a Direct-to-LDS global load (tracked by vmcnt).
  /// ldsAddrs is per-active-lane; padded to waveSize internally.
  void globalToLds(int wave, std::vector<uint32_t> ldsAddrs, int bytesPerLane, uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    constexpr MemoryOrdering ordering{MemoryOrderClass::VMEM, MemoryOrderClass::VMEM};
    waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT);
    ldsAddrs.resize(waveSize_, 0);
    waves_[wave]->registerLdsEvent(pc_++, MemoryEventType::GLOBAL_TO_LDS,
                                   /*registers=*/{}, exec, waveSize_, ldsAddrs, bytesPerLane,
                                   /*byteMask=*/0xF, amdgpu::WaitCounterType::VMCNT, ordering);
  }

  /// Register a global store from VGPRs (tracked by vmcnt).
  /// Stores read VGPRs at issue time — no destination registers.
  void globalStore(int wave, uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    constexpr MemoryOrdering ordering{MemoryOrderClass::VMEM, MemoryOrderClass::VMEM};
    waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT);
    waves_[wave]->registerEvent(pc_++, MemoryEventType::VGPR_TO_GLOBAL,
                                /*registers=*/{}, exec, /*byteMask=*/0xF,
                                amdgpu::WaitCounterType::VMCNT, ordering);
  }

  /// Register a scalar load into SGPRs with its architecture-specific counter.
  void scalarLoad(int wave, int sgprBase, int numRegs,
                  amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT) {
    waves_[wave]->prepareForCounterIncrement(waitCounterType, numRegs > 1 ? 2 : 1);
    waves_[wave]->registerScalarLoad(
        pc_++,
        RegisterRef{RegClass::SGPR, static_cast<uint16_t>(sgprBase), static_cast<uint8_t>(numRegs)},
        defaultExec_, waitCounterType);
  }

  /// Register a scalar load into TTMPs with its architecture-specific counter.
  void ttmpLoad(int wave, int ttmpBase, int numRegs,
                amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT) {
    waves_[wave]->prepareForCounterIncrement(waitCounterType, numRegs > 1 ? 2 : 1);
    waves_[wave]->registerScalarLoad(
        pc_++,
        RegisterRef{RegClass::TTMP, static_cast<uint16_t>(ttmpBase), static_cast<uint8_t>(numRegs)},
        defaultExec_, waitCounterType);
  }

  /// Register a scalar store so partial waits retain counter ordering.
  void scalarStore(int wave,
                   amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT) {
    waves_[wave]->prepareForCounterIncrement(waitCounterType);
    waves_[wave]->registerEvent(pc_++, MemoryEventType::SCALAR_TO_GLOBAL, {}, defaultExec_, 0xF,
                                waitCounterType);
  }

  /// Apply the issue-time backpressure for a memory operation before checking
  /// any of that operation's register or LDS accesses.
  void prepareCounterIncrement(int wave, amdgpu::WaitCounterType counter, int increment = 1) {
    waves_[wave]->prepareForCounterIncrement(counter, increment);
  }

  /// Register an LDS write and validate against outstanding reads.
  void ldsWrite(int wave, int lane, int addr, int bytes) {
    constexpr MemoryOrdering ordering{MemoryOrderClass::LDS, MemoryOrderClass::LDS};
    waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::LGKMCNT);
    detector_->validateWrite(addr, WaveId{wave}, lane, bytes);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    waves_[wave]->registerLdsEvent(pc_++, MemoryEventType::VGPR_TO_LDS,
                                   /*registers=*/{}, laneMask, waveSize_, ldsAddrs, bytes,
                                   /*byteMask=*/0xF, amdgpu::WaitCounterType::LGKMCNT, ordering);
  }

  /// Register an LDS read and validate against outstanding writes.
  /// byteMask: which bytes of the destination VGPR are written by this load
  /// (0xF=full, 0x3=lo D16, 0xC=hi D16). Used for byte-level race tracking.
  void ldsRead(int wave, int lane, int addr, int bytes, int vgprDst, uint8_t byteMask = 0xF,
               amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT) {
    constexpr MemoryOrdering ordering{MemoryOrderClass::LDS, MemoryOrderClass::LDS};
    waves_[wave]->prepareForCounterIncrement(waitCounterType);
    detector_->validateRead(addr, WaveId{wave}, lane, bytes);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    std::vector<uint32_t> regs = {static_cast<uint32_t>(vgprDst)};
    waves_[wave]->registerLdsEvent(pc_++, MemoryEventType::LDS_TO_VGPR, std::move(regs), laneMask,
                                   waveSize_, ldsAddrs, bytes, byteMask, waitCounterType, ordering);
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

  /// Check a VGPR read for races against pending loads.
  /// byteMask selects which bytes of the 4-byte register are read:
  ///   0xF = all 4 bytes (default, full 32-bit read)
  ///   0x3 = lo 2 bytes (D16 lo half)
  ///   0xC = hi 2 bytes (D16 hi half)
  /// A race fires only when the pending load's bytes overlap the read's bytes.
  void checkVgprRead(int wave, int reg, int lane, uint8_t byteMask = 0xF) {
    waves_[wave]->checkVgprRead(reg, lane, byteMask);
  }

  void checkVgprReadLanes(int wave, int reg, uint64_t laneMask, uint8_t byteMask = 0xF) {
    waves_[wave]->checkVgprReadLanes(reg, laneMask, byteMask);
  }

  void checkVgprWrite(int wave, int reg, int lane, uint8_t byteMask = 0xF) {
    waves_[wave]->checkVgprWrite(reg, lane, byteMask);
  }

  void checkVgprWriteLanes(int wave, int reg, uint64_t laneMask, uint8_t byteMask = 0xF) {
    waves_[wave]->checkVgprWriteLanes(reg, laneMask, byteMask);
  }

  void checkSgprRead(int wave, int reg) {
    waves_[wave]->checkScalarRead(RegisterRef{RegClass::SGPR, static_cast<uint16_t>(reg), 1});
  }

  void checkTtmpRead(int wave, int reg) {
    waves_[wave]->checkScalarRead(RegisterRef{RegClass::TTMP, static_cast<uint16_t>(reg), 1});
  }

  void checkLdsRead(int wave, int lane, int addr, int bytes) {
    detector_->validateRead(addr, WaveId{wave}, lane, bytes);
  }

  void checkLdsWrite(int wave, int lane, int addr, int bytes) {
    detector_->validateWrite(addr, WaveId{wave}, lane, bytes);
  }

  // -- Results --

  bool hasRace() const { return !violations_.empty(); }
  int raceCount() const { return static_cast<int>(violations_.size()); }
  const std::vector<RaceViolation> &violations() const { return violations_; }

  bool hasVgprRace(int reg) const {
    for (const auto &v : violations_) {
      if (v.space == RaceViolation::Space::VGPR && v.index == reg) {
        return true;
      }
    }
    return false;
  }

  bool hasSgprRace(int reg) const {
    for (const auto &v : violations_) {
      if (v.space == RaceViolation::Space::SGPR && v.index == reg) {
        return true;
      }
    }
    return false;
  }

  bool hasTtmpRace(int reg) const {
    for (const auto &v : violations_) {
      if (v.space == RaceViolation::Space::TTMP && v.index == reg)
        return true;
    }
    return false;
  }

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

private:
  std::unique_ptr<RaceDetector> detector_;
  std::vector<WaveRaceState *> waves_;
  std::vector<RaceViolation> violations_;
  int waveSize_;
  uint64_t defaultExec_;
  int pc_ = 0; // Auto-incrementing fake PC for event registration.
};

} // namespace rocjitsu::plugins::race_detector
