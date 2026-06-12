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
  RaceTestBuilder(int numWaves, int vgprs, int sgprs, int waveSize = 64, Dim3d wgId = Dim3d(0))
      : waveSize_(waveSize), defaultExec_(waveSize == 64 ? ~0ULL : (1ULL << waveSize) - 1) {
    detector_ = std::make_unique<RaceDetector>(
        numWaves, vgprs, sgprs, wgId, [this](RaceViolation v) { violations_.push_back(v); });
    for (int w = 0; w < numWaves; ++w) {
      waves_.push_back(&detector_->getWaveRaceState(w));
    }
  }

  // -- Memory events --

  /// Register a global load into VGPRs (tracked by vmcnt).
  void globalLoad(int wave, int vgprBase, int numRegs, uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    std::vector<uint32_t> regs(numRegs);
    for (int i = 0; i < numRegs; ++i) {
      regs[i] = vgprBase + i;
    }
    waves_[wave]->registerEvent(pc_++, MemoryEventType::GLOBAL_TO_VGPR, std::move(regs), exec);
  }

  /// Register a Direct-to-LDS global load (tracked by vmcnt).
  /// ldsAddrs is per-active-lane; padded to waveSize internally.
  void globalToLds(int wave, std::vector<uint32_t> ldsAddrs, int bytesPerLane, uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    ldsAddrs.resize(waveSize_, 0);
    waves_[wave]->registerLdsEvent(pc_++, MemoryEventType::GLOBAL_TO_LDS,
                                   /*registers=*/{}, exec, waveSize_, ldsAddrs, bytesPerLane);
  }

  /// Register a global store from VGPRs (tracked by vmcnt).
  /// Stores read VGPRs at issue time — no destination registers.
  void globalStore(int wave, uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    waves_[wave]->registerEvent(pc_++, MemoryEventType::VGPR_TO_GLOBAL,
                                /*registers=*/{}, exec);
  }

  /// Register a scalar load into SGPRs (tracked by lgkmcnt).
  void scalarLoad(int wave, int sgprBase, int numRegs) {
    std::vector<uint32_t> regs(numRegs);
    for (int i = 0; i < numRegs; ++i) {
      regs[i] = sgprBase + i;
    }
    waves_[wave]->registerEvent(pc_++, MemoryEventType::GLOBAL_TO_SGPR, std::move(regs),
                                defaultExec_);
  }

  /// Register an LDS write and validate against outstanding reads.
  void ldsWrite(int wave, int lane, int addr, int bytes, uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    detector_->validateWrite(addr, WaveId{wave}, lane, bytes);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    waves_[wave]->registerLdsEvent(pc_++, MemoryEventType::VGPR_TO_LDS,
                                   /*registers=*/{}, laneMask, waveSize_, ldsAddrs, bytes);
  }

  /// Register an LDS read and validate against outstanding writes.
  /// byteMask: which bytes of the destination VGPR are written by this load
  /// (0xF=full, 0x3=lo D16, 0xC=hi D16). Used for byte-level race tracking.
  void ldsRead(int wave, int lane, int addr, int bytes, int vgprDst, uint8_t byteMask = 0xF,
               uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    detector_->validateRead(addr, WaveId{wave}, lane, bytes);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    std::vector<uint32_t> regs = {static_cast<uint32_t>(vgprDst)};
    waves_[wave]->registerLdsEvent(pc_++, MemoryEventType::LDS_TO_VGPR, std::move(regs), laneMask,
                                   waveSize_, ldsAddrs, bytes, byteMask);
  }

  // -- Sync --

  /// Dispatch s_waitcnt. -1 means "don't change this counter".
  void waitcnt(int wave, int vmcnt = -1, int lgkmcnt = -1) {
    waves_[wave]->dispatch(PendingWaitCount{vmcnt, lgkmcnt});
  }

  /// Barrier: flush all waves' WAVE_COMPLETE events (simulates s_barrier).
  void barrier() {
    for (auto *w : waves_) {
      w->flushWaveCompleteMemoryEvents();
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

  void checkSgprRead(int wave, int reg) { waves_[wave]->checkSgprRead(reg); }

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
